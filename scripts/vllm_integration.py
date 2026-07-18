#!/usr/bin/env python3
"""
Step 5 (stretch): real vLLM serving with PARS-driven priority scheduling.

This replaces the discrete-event simulator with an actual vLLM engine.
vLLM natively supports priority-based scheduling (lower priority value =
served first, ties broken by arrival time) via:

    EngineArgs(..., scheduling_policy="priority")
    llm.generate(prompts, sampling_params, priority=[...])

We convert each PARS ranker score into an integer priority (shorter
predicted output -> lower priority value -> scheduled sooner), run the
batch through vLLM, and pull real GPU / KV-cache / preemption stats from
vLLM's engine metrics instead of a simulated clock.

Requires: pip install vllm  (needs a real GPU; run this on Colab/cloud,
not the same CPU box used for quick unit tests).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_labels
from src.ranker import load_ranker
from src.utils import load_config


def scores_to_priorities(scores):
    """
    vLLM priority is an int; LOWER value = served first.
    Our ranker score is trained so HIGHER score = longer predicted output.
    So we just rank scores ascending (shortest predicted first) and use
    the rank position itself as the integer priority.
    """
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    priorities = [0] * len(scores)
    for rank, idx in enumerate(order):
        priorities[idx] = rank
    return priorities


def run_policy(llm, prompts, sampling_params, priorities=None, policy_name="fcfs"):
    from vllm.engine.metrics import Stats  # noqa: F401  (import check only)

    start = time.perf_counter()
    if priorities is not None:
        outputs = llm.generate(prompts, sampling_params, priority=priorities)
    else:
        outputs = llm.generate(prompts, sampling_params)
    wall_time = time.perf_counter() - start

    # Real engine-level stats (GPU / KV-cache / preemptions), not simulated.
    # vLLM exposes these via the engine's stat logger; the exact attribute
    # path has moved between versions, so we defensively probe a couple.
    stats = {}
    try:
        engine = llm.llm_engine
        if hasattr(engine, "stat_loggers"):
            logger = engine.stat_loggers.get("prometheus") or next(
                iter(engine.stat_loggers.values())
            )
            stats = {
                "gpu_cache_usage": getattr(logger, "gpu_cache_usage_sys", None),
                "num_preemptions": getattr(logger, "num_preemptions_total", None),
            }
    except Exception as e:
        stats["stats_error"] = str(e)

    print(f"\n=== {policy_name.upper()} (real vLLM) ===")
    print(f"  wall time:    {wall_time:.2f}s")
    print(f"  throughput:   {len(prompts) / wall_time:.2f} req/s")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    return outputs, wall_time, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--labels", default="data/processed/prod_labels.json")
    parser.add_argument("--ranker", default="checkpoints/pairwise_ranker.pt")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("ERROR: vLLM is not installed. Run this on a GPU box: pip install vllm")
        sys.exit(1)

    cfg = load_config(args.config)
    records, meta = load_labels(args.labels)
    records = records[: args.limit]
    prompts = [r.text for r in records]

    ranker = load_ranker(args.ranker, device=args.device)
    scores = ranker.score(prompts)
    priorities = scores_to_priorities(scores)

    model_name = meta.get("llm") or cfg["llm"]["profiles"][cfg["llm"]["profile"]]["model"]
    print(f"Loading {model_name} in vLLM with priority scheduling enabled...")

    llm = LLM(
        model=model_name,
        scheduling_policy="priority",  # <-- the real integration point
        gpu_memory_utilization=0.85,
    )
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512)

    # FCFS baseline: no priority argument, so vLLM falls back to arrival order.
    run_policy(llm, prompts, sampling_params, priorities=None, policy_name="fcfs")

    # PARS: our ranker's scores drive real vLLM scheduling decisions.
    run_policy(llm, prompts, sampling_params, priorities=priorities, policy_name="pars")


if __name__ == "__main__":
    main()
