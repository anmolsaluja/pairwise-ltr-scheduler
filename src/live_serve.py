"""
Live LLM serving helpers (vLLM).

Maps FCFS / LTR / PARS(+priority) into vLLM's integer priority schedule
(lower value = served first) and measures real wall-clock throughput.
"""

from __future__ import annotations

import gc
import json
import os
import time
from dataclasses import asdict, dataclass

import torch

from src.requests import parse_priority


@dataclass
class LivePolicyResult:
    policy: str
    num_requests: int
    wall_time_s: float
    throughput_rps: float
    avg_output_tokens: float
    total_output_tokens: int
    notes: str = ""


def assign_eval_priorities(records):
    """Same mix as scripts/evaluate.py: ~1/8 high, ~1/5 low, rest normal."""
    n_high = n_low = 0
    for i, rec in enumerate(records):
        if i % 8 == 0:
            rec.priority = "high"
            n_high += 1
        elif i % 5 == 0:
            rec.priority = "low"
            n_low += 1
        else:
            rec.priority = "normal"
    return n_high, n_low


def scores_to_vllm_priorities(scores):
    """
    vLLM: lower int = served sooner.
    Our rank / length scores: higher => longer job => should wait.
    Rank ascending by score and use that rank as the priority int.
    """
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    priorities = [0] * len(scores)
    for rank, idx in enumerate(order):
        priorities[idx] = rank
    return priorities


def effective_scores(rank_scores, priorities, boosts, use_priority):
    out = []
    for score, pri in zip(rank_scores, priorities):
        b = boosts.get(pri, 0.0) if use_priority else 0.0
        out.append(float(score) + float(b))
    return out


def free_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def build_llm(model_name, gpu_memory_utilization=0.85, max_model_len=4096):
    from vllm import LLM

    kwargs = {
        "model": model_name,
        "scheduling_policy": "priority",
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "trust_remote_code": True,
    }
    # dtype auto works on most Colab / cloud GPUs
    try:
        return LLM(**kwargs, dtype="auto")
    except TypeError:
        return LLM(**kwargs)


def run_vllm_policy(llm, prompts, sampling_params, priorities, policy_name):
    start = time.perf_counter()
    if priorities is None:
        outputs = llm.generate(prompts, sampling_params)
    else:
        outputs = llm.generate(prompts, sampling_params, priority=priorities)
    wall = time.perf_counter() - start

    out_lens = [len(o.outputs[0].token_ids) for o in outputs]
    total_tok = int(sum(out_lens))
    result = LivePolicyResult(
        policy=policy_name,
        num_requests=len(prompts),
        wall_time_s=wall,
        throughput_rps=(len(prompts) / wall) if wall > 0 else 0.0,
        avg_output_tokens=(total_tok / len(out_lens)) if out_lens else 0.0,
        total_output_tokens=total_tok,
        notes="vLLM priority scheduling" if priorities is not None else "vLLM arrival/FCFS order",
    )
    return result, outputs


def print_live_result(result: LivePolicyResult, title: str):
    print(f"\n=== {title} ===")
    print(f"  policy id:          {result.policy}")
    print(f"  requests:           {result.num_requests}")
    print(f"  wall time:          {result.wall_time_s:.2f}s")
    print(f"  throughput:         {result.throughput_rps:.2f} req/s")
    print(f"  avg output tokens:  {result.avg_output_tokens:.1f}")
    print(f"  total output tokens:{result.total_output_tokens}")
    if result.notes:
        print(f"  notes:              {result.notes}")


def pct_gain(baseline, improved):
    """Positive => improved is faster (lower wall time)."""
    if baseline is None or improved is None or baseline <= 0:
        return None
    return 100.0 * (baseline - improved) / baseline


def save_live_results(path, payload):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved live results -> {path}")


def results_to_payload(meta, results: list[LivePolicyResult]):
    by = {r.policy: r for r in results}
    gains = {}
    if "fcfs" in by and "ltr" in by:
        gains["ltr_vs_fcfs_wall_pct"] = pct_gain(by["fcfs"].wall_time_s, by["ltr"].wall_time_s)
    if "ltr" in by and "pars" in by:
        gains["ours_vs_ltr_wall_pct"] = pct_gain(by["ltr"].wall_time_s, by["pars"].wall_time_s)
    if "fcfs" in by and "pars" in by:
        gains["ours_vs_fcfs_wall_pct"] = pct_gain(by["fcfs"].wall_time_s, by["pars"].wall_time_s)
    return {
        "meta": meta,
        "results": [asdict(r) for r in results],
        "wall_time_improvements_pct": gains,
    }


def parse_priorities_from_records(records):
    return [parse_priority(getattr(r, "priority", "normal")) for r in records]
