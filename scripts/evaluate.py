#!/usr/bin/env python3
"""
Step 4: evaluate prediction quality + scheduler latency.

Baselines (from midterm slides):
  - FCFS
  - ProD-M pointwise SJF  (original LTR-style length prediction)
  - PARS pairwise SJF     (our proposal extension)
  - Oracle SJF            (true median length — upper bound)

Also reports MAE, Kendall Tau, pairwise accuracy, NDCG, and approx TTFT.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_labels
from src.metrics import kendall_tau, mae, ndcg_at_k, pairwise_accuracy
from src.prod_m import load_hidden, load_prod_m
from src.ranker import load_ranker
from src.simulate import SimConfig, compare
from src.utils import load_config


def print_summary(s):
    print(f"\n=== {s.policy.upper()} ===")
    print(f"  requests:    {s.num_requests}")
    print(f"  avg latency: {s.avg_latency:.3f}s")
    print(f"  p50:         {s.p50_latency:.3f}s")
    print(f"  p95:         {s.p95_latency:.3f}s")
    print(f"  p99:         {s.p99_latency:.3f}s")
    print(f"  avg wait:    {s.avg_wait:.3f}s")
    print(f"  avg TTFT:    {s.avg_ttft:.3f}s")
    print(f"  throughput:  {s.throughput_rps:.2f} req/s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--labels", default="data/processed/prod_labels.json")
    parser.add_argument("--prod-m", default="checkpoints/prod_m.pt")
    parser.add_argument("--ranker", default="checkpoints/pairwise_ranker.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.labels):
        print(f"ERROR: {args.labels} not found. Run generate_labels.py first.")
        sys.exit(1)

    cfg = load_config(args.config)
    records, meta = load_labels(args.labels)
    limit = args.limit or cfg["datasets"].get("eval_limit", len(records))
    records = records[:limit]

    # assign a few high/low priorities so the priority boost path is exercised
    for i, rec in enumerate(records):
        if i % 8 == 0:
            rec.priority = "high"
        elif i % 5 == 0:
            rec.priority = "low"

    prod_m = load_prod_m(args.prod_m, device=args.device) if os.path.exists(args.prod_m) else None
    ranker = load_ranker(args.ranker, device=args.device) if os.path.exists(args.ranker) else None

    hidden = None
    hidden_path = meta.get("hidden_states_path", "data/processed/prod_hidden.pt")
    if os.path.exists(hidden_path):
        hidden = load_hidden(hidden_path)[: len(records)]

    print("--- Prediction quality ---")
    if prod_m is not None and hidden is not None:
        preds = prod_m.predict_lengths(hidden.to(args.device))
        true = [r.output_length for r in records]
        print(f"ProD-M MAE vs median: {mae(true, preds):.2f} tokens")
    else:
        print("ProD-M checkpoint / hidden states missing — skip MAE")

    if ranker is not None:
        lengths = [r.output_length for r in records]
        scores = ranker.score([r.text for r in records])
        # lower score first ~= shorter job first; compare that order to sorted lengths
        order = [lengths[i] for i in sorted(range(len(scores)), key=lambda k: scores[k])]
        print(f"PARS Kendall Tau:       {kendall_tau(order, sorted(lengths)):.3f}")
        print(f"PARS Pairwise Accuracy: {pairwise_accuracy(scores, lengths):.3f}")
        print(f"PARS NDCG:              {ndcg_at_k(scores, lengths):.3f}")
    else:
        print("PARS checkpoint missing — skip ranking metrics")

    boosts = {
        "high": cfg["priority"]["high_boost"],
        "normal": cfg["priority"]["normal_boost"],
        "low": cfg["priority"]["low_boost"],
    }
    sim_cfg = SimConfig(
        batch_size=cfg["scheduler"]["batch_size"],
        arrival_rate=cfg["simulation"]["arrival_rate"],
        seed=cfg["simulation"]["seed"],
        boosts=boosts,
    )

    print("\n--- Scheduler comparison ---")
    summaries = compare(
        records,
        ["fcfs", "prod_m", "pars", "oracle"],
        sim_cfg,
        ranker=ranker,
        prod_m=prod_m,
        hidden=hidden,
        device=args.device,
    )
    for s in summaries:
        print_summary(s)

    # quick relative improvement note for the report
    by_name = {s.policy: s for s in summaries}
    if "fcfs" in by_name and "pars" in by_name and by_name["fcfs"].p95_latency > 0:
        gain = (by_name["fcfs"].p95_latency - by_name["pars"].p95_latency) / by_name["fcfs"].p95_latency
        print(f"\nPARS vs FCFS p95 improvement: {100 * gain:.1f}%")


if __name__ == "__main__":
    main()
