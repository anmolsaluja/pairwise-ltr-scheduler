#!/usr/bin/env python3
"""
Step 4: compare schedulers the way the proposal describes.

  FCFS   - baseline
  LTR    - MAIN PAPER style (pointwise, trained on single-sample lengths)
  ProD-M - OUR robust pointwise predictor (median labels) — not in main paper
  PARS   - OUR pairwise ranking + priority + starvation
  Oracle - upper bound (true median lengths)
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

POLICY_TITLE = {
    "fcfs": "FCFS (baseline)",
    "ltr": "LTR pointwise (MAIN PAPER style, single-sample labels)",
    "prod_m": "ProD-M pointwise (OURS: median labels — not in main paper)",
    "pars": "PARS pairwise (OURS: ranking + priority)",
    "oracle": "Oracle SJF (true median length, upper bound)",
}


def print_summary(s):
    title = POLICY_TITLE.get(s.policy, s.policy.upper())
    print(f"\n=== {title} ===")
    print(f"  policy id:   {s.policy}")
    print(f"  requests:    {s.num_requests}")
    print(f"  avg latency: {s.avg_latency:.3f}s")
    print(f"  p50:         {s.p50_latency:.3f}s")
    print(f"  p95:         {s.p95_latency:.3f}s")
    print(f"  p99:         {s.p99_latency:.3f}s")
    print(f"  avg wait:    {s.avg_wait:.3f}s")
    print(f"  avg TTFT:    {s.avg_ttft:.3f}s")
    print(f"  throughput:  {s.throughput_rps:.2f} req/s")


def _pct_gain(base, other):
    if base <= 0:
        return None
    return 100.0 * (base - other) / base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--labels", default="data/processed/prod_labels.json")
    parser.add_argument("--ltr", default="checkpoints/ltr_pointwise.pt",
                        help="main-paper style pointwise model (single-sample)")
    parser.add_argument("--prod-m", default="checkpoints/prod_m.pt",
                        help="our ProD-M model (median)")
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

    n_high = n_low = 0
    for i, rec in enumerate(records):
        if i % 8 == 0:
            rec.priority = "high"
            n_high += 1
        elif i % 5 == 0:
            rec.priority = "low"
            n_low += 1
    print(
        f"Priority mix: high={n_high}, low={n_low}, "
        f"normal={len(records) - n_high - n_low}"
    )

    ltr_model = load_prod_m(args.ltr, device=args.device) if os.path.exists(args.ltr) else None
    prod_m = load_prod_m(args.prod_m, device=args.device) if os.path.exists(args.prod_m) else None
    ranker = load_ranker(args.ranker, device=args.device) if os.path.exists(args.ranker) else None

    hidden = None
    hidden_path = meta.get("hidden_states_path", "data/processed/prod_hidden.pt")
    if os.path.exists(hidden_path):
        hidden = load_hidden(hidden_path)[: len(records)]

    true = [r.output_length for r in records]

    print("\n--- 1) Prediction / ranking quality ---")
    print("Note: ProD-M is OUR addition (ProD paper). It is NOT in the main LTR paper.")
    if ltr_model is not None and hidden is not None:
        print(f"LTR  (single-sample train) MAE vs median: "
              f"{mae(true, ltr_model.predict_lengths(hidden.to(args.device))):.2f}")
    else:
        print("LTR checkpoint missing — run: python scripts/train_prod_m.py --target single "
              "--output checkpoints/ltr_pointwise.pt")

    if prod_m is not None and hidden is not None:
        print(f"ProD-M (median train)      MAE vs median: "
              f"{mae(true, prod_m.predict_lengths(hidden.to(args.device))):.2f}")
    else:
        print("ProD-M checkpoint missing")

    if ranker is not None:
        lengths = true
        scores = ranker.score([r.text for r in records])
        order = [lengths[i] for i in sorted(range(len(scores)), key=lambda k: scores[k])]
        print(f"PARS Kendall Tau:       {kendall_tau(order, sorted(lengths)):.3f}")
        print(f"PARS Pairwise Accuracy: {pairwise_accuracy(scores, lengths):.3f}")
        print(f"PARS NDCG:              {ndcg_at_k(scores, lengths):.3f}")
    else:
        print("PARS checkpoint missing")

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

    print("\n--- 2) Scheduler comparison ---")
    print("FCFS  = baseline")
    print("LTR   = main paper (pointwise, single-sample labels)")
    print("ProD-M = our robust pointwise (median labels)")
    print("PARS  = our pairwise improvement (+ priority)")
    print("Oracle = upper bound")

    summaries = compare(
        records,
        ["fcfs", "ltr", "prod_m", "pars", "oracle"],
        sim_cfg,
        ranker=ranker,
        ltr_model=ltr_model,
        prod_m=prod_m,
        hidden=hidden,
        device=args.device,
    )
    for s in summaries:
        print_summary(s)

    by_name = {s.policy: s for s in summaries}

    def show(label, a, b):
        if a in by_name and b in by_name:
            g = _pct_gain(by_name[a].p95_latency, by_name[b].p95_latency)
            if g is not None:
                print(f"{label}: {g:.1f}%")

    print("\n--- p95 latency improvements ---")
    show("LTR vs FCFS", "fcfs", "ltr")
    show("ProD-M vs LTR (median labels help pointwise)", "ltr", "prod_m")
    show("PARS vs LTR (our pairwise vs main paper)", "ltr", "pars")
    show("PARS vs ProD-M", "prod_m", "pars")
    show("PARS vs FCFS", "fcfs", "pars")


if __name__ == "__main__":
    main()
