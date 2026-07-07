#!/usr/bin/env python3
"""
Compare FCFS vs pointwise LTR vs pairwise LTR on simulated traffic.

Example:
  python scripts/run_simulation.py --policy pairwise_ltr
  python scripts/run_simulation.py --compare-all --checkpoint checkpoints/pairwise_ranker.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_alpaca_prompts
from src.metrics import kendall_tau
from src.pairwise_predictor import load_model
from src.simulator import SimConfig, compare_policies, run_simulation


def print_summary(summary):
    print(f"\n=== {summary.policy.upper()} ===")
    print(f"  Requests:     {summary.num_requests}")
    print(f"  Avg latency:  {summary.avg_latency:.3f}s")
    print(f"  P50 latency:  {summary.p50_latency:.3f}s")
    print(f"  P95 latency:  {summary.p95_latency:.3f}s")
    print(f"  P99 latency:  {summary.p99_latency:.3f}s")
    print(f"  Avg wait:     {summary.avg_wait:.3f}s")
    print(f"  Throughput:   {summary.throughput_rps:.2f} req/s")


def main():
    parser = argparse.ArgumentParser(description="Run scheduling simulation")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--policy", default=None)
    parser.add_argument("--compare-all", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--num-requests", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local-data", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    num_req = args.num_requests or cfg["simulation"]["num_requests"]
    records = load_alpaca_prompts(split="train", limit=num_req, use_local=args.local_data)

    # Mix in some priority prompts for demo
    for i, rec in enumerate(records):
        if i % 10 == 0:
            rec.priority = "high"
        elif i % 7 == 0:
            rec.priority = "low"

    ranker = None
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading ranker from {args.checkpoint}")
        ranker = load_model(args.checkpoint, device=args.device)

    base_config = SimConfig(
        policy=cfg["scheduler"]["policy"],
        batch_size=cfg["scheduler"]["batch_size"],
        arrival_rate=cfg["simulation"]["arrival_rate"],
        seed=cfg["simulation"]["seed"],
        priority_boosts=cfg["priority"],
    )

    if args.compare_all:
        policies = ["fcfs", "ltr_pointwise", "pairwise_ltr"]
        summaries = compare_policies(records, ranker, policies, base_config, device=args.device)
        for s in summaries:
            print_summary(s)

        # Ranking quality check
        if ranker:
            lengths = [r.output_length for r in records]
            true_order = sorted(lengths)
            scores = ranker.score_prompts([r.text for r in records])
            pred_order = [lengths[i] for i in sorted(range(len(scores)), key=lambda k: scores[k])]
            tau = kendall_tau(pred_order, true_order)
            print(f"\nKendall Tau (ranking quality): {tau:.3f}")
        return

    policy = args.policy or cfg["scheduler"]["policy"]
    config = SimConfig(
        policy=policy,
        batch_size=base_config.batch_size,
        arrival_rate=base_config.arrival_rate,
        seed=base_config.seed,
        priority_boosts=base_config.priority_boosts,
    )

    _, summary = run_simulation(records, config, ranker=ranker, device=args.device)
    print_summary(summary)


if __name__ == "__main__":
    main()
