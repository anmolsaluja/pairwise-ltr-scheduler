#!/usr/bin/env python3
"""
Run the full project pipeline end-to-end.

  1) generate median labels with Llama
  2) train ProD-M
  3) train pairwise ranker
  4) compare FCFS / ProD-M / PARS

Example (Colab):
  export HF_TOKEN=hf_...
  python scripts/run_all.py --limit 50 --device cuda
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import get_hf_token, load_config, resolve_llm


def run(cmd):
    print("\n>> " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--llm-profile", default=None)
    parser.add_argument("--skip-train", action="store_true",
                        help="only run evaluate.py (needs existing checkpoints)")
    parser.add_argument("--labels", default="data/processed/prod_labels.json")
    parser.add_argument("--prod-m", default="checkpoints/prod_m.pt")
    parser.add_argument("--ranker", default="checkpoints/pairwise_ranker.pt")
    args = parser.parse_args()

    if not get_hf_token() and not args.skip_train:
        print("ERROR: set HF_TOKEN before running")
        sys.exit(1)

    cfg = load_config(args.config)
    llm = resolve_llm(cfg, args.llm_profile)
    dataset = args.dataset or cfg["datasets"]["name"]
    limit = args.limit or cfg["datasets"]["limit"]

    print(f"Using {llm['profile']} -> {llm['model']}")
    print(f"Dataset={dataset}, limit={limit}, device={args.device}")

    py = sys.executable

    if not args.skip_train:
        run([
            py, "scripts/generate_labels.py",
            "--dataset", dataset,
            "--limit", str(limit),
            "--output", args.labels,
            "--device", args.device,
            "--llm-profile", llm["profile"],
        ])
        run([
            py, "scripts/train_prod_m.py",
            "--labels", args.labels,
            "--output", args.prod_m,
            "--device", args.device,
        ])
        run([
            py, "scripts/train_ranker.py",
            "--labels", args.labels,
            "--train-samples", str(limit),
            "--output", args.ranker,
            "--device", args.device,
        ])

    run([
        py, "scripts/evaluate.py",
        "--labels", args.labels,
        "--prod-m", args.prod_m,
        "--ranker", args.ranker,
        "--device", args.device,
        "--limit", str(limit),
    ])


if __name__ == "__main__":
    main()
