#!/usr/bin/env python3
"""
End-to-end live pipeline at high prompt count:

  1) chunked ProD-M labeling
  2) train LTR (single-sample) + PARS (median)
  3) live vLLM FCFS | LTR | OURS comparison

Example (Colab / cloud GPU):
  python scripts/run_live.py --limit 1000 --chunk-size 50 --num-samples 3 \\
      --backup-dir /content/drive/MyDrive/capstone_results --device cuda
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import get_hf_token


def run(cmd):
    print("\n>> " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/live_run.yaml")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backup-dir", default="")
    parser.add_argument("--labels", default="data/processed/prod_labels.json")
    parser.add_argument("--ltr", default="checkpoints/ltr_pointwise.pt")
    parser.add_argument("--ranker", default="checkpoints/pairwise_ranker.pt")
    parser.add_argument("--skip-label", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-live", action="store_true")
    parser.add_argument(
        "--live-output",
        default="data/processed/live_eval_results.json",
    )
    args = parser.parse_args()

    if not get_hf_token() and not args.skip_label:
        print("ERROR: set HF_TOKEN before labeling / live Llama serving")
        sys.exit(1)

    py = sys.executable

    if not args.skip_label:
        cmd = [
            py, "scripts/generate_labels.py",
            "--config", args.config,
            "--limit", str(args.limit),
            "--chunk-size", str(args.chunk_size),
            "--num-samples", str(args.num_samples),
            "--resume",
            "--device", args.device,
            "--output", args.labels,
        ]
        if args.backup_dir:
            cmd += ["--backup-dir", args.backup_dir]
        run(cmd)

    if not args.skip_train:
        run([
            py, "scripts/train_prod_m.py",
            "--config", args.config,
            "--labels", args.labels,
            "--target", "single",
            "--output", args.ltr,
            "--device", args.device,
        ])
        run([
            py, "scripts/train_ranker.py",
            "--config", args.config,
            "--labels", args.labels,
            "--train-samples", str(args.limit),
            "--output", args.ranker,
            "--device", args.device,
        ])

    if not args.skip_live:
        run([
            py, "scripts/evaluate_live.py",
            "--config", args.config,
            "--labels", args.labels,
            "--ltr", args.ltr,
            "--ranker", args.ranker,
            "--limit", str(args.limit),
            "--device", args.device,
            "--output", args.live_output,
        ])


if __name__ == "__main__":
    main()
