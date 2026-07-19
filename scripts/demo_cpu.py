#!/usr/bin/env python3
"""
Tiny CPU-only demo of the scheduler (no GPU / no HF token needed).

Useful to sanity-check FCFS HOL blocking vs SJF before the full cloud run.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import PromptRecord
from src.simulate import SimConfig, compare


def main():
    # fake workload: mix of short and long jobs
    records = []
    lengths = [20, 20, 400, 25, 30, 350, 15, 40, 300, 22]
    for i, n in enumerate(lengths):
        records.append(
            PromptRecord(
                prompt_id=f"demo_{i}",
                text=f"prompt {i}",
                output_length=n,
                priority="high" if i == 3 else "normal",
            )
        )

    cfg = SimConfig(batch_size=2, arrival_rate=6.0, seed=0, boosts={
        "high": -3.0, "normal": 0.0, "low": 3.0,
    })

    print("CPU demo — synthetic lengths:", lengths)
    print("(oracle = perfect SJF using true lengths)\n")
    for s in compare(records, ["fcfs", "oracle"], cfg):
        print(
            f"{s.policy:8s}  avg={s.avg_latency:.2f}s  p95={s.p95_latency:.2f}s  "
            f"wait={s.avg_wait:.2f}s  ttft={s.avg_ttft:.2f}s"
        )
    print("\nOracle should beat FCFS on avg wait / p95 when lengths are skewed.")
    print("For real PARS / ProD-M numbers, run the GPU pipeline on Colab.")


if __name__ == "__main__":
    main()
