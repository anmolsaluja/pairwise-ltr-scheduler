"""
Report-ready plots styled after the main LTR paper figures:

  Fig-style A: input / output length distributions
  Fig-style B: latency vs request rate (FCFS / LTR / OURS)
  Plus: bar charts for avg/p95 latency, throughput, % gains
"""

from __future__ import annotations

import os
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np


# Consistent colors across all figures (report-friendly, not purple-default)
COLORS = {
    "fcfs": "#4C566A",
    "ltr": "#D08770",
    "pars": "#2E6F6A",
}
LABELS = {
    "fcfs": "FCFS (baseline)",
    "ltr": "LTR (main paper)",
    "pars": "PARS+ProD-M+Priority (ours)",
}


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_fig(fig, out_dir, name):
    _ensure_dir(out_dir)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {path}")
    return path


def plot_length_distributions(input_lens, output_lens, out_dir, title_suffix=""):
    """Main-paper Fig. 2 style: input vs output length histograms."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].hist(input_lens, bins=30, color="#5E81AC", edgecolor="white", alpha=0.9)
    axes[0].set_title("Input length distribution")
    axes[0].set_xlabel("Tokens")
    axes[0].set_ylabel("Count")
    axes[0].axvline(np.mean(input_lens), color="#BF616A", linestyle="--", label=f"mean={np.mean(input_lens):.0f}")
    axes[0].legend(frameon=False)

    axes[1].hist(output_lens, bins=30, color="#A3BE8C", edgecolor="white", alpha=0.9)
    axes[1].set_title("Output length distribution")
    axes[1].set_xlabel("Tokens")
    axes[1].set_ylabel("Count")
    axes[1].axvline(np.mean(output_lens), color="#BF616A", linestyle="--", label=f"mean={np.mean(output_lens):.0f}")
    axes[1].legend(frameon=False)

    fig.suptitle(f"Prompt length distributions{title_suffix}", y=1.02, fontsize=12)
    fig.tight_layout()
    return save_fig(fig, out_dir, "fig_length_distributions.png")


def plot_latency_vs_rate(rate_rows: List[dict], out_dir, metric="avg_latency"):
    """
    Main-paper Fig. 3 style: latency vs request rate.
    rate_rows: list of {rate, fcfs, ltr, pars} latency values.
    """
    rates = [r["rate"] for r in rate_rows]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for key in ("fcfs", "ltr", "pars"):
        ys = [r[key] for r in rate_rows]
        ax.plot(
            rates,
            ys,
            marker="o",
            linewidth=2,
            markersize=5,
            color=COLORS[key],
            label=LABELS[key],
        )
    ax.set_xlabel("Request rate (req/s)")
    ylabel = "Average latency (s)" if metric == "avg_latency" else "p95 latency (s)"
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs request rate")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    name = "fig_latency_vs_rate.png" if metric == "avg_latency" else "fig_p95_vs_rate.png"
    return save_fig(fig, out_dir, name)


def plot_policy_bars(summaries: Dict[str, dict], out_dir):
    """Grouped bars: avg / p95 latency and throughput for 3 policies."""
    order = [k for k in ("fcfs", "ltr", "pars") if k in summaries]
    names = [LABELS[k] for k in order]
    x = np.arange(len(order))
    width = 0.36

    # Latency bars
    fig, ax = plt.subplots(figsize=(8, 4.2))
    avg = [summaries[k]["avg_latency"] for k in order]
    p95 = [summaries[k]["p95_latency"] for k in order]
    ax.bar(x - width / 2, avg, width, label="Avg latency", color="#5E81AC")
    ax.bar(x + width / 2, p95, width, label="p95 latency", color="#BF616A")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Latency (s)")
    ax.set_title("Scheduler latency comparison")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    p1 = save_fig(fig, out_dir, "fig_latency_bars.png")

    # Throughput bars
    fig, ax = plt.subplots(figsize=(7, 4))
    tput = [summaries[k]["throughput_rps"] for k in order]
    ax.bar(x, tput, color=[COLORS[k] for k in order], width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Throughput (req/s)")
    ax.set_title("Scheduler throughput comparison")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    p2 = save_fig(fig, out_dir, "fig_throughput_bars.png")
    return p1, p2


def plot_improvement_bars(gains: Dict[str, float], out_dir):
    """Percent improvement bars (positive = better / lower latency)."""
    keys = list(gains.keys())
    vals = [gains[k] for k in keys]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    colors = ["#D08770" if v >= 0 else "#BF616A" for v in vals]
    ax.barh(keys, vals, color=colors)
    ax.axvline(0, color="#2E3440", linewidth=0.8)
    ax.set_xlabel("p95 latency improvement (%)")
    ax.set_title("Relative improvements")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    return save_fig(fig, out_dir, "fig_improvements.png")


def plot_ranking_quality(metrics: dict, out_dir):
    """Optional ranking-quality bars for PARS (Kendall / pairwise / NDCG)."""
    keys = [k for k in ("kendall", "pairwise_acc", "ndcg") if k in metrics and metrics[k] is not None]
    if not keys:
        return None
    labels = {"kendall": "Kendall Tau", "pairwise_acc": "Pairwise Acc", "ndcg": "NDCG"}
    fig, ax = plt.subplots(figsize=(6, 3.8))
    vals = [metrics[k] for k in keys]
    ax.bar([labels[k] for k in keys], vals, color="#2E6F6A", width=0.55)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("PARS ranking quality (ProD-M median labels)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return save_fig(fig, out_dir, "fig_ranking_quality.png")


def print_results_table(summaries: Dict[str, dict], gains: Dict[str, float]):
    print("\n" + "=" * 72)
    print(" RESULTS TABLE (for report)")
    print("=" * 72)
    print(f"{'Policy':<32} {'Avg lat':>10} {'p95':>10} {'Throughput':>12}")
    print("-" * 72)
    for key in ("fcfs", "ltr", "pars"):
        if key not in summaries:
            continue
        s = summaries[key]
        print(
            f"{LABELS[key]:<32} "
            f"{s['avg_latency']:10.3f} "
            f"{s['p95_latency']:10.3f} "
            f"{s['throughput_rps']:10.2f} r/s"
        )
    print("-" * 72)
    print("p95 improvements:")
    for k, v in gains.items():
        print(f"  {k}: {v:.1f}%")
    print("=" * 72)
