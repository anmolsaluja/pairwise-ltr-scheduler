"""Latency / ranking metrics used in the evaluation."""

from __future__ import annotations

import statistics
from dataclasses import dataclass


@dataclass
class RequestMetrics:
    request_id: str
    wait_time: float
    service_time: float
    total_latency: float
    output_length: int
    priority: str


@dataclass
class RunSummary:
    policy: str
    num_requests: int
    avg_latency: float
    p50_latency: float
    p95_latency: float
    p99_latency: float
    avg_wait: float
    throughput_rps: float


def percentile(values, p):
    if not values:
        return 0.0
    vals = sorted(values)
    i = min(int(len(vals) * p / 100), len(vals) - 1)
    return vals[i]


def summarize(policy, rows, total_time):
    lats = [r.total_latency for r in rows]
    waits = [r.wait_time for r in rows]
    return RunSummary(
        policy=policy,
        num_requests=len(rows),
        avg_latency=statistics.mean(lats) if lats else 0.0,
        p50_latency=percentile(lats, 50),
        p95_latency=percentile(lats, 95),
        p99_latency=percentile(lats, 99),
        avg_wait=statistics.mean(waits) if waits else 0.0,
        throughput_rps=(len(rows) / total_time) if total_time > 0 else 0.0,
    )


def mae(true_vals, pred_vals):
    if not true_vals:
        return 0.0
    return sum(abs(t - p) for t, p in zip(true_vals, pred_vals)) / len(true_vals)


def kendall_tau(pred_order, true_order):
    """
    Ranking quality (used in the LTR papers).
    Both lists should be lengths ordered by the ranking method / ground truth.
    """
    n = len(pred_order)
    if n < 2:
        return 0.0

    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            ps = pred_order[i] - pred_order[j]
            ts = true_order[i] - true_order[j]
            if ps == 0 or ts == 0:
                continue
            if ps * ts > 0:
                conc += 1
            else:
                disc += 1
    total = conc + disc
    return 0.0 if total == 0 else (conc - disc) / total
