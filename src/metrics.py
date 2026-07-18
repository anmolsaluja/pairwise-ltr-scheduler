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


def pairwise_accuracy(pred_scores, true_lengths):
    """
    Fraction of prompt pairs where the predicted score correctly orders
    which one has the longer true output. This is the direct "pairwise
    ranking accuracy" metric from the proposal (distinct from Kendall's
    Tau, which is a correlation rather than a plain accuracy count).
    """
    n = len(pred_scores)
    if n < 2:
        return 0.0

    correct = total = 0
    for i in range(n):
        for j in range(i + 1, n):
            true_diff = true_lengths[i] - true_lengths[j]
            if true_diff == 0:
                continue  # tie in ground truth, not a valid comparison
            pred_diff = pred_scores[i] - pred_scores[j]
            if pred_diff == 0:
                continue  # model didn't express a preference
            total += 1
            if (true_diff > 0) == (pred_diff > 0):
                correct += 1
    return 0.0 if total == 0 else correct / total


def ndcg_at_k(pred_scores, true_lengths, k=None):
    """
    NDCG over the ranking induced by pred_scores, using true output
    length as relevance. Longer outputs are treated as "more relevant"
    to keep this consistent with SJF-style evaluation (we care whether
    the model correctly identifies the longest/shortest jobs near the
    top of its ranking, not just an aggregate correlation).

    k defaults to the full list length (i.e. standard NDCG, not NDCG@k).
    """
    n = len(pred_scores)
    if n == 0:
        return 0.0
    k = k or n

    # DCG using the model's predicted ordering (descending score = "most confident longest")
    order = sorted(range(n), key=lambda i: pred_scores[i], reverse=True)
    dcg = 0.0
    for rank, idx in enumerate(order[:k], start=1):
        rel = true_lengths[idx]
        dcg += rel / _log2(rank + 1)

    # Ideal DCG using the best possible ordering (true lengths, descending)
    ideal_order = sorted(range(n), key=lambda i: true_lengths[i], reverse=True)
    idcg = 0.0
    for rank, idx in enumerate(ideal_order[:k], start=1):
        rel = true_lengths[idx]
        idcg += rel / _log2(rank + 1)

    return 0.0 if idcg == 0 else dcg / idcg


def _log2(x):
    import math

    return math.log2(x)
