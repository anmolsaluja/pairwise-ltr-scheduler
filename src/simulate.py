"""
Discrete-event simulator.

Main comparison (three methods):
  fcfs  - baseline
  ltr   - MAIN PAPER pointwise LTR (single-sample labels)
  pars  - OURS: PARS pairwise + ProD-M median training + priority

(prod_m / oracle helpers remain available but are not used in evaluate.py)
"""

from __future__ import annotations

from dataclasses import dataclass

from src.data import poisson_arrivals
from src.metrics import RequestMetrics, summarize
from src.requests import Request, parse_priority
from src.scheduler import Scheduler

PREFILL = 0.05
PER_TOKEN = 0.01


@dataclass
class SimConfig:
    policy: str = "pars"
    batch_size: int = 8
    arrival_rate: float = 8.0
    seed: int = 42
    boosts: dict = None


def _service_time(req):
    return PREFILL + req.output_length * PER_TOKEN


def _normalize_policy(policy):
    if policy in ("pairwise_ltr", "prod_m_pars"):
        return "pars"
    if policy in ("ltr_pointwise", "main_ltr"):
        return "ltr"
    return policy


def _score_requests(
    records,
    policy="fcfs",
    ranker=None,
    ltr_model=None,
    prod_m=None,
    hidden=None,
    device="cpu",
):
    """Fill rank_score used by the length-aware scheduler."""
    policy = _normalize_policy(policy)
    reqs = []
    for rec in records:
        reqs.append(
            Request(
                request_id=rec.prompt_id,
                prompt=rec.text,
                output_length=rec.output_length,
                priority=parse_priority(rec.priority),
            )
        )

    if policy == "oracle":
        for req in reqs:
            req.rank_score = float(req.output_length)
            req.predicted_length = req.output_length
        return reqs

    # Main-paper style LTR: pointwise model trained on single-sample lengths
    if policy == "ltr" and ltr_model is not None and hidden is not None:
        ltr_model.to(device)
        lengths = ltr_model.predict_lengths(hidden.to(device))
        for req, length in zip(reqs, lengths):
            req.rank_score = float(length)
            req.predicted_length = int(round(length))
        return reqs

    # Our ProD-M pointwise (median-trained) — separate from the main paper
    if policy == "prod_m" and prod_m is not None and hidden is not None:
        prod_m.to(device)
        lengths = prod_m.predict_lengths(hidden.to(device))
        for req, length in zip(reqs, lengths):
            req.rank_score = float(length)
            req.predicted_length = int(round(length))
        return reqs

    # Our PARS pairwise method
    if policy == "pars" and ranker is not None:
        ranker.to(device)
        scores = ranker.score([r.prompt for r in reqs])
        for req, s in zip(reqs, scores):
            req.rank_score = float(s)
        return reqs

    return reqs


def run_sim(
    records,
    config,
    ranker=None,
    ltr_model=None,
    prod_m=None,
    hidden=None,
    device="cpu",
):
    policy = _normalize_policy(config.policy)
    sched_policy = "fcfs" if policy == "fcfs" else "ltr"
    sched = Scheduler(
        policy=sched_policy,
        batch_size=config.batch_size,
        boosts=config.boosts,
    )

    all_reqs = _score_requests(
        records,
        policy=policy,
        ranker=ranker,
        ltr_model=ltr_model,
        prod_m=prod_m,
        hidden=hidden,
        device=device,
    )
    by_id = {r.request_id: r for r in all_reqs}

    arrivals = list(poisson_arrivals(records, config.arrival_rate, config.seed))
    idx = 0
    clock = 0.0
    active = []
    done = []

    while idx < len(arrivals) or active or sched.waiting:
        while idx < len(arrivals) and arrivals[idx][0] <= clock:
            t, rec = arrivals[idx]
            req = by_id[rec.prompt_id]
            req.arrival_time = t
            sched.add(req)
            idx += 1

        still = []
        for finish, req in active:
            if finish <= clock:
                service = _service_time(req)
                done.append(
                    RequestMetrics(
                        request_id=req.request_id,
                        wait_time=max(0.0, clock - req.arrival_time - service),
                        service_time=service,
                        total_latency=clock - req.arrival_time,
                        output_length=req.output_length,
                        priority=req.priority,
                    )
                )
            else:
                still.append((finish, req))
        active = still

        free = config.batch_size - len(active)
        if free > 0 and sched.waiting:
            for req in sched.next_batch(now=clock, n=free):
                active.append((clock + _service_time(req), req))

        next_arr = arrivals[idx][0] if idx < len(arrivals) else float("inf")
        next_fin = min((t for t, _ in active), default=float("inf"))
        nxt = min(next_arr, next_fin)
        if nxt == float("inf"):
            break
        clock = nxt

    return summarize(policy, done, clock)


def compare(
    records,
    policies,
    config,
    ranker=None,
    ltr_model=None,
    prod_m=None,
    hidden=None,
    device="cpu",
):
    results = []
    for policy in policies:
        policy = _normalize_policy(policy)
        cfg = SimConfig(
            policy=policy,
            batch_size=config.batch_size,
            arrival_rate=config.arrival_rate,
            seed=config.seed,
            boosts=config.boosts,
        )
        results.append(
            run_sim(
                records,
                cfg,
                ranker=ranker if policy == "pars" else None,
                ltr_model=ltr_model if policy == "ltr" else None,
                prod_m=prod_m if policy == "prod_m" else None,
                hidden=hidden if policy in ("ltr", "prod_m") else None,
                device=device,
            )
        )
    return results
