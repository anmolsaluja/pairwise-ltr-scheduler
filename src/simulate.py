"""
Simple discrete-event simulator.

We don't need a real GPU here — each token just costs a fixed amount
of simulated time. That's enough to show HOL blocking under FCFS vs
length-aware scheduling.
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


def _score_requests(records, ranker=None, prod_m=None, hidden=None, device="cpu"):
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

    if ranker is not None:
        ranker.to(device)
        scores = ranker.score([r.prompt for r in reqs])
        for req, s in zip(reqs, scores):
            req.rank_score = float(s)

    if prod_m is not None and hidden is not None:
        prod_m.to(device)
        lengths = prod_m.predict_lengths(hidden.to(device))
        for req, length in zip(reqs, lengths):
            req.rank_score = float(length)
            req.predicted_length = int(round(length))

    return reqs


def run_sim(records, config, ranker=None, prod_m=None, hidden=None, device="cpu"):
    sched = Scheduler(
        policy=config.policy,
        batch_size=config.batch_size,
        boosts=config.boosts,
    )

    all_reqs = _score_requests(records, ranker, prod_m, hidden, device)
    by_id = {r.request_id: r for r in all_reqs}

    arrivals = list(poisson_arrivals(records, config.arrival_rate, config.seed))
    idx = 0
    clock = 0.0
    active = []  # (finish_time, request)
    done = []

    while idx < len(arrivals) or active or sched.waiting:
        # admit arrivals
        while idx < len(arrivals) and arrivals[idx][0] <= clock:
            t, rec = arrivals[idx]
            req = by_id[rec.prompt_id]
            req.arrival_time = t
            sched.add(req)
            idx += 1

        # finish completed jobs
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

        # start new work if we have free slots
        free = config.batch_size - len(active)
        if free > 0 and sched.waiting:
            for req in sched.next_batch(now=clock)[:free]:
                active.append((clock + _service_time(req), req))

        next_arr = arrivals[idx][0] if idx < len(arrivals) else float("inf")
        next_fin = min((t for t, _ in active), default=float("inf"))
        nxt = min(next_arr, next_fin)
        if nxt == float("inf"):
            break
        clock = nxt

    return summarize(config.policy, done, clock)


def compare(records, policies, config, ranker=None, prod_m=None, hidden=None, device="cpu"):
    results = []
    for policy in policies:
        cfg = SimConfig(
            policy=policy,
            batch_size=config.batch_size,
            arrival_rate=config.arrival_rate,
            seed=config.seed,
            boosts=config.boosts,
        )
        use_pars = policy in ("pars", "prod_m_pars", "pairwise_ltr")
        use_prod = policy in ("prod_m", "ltr")
        results.append(
            run_sim(
                records,
                cfg,
                ranker=ranker if use_pars else None,
                prod_m=prod_m if use_prod else None,
                hidden=hidden if use_prod else None,
                device=device,
            )
        )
    return results
