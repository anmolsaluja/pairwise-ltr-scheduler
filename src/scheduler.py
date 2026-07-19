"""
Schedulers:

  fcfs  - baseline
  ltr   - main-paper pointwise LTR
  pars  - ours (PARS ranking + priority + starvation; trained with ProD-M medians)
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

from src.requests import Request


@dataclass(order=True)
class _Item:
    key: float
    req: Request = field(compare=False)


class Scheduler:
    def __init__(self, policy="fcfs", batch_size=8, starvation_sec=120.0, boosts=None):
        if policy in ("pairwise_ltr", "prod_m_pars"):
            policy = "pars"
        if policy in ("ltr_pointwise", "main_ltr"):
            policy = "ltr"

        self.policy = policy
        self.batch_size = batch_size
        self.starvation_sec = starvation_sec
        self.boosts = boosts or {"high": -3.0, "normal": 0.0, "low": 3.0}
        self.waiting = []

    def add(self, req: Request):
        self.waiting.append(req)

    def _maybe_promote(self, now):
        # fairness (PARS-style): long wait -> treat as high priority
        for req in self.waiting:
            if now - req.arrival_time >= self.starvation_sec:
                req.priority = "high"

    def next_batch(self, now=0.0, n=None):
        self._maybe_promote(now)
        n = self.batch_size if n is None else max(0, n)
        if n == 0 or not self.waiting:
            return []

        if self.policy == "fcfs":
            batch = self.waiting[:n]
            self.waiting = self.waiting[n:]
            return batch

        # length-aware policies: lowest effective score first
        heap = []
        for req in self.waiting:
            heapq.heappush(heap, _Item(key=req.effective_score(self.boosts), req=req))

        batch = []
        for _ in range(min(n, len(heap))):
            batch.append(heapq.heappop(heap).req)

        picked = {r.request_id for r in batch}
        self.waiting = [r for r in self.waiting if r.request_id not in picked]
        return batch
