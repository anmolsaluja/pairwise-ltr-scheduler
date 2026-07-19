"""
Schedulers compared in this project:

  fcfs      - first come first serve (baseline, suffers from HOL blocking)
  prod_m    - sort by ProD-M predicted length (pointwise LTR / SJF)
  pars      - sort by pairwise ranker score (our proposal extension)
  oracle    - sort by true output length (upper bound for SJF)

Priority boosts and a simple starvation rule are applied on top
(PARS paper ~2 min wait threshold).
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
        self.policy = policy
        self.batch_size = batch_size
        self.starvation_sec = starvation_sec
        self.boosts = boosts or {"high": -3.0, "normal": 0.0, "low": 3.0}
        self.waiting = []

    def add(self, req: Request):
        self.waiting.append(req)

    def _maybe_promote(self, now):
        # if something waits too long, bump it to high priority
        for req in self.waiting:
            if now - req.arrival_time >= self.starvation_sec:
                req.priority = "high"

    def next_batch(self, now=0.0, n=None):
        """Pick up to n requests (default = batch_size)."""
        self._maybe_promote(now)
        n = self.batch_size if n is None else max(0, n)
        if n == 0 or not self.waiting:
            return []

        if self.policy == "fcfs":
            batch = self.waiting[:n]
            self.waiting = self.waiting[n:]
            return batch

        # prod_m / pars / oracle: lowest effective score first (approx SJF)
        heap = []
        for req in self.waiting:
            heapq.heappush(heap, _Item(key=req.effective_score(self.boosts), req=req))

        batch = []
        for _ in range(min(n, len(heap))):
            batch.append(heapq.heappop(heap).req)

        picked = {r.request_id for r in batch}
        self.waiting = [r for r in self.waiting if r.request_id not in picked]
        return batch
