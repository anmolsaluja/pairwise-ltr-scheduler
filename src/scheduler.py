"""
Schedulers: FCFS, pointwise LTR, and pairwise LTR (our proposal).

The pairwise scheduler sorts by predicted length + user priority.
Starvation prevention boosts requests that wait too long (from PARS).
"""

from __future__ import annotations

import heapq
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.priority import InferenceRequest, PriorityLevel


@dataclass(order=True)
class QueueItem:
    """Heap entry: sort key first, then request."""

    sort_key: float
    request: InferenceRequest = field(compare=False)


class BaseScheduler(ABC):
    """Common interface for all scheduling policies."""

    def __init__(self, batch_size: int = 8, starvation_seconds: float = 120.0):
        self.batch_size = batch_size
        self.starvation_seconds = starvation_seconds
        self.waiting: list[InferenceRequest] = []
        self.running: list[InferenceRequest] = []

    def add_request(self, request: InferenceRequest) -> None:
        self.waiting.append(request)

    @abstractmethod
    def pick_next_batch(self, now: float | None = None) -> list[InferenceRequest]:
        """Choose which requests move from waiting -> running."""
        pass

    def _apply_starvation(self, now: float, boosts: dict[str, float]) -> None:
        """After too long in queue, treat request as high priority."""
        for req in self.waiting:
            waited = now - req.arrival_time
            if waited >= self.starvation_seconds:
                req.priority = PriorityLevel.HIGH


class FCFSScheduler(BaseScheduler):
    """First-come-first-serve baseline."""

    def pick_next_batch(self, now: float | None = None) -> list[InferenceRequest]:
        now = now or time.time()
        self._apply_starvation(now, {})

        batch = self.waiting[: self.batch_size]
        self.waiting = self.waiting[self.batch_size :]
        self.running.extend(batch)
        return batch


class PairwiseLTRScheduler(BaseScheduler):
    """
    Our proposed scheduler.

    1. Predictor assigns rank_score to each prompt
    2. User priority adjusts the score
    3. Serve shortest jobs first (lowest effective score)
    """

    def __init__(
        self,
        batch_size: int = 8,
        starvation_seconds: float = 120.0,
        priority_boosts: dict[str, float] | None = None,
    ):
        super().__init__(batch_size, starvation_seconds)
        self.priority_boosts = priority_boosts or {
            "high": -2.0,
            "normal": 0.0,
            "low": 2.0,
        }

    def pick_next_batch(self, now: float | None = None) -> list[InferenceRequest]:
        now = now or time.time()
        self._apply_starvation(now, self.priority_boosts)

        # Min-heap: lowest effective score goes first
        heap: list[QueueItem] = []
        for req in self.waiting:
            key = req.effective_score(self.priority_boosts)
            heapq.heappush(heap, QueueItem(sort_key=key, request=req))

        batch = []
        for _ in range(min(self.batch_size, len(heap))):
            batch.append(heapq.heappop(heap).request)

        picked_ids = {r.request_id for r in batch}
        self.waiting = [r for r in self.waiting if r.request_id not in picked_ids]
        self.running.extend(batch)
        return batch


class PointwiseLTRScheduler(BaseScheduler):
    """
    Baseline from the main paper: sort by predicted absolute length.

    Same queue logic as pairwise, but meant for pointwise predictors.
    """

    def __init__(
        self,
        batch_size: int = 8,
        starvation_seconds: float = 120.0,
        priority_boosts: dict[str, float] | None = None,
    ):
        super().__init__(batch_size, starvation_seconds)
        self.priority_boosts = priority_boosts or {
            "high": -2.0,
            "normal": 0.0,
            "low": 2.0,
        }

    def pick_next_batch(self, now: float | None = None) -> list[InferenceRequest]:
        # Same serving policy; only training differs
        scheduler = PairwiseLTRScheduler(
            self.batch_size,
            self.starvation_seconds,
            self.priority_boosts,
        )
        scheduler.waiting = self.waiting
        batch = scheduler.pick_next_batch(now)
        self.waiting = scheduler.waiting
        self.running.extend(batch)
        return batch


def make_scheduler(policy: str, **kwargs) -> BaseScheduler:
    """Factory helper."""
    policy = policy.lower()
    if policy == "fcfs":
        kwargs.pop("priority_boosts", None)
        return FCFSScheduler(**kwargs)
    if policy == "ltr_pointwise":
        return PointwiseLTRScheduler(**kwargs)
    if policy == "pairwise_ltr":
        return PairwiseLTRScheduler(**kwargs)
    raise ValueError(f"Unknown policy: {policy}")
