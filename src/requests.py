"""Request object used by the scheduler / simulator."""

from __future__ import annotations

from dataclasses import dataclass, field

#stores all scheduling- related informationfor an inference request
@dataclass
class Request:
    request_id: str
    prompt: str
    output_length: int = 0
    priority: str = "normal"       # high / normal / low
    arrival_time: float = 0.0
    rank_score: float = 0.0        # higher => expected longer job
    predicted_length: int = 0

    def effective_score(self, priority_boosts):
        """
        Lower score gets served first.
        High priority subtracts from the score so urgent jobs jump ahead.
        """
        return self.rank_score + priority_boosts.get(self.priority, 0.0)


def parse_priority(value):
    value = (value or "normal").lower().strip()
    if value in ("high", "urgent", "p1"):
        return "high"
    if value in ("low", "background", "p3"):
        return "low"
    return "normal"
