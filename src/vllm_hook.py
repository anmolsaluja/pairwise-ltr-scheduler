"""
Optional vLLM integration hook.

Full vLLM serving needs a GPU cloud machine. This module shows how
to plug the pairwise scheduler into a vLLM-style waiting queue.

For the class project we use src/simulator.py on CPU first, then
run this on cloud when you have GPU access.
"""

from __future__ import annotations

from src.pairwise_predictor import PairwiseRanker, load_model
from src.priority import InferenceRequest, parse_priority
from src.scheduler import PairwiseLTRScheduler


class VLLMSchedulerHook:
    """
    Drop-in helper for ranking vLLM waiting-queue requests.

    Usage on cloud (after installing vllm):
      hook = VLLMSchedulerHook("checkpoints/pairwise_ranker.pt")
      ordered = hook.rank_waiting_queue(waiting_requests)
    """

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.ranker = load_model(checkpoint_path, device=device)
        self.scheduler = PairwiseLTRScheduler()

    def rank_waiting_queue(self, raw_requests: list[dict]) -> list[dict]:
        """
        raw_requests: list of {"id", "prompt", "priority" (optional)}

        Returns the same list sorted shortest-job-first.
        """
        requests = []
        for item in raw_requests:
            req = InferenceRequest(
                request_id=item["id"],
                prompt=item["prompt"],
                output_length=0,
                priority=parse_priority(item.get("priority", "normal")),
            )
            requests.append(req)

        scores = self.ranker.score_prompts([r.prompt for r in requests])
        for req, score in zip(requests, scores):
            req.rank_score = score

        self.scheduler.waiting = requests
        batch = self.scheduler.pick_next_batch()
        return [{"id": r.request_id, "prompt": r.prompt} for r in batch]
