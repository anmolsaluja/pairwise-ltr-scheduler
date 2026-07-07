"""
Pairwise learning-to-rank predictor.

Uses BERT [CLS] embeddings + a linear head.
Trained with margin ranking loss (same idea as PARS).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class PairwiseRanker(nn.Module):
    """Score prompts: higher score means longer expected output."""

    def __init__(self, backbone: str = "bert-base-uncased"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(backbone)
        self.encoder = AutoModel.from_pretrained(backbone)
        hidden = self.encoder.config.hidden_size
        self.scorer = nn.Linear(hidden, 1)

    def encode(self, texts: list[str], max_length: int = 512) -> torch.Tensor:
        """Get BERT [CLS] vectors for a batch of prompts."""
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        device = next(self.parameters()).device
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = self.encoder(**batch)
        # Pooler output is the [CLS] summary BERT was trained with
        return outputs.pooler_output

    def forward(self, texts: list[str], max_length: int = 512) -> torch.Tensor:
        features = self.encode(texts, max_length=max_length)
        return self.scorer(features).squeeze(-1)

    @torch.no_grad()
    def score_prompts(self, texts: list[str], max_length: int = 512) -> list[float]:
        """Return a length score for each prompt."""
        self.eval()
        scores = self.forward(texts, max_length=max_length)
        return scores.cpu().tolist()


def margin_ranking_step(
    model: PairwiseRanker,
    prompt_a: list[str],
    prompt_b: list[str],
    labels: torch.Tensor,
    margin: float,
    max_length: int,
) -> torch.Tensor:
    """
    One training step.

    Loss pushes score(A) > score(B) + margin when label=1.
    """
    score_a = model(prompt_a, max_length=max_length)
    score_b = model(prompt_b, max_length=max_length)
    loss_fn = nn.MarginRankingLoss(margin=margin)
    target = labels.float()
    return loss_fn(score_a, score_b, target)


def save_model(model: PairwiseRanker, path: str) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "backbone": model.encoder.config._name_or_path,
        },
        path,
    )


def load_model(path: str, device: str = "cpu") -> PairwiseRanker:
    checkpoint = torch.load(path, map_location=device)
    model = PairwiseRanker(backbone=checkpoint["backbone"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    return model
