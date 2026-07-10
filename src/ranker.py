"""
Pairwise ranker (PARS-style).

Instead of predicting exact length, we learn to say which of two
prompts is likely to produce a longer answer. That ranking is enough
to approximate Shortest-Job-First.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class PairwiseRanker(nn.Module):
    def __init__(self, backbone="bert-base-uncased"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(backbone)
        self.encoder = AutoModel.from_pretrained(backbone)
        self.scorer = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, texts, max_length=512):
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        device = next(self.parameters()).device
        batch = {k: v.to(device) for k, v in batch.items()}
        cls = self.encoder(**batch).pooler_output
        return self.scorer(cls).squeeze(-1)

    @torch.no_grad()
    def score(self, texts, max_length=512):
        self.eval()
        return self.forward(texts, max_length=max_length).cpu().tolist()


def ranking_loss(model, prompts_a, prompts_b, labels, margin, max_length):
    # labels are +1 when A should score higher (longer) than B
    sa = model(prompts_a, max_length=max_length)
    sb = model(prompts_b, max_length=max_length)
    return nn.MarginRankingLoss(margin=margin)(sa, sb, labels.float())


def save_ranker(model, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "backbone": model.encoder.config._name_or_path,
        },
        path,
    )


def load_ranker(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model = PairwiseRanker(backbone=ckpt["backbone"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device)
