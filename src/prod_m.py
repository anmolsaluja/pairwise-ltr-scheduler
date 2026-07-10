"""
ProD-M length predictor.

Takes Llama hidden states and predicts which length bin the output
will fall into. We convert the bin back to a length estimate for SJF.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn


def make_bins(max_length, num_bins):
    return torch.linspace(1, max_length, num_bins + 1)


def length_to_bin(length, bin_edges):
    idx = torch.bucketize(torch.tensor([float(length)]), bin_edges[1:]).item()
    return min(idx, len(bin_edges) - 2)


def bin_to_length(bin_idx, bin_edges):
    lo = bin_edges[bin_idx].item()
    hi = bin_edges[bin_idx + 1].item()
    return (lo + hi) / 2.0


class ProDMPredictor(nn.Module):
    def __init__(self, hidden_dim, num_bins, bin_edges):
        super().__init__()
        self.num_bins = num_bins
        self.register_buffer("bin_edges", bin_edges)
        mid = max(hidden_dim // 2, 256)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mid),
            nn.ReLU(),
            nn.Linear(mid, num_bins),
        )

    def forward(self, x):
        return self.mlp(x)

    @torch.no_grad()
    def predict_lengths(self, hidden_states):
        self.eval()
        bins = self.forward(hidden_states).argmax(dim=-1).cpu().tolist()
        return [bin_to_length(b, self.bin_edges) for b in bins]


def save_prod_m(model, path, meta=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hidden_dim": model.mlp[0].in_features,
            "num_bins": model.num_bins,
            "bin_edges": model.bin_edges,
            "meta": meta or {},
        },
        path,
    )


def load_prod_m(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model = ProDMPredictor(ckpt["hidden_dim"], ckpt["num_bins"], ckpt["bin_edges"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device)


def save_hidden(path, tensor):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(tensor, path)


def load_hidden(path):
    return torch.load(path, map_location="cpu")
