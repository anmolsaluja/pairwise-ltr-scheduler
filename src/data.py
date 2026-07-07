"""
Load prompts and build training pairs for the pairwise ranker.

We use HuggingFace datasets on cloud so you do not need local GPU storage.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Iterator

from datasets import load_dataset

SAMPLE_DATA = os.path.join(os.path.dirname(__file__), "..", "data", "sample_prompts.json")


@dataclass
class PromptRecord:
    """One prompt with its measured output length."""

    prompt_id: str
    text: str
    output_length: int
    priority: str = "normal"  # high | normal | low


def load_local_sample_prompts(limit: int | None = None) -> list[PromptRecord]:
    """Offline fallback — works without HuggingFace or GPU."""
    with open(SAMPLE_DATA) as f:
        rows = json.load(f)

    records = []
    for i, row in enumerate(rows):
        instruction = row.get("instruction", "")
        inp = row.get("input", "")
        output = row.get("output", "")
        text = f"{instruction}\n{inp}".strip() if inp else instruction
        length = max(1, len(output.split()))
        records.append(
            PromptRecord(
                prompt_id=f"sample_{i}",
                text=text,
                output_length=length,
            )
        )

    if limit:
        records = records[:limit]
    return records


def load_alpaca_prompts(
    split: str = "train",
    limit: int | None = None,
    seed: int = 42,
    use_local: bool = False,
) -> list[PromptRecord]:
    """
    Load Alpaca-style prompts from HuggingFace.

    Output length is estimated from the reference answer token count.
    On cloud you can replace this with real generation lengths from vLLM.

    Set use_local=True (or env USE_LOCAL_DATA=1) to use bundled sample data.
    """
    if use_local or os.environ.get("USE_LOCAL_DATA") == "1":
        return load_local_sample_prompts(limit=limit)

    try:
        ds = load_dataset("tatsu-lab/alpaca", split=split)
        rows = list(ds)
    except Exception:
        # HuggingFace unavailable — fall back to bundled samples
        return load_local_sample_prompts(limit=limit)

    random.Random(seed).shuffle(rows)

    if limit:
        rows = rows[:limit]

    records = []
    for i, row in enumerate(rows):
        instruction = row.get("instruction", "")
        inp = row.get("input", "")
        output = row.get("output", "")

        if inp:
            text = f"{instruction}\n{inp}"
        else:
            text = instruction

        # Simple length proxy: word count of expected answer
        length = max(1, len(output.split()))

        records.append(
            PromptRecord(
                prompt_id=f"alpaca_{split}_{i}",
                text=text.strip(),
                output_length=length,
            )
        )

    return records


def make_pairwise_samples(
    records: list[PromptRecord],
    min_length_diff: float = 0.2,
) -> list[tuple[str, str, int]]:
    """
    Build (prompt_a, prompt_b, label) pairs for margin ranking loss.

    label = 1  -> A is longer than B
    label = -1 -> B is longer than A

    Pairs with tiny length gaps are skipped (PARS filtering).
    """
    pairs = []

    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            len_a = records[i].output_length
            len_b = records[j].output_length

            max_len = max(len_a, len_b)
            if max_len == 0:
                continue

            rel_diff = abs(len_a - len_b) / max_len
            if rel_diff < min_length_diff:
                continue

            if len_a > len_b:
                pairs.append((records[i].text, records[j].text, 1))
            else:
                pairs.append((records[j].text, records[i].text, 1))

    random.shuffle(pairs)
    return pairs


def stream_requests(
    records: list[PromptRecord],
    arrival_rate: float,
    seed: int = 42,
) -> Iterator[tuple[float, PromptRecord]]:
    """
    Yield (arrival_time, request) for simulation.

    arrival_rate = average requests per second (Poisson-like spacing).
    """
    rng = random.Random(seed)
    t = 0.0

    for record in records:
        yield t, record
        gap = rng.expovariate(arrival_rate)
        t += gap
