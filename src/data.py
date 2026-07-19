"""
Prompt records, label I/O, and pairwise training pairs.

ProD-M idea: sample the LLM a few times per prompt, take the median
length as a more stable label than a single noisy sample.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field


@dataclass
class PromptRecord:
    prompt_id: str
    text: str
    output_length: int = 0          # median length after labeling
    priority: str = "normal"        # high / normal / low
    sample_lengths: list = field(default_factory=list)
    single_sample_length: int = 0


def save_labels(path, records, meta, hidden_path=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if hidden_path:
        meta = {**meta, "hidden_states_path": hidden_path}

    payload = {
        "meta": meta,
        "records": [
            {
                "prompt_id": r.prompt_id,
                "text": r.text,
                "priority": r.priority,
                "sample_lengths": r.sample_lengths,
                "median_length": r.output_length,
                "single_sample_length": r.single_sample_length
                or (r.sample_lengths[0] if r.sample_lengths else r.output_length),
            }
            for r in records
        ],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_labels(path):
    with open(path) as f:
        payload = json.load(f)

    records = []
    for row in payload["records"]:
        records.append(
            PromptRecord(
                prompt_id=row["prompt_id"],
                text=row["text"],
                output_length=int(row["median_length"]),
                priority=row.get("priority", "normal"),
                sample_lengths=row.get("sample_lengths", []),
                single_sample_length=int(
                    row.get("single_sample_length", row["median_length"])
                ),
            )
        )
    return records, payload.get("meta", {})


def _length_for_pairs(rec, use_single_sample=False):
    # ablation: single-sample labels are noisier than the ProD-M median
    if use_single_sample:
        return rec.single_sample_length or (
            rec.sample_lengths[0] if rec.sample_lengths else rec.output_length
        )
    return rec.output_length


def build_pairs(records, min_diff=0.2, max_pairs=5000, use_single_sample=False):
    """
    Build (prompt_a, prompt_b, label) for margin ranking.

    label=1 means A is longer than B.
    We skip pairs that are too close in length (same idea as PARS).
    """
    pairs = []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            a = _length_for_pairs(records[i], use_single_sample)
            b = _length_for_pairs(records[j], use_single_sample)
            longer = max(a, b)
            if longer == 0:
                continue
            if abs(a - b) / longer < min_diff:
                continue

            if a > b:
                pairs.append((records[i].text, records[j].text, 1))
            else:
                pairs.append((records[j].text, records[i].text, 1))

    random.shuffle(pairs)
    if len(pairs) > max_pairs:
        pairs = pairs[:max_pairs]
    return pairs


def dataset_tag(prompt_id):
    """gsm8k_12 -> gsm8k. Used for ID / OOD splits."""
    if "_" not in prompt_id:
        return "unknown"
    return prompt_id.split("_", 1)[0]


def split_id_ood(records, id_datasets=None, ood_datasets=None):
    """
    Split labeled prompts into in-distribution vs out-of-distribution.

    Default (matches our midterm plan):
      ID  = math-ish sets (gsm8k, math)
      OOD = chat / long-context / coding (wildchat, longbench, livecodebench)
    """
    id_datasets = set(id_datasets or ("gsm8k", "math"))
    ood_datasets = set(ood_datasets or ("wildchat", "lb2", "longbench", "lcb", "livecodebench"))

    id_recs, ood_recs = [], []
    for rec in records:
        tag = dataset_tag(rec.prompt_id)
        # livecodebench ids look like lcb_...
        if tag == "lcb":
            tag = "livecodebench"
        if tag == "lb2":
            tag = "longbench"
        if tag in id_datasets:
            id_recs.append(rec)
        elif tag in ood_datasets:
            ood_recs.append(rec)
    return id_recs, ood_recs


def poisson_arrivals(records, rate, seed=42):
    """Yield (arrival_time, record) with exponential inter-arrival times."""
    rng = random.Random(seed)
    t = 0.0
    for rec in records:
        yield t, rec
        t += rng.expovariate(rate)
