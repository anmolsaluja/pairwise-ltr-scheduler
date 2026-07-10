"""
Dataset loaders.

We mix short and long prompts on purpose — the scheduler only helps
when output lengths vary a lot (that's the whole point of the papers).
"""

from __future__ import annotations

import os
import random

from datasets import load_dataset

from src.data import PromptRecord


def _shuffle(rows, seed, limit):
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    if limit:
        rows = rows[:limit]
    return rows


def load_gsm8k(limit=None, seed=42):
    ds = load_dataset("openai/gsm8k", "main", split="train")
    rows = _shuffle(ds, seed, limit)
    out = []
    for i, row in enumerate(rows):
        out.append(
            PromptRecord(
                prompt_id=f"gsm8k_{i}",
                text="Solve step by step.\n\n" + row["question"].strip(),
            )
        )
    return out


def load_math(limit=None, seed=42):
    # competition math tends to need longer answers than gsm8k
    parts = []
    for subset in ("algebra", "number_theory", "counting_and_probability"):
        try:
            parts.extend(list(load_dataset("EleutherAI/hendrycks_math", subset, split="train")))
        except Exception:
            continue
    if not parts:
        parts = list(load_dataset("EleutherAI/hendrycks_math", "algebra", split="train"))

    rows = _shuffle(parts, seed, limit)
    out = []
    for i, row in enumerate(rows):
        out.append(
            PromptRecord(
                prompt_id=f"math_{i}",
                text="Solve this math problem and show your work.\n\n" + row["problem"].strip(),
            )
        )
    return out


def load_livecodebench(limit=None, seed=42):
    ds = load_dataset("livecodebench/code_generation", split="test", streaming=True)
    out = []
    for row in ds:
        q = (row.get("question_content") or "").strip()
        if not q:
            continue
        out.append(
            PromptRecord(
                prompt_id=f"lcb_{row.get('question_id', len(out))}",
                text="Write a complete Python solution.\n\n" + q,
            )
        )
        if limit and len(out) >= limit:
            break
    random.Random(seed).shuffle(out)
    return out


def load_wildchat(limit=None, seed=42):
    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    out = []
    seen = set()
    for row in ds:
        convo = row.get("conversation") or []
        if not convo:
            continue
        msg = (convo[0].get("content") or "").strip()
        if not msg or msg in seen:
            continue
        seen.add(msg)
        out.append(PromptRecord(prompt_id=f"wildchat_{len(out)}", text=msg))
        if limit and len(out) >= limit:
            break
    random.Random(seed).shuffle(out)
    return out


def load_longbench(limit=None, seed=42):
    try:
        ds = load_dataset("THUDM/LongBench-v2", split="train")
        rows = _shuffle(ds, seed, limit)
        out = []
        for i, row in enumerate(rows):
            q = str(row.get("question", "")).strip()
            choices = []
            for key in ("choice_A", "choice_B", "choice_C", "choice_D"):
                if row.get(key):
                    choices.append(f"{key[-1]}. {row[key]}")
            text = q + "\n\n" + "\n".join(choices) + "\n\nPick the best answer and explain briefly."
            out.append(PromptRecord(prompt_id=f"lb2_{row.get('_id', i)}", text=text))
        return out
    except Exception:
        # fallback if LongBench-v2 is unavailable
        ds = load_dataset("hotpot_qa", "fullwiki", split="train", streaming=True)
        out = []
        for row in ds:
            titles = " ".join(row.get("context", {}).get("title", [])[:3])
            sents = row.get("context", {}).get("sentences", [[]])[0][:15]
            ctx = titles + "\n" + " ".join(s[:200] for s in sents if s)
            q = (row.get("question") or "").strip()
            out.append(
                PromptRecord(
                    prompt_id=f"hotpot_{len(out)}",
                    text=f"Context:\n{ctx[:4000]}\n\nQuestion: {q}\n\nAnswer:",
                )
            )
            if limit and len(out) >= limit:
                break
        random.Random(seed).shuffle(out)
        return out


LOADERS = {
    "gsm8k": load_gsm8k,
    "math": load_math,
    "livecodebench": load_livecodebench,
    "wildchat": load_wildchat,
    "longbench": load_longbench,
}


def load_prompts(name="all", limit=None, seed=42, per_dataset=None):
    name = name.lower()
    per_dataset = per_dataset or {}

    if name == "all":
        names = ["gsm8k", "math", "livecodebench", "wildchat", "longbench"]
        parts = []
        for ds in names:
            n = per_dataset.get(ds)
            if n is None and limit:
                n = max(1, limit // len(names))
            try:
                parts.extend(LOADERS[ds](limit=n, seed=seed))
            except Exception as e:
                print(f"warning: could not load {ds}: {e}")
        random.Random(seed).shuffle(parts)
        if limit:
            parts = parts[:limit]
        return parts

    if name not in LOADERS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {list(LOADERS)} or 'all'")
    return LOADERS[name](limit=limit, seed=seed)
