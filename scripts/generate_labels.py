#!/usr/bin/env python3
"""
Step 1: generate ProD-M labels with the real LLM.

For each prompt we sample the model r times and take the median
output length. We also save last-token hidden states for ProD-M.
"""

from __future__ import annotations

import argparse
import os
import sys
from statistics import median

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import PromptRecord, save_labels
from src.datasets import load_prompts
from src.llama import LlamaServer
from src.prod_m import save_hidden
from src.utils import get_hf_token, load_config, resolve_llm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--llm-profile", default=None)
    parser.add_argument("--output", default="data/processed/prod_labels.json")
    parser.add_argument("--hidden-output", default="data/processed/prod_hidden.pt")
    args = parser.parse_args()

    if not get_hf_token():
        print("ERROR: export HF_TOKEN=hf_... first")
        print("Also accept the license: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct")
        sys.exit(1)

    cfg = load_config(args.config)
    llm = resolve_llm(cfg, args.llm_profile)
    dataset = args.dataset or cfg["datasets"]["name"]
    limit = args.limit or cfg["datasets"]["limit"]
    r = cfg["prod_m"]["num_samples"]

    print(f"LLM: {llm['profile']} -> {llm['model']}")
    print(f"Loading {limit} prompts from {dataset}...")
    records = load_prompts(
        dataset,
        limit=limit,
        per_dataset=cfg["datasets"].get("per_dataset"),
    )
    print(f"Got {len(records)} prompts")

    server = LlamaServer(
        llm["model"],
        device=args.device,
        load_in_4bit=llm["load_in_4bit"],
        max_prompt_tokens=llm["max_prompt_tokens"],
    )

    labeled = []
    print(f"Sampling {r} times per prompt...")
    for i, rec in enumerate(records):
        samples = server.generate_lengths(
            rec.text,
            num_samples=r,
            max_new_tokens=llm["max_new_tokens"],
            temperature=cfg["prod_m"]["temperature"],
            top_p=cfg["prod_m"]["top_p"],
        )
        med = int(median(samples))
        labeled.append(
            PromptRecord(
                prompt_id=rec.prompt_id,
                text=rec.text,
                output_length=med,
                priority=rec.priority,
                sample_lengths=samples,
                single_sample_length=samples[0],
            )
        )
        if (i + 1) % 5 == 0 or i + 1 == len(records):
            print(f"  {i + 1}/{len(records)}  (last median={med})")

    print("Extracting hidden states...")
    hidden = server.encode(
        [r.text for r in labeled],
        batch_size=cfg["prod_m"]["encode_batch_size"],
    )
    save_hidden(args.hidden_output, hidden)
    save_labels(
        args.output,
        labeled,
        meta={
            "dataset": dataset,
            "llm": llm["model"],
            "profile": llm["profile"],
            "num_samples": r,
        },
        hidden_path=args.hidden_output,
    )
    print(f"Saved labels -> {args.output}")
    print(f"Saved hidden -> {args.hidden_output}")


if __name__ == "__main__":
    main()
