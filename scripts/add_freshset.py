#!/usr/bin/env python3
"""
Adds fresh-prompt (held-out) labeling to generate_labels.py.

Run once from the repo root:   python add_freshset.py

New flags:
  --exclude-labels PATH   skip any prompt whose TEXT already appears in PATH
  --oversample N          multiply per-dataset limits to build a bigger pool
                          before filtering (default 3)

Why text-matching and not a different seed: gsm8k/math/longbench shuffle before
slicing (seed changes selection), but wildchat/livecodebench take the first N
from a stream and shuffle after (seed only reorders). Text comparison is the
only reliable way to guarantee a disjoint set.

Prompt ids get an "_x2" SUFFIX so dataset_tag() still resolves correctly
for eval_ood.py (a prefix would break it).
"""
import sys

BASE = sys.argv[1] if len(sys.argv) > 1 else "scripts"
p = f"{BASE}/generate_labels.py"
s = open(p).read()

s = s.replace('''    parser.add_argument(
        "--num-samples",''', '''    parser.add_argument(
        "--exclude-labels",
        default="",
        help="path to an existing labels file; prompts whose TEXT appears there are skipped",
    )
    parser.add_argument(
        "--oversample",
        type=int,
        default=3,
        help="multiply per-dataset limits to build a bigger pool before filtering",
    )
    parser.add_argument(
        "--num-samples",''', 1)

s = s.replace('''    records = load_prompts(
        dataset,
        limit=limit,
        per_dataset=cfg["datasets"].get("per_dataset"),
    )
    print(f"Got {len(records)} prompts")''', '''    per_ds = dict(cfg["datasets"].get("per_dataset") or {})
    if args.exclude_labels:
        per_ds = {k: v * args.oversample for k, v in per_ds.items()}
        records = load_prompts(dataset, limit=None, per_dataset=per_ds)
    else:
        records = load_prompts(dataset, limit=limit, per_dataset=per_ds)
    print(f"Got {len(records)} prompts")

    if args.exclude_labels:
        if not os.path.exists(args.exclude_labels):
            print(f"ERROR: --exclude-labels file not found: {args.exclude_labels}")
            sys.exit(1)
        old_recs, _ = load_labels(args.exclude_labels)
        seen_texts = {r.text for r in old_recs}
        before = len(records)
        records = [r for r in records if r.text not in seen_texts]
        print(f"Excluded {before - len(records)} prompts already in {args.exclude_labels}")
        for rec in records:
            rec.prompt_id = f"{rec.prompt_id}_x2"
        records = records[:limit]
        print(f"Fresh held-out set: {len(records)} prompts")
        if len(records) < limit:
            print(f"  (wanted {limit}; raise --oversample for a larger pool)")''', 1)

open(p, "w").write(s)
print("patched generate_labels.py")
