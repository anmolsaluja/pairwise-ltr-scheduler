#!/usr/bin/env bash
# One-shot cloud setup: install deps, train ranker, run comparison.
# Works on Google Colab, RunPod, Lambda Labs, or any Linux GPU box.

set -e

echo "=== Pairwise LTR Scheduler — Cloud Setup ==="

# Install dependencies
pip install -q -r requirements.txt

# Train the pairwise predictor (uses GPU if available)
python scripts/train_predictor.py \
  --epochs 3 \
  --batch-size 32 \
  --train-samples 2000 \
  --output checkpoints/pairwise_ranker.pt

# Compare all three scheduling policies
python scripts/run_simulation.py \
  --compare-all \
  --checkpoint checkpoints/pairwise_ranker.pt \
  --num-requests 200

echo ""
echo "Done. Check latency numbers above."
echo "FCFS should have highest P95; pairwise_ltr should be lowest."
