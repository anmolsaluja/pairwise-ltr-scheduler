# Pairwise LTR Scheduler for LLM Serving

Implementation of our proposed **Pairwise Learning-to-Rank scheduler** for reducing LLM inference latency. Based on:

- *An Empirical Study on Latency Reduction Techniques for Large Language Models*
- *PARS: Low-Latency LLM Serving via Pairwise Learning-to-Rank*
- *Efficient LLM Scheduling by Learning to Rank*

## What this does

```
Incoming Requests
       ↓
Pairwise Ranking Predictor  (BERT + margin ranking loss)
       ↓
Priority Prompt Boost     (high / normal / low)
       ↓
LTR Scheduler             (shortest-job-first approximation)
       ↓
Inference Engine          (simulator locally, vLLM on cloud GPU)
```

We compare three policies:

| Policy | Description |
|--------|-------------|
| `fcfs` | First-come-first-serve baseline |
| `ltr_pointwise` | Sort by predicted absolute length |
| `pairwise_ltr` | **Our proposal** — pairwise ranking + priority prompts |

## Quick start (local CPU — simulation only)

```bash
pip install -r requirements.txt

# Train the pairwise ranker (CPU works, but slow — use cloud for real training)
python scripts/train_predictor.py --epochs 1 --train-samples 500

# Compare all schedulers
python scripts/run_simulation.py --compare-all --checkpoint checkpoints/pairwise_ranker.pt
```

## Cloud GPU (recommended)

Your local machine does not need a GPU. Run everything on Google Colab, RunPod, or Lambda Labs:

```bash
git clone https://github.com/YOUR_USERNAME/pairwise-ltr-scheduler.git
cd pairwise-ltr-scheduler
bash cloud/run_on_cloud.sh
```

Or open `notebooks/cloud_experiment.ipynb` in Google Colab.

### Google Colab steps

1. Upload this repo to Colab (or clone from GitHub)
2. Set runtime to **GPU** (T4 is enough for BERT training)
3. Run all cells in `notebooks/cloud_experiment.ipynb`

## Priority prompts

Assign priority when submitting a request:

```python
from src.priority import InferenceRequest, PriorityLevel

req = InferenceRequest(
    request_id="user_42",
    prompt="Summarize this document in 2 sentences.",
    output_length=50,
    priority=PriorityLevel.HIGH,   # served before normal/low
)
```

Priority boosts are configured in `configs/default.yaml`:

```yaml
priority:
  high_boost: -2.0    # lower score = served sooner
  normal_boost: 0.0
  low_boost: 2.0
```

## Project structure

```
├── configs/default.yaml       # All tunable settings
├── src/
│   ├── pairwise_predictor.py  # BERT + margin ranking loss
│   ├── scheduler.py           # FCFS, LTR, Pairwise LTR
│   ├── priority.py            # Priority prompt handling
│   ├── simulator.py           # CPU simulation (no GPU needed)
│   └── vllm_hook.py           # Optional vLLM integration
├── scripts/
│   ├── train_predictor.py     # Train on cloud GPU
│   └── run_simulation.py      # Evaluate schedulers
├── cloud/run_on_cloud.sh      # One-command cloud setup
└── notebooks/cloud_experiment.ipynb
```

## Evaluation metrics

- Average / P50 / P95 / P99 latency
- Queue waiting time
- Throughput (requests/sec)
- Kendall's Tau (ranking quality)

## References

1. Fu et al., "Efficient LLM Scheduling by Learning to Rank," 2024
2. Tao et al., "PARS: Low-Latency LLM Serving via Pairwise Learning-to-Rank," 2025
3. Agrawal et al., "Sarathi-Serve," 2024
4. Kwon et al., "vLLM," SOSP 2023
