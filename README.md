# Improving LTR with PARS + ProD-M + Priority

**FDU Vancouver Capstone**

## Three-way comparison

| # | Method | Whose? |
|---|--------|--------|
| 1 | **FCFS** | Baseline |
| 2 | **LTR scheduler** | Main paper (pointwise, single-sample labels) |
| 3 | **PARS + ProD-M + Priority** | **Ours** |

- **ProD-M** (not in the main paper): sample Llama `r` times, take the **median** length as the label for training our ranker.
- **PARS**: pairwise BERT ranker.
- **Priority**: high / normal / low + starvation prevention.

## Repository structure

```
configs/
  live_run.yaml          # single config for the primary 1000-prompt run
src/
  data.py                # prompt records, label I/O, pairwise training pairs
  datasets.py            # GSM8K / MATH / LiveCodeBench / WildChat / LongBench loaders
  llama.py               # served LLM wrapper (labels + hidden states)
  prod_m.py              # ProD-M pointwise length predictor (bins on hidden states)
  ranker.py              # PARS pairwise BERT ranker
  requests.py            # request objects + priority parsing
  scheduler.py           # FCFS / LTR / PARS(+priority, starvation) scheduling
  simulate.py            # discrete-event simulator
  metrics.py             # latency / ranking metrics
  plots.py               # report-style figures
  live_serve.py          # vLLM live-serving helpers
scripts/
  check_setup.py         # pre-flight checks (HF token, GPU, imports)
  generate_labels.py     # Step 1: Llama x r -> median labels (chunked, --resume safe)
  train_prod_m.py        # Step 2: pointwise predictor (--target single = main-paper LTR)
  train_ranker.py        # Step 3: PARS pairwise ranker on median pairs
  evaluate.py            # Step 4: FCFS | LTR | OURS in the simulator
  plot_results.py        # Step 5: report tables + figures
  run_all.py             # steps 1-4 in one command
  run_live.py            # end-to-end live pipeline (labels -> train -> live vLLM eval)
  evaluate_live.py       # live three-way comparison on a real vLLM engine
  evaluate_live_hf.py    # live comparison via HuggingFace (when vLLM won't import)
  ensure_vllm.py         # install/verify vLLM on Colab
  eval_ood.py            # extra: in-distribution vs out-of-distribution study
  ablation_labels.py     # extra: median (ProD-M) vs single-sample label ablation
  demo_cpu.py            # tiny CPU-only scheduler demo (no GPU / token needed)
notebooks/
  colab_run.ipynb        # full Colab workflow (resume-safe, Drive backups)
cloud/
  run_on_cloud.sh        # one-shot pipeline for Colab / RunPod GPUs
docs/
  PROJECT_OVERVIEW.md    # report write-up notes
```

## Primary run scale: **1000 prompts**

Config: `configs/live_run.yaml` (1000 prompts, chunk size 50, 3 samples/prompt).
Use Colab Pro / A100 when possible. Labels resume from Drive after disconnects.

```python
import os
from google.colab import drive
os.environ["HF_TOKEN"] = "hf_YOUR_TOKEN"
drive.mount("/content/drive")

!git clone https://github.com/anmolsaluja/pairwise-ltr-scheduler.git
%cd pairwise-ltr-scheduler
!pip install -q -r requirements.txt
!python scripts/check_setup.py

# 1000 prompts in chunks of 50 (--resume safe)
!python scripts/generate_labels.py --config configs/live_run.yaml \
  --limit 1000 --chunk-size 50 --num-samples 3 --resume --device cuda \
  --backup-dir /content/drive/MyDrive/capstone_results

!python scripts/train_prod_m.py --config configs/live_run.yaml \
  --target single --output checkpoints/ltr_pointwise.pt --device cuda
!python scripts/train_ranker.py --config configs/live_run.yaml \
  --train-samples 1000 --device cuda
!python scripts/evaluate.py --config configs/live_run.yaml --limit 1000 --device cuda

# Report graphs (paper Fig. 2 / Fig. 3 style)
!python scripts/plot_results.py --config configs/live_run.yaml --limit 1000 --device cuda \
  --out-dir /content/drive/MyDrive/capstone_results/figures
```

Or use `notebooks/colab_run.ipynb` / `python scripts/run_live.py --limit 1000 ...`.

Accept the model license first: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct

**Time (A100):** labeling is the long step (multi-session OK with `--resume`); train/eval/plots are much faster.

## Final printed result looks like

```text
=== FCFS (baseline) ===
=== LTR scheduler (MAIN PAPER) ===
=== PARS + ProD-M + Priority (OURS) ===

LTR vs FCFS: ...%
OURS vs LTR (main paper): ...%
OURS vs FCFS: ...%
```

See `docs/PROJECT_OVERVIEW.md` for the report write-up.

## Live GPU serving (optional)

```bash
# HuggingFace live path (recommended on Colab if vLLM import fails)
python scripts/evaluate_live_hf.py --config configs/live_run.yaml --limit 1000 --device cuda

# vLLM path (when install works)
python scripts/ensure_vllm.py --install
python scripts/evaluate_live.py --config configs/live_run.yaml --limit 1000 --device cuda
```

Results save to `data/processed/live_eval_results.json` and figures under Drive `capstone_results/figures/`.

## Extra experiments (report appendix)

```bash
# Median-of-r vs single-sample labels — shows the gain comes from the supervision target
python scripts/ablation_labels.py --config configs/live_run.yaml --device cuda

# Train on math prompts, test on chat / code / long-context (generalization)
python scripts/eval_ood.py --device cuda

# Quick CPU sanity check of FCFS vs SJF (no GPU, no token)
python scripts/demo_cpu.py
```

## Quick start (no GPU)

```bash
pip install -r requirements.txt   # install torch separately first
python scripts/demo_cpu.py
```
