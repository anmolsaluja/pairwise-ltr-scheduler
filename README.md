# Improving Main-Paper LTR with PARS + ProD-M + Priority

**FDU Vancouver Capstone (CS Master's)**

## Three-way comparison only

| # | Method | Whose? |
|---|--------|--------|
| 1 | **FCFS** | Baseline |
| 2 | **LTR scheduler** | Main paper (pointwise, single-sample labels) |
| 3 | **PARS + ProD-M + Priority** | **Ours** |

- **ProD-M** (not in the main paper): sample Llama `r` times, take the **median** length as the label for training our ranker.  
- **PARS**: pairwise BERT ranker.  
- **Priority**: high / normal / low + starvation prevention.

## Run (Colab GPU) — 4 checkpoints of 25 prompts

Labeling is the slow part. We save every **25 prompts** and copy to Drive.

```python
import os
from google.colab import drive
os.environ["HF_TOKEN"] = "hf_YOUR_TOKEN"
drive.mount("/content/drive")

!git clone https://github.com/anmolsaluja/pairwise-ltr-scheduler.git
%cd pairwise-ltr-scheduler
!pip install -q -r requirements.txt
!python scripts/check_setup.py

# Chunk 1-4: safe to re-run if disconnected (--resume)
!python scripts/generate_labels.py \
  --limit 100 --chunk-size 25 --resume --device cuda \
  --backup-dir /content/drive/MyDrive/capstone_results

# After 100 labels exist:
!python scripts/train_prod_m.py --target single --output checkpoints/ltr_pointwise.pt --device cuda
!python scripts/train_ranker.py --train-samples 100 --device cuda
!python scripts/evaluate.py --limit 100 --device cuda
```

Or use `notebooks/colab_run.ipynb`.

Accept license: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct  

**Time (T4):** ~30–60 min per 25-prompt chunk × 4, then ~1 hr train/eval.

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

## Live vLLM + higher prompt count (~1000)

Uses a real GPU engine instead of the discrete-event simulator.
Config: `configs/live_run.yaml` (1000 prompts, 3 samples/prompt for faster labeling).

```bash
pip install vllm   # GPU only

# Full pipeline (chunked labels → train → live FCFS/LTR/PARS)
python scripts/run_live.py \
  --limit 1000 --chunk-size 50 --num-samples 3 --device cuda \
  --backup-dir /content/drive/MyDrive/capstone_results

# Or live eval only (after labels + checkpoints exist)
python scripts/evaluate_live.py --config configs/live_run.yaml --limit 1000 --device cuda
```

Results print to the terminal and save to `data/processed/live_eval_results.json`.

**Note:** Labeling 1000 prompts is multi-session work on a T4 (use `--resume` + Drive backup). Live eval itself needs enough GPU RAM for vLLM (T4 + Llama-3.2-3B is the intended Colab path). Prefer `scripts/evaluate_live_hf.py` on Colab if vLLM import fails.

## Report graphs (Results section)

After labels + checkpoints exist:

```bash
python scripts/plot_results.py --config configs/live_run.yaml --limit 100 --device cuda \
  --out-dir /content/drive/MyDrive/capstone_results/figures
```

Writes paper-style PNGs (length distributions, latency vs request rate, FCFS/LTR/OURS bars) plus `results_section.md` for the report.
