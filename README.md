# Pairwise LTR Scheduling for LLM Serving

**FDU Vancouver Capstone (CS Master's)**  
Team: Mohammed Sirajuddin, Anmol Saluja, Chandra Sekhar Venigalla,
Daisy Lucia Clavijo Navas, Veera Venkata Sai Sravan Bhamidipati

## Problem

LLM decode time grows with output length. Default **FCFS** scheduling causes
**head-of-line (HOL) blocking**: one long request stalls many short ones.
Shortest-Job-First would help, but the true length is unknown before generation.

## What we implemented

We follow the midterm plan and the proposal roadmap:

| Phase | Deliverable | Code |
|------|-------------|------|
| 1 | Repeated sampling → **median labels** (ProD-M) | `scripts/generate_labels.py` |
| 2 | Train **ProD-M** MLP on Llama hidden states; report MAE + **ID/OOD** | `train_prod_m.py`, `eval_ood.py` |
| 3 | Train **PARS** pairwise ranker on median pairs; Kendall Tau | `train_ranker.py` |
| 4 | Compare **FCFS / ProD-M / PARS / Oracle** in a simulator | `evaluate.py`, `simulate.py` |
| Stretch | Wire ranker scores into **vLLM priority scheduling** | `vllm_integration.py` |

Extra pieces from the slides / proposal:
- starvation prevention (~2 min wait → high priority)
- user priorities (high / normal / low)
- ablation: median labels vs single-sample labels (`ablation_labels.py`)

---

## Quick start (Google Colab, recommended)

1. Runtime → GPU (T4 is fine for Llama 3.2 3B)
2. Accept license: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
3. Create token: https://huggingface.co/settings/tokens
4. Run:

```python
import os
os.environ["HF_TOKEN"] = "hf_YOUR_TOKEN"

!git clone https://github.com/anmolsaluja/pairwise-ltr-scheduler.git
%cd pairwise-ltr-scheduler
!pip install -q -r requirements.txt

!python scripts/check_setup.py
!python scripts/run_all.py --limit 50 --device cuda
```

Or open `notebooks/colab_run.ipynb`.

After labels exist, also run (for the report):

```python
!python scripts/eval_ood.py --device cuda
!python scripts/ablation_labels.py --device cuda --epochs 3
```

Mount Drive and copy `checkpoints/` + `data/processed/` when a step finishes —
Colab can wipe `/content` on disconnect.

---

## Local CPU sanity check (no GPU)

```bash
pip install -r requirements.txt
python scripts/demo_cpu.py
```

This only demos FCFS vs Oracle SJF on fake lengths.

---

## Full pipeline commands

```bash
export HF_TOKEN=hf_...
pip install -r requirements.txt

python scripts/check_setup.py
python scripts/generate_labels.py --limit 100 --device cuda
python scripts/train_prod_m.py --device cuda
python scripts/train_ranker.py --device cuda
python scripts/evaluate.py --device cuda

# report extras
python scripts/eval_ood.py --device cuda
python scripts/ablation_labels.py --device cuda

# stretch (needs `pip install vllm` + GPU)
python scripts/vllm_integration.py --device cuda
```

One-shot:

```bash
python scripts/run_all.py --limit 100 --device cuda
```

For the presentation-scale model (Llama 3.1 8B), use a larger GPU:

```bash
python scripts/run_all.py --config configs/full_run.yaml --device cuda
```

---

## Project layout

```
src/
  llama.py        # Llama sampling + hidden states
  prod_m.py       # 2-layer MLP length-bin predictor
  ranker.py       # BERT + margin ranking loss
  scheduler.py    # FCFS / length-aware SJF + starvation
  simulate.py     # discrete-event serving simulator
  datasets.py     # GSM8K, MATH, LiveCodeBench, WildChat, LongBench
  data.py         # labels, pairs, ID/OOD split helpers
  metrics.py      # latency, MAE, Kendall, NDCG, pairwise acc
  requests.py     # request + priority
  utils.py        # config / model profiles

scripts/
  run_all.py              # end-to-end
  generate_labels.py      # Phase 1
  train_prod_m.py         # Phase 2
  train_ranker.py         # Phase 3
  evaluate.py             # Phase 4
  eval_ood.py             # ID vs OOD
  ablation_labels.py      # median vs single-sample
  vllm_integration.py     # stretch
  demo_cpu.py             # offline scheduler check
```

---

## How this maps to the proposal

Proposal architecture:

```
Incoming requests → Pairwise ranking predictor → LTR scheduler → (vLLM) → responses
```

We implement that, and strengthen the training signal with **ProD-M medians**
(from the midterm), because one noisy sample per prompt is a weak label.

Evaluation plan from the proposal / slides:
- Latency: avg / p50 / p95 / p99, queue wait, approx TTFT
- Ranking: Kendall Tau, pairwise accuracy, NDCG
- Throughput: requests/sec in the simulator (and real wall time under vLLM)
- Baselines: FCFS, pointwise LTR (ProD-M), pairwise (PARS), Oracle SJF

See `docs/PROJECT_OVERVIEW.md` for a longer write-up suitable for the final report.

---

## References

1. Saravana Kumar et al. — An Empirical Study on Latency Reduction Techniques for LLMs (main paper)
2. Fu et al. — Efficient LLM Scheduling by Learning to Rank (arXiv:2408.15792)
3. Wang et al. — ProD: Robust Length Prediction (arXiv:2604.07931)
4. Tao et al. — PARS: Pairwise Learning-to-Rank Serving (arXiv:2510.03243)
5. Kwon et al. — vLLM / PagedAttention (SOSP 2023)
