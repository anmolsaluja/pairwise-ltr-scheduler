# Improving Main-Paper LTR Scheduling with ProD-M + PARS

**FDU Vancouver Capstone (CS Master's)**  
Team: Mohammed Sirajuddin, Anmol Saluja, Chandra Sekhar Venigalla,
Daisy Lucia Clavijo Navas, Veera Venkata Sai Sravan Bhamidipati

## Idea

The **main paper** uses **Learning-to-Rank (LTR)** / pointwise output-length
prediction to schedule shorter LLM requests first (better than FCFS).

**ProD-M is not in the main paper.** We add it (from the ProD work) plus **PARS**
pairwise ranking and **request priority**, then compare everything.

## What we compare

| Policy | Whose idea? | What it is |
|--------|-------------|------------|
| **FCFS** | Baseline | Arrival order |
| **LTR** | **Main paper** | Pointwise length model trained on **single-sample** labels |
| **ProD-M** | **Ours** (ProD paper) | Pointwise model trained on **median-of-r** labels |
| **PARS** | **Ours** (proposal) | Pairwise ranking + priority + starvation |
| **Oracle** | Upper bound | True median lengths |

## Pipeline

```
Prompts
  -> Llama x r samples -> keep single sample + take MEDIAN
  -> LTR model  (train on single sample)     [main paper style]
  -> ProD-M     (train on median)            [ours]
  -> PARS       (pairs from medians)         [ours]
  -> Compare FCFS | LTR | ProD-M | PARS | Oracle
```

---

## Run on Google Colab (GPU)

```python
import os
from google.colab import drive
os.environ["HF_TOKEN"] = "hf_YOUR_TOKEN"
drive.mount("/content/drive")

!git clone https://github.com/anmolsaluja/pairwise-ltr-scheduler.git
%cd pairwise-ltr-scheduler
!pip install -q -r requirements.txt

!python scripts/check_setup.py
!python scripts/run_all.py --limit 100 --device cuda

!mkdir -p /content/drive/MyDrive/capstone_results
!cp -r checkpoints data/processed /content/drive/MyDrive/capstone_results/
```

Accept Llama license first: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct  

**Time (T4, limit=100):** about 2.5–5 hours (extra ProD-M train is only a few minutes; labels + PARS dominate).

---

## Manual steps

```bash
export HF_TOKEN=hf_...
pip install -r requirements.txt

python scripts/generate_labels.py --limit 100 --device cuda

# main-paper style LTR
python scripts/train_prod_m.py --target single --output checkpoints/ltr_pointwise.pt --device cuda

# our ProD-M
python scripts/train_prod_m.py --target median --output checkpoints/prod_m.pt --device cuda

# our PARS
python scripts/train_ranker.py --device cuda

python scripts/evaluate.py --device cuda
```

See `docs/PROJECT_OVERVIEW.md` for the report write-up.

## References

1. Main paper — LTR scheduling for LLM latency  
2. Fu et al. — Efficient LLM Scheduling by Learning to Rank  
3. Wang et al. — ProD / ProD-M (median length prediction) — **our addition**  
4. Tao et al. — PARS (pairwise LTR) — **our addition**  
5. Kwon et al. — vLLM  
