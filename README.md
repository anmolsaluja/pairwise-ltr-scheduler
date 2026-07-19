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

## Run (Colab GPU)

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
```

Accept license: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct  

**Time (T4, limit=100):** about 2.5–4.5 hours.

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
