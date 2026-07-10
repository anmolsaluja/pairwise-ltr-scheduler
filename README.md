# Pairwise LTR Scheduling for LLM Serving

FDU Vancouver Capstone — CS Master's project.

We look at latency in LLM serving. FCFS causes head-of-line blocking when a
long request sits at the front of the queue. The papers we studied (LTR
scheduling, PARS, ProD-M) suggest ranking requests by expected output length
and serving shorter ones first.

**What we built**

1. Sample Llama a few times per prompt → take the **median** length (ProD-M labels)
2. Train a small MLP on Llama hidden states to predict length bins (**ProD-M**)
3. Train a BERT pairwise ranker on those median pairs (**PARS-style**)
4. Simulate **FCFS vs ProD-M vs Pairwise** and compare latency / throughput

Priority levels (high / normal / low) and a simple starvation rule are included.

---

## Quick start (Google Colab)

1. Runtime → Change runtime type → **GPU (T4)**
2. Accept Llama license: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
3. Create a HF token: https://huggingface.co/settings/tokens
4. Open `notebooks/colab_run.ipynb` and run the cells top to bottom

Or paste this in a Colab cell:

```python
import os
os.environ["HF_TOKEN"] = "hf_YOUR_TOKEN"

!git clone https://github.com/anmolsaluja/pairwise-ltr-scheduler.git
%cd pairwise-ltr-scheduler
!pip install -q -r requirements.txt

!python scripts/check_setup.py
!python scripts/run_all.py --limit 50 --device cuda
```

`--limit 50` is a good first run (~30–90 min on T4). Use 100 for the report.

---

## Project layout

```
src/
  llama.py       # load Llama, sample lengths, pull hidden states
  prod_m.py      # ProD-M MLP
  ranker.py      # pairwise BERT ranker
  scheduler.py   # FCFS / ProD-M / PARS
  simulate.py    # discrete-event simulator
  datasets.py    # gsm8k, math, livecodebench, wildchat, longbench
  data.py        # labels + training pairs
  metrics.py     # latency, MAE, Kendall tau
  requests.py    # request object + priority
  utils.py       # config / LLM profile helpers

scripts/
  check_setup.py
  generate_labels.py
  train_prod_m.py
  train_ranker.py
  evaluate.py
  run_all.py          # runs everything
```

---

## Step by step (local / cloud)

```bash
export HF_TOKEN=hf_...
pip install -r requirements.txt
python scripts/check_setup.py

# full pipeline
python scripts/run_all.py --limit 100 --device cuda

# or one step at a time
python scripts/generate_labels.py --limit 100 --device cuda
python scripts/train_prod_m.py --device cuda
python scripts/train_ranker.py --device cuda
python scripts/evaluate.py --device cuda
```

If training already finished and you only want metrics again:

```bash
python scripts/run_all.py --skip-train --device cuda
```

---

## What to look for in the results

`evaluate.py` prints something like:

```
ProD-M MAE: ...
PARS Kendall Tau: ...

=== FCFS ===
  p95: ...
=== PROD_M ===
  p95: ...
=== PARS ===
  p95: ...
```

We expect length-aware policies (ProD-M / PARS) to beat FCFS on p95 latency
when the workload has mixed short and long outputs.

---

## Notes

- Default model is `meta-llama/Llama-3.2-3B-Instruct` (fits Colab T4 in 4-bit).
- Colab can wipe `/content` if the runtime disconnects — mount Drive and copy
  `checkpoints/` + `data/processed/` when a step finishes.
- Real vLLM integration is left as future work (same as in the proposal).

## References

- Fu et al., Efficient LLM Scheduling by Learning to Rank
- Tao et al., PARS: Low-Latency LLM Serving via Pairwise Learning-to-Rank
- Wang et al., ProD-M (median-supervised length prediction)
- Kwon et al., vLLM / PagedAttention
