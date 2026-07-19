# Project Overview (final report / viva)

## Correct story (important)

The **main paper** proposes **LTR-based scheduling**: predict/rank expected
output length and serve shorter jobs first to beat FCFS.

**ProD-M is not part of the main paper.**  
We bring ProD-M in from the ProD work (median-of-r labels + hidden-state MLP)
as **our improvement to the training signal**, then add **PARS pairwise
ranking** and **priority / starvation** as in the proposal.

## Methods we compare

| Policy | Source | Training signal | Scheduling |
|--------|--------|-----------------|------------|
| FCFS | Baseline | — | Arrival order |
| LTR | Main paper style | Single-sample length | Pointwise SJF |
| ProD-M | Ours (ProD paper) | Median of r samples | Pointwise SJF |
| PARS | Ours (proposal) | Pairs from medians | Pairwise SJF + priority |
| Oracle | Upper bound | True median | Perfect SJF |

## Pipeline

```
Prompts
   |
   +--> Llama x r --> single sample + MEDIAN label + hidden state
   |
   +--> train pointwise on SINGLE  --> LTR policy     [main paper style]
   +--> train pointwise on MEDIAN  --> ProD-M policy  [ours]
   +--> train pairwise on MEDIAN pairs --> PARS       [ours]
   |
   v
Compare FCFS | LTR | ProD-M | PARS | Oracle
```

## Why this matches the proposal

Proposal: extend the main paper by replacing / improving its length-prediction
path with **pairwise ranking**.  

Midterm: also use **median supervision (ProD-M)** so labels are robust.  

So the “ours” stack is **ProD-M labels + PARS + priority**, evaluated against
**FCFS** and **main-paper-style LTR**.

## Code map

| Piece | File |
|-------|------|
| Median + single-sample labels | `scripts/generate_labels.py` |
| LTR / ProD-M training | `scripts/train_prod_m.py --target single\|median` |
| PARS training | `scripts/train_ranker.py` |
| Comparison table | `scripts/evaluate.py` |
| Priority + starvation | `src/scheduler.py`, `src/requests.py` |

## Reproduce

```bash
python scripts/run_all.py --limit 100 --device cuda
```

Expect printed lines like:

```text
LTR  (single-sample train) MAE vs median: ...
ProD-M (median train)      MAE vs median: ...
=== LTR pointwise (MAIN PAPER style ...) ===
=== ProD-M pointwise (OURS: median labels ...) ===
=== PARS pairwise (OURS ...) ===
PARS vs LTR (our pairwise vs main paper): ...%
```
