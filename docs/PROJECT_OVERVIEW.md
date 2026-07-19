# Project Overview

## Evaluation (only three methods)

1. **FCFS** — baseline  
2. **LTR** — main paper (pointwise length prediction, single-sample labels)  
3. **PARS + ProD-M + Priority** — **ours**

ProD-M is **not** in the main paper. In our method it means: generate **median-of-r** labels, build pairs from those medians, train PARS, and schedule with priority / starvation.

## Pipeline

```
Prompts
  -> Llama x r -> median labels          [ProD-M labeling, for OURS]
  -> train LTR on single-sample          [MAIN PAPER]
  -> train PARS on median pairs          [OURS]
  -> compare FCFS | LTR | OURS
```

## Why this matches the proposal

Main paper: LTR beats FCFS.  
We improve it with median supervision (ProD-M) + pairwise ranking (PARS) + priority.
