# Project Overview (for final report / demo)

This note is written the way we would explain the project in a viva or final paper.
It matches the submitted proposal and midterm slides.

## 1. Motivation

Serving LLMs is expensive and latency-sensitive. Generation is autoregressive, so
runtime scales with the number of output tokens. Production stacks often use
**FCFS**. When a long request sits at the head of the queue, short requests wait
behind it — classic **HOL blocking**.

**SJF** would reduce average wait, but we do not know the output length in advance.
Recent work replaces exact length knowledge with **learning-to-rank**: if we can
order requests from short to long well enough, we can approximate SJF.

## 2. Related work we build on

- **Main paper (LTR + vLLM):** rank requests, then serve shorter ones earlier.
- **Fu et al.:** ranking quality (e.g. Kendall Tau) is enough for scheduling gains.
- **ProD / ProD-M:** output length is heavy-tailed; train on the **median of r samples**
  and use the served LLM's **hidden states** + a small MLP.
- **PARS:** train a **pairwise** BERT ranker with margin loss; add **starvation prevention**.
- **vLLM:** real engine with paged KV cache; stretch goal for end-to-end serving.

## 3. Our pipeline

```
Datasets (mixed short/long prompts)
        |
        v
Llama x r samples / prompt  ----> median length label   (Phase 1)
        |
        +----> last-token hidden state
        |
        v
ProD-M MLP (length bins)  ----> pointwise SJF baseline  (Phase 2)
        |
        v
Build pairs from medians (filter small gaps)
        |
        v
PARS BERT ranker (margin loss) ----> pairwise SJF       (Phase 3)
        |
        v
Simulator: FCFS vs ProD-M vs PARS vs Oracle             (Phase 4)
        |
        v
(optional) vLLM priority scheduling                     (Stretch)
```

The novel link from our midterm: **median labels feed the pairwise ranker**,
so robust supervision and fair ranking are connected.

## 4. What each module does

| File | Role |
|------|------|
| `src/llama.py` | Load Llama (4-bit on Colab), sample lengths, extract hidden states |
| `src/prod_m.py` | 2-layer MLP over length bins; save/load checkpoint |
| `src/ranker.py` | BERT encoder + linear score; margin ranking loss |
| `src/scheduler.py` | Waiting queue; FCFS or score-sorted batching; starvation bump |
| `src/simulate.py` | Discrete-event sim with Poisson arrivals |
| `src/datasets.py` | GSM8K, MATH, LiveCodeBench, WildChat, LongBench-v2 |
| `scripts/eval_ood.py` | Train on math-like data, test on chat/code/long (ID/OOD) |
| `scripts/ablation_labels.py` | Median labels vs single-sample labels |
| `scripts/vllm_integration.py` | Map PARS scores → vLLM `priority=` |

## 5. Metrics we report

**Prediction / ranking**
- MAE of ProD-M vs median target
- Kendall Tau, pairwise accuracy, NDCG of PARS
- ID vs OOD MAE / Tau gap

**Serving (simulator)**
- Average, p50, p95, p99 latency
- Average queue wait
- Approximate TTFT (= wait + prefill)
- Throughput (req/s)

**Serving (vLLM stretch)**
- Wall-clock time, req/s, optional GPU/KV/preemption stats

## 6. How to reproduce results for submission

### Minimum (T4 Colab, ~1–3 hours for limit=50–100)

```bash
export HF_TOKEN=hf_...
python scripts/run_all.py --limit 100 --device cuda
python scripts/eval_ood.py --device cuda
python scripts/ablation_labels.py --device cuda --epochs 3
```

Copy the printed tables into the report.

### Stronger (A100 / L4, matches slide model)

```bash
python scripts/run_all.py --config configs/full_run.yaml --device cuda
```

### Stretch

```bash
pip install vllm
python scripts/vllm_integration.py --limit 50 --device cuda
```

## 7. Expected story in the results

1. ProD-M MAE should be reasonable vs the median target (not perfect — bins + small data).
2. PARS Kendall Tau / pairwise accuracy should be clearly above random (0.0).
3. In the simulator, **Oracle ≤ PARS ≈ ProD-M < FCFS** on p95 / avg wait when
   the workload is skewed (mixed short and long outputs).
4. Ablation: median supervision should beat or match single-sample labels.
5. OOD: some degradation is expected; we discuss the gap honestly.

## 8. Honest scope notes (good for viva)

- Full production vLLM latency study (TTFT/TBT under high QPS with preemption
  traces) is the **stretch** goal — same as midterm slide 7.
- The discrete-event simulator uses a simple cost model
  (`prefill + tokens * decode_time`). It is enough to show HOL effects and
  compare policies fairly under the same assumptions.
- Default Colab model is **Llama-3.2-3B-Instruct** for VRAM. The slides mention
  Llama-3.1-8B; that is available via `configs/full_run.yaml`.

## 9. Team contribution tip

When writing the report, split work by phase (data, ProD-M, PARS, evaluation,
docs/demo). The code is intentionally organized that way.
