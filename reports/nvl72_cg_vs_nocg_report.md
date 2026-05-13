# NVL72 EP72: CUDA Graph vs No CUDA Graph Comparison

## Profile Configurations

| Setting           | No CG                                   | CG                                      |
|-------------------|----------------------------------------|----------------------------------------|
| CUDA Graphs       | OFF                                     | **Full-iteration CG**                   |
| 1F1B Overlap      | OFF                                     | OFF                                     |
| Combine config    | SM32, Comb_36_4_1_**18**                | SM32, Comb_36_4_1_**4**                 |
| Model             | Qwen3-Next-80B-A3B_E72                  | Same                                    |
| GPUs              | 72× B300, TP1 PP1 EP72                  | Same                                    |

---

## Headline: CUDA Graphs Eliminate Barrier Overhead

### Backward MoE Cycle Comparison (Rank 0 Medians)

| Phase              | No CG (us) | CG (us)  | Delta      | Notes                |
|--------------------|------------|----------|------------|----------------------|
| pre_dispatch_sync  | 2,914      | **67**   | **-2,847** | 97.7% reduction      |
| dispatch           | 370        | 357      | -13        |                      |
| post_dispatch_sync | 34         | 36       | +2         |                      |
| permute            | 788        | 791      | +3         |                      |
| expert_compute     | 2,112      | 2,080    | -32        |                      |
| unpermute          | 1,035      | 1,036    | +1         |                      |
| pre_combine_sync   | 3,870      | **59**   | **-3,811** | 98.5% reduction      |
| combine            | 3,510      | 2,687    | -823       | Different batch param |
| post_combine_sync  | 191        | **22**   | **-169**   | 88.5% reduction      |
| **TOTAL**          | **14,824** | **7,137**| **-7,687** | **2.08x speedup**    |

### Barrier Overhead: 46.5% → 2.6%

| Metric                       | No CG       | CG         |
|------------------------------|-------------|------------|
| Total sync time per cycle    | 6,975 us    | 184 us     |
| % of cycle in barriers       | 46.5%       | 2.6%       |
| Dispatch-to-dispatch spread  | 13 us       | **3 us**   |

---

## Why CUDA Graphs Fix This

### Root Cause in No-CG: Kernel Launch Jitter

Without CUDA graphs, each kernel is launched individually by the CPU driver:
- **Kernel launch gap: ~6 us** per kernel (CPU → driver → GPU submission)
- With ~79 kernels per backward MoE cycle, that's **~474 us of total launch overhead**
- More critically, the launch overhead is **variable**: it depends on CPU scheduling, cache state, driver contention
- With 72 ranks, each with stochastic launch jitter, the **extreme value (worst of 72)** is always large
- This creates per-occurrence cross-rank spread of **2,962 us (Seg A) and 5,094 us (Seg B)**

### CUDA Graphs Eliminate the Jitter

With full-iteration CUDA graphs:
- **Kernel launch gap: ~0.6 us** (pre-recorded command buffer, no CPU involvement)
- Launch timing is **deterministic** — no per-kernel CPU jitter
- All 72 ranks execute the same pre-recorded graph with identical timing
- **Result: per-occurrence compute is nearly identical across ranks**

Evidence:
- Post-dispatch sync (where ranks enter nearly together): **36 us** — identical to No-CG (34 us)
- Pre-dispatch sync: **67 us** — reflecting only the true compute spread
- Pre-combine sync: **59 us** — reflecting only the true compute spread
- Dispatch-to-dispatch cycle spread across 72 ranks: **3 us** (was 13 us)

### Per-Rank Distribution (CG)

Backward pre-dispatch sync:
```
Global: median=68 us, p95=152 us, max=1465 us
Range of per-rank medians: spread=134 us
```

Backward pre-combine sync:
```
Global: median=74 us, p95=131 us, max=265 us
Range of per-rank medians: spread=104 us
```

These are ~40-50x smaller than the No-CG case (2,914 / 3,870 us).

---

## Remaining Bottleneck: Combine Kernel

With barriers eliminated as the bottleneck, the backward MoE cycle is now dominated by:

| Phase          | CG Median | % of Cycle |
|----------------|-----------|------------|
| combine        | 2,687 us  | **37.7%**  |
| expert_compute | 2,080 us  | 29.1%      |
| unpermute      | 1,036 us  | 14.5%      |
| permute        | 791 us    | 11.1%      |
| dispatch       | 357 us    | 5.0%       |
| barriers (all) | 184 us    | 2.6%       |

Note: the CG profile uses batch=4 (vs batch=18 in no-CG), giving combine=2,687 us (vs 3,510 us). Our batched accumulation optimization (batch=8) should bring this further down.

---

## Data Files

- `nvl72_cg_device_sync_per_rank.csv`: Per-rank sync distribution (CG profile)
- `nvl72_cg_per_occurrence_spread.csv`: Per-occurrence cross-rank spread (CG profile)
- `nvl72_cg_vs_nocg_comparison.csv`: Side-by-side comparison
