# NVL72 EP72 device_sync_kernel Analysis

**Profile**: `mcore-benchmarking-v0.16-dev-Head128-hepS-AG1-noCG-no1f1b-profile-SM32-Comb_36_4_1_18`
**Config**: Qwen3-Next-80B-A3B_E72, TP1 PP1 EP72, 72× B300, CUDA graphs OFF, 1F1B overlap OFF
**Combine**: SM=32, G2S_stages=36, S2G_stages=4, group=1, chunk=64, batch=18
**Data**: All 72 ranks profiled

---

## Key Finding: Backward device_sync is Large Due to Per-Occurrence Compute Jitter

### Forward vs Backward Sync Comparison

| Sync Position   | Forward Median | Backward Median | Notes                          |
|-----------------|----------------|-----------------|--------------------------------|
| pre-dispatch    | 80 us          | 2,914 us        | Before dispatch_kernel         |
| post-dispatch   | 34 us          | 34 us           | After dispatch_kernel          |
| pre-combine     | 74 us          | 3,870 us        | Before combine_kernel          |
| post-combine    | 23 us          | 193 us          | After combine_kernel           |

### Backward MoE Cycle Breakdown (Rank 0 Median)

| Phase              | Duration (us) | % of Cycle |
|--------------------|---------------|------------|
| pre_dispatch_sync  | 2,914         | 19.4%      |
| dispatch           | 370           | 2.5%       |
| post_dispatch_sync | 34            | 0.2%       |
| permute            | 788           | 5.3%       |
| expert_compute     | 2,112         | 14.1%      |
| unpermute          | 1,035         | 6.9%       |
| pre_combine_sync   | 3,870         | 25.8%      |
| combine            | 3,510         | 23.4%      |
| post_combine_sync  | 191           | 1.3%       |
| **TOTAL**          | **~15,000**   | **100%**   |

**Barrier overhead = 46.5%** of the backward MoE cycle.

---

## Root Cause: Per-Occurrence Stochastic Compute Jitter

### What Was Misleading: Median-of-Medians Analysis

Comparing per-rank **medians** shows tight spreads:

| Segment                               | Median Spread (across 72 ranks) |
|----------------------------------------|---------------------------------|
| Seg A: permute + expert + unpermute    | 119 us                          |
| Seg B: combine + inter-layer compute   | 118 us                          |
| Dispatch kernel duration               | 48 us                           |
| Combine kernel duration                | 51 us                           |
| Post-dispatch sync (barrier protocol)  | 49 us                           |

All extremely tight — seemingly incompatible with 3-4ms sync times.

### What's Actually Happening: Per-Occurrence Cross-Rank Spread

When we measure the spread **within each individual occurrence** (same MoE layer in the same iteration across all 72 ranks):

| Segment | Per-Occurrence Spread Median | P95          | Max          |
|---------|------------------------------|--------------|--------------|
| Seg A   | **2,962 us**                 | 20,675 us    | 25,306 us    |
| Seg B   | **5,094 us**                 | 24,489 us    | 27,206 us    |

The per-occurrence spread distribution for Seg A (permute + expert + unpermute):
```
  [    0,   100) us:    0 ( 0.0%)
  [  100,   200) us:   10 ( 5.0%)
  [  200,   500) us:   22 (11.0%)
  [  500,  1000) us:    4 ( 2.0%)
  [ 1000,  2000) us:   19 ( 9.5%)
  [ 2000,  5000) us:  122 (61.0%)  ← MAJORITY
  [ 5000, 10000) us:    2 ( 1.0%)
  [10000, 30000) us:   21 (10.5%)
```

For Seg B (combine + inter-layer bwd compute):
```
  [    0,   100) us:    0 ( 0.0%)
  [  100,   200) us:   29 (14.5%)
  [  200,   500) us:    7 ( 3.5%)
  [  500,  1000) us:    8 ( 4.0%)
  [ 1000,  2000) us:   34 (17.0%)
  [ 2000,  5000) us:   21 (10.5%)
  [ 5000, 10000) us:    9 ( 4.5%)
  [10000, 30000) us:   92 (46.0%)  ← HALF of occurrences
```

### Interpretation

The device_sync **median-of-medians** is deceptive because it captures systematic inter-rank bias (which is tiny). The actual sync time at each barrier is determined by the **worst single rank at that specific occurrence**, which varies wildly due to **stochastic GPU jitter**:

- Each rank's per-occurrence compute time has a heavy tail (p95/median ratio ~2x)
- With 72 ranks, the probability of at least one rank hitting its tail is very high
- The expected maximum of 72 i.i.d. draws from a heavy-tailed distribution is much larger than the median

**The ~3ms pre-combine sync** ≈ **2,962 us median per-occurrence spread** of Seg A across 72 ranks. The fastest rank at each occurrence waits ~3ms for the slowest.

**The ~2.9ms pre-dispatch sync** is determined by Seg B spread, but Seg B also has a bimodal distribution (14.5% at ~150 us when ranks are tight, 46% at 10-30ms when they're spread). The pre-dispatch sync averages these effects.

---

## Why Forward is Small

In the forward pass, the `ag_nvl_kernel` (custom NVLink allgather, ~7.8ms) acts as a strong synchronization point with its own internal barrier. After the allgather, all ranks proceed through `scan → permute_preprocessing → pre-dispatch-sync → dispatch → ... → pre-combine-sync` nearly in lockstep. The compute jitter within a single forward MoE cycle (expert compute median ~991 us, p95 ~1800 us) is much smaller relative to the 7.8ms synchronization window.

In the backward pass, there is no allgather. The inter-layer backward compute (2.5ms) and expert backward compute (2.1ms) both have heavy tails that frequently create 2-5ms of cross-rank spread.

---

## Barrier Protocol Overhead: Negligible

The `device_sync_kernel` barrier protocol itself is cheap:
- **Post-dispatch sync: 34 us median** (range 4-53 us across ranks)
- **Rank 0 (flag owner): 34 us** — no systematic advantage from hosting the atomic flag
- The `__threadfence_system()` + atomic add + polling loop costs ~34-50 us total

There is **no significant barrier release skew**. The post-dispatch sync (where all ranks enter nearly simultaneously after dispatch) shows only 49 us spread across all 72 ranks. The 3ms sync times are purely due to straggler waiting, not protocol overhead.

---

## Notable Outliers

- **100-127ms pre-dispatch outliers** (first backward MoE layer): Loss backward + non-MoE AllReduce must complete before the first MoE backward dispatch
- **14-15ms intra-cycle spikes** (~0.4% at p99): Occasional large expert compute on individual ranks
- **Rank 41** is the most frequent straggler (shortest sync = arrived latest), with avg backward expert compute 3,730 us vs 2,112 us median — likely receives more tokens due to routing randomness

---

## Data Files

- `nvl72_device_sync_per_rank.csv`: Per-rank median/p95 for each sync type and compute segment
- `nvl72_device_sync_full.json`: Complete per-rank statistics
- `nvl72_per_occurrence_spread.csv`: Per-occurrence cross-rank spread for Seg A and Seg B
