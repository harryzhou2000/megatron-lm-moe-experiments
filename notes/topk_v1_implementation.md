# `naive_topk_and_mask_v1` — Warp-level Bitonic Top-K

## Overview

Replacement for the original `naive_topk_and_mask` in the fused router kernel.
Uses a warp-level bitonic sort network operating on register files instead of
the original repeated-scan approach.

**Files modified:**
- `TE/transformer_engine/common/fused_router/utils.h` — new top-K + sort/merge functions
- `TE/transformer_engine/common/fused_router/fused_topk_with_score_function.cu` — callsite,
  init optimization, double→float arithmetic

## 1. Algorithm Design

### Data layout

Each of the 32 warp lanes holds `N_REGS` register pairs `(float val, int idx)`.
The registers form a virtual array of `N_REGS × 32` elements in row-major order:

```
virtual_pos(reg_r, lane_l) = r * 32 + l
```

`N_REGS` is chosen from `{1, 2, 4}` — powers of two only, so that
`N_REGS × 32` equals the bitonic network's padded size `TOTAL_P2` and every
register index referenced by the network exists.

### Why powers of two only

The bitonic sort's cross-register CAS step pairs register `r` with
`r ^ (j/32)`.  When `N_REGS` is not a power of two (e.g. 3), some partner
indices fall outside `[0, N_REGS)` and the comparison is silently skipped,
breaking the bitonic invariants.

### N_REGS dispatch

The dispatch wrapper `naive_topk_and_mask_v1()` selects `N_REGS` based on
both `topk` and `data_size`:

```
keep_regs     = ceil(topk / 32)
needs_stream  = (data_size > keep_regs * 32)
min_n_regs    = needs_stream ? keep_regs + 1 : keep_regs
min_n_regs    = round_up_to_power_of_2(min_n_regs)   // skip 3 → 4
```

The streaming path requires at least one spare register row (`new_slots > 0`)
to avoid an infinite loop.  With `N_REGS ∈ {1, 2, 4}`:

| topk range | E ≤ N*32 | E > N*32 (streaming) | N_REGS | new_slots |
|------------|----------|----------------------|--------|-----------|
| 1–32       | N=1      | N=2, new=1           | 1 or 2 | 0 or 32   |
| 33–64      | N=2      | N=4, new=2           | 2 or 4 | 0 or 64   |
| 65–96      | —        | N=4, new=1           | 4      | 32        |
| 97–128     | N=4      | (unsupported)        | 4      | —         |

Maximum supported: `topk ≤ 128` if `data_size ≤ 128`, or `topk ≤ 96` if
streaming is needed.

### Bitonic sort

`warp_bitonic_sort_N_descending<N>` implements a standard bitonic sort over
`N × 32` virtual positions, padded to `TOTAL_P2`:

- **Cross-lane steps** (`j < 32`): each register independently does
  `__shfl_xor_sync` to exchange with partner lane.  Each lane decides whether
  to keep its own or the partner's value based on `(is_low, descending)`.

- **Cross-register steps** (`j ≥ 32`): same-lane CAS between registers
  `r` and `r ^ (j/32)`.  Uses `bitonic_cas_descending` helper.

Comparator: **(value DESC, index ASC)** — higher value wins; on tie, lower
original index wins.  This matches the original `naive_topk_and_mask` behavior
and PyTorch's `torch.topk` deterministic tie-breaking.

Sub-stage count for `TOTAL_P2 = 2^n`:  `n*(n+1)/2` total (k,j) pairs.

| N_REGS | TOTAL_P2 | Sub-stages | Shuffles (j<32 steps × N × 2) |
|--------|----------|------------|-------------------------------|
| 1      | 32       | 15         | 15 × 1 × 2 = 30              |
| 2      | 64       | 21         | 20 × 2 × 2 = 80              |
| 4      | 128      | 28         | 25 × 4 × 2 = 200             |

### Streaming merge (optimization)

When `data_size > N_REGS × 32`, the data is processed in chunks:

1. **First chunk**: load all `N_REGS × 32` elements, run a **full** bitonic
   sort.  Discard positions beyond `topk` (set to sentinel).

2. **Subsequent chunks**: load new data into the spare register rows, then:
   - **Sort new chunk ascending** (`warp_bitonic_sort_range_ascending<M>`)
   - **Bitonic merge** (`warp_bitonic_merge_descending<N>`)

   The keep set (regs 0..keep_regs-1) is already sorted descending.
   The new chunk (regs keep_regs..N_REGS-1), once sorted ascending,
   forms a bitonic sequence with the keep set.  A single merge pass
   (the final stage of the bitonic network: `k = TOTAL_P2`, sub-stages
   `j = TOTAL_P2/2 .. 1`) produces a fully sorted descending result.

3. **Discard tail** after each merge: reset non-keep registers to sentinel.

**Why this is faster than a full sort per chunk:**

The full sort has `n*(n+1)/2` sub-stages.  The merge approach has:
- Sub-sort on M registers: `m*(m+1)/2` sub-stages (smaller network, fewer regs)
- Merge on N registers: `n` sub-stages (single stage of the full network)

For the primary case (N_REGS=4, keep_regs=2, new_regs=2):
- Full sort: 28 sub-stages × 200 shuffles = 200 shuffles
- Sub-sort(M=2): 21 sub-stages, 20 shuffle-steps × 2 regs × 2 = 80 shuffles
- Merge(N=4): 7 sub-stages, 5 shuffle-steps × 4 regs × 2 = 40 shuffles
- Total: 120 shuffles per streaming chunk (−40% vs full sort)

### Ascending sort for merge

`warp_bitonic_sort_range_ascending<M>` sorts registers `[start_r..start_r+M-1]`
in ascending order: **(value ASC, index DESC)**.

The direction logic is flipped vs descending:
- `ascending = ((local_vpos & k) == 0)` → this sub-sequence goes ascending
- `want_larger = (is_low != ascending)` — low position keeps smaller for
  ascending sub-sequences

For cross-register CAS in ascending mode:
```cpp
if (ascending)
  bitonic_cas_descending(vals[partner_r], ..., vals[r], ...);
  // ensures partner_r (high pos) ≥ r (low pos) → ascending
```

### Bitonic merge

`warp_bitonic_merge_descending<N>` runs a single merge pass (`k = TOTAL_P2`).
Since `vpos < TOTAL_P2` for all elements, `(vpos & k) == 0` is always true,
so direction is always **descending**.  This simplifies the inner loop:
no need to compute direction per-element.

## 2. Other Optimizations Applied

### 2a. Float arithmetic (was: double)

All warp reduce functions (`warp_reduce_on_shmem`, `masked_warp_reduce_on_shmem`)
and kernel arithmetic changed from `double` to `float`:

- `volatile double val` → `volatile float val`
- `static_cast<double>()` → `static_cast<float>()` throughout
- `double default_val` → `float default_val`
- Backward kernel: `double sum_fwd_input` → `float`, etc.

Rationale: router probabilities are float32.  Double precision wastes register
space and halves ALU throughput with no benefit for this workload.

### 2b. Init loop fusion

The forward kernel's init phase previously ran 2–3 separate loops over all
`num_experts` positions:
1. Clear `probs`, `routing_map`, and `intermediate_output` (3 global stores)
2. Load `logits` into shared memory (1 global load + 1 shmem store)

Now fused into a single loop.  Additionally, `intermediate_output` initialization
is skipped for pre-softmax and sigmoid paths (it gets fully overwritten in the
preprocess section), saving `E` global writes per token in those cases.

## 3. Performance Results

**Config**: B300 SXM6 (sm_103), 148 SMs, CUDA 13.1, PyTorch 2.11

| topk | E    | Fused (before) | Fused (after) | PyTorch ref | Ratio  |
|------|------|----------------|---------------|-------------|--------|
| 32   | 2304 | 0.698 ms       | 0.395 ms      | 0.339 ms    | 0.86×  |
| 64   | 2304 | 0.731 ms       | 0.533 ms      | 0.359 ms    | 0.67×  |

(4096 tokens, softmax, pre-softmax, float32, random input)

## 4. Remaining Optimization Opportunities

### Tier 2 (moderate refactoring)

**2a. Fuse online-softmax into topk streaming loop**

Instead of 4 passes over shared memory for softmax (max-reduce, exp, sum-reduce,
divide), then streaming topk from shmem, compute softmax on-the-fly during
the topk load using the "online softmax" trick (running max + running sum,
correction at end).  Eliminates ~3 of the 4 softmax shmem passes.

Expected impact: significant for the pre-softmax path where softmax is the
second-largest cost (~20-25% of kernel time).

**2b. Vectorized shared memory loads (float4)**

Each lane currently loads one float per shmem access.  Using `float4` loads
would quadruple throughput for the sequential shmem access patterns in softmax
and topk data loading.

**2c. Multi-warp per token for large E**

Currently 1 warp = 1 token.  For E=2304, each lane serializes 72 iterations
per softmax pass.  Using 2–4 warps cooperatively per token would parallelize
the serial loops at the cost of inter-warp synchronization.

### Tier 3 (algorithmic redesign)

**3a. Radix-based top-K (O(E) single pass)**

Replace bitonic streaming with a radix select.  Partition elements by bit
from MSB to LSB, tracking the count on each side.  After `log2(value_range)`
passes, the top-K boundary is identified and elements are selected.  This
is O(E) per token with a small constant — no streaming needed.

**3b. Tournament tree (O(E log K))**

Maintain a K-element min-heap in registers.  For each new element, compare
against the heap minimum.  If larger, replace and sift down in O(log K).
Total work: O(E log K).  For K=36, log K ≈ 5.2, giving ~12,000 comparisons
for E=2304 vs ~7,000 shuffles in the current streaming approach.  The
advantage is simpler control flow and no full-sort overhead, but the heap
operations are serial per lane.

### Performance bottleneck summary (after Tier 1)

For topk=32, E=2304, the kernel is at 0.86× PyTorch.  The remaining gap
comes from:

1. **Softmax overhead** (~20-25%): 4 serial passes over 2304 shmem elements
   per token, each with 72 iterations per lane.  PyTorch's cuDNN softmax
   parallelizes across tokens AND within tokens.

2. **Streaming topk** (~40-50%): Even with merge optimization, 34 streaming
   chunks × (sub-sort + merge) is substantial.  The merge alone has 7
   sub-stages with 4 registers, each requiring `__syncwarp()`.

3. **Global memory init/write** (~10-15%): Clearing and writing dense
   [num_tokens × num_experts] output tensors, even though only `topk` of
   `num_experts` positions are non-zero.

The fundamental tension: the fused kernel's advantage (avoiding global memory
round-trips between softmax and topk) is partially offset by the
single-warp-per-token execution model, which serializes work that
PyTorch's separate kernels can parallelize more aggressively.  The
fusion wins when E is small (the separate-kernel overhead dominates),
but struggles when E is large (the serial work dominates).
