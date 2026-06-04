## Summary

Optimizes the fused router CUDA kernels introduced in #2821 (`fused_topk_with_score_function` and `fused_score_for_moe_aux_loss`). Achieves significant bandwidth improvements for large expert counts and topk values while preserving identical performance for smaller configurations (e.g., E=256, topk=4).

**Key results (B300, float32, 8192 tokens):**
- Forward (E=2304, K=36, softmax): 673 → 964 GB/s (**+43%**)
- Backward (E=2304, K=36, softmax): 543 → 2766 GB/s (**+410%**)
- Forward (E=512, K=4): no regression (±0.3%)

## Changes

### Forward kernels
- **Persistent grid with async double-buffered prefetch:** `RawAsyncLoader<T>` uses `cp.async` (sm_80+) for non-blocking global→shmem loads. Occupancy-aware grid sizing (`compute_persistent_grid`) keeps all SMs saturated across multiple rounds.
- **Packed 8-bit radix histogram:** Reduces radix topk register usage from 32 to 4 registers by packing 16 bucket counts into 4×u32 with 8-bit fields. Eliminates local memory spill at large E.
- **Compile-time score function dispatch:** `ScoreFunc` template parameter with `if constexpr` removes runtime branches from the hot loop.
- **Simple kernel path for small topk:** When `topk < NVTE_RADIX_TOPK_THRESHOLD` (default 8), dispatches to a lightweight kernel matching the original structure — no async loader, no persistent grid — avoiding scheduling overhead that dominates at small K.

### Backward kernels
- **Two-pass fused design:** Pass 1 accumulates warp-level sums via register reduction + `warp_allreduce_sum`. Pass 2 computes per-element gradients using scalar helpers. Eliminates the `comp_buf` shared memory buffer (saves `E × warps × 4` bytes per block).
- **Double-buffered async loading:** All backward inputs (grad, activation, mask) loaded through `RawAsyncLoader` with always-on double buffering.

### Infrastructure
- `async_loader.h`: `RawAsyncLoader<T>`, `compute_persistent_grid()`, `choose_num_buffers()`, vectorized global store/fill helpers.
- `NVTE_RADIX_TOPK_THRESHOLD` env var (default 8): configurable naive↔radix crossover.
- Templated `warp_reduce_on_shmem<T, ReduceFuncType>` eliminates function-pointer overhead.

### Hardening
- Host-side: `num_tokens * num_experts <= INT_MAX`, `topk ∈ [1, E]`, `topk % group_topk == 0`
- Device-side: `assert(data_size <= kMaxExpertsRadixTopk)` in radix path
- Correct `cudaDevAttrMaxSharedMemoryPerMultiprocessor` for buffer-count decision
- Fix: single-buffer prefetch clobber when shmem is too tight for double buffering

## Compatibility

- **No regression for small configs:** The simple forward kernel path is an exact replica of the original kernel structure, ensuring E=256/topk=4 (common in standard MoE) performs identically.
- **All existing tests pass:** 891/891 `test_fused_router.py` tests pass, 117 skipped (fp8/multi-node).
- **No API changes:** Same Python/C++ interface, same output semantics.
- **Tunable:** Set `NVTE_RADIX_TOPK_THRESHOLD=0` to force radix everywhere, or `=16` to use naive for topk<16.

## Performance (B300 SXM6, sm_103, float32, 8192 tokens)

Effective bandwidth (GB/s) is computed as the minimum bytes that must be transferred to/from global memory for one kernel invocation, divided by the measured wall time. For example, the topk forward kernel reads logits (`T×E×dtype`) and writes probs (`T×E×dtype`), routing_map (`T×E×1`), and intermediate_output (`T×E×4`). This metric captures how well the kernel utilizes memory bandwidth — higher is better, with the device peak around 8 TB/s on B300. Config format is `num_experts/topk`.

<details>
<summary>Full benchmark table (softmax)</summary>

| kernel   | pass  | config  | before   | after           |
| -------- | ----- | ------- | -------- | --------------- |
| topk     | fprop | 512/4   | 1779     | 1784 (+0.3%)    |
| topk     | fprop | 512/8   | 798      | 904 (+13%)      |
| topk     | fprop | 512/22  | 514      | 924 (+80%)      |
| topk     | fprop | 512/36  | 499      | 908 (+82%)      |
| topk     | fprop | 2304/4  | 1803     | 1802 (0%)       |
| topk     | fprop | 2304/8  | 660      | 993 (+51%)      |
| topk     | fprop | 2304/22 | 602      | 972 (+61%)      |
| topk     | fprop | 2304/36 | 673      | 964 (+43%)      |
| topk     | bprop | 512/22  | 3391     | 5362 (+58%)     |
| topk     | bprop | 2304/36 | 543      | 2766 (+410%)    |
| aux_loss | fprop | 512/22  | 519      | 896 (+73%)      |
| aux_loss | fprop | 2304/36 | 645      | 891 (+38%)      |
| aux_loss | bprop | 512/22  | 5289     | 6155 (+16%)     |
| aux_loss | bprop | 2304/36 | 2272     | 4201 (+85%)     |

</details>

<details>
<summary>Full benchmark table (sigmoid)</summary>

| kernel   | pass  | config  | before   | after           |
| -------- | ----- | ------- | -------- | --------------- |
| topk     | fprop | 512/4   | 1728     | 1736 (+0.5%)    |
| topk     | fprop | 512/22  | 470      | 891 (+90%)      |
| topk     | fprop | 2304/36 | 639      | 798 (+25%)      |
| topk     | bprop | 512/22  | 3169     | 4398 (+39%)     |
| topk     | bprop | 2304/36 | 533      | 2274 (+327%)    |
| aux_loss | fprop | 512/22  | 475      | 912 (+92%)      |
| aux_loss | fprop | 2304/36 | 598      | 867 (+45%)      |
| aux_loss | bprop | 2304/36 | 1965     | 2757 (+40%)     |

</details>
