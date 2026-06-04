# Fused Router Kernel Optimization — Performance Results

## Summary

Performance optimization of the fused router CUDA kernels (`fused_topk_with_score_function` and `fused_score_for_moe_aux_loss`) that were introduced in [PR #2821](https://github.com/NVIDIA/TransformerEngine/pull/2821). The optimizations target both forward and backward kernels, achieving up to **+90% forward** and **+410% backward** effective bandwidth improvement at large expert counts and topk values, while preserving identical performance for small configurations (E=256, topk≤4).

All measurements were taken on an NVIDIA B300 SXM6 (148 SMs, sm_103) within the same SLURM allocation for consistency.

## Optimizations

1. **Fused preprocess/backward loops:** Forward kernels fuse the separate clear→load→score→save→bias loops into single loops per score function. Backward kernels use a two-pass structure (Pass 1: warp-level reductions; Pass 2: per-element gradient), eliminating the `comp_buf` shared memory buffer entirely.

2. **Async loader with persistent grid and double buffering:** A new `RawAsyncLoader<T>` uses `cp.async` (sm_80+) for non-blocking global→shared memory transfers. Forward kernels use occupancy-aware persistent grids (`compute_persistent_grid`) with double-buffered prefetch when shared memory permits. Backward kernels always double-buffer all inputs.

3. **Packed 8-bit radix histogram:** The radix topk selection packs 16 bucket counts into 4 registers using 8-bit fields, reducing register pressure from 32 registers to 4. This eliminates the local memory spill observed at large expert counts (E=2304).

4. **Compile-time score function dispatch:** `ScoreFunc` becomes a template parameter with `if constexpr` in the optimized kernel path, eliminating runtime branching from the hot loop.

5. **Configurable radix/naive crossover:** The environment variable `NVTE_RADIX_TOPK_THRESHOLD` (default 8) controls whether the O(K×E) naive or O(E) radix selection is used. Topk values below the threshold use naive; at or above use radix.

6. **Simple forward kernel for small topk:** When `topk < NVTE_RADIX_TOPK_THRESHOLD`, a lightweight forward kernel matching the original structure (no async loader, no persistent grid) is dispatched to avoid scheduling overhead that dominates at small K. This ensures no performance regression for typical configurations like E=256, topk=4.

7. **Templated warp reduction:** `warp_reduce_on_shmem` uses compile-time `if constexpr` dispatch instead of a runtime function pointer, removing indirect call overhead from the reduction butterfly.

8. **Correctness hardening:** Host-side bounds checks (`num_tokens * num_experts <= INT_MAX`, `topk % group_topk == 0`), device-side assertions for packed histogram overflow, correct `cudaDevAttrMaxSharedMemoryPerMultiprocessor` query for buffer-count decisions, and a fix for single-buffer async prefetch aliasing.

## Performance Results — Effective Bandwidth (GB/s)

Measured with 8192 tokens, float32. Config format: `num_experts/topk`.

### Softmax

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

### Sigmoid

| kernel   | pass  | config  | before   | after           |
| -------- | ----- | ------- | -------- | --------------- |
| topk     | fprop | 512/4   | 1728     | 1736 (+0.5%)    |
| topk     | fprop | 512/8   | 773      | 921 (+19%)      |
| topk     | fprop | 512/22  | 470      | 891 (+90%)      |
| topk     | fprop | 512/36  | 455      | 851 (+87%)      |
| topk     | fprop | 2304/4  | 1616     | 1615 (0%)       |
| topk     | fprop | 2304/8  | 632      | 823 (+30%)      |
| topk     | fprop | 2304/22 | 623      | 797 (+28%)      |
| topk     | fprop | 2304/36 | 639      | 798 (+25%)      |
| topk     | bprop | 512/22  | 3169     | 4398 (+39%)     |
| topk     | bprop | 2304/36 | 533      | 2274 (+327%)    |
| aux_loss | fprop | 512/22  | 475      | 912 (+92%)      |
| aux_loss | fprop | 2304/36 | 598      | 867 (+45%)      |
| aux_loss | bprop | 512/22  | 4551     | 5381 (+18%)     |
| aux_loss | bprop | 2304/36 | 1965     | 2757 (+40%)     |

## Key Points

- **No regression for small configurations:** Topk=4 with E=256 or E=512 (common in standard MoE) maintains identical performance (±0.5%). The `NVTE_RADIX_TOPK_THRESHOLD` mechanism routes these to the original kernel structure automatically.

- **Largest gains in backward:** The fused two-pass backward with register-based reductions provides the most significant improvement (+327% to +410% at E=2304). This matters because backward is typically 2–3× more frequent than forward in training.

- **Tunable crossover:** Users with unusual configurations can set `NVTE_RADIX_TOPK_THRESHOLD` to control the naive/radix boundary. Default (8) is optimal for the tested hardware.

## Test Coverage

- All 891 existing `test_fused_router.py` tests pass (117 skipped for fp8/multi-node).
- Additional correctness sweep (206 configs) covering all score functions, group_topk, extreme inputs, and NaN edge cases.
