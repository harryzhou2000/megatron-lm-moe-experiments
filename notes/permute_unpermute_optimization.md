# Permute / Unpermute Kernel Optimization

Ballot-based expert skip for DeepEP hybrid-ep permute and unpermute kernels.

**Branch**: `hhanyu/hybrid-ep-sparse-opt`
**Hardware**: B300 NVL8 SXM6 (8 TB/s HBM, 900 GB/s NVLink)
**Workload**: T=8192 tokens/rank, E=32 local experts, TOPK=36, bf16

## Problem Statement

The reference permute/unpermute kernels iterate over all `num_of_local_experts`
(32) routing slots per token using a serial loop, even though only ~4.5 experts
are active per token on average (TOPK=36 across R=8 ranks → 36/8 ≈ 4.5 per
rank). This wastes ~86% of loop iterations on zero-valued routing entries.

Additionally, the reference uses a fixed 128-thread "extended warp" per token,
which is excessive for small hidden sizes (H=512 → 32 float4 loads per token).

## Optimization: Ballot + FFS Skip

### Core idea

Replace the serial `for (i = 0; i < num_of_local_experts; i++)` loop with:

1. **`__ballot_sync`**: Each lane in the first warp evaluates whether its
   routing entry is active (`dest_token_id > 0` for permute, `src_id > 0` for
   unpermute). This produces a 32-bit bitmask of active experts.
2. **`__ffs` iteration**: Loop via `while (mask) { e = __ffs(mask) - 1; mask &= mask - 1; ... }`
   to visit only the set bits (active experts).
3. **`__shfl_sync`**: For THREADS_PER_TOKEN=32, broadcast the dest/src token ID
   from the lane that owns expert `e` to all threads in the warp, avoiding
   shared memory entirely.

### Configurable THREADS_PER_TOKEN

Thread count is selected at launch time based on `hidden_size_fp4` (number of
float4 elements per token = H / (16/sizeof(dtype))):

| hidden_size_fp4 | THREADS_PER_TOKEN | Rationale |
| --------------- | ----------------- | --------- |
| ≤ 64            | 32 (1 warp)       | Few loads per thread; maximize tokens in flight |
| ≤ 128           | 64 (2 warps)      | Moderate ILP needed |
| > 128           | 128 (4 warps)     | Saturate HBM bandwidth for large H |

For H=512 bf16: `hidden_size_fp4 = 512 / 4 = 128` → but sizeof(bf16)=2, so
`float4` holds 8 elements → `hidden_size_fp4 = 512 / 8 = 64` → 32 threads.

For H=7168 bf16: `hidden_size_fp4 = 7168 / 8 = 896` → 128 threads.

### No shared memory needed (32-thread path)

When THREADS_PER_TOKEN=32:
- All 32 expert routing entries fit in a single warp (lane `i` reads `row_id_map[token * E + i]`)
- `__ballot_sync` computes active mask across the full expert set
- `__shfl_sync(0xFFFFFFFF, my_dest, e)` distributes the dest token ID
- Zero SMEM usage → higher occupancy

For 64/128-thread groups, routing is read directly from GMEM (L1-cached) since
only the first warp can participate in ballot. Each warp independently computes
the active_mask from the same routing data.

## Performance Results

### Permute (scatter: 1 GMEM read → K coalesced writes per token)

| Config | Reference (128 thr, serial) | Optimized (ballot) | Speedup |
| ------ | --------------------------- | ------------------ | ------- |
| H=512  | 381 μs | **92 μs** | **4.1x** |
| H=7168 | 937 μs | **912 μs** | 1.03x (parity) |

### Unpermute (gather+reduce: K scattered reads → 1 coalesced write per token)

| Config | Reference (128 thr, serial) | Optimized (ballot) | Speedup |
| ------ | --------------------------- | ------------------ | ------- |
| H=512  | 265 μs | **128 μs** | **2.1x** |
| H=7168 | 1475 μs | **1098 μs** | **1.34x** |

### Why unpermute gains less than permute

- **Permute** is a pure scatter: read 1 token (coalesced), write K copies.
  The reference wastes time evaluating 32 routing slots; the ballot path
  directly jumps to the ~4.5 active ones. For H=512, the reduction in
  instruction count dominates since data movement is small.

- **Unpermute** is a gather+accumulate: read K source tokens (scattered
  addresses), accumulate in registers, write 1 output token. The bottleneck
  is the K random GMEM reads (~4.5 × 64B float4 loads from random positions).
  Ballot reduces the loop overhead but not the memory latency.

- For H=7168, unpermute's 1.34x comes from: (a) skipping empty-slot
  accumulator iterations, and (b) fewer predicated branches in the inner
  loop. The kernel is still memory-bandwidth-bound.

## Speed of Light Analysis

### Permute SOL (H=512)

- Data movement: 8192 tokens × 512 × 2B (read) + 8192 × 4.5 × 512 × 2B (write) = 8 MB + 37.7 MB ≈ 46 MB
- B300 HBM: 8 TB/s → theoretical = 46 MB / 8 TB/s ≈ **5.7 μs**
- Measured: 92 μs → **16x above SOL**
- Reason: writes are scattered (not coalesced per-expert), L2 sector waste

### Unpermute SOL (H=512)

- Data movement: 8192 × 4.5 × 512 × 2B (scattered reads) + 8192 × 512 × 2B (coalesced write) = 37.7 MB + 8 MB ≈ 46 MB
- Theoretical: 46 MB / 8 TB/s ≈ **5.7 μs**
- Measured: 128 μs → **22x above SOL**
- Reason: scattered reads cause L2 cache thrashing, each float4 load pulls a full 128B sector

## Fallback / Debug

Reference kernels are preserved and selectable at runtime:

```bash
HYBRID_EP_USE_PERMUTE_REFERENCE=1    # use permute_kernel_reference
HYBRID_EP_USE_UNPERMUTE_REFERENCE=1  # use unpermute_kernel_reference
```

## Commits

| Hash | Message |
| ---- | ------- |
| `1c3989c` | [DeepEP] Optimize permute kernel with ballot-based expert skip |
| `7ec3238` | [DeepEP] Optimize unpermute kernel with ballot-based expert skip |

## Key Takeaways

1. **Ballot intrinsics eliminate branch divergence** for sparse routing patterns.
   With ~86% of expert slots empty, skipping them via bitmask iteration is a
   massive win for instruction-bound kernels (small H).

2. **Thread count must match workload**: 128 threads/token wastes occupancy when
   H=512 only needs 32 threads to cover all float4 loads. Fewer threads per
   token = more tokens per SM = better latency hiding.

3. **Shuffles replace shared memory** for the 32-thread path. No SMEM barriers
   needed → simpler code, higher occupancy.

4. **Diminishing returns at large H**: when the kernel becomes memory-bound
   (H=7168), reducing loop iterations helps modestly but can't beat the
   fundamental scattered-access latency.
