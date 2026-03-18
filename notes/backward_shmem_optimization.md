# Backward Kernel Shared Memory Optimization

## Status: Deferred

## Problem Summary

The topk backward kernel (`fused_topk_with_score_function_backward_kernel`) has two
performance issues causing it to be slower than PyTorch for large expert counts,
particularly for sqrtsoftplus.

## Issue 1: Sqrtsoftplus recomputes the forward activation

For sqrtsoftplus, `intermediate_output` stores the **original logits** (not the
activation output). The backward must recompute `sqrtsoftplus(x)` for all E experts:

```cpp
// Lines 354-362 in fused_topk_with_score_function.cu
if (score_function == 2) {
  // Copy original logits to local_comp_buf and apply sqrtsoftplus in-place
  for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
    local_comp_buf[i] = local_act_from_fwd[i];
  }
  __syncwarp();
  apply_sqrtsoftplus_on_float(local_comp_buf, num_experts, lane_id);  // expf + log1pf + sqrtf
  __syncwarp();
}
```

This costs `expf()`, `log1pf()`, `sqrtf()` per element — expensive transcendentals on
all E elements per token. By contrast:
- **Sigmoid**: `intermediate_output` already stores `sigmoid(x)` — zero recomputation
- **Softmax**: `intermediate_output` already stores `softmax(x)` — zero recomputation

### Potential fix

Store `sqrtsoftplus(x)` in `intermediate_output` (like sigmoid does), then recover
`sigmoid(x)` in the backward from `y = sqrtsoftplus(x)`:
```
softplus(x) = y²
sigmoid(x) = 1 - exp(-y²)
```
This replaces `expf + log1pf + sqrtf` with a single `expf`, and eliminates the
extra shmem copy + pass.

## Issue 2: `comp_buf` wastes shared memory → reduces occupancy

The backward allocates **3 × E** float buffers per warp plus routing_map bools:

| Buffer            | Size per warp | Used by                         |
| ----------------- | ------------- | ------------------------------- |
| `grad_probs_buf`  | E × 4 bytes   | All paths                       |
| `act_from_fwd_buf`| E × 4 bytes   | All paths                       |
| `comp_buf`        | E × 4 bytes   | softmax_bwd scratch, sqrtsoftplus recompute |
| `routing_map_buf` | E × 1 byte    | All paths                       |

For E=2304, 8 warps/block:
```
shmem = 3 × 2304 × 8 × 4 + 2304 × 8 × 1 = 239,616 bytes
```

This is massive — limits occupancy to 1 block per SM. The `comp_buf` is entirely
unused for sigmoid and pre-softmax paths, wasting 33% of shmem.

### Potential fix

- **Sigmoid/pre-softmax**: Skip `comp_buf` allocation entirely (or use 2-buffer layout)
- **Sqrtsoftplus**: If we store activation output in `intermediate_output` (Issue 1 fix),
  `comp_buf` is no longer needed for recomputation
- **Softmax bwd**: Could potentially use register-based accumulation instead of scratch buffer

Eliminating `comp_buf` cuts shmem by 33%:
```
shmem = 2 × 2304 × 8 × 4 + 2304 × 8 × 1 = 165,888 bytes
```
This may allow higher occupancy or more warps per block.

## Performance Data (B300 SXM6, sm_103)

Backward benchmark, format: `fused_ms / pytorch_ms = speedup`

### Sqrtsoftplus (~2x slower than sigmoid/softmax fused kernel)

| Tokens  | Experts | Topk | Fused (ms) | PyTorch (ms) | Speedup |
| ------- | ------- | ---- | ---------- | ------------ | ------- |
| 8192    | 2304    | 8    | 0.94       | 0.55         | 0.59x   |
| 8192    | 2304    | 32   | 0.95       | 0.54         | 0.57x   |
| 32768   | 2304    | 8    | 2.88       | 0.93         | 0.32x   |
| 32768   | 2304    | 32   | 2.93       | 0.94         | 0.32x   |
| 131072  | 2304    | 8    | 10.58      | 2.62         | 0.25x   |
| 131072  | 2304    | 32   | 10.79      | 2.71         | 0.25x   |

### Sigmoid (better, but still slow at large E)

| Tokens  | Experts | Topk | Fused (ms) | PyTorch (ms) | Speedup |
| ------- | ------- | ---- | ---------- | ------------ | ------- |
| 8192    | 2304    | 8    | 0.66       | 0.51         | 0.77x   |
| 8192    | 2304    | 32   | 0.68       | 0.49         | 0.73x   |
| 32768   | 2304    | 8    | 1.85       | 0.69         | 0.37x   |
| 32768   | 2304    | 32   | 1.90       | 0.71         | 0.37x   |
| 131072  | 2304    | 8    | 6.53       | 1.61         | 0.25x   |
| 131072  | 2304    | 32   | 6.71       | 1.71         | 0.25x   |

### Softmax (similar to sigmoid)

| Tokens  | Experts | Topk | Fused (ms) | PyTorch (ms) | Speedup |
| ------- | ------- | ---- | ---------- | ------------ | ------- |
| 8192    | 2304    | 8    | 0.65       | 0.44         | 0.68x   |
| 8192    | 2304    | 32   | 0.67       | 0.43         | 0.64x   |
| 32768   | 2304    | 8    | 1.89       | 0.52         | 0.27x   |
| 32768   | 2304    | 32   | 1.89       | 0.52         | 0.28x   |
| 131072  | 2304    | 8    | 6.62       | 1.04         | 0.16x   |
| 131072  | 2304    | 32   | 6.62       | 1.14         | 0.17x   |

### Key observations

- All score functions regress at E=2304 due to shmem pressure (3×E buffers)
- Sqrtsoftplus is ~1.6x slower than sigmoid/softmax fused, due to forward recomputation
- PyTorch backward scales much better because it uses optimized element-wise kernels
  (no shmem bottleneck, high occupancy)
- At small E (≤512), the fused kernel is competitive (0.9-1.2x) for all score functions
- The regression is proportional to E because shmem = O(E × warps)

## Related: Stashed optimizations (from prior sessions)

These were developed but not yet applied:
- `_with_writeback` variants of activation functions (fuse activation + global write)
- `comp_buf` elimination from backward shmem layouts
- Init loop fusions
- Aux loss backward replaced with PyTorch-native ops
