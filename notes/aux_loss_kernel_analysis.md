# Fused MoE Auxiliary Loss — Kernel Analysis

## Overview

The MoE auxiliary (load-balancing) loss encourages even distribution of tokens
across experts.  TE provides two fused CUDA kernels for this:

1. **`fused_score_for_moe_aux_loss`** — Computes per-expert scores and routing
   maps from logits (forward + backward).  This is a *subset* of
   `fused_topk_with_score_function`: it applies the score function (softmax or
   sigmoid), runs top-K to get the routing map, and outputs the full score
   matrix.  Unlike the main routing kernel, it does **not** produce
   probability-weighted gated outputs — it outputs raw scores for all experts.

2. **`fused_moe_aux_loss`** — Computes the scalar auxiliary loss value from
   the per-expert scores and token counts (forward + backward).

### Python API (`router.py`)

```python
# Step 1: get scores + routing map
routing_map, scores = fused_compute_score_for_moe_aux_loss(logits, topk, score_function)

# Step 2: compute the loss
aux_loss = fused_moe_aux_loss(probs, tokens_per_expert, total_num_tokens,
                              num_experts, topk, coeff)
```

---

## Kernel 1: `fused_score_for_moe_aux_loss`

### File
`TE/transformer_engine/common/fused_router/fused_score_for_moe_aux_loss.cu`

### Forward kernel

**Launch config**: Same as `fused_topk_with_score_function` — 4 warps/block
(128 threads), one warp per token, grid-stride loop.

**Shared memory layout** (per block, 4 warps):
- `logits_buf`:       E × 4 × sizeof(DataType)  — per-warp logits/scores
- `topk_logits_buf`:  K × 4 × sizeof(DataType)  — per-warp topk scores
- `topk_indices_buf`: K × 4 × sizeof(int)        — per-warp topk indices

**Algorithm per token** (single warp):

1. **Init**: Load logits from global → shmem. Clear `routing_map[E]` in global.
   For softmax: also init `intermediate_output[E]` to -inf.

2. **Preprocess** (in shmem):
   - **Softmax** (score_function=1): Apply softmax to `local_logits[E]`,
     write result to `intermediate_output[E]` in global (for backward).
   - **Sigmoid** (score_function=0): Apply sigmoid, save to
     `intermediate_output[E]`.  Then if topk > 1: normalize all E scores by
     dividing by their sum (creates a probability distribution over all experts,
     not just the top-K — this differs from the main routing kernel which
     normalizes only the top-K).

3. **Top-K**: `naive_topk_and_mask_v2(local_logits, E, K)` — radix selection
   over the (possibly normalized) scores.

4. **Output**:
   - Write `routing_map[topk_indices] = true` (K positions in global)
   - Write `scores[E] = local_logits[E]` — the **full** score vector (all E
     experts), not just top-K.  This is the key difference from the main
     routing kernel.

### Backward kernel

**Shared memory**: 3 × E × 4 × sizeof(DataType) — grad, act_from_fwd, comp_buf
per warp. (No routing_map buffer — the backward doesn't need it.)

**Algorithm per token**:

1. **Init**: Load `grad_scores[E]` and `intermediate_output[E]` from global
   to shmem. Clear `grad_logits[E]` in global.

2. **Sigmoid post-process backward** (if sigmoid + topk > 1):
   Computes the Jacobian of the all-expert normalization:
   ```
   grad[i] = grad[i] / (sum_fwd + eps) - sum(grad * act) / (sum_fwd + eps)^2
   ```
   Uses `warp_reduce_on_shmem` for the two reductions.

3. **Softmax backward** (if softmax): Standard softmax backward via
   `apply_softmax_bwd_on_float` over all E experts (no mask — all positions).

4. **Sigmoid backward** (if sigmoid): Element-wise `grad * σ(x) * (1 - σ(x))`
   via `apply_sigmoid_bwd_on_float`.

5. **Write**: `grad_logits[E] = local_grad[E]` to global.

### Key difference from main routing kernel

| Aspect | `fused_topk_with_score_function` | `fused_score_for_moe_aux_loss` |
|--------|----------------------------------|-------------------------------|
| Output scores | Only top-K positions (gated probs) | All E experts (raw scores) |
| Sigmoid normalization | Over top-K only (in postprocess) | Over all E experts (in preprocess) |
| Expert bias | Supported | Not supported |
| Group topk | Supported | Not supported |
| Backward topk | Mask gradient to top-K positions | No masking (grad flows to all E) |
| `intermediate_output` init | Only for post-softmax path | Always for softmax |

---

## Kernel 2: `fused_moe_aux_loss`

### File
`TE/transformer_engine/common/fused_router/fused_moe_aux_loss.cu`

### The loss formula

The MoE load-balancing auxiliary loss is:

```
aux_loss = C_coeff * sum_j( aggregated_probs[j] * tokens_per_expert[j] )
```

where:
- `aggregated_probs[j] = sum_i( probs[i, j] )` — sum of probability for expert
  j across all tokens
- `tokens_per_expert[j]` — count of tokens routed to expert j
- `C_coeff = (num_experts * coeff) / (topk * total_num_tokens^2)`

### Forward kernel

Two architecture-specific paths:

**sm_90+ (Hopper/Blackwell): Cluster-based reduction**

Uses CUDA thread block clusters (`cooperative_groups::cluster_group`) with up
to 8 blocks sharing distributed shared memory.

1. Each block/warp accumulates `sum_i(probs[i, j])` for its assigned rows
   into per-block shmem using `atomicAdd` (double precision).
2. Cluster sync, then block 0 reduces across all cluster blocks via
   `cluster.map_shared_rank()` — reads remote shmem directly.
3. Multiply `aggregated_probs[j] *= tokens_per_expert[j]`.
4. Warp 0 reduces the E-element product array to a scalar via
   `warp_reduce_on_shmem`.
5. Lane 0 computes `aux_loss = result * C_coeff` and saves `C_coeff` for
   backward.

**Pre-sm_90: Single-block fallback**

Uses a single block of 1024 threads.  Same algorithm but without cluster
synchronization — all warps in one block cooperate via `__syncthreads()` and
`atomicAdd` to shmem.

**Computation type**: `double` throughout (`using CompType = double`) to avoid
accumulation errors when summing over many tokens.

**Shared memory**: `num_cols × sizeof(double)` — the aggregated probs vector.

### Backward kernel

The backward is trivially parallel — each element gets the same gradient:

```
grad_probs[i, j] = C_coeff * tokens_per_expert[j] * grad_aux_loss
```

**Launch config**: Grid of 256-thread blocks, one warp handles a column stripe,
grid-stride over rows.

No shared memory needed — reads `C_coeff` and `tokens_per_expert` from global,
broadcasts `grad_aux_loss` scalar.

---

## Optimization notes

### Current state
- The aux loss score kernel now uses `naive_topk_and_mask_v2` (radix selection)
  and `apply_softmax_on_float_with_writeback` (online softmax with fused
  write-back), matching the optimizations in the main routing kernel.

### Potential improvements
1. **Sigmoid backward in `fused_score_for_moe_aux_loss`**: Still uses
   `static_cast<double>` in the normalization backward. Could switch to float
   for consistency (same as the main routing kernel).

2. **`fused_moe_aux_loss` forward**: The single-block fallback (pre-sm_90) is
   limited to 1024 threads = 32 warps.  For large token counts (131K+), each
   warp processes ~4K rows serially.  A multi-block reduction with atomic
   global accumulation could help on older architectures.

3. **`fused_moe_aux_loss` backward**: Already embarrassingly parallel. Could
   use vectorized stores (`float4`) for the output.

4. **Init loop in score kernel**: Two separate loops (clear routing_map, load
   logits) could be fused into one, matching the optimization done in the main
   routing kernel.
