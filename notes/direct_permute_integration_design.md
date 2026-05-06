# Direct-Permute Integration Design

## Overview

Eliminate the staging buffer and permute kernel from the dispatch path by having
the S2G warp write directly to expert-grouped positions on the target rank's
buffer via NVLink, using precomputed `direct_write_map` addressing.

## Current Dispatch Path (Non-Fused)

```
allgather_routing_map → scan kernel → permute_preprocessing
                                         ↓
dispatch kernel (S2G writes to staging buffer on target rank via NVLink)
                                         ↓
device_sync (wait for all ranks to finish writing)
                                         ↓
permute kernel (staging → expert-grouped output, local HBM)
                                         ↓
(output: permuted_tokens, permuted_probs, padded_tokens_per_expert)
```

## Proposed Direct Path

```
allgather_routing_map → scan kernel → compute_direct_write_map (3 kernels)
                                         ↓
memset(target_buffer, 0) — async zero the output buffer (for padding)
                                         ↓
dispatch_direct kernel (S2G writes directly to expert-grouped positions via NVLink)
                                         ↓
device_sync
                                         ↓
(output: permuted_tokens, permuted_probs, padded_tokens_per_expert)
```

Eliminates: permute_preprocessing + permute kernel (~92 μs for H=512).
Adds: compute_direct_write_map (~259 μs currently, optimizable to ~30 μs).
Net: Currently slower due to unoptimized Kernel 1, but once optimized the
staging write + permute is fully replaced.

---

## Kernel Integration Design

### 1. S2G Warp Device Function — Template Parameter

```cuda
// In hybrid_ep_backend.cuh, S2G warp group device function:
template <bool DIRECT_PERMUTE, ...>
__device__ void s2g_warp_group_device_function(...) {
    // ...
    if constexpr (DIRECT_PERMUTE) {
        // Use direct_write_map to determine destination:
        // For each token being sent to target_rank:
        //   For each topk slot k of this token hitting target_rank:
        //     dest_row = direct_write_map[local_token_id * TOPK + k]
        //     TMA write to target_rank's expert_output_token[dest_row * H]
        //     TMA write to target_rank's expert_output_prob[dest_row] (scalar)
    } else {
        // Existing path: write to staging buffer at sparse_to_dense_map position
    }
}
```

### 2. Key Difference in S2G Write Pattern

**Current (staging):** One token → one write per target rank (contiguous 1 KB).
```
target_buffer[dense_index * H .. dense_index * H + H-1] = token_data
```

**Direct:** One token → multiple writes per target rank (one per active expert on that rank).
```
For each expert e on target_rank that this token is routed to:
    target_buffer[direct_write_map[t][k] * H .. + H-1] = token_data  (replicate)
    target_prob_buffer[direct_write_map[t][k]] = probs[t][e]          (scalar)
```

The token data is replicated to each expert slot it belongs to (multicast).
The prob is a scalar unique to each (token, expert) pair.

### 3. How S2G Knows Which Experts Are Active Per Token Per Target

The S2G warp currently reads `sparse_to_dense_map[token][rank]` to get the
staging buffer index. For direct-write, it needs:

- `direct_write_map[local_token_id, k]` for each topk slot k
- To determine which topk slots map to the current target_rank

**Approach A (simpler):** S2G warp reads the token's TOPK entries from
`global_routing_map[token * TOPK .. + TOPK]` and filters for current target rank.
For each match, looks up `direct_write_map[local_token_id * TOPK + k]`.

**Approach B (precomputed):** A per-target-rank mapping that directly says
"for token t, write to these positions on rank r." Would require an extra
data structure but avoids re-reading the routing map in S2G.

Recommendation: **Approach A** — re-reading TOPK int16 values (72 bytes) from
L1-cached global routing map is negligible cost. Avoids extra precomputation.

### 4. Buffer Layout on Target Rank

The target rank's output buffer is pre-allocated as:
```
expert_output_token: [num_permuted_tokens, H] — expert-grouped, padded
expert_output_prob:  [num_permuted_tokens]     — same layout, scalar per pos
```

Layout within the buffer:
```
[expert_0: padded_count_0 rows | expert_1: padded_count_1 rows | ... | expert_E-1]
```

The `padded_tokens_per_expert` (output of compute_direct_write_map) tells
the grouped GEMM where each expert's region starts (via cumsum internally).

### 5. Memset Zero Before Dispatch

Padding positions must be zero (the GEMM processes them but they contribute
nothing). With direct-write, padding positions are never written to — only
real token positions get data. So the buffer must be zeroed first.

```cuda
// Before dispatch_direct kernel:
cudaMemsetAsync(expert_output_token, 0, num_permuted_tokens * H * sizeof(dtype), stream);
cudaMemsetAsync(expert_output_prob, 0, num_permuted_tokens * sizeof(float), stream);
```

Cost: num_permuted_tokens * H * 2 bytes = ~4.5K * 512 * 2 ≈ 4.6 MB for NVL8 latent MoE.
At 8 TB/s HBM bandwidth: ~0.6 μs. Negligible.

### 6. Prob Handling

Probs are scalar (1 float per position). In the permute kernel today:
```cuda
permuted_probs[dest_id - 1] = probs[token_id * E_per_rank * R + local_rank * E + expert_e];
```

For direct-write, the S2G warp writes the prob alongside the token:
```cuda
// For each active expert e on target_rank for this token:
int dest = direct_write_map[local_t * TOPK + k];
// TMA write token data to target's expert_output_token[dest * H]
// TMA write (or regular store) prob to target's expert_output_prob[dest]
```

The prob source is `probs[token_id * (E_per_rank * R_per_node) + local_rank * E_per_rank + local_expert]`
— same as today's permute kernel. The S2G warp has access to the probs buffer
(already part of the dispatch pipeline).

---

## Combine (Backward) Path: combine_direct

### Forward Combine (combine_with_unpermute)

Currently:
```
expert_output (expert-grouped) → unpermute kernel (gather + sum) → combined output
                                → NVLink combine kernel reads permuted positions,
                                  writes to source rank's output
```

With direct-write, the combine kernel can read directly from expert-grouped
positions using the same `direct_write_map` (or the inverse `row_id_map`).

**However**, combine already reads from the permuted buffer — the unpermute
happens on the source (attention) rank AFTER receiving data. The combine S2G
warp writes the permuted data back to the source rank, and the source rank
runs unpermute to gather + sum.

For combine_direct: the combine kernel on the expert rank reads from
expert-grouped positions (same buffer) and sends over NVLink. The receiving
(attention) rank's G2S warp can accumulate directly into the output position.
**The unpermute kernel is eliminated because the combine kernel does the
gather-accumulate during the NVLink transfer.**

This is exactly what the existing combine kernel does (it reads from permuted
positions specified by row_id_map on the expert side). The difference is that
with direct-write, we use `direct_write_map` for addressing instead of `row_id_map`.

### Backward Dispatch (HybridEPCombine.backward → dispatch_with_permute)

In backward, `dispatch_with_permute(grad_combined)` re-permutes gradients.
The direct-write path works identically: the same `direct_write_map` is reused
(it's stored in the handle from forward).

### Backward Combine (HybridEPDispatch.backward → combine_with_unpermute)

In backward, `combine_with_unpermute(grad_expert_output, probs=grad_probs)`
gathers expert gradients back and unpermutes prob gradients. For direct-write,
the prob gradient unpermute can use `direct_write_map` inversely — but this
is conceptually the same as the current path.

---

## API Design

### Python Interface

```python
class HybridEPBuffer:
    def dispatch_with_permute(self, ..., direct_permute=False):
        """
        If direct_permute=True:
          - Runs compute_direct_write_map instead of permute_preprocessing
          - Dispatch kernel writes directly to expert-grouped positions
          - Returns same outputs as non-direct path
        """
        
    def combine_with_unpermute(self, ..., direct_permute=False):
        """
        If direct_permute=True:
          - Combine kernel reads from expert-grouped positions using direct_write_map
          - Unpermute kernel is skipped
        """
```

### C++ Interface Changes

```cpp
// HandleImpl — add direct_write_map field
struct HandleImpl {
    // ... existing fields ...
    torch::Tensor direct_write_map;  // [T_per_rank, TOPK] int32 (optional)
};

// Executor — new methods or mode flag
void Executor::dispatch_direct(HybridEpConfigInstance config, DispatchArgs& args);
void Executor::combine_direct(HybridEpConfigInstance config, CombineArgs& args);
```

### Buffer Allocation

`direct_write_map` is per-dispatch (changes each iteration with routing).
Allocated during metadata_preprocessing (same time as row_id_map today).
Size: T_per_rank * TOPK * 4 bytes = 8192 * 36 * 4 = 1.2 MB.

`position_counters` is a temporary: R * E * 4 bytes = 72 * 32 * 4 = 9 KB.
Can be allocated once and reused.

---

## Execution Timeline Comparison

### Current (NVL8, H=512, T=8192, E=32, K=36)

```
allgather:          ~50 us (routing map)
scan:               ~10 us
permute_preprocess: ~15 us (cooperative kernel)
device_sync:        ~5  us
dispatch S2G:       ~9  us (8 MB NVLink, staging)
device_sync:        ~5  us
permute:            ~92 us (our optimized ballot kernel)
────────────────────────────
Total:              ~186 us
```

### Direct Path (projected, after Kernel 1 optimization)

```
allgather:                  ~50 us (routing map)
scan:                       ~10 us
compute_direct_write_map:   ~30 us (optimized, projected)
memset output buffer:       ~1  us
device_sync:                ~5  us
dispatch_direct S2G:        ~40 us (36 MB NVLink, multicast to expert positions)
device_sync:                ~5  us
────────────────────────────
Total:                      ~141 us (projected)
```

Savings: ~45 μs per dispatch (24% reduction).
Additionally eliminates the staging buffer memory (8 MB per rank).

---

## Implementation Order

1. ~~**Optimize Kernel 1** (count_per_source) — reduce from 184 μs to ~20 μs~~
   **Deferred** — does not block correctness.
2. ~~**Template S2G device function** in hybrid_ep_backend.cuh with DIRECT_PERMUTE~~
   **DONE** — `if constexpr(DIRECT_PERMUTE)` path in S2G, single elected thread
   iterates TOPK entries, issues TMA per expert, scalar prob write.
3. ~~**Add dispatch_direct to executor.cu**~~ **DONE** — integrated into existing
   `dispatch_with_permute` with `direct_permute=True` flag:
   - Dedicated NVLink-accessible `direct_output_token/prob` buffers (IPC-shared)
   - Sized by `num_permuted_tokens_direct` (HybridEPBuffer ctor argument)
   - Output tensor is `torch::from_blob` view (zero-copy)
   - Assertion: `num_permuted_tokens <= num_permuted_tokens_direct`
4. ~~**Python API** (direct_permute=True flag on dispatch_with_permute)~~ **DONE**
   - `direct_permute=True` + `global_routing_map=` + `num_permuted_tokens=` (required)
   - `num_permuted_tokens_direct=` on `HybridEPBuffer` constructor
5. ~~**Test with multi-GPU**~~ **DONE** — `test_hybrid_ep_direct.py`:
   - Pure-torch reference (allgather + group-by-expert)
   - Verifies tokens AND probs on ALL ranks (256 experts total for NVL8)
   - Base-16 encoded 4-component token IDs (safe for bf16/fp8/mxfp8)
   - Result: 256/256 PASS on B200 NVL8 (H=512, T=8192, E=32, K=36)
6. **Combine path** (reuse same map for backward) — **NOT YET IMPLEMENTED**
7. **Kernel 1 optimization** — **NOT YET IMPLEMENTED**

---

## Buffer Management (Updated — Implemented)

The direct-permute path uses **dedicated pre-allocated NVLink-accessible buffers**
separate from the staging buffers:

```
IntraNodeDispatchBuffers:
  expert_output_token/prob       — existing staging buffer (used by non-direct path)
  direct_output_token/prob       — NEW: sized num_permuted_tokens_direct * H
  direct_output_token_all_ranks  — NEW: IPC pointers to each rank's direct buffer
  direct_output_prob_all_ranks   — NEW: IPC pointers to each rank's prob buffer
```

### Allocation Flow

1. User passes `num_permuted_tokens_direct` to `HybridEPBuffer` constructor
2. `BufferConfig.num_permuted_tokens_direct` is set before C++ buffer allocation
3. `allocate_dispatch_buffers()` allocates `direct_output_token/prob` via
   `remote_allocator` (NVLink-accessible), IPC-shares via handle exchange
4. Handles are at positions [5] and [6] in the dispatch handles tensor
5. `open_handles_from_other_ranks()` opens IPC handles for remote direct buffers

### Per-Call Flow (dispatch_with_permute, direct_permute=True)

1. Assert `num_permuted_tokens > 0` (must be explicit)
2. Assert `num_permuted_tokens <= num_permuted_tokens_direct` (buffer bounds)
3. `compute_direct_write_map()` — 3-kernel pipeline
4. `cudaMemsetAsync(direct_output_token, 0, ...)` — zero padding
5. `cudaMemsetAsync(direct_output_prob, 0, ...)`
6. `executor.dispatch_core()` — S2G uses `direct_output_token_all_ranks[]`
7. Return `torch::from_blob(direct_output_token, {num_permuted_tokens, H})`

### Size Requirements

- Buffer: `num_permuted_tokens_direct * H * sizeof(dtype) + num_permuted_tokens_direct * sizeof(float)`
- For NVL8 latent MoE (H=512, T=8192, E=32, K=36, capacity_factor=1.05):
  `309657 * 512 * 2 + 309657 * 4 ≈ 318 MB + 1.2 MB ≈ 319 MB` per rank
- The user sets this based on their capacity factor / token budget

---

## Probs: Scalar NVLink Access

### Dispatch (forward)

Token data uses TMA (`cp.async.bulk`) for efficient NVLink writes (1 KB per position).
Probs are a single float32 per position — too small for TMA (minimum 128B).

**Approach:** Use a regular global store (`st.global.relaxed.sys`) from register
to the target rank's NVLink-mapped `expert_output_prob[dest]`. A 4-byte store
over NVLink completes as a single flit. Since each S2G thread already has the
prob value in a register (loaded alongside the token's routing info), this is:
```cuda
// After TMA write of token data:
float prob_val = probs_src[token_id * E_total_local + local_rank * E_per_rank + expert_e];
float* remote_prob_addr = remote_expert_output_prob[target_rank] + dest_row;
asm volatile("st.relaxed.sys.global.f32 [%0], %1;" :: "l"(remote_prob_addr), "f"(prob_val));
```

Cost: negligible — one 4B store per active expert slot, pipelined with TMA writes.

### Combine (backward)

In combine, the G2S warp needs to read `permuted_probs[source_pos]` from the
expert rank's buffer over NVLink. Since it's a scalar, TMA is not suitable.

**Approach:** Use `cp.async` (not `cp.async.bulk`) to load 4B into SMEM, hiding
NVLink latency behind the token TMA loads:
```cuda
// Issue cp.async for prob alongside TMA for token:
cp_async_ca_shared_global<4>(&smem_prob[stage], &remote_prob[source_pos]);
// ... later, after mbarrier wait ...
float prob = smem_prob[stage];
```

Or batch probs into a larger `cp.async` if multiple probs for the same token
are contiguous (unlikely in direct path). Alternatively, just use `ld.global`
and rely on the fact that L2 caching hides most of the latency for repeated
accesses to the same 9 KB region of prob data.

### Memset for Probs

Yes — `expert_output_prob` must also be zeroed before dispatch (padding positions
need prob=0 so the GEMM's prob-weighted activation produces zeros):
```cuda
cudaMemsetAsync(expert_output_prob, 0, num_permuted_tokens * sizeof(float), stream);
```
Cost: num_permuted_tokens * 4B ≈ 18 KB for NVL8 latent MoE. Negligible.

---

## Open Questions

1. **NVLink multicast efficiency:** Writing 4.5 copies of each token (1 KB each)
   to scattered expert positions — does NVLink handle this efficiently vs one
   contiguous 1 KB write? The addresses are on the same target rank but different
   offsets. NVLink routes by address but each copy is a separate TMA operation.

2. **TMA granularity:** Each expert-position write is 1 KB (H=512, bf16). TMA
   minimum is 128B. So 1 KB = 8 TMA flits per write × 4.5 writes per token =
   36 flits per token. Currently: 8 flits per token (one write to staging).
   4.5x more NVLink traffic vs current staging path.

3. **When is direct-write a net win?** When `compute_direct_write_map + multicast_write`
   < `staging_write + permute_kernel`. For H=512: the permute is expensive (92 μs)
   so direct-write wins. For H=7168: permute is ~1 ms (BW-bound on local HBM),
   and direct NVLink multicast would be ~4.5x more NVLink BW — may be worse.
   **Direct-write is optimal for latent MoE (small H, sparse routing).**

4. **Combine path with direct addressing:** Does combine need its own precomputed
   map, or can it reuse `direct_write_map` from the forward? The combine reads
   from permuted positions (same buffer the dispatch wrote to) — so yes, the same
   `direct_write_map` gives the source positions for combine's gather operation.
   The combine G2S warp on the attention rank reads: for each of my tokens,
   for each topk slot k, gather from `expert_output[direct_write_map[t][k]]`
   on the source (expert) rank and accumulate locally.
   **Direct-write is optimal for latent MoE (small H, sparse routing).**
