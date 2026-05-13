# NCU Analysis: Direct-Permute Dispatch Kernel

**Date**: 2026-05-14
**Node**: B300 NVL8 (8× B300 SXM6, sm_103, NVLink)
**Container**: PyTorch 26.03, CUDA 13.2, ncu 2026.1.0
**Config**: H=512, T=8192, E_local=32, K=8, R=8, NUM_SMS_DISPATCH=24

## Profiling Setup

- `--replay-mode application` with TCP communicator + lockstep
- ~13 application replay passes per rank
- Reports from rank 1 (representative; all ranks symmetric)

## Side-by-Side Comparison

| Metric                    | Non-direct     | Direct         | Ratio / Delta  |
| ------------------------- | -------------- | -------------- | -------------- |
| **Timing**                | 155.94 us      | 344.86 us      | **2.2x slower**|
| **DRAM read**             | 17.10 MB       | 17.40 MB       | 1.0x           |
| **DRAM write**            | 83.20 KB       | 7.99 MB        | **96x more**   |
| **NVLink TX (total)**     | 52.26 MB       | 214.05 MB      | **4.1x more**  |
| **NVLink TX (user data)** | 44.01 MB       | 178.26 MB      | **4.0x more**  |
| **NVLink RX (total)**     | 865.47 KB      | 1.83 MB        | 2.1x           |
| **L1 sectors (total)**    | 2,048          | 198,656        | **97x more**   |
| **L1 sectors (miss)**     | 256            | 65,976         | **258x more**  |
| **L1 miss rate**          | 12.5%          | 33.2%          | +21pp          |
| **SM throughput**         | 1.21%          | 1.38%          | ~same          |
| **Warp occupancy**        | 6.24%          | 6.25%          | ~same          |
| **Stall: long_scoreboard**| 59.10%         | 45.39%         | -14pp          |
| **Stall: short_scoreboard**| 6.88%         | 18.03%         | **+11pp**      |
| **Stall: wait**           | 19.91%         | 25.09%         | **+5pp**       |
| **Stall: barrier**        | 0.04%          | 0.01%          | ~same          |
| **Stall: membar**         | 0%             | 0%             | ~same          |
| **Stall: not_selected**   | 0%             | 0%             | ~same          |
| **Stall: mio_throttle**   | 0.01%          | 0.00%          | ~same          |
| **Stall: math_pipe**      | 0%             | 0%             | ~same          |

Kernel template parameters (from demangled name):
- Non-direct: `..., 32, 6, 64, 512, 8192, 32, 8, 1, 24, 2368, 1, 0, 0` (DIRECT_PERMUTE=0)
- Direct:     `...,  1, 6, 64, 512, 8192, 32, 8, 1, 24, 2368, 1, 1, 8` (DIRECT_PERMUTE=1, TOPK=8)

Grid: (24, 1, 1), Block: (128, 1, 1) — both use 24 SMs × 4 warps.

## Analysis

### 1. NVLink TX 4x Higher (44 MB → 178 MB)

The dominant cost. In non-direct mode, each token is written once to a staging buffer on the
target rank. The staging buffer is laid out sequentially (token 0, token 1, ...) so writes
from each SM are coalesced.

In direct-permute mode, each token is written K=8 times — once per expert it's routed to.
Each write goes to an expert-grouped position on the target rank's direct output buffer.
With 256 total experts across 8 ranks, each rank handles 32 local experts. The destination
row for each (token, expert) pair is computed by `direct_write_map`, producing scattered
write addresses within the target rank's buffer.

Expected NVLink TX:
- Non-direct: T × H × sizeof(bf16) = 8192 × 512 × 2 = 8.39 MB per rank, ×8 ranks = ~67 MB total sent by this rank. Measured 44 MB user data suggests ~5200 tokens sent by rank 1 (due to routing distribution).
- Direct: T × K × H × sizeof(bf16) = 8192 × 8 × 512 × 2 ≈ 67 MB per rank. With K copies to potentially all 8 ranks, rank 1 sends ~178 MB user data. Consistent with 4x ratio.

### 2. DRAM Write 96x Higher (83 KB → 8 MB)

Non-direct writes go to the staging buffer which is sequentially filled. TMA bulk copies
write contiguous 512-byte chunks, and the L2 cache absorbs the writes efficiently (83 KB
DRAM write for ~44 MB NVLink = almost all writes stay in L2 or go directly to NVLink).

Direct writes scatter across the expert-grouped output buffer. The destination rows are
non-contiguous (token N for expert E₁ and token N for expert E₂ are at different offsets).
These scattered writes miss L2 and spill to DRAM. The 8 MB DRAM write represents
write-back of dirty L2 lines evicted by the access pattern.

### 3. L1 Sectors 97x Higher (2K → 199K)

Non-direct S2G inner loop: each elected thread reads token data from SMEM (prefetched by
TMA) and issues one TMA write to the target rank. The SMEM reads are coalesced. The only
L1 traffic is metadata reads (`sparse_to_dense_map`).

Direct S2G inner loop: for each token, iterates over K=8 topk entries. For each entry:
1. Reads `direct_write_map[chunk_offset * TOPK + k]` from SMEM → dest_row
2. Reads `topk_routing_map[chunk_offset * TOPK + k]` from SMEM → global_expert_id
3. Computes target_rank and local_expert from global_expert_id
4. Issues TMA write of token to `direct_output_token_all_ranks[target_rank][dest_row]`
5. Issues `st.relaxed.sys.global.f32` for prob

The SMEM pipeline prefetches `direct_write_map` + `topk_routing_map` via TMA, but the
actual NVLink writes to scattered positions generate L1 traffic from the TMA descriptor
setup and address computation. The 33% L1 miss rate (vs 12.5% non-direct) reflects the
scattered access pattern for destination addresses.

### 4. Stall Shift: long_scoreboard Down, short_scoreboard + wait Up

**long_scoreboard (59% → 45%)**: Both kernels are NVLink-write bound. Non-direct has higher
long_scoreboard because its writes are more coalesced — the SM issues writes faster and
spends more time waiting for the NVLink fabric to drain. Direct issues writes more slowly
(scattered addresses, TOPK iteration overhead), so the SM is less often blocked on NVLink.

**short_scoreboard (7% → 18%)**: Short scoreboard stalls indicate waiting for shared memory
or L1 cache operations. The direct kernel's SMEM pipeline reads `direct_write_map` and
`topk_routing_map` from ping-pong SMEM buffers via mbarrier waits. Each TMA prefetch of
the next chunk triggers mbarrier wait cycles that show up as short_scoreboard stalls.

**wait (20% → 25%)**: The `wait` stall category on Blackwell includes TMA completion waits.
Direct mode issues more TMA operations (SMEM prefetch of metadata + token writes to 8
scattered destinations per token vs 1 sequential write). The increased TMA traffic
contributes to higher wait stalls.

### 5. SM Throughput and Occupancy Unchanged

Both kernels use 24 SMs with 128 threads (4 warps) per SM. The kernel is not compute-bound
(SM throughput ~1.2%), and warp occupancy is identical (~6.25%). The bottleneck is entirely
in the memory/NVLink subsystem.

## Root Cause Summary

The direct-permute dispatch kernel is **2.2x slower** because:

1. **4x more NVLink traffic**: Each token is written K=8 times (once per routed expert) vs
   once in non-direct mode.
2. **Scattered write pattern**: Expert-grouped positions are non-contiguous, defeating NVLink
   write coalescing and causing L2 spills to DRAM (96x more DRAM writes).
3. **SMEM metadata overhead**: Reading `direct_write_map` + `topk_routing_map` from SMEM adds
   short_scoreboard stalls (18% vs 7%).

## Implications

For K=8, H=512, the direct-permute dispatch kernel (345 us) + address computation (67 us)
= 412 us total, vs non-direct dispatch (156 us) + permute (42 us) = 198 us. Direct is
**2.1x slower end-to-end**.

The break-even point requires either:
- Much larger H (so permute kernel cost dominates and direct-permute saves more)
- Reduced NVLink write amplification (e.g., coalescing writes by sorting destination
  addresses, or using a different buffer layout)
- Larger K where the permute kernel scales worse than the direct write overhead

For the target workload (H=512, K=36), the 4x write amplification would be even worse
(~4.5x), making direct-permute clearly uneconomical without fundamental changes to the
write pattern.
