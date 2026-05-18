# NCU Analysis: Direct-Permute Dispatch Kernel

**Date**: 2026-05-14
**Node**: B300 NVL8 (8× B300 SXM6, sm_103, NVLink)
**Container**: PyTorch 26.03, CUDA 13.2, ncu 2026.1.0
**Config**: H=512, T=8192, E_local=32, R=8, NUM_SMS_DISPATCH=24

## Profiling Setup

- `--replay-mode application` with TCP communicator + lockstep
- ~13 application replay passes per rank
- Reports from rank 1 (representative; all ranks symmetric)

## K=4 Results

| Metric                     | Non-direct    | Direct        | Ratio / Delta   |
| -------------------------- | ------------- | ------------- | --------------- |
| **Timing**                 | 144.99 us     | 197.38 us     | **1.36x slower**|
| **DRAM read**              | 17.10 MB      | 17.12 MB      | 1.0x            |
| **DRAM write**             | 114.43 KB     | 72.96 KB      | 0.6x            |
| **NVLink TX (total)**      | 32.77 MB      | 106.69 MB     | **3.3x more**   |
| **NVLink TX (user data)**  | 27.60 MB      | 89.00 MB      | **3.2x more**   |
| **NVLink RX (total)**      | 812.26 KB     | 1.15 MB       | 1.4x            |
| **L1 sectors (total)**     | 2,048         | 100,352       | **49x more**    |
| **L1 sectors (miss)**      | 256           | 28,480        | **111x more**   |
| **L1 miss rate**           | 12.5%         | 28.4%         | +16pp           |
| **SM throughput**          | 1.09%         | 1.42%         | ~same           |
| **Warp occupancy**         | 6.24%         | 6.25%         | ~same           |
| **Stall: long_scoreboard** | 59.63%        | 50.84%        | -9pp            |
| **Stall: short_scoreboard**| 6.53%         | 15.26%        | **+9pp**        |
| **Stall: wait**            | 18.89%        | 22.89%        | **+4pp**        |
| **Stall: barrier**         | 0.02%         | 0.03%         | ~same           |
| **Stall: membar**          | 0%            | 0%            | ~same           |
| **Stall: not_selected**    | 0%            | 0%            | ~same           |
| **Stall: mio_throttle**    | 0.01%         | 0.01%         | ~same           |
| **Stall: math_pipe**       | 0%            | 0%            | ~same           |

Kernel template parameters (from demangled name):
- Non-direct: `..., 32, 6, 64, 512, 8192, 32, 8, 1, 24, 2368, 1, 0, 0` (DIRECT_PERMUTE=0)
- Direct:     `...,  1, 6, 64, 512, 8192, 32, 8, 1, 24, 2368, 1, 1, 4` (DIRECT_PERMUTE=1, TOPK=4)

## K=8 Results

| Metric                     | Non-direct    | Direct        | Ratio / Delta   |
| -------------------------- | ------------- | ------------- | --------------- |
| **Timing**                 | 155.94 us     | 344.86 us     | **2.2x slower** |
| **DRAM read**              | 17.10 MB      | 17.40 MB      | 1.0x            |
| **DRAM write**             | 83.20 KB      | 7.99 MB       | **96x more**    |
| **NVLink TX (total)**      | 52.26 MB      | 214.05 MB     | **4.1x more**   |
| **NVLink TX (user data)**  | 44.01 MB      | 178.26 MB     | **4.0x more**   |
| **NVLink RX (total)**      | 865.47 KB     | 1.83 MB       | 2.1x            |
| **L1 sectors (total)**     | 2,048         | 198,656       | **97x more**    |
| **L1 sectors (miss)**      | 256           | 65,976        | **258x more**   |
| **L1 miss rate**           | 12.5%         | 33.2%         | +21pp           |
| **SM throughput**          | 1.21%         | 1.38%         | ~same           |
| **Warp occupancy**         | 6.24%         | 6.25%         | ~same           |
| **Stall: long_scoreboard** | 59.10%        | 45.39%        | -14pp           |
| **Stall: short_scoreboard**| 6.88%         | 18.03%        | **+11pp**       |
| **Stall: wait**            | 19.91%        | 25.09%        | **+5pp**        |
| **Stall: barrier**         | 0.04%         | 0.01%         | ~same           |
| **Stall: membar**          | 0%            | 0%            | ~same           |
| **Stall: not_selected**    | 0%            | 0%            | ~same           |
| **Stall: mio_throttle**    | 0.01%         | 0.00%         | ~same           |
| **Stall: math_pipe**       | 0%            | 0%            | ~same           |

Grid: (24, 1, 1), Block: (128, 1, 1) — both use 24 SMs × 4 warps for all configs.

## Cross-K Scaling

| Metric                      | K=4 non-direct | K=4 direct | K=8 non-direct | K=8 direct |
| --------------------------- | -------------- | ---------- | -------------- | ---------- |
| **Timing (us)**             | 145.0          | 197.4      | 155.9          | 344.9      |
| **Direct/non-direct ratio** | —              | 1.36x      | —              | 2.21x      |
| **NVLink TX user (MB)**     | 27.60          | 89.00      | 44.01          | 178.26     |
| **NVLink TX ratio**         | —              | 3.2x       | —              | 4.0x       |
| **DRAM write**              | 114 KB         | 73 KB      | 83 KB          | 7.99 MB    |
| **L1 sectors (total)**      | 2,048          | 100,352    | 2,048          | 198,656    |
| **short_scoreboard**        | 6.53%          | 15.26%     | 6.88%          | 18.03%     |

Key observations from K scaling:
- **Non-direct kernel is stable**: 145 → 156 us (+7%) from K=4 to K=8. The non-direct
  kernel does not iterate over K; it writes each token once. The slight increase is from
  more tokens being routed to remote ranks with higher K.
- **Direct kernel scales ~linearly with K**: 197 → 345 us (+75%) for K doubling. Each
  token requires K TMA writes, so doubling K roughly doubles S2G work.
- **NVLink TX user data scales linearly with K**: 89 → 178 MB (2.0x) for direct, matching
  the K=4→K=8 doubling.
- **DRAM write anomaly at K=4**: Direct K=4 writes LESS to DRAM (73 KB) than non-direct
  (114 KB). At K=4 the scattered writes fit in L2 cache. At K=8 this breaks down
  catastrophically (7.99 MB) as the working set exceeds L2 capacity.
- **L1 sectors scale linearly**: 100K → 199K (2.0x) for K doubling, matching the K
  metadata reads per token.

## Analysis

### 1. NVLink TX Amplification

The dominant cost. In non-direct mode, each token is written once to a staging buffer on
the target rank. In direct-permute mode, each token is written K times — once per routed
expert — to scattered expert-grouped positions via NVLink.

| K | Non-direct TX (user) | Direct TX (user) | Amplification |
|---|----------------------|------------------|---------------|
| 4 | 27.60 MB             | 89.00 MB         | 3.2x          |
| 8 | 44.01 MB             | 178.26 MB        | 4.0x          |

The amplification factor approaches K as more tokens route to distinct remote ranks. At
K=4, some tokens share destinations, giving 3.2x instead of 4x. At K=8, routing is more
spread across ranks, approaching 4x. At K=36, this would be ~4.5x.

### 2. DRAM Write: L2 Capacity Cliff at K=8

A striking result: direct K=4 has *lower* DRAM writes than non-direct (73 KB vs 114 KB),
while direct K=8 explodes to 7.99 MB.

At K=4: the total scattered write footprint per rank is T × K × H × 2 ≈ 33.5 MB. The
B300's 55 MB L2 cache can hold this, so scattered writes are absorbed by L2 and never
spill to DRAM.

At K=8: the footprint doubles to ~67 MB, exceeding L2 capacity. Eviction-driven writeback
produces 7.99 MB of DRAM traffic. This is a capacity cliff, not a gradual degradation.

### 3. L1 Sectors: Metadata Overhead Scales with K

Non-direct: 2,048 L1 sectors regardless of K (reads only `sparse_to_dense_map`).

Direct: L1 sectors scale linearly with K (100K at K=4, 199K at K=8) because the S2G
inner loop reads `direct_write_map[chunk_offset * TOPK + k]` and
`topk_routing_map[chunk_offset * TOPK + k]` from SMEM for each of K entries per token.
The L1 traffic comes from TMA descriptor management and prob store operations, both
proportional to K.

### 4. Stall Profile

**long_scoreboard**: Decreases from ~60% (non-direct) to ~50% (K=4 direct) to ~45% (K=8
direct). Non-direct issues coalesced writes faster, spending more time waiting for NVLink
to drain. Direct's per-token loop over K entries introduces instruction-level overhead that
shifts the bottleneck away from pure NVLink stalls.

**short_scoreboard**: Increases from ~6.5% to ~15% (K=4) to ~18% (K=8). The SMEM pipeline
reads of `direct_write_map` + `topk_routing_map` via mbarrier waits drive this. Scales
with K as more metadata entries are read per token.

**wait**: Increases from ~19% to ~23% (K=4) to ~25% (K=8). More TMA operations (metadata
prefetch + K scattered writes per token) increase TMA completion waits.

### 5. SM Throughput and Occupancy Unchanged

All configs use 24 SMs × 128 threads. SM throughput ~1.1-1.4%, warp occupancy ~6.25%.
The kernel is entirely NVLink/memory bound; compute is not a factor.

## Root Cause Summary

The direct-permute dispatch kernel is slower because:

1. **K× NVLink write amplification**: Each token written K times vs once. At K=4: 3.2x
   more NVLink traffic, 1.36x slower. At K=8: 4.0x more traffic, 2.2x slower.
2. **L2 capacity cliff at K=8**: Scattered writes fit in L2 at K=4 (73 KB DRAM write) but
   overflow at K=8 (7.99 MB). This non-linear degradation is why K=8 is disproportionately
   worse (2.2x) vs K=4 (1.36x).
3. **Linear metadata overhead**: L1 sectors and short_scoreboard stalls grow linearly with K
   from per-token SMEM reads of routing metadata.

## End-to-End Comparison

| Path                        | K=4    | K=8    |
| --------------------------- | ------ | ------ |
| Non-direct dispatch kernel  | 145 us | 156 us |
| + permute kernel            | +32 us | +32 us |
| **= Non-direct total**      | **177 us** | **188 us** |
| Direct dispatch kernel      | 197 us | 345 us |
| + address computation       | +48 us | +67 us |
| **= Direct total**          | **245 us** | **412 us** |
| **Direct overhead**         | **1.38x** | **2.19x** |

Note: permute kernel cost (~32 us at H=512) is independent of K. Address computation
scales sub-linearly with K (48 us at K=4, 67 us at K=8).

## Implications

At K=4, the gap narrows to 1.38x — only 68 us overhead. This is still not favorable for
H=512 since permute only costs 32 us. But there are two important observations:

1. **Permute scales with H, direct does not**: At larger H, the permute kernel grows
   linearly (32 us at H=512 → ~450 us at H=7168) while the direct dispatch kernel's
   overhead is dominated by NVLink writes which already transfer the full token. The
   break-even H scales with K.

2. **K=4 avoids the L2 cliff**: The dramatically better DRAM behavior at K=4 (73 KB vs
   7.99 MB at K=8) suggests that for small K, direct-permute's scattered writes are
   efficiently absorbed by L2. The L2 cliff occurs around K=6-7 for this configuration.

For the target workload (H=512, K=36), direct-permute is clearly uneconomical. The ~4.5x
NVLink amplification, L2 overflow, and O(K) metadata overhead would produce a kernel
~4-5x slower than non-direct, far exceeding the permute kernel savings.

Direct-permute may be viable for:
- Large H (H≥4096) where permute cost dominates
- Small K (K≤4) where L2 absorbs scattered writes
- Both together (large H + small K) for maximum benefit
