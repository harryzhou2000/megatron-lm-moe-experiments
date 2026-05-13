# hybrid-ep-sparse-opt Performance Comparison

## Test Environment
- **Hardware**: 8x NVIDIA B200 SXM (NVLink)
- **Container**: PyTorch 26.03, Python 3.12, CUDA 13.1
- **Node**: umb-b200-247 (132 SMs, sm_100)
- **Build**: `TORCH_CUDA_ARCH_LIST="10.0"`, ccache, `--no-build-isolation`
- **Config**: `NUM_SMS_DISPATCH=32`, `NUM_SMS_COMBINE=32`, `HIDDEN_DIM=512`, `NUM_TOKENS_PER_RANK=8192`, BF16

## Branches
- **`hybrid-ep`** (upstream): commit `1b8f467` — original permute/unpermute kernels
- **`hhanyu/hybrid-ep-sparse-opt`**: commit `7ec3238` — ballot+ffs optimized permute/unpermute

## Results

### Config 1: E=32, TOPK=36 (latent MoE canonical)

| Metric                          | hybrid-ep   | sparse-opt  | Speedup |
| ------------------------------- | ----------- | ----------- | ------- |
| dispatch kernel                 | 185.1 μs    | 103.4 μs    | 1.79x   |
| dispatch no-prob kernel         | 95.8 μs     | 95.7 μs     | 1.00x   |
| **permute_kernel**              | **393.6 μs** | **94.4 μs** | **4.17x** |
| **unpermute_kernel**            | **267.3 μs** | **131.6 μs** | **2.03x** |
| combine kernel (w/ probs)       | 972.5 μs    | 960.8 μs    | 1.01x   |
| combine kernel (no probs)       | 876.7 μs    | 877.4 μs    | 1.00x   |
| dispatch+permute (API, w/ prob) | 599.8 μs    | 218.0 μs    | 2.75x   |
| combine+unpermute (API)         | 1283.2 μs   | 1125.5 μs   | 1.14x   |
| fused dispatch+permute kernel   | 619.2 μs    | 597.1 μs    | 1.04x   |
| fused combine+unpermute kernel  | 1144.8 μs   | 1143.5 μs   | 1.00x   |

### Config 2: E=32, TOPK=8

| Metric                          | hybrid-ep   | sparse-opt  | Speedup |
| ------------------------------- | ----------- | ----------- | ------- |
| dispatch kernel                 | 130.0 μs    | 96.9 μs     | 1.34x   |
| dispatch no-prob kernel         | 90.3 μs     | 90.1 μs     | 1.00x   |
| **permute_kernel**              | **257.2 μs** | **43.3 μs** | **5.94x** |
| **unpermute_kernel**            | **127.3 μs** | **64.4 μs** | **1.98x** |
| combine kernel (w/ probs)       | 647.0 μs    | 644.9 μs    | 1.00x   |
| combine kernel (no probs)       | 577.4 μs    | 577.2 μs    | 1.00x   |
| dispatch+permute (API, w/ prob) | 397.9 μs    | 162.4 μs    | 2.45x   |
| combine+unpermute (API)         | 809.7 μs    | 730.3 μs    | 1.11x   |
| fused dispatch+permute kernel   | 412.9 μs    | 408.1 μs    | 1.01x   |
| fused combine+unpermute kernel  | 667.5 μs    | 654.5 μs    | 1.02x   |

### Config 3: E=16, TOPK=8

| Metric                          | hybrid-ep   | sparse-opt  | Speedup |
| ------------------------------- | ----------- | ----------- | ------- |
| dispatch kernel                 | 96.9 μs     | 96.8 μs     | 1.00x   |
| dispatch no-prob kernel         | 90.3 μs     | 90.4 μs     | 1.00x   |
| **permute_kernel**              | **153.8 μs** | **43.3 μs** | **3.55x** |
| **unpermute_kernel**            | **103.5 μs** | **58.2 μs** | **1.78x** |
| combine kernel (w/ probs)       | 626.4 μs    | 625.6 μs    | 1.00x   |
| combine kernel (no probs)       | 581.5 μs    | 582.2 μs    | 1.00x   |
| dispatch+permute (API, w/ prob) | 276.9 μs    | 163.6 μs    | 1.69x   |
| combine+unpermute (API)         | 772.6 μs    | 713.4 μs    | 1.08x   |
| fused dispatch+permute kernel   | 272.4 μs    | 291.3 μs    | 0.94x   |
| fused combine+unpermute kernel  | 647.6 μs    | 641.3 μs    | 1.01x   |

## Key Observations

1. **Permute kernel**: 3.5-5.9x speedup from ballot+ffs optimization. Largest gain at E=32 TOPK=8 where routing is sparsest (only ~1 active expert per token locally vs scanning 32).

2. **Unpermute kernel**: 1.8-2.0x speedup from same ballot optimization applied to reverse path.

3. **Dispatch kernel (with prob)**: 1.3-1.8x speedup for TOPK=36 config due to prob write optimization (E_per_rank slice TMA instead of full E*R vector). No change for TOPK=8 (smaller prob vectors).

4. **Dispatch no-prob / Combine kernels**: Unchanged — these paths were not modified.

5. **Fused kernels**: Essentially unchanged (< 5% difference). The fused kernel bottleneck is SM contention and chunk-flag latency, not the permute S2G map scan. The ballot optimization inside the fused permute blocks has negligible impact.

6. **End-to-end dispatch+permute API**: 1.7-2.8x improvement driven entirely by the permute/dispatch kernel speedups.

## Conclusion

The `sparse-opt` branch provides major gains for the **non-fused** dispatch+permute path through:
- Ballot+ffs expert skip in permute/unpermute (avoids iterating inactive experts)
- E_per_rank prob slice optimization in dispatch S2G (reduces NVLink traffic)

For latent MoE (H=512), the **non-fused path is always preferred** over fused:
- Non-fused dispatch+permute: 163-218 μs (sparse-opt) vs 291-597 μs (fused)
- Fused provides no overlap benefit when permute is already fast (43-94 μs)

---

## Combine Kernel with Tuned Pipeline Config

Additional test with `NUM_OF_STAGES_G2S_COMBINE_API=8`, `NUM_OF_STAGES_S2G_COMBINE_API=4`, `NUM_TOKENS_COMBINE_REDUCE_BATCH_COMBINE_API=4`.

### Config 1: E=32, TOPK=36

| Metric                    | hybrid-ep (default) | hybrid-ep (tuned) | sparse-opt (default) | sparse-opt (tuned) |
| ------------------------- | ------------------- | ----------------- | -------------------- | ------------------ |
| combine kernel (w/ probs) | 972.5 μs            | 1193.8 μs         | 960.8 μs             | 1473.1 μs          |
| combine kernel (no probs) | 876.7 μs            | 1068.1 μs         | 877.4 μs             | 1332.6 μs          |
| combine+unpermute (API)   | 1283.2 μs           | 1491.6 μs         | 1125.5 μs            | 1624.2 μs          |

### Config 2: E=32, TOPK=8

| Metric                    | hybrid-ep (default) | hybrid-ep (tuned) | sparse-opt (default) | sparse-opt (tuned) |
| ------------------------- | ------------------- | ----------------- | -------------------- | ------------------ |
| combine kernel (w/ probs) | 647.0 μs            | 788.8 μs          | 644.9 μs             | 1176.8 μs          |
| combine kernel (no probs) | 577.4 μs            | 700.7 μs          | 577.2 μs             | 1049.6 μs          |
| combine+unpermute (API)   | 809.7 μs            | 953.7 μs          | 730.3 μs             | 1311.4 μs          |

### Config 3: E=16, TOPK=8

| Metric                    | hybrid-ep (default) | hybrid-ep (tuned) | sparse-opt (default) | sparse-opt (tuned) |
| ------------------------- | ------------------- | ----------------- | -------------------- | ------------------ |
| combine kernel (w/ probs) | 626.4 μs            | 761.9 μs          | 625.6 μs             | 1139.4 μs          |
| combine kernel (no probs) | 581.5 μs            | 705.7 μs          | 582.2 μs             | 1053.9 μs          |
| combine+unpermute (API)   | 772.6 μs            | 909.5 μs          | 713.4 μs             | 1279.8 μs          |

### Observation

The tuned combine pipeline config (G2S=8, S2G=4, batch=4) **regresses** the combine kernel on B200 for H=512:
- hybrid-ep: 1.2-1.3x slower
- sparse-opt: 1.5-1.8x slower

The regression is worse on sparse-opt, likely because the larger SMEM footprint (8 G2S stages + 4 S2G stages) conflicts with the combine kernel's occupancy needs. The default config (G2S=10, S2G=2, batch=1) is better for this hardware/hidden-dim combination.

The batch-reduce feature (`NUM_TOKENS_COMBINE_REDUCE_BATCH=4`) accumulates 4 tokens before doing the reduction barrier sync. This was designed for larger hidden dims where the reduction compute dominates — for H=512 the tokens are small and the extra accumulation just adds latency without amortizing meaningful compute.

---

## Combine Kernel with Documented Best Config

Config from `docs/Hybrid-EP_Implementation.md`:
```
NUM_OF_STAGES_G2S_COMBINE_API=64  NUM_OF_STAGES_S2G_COMBINE_API=8
NUM_TOKENS_COMBINE_REDUCE_BATCH_COMBINE_API=16  NUM_OF_TOKENS_PER_GROUP_COMBINE_API=1
NUM_SMS_COMBINE=64  NUM_SMS_DISPATCH=32
```

Tested on B200 NVL8, H=512, E=32, TOPK=36, T=8192.

### Cross-branch comparison (tuned combine config)

| Metric                          | hybrid-ep (tuned) | sparse-opt (tuned) | Speedup (sparse-opt) |
| ------------------------------- | ----------------- | ------------------ | -------------------- |
| dispatch kernel (w/ prob)       | 185.1 μs          | 103.5 μs           | **1.79x**            |
| dispatch no-prob kernel         | 95.7 μs           | 95.7 μs            | 1.00x                |
| **permute_kernel**              | **396.1 μs**      | **92.7 μs**        | **4.27x**            |
| **unpermute_kernel**            | **269.5 μs**      | **131.8 μs**       | **2.04x**            |
| combine kernel (w/ probs)       | 241.8 μs          | 252.3 μs           | 0.96x                |
| combine kernel (no probs)       | 228.0 μs          | 210.9 μs           | 1.08x                |
| dispatch+permute (API)          | 602.1 μs          | 217.9 μs           | **2.76x**            |
| combine+unpermute (API)         | 531.6 μs          | 399.2 μs           | **1.33x**            |

### Default vs Tuned combine (within each branch, E=32 TOPK=36)

| Metric                    | sparse-opt default | sparse-opt tuned | Speedup |
| ------------------------- | ------------------ | ---------------- | ------- |
| combine kernel (w/ probs) | 960.8 μs           | 252.3 μs         | **3.8x**|
| combine kernel (no probs) | 877.4 μs           | 210.9 μs         | **4.2x**|
| combine+unpermute (API)   | 1125.5 μs          | 399.2 μs         | **2.8x**|
| dispatch+permute (API)    | 218.0 μs           | 217.9 μs         | 1.00x   |

| Metric                    | hybrid-ep default | hybrid-ep tuned  | Speedup |
| ------------------------- | ----------------- | ---------------- | ------- |
| combine kernel (w/ probs) | 972.5 μs          | 241.8 μs         | **4.0x**|
| combine kernel (no probs) | 876.7 μs          | 228.0 μs         | **3.8x**|
| combine+unpermute (API)   | 1283.2 μs         | 531.6 μs         | **2.4x**|
| dispatch+permute (API)    | 599.8 μs          | 602.1 μs         | 1.00x   |

### Summary

- The tuned combine config successfully replicates documented results on B200 (combine kernel 242-252 μs, ~4x over default).
- The tuned config does **not** degrade the dispatch+permute path (dispatch kernel and permute kernel unchanged from default).
- **sparse-opt with tuned combine** gives the best overall end-to-end:
  - dispatch+permute: **218 μs** (2.8x faster than hybrid-ep's 602 μs)
  - combine+unpermute: **399 μs** (1.3x faster than hybrid-ep's 532 μs)
  - **Total: 617 μs** (vs hybrid-ep tuned 1134 μs = **1.84x** end-to-end improvement)

---

## Combine Kernel Pipeline Analysis

### Pipeline Architecture (single-node, B200 NVL8)

Each combine block has 6 warps split into 2 independent data pipelines:
- **G2S warp** (1 per pipeline): Single elected thread issues `cp_async_bulk` TMAs to load source tokens (1 KB each for BF16 H=512) from remote NVLink buffers into SMEM FIFO.
- **Reduction warp group** (2 warps per pipeline, 64 threads): Waits for G2S FIFO slots, accumulates source tokens in FP32 registers, writes result to S2G SMEM, issues S2G TMA.

With `NUM_OF_STAGES_G2S=64` and 2 pipelines: each pipeline gets 32 G2S FIFO slots.

### Batched Reduction Loop (per output token, ~36 source tokens)

For each batch of up to `BATCH=16` sources:
1. **Phase 1**: Elected thread waits on mbarriers sequentially (spin-wait for TMA completion)
2. **Phase 2**: `arrive_and_wait` barrier (sync 64 threads)
3. **Phase 3**: All threads read SMEM and accumulate (trivial compute: 4 FMAs/thread/source)
4. **Phase 4**: `arrive_and_wait` barrier, then free FIFO slots

With ~36 sources and BATCH=16: ~3 batches → 6 named barriers per output token.

### GROUP=1 vs GROUP=2 Tradeoff

| Config                   | combine kernel (w/ probs) | GB/s (output) | SMs  | Pipeline depth |
| ------------------------ | ------------------------- | ------------- | ---- | -------------- |
| GROUP=1, SMS=64          | 252 μs                    | 265 GB/s      | 64   | 64 stages      |
| GROUP=2, SMS=32          | 254 μs                    | 263 GB/s      | 32   | 32 stages/pipe |
| GROUP=4, SMS=32          | 502 μs                    | 133 GB/s      | 32   | 16 stages/pipe |

GROUP=1 wastes half the warps but gives full 64-stage depth per pipeline. GROUP=2 uses both pipelines at 32 stages each. **Same throughput, half the SMs** — GROUP=2 with SMS=32 is more SM-efficient.

GROUP=4 is too shallow (16 stages per output token) — can't hide NVLink latency.

### Actual NVLink Read Bandwidth

The reported "GB/s" metric (265 GB/s) is based on **output bytes** (66.73 MB). The actual NVLink **read** traffic is much larger:
- Each output token reads from ~8 source ranks × 1 KB = ~8 KB (not 36 — combine reads one aggregated copy per rank, not per expert)
- 8192 tokens × 8 KB = ~65.5 MB total NVLink reads
- 65.5 MB / 246 μs = **~266 GB/s actual NVLink read BW** — similar to the output metric, as expected (read ≈ write for TOPK~36 with 8 ranks)

### Pipeline Depth Investigation (B300 NVL8)

Tested whether deeper G2S pipeline improves throughput:

| G2S stages | Per-pipeline depth | Tokens prefetched | combine kernel (w/ probs) |
| ---------- | ------------------ | ----------------- | ------------------------- |
| 64         | 32 stages          | ~4 output tokens  | 246.1 μs                  |
| 128        | 64 stages          | ~8 output tokens  | 245.0 μs                  |

**Doubling pipeline depth had zero effect.** The pipeline is already deep enough: with ~8 sources per output token and 32 slots, 4 tokens of prefetch covers NVLink latency. The G2S warp can fill ahead across output token boundaries — there is no inter-token pipeline bubble.

### True Bottleneck: Per-Token Pipeline Overhead

Per output token, the reduction pipeline executes:
- **Phase 1**: Elected thread waits for batch of mbarriers (G2S TMA completions)
- **Phase 2**: `arrive_and_wait` (sync 64 threads before reading batch)
- **Phase 3**: Accumulate from SMEM (trivial compute: 4 bf16x2 FMAs/thread)
- **Phase 4**: `arrive_and_wait` (sync after accumulate, before freeing slots)
- **S2G wait**: `arrive_and_wait` (before store to S2G SMEM)
- **S2G store**: All threads write accumulator to SMEM
- **S2G fence**: `arrive_and_wait` (after store, before TMA issue)
- **S2G TMA**: Elected thread issues cp_async_bulk

= **4 named barriers** + mbarrier waits + SMEM R/W + TMA issue per output token.
With BATCH=16 covering all ~8 sources in 1 batch, the reduction loop runs once per token.

**Estimated per-token cost**: ~1.9 μs (from 128 tokens/pipeline × 1.9 μs ≈ 243 μs → matches measured 246 μs).

### Experiments: Single-Warp Reduce and GROUP Scaling (B300 NVL8)

**Single-warp reduce** (1 warp = 32 threads per pipeline, `__syncwarp` instead of `bar.sync`):
- combine kernel (w/ probs): 269 μs (+9% vs 246 μs default)
- **Slower** — the 2x more SMEM loop iterations per thread (8 vs 4 elements) offsets barrier savings. B300 named barriers are faster than estimated. Warp scheduling between 2 warps hides SMEM latency.

**GROUP scaling** (tokens per group, with SMS_COMBINE=64):

| GROUP | Tokens/pipeline/group | G2S headroom (slots) | combine kernel (w/ probs) |
| ----- | --------------------- | -------------------- | ------------------------- |
| 1     | 1                     | 64 - 8 = 56          | 245 μs (SMS=64)           |
| 2     | 1                     | 32 - 8 = 24          | 246 μs (SMS=32)           |
| 4     | 2                     | 32 - 16 = 16         | 486 μs                    |
| 8     | 4                     | 32 - 32 = 0          | 955 μs                    |

**GROUP ≥ 4 degrades** because the G2S FIFO fills up (no headroom for prefetch). GROUP=1 and GROUP=2 are equivalent — the pipeline has sufficient headroom (24-56 spare slots) to keep the G2S warp running ahead.

### Pipeline Depth vs Headroom

With G2S=64, 2 pipelines (32 slots each), ~8 sources/token:
- G2S fills 4 tokens of sources ahead (32/8 = 4 tokens)
- Reduction processes 1 token at a time (~1.9 μs each)
- G2S fills 1 token's sources in ~0.4-0.8 μs (8 TMAs at ~50-100 ns issue rate)
- **G2S is ~2-4x faster than reduction** → no pipeline bubble

Increasing G2S to 128 confirmed no improvement (245 μs vs 246 μs). The pipeline is not depth-limited.

### Conclusion

The combine kernel at **246 μs** (271 GB/s output BW) for H=512 BF16 with tuned config (G2S=64, S2G=8, BATCH=16, GROUP=2, SMS=32) appears to be at or near the practical hardware limit for this access pattern. The per-token overhead (~1.9 μs) is dominated by:
1. Named barrier synchronization (4 per token)
2. SMEM read/write for accumulation and S2G store
3. TMA issue overhead for 1 KB transfers

Further improvement would require fundamental restructuring (e.g., eliminating per-token barriers by fusing multiple tokens' accumulations into a single barrier-free loop), which would be a major rewrite of the combine kernel architecture.

---

## Combine Kernel at H=7168 (Standard MoE Config)

B300 NVL8, H=7168, E_per_rank=8, TOPK=8, T=4096, BF16. Per TMA: 14 KB.

### Results

| Config                                     | dispatch kernel | combine kernel (w/ probs) | combine kernel (no probs) |
| ------------------------------------------ | --------------- | ------------------------- | ------------------------- |
| Default (SMS_C=32, G2S=10, GROUP=2)        | 409 μs (778 GB/s) | 482 μs (659 GB/s)       | 478 μs (665 GB/s)         |
| Tuned (SMS_C=64, G2S=10, BATCH=8, GROUP=2) | 409 μs (778 GB/s) | 471 μs (675 GB/s)       | 468 μs (679 GB/s)         |
| GROUP=1 (SMS_C=64, 4 red warps, 1 pipe)   | 409 μs (778 GB/s) | 473 μs (673 GB/s)       | 471 μs (676 GB/s)         |

### Analysis: Why Combine Lags Dispatch by ~15%

Dispatch: 778 GB/s (86% of 900 GB/s NVLink peak).
Combine: 675 GB/s (75% of peak). Gap: ~1.15x.

**Root cause: G2S TMA issue rate.** The dispatch S2G warp group has **3 warps** (single-node), each with 1 elected TMA issuer = 3 concurrent TMA issuers distributing writes across 8 remote ranks. The combine G2S warp group has **2 warps** (1 per pipeline), each issuing reads from remote ranks = 2 concurrent TMA issuers.

Per output token:
- Dispatch S2G: 8 TMAs of 14 KB, distributed across 3 threads → ~2.7 TMAs/thread
- Combine G2S: 8 TMAs of 14 KB, distributed across 2 threads (2 pipelines) → ~4 TMAs/thread

At ~50 ns per TMA issue, dispatch takes ~135 ns vs combine ~200 ns per token for TMA issue alone. This ~1.5x issue rate gap explains the ~1.15x throughput gap (partially hidden by pipeline).

GROUP=1 with 4 red warps (128 threads) didn't help because the bottleneck is G2S TMA issue rate, not reduction compute speed. More red warps reduce accumulation time but the pipeline is already G2S-bound.

### Improvement Path

To match dispatch's TMA issue rate, combine would need 3 G2S warps (matching dispatch's 3 S2G warps). This requires either:
- `NUM_OF_DATA_PIPELINE_PER_BLOCK=3` with 3 G2S warps + 3-6 red warps
- A redesign allowing multiple G2S threads within a single pipeline

This is a non-trivial architectural change to the combine kernel.

### H=512 vs H=7168 Combine Comparison

| Hidden dim | Per TMA | combine BW  | % of dispatch | Bottleneck                           |
| ---------- | ------- | ----------- | ------------- | ------------------------------------ |
| H=512      | 1 KB    | 271 GB/s    | 42%           | Per-token overhead (barriers, TMA issue) |
| H=7168     | 14 KB   | 675 GB/s    | 87%           | G2S TMA issue rate                   |

Combine efficiency scales with hidden dimension because the per-token fixed overhead (barriers, TMA issue) is amortized over more data per token.

---

## NCU Stall Profile: Combine Kernel (H=512, B300)

**Config**: H=512, E=32, TOPK=36, 8 ranks, NUM_SMS=32, G2S=64, S2G=8, BATCH=16, GROUP=2  
**Hardware**: B300 SXM6 (sm_103, 128 SMs, NVLink)  
**Method**: Collective ncu (`--communicator tcp --lockstep-kernel-launch`), kernel replay across all 8 ranks

### Timing & Throughput

| Metric          | Value         |
| --------------- | ------------- |
| Kernel duration | 254.43 μs     |
| SM throughput   | 5.09% of peak |
| Warp occupancy  | 9.08% of peak |

### Warp Stall Breakdown

| Stall reason                              | %      | Interpretation                                    |
| ----------------------------------------- | ------ | ------------------------------------------------- |
| **long_scoreboard** (L1TEX/TMA)           | **26.59%** | Waiting for TMA G2S loads (NVLink read round-trip) |
| **wait** (fixed-latency)                  | **25.92%** | mbarrier_try_wait_parity for TMA completion        |
| **barrier** (named barrier)               | **15.23%** | Red warps waiting for G2S to signal data ready     |
| short_scoreboard (MIO)                    | 7.22%  | SMEM loads during reduction                        |
| not_selected                              | 1.09%  | Warp scheduler contention                          |
| math_pipe_throttle                        | 0.39%  | FMA pipe (negligible for H=512)                    |
| misc                                      | 0.05%  | —                                                  |
| mio_throttle                              | 0.04%  | —                                                  |
| membar                                    | 0.00%  | —                                                  |

### NVLink Traffic (per rank)

| Metric                   | Value         | Notes                                        |
| ------------------------ | ------------- | -------------------------------------------- |
| NVL RX total             | 73.90 MB      | All data comes over NVLink                   |
| NVL RX user data         | 65.69 MB      | Token data: 8192 × (36/8) × 512B × 7 peers  |
| NVL RX protocol overhead | 8.21 MB       | 11.1% overhead — high for 1 KB TMA requests  |
| NVL TX total             | 12.32 MB      | Request packets only (read requests to peers) |
| NVL TX user data         | 0 bytes       | No S2G writes over NVLink (output is local)  |

### DRAM & L1 Cache

| Metric            | Value        | Notes                                   |
| ----------------- | ------------ | --------------------------------------- |
| DRAM read         | 959 KB       | Negligible — kernel is pure NVLink-bound |
| DRAM write        | 256 bytes    | Only metadata/flags                      |
| L1TEX LSU sectors | 40,960       | SMEM access during reduction             |
| L1TEX LSU misses  | 12,288 (30%) | Metadata lookups (routing maps, indices) |

### Analysis

**All three top stall reasons trace back to NVLink read latency:**

1. `long_scoreboard` (26.6%) — G2S TMA issued, warp waiting for data to arrive from remote rank
2. `wait` (25.9%) — mbarrier wait for TMA completion notification (same root cause as #1)
3. `barrier` (15.2%) — red warps blocked at named barrier because G2S hasn't signaled yet (also waiting for NVLink)

Combined: **67.7% of warp stall time** is waiting for NVLink reads.

The reduction compute itself is trivially cheap:
- `math_pipe_throttle` = 0.39% (FMA throughput is not limiting)
- `short_scoreboard` = 7.22% (SMEM read latency during reduce-accumulate)

**Why deeper pipeline (G2S=128) didn't help:**
The single G2S issuer thread issues TMAs at hardware-limited rate. Pipeline depth hides latency only if consumer is slower than producer. For H=512, each token is only 512B — TMA issue latency (~50ns) dominates over the actual NVLink transfer time. Doubling pipeline stages just adds more in-flight requests that still bottleneck on the same issue thread.

**Why the access pattern is inherently limited for H=512:**
- 1 KB per TMA request (GROUP=2 → 2 tokens × 512B) — high per-request overhead (11% protocol)
- Pull (G2S read) vs Push (S2G write): reads require round-trip (request → remote HBM → response). Dispatch uses push (fire-and-forget S2G writes) achieving 650 GB/s vs combine's 271 GB/s.
- 7-way fan-in: each rank reads from 7 peers, all 8 ranks reading simultaneously — NVLink fabric contention
- Non-sequential remote addresses (token routing): limits NVLink request coalescing

---

## Fused vs Standalone Kernel Analysis (H=512, B300)

### Performance Gap

| Operation             | Standalone (serial)    | Fused (single launch) | Ratio    |
| --------------------- | ---------------------- | --------------------- | -------- |
| dispatch + permute    | 102 + 117 = **219 μs** | **591 μs**            | **2.7x slower** |
| combine + unpermute   | 246 + 131 = **377 μs** | **1093 μs**           | **2.9x slower** |

### Root Causes

#### 1. Massive occupancy reduction in fused mode

| Factor              | Standalone permute    | Fused permute blocks    |
| ------------------- | --------------------- | ----------------------- |
| Grid size           | 2048 blocks (sm×16)   | 96 blocks               |
| Blocks per SM       | 16                    | 1 (due to `__launch_bounds__(..., 1)`) |
| Tokens per block    | ~4 tokens/block       | ~85 tokens/block        |
| Latency hiding      | High (16 blocks/SM)   | None (1 block/SM)       |

Standalone permute has 2048 blocks across 128 SMs = 16 blocks/SM. This massive parallelism
amortizes per-token overhead (flag checks, TMA issue) and hides memory latency. Fused permute
has 96 blocks with 0 occupancy overlap — every memory stall is paid in full.

#### 2. Producer-consumer flag polling serialization

In fused mode, permute blocks must **poll chunk completion flags** (`ld.relaxed.sys`) to wait
for dispatch S2G to write each data chunk:

```
dispatch S2G writes chunk → sets flag → permute G2S polls flag → TMA G2S → permute S2G writes
```

This creates a per-chunk serial dependency. With 1 block/SM and no other work to hide behind,
every poll adds latency directly to the critical path.

In standalone mode, dispatch fully completes before permute launches. Permute reads a complete
buffer with no polling — just straight TMA loads.

#### 3. Thread utilization waste

Fused dispatch kernel uses `__launch_bounds__(128, 1)` — 4 warps/block. But:
- Permute G2S: only **1 elected thread** per warp does flag polling + TMA issue (31 idle)
- Permute S2G: only **1 elected thread** per warp does TMA multicast (31 idle per warp, 3 warps)

Only 4 threads out of 128 per permute block are doing useful work most of the time.

#### 4. SM resource contention (fused dispatch)

Fused dispatch grid = 32 dispatch + 96 permute = 128 blocks on 128 SMs:
- All blocks scheduled simultaneously (1 block/SM)
- 96 permute blocks immediately start spin-polling (active warps consuming SM resources)
- 32 dispatch blocks do real work
- No SM "starvation" per se, but the permute blocks are wasting 96 SMs on spin-wait

#### 5. Same issue for fused combine+unpermute

Fused combine grid = 32 combine + 96 unpermute = 128 blocks:
- Unpermute blocks depend on combine writing results to local expert buffers
- Unpermute polls flags or mbarriers waiting for combine to produce output
- With 1 block/SM, no latency hiding for the poll-wait-process-write cycle
- Standalone unpermute: 2048 blocks with 16 blocks/SM, processes pre-filled buffer

### Conclusion

The fused kernels are architecturally designed for **overlapping** dispatch/combine with
permute/unpermute. The idea: while dispatch writes chunks, permute immediately processes them
without waiting for dispatch to fully finish. In theory this should be faster.

In practice for H=512, it's 2.7-2.9x slower because:
1. The per-chunk flag polling overhead dominates (tiny 512B tokens = many chunks, many polls)
2. `__launch_bounds__(..., 1)` eliminates all occupancy-based latency hiding
3. Standalone permute/unpermute kernels use 21x more blocks with massive occupancy

The fused approach might break even at larger H (H=7168) where each chunk is 14 KB and the
per-chunk overhead is amortized. But for latent MoE (H=512), non-fused is always preferred.

### Fused Block Count Sweep (B300, H=512, E=32, TOPK=36)

Default fused block count uses `min(108, sm_count - num_sms_dispatch)` = 96, which severely
underutilizes the GPU. Increasing permute/unpermute blocks dramatically helps:

| Permute/Unpermute blocks | Fused dispatch+permute (kernel) | Fused combine+unpermute (kernel) |
| ------------------------ | ------------------------------- | -------------------------------- |
| 32                       | 1666 μs                         | 3321 μs                          |
| 96 (default)             | 591 μs                          | 1093 μs                          |
| 256                      | 283 μs                          | 557 μs                           |
| 512                      | **230 μs**                      | 612 μs                           |
| 1024                     | 233 μs                          | **544 μs**                       |
| 2048                     | 233 μs                          | 545 μs                           |
| **Non-fused standalone**     | **219 μs** (dispatch+permute)   | **377 μs** (combine+unpermute)   |

**Key observations:**
- Saturates at ~512 blocks for dispatch, ~1024 for combine
- Fused dispatch+permute with 512 blocks (230 μs) is within **5% of standalone** (219 μs)
  but does NOT fully match — residual overhead from flag-polling coordination
- **Fused combine+unpermute never matches standalone** — even at 2048 blocks, 544 μs vs
  377 μs = **44% slower**. The producer-consumer serialization between combine and unpermute
  blocks is a structural overhead that cannot be hidden by occupancy alone
- The combine→unpermute dependency is deeper than dispatch→permute because unpermute must
  accumulate (reduce) tokens from multiple experts, requiring the full combine output for
  each destination token before it can produce final results

**Conclusion:** For H=512, non-fused remains strictly preferred. The default fused block count
(96) is catastrophically bad; even with optimal block count the gap persists for combine.

### Fused Dispatch+Permute Pipeline Tuning (B300, H=512, E=32, TOPK=36)

Sweeping pipeline parameters with NUM_BLOCKS_PERMUTE (where not specified) to find optimal config.
Standalone baseline: dispatch=102.7 µs + permute=122.6 µs = **225.3 µs**.

| d_stages | p_stages | infl_d | infl_p | add_s2g | blocks  | Time (μs)         |
| -------- | -------- | ------ | ------ | ------- | ------- | ----------------- |
| 10 (def) | 10 (def) | 8      | 8      | 6 (def) | 96 (def) | **591.0**         |
| 10       | 10       | 8      | 8      | 6       | 512     | 232.8             |
| 5        | 5        | 4      | 4      | 6       | 512     | 364.8             |
| 10       | 20       | 8      | 16     | 6       | 512     | 232.5             |
| 20       | 10       | 16     | 8      | 6       | 512     | 212.6             |
| 10       | 10       | 8      | 8      | 0       | 512     | 237.6             |
| 20       | 20       | 16     | 16     | 6       | 512     | 209.7             |
| 20       | 20       | 16     | 16     | 10      | 512     | 209.9             |
| 16       | 16       | 12     | 12     | 6       | 512     | 209.6             |
| 24       | 24       | 20     | 20     | 6       | 512     | 208.7             |
| 30       | 30       | 24     | 24     | 6       | 512     | 272.7 (SMEM limit) |
| 20       | 20       | 16     | 16     | 6       | 768     | 207.9             |
| 20       | 20       | 16     | 16     | 6       | 1024    | **207.8**         |
| 20       | 20       | 16     | 16     | 6       | 2048    | 207.9             |
| **standalone**    | —        | —      | —      | —       | —       | **225.3**         |

**Best fused: 207.8 μs = 7.8% faster than standalone (225.3 μs).**

**Key findings:**
- Default config (96 blocks, 10 stages) is **2.8x worse** than tuned (591 vs 208 μs)
- Deep pipelines (20-24 stages) are critical — hide NVLink + flag-polling latency
- Dispatch pipeline depth matters more than permute (d=20,p=10=213 beats d=10,p=20=233)
- Too deep backfires (d=30=273) — SMEM constraint forces stage reduction, hurting both dispatch and permute
- `additional_s2g_dispatch=6` vs 10 vs 0: 6 is fine, 0 is worse
- Block count saturates at ~768-1024; 512 is within 0.9% of optimum
- The tuning closes the standalone gap (unlike combine+unpermute which stays 14% behind)

### Fused Combine+Unpermute Pipeline Tuning (B300, H=512, E=32, TOPK=36)

Sweeping unpermute stages, block count, and combine pipeline. Standalone baseline:
combine kernel = 246.0 µs + unpermute kernel = 134.6 µs = **380.6 µs**.

| combine G2S/S2G | ug2s/us2g | blocks    | Fused kernel (μs) | vs standalone |
| --------------- | --------- | --------- | ----------------- | ------------- |
| 64/8            | 2/2 (def) | 96 (def)  | **1095.1**            | 2.88x slower  |
| 64/8            | 2/2       | 512       | 618.6             |               |
| 64/8            | 2/2       | 1024      | 548.9             |               |
| 64/8            | 2/2       | 2048      | 548.8             |               |
| 64/8            | 2/2       | 4096      | 548.9             |               |
| 64/8            | 4/4       | 512       | 504.2             |               |
| 64/8            | 4/4       | 1024      | 463.4             |               |
| 64/8            | 8/8       | 1024      | 438.6             |               |
| 64/8            | 32/32     | 1024      | **434.4**             | **14% slower** |
| 16/2            | 24/24     | 1024      | 674.1             | combine killed† |
| 64/8, group=4   | 8/8       | 1024      | 585.9             | combine killed† |
| **standalone**  | —         | —         | **380.6**             | —              |

† Reducing combine pipeline (G2S=16→2) or increasing group size (2→4) cripples
the combine kernel itself, making fused overall worse despite better unpermute.

**Key findings:**
- Default fused (96 blocks, ug2/us2): **2.88x slower** than standalone
- Deeper unpermute stages help (2→32 = 548→434 µs, 21% improvement) but plateau
- More blocks help (96→1024 = 1095→549 µs) but plateau completely after 1024
- **Best fused (434 µs) is 14% slower than standalone (381 µs)** — irreducible gap
- Combine pipeline must remain deep (G2S=64, S2G=8) — reducing it kills combine perf
- Unlike dispatch+permute which closed the gap at 7.8% faster, combine+unpermute
  has structural overhead that tuning cannot eliminate

**Why the gap persists for combine but not dispatch:**

1. **Producer-consumer serialization**: Unpermute blocks must wait for combine blocks
   to produce output chunks (chunk flag polling). This is a serial dependency that
   adds per-chunk latency. Dispatch→permute has the same pattern but dispatch is
   push-based (S2G writes) which completes chunks faster than combine's pull-based
   G2S reads.

2. **Unpermute does more work**: Unpermute must gather tokens from multiple local
   experts per destination token (reduction across experts), while permute just
   scatters already-dispatched tokens. More work per chunk = more latency exposed.

3. **Combine is already the slower half**: At 246 µs vs unpermute's 135 µs,
   combine dominates the fused timeline. The unpermute blocks finish faster than
   permute blocks (135 vs 122 µs standalone) but must wait for combine to produce
   data — so unpermute's speed advantage is wasted on flag polling.

**Conclusion:** Non-fused combine+unpermute (381 µs) is always preferred over fused
(434 µs at best). Fused dispatch+permute can beat standalone with proper tuning
(208 vs 225 µs), but fused combine+unpermute has a permanent ~14% penalty.

### NCU Stall Profile: Standalone Dispatch Kernel (H=512, B300)

**Config**: Same as combine profile above. 32 SMs, non-fused dispatch_with_permute.

| Metric           | Dispatch (standalone) | Combine (standalone) |
| ---------------- | --------------------- | -------------------- |
| **Duration**         | **116.83 μs**             | **254.43 μs**            |
| Warp occupancy   | 6.24%                 | 9.08%                |
| **long_scoreboard**  | **57.95%**                | **26.59%**               |
| **wait**             | **21.36%**                | **25.92%**               |
| **barrier**          | **0.05%**                 | **15.23%**               |
| short_scoreboard | 7.40%                 | 7.22%                |
| not_selected     | 0.00%                 | 1.09%                |
| NVL RX           | 531 KB                | 73.90 MB             |
| NVL TX           | 78.01 MB              | 12.32 MB             |

#### Dispatch vs Combine Architecture

| Property        | Dispatch (push/S2G)            | Combine (pull/G2S)              |
| --------------- | ------------------------------ | ------------------------------- |
| Data direction  | TX-dominant (78 MB TX)         | RX-dominant (74 MB RX)         |
| TMA pattern     | S2G writes (fire-and-forget)   | G2S reads (request-response)   |
| Warp sync       | No inter-warp barriers needed  | Named barriers (G2S → reduce)  |
| Top stall       | long_scoreboard 58% (S2G backpressure) | long_scoreboard 27% + wait 26% + barrier 15% |
| Throughput      | 650 GB/s (72% NVL peak)        | 271 GB/s (30% NVL peak)        |

**Why dispatch is 2.4x faster than combine (same data volume):**

Dispatch uses **push** (S2G writes) — fire-and-forget TMA operations. The warp issues TMA S2G
and moves on; NVLink handles delivery asynchronously. The 58% long_scoreboard is TMA S2G
backpressure (TMA issue queue full, waiting for NVLink to drain writes).

Combine uses **pull** (G2S reads) — each TMA requires a round-trip (request → remote HBM read →
NVLink response). This 2-way latency is fundamentally slower. Additionally, combine requires
inter-warp barriers (15%) for the G2S → reduce handoff, adding overhead that dispatch avoids.

#### Why NCU Failed for Fused Kernels

Fused kernels allocate buffers for both operations (dispatch+permute or combine+unpermute),
resulting in a larger IPC-mapped memory footprint. ncu's kernel replay must save/restore all
accessible GPU memory before each replay pass. On B300 (275 GB HBM + IPC mapped remote memory),
this exceeds ncu's capacity for the fused variants, causing "Failed to save memory for replay"
on all attempts. With `--lockstep-kernel-launch`, one rank's failure causes all others to
deadlock (they wait forever for the failed rank to participate in replay).

---

## Summary of Findings (May 2026)

### Test Configuration
- **Hardware**: B300 SXM6 NVL8, 128 SMs, 275 GB HBM, NVLink 900 GB/s per GPU
- **Problem**: H=512, E=32, TOPK=36, T=8192, bf16 (latent MoE canonical)
- **Branches**: `hhanyu/hybrid-ep-sparse-opt` (ballot+ffs permute/unpermute + dispatch prob opt)

### Best Standalone Results (Non-Fused)

| Kernel             | Time     | BW/Throughput    | Config                                      |
| ------------------ | -------- | ---------------- | ------------------------------------------- |
| dispatch           | 102.5 us | 651 GB/s (72%)   | SMS=32, prob opt (E-per-rank slice TMA)     |
| permute            | 109.7 us | --               | ballot+ffs, standalone 2048 blocks          |
| combine (w/ probs) | 246.0 us | 271 GB/s (30%)   | SMS=32, G2S=64, S2G=8, GROUP=2, BATCH=16   |
| unpermute          | 134.6 us | --               | ballot+ffs, standalone 2048 blocks          |
| **Total (kernel only)** | **592.8 us** | --          | --                                          |

### Fused vs Standalone: Final Verdict

| Operation         | Best standalone | Best fused     | Verdict                                     |
| ----------------- | --------------- | -------------- | ------------------------------------------- |
| dispatch+permute  | 212.2 us        | **207.8 us**   | **Fused 2.1% faster** (d20,p20,inflight16,blocks1024) |
| combine+unpermute | 380.6 us        | 434.4 us       | **Standalone 14% faster** (gap irreducible) |

### Key Findings

**1. Dispatch is push, combine is pull -- fundamental asymmetry.**
Dispatch uses S2G TMA writes (fire-and-forget): 651 GB/s (72% of NVLink peak). Combine uses G2S
TMA reads (request-response round-trip): 271 GB/s (30%). Dispatch is 2.4x faster for the same
data volume. This is structural -- combine can never match dispatch throughput.

**2. Fused dispatch+permute can beat standalone with proper tuning.**
Default fused config (96 permute blocks, 10 stages): 591 us (2.8x worse than standalone).
Tuned (d=20, p=20, inflight=16, blocks=1024): 207.8 us (2.1% faster). Deep pipelines hide
flag-polling latency; enough blocks ensure occupancy.

**3. Fused combine+unpermute has an irreducible ~14% penalty.**
Even with optimal tuning (ug=32, blocks=1024, BATCH=16): 434 us vs standalone 381 us.
More blocks, deeper stages, reduced batch sizes -- none close the gap. The combine's pull-based
G2S reads produce output chunks slowly, and unpermute blocks serialize behind them via flag
polling. The gap is structural: unpermute must wait for combine, and combine is inherently slow.

**4. Combine pipeline depth (G2S=64) is already at the ceiling.**
Doubling to G2S=128: zero improvement. Pipeline is not depth-limited. The bottleneck is the
G2S TMA issue rate (1 elected thread per warp x 2 warps = 2 TMA issuers vs dispatch's 3).
Combine is not limited by SMEM, occupancy, or pipeline prefetch -- it's limited by NVLink
read round-trip latency of 1 KB transfers.

**5. GROUP=2, BATCH=16 is optimal for H=512 combine.**
GROUP=1 (SMS=64): same perf, wastes SMs. GROUP=4+: FIFO too shallow (<=16 slots per pipeline),
fails to hide NVLink latency. BATCH=16 covers all ~8 sources in one barrier cycle; smaller
batches add per-barrier overhead with no benefit.

**6. NCU profiling of collective kernels requires the TCP communicator.**
`--communicator tcp --lockstep-kernel-launch` coordinates kernel replay across all 8 ranks
simultaneously. Without it, single-rank replay fails (can't save 275 GB HBM) and application
replay fails (multi-process inconsistency). NCU injects LD_PRELOAD into child processes,
breaking nvcc/gcc -- JIT cache must be pre-warmed in a separate run.

**7. ~1 in 8 ranks consistently fails ncu memory save on B300.**
The other 7 succeed. This is sufficient for gathering data (all ranks execute identical
kernels). Fused kernels fail more frequently due to larger buffer allocations.

### Overall Recommendation for Latent MoE (H=512)

Use **non-fused standalone** kernels for all operations:
- `dispatch_with_permute` (non-fused): dispatch 102 us + permute 110 us = 212 us
- `combine_with_unpermute` (non-fused): combine 246 us + unpermute 135 us = 381 us
- Total kernel time: **593 us**

Fused dispatch+permute is slightly faster (208 us) but the difference is marginal (2%) and not
worth the additional JIT variant complexity. Fused combine+unpermute is strictly worse (14%).
The tuning effort for fused kernels is significant (pipeline stages, block counts, inflight
TMAs) and the results are fragile across hardware/configurations.

### Best Configs (Env Vars)

**Recommended: Non-fused standalone**

```bash
HIDDEN_DIM=512
NUM_TOKENS_PER_RANK=8192
MAX_NUM_OF_TOKENS_PER_RANK=8192
NUM_LOCAL_EXPERTS=32
TOPK=36
NUM_SMS_DISPATCH=32
NUM_SMS_COMBINE=32
NUM_OF_STAGES_G2S_COMBINE_API=64
NUM_OF_STAGES_S2G_COMBINE_API=8
NUM_TOKENS_COMBINE_REDUCE_BATCH_COMBINE_API=16
NUM_OF_TOKENS_PER_GROUP_COMBINE_API=2
```

**Optional: Fused dispatch+permute** (adds 2% speedup with significant tuning complexity)

```bash
# In addition to the above:
NUM_BLOCKS_PERMUTE=1024
NUM_OF_STAGES_DISPATCH_API=20
NUM_OF_STAGES_PERMUTE_BLOCK_DISPATCH_API=20
NUM_OF_IN_FLIGHT_S2G_DISPATCH_API=16
NUM_OF_IN_FLIGHT_S2G_PERMUTE_BLOCK_DISPATCH_API=16
```

**Fused combine+unpermute** (434 us, 14% slower than standalone — not recommended)

```bash
# In addition to the standalone config above:
NUM_BLOCKS_UNPERMUTE=1024
NUM_OF_STAGES_G2S_UNPERMUTE_BLOCK=8
NUM_OF_STAGES_S2G_UNPERMUTE_BLOCK=8
```
