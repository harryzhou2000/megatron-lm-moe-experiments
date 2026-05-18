# Nsys Profiling Analysis: Qwen3-Next-80B-A3B E72 Full Training Iteration

## Test Environment

- **Model**: Qwen3-Next-80B-A3B_E72 (72 experts, top-K=36)
- **Parallelism**: TP1 PP1 EP72 VPP1, MBS3 GBS864
- **Hardware**: 72x NVIDIA GB200 SXM6 (152 SMs, 197.6 GB HBM, Compute 10.0)
- **Profile**: 3 training iterations captured via CUDA graph replay
- **Reports**: 72 ranks, converted from `.nsys-rep` → `.sqlite` via `nsys export`
- **Config tag**: `Head128-hepS-AG1-profile-SM32-Comb_36_4_2_4`

## Iteration Structure

Iterations were detected from kernel timeline density (NVTX is useless with
CUDA graph replay). The algorithm bins kernel GPU-time into 10ms windows and
finds contiguous "dense compute" phases.

| Iter | Compute Window (ms) | Wall-Clock | GPU Time | Optimizer/Sync |
| ---- | ------------------- | ---------- | -------- | -------------- |
| 1    | 2110 – 3880         | 1770 ms    | 2332.6 ms | 10 ms         |
| 2    | 4760 – 6530         | 1770 ms    | 2330.7 ms | 450 ms        |
| 3    | 7410 – 9180         | 1770 ms    | 2330.7 ms | 10 ms         |

- **GPU/Wall ratio**: 1.32x (multi-stream overlap)
- All 3 iterations are identical in structure (CUDA graph replay)
- Iteration 2 used as steady-state reference below

## Per-Kernel Time Budget (Rank 0, Iteration 2)

Wall-clock = 1770ms. "% Wall" shows how much of the iteration each kernel fills.
Cumulative exceeds 100% due to multi-stream overlap (1.32x).

### Top 20 Kernels

| #  | Kernel                         | Category                   | Invoc | GPU (ms) | % Wall | Cum %  |
| -- | ------------------------------ | -------------------------- | ----- | -------- | ------ | ------ |
| 1  | **combine_kernel**             | **MoE combine**            | 192   | 374.1    | **21.1%** | 21.1%  |
| 2  | device_sync_kernel             | MoE device_sync            | 768   | 146.5    | 8.3%   | 29.4%  |
| 3  | permute_preprocessing_kernel   | MoE permute_preprocess     | 96    | 128.6    | 7.3%   | 36.7%  |
| 4  | dispatch_kernel                | MoE dispatch               | 192   | 123.0    | 6.9%   | 43.6%  |
| 5  | unpermute_kernel               | MoE permute                | 192   | 123.0    | 6.9%   | 50.6%  |
| 6  | scan                           | MoE scan                   | 96    | 100.2    | 5.7%   | 56.2%  |
| 7  | _paged_stash_pop_kernel        | MoE paged_stash_pop        | 552   | 84.8     | 4.8%   | 61.0%  |
| 8  | _paged_stash_copy_kernel       | MoE paged_stash_copy       | 552   | 83.9     | 4.7%   | 65.8%  |
| 9  | flash_bprop (sdpa bwd)         | Attention (cuDNN Flash)    | 96    | 73.8     | 4.2%   | 69.9%  |
| 10 | grouped_gemm_dglu_dbias        | MoE grouped_gemm           | 96    | 73.3     | 4.1%   | 74.1%  |
| 11 | grouped_gemm_quant             | MoE grouped_gemm           | 192   | 69.6     | 3.9%   | 78.0%  |
| 12 | NCCL AllGather                 | NCCL                       | 105   | 53.4     | 3.0%   | 81.0%  |
| 13 | fused_moe_aux_loss_fwd         | MoE aux_loss               | 96    | 50.3     | 2.8%   | 83.9%  |
| 14 | grouped_gemm_wgrad             | MoE grouped_gemm           | 192   | 48.1     | 2.7%   | 86.6%  |
| 15 | grouped_gemm_glu_bias          | MoE grouped_gemm           | 96    | 45.7     | 2.6%   | 89.2%  |
| 16 | ln_tma_fwd_2D_kernel           | LayerNorm/RMSNorm          | 96    | 42.3     | 2.4%   | 91.5%  |
| 17 | fused_score_aux_loss_fwd       | MoE score_aux_loss         | 96    | 37.3     | 2.1%   | 93.7%  |
| 18 | permute_kernel                 | MoE permute                | 192   | 35.7     | 2.0%   | 95.7%  |
| 19 | quantize_mxfp8_kernel          | Quantization (MXFP8)       | 2928  | 35.0     | 2.0%   | 97.6%  |
| 20 | fused_topk_fwd                 | MoE topk                   | 96    | 34.8     | 2.0%   | 99.6%  |

**Top 6 kernels = 56% of wall-clock. Top 15 = 89%.**

### By Category

| Category                  | GPU (ms) | % Wall | Cum %  | Invocations |
| ------------------------- | -------- | ------ | ------ | ----------- |
| **MoE combine**           | **374.1**| **21.1%** | 21.1% | 192         |
| GEMM (CUTLASS/nvjet)      | 183.3    | 10.4%  | 31.5%  | 2220        |
| MoE permute+unpermute     | 158.7    | 9.0%   | 40.5%  | 384         |
| MoE device_sync           | 146.5    | 8.3%   | 48.7%  | 768         |
| Other (sort, rope, etc.)  | 134.4    | 7.6%   | 56.3%  | 6800        |
| MoE permute_preprocess    | 128.6    | 7.3%   | 63.6%  | 96          |
| MoE dispatch              | 123.0    | 6.9%   | 70.5%  | 192         |
| Attention (cuDNN Flash)   | 101.2    | 5.7%   | 76.3%  | 192         |
| MoE scan                  | 100.2    | 5.7%   | 81.9%  | 108         |
| Elementwise               | 95.2     | 5.4%   | 87.3%  | 6580        |
| MoE paged_stash_pop       | 84.8     | 4.8%   | 92.1%  | 552         |
| MoE paged_stash_copy      | 83.9     | 4.7%   | 96.8%  | 552         |
| MoE grouped_gemm (all 4)  | 237.0    | 13.4%  | —      | 864         |
| LayerNorm/RMSNorm         | 71.4     | 4.0%   | —      | 1164        |
| Quantization (MXFP8)      | 66.4     | 3.7%   | —      | 3120        |
| MoE aux_loss              | 57.6     | 3.3%   | —      | 192         |
| MoE topk                  | 54.1     | 3.1%   | —      | 192         |
| NCCL AllGather             | 53.4     | 3.0%   | —      | 105         |
| MoE score_aux_loss        | 48.5     | 2.7%   | —      | 192         |

**MoE-specific kernels account for ~76% of wall-clock time.**

## Cross-Rank Analysis (72 Ranks, All 3 Iterations)

### Total Kernel Time Distribution

- **Mean**: 9722 ms
- **Min**: 7898 ms (rank 55)
- **Max**: 11533 ms (rank 65)
- Spread: **1.46x** between lightest and heaviest rank

### Load Imbalance by Category (Coefficient of Variation)

| Category                   | CV    | Mean (ms) | Min     | Max     | Interpretation                          |
| -------------------------- | ----- | --------- | ------- | ------- | --------------------------------------- |
| **NCCL AllGather**         | **0.418** | 2030  | 205     | 3856    | EP routing → wildly uneven collectives  |
| NCCL AllReduce u64         | 0.386 | 150       | 18      | 290     | Gradient sync varies by rank            |
| NCCL AllReduce u32         | 0.113 | 24        | 1.3     | 24      | Minor variance                          |
| NCCL AllReduce f32         | 0.112 | 596       | 42      | 631     | Large but relatively stable             |
| **MoE device_sync**       | **0.091** | 435   | 346     | 526     | SM contention from expert load imbalance |
| LayerNorm/RMSNorm          | 0.047 | 216       | 190     | 236     | Token count differences                 |
| MoE permute_preprocess     | 0.043 | 387       | 354     | 416     | Routing prep varies with expert assign  |
| MoE dispatch               | 0.026 | 363       | 347     | 385     | Moderate                                |
| Attention (cuDNN Flash)    | 0.018 | 304       | 293     | 317     | Stable                                  |

### Per-Iteration View (Iteration 2 Only, 72 Ranks)

When restricting to iteration 2 (4760–6530ms), the imbalance becomes more
pronounced because the NCCL warmup AllGather at profile start is excluded:

| Category                   | CV    | Mean (ms) | Min   | Max   |
| -------------------------- | ----- | --------- | ----- | ----- |
| NCCL AllReduce f32         | 0.592 | 147       | 0.2   | 212   |
| NCCL AllReduce u32         | 0.536 | 9.0       | 0.3   | 12    |
| NCCL AllReduce u64         | 0.419 | 51        | 8.4   | 100   |
| NCCL AllGather             | 0.371 | 47        | 19    | 86    |
| MoE permute_preprocess     | 0.336 | 85        | 46    | 138   |
| MoE paged_stash_copy       | 0.333 | 55        | 30    | 84    |
| MoE grouped_gemm_glu_bias  | 0.328 | 30        | 17    | 46    |
| MoE scan                   | 0.324 | 67        | 39    | 104   |
| MoE device_sync            | 0.288 | 94        | 52    | 146   |

## Key Observations

1. **combine_kernel is the #1 optimization target** — 21.1% of iteration
   wall-clock, 192 invocations per iteration (96 fwd + 96 bwd for 96 layers
   across the microbatches). Currently running with config
   `NUM_BLOCKS_UNPERMUTE=1024, NUM_OF_STAGES_G2S_UNPERMUTE_BLOCK=8,
   NUM_OF_STAGES_S2G_UNPERMUTE_BLOCK=8`.

2. **MoE routing overhead is massive** — combine + dispatch + device_sync +
   permute_preprocess + permute + unpermute + scan + paged_stash totals
   ~68% of wall-clock. The actual expert compute (grouped GEMMs) is only
   ~13.4%.

3. **NCCL AllGather has 19x spread across ranks** (full profile). This is
   inherent to expert parallelism — different ranks route different numbers
   of tokens, causing asymmetric collective sizes.

4. **device_sync at 8.3% is suspicious** — 768 invocations per iteration
   suggests 4 syncs per layer per microbatch. This is inter-SM
   synchronization within the dispatch/combine pipeline. Reducing sync
   points or overlapping with compute could be impactful.

5. **paged_stash_pop + paged_stash_copy together = 9.5%** — these manage
   paged memory for the token stash. 552 invocations each suggests 5-6 per
   layer. Could benefit from batching or fusion.

6. **Attention is only 5.7%** — not the bottleneck in this EP72 MoE config.
   The cuDNN Flash attention kernels are well-optimized.

7. **GPU/Wall ratio of 1.32x** means ~24% of GPU time overlaps across
   streams (dispatch/combine pipelining working).
