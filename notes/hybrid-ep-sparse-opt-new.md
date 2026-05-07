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
