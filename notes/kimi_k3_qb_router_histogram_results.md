# Kimi K3 QB Router Histogram Results

## Outcome

Transformer Engine now has two opt-in Kimi K3 Quantile Balancing (QB)
histogram paths on branch `hhanyu/qb-fused-router-histogram`, based directly on
main-branch commit `f1d5f8d5222b5789351dd6dc145253a7ac727b78`:

1. `two_kernel`: the router emits the Top-(k+1) cutoff and a second hierarchical
   histogram kernel accumulates counts.
2. `fused_atomic`: the router computes the same bins in its epilogue and directly
   performs global int32 atomic accumulation.

Both paths retain only the actual Top-k routes. The extra selected expert supplies
the QB cutoff and is never written into the routing map or dense routing indices.
The existing non-QB specialization is unchanged.

The implementation does not materialize the proposed
`int32[num_tokens, num_experts]` bin-index buffer. Both algorithms reuse the
router's existing FP32 raw sigmoid scores. The two-kernel path additionally writes
one FP32 cutoff per token; the fused path keeps the cutoff local.

## Correctness

The oracle is a dedicated pure-PyTorch implementation. It independently computes
sigmoid scores, biased Top-(k+1), deterministic cutoff-tie compaction, normalized
Top-k probabilities, bin indices, and histogram accumulation.

Focused B300 results:

```text
25 passed, 3648 deselected, 4 warnings in 21.37s
```

Coverage includes:

- both QB implementations;
- Top-8 and Top-16, exercising simple and radix router paths;
- BYTEMAP, BITMAP_U8, and dense int16/int32/int64 routing outputs;
- exact routing-map and histogram equality;
- forward probability and backward-gradient parity;
- constructed cutoff ties and bin-boundary clamping;
- accumulation across two microbatches; and
- unsupported-configuration validation.

The final editable TE build and the post-build test both passed with:

```text
NVTE_BUILD_THREADS_PER_JOB=4
NVTE_CUDA_ARCHS="100;103a;"
NVTE_USE_CCACHE=1
/usr/bin/python3 -m pip install --no-build-isolation -e '.[test]' --verbose
```

## B300 Performance

The standalone benchmark is `scripts/benchmark_qb_router.py`. It checks all four
implementations for matching routes, probabilities, and QB histograms before
timing. Measurements use 100 warmups, 500 timed calls, ten CUDA-event samples,
896 experts, Top-16, 1,000 bins, and FP32 logits.

Median latency in milliseconds:

| Routing | Tokens | PyTorch QB | TE no QB | QB two-kernel | QB fused atomic | Fused / PyTorch | Fused / TE no QB | Fused / two-kernel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BYTEMAP | 256 | 0.297896 | 0.030832 | 0.040714 | 0.036333 | 0.122 | 1.178 | 0.892 |
| BYTEMAP | 1,024 | 0.343485 | 0.032320 | 0.050766 | 0.038981 | 0.113 | 1.206 | 0.768 |
| BYTEMAP | 4,096 | 0.460332 | 0.063559 | 0.087384 | 0.085284 | 0.185 | 1.342 | 0.976 |
| BYTEMAP | 8,192 | 0.646173 | 0.105448 | 0.141065 | 0.133098 | 0.206 | 1.262 | 0.944 |
| BYTEMAP | 16,384 | 1.135861 | 0.194089 | 0.280996 | 0.247061 | 0.218 | 1.273 | 0.879 |
| dense int16 | 256 | 0.285194 | 0.030907 | 0.040600 | 0.037092 | 0.130 | 1.200 | 0.914 |
| dense int16 | 1,024 | 0.333949 | 0.031876 | 0.050848 | 0.038818 | 0.116 | 1.218 | 0.763 |
| dense int16 | 4,096 | 0.453073 | 0.062932 | 0.085379 | 0.086220 | 0.190 | 1.370 | 1.010 |
| dense int16 | 8,192 | 0.632043 | 0.104072 | 0.136491 | 0.131487 | 0.208 | 1.263 | 0.963 |
| dense int16 | 16,384 | 1.119778 | 0.192712 | 0.277453 | 0.245895 | 0.220 | 1.276 | 0.886 |

The fused QB kernel is not slower than the unfused PyTorch QB implementation. It
uses 11.3% to 22.0% of the PyTorch latency, a 4.6x to 8.8x speedup.

Relative to the two-kernel implementation, fused atomic is normally 2.4% to 23.7%
faster. The dense 4,096-token case is effectively tied, with fused atomic 1.0%
slower in this sample. The fused QB work costs 17.8% to 37.0% over the existing TE
router without QB; that comparison is the QB feature overhead, not a PyTorch
regression.

Recommendation: use `fused_atomic` as the default experimental QB path, while
retaining `two_kernel` as a reference and as a possible alternative if larger
expert/bin configurations make global atomic contention dominant.

## MCore integration

MCore branch `hhanyu/qwen35-opt` now feature-detects the three TE QB arguments and
threads only the recommended `fused_atomic` path. Each router owns a persistent FP32
`qb_bin_bounds[2]` buffer and a nonpersistent int32
`qb_histogram[num_experts, num_bins]`. `finalize_model_grads` performs one stacked
TPxDPxCP histogram all-reduce for all local MoE layers, recovers and mean-centers the
quantile biases, updates both bias and bounds in place, then clears the histogram.

The K3-like perf launcher supports either:

```text
BIAS_UPDATE_METHOD=quantile
BIAS_UPDATE_METHOD=sign
```

EP2 validation on `umb-b300-dp-189` used HybridEP dense routing and MXFP8 grouped
SiTU-GLU. At the K3 router shape (`E=896`, Top-16, 1,000 bins), five measured
warm-cache iterations reported:

| Method | Forward | Backward | Finalization | End to end |
| --- | ---: | ---: | ---: | ---: |
| sign | 15.573 ms | 13.225 ms | 0.589 ms | 29.387 ms |
| quantile | 15.333 ms | 13.160 ms | 0.761 ms | 29.254 ms |

The QB-specific step-end overhead versus signed updating was about 0.17 ms in this
small one-layer EP2 harness. Full-iteration CUDA graph capture failed in HybridEP for
both QB and signed controls at the same `cudaErrorStreamCaptureInvalidated` site, so
that limitation is not caused by the QB integration.

## Environment and Logs

Final environment:

| Component | Version |
| --- | --- |
| Node / GPU | `umb-b300-dp-141`, NVIDIA B300 |
| Container | `test_container_2606` |
| PyTorch | `2.13.0a0+8145d630e8.nv26.6.54250401` |
| Transformer Engine | `2.19.0.dev0+c0517995` after the editable build |
| nvidia-cutlass-dsl | 4.4.2 |
| cuDNN Frontend | 1.26.0 |

Persistent remote logs:

```text
/home/scratch.hhanyu_gpu/projects/moe/TE/logs/qb_histogram/build_final_1260_442.log
/home/scratch.hhanyu_gpu/projects/moe/TE/logs/qb_histogram/test_qb_after_final_build.log
/home/scratch.hhanyu_gpu/projects/moe/TE/logs/qb_histogram/final_evidence_1260_442.log
/home/scratch.hhanyu_gpu/projects/moe/logs/qb_histogram/benchmark_qb_vs_pytorch_b300_100x500.log
/home/scratch.hhanyu_gpu/projects/moe/logs/qb_histogram/benchmark_qb_vs_pytorch_b300_100x500.json
```
