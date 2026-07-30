# Initial-profile GPU timing

Source report:
`/Users/harry/Downloads/Qwen3-Next-80B-A3B-syncfree_profile-rank0-date_26-01-10_time_16-12-55-1482268.nsys-rep`

The report was exported on 2026-07-30 with:

```sh
cd /Users/harry/Downloads
/Users/harry/projects/nsys-docker/nsys export -t sqlite \
  -o Qwen3-Next-80B-A3B-syncfree_profile-rank0-date_26-01-10_time_16-12-55-1482268.sqlite \
  Qwen3-Next-80B-A3B-syncfree_profile-rank0-date_26-01-10_time_16-12-55-1482268.nsys-rep
```

`gpu_kernel_timing.csv` sums `end - start` from every
`CUPTI_ACTIVITY_KIND_KERNEL` event in the rank-0 trace. Consequently, the
values are aggregate GPU execution time over the captured forward and backward
work, not a single-forward latency or wall-clock critical path. They can exceed
wall-clock time because the trace has three CUDA streams and overlapping work.

Classification rules:

- **Fused router:** Transformer Engine fused score, top-k, and MoE auxiliary-loss kernels.
- **Grouped GEMM (expert MLP):** CUTLASS `GroupProblemShape` kernels. This is the
  grouped expert-MLP compute path.
- **Other GEMM:** `nvjet`, non-grouped CUTLASS GEMM, cuBLASLt, and GEMV kernel
  families; FlashAttention is excluded.
- **Attention:** FlashAttention kernel family.
- **HybridEP dispatch/combine:** only the corresponding `hybrid_ep::*_kernel` symbols;
  permute, unpermute, scan, and preprocessing are excluded.

The exported SQLite is retained alongside the source report in `~/Downloads`.
