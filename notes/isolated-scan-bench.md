# Isolated Hybrid-EP Scan Benchmark

## Setup

- Hardware: B300 NVL node (`umb-b300-003`)
- Benchmark source: `scripts/isolated_scan_bench.cu`
- Runner: `scripts/run_isolated_scan_bench.sh`
- This benchmark is standalone CUDA. It does not construct `HybridEPBuffer`, does not use IPC,
  and does not use process groups.
- Input routing is dense `int16` top-k indices.
- Template shape matches the real NVL72 scan shape:
  `scan<threads, blocks, 256, 12288, 64, 72, 1, localE, topk>`

## 32 Local Experts, TopK 36

Log: `/home/scratch.hhanyu_gpu/projects/moe/bench_logs/isolated_scan_20260525_213124`

These results were compiled with `HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE=1`.

| Version | Template | Time |
| --- | ---: | ---: |
| before | `scan<256,108,256,12288,64,72,1,32,36>` | 2044.7 us |
| after | `scan<256,108,256,12288,64,72,1,32,36>` | 686.4 us |
| before | `scan<256,64,256,12288,64,72,1,32,36>` | 3391.8 us |
| after | `scan<256,64,256,12288,64,72,1,32,36>` | 1095.9 us |
| before | `scan<512,108,256,12288,64,72,1,32,36>` | 1384.7 us |
| after | `scan<512,108,256,12288,64,72,1,32,36>` | 583.1 us |
| before | `scan<512,64,256,12288,64,72,1,32,36>` | 2222.5 us |
| after | `scan<512,64,256,12288,64,72,1,32,36>` | 907.1 us |

Best tested config after the local-expert bitset patch is `512/108` at 583.1 us.

## 32 Local Experts, TopK 8

Log: `/home/scratch.hhanyu_gpu/projects/moe/bench_logs/isolated_scan_20260525_213822`

These results were compiled with `HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE=1`.

| Version | Template | Time |
| --- | ---: | ---: |
| after | `scan<256,108,256,12288,64,72,1,32,8>` | 391.2 us |
| after | `scan<256,64,256,12288,64,72,1,32,8>` | 633.1 us |
| after | `scan<512,108,256,12288,64,72,1,32,8>` | 345.5 us |
| after | `scan<512,64,256,12288,64,72,1,32,8>` | 522.3 us |

## 8 Local Experts, TopK 6

Log: `/home/scratch.hhanyu_gpu/projects/moe/bench_logs/isolated_scan_20260525_214207`

These results were compiled with `HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE=1`.

| Version | Template | Time |
| --- | ---: | ---: |
| before | `scan<256,108,256,12288,64,72,1,8,6>` | 579.8 us |
| after | `scan<256,108,256,12288,64,72,1,8,6>` | 427.5 us |
| before | `scan<256,64,256,12288,64,72,1,8,6>` | 914.4 us |
| after | `scan<256,64,256,12288,64,72,1,8,6>` | 682.1 us |
| before | `scan<512,108,256,12288,64,72,1,8,6>` | 412.4 us |
| after | `scan<512,108,256,12288,64,72,1,8,6>` | 350.8 us |
| before | `scan<512,64,256,12288,64,72,1,8,6>` | 650.1 us |
| after | `scan<512,64,256,12288,64,72,1,8,6>` | 541.7 us |

## Notes

- The implemented optimization is a local-expert register bitset in dense fused scan.
- The local-expert register bitset was moved into the common dense scan path, so it now applies
  to `HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE=0` as well as `=1`.
- No rank-bitset optimization has been implemented yet.
- Naming is confusing: `HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE` has different effective meanings
  depending on which JIT kernel is being built.
  - For the scan kernel, the macro means scan also produces permute-preprocessing metadata
    (`dense_chunk_layout`, `dense_to_expert_map`, `tokens_per_expert`). This can be true even when
    dispatch and permute are not fused at runtime.
  - For the dispatch kernel, the macro means dispatch and permute are fused into one kernel.
  - Current `opt2` compiles the scan with permute-preprocessing metadata whenever
    `enable_permute=True`. Therefore both fused and non-fused `dispatch_with_permute` use a
    "fat" scan. In non-fused mode, a standalone `permute_kernel` still appears after the normal
    dispatch kernel.
- Smaller top-k variants (`32/8`, `8/6`) were requested but large generated scan templates caused
  `ptxas` to be killed or exceed the 120 second per-template compile timeout in this environment.
  The runner now skips compile failures/timeouts instead of aborting the sweep.
- The real trace template `scan<256,108,256,12288,64,72,1,32,36>` is represented directly by
  this isolated benchmark.

## No Permute-Fusion Macro (`HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE=0`)

These results match the use case where scan is compiled without `HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE`.
They use the common dense local-expert register bitset path.

### 32 Local Experts, TopK 36

Log: `/home/scratch.hhanyu_gpu/projects/moe/bench_logs/isolated_scan_20260525_221837`
Before log: `/home/scratch.hhanyu_gpu/projects/moe/bench_logs/isolated_scan_20260525_222750`

| Version | Template | Time |
| --- | ---: | ---: |
| before | `scan<256,108,256,12288,64,72,1,32,36>` | 702.3 us |
| after | `scan<256,108,256,12288,64,72,1,32,36>` | 625.4 us |
| before | `scan<256,64,256,12288,64,72,1,32,36>` | 1152.5 us |
| after | `scan<256,64,256,12288,64,72,1,32,36>` | 1018.2 us |
| before | `scan<512,108,256,12288,64,72,1,32,36>` | 600.9 us |
| after | `scan<512,108,256,12288,64,72,1,32,36>` | 525.6 us |
| before | `scan<512,64,256,12288,64,72,1,32,36>` | 967.3 us |
| after | `scan<512,64,256,12288,64,72,1,32,36>` | 854.9 us |

### 32 Local Experts, TopK 8

Log: `/home/scratch.hhanyu_gpu/projects/moe/bench_logs/isolated_scan_20260525_222023`

| Version | Template | Time |
| --- | ---: | ---: |
| after | `scan<256,108,256,12288,64,72,1,32,8>` | 389.3 us |
| after | `scan<256,64,256,12288,64,72,1,32,8>` | 607.3 us |
| after | `scan<512,108,256,12288,64,72,1,32,8>` | 324.3 us |
| after | `scan<512,64,256,12288,64,72,1,32,8>` | 527.6 us |

### 8 Local Experts, TopK 6

Log: `/home/scratch.hhanyu_gpu/projects/moe/bench_logs/isolated_scan_20260525_222204`

| Version | Template | Time |
| --- | ---: | ---: |
| after | `scan<256,108,256,12288,64,72,1,8,6>` | 416.6 us |
| after | `scan<256,64,256,12288,64,72,1,8,6>` | 656.8 us |
| after | `scan<512,108,256,12288,64,72,1,8,6>` | 336.2 us |
| after | `scan<512,64,256,12288,64,72,1,8,6>` | 537.2 us |

The original no-permute-fusion `before` run initially failed because an older isolated harness
allocated `local_expert_routing_map` as a 1-byte placeholder. The no-permute-fusion scan path
writes that output, unlike the permute-fusion path. After allocating
`rows * local_experts` bytes, the `before` no-permute-fusion benchmark ran successfully.
