# Fused Router Trace Benchmarks

This directory contains matched measurements for the fused-router trace checkpoints.

## Checkpoints

| Label | Branch | Router checkpoint |
| --- | --- | --- |
| `trace-back` | `hhanyu/trace-back` | Before large top-K support |
| `trace-back-2821` | `hhanyu/trace-back-2821` | #2821 large top-K and expert counts |
| `trace-back-3012` | `hhanyu/trace-back-3012` | #3012 fused router optimizations |
| `trace-back-3129` | `hhanyu/trace-back-3129` | #3129 dense router output |

## Environment

- Node: `umb-b300-dp-146`
- GPU: NVIDIA B300 SXM6 AC, SM 10.3
- Container: `test_container_2606`
- CUDA: 13.3
- PyTorch: `2.13.0a0+8145d630e8.nv26.06`
- Each checkpoint was built sequentially in the same `TE/` checkout.
- Builds sourced `~/.bashrc` and enabled ccache.

```bash
source ~/.bashrc
export PATH="/home/hhanyu/.pixi.x86_64/bin:${PATH}"
export NVTE_BUILD_THREADS_PER_JOB=4
export NVTE_CUDA_ARCHS="100;103a;"
export NVTE_USE_CCACHE=1

/usr/bin/python3 -m pip install \
  --no-build-isolation \
  -e '.[test]' \
  --verbose
```

## Benchmark Configuration

All runs used `scripts/test_fused_topk.py` with:

- Tokens: `4096`, `16384`, `65536`, `262144`
- Expert/top-K pairs: `256/8`, `384/6`, `512/10`, `512/22`, `896/16`, `2304/16`, `2304/36`
- Score functions: post-softmax and sigmoid
- Kernels: top-K and aux-loss score
- Passes: forward and raw backward
- Dtype: FP32
- Grouped top-K: disabled (`group_topk=0`)
- Warmup / timed iterations: `20` / `100`

```bash
export NVTE_RADIX_TOPK_THRESHOLD=10

/usr/bin/python3 scripts/test_fused_topk.py \
  --mode benchmark \
  --kernel topk aux_loss \
  --pass forward backward_raw \
  --router-shape 256/8 384/6 512/10 512/22 896/16 2304/16 2304/36 \
  --num-tokens 4096 16384 65536 262144 \
  --score-function softmax sigmoid \
  --group-topk 0 \
  --topk-output-mode sparse \
  --warmup 20 \
  --iters 100
```

`trace-back` has no radix implementation. `trace-back-2821` has a hard-coded
radix threshold of 16. The threshold-10 environment setting applies to the p3R
dispatch in `trace-back-3012` and `trace-back-3129`, so their `512/10` cases use
radix selection.

## Dense Output

`trace-back-3129-dense-int16-topk-forward.csv` measures dense int16 top-K index
output for top-K forward only. `trace-back-3129.csv` and the top-K forward plot
substitute those rows for the sparse routing-map rows; all other 3129 measurements
remain sparse.

## Files

- `trace-back*.csv`: per-checkpoint raw benchmark rows
- `router_benchmark_combined.csv`: all checkpoints after the 3129 dense-forward substitution
- `topk_*.png`, `aux_loss_*.png`: effective-bandwidth comparison plots
- `../plot_trace_back_benchmark.py`: regenerates the merged CSV and plots

## Plotting

Each panel includes the mean `ref_gbps` across checkpoints as a black dashed
unfused-reference line. Each score-function row shares a y-axis range across all
token counts, starts at zero, and extends five percent above the row maximum.
