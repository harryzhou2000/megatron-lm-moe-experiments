# Hybrid-EP Rebased Branch Bandwidth Results

## Setup

- Branch: `hhanyu/hybrid-ep-sparse-opt-2`
- DeepEP head: `e944b0d [HybridEP] Use rank bitsets in dense scan`
- Base: rebased onto `upstream/hybrid-ep`
- Node: `umb-b300-020`
- Container: `test_container_2603`
- Test: `DeepEP/tests/test_hybrid_ep.py --num-processes 8`

Environment:

```bash
HIDDEN_DIM=512
NUM_TOKENS_PER_RANK=8192
MAX_NUM_OF_TOKENS_PER_RANK=8192
NUM_OF_THREADS_PER_BLOCK_PREPROCESSING_API=512
NUM_SMS_DISPATCH=32
NUM_SMS_COMBINE=32
NUM_OF_STAGES_G2S_COMBINE_API=64
NUM_OF_STAGES_S2G_COMBINE_API=8
NUM_TOKENS_COMBINE_REDUCE_BATCH_COMBINE_API=16
NUM_OF_TOKENS_PER_GROUP_COMBINE_API=2
```

Both BF16 and FP8 correctness checks passed for:

- `dispatch+combine API`
- `dispatch+combine API (dense routing)`
- `dispatch_with_permute + combine_with_unpermute API (non-fused)`
- `dispatch_with_permute + combine_with_unpermute API (fused)`

## E=32, TopK=36

### Torch API Bandwidth

| Path | BF16 BW | BF16 Time | FP8 BW | FP8 Time |
| --- | ---: | ---: | ---: | ---: |
| dispatch, probs=True | 302.05 GB/s | 220.9 us | 159.06 GB/s | 216.2 us |
| combine, probs=True | 210.22 GB/s | 317.4 us | 210.49 GB/s | 316.8 us |
| dispatch, probs=False | 391.21 GB/s | 170.6 us | 221.11 GB/s | 155.5 us |
| combine, probs=False | 260.92 GB/s | 255.8 us | 261.22 GB/s | 255.3 us |
| dispatch+permute | 327.04 GB/s | 204.0 us | 184.12 GB/s | 186.7 us |
| combine+unpermute | 176.51 GB/s | 378.1 us | 176.94 GB/s | 376.9 us |
| fused dispatch+permute | 108.92 GB/s | 612.7 us | 56.44 GB/s | 609.2 us |
| fused combine+unpermute | 59.67 GB/s | 1118.3 us | 60.17 GB/s | 1108.3 us |

### Kernel-Only Bandwidth

| Kernel | BF16 BW | BF16 Time | FP8 BW | FP8 Time |
| --- | ---: | ---: | ---: | ---: |
| dispatch, probs=True | 647.93 GB/s | 103.0 us | 337.32 GB/s | 101.9 us |
| combine, probs=True | 267.57 GB/s | 249.4 us | 267.59 GB/s | 249.2 us |
| dispatch, probs=False | 706.65 GB/s | 94.4 us | 360.72 GB/s | 95.3 us |
| combine, probs=False | 317.83 GB/s | 210.0 us | 317.59 GB/s | 210.0 us |
| fused dispatch+permute | 112.57 GB/s | 592.8 us | 58.71 GB/s | 585.6 us |
| fused combine+unpermute | 60.87 GB/s | 1096.4 us | 61.25 GB/s | 1088.7 us |

Standalone permute/unpermute kernel breakdown:

| Kernel | BF16 Time | FP8 Time |
| --- | ---: | ---: |
| dispatch kernel in dispatch+permute | 114.4 us | 113.9 us |
| permute kernel | 64.2 us | 49.3 us |
| unpermute kernel | 104.1 us | 103.9 us |
| combine kernel in combine+unpermute | 249.2 us | 249.0 us |

## E=8, TopK=8

### Torch API Bandwidth

| Path | BF16 BW | BF16 Time | FP8 BW | FP8 Time |
| --- | ---: | ---: | ---: | ---: |
| dispatch, probs=True | 242.49 GB/s | 185.4 us | 113.53 GB/s | 204.5 us |
| combine, probs=True | 190.71 GB/s | 235.8 us | 189.93 GB/s | 237.1 us |
| dispatch, probs=False | 288.52 GB/s | 155.8 us | 163.62 GB/s | 141.9 us |
| combine, probs=False | 213.66 GB/s | 210.5 us | 211.91 GB/s | 212.5 us |
| dispatch+permute | 295.40 GB/s | 152.2 us | 184.24 GB/s | 126.0 us |
| combine+unpermute | 178.30 GB/s | 252.2 us | 177.73 GB/s | 253.4 us |
| fused dispatch+permute | 169.89 GB/s | 264.7 us | 112.80 GB/s | 205.8 us |
| fused combine+unpermute | 132.29 GB/s | 339.9 us | 133.52 GB/s | 337.3 us |

### Kernel-Only Bandwidth

| Kernel | BF16 BW | BF16 Time | FP8 BW | FP8 Time |
| --- | ---: | ---: | ---: | ---: |
| dispatch, probs=True | 473.58 GB/s | 94.9 us | 237.86 GB/s | 97.6 us |
| combine, probs=True | 237.77 GB/s | 189.1 us | 236.59 GB/s | 190.3 us |
| dispatch, probs=False | 503.89 GB/s | 89.2 us | 254.41 GB/s | 91.3 us |
| combine, probs=False | 257.22 GB/s | 174.8 us | 255.11 GB/s | 176.5 us |
| fused dispatch+permute | 183.76 GB/s | 244.7 us | 96.56 GB/s | 240.5 us |
| fused combine+unpermute | 140.99 GB/s | 318.9 us | 141.24 GB/s | 318.8 us |

Standalone permute/unpermute kernel breakdown:

| Kernel | BF16 Time | FP8 Time |
| --- | ---: | ---: |
| dispatch kernel in dispatch+permute | 104.6 us | 82.8 us |
| permute kernel | 24.1 us | 18.5 us |
| unpermute kernel | 39.1 us | 39.2 us |
| combine kernel in combine+unpermute | 189.1 us | 190.2 us |

## Scan Reference

Isolated scan, best tested `512/108` template:

| Case | Permute metadata | Time |
| --- | ---: | ---: |
| E=32, topk=36 | yes | 462.093 us |
| E=32, topk=36 | no | 323.519 us |
| E=8, topk=8 | yes | 239.158 us |
| E=8, topk=8 | no | 215.427 us |

Scan logs:

- `/home/scratch.hhanyu_gpu/projects/moe/bench_logs/isolated_scan_20260602_031431`
- `/home/scratch.hhanyu_gpu/projects/moe/bench_logs/isolated_scan_20260602_031518`

## H=4096, E=8, TopK=8

Same branch, node, and tuned env as above except `HIDDEN_DIM=4096`.

### Torch API Bandwidth

| Path | BF16 BW | BF16 Time | FP8 BW | FP8 Time |
| --- | ---: | ---: | ---: | ---: |
| dispatch, probs=True | 532.62 GB/s | 685.2 us | 451.44 GB/s | 417.5 us |
| combine, probs=True | 561.39 GB/s | 650.1 us | 566.20 GB/s | 645.6 us |
| dispatch, probs=False | 554.14 GB/s | 658.6 us | 486.20 GB/s | 387.7 us |
| combine, probs=False | 572.54 GB/s | 637.5 us | 577.05 GB/s | 633.5 us |
| dispatch+permute | 550.29 GB/s | 663.2 us | 504.91 GB/s | 373.3 us |
| combine+unpermute | 530.89 GB/s | 687.5 us | 534.92 GB/s | 683.4 us |
| fused dispatch+permute | 597.76 GB/s | 610.6 us | 489.68 GB/s | 384.9 us |
| fused combine+unpermute | 594.42 GB/s | 614.0 us | 594.94 GB/s | 614.4 us |

### Kernel-Only Bandwidth

| Kernel | BF16 BW | BF16 Time | FP8 BW | FP8 Time |
| --- | ---: | ---: | ---: | ---: |
| dispatch, probs=True | 775.88 GB/s | 470.4 us | 737.73 GB/s | 255.5 us |
| combine, probs=True | 724.72 GB/s | 503.6 us | 727.70 GB/s | 502.3 us |
| dispatch, probs=False | 782.48 GB/s | 466.4 us | 752.16 GB/s | 250.6 us |
| combine, probs=False | 730.61 GB/s | 499.5 us | 733.55 GB/s | 498.3 us |
| fused dispatch+permute | 619.49 GB/s | 589.1 us | 515.77 GB/s | 365.5 us |
| fused combine+unpermute | 611.06 GB/s | 597.3 us | 611.90 GB/s | 597.4 us |

Standalone permute/unpermute kernel breakdown:

| Kernel | BF16 Time | FP8 Time |
| --- | ---: | ---: |
| dispatch kernel in dispatch+permute | 473.1 us | 258.1 us |
| permute kernel | 142.4 us | 75.2 us |
| unpermute kernel | 154.2 us | 154.7 us |
| combine kernel in combine+unpermute | 503.5 us | 502.3 us |

## Upstream Hybrid-EP Comparison

Upstream baseline:

- Worktree: `upstream/hybrid-ep`
- Upstream head: `e0a5b1d Hybrid ep nixl restripe (#634)`
- Node: `umb-b300-020`
- Container: `test_container_2603`
- Same env knobs as the current-branch runs.

Note: upstream's test is branch-native and does not include the dense-routing correctness/perf line
added by this branch. The upstream `dispatch`/`combine` rows correspond to the branch's standard
probs-enabled benchmark rows.

### Torch API, BF16

| Config | Branch | dispatch | combine | dispatch+permute | combine+unpermute | fused dispatch+permute | fused combine+unpermute |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H=512, E=32, K=36 | current | 302.05 GB/s / 220.9 us | 210.22 GB/s / 317.4 us | 327.04 GB/s / 204.0 us | 176.51 GB/s / 378.1 us | 108.92 GB/s / 612.7 us | 59.67 GB/s / 1118.3 us |
| H=512, E=32, K=36 | upstream | 209.97 GB/s / 317.8 us | 219.41 GB/s / 304.1 us | 229.42 GB/s / 290.9 us | 183.17 GB/s / 364.3 us | 104.21 GB/s / 640.4 us | 60.08 GB/s / 1110.7 us |
| H=512, E=8, K=8 | current | 242.49 GB/s / 185.4 us | 190.71 GB/s / 235.8 us | 295.40 GB/s / 152.2 us | 178.30 GB/s / 252.2 us | 169.89 GB/s / 264.7 us | 132.29 GB/s / 339.9 us |
| H=512, E=8, K=8 | upstream | 244.10 GB/s / 184.2 us | 204.62 GB/s / 219.8 us | 296.73 GB/s / 151.5 us | 189.94 GB/s / 236.7 us | 185.91 GB/s / 241.9 us | 133.60 GB/s / 336.6 us |
| H=4096, E=8, K=8 | current | 532.62 GB/s / 685.2 us | 561.39 GB/s / 650.1 us | 550.29 GB/s / 663.2 us | 530.89 GB/s / 687.5 us | 597.76 GB/s / 610.6 us | 594.42 GB/s / 614.0 us |
| H=4096, E=8, K=8 | upstream | 524.95 GB/s / 695.2 us | 555.22 GB/s / 657.3 us | 541.83 GB/s / 673.6 us | 525.25 GB/s / 694.9 us | 588.05 GB/s / 620.6 us | 556.04 GB/s / 656.4 us |

### Torch API, FP8

| Config | Branch | dispatch | combine | dispatch+permute | combine+unpermute | fused dispatch+permute | fused combine+unpermute |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H=512, E=32, K=36 | current | 159.06 GB/s / 216.2 us | 210.49 GB/s / 316.8 us | 184.12 GB/s / 186.7 us | 176.94 GB/s / 376.9 us | 56.44 GB/s / 609.2 us | 60.17 GB/s / 1108.3 us |
| H=512, E=32, K=36 | upstream | 124.52 GB/s / 276.6 us | 220.12 GB/s / 303.4 us | 142.12 GB/s / 242.3 us | 183.29 GB/s / 364.4 us | 56.00 GB/s / 615.0 us | 60.20 GB/s / 1109.4 us |
| H=512, E=8, K=8 | current | 113.53 GB/s / 204.5 us | 189.93 GB/s / 237.1 us | 184.24 GB/s / 126.0 us | 177.73 GB/s / 253.4 us | 112.80 GB/s / 205.8 us | 133.52 GB/s / 337.3 us |
| H=512, E=8, K=8 | upstream | 118.43 GB/s / 196.5 us | 204.11 GB/s / 221.1 us | 178.23 GB/s / 130.6 us | 190.36 GB/s / 237.1 us | 111.69 GB/s / 208.4 us | 135.72 GB/s / 332.6 us |
| H=4096, E=8, K=8 | current | 451.44 GB/s / 417.5 us | 566.20 GB/s / 645.6 us | 504.91 GB/s / 373.3 us | 534.92 GB/s / 683.4 us | 489.68 GB/s / 384.9 us | 594.94 GB/s / 614.4 us |
| H=4096, E=8, K=8 | upstream | 392.36 GB/s / 481.1 us | 559.98 GB/s / 653.7 us | 493.63 GB/s / 382.4 us | 528.24 GB/s / 693.0 us | 484.87 GB/s / 389.3 us | 560.39 GB/s / 653.3 us |
