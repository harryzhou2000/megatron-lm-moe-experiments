# Hybrid-EP Dense Routing PR Test Results

## Setup

- Branch under test: `hhanyu/hybrid-ep-dense-routing-scan-opt`
- Base branch: `hybrid-ep` / `upstream/hybrid-ep` at `e0a5b1d`
- Compute node: `umb-b300-023`
- Container: `test_container_2603`
- Build command:

```bash
PYTORCH_NVCC="ccache nvcc" \
NVCC_APPEND_FLAGS="--threads 8" \
TORCH_CUDA_ARCH_LIST="10.0;10.3+PTX" \
pip install --no-build-isolation . -v
```

## Test Script Behavior

- The updated test script exercises dense `topk_idx` scan input in two places:
  - `dispatch(..., dense_routing=True)`, which tests dense scan metadata without permute metadata.
  - `dispatch_with_permute(..., dense_routing=True)`, which tests dense scan with `enable_permute=True`, producing `dense_chunk_layout` and `dense_to_expert_map`.
- The script is backward-compatible with old `hybrid-ep`: it detects whether `dense_routing` is accepted by `dispatch` and `dispatch_with_permute` and prints `SKIP (unsupported)` when absent.
- Current branch correctness includes dense routing PASS lines. Upstream `hybrid-ep` correctness includes dense routing SKIP lines.

## Configs Tested

| Name | Hidden | Tokens/rank | Local experts | TopK | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| default | 7168 | 4096 | 8 | 8 | Test script defaults |
| sparser-H512 | 512 | 8192 | 8 | 8 | Tuned env |
| sparser-H4096 | 4096 | 8192 | 8 | 8 | Tuned env |

Tuned env for sparser configs:

```bash
NUM_OF_THREADS_PER_BLOCK_PREPROCESSING_API=512
NUM_SMS_DISPATCH=32
NUM_SMS_COMBINE=32
NUM_OF_STAGES_G2S_COMBINE_API=64
NUM_OF_STAGES_S2G_COMBINE_API=8
NUM_TOKENS_COMBINE_REDUCE_BATCH_COMBINE_API=16
NUM_OF_TOKENS_PER_GROUP_COMBINE_API=2
```

## Correctness Summary

| Branch | Config | BF16 | FP8 | Dense routing checks |
| --- | --- | --- | --- | --- |
| current | default | PASS | PASS | PASS |
| current | sparser-H512 | PASS | PASS | PASS |
| current | sparser-H4096 | PASS | PASS | PASS |
| upstream | default | PASS | PASS | SKIP unsupported |
| upstream | sparser-H512 | PASS | PASS | SKIP unsupported |
| upstream | sparser-H4096 | PASS | PASS | SKIP unsupported |

## Torch API Bandwidth, BF16

Values are `GB/s / us`.

| Config | Branch | dispatch | combine | dispatch+permute | combine+unpermute | fused dispatch+permute | fused combine+unpermute |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| default | current | 496.37 / 640.5 | 518.00 / 613.7 | 516.15 / 615.9 | 492.73 / 645.2 | 615.36 / 516.6 | 567.57 / 560.1 |
| default | upstream | 491.58 / 646.7 | 516.06 / 616.0 | 510.82 / 622.4 | 491.15 / 647.3 | 610.66 / 520.6 | 563.62 / 564.1 |
| sparser-H512 | current | 242.79 / 185.2 | 194.89 / 230.7 | 296.60 / 151.6 | 181.63 / 247.6 | 168.86 / 266.3 | 132.45 / 339.5 |
| sparser-H512 | upstream | 243.37 / 184.8 | 205.02 / 219.3 | 297.43 / 151.2 | 190.18 / 236.4 | 184.40 / 243.9 | 134.23 / 335.0 |
| sparser-H4096 | current | 544.77 / 669.9 | 564.12 / 647.0 | 561.12 / 650.4 | 533.93 / 683.6 | 607.23 / 601.0 | 559.77 / 652.0 |
| sparser-H4096 | upstream | 534.45 / 682.9 | 554.54 / 658.1 | 551.26 / 662.1 | 524.59 / 695.7 | 600.35 / 607.9 | 553.51 / 659.4 |

## Torch API Bandwidth, FP8

Values are `GB/s / us`.

| Config | Branch | dispatch | combine | dispatch+permute | combine+unpermute | fused dispatch+permute | fused combine+unpermute |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| default | current | 426.30 / 378.1 | 508.32 / 615.0 | 481.54 / 334.8 | 484.91 / 644.7 | 514.63 / 313.2 | 554.07 / 564.3 |
| default | upstream | 421.55 / 382.4 | 506.28 / 617.5 | 473.82 / 340.2 | 482.92 / 647.4 | 507.61 / 317.6 | 551.39 / 567.0 |
| sparser-H512 | current | 115.98 / 200.2 | 194.21 / 231.9 | 181.38 / 128.0 | 181.19 / 248.5 | 112.27 / 206.8 | 133.10 / 338.3 |
| sparser-H512 | upstream | 116.24 / 199.8 | 203.43 / 221.4 | 177.13 / 131.1 | 189.53 / 237.6 | 106.34 / 218.4 | 133.97 / 336.1 |
| sparser-H4096 | current | 463.15 / 407.0 | 568.77 / 642.7 | 517.58 / 364.2 | 536.19 / 681.8 | 491.59 / 383.4 | 563.93 / 648.2 |
| sparser-H4096 | upstream | 450.35 / 418.5 | 559.56 / 653.3 | 502.31 / 375.2 | 528.82 / 691.3 | 482.63 / 390.6 | 557.47 / 655.7 |

## Kernel-Only Highlights

- Current default BF16 fused dispatch+permute kernel: `639.95 GB/s, 496.8 us`.
- Upstream default BF16 fused dispatch+permute kernel: `632.73 GB/s, 502.5 us`.
- Current sparser-H512 FP8 dispatch kernel, probs=False: `254.28 GB/s, 91.3 us`.
- Upstream sparser-H512 FP8 dispatch kernel, probs=False: `254.19 GB/s, 91.3 us`.
- Current sparser-H4096 FP8 dispatch kernel, probs=True: `745.58 GB/s, 252.8 us`.
- Upstream sparser-H4096 FP8 dispatch kernel, probs=True: `715.64 GB/s, 263.4 us`.

## Logs

- Current default: tee log was not saved because `bench_logs/` did not exist yet; stdout was captured in the session.
- Current sparser-H512: `/home/scratch.hhanyu_gpu/projects/moe/bench_logs/dense_routing_pr_current_h512_e8_k8.log`
- Current sparser-H4096: `/home/scratch.hhanyu_gpu/projects/moe/bench_logs/dense_routing_pr_current_h4096_e8_k8.log`
- Upstream default: `/home/scratch.hhanyu_gpu/projects/moe/bench_logs/dense_routing_pr_upstream_default.log`
- Upstream sparser-H512: `/home/scratch.hhanyu_gpu/projects/moe/bench_logs/dense_routing_pr_upstream_h512_e8_k8.log`
- Upstream sparser-H4096: `/home/scratch.hhanyu_gpu/projects/moe/bench_logs/dense_routing_pr_upstream_h4096_e8_k8.log`

## Notes

- Current and upstream performance are similar for default and H512 sparse configs.
- Current branch is modestly faster on the H4096 sparse config, especially FP8 dispatch and dispatch+permute paths.
- The main functional delta is dense `topk_idx` scan support and dense scan optimization, not a broad sparse-route performance regression/improvement.
