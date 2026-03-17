# Analysis: `fused_topk_with_score_function_forward_kernel`

## 1. Source file locations

| Layer | File |
|---|---|
| CUDA kernel | `TE/transformer_engine/common/fused_router/fused_topk_with_score_function.cu` |
| Kernel helpers | `TE/transformer_engine/common/fused_router/utils.h` |
| C API header | `TE/transformer_engine/common/include/transformer_engine/fused_router.h` |
| Python binding | `TE/transformer_engine/pytorch/router.py` |
| Existing tests | `TE/tests/pytorch/test_fused_router.py` |

## 2. What the kernel does

Fuses **topk selection + score function (softmax or sigmoid)** into a single CUDA kernel for MoE routing. A single kernel call replaces what would otherwise be 3–5 separate PyTorch ops (softmax/sigmoid → topk → scatter → optional normalize).

### Forward pipeline (per token, handled by one warp)

```
logits [num_tokens, num_experts]
    │
    ├─ if use_pre_softmax && softmax:  softmax(logits) → scores → topk(scores)
    ├─ if !use_pre_softmax && softmax: topk(logits) → softmax(top_k_logits)
    ├─ if sigmoid:                     sigmoid(logits) + optional expert_bias → topk → revert bias → normalize
    │
    └─ outputs:
        probs              [num_tokens, num_experts]  — sparse, zero except topk positions
        routing_map        [num_tokens, num_experts]  — bool mask of selected experts
        intermediate_output[num_tokens, num_experts]  — softmax/sigmoid values saved for backward
```

### Key parameters

| Parameter | Meaning |
|---|---|
| `score_function` | 0 = sigmoid, 1 = softmax |
| `use_pre_softmax` | softmax-before-topk (True) vs topk-then-softmax (False); only used with softmax |
| `num_groups` / `group_topk` | Grouped topk: pick top `group_topk` groups first, then topk within selected groups |
| `scaling_factor` | Multiply final probs by this scalar |
| `expert_bias` | Added to sigmoid scores before topk (for biased routing), reverted after topk |

## 3. Kernel launch configuration

```
kThreadsPerWarp  = 32
kThreadsPerBlock = 128  (4 warps per CTA)
num_token_per_block = kThreadsPerBlock / kThreadsPerWarp = 4
grid_size = ceil(num_tokens / num_token_per_block)
```

- **Each warp processes exactly 1 token** — intra-warp shuffles for reductions.
- **All per-token work (scores, topk, softmax/sigmoid) in shared memory.**
- Shared memory per block:
  - `scores`: `num_experts * 4 * sizeof(DataType)`
  - `topk_scores`: `topk * 4 * sizeof(DataType)`
  - `topk_indices`: `topk * 4 * sizeof(int)`
  - If `group_topk > 0`: `+group_scores + masked_scores` buffers
- Dynamic shared memory (`extern __shared__`), set via `cudaFuncSetAttribute`.

## 4. Supported dtypes

From `TE_ROUTER_PROBS_TYPE_SWITCH_ALL` in `utils.h`:
- `float32` (float)
- `float16` (fp16 / half)
- `bfloat16` (bf16 / nv_bfloat16)

The existing tests **only test float32**. fp16/bf16 coverage is missing.

## 5. Existing test analysis (`test_fused_router.py`)

### Reference implementation

`topk_softmax_sigmoid_pytorch()` (line 50) — pure PyTorch implementing the same logic as the CUDA kernel: softmax/sigmoid → topk → post-processing. Used as the ground truth for `torch.testing.assert_close`.

### Correctness tests

| Test | Score function | Parameters swept |
|---|---|---|
| `test_topk_sigmoid` | sigmoid | tokens={2048,7168,8992}, experts={128,32}, topk={4,8}, group_topk={None,4}, scaling={None,1.2}, bias={T,F} |
| `test_topk_softmax` | softmax | tokens={2048,7168,14234}, experts={128,32}, topk={4,8}, pre_softmax={T,F}, group_topk={None,4}, scaling={None,1.2} |

Both check: forward probs match, routing_map match, backward gradients match.

### Batch dimension

The Python binding (`router.py` line 32) does `logits.view(-1, tensor_shape[-1])` — so a
`[batch, seq, num_experts]` input is flattened to `[batch*seq, num_experts]` before the
kernel sees it. The kernel always operates on a 2D `[num_tokens, num_experts]` tensor.

All existing tests pass 2D inputs directly, so they effectively test **batch=1** only.
The kernel itself doesn't care about batch — it's just the total token count that matters.
But the view/reshape path for 3D+ inputs is never exercised in the tests.

### Gaps in the existing tests

1. **Dtypes**: Only `float32`. No `float16` or `bfloat16`.
2. **Input data**: Deterministic arange-based inputs (monotonic, no ties, no extreme values).
   - Never tests random data, all-same values, large magnitudes, or near-zero logits.
3. **Edge cases**: No tests for `topk=1`, `num_experts=1`, `num_tokens=1`, `topk=num_experts`.
4. **Expert counts > 32**: Tests do use 128 experts, but `naive_topk_and_mask` is O(topk × num_experts) per warp — perf degrades with large expert counts.
5. **Performance**: No benchmarks at all. The `profile_topk_softmax` function at the bottom is just a thin wrapper calling correctness tests.
6. **Numerical stability**: No tests with logits that would cause softmax overflow/underflow (very large/small magnitudes).

## 6. Python API for calling the kernel

```python
from transformer_engine.pytorch.router import fused_topk_with_score_function

probs, routing_map = fused_topk_with_score_function(
    logits=logits,           # [num_tokens, num_experts], float32/fp16/bf16, CUDA, requires_grad ok
    topk=4,                  # int, number of experts per token
    use_pre_softmax=False,   # bool, only relevant for softmax
    num_groups=8,            # int, set 0 or None to disable grouped topk
    group_topk=4,            # int, set 0 or None to disable grouped topk
    scaling_factor=1.0,      # float
    score_function="softmax",# "softmax" or "sigmoid"
    expert_bias=None,        # [num_experts], optional, only used with sigmoid
)
# probs:       [num_tokens, num_experts] — sparse probabilities
# routing_map: [num_tokens, num_experts] — bool mask
```

To run on the remote machine:
```bash
python3 -m pytest -xvs TE/tests/pytorch/test_fused_router.py
```

## 7. Plan: Correctness testing with controlled input

### Test categories

#### A. Input distribution control
- **Uniform random**: `torch.randn(num_tokens, num_experts)`
- **Extreme magnitudes**: logits in [-1000, 1000] (tests softmax numerical stability)
- **Near-zero**: logits ~ 1e-7 (sigmoid near 0.5, softmax near uniform)
- **Identical values**: all logits equal (tie-breaking behavior)
- **One-hot-like**: one expert has logit=100, rest=0 (degenerate topk)
- **Adversarial**: duplicate max values, negative infinities in some positions

#### B. Edge cases
- `num_tokens=1` (single token)
- `num_experts=topk` (all experts selected)
- `topk=1` (single expert per token)
- `num_experts` not a multiple of 32 (warp stride edge case)
- `num_experts` very large (e.g. 256, 512, 1024) — stress shared memory

#### C. Dtype coverage
- `torch.float32`, `torch.float16`, `torch.bfloat16`
- Check that fp16/bf16 match a float32 reference within appropriate tolerances

#### D. Gradient correctness
- `torch.autograd.gradcheck` with float64-promoted reference (or double-sided finite differences)
- Per-element gradient comparison: fused vs PyTorch reference

## 8. Plan: Performance testing

### Benchmarking approach

```python
import torch
from torch.cuda import Event

def benchmark_kernel(fn, *args, warmup=20, iters=100):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    start = Event(enable_timing=True)
    end = Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms
```

### Key dimensions to sweep

| Dimension | Values |
|---|---|
| `num_tokens` | 128, 512, 2048, 8192, 32768, 131072 |
| `num_experts` | 8, 32, 64, 128, 256, 512 |
| `topk` | 1, 2, 4, 8 |
| `score_function` | softmax, sigmoid |
| `group_topk` | None, 4 |
| `dtype` | float32, bfloat16 |

### Metrics to collect
- **Latency** (ms per call)
- **Throughput** (tokens/sec)
- **Compare vs PyTorch reference** (speedup factor)
- **Shared memory usage** (will hit limits with large num_experts)

### Shared memory limit concern

`shared_memory_size = (num_experts + topk) * 4 * sizeof(float) + topk * 4 * sizeof(int)`

For num_experts=512, topk=8, no grouping:
`(512 + 8) * 4 * 4 + 8 * 4 * 4 = 8320 + 128 = 8448 bytes` — fine.

For num_experts=1024 with grouping: double the scores buffer → ~32KB — still within 48KB default, but approaching limit. For very large expert counts, need `cudaFuncSetAttribute` to extend (kernel already does this).

## 9. Seeing which GPU device pytest runs on

pytest itself doesn't print GPU info. Options:
- **In-script**: `torch.cuda.get_device_properties(torch.cuda.current_device())` at the top of
  the test file / conftest.
- **Before pytest**: `python3 -c "import torch; print(torch.cuda.get_device_name())"`
- **nvidia-smi**: `nvidia-smi -L` lists all GPUs.
- **Select device**: `CUDA_VISIBLE_DEVICES=3 python3 -m pytest ...` pins to GPU 3.

## 10. Custom test/benchmark script

Written at `scripts/test_fused_topk.py`. Covers correctness (forward only, vs PyTorch reference)
and performance (CUDA-event-timed, fused vs reference).

```bash
# Full sweep (correctness + benchmark), prints GPU info first
python scripts/test_fused_topk.py

# Correctness only, single config
python scripts/test_fused_topk.py --mode correctness \
    --num-tokens 4096 --num-experts 64 --topk 4 --score-function softmax

# Benchmark only
python scripts/test_fused_topk.py --mode benchmark

# Different dtype
python scripts/test_fused_topk.py --dtype bf16

# Random input instead of deterministic arange
python scripts/test_fused_topk.py --mode correctness --input-type random

# Pin GPU
CUDA_VISIBLE_DEVICES=2 python scripts/test_fused_topk.py
```
