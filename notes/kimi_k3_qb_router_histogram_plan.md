# Kimi K3 QB Router Histogram Implementation Plan

## Objective

Implement and compare two Quantile Balancing (QB) histogram accumulation paths in
Transformer Engine (TE), starting from main-branch commit
`f1d5f8d5222b5789351dd6dc145253a7ac727b78`:

1. **Two-kernel path**: the fused router emits the per-token Top-(k+1) cutoff
   `alpha`; a second kernel reads `alpha` and the router's existing raw FP32 sigmoid
   scores, computes histogram bins, and accumulates a caller-owned histogram.
2. **One-kernel path**: the statically dispatched QB router computes the same bin
   indices after Top-(k+1) and directly atomically accumulates the caller-owned
   histogram as a router epilogue.

Both paths must produce identical Top-k routes, probabilities, cutoffs, and integer
histogram counts. The benchmark will determine the cost of avoiding the second
kernel and score reread versus issuing more global atomics on the router's critical
path.

The implementation branch is:

```text
hhanyu/qb-fused-router-histogram
```

An isolated worktree for this branch was created from the exact base commit so the
existing dirty `TE/` checkout does not need to be switched or cleaned.

## QB Semantics

This implementation targets the Kimi K3 router:

- Score function: sigmoid only.
- Router bias: FP32 expert bias, used for selection but not mixture weights.
- Grouped Top-k: unsupported in the QB path.
- `0 < topk < num_experts`.
- K3 primary configuration: 896 routed experts, Top-16, 1,000 histogram bins.

For token `i`, the router computes raw scores and biased selection scores:

```text
s[i, j]       = sigmoid(logits[i, j])
biased[i, j]  = s[i, j] + expert_bias[j]
```

The QB route selects Top-(k+1) from `biased[i, :]`:

- The best `k` experts are the actual routes.
- The remaining value is `alpha[i]`, the biased cutoff.
- Mixture weights are computed from the raw, unbiased scores of only the actual
  Top-k experts, exactly as in the existing router.

Each token-expert pair contributes:

```text
r[i, j] = alpha[i] - s[i, j]
```

to one uniform bin. For bounds `[lower, upper]` and `B` bins:

```text
bin = clamp(floor((r - lower) * B / (upper - lower)), 0, B - 1)
```

The caller computes one fixed range for the whole training step:

```text
lower = min(expert_bias) - 1
upper = max(expert_bias) + 1
```

and passes it as a CUDA FP32 tensor to avoid a host synchronization. The histogram
is a contiguous CUDA int32 tensor with shape `[num_experts, num_bins]`. The caller
zeros it once per training step and reuses it across all gradient-accumulation
microbatches.

TE performs only rank-local histogram accumulation. The data-parallel all-reduce,
quantile interpolation, mean-centering, optional EMA, and next-step bias update
remain outside this TE branch.

## Static Router Dispatch

Use a compile-time QB mode so the existing router specialization remains unchanged:

```cpp
enum class QBHistogramMode {
  Disabled,
  TwoKernel,
  FusedAtomic,
};
```

The CUDA kernel and launcher should statically dispatch `Disabled` versus a QB
specialization. The QB specialization may share Top-(k+1) and cutoff extraction
between `TwoKernel` and `FusedAtomic`; only `FusedAtomic` executes the histogram
epilogue.

The dispatch must cover:

- The simple/naive Top-k kernel.
- The optimized persistent/radix Top-k kernel.
- BYTEMAP routing output.
- BITMAP_U8 routing output.
- Dense Top-k indices with int16, int32, and int64 output types.

No QB checks or additional writes should remain in the `Disabled` specialization.

### Radix Top-(k+1)

K3's Top-16 uses the radix path at the current default threshold, so cutoff handling
must not assume the radix output array is sorted.

For the QB specialization:

1. Allocate shared scratch for `k+1` scores and indices.
2. Select the Top-(k+1) set.
3. Find its minimum biased score; this is `alpha`.
4. If several selected entries equal `alpha`, discard the selected entry with the
   largest expert index. This preserves the router's ascending-index tie rule.
5. Compact the remaining `k` entries and use only those entries for route output,
   bias removal, normalization, and probabilities.

The non-QB specialization must continue to call the existing Top-k code with the
original scratch size and ordering.

## Option 1: Two-Kernel Histogram

The QB router writes an internal FP32 `alpha[num_tokens]` tensor. It already writes
the raw, unbiased FP32 sigmoid scores to `intermediate_output[num_tokens,
num_experts]` for backward, so no score or bin-index matrix is added.

The histogram kernel receives:

```text
raw_scores  float32 [num_tokens, num_experts]
alpha       float32 [num_tokens]
bin_bounds  float32 [2]
histogram   int32   [num_experts, num_bins]  # accumulated in place
```

Initial kernel structure:

- 256 threads per CTA.
- Eight contiguous experts per CTA tile.
- Dynamic shared-memory counts `[8, num_bins]`; 32 KiB at 1,000 bins.
- Token partitions:
  `min(4, max(1, ceil(num_tokens / 1024)))` CTAs per expert tile.
- Load contiguous expert segments from the raw-score matrix.
- Fuse `r`, uniform-bin calculation, and shared-memory atomic accumulation in the
  load path.
- Flush only nonzero shared counts with global int32 atomics into the persistent
  histogram.

This is a one-pass hierarchical histogram. It does not materialize per-CTA partial
histograms or perform a second tree-reduction pass.

## Option 2: One-Kernel Fused-Atomic Histogram

After Top-(k+1) and cutoff extraction, each token warp revisits its shared-memory
biased scores:

```text
raw_score = biased_score - expert_bias[j]
r         = alpha - raw_score
bin       = uniform_bin(r)
atomicAdd(&histogram[j, bin], 1)
```

This path:

- Does not write `alpha` to global memory.
- Does not reread the FP32 score matrix.
- Does not launch a histogram kernel.
- Issues one global integer atomic per token-expert pair.

A complete CTA-private `[num_experts, num_bins]` histogram is not feasible. Tiling
an `[8, 1000]` shared histogram inside the existing four-token router CTA would
require 112 zero/accumulate/flush phases for 896 experts and only 32 samples per
phase, so it is intentionally not used. Cross-warp deduplication is also omitted
initially because four tokens are unlikely to hit the same one of 1,000 bins for
the same expert often enough to repay the synchronization cost.

## PyTorch and Common APIs

Keep existing C APIs and Python behavior backward compatible. Add dedicated common
entry points for:

- QB forward with routing-map output and FP32 `alpha`.
- QB forward with dense Top-k indices and FP32 `alpha`.
- Two-kernel histogram accumulation.
- Fused-atomic QB forward, which accepts the histogram and bin bounds directly.

Add optional trailing Python arguments:

```python
qb_histogram: Optional[torch.Tensor] = None
qb_bin_bounds: Optional[torch.Tensor] = None
qb_histogram_mode: Optional[str] = None
```

Accepted modes are `"two_kernel"` and `"fused_atomic"`. All three arguments must be
absent for the existing path, or all required QB arguments must be present.

The public return remains:

```python
probs, routing_output
```

The QB autograd wrapper keeps `alpha` and raw scores internal. Histogram counts and
cutoffs are non-differentiable; backward remains the existing Top-k router
backward over the actual `k` routes.

Validation requirements:

- CUDA, contiguous, same device for all QB tensors.
- `qb_histogram.dtype == torch.int32`.
- `qb_histogram.shape == [num_experts, num_bins]`.
- `qb_bin_bounds.dtype == torch.float32` and shape `[2]`.
- Sigmoid score function, FP32 expert bias, no grouped Top-k.
- `topk < num_experts`, `num_bins > 0`, and `upper > lower`.
- The int32 counter contract requires fewer than `INT_MAX` local accumulated
  tokens per histogram reset.

## Correctness Tests

Add a reusable, pure-PyTorch QB reference helper alongside the fused-router tests.
It must use ordinary FP32 PyTorch operations and no TE fused-router calls. Its
interface should accept logits, Top-k, expert bias, bin bounds, number of bins, and
an optional existing histogram, and return:

```text
probs
routing map or dense Top-k indices
raw sigmoid scores
alpha
bin indices
updated histogram
```

The helper computes:

1. FP32 sigmoid raw scores.
2. Biased `torch.topk(..., k=topk+1)`.
3. Actual Top-k routes and unbiased normalized probabilities.
4. `alpha`, `r`, bin indices, and an exact int32 histogram.

Use this helper as the single numerical oracle for both CUDA implementations. Keep
it intentionally straightforward, even if it materializes `[tokens, experts]`
values, because it is test-only. Support accumulation into an existing histogram
so the same helper verifies gradient-accumulation behavior across microbatches.

Cover:

- `num_experts=896`, `topk=16`, `num_bins=1000`.
- Both naive and radix selection thresholds.
- FP32 and BF16 logits where supported by the existing router.
- Random inputs and constructed Top-(k+1) ties.
- Values exactly on the lower bound, internal bin edges, and upper bound.
- Non-multiple token and expert dimensions.
- BYTEMAP, BITMAP_U8, and dense int16/int32/int64 routes.
- Two microbatches accumulated into one histogram.
- Exact equality between two-kernel and fused-atomic histograms.
- Forward probabilities and backward gradients against the existing/reference
  router.
- Rejection of every unsupported mode and malformed tensor combination.
- QB-disabled output and backward regression tests.

## Benchmark

Add a standalone benchmark under the parent repository's `scripts/` directory
that checks correctness first and reports four paths:

1. Pure PyTorch unfused QB reference.
2. Existing fused router, QB disabled.
3. QB two-kernel.
4. QB fused-atomic.

Primary parameters:

```text
num_experts = 896
topk        = 16
num_bins    = 1000
tokens      = 256, 1024, 4096, 8192, 16384
dtype       = float32, bfloat16 where supported
output      = dense int16 and BYTEMAP
```

Use CUDA events, synchronize only outside the timed region, perform at least 100
warmup iterations and 500 timed iterations, and report median and mean latency,
the absolute QB overhead, the ratio to the unfused PyTorch implementation, and
the ratio to the QB-disabled router. Histogram buffers intentionally accumulate
across timed calls without clearing between calls, matching gradient accumulation.

Save complete, unfiltered build, test, and benchmark output under a persistent
remote log directory and copy the final benchmark summary into `notes/`.

## Remote Build and Test Workflow

Use `test_container_2606` on an active B200/B300 allocation, preferring the
available B300 node. Sync the isolated TE worktree contents to the remote
`~/projects/moe/TE/` checkout without copying `.git`.

The 26.06 container uses system Python. Inside the container:

```bash
cd /home/scratch.hhanyu_gpu/projects/moe/TE

export PATH="/home/hhanyu/.pixi.x86_64/bin:${PATH}"
export NVTE_BUILD_THREADS_PER_JOB=4
export NVTE_CUDA_ARCHS="100;103a;"
export NVTE_USE_CCACHE=1

/usr/bin/python3 -m pip install --no-build-isolation -e '.[test]' --verbose
```

Do not filter online output. Redirect complete output to timestamped log files,
then inspect those files after the command finishes.

Run focused fused-router tests first, followed by the full relevant PyTorch router
test file and both benchmark variants. Record:

- Compute node and GPU model.
- Container name.
- CUDA, PyTorch, TE, compiler, and ccache versions.
- Exact build command.
- Test results.
- Per-shape benchmark results and the selected recommendation.

## Deliverables

- TE branch `hhanyu/qb-fused-router-histogram`, based directly on `f1d5f8d`.
- Both histogram implementations behind explicit static/runtime dispatch.
- Focused correctness and performance tests.
- Remote build/test/benchmark logs.
- A benchmark-results note with a recommendation for the MLM integration.
- Updated project-local `computelab-run` skill if its container/build instructions
  need correction.
- Signed TE commit with an imperative `[PyTorch]`/`[Common]` title.
