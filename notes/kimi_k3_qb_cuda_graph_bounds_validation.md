# Kimi K3 QB mutable bounds and full-iteration CUDA graphs

## Status

This note records a CUDA-graph regression in the fused Quantile Balancing (QB)
router, identifies the change that introduced it, and documents the focused
Transformer Engine fix and validation.

The implementation is Python-only in Transformer Engine, with a companion
feature-detected MCore call after the trusted QB update. It does not change a
CUDA kernel, C++ binding, fused-router launch, or ABI.

## Observed failure

The failure was reproduced by the K3-like MoE performance recipe:

```text
configs/benchmarking/recipes/nemotron3_ultra_150b_proxy/gb300/
  mxfp8_64GPU_TP1PP1EP64_moe_perf_k3_moe_mock.yaml
```

Run information:

- Remote job/session: `20260826-032410-449a`
- Remote experiment checkout: `~/projects/agentic-mcore-dev-2`
- Megatron-LM commit: `1898ad6e674031f372c70421865dab8d54e1e095`
- Full-iteration CUDA-graph warmups: 8
- Fused QB histogram mode: single-kernel atomic accumulation

All ranks completed the configured eager warmups and entered full-iteration
capture. They then failed in the first fused QB router invocation inside capture:

```text
RuntimeError: QB bin_bounds must be validated by an eager router call before CUDA graph capture
```

The relevant call chain was:

```text
recipe_frontend.py::_call_full_iteration_wrapper
  -> FullCudaGraphWrapper.__call__
  -> forward_backward_func
  -> TopKRouter.routing
  -> fused_topk_with_score_function
  -> FusedTopkScoreFunctionQB.forward
  -> _validate_qb_bin_bounds
```

This fails before HybridEP dispatch, expert-capacity handling, paged stash, or
the expert MLP. Those features are not causal.

## Why the older TE implementation worked

The older TE implementation at
[`c850084f403764277eddb0edf2189617365849cf`](https://github.com/harryzhou2000/TransformerEngine/commit/c850084f403764277eddb0edf2189617365849cf)
passed `qb_bin_bounds` directly to the CUDA implementation. It validated tensor
metadata such as device, dtype, contiguity, and shape, but it did not copy the
two bound values to the host or associate validation with the tensor's PyTorch
version.

The fused kernels load `lower` and `upper` through the device pointer when they
execute. MCore updates the persistent bounds tensor in place:

```python
bounds.copy_(next_bounds)
```

Therefore the pointer captured by the CUDA graph remains stable while each
replay observes the latest values. This is the intended mutable-input pattern
for the QB state.

The old behavior was CUDA-graph-compatible, but it did not produce a clean,
recoverable error for invalid values such as non-finite bounds or
`lower >= upper`.

## Regression source

TE commit
[`e363e628bdc0a37ab4b34ec39fbdbcee7a6185b2`](https://github.com/NVIDIA/TransformerEngine/commit/e363e628bdc0a37ab4b34ec39fbdbcee7a6185b2),
`[Common][PyTorch] Make QB bounds validation recoverable`, added the current
host validation and version cache. This work landed as part of
[Transformer Engine PR #3395](https://github.com/NVIDIA/TransformerEngine/pull/3395).

The essential validation logic is:

```python
version = bin_bounds._version
if getattr(bin_bounds, _QB_BOUNDS_VALIDATED_VERSION_ATTR, None) == version:
    return True

if torch.cuda.is_current_stream_capturing():
    raise RuntimeError(
        "QB bin_bounds must be validated by an eager router call before CUDA graph capture"
    )

lower, upper = bin_bounds.detach().cpu().tolist()
```

The corresponding C++ binding selects an unchecked common entry point only
when Python reports that the values have already been validated. This avoids a
device-to-host copy and stream synchronization inside CUDA graph capture.

The validation itself is sound for an immutable tensor. The missing case is a
persistent, graph-address-stable tensor whose contents are intentionally
updated between executions.

## Exact failure sequence

For each eager warmup iteration:

1. The fused router validates `qb_bin_bounds` at tensor version `N`.
2. TE caches version `N` on the tensor object.
3. QB finalization computes the next bias and histogram bounds.
4. MCore executes `bounds.copy_(next_bounds)`.
5. The in-place copy preserves `data_ptr()` but increments the PyTorch tensor
   version to `N + 1`.

After the eighth warmup, capture begins before another eager router call. TE
finds that the cached version is stale and tries to validate the new values.
The validation requires `detach().cpu().tolist()`, which is not legal inside
capture, so TE raises the observed exception.

In compact form:

```text
Old TE:
  eager router -> update device values -> capture stable pointer -> replay succeeds

Current TE:
  eager validates version N -> update creates version N+1
  -> capture requests host validation -> RuntimeError
```

Increasing the CUDA-graph warmup count cannot fix the issue: every warmup is
followed by another QB bounds update. Capturing with zero warmups also fails
because the tensor has never passed eager value validation.

## Graph-safe validation design

An exact PyTorch tensor-version handshake is insufficient for a full-iteration
graph. `finalize_model_grads_func` is part of MCore's captured pipeline
schedule, and QB finalization updates the persistent bounds near the end of
every global-batch iteration. Python dispatch increments the tensor version
once while the graph is captured, but subsequent `cudaGraphLaunch` calls
execute the device copy without re-entering Python or incrementing the version.

A focused B200 probe observed exactly this behavior:

```text
before          version=0 value=1
after capture   version=1 value=1
after replay 1  version=1 value=2
after replay 2  version=1 value=3
```

The graph executable is not mutated. The state behind its stable device
pointers is intentionally mutated once per replayed training iteration.

Within one global-batch iteration, the bounds remain unchanged across all
router invocations. QB finalization performs the single in-place update only
after the iteration's histograms have been accumulated, and the next graph
replay begins by routing with those new bounds. Thus the replay sequence is:

```text
route with bounds N -> accumulate histogram N -> finalize/write bounds N+1
-> next replay routes with bounds N+1
```

This mutates persistent training state, not the captured graph topology,
kernel arguments, or device pointers.

The final design uses an explicit trusted-producer handshake:

1. Eager calls retain recoverable host validation and the exact tensor-version
   cache.
2. `mark_qb_bin_bounds_validated()` records the current version without
   inspecting values or synchronizing the stream.
3. The marker may run immediately after a trusted in-place update is
   dispatched during capture on the same stream. The PyTorch `copy_` dispatch
   has already advanced the version even though the captured device copy has
   not executed yet.
4. The router near the start of capture accepts only the exact version marked
   by the last eager warmup. A never-validated tensor or an unmarked stale
   version remains an error.
5. MCore's captured finalization dispatches the next mathematically bounded
   update and marks its new version. Later replays do not re-enter Python and
   trust that captured producer while reading new values through the stable
   pointer.

This adds no graph kernel, global status write, host synchronization, or change
to the fused-router launch sequence.

## Correctness rationale

The design preserves the useful protections added by the recoverable
validation change:

- Invalid initial bounds still fail eagerly with a recoverable exception.
- A changed tensor used by another eager router call is revalidated at its new
  PyTorch version.
- A never-validated tensor is rejected during capture.
- A tensor changed after its last validation is rejected during capture unless
  the trusted producer explicitly marks that exact version.
- Tensor device, dtype, shape, and contiguity remain checked by the binding on
  every invocation.

The MCore updater constructs `[bias_min - 1, bias_max + 1]`, so finite valid
input bounds inductively produce ordered next bounds. A recoverable device-side
check for an untrusted graph producer would require a caller-owned status
tensor, a validation kernel or producer epilogue that records failure in that
tensor, and a later host observation point. If invalid bounds must be prevented
from reaching the next replay, the router would also need a device-side safe
fallback or conditional path. An assertion or trap would poison the CUDA
context, so this focused trusted-producer change does not use one.

## Required TE regression coverage

Extend the existing QB CUDA-graph tests to cover the production transition,
for both `two_kernel` and `fused_atomic`:

1. Validate bounds eagerly and run a trusted update that marks the new version.
2. Capture and replay the fused router successfully.
3. Verify probabilities, routing output, histogram totals, and histogram bins
   against an eager/PyTorch reference using the updated range.
4. Update the bounds in place between replays and verify that replay
   observes the new values through the stable pointer.
5. Capture an iteration in which the router runs before an in-graph bounds
   update, matching the production pipeline schedule.
6. Verify that the marker may execute after the in-graph update and that replay
   continues to read the updated pointer.

Retain or add the complementary negative fences:

- Invalid initial bounds must fail recoverably.
- Invalid bounds introduced by an in-place update must still fail on the next
  eager call.
- Never-validated and unmarked stale versions must fail capture.
- Cover both routing-map output and caller-provided dense Top-k indices where
  supported.

## Implemented TE change

The TE implementation keeps exact-version capture gating and exposes the
trusted-producer marker from `transformer_engine/pytorch/router.py`. Unlike the
initial revision, the marker is pure Python metadata and is allowed during
capture. Its docstring requires same-stream ordering immediately after the
trusted device update.

Regression coverage exercises `two_kernel` and `fused_atomic`, sparse routing
maps and caller-provided dense Top-k indices, updated bounds between replays,
the marker after an update captured later in the iteration, recoverable eager
errors, and never-validated and stale-version capture rejection.

An A/B run against the unchanged upstream implementation reproduced the
failure at capture. The same focused selection on the fixed source passed:

```text
22 passed, 3675 deselected
```

The complete QB selection passed:

```text
47 passed, 2 skipped, 3648 deselected
```

The two skips are the pre-existing multi-GPU device tests in the one-GPU
allocation. Black left both changed files unchanged, and Pylint rated the
production module 10.00/10.

## MCore integration validation

After the focused TE tests pass, validate with a private `gcp-nrt` allocation:

1. Run a small fused-QB MoE forward/backward with at least one warmup,
   QB finalization, full-iteration capture, and multiple replays.
2. Assert that `qb_bin_bounds.data_ptr()` remains stable across finalizations.
3. Assert that bounds and expert bias change across QB updates.
4. Assert the histogram count invariant before finalization and zeroed
   histogram state afterward.
5. Exercise the fused-atomic path used by the K3 recipe.
6. Rerun the K3-like full-iteration CUDA-graph performance frontend before
   returning to a large distributed performance run.

The focused MCore integration ran on one B200 with the fused-atomic TE API. A
real MCore `TopKRouter` completed three eager full iterations, captured a graph
containing both routing and the later QB finalization, and replayed it four
times. The bounds pointer remained stable and the bounds stayed finite and
ordered while changing on every replay. The Python tensor version advanced
from 1 through 3 in eager execution, advanced once to 4 during capture, and
remained 4 across all four replays even though device contents continued to
change. This directly confirms why host version bookkeeping cannot validate
mutable replay state.

The companion MCore change feature-detects the TE marker and calls it
immediately after each CUDA-resident QB bounds update. The marker executes in
eager warmups and once while the full iteration is captured; graph replays need
no Python callback.

The K3-like MoE performance frontend also completed three eager QB updates and
entered full-iteration capture without the former TE bounds-validation error.
Its one-rank run then encountered dispatcher-specific capture limitations:

- AllToAll synchronizes a D2H event recorded in the capture stream.
- AllGather copies token counts to pageable CPU memory during capture.
- Flex/HybridEP requires `TP x EP > 1`, which the private one-GPU allocation
  cannot provide.

These failures occur after the fixed TE router accepts the mutable bounds and
are outside this TE-only change. The router-only MCore graph test isolates and
passes the exact QB state transition without altering unrelated dispatcher
code.

## GCP-NRT build observations

The exact TE branch was built in a private `/workspace/venv-te-qb-cg` using the
`dev_2604` container on one B200. The build used:

```text
NVTE_FRAMEWORK=pytorch
NVTE_CUDA_ARCHS=100
NVTE_BUILD_THREADS_PER_JOB=4
MAX_JOBS=4
NVTE_USE_CCACHE=1
NVTE_WITH_NCCL_EP=0
```

The cold editable build completed in 2,125 seconds (35m25s). It made 543
cacheable compiler calls, with zero hits because the persistent cache had no
matching entries. Although `NVTE_CUDA_ARCHS=100` was requested, compiler output
also showed `sm_100a`, `sm_103a`, and `sm_90a` specializations generated by the
build.

The longest translation units recorded in `build/cmake/.ninja_log` were:

| Translation unit | Time |
|---|---:|
| `gelu_grouped.cu` | 883.934 s |
| `swiglu_grouped.cu` | 501.608 s |
| `relu_grouped.cu` | 496.940 s |
| `fused_topk_with_score_function.cu` | 373.474 s |
| `gelu_grouped_dbias.cu` | 318.308 s |
| `gelu.cu` | 293.479 s |
| `swiglu.cu` | 290.616 s |
| `relu.cu` | 238.473 s |
| `gelu_dbias.cu` | 178.414 s |
| `scaled_activation.cu` | 170.529 s |

CPU samples showed 224 logical CPUs, no swap or I/O wait, and approximately
94--96% aggregate idle time. Individual compiler workers saturated their cores,
but the build was dominated by a few long CUDA translation units rather than
host-wide CPU capacity.

## Source references

- Failed MCore revision:
  [`1898ad6e674031f372c70421865dab8d54e1e095`](https://github.com/harryzhou2000/Megatron-LM/commit/1898ad6e674031f372c70421865dab8d54e1e095)
- Original graph-compatible QB TE revision:
  [`c850084f403764277eddb0edf2189617365849cf`](https://github.com/harryzhou2000/TransformerEngine/commit/c850084f403764277eddb0edf2189617365849cf)
- Recoverable bounds-validation change:
  [`e363e628bdc0a37ab4b34ec39fbdbcee7a6185b2`](https://github.com/NVIDIA/TransformerEngine/commit/e363e628bdc0a37ab4b34ec39fbdbcee7a6185b2)
- Graph-safe trusted-update fix:
  [Transformer Engine PR #3426](https://github.com/NVIDIA/TransformerEngine/pull/3426)
- Companion MCore trusted-update marker:
  [`78e971efa`](https://github.com/harryzhou2000/Megatron-LM/commit/78e971efa)
- Merged QB implementation and its existing static-bounds graph test:
  [Transformer Engine PR #3395](https://github.com/NVIDIA/TransformerEngine/pull/3395)
- Kimi K3 architecture and QB algorithm:
  [Kimi K3 technical report](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf)
