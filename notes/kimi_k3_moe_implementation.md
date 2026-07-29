# Kimi K3 Stable LatentMoE implementation

## Purpose and scope

This note specifies and records the test-oriented Megatron Core implementation of:

1. Transformer Engine RMSNorm immediately before the routed latent up-projection.
2. SiTU-GLU forward and backward, including dense FFNs, routed experts, and shared
   experts.
3. Auxiliary-loss-free routing with either the existing signed bias update or K3
   Quantile Balancing (QB).

The implementation is exercised through
`tests/functional_tests/test_cases/common/moe_perf/recipe_frontend.py`. It is opt-in
and lives in Megatron-LM (MLM), so it can be tested with an existing Transformer Engine
(TE) installation without rebuilding TE or the container image.

Shared-expert overlap is out of scope. The fused TE router owns the QB Top-(k+1)
selection and local histogram accumulation; MCore owns persistent bounds, distributed
histogram reduction, and step-end bias updates.

## Architecture and operation order

K3 router scoring and shared experts consume the full-width model representation. Only
the routed expert payload enters latent space:

```text
                              +--> router(x) --> routes and probabilities
full-width input x -----------+
                              +--> W_down(x) --> latent routed experts

latent routed aggregate u --> RMSNorm(u) --> W_up --> routed output

full-width input x --> full-width shared experts ------------------+
                                                                  +--> layer output
full-width routed output ------------------------------------------+
```

This order is already present in MCore:

1. `MoELayer.route(hidden_states)` scores the original full-width `hidden_states`.
2. `MoELayer.preprocess(...)` applies the latent down-projection after routing.
3. Dispatch, routed expert computation, combine, and routed aggregation occur in latent
   width.
4. The new native TE `LayerNormLinear` computes `W_up(RMSNorm(u))`.
5. The full-width shared-expert result is added after the routed up-projection.

The norm is therefore before the linear in forward, and the reverse order in backward:

```text
forward:  u -> RMSNorm -> W_up
backward: dY -> W_up dgrad -> RMSNorm backward
```

It is not `RMSNorm(W_up(u))`.

## Dependency baseline

The current `test_container_2606` runtime baseline is:

| Component | Version | Source commit |
| --- | --- | --- |
| Transformer Engine | `2.19.0.dev0+c0517995` | editable experimental build |
| DeepEP / HybridEP | `1.2.1+e944b0d` | installed in `test_container_2606` |
| CUTLASS / CuTe DSL | 4.4.2 | installed wheel |
| cuDNN Frontend | 1.26.0 | `35fd7b0d0e1d4952b904c79341c5e84e3af0a328` |
| cuda-python | 13.3.1 | installed in `test_container_2606` |

The first validation used CuTe DSL 4.5.0. Its source reference remains the official
CUTLASS `v4.5.0` tag at
`e406c186f510a15091cce01f782020ceb7ba8eb5`. The local cuDNN Frontend reference
checkout is v1.26.0. Upstream cuDNN Frontend 1.26.0 metadata pins 4.5.0, but the
current container deliberately carries the 1.26.0 + 4.4.2 combination. MLM therefore
feature-detects the API differences needed by that combination instead of assuming
the upstream wheel pin.

The source references used by the implementation are:

```text
TE/3rdparty/cutlass/python/CuTeDSL/cutlass/cute/
TE/3rdparty/cudnn-frontend/python/cudnn/grouped_gemm/
TE/transformer_engine/pytorch/ops/fused/grouped_mlp.py
```

## Activation knob design

### Existing MCore activation knobs

MCore exposes model-wide semantic activation choices near the network-size arguments:

- default GELU;
- `--swiglu`;
- `--quick-geglu`;
- `--squared-relu`; and
- the older `--openai-gelu`.

It separately exposes implementation and fusion controls, including:

- `use_te_activation_func`;
- `bias_activation_fusion` and `--no-bias-swiglu-fusion`;
- `--use-transformer-engine-op-fuser`;
- `--moe-mlp-glu-interleave-size`; and
- activation clamp/offset controls.

The semantic activation must not be encoded as an MoE-only backend flag because K3 uses
SiTU-GLU in dense FFNs, routed experts, and shared experts.

### New knobs

The semantic command-line flag is:

```text
--situ-glu
```

For recipe compatibility, the same parser action also accepts:

```text
--moe-use-situ-glu
```

Despite its name, the alias is not MoE-local. Both forms set:

```python
use_situ_glu: bool = True
```

The implementation controls are separate MCore configuration fields:

```python
situ_glu_impl: Literal["cutedsl"] = "cutedsl"
situ_glu_beta1: float = 4.0
situ_glu_beta2: float = 25.0
moe_latent_up_projection_rmsnorm: bool = False
```

The generated command-line forms for those fields are:

```text
--situ-glu-impl cutedsl
--situ-glu-beta1 4
--situ-glu-beta2 25
--moe-latent-up-projection-rmsnorm
```

`--situ-glu` configures the same FFN sizing rule as `--swiglu`, sets
`gated_linear_unit=True`, uses `F.silu` as the activation-family marker, selects the TE
activation submodule path, and disables the incompatible bias-activation fusion. Argument
validation also updates the raw `use_te_activation_func` and `bias_swiglu_fusion` values,
so argument logging and any pre-config consumers see the same backend selection as the
final `TransformerConfig`.

### Validation

The initial implementation deliberately has a narrow valid configuration:

- SiTU-GLU requires a GLU FFN and the TE activation module path.
- It is mutually exclusive with `--swiglu`, `--quick-geglu`, and `--squared-relu`.
- `situ_glu_impl` must be `cutedsl`.
- Both beta values must be positive and finite.
- SiTU-GLU currently requires MXFP8, even for dense and shared FFNs. This is an explicit
  product constraint for this test implementation, not a mathematical kernel limitation.
- Grouped routed experts additionally require the TE operation fuser and a 32-element GLU
  interleave matching TE's fused grouped-MLP layout.
- The grouped CuTe path requires
  `NVTE_CUTEDSL_FUSED_GROUPED_MLP=1`.
- Latent up-projection RMSNorm requires `moe_latent_size` and model
  `normalization="RMSNorm"`.

All features default to disabled, leaving existing activation and latent-MoE behavior
unchanged.

## 1. Native TE RMSNorm and latent up-projection

### Implementation

When `moe_latent_up_projection_rmsnorm=False`, `fc2_latent_proj` remains the existing
MCore `TELinear`.

When enabled, `MoELayer` directly instantiates:

```python
transformer_engine.pytorch.LayerNormLinear(
    in_features=moe_latent_size,
    out_features=hidden_size,
    normalization="RMSNorm",
    parallel_mode=None,
    tp_group=None,
    tp_size=1,
    return_bias=False,
    ...
)
```

There is no extra MCore forward wrapper. The only adaptation is parameter metadata needed
for the existing duplicated, non-tensor-parallel latent projection:

```text
allreduce=True
sequence_parallel=<MCore config>
tensor_model_parallel=False
```

The post-combine call accepts the native tensor return. It also retains tuple handling for
the old `TELinear` path.

The down-projection remains the existing full-width-to-latent `TELinear`. The
inference-optimized implementation rejects the new norm path until it gains an equivalent
native operation.

### Fusion semantics

TE `LayerNormLinear` is the correct reusable TE module for `RMSNorm -> Linear`. It can
produce the quantized normalized tensor directly for the following block-scaled GEMM and
avoid an extra high-precision materialization/quantization boundary. It is a module-level
fusion; it does not promise that RMSNorm and GEMM execute as one physical GPU kernel.

This choice minimizes Python overhead and avoids duplicating TE implementation logic.

## 2. CuTe DSL SiTU-GLU

### Definition

For the gate projection `g` and up projection `v`:

```text
A(g) = beta1 * tanh(g / beta1) * sigmoid(g)
B(v) = beta2 * tanh(v / beta2)

SiTU-GLU(g, v) = A(g) * B(v)
```

The K3 defaults are `beta1=4` and `beta2=25`.

The derivatives implemented in backward are:

```text
t_g = tanh(g / beta1)
t_v = tanh(v / beta2)
s_g = sigmoid(g)

dA/dg = (1 - t_g^2) * s_g
         + beta1 * t_g * s_g * (1 - s_g)
dB/dv = 1 - t_v^2

dG = dY * B(v) * dA/dg
dV = dY * A(g) * dB/dv
```

For routed experts, the cuDNN grouped dGLU specialization also applies router probability
`p` to `dG`/`dV` and produces:

```text
dP = sum_last_dim(dY * A(g) * B(v))
```

Nonlinear evaluation and accumulator math use FP32.

### Standalone activation

`megatron/core/fusions/cutedsl_situ_glu.py` owns:

- a PyTorch reference;
- CuTe DSL forward and backward kernels;
- an autograd function;
- `CuTeDSLSiTUGLU`; and
- the scaled shell used by TE's operation matcher.

The standalone kernels:

- compile lazily with `cute.compile`;
- use `cute.math.tanh(..., fastmath=True)`;
- use the current PyTorch CUDA stream;
- flatten arbitrary leading dimensions;
- predicate non-multiple element tails;
- compute in FP32 and explicitly cast BF16/FP16 stores;
- cache by dtype, device, and shape; and
- support `sm_90a`, `sm_100a`, and `sm_103a`.

The current standalone storage dtypes are BF16 and FP16. MXFP8 is applied at the
surrounding TE linear/quantization boundaries.

`TEActivationOp` returns `CuTeDSLSiTUGLU` whenever `use_situ_glu=True`. This is the common
activation builder used by dense FFNs, shared experts, and unfused expert fallbacks, so the
model-wide flag reaches all three paths without duplicating their MLP code.

### Routed grouped MLP

For MXFP8 grouped experts, TE's selected operation is:

```text
GroupedMLP_CuTeGEMMGLU
```

This is a joint TE operation covering:

```text
grouped FC1 -> GLU/probability/quantization -> grouped FC2
```

It is not one monolithic physical kernel containing both GEMMs. The first grouped GEMM
uses a fused GLU epilogue, while FC2 and backward weight/data-gradient work use their
corresponding grouped kernels.

MLM does not copy or replace TE's orchestration. `ScaledSiTUGLU` subclasses TE's
`ScaledSwiGLU`, allowing the existing TE matcher to select
`GroupedMLP_CuTeGEMMGLU`. Before its first compilation, MLM installs small subclasses of
the cuDNN Frontend 1.26.0 Python CuTe DSL kernel classes and overrides only:

```text
swiglu_act(...)  -> SiTU-GLU forward equations
dswiglu(...)     -> SiTU-GLU backward equations
```

The existing cuDNN Frontend code continues to own:

- grouped FC1 and FC2 GEMMs;
- MXFP8 scale handling and quantization;
- router-probability multiplication and `dprob`;
- dynamic scheduling;
- output layouts;
- dgrad/wgrad; and
- kernel caching.

The implementation passes `"swiglu"` through the current TE/cuDNN API because that API
does not expose a SiTU-GLU enum. Only the virtual activation math is specialized.

The installation is process-wide, idempotent for identical beta values, and rejects a
second model requesting different beta values. The performance frontend inspects the
fuser after warmup and fails unless `GroupedMLP_CuTeGEMMGLU` was actually selected.

### CuTe DSL compatibility shims

The CuTe DSL 4.5 binding exposes `nvvm.atomicrmw` with an explicit result-type argument,
while the cuDNN Frontend 1.26 grouped scheduler calls it without that argument. An
MLM-local feature-detected shim infers the result type from the atomic input only when
the installed signature requires it.

CuTe DSL 4.4.2 defines `OperandMajorMode` under
`cutlass.cute.nvgpu.tcgen05`, while cuDNN Frontend 1.26.0 imports it from
`cutlass.cute.nvgpu`. A second feature-detected shim re-exports the existing symbol
before importing the grouped-GEMM modules. It does not replace the class or edit an
installed package.

Two additional 4.4.2 adapters bridge 4.5-era calls made by cuDNN Frontend 1.26.0:

- `make_blockscaled_trivial_tiled_mma` accepts both the 4.4 single-A/B-dtype form
  used by wgrad and the 4.5 separate-A/B-dtype form used by the grouped GLU
  kernels; and
- a raw CuTe `_Pointer` exposes the identity `.ptr` accessor expected by the
  1.26.0 generated kernel.

TE may cache grouped-MLP support as false before the `OperandMajorMode` export is
installed. On that specific compatibility path, MLM clears the cached support
decision and registers TE's existing `fuse_ops` rule. It does not implement or copy
the TE grouped-MLP orchestration.

All adapters are installed only when the corresponding API discrepancy is present.
They avoid editing TE/cuDNN packages or rebuilding the image.

## 3. QB integration remains deferred in MLM

K3 QB requests Top-(k+1) on biased router scores. The first `k` experts are dispatched;
the additional score supports the quantile/bias update and must not be dispatched.

The MLM K3 frontend still omits both the extra result and QB bias updates. A separate
TE experiment now implements Top-(k+1) plus QB histogram accumulation without
dispatching the extra expert; see
`notes/kimi_k3_qb_router_histogram_results.md`. That TE API is not wired into MCore's
bias-update loop in this change.

The original integration concern remains:

- its APIs and output layouts are sized for exactly `topk`;
- route indices, probabilities, workspaces, and static/CUDA-graph buffers assume `k`;
- dispatch consumes every selected entry; and
- a retained but non-dispatched `(k+1)`th entry requires a new output contract.

The MLM performance frontend therefore continues to use current auxiliary-loss-free
bias balancing and measures only the latent RMSNorm and SiTU-GLU changes.

## Testing and performance frontend

The frontend reports whether latent RMSNorm and SiTU-GLU are enabled and which
implementation is requested. After the first warmup iteration it inspects TE's fuser and
prints:

```text
[moe_perf] verified_situ_glu_backend=GroupedMLP_CuTeGEMMGLU
```

If TE falls back to separate `GroupedLinear`, activation, and `GroupedLinear` operations,
the frontend raises an actionable error instead of silently benchmarking the wrong path.

Tests cover:

- both `--situ-glu` and `--moe-use-situ-glu` parser spellings;
- PyTorch reference formula;
- BF16 and FP16 standalone CuTe DSL forward/backward parity;
- direct native TE `LayerNormLinear` selection for latent up-projection; and
- the full frontend forward/backward smoke path.

The reproducible launcher is:

```text
tests/functional_tests/test_cases/common/moe_perf/launch_kimi_k3.sh
```

It runs the focused unit tests on GPU 0 and then launches the frontend with `torchrun`.
Its distributed configuration is fixed to TP=1, ETP=1, EP=8, eight GPUs, and eight
experts. HybridEP is the default dispatcher; `DISPATCHER_BACKEND=alltoall` selects the
plain all-to-all comparison. `RUN_UNIT_TESTS=0` skips the already-run unit suite when
repeating only the EP8 frontend.

The launcher enables `MCORE_DEBUG_DENSE_ROUTING=1` by default and writes the complete,
unfiltered eight-rank stdout/stderr stream to a timestamped file under
`logs/moe_perf/`. `KIMI_K3_LOG_FILE` selects an exact destination.

## Original CuTe DSL 4.5 remote validation result

Validation ran on:

| Property | Value |
| --- | --- |
| Nodes | `umb-b300-dp-186` (unit/1-GPU), `umb-b300-dp-217` (EP8) |
| GPU | NVIDIA B300 |
| Compute capability | 10.3 |
| Container | `test_container_2606` |
| CUDA compatibility driver | 610.43.02 over kernel driver 595.58.03 |
| Python | 3.12 |
| PyTorch | `2.13.0a0+8145d630e8.nv26.06` |
| CuTe DSL | 4.5.0 |
| cuDNN Frontend | 1.26.0 |
| TE | `2.19.0.dev0+f1d5f8d` |

Passed results:

```text
standalone SiTU-GLU: 3 passed
CLI aliases:         2 passed
latent TE module:    1 passed
frontend:            forward + backward completed
frontend backend:    GroupedMLP_CuTeGEMMGLU verified
EP8 frontend:        forward + backward completed
HybridEP dense path: TE router and HybridEP debug markers verified
```

The EP8 frontend used TP=1, ETP=1, EP=8, eight experts, one warmup, and one measured
iteration. All ranks selected the grouped CuTe DSL forward, dGLU, quantization, and wgrad
kernels. Kernel JIT made the first iteration 56.9 seconds. The warm measured iteration
reported 8.024 ms forward and 4.560 ms backward for the small smoke shape.

These timings validate distributed integration and warm-cache behavior; they are not a
representative K3 throughput result.

The HybridEP rerun additionally used the fused TE router with FP32 router probabilities.
The persisted debug log recorded:

```text
TE fused router requested dense topk_indices dtype=torch.int16 shape=(256, 1)
HybridEP metadata received dense topk_idx and skipped bool-map conversion
HybridEP dispatch used routing_map=None and consumed topk_idx directly
```

It also verified `GroupedMLP_CuTeGEMMGLU` and completed forward/backward. Its warm smoke
timings were 75.866 ms forward and 4.125 ms backward. The complete log is:

```text
MLM/logs/moe_perf/kimi_k3_hybridep_dense_routing_ep8.log
```

The EP8 node successfully enabled CUDA forward compatibility. All standalone and grouped
kernels compiled and ran successfully.

## CuTe DSL 4.4.2 revalidation

The current 1.26.0 + 4.4.2 baseline was revalidated on an NVIDIA B300 in
`test_container_2606`.

The focused suite passed:

```text
6 passed
```

It covers the PyTorch formula, activation selection and validation, BF16 and FP16
standalone CuTe DSL forward/backward JIT execution, installation of the actual
cuDNN grouped forward/backward SiTU-GLU subclasses, and TE grouped-MLP support
registration. Without the `OperandMajorMode` compatibility shim, importing the
cuDNN grouped kernels fails before compilation.

An actual one-GPU MoE frontend run then selected:

```text
GroupedMLP_CuTeGEMMGLU
```

It compiled and executed grouped FC1/SiTU-GLU/FC2 forward, dGLU, quantization, and
wgrad, then completed backward. The first JIT iteration took 22.2 seconds. The warm
smoke iteration reported 5.067 ms forward and 3.306 ms backward. This validates
kernel compatibility; it is not a representative K3 throughput measurement.

The complete focused-suite log is:

```text
/home/scratch.hhanyu_gpu/projects/moe/MLM/logs/moe_perf/test_situ_glu_cutedsl_442_with_grouped_hook.log
/home/scratch.hhanyu_gpu/projects/moe/MLM/logs/moe_perf/test_situ_glu_cutedsl_442_final.log
/home/scratch.hhanyu_gpu/projects/moe/MLM/logs/moe_perf/situ_glu_grouped_runtime_1260_442_full_compat.log
```

## Version compatibility assessment

### Confirmed

The original complete distributed forward/backward path is confirmed for:

```text
TE 2.19.0.dev0+f1d5f8d
CuTe DSL 4.5.0
cuDNN Frontend 1.26.0
SM103 / B300, including TP1/EP8
```

The current standalone and complete one-GPU grouped forward/backward path is
confirmed for:

```text
TE 2.19.0.dev0+c0517995
CuTe DSL 4.4.2
cuDNN Frontend 1.26.0
SM103 / B300, TP1/EP1 compatibility smoke
```

TE must provide the operation-fuser APIs and `GroupedMLP_CuTeGEMMGLU`. The current MCore
integration uses TE 2.14.0 as its minimum feature gate, but versions below the tested
2.19 development build have not been run with this patch.

### Source-compatible candidates, not runtime-confirmed

cuDNN Frontend tags 1.23.0 through 1.26.0 contain the same
`swiglu_act(...)` and `dswiglu(...)` parameter sets used by the feature checks.

- cuDNN Frontend 1.23.0 pins CuTe DSL 4.4.1.
- cuDNN Frontend 1.24.0, 1.24.1, 1.25.0, and 1.26.0 pin CuTe DSL 4.5.0.
- `cute.math.tanh` exists in CUTLASS/CuTe DSL 4.2.0 through 4.5.0.
- The TVM-FFI DLPack and fake-stream interfaces used by the standalone launcher exist
  from CuTe DSL 4.3.0 onward.

This makes cuDNN Frontend 1.23.0 with CuTe DSL 4.4.1 a plausible lower pair, and
1.24-1.26 with CuTe DSL 4.5.0 source-compatible pairs. Apart from the explicitly
bridged and tested 1.26.0 + 4.4.2 baseline, those other combinations remain
unverified because the grouped implementation subclasses private Python kernel
classes; matching signatures do not guarantee identical kernel layouts or compiler
behavior.

CuTe DSL 4.2.0 is not compatible with the standalone launcher as written because it lacks
the required TVM-FFI runtime interface. CuTe DSL 4.3.x has the needed public launcher
surface but is not paired with the inspected cuDNN grouped implementation and is not
claimed supported.

At runtime the implementation:

- checks the private cuDNN forward/backward method signatures;
- checks TE's operation-fuser feature level;
- relies on TE's own architecture and cuDNN version support tests;
- checks the selected fused op in the frontend; and
- feature-detects the `nvvm.atomicrmw` signature.

The support policy for this experimental path should therefore be:

```text
Current baseline: CuTe DSL 4.4.2 + cuDNN Frontend 1.26.0 + MLM adapters + tested TE build.
Historical full EP8 validation: CuTe DSL 4.5.0 + cuDNN Frontend 1.26.0.
Other versions: best-effort only; accept only after the same GPU test suite passes.
```

## 3. Quantile Balancing integration

### Knobs and compatibility gate

The auxiliary-loss-free router keeps its existing behavior by default:

```text
--moe-router-bias-update-method sign
```

K3 QB is selected with:

```text
--moe-router-enable-expert-bias
--moe-router-bias-update-method quantile
--moe-router-qb-num-bins 1000
--moe-router-score-function sigmoid
--moe-router-pre-softmax
--moe-router-fusion
```

MCore inspects the installed TE Python signature and enables QB only when
`fused_topk_with_score_function` exposes `qb_histogram`, `qb_bin_bounds`, and
`qb_histogram_mode`. It always requests `qb_histogram_mode="fused_atomic"`; the
two-kernel TE path remains available for isolated TE testing but is not threaded into
MCore.

The current MCore QB path rejects grouped routing, token dropping, non-sigmoid scoring,
post-Top-k normalization, unfused routing, and padding masks. Padding is rejected because
the current TE histogram API has no valid-token mask.

### Router state and step lifecycle

Each QB router follows the existing `expert_bias` ownership pattern and registers:

```text
expert_bias    float32 [num_experts]             persistent
qb_bin_bounds  float32 [2]                       persistent
qb_histogram   int32   [num_experts, num_bins]   nonpersistent
```

`qb_bin_bounds` starts at `[-1, 1]` and remains the same CUDA allocation so graph-facing
kernel arguments keep a stable pointer. The step-local histogram also remains in place
but is omitted from checkpoints.

For every training forward that records gradients, the TE fused router:

1. selects Top-(k+1) using raw sigmoid score plus the current expert bias;
2. emits only the actual Top-k routes and probabilities;
3. computes `r[i,j] = alpha[i] - sigmoid(logit[i,j])`; and
4. atomically accumulates one int32 bin for every token-expert pair.

Histogram counts accumulate across gradient-accumulation microbatches. During
`finalize_model_grads`, MCore stacks all local MoE-layer histograms, performs one int32
all-reduce over the existing TPxDPxCP group, recovers each expert's target quantile,
linearly interpolates inside its selected bin, mean-centers the resulting biases, and
updates `expert_bias` and the next bounds in place. The next range is exactly:

```text
[min(updated_expert_bias) - 1, max(updated_expert_bias) + 1]
```

The histogram is then zeroed by the normal temporary-tensor reset. No score matrix,
bin-index matrix, bias-handle registry, or extra broadcast is introduced.

### EP2 validation and performance

Validation used `umb-b300-dp-189`, `test_container_2606`, two B300 GPUs, TP1/EP2,
HybridEP dense routing, MXFP8 grouped SiTU-GLU, and the installed TE fused-atomic QB API.
The focused router suite passed 4 tests and the step-finalization test passed. Pointer
stability, repeated-forward histogram accumulation, persistent bounds, nonpersistent
histograms, exact quantile recovery, and reset behavior were checked.

Warm-cache one-layer results:

| Experts / Top-k | Method | Forward | Backward | Bias finalization | End to end |
| --- | --- | ---: | ---: | ---: | ---: |
| 8 / 1 | sign | 2.896 ms | 2.672 ms | 0.426 ms | 5.995 ms |
| 8 / 1 | quantile | 2.771 ms | 2.252 ms | 0.758 ms | 5.780 ms |
| 896 / 16 | sign | 15.573 ms | 13.225 ms | 0.589 ms | 29.387 ms |
| 896 / 16 | quantile | 15.333 ms | 13.160 ms | 0.761 ms | 29.254 ms |

The small timing differences in forward/backward are noise-scale for this smoke harness.
The measurable QB cost is the histogram all-reduce and quantile update: about 0.17 ms
over signed updating at the K3 router shape in this EP2 run. An
`896 x 1000 x int32` histogram occupies about 3.42 MiB per MoE layer.

`MCORE_DEBUG_DENSE_ROUTING=1` confirmed that the fused TE router returned dense int16
Top-k indices and HybridEP consumed them without reconstructing indices from a sparse or
boolean routing map.

Full-iteration CUDA graph capture currently fails inside
`deep_ep/backend/hybrid_ep_backend.cuh:6000` with
`cudaErrorStreamCaptureInvalidated`. The identical signed-bias control fails at the same
point, so this is a pre-existing HybridEP/full-iteration graph limitation rather than a
QB buffer or pointer issue. Eager EP2 is the validated path.

Persistent logs:

```text
MLM/logs/qb_mcore/test_router_quantile_final.log
MLM/logs/qb_mcore/test_finalize_qb.log
MLM/logs/qb_mcore/k3_qb_ep2_eager_warm.log
MLM/logs/qb_mcore/k3_sign_ep2_eager.log
MLM/logs/qb_mcore/k3_qb_e896_top16_ep2.log
MLM/logs/qb_mcore/k3_sign_e896_top16_ep2.log
MLM/logs/qb_mcore/k3_qb_ep2_graph.log
MLM/logs/qb_mcore/k3_sign_ep2_graph.log
```

## Follow-up work intentionally excluded

- Padding-mask support in the TE QB histogram API.
- HybridEP full-iteration CUDA graph capture support.
- Shared-expert overlap.
- A public TE or cuDNN Frontend SiTU-GLU operation.
- Mixed SiTU-GLU and SwiGLU models in one Python process.
- A production performance sweep across expert counts, token distributions, EP sizes,
  B200, and B300.
- CUDA graph capture validation for first-time JIT and cache-reuse paths.
