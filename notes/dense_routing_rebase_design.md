# Dense routing design and rebase validation

## Scope

This note covers the dense top-k routing design across:

- Transformer Engine branch `hhanyu/router_dense_route`, rebased on latest `hhanyu/router_fix_p3R`.
- Upstream p3R commit `2f17b7fee25be6ed5f58a6adcb29ed165e02331a`.
- Megatron-LM plumbing since `a9098e478885de81a3d8156286a213750748988c` on MLM branch `hhanyu/dev_p3`.

## Rebase validation

Upstream commit `2f17b7fee25be6ed5f58a6adcb29ed165e02331a` changed router top-k dispatch policy:

- Add early `score_function` validation in fused router launchers.
- Use radix top-k only when `topk >= get_radix_topk_threshold()` and
  `num_experts <= kMaxExpertsRadixTopk`.
- Fall back to naive top-k when expert count exceeds the radix histogram limit, instead of failing.
- Apply the same fallback policy to `fused_score_for_moe_aux_loss.cu`.

The sparse fused-router path already contained those upstream changes after the rebase. The dense
`forward_with_indices` launcher initially still used the old policy:

```cpp
if (topk < get_radix_topk_threshold()) {
  // naive
} else {
  NVTE_CHECK(num_experts <= kMaxExpertsRadixTopk, ...);
  // radix
}
```

That would interfere with dense routing for large expert counts: sparse routing would fall back to
naive top-k, but dense routing would still fail when `num_experts > kMaxExpertsRadixTopk`.

The dense launcher has been aligned with upstream policy:

```cpp
NVTE_CHECK(score_function >= 0 && score_function <= 2, ...);
const bool use_radix = topk >= get_radix_topk_threshold()
                       && num_experts <= kMaxExpertsRadixTopk;
if (!use_radix) {
  // naive
} else {
  // radix
}
```

With this parity fix, the upstream p3R change does not otherwise conflict with dense routing.
Backward dense routing is independent of radix/naive top-k selection because it consumes the already
materialized `[num_tokens, topk]` selected expert ids.

## TE dense-route API

The legacy API remains unchanged when no dense output buffer is supplied:

```python
probs, routing_map = fused_topk_with_score_function(..., topk_indices=None)
```

When the caller supplies `topk_indices`, TE writes selected expert ids into that tensor and returns it
as the second output:

```python
topk_indices = torch.empty((num_tokens, topk), device="cuda", dtype=torch.int16)
probs, routing_output = fused_topk_with_score_function(..., topk_indices=topk_indices)
```

The PyTorch extension validates that `topk_indices` is CUDA, contiguous, shaped
`[num_tokens, topk]`, and has dtype `torch.int16`, `torch.int32`, or `torch.int64`.

The C API adds matching dense-index entry points:

- `nvte_fused_topk_with_score_function_forward_with_indices`
- `nvte_fused_topk_with_score_function_backward_with_indices`

The Python autograd wrapper saves the returned routing representation and records whether it is dense
so backward dispatches to the matching C++ path.

## TE forward design

The existing fused top-k kernels are templated on `IndexType` and accept two optional routing outputs:

- `bool *routing_map` for the legacy sparse bool representation `[T, E]`.
- `IndexType *topk_indices_output` for dense selected indices `[T, K]`.

The sparse path passes a bool routing map and `nullptr` dense output. The dense path passes
`nullptr` routing map and the caller-provided index buffer.

Both forward kernel families support dense output:

- Simple naive kernel for small top-k or expert counts unsupported by radix.
- Persistent/radix kernel when `topk >= get_radix_topk_threshold()` and
  `num_experts <= kMaxExpertsRadixTopk`.

Dense forward still writes the same sparse probability tensor `[T, E]` and intermediate tensor
`[T, E]`. Only the routing representation changes from bool `[T, E]` to integer `[T, K]`.

For `torch.int16` dense output, TE checks `num_experts <= INT16_MAX` before launch.

## TE backward design

The legacy sparse bool backward remains separate and unchanged. It consumes:

- `routing_map [T, E]`
- `intermediate_output [T, E]`
- `grad_probs [T, E]`

The dense backward consumes:

- `topk_indices [T, K]`
- `intermediate_output [T, E]`
- `grad_probs [T, E]`

The optimized dense backward kernel is:

```cpp
fused_topk_backward_selected_indices_kernel<DataType, IndexType, ScoreFunc>
```

It avoids scanning every expert for membership in the selected set. Instead, each token warp loops
over the `K` selected indices directly, computes normalization reductions from selected experts, zero
fills `grad_logits [T, E]` as needed, and writes gradients for selected experts.

Score-function behavior:

- `sigmoid`: selected experts only have nonzero gradients after top-k masking; zero-fill then write
  selected gradients.
- `sqrtsoftplus`: same sparse selected-gradient behavior as sigmoid, with sqrt-softplus derivative.
- `softmax` post-top-k: non-selected logits receive zero gradient; zero-fill then write selected
  softmax gradients.
- `softmax` pre-top-k: softmax coupling makes all experts receive gradients; dense backward writes
  the full row using the selected-gradient reduction term, then overwrites selected entries with their
  direct gradient contribution.

This keeps one dense selected-indices kernel template shared across score functions and index dtypes.

## TE test coverage

Dense output coverage is integrated into existing PyTorch fused-router Cartesian tests:

- `test_topk_sigmoid`
- `test_topk_sqrtsoftplus`
- `test_topk_softmax`

Each now parameterizes `topk_index_dtype` over:

- `None`: legacy bool routing map path.
- `torch.int16`: dense route path.
- `torch.int32`: dense route path.
- `torch.int64`: dense route path.

The tests convert returned dense indices back to a bool routing map for comparison against the
reference implementation and assert that the returned tensor aliases the caller-provided buffer.

## MLM plumbing since `a9098e478885de81a3d8156286a213750748988c`

MLM uses dense routing opportunistically, not as a required feature. The plumbing is guarded so MLM
still works with TE builds that do not expose the `topk_indices` argument.

### Capability detection

`megatron/core/extensions/transformer_engine.py` inspects the TE Python signature:

```python
fused_topk_with_score_function_supports_topk_indices = (
    "topk_indices" in inspect.signature(fused_topk_with_score_function).parameters
)
```

`megatron/core/transformer/moe/moe_utils.py` only forwards `topk_indices` into TE when that flag is
true. Otherwise the argument is omitted and the legacy bool routing map path is used.

### Router dtype selection

`TopKRouter._dense_route_indices_dtype()` enables dense routing only when all conditions hold:

- `moe_router_fusion=True`, so TE can produce dense indices directly.
- `moe_token_dispatcher_type == "flex"`.
- `moe_expert_capacity_factor is None`, because token dropping currently mutates sparse routing maps.
- Expert bias is disabled, because expert-bias token accounting expects a bool routing map.
- Installed TE supports `topk_indices`.

Backend dtype policy:

- `moe_flex_dispatcher_backend == "deepep"`: request `torch.int64` indices.
- `moe_flex_dispatcher_backend == "hybridep"`: request `torch.int16` only when HybridEP exposes
  dense routing and `tp_group_size * num_moe_experts <= torch.iinfo(torch.int16).max`.

If any condition fails, MLM falls back to the bool routing map path. `MCORE_DEBUG_DENSE_ROUTING=1`
prints one-shot explanations for disabled dense routing and one-shot confirmation when dense routing
is active.

### Router call path

In `TopKRouter.routing()`:

1. MLM asks `_dense_route_indices_dtype()` for an optional dtype.
2. If a dtype is returned, MLM preallocates `topk_indices [T, K]` on the logits device.
3. `topk_routing_with_score_function()` forwards the buffer to TE when supported.
4. TE returns `probs [T, E]` and `routing_map`, where `routing_map` is actually the dense
   `topk_indices` tensor in dense mode.

The rest of the MoE stack intentionally keeps the variable name `routing_map` for interface
compatibility, but consumers branch on dtype/shape where dense routing is supported.

### HybridEP dispatcher path

`_HybridEPManager.setup_metadata()` accepts both representations:

- Bool `[T, E]`: preserve legacy `routing_map` and derive `topk_idx` from `torch.topk(probs)` when
  HybridEP dense routing is available.
- Integer `[T, K]`: set `routing_map=None`, store `topk_idx`, and skip the bool map-to-indices
  conversion.

`_HybridEPManager.dispatch()` then calls `hybrid_ep_dispatch()` with:

- `routing_map=None`
- `topk_idx=[T, K]`
- `num_of_experts=self.num_experts`

`megatron/core/transformer/moe/fused_a2a.py` checks whether installed `HybridEPBuffer` supports the
`dense_routing` parameter. If yes and `topk_idx` is provided, it calls:

```python
HybridEPBuffer.dispatch_with_permute(
    hidden=x,
    topk_idx=topk_idx,
    probs=probs,
    num_of_experts=num_of_experts,
    num_of_experts_per_rank=num_local_experts,
    dense_routing=True,
    ...,
)
```

Otherwise it falls back to the legacy bool `routing_map` call.

### DeepEP dispatcher path

The router requests `torch.int64` dense indices for `moe_flex_dispatcher_backend == "deepep"`.
DeepEP's regular dispatch API consumes dense top-k indices, so this avoids reconstructing indices
from the sparse bool routing map in the flex path. The probability tensor remains `[T, E]` and is
used to gather/dispatch weights according to those indices.

## Remaining limitations

- TE dense routing still returns probabilities as sparse dense `[T, E]`; only the route map becomes
  compact `[T, K]`.
- Dense routing is disabled when router token dropping is enabled.
- Dense routing is disabled when expert bias accounting is enabled.
- HybridEP dense routing currently requires int16 expert ids, so MLM falls back when the TP-expanded
  expert id range exceeds int16.
- Pre-softmax dense backward remains inherently full-row because softmax couples all experts.

## Validation status

Static validation performed during rebase review:

- Confirmed `2f17b7fee25be6ed5f58a6adcb29ed165e02331a` is the base commit under the dense-route
  commits on `hhanyu/router_dense_route`.
- Compared upstream sparse fused-router fallback logic against dense `forward_with_indices`.
- Patched dense `forward_with_indices` to match upstream score-function validation and radix/naive
  fallback policy.
- Ran `git diff --check` in `TE` with no whitespace errors.

Recommended GPU validation after rebuilding TE:

- Run fused-router PyTorch tests covering dense dtypes in `test_topk_sigmoid`,
  `test_topk_sqrtsoftplus`, and `test_topk_softmax`.
- Include one large-expert dense forward case with `num_experts > kMaxExpertsRadixTopk` to verify the
  dense path now falls back to naive top-k instead of failing.
