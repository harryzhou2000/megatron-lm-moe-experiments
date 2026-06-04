# Dense top-k indices for fused router dispatch

## Motivation

The fused router currently materializes `routing_map` as a boolean tensor with shape
`[num_tokens, num_experts]`. This is expensive when `num_experts` is large, while the Flex
DeepEP-family dispatchers consume a dense top-k representation after router metadata setup.

This change adds an optional dense top-k output path to Transformer Engine and lets MLM select it
for Flex dispatcher backends that can consume it. The dense tensor has shape
`[num_tokens, topk]` and stores selected expert ids. The dense tensor replaces the boolean routing
map in the router output, but the probability tensor remains the existing sparse dense tensor with
shape `[num_tokens, num_experts]`.

## Backend dtype requirements

Current DeepEP and HybridEP use different index widths:

- DeepEP accepts top-k expert indices as `torch.int64`. Its Python and C++ wrappers use
  `data_ptr<int64_t>()`.
- HybridEP dense routing accepts `torch.int16`. The HybridEP dispatcher converts `topk_idx` to
  `torch.int16`, allgathers it as bytes, and the backend scans `int16_t` indices with `-1` as the
  invalid sentinel.

TE therefore supports `int16`, `int32`, and `int64` dense index outputs. The PyTorch API infers the
desired output dtype from the optional `topk_indices` tensor supplied by the caller. Passing `None`
keeps the previous boolean routing map behavior.

## TE API shape

The existing API is unchanged:

```python
probs, routing_map = fused_topk_with_score_function(..., topk_indices=None)
```

When the caller provides a preallocated output:

```python
topk_indices = torch.empty((num_tokens, topk), device=logits.device, dtype=torch.int64)
probs, routing_output = fused_topk_with_score_function(..., topk_indices=topk_indices)
```

`routing_output` is the same tensor as `topk_indices`. The C++ binding validates CUDA placement,
contiguity, shape `[num_tokens, topk]`, and dtype `int16`, `int32`, or `int64`.

The CUDA forward path templates the top-k kernels on `IndexType`. The boolean map write is
guarded, and the dense output write is optional. Both the legacy small-topk simple kernel and the
persistent/radix kernel can write either a boolean route map or dense top-k indices.

Backward has a matching dense-index API. Instead of loading a boolean `[T, E]` mask, it scans the
token's `K` selected indices to decide whether an expert was routed. This preserves gradients while
avoiding the boolean routing map allocation.

## MLM selection

MLM chooses dense output automatically in `TopKRouter` only when all conditions are true:

- router fusion is enabled, so TE can produce the dense output directly;
- token dispatcher type is `flex`;
- router token dropping is disabled, because capacity-based dropping mutates routing maps;
- expert bias is disabled, because expert-bias accounting currently expects a boolean map.

Backend-specific dtype selection:

- `moe_flex_dispatcher_backend == "deepep"`: allocate `topk_indices` as `torch.int64`;
- `moe_flex_dispatcher_backend == "hybridep"`: allocate `torch.int16` only if HybridEP dense
  routing is available and `tp_size * num_moe_experts <= int16_max`.

Otherwise MLM falls back to the existing boolean routing map path.

## Flex dispatcher handling

`MoEFlexTokenDispatcher._initialize_metadata` now accepts either representation:

- bool `[T, E]`: existing expansion to `[T, world_size, num_local_experts]`;
- dense `[T, K]`: expand expert ids across TP to `[T, K * tp_size]`.

The dense expansion maps an original expert id `ep_idx * num_local_experts + local_idx` to each
TP replica:

```text
((ep_idx * tp_size + tp_idx) * num_local_experts + local_idx)
```

The probability tensor remains expanded with the existing dense `[T, E]` path.

DeepEP metadata now skips `torch.topk(probs)` when it receives dense indices and gathers matching
top-k probabilities with those indices. HybridEP metadata uses the provided dense indices directly
and passes `num_of_experts` separately to the dense HybridEP dispatcher, because there is no boolean
map shape to inspect in that path.

## Aux loss interaction

Aux loss still computes its own score and routing tensors through
`compute_routing_scores_for_aux_loss`. This is intentional: aux loss score semantics are not the
same as dispatch probabilities in all router modes, so it cannot blindly reuse the dispatch scores.
The dense top-k dispatch optimization does not change aux-loss tensors.

Aux loss can potentially reuse dispatch top-k indices only in cases where the selected experts are
provably identical to the aux-loss selection. Even then, aux loss still needs its own score tensor,
so the reusable part is only selected expert ids or token counts, not the full aux-loss output.

## Limitations and follow-ups

- TE still returns sparse dense probabilities `[T, E]`. A future optimization could add optional
  dense `[T, K]` probabilities for dispatchers that do not need `[T, E]`.
- The dense-index backward scans `K` indices per expert. This trades compute for avoiding the
  `[T, E]` boolean map. It is most attractive when `E` is large and `K` is moderate.
- HybridEP dense routing is limited to `int16` expert ids today. MLM falls back automatically when
  the TP-expanded expert space does not fit.
- Token dropping and expert-bias accounting still use the boolean route map path.
