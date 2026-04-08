# Removing `record_stream()` from combined_1f1b for CUDA Graph Compatibility

## Config

Qwen3-Next-80B-A3B_E72, 24 layers, EP=8, GBS=64, MBS=2, TP=1, PP=1, DP=8,
num_microbatches=4, hybridep dispatcher, mxfp8 precision, shared experts enabled
(`--moe-shared-expert-intermediate-size 512`, `--moe-shared-expert-gate`).

## Problem

`combined_1f1b` with full-iteration CUDA graph capture causes ~190 GB GPU
memory vs ~110 GB in eager mode (both with combined_1f1b). The 80 GB inflation
is caused by `record_stream()` calls that prevent freed memory from being
reused during CUDA graph capture.

### How `record_stream()` causes inflation under CUDA graph capture

PyTorch's caching allocator tracks cross-stream tensor usage via
`record_stream()`, which adds entries to a tensor's `stream_uses` set
(`c10/cuda/CUDACachingAllocator.cpp`). When a tensor is freed during CUDA
graph capture:

- **`stream_uses` empty** (single-stream): `free()` calls `free_block()`
  immediately -- the block returns to the graph's private `BlockPool` and can
  be reused by subsequent allocations within the same capture.
- **`stream_uses` non-empty** (cross-stream): the block is placed in
  `deferred_blocks` and is **NOT reusable** for the remainder of the capture.
  This is because the allocator cannot call `cudaEventQuery()` during capture
  to determine whether a stream has actually completed.

In contrast, during eager execution, `process_events()` periodically reclaims
deferred blocks once their stream events complete. During CUDA graph capture,
`process_events()` is skipped entirely, so deferred blocks are permanently
stuck.

The `combined_1f1b` schedule uses two CUDA streams (compute and comm) and the
original code had `record_stream()` calls at three sites. Every such call on
a tensor that is later freed creates a permanently deferred block. With
4 microbatches x 24 layers x multiple tensors per layer, hundreds of blocks
accumulate as deferred, inflating the private pool by ~80 GB.

### Why `record_stream()` can be conditionally skipped

The `combined_1f1b` schedule uses a per-microbatch CUDA event shared across
all nodes within a microbatch. Every node's forward and backward runs inside
`stream_acquire_context()` (`pipeline_parallel/utils.py`):

```python
@contextmanager
def stream_acquire_context(self, name=None):
    self.event.wait(self.stream)      # wait for previous node's stream
    with torch.cuda.stream(self.stream):
        yield
    self.event.record(self.stream)    # signal completion on this stream
```

Each `TransformerModelChunkSchedulePlan` creates ONE `torch.cuda.Event()` that
is shared by all layers and all nodes (attn, dispatch, mlp, combine) within
that microbatch. Two microbatches in an overlapped phase use two different
events (Ev_f for the forward microbatch, Ev_b for the backward microbatch).

`TransformerLayerSchedulePlan.run()` interleaves operations from both
microbatches, alternating between compute and comm streams:

```
CPU order | Operation              | Stream  | Event ops
----------|------------------------|---------|-----------------------------------
1         | B.combine(mb0, Li)     | comm    | Ev_b.wait(comm), Ev_b.record(comm)
2         | F.attn(mb1, Lj)        | compute | Ev_f.wait(compute), Ev_f.record(compute)
3         | B.mlp(mb0, Li)         | compute | Ev_b.wait(compute), Ev_b.record(compute)
4         | F.dispatch(mb1, Lj)    | comm    | Ev_f.wait(comm), Ev_f.record(comm)
5         | B.mlp_dw(mb0, Li)      | compute | (no event ops)
6         | B.dispatch(mb0, Li)    | comm    | Ev_b.wait(comm), Ev_b.record(comm)
7         | F.mlp(mb1, Lj)         | compute | Ev_f.wait(compute), Ev_f.record(compute)
8         | F.combine(mb1, Lj)     | comm    | Ev_f.wait(comm), Ev_f.record(comm)
9         | B.attn(mb0, Li)        | compute | Ev_b.wait(compute), Ev_b.record(compute)
10        | B.attn_dw(mb0, Li)     | compute | (no event ops)
```

The two events create a **zigzag synchronization chain**: each operation on
one stream waits for the latest event, which was recorded on the other stream
by the previous operation. This creates a total order -- every operation in the
CPU-side schedule happens-after all previous operations, regardless of which
stream they're on.

Additionally, operations on the **same stream** are ordered by CUDA's stream
FIFO guarantee.

Together, these two properties guarantee that **GPU operations** on freed
memory are correctly ordered. However, this event-based ordering alone is NOT
sufficient in eager mode, because it does not inform the **CPU-side CUDA
caching allocator** about cross-stream usage.

### Why `record_stream()` is still needed in eager mode (lesson from test failure)

The initial approach unconditionally removed all `record_stream()` calls in
`utils.py`. This caused test failures
(`test_schedule_chunk_1f1b::test_1f1b_schedule_model_chunk` on
`pre_mlp_layernorm.weight` gradient mismatch).

**Root cause**: In eager mode, when a tensor's Python reference is dropped
(e.g., `output_grad` going out of scope after `_backward` returns), PyTorch's
caching allocator decides when to recycle the memory. The allocator tracks
which stream a block was **allocated** on, and with no `record_stream` calls,
it has no knowledge of cross-stream readers. It may immediately return the
block to the allocation stream's free list, where a concurrent operation on
that same stream (from a different chunk or microbatch) allocates from it --
while GPU kernels on the **consuming** stream (e.g., `run_backward` inside
`attn.backward()` on comp_stream) are still reading asynchronously.

The event synchronization orders GPU **kernel launches**, but the allocator
runs on the CPU and makes recycling decisions independently. `record_stream()`
bridges this gap by telling the allocator "this block is also used on stream X;
don't recycle until stream X has caught up."

### Why `record_stream()` can be skipped during CUDA graph capture

During CUDA graph capture, no GPU work actually executes. The runtime records
a dependency graph of operations. When replayed:

- Memory is pre-allocated in a **private pool** sized for all concurrent
  allocations observed during capture.
- The captured stream ordering is **faithfully reproduced** by the GPU.
- The allocator does NOT make runtime recycling decisions -- the private pool
  is static.

So during capture, `record_stream()` only causes harm (deferred-free inflation
preventing block reuse within the capture) with no benefit (the replay handles
cross-stream safety). Skipping it during capture is both safe and necessary.

### Why `record_stream()` removal is safe for `residual`/`shared_expert_output`

Unlike `free_input` tensors and `output_grad` tensors, `residual` and
`shared_expert_output` do NOT need `record_stream()` even in eager mode. This
is because they are held alive by `node.before_detached` and `node.detached`
tuples (set via `node.detach()`) until `_release_state()` runs at the end of
`attn._backward()`. By that point, `attn.backward()` has already synchronized
with the comm stream (via the event chain), so the comm stream's reads of
these tensors in `combine.forward()` are guaranteed complete before the Python
references are released.

## Changes

### 1. Forward `free_input` path: `megatron/core/pipeline_parallel/utils.py`

`ScheduleNode._forward` -- conditional `record_stream()` on the
`free_input=True` input-freeing path (skip during CUDA graph capture):

```python
# Before:
if self.free_input:
    for input in inputs:
        if input is not None:
            input.record_stream(self.stream)        # always called
            input.untyped_storage().resize_(0)

# After (inside stream_acquire_context):
if self.free_input:
    for input in inputs:
        if input is not None:
            if not is_graph_capturing():
                input.record_stream(self.stream)    # skip during CG capture
            input.untyped_storage().resize_(0)
```

**Affected nodes**: `mlp` (`free_input=True`) and `moe_combine`
(`free_input=True`), per `should_free_input()` in `fine_grained_callables.py`.

**Tensor sizes**: mlp input is `[~32768, 3584]` bf16 (~224 MB); combine input
is `[~32768, 3584]` bf16 (~224 MB). 4 microbatches x 24 layers x 2 nodes =
192 deferred blocks eliminated during CG capture.

### 2. Backward `output_grad` path: `megatron/core/pipeline_parallel/utils.py`

`ScheduleNode._backward` -- conditional `record_stream()` on `output_grad`
tensors (skip during CUDA graph capture):

```python
# Before:
for g in output_grad:
    if g is not None:
        g.record_stream(self.stream)                # always called
        if self.manual_release_grads and not self.delay_grads_release:
            g.untyped_storage().resize_(0)

# After:
for g in output_grad:
    if g is not None:
        if not is_graph_capturing():
            g.record_stream(self.stream)            # skip during CG capture
        if self.manual_release_grads and not self.delay_grads_release:
            g.untyped_storage().resize_(0)
```

**Why `record_stream` is needed in eager mode**: `output_grad` tensors are
produced by the downstream node's backward on one stream (e.g., dispatch
backward on comm_stream) and consumed by this node's backward on `self.stream`
(e.g., attn backward on comp_stream). The event synchronization orders the GPU
work correctly, but when the Python references to these grad tensors are
dropped after `_backward` returns, the CUDA allocator only knows about the
allocation stream. Without `record_stream(self.stream)`, the allocator may
recycle the memory for the allocation stream's pool while `self.stream`'s GPU
kernels (from `backward_func`/`run_backward`) are still reading
asynchronously.

**Why safe to skip during CG capture**: During CUDA graph capture, no GPU work
actually executes; the runtime records a dependency graph. When the captured
graph is replayed, PyTorch's CG machinery pre-allocates all memory in a
private pool and replays the exact stream ordering from capture. The hardware
faithfully reproduces the stream dependencies, so cross-stream safety is
enforced by the graph structure itself, not by the allocator.

Note: `manual_release_grads` is always `False` in the current config (there is
a typo in `model_chunk_schedule_plan.py` -- it sets `manual_grads_release`
instead of `manual_release_grads`, so the attribute never takes effect). The
`resize_(0)` path is therefore never taken, but the `record_stream()` was
still causing cross-stream tagging on every grad tensor.

### 3. `residual` and `shared_expert_output`: `megatron/core/models/gpt/fine_grained_callables.py`

`submodule_combine_forward` -- removed `record_stream()` on `residual` and
`shared_expert_output`:

```python
# Before:
node.layer_state.residual.record_stream(torch.cuda.current_stream())
if shared_expert_output is not None:
    shared_expert_output.record_stream(torch.cuda.current_stream())

# After:
# (both calls removed, replaced with explanatory comment)
```

**Context**: `residual` and `shared_expert_output` are created by the `attn`
node on the compute stream via `node.detach()`, which stores them in both
`layer_state` (for `combine` to read) and `before_detached` (for backward).
They are consumed by `combine` on the comm stream. The `record_stream()` was
marking the comm stream as a user so that when these tensors are eventually
freed (after `attn`'s backward, during `_release_state()`), the allocator
would defer the free.

**Why safe**: These tensors live until `_release_state()` runs after `attn`'s
backward on the compute stream. By that point, the event chain guarantees both
streams have progressed well past the `combine` forward where the comm stream
last touched them. Without `record_stream()`, `stream_uses` is empty, so the
free is immediate and the block returns to the pool for reuse.

**Tensor sizes**: `residual` is `[4096, 2, 3584]` bf16 (~56 MB);
`shared_expert_output` is similar (~56 MB). 4 microbatches x 24 layers x
2 tensors = 192 deferred blocks eliminated.

## What is NOT changed

- **`record_stream()` in token dispatcher** (`token_dispatcher.py:900-916`):
  On small metadata tensors (splits, token counts) being D2H-transferred.
  Negligible memory impact.
- **`record_stream()` in activation offloading**
  (`fine_grained_activation_offload.py`): Separate system, not in the
  combined_1f1b hot path for this config.
- **`record_stream()` in optimizer offloading**
  (`optimizer_state_offloader.py`): Separate system.

## Correctness argument: cross-microbatch buffer reuse

The trickiest case is when a tensor freed by microbatch 0's node is reused by
microbatch 1's node on a **different stream**. Concrete example:

1. **`combine(mb0, L5)`** runs on comm stream, produces output into pool
   buffer P, records `Ev_f` on comm stream.
2. **`attn(mb0, L6)`** runs on compute stream (waits `Ev_f` first), reads P
   via `make_viewless().detach()`. Has `free_input=False`, so P stays alive
   in `_pool_inputs` / `self.inputs`.
3. **`attn(mb0, L6)._backward()`** runs on compute stream, reads P's data for
   gradient computation, then `_release_state()` drops references. CPython
   refcounting triggers `free()` on P's storage.
4. **`combine(mb1, Lk)`** runs on comm stream, allocates memory via
   `forward_func` -- caching allocator may return P's block.

**Is step 4 safe?** Between steps 3 and 4, the schedule executes:

- Step 3 records `Ev_b` on compute stream.
- Next layer's `B.combine` waits `Ev_b` on comm stream (syncs comm with
  compute through step 3).
- Multiple further operations zigzag between streams via events.
- Eventually, step 4's `F.combine(mb1)` waits `Ev_f` on comm stream. `Ev_f`
  was last recorded on compute stream, which has executed step 3's backward
  and everything after it (CUDA stream FIFO).

Therefore all GPU reads of P have completed before step 4's writer starts.

## Discarded alternative: ActivationPool

Before arriving at this fix, we implemented an `ActivationPool` class that
pre-allocated static buffers and cycled them between microbatches, avoiding
`record_stream()` and `resize_(0)` entirely. This worked correctly but had
two drawbacks:

1. **Extra memory**: Pool buffers + the original `self.output` tensors (which
   must stay alive for backward's `grad_fn` graph) created duplicate storage,
   adding ~20 GB vs eager baseline (130 GB vs 110 GB).
2. **Extra copies**: Each `free_input=True` node's output was copied into a
   pool buffer via `buf.copy_(tensor)`, adding latency.

The `record_stream()` removal approach is strictly better: zero copies, zero
extra memory, and the same correctness guarantee from the schedule's event
chain. The pool implementation is preserved in the git stash for reference.

## Summary of approach

| Site | Eager mode | CUDA graph capture |
|------|-----------|-------------------|
| `utils.py` `_forward` free_input | `record_stream()` ✓ | skip ✓ |
| `utils.py` `_backward` output_grad | `record_stream()` ✓ | skip ✓ |
| `fine_grained_callables.py` residual/shared_expert_output | removed ✓ | removed ✓ |

The guard `torch.cuda.is_current_stream_capturing()` controls whether
`record_stream()` is called. This queries the CUDA driver directly and works
for any capture method (both `TECudaGraphHelper` partial CG and
`FullCudaGraphWrapper` full-iteration CG).

**Earlier bug with `is_graph_capturing()`**: The original implementation used
`is_graph_capturing()` from `megatron.core.transformer.cuda_graphs`, which is
a manually-managed boolean (`_IS_GRAPH_CAPTURING`) set by
`_set_capture_start()`/`_set_capture_end()` inside `TECudaGraphHelper.create_cudagraphs()`.
`FullCudaGraphWrapper` uses raw `torch.cuda.graph()` and never calls these
functions, so `is_graph_capturing()` always returned `False` during
full-iteration capture. Switched to `torch.cuda.is_current_stream_capturing()`
which queries the CUDA driver directly and is correct for both capture paths.

## Expected impact

- **Eager mode**: Functionally identical to upstream. `record_stream()` is
  called in `utils.py` as before. The `fine_grained_callables.py` removal of
  `record_stream()` on `residual`/`shared_expert_output` has no functional
  impact because those tensors are held alive by `before_detached`/`detached`
  references.
- **CUDA graph capture**: Eliminates ~80 GB of deferred-free memory inflation
  from the `utils.py` sites (the `fine_grained_callables.py` site is
  unconditionally removed). Target: ~110 GB, down from ~190 GB.

## Grouped GEMM and CUDA Graph Compatibility

Full-iteration CUDA graph capture places MoE expert compute (grouped GEMM)
inside the captured graph. Not all grouped GEMM backends in TE are CG-safe:

| Backend | Path | CG-safe? | Issue |
|---------|------|----------|-------|
| cuBLASLt | `general_grouped_gemm_for_grouped_tensor()` | **No** | Calls `cublasLtMatmulAlgoGetHeuristic` (CPU-side API) every forward; allocates `torch.empty()` for setup workspace on every call |
| CUTLASS Hopper | `cutlass_grouped_gemm.cuh` | **No** | `cudaMemcpyAsync` from non-pinned host memory (`std::malloc`) every call; incompatible with graph capture |
| CUTLASS Device-Initiated (Blackwell MXFP8) | `device_init_grouped_gemm.py` | **Yes** | Pre-allocated 16 MiB device workspace + pinned host buffer (global singleton); device-side argument setup kernel; no CPU-side operations during execution |

**Only the device-initiated CUTLASS path is CG-safe.** This requires:
- Blackwell GPU (SM >= 100) for MXFP8 support
- `--moe-use-device-initiated-grouped-gemm` (`moe_use_device_initiated_grouped_gemm=True`)
- MXFP8 quantization (`fp8='e4m3'`, `fp8_recipe='mxfp8'`)
- HybridEP flex dispatcher (`moe_token_dispatcher_type='flex'`, `moe_flex_dispatcher_backend='hybridep'`)
- Static token capacity (`moe_received_token_capacity` set)

MCore enforces this at config validation time (when `cuda_graph_impl != "none"`)
in `transformer_config.py:2178-2189`:

```python
moe_compute_in_graph = (
    CudaGraphScope.moe in self.cuda_graph_scope
    or not self.cuda_graph_scope
    or CudaGraphScope.full_iteration in self.cuda_graph_scope
)
if moe_compute_in_graph:
    assert (
        self.moe_token_dispatcher_type == 'flex'
        and self.moe_flex_dispatcher_backend == 'hybridep'
        and self.moe_received_token_capacity is not None
        and self.moe_use_device_initiated_grouped_gemm
    ), 'moe cuda graph is only supported with sync-free MoE.'
```

The existing partial CG test (`test_cuda_graphed_schedule_chunk_1f1b.py`)
only captures `attn`, `moe_router`, and `moe_preprocess` scopes — expert
compute (grouped GEMM) runs outside the graph.

The `GroupedLinear` TE module also lacks the `is_graph_capturing()` guard for
FP8 scale factor reduction that other TE modules (`Linear`, `LayerNormLinear`,
`LayerNormMLP`) have. This is safe only because the device-initiated path
handles scale factors differently via MXFP8.

## `manual_release_grads` Typo Bug

`model_chunk_schedule_plan.py:178-179` had a typo:

```python
# Bug (sets a nonexistent attribute):
self.mlp.manual_grads_release = False
self.moe_combine.manual_grads_release = False

# Fixed (correct attribute name, matches utils.py:180):
self.mlp.manual_release_grads = False
self.moe_combine.manual_release_grads = False
```

This typo meant that when `CudaGraphScope.attn` was in `cuda_graph_scope`,
the code intended to disable manual grad release for `mlp` and `moe_combine`
(since their dgrad comes from attn, which is managed by cuda graph), but
the flag was never actually set on the correct attribute. The `resize_(0)` in
`_backward` was still being called (or not, depending on the default value of
`manual_release_grads`). Fixed in this change.

## Files changed

```
megatron/core/pipeline_parallel/utils.py              # _forward and _backward (conditional record_stream)
megatron/core/models/gpt/fine_grained_callables.py    # submodule_combine_forward (conditional record_stream)
megatron/core/models/common/model_chunk_schedule_plan.py  # manual_grads_release → manual_release_grads typo fix
tests/unit_tests/a2a_overlap/test_full_cg_schedule_chunk_1f1b.py  # New: full-iteration CG capture+replay test
```

## Test design

See [test_full_cg_design.md](test_full_cg_design.md) for the detailed design
of `test_full_cg_schedule_chunk_1f1b.py`, including the multistream schedule
exercised, the `forward_step_func` / `loss_func` pattern, state reset between
runs, and known coverage gaps.
