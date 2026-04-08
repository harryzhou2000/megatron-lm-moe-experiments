# Test Design: `test_full_cg_schedule_chunk_1f1b.py`

Full-iteration CUDA graph capture + replay correctness test for the
`combined_1f1b` A2A overlap schedule with device-initiated grouped GEMM.

File: `tests/unit_tests/a2a_overlap/test_full_cg_schedule_chunk_1f1b.py`

## Purpose

Validate that `FullCudaGraphWrapper` (the production full-iteration CG path)
produces bit-exact loss values compared to eager execution when the model runs
the multistream `combined_1f1b` EP overlap schedule. This is the primary
regression test for the conditional `record_stream()` changes and confirms
that device-initiated CUTLASS grouped GEMM is CG-safe.

## What it exercises

### The multistream combined_1f1b schedule

With `num_microbatches=4`, `combined_1f1b_schedule_for_no_pipelining` runs:

```
Phase 0:  MB0 forward only         (f_model=model, b_model=None)
Phase 1:  MB0 backward + MB1 fwd   (f_model=model, b_model=model)  ← OVERLAP
Phase 2:  MB1 backward + MB2 fwd   (f_model=model, b_model=model)  ← OVERLAP
Phase 3:  MB2 backward + MB3 fwd   (f_model=model, b_model=model)  ← OVERLAP
Phase 4:  MB3 backward only        (f_model=None,  b_model=model)
```

Phases 1-3 call `TransformerLayerSchedulePlan.run(f_layer, b_layer)` with both
non-None, which runs the full interleaved two-stream schedule:

```
comm_stream:  combine_bwd | dispatch_fwd → dispatch_bwd  | combine_fwd
comp_stream:  attn_fwd    | mlp_bwd → mlp_bwd_dw → mlp_fwd | attn_bwd
```

This is where `record_stream()` is critical in eager mode and must be skipped
during CG capture. Each node runs inside `stream_acquire_context()` which
does `event.wait(stream)` / `event.record(stream)` to synchronize across the
two streams.

**Why `num_microbatches >= 2` is required**: With `num_microbatches=1`, the
overlap loop is `range(num_microbatches - 1) = range(0)` and never executes.
`overlapped_layers = min(f_num_layers, b_num_layers) = 0`. The model goes
through the `combined_1f1b` code path (schedule plan, ScheduleNode wrappers)
but the actual multistream forward+backward overlap never runs. An earlier
version of this test had `num_microbatches=1` and passed trivially without
exercising the schedule that makes `record_stream()` relevant.

### The full-iteration CUDA graph capture path

`FullCudaGraphWrapper` (`megatron/core/full_cuda_graph.py`) wraps the entire
`forward_backward_func` into a single `torch.cuda.CUDAGraph`:

1. **Warmup** (steps 0..W-1): Runs `forward_backward_func` eagerly to let
   cuBLAS/cuDNN autotuners settle. `StaticBufferLoader` creates static CUDA
   buffers from the data iterator on the first call.
2. **Capture** (step W): Creates a `torch.cuda.CUDAGraph`, registers RNG
   generator states, captures `forward_backward_func` inside
   `torch.cuda.graph()`. The captured result (a reference to
   `forward_data_store`) is stored as `result['training']`.
3. **Replay** (steps W+1..N): `StaticBufferLoader` copies new data into static
   buffers via `clone_tensors_in_struct` (in-place update). Then `.replay()`
   re-executes the captured graph. The result references are stable — tensor
   storage is updated in-place by the replay.

### The production forward_step_func pattern

The test's `forward_step_func` mirrors `pretrain_gpt.py:forward_step` exactly:

```python
def forward_step_func(data_iterator, model, return_schedule_plan=False):
    batch = next(data_iterator)
    loss_mask = batch["loss_mask"]
    if return_schedule_plan:
        schedule_plan = model.build_schedule_plan(**batch)
        return schedule_plan, partial(_loss_func, loss_mask, model=model)
    output = model.forward(**batch)
    return output, partial(_loss_func, loss_mask, model=model)
```

Key design points:

- **`return_schedule_plan=True`**: Returns `(schedule_plan, loss_func)` where
  `loss_func` is a `functools.partial` with `loss_mask` captured. This is the
  path used by `combined_forward_backward_step` (line 347 of
  `combined_1f1b.py`).
- **`loss_func` signature**: `_loss_func(loss_mask, output_tensor, model=None)`
  → `(loss, num_tokens, report)`. The `partial` binds `loss_mask` as the first
  positional arg, so when called as `loss_func(output_tensor)` by
  `forward_step_calc_loss`, it becomes `_loss_func(loss_mask, output_tensor)`.
- **`_loss_func`** is a simplified version of `pretrain_gpt.py:loss_func`.
  It applies `loss_mask` to per-token cross-entropy loss and returns the
  3-tuple expected by `forward_step_calc_loss`. The original had ModelOpt,
  rerun state machine, and spiky loss checks — all removed for the test.

**Earlier bug**: The test originally returned `model.compute_language_model_loss`
as `loss_func`. This method has signature `(self, labels, logits)` and requires
two positional args, but `forward_step_calc_loss` calls `loss_func(output_tensor)`
with only one arg. Production `forward_step` returns a closure via `partial()`.

### The loss extraction path

`forward_backward_no_pipelining` returns `forward_data_store`, a list of dicts.
`forward_step_calc_loss` appends `loss_reduced` from `_loss_func`:

```python
loss_reduced = {'lm loss': torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])}
```

The test extracts `result['lm loss'][0]` (the scalar loss value).

Note: `forward_step_calc_loss` also divides `output_tensor` (used for backward)
by `num_microbatches` for gradient accumulation scaling. This in-place
modification affects the autograd graph but NOT the `loss_reduced` dict, since
`loss_reduced` was created from `loss.clone().detach()`.

## Test structure

### Two-run comparison

```
Run 1 (eager):  Same CG config, no FullCudaGraphWrapper  →  eager_losses
Run 2 (CG):     Same CG config, with FullCudaGraphWrapper →  cg_losses
Compare:        torch.equal(eager_losses[i], cg_losses[i]) for all i
```

Both runs:
- Use identical model config (`cuda_graph_impl="local"`, `cuda_graph_scope=[full_iteration]`)
- Start from the same seed (`torch.manual_seed(123)` + `model_parallel_cuda_manual_seed(123)`)
- Process the same data (4 microbatches × 5 steps = 20 forward passes each)
- Run `optimizer.step()` after each training step

The only difference is whether `FullCudaGraphWrapper` wraps `forward_backward_func`.

### Step breakdown (5 steps, warmup=3)

| Step | Eager run (Run 1) | CG run (Run 2) |
|------|-------------------|-----------------|
| 0    | Eager             | Eager (warmup)  |
| 1    | Eager             | Eager (warmup)  |
| 2    | Eager             | Eager (warmup)  |
| 3    | Eager             | **CG capture**  |
| 4    | Eager             | **CG replay**   |

Each step produces 4 loss values (one per microbatch), totalling 20 comparisons.

### Comparison strength

- `torch.equal` is **bit-exact**. If the CG path produced even slightly
  different numerics, the test would fail.
- Distinct data per microbatch (offset by `microbatch_idx * 7`) ensures
  different tokens flow through forward/backward of overlapped microbatches.
  This prevents the degenerate case where all microbatches produce identical
  loss and a broken schedule could go undetected.

## Setup / teardown details

### Environment variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `CUDA_DEVICE_MAX_CONNECTIONS` | `1` | Required by combined_1f1b overlap (single HW queue per context) |
| `CUBLAS_WORKSPACE_CONFIG` | `:4096:8` | Deterministic cuBLAS algorithms |
| `NVTE_ALLOW_NONDETERMINISTIC_ALGO` | `0` | Deterministic TE kernels |
| `NCCL_NVLS_ENABLE` | `0` | Disable NVLS (not CG-safe) |
| `NCCL_ALGO` | `^NVLS` | Exclude NVLS algorithm |

All are saved/restored in `setup_method`/`teardown_method`.

### State reset between runs

The test must clean up ALL process-global singletons between Run 1 and Run 2:

| Singleton | Reset method |
|-----------|-------------|
| `FullCudaGraphWrapper.curr_iteration`, `.cuda_graph`, `.result` | `_reset_full_cuda_graph_wrapper_state()` |
| `StaticBufferLoader.static_buffers` | Same helper |
| `_COMM_STREAM` in `pipeline_parallel.utils` | `pp_utils._COMM_STREAM = None` |
| `_CUDA_RNG_STATE_TRACKER` | `force_reset_rng=True` in `model_parallel_cuda_manual_seed()` |
| `_CUDA_RNG_STATE_TRACKER_INITIALIZED` | Reset by `force_reset_rng=True` |
| Global vars (args, timers, etc.) | `destroy_global_vars()` |
| Num microbatches calculator | `destroy_num_microbatches_calculator()` |
| Model parallel state | `Utils.destroy_model_parallel()` |

**Why `_COMM_STREAM` must be reset**: `set_streams()` creates `_COMM_STREAM`
as a module-level singleton on the current CUDA device. Between runs,
`Utils.destroy_model_parallel()` does NOT reset it. If left stale, the Run 2
call to `set_streams()` would be a no-op (since `_COMM_STREAM is not None`).
While both runs use the same device in practice, explicitly resetting ensures
`set_streams()` creates a fresh stream after `Utils.initialize_model_parallel()`
has called `torch.cuda.set_device(local_rank)`.

**Why `set_streams()` must be AFTER `Utils.initialize_model_parallel()`**:
`set_streams()` calls `torch.cuda.Stream(device="cuda")`, which creates a
stream on the current default device. `Utils.initialize_model_parallel()`
calls `torch.cuda.set_device(local_rank)`. If `set_streams()` is called
first, all ranks create the stream on `cuda:0`. Then `torch.cuda.Event()`
created later on the correct per-rank device doesn't match the stream's device,
causing `RuntimeError: Event device does not match recording stream's device`.

**Why `force_reset_rng=True`**: `model_parallel_cuda_manual_seed` uses
`_CUDA_RNG_STATE_TRACKER`, guarded by `_CUDA_RNG_STATE_TRACKER_INITIALIZED`.
Without `force_reset_rng=True`, Run 2 inherits Run 1's tracker type (Megatron
vs TE). With `use_te_rng_tracker=True`, this means the TE tracker from Run 1
persists but the model expects a fresh one. `force_reset_rng=True` clears
the initialized flag and creates a new tracker.

## Model config

Scaled-down MoE model matching the production Qwen3-style config:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `num_layers` | 1 | Minimal, but sufficient (1 layer = 1 MoE block with attn+dispatch+mlp+combine) |
| `num_experts` | 8 | EP=4, so 2 local experts per rank |
| `expert_model_parallel_size` | 4 | Requires 4 GPUs |
| `moe_router_topk` | 2 | Standard top-2 routing |
| `vocab_size` | 1024 | Small for fast testing |
| `hidden_size` | 128 | Small for fast testing |
| `seq_length` | 512 | |
| `micro_batch_size` | 2 | |
| `num_microbatches` | 4 | Gives 3 overlapped phases |
| `fp8` | `e4m3` / `mxfp8` | Required for device-initiated CUTLASS |
| `moe_token_dispatcher_type` | `flex` / `hybridep` | Required for CG-safe dispatch |
| `moe_use_device_initiated_grouped_gemm` | True | Only CG-safe grouped GEMM path |
| `cuda_graph_impl` | `local` | Full-iteration CG via `FullCudaGraphWrapper` |
| `cuda_graph_scope` | `[full_iteration]` | Captures entire forward+backward |
| `overlap_moe_expert_parallel_comm` | True | Enables `combined_1f1b` path |
| `use_te_rng_tracker` | True | Required for CG-safe RNG |

## Call chain

```
test_full_iter_cg_combined_1f1b
  ├── _run_training_steps(use_full_iter_cg=False)   # Run 1: eager
  │     └── forward_backward_func(...)
  │           └── forward_backward_no_pipelining(...)
  │                 └── combined_1f1b_schedule_for_no_pipelining(...)
  │                       ├── combined_forward_backward_step(f_model, b_model=None)      # MB0 fwd
  │                       ├── combined_forward_backward_step(f_model, b_model)  ×3        # overlap
  │                       │     ├── forward_step_func(data_iter, model, return_schedule_plan=True)
  │                       │     │     └── model.build_schedule_plan(**batch)
  │                       │     │     └── return (schedule_plan, partial(_loss_func, loss_mask))
  │                       │     ├── TransformerModelChunkSchedulePlan.run(f_plan, b_plan)
  │                       │     │     └── TransformerLayerSchedulePlan.run(f_layer, b_layer)
  │                       │     │           ├── b_layer.moe_combine.backward()   [comm_stream]
  │                       │     │           ├── f_layer.attn.forward()            [comp_stream]
  │                       │     │           ├── b_layer.mlp.backward()            [comp_stream]
  │                       │     │           ├── f_layer.moe_dispatch.forward()    [comm_stream]
  │                       │     │           ├── f_layer.mlp.forward()             [comp_stream]
  │                       │     │           ├── f_layer.moe_combine.forward()     [comm_stream]
  │                       │     │           └── b_layer.attn.backward()           [comp_stream]
  │                       │     └── forward_step_calc_loss(output, loss_func, ...)
  │                       │           └── loss_func(output_tensor) → (loss, num_tokens, report)
  │                       └── combined_forward_backward_step(f_model=None, b_model)      # MB3 bwd
  │
  ├── _run_training_steps(use_full_iter_cg=True)    # Run 2: CG
  │     └── FullCudaGraphWrapper(forward_backward_func)(...)
  │           ├── data_read() → StaticBufferLoader copies to static CUDA buffers
  │           ├── [step < W]: forward_backward_func(...)  # warmup (same as Run 1)
  │           ├── [step == W]: torch.cuda.graph() { forward_backward_func(...) }  # capture
  │           └── [step > W]: .replay()  # replay with updated static buffers
  │
  └── Compare: torch.equal(eager_losses[i], cg_losses[i])
```

## What the test does NOT cover

- **Multiple transformer layers**: `num_layers=1`. Production uses 24+.
  The schedule interleaves forward-of-layer-i with backward-of-layer-j
  across microbatches, but with 1 layer the intra-model-chunk loop is trivial.
  This is acceptable because the multistream overlap is at the microbatch
  level (tested) and the layer-level schedule is a loop of the same pattern.
- **Pipeline parallelism**: PP=1. The interleaved VP pipeline schedule
  (`combined_1f1b_schedule_for_interleaved_pipelining`) is not exercised.
- **Shared experts**: Not enabled in this test config. The
  `residual`/`shared_expert_output` `record_stream()` removal in
  `fine_grained_callables.py` is not directly validated here.
- **Memory reduction measurement**: The test checks correctness, not that CG
  private pool memory is reduced from ~190 GB to ~110 GB. That requires a
  production-scale model.
- **Dynamic token counts**: `moe_received_token_capacity=64` pads to fixed
  capacity. Variable-length dispatch is not tested.
- **Gradient accumulation all-reduce correctness**: The test runs with EP=4
  and DP is implicit, but it only checks loss values, not that gradients
  are correctly reduced across ranks.

## Requirements

- Blackwell GPU (SM >= 100) for MXFP8 device-initiated CUTLASS grouped GEMM
- 4 GPUs (EP=4)
- HybridEP dispatcher (`HAVE_HYBRIDEP`)
- TE >= 1.9.0.dev0

```bash
torchrun --nproc_per_node 4 -m pytest -xvs \
    "tests/unit_tests/a2a_overlap/test_full_cg_schedule_chunk_1f1b.py"
```
