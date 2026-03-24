# DeepEP permute assertion num_of_local_experts

**Session ID:** ses_2e6ec963effefxS3Mx64PUsfxI
**Created:** 3/23/2026, 1:03:15 PM
**Updated:** 3/23/2026, 1:06:48 PM

---

## User

I got python: /home/scratch.hhanyu_gpu/projects/moe/DeepEP/csrc/hybrid_ep/extension/permute.cu:265: std::tuple<at::Tensor, at::Tensor, at::Tensor> permute_preprocessing(bool*, at::Tensor, int, int, int, int, int, bool, cudaStream_t): Assertion `num_of_local_experts <= block_size' failed. , please tell me what the sizes are and how to configure them in megatron.

---

## Assistant (Build · aws/anthropic/bedrock-claude-opus-4-6 · 194.8s)



Let me investigate the assertion failure in the DeepEP permute preprocessing code and trace how these sizes are configured.

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/harry/projects/moe/DeepEP/csrc/hybrid_ep/extension/permute.cu",
  "offset": 250,
  "limit": 40
}
```

**Output:**
```
<path>/Users/harry/projects/moe/DeepEP/csrc/hybrid_ep/extension/permute.cu</path>
<type>file</type>
<content>250:  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> 
251:  permute_preprocessing(
252:      bool* routing_map,
253:      torch::Tensor num_dispatched_token_tensor,
254:      // Used in the permute case, use up-bound to avoid synchronization to get the real num_dispatched_tokens from the pinned memory
255:      int max_num_dispatched_tokens,
256:      int num_of_local_experts,
257:      int pad_multiple,
258:      int num_of_blocks,
259:      int num_permuted_tokens,
260:      bool non_blocking,
261:      cudaStream_t stream) {
262:    constexpr int block_size = 256;
263:    const int warp_size = 32;
264: 
265:    assert(num_of_local_experts <= block_size);
266:    assert(num_of_local_experts > 0);
267:  
268:    auto row_id_map = torch::empty({max_num_dispatched_tokens + pad_multiple, num_of_local_experts},
269:                                   torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
270:    torch::Tensor tokens_per_expert;
271:    if (non_blocking) {
272:      tokens_per_expert =
273:          torch::empty({num_of_local_experts}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
274:    } else {
275:      tokens_per_expert =
276:          torch::empty({num_of_local_experts}, torch::TensorOptions().dtype(torch::kInt64).pinned_memory(true));
277:    }
278:    torch::Tensor overflow_flag = torch::empty({1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
279:  
280:    // Construct the template buffers
281:    int rows_workspace_1 = (max_num_dispatched_tokens + block_size - 1) / block_size;
282:    int rows_workspace_2 = (rows_workspace_1 + block_size - 1) / block_size;
283:    auto workspace1 = torch::empty({rows_workspace_1, num_of_local_experts},
284:                                   torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
285:    auto workspace2 = torch::empty({rows_workspace_2, num_of_local_experts},
286:                                   torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
287:  
288:    // Compute the size of the shared memory
289:    int shared_mem_size_pass_1_2 = block_size * num_of_local_experts * sizeof(int);

(Showing lines 250-289 of 581. Use offset=290 to continue.)
</content>
```

**Tool: task**

**Input:**
```json
{
  "description": "Find DeepEP permute callers",
  "prompt": "In the repository at /Users/harry/projects/moe, search the entire codebase (including all subdirectories like DeepEP/, TE/, etc.) for:\n\n1. All call sites that invoke `permute_preprocessing` — find how `num_of_local_experts` and `block_size` arguments are passed.\n2. Any Python bindings or wrappers that call into this CUDA function (look for pybind, torch extension, or similar).\n3. Any Megatron or configuration code that sets `num_local_experts`, `num_of_local_experts`, `block_size`, or similar MoE expert count / block size parameters.\n\nReturn:\n- The full function signature of `permute_preprocessing`\n- Every call site with file paths and line numbers\n- How `block_size` is determined (is it a CUDA block dim? a config param?)\n- How `num_of_local_experts` flows from Python config down to this CUDA assertion\n- Any relevant Megatron config fields (like `--num-experts`, `--expert-model-parallel-size`, etc.) that affect these values",
  "subagent_type": "explore"
}
```

**Output:**
```
task_id: ses_2e6ec6092ffe2lgVY2HuN1kMYv (for resuming to continue this task if needed)

<task_result>
Now I have all the information needed. Let me compile a comprehensive answer.

---

## Complete Analysis of `permute_preprocessing` and MoE Expert Count / Block Size Parameters

### 1. Full Function Signature of `permute_preprocessing`

**File:** `/Users/harry/projects/moe/DeepEP/csrc/hybrid_ep/extension/permute.cu`, lines 250-261
**Declaration:** `/Users/harry/projects/moe/DeepEP/csrc/hybrid_ep/extension/permute.cuh`, lines 76-86

```cpp
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
permute_preprocessing(
    bool* routing_map,                        // shape: [num_dispatched_tokens, num_of_local_experts]
    torch::Tensor num_dispatched_token_tensor, // GPU tensor with actual dispatch count
    int max_num_dispatched_tokens,             // upper-bound to avoid GPU sync
    int num_of_local_experts,                  // number of experts on this rank
    int pad_multiple,                          // alignment padding (e.g. 32 for FP8 GEMM)
    int num_of_blocks,                         // CUDA grid size (SM count)
    int num_permuted_tokens,                   // upper bound for permuted output, or -1
    bool non_blocking,                         // if true, output stays on GPU; if false, pinned mem + sync
    cudaStream_t stream);                      // CUDA stream
```

Returns `(row_id_map, tokens_per_expert, overflow_flag)`.

The underlying kernel is:

```cpp
template <const int block_size = 512, const int warp_size = 32>
__global__ void permute_preprocessing_kernel(
    bool* routing_map, int* num_dispatched_tokens_ptr, int num_of_local_experts,
    int* workspace_1, int rows_workspace_1, int* workspace_2, int rows_workspace_2,
    int pad_multiple, int64_t* tokens_per_expert, int* row_id_map,
    int* overflow_flag, int num_permuted_tokens);
```

---

### 2. Every Call Site

| #   | File                                                                 | Line(s) | How `num_of_local_experts` is passed                           | How `block_size` is determined                                                                                                                    |
| --- | -------------------------------------------------------------------- | ------- | ------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| **1**   | `/Users/harry/projects/moe/DeepEP/csrc/hybrid_ep/executor/executor.cu` | 145-155 | `config.num_of_experts_per_rank` (from `HybridEpConfigInstance`) | `config.num_of_blocks_preprocessing_api` (for grid `num_of_blocks`); the kernel template `block_size` is hardcoded `256` inside `permute_preprocessing()` |

This is the **only direct call site** of the C++ `permute_preprocessing()` function in the entire codebase. It lives inside `Executor::dispatch_preprocess()`:

```cpp
// executor.cu:145-155
std::tie(row_id_map, tokens_per_expert, overflow_flag) = permute_preprocessing(
    args.local_expert_routing_map.value().data_ptr<bool>(),
    args.num_dispatched_tokens_tensor.value(),
    args.max_num_dispatched_tokens,
    config.num_of_experts_per_rank,          // <-- num_of_local_experts
    args.pad_multiple,
    config.num_of_blocks_preprocessing_api,  // <-- num_of_blocks (grid dim)
    args.num_permuted_tokens,
    args.non_blocking,
    args.stream
);
```

The `Executor::dispatch_postprocess()` at lines 259-276 also references `num_of_local_experts` indirectly by setting `permute_args.num_of_local_experts = config.num_of_experts_per_rank` and `unpermute_args.num_of_local_experts = config.num_of_experts_per_rank` (lines 269, 345), but those go to `permute_launcher` and `unpermute_launcher`, not `permute_preprocessing`.

---

### 3. How `block_size` Is Determined

**`block_size` is NOT a config param. It is a compile-time CUDA constant (template parameter).**

Inside `permute_preprocessing()` at line 262 of `permute.cu`:

```cpp
constexpr int block_size = 256;  // threads per CUDA block
```

This value is used as:
- The template argument to `permute_preprocessing_kernel<block_size, warp_size>` (line 306-308)
- The CUDA launch block dimension: `cudaLaunchCooperativeKernel(..., block_size, ...)` (line 308-309)
- Shared memory sizing: `block_size * num_of_local_experts * sizeof(int)` (line 289)
- Workspace row calculations: `rows_workspace_1 = (max_num_dispatched_tokens + block_size - 1) / block_size` (line 281)

The **critical assertion** at line 265 is:
```cpp
assert(num_of_local_experts <= block_size);  // i.e. num_of_local_experts <= 256
```

This exists because:
- Pass 1-2: shared memory is used as tiles of shape `[block_size, num_of_local_experts]`, and each thread in the block processes one row of the tile, iterating over `num_of_local_experts` columns.
- Pass 3: A `BlockScan` prefix-sum is computed over `num_of_local_experts` values loaded into the first `num_of_local_experts` threads, which requires `num_of_local_experts <= block_size` (line 172: `int value = threadIdx.x < num_of_local_experts ? ... : 0;`).

The other kernels (`permute_kernel`, `unpermute_kernel`) use `constexpr int block_size = 512` (lines 439, 561), but those are separate kernels.

The **grid size** (`num_of_blocks`) IS a config param. It comes from `config.num_of_blocks_preprocessing_api`, which defaults to **108** SMs (set in Python at `hybrid_ep_buffer.py` line 90).

---

### 4. How `num_of_local_experts` Flows from Python Config Down to the CUDA Assertion

Here is the full data flow, traced end-to-end:

#### Layer 1: Megatron CLI / TransformerConfig

```
--num-experts N                    (CLI argument, arguments.py:2724)
  --> args.num_experts = N         (argparse)
  --> kw_args['num_moe_experts'] = args.num_experts   (arguments.py:1350)
  --> TransformerConfig.num_moe_experts = N            (transformer_config.py:190)

--expert-model-parallel-size E     (CLI argument, via ModelParallelConfig dataclass)
  --> ModelParallelConfig.expert_model_parallel_size = E   (model_parallel_config.py:75)
```

#### Layer 2: MoELayer computes `num_local_experts`

File: `/Users/harry/projects/moe/MLM/megatron/core/transformer/moe/moe_layer.py`, line 102:
```python
self.num_local_experts = self.config.num_moe_experts // ep_size
```
where `ep_size = get_pg_size(self.ep_group)` (which equals `expert_model_parallel_size`).

#### Layer 3: Token Dispatcher passes it to HybridEP

File: `/Users/harry/projects/moe/MLM/megatron/core/transformer/moe/token_dispatcher.py`

The `_HybridEPManager` (line 959-1003) stores `self.num_local_experts = num_local_experts`.

On dispatch (line 1048-1059), it calls:
```python
hybrid_ep_dispatch(
    x=hidden_states,
    ...
    num_local_experts=self.num_local_experts,
    ...
)
```

#### Layer 4: `hybrid_ep_dispatch` -> `HybridEPDispatch.apply`

File: `/Users/harry/projects/moe/MLM/megatron/core/transformer/moe/fused_a2a.py`, lines 446-494:

`hybrid_ep_dispatch()` calls `HybridEPDispatch.apply(x, routing_map, probs, group, num_local_experts, ...)`.

#### Layer 5: `HybridEPDispatch.forward` -> `init_hybrid_ep_buffer` -> `HybridEPBuffer.__init__`

File: `fused_a2a.py`, lines 337-384:

```python
init_hybrid_ep_buffer(group, hidden_dim, seq_len, num_local_experts, ...)
```

This creates a `HybridEPBuffer(... num_local_experts=num_local_experts ...)`.

#### Layer 6: `HybridEPBuffer.__init__` -> `BufferConfig`

File: `/Users/harry/projects/moe/DeepEP/deep_ep/hybrid_ep_buffer.py`, line 112:
```python
self.config.num_of_experts_per_rank = num_local_experts
```

#### Layer 7: `dispatch_with_permute` -> `update_template_config` -> `HybridEpConfigInstance`

File: `hybrid_ep_buffer.py`, lines 456-461:
```python
config = self.update_template_config(
    ...
    num_local_experts=num_of_experts_per_rank,
    ...
)
```

Inside `update_template_config` (lines 183-186):
```python
config.num_of_experts_per_rank = (
    num_local_experts if num_local_experts is not None
    else self.config.num_of_experts_per_rank
)
```

#### Layer 8: Pybind -> C++ `HybridEpConfigInstance`

File: `/Users/harry/projects/moe/DeepEP/csrc/hybrid_ep/pybind_hybrid_ep.cu`, line 69-70:
```cpp
.def_readwrite("num_of_experts_per_rank",
               &HybridEpConfigInstance::num_of_experts_per_rank)
```

The C++ struct is defined in `/Users/harry/projects/moe/DeepEP/csrc/hybrid_ep/config.cuh`, line 57:
```cpp
int num_of_experts_per_rank;
```

#### Layer 9: `Executor::dispatch_preprocess` -> `permute_preprocessing`

File: `/Users/harry/projects/moe/DeepEP/csrc/hybrid_ep/executor/executor.cu`, line 149:
```cpp
config.num_of_experts_per_rank   // passed as num_of_local_experts arg
```

#### Layer 10: The assertion

File: `/Users/harry/projects/moe/DeepEP/csrc/hybrid_ep/extension/permute.cu`, lines 265-266:
```cpp
assert(num_of_local_experts <= block_size);  // block_size = 256
assert(num_of_local_experts > 0);
```

**Summary formula:**
```
num_of_local_experts = num_moe_experts / expert_model_parallel_size
                     = --num-experts  / --expert-model-parallel-size

Assertion requires: num_of_local_experts <= 256
```

---

### 5. Relevant Megatron Config Fields

| Config Field / CLI Arg                          | Location                                                                          | Type            | Effect on `num_of_local_experts`                                                                                 |
| ----------------------------------------------- | --------------------------------------------------------------------------------- | --------------- | -------------------------------------------------------------------------------------------------------------- |
| `--num-experts N`                                 | `/Users/harry/projects/moe/MLM/megatron/training/arguments.py:2724`                 | `int`             | Total number of MoE experts globally. Maps to `TransformerConfig.num_moe_experts`.                               |
| `TransformerConfig.num_moe_experts`               | `/Users/harry/projects/moe/MLM/megatron/core/transformer/transformer_config.py:190` | `Optional[int]`   | The core config field. Set from `args.num_experts` at `arguments.py:1350`.                                         |
| `--expert-model-parallel-size E`                  | Via `ModelParallelConfig` (dataclass field picked up by argparse)                   | `int` (default 1) | Expert parallelism degree. Defined at `/Users/harry/projects/moe/MLM/megatron/core/model_parallel_config.py:75`. |
| `ModelParallelConfig.expert_model_parallel_size`  | `/Users/harry/projects/moe/MLM/megatron/core/model_parallel_config.py:75`           | `int` (default 1) | Distributes MoE experts across sub-data-parallel dimension.                                                    |
| `ModelParallelConfig.expert_tensor_parallel_size` | `/Users/harry/projects/moe/MLM/megatron/core/model_parallel_config.py:78`           | `Optional[int]`   | Intra-layer tensor parallelism for expert layers. Affects the EP group size when combined with EP.             |
| `BaseMoELayer.num_local_experts`                  | `/Users/harry/projects/moe/MLM/megatron/core/transformer/moe/moe_layer.py:102`      | computed        | `= config.num_moe_experts // ep_size`                                                                            |
| `BufferConfig.num_of_experts_per_rank`            | `/Users/harry/projects/moe/DeepEP/csrc/hybrid_ep/config.cuh:16`                     | `int`             | C++ config struct field, set from Python `num_local_experts` via pybind.                                         |
| `HybridEpConfigInstance.num_of_experts_per_rank`  | `/Users/harry/projects/moe/DeepEP/csrc/hybrid_ep/config.cuh:57`                     | `int`             | Runtime config struct, passed directly to `permute_preprocessing`.                                               |
| `HybridEPBuffer.__init__` param `num_local_experts` | `/Users/harry/projects/moe/DeepEP/deep_ep/hybrid_ep_buffer.py:41`                   | `int`             | Python constructor parameter; stored as `self.config.num_of_experts_per_rank` at line 112.                       |
| `NUM_LOCAL_EXPERTS` (env var in tests)            | `/Users/harry/projects/moe/DeepEP/tests/test_hybrid_ep.py:16`                       | `int` (default 8) | Test-level default.                                                                                            |

### Validation constraints in the C++ config

From `config.cuh` line 37/101:
```cpp
valid &= ((num_of_experts_per_rank * num_of_ranks_per_node) % 4 == 0);
```

And the critical `permute_preprocessing` constraint at `permute.cu:265`:
```cpp
assert(num_of_local_experts <= block_size);  // block_size is hardcoded 256
assert(num_of_local_experts > 0);
```

This means: **if `--num-experts / --expert-model-parallel-size > 256`, the `permute_preprocessing` CUDA kernel will fail with an assertion error.**
</task_result>
```

---

## Assistant (Build · aws/anthropic/bedrock-claude-opus-4-6 · 18.0s)

## The Assertion Explained

The failing assertion is at `DeepEP/csrc/hybrid_ep/extension/permute.cu:265`:

```cpp
constexpr int block_size = 256;   // line 262 — hardcoded CUDA block size
assert(num_of_local_experts <= block_size);  // line 265
```

### The two sizes

| Value                | What it is                                                                                            | How it's set                                             |
| -------------------- | ----------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| `block_size`           | **Hardcoded to 256** — a compile-time CUDA constant (threads per block). It cannot be changed at runtime. | `constexpr int block_size = 256;` in `permute.cu:262`        |
| `num_of_local_experts` | Number of MoE experts **on this rank**                                                                    | Computed as `num_moe_experts / expert_model_parallel_size` |

The kernel requires `num_of_local_experts <= 256` because it uses shared memory tiles of shape `[block_size, num_of_local_experts]` and a block-level prefix-sum over `num_of_local_experts` elements — both of which require that value to fit within a single CUDA block of 256 threads.

### How to fix it in Megatron

The formula is:

```
num_local_experts = --num-experts / --expert-model-parallel-size
```

**You must ensure `num_local_experts <= 256`.**

The two relevant Megatron CLI arguments are:

| CLI Argument                   | Config Field                                   | File                                          |
| ------------------------------ | ---------------------------------------------- | --------------------------------------------- |
| `--num-experts N`                | `TransformerConfig.num_moe_experts`              | `MLM/megatron/training/arguments.py:2724`       |
| `--expert-model-parallel-size E` | `ModelParallelConfig.expert_model_parallel_size` | `MLM/megatron/core/model_parallel_config.py:75` |

**Example**: If you have `--num-experts 512`, you need at least `--expert-model-parallel-size 2` so that `512 / 2 = 256` experts per rank (the maximum). With `--num-experts 1024`, you'd need at least `--expert-model-parallel-size 4`, and so on.

### Quick reference

```
--num-experts 64  --expert-model-parallel-size 1  => 64 local experts   (OK)
--num-experts 256 --expert-model-parallel-size 1  => 256 local experts  (OK, at limit)
--num-experts 512 --expert-model-parallel-size 1  => 512 local experts  (FAILS)
--num-experts 512 --expert-model-parallel-size 2  => 256 local experts  (OK)
--num-experts 512 --expert-model-parallel-size 4  => 128 local experts  (OK)
```

There is also a secondary constraint in the DeepEP config validation (`config.cuh:37`):

```cpp
valid &= ((num_of_experts_per_rank * num_of_ranks_per_node) % 4 == 0);
```

So `num_local_experts * GPUs_per_node` must be divisible by 4.

### If you truly need >256 local experts

The `block_size = 256` is hardcoded in DeepEP's CUDA source. You would need to modify `permute.cu:262` to a larger power-of-two (e.g., 512 or 1024), but this increases shared memory usage (`block_size * num_of_local_experts * sizeof(int)`) and may exceed the per-SM shared memory limit or reduce occupancy. This is not a trivial change.

---

