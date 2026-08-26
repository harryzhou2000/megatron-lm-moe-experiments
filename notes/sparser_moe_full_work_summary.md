# Sparser MoE: keeping sparsity end to end

## Executive summary

The Sparser MoE work started with a 2,304-expert profile and ended as a cross-stack
optimization of the router, MCore, and HybridEP. With many experts, large top-k, high
expert parallelism, and a small latent dimension, the expert GEMMs are relatively cheap.
Costs that conventional MoE models can hide—selection, route metadata, NVLink transfers,
synchronization, and pre/post-processing—become the bottleneck.

The work fell into six connected parts:

1. Fix the large-expert, large-top-k router algorithm in Transformer Engine.
2. Preserve compact selected-expert metadata through TE, MCore, and HybridEP.
3. Stop communicating probability columns and visiting local expert slots that a token
   did not select.
4. Tune the pull-and-reduce combine pipeline for the actual latent-MoE shape.
5. Use profiling to reject fusion and direct-write designs that merely moved or amplified
   the work.
6. Validate the complete stack with matched full-model runs, broader MoE-module coverage,
   and real distributed routing tests.

The original target progressed historically from roughly 92 TFLOP/s/GPU to 158, then
310, and later about 400 as the bottleneck moved across the stack. Those snapshots came
from an evolving training stack and are not a controlled 92-to-400 ablation. The cleaner
matched full-model results compare the complete sparse stack with a no-tune/no-radix
baseline under the same paged-stash, full-iteration CUDA graph, and expert-1F1B recipe:

| System | Shape | Baseline | Full sparse stack | Gain |
| --- | --- | ---: | ---: | ---: |
| GB200 | 2,304 experts / EP72 | 125.3 | 403.7 TFLOP/s/GPU | 3.22x |
| GB200 | 2,048 experts / EP64 | 159.4 | 421.0 TFLOP/s/GPU | 2.64x |
| GB300 | 2,304 experts / EP72 | 127.2 | 419.7 TFLOP/s/GPU | 3.30x |
| GB300 | 2,048 experts / EP64 | 162.2 | 435.1 TFLOP/s/GPU | 2.68x |

The matched results came from preserving sparsity end to end. Sparse selection alone is
not enough if later stages rebuild expert-wide tensors, transfer irrelevant columns, or
loop over inactive experts.

## Project timeline

| Period | Milestone |
| --- | --- |
| March | Profiled the original 2,304E/top-36/EP72 run, identified router and expert-parallel overhead, and developed the first radix router path. |
| April | Merged the initial router optimization, investigated full-iteration CUDA graph + expert-1F1B memory behavior, and shifted the critical path toward HybridEP. |
| April-May | Implemented probability slicing, compact route handling, ballot permute/unpermute, combine tuning, and NCU-guided negative experiments. |
| May-June | Advanced the historical target from the post-router ~158 result through the ~310 HybridEP stack and later ~400 dense-scan snapshot. |
| June | Merged the deeper TE router and selected-index API PRs and established matched Qwen full-model comparisons. |
| July | Ported the 2,304E/2,048E benchmark recipes, ran the broader MoE matrix, and consolidated the technical presentation/evidence package. |
| August | Merged the DeepEP dense top-k scan and extracted the paired MCore dense-route PRs with focused and eight-rank validation. |

## 1. Target workload and why it changes the bottleneck

The investigation began with a Qwen3-Next-style Sparser MoE configuration for a GB200
NVL72 system:

| Dimension | Target value | Performance consequence |
| --- | ---: | --- |
| Total experts | 2,304 | Router rows and expert-wide metadata become very wide |
| Router top-k | 36 | Repeated selection and membership work is no longer “small-k” |
| Expert parallelism | 72 | Routing information and token payloads cross a large NVLink domain |
| Local experts/rank | 32 | Most local expert slots are inactive for each token |
| Latent/expert hidden size | 512 | Expert GEMMs and copies are small and latency-sensitive |
| Micro/global batch | 2 / 576 | Original full-model EP72 setup |

At the representative eight-rank microbenchmark shape, top-36 routing activates about
`36 / 8 = 4.5` experts per rank on average. A kernel that checks all 32 local slots is
therefore spending roughly 86% of those slot checks on inactive entries.

The first full profile made the imbalance visible:

| Kernel group | Captured GPU time |
| --- | ---: |
| Fused router | 3947.499 ms |
| HybridEP combine | 1713.859 ms |
| HybridEP dispatch | 1259.542 ms |
| FlashAttention | 903.899 ms |
| Other GEMM | 371.136 ms |
| Grouped GEMM (expert MLP) | 286.147 ms |

The expert MLP was not the dominant cost. Router and expert-parallel work together consumed
far more captured GPU time, so optimizing only GEMM would have missed the critical path.

![HybridEP workflow](https://github.com/harryzhou2000/megatron-lm-moe-experiments/blob/945e8fe0091b315e7328111fff9877cab4eb65df/notes/final_presentation/assets/hybrid_ep_workflow.svg)

The end-to-end ownership is split across three repositories:

- **Transformer Engine (TE):** score transform, top-k selection, route outputs, and router
  backward.
- **Megatron Core (MCore):** decides whether a compact route is legal, expands it for
  expert tensor parallelism, manages padding/recompute/expert bias, and selects the Flex
  dispatcher backend.
- **DeepEP HybridEP:** consumes route metadata, scans destinations, gathers metadata,
  dispatches and combines tokens/probabilities, and permutes data into expert order.

An optimization only paid off when its representation and correctness invariants survived
all three layers.

## 2. Phase one: fix the router algorithm

### 2.1 Failure mode of repeated max selection

The original large-`E`, large-`K` fused path repeatedly scanned the expert dimension to
select one maximum at a time. Conceptually:

```text
for selected_slot in 0..K-1:
    scan all E experts
    choose the next maximum
    mark it selected
```

The selection work scales approximately with `T * E * K`. At `E=2304` and `K=36`, that
repeated scan was large enough to dominate the profile.

The first fix replaced it with radix selection. A radix pass builds a small digit histogram,
finds the bucket containing the kth threshold, and narrows the candidate prefix. The work
tracks the expert row plus a bounded number of histogram passes rather than rescanning the
full row once per selected expert.

That work became https://.
For the target shape, the forward router kernels improved by more than 10x, and the
historical full-model result moved from about 92 to about 158 TFLOP/s/GPU. The same router
fix was useful outside this experiment, including large-expert Nemotron work.

### 2.2 Turn the initial fix into a robust kernel family

The next router phase addressed forward and backward together. The main changes were:

- fuse preprocessing and backward loops so the kernels traverse route data fewer times;
- remove large shared-memory backward temporaries and use register reductions;
- load score rows asynchronously with `cp.async` and choose one or two shared-memory
  buffers according to the shape;
- size persistent grids to reuse blocks across token rows;
- pack radix histogram state to reduce register pressure;
- specialize score functions at compile time instead of branching dynamically;
- retain the simple kernel for small top-k so the large-shape fix does not regress common
  cases.

The final B300 effective-bandwidth results from
https:// show why this needed to
cover backward rather than only selection:

| Kernel | Shape | Before | After | Improvement |
| --- | --- | ---: | ---: | ---: |
| Top-k forward, softmax | E=2304, K=36 | 673 GB/s | 964 GB/s | +43% |
| Top-k backward | E=2304, K=36 | 543 GB/s | 2766 GB/s | +410% |
| Aux-loss forward | E=2304, K=36 | 645 GB/s | 891 GB/s | +38% |
| Aux-loss backward | E=2304, K=36 | 2272 GB/s | 4201 GB/s | +85% |
| Top-k forward, softmax | E=512, K=4 | 1779 GB/s | 1784 GB/s | +0.3% |

The small-k row shows that the dispatch policy retained parity instead of forcing radix
selection everywhere. The complete local benchmark
and implementation record is in [TE fused-router optimization](https://github.com/harryzhou2000/megatron-lm-moe-experiments/blob/945e8fe0091b315e7328111fff9877cab4eb65df/notes/te_fused_router_optimization.md)
and [final router results](https://github.com/harryzhou2000/megatron-lm-moe-experiments/blob/945e8fe0091b315e7328111fff9877cab4eb65df/notes/te_fused_router_p3R_results.md).

## 3. Phase two: make the routing representation sparse

### 3.1 Bool `[T,E]` versus selected indices `[T,K]`

The router originally exposed a boolean routing map with one column per expert even though
only `K` entries per token were selected. For `T=8192`, `E=2304`, and `K=36`:

```text
bool route map = T * E * 1 byte
               = 18,874,368 bytes = 18.87 MB

int16 selected indices = T * K * 2 bytes
                       = 589,824 bytes = 0.59 MB
```

The compact representation is a **32x logical metadata reduction** at the target shape.
More importantly, it changes the algorithmic interface: downstream code can traverse the
selected expert IDs directly rather than search an expert-wide bitmap.

https:// added optional dense
`topk_indices` output and matching backward support. “Dense” here means a dense list of
selected IDs shaped `[T,K]`; it is compact compared with the sparse boolean `[T,E]` map.

### 3.2 Correctness invariants through MCore

Compact routing is not a dtype-only substitution. The MCore integration has to preserve:

- `-1` for padded or invalid route slots;
- expert-TP expansion in the same expert-ID domain used by the dispatcher;
- expert-bias token counts without reconstructing a boolean map;
- padding semantics before and after dynamic token equalization;
- recompute behavior that updates routing statistics exactly once;
- backend capability and expert-ID range checks;
- fallback to the existing bool route whenever capacity/drop modes, TE version, backend,
  or shape do not support selected indices.

HybridEP uses int16 selected IDs only when the expanded expert range is representable;
other dense-route backends use int64. A July rebase audit caught an important boundary bug:
the int16 range calculation initially used the attention TP group, while Flex expands over
the **expert** TP group. Using expert TP in the gate prevents dense routing from being
selected above the 32,768-ID limit.

The final focused MCore work is in
https:// for `dev` and
https:// for `main`. At the 2026/08/26
snapshot both are open and mergeable. Focused suites passed 27/27 on both branches; real
eight-rank tests passed the bool route, dense-index route, and dense expert-bias paths on
every worker.

### 3.3 Dense metadata is independent of the collective

Two separate choices are involved:

1. **Representation:** bool `[T,E]` versus selected indices `[T,K]`.
2. **Transport:** custom TMA all-gather versus NCCL/fallback all-gather.

The representation controls how much metadata exists and how scan consumes it. The
collective controls how ranks exchange that metadata. Either representation can use a
supported collective path; enabling compact routes must not silently force the custom
all-gather.

## 4. Phase three: remove irrelevant HybridEP work

### 4.1 Transfer only destination-local probabilities

Training cannot remove route probabilities: selected weights are needed for the expert
activation and for probability gradients in backward. The avoidable part was transferring
the entire expert-probability row to every destination.

For one destination rank:

```text
old payload: all E_local * R probabilities
new payload: only that destination's E_local probabilities
```

At the EP72 target, `E_local=32`:

```text
full FP32 row = 32 * 72 * 4 bytes = 9,216 bytes
local slice   = 32 * 4 bytes      =   128 bytes
BF16 H=512 token                     1,024 bytes
```

The old probability row was nine times larger than the token itself. Destination slicing
reduces probability bytes per peer by up to 72x while retaining all selected values needed
for training.

The controlled `H=512`, `E_local=32`, `K=36` result isolated the effect:

| Dispatch path | Before | After | Result |
| --- | ---: | ---: | ---: |
| With probabilities | 185.1 us | 103.4 us | 1.79x |
| Without probabilities | 95.8 us | 95.7 us | parity |

The no-probability control remained unchanged, which ties the gain to the removed
probability traffic rather than an unrelated launch change.

### 4.2 Visit active experts with a warp ballot

The reference permute/unpermute preprocessing assigned 128 threads per token and tested all
32 local expert slots. The optimized `H=512` path uses a 32-thread warp:

```cpp
bool active = route_slot[lane] != EMPTY;
uint32_t mask = __ballot_sync(FULL_MASK, active);

while (mask != 0) {
  int expert = __ffs(mask) - 1;
  mask &= mask - 1;
  int token = __shfl_sync(FULL_MASK, my_token, expert);
  copy_or_accumulate(token, expert);
}
```

Only set bits are visited. The result is largest for the small latent dimension, where
control overhead was a large fraction of the kernel:

| Kernel | Reference | Ballot path | Speedup |
| --- | ---: | ---: | ---: |
| Permute, H=512 | 381 us | 92 us | 4.1x |
| Unpermute, H=512 | 265 us | 128 us | 2.1x |
| Permute, H=7168 | 937 us | 912 us | parity |
| Unpermute, H=7168 | 1475 us | 1098 us | 1.34x |

The production path kept permute and unpermute as standalone kernels. An independently
upstreamed ballot traversal in https://
used the same approach.

### 4.3 Tune combine as a pipeline, not a single knob

Dispatch and combine are not symmetric:

```text
dispatch: push / S2G
  issue remote writes and continue

combine: pull / G2S + reduce
  request remote data
  wait for arrival
  synchronize producer and reduction warps
  reduce in FP32 and store
```

The representative NCU data makes the asymmetry concrete:

| Metric | Dispatch | Combine |
| --- | ---: | ---: |
| Duration | 116.83 us | 254.43 us |
| Dominant traffic | 78.01 MB TX | 73.90 MB RX |
| Throughput | 650 GB/s | 271 GB/s |
| Long-scoreboard stall | 57.95% | 26.59% |
| Wait stall | 21.36% | 25.92% |
| Barrier stall | 0.05% | 15.23% |

Combine needs enough G2S FIFO depth to cover the remote-read round trip, enough S2G stages
to drain reduced outputs, and enough warp groups to use the block without dividing the FIFO
too finely. The selected controlled NVL8 tuple was:

| Parameter | Value |
| --- | ---: |
| Combine SMs | 32 |
| G2S stages | 64 |
| S2G stages | 8 |
| Tokens/group | 2 |
| Reduction batch | 16 |

The historical EP72 launch used G2S 72, S2G 8, group 2, and 32 combine SMs. These are
workload-specific results, not universal defaults.

Why the tuple worked:

- group 2 kept both data pipelines active while preserving 32 G2S slots per pipeline;
- those slots provide roughly four tokens of lookahead for an eight-source reduction;
- group 4 cut the per-pipeline FIFO too far and fell to about 133 GB/s;
- doubling G2S depth from 64 to 128 was effectively flat, showing that latency was already
  covered;
- more SMs alone did not help once shared-memory depth and warp utilization were balanced.

Controlled default-to-tuned results:

| Measurement | Default | Tuned | Improvement |
| --- | ---: | ---: | ---: |
| Combine with probabilities | 960.8 us | 252.3 us | 3.8x |
| Combine without probabilities | 877.4 us | 210.9 us | 4.2x |
| Combine + unpermute API | 1125.5 us | 399.2 us | 2.8x |

With both sparse preprocessing and combine tuning, dispatch+permute plus
combine+unpermute fell from 1133.7 to 617.1 us, a 1.84x total improvement in the
controlled study. Full details are in
[HybridEP sparse optimization](https://github.com/harryzhou2000/megatron-lm-moe-experiments/blob/945e8fe0091b315e7328111fff9877cab4eb65df/notes/hybrid-ep-sparse-opt-new.md) and
[rebased bandwidth results](https://github.com/harryzhou2000/megatron-lm-moe-experiments/blob/945e8fe0091b315e7328111fff9877cab4eb65df/notes/hybrid_ep_rebased_bw_results.md).

### 4.4 Replace dense-scan membership loops with bitsets

Compact `[T,K]` metadata still has to be scanned into destination-rank and local-expert
counts. The optimized path derives both from the selected IDs:

```text
for expert_id in topk_indices[token]:
    owner = owner_rank(expert_id)
    rank_mask[word(owner)] |= bit(owner)
    if owner == my_rank:
        local_expert_mask |= bit(local_index(expert_id))
```

Later membership tests are register bit operations rather than nested comparisons. The
local-expert bitset improved tested fused-scan templates by 2.37x-2.98x; the rank bitset
improved the no-permute `512/108` scan from 525.6 to 331.7 us (1.58x).

The dense top-k scan path merged as
https://. Distributed BF16 and FP8
correctness passed for dispatch/combine, selected-index routing, standalone and fused
pre/post-processing. See [dense scan benchmarks](https://github.com/harryzhou2000/megatron-lm-moe-experiments/blob/945e8fe0091b315e7328111fff9877cab4eb65df/notes/isolated-scan-bench.md) and
[HybridEP dense-routing tests](https://github.com/harryzhou2000/megatron-lm-moe-experiments/blob/945e8fe0091b315e7328111fff9877cab4eb65df/notes/hybrid_ep_dense_routing_pr_test_results.md).

## 5. Runtime interaction: CUDA graphs, 1F1B, and skew

Kernel improvements did not always carry over unchanged to full training overlap. Two
runtime effects explained the difference.

### 5.1 CUDA-graph memory inflation from cross-stream lifetime tracking

Combining expert 1F1B with full-iteration CUDA graph capture initially grew memory from
about 108-110 GiB in eager 1F1B to 190-265+ GiB. The diagnosis traced this to
`record_stream()` and the CUDA caching allocator:

- eager execution can query completion events and reclaim deferred blocks;
- graph capture cannot query those events, so blocks with cross-stream uses remain deferred
  inside the graph-private pool;
- repeated layers and microbatches accumulate many large deferred blocks.

The schedule's CUDA events correctly order GPU work, but that ordering does not by itself
tell the eager CPU-side allocator when cross-stream readers are finished. Unconditionally
removing `record_stream()` therefore caused gradient mismatches in eager tests. The safe
lesson was to distinguish eager allocator ownership from graph-capture lifetime rather than
apply a global removal.

An early deferred-release approach brought memory back to about 107.8 GiB. Later review
showed that a custom allocator patch was unnecessary because the supported PyTorch
mechanism addresses the practical case. The root-cause analysis and correctness boundary
are documented in
[`record_stream()` investigation](https://github.com/harryzhou2000/megatron-lm-moe-experiments/blob/945e8fe0091b315e7328111fff9877cab4eb65df/notes/record_stream_removal.md).

### 5.2 Full-iteration graphs exposed, but did not create, cross-rank skew

The NVL72 investigation found that explicit pre-dispatch/pre-combine synchronization in
the no-graph path could dominate measured kernel regions. Full-iteration CUDA graphs
removed much of the CPU launch/synchronization noise, but expert 1F1B still exposed
cross-rank skew when compute and communication contested SMs.

Switching the grouped-expert GEMM path changed that contention enough to mitigate the
observed skew. This is another system-level lesson: a faster compute kernel can improve
communication overlap even when it does not touch the communication code. The complete
analysis is in [NVL72 graph-vs-eager report](https://github.com/harryzhou2000/megatron-lm-moe-experiments/blob/945e8fe0091b315e7328111fff9877cab4eb65df/reports/nvl72_cg_vs_nocg_report.md).

## 6. Negative results that narrowed the design space

### 6.1 Direct dispatch into expert-contiguous output

Direct-permute attempted to eliminate the local permute by writing each selected expert
route directly into the final expert-contiguous output. For small `H` and large `K`, it
eliminates one local kernel but multiplies remote writes:

| Test | Direct path versus staged dispatch + permute |
| --- | --- |
| K=4 | 1.36x slower, 3.2x more NVLink TX, 49x more L1 sectors |
| K=8 | 2.2x slower, 4.0x more NVLink TX, 97x more L1 sectors |
| Target K=36 | projected ~4.5x NVLink amplification and ~4-5x slowdown |

The staged design writes one token per destination rank, then performs a local optimized
permute. Direct-permute writes once per selected expert, producing scattered sectors and an
L2-capacity cliff. It may still be interesting for much smaller `K` or larger `H`, but it is
the wrong design for H=512/top-36. See [direct-permute NCU analysis](https://github.com/harryzhou2000/megatron-lm-moe-experiments/blob/945e8fe0091b315e7328111fff9877cab4eb65df/notes/direct_permute_ncu_analysis.md).

### 6.2 Fused dispatch/permute and combine/unpermute

For the target latent shape, standalone preprocessing was already fast enough that fusion
introduced more polling and SM-allocation constraints than it removed. Fused
combine+unpermute was about 434 us versus 381 us standalone—roughly 14% slower—because
unpermute consumers serialized behind slowly produced pull/reduce chunks.

A tuned fused dispatch+permute was about 2% faster in one configuration, but the gain was
fragile and added more JIT variants and tuning surface. The standalone path was the better
production choice.

### 6.3 Warp-pruned dense scan

Skipping entire expert loops appeared attractive, but warp reductions, set-bit iteration,
and row initialization cost more than the skipped work in several shapes. The simpler rank
bitset was kept because it gave a robust constant-time membership test without another
control hierarchy.

The profiling established a clear boundary for future work: removing a kernel helps only
when the replacement also reduces total data movement and synchronization.

## 7. Performance progression and generalization

### 7.1 Historical target timeline

| Period / stack snapshot | Approx. TFLOP/s/GPU | Main change |
| Initial profile | ~92 | Router and expert-parallel overhead exposed |
| Initial TE router fix | ~158 | Radix large-E/large-K router path |
| Refined HybridEP stack | ~310 | Probability slicing, compact routing, ballot preprocessing, combine tuning |
| Dense scan/bitset stack | ~400 | Compact route plumbing plus local-expert/rank bitsets |

These are historical snapshots, not an additive waterfall. Grouped GEMM, MXFP8, paged
stash, CUDA graph, and 1F1B also changed during the same period.

### 7.2 Matched staged full-model results

The later runs used the same five stages on both GPU generations:

| System | Shape | No tune / no radix | Tuned HybridEP | Router #2821 | Router #3012 | Full sparse stack |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GB200 | 2304E / EP72 | 125.3 | 134.7 | 214.9 | 225.7 | 403.7 |
| GB200 | 2048E / EP64 | 159.4 | 175.5 | 244.5 | 255.2 | 421.0 |
| GB300 | 2304E / EP72 | 127.2 | 137.1 | 219.9 | 230.8 | 419.7 |
| GB300 | 2048E / EP64 | 162.2 | 178.3 | 249.9 | 261.4 | 435.1 |

The router stages matter, but the largest final step comes from preserving the compact
sparse representation through MCore and HybridEP rather than optimizing TE in isolation.

### 7.3 Established Qwen recipe comparisons

Matched median TFLOP/s/GPU gains over the June base stack were largest on the most
routing-intensive models:

| Model | Baseline recipe | Paged stash | Paged stash + 1F1B |
| --- | ---: | ---: | ---: |
| Qwen3.5 Sparser 40B, EP72 | +7.3% | +28.4% | +29.7% |
| Qwen3 Sparser 80B, EP64 | +33.2% | +53.8% | +53.1% |
| Qwen3.5 397B EP64 proxy | +2.6% | +5.3% | +4.9% |
| Qwen3.5 397B sparser EP64 proxy | +8.1% | +10.9% | +9.7% |

![Qwen3.5 Sparser 40B EP72](https://github.com/harryzhou2000/megatron-lm-moe-experiments/blob/945e8fe0091b315e7328111fff9877cab4eb65df/notes/final_presentation/assets/qwen3_5_sparser_40b_ep72.png)

![Qwen3 Sparser 80B EP64](https://github.com/harryzhou2000/megatron-lm-moe-experiments/blob/945e8fe0091b315e7328111fff9877cab4eb65df/notes/final_presentation/assets/qwen3_sparser_80b_ep64.png)

![Qwen3.5 397B EP64 proxy](https://github.com/harryzhou2000/megatron-lm-moe-experiments/blob/945e8fe0091b315e7328111fff9877cab4eb65df/notes/final_presentation/assets/qwen3_5_397b_ep64_proxy.png)

![Qwen3.5 397B sparser EP64 proxy](https://github.com/harryzhou2000/megatron-lm-moe-experiments/blob/945e8fe0091b315e7328111fff9877cab4eb65df/notes/final_presentation/assets/qwen3_5_397b_sparser_ep64_proxy.png)

### 7.4 Broader MoE-module matrix

The July matrix ran 160/160 MoE performance iterations per row with paged stash and a
full-iteration CUDA graph, without expert 1F1B. It is intentionally separate from the
full-model results:

| Shape | Base | Optimized | Gain |
| --- | ---: | ---: | ---: |
| NT3 Super estimate | 511.8 | 553.65 | +8.2% |
| Nemotron Ultra proxy | 1072.0 | 1123.8 | +4.8% |
| Qwen397 EP64 | 1014.4 | 1198.3 | +18.1% |
| Qwen397 Sparser EP64 | 826.2 | 1041.5 | +26.1% |
| Qwen40 Sparser EP72 | 234.5 | 484.5 | +106.6% |

![MoE-module performance matrix](https://github.com/harryzhou2000/megatron-lm-moe-experiments/blob/945e8fe0091b315e7328111fff9877cab4eb65df/notes/final_presentation/assets/moe_module_perf_20260719.png)

The matrix validates the expected trend: conventional 512-expert shapes still benefit,
but the gain grows sharply when expert count, top-k, and small latent payloads make route
and communication overhead a larger fraction of the step.

## 8. Upstream and integration status

Status snapshot: 2026/08/26.

| Repository / PR | Contribution | Status |
| --- | --- | --- |
| https:// | Radix top-k foundation and large-shape router fix | Merged |
| https:// | Async/persistent forward and fused backward optimization | Merged |
| https:// | Optional selected-index output and backward support | Merged |
| https:// | Upstream standalone permute ballot traversal | Merged |
| https:// | Dense top-k routing scan | Merged |
| https:// | Dense Flex routing on `dev` | Open, mergeable |
| https:// | Dense Flex routing on `main` | Open, mergeable |

The TE PRs were already merged by the 2026/06/29 T5T. The post-June status movement was
the DeepEP scan merge and the extraction/validation of the paired MCore integration PRs.

Relevant source entry points:

| Layer | Source |
| --- | --- |
| TE radix helpers |  |
| TE async loader |  |
| TE route launch/dense output |  |
| TE PyTorch route API |  |
| MCore capability gate |  |
| MCore HybridEP forwarding |  |
| DeepEP permute/unpermute |  |
| DeepEP custom all-gather |  |
| DeepEP route scan |  |
| DeepEP combine configuration |  |

## 9. Validation record

Validation covered the individual kernels, distributed integration, and full-model runs:

- **TE router correctness/performance:** all 891 existing fused-router tests passed in the
  final P3 record, with 117 expected skips for unsupported FP8/multi-node cases.
- **DeepEP correctness:** BF16 and FP8 dispatch/combine, selected-index routing,
  standalone pre/post-processing, and fused variants passed on the controlled eight-rank
  topology.
- **MCore focused tests:** 27/27 passed on both `dev` and `main` extracted PR branches.
- **MCore distributed integration:** eight-rank bool/dense HybridEP paths and dense
  expert-bias accumulation passed on every worker.
- **Full-model performance:** staged 2,304E/2,048E runs completed on both GB200 and GB300.
- **Broader MoE coverage:** the July module matrix completed all 160 measured iterations per
  row and preserved the performance trend across multiple architectures.

Each level supports a different claim. A one-rank fixture cannot validate EP72 transport;
a router microbenchmark cannot establish a full-model speedup; and a full-model result
cannot attribute the gain to one kernel without the lower-level measurements.

## 10. What is reusable

### Algorithmic lessons

- Replace repeated `O(E*K)` selection with a threshold/radix algorithm once `E` and `K`
  leave the small regime.
- Carry selected indices through the API instead of regenerating expert-wide state.
- Use compact bitsets for small fixed membership domains such as local experts or ranks.

### Communication lessons

- Preserve only the probability slice required by the receiving rank, but retain selected
  probabilities and their gradients.
- Treat push dispatch and pull/reduce combine as different pipelines.
- Tune SM count, FIFO depth, groups, and reduction batch together; each changes the resource
  budget available to the others.

### Profiling lessons

- Report effective payload bandwidth separately from hardware-counter bandwidth.
- Use probability-disabled paths as upper-bound controls, not as training-equivalent
  performance numbers.
- A removed kernel can increase total work through write amplification or serialization.
- Separate historical stack snapshots, controlled microbenchmarks, MoE-module tests, and
  matched full-model comparisons.

### Integration lessons

- Feature-gate selected indices by TE version, dispatcher backend, expert-ID range,
  capacity/drop semantics, and padding mode.
- Preserve invalid `-1` routes through TP expansion and token equalization.
- Keep metadata representation and collective choice as independent controls.
- Validate real distributed topology before classifying a failure as a transport defect.

## 11. Remaining opportunities

The main bottlenecks for this workload are addressed. Remaining work includes:

- merge and release the MCore dense-route integration so users do not need a feature branch;
- test more expert-count/top-k boundaries and mixed expert-TP layouts;
- investigate a deeper combine redesign that reduces remote-read round trips or barrier cost,
  rather than only increasing queue depth;
- revisit direct placement only if a design avoids per-expert remote-write amplification;
- use the canonical 2,304E/2,048E matrix as regression coverage for future TE, MCore,
  DeepEP, grouped-GEMM, and CUDA-graph changes;
- keep full-model attribution staged so later runtime or kernel updates can be separated
  cleanly from the compact-routing gains.

## 12. Evidence map

Supporting records:

- [Final presentation content and claims map](https://github.com/harryzhou2000/megatron-lm-moe-experiments/blob/945e8fe0091b315e7328111fff9877cab4eb65df/notes/final_presentation/sparser_moe_final_presentation_content.md)
- [TE fused-router optimization](https://github.com/harryzhou2000/megatron-lm-moe-experiments/blob/945e8fe0091b315e7328111fff9877cab4eb65df/notes/te_fused_router_optimization.md)
- [TE final router benchmark](https://github.com/harryzhou2000/megatron-lm-moe-experiments/blob/945e8fe0091b315e7328111fff9877cab4eb65df/notes/te_fused_router_p3R_results.md)
- [HybridEP sparse optimization](https://github.com/harryzhou2000/megatron-lm-moe-experiments/blob/945e8fe0091b315e7328111fff9877cab4eb65df/notes/hybrid-ep-sparse-opt-new.md)
- [HybridEP rebased results](https://github.com/harryzhou2000/megatron-lm-moe-experiments/blob/945e8fe0091b315e7328111fff9877cab4eb65df/notes/hybrid_ep_rebased_bw_results.md)
- [Dense scan benchmarks](https://github.com/harryzhou2000/megatron-lm-moe-experiments/blob/945e8fe0091b315e7328111fff9877cab4eb65df/notes/isolated-scan-bench.md)
- [Direct-permute NCU analysis](https://github.com/harryzhou2000/megatron-lm-moe-experiments/blob/945e8fe0091b315e7328111fff9877cab4eb65df/notes/direct_permute_ncu_analysis.md)
- [Full-iteration NVL72 analysis](https://github.com/harryzhou2000/megatron-lm-moe-experiments/blob/945e8fe0091b315e7328111fff9877cab4eb65df/reports/nvl72_cg_vs_nocg_report.md)
- [`record_stream()` / CUDA graph investigation](https://github.com/harryzhou2000/megatron-lm-moe-experiments/blob/945e8fe0091b315e7328111fff9877cab4eb65df/notes/record_stream_removal.md)
- [June 8 T5T](https://github.com/harryzhou2000/megatron-lm-moe-experiments/blob/945e8fe0091b315e7328111fff9877cab4eb65df/notes/top5_things_2026_06_08.md)
- [June 29 T5T](https://github.com/harryzhou2000/megatron-lm-moe-experiments/blob/945e8fe0091b315e7328111fff9877cab4eb65df/notes/top5_things_2026_06_29.md)
- [Confluence T5T transcript](https://github.com/harryzhou2000/megatron-lm-moe-experiments/blob/945e8fe0091b315e7328111fff9877cab4eb65df/reports/top5_confluence_transcript.md)
- [Published evidence notebook](https://github.com/harryzhou2000/megatron-lm-moe-experiments/blob/945e8fe0091b315e7328111fff9877cab4eb65df/notes/top5_since_2026_06_29/evidence.md), containing the
  sanitized July MoE-module and canonical-matrix measurements used above

Related shared documents:

- https://
- https://

The measurements retain the scope used in their source records. Future results should
update the matched table and provenance without rewriting the historical snapshots as if
they came from one fixed stack.
