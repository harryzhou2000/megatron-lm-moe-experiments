# CuTe DSL API Overview

> **Full API reference**: https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api.html

This document provides a navigational overview of the CuTe DSL API modules and their most important functions. For complete signatures, parameter details, and advanced usage, consult the official documentation linked above.

## API Module Structure

```
cutlass.cute          # Core CuTe DSL: layouts, tensors, copy, MMA, printf
  cute.arch           # Architecture intrinsics (threadIdx, blockIdx, syncthreads, etc.)
  cute.runtime        # Host-side runtime: from_dlpack, make_ptr
cutlass.cute.nvgpu    # GPU-specific MMA and Copy operations
  nvgpu.common        # Arch-agnostic operations
  nvgpu.warp          # Warp-level operations (SM80+)
  nvgpu.warpgroup     # Warpgroup-level operations (SM90+)
  nvgpu.cpasync       # cp.async operations (SM80+)
  nvgpu.tcgen05       # Blackwell tcgen05 MMA/copy (SM100)
cutlass.pipeline      # Synchronization primitives: pipelines, barriers
cutlass.utils         # Utilities: SmemAllocator, TmemAllocator, tile schedulers, HardwareInfo
  utils.sm90          # SM90-specific utilities
  utils.sm100         # SM100-specific utilities
```

## `cutlass.cute` -- Core Module

### Layout Construction

| Function | Purpose |
|---|---|
| `cute.make_layout(shape, stride=None)` | Create layout from shape + optional stride. Default: compact column-major. |
| `cute.make_ordered_layout(shape, order)` | Create layout with dimension ordering. `order=(1,0)` = row-major. |
| `cute.make_identity_layout(shape)` | Identity layout: coord maps to itself. Stride = `(1@0, 1@1, ...)`. |
| `cute.make_layout_like(input)` | Create layout with same shape/stride as input layout or tensor. |
| `cute.make_composed_layout(inner, offset, outer)` | Compose inner transform (Swizzle/Layout) with offset and outer layout. |
| `cute.make_swizzle(b, m, s)` | Create a Swizzle with BBits, MBase, SShift parameters. |

### Layout Queries

| Function | Purpose |
|---|---|
| `cute.size(a, mode=[])` | Total number of elements in domain. With `mode`, size of specific mode. |
| `cute.cosize(a, mode=[])` | Size of codomain (min storage needed). |
| `cute.rank(a, mode=[])` | Dimensionality (number of modes). |
| `cute.depth(a)` | Nesting level. `depth(1)=0`, `depth((1,2))=1`, `depth(((1,2),(3,4)))=2`. |
| `cute.size_in_bytes(dtype, layout)` | Byte size for dtype + layout, rounded up. |

### Layout Algebra

| Function | Purpose |
|---|---|
| `cute.composition(a, b)` | Compose layout A with layout/function B. |
| `cute.complement(a, cosize)` | Compute the complement layout. |
| `cute.right_inverse(layout)` | Right inverse of a layout. |
| `cute.left_inverse(layout)` | Left inverse of a layout. |
| `cute.logical_divide(layout, tiler)` | Divide layout into tiles. |
| `cute.zipped_divide(layout, tiler)` | Zipped tile division. |
| `cute.tiled_divide(layout, tiler)` | Tiled division. |
| `cute.logical_product(a, b)` | Logical product of layouts. |
| `cute.zipped_product(a, b)` | Zipped product. |
| `cute.tiled_product(a, b)` | Tiled product. |
| `cute.blocked_product(a, b)` | Blocked product. |
| `cute.raked_product(a, b)` | Raked product. |
| `cute.coalesce(input, target_profile=None)` | Merge adjacent modes with contiguous strides. |
| `cute.flatten(a)` | Flatten hierarchical layout/tuple to single level. |
| `cute.filter(input)` | Filter a layout according to CuTe rules. |
| `cute.filter_zeros(input)` | Remove zero-stride dimensions. |
| `cute.ceil_div(input, tiler)` | Number of tiles needed to cover input. |
| `cute.shape_div(lhs, rhs)` | Element-wise shape division. |

### Coordinate and Index Operations

| Function | Purpose |
|---|---|
| `cute.crd2idx(coord, layout)` | Convert multi-dim coordinate to linear index via layout. |
| `cute.idx2crd(idx, layout)` | Convert linear index to multi-dim coordinate. |
| `cute.slice_(src, coord)` | Slice tensor/layout: `None` keeps all elements in that mode. |
| `cute.get(input, mode)` | Extract sub-layout at specific mode path. |
| `cute.select(input, mode)` | Select specific modes from layout. |
| `cute.group_modes(input, begin, end)` | Group a range of modes into a single hierarchical mode. |

### Tuple Manipulation

| Function | Purpose |
|---|---|
| `cute.flatten(a)` | Flatten nested tuple to single level. |
| `cute.prepend(input, elem, up_to_rank=None)` | Prepend element(s) to reach target rank. |
| `cute.append(input, elem, up_to_rank=None)` | Append element(s) to reach target rank. |
| `cute.repeat(x, n)` | Repeat value n times (returns x if n=1, else tuple). |
| `cute.repeat_like(x, target)` | Create structure matching target, filled with x. |
| `cute.repeat_as_tuple(x, n)` | Always returns tuple of x repeated n times. |

### Tensor Construction

| Function | Purpose |
|---|---|
| `cute.make_tensor(ptr, layout)` | Create tensor from pointer + layout. |
| `cute.make_fragment_like(tensor)` | Create register-space tensor with same layout. |
| `cute.make_rmem_tensor(layout)` | Create register-memory tensor. |
| `cute.local_partition(tensor, ...)` | Partition tensor for local thread. |
| `cute.local_tile(tensor, ...)` | Tile tensor for local thread block. |

### Copy and MMA Operations

| Function | Purpose |
|---|---|
| `cute.copy(atom, src, dst)` | Copy data using a CopyAtom. |
| `cute.gemm(tiled_mma, A, B, C)` | Perform tiled matrix multiply-accumulate. |
| `cute.CopyAtom(...)` | Fundamental copy operation atom. |
| `cute.TiledCopy(...)` | Tiled copy across threads. |
| `cute.TiledMma(...)` | Tiled MMA across threads. |
| `cute.make_tma_copy(...)` | Create TMA (Tensor Memory Accelerator) copy atom. |

### Miscellaneous

| Function | Purpose |
|---|---|
| `cute.printf(fmt, *args)` | GPU runtime printf (C-style or `{}` format). |
| `cute.assume(src, divby=None)` | Attach divisibility hint to dynamic value. |
| `cute.is_static(x)` | Check if value is known at compile time. |
| `cute.is_major(mode, stride)` | Check if mode is the major (contiguous) mode. |
| `cute.E(mode)` | Create unit ScaledBasis element: `ScaledBasis(1, mode)`. |
| `cute.front(input)` | Recursively get first element of hierarchical input. |
| `cute.is_congruent(a, b)` | Check structural equivalence of two objects. |
| `cute.where(cond, true_val, false_val)` | Element-wise conditional select. |
| `cute.full_like(tensor, value)` | Create tensor filled with value, matching input shape. |

## `cutlass.cute.nvgpu` -- GPU Operations

Architecture-specific MMA and Copy operations. Top-level exposes arch-agnostic ops; submodules provide arch-specific ones.

| Submodule | Target | Key Operations |
|---|---|---|
| `nvgpu.common` | All archs | Shared helpers |
| `nvgpu.warp` | SM80+ | Warp-level MMA atoms |
| `nvgpu.warpgroup` | SM90+ | Warpgroup MMA (e.g., `WarpgroupMma`) |
| `nvgpu.cpasync` | SM80+ | `cp.async` copy operations |
| `nvgpu.tcgen05` | SM100 | Blackwell tcgen05 MMA, `CtaGroup`, TMEM operations |

## `cutlass.pipeline` -- Synchronization

Producer-consumer pipeline abstractions using mbarriers and named barriers.

| Class | Purpose |
|---|---|
| `PipelineAsync` | Base async pipeline: both producer and consumer are async threads. |
| `PipelineTmaAsync` | TMA producer + async thread consumer (Hopper mainloops). |
| `PipelineTmaUmma` | TMA producer + UMMA consumer (Blackwell mainloops). |
| `PipelineAsyncUmma` | Async thread producer + UMMA consumer (Blackwell input fusion). |
| `PipelineUmmaAsync` | UMMA producer + async thread consumer (Blackwell acc pipelines). |
| `PipelineCpAsync` | CpAsync producer + async thread consumer. |
| `PipelineState` | Circular buffer index + phase bit state. |
| `PipelineOrder` | Ordered execution across multiple groups. |
| `MbarrierArray` | Array of shared memory barriers with arrive/wait. |
| `NamedBarrier` | Hardware named barriers (16 available, ids 0-15). |
| `TmaStoreFence` | Multi-stage epilogue buffer sync. |
| `Agent` | Enum: `Thread`, `ThreadBlock`, `ThreadBlockCluster`. |
| `PipelineOp` | Enum: `AsyncThread`, `TCGen05Mma`, `TmaLoad`, `TmaStore`, `ClcLoad`, etc. |

### Pipeline Usage Pattern

```python
pipeline = PipelineAsync.create(
    num_stages=5,
    producer_group=producer_warp,
    consumer_group=consumer_warp,
    barrier_storage=smem_ptr,
)

producer, consumer = pipeline.make_participants()

# Producer side
for i in range(num_iterations):
    handle = producer.acquire_and_advance()
    # Write data
    handle.commit()

# Consumer side
for i in range(num_iterations):
    handle = consumer.wait_and_advance()
    # Read data
    handle.release()
```

## `cutlass.utils` -- Utilities

### Memory Allocation

| Class/Function | Purpose |
|---|---|
| `SmemAllocator()` | Shared memory allocator. Auto-calculates usage on kernel launch. Base pointer aligned to 1024 bytes. |
| `SmemAllocator.allocate(size_or_type, alignment)` | Allocate raw bytes, numeric type, or `@cute.struct`. |
| `SmemAllocator.allocate_array(element_type, num_elems)` | Allocate array in shared memory. |
| `SmemAllocator.allocate_tensor(element_type, layout)` | Allocate tensor in shared memory (static layouts only). |
| `SmemAllocator.capacity_in_bytes(compute_cap)` | Query max shared memory for a given compute capability. |
| `TmemAllocator(...)` | Tensor memory allocator for Blackwell (SM100). |
| `get_num_tmem_alloc_cols(tmem_tensors)` | Get total TMEM allocation columns. |

### Tile Scheduling

| Class | Purpose |
|---|---|
| `PersistentTileSchedulerParams` | Configure persistent tile scheduling (cluster shape, problem layout, swizzle). |
| `StaticPersistentTileScheduler` | Static persistent GEMM tile scheduler with work tile iteration. |
| `StaticPersistentRuntimeTileScheduler` | Runtime variant that always launches all SMs. |
| `WorkTileInfo` | Tile index + validity from scheduler. |
| `GroupedGemmTileSchedulerHelper` | Grouped GEMM: maps linear block index to per-group tile coordinates. |
| `GroupSearchResult` | Result of group search (group_idx, tile coords, problem shape). |

### Hardware and Layout Helpers

| Class/Function | Purpose |
|---|---|
| `HardwareInfo(device_id)` | Query GPU: max active clusters, L2 cache size, SM count. |
| `LayoutEnum` | `ROW_MAJOR` / `COL_MAJOR` enum with MMA-major helpers. |
| `TensorMapManager` | Manage TensorMap initialization and updates (GMEM/SMEM modes). |
| `TransformMode` | Enum for mixed-input GEMM: `ConvertOnly`, `ConvertScale`. |

## `cutlass.cute.runtime` -- Host-Side Runtime

| Function | Purpose |
|---|---|
| `from_dlpack(tensor, assumed_align, use_32bit_stride)` | Convert DLPack tensor to `cute.Tensor` (static layout, zero-copy). |
| `make_ptr(dtype, data_ptr, address_space, assumed_align)` | Create `cute.Pointer` from raw address (bypass DLPack). |

## `cutlass.cute.arch` -- Architecture Intrinsics

Thread/block/grid indexing, synchronization, and cluster operations (threadIdx, blockIdx, blockDim, syncthreads, cluster_arrive, etc.).
