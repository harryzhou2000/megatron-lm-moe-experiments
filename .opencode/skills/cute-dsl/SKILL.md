---
name: cute-dsl
description: Expert guidance for writing NVIDIA CuTe DSL kernels -- Python-based GPU kernel authoring using the CUTLASS 4.x CuTe DSL framework. Use this skill when writing, reviewing, or debugging CuTe DSL code, when the user mentions "cute dsl", "cutlass dsl", "cutlass python", "@cute.jit", "@cute.kernel", or when working with CuTe layout algebra, tensor operations, or GEMM kernels in Python.
license: MIT
compatibility: opencode
---

# CuTe DSL Skill

Expert guidance for writing NVIDIA CuTe DSL kernels -- Python-based GPU kernel authoring using the CUTLASS 4.x CuTe DSL framework. Use this skill when writing, reviewing, or debugging CuTe DSL code, when the user mentions "cute dsl", "cutlass dsl", "cutlass python", "@cute.jit", "@cute.kernel", or when working with CuTe layout algebra, tensor operations, or GEMM kernels in Python.

## Reference Documentation

Detailed documentation is provided in the `docs/` directory alongside this skill:

- [docs/01-overview.md](docs/01-overview.md) -- Architecture, compilation pipeline, core concepts
- [docs/02-decorators-and-calling.md](docs/02-decorators-and-calling.md) -- `@jit`, `@kernel`, calling conventions
- [docs/03-type-system.md](docs/03-type-system.md) -- Types, static vs dynamic, layouts, tensors, pointers, structs
- [docs/04-control-flow.md](docs/04-control-flow.md) -- `for`/`while`/`if`, compile-time vs runtime, software pipelining
- [docs/05-jit-arguments.md](docs/05-jit-arguments.md) -- Argument generation, static/dynamic args, custom types
- [docs/06-framework-integration.md](docs/06-framework-integration.md) -- DLPack, PyTorch interop, `from_dlpack`, dynamic layouts
- [docs/07-caching-and-compilation.md](docs/07-caching-and-compilation.md) -- JIT caching, `cute.compile`, environment variables
- [docs/08-debugging.md](docs/08-debugging.md) -- Logging, IR/PTX/CUBIN dumps, `cute.printf`, compute-sanitizer
- [docs/09-limitations-and-pitfalls.md](docs/09-limitations-and-pitfalls.md) -- Known constraints, workarounds, design limitations
- [docs/10-api-overview.md](docs/10-api-overview.md) -- API module map: `cute`, `nvgpu`, `pipeline`, `utils` key functions

For the **full official API reference** with complete signatures and all parameters, see:
https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api.html

## Installation

```bash
# For CUDA Toolkit 12.9:
pip install nvidia-cutlass-dsl

# For CUDA Toolkit 13.1:
pip install nvidia-cutlass-dsl[cu13]

# From source (ensures compatibility with GitHub examples):
git clone https://github.com/NVIDIA/cutlass.git
./cutlass/python/CuTeDSL/setup.sh --cu12   # or --cu13
```

Requirements: Linux (x86_64 or ARM64), Python 3.10-3.14, NVIDIA driver >= 575.51.03.

## Core Rules -- MUST Follow When Writing CuTe DSL Code

### 1. Understand the Two-World Model

CuTe DSL code executes in two stages:

- **Meta-stage (compile time)**: Python interpreter runs with proxy arguments. `print()` works here. Python metaprogramming, list/dict manipulation, class instantiation all happen here.
- **Object-stage (runtime)**: Compiled CUDA kernel executes on GPU. `cute.printf()` works here. Only dynamic values exist.

**Rule**: Never confuse what runs at compile time vs runtime. Use `print()` for compile-time debugging, `cute.printf()` for runtime debugging.

### 2. Static Typing -- No Dependent Types

CuTe DSL is statically typed. The type of every expression must be determinable at compile time.

```python
# WRONG: dependent type -- result type depends on runtime value
res[0] = a if cond else b   # where a: Int32, b: Float32

# CORRECT: explicit type conversion
res[0] = a.to(Float32) if cond else b
```

**Rules**:
- Never change a variable's type inside a loop or branch body.
- Never use `max(int_val, float_val)` where result type would depend on values.
- Always use explicit type conversions when mixing types.

### 3. Control Flow Categories

Every control-flow construct is either compile-time or runtime:

| Construct | Compile-time | Runtime |
|---|---|---|
| `if cutlass.const_expr(x)` | Yes | No |
| `if dynamic_var == 10` | No | Yes |
| `for i in cutlass.range_constexpr(n)` | Yes (unrolled) | No |
| `for i in range(n)` | No | Yes (IR loop) |
| `for i in cutlass.range(n, unroll=2)` | No | Yes (with unroll hint) |
| `while cutlass.const_expr(cond)` | Yes | No |
| `while dynamic_cond` | No | Yes |

**Rules**:
- Never pass a dynamic value to `cutlass.const_expr()` or `cutlass.range_constexpr()`.
- Never use `break`, `continue`, `pass`, `return`, or `raise` inside dynamic control flow bodies.
- Variables defined inside dynamic control flow are NOT visible outside.
- Variables used in dynamic control flow must be defined BEFORE the control flow statement.

### 4. Decorator Usage

```python
import cutlass
import cutlass.cute as cute

@cute.jit                    # Host-side JIT function
def host_func(...):
    ...

@cute.kernel                 # GPU kernel
def my_kernel(...):
    ...
```

**Calling convention rules**:
- Python can call `@jit` directly. Python CANNOT call `@kernel` directly.
- `@jit` can call `@jit`, `@kernel`, and plain Python functions.
- `@kernel` can call `@jit` and plain Python functions. `@kernel` CANNOT call `@kernel`.
- `@jit`-to-`@jit` and `@jit`-to-Python calls are inlined at compile time.
- `@jit`-to-`@kernel` is a dynamic GPU launch.

### 5. Kernel Launch Parameters

```python
@cute.kernel
def my_kernel(tensor_a: cute.Tensor, tensor_b: cute.Tensor):
    ...

@cute.jit
def launch():
    my_kernel(a, b).launch(
        grid=[grid_x, grid_y, grid_z],
        block=[block_x, block_y, block_z],
        cluster=[cx, cy, cz],          # optional
        smem=None,                       # None = auto-calculate via SmemAllocator
    )
```

Launch parameters:
- `grid`: list of 3 ints
- `block`: list of 3 ints
- `cluster`: optional list of 3 ints
- `smem`: `None` (auto) or `int` (bytes)
- `use_pdl`: `False` (default) -- Programmatic Dependent Launch
- `cooperative`: `False` (default) -- cooperative kernel launch
- `fallback_cluster`: optional minimum-guaranteed cluster size

### 6. Tensor and Layout Rules

```python
# Creating layouts
layout = cute.make_layout((M, N), stride=(N, 1))         # row-major
layout = cute.make_layout((M, N))                          # default (col-major)
layout = cute.make_ordered_layout((M, N, L), order=(0, 1, 2))

# Creating tensors
tensor = cute.make_tensor(ptr, layout=layout)
```

**Rules**:
- CuTe layout shapes/strides are 32-bit only.
- CuTe layout algebra operations (composition, complement, etc.) are ONLY available inside `@jit`/`@kernel` functions, NOT in native Python.
- Supported JIT argument types: `Tensor`, `Pointer`, `Shape`, `Stride`, `Coord`, `IntTuple`. `Layout` cannot be passed from native Python.

### 7. Framework Tensor Integration

```python
import torch
from cutlass.cute.runtime import from_dlpack, make_ptr

# Implicit conversion (auto dynamic layout, ~2-3us overhead per tensor):
@cute.jit
def foo(tensor):          # Pass torch.Tensor directly
    print(tensor.layout)  # (?,?):(?,1) -- dynamic

# Explicit conversion (static layout, cacheable):
t = from_dlpack(torch_tensor)
t.mark_layout_dynamic()                                    # all shapes dynamic
t.mark_compact_shape_dynamic(mode=0, divisibility=16)      # fine-grained

# Zero-overhead bypass (no DLPack):
ptr = make_ptr(cutlass.Float16, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)
```

**Rules**:
- Passing `torch.Tensor` directly to `@jit` triggers implicit DLPack conversion with dynamic layout.
- `from_dlpack()` produces static layout; use `mark_layout_dynamic()` for reuse across shapes.
- For performance-critical paths, bypass DLPack with `make_ptr()` + `cute.make_tensor()`.

### 8. JIT Caching

- Caching is ON by default. Same IR = cache hit, skip compilation.
- Cache key = hash of MLIR bytecode + source files + shared libs + env vars.
- Use `cute.compile(fn, *args)` for explicit compilation returning a reusable JIT Executor.
- Use `no_cache=True` on call site to force recompilation.
- File cache dir: `/tmp/{user}/cutlass_python_cache` (override with `CUTE_DSL_CACHE_DIR`).

### 9. Data Types and Automatic Conversion

Python primitives auto-convert when passed as dynamic arguments:
- `int` -> `Int32`
- `bool` -> `Bool`
- `float` -> `Float32`

Annotate with `cutlass.Constexpr` for compile-time constants:
```python
@cute.jit
def foo(x: cutlass.Int32, tile_size: cutlass.Constexpr):
    ...
```

### 10. Common Patterns

**Compile-time specialization (epilogue fusion)**:
```python
@cute.kernel
def gemm_kernel(..., do_relu: cutlass.Constexpr):
    # ... main GEMM work ...
    if cutlass.const_expr(do_relu):
        # ReLU code only emitted when do_relu=True
        acc = cute.where(acc > 0, acc, cute.full_like(acc, 0))
```

**Software pipelining**:
```python
for i in cutlass.range(bound, prefetch_stages=num_stages):
    cute.copy(atom, gmem[i], smem_buf[i % total_stages], ...)
    use(smem_buf[i % total_stages])
```

**Shared memory allocation**:
```python
@cute.struct
class SmemStorage:
    buf_a: cute.struct.Align[cute.struct.MemRange[cutlass.Float16, size_a], 128]
    buf_b: cute.struct.Align[cute.struct.MemRange[cutlass.Float16, size_b], 128]

allocator = utils.SmemAllocator()
storage = allocator.allocate(SmemStorage)
```

**Dynamic shape with divisibility assumption**:
```python
@cute.jit
def foo(a: cute.Tensor):
    new_shape = a.shape
    new_shape[0] = cute.assume(new_shape[0], 16)  # assume divisible by 16
    new_layout = cute.make_layout(new_shape, stride=a.stride)
    new_a = cute.make_tensor(a.iterator, new_layout)
```

### 11. Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `CUTE_DSL_LOG_TO_CONSOLE` | `0` | Enable console logging |
| `CUTE_DSL_LOG_LEVEL` | `10` | Log verbosity (0=off, 10=debug, 20=info, 30=warn) |
| `CUTE_DSL_PRINT_IR` | `0` | Dump generated MLIR IR |
| `CUTE_DSL_KEEP_IR` | `0` | Save IR to file |
| `CUTE_DSL_KEEP_PTX` | `0` | Save PTX to file |
| `CUTE_DSL_KEEP_CUBIN` | `0` | Save CUBIN to file |
| `CUTE_DSL_LINEINFO` | `0` | Generate debug line info for profiling |
| `CUTE_DSL_CACHE_DIR` | `/tmp/{user}/...` | Persistent cache directory |
| `CUTE_DSL_DISABLE_FILE_CACHING` | `0` | Disable file-based JIT cache |
| `CUTE_DSL_DUMP_DIR` | `.` | Directory for dumped files |

### 12. Supported Hardware

| Architecture | Supported MMA Types |
|---|---|
| Ampere (SM80) | FP16, BF16 |
| Hopper (SM90) | FP16, BF16, FP8 |
| Blackwell (SM100) | FP16, BF16, TF32, I8, F8 |

### 13. Critical Don'ts

- **Don't** use `functools.lru_cache` with `@cute.jit` (MLIR context sensitivity).
- **Don't** index a Python list with a dynamic index inside JIT.
- **Don't** use `_` as a readable variable (it is write-only in the DSL).
- **Don't** modify list/dict structure at runtime inside JIT (append, pop, etc.).
- **Don't** pass dynamic values between class methods via `self` state without implementing `DynamicExpression` protocol.
- **Don't** use `return` to return dynamic values from `@jit` functions (only `Constexpr` returns work).
- **Don't** hash DSL API objects across different MLIR contexts.

## Upstream Documentation

- Overview & guides: https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html
- Full API reference: https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api.html
