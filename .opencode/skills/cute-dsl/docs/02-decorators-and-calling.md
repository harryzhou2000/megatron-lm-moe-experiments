# Decorators and Calling Conventions

## Two Main Decorators

CuTe DSL provides two Python decorators for generating optimized code via JIT compilation:

### `@cute.jit` -- Host-Side JIT Functions

Declares JIT-compiled functions invocable from Python or from other CuTe DSL functions.

```python
import cutlass
import cutlass.cute as cute

@cute.jit
def my_host_function(a: cutlass.Int32, b: cute.Tensor):
    # This runs as a JIT-compiled host function
    ...
```

**Decorator Parameters**:
- `preprocessor` (default: `True`) -- `True`: automatically translate Python control flow into IR operations. `False`: no automatic expansion (tracing only).

**Call-site Parameters**:
- `no_cache` (default: `False`) -- `True`: disables JIT caching, forces fresh compilation each call.

### `@cute.kernel` -- GPU Kernel Functions

Defines GPU kernel functions compiled as device code.

```python
@cute.kernel
def my_kernel(tensor_a: cute.Tensor, tensor_b: cute.Tensor):
    # This is GPU device code
    ...
```

**Decorator Parameters**:
- `preprocessor` (default: `True`) -- same as `@jit`.

**Kernel Launch Parameters** (passed to `.launch()`):

| Parameter | Type | Default | Description |
|---|---|---|---|
| `grid` | `list[int]` | required | Grid size `[x, y, z]` |
| `block` | `list[int]` | required | Block size `[x, y, z]` |
| `cluster` | `list[int]` | `None` | Cluster size `[x, y, z]` |
| `smem` | `int \| None` | `None` | Shared memory bytes. `None` = auto via `SmemAllocator` |
| `fallback_cluster` | `list[int] \| None` | `None` | Minimum-guaranteed cluster size (graceful degradation) |
| `max_number_threads` | `list[int]` | `[0,0,0]` | Max threads per block (maxntid). `[0,0,0]` = auto from `block` |
| `min_blocks_per_mp` | `int` | `0` | Min blocks per multiprocessor occupancy hint |
| `use_pdl` | `bool` | `False` | Programmatic Dependent Launch |
| `cooperative` | `bool` | `False` | Cooperative kernel launch (grid-wide sync) |

**Launch example**:
```python
@cute.jit
def launch_gemm(mA, mB, mC):
    my_kernel(mA, mB, mC).launch(
        grid=[grid_m, grid_n, 1],
        block=[128, 1, 1],
        cluster=[2, 1, 1],
        smem=None,  # auto-calculate
    )
```

## Calling Conventions

| Caller | Callee | Allowed | Behavior |
|---|---|---|---|
| Python function | `@jit` | Yes | DSL runtime invocation |
| Python function | `@kernel` | **No** | Error raised |
| `@jit` | `@jit` | Yes | Compile-time call, inlined |
| `@jit` | Python function | Yes | Compile-time call, inlined |
| `@jit` | `@kernel` | Yes | Dynamic GPU launch via driver |
| `@kernel` | `@jit` | Yes | Compile-time call, inlined |
| `@kernel` | Python function | Yes | Compile-time call, inlined |
| `@kernel` | `@kernel` | **No** | Error raised |

**Key points**:
- You cannot call `@kernel` directly from Python. You must go through a `@jit` function.
- `@kernel` cannot call another `@kernel`.
- All `@jit`-to-`@jit` and `@jit`-to-Python calls are inlined at compile time (zero overhead).
- Only `@jit`-to-`@kernel` is a true dynamic call dispatched via the GPU driver.

## Custom Kernel Name Prefix

For profiling/debugging, you can set a custom name prefix on kernels:

```python
@cute.kernel
def my_kernel(arg1, arg2):
    ...

@cute.jit
def launch():
    my_kernel.set_name_prefix("my_gemm_fp16_128x128")
    my_kernel(arg1, arg2).launch(grid=[1,1,1], block=[128,1,1])
    # Generated kernel name: "my_gemm_fp16_128x128_xxx"
```
