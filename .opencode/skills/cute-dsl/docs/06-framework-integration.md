# Framework Integration

## DLPack Protocol

CuTe DSL integrates with frameworks (PyTorch, JAX, NumPy) via the DLPack protocol. Framework tensors are converted to `cute.Tensor` for use in JIT functions.

## Implicit Conversion

Pass framework tensors directly to `@jit` functions. CuTe DSL auto-converts via DLPack with a **fully dynamic layout** (except leading dimension stride = 1).

```python
import torch
import cutlass.cute as cute

@cute.jit
def foo(src):
    print(src)        # ptr<f32, generic> o (?,?,?):(?,?,1)
    print(type(src))  # <class 'cutlass.cute.core._Tensor'>

a = torch.randn(30, 20, 32, device="cuda")
foo(a)  # implicit conversion, dynamic layout
```

**Overhead**: ~2-3 microseconds per tensor per call.

## Explicit Conversion with `from_dlpack`

```python
from cutlass.cute.runtime import from_dlpack

x = torch.randn(30, 20, device="cuda")
y = from_dlpack(x)

print(y.shape)         # (30, 20)
print(y.stride)        # (20, 1)
print(y.memspace)      # gmem
print(y.element_type)  # Float32
# Tensor<0x000000000875f580@gmem o (30, 20):(20, 1)>
```

**Result**: Static layout (all shapes/strides known at compile time). Zero-copy -- shares memory with source tensor.

### `from_dlpack` Signature

```python
def from_dlpack(tensor, assumed_align=None, use_32bit_stride=False):
```

| Parameter | Description |
|---|---|
| `assumed_align` | Alignment in bytes. Default: natural alignment of element type. Affects IR and caching. |
| `use_32bit_stride` | Use 32-bit strides for dynamic values. Default: `False` (64-bit). Set `True` for small problems to reduce register usage. |

### When to Use Explicit Conversion

1. **Caching**: Avoid repeated DLPack overhead by caching converted tensors.
2. **Fine-grained layout control**: Control which modes are static vs dynamic.
3. **Alignment control**: Specify custom alignment.

```python
if key not in cached_tensors:
    cached_tensors[key] = cute.runtime.from_dlpack(x)
foo(cached_tensors[key])
```

## Making Layouts Dynamic

### `mark_layout_dynamic(leading_dim=None)`

After calling, all shape modes become dynamic. Stride modes become dynamic except:
- Leading dimension stride remains 1.
- Stride elements equal to 0 (broadcasting) are retained.

```python
t = from_dlpack(torch_tensor)    # static: (30, 20):(20, 1)
t.mark_layout_dynamic()          # dynamic: (?,?):(?,1)
```

**Leading dimension deduction** (when `leading_dim=None`):
1. If exactly one dimension has stride 1, that's the leading dim.
2. If multiple dims have stride 1, succeeds only if exactly one has size > 1.
3. If no dimension has stride 1, all strides remain dynamic.

```python
# (8,4,16,2):(2,16,64,1) -> leading_dim=3 -> (?,?,?,?):(?,?,?,1)
a = torch.empty(16, 4, 8, 2).permute(2, 1, 0, 3)
t = from_dlpack(a).mark_layout_dynamic()

# (2,2):(8,2) -> no stride-1 dim -> (?,?):(?,?)
c = torch.empty(3, 4)[::2, ::2]
t = from_dlpack(c).mark_layout_dynamic()

# Broadcasting: (3,4,2,5):(5,0,0,1) -> (?,?,?,?):(?,0,0,1)
d = torch.empty(3, 1, 1, 5).expand(3, 4, 2, 5)
t = from_dlpack(d).mark_layout_dynamic()
```

### `mark_compact_shape_dynamic(mode, stride_order=None, divisibility=1)`

Fine-grained control: make a specific mode dynamic with divisibility constraint.

```python
# (8,4,16,2):(2,16,64,1) -- make mode 0 dynamic, div by 2
t = from_dlpack(a).mark_compact_shape_dynamic(mode=0, divisibility=2)
# (?{div=2},4,16,2):(2,?{div=4},?{div=16},1)

# Chain multiple calls
t = from_dlpack(a) \
    .mark_compact_shape_dynamic(mode=1, divisibility=2) \
    .mark_compact_shape_dynamic(mode=3, divisibility=2)
# (8,?{div=2},16,?{div=2}):(?{div=2},?{div=16},?{div=32},1)
```

**`stride_order`**: Ordering of strides (like `torch.Tensor.dim_order()`). Required when auto-deduction fails (e.g., multiple stride-1 dims).

**Only works for compact tensors**. For non-compact tensors, use `cute.assume` inside JIT:
```python
@cute.jit
def foo(a: cute.Tensor):
    new_shape = a.shape
    new_shape[0] = cute.assume(new_shape[0], 16)
    new_layout = cute.make_layout(new_shape, stride=a.stride)
    new_a = cute.make_tensor(a.iterator, new_layout)
```

## Bypassing DLPack (Zero Overhead)

For maximum performance, bypass DLPack entirely:

```python
from cutlass.cute.runtime import make_ptr

# Create raw pointer from PyTorch tensor
a_ptr = make_ptr(
    cutlass.Float16,
    a.data_ptr(),
    cute.AddressSpace.gmem,
    assumed_align=32
)

# JIT wrapper constructs tensors from pointers
@cute.jit
def wrapper(a_ptr: cute.Pointer, m: cutlass.Int32, n: cutlass.Int32):
    m = cute.assume(m, divby=8)
    n = cute.assume(n, divby=8)
    a_layout = cute.make_ordered_layout((m, n), order=(0, 1))
    mA = cute.make_tensor(a_ptr, layout=a_layout)
    my_kernel(mA).launch(...)
```

**Use cases for bypassing DLPack**:
1. Avoid 2-3 us overhead per tensor.
2. DLPack canonicalizes stride-1 dimensions, which may break alignment propagation.
3. DLPack may lack support for narrow data types.

## TVM FFI Integration

CuTe DSL supports TVM FFI for faster PyTorch interop:
- Faster JIT function invocation
- Direct `torch.Tensor` acceptance
- Enhanced error handling
