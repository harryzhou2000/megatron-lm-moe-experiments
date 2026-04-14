# JIT Function Arguments

## Argument Classification

CuTe DSL arguments fall into two categories:

### Static Arguments (`cutlass.Constexpr`)

Values known at compile time. NOT included in the generated JIT function signature. Changing them triggers recompilation.

```python
@cute.jit
def foo(tile_size: cutlass.Constexpr, do_relu: cutlass.Constexpr):
    print("tile_size =", tile_size)  # prints actual value at compile time
```

### Dynamic Arguments (default)

Values known only at runtime. Included in the JIT function signature.

```python
@cute.jit
def foo(x: cutlass.Int32, tensor: cute.Tensor):
    cute.printf("x = %d\n", x)  # prints at runtime on GPU
```

**Default behavior**: Arguments are dynamic unless annotated with `cutlass.Constexpr`.

## Type Annotations and Safety

CuTe DSL validates argument types at compile time:

```python
@cute.jit
def foo(x: cute.Tensor, y: cutlass.Float16):
    ...

a = np.random.randn(10, 10).astype(np.float16)
b = 32

foo(a, b)     # OK
foo(b, a)     # COMPILE ERROR: type mismatch
# DSLRuntimeError: expects argument #1 (x) to be Tensor, but got int
```

## Supported JIT Argument Types

From native Python context, these types can be passed to JIT functions:

| Type | Description |
|---|---|
| `cute.Tensor` | CuTe tensor (from `from_dlpack` or implicit conversion) |
| `cute.Pointer` | Raw pointer (from `make_ptr`) |
| `cute.Shape` | Shape tuple |
| `cute.Stride` | Stride tuple (no `ScaledBasis` from native Python) |
| `cute.Coord` | Coordinate tuple |
| `cute.IntTuple` | Integer tuple |
| `cutlass.Int32`, `Float32`, etc. | Scalar types |
| `cutlass.Constexpr` | Compile-time constant (any Python value) |
| `int`, `float`, `bool` | Auto-converted to `Int32`, `Float32`, `Bool` |

**NOT passable from native Python**: `Layout` objects. Create layouts inside JIT functions using `cute.make_layout()`.

## Constexpr for Epilogue Fusion

A powerful pattern: pass lambdas as `Constexpr` for compile-time specialization:

```python
@cute.kernel
def gemm_kernel(
    self,
    tiled_mma: cute.TiledMma,
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    epilogue_op: cutlass.Constexpr,  # compile-time lambda
):
    # ... main GEMM ...
    acc_vec = tTR_rAcc.load()
    acc_vec = epilogue_op(acc_vec.to(self.c_dtype))
    tTR_rC.store(acc_vec)

# Usage: ReLU epilogue
gemm_kernel(..., epilogue_op=lambda x: cute.where(x > 0, x, cute.full_like(x, 0)))

# Usage: identity epilogue
gemm_kernel(..., epilogue_op=lambda x: x)
```

## Custom Types via Protocols

CuTe DSL provides two runtime-checkable protocols for custom argument types:

### `JitArgument` Protocol (host JIT functions called from Python)

```python
class JitArgument(Protocol):
    def __c_pointers__(self) -> list:          # ctypes pointers for C ABI
        ...
    def __get_mlir_types__(self) -> list:       # MLIR types for this object
        ...
    def __new_from_mlir_values__(self, values): # reconstruct from MLIR values
        ...
```

### `DynamicExpression` Protocol (device JIT functions called from host JIT)

```python
class DynamicExpression(Protocol):
    def __extract_mlir_values__(self) -> list:  # extract dynamic MLIR values
        ...
    def __new_from_mlir_values__(self, values): # reconstruct from MLIR values
        ...
```

### Direct Implementation

```python
class MyDynExpr:
    def __init__(self, tensor, offset):
        self._tensor = tensor
        self._offset = offset

    def __extract_mlir_values__(self):
        return [self._tensor.__extract_mlir_values__(),
                self._offset.__extract_mlir_values__()]

    def __new_from_mlir_values__(self, values):
        return MyDynExpr(values[0], values[1])

@cute.kernel
def my_kernel(x: MyDynExpr):
    ...
```

### Adapter-Based Registration

For types you cannot modify:

```python
@cutlass.register_jit_arg_adapter(MyFrameworkObject)
class MyAdapter:
    def __init__(self, arg):
        self._arg = arg

    def __c_pointers__(self):
        return [self._arg.get_cabi_pointer()]

    def __get_mlir_types__(self):
        return [self._arg.get_data().mlir_type]

    def __new_from_mlir_values__(self, values):
        return MyFrameworkObject(values[0])
```

After registration, `MyFrameworkObject` instances are automatically handled by CuTe DSL.
