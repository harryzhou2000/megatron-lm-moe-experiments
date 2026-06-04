# Limitations and Pitfalls

## Platform Constraints

- **Linux only** (x86_64 and ARM64). No Windows, no macOS.
- **Python 3.10-3.14**.
- **NVIDIA driver** >= 575.51.03 (CUDA Toolkit 12.9 compatible).
- **No convolution support** yet.
- **No preferred clusters** support yet.

## 32-Bit Layout Algebra

CuTe layout shapes and strides are 32-bit only. 64-bit or arbitrary-width support is planned for future releases. This means:
- Maximum shape dimension value: ~2 billion.
- When using `use_32bit_stride=True` with `from_dlpack`, a runtime check ensures no overflow. For large tensors, keep `use_32bit_stride=False` (64-bit, default).

## Static Typing -- No Dependent Types

CuTe DSL is statically typed. The type of every expression must be determinable at compile time.

### What does NOT work:

```python
# Result type depends on runtime value -- NOT ALLOWED
max(int(1), float(2.0))   # Python returns float
max(int(3), float(2.0))   # Python returns int

# Ternary with different types -- NOT ALLOWED
res[0] = a if cond else b  # where a: Int32, b: Float32

# Type change inside loop -- NOT ALLOWED
a = Int32(1)
for i in range(10):
    a = Float32(2)  # WRONG
```

### What works:

```python
# Explicit type promotion
res[0] = max(a, b)  # auto-promoted to Float32 if a: Int32, b: Float32

# Same-type ternary
res[0] = a if cond else cutlass.Int32(0)  # both Int32
```

## Python Native Data Types

Lists, tuples, and dicts are **static containers** -- their structure is fixed at compile time.

```python
@cute.jit
def foo(a: Float32, b: Float32, i: Int32, res: cute.Tensor):
    xs = [a, b]

    # WRONG: indexing list with dynamic index
    res[0] = xs[i]  # NOT SUPPORTED

    # WRONG: list structure changes are compile-time only
    if i == 0:
        xs.append(Float32(3.0))  # Always appends, regardless of i

    # WRONG: loop doesn't unroll -- only one append at compile time
    for i in range(10):
        xs.append(Float32(1.0))  # Only 1 element added
```

## Function Return Values

Only `Constexpr` values can be returned from `@jit` functions. Returning **dynamic values** is NOT supported.

```python
@cute.jit
def returns_constexpr(a: cutlass.Constexpr):
    return a + 1  # OK: Constexpr return

@cute.jit
def returns_dynamic(a: cutlass.Int32):
    return a + 1  # PARTIALLY SUPPORTED: works when called from other JIT

returns_constexpr(10)     # OK
returns_dynamic(10)       # NOT SUPPORTED when called from Python
```

## Dynamic Control Flow Restrictions

Inside dynamic `if`/`for`/`while` bodies:
- **No** `break`, `continue`, `pass`
- **No** `return`
- **No** `raise` / exceptions
- Variables defined inside are **NOT visible** outside
- Variable types **CANNOT change**

## Object-Oriented Programming

Limited OOP support for objects containing dynamic values. Avoid passing dynamic values between methods via `self` state.

```python
class Foo:
    def __init__(self, a: Int32):
        self.a = a

    def set_a(self, i: Int32):
        self.a = i

    def get_a(self):
        return self.a

@cute.jit
def foo(a: Int32, res: cute.Tensor):
    foo = Foo(a)
    for i in range(10):
        foo.set_a(i)
    # FAILS: self.a was assigned inside loop body, not visible outside
    res[0] = foo.get_a()
```

**Workaround**: Implement the `DynamicExpression` protocol on your class (see [JIT Arguments](05-jit-arguments.md)).

## The `_` Variable

`_` is a special write-only variable. Reading it is NOT allowed:

```python
@cute.jit
def foo():
    _ = 1
    print(_)  # NOT ALLOWED
```

## CuTe Layout Algebra in Native Python

All CuTe layout algebra operations **require JIT compilation**. They cannot be used in standard Python outside `@jit`/`@kernel`.

Only these types can be passed as arguments from native Python:
`Tensor`, `Pointer`, `Shape`, `Stride`, `Coord`, `IntTuple`.

`Layout` objects **CANNOT** be passed from native Python. Create them inside JIT functions.

`ScaledBasis` in `Stride` is **NOT** supported from native Python context.

## Hashing and `functools.lru_cache`

DSL API objects are sensitive to MLIR context. **Do NOT** use `functools.lru_cache` with `@cute.jit` -- it caches MLIR objects from one context that may be invalid in another.

## DLPack Quirks

- DLPack canonicalizes stride-1 dimensions for shape-1 dims. This can produce multiple stride-1 dimensions, causing `mark_layout_dynamic` deduction to fail.
- DLPack conversion overhead: ~2-3 us per tensor.
- DLPack may lack support for some narrow data types.
- For performance-critical paths, bypass DLPack with `make_ptr()`.

## CuTe DSL 4.4.2 Pointer Operand Pitfalls

CuTe DSL 4.4.2 is stricter about operands passed to low-level NVVM ops. Dynamic pointer arithmetic from tensor iterators can produce proxy objects that are not accepted by `cute.arch.load()` or atomic wrappers, causing errors like:

```text
ValueError: Operand 0 of operation "nvvm.load.ext" must be a Value (is not a Value)
ValueError: Operand 0 of operation "nvvm.atomicrmw" must be a Value (is not a Value)
```

Use these workarounds:

- Prefer tensor indexing (`tensor[idx]`) over `cute.arch.load(tensor.iterator + idx, dtype)` for dynamic offsets.
- For atomics, pass a real `cute.Pointer` created with `make_ptr()` when the base address is known on the host.
- If pointer arithmetic still fails through the atomic wrapper, pass `(ptr + offset).llvm_ptr` explicitly.

Example:

```python
from cutlass.cute.runtime import make_ptr

tmp_p = make_ptr(cutlass.Int64, tmp_tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=8)

@cute.kernel
def kernel(tmp_p: cute.Pointer, idx: cutlass.Int32):
    ptr = tmp_p + idx
    cute.arch.atomic_exch(ptr.llvm_ptr, cutlass.Int64(1), sem="relaxed", scope="gpu")
```

## Debugging Limitations

- **No single-stepping** through JIT-compiled code.
- **No exception handling** inside JIT-compiled code.
- Use `print()` (compile-time) and `cute.printf()` (runtime) as primary debugging tools.
- Use `compute-sanitizer` for memory errors.

## Design Limitations (Will Likely Remain)

These are by-design constraints:

1. **Complex data structures as dynamic values**: Lists, tuples, dicts remain static containers. Structure cannot change at runtime.

2. **Dependent types**: Not supported. Would introduce complexity and hurt generated code performance.

3. **CuTe layout algebra in native Python**: No plans to extend. CuTe algebra will remain JIT-only.

## Common Error Messages and Their Meaning

| Error | Cause |
|---|---|
| `DSLRuntimeError: expects argument #N to be X, but got Y` | Type mismatch in JIT function call |
| `Can't deduce the leading dimension from layout` | `mark_layout_dynamic` can't find a unique stride-1 dim |
| `Expected strides[leading_dim] == 1, but got N` | Specified `leading_dim` doesn't have stride 1 |
| `Layout in DLTensorWrapper has int32 overflow risk` | Tensor too large for 32-bit strides; set `use_32bit_stride=False` |
| `oldString not found in content` | (Edit tool error, not DSL) |

## Workarounds Summary

| Problem | Workaround |
|---|---|
| Dynamic list indexing | Use compile-time indices or restructure as tensor operations |
| Return dynamic values | Use output tensors instead of return values |
| Type change in loop | Define variables with correct type before the loop |
| OOP with dynamic state | Implement `DynamicExpression` protocol |
| Layout from Python | Create layout inside `@jit` function using `cute.make_layout` |
| DLPack overhead | Cache `from_dlpack` results or bypass with `make_ptr` |
| Large tensor strides | Use `use_32bit_stride=False` (default) |
| `nvvm.load.ext` / `nvvm.atomicrmw` operand is not a Value | Use tensor indexing, host-created `make_ptr`, or `(ptr + offset).llvm_ptr` |

## CuTe DSL 4.5.x Static Attribute Strictness

Some APIs that lower to NVVM attributes require Python compile-time literals, not DSL numeric values. In 4.5.2, `arch.cp_async_bulk_wait_group(cutlass.Int32(N), read=True)` fails with an `IntegerAttr.get(...)` type error because the argument is an `ArithValue` instead of a Python `int`.

Use a plain Python constexpr integer:

```python
# WRONG on CuTe DSL 4.5.2
arch.cp_async_bulk_wait_group(cutlass.Int32(NUM_IN_FLIGHT), read=True)

# Works on 4.4.2 and 4.5.2
arch.cp_async_bulk_wait_group(NUM_IN_FLIGHT, read=True)
arch.cp_async_bulk_wait_group(0, read=True)
```
