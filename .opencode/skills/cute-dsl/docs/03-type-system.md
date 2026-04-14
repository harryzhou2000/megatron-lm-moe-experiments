# Type System

## Overview

CuTe DSL provides a set of core types for tensor layout algebra and GPU programming. Types divide into **static** (compile-time, known during meta-stage) and **dynamic** (runtime, known only during kernel execution).

## Primitive Types and Auto-Conversion

When Python primitives are passed as dynamic arguments to JIT functions, they auto-convert:

| Python Type | CuTe DSL Type |
|---|---|
| `int` | `cutlass.Int32` |
| `bool` | `cutlass.Bool` |
| `float` | `cutlass.Float32` |

For compile-time constants, annotate with `cutlass.Constexpr`:
```python
@cute.jit
def foo(x: cutlass.Int32, tile_size: cutlass.Constexpr):
    print("tile_size =", tile_size)  # known at compile time
    cute.printf("x = %d\n", x)       # known only at runtime
```

Explicit numeric types available:
- Integer: `cutlass.Int8`, `cutlass.Int16`, `cutlass.Int32`, `cutlass.Int64`
- Unsigned: `cutlass.UInt8`, `cutlass.UInt16`, `cutlass.UInt32`
- Float: `cutlass.Float16`, `cutlass.BFloat16`, `cutlass.Float32`, `cutlass.Float64`
- FP8: `cutlass.Float8e4m3`, `cutlass.Float8e5m2`
- Special: `cutlass.Boolean`

## Core Numeric Types

### IntValue

Internal proxy for constrained integer types with divisibility tracking. Used inside JIT functions.

```python
# IntValue with divisibility 1 prints as "?"
# IntValue with divisibility 4 prints as "?{div=4}"
```

Supports arithmetic with divisibility propagation: `+`, `-`, `*`, `//`, `%`.

### Ratio

Rational number as numerator/denominator pair. Arises in layout composition where divisibility conditions may not hold.

```python
r = cute.Ratio(1, 2)
r.is_integral()   # False
r.reduced()        # Ratio(1, 2)
r.to(float)        # 0.5
r * 3              # Ratio(3, 2)
```

## Layout Algebra Types

### Layout

Core abstraction: maps logical coordinates to linear indices via `(Shape, Stride)` pair.

```python
# Column-major (default)
layout = cute.make_layout((4, 8))           # (4,8):(1,4)

# Row-major
layout = cute.make_layout((4, 8), stride=(8, 1))  # (4,8):(8,1)

# Ordered layout
layout = cute.make_ordered_layout((M, N, L), order=(0, 1, 2))

# Properties
layout.shape    # (4, 8)
layout.stride   # (8, 1)
```

**Layout operations**: concatenation, coalescence, composition, complement, inversion.

**Important**: CuTe layout algebra operations are ONLY available inside `@jit`/`@kernel` functions, NOT in native Python.

### ComposedLayout

Composition of layouts and transformations. Three components:
- `inner` -- inner transformation (Swizzle or Layout)
- `offset` -- offset applied to coordinates
- `outer` -- outer layout

```python
# Typically created through composition operations
composed = cute.composition(swizzle, layout)
composed.inner   # the swizzle
composed.outer   # the layout
composed.offset  # the offset
```

String format: `inner o offset o outer`.

### ScaledBasis

Scaled basis element in CuTe's coordinate system. Scale value + mode identifier.

```python
sb = cute.ScaledBasis(2, 0)         # 2 * E(0)
sb = cute.ScaledBasis(cute.Ratio(1, 2), 1)  # (1/2) * E(1)
basis = cute.E(mode)                # unit-scale basis: ScaledBasis(1, mode)

# Used in layout strides for multi-modal layouts
layout = cute.make_layout((4, 8), stride=(cute.ScaledBasis(2, 0), cute.ScaledBasis(1, 1)))
```

### Swizzle

Bit-manipulation transformation to avoid shared memory bank conflicts.

Parameters:
- **MBase**: least-significant bits kept constant
- **BBits**: number of bits in the XOR mask
- **SShift**: distance to shift the mask

```
Given:    0bxxxxxxxxxxxxxxxxYYxxxxxxxxxZZxxx
Result:   0bxxxxxxxxxxxxxxxxYYxxxxxxxxxAAxxx
          where AA = ZZ xor YY
```

## Memory and Pointer Types

### Pointer

Memory address with type, memory space, and alignment information.

```python
# Properties
ptr.dtype        # value type (e.g., Float16)
ptr.memspace     # gmem, smem, rmem, generic
ptr.alignment    # alignment in bytes

# Arithmetic
ptr2 = ptr + offset
ptr3 = ptr - offset

# Conversion
addr = ptr.toint()       # Int64 for gmem/generic, Int32 otherwise
aligned = ptr.align(16)  # align to 16-byte boundary
```

**Tensor = Pointer ∘ Layout**: `T(c) = *(E + L(c))`

### Tensor

Composition of a pointer/iterator (engine) with a layout.

```python
# Create tensor from pointer and layout
tensor = cute.make_tensor(ptr, layout=layout)

# Properties
tensor.shape      # shape tuple
tensor.stride     # stride tuple
tensor.layout     # the full layout
tensor.iterator   # the pointer/engine
tensor.data()     # get underlying pointer

# Indexing
val = tensor[i]
tensor[i] = val
```

## Structured Data Types

### `@cute.struct`

Abstracts C structures with precise layout, alignment, and nesting control.

```python
@cute.struct
class complex:
    real: cutlass.Float32
    imag: cutlass.Float32

@cute.struct
class SmemStorage:
    mbar: cute.struct.MemRange[cutlass.Int64, num_stages]
    data: cute.struct.Align[cute.struct.MemRange[cutlass.Float16, buf_size], 1024]
    flag: cutlass.Int32

# Static queries
SmemStorage.__sizeof__()    # size in bytes
SmemStorage.__alignof__()   # alignment in bytes
```

### `cute.struct.MemRange[dtype, size]`

Contiguous memory range with element type and count.

```python
@cute.struct
class Buffer:
    data: cute.struct.MemRange[cutlass.Float32, 128]

buf = allocator.allocate(Buffer)
ptr = buf.data.data_ptr()
elem = buf.data[5]
tensor = buf.data.get_tensor(layout, swizzle=None, dtype=None)
```

### `cute.struct.Align[dtype, alignment]`

Explicit alignment for struct members.

```python
@cute.struct
class AlignedStorage:
    buffer: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, 256], 1024]
    counter: cute.struct.Align[cutlass.Int32, 16]
```

### `@cute.union`

C union: all members start at offset 0, size = max member size.

```python
@cute.union
class ValueUnion:
    as_int: cutlass.Int32
    as_float: cutlass.Float32
```

## Static vs Dynamic Layouts

**Static layout**: All shape/stride values known at compile time. Enables aggressive optimization (loop unrolling, constant folding). Requires recompilation for different shapes.

**Dynamic layout**: Shape/stride values determined at runtime (`?` markers). Single compilation works for varying shapes. Less optimization opportunity.

```python
# Static layout (from from_dlpack):
t = from_dlpack(torch_tensor)   # e.g., (3):(1)

# Dynamic layout (mark_layout_dynamic):
t.mark_layout_dynamic()         # e.g., (?,?):(?,1)

# Fine-grained dynamic (specific mode):
t.mark_compact_shape_dynamic(mode=0, divisibility=16)  # (?{div=16},8):(8,1)
```

**When passing `torch.Tensor` directly** to `@jit`, implicit conversion uses `mark_layout_dynamic` automatically.

## Best Practices

1. Use **static values** (Python `int`) when dimensions are known at compile time.
2. Use **dynamic values** when dimensions vary across calls.
3. Specify **alignment** for shared memory structures to avoid bank conflicts.
4. Use **type annotations** in `@jit`/`@kernel` signatures for type safety.
5. Prefer built-in layout operations (`make_layout`, `composition`) over manual construction.
6. Use `cute.assume(value, divby)` to attach divisibility hints to dynamic shapes.
