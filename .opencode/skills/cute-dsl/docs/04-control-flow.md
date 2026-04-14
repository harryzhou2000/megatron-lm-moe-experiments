# Control Flow

## Overview

CuTe DSL walks Python's AST and converts each control-flow construct into structured IR. Each construct is either **compile-time** (evaluated during meta-stage) or **runtime** (emitted as IR for GPU execution).

Passing an IR/dynamic value to a compile-time construct raises an error.

## For Loops

Three kinds of ranges:

### `range()` / `cutlass.range()` -- Runtime Loops

Always lowered to IR, even if inputs are Python values.

```python
@cute.jit
def foo(bound: cutlass.Int32):
    n = 10

    # Runtime loop (IR), even though n is a Python int
    for i in range(n):
        cute.printf("%d\n", i)

    # Runtime loop with dynamic bound
    for i in range(bound):
        cute.printf("%d\n", i)

    # Runtime loop with unroll hint
    for i in cutlass.range(bound, unroll=2):
        cute.printf("%d\n", i)
```

### `cutlass.range_constexpr()` -- Compile-Time Loops

Fully unrolled at compile time. All loop indices must be `Constexpr`.

```python
@cute.jit
def foo():
    n = 10

    # Compile-time loop -- fully unrolled
    for i in cutlass.range_constexpr(n):
        cute.printf("%d\n", i)

    # ERROR: dynamic bound in range_constexpr
    # for i in cutlass.range_constexpr(dynamic_var):  # WRONG
```

### Software Pipelining (Experimental, SM90+)

CuTe DSL can auto-generate prefetch + main loop from a single loop body:

```python
# Manual pipelining (tedious):
for i in range(prefetch_stages):
    cute.copy(atom, gmem[i], buffer[i], ...)
for i in range(bound):
    if i + prefetch_stages < bound:
        cute.copy(atom, gmem[i + prefetch_stages], buffer[(i + prefetch_stages) % total_stages], ...)
    use(buffer[i % total_stages])

# Automatic pipelining:
for i in cutlass.range(bound, prefetch_stages=prefetch_stages):
    cute.copy(atom, gmem[i], buffer[i % total_stages], ...)
    use(buffer[i % total_stages])
```

The compiler generates the prefetch loop and main loop automatically.

## If-Else Statements

### Runtime Branches (default)

```python
@cute.jit
def foo(dynamic_var: cutlass.Int32):
    # Runtime branch -- emitted as IR
    if dynamic_var == 10:
        cute.printf("True\n")
    else:
        cute.printf("False\n")
```

### Compile-Time Branches

```python
@cute.jit
def foo(const_var: cutlass.Constexpr):
    # Compile-time branch -- only taken path emitted
    if cutlass.const_expr(const_var):
        cute.printf("Const branch\n")
    else:
        cute.printf("Const else\n")

    # ERROR: dynamic value in const_expr
    # if cutlass.const_expr(dynamic_var == 10):  # WRONG
```

## While Loops

### Runtime While

```python
@cute.jit
def foo(dynamic_var: cutlass.Int32):
    while dynamic_var == 10:
        cute.printf("looping\n")
        dynamic_var = dynamic_var - 1
```

### Compile-Time While

```python
@cute.jit
def foo():
    n = 0
    while cutlass.const_expr(n < 10):
        cute.printf("Const branch\n")
        n += 1
```

## Summary Table

| Construct | Compile-Time | Runtime |
|---|---|---|
| `if cutlass.const_expr(x)` | Yes | No |
| `if pred` | No | Yes |
| `while cutlass.const_expr(x)` | Yes | No |
| `while pred` | No | Yes |
| `for i in cutlass.range_constexpr(n)` | Yes (unrolled) | No |
| `for i in range(n)` | No | Yes |
| `for i in cutlass.range(n, ...)` | No | Yes (with hints) |

## Compile-Time Metaprogramming Pattern

Mix compile-time guards with runtime code:

```python
@cute.kernel
def gemm(..., do_relu: cutlass.Constexpr):
    # ... main GEMM work ...

    if cutlass.const_expr(do_relu):
        # ReLU code ONLY emitted when do_relu=True
        acc = cute.where(acc > 0, acc, cute.full_like(acc, 0))

# Two different kernels compiled:
gemm(..., False)   # no ReLU in generated code
gemm(..., True)    # ReLU included
```

## Limitations of Dynamic Control Flow

**Not supported inside dynamic control flow bodies**:
- `break`
- `continue`
- `pass`
- `return`
- `raise` / exceptions

**Scoping rules**:
- Variables defined inside a dynamic branch/loop body are NOT visible outside.
- Variables used in dynamic control flow MUST be defined before the construct.
- Variable type CANNOT change inside a dynamic body.

```python
@cute.jit
def negative_examples(predicate: cutlass.Boolean):
    # WRONG: early exit
    for i in range(10):
        if i == 5:
            break           # NOT SUPPORTED

    # WRONG: variable defined inside dynamic branch
    if predicate:
        val = 10
    cute.printf("%d\n", val)  # val not visible here

    # WRONG: type change in dynamic branch
    n = cutlass.Int32(10)
    if predicate:
        n = cutlass.Float32(10.0)  # type change NOT ALLOWED
```

## Static vs Dynamic Layout Interaction with Control Flow

When tensor layouts are static, the compiler can optimize control flow at compile time:

```python
@cute.jit
def foo(tensor, x: cutlass.Constexpr[int]):
    # Static layout: cute.size(tensor) is a compile-time constant
    # -> if/for can be resolved at compile time
    if cute.size(tensor) > x:
        cute.printf("tensor[2]: {}", tensor[2])

    # Dynamic layout: cute.size(tensor) is runtime value
    # -> if/for emit as runtime IR
    for i in range(cute.size(tensor)):
        cute.printf("tensor[{}]: {}", i, tensor[i])
```
