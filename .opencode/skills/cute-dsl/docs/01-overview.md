# CuTe DSL Overview

## What is CuTe DSL?

CuTe DSL is a Python-based domain-specific language released as part of CUTLASS 4.x. It provides a Python front-end to the CuTe C++ abstractions for writing high-performance CUDA GPU kernels -- primarily targeting linear algebra (GEMM) workloads -- without writing C++ templates.

It is a **low-level** kernel authoring DSL. You still think in terms of tiles, warps, shared memory, TMA, and MMA atoms, but you express everything in Python. It is part of the CUTLASS family and shares all concepts with CUTLASS 3.x C++ (CuTe layouts, tensors, atoms, tiled operations, pipelines, schedulers).

## Core Abstractions

| Concept | Description |
|---|---|
| **Layouts** | Describe how data is organized in memory and across threads. A `(Shape, Stride)` pair mapping logical coordinates to linear indices. Currently 32-bit shapes/strides only. |
| **Tensors** | Data pointer (engine/iterator) composed with layout metadata: `T = E ∘ L`. Dereferencing: `T(c) = *(E + L(c))`. |
| **Atoms** | Fundamental hardware operations: matrix multiply-accumulate (MMA) or memory copy. |
| **Tiled Operations** | How atoms are applied across thread blocks and warps: `TiledMma`, `TiledCopy`. |

## The Hybrid DSL Architecture

CuTe DSL's key innovation is its **hybrid compilation** approach combining two techniques:

### AST Rewrite
Before execution, the Python AST is analyzed. Control flow (`for`, `while`, `if/else`) is rewritten into structured MLIR IR constructs. This preserves loop/branch structure that pure tracing would lose.

**Advantages**: Sees the entire program; preserves every branch and loop; keeps loop structure intact for optimization (tiling, vectorization, GPU thread mapping).

**Disadvantages**: Requires a well-defined Python subset.

### Tracing
Inside each structured region, the function executes with proxy tensor arguments. Overloaded operators record every tensor operation into IR.

**Advantages**: Near-zero compile latency for straight-line arithmetic; supports dynamic Python features naturally.

**Disadvantages**: Untaken branches vanish; loops flatten; data-dependent control flow freezes.

### Why Hybrid Works
GPU kernels are structurally simple at runtime (no deep call hierarchies, minimal branching), but *authoring* them benefits from Python abstractions. The hybrid approach resolves this:

1. **AST rewrite handles structure** -- loops and branches compile to structured IR, solving tracing's control-flow problem.
2. **Tracing handles arithmetic** -- tensor operations are recorded as-is, solving AST rewriting's complexity problem.

Result: Loops compile to real loops (not unrolled traces), all branches are preserved, and Python metaprogramming works naturally.

## Three-Stage Compilation Pipeline

```
Python source
    |
    v
[Stage 1: Pre-Staging -- AST Preprocessing]
    - Rewrites AST, inserts callbacks around control-flow constructs
    |
    v
[Stage 2: Meta-Stage -- Python Interpreter Tracing]
    - Executes rewritten function with proxy arguments
    - Callbacks emit structured IR (loops, branches)
    - Tensor operations traced via overloaded operators
    - Compile-time constants partially evaluated
    - print() runs here
    |
    v
[Stage 3: Object-Stage -- Compiler Backend]
    - MLIR optimization passes (tiling, vectorization, memory promotion)
    - Lowering to PTX/SASS via ptxas
    - cute.printf() runs here on GPU
    |
    v
Device binary (loaded and launched)
```

## Meta-Programming vs Runtime

Your Python code runs twice, in two different contexts:

1. **Meta-programming time (compilation)** -- Python executes to *build* the kernel. Happens on host CPU when you call a `@jit` function. `print()` works here; proxy values show as `<Float32 proxy>`.

2. **Runtime (execution)** -- The compiled kernel runs on GPU with actual tensor data. `cute.printf()` works here; actual values are printed.

### Practical Implications

- Use `print()` to debug your meta-program (shapes, strides, tile sizes, compile-time decisions).
- `Constexpr` parameters enable specialization -- compiler generates tighter code when values are known at JIT time.
- Dynamic parameters preserve generality -- a single compiled kernel handles varying input sizes without recompilation.

## Code-Generation Modes

Two mutually exclusive modes (selectable via `preprocessor` flag):

1. **Tracing mode** (`@jit(preprocessor=False)`) -- Tracing only. Fastest compilation. Recommended only for straight-line arithmetic kernels. Suffers from all tracing limitations.

2. **Preprocessor mode** (`@jit(preprocessor=True)`, **default**) -- AST rewrite + tracing. Captures every loop and branch, then traces arithmetic. This is the recommended mode.

## Supported Hardware

| Architecture | Supported MMA Types |
|---|---|
| Ampere (SM80) | FP16, BF16 |
| Hopper (SM90) | FP16, BF16, FP8 |
| Blackwell (SM100) | FP16, BF16, TF32, I8, F8 |

## Platform Requirements

- **OS**: Linux (x86_64 and ARM64)
- **Python**: 3.10 - 3.14
- **CUDA Driver**: >= 575.51.03 (same as CUDA Toolkit 12.9)
- **Frameworks**: PyTorch, JAX (via DLPack protocol)
