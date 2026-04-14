# Debugging

## Two-Level Printing

| Function | Stage | Purpose |
|---|---|---|
| `print()` | Meta-stage (compile time) | Inspect shapes, strides, layouts, tile sizes, compile-time decisions |
| `cute.printf()` | Object-stage (GPU runtime) | Print actual tensor values, runtime diagnostics |

```python
@cute.jit
def foo(a: cutlass.Float32, b: cutlass.Float32):
    result = a + b
    print("[meta-stage] result =", result)          # <Float32 proxy>
    cute.printf("[object-stage] result = %f\n", result)  # 7.000000
```

If both operands are `Constexpr`, `print()` shows the actual computed value:
```python
@cute.jit
def foo(b: cutlass.Constexpr):
    a = 2.0
    result = a + b
    print("[meta-stage] result =", result)  # 7.0 (computed at compile time)
```

## Source Code Correlation

Enable Python-to-PTX/SASS line mapping for profiling tools (NSight Compute, etc.):

```bash
# Global enable:
export CUTE_DSL_LINEINFO=1

# Per-kernel: use JIT compilation options (see JIT Compilation Options docs)
```

## Logging

```bash
# Enable console logging
export CUTE_DSL_LOG_TO_CONSOLE=1

# Log to file
export CUTE_DSL_LOG_TO_FILE=my_log.txt

# Control verbosity
export CUTE_DSL_LOG_LEVEL=20
```

| Level | Description |
|---|---|
| 0 | Disabled |
| 10 | Debug (default) |
| 20 | Info |
| 30 | Warning |
| 40 | Error |
| 50 | Critical |

## Dumping Generated Code

### IR (MLIR)

```bash
export CUTE_DSL_PRINT_IR=1    # Print IR to stdout
export CUTE_DSL_KEEP_IR=1     # Save IR to file
```

### PTX and CUBIN

```bash
export CUTE_DSL_KEEP_PTX=1    # Save .ptx file
export CUTE_DSL_KEEP_CUBIN=1  # Save .cubin file
```

Get SASS from CUBIN:
```bash
nvdisasm your_dsl_code.cubin > your_dsl_code.sass
```

### Change Dump Directory

```bash
export CUTE_DSL_DUMP_DIR=/path/to/dump/dir
```

### Programmatic Access

```python
compiled_foo = cute.compile(foo, ...)
print(compiled_foo.__ptx__)
print(compiled_foo.__mlir__)
with open("foo.cubin", "wb") as f:
    f.write(compiled_foo.__cubin__)
```

## Kernel Functional Debugging

### Compute-Sanitizer

Detect memory errors and race conditions:
```bash
compute-sanitizer --tool memcheck python your_dsl_code.py
compute-sanitizer --tool racecheck python your_dsl_code.py
```

### Handling Hung Kernels

When a kernel becomes unresponsive and `Ctrl+C` fails:
1. Press `Ctrl+Z` to suspend the process.
2. Kill the suspended process:
```bash
kill -9 $(jobs -p | tail -1)
```

### Custom Kernel Name Prefix

Attach runtime info to kernel names for profiling:
```python
@cute.kernel
def kernel(arg1, arg2):
    ...

@cute.jit
def launch():
    kernel.set_name_prefix("rank0_gemm_fp16_128x128")
    kernel(arg1, arg2).launch(grid=[1,1,1], block=[128,1,1])
    # Kernel name in profiler: "rank0_gemm_fp16_128x128_xxx"
```

## All Debugging Environment Variables

| Variable | Default | Description |
|---|---|---|
| `CUTE_DSL_LOG_TO_CONSOLE` | `0` | Enable console logging |
| `CUTE_DSL_LOG_TO_FILE` | (none) | Log file path |
| `CUTE_DSL_LOG_LEVEL` | `10` | Log verbosity |
| `CUTE_DSL_PRINT_IR` | `0` | Print MLIR IR to stdout |
| `CUTE_DSL_KEEP_IR` | `0` | Save IR to file |
| `CUTE_DSL_KEEP_PTX` | `0` | Save PTX to file |
| `CUTE_DSL_KEEP_CUBIN` | `0` | Save CUBIN to file |
| `CUTE_DSL_LINEINFO` | `0` | Generate debug line info |
| `CUTE_DSL_DUMP_DIR` | `.` | Dump directory |
