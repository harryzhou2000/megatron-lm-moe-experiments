# Caching and Compilation

## JIT Executor

A **JIT Executor** is the compiled artifact returned by `cute.compile()` or created implicitly. It is callable and contains:

- Host function pointer and MLIR execution engine
- CUDA modules (loaded via `cuModuleLoad`)
- Kernel function pointers (via `cuModuleGetFunction`)
- Argument specifications for Python-to-C-ABI conversion

When called, the JIT Executor parses Python runtime arguments, converts them to C ABI types, and invokes the host function.

**Note**: Arguments annotated with `cutlass.Constexpr` are evaluated at compile time and excluded from the executor's argument spec.

## Explicit Compilation with `cute.compile`

`cute.compile` always performs compilation (bypasses cache), returning a fixed JIT Executor:

```python
@cute.jit
def add(a, b, print_result: cutlass.Constexpr):
    if print_result:
        cute.printf("Result: %d\n", a + b)
    return a + b

# Compile once
jit_executor = cute.compile(add, 1, 2, True)

# Reuse without recompilation
jit_executor(1, 2)    # output: Result: 3
jit_executor(3, 4)    # output: Result: 7
```

### Custom Caching Strategy

```python
custom_cache = {}

a = 1
custom_cache[1] = cute.compile(add_with_global_a, 2)
# a=1: result=3

a = 2
custom_cache[2] = cute.compile(add_with_global_a, 2)
# a=2: result=4

custom_cache[1](2)  # result = 3 (uses cached a=1 version)
custom_cache[2](2)  # result = 4 (uses cached a=2 version)
```

## Implicit JIT Caching

By default, CuTe DSL caches compiled JIT Executors to avoid recompilation.

**Cache key** = hash of:
- MLIR bytecode of the generated program
- All CuTe DSL Python source files
- All CuTe DSL shared libraries
- All CuTe DSL environment variables

**Cache behavior**:
- **Hit**: Compilation skipped, cached executor reused.
- **Miss**: Kernel compiled, new executor stored in cache.

```python
a = 1

@cute.jit
def add(b):
    return a + b

# Call 1: cache miss -> compile
result = add(2)  # 3

# Call 2: cache hit -> reuse
result = add(2)  # 3

a = 2
# Call 3: cache miss (IR changed) -> recompile
result = add(2)  # 4
```

### File-Based Caching

The cache can serialize to files for persistence across runs.

**Default directory**: `/tmp/{current_user}/cutlass_python_cache`

**Warning**: The default temp directory is not persistent (cleared on reboot). Set `CUTE_DSL_CACHE_DIR` for persistence.

### Caching Limitation

The MLIR program must ALWAYS be regenerated to verify the IR matches what was previously built, because dynamic factors (global variables, etc.) can change the IR. For optimal host launch latency, use `cute.compile` with manual caching.

## Programmatic Access to Compiled Artifacts

```python
compiled_foo = cute.compile(foo, ...)

# Access generated code
print(compiled_foo.__ptx__)    # PTX source
print(compiled_foo.__mlir__)   # MLIR IR

with open("foo.cubin", "wb") as f:
    f.write(compiled_foo.__cubin__)  # CUBIN binary
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `CUTE_DSL_CACHE_DIR` | `/tmp/{user}/cutlass_python_cache` | Cache directory location |
| `CUTE_DSL_DISABLE_FILE_CACHING` | `False` | Disable file caching (keep in-memory) |
