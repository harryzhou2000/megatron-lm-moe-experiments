# AGENTS.md — Transformer Engine (MoE fork)

This repo is a fork of NVIDIA's [TransformerEngine](https://github.com/NVIDIA/TransformerEngine) with MoE (Mixture of Experts) extensions. The main source lives in the `TE/` git submodule. It is a mixed Python + C++/CUDA project built with setuptools + CMake.

## Repository layout

```
TE/                              # TransformerEngine submodule (all source here)
  transformer_engine/
    common/                      # C/C++/CUDA core library (CMake build)
      include/transformer_engine/ # Public C API headers
      fused_router/              # MoE router CUDA kernels
      gemm/, normalization/, ...
    pytorch/                     # PyTorch bindings & Python modules
      module/                    # nn.Module wrappers (Linear, LayerNorm, etc.)
      csrc/                      # PyTorch C++/CUDA extensions
      tensor/                    # Custom tensor types (Float8, MXFP8)
    jax/                         # JAX bindings
    debug/                       # Debug/inspection utilities
  tests/pytorch/                 # PyTorch test suite (pytest)
  tests/cpp/                     # C++ unit tests (gtest + CTest)
  qa/                            # CI test scripts (L0_*, L1_*, L2_*)
  3rdparty/                      # Submodules: CUTLASS, cuDNN-frontend, googletest
```

## Build commands

```bash
# Full build (requires CUDA toolkit 12.1+, PyTorch, CMake, Ninja)
cd TE && pip install -e ".[test]"

# Set NVTE_FRAMEWORK to choose backends
NVTE_FRAMEWORK=pytorch pip install -e .

# Custom CUDA architectures
NVTE_CUDA_ARCHS="80;90" pip install -e .

# C++ unit tests
cd TE/tests/cpp && cmake -GNinja -Bbuild . && cmake --build build
ctest --test-dir build -j4
```

## Lint commands

```bash
# Python formatting (Black, line-length 100)
cd TE && python3 -m pre_commit run --all-files

# Python linting
cd TE && python3 -m pylint --recursive=y transformer_engine/common transformer_engine/pytorch transformer_engine/debug

# C++ linting
cd TE && python3 -m cpplint --root transformer_engine/common/include --recursive transformer_engine/common/include
python3 -m cpplint --recursive --exclude=transformer_engine/common/include --exclude=transformer_engine/build_tools/build transformer_engine/common
python3 -m cpplint --recursive transformer_engine/pytorch

# Full QA lint (from CI)
TE_PATH=$(pwd)/TE bash TE/qa/L0_pytorch_lint/test.sh
```

## Test commands

```bash
# Run the full PyTorch test suite
TE_PATH=$(pwd)/TE bash TE/qa/L0_pytorch_unittest/test.sh

# Run a single test file
python3 -m pytest -xvs TE/tests/pytorch/test_sanity.py

# Run a single test function
python3 -m pytest -xvs TE/tests/pytorch/test_sanity.py::test_name

# Run a single test with keyword match
python3 -m pytest -xvs TE/tests/pytorch/test_sanity.py -k "some_keyword"

# Some tests need specific env vars to disable JIT/compile
PYTORCH_JIT=0 NVTE_TORCH_COMPILE=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 NVTE_FUSED_ATTN=0 \
  python3 -m pytest -xvs TE/tests/pytorch/test_numerics.py
```

## Code style — Python

- **Formatter**: Black, line length **100**, with `--preview --enable-unstable-feature=string_processing`
- **Linter**: pylint (see `TE/pylintrc` for disabled checks)
- **Min Python version**: 3.10 (enforced by vermin pre-commit hook)
- **Copyright header** on every file (required, checked by `qa/L0_license`):
  ```python
  # Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  #
  # See LICENSE for license information.
  ```
- **Imports**: stdlib first, then third-party (`torch`, `numpy`), then TE absolute imports (`transformer_engine.common.recipe`), then relative imports (`.base`, `..distributed`). No wildcard imports; use explicit names.
- **Type hints**: Use `typing` module (`Optional`, `List`, `Tuple`, `Union`, `Dict`, `Callable`). Annotate function signatures.
- **Docstrings**: Triple-quoted module-level docstring. Function/class docstrings where public.
- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for module-level constants. Private helpers prefixed with `_`.
- **`__all__`**: Define in public API modules to control exports.
- **Error handling**: Use `assert` for build-time checks. Use `warnings.warn()` with `DeprecationWarning`/`RuntimeWarning` for soft failures. Raise explicit exceptions (`ValueError`, `RuntimeError`) for invalid arguments.
- **Environment variables**: Access via `os.getenv("NVTE_*")` or `os.environ.get()`. Key vars: `NVTE_FRAMEWORK`, `NVTE_FUSED_ATTN`, `NVTE_TORCH_COMPILE`, `NVTE_FLASH_ATTN`.

## Code style — C++/CUDA

- **Standard**: C++17 / CUDA 17
- **Style**: Google C++ Style Guide, enforced by `.clang-format` (based on Google) and `cpplint`
- **Line length**: 100 characters
- **Indent**: 2 spaces (C++), no tabs
- **Braces**: Attach style (K&R). No brace wrapping after control statements.
- **Pointers**: Left-aligned (`void* ptr`)
- **Copyright header** (required):
  ```cpp
  /*************************************************************************
   * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   *
   * See LICENSE for license information.
   ************************************************************************/
  ```
- **Include order**: System headers (`<cuda_runtime.h>`), then TE public headers (`<transformer_engine/...>`), then local headers (`"../common.h"`)
- **Namespaces**: `transformer_engine` for all library code. No indentation inside namespaces.
- **Naming**: `snake_case` for functions/variables, `PascalCase` for types/classes, `kPascalCase` for constants (e.g., `kThreadsPerWarp`). Kernel names: `snake_case_kernel`.
- **Error checking**: Use `NVTE_CHECK(cond, msg...)` macro for runtime assertions. Check CUDA errors.
- **Templates**: CUDA kernel templates use `<typename DataType, ...>`. Break template declarations.

## Commit and PR conventions

- Write commit titles in **imperative mood** (e.g., "Add cutlass grouped gemm support")
- Prefix scope in brackets: `[PyTorch]`, `[Common]`, `[JAX]`, `[PyTorch Debug]`
- Sign-off required: `git commit -s -m "message"`
- Keep PRs focused on a single concern; avoid commented-out code

## Key environment variables

| Variable | Purpose |
|---|---|
| `NVTE_FRAMEWORK` | `pytorch`, `jax`, or `all` — selects which framework extensions to build |
| `NVTE_FUSED_ATTN` | `0`/`1` — enable/disable fused attention |
| `NVTE_FLASH_ATTN` | `0`/`1` — enable/disable Flash Attention |
| `NVTE_TORCH_COMPILE` | `0`/`1` — enable/disable torch.compile |
| `NVTE_CUDA_ARCHS` | Semicolon-separated CUDA architectures |
| `NVTE_UB_WITH_MPI` | `1` to enable MPI-based userbuffer comm (requires `MPI_HOME`) |
| `TE_PATH` | Root of TE source tree (used by QA scripts, defaults to `/opt/transformerengine`) |

## Testing notes

- Tests require a CUDA GPU. Most tests assume Hopper (sm_90) or later for FP8.
- Test files use `pytest`. Common test utilities are in `TE/tests/pytorch/utils.py`.
- Distributed tests use `torchrun` or `pytest` with MPI and are in `tests/pytorch/distributed/`.
- The `qa/` directory contains CI-ready scripts; set `TE_PATH` to your TE checkout to run locally.
