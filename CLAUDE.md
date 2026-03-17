# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fork of NVIDIA's TransformerEngine with Mixture of Experts (MoE) extensions. The main source lives in the `TE/` git submodule. Mixed Python + C++/CUDA project built with setuptools + CMake.

## Build Commands

```bash
# Full build (requires CUDA 12.1+, PyTorch, CMake, Ninja)
cd TE && pip install -e ".[test]"

# PyTorch-only build
NVTE_FRAMEWORK=pytorch pip install -e .

# Custom CUDA architectures
NVTE_CUDA_ARCHS="80;90" pip install -e .

# C++ unit tests
cd TE/tests/cpp && cmake -GNinja -Bbuild . && cmake --build build
ctest --test-dir build -j4
```

## Test Commands

```bash
# Full PyTorch test suite
TE_PATH=$(pwd)/TE bash TE/qa/L0_pytorch_unittest/test.sh

# Single test file
python3 -m pytest -xvs TE/tests/pytorch/test_fused_router.py

# Single test function
python3 -m pytest -xvs TE/tests/pytorch/test_fused_router.py::test_topk_sigmoid

# Keyword match
python3 -m pytest -xvs TE/tests/pytorch/test_fused_router.py -k "some_keyword"

# Some tests need env vars
PYTORCH_JIT=0 NVTE_TORCH_COMPILE=0 NVTE_FUSED_ATTN=0 \
  python3 -m pytest -xvs TE/tests/pytorch/test_numerics.py
```

## Lint Commands

```bash
# Python formatting + linting (Black, line-length 100)
cd TE && python3 -m pre_commit run --all-files

# Python linting only
cd TE && python3 -m pylint --recursive=y transformer_engine/common transformer_engine/pytorch

# C++ linting
cd TE && python3 -m cpplint --root transformer_engine/common/include --recursive transformer_engine/common/include

# Full QA lint
TE_PATH=$(pwd)/TE bash TE/qa/L0_pytorch_lint/test.sh
```

## Architecture

```
TE/transformer_engine/
  common/                        # C/C++/CUDA core (CMake build)
    include/transformer_engine/  # Public C API headers
    fused_router/                # MoE router CUDA kernels (topk, aux loss)
  pytorch/                       # PyTorch bindings
    router.py                    # Python API: fused_topk_with_score_function()
    module/                      # nn.Module wrappers (Linear, LayerNorm, etc.)
    tensor/                      # Custom tensor types (Float8, MXFP8)
  jax/                           # JAX bindings
TE/tests/pytorch/                # pytest test suite
scripts/                         # Custom test/benchmark scripts
notes/                           # Design documentation
```

**Call chain**: Python (`router.py`) -> C++ binding (`cpp_extensions/`) -> C API (`fused_router.h`) -> CUDA kernel (`fused_router/*.cu`)

## Code Style

**Python**: Black formatter, line length 100, `--preview --enable-unstable-feature=string_processing`. Min Python 3.10. pylint for linting (see `TE/pylintrc`). `snake_case` functions, `PascalCase` classes, `UPPER_SNAKE_CASE` constants.

**C++/CUDA**: C++17, Google Style, 2-space indent, 100-char lines. `snake_case` functions, `PascalCase` types, `kPascalCase` constants. Kernel names: `snake_case_kernel`. Error checking via `NVTE_CHECK()`.

**Imports**: stdlib -> third-party (torch, numpy) -> TE absolute -> relative. No wildcard imports.

**Copyright header required** on every file (checked by CI):
- Python: `# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.`
- C++: Block comment with same text

## Commit Conventions

- Imperative mood: "Add cutlass grouped gemm support"
- Prefix scope: `[PyTorch]`, `[Common]`, `[JAX]`
- Sign-off required: `git commit -s`

## Key Environment Variables

| Variable | Purpose |
|---|---|
| `NVTE_FRAMEWORK` | `pytorch`, `jax`, or `all` — selects framework extensions |
| `NVTE_CUDA_ARCHS` | Semicolon-separated CUDA architectures (e.g., "80;90") |
| `NVTE_FUSED_ATTN` | `0`/`1` — enable/disable fused attention |
| `NVTE_TORCH_COMPILE` | `0`/`1` — enable/disable torch.compile |
| `TE_PATH` | Root of TE source tree (used by QA scripts) |

## Testing Notes

- Tests require a CUDA GPU. Most assume Hopper (sm_90) or later for FP8.
- Test utilities in `TE/tests/pytorch/utils.py`.
- Distributed tests use `torchrun` and are in `tests/pytorch/distributed/`.
- QA scripts in `TE/qa/` are CI-ready; set `TE_PATH` to run locally.
