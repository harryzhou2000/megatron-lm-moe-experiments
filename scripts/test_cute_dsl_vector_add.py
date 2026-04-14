# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
CuTe DSL Vector Add Test
=========================

A simple element-wise vector addition kernel using CuTe DSL to demonstrate:
  - @cute.kernel and @cute.jit decorators
  - Layout construction and tensor creation
  - Thread indexing with cute.arch
  - Dynamic control flow with range()
  - Framework integration via from_dlpack / implicit conversion

Usage:
    # Run on compute node (requires CUDA GPU):
    python scripts/test_cute_dsl_vector_add.py

    # Or with custom size:
    python scripts/test_cute_dsl_vector_add.py --n 1048576
"""

import argparse

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


# ---------------------------------------------------------------------------
# Kernel: each thread processes one element
# ---------------------------------------------------------------------------
@cute.kernel
def vector_add_kernel(
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    n: cutlass.Int32,
):
    """Element-wise c = a + b for 1-D tensors of length n."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    idx = bidx * bdim + tidx
    if idx < n:
        c[idx] = a[idx] + b[idx]


# ---------------------------------------------------------------------------
# Host-side JIT wrapper: sets up grid/block and launches
# ---------------------------------------------------------------------------
@cute.jit
def vector_add_launch(
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    n: cutlass.Int32,
    block_size: cutlass.Constexpr,
):
    """Launch the vector_add_kernel with a 1-D grid."""
    grid_size = (n + block_size - 1) // block_size
    vector_add_kernel(a, b, c, n).launch(
        grid=[grid_size, 1, 1],
        block=[block_size, 1, 1],
    )


# ---------------------------------------------------------------------------
# Tiled vector add: each thread processes ELEMS_PER_THREAD elements
# ---------------------------------------------------------------------------
@cute.kernel
def vector_add_tiled_kernel(
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    n: cutlass.Int32,
    elems_per_thread: cutlass.Constexpr,
):
    """Tiled vector add -- each thread processes elems_per_thread elements."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    base = (bidx * bdim + tidx) * elems_per_thread
    for i in cutlass.range_constexpr(elems_per_thread):
        idx = base + i
        if idx < n:
            c[idx] = a[idx] + b[idx]


@cute.jit
def vector_add_tiled_launch(
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    n: cutlass.Int32,
    block_size: cutlass.Constexpr,
    elems_per_thread: cutlass.Constexpr,
):
    """Launch the tiled vector_add_kernel."""
    threads_total = (n + elems_per_thread - 1) // elems_per_thread
    grid_size = (threads_total + block_size - 1) // block_size
    vector_add_tiled_kernel(a, b, c, n, elems_per_thread).launch(
        grid=[grid_size, 1, 1],
        block=[block_size, 1, 1],
    )


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------
def test_vector_add(n: int = 65536, block_size: int = 256):
    """Test the basic vector add kernel."""
    print(f"[vector_add] n={n}, block_size={block_size}")
    a = torch.randn(n, dtype=torch.float32, device="cuda")
    b = torch.randn(n, dtype=torch.float32, device="cuda")
    c = torch.zeros(n, dtype=torch.float32, device="cuda")

    # Convert to CuTe tensors with dynamic layout
    a_cute = from_dlpack(a).mark_layout_dynamic()
    b_cute = from_dlpack(b).mark_layout_dynamic()
    c_cute = from_dlpack(c).mark_layout_dynamic()

    vector_add_launch(a_cute, b_cute, c_cute, n, block_size)

    # Verify
    ref = a + b
    max_err = (c - ref).abs().max().item()
    print(f"  max error: {max_err:.2e}")
    assert max_err < 1e-5, f"Vector add failed: max error {max_err}"
    print("  PASSED")


def test_vector_add_tiled(n: int = 65536, block_size: int = 256, elems_per_thread: int = 4):
    """Test the tiled vector add kernel."""
    print(f"[vector_add_tiled] n={n}, block_size={block_size}, elems_per_thread={elems_per_thread}")
    a = torch.randn(n, dtype=torch.float32, device="cuda")
    b = torch.randn(n, dtype=torch.float32, device="cuda")
    c = torch.zeros(n, dtype=torch.float32, device="cuda")

    a_cute = from_dlpack(a).mark_layout_dynamic()
    b_cute = from_dlpack(b).mark_layout_dynamic()
    c_cute = from_dlpack(c).mark_layout_dynamic()

    vector_add_tiled_launch(a_cute, b_cute, c_cute, n, block_size, elems_per_thread)

    ref = a + b
    max_err = (c - ref).abs().max().item()
    print(f"  max error: {max_err:.2e}")
    assert max_err < 1e-5, f"Tiled vector add failed: max error {max_err}"
    print("  PASSED")


def test_vector_add_fp16(n: int = 65536, block_size: int = 256):
    """Test vector add with FP16 to verify dtype handling."""
    print(f"[vector_add_fp16] n={n}, block_size={block_size}")
    a = torch.randn(n, dtype=torch.float16, device="cuda")
    b = torch.randn(n, dtype=torch.float16, device="cuda")
    c = torch.zeros(n, dtype=torch.float16, device="cuda")

    a_cute = from_dlpack(a).mark_layout_dynamic()
    b_cute = from_dlpack(b).mark_layout_dynamic()
    c_cute = from_dlpack(c).mark_layout_dynamic()

    vector_add_launch(a_cute, b_cute, c_cute, n, block_size)

    ref = a + b
    max_err = (c - ref).abs().max().item()
    print(f"  max error: {max_err:.2e}")
    assert max_err < 1e-2, f"FP16 vector add failed: max error {max_err}"
    print("  PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CuTe DSL vector add test")
    parser.add_argument("--n", type=int, default=65536, help="Vector length")
    parser.add_argument("--block-size", type=int, default=256, help="Threads per block")
    args = parser.parse_args()

    test_vector_add(n=args.n, block_size=args.block_size)
    test_vector_add_tiled(n=args.n, block_size=args.block_size, elems_per_thread=4)
    test_vector_add_fp16(n=args.n, block_size=args.block_size)
    print("\nAll vector add tests passed!")
