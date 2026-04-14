# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
CuTe DSL SM100 (Blackwell) Dense GEMM Test
============================================

Demonstrates a high-performance dense GEMM on Blackwell (SM100) GPUs using CuTe DSL:
  - tcgen05 MMA (warpgroup matrix multiply-accumulate via TMEM)
  - TMA (Tensor Memory Accelerator) for global <-> shared memory transfers
  - Multi-stage software pipelining with mbarrier synchronization
  - Cluster-level TMA multicast
  - Optional 2-CTA cooperative instructions
  - Optional TMA epilogue store

This script downloads and wraps the CUTLASS DenseGemmKernel example from:
  https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell/dense_gemm.py

The kernel flow (simplified):
  1. TMA loads A/B tiles from GMEM -> SMEM (pipelined with mbarrier full/empty)
  2. tcgen05.mma reads A/B from SMEM, accumulates into TMEM
  3. Epilogue: tcgen05.ld copies accumulator from TMEM -> RMEM
  4. Type-convert and store to GMEM (direct store or via TMA store through SMEM)

Requirements:
  - Blackwell GPU (SM100, e.g., B200/B300)
  - nvidia-cutlass-dsl >= 4.0
  - PyTorch with CUDA support
  - cuda-python (cuda.bindings)

Usage:
    # Basic FP16 GEMM (1-CTA, 128x128 tile):
    python scripts/test_cute_dsl_gemm_sm100.py

    # 2-CTA FP16 GEMM with 256x128 tile, TMA store, cluster (2,1):
    python scripts/test_cute_dsl_gemm_sm100.py \\
        --ab-dtype Float16 --c-dtype Float16 --acc-dtype Float32 \\
        --mma-tiler-mn 256,128 --cluster-shape-mn 2,1 \\
        --use-2cta --use-tma-store --mnkl 4096,4096,4096,1

    # BF16 GEMM:
    python scripts/test_cute_dsl_gemm_sm100.py \\
        --ab-dtype BFloat16 --c-dtype BFloat16 --acc-dtype Float32 \\
        --mnkl 2048,2048,2048,1

    # Run on compute node via helper:
    ssh computelab "bash ~/projects/moe/scripts/run_on_compute.sh \\
        'cd /home/scratch.hhanyu_gpu/projects/moe && \\
         python scripts/test_cute_dsl_gemm_sm100.py'"
"""

import argparse
import importlib.util
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Tuple, Type

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute import testing as cute_testing
from cutlass.cute.runtime import from_dlpack

# ---------------------------------------------------------------------------
# Verify tcgen05 is available (requires SM100-capable cutlass-dsl)
# ---------------------------------------------------------------------------
try:
    from cutlass.cute.nvgpu import tcgen05
except ImportError:
    print(
        "ERROR: cutlass.cute.nvgpu.tcgen05 not available. "
        "Requires nvidia-cutlass-dsl with SM100 support."
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Download and import the CUTLASS DenseGemmKernel example
# ---------------------------------------------------------------------------
_CUTLASS_EXAMPLE_BASE = (
    "https://raw.githubusercontent.com/NVIDIA/cutlass/"
    "v4.4.2/examples/python/CuTeDSL/blackwell"
)

_EXAMPLE_CACHE_DIR = Path(
    os.environ.get("CUTLASS_EXAMPLE_CACHE", "/tmp/cutlass_dsl_examples")
)


def _download_example(filename: str) -> Path:
    """Download a CUTLASS example file from GitHub, cached locally."""
    _EXAMPLE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = _EXAMPLE_CACHE_DIR / filename
    if dest.exists():
        print(f"  Using cached: {dest}")
        return dest
    url = f"{_CUTLASS_EXAMPLE_BASE}/{filename}"
    print(f"  Downloading: {url}")
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as e:
        raise RuntimeError(
            f"Failed to download {url}: {e}\n"
            f"You can manually place the file at {dest}"
        ) from e
    return dest


def _import_dense_gemm():
    """Import DenseGemmKernel from the CUTLASS example."""
    # Try local import first (if CUTLASS repo is present)
    try:
        from cutlass.cute.testing import DenseGemmKernel
        return DenseGemmKernel
    except (ImportError, AttributeError):
        pass

    # Download the example files from GitHub
    dense_gemm_path = _download_example("dense_gemm.py")

    # The dense_gemm module needs its directory on sys.path
    example_dir = str(dense_gemm_path.parent)
    if example_dir not in sys.path:
        sys.path.insert(0, example_dir)

    spec = importlib.util.spec_from_file_location("dense_gemm", dense_gemm_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.DenseGemmKernel


def _get_cuda_stream() -> cuda.CUstream:
    """Get a cuda.bindings.driver.CUstream wrapping PyTorch's current stream.

    This follows the same pattern as the CUTLASS example: wrap the raw
    PyTorch stream pointer in a CUstream object so the CUTLASS DSL runtime
    accepts it as the correct type.
    """
    torch_stream = torch.cuda.current_stream()
    return cuda.CUstream(torch_stream.cuda_stream)


# ---------------------------------------------------------------------------
# GEMM test
# ---------------------------------------------------------------------------
class SM100GemmTest:
    """Test harness for Blackwell SM100 dense GEMM.

    Wraps the CUTLASS DenseGemmKernel which internally implements:
      - tcgen05 warpgroup MMA (reads A/B from SMEM, writes accumulator to TMEM)
      - TMA bulk copy for GMEM <-> SMEM data movement
      - Multi-stage pipeline with mbarrier-based producer/consumer sync
      - Cluster-level TMA multicast for A and B operands
      - Optional 2-CTA cooperative MMA instructions (CtaGroup.TWO)
      - TMEM allocator for accumulator buffer management
    """

    # Mapping from string names to cutlass dtype objects
    DTYPE_MAP = {
        "Float16": (cutlass.Float16, torch.float16),
        "BFloat16": (cutlass.BFloat16, torch.bfloat16),
        "Float32": (cutlass.Float32, torch.float32),
        "TFloat32": (cutlass.TFloat32, torch.float32),
        "Int8": (cutlass.Int8, torch.int8),
    }

    def __init__(
        self,
        ab_dtype_str: str = "Float16",
        c_dtype_str: str = "Float16",
        acc_dtype_str: str = "Float32",
        mma_tiler_mn: Tuple[int, int] = (128, 128),
        cluster_shape_mn: Tuple[int, int] = (1, 1),
        use_2cta_instrs: bool = False,
        use_tma_store: bool = False,
    ):
        self.ab_dtype_str = ab_dtype_str
        self.c_dtype_str = c_dtype_str
        self.ab_cutlass_dtype, self.ab_torch_dtype = self.DTYPE_MAP[ab_dtype_str]
        self.c_cutlass_dtype, self.c_torch_dtype = self.DTYPE_MAP[c_dtype_str]
        self.acc_cutlass_dtype, _ = self.DTYPE_MAP[acc_dtype_str]
        self.mma_tiler_mn = mma_tiler_mn
        self.cluster_shape_mn = cluster_shape_mn
        self.use_2cta_instrs = use_2cta_instrs
        self.use_tma_store = use_tma_store

    def _create_tensors(
        self, m: int, n: int, k: int, l: int, a_major: str = "k", b_major: str = "k",
        c_major: str = "n",
    ):
        """Create A, B, C tensors using cutlass.torch helpers.

        Returns (a_cute, b_cute, c_cute, a_torch_cpu, b_torch_cpu, c_torch_gpu)
        matching the official CUTLASS example's tensor creation pattern.
        """
        a_torch_cpu = cutlass_torch.matrix(l, m, k, a_major == "m", self.ab_cutlass_dtype)
        b_torch_cpu = cutlass_torch.matrix(l, n, k, b_major == "n", self.ab_cutlass_dtype)
        c_torch_cpu = cutlass_torch.matrix(l, m, n, c_major == "m", self.c_cutlass_dtype)

        a_cute, _ = cutlass_torch.cute_tensor_like(
            a_torch_cpu, self.ab_cutlass_dtype, is_dynamic_layout=True, assumed_align=16,
        )
        b_cute, _ = cutlass_torch.cute_tensor_like(
            b_torch_cpu, self.ab_cutlass_dtype, is_dynamic_layout=True, assumed_align=16,
        )
        c_cute, c_torch_gpu = cutlass_torch.cute_tensor_like(
            c_torch_cpu, self.c_cutlass_dtype, is_dynamic_layout=True, assumed_align=16,
        )

        return a_cute, b_cute, c_cute, a_torch_cpu, b_torch_cpu, c_torch_gpu

    def run(
        self,
        m: int,
        n: int,
        k: int,
        l: int = 1,
        check: bool = True,
        warmup: int = 3,
        iterations: int = 10,
    ):
        """Run the SM100 GEMM kernel and optionally verify against reference."""
        print(f"\n{'='*70}")
        print(f"SM100 GEMM: M={m}, N={n}, K={k}, L={l}")
        print(
            f"  ab_dtype={self.ab_cutlass_dtype}, c_dtype={self.c_cutlass_dtype}, "
            f"acc_dtype={self.acc_cutlass_dtype}"
        )
        print(f"  mma_tiler_mn={self.mma_tiler_mn}, cluster={self.cluster_shape_mn}")
        print(f"  use_2cta={self.use_2cta_instrs}, tma_store={self.use_tma_store}")
        print(f"{'='*70}")

        # Get CUDA stream (wraps PyTorch's current stream as CUstream)
        stream = _get_cuda_stream()

        # Create tensors using cutlass.torch helpers (matches CUTLASS example)
        a_cute, b_cute, c_cute, a_cpu, b_cpu, c_gpu = self._create_tensors(m, n, k, l)

        # Import and instantiate the CUTLASS Blackwell GEMM kernel
        print("  Importing DenseGemmKernel...")
        try:
            DenseGemmKernel = _import_dense_gemm()
        except Exception as e:
            print(f"  ERROR: Could not import DenseGemmKernel: {e}")
            print("  Skipping kernel execution.")
            return

        print("  Creating kernel instance...")
        try:
            gemm_kernel = DenseGemmKernel(
                acc_dtype=self.acc_cutlass_dtype,
                use_2cta_instrs=self.use_2cta_instrs,
                mma_tiler_mn=self.mma_tiler_mn,
                cluster_shape_mn=self.cluster_shape_mn,
                use_tma_store=self.use_tma_store,
            )
        except Exception as e:
            print(f"  ERROR: Failed to create kernel: {e}")
            import traceback
            traceback.print_exc()
            return

        # Check if configuration can be implemented
        if hasattr(gemm_kernel, "can_implement"):
            if not gemm_kernel.can_implement(a_cute, b_cute, c_cute):
                print("  ERROR: Configuration is invalid/unsupported by the kernel.")
                return

        # Pre-compile via cute.compile() to eliminate per-call host overhead.
        # Without this, each __call__ recreates TMA descriptors, layouts, and
        # tiled_mma objects in Python (~120ms). cute.compile() traces the host
        # code once and produces a fast compiled callable.
        print("  Compiling kernel (cute.compile)...")
        compiled_gemm = cute.compile(gemm_kernel, a_cute, b_cute, c_cute, stream)

        # Warmup (first call triggers any remaining JIT compilation)
        print("  Running warmup...")
        for _ in range(warmup):
            compiled_gemm(a_cute, b_cute, c_cute, stream)
        torch.cuda.synchronize()

        # Benchmark using the CUTLASS testing.benchmark utility if available,
        # otherwise fall back to CUDA events.
        print(f"  Benchmarking ({iterations} iters)...")

        def generate_tensors():
            """Generate fresh JitArguments for each benchmark iteration."""
            a_t, b_t, c_t, _, _, _ = self._create_tensors(m, n, k, l)
            return cute_testing.JitArguments(a_t, b_t, c_t, stream)

        try:
            exec_time_us = cute_testing.benchmark(
                compiled_gemm,
                workspace_generator=generate_tensors,
                workspace_count=1,
                stream=stream,
                warmup_iterations=warmup,
                iterations=iterations,
            )
            avg_ms = exec_time_us / 1000.0
            flops = 2.0 * m * n * k * l
            tflops = flops / (avg_ms / 1000) / 1e12
            print(f"  avg GPU kernel time: {avg_ms:.3f} ms")
            print(f"  throughput: {tflops:.2f} TFLOP/s")
        except Exception as e:
            print(f"  testing.benchmark failed ({e}), falling back to CUDA events...")
            self._benchmark_cuda_events(compiled_gemm, a_cute, b_cute, c_cute,
                                        stream, iterations, m, n, k, l)

        # Correctness check — run once more to get a fresh result
        if check:
            compiled_gemm(a_cute, b_cute, c_cute, stream)
            torch.cuda.synchronize()

            c_result = c_gpu.cpu().float()

            # Reference: einsum mkl,nkl->mnl in FP32
            ref = torch.einsum(
                "mkl,nkl->mnl",
                a_cpu.to(dtype=torch.float32),
                b_cpu.to(dtype=torch.float32),
            ).to(self.c_torch_dtype).float()

            abs_err = (c_result - ref).abs()
            max_err = abs_err.max().item()
            mean_err = abs_err.mean().item()

            if self.ab_torch_dtype == torch.float16:
                rtol = 1e-2 * (k / 1024)
            elif self.ab_torch_dtype == torch.bfloat16:
                rtol = 5e-2 * (k / 1024)
            else:
                rtol = 1e-1

            print(f"  max abs error: {max_err:.4e}")
            print(f"  mean abs error: {mean_err:.4e}")

            if max_err < rtol * ref.abs().max().item() + 1e-5:
                print("  PASSED")
            else:
                print(f"  WARNING: error exceeds tolerance (rtol={rtol:.1e})")

    def _benchmark_cuda_events(
        self, compiled_gemm, a_cute, b_cute, c_cute, stream, iterations,
        m, n, k, l,
    ):
        """Fallback benchmark using CUDA events for GPU-only timing."""
        err, ev_start = cuda.cuEventCreate(0)
        err, ev_stop = cuda.cuEventCreate(0)

        gpu_times_ms = []
        for _ in range(iterations):
            err, = cuda.cuEventRecord(ev_start, stream)
            compiled_gemm(a_cute, b_cute, c_cute, stream)
            err, = cuda.cuEventRecord(ev_stop, stream)
            err, = cuda.cuStreamSynchronize(stream)
            err, elapsed_ms = cuda.cuEventElapsedTime(ev_start, ev_stop)
            gpu_times_ms.append(elapsed_ms)

        cuda.cuEventDestroy(ev_start)
        cuda.cuEventDestroy(ev_stop)

        gpu_times_ms.sort()
        median_ms = gpu_times_ms[len(gpu_times_ms) // 2]
        min_ms = gpu_times_ms[0]
        avg_ms = sum(gpu_times_ms) / len(gpu_times_ms)

        flops = 2.0 * m * n * k * l
        tflops = flops / (median_ms / 1000) / 1e12
        print(f"  GPU kernel time ({iterations} iters):")
        print(f"    median: {median_ms:.3f} ms  ->  {tflops:.2f} TFLOP/s")
        print(f"    min:    {min_ms:.3f} ms")
        print(f"    avg:    {avg_ms:.3f} ms")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="SM100 (Blackwell) Dense GEMM using CuTe DSL with TMA + tcgen05 WGMMA"
    )
    parser.add_argument(
        "--ab-dtype", type=str, default="Float16",
        choices=["Float16", "BFloat16", "TFloat32", "Int8"],
        help="A/B operand data type",
    )
    parser.add_argument(
        "--c-dtype", type=str, default="Float16",
        choices=["Float16", "BFloat16", "Float32"],
        help="Output C data type",
    )
    parser.add_argument(
        "--acc-dtype", type=str, default="Float32",
        choices=["Float32", "Float16"],
        help="Accumulator data type",
    )
    parser.add_argument(
        "--mma-tiler-mn", type=str, default="128,128",
        help="MMA tile shape M,N (e.g., 128,128 for 1-CTA or 256,128 for 2-CTA)",
    )
    parser.add_argument(
        "--cluster-shape-mn", type=str, default="1,1",
        help="Cluster shape M,N (e.g., 2,1). Total cluster size <= 16.",
    )
    parser.add_argument(
        "--mnkl", type=str, default="2048,2048,2048,1",
        help="Problem shape M,N,K,L",
    )
    parser.add_argument(
        "--use-2cta", action="store_true", help="Use 2-CTA tcgen05 instructions"
    )
    parser.add_argument(
        "--use-tma-store", action="store_true", help="Use TMA for epilogue store"
    )
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=10, help="Timed iterations")
    parser.add_argument("--skip-check", action="store_true", help="Skip correctness check")
    args = parser.parse_args()

    mma_tiler_mn = tuple(int(x) for x in args.mma_tiler_mn.split(","))
    cluster_shape_mn = tuple(int(x) for x in args.cluster_shape_mn.split(","))
    mnkl = tuple(int(x) for x in args.mnkl.split(","))
    assert len(mma_tiler_mn) == 2, "mma-tiler-mn must be M,N"
    assert len(cluster_shape_mn) == 2, "cluster-shape-mn must be M,N"
    assert len(mnkl) == 4, "mnkl must be M,N,K,L"

    test = SM100GemmTest(
        ab_dtype_str=args.ab_dtype,
        c_dtype_str=args.c_dtype,
        acc_dtype_str=args.acc_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        use_2cta_instrs=args.use_2cta,
        use_tma_store=args.use_tma_store,
    )

    m, n, k, l = mnkl
    test.run(
        m=m, n=n, k=k, l=l,
        check=not args.skip_check,
        warmup=args.warmup,
        iterations=args.iterations,
    )


if __name__ == "__main__":
    main()
