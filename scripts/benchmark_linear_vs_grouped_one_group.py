#!/usr/bin/env python3
"""Compare TE Linear with TE GroupedLinear using one BF16 CUDA split.

The grouped call uses the public module API:

    x:        [M, K] BF16 CUDA tensor
    m_splits: [1] int64 CUDA tensor containing M
    weight0:  [N, K] BF16 CUDA parameter
    output:   [M, N] BF16 CUDA tensor

The FLOP count for every call is 2*M*N*K. Both eager submission and CUDA-graph
replay are reported. ``use_grouped_tensor=True`` explicitly requests the native
device-resident split-size path.
"""

from __future__ import annotations

import argparse
import inspect
import json
import statistics
from dataclasses import asdict, dataclass
from typing import Callable

import torch


@dataclass
class Result:
    shape_name: str
    recipe: str
    backend: str
    mode: str
    m: int
    n: int
    k: int
    latency_us: float
    tflops: float


def _time_eager(fn: Callable[[], torch.Tensor], warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1e3 / iterations


def _time_graph(fn: Callable[[], torch.Tensor], warmup: int, iterations: int) -> float:
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(side_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_output = fn()
    graph.replay()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    del static_output, graph
    return start.elapsed_time(end) * 1e3 / iterations


def _measure(
    results: list[Result],
    shape_name: str,
    recipe: str,
    backend: str,
    mode: str,
    fn: Callable[[], torch.Tensor],
    m: int,
    n: int,
    k: int,
    warmup: int,
    iterations: int,
) -> None:
    samples = []
    timer = _time_graph if mode == "graph" else _time_eager
    for _ in range(3):
        samples.append(timer(fn, warmup, iterations))
    latency_us = statistics.median(samples)
    tflops = 2.0 * m * n * k / (latency_us * 1e-6) / 1e12
    results.append(
        Result(
            shape_name=shape_name,
            recipe=recipe,
            backend=backend,
            mode=mode,
            m=m,
            n=n,
            k=k,
            latency_us=latency_us,
            tflops=tflops,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--m-values", default="2048,4096,8192,16384")
    args = parser.parse_args()

    import transformer_engine
    import transformer_engine.pytorch as te
    import transformer_engine_torch as tex
    from transformer_engine.common.recipe import MXFP8BlockScaling
    from transformer_engine.pytorch.module.grouped_linear import (
        is_module_grouped_tensor_path_supported,
    )
    from transformer_engine.pytorch.quantization import autocast

    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    torch.set_grad_enabled(False)
    dtype = torch.bfloat16
    device = torch.device("cuda")

    m_values = [int(value) for value in args.m_values.split(",")]
    shapes = (
        ("k3_shared_fc1", 3584, 6144),
        ("k3_shared_fc2", 3072, 3584),
    )

    grouped_path_supported = {
        "bf16": is_module_grouped_tensor_path_supported(None, dtype),
        "mxfp8": is_module_grouped_tensor_path_supported(MXFP8BlockScaling(), dtype),
    }

    metadata = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "transformer_engine_version": transformer_engine.__version__,
        "transformer_engine_path": transformer_engine.__file__,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": torch.cuda.get_device_capability(0),
        "cublaslt_version": tex.get_cublasLt_version(),
        "grouped_tensor_path_supported": grouped_path_supported,
        "grouped_linear_forward_signature": str(inspect.signature(te.GroupedLinear.forward)),
        "linear_forward_signature": str(inspect.signature(te.Linear.forward)),
        "dtype": str(dtype),
        "recipes": ["bf16", "mxfp8"],
        "timing_scope": "TE module forward; MXFP8 input quantization included, weight cache reused",
        "warmup": args.warmup,
        "iterations": args.iterations,
        "m_values": m_values,
        "shapes": [
            {"name": name, "input": ["M", k], "weight": [n, k], "output": ["M", n]}
            for name, k, n in shapes
        ],
        "grouped_api": {
            "constructor": (
                "te.GroupedLinear(num_gemms=1, in_features=K, out_features=N, "
                "bias=False, use_grouped_tensor=True)"
            ),
            "input": "BF16 CUDA [M,K]",
            "m_splits": "int64 CUDA [1] with value [M]",
            "weight0": "BF16 CUDA [N,K]",
            "output": "BF16 CUDA [M,N]",
        },
    }
    print("METADATA " + json.dumps(metadata, sort_keys=True), flush=True)
    if not all(grouped_path_supported.values()):
        raise RuntimeError("TE reports that the device-resident grouped-tensor path is unsupported")

    results: list[Result] = []
    for recipe_name in ("bf16", "mxfp8"):
        recipe = None if recipe_name == "bf16" else MXFP8BlockScaling()
        for shape_name, k, n in shapes:
            for m in m_values:
                x = torch.randn((m, k), device=device, dtype=dtype)
                te_linear = te.Linear(k, n, bias=False, device=device, params_dtype=dtype)
                grouped_linear = te.GroupedLinear(
                    num_gemms=1,
                    in_features=k,
                    out_features=n,
                    bias=False,
                    device=device,
                    params_dtype=dtype,
                    use_grouped_tensor=True,
                )
                with torch.no_grad():
                    grouped_linear.weight0.copy_(te_linear.weight)
                m_splits = torch.tensor([m], device=device, dtype=torch.int64)

                fns = {
                    "te_linear": lambda: te_linear(x, is_first_microbatch=False),
                    "te_grouped_linear_g1": lambda: grouped_linear(
                        x, m_splits, is_first_microbatch=False
                    ),
                }

                quantization_context = (
                    torch.autocast(device_type="cuda", dtype=dtype)
                    if recipe is None
                    else autocast(enabled=True, recipe=recipe)
                )
                with quantization_context:
                    # Prime both weight caches before timing. Timed calls use the cached weights.
                    te_linear(x, is_first_microbatch=True)
                    grouped_linear(x, m_splits, is_first_microbatch=True)
                    reference = torch.nn.functional.linear(x, te_linear.weight)
                    te_output = fns["te_linear"]()
                    grouped_output = fns["te_grouped_linear_g1"]()
                    correctness = {
                        "shape": shape_name,
                        "recipe": recipe_name,
                        "m": m,
                        "te_linear_max_abs": (te_output - reference).abs().max().item(),
                        "grouped_max_abs": (grouped_output - reference).abs().max().item(),
                        "te_vs_grouped_max_abs": (te_output - grouped_output).abs().max().item(),
                        "m_splits_device": str(m_splits.device),
                        "m_splits_dtype": str(m_splits.dtype),
                        "m_splits_shape": list(m_splits.shape),
                        "m_splits_value": m,
                    }
                    print("CORRECTNESS " + json.dumps(correctness, sort_keys=True), flush=True)

                    for mode in ("eager", "graph"):
                        for backend, fn in fns.items():
                            _measure(
                                results,
                                shape_name,
                                recipe_name,
                                backend,
                                mode,
                                fn,
                                m,
                                n,
                                k,
                                args.warmup,
                                args.iterations,
                            )
                            print(
                                "RESULT " + json.dumps(asdict(results[-1]), sort_keys=True),
                                flush=True,
                            )

                del x, te_linear, grouped_linear, m_splits
                torch.cuda.empty_cache()

    print("CSV shape,recipe,backend,mode,M,N,K,latency_us,tflops")
    for result in results:
        print(
            f"CSV {result.shape_name},{result.recipe},{result.backend},{result.mode},"
            f"{result.m},{result.n},{result.k},{result.latency_us:.3f},{result.tflops:.3f}"
        )


if __name__ == "__main__":
    main()
