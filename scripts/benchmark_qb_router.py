#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Benchmark Kimi K3 QB router implementations against an unfused PyTorch reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Callable, Optional

import torch

from transformer_engine.pytorch.router import fused_topk_with_score_function


def _benchmark_cuda(
    fn: Callable[[], object], warmup: int, iterations: int, repeats: int
) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    repeats = min(repeats, iterations)
    iterations_per_repeat = [iterations // repeats] * repeats
    for index in range(iterations % repeats):
        iterations_per_repeat[index] += 1
    samples = []
    for repeat_iterations in iterations_per_repeat:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeat_iterations):
            fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / repeat_iterations)
    return {
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def _compact_topk_plus_one(
    scores: torch.Tensor, indices: torch.Tensor, topk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    cutoff = scores.min(dim=-1).values
    cutoff_candidates = indices.masked_fill(scores != cutoff.unsqueeze(1), -1)
    dropped_expert = cutoff_candidates.max(dim=-1).values
    selected = indices[indices != dropped_expert.unsqueeze(1)].reshape(-1, topk)
    return selected, cutoff


def _pytorch_qb(
    logits: torch.Tensor,
    topk: int,
    expert_bias: torch.Tensor,
    histogram: torch.Tensor,
    bin_bounds: torch.Tensor,
    dense_indices: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens, num_experts = logits.shape
    num_bins = histogram.shape[1]
    raw_scores = torch.sigmoid(logits.float())
    topk_plus_one_scores, topk_plus_one_indices = torch.topk(
        raw_scores + expert_bias, k=topk + 1, dim=-1
    )
    topk_indices, cutoff = _compact_topk_plus_one(topk_plus_one_scores, topk_plus_one_indices, topk)
    selected_scores = torch.gather(raw_scores, 1, topk_indices)
    selected_probs = selected_scores / (selected_scores.sum(dim=-1, keepdim=True) + 1.0e-20)
    probs = torch.zeros_like(raw_scores).scatter(1, topk_indices, selected_probs)

    lower, upper = bin_bounds
    bin_scale = num_bins / (upper - lower)
    bin_indices = torch.floor((cutoff.unsqueeze(1) - raw_scores - lower) * bin_scale).to(
        torch.int64
    )
    bin_indices.clamp_(0, num_bins - 1)
    expert_offsets = torch.arange(num_experts, device=logits.device, dtype=torch.int64) * num_bins
    counts = torch.bincount(
        (bin_indices + expert_offsets).reshape(-1),
        minlength=num_experts * num_bins,
    )
    histogram.add_(counts.reshape(num_experts, num_bins).to(torch.int32))

    if dense_indices:
        routing_output = topk_indices.to(torch.int16)
    else:
        routing_output = torch.zeros(
            num_tokens, num_experts, device=logits.device, dtype=torch.bool
        ).scatter(1, topk_indices, True)
    return probs.to(logits.dtype), routing_output


def _routing_map(
    routing_output: torch.Tensor, num_experts: int, dense_indices: bool
) -> torch.Tensor:
    if not dense_indices:
        return routing_output
    return torch.zeros(
        routing_output.shape[0],
        num_experts,
        device=routing_output.device,
        dtype=torch.bool,
    ).scatter(1, routing_output.to(torch.int64), True)


def _make_topk_indices(num_tokens: int, topk: int, dense_indices: bool) -> Optional[torch.Tensor]:
    if not dense_indices:
        return None
    return torch.empty(num_tokens, topk, device="cuda", dtype=torch.int16)


def _run_case(
    num_tokens: int,
    num_experts: int,
    topk: int,
    num_bins: int,
    dense_indices: bool,
    warmup: int,
    iterations: int,
    repeats: int,
) -> dict[str, object]:
    torch.manual_seed(42 + num_tokens + int(dense_indices))
    logits = torch.randn(num_tokens, num_experts, device="cuda", dtype=torch.float32)
    expert_bias = torch.linspace(-0.2, 0.2, num_experts, device="cuda", dtype=torch.float32)
    bin_bounds = torch.tensor([-1.2, 1.2], device="cuda", dtype=torch.float32)
    pytorch_histogram = torch.zeros(num_experts, num_bins, device="cuda", dtype=torch.int32)
    two_kernel_histogram = torch.zeros_like(pytorch_histogram)
    fused_atomic_histogram = torch.zeros_like(pytorch_histogram)
    baseline_indices = _make_topk_indices(num_tokens, topk, dense_indices)
    two_kernel_indices = _make_topk_indices(num_tokens, topk, dense_indices)
    fused_atomic_indices = _make_topk_indices(num_tokens, topk, dense_indices)

    def run_pytorch():
        return _pytorch_qb(
            logits,
            topk,
            expert_bias,
            pytorch_histogram,
            bin_bounds,
            dense_indices,
        )

    def run_te_baseline():
        return fused_topk_with_score_function(
            logits=logits,
            topk=topk,
            use_pre_softmax=False,
            num_groups=None,
            group_topk=None,
            scaling_factor=None,
            score_function="sigmoid",
            expert_bias=expert_bias,
            topk_indices=baseline_indices,
        )

    def run_te_qb(mode, histogram, topk_indices):
        return fused_topk_with_score_function(
            logits=logits,
            topk=topk,
            use_pre_softmax=False,
            num_groups=None,
            group_topk=None,
            scaling_factor=None,
            score_function="sigmoid",
            expert_bias=expert_bias,
            topk_indices=topk_indices,
            qb_histogram=histogram,
            qb_bin_bounds=bin_bounds,
            qb_histogram_mode=mode,
        )

    pytorch_output = run_pytorch()
    baseline_output = run_te_baseline()
    two_kernel_output = run_te_qb("two_kernel", two_kernel_histogram, two_kernel_indices)
    fused_atomic_output = run_te_qb("fused_atomic", fused_atomic_histogram, fused_atomic_indices)
    pytorch_map = _routing_map(pytorch_output[1], num_experts, dense_indices)
    for output in (baseline_output, two_kernel_output, fused_atomic_output):
        torch.testing.assert_close(output[0], pytorch_output[0])
        torch.testing.assert_close(_routing_map(output[1], num_experts, dense_indices), pytorch_map)
    torch.testing.assert_close(two_kernel_histogram, pytorch_histogram)
    torch.testing.assert_close(fused_atomic_histogram, pytorch_histogram)

    pytorch_timing = _benchmark_cuda(run_pytorch, warmup, iterations, repeats)
    baseline_timing = _benchmark_cuda(run_te_baseline, warmup, iterations, repeats)
    two_kernel_timing = _benchmark_cuda(
        lambda: run_te_qb("two_kernel", two_kernel_histogram, two_kernel_indices),
        warmup,
        iterations,
        repeats,
    )
    fused_atomic_timing = _benchmark_cuda(
        lambda: run_te_qb("fused_atomic", fused_atomic_histogram, fused_atomic_indices),
        warmup,
        iterations,
        repeats,
    )
    pytorch_ms = pytorch_timing["median_ms"]
    baseline_ms = baseline_timing["median_ms"]
    two_kernel_ms = two_kernel_timing["median_ms"]
    fused_atomic_ms = fused_atomic_timing["median_ms"]
    result = {
        "num_tokens": num_tokens,
        "num_experts": num_experts,
        "topk": topk,
        "num_bins": num_bins,
        "routing": "dense_int16" if dense_indices else "bytemap",
        "pytorch_unfused": pytorch_timing,
        "te_baseline": baseline_timing,
        "qb_two_kernel": two_kernel_timing,
        "qb_fused_atomic": fused_atomic_timing,
        "fused_vs_pytorch": fused_atomic_ms / pytorch_ms,
        "fused_vs_te_baseline": fused_atomic_ms / baseline_ms,
        "fused_vs_two_kernel": fused_atomic_ms / two_kernel_ms,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[256, 1024, 4096, 8192, 16384])
    parser.add_argument("--num-experts", type=int, default=896)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--num-bins", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--routing",
        choices=("bytemap", "dense_int16", "both"),
        default="both",
    )
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    if args.topk <= 0 or args.topk >= args.num_experts:
        raise ValueError("topk must be in [1, num_experts).")
    if args.warmup < 0 or args.iterations <= 0 or args.repeats <= 0:
        raise ValueError("warmup must be nonnegative; iterations and repeats must be positive.")

    dense_modes = {
        "bytemap": [False],
        "dense_int16": [True],
        "both": [False, True],
    }[args.routing]
    results = [
        _run_case(
            num_tokens,
            args.num_experts,
            args.topk,
            args.num_bins,
            dense_indices,
            args.warmup,
            args.iterations,
            args.repeats,
        )
        for dense_indices in dense_modes
        for num_tokens in args.tokens
    ]
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
