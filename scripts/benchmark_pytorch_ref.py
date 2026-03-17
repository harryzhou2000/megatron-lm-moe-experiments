#!/usr/bin/env python3
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
Per-operation performance breakdown of the PyTorch reference MoE routing path.

Measures each internal operation (softmax, sigmoid, topk, gather, scatter,
normalization) independently with CUDA events so we can see exactly where time
is spent versus the fused kernel.

Usage
-----
  # Default: sweep representative configs
  python scripts/benchmark_pytorch_ref.py

  # Single config matching the fused kernel test
  python scripts/benchmark_pytorch_ref.py \
      --num-tokens 4096 --num-experts 2304 --topk 36 \
      --score-function softmax --use-pre-softmax

  # Sigmoid path with bias
  python scripts/benchmark_pytorch_ref.py \
      --num-tokens 4096 --num-experts 2304 --topk 36 \
      --score-function sigmoid --enable-bias

  # Custom dtype
  python scripts/benchmark_pytorch_ref.py --dtype bf16

  # More iterations for stable numbers
  python scripts/benchmark_pytorch_ref.py --warmup 50 --iters 500
"""

import argparse
import sys
from typing import Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DTYPE_MAP = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}


def parse_dtype(s: str) -> torch.dtype:
    s = s.strip().lower()
    if s not in DTYPE_MAP:
        raise argparse.ArgumentTypeError(
            f"Unknown dtype '{s}'. Choose from: {', '.join(DTYPE_MAP.keys())}"
        )
    return DTYPE_MAP[s]


def print_gpu_info() -> None:
    """Print current CUDA device properties."""
    if not torch.cuda.is_available():
        print("ERROR: No CUDA device available.")
        sys.exit(1)
    dev = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(dev)
    print("=" * 80)
    print(f"Device {dev}: {props.name}")
    print(f"  Compute capability : {props.major}.{props.minor}")
    print(f"  SMs                : {props.multi_processor_count}")
    print(f"  Global memory      : {props.total_mem / (1024**3):.1f} GiB"
          if hasattr(props, "total_mem") else
          f"  Global memory      : {props.total_memory / (1024**3):.1f} GiB")
    print(f"  CUDA version       : {torch.version.cuda}")
    print(f"  PyTorch version    : {torch.__version__}")
    print("=" * 80)


# ---------------------------------------------------------------------------
# CUDA-event-based microbenchmark helper
# ---------------------------------------------------------------------------

def _bench_op(fn, warmup: int, iters: int) -> float:
    """
    Time a callable `fn()` using CUDA events.
    Returns average elapsed time in milliseconds.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / iters


# ---------------------------------------------------------------------------
# Individual op benchmarks for SOFTMAX path
# ---------------------------------------------------------------------------

def bench_softmax_path(
    logits: torch.Tensor,
    topk: int,
    use_pre_softmax: bool,
    scaling_factor: float,
    warmup: int,
    iters: int,
) -> Dict[str, float]:
    """
    Break down the softmax routing path into individual ops.

    Pre-softmax path:
      1. softmax(logits)       -> scores           [T, E]
      2. topk(scores, k)       -> (vals, indices)   [T, K]
      3. (optional) scale       -> probs * factor
      4. scatter(probs, indices) -> output           [T, E]

    Post-softmax path:
      1. topk(logits, k)       -> (vals, indices)   [T, K]
      2. softmax(vals, dim=-1)  -> probs             [T, K]
      3. (optional) scale       -> probs * factor
      4. scatter(probs, indices) -> output           [T, E]
    """
    num_tokens, num_experts = logits.shape
    results = {}

    if use_pre_softmax:
        # --- Step 1: softmax over full E dimension ---
        results["softmax_full_E"] = _bench_op(
            lambda: torch.softmax(logits, dim=-1, dtype=torch.float32).to(logits.dtype),
            warmup, iters,
        )
        scores = torch.softmax(logits, dim=-1, dtype=torch.float32).to(logits.dtype)

        # --- Step 2: topk on scores ---
        results["topk"] = _bench_op(
            lambda: torch.topk(scores, k=topk, dim=1),
            warmup, iters,
        )
        probs, top_indices = torch.topk(scores, k=topk, dim=1)

        # --- Step 3: scaling ---
        if scaling_factor != 1.0:
            results["scale"] = _bench_op(
                lambda: probs * scaling_factor,
                warmup, iters,
            )
        else:
            results["scale"] = 0.0

        # --- Step 4: scatter into [T, E] output ---
        output_template = torch.zeros_like(logits)
        results["scatter"] = _bench_op(
            lambda: output_template.zero_().scatter_(1, top_indices, probs),
            warmup, iters,
        )

        # --- Step 5: build topk_map (int32 scatter) ---
        map_template = torch.zeros_like(logits, dtype=torch.int32)
        ones = torch.ones_like(top_indices, dtype=torch.int32)
        results["topk_map_scatter"] = _bench_op(
            lambda: map_template.zero_().scatter_(1, top_indices, ones).bool(),
            warmup, iters,
        )

    else:
        # Post-softmax: topk on raw logits, then softmax over K
        # --- Step 1: topk on logits ---
        results["topk"] = _bench_op(
            lambda: torch.topk(logits, k=topk, dim=1),
            warmup, iters,
        )
        top_scores, top_indices = torch.topk(logits, k=topk, dim=1)

        # --- Step 2: softmax over K dimension ---
        results["softmax_over_K"] = _bench_op(
            lambda: torch.softmax(top_scores, dim=-1, dtype=torch.float32).to(logits.dtype),
            warmup, iters,
        )
        probs = torch.softmax(top_scores, dim=-1, dtype=torch.float32).to(logits.dtype)

        # --- Step 3: scaling ---
        if scaling_factor != 1.0:
            results["scale"] = _bench_op(
                lambda: probs * scaling_factor,
                warmup, iters,
            )
        else:
            results["scale"] = 0.0

        # --- Step 4: scatter ---
        output_template = torch.zeros_like(logits)
        results["scatter"] = _bench_op(
            lambda: output_template.zero_().scatter_(1, top_indices, probs),
            warmup, iters,
        )

        # --- Step 5: topk_map ---
        map_template = torch.zeros_like(logits, dtype=torch.int32)
        ones = torch.ones_like(top_indices, dtype=torch.int32)
        results["topk_map_scatter"] = _bench_op(
            lambda: map_template.zero_().scatter_(1, top_indices, ones).bool(),
            warmup, iters,
        )

    # --- End-to-end reference ---
    def _reference_e2e():
        if use_pre_softmax:
            s = torch.softmax(logits, dim=-1, dtype=torch.float32).to(logits.dtype)
            p, idx = torch.topk(s, k=topk, dim=1)
        else:
            s, idx = torch.topk(logits, k=topk, dim=1)
            p = torch.softmax(s, dim=-1, dtype=torch.float32).to(logits.dtype)
        if scaling_factor != 1.0:
            p = p * scaling_factor
        out = torch.zeros_like(logits).scatter_(1, idx, p)
        m = torch.zeros_like(logits, dtype=torch.int32).scatter_(1, idx, 1).bool()
        return out, m

    results["end_to_end"] = _bench_op(_reference_e2e, warmup, iters)

    return results


# ---------------------------------------------------------------------------
# Individual op benchmarks for SIGMOID path
# ---------------------------------------------------------------------------

def bench_sigmoid_path(
    logits: torch.Tensor,
    topk: int,
    enable_bias: bool,
    scaling_factor: float,
    warmup: int,
    iters: int,
) -> Dict[str, float]:
    """
    Break down the sigmoid routing path into individual ops.

    Steps:
      1. sigmoid(logits)                    -> scores         [T, E]
      2. (optional) scores + expert_bias    -> routing_scores  [T, E]
      3. topk(routing_scores, k)            -> (vals, indices) [T, K]
      4. gather(scores, indices)            -> selected_scores [T, K]
      5. normalize: selected / sum(selected) -> probs          [T, K]
      6. (optional) scale
      7. scatter(probs, indices)             -> output         [T, E]
    """
    num_tokens, num_experts = logits.shape
    results = {}

    expert_bias = None
    if enable_bias:
        expert_bias = torch.randn(num_experts, dtype=logits.dtype, device="cuda") * 0.1

    # --- Step 1: sigmoid ---
    results["sigmoid"] = _bench_op(
        lambda: torch.sigmoid(logits.float()).to(logits.dtype),
        warmup, iters,
    )
    scores = torch.sigmoid(logits.float()).to(logits.dtype)

    # --- Step 2: add bias ---
    if enable_bias and expert_bias is not None:
        results["add_bias"] = _bench_op(
            lambda: scores + expert_bias,
            warmup, iters,
        )
        routing_scores = scores + expert_bias
    else:
        results["add_bias"] = 0.0
        routing_scores = scores

    # --- Step 3: topk ---
    results["topk"] = _bench_op(
        lambda: torch.topk(routing_scores, k=topk, dim=1),
        warmup, iters,
    )
    _, top_indices = torch.topk(routing_scores, k=topk, dim=1)

    # --- Step 4: gather (re-fetch original scores, not biased) ---
    if enable_bias:
        results["gather"] = _bench_op(
            lambda: torch.gather(scores, dim=1, index=top_indices),
            warmup, iters,
        )
        selected = torch.gather(scores, dim=1, index=top_indices)
    else:
        results["gather"] = 0.0
        selected, top_indices = torch.topk(routing_scores, k=topk, dim=1)

    # --- Step 5: normalize ---
    if topk > 1:
        results["normalize"] = _bench_op(
            lambda: selected / (selected.sum(dim=-1, keepdim=True) + 1e-20),
            warmup, iters,
        )
    else:
        results["normalize"] = 0.0

    # --- Step 6: scaling ---
    probs = selected / (selected.sum(dim=-1, keepdim=True) + 1e-20) if topk > 1 else selected
    if scaling_factor != 1.0:
        results["scale"] = _bench_op(
            lambda: probs * scaling_factor,
            warmup, iters,
        )
    else:
        results["scale"] = 0.0

    # --- Step 7: scatter ---
    output_template = torch.zeros_like(logits)
    results["scatter"] = _bench_op(
        lambda: output_template.zero_().scatter_(1, top_indices, probs),
        warmup, iters,
    )

    # --- Step 8: topk_map ---
    map_template = torch.zeros_like(logits, dtype=torch.int32)
    ones = torch.ones_like(top_indices, dtype=torch.int32)
    results["topk_map_scatter"] = _bench_op(
        lambda: map_template.zero_().scatter_(1, top_indices, ones).bool(),
        warmup, iters,
    )

    # --- End-to-end ---
    def _reference_e2e():
        s = torch.sigmoid(logits.float()).to(logits.dtype)
        if enable_bias and expert_bias is not None:
            rs = s + expert_bias
            _, idx = torch.topk(rs, k=topk, dim=1)
            sel = torch.gather(s, dim=1, index=idx)
        else:
            sel, idx = torch.topk(s, k=topk, dim=1)
        p = sel / (sel.sum(dim=-1, keepdim=True) + 1e-20) if topk > 1 else sel
        if scaling_factor != 1.0:
            p = p * scaling_factor
        out = torch.zeros_like(logits).scatter_(1, idx, p)
        m = torch.zeros_like(logits, dtype=torch.int32).scatter_(1, idx, 1).bool()
        return out, m

    results["end_to_end"] = _bench_op(_reference_e2e, warmup, iters)

    return results


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_breakdown(
    label: str,
    results: Dict[str, float],
    num_tokens: int,
    num_experts: int,
    topk: int,
) -> None:
    """Print a single config's per-op breakdown."""
    total = results["end_to_end"]
    print(f"\n{'─' * 80}")
    print(f"  {label}")
    print(f"  Shape: tokens={num_tokens}, experts={num_experts}, topk={topk}")
    print(f"{'─' * 80}")
    print(f"  {'Operation':<30s} {'Time (ms)':>10s} {'% of E2E':>10s} {'Notes':>20s}")
    print(f"  {'─' * 72}")

    sum_parts = 0.0
    for op_name, ms in results.items():
        if op_name == "end_to_end":
            continue
        if ms < 1e-6:
            continue
        pct = ms / total * 100 if total > 0 else 0
        sum_parts += ms

        # Add notes for significant ops
        note = ""
        if op_name == "topk":
            note = f"K={topk}, E={num_experts}"
        elif "softmax" in op_name:
            dim = num_experts if "full_E" in op_name else topk
            note = f"dim={dim}"
        elif op_name == "scatter":
            note = f"{num_tokens}×{num_experts}"

        print(f"  {op_name:<30s} {ms:>10.4f} {pct:>9.1f}% {note:>20s}")

    overhead = total - sum_parts
    overhead_pct = overhead / total * 100 if total > 0 else 0
    print(f"  {'─' * 72}")
    print(f"  {'(sum of parts)':<30s} {sum_parts:>10.4f} {sum_parts / total * 100:>9.1f}%")
    print(f"  {'(launch/sync overhead)':<30s} {overhead:>10.4f} {overhead_pct:>9.1f}%")
    print(f"  {'END-TO-END':<30s} {total:>10.4f} {'100.0%':>10s}")
    print()
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Main benchmark driver
# ---------------------------------------------------------------------------

def run_breakdown(args) -> None:
    """Run per-op breakdown benchmarks."""
    warmup = args.warmup
    iters = args.iters
    dtype = args.dtype

    if args.user_specified:
        token_list = (
            [args.num_tokens] if args.num_tokens is not None
            else [128, 1024, 4096, 16384, 65536]
        )
    else:
        token_list = [1024, 4096, 16384]

    expert_list = [args.num_experts] if args.user_specified else [64, 256, 2304]
    topk_list = [args.topk] if args.user_specified else [4, 32, 64]
    sf_list = [args.score_function] if args.user_specified else ["softmax", "sigmoid"]

    print(f"\nPer-op breakdown (warmup={warmup}, iters={iters}, dtype={dtype})")
    print(f"Config matrix: tokens={token_list}, experts={expert_list}, "
          f"topk={topk_list}, score_fn={sf_list}")
    sys.stdout.flush()

    for sf in sf_list:
        for ne in expert_list:
            for tk in topk_list:
                if tk > ne:
                    continue
                for nt in token_list:
                    logits = torch.randn(
                        nt, ne, dtype=dtype, device="cuda", requires_grad=True,
                    )

                    if sf == "softmax":
                        pre = args.use_pre_softmax if args.user_specified else True
                        results = bench_softmax_path(
                            logits, tk, pre, args.scaling_factor, warmup, iters,
                        )
                        label = f"softmax ({'pre' if pre else 'post'}-softmax)"

                        print_breakdown(label, results, nt, ne, tk)

                        # Also benchmark the other softmax variant if sweeping
                        if not args.user_specified:
                            results2 = bench_softmax_path(
                                logits, tk, False, args.scaling_factor, warmup, iters,
                            )
                            print_breakdown(
                                "softmax (post-softmax)", results2, nt, ne, tk,
                            )

                    elif sf == "sigmoid":
                        results = bench_sigmoid_path(
                            logits, tk, args.enable_bias, args.scaling_factor,
                            warmup, iters,
                        )
                        label = f"sigmoid (bias={'yes' if args.enable_bias else 'no'})"
                        print_breakdown(label, results, nt, ne, tk)


def run_isolated(args) -> None:
    """Run isolated op benchmarks."""
    warmup = args.warmup
    iters = args.iters
    dtype = args.dtype

    print(f"\nIsolated op benchmarks (warmup={warmup}, iters={iters}, dtype={dtype})")
    sys.stdout.flush()

    # --- torch.topk ---
    print(f"\n{'=' * 80}")
    print("  torch.topk Isolated Benchmark")
    print(f"{'=' * 80}")
    hdr = f"  {'tokens':>8s} {'experts':>8s} {'topk':>6s} {'ms':>10s} {'M elem/us':>12s}"
    print(hdr)
    print(f"  {'─' * (len(hdr) - 2)}")
    sys.stdout.flush()
    for nt in [128, 1024, 4096, 16384]:
        for ne in [64, 256, 512, 2304]:
            for tk in [1, 4, 8, 32, 64, 128]:
                if tk > ne:
                    continue
                data = torch.randn(nt, ne, dtype=dtype, device="cuda", requires_grad=True)
                ms = _bench_op(lambda: torch.topk(data, k=tk, dim=1), warmup, iters)
                elems_per_us = nt * ne / (ms * 1000)
                print(
                    f"  {nt:>8d} {ne:>8d} {tk:>6d} "
                    f"{ms:>10.4f} {elems_per_us:>12.2f}"
                )
                sys.stdout.flush()
    print()

    # --- torch.softmax ---
    print(f"\n{'=' * 80}")
    print("  torch.softmax Isolated Benchmark")
    print(f"{'=' * 80}")
    hdr = f"  {'tokens':>8s} {'experts':>8s} {'ms':>10s} {'M elem/us':>12s}"
    print(hdr)
    print(f"  {'─' * (len(hdr) - 2)}")
    sys.stdout.flush()
    for nt in [128, 1024, 4096, 16384]:
        for ne in [64, 256, 512, 2304]:
            data = torch.randn(nt, ne, dtype=dtype, device="cuda", requires_grad=True)
            ms = _bench_op(
                lambda: torch.softmax(data, dim=-1, dtype=torch.float32).to(dtype),
                warmup, iters,
            )
            elems_per_us = nt * ne / (ms * 1000)
            print(
                f"  {nt:>8d} {ne:>8d} "
                f"{ms:>10.4f} {elems_per_us:>12.2f}"
            )
            sys.stdout.flush()
    print()

    # --- scatter_ ---
    print(f"\n{'=' * 80}")
    print("  scatter_ Isolated Benchmark")
    print(f"{'=' * 80}")
    hdr = f"  {'tokens':>8s} {'experts':>8s} {'topk':>6s} {'ms':>10s}"
    print(hdr)
    print(f"  {'─' * (len(hdr) - 2)}")
    sys.stdout.flush()
    for nt in [128, 1024, 4096, 16384]:
        for ne in [64, 256, 512, 2304]:
            for tk in [4, 32, 64]:
                if tk > ne:
                    continue
                output = torch.zeros(nt, ne, dtype=dtype, device="cuda")
                indices = torch.randint(0, ne, (nt, tk), device="cuda")
                values = torch.randn(nt, tk, dtype=dtype, device="cuda")
                ms = _bench_op(
                    lambda: output.zero_().scatter_(1, indices, values),
                    warmup, iters,
                )
                print(
                    f"  {nt:>8d} {ne:>8d} {tk:>6d} "
                    f"{ms:>10.4f}"
                )
                sys.stdout.flush()
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _any_kernel_arg_set(argv: List[str]) -> bool:
    """Return True if the user passed any kernel-shape / config flag."""
    kernel_flags = {
        "--num-tokens", "--num-experts", "--topk", "--score-function",
        "--use-pre-softmax", "--enable-bias", "--scaling-factor",
    }
    for arg in argv:
        if arg.split("=")[0] in kernel_flags:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Per-operation performance breakdown of PyTorch MoE routing reference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["breakdown", "isolated", "all"],
        default="all",
        help="breakdown = per-op in routing path, isolated = individual op scaling, "
             "all = both (default: all)",
    )

    # Kernel shape params
    parser.add_argument("--num-tokens", type=int, default=None)
    parser.add_argument("--num-experts", type=int, default=2304)
    parser.add_argument("--topk", type=int, default=36)
    parser.add_argument("--score-function", choices=["softmax", "sigmoid"], default="softmax")
    parser.add_argument("--use-pre-softmax", action="store_true", default=False)
    parser.add_argument("--enable-bias", action="store_true", default=False)
    parser.add_argument("--scaling-factor", type=float, default=1.0)

    # Dtype
    parser.add_argument("--dtype", type=parse_dtype, default=torch.float32,
                        help="fp32 | fp16 | bf16 (default: fp32)")

    # Benchmark tuning
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)

    args = parser.parse_args()
    args.user_specified = _any_kernel_arg_set(sys.argv[1:])

    print_gpu_info()

    if args.mode in ("breakdown", "all"):
        run_breakdown(args)
    if args.mode in ("isolated", "all"):
        run_isolated(args)


if __name__ == "__main__":
    main()
