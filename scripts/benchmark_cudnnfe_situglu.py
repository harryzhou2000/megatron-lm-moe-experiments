#!/usr/bin/env python3
"""Benchmark cuDNN Frontend fused grouped SwiGLU and SiTU-GLU kernels on one GPU.

The model shapes are derived from the official DeepSeek-V4-Flash config and the
Kimi K3 technical report. EP64 affects only the number of local expert groups;
``--tokens`` is the total number of routed-token rows processed by those local
experts before the kernel's per-expert padding.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
from typing import Callable

import torch


MODEL_CONFIGS = {
    "deepseek-v4-flash": {
        "hidden_size": 4096,
        "expert_intermediate_size": 2048,
        "total_experts": 256,
        "topk": 6,
        "source": "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/config.json",
    },
    "kimi-k3": {
        "hidden_size": 3584,
        "expert_intermediate_size": 3072,
        "total_experts": 896,
        "topk": 16,
        "source": "https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf",
    },
}

FORMAT_CONFIGS = {
    "mxfp4": {
        "ab_dtype": torch.float4_e2m1fn_x2,
        "sf_dtype": torch.float8_e8m0fnu,
        "sf_vec_size": 32,
        "forward_d_dtype": torch.bfloat16,
        "backward_d_dtype": torch.bfloat16,
    },
    "mxfp8": {
        "ab_dtype": torch.float8_e4m3fn,
        "sf_dtype": torch.float8_e8m0fnu,
        "sf_vec_size": 32,
        "forward_d_dtype": torch.float8_e4m3fn,
        "backward_d_dtype": torch.float8_e4m3fn,
    },
    "nvfp4": {
        "ab_dtype": torch.float4_e2m1fn_x2,
        "sf_dtype": torch.float8_e4m3fn,
        "sf_vec_size": 16,
        "forward_d_dtype": torch.bfloat16,
        "backward_d_dtype": torch.bfloat16,
    },
}

KIMI_SITU_BETA1 = 4.0
KIMI_SITU_BETA2 = 25.0


def _balanced_random_split(total: int, groups: int, seed: int, jitter: float) -> list[int]:
    """Return positive, bounded-random group sizes that sum exactly to ``total``."""
    if total < groups:
        raise ValueError(f"tokens={total} must be at least local_experts={groups}")
    rng = random.Random(seed)
    weights = [1.0 + rng.uniform(-jitter, jitter) for _ in range(groups)]
    scale = total / sum(weights)
    raw = [weight * scale for weight in weights]
    sizes = [max(1, math.floor(value)) for value in raw]
    remainder = total - sum(sizes)
    if remainder > 0:
        order = sorted(range(groups), key=lambda idx: raw[idx] - math.floor(raw[idx]), reverse=True)
        for idx in order[:remainder]:
            sizes[idx] += 1
    elif remainder < 0:
        order = sorted(range(groups), key=lambda idx: raw[idx] - math.floor(raw[idx]))
        for idx in order:
            if remainder == 0:
                break
            if sizes[idx] > 1:
                sizes[idx] -= 1
                remainder += 1
    if sum(sizes) != total:
        raise RuntimeError(f"Failed to construct an exact split: {sizes}")
    return sizes


def _distribution_stats(sizes: list[int], alignment: int) -> dict[str, object]:
    padded = [((size + alignment - 1) // alignment) * alignment for size in sizes]
    return {
        "expert_tokens": sizes,
        "min_expert_tokens": min(sizes),
        "max_expert_tokens": max(sizes),
        "padded_expert_tokens": padded,
        "padded_tokens": sum(padded),
        "padding_ratio": sum(padded) / sum(sizes),
    }


def _measure_cuda_ms(
    fn: Callable[[], None],
    *,
    warmup: int,
    iterations: int,
    rounds: int,
) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(rounds):
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        for start, end in zip(starts, ends):
            start.record()
            fn()
            end.record()
        torch.cuda.synchronize()
        samples.extend(start.elapsed_time(end) for start, end in zip(starts, ends))
    return samples


def _summarize_tflops(samples_ms: list[float], gemm_flops: int) -> dict[str, float]:
    samples = [gemm_flops / 1.0e9 / sample_ms for sample_ms in samples_ms]
    ordered = sorted(samples)
    return {
        "median_tflops": statistics.median(ordered),
        "mean_tflops": statistics.fmean(ordered),
        "min_tflops": ordered[0],
        "p25_tflops": ordered[len(ordered) // 4],
        "p75_tflops": ordered[(3 * len(ordered)) // 4],
    }


def _make_forward_runner(
    *,
    format_name: str,
    act_func: str,
    hidden_size: int,
    intermediate_size: int,
    group_m_list: list[int],
    use_dynamic_sched: bool,
    vector_f32: bool,
    mma_tiler_m: int,
):
    from cuda.bindings import driver as cuda
    from cudnn.gemm.cutedsl.grouped.glu.api import GroupedGemmGluSm100
    from fe_api.grouped_gemm.test_discrete_grouped_gemm_swiglu_utils import (
        allocate_discrete_input_tensors,
        allocate_discrete_output_tensors,
    )

    num_experts = len(group_m_list)
    format_config = FORMAT_CONFIGS[format_name]
    n = 2 * intermediate_size
    inputs = allocate_discrete_input_tensors(
        n=n,
        k=hidden_size,
        num_experts=num_experts,
        group_m_list=group_m_list,
        ab_dtype=format_config["ab_dtype"],
        sf_dtype=format_config["sf_dtype"],
        sf_vec_size=format_config["sf_vec_size"],
        m_aligned=256,
        b_major="k",
    )
    outputs = allocate_discrete_output_tensors(
        tensor_m=inputs["tensor_m"],
        n=n,
        num_experts=num_experts,
        ab_dtype=format_config["ab_dtype"],
        c_dtype=torch.bfloat16,
        d_dtype=format_config["forward_d_dtype"],
        cd_major="n",
        sf_dtype=format_config["sf_dtype"],
        sf_vec_size=format_config["sf_vec_size"],
    )
    api = GroupedGemmGluSm100(
        sample_a=inputs["a_tensor"],
        sample_c=outputs["c_tensor"],
        sample_d=outputs["d_tensor"],
        sample_sfa=inputs["sfa_tensor"],
        sample_padded_offsets=inputs["padded_offsets_tensor"],
        sample_alpha=inputs["alpha_tensor"],
        sample_d_col=outputs["d_col_tensor"],
        num_experts=num_experts,
        b_shape=(n, hidden_size),
        b_dtype=inputs["b_list"][0].dtype,
        sample_sfd_row=outputs["sfd_row_tensor"],
        sample_sfd_col=outputs["sfd_col_tensor"],
        sample_norm_const=inputs["norm_const_tensor"],
        sample_prob=inputs["prob_tensor"],
        acc_dtype=torch.float32,
        mma_tiler_mn=(mma_tiler_m, 256),
        cluster_shape_mn=(2 if mma_tiler_m == 256 else 1, 1),
        sf_vec_size=format_config["sf_vec_size"],
        vector_f32=vector_f32,
        m_aligned=256,
        discrete_col_sfd=False,
        act_func=act_func,
        situ_beta1=KIMI_SITU_BETA1,
        b_major="k",
        use_dynamic_sched=use_dynamic_sched,
    )
    if not api.check_support():
        raise RuntimeError(f"Forward act_func={act_func} is unsupported")
    api.compile()
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    def run() -> None:
        api.execute(
            a_tensor=inputs["a_tensor"],
            c_tensor=outputs["c_tensor"],
            d_tensor=outputs["d_tensor"],
            sfa_tensor=inputs["sfa_tensor"],
            padded_offsets=inputs["padded_offsets_tensor"],
            alpha_tensor=inputs["alpha_tensor"],
            b_ptrs=inputs["b_ptrs_tensor"],
            sfb_ptrs=inputs["sfb_ptrs_tensor"],
            d_col_tensor=outputs["d_col_tensor"],
            sfd_row_tensor=outputs["sfd_row_tensor"],
            sfd_col_tensor=outputs["sfd_col_tensor"],
            norm_const_tensor=inputs["norm_const_tensor"],
            prob_tensor=inputs["prob_tensor"],
            situ_beta1=KIMI_SITU_BETA1,
            situ_beta2=KIMI_SITU_BETA2,
            current_stream=stream,
        )

    return run, (api, inputs, outputs)


def _make_backward_runner(
    *,
    format_name: str,
    act_func: str,
    hidden_size: int,
    intermediate_size: int,
    group_m_list: list[int],
    use_dynamic_sched: bool,
    vector_f32: bool,
    mma_tiler_m: int,
):
    from cuda.bindings import driver as cuda
    from cudnn.gemm.cutedsl.grouped.dglu.api import GroupedGemmDgluSm100
    from fe_api.grouped_gemm.test_discrete_grouped_gemm_dswiglu_utils import (
        allocate_discrete_dswiglu_input_tensors,
        allocate_discrete_dswiglu_output_tensors,
    )

    num_experts = len(group_m_list)
    format_config = FORMAT_CONFIGS[format_name]
    inputs = allocate_discrete_dswiglu_input_tensors(
        n=intermediate_size,
        k=hidden_size,
        num_experts=num_experts,
        group_m_list=group_m_list,
        ab_dtype=format_config["ab_dtype"],
        c_dtype=torch.bfloat16,
        sf_dtype=format_config["sf_dtype"],
        sf_vec_size=format_config["sf_vec_size"],
        m_aligned=256,
        b_major="k",
    )
    outputs = allocate_discrete_dswiglu_output_tensors(
        tensor_m=inputs["tensor_m"],
        n=intermediate_size,
        num_experts=num_experts,
        ab_dtype=format_config["ab_dtype"],
        d_dtype=format_config["backward_d_dtype"],
        cd_major="n",
        sf_dtype=format_config["sf_dtype"],
        sf_vec_size=format_config["sf_vec_size"],
    )
    api = GroupedGemmDgluSm100(
        sample_a=inputs["a_tensor"],
        sample_c=inputs["c_tensor"],
        sample_d_row=outputs["d_row_tensor"],
        sample_d_col=outputs["d_col_tensor"],
        sample_sfa=inputs["sfa_tensor"],
        sample_padded_offsets=inputs["padded_offsets_tensor"],
        sample_alpha=inputs["alpha_tensor"],
        sample_beta=inputs["beta_tensor"],
        sample_prob=inputs["prob_tensor"],
        sample_dprob=inputs["dprob_tensor"],
        num_experts=num_experts,
        b_shape=(intermediate_size, hidden_size),
        b_dtype=inputs["b_list"][0].dtype,
        sample_sfd_row=outputs["sfd_row_tensor"],
        sample_sfd_col=outputs["sfd_col_tensor"],
        sample_norm_const=inputs["norm_const_tensor"],
        acc_dtype=torch.float32,
        mma_tiler_mn=(mma_tiler_m, 256),
        cluster_shape_mn=(2 if mma_tiler_m == 256 else 1, 1),
        sf_vec_size=format_config["sf_vec_size"],
        vector_f32=vector_f32,
        m_aligned=256,
        discrete_col_sfd=False,
        act_func=act_func,
        situ_beta1=KIMI_SITU_BETA1,
        situ_beta2=KIMI_SITU_BETA2,
        b_major="k",
        use_dynamic_sched=use_dynamic_sched,
    )
    if not api.check_support():
        raise RuntimeError(f"Backward act_func={act_func} is unsupported")
    api.compile()
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    def run() -> None:
        api.execute(
            a_tensor=inputs["a_tensor"],
            c_tensor=inputs["c_tensor"],
            d_row_tensor=outputs["d_row_tensor"],
            d_col_tensor=outputs["d_col_tensor"],
            sfa_tensor=inputs["sfa_tensor"],
            padded_offsets=inputs["padded_offsets_tensor"],
            alpha_tensor=inputs["alpha_tensor"],
            beta_tensor=inputs["beta_tensor"],
            prob_tensor=inputs["prob_tensor"],
            dprob_tensor=inputs["dprob_tensor"],
            b_ptrs=inputs["b_ptrs_tensor"],
            sfb_ptrs=inputs["sfb_ptrs_tensor"],
            sfd_row_tensor=outputs["sfd_row_tensor"],
            sfd_col_tensor=outputs["sfd_col_tensor"],
            norm_const_tensor=inputs["norm_const_tensor"],
            current_stream=stream,
        )

    return run, (api, inputs, outputs)


def _benchmark_case(args, model_name: str, format_name: str, tokens: int) -> list[dict[str, object]]:
    model = MODEL_CONFIGS[model_name]
    format_config = FORMAT_CONFIGS[format_name]
    total_experts = int(model["total_experts"])
    if total_experts % args.ep != 0:
        raise ValueError(f"{model_name}: total_experts={total_experts} is not divisible by EP={args.ep}")
    local_experts = total_experts // args.ep
    group_m_list = _balanced_random_split(
        tokens,
        local_experts,
        seed=args.seed + tokens + total_experts,
        jitter=args.jitter,
    )
    dist = _distribution_stats(group_m_list, alignment=256)
    print(
        f"case model={model_name} format={format_name} local_experts={local_experts} local_tokens={tokens} "
        f"padded_tokens={dist['padded_tokens']} split={group_m_list}",
        flush=True,
    )

    runners = {}
    keepalive = []
    case_seed = args.seed + tokens + total_experts
    for direction, activation in (
        ("forward", "swiglu"),
        ("forward", "situglu"),
        ("backward", "dswiglu"),
        ("backward", "dsituglu"),
    ):
        # Keep each activation's tensors identical and make a case independent
        # of which model shapes were benchmarked before it.
        torch.manual_seed(case_seed)
        torch.cuda.manual_seed_all(case_seed)
        factory = _make_forward_runner if direction == "forward" else _make_backward_runner
        runner, state = factory(
            format_name=format_name,
            act_func=activation,
            hidden_size=int(model["hidden_size"]),
            intermediate_size=int(model["expert_intermediate_size"]),
            group_m_list=group_m_list,
            use_dynamic_sched=args.dynamic_sched,
            vector_f32=args.vector_f32,
            mma_tiler_m=args.mma_tiler_m,
        )
        runners[(direction, activation)] = runner
        keepalive.append(state)

    results = []
    for direction, baseline_activation, situ_activation in (
        ("forward", "swiglu", "situglu"),
        ("backward", "dswiglu", "dsituglu"),
    ):
        samples = {baseline_activation: [], situ_activation: []}
        orders = (
            (baseline_activation, situ_activation),
            (situ_activation, baseline_activation),
        )
        for round_idx in range(args.rounds):
            for activation in orders[round_idx % len(orders)]:
                samples[activation].extend(
                    _measure_cuda_ms(
                        runners[(direction, activation)],
                        warmup=args.warmup if round_idx == 0 else 0,
                        iterations=args.iterations,
                        rounds=1,
                    )
                )

        gemm_n = (
            2 * int(model["expert_intermediate_size"])
            if direction == "forward"
            else int(model["expert_intermediate_size"])
        )
        gemm_flops = 2 * int(dist["padded_tokens"]) * int(model["hidden_size"]) * gemm_n
        summaries = {
            activation: _summarize_tflops(values, gemm_flops)
            for activation, values in samples.items()
        }
        baseline_tflops = summaries[baseline_activation]["median_tflops"]
        situ_tflops = summaries[situ_activation]["median_tflops"]
        for activation in (baseline_activation, situ_activation):
            row = {
                "model": model_name,
                "format": format_name,
                "ab_dtype": str(format_config["ab_dtype"]),
                "sf_dtype": str(format_config["sf_dtype"]),
                "sf_vec_size": format_config["sf_vec_size"],
                "direction": direction,
                "activation": activation,
                "ep": args.ep,
                "total_experts": total_experts,
                "local_experts": local_experts,
                "hidden_size": model["hidden_size"],
                "expert_intermediate_size": model["expert_intermediate_size"],
                "local_tokens": tokens,
                **dist,
                **summaries[activation],
                "situ_throughput_retention": situ_tflops / baseline_tflops,
                "samples": len(samples[activation]),
                "seed": args.seed,
                "jitter": args.jitter,
                "dynamic_sched": args.dynamic_sched,
                "vector_f32": args.vector_f32,
                "mma_tiler_m": args.mma_tiler_m,
                "situ_beta1": KIMI_SITU_BETA1,
                "situ_beta2": KIMI_SITU_BETA2,
            }
            results.append(row)
        print(
            f"result model={model_name} format={format_name} tokens={tokens} direction={direction} "
            f"{baseline_activation}={baseline_tflops:.3f}TFLOP/s "
            f"{situ_activation}={situ_tflops:.3f}TFLOP/s "
            f"situ_throughput_retention={situ_tflops / baseline_tflops:.4f}",
            flush=True,
        )
    del keepalive
    torch.cuda.empty_cache()
    return results


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    serializable = []
    for row in rows:
        record = dict(row)
        record["expert_tokens"] = json.dumps(record["expert_tokens"])
        record["padded_expert_tokens"] = json.dumps(record["padded_expert_tokens"])
        serializable.append(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(serializable[0]))
        writer.writeheader()
        writer.writerows(serializable)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cudnn-fe-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=MODEL_CONFIGS, default=list(MODEL_CONFIGS))
    parser.add_argument("--formats", nargs="+", choices=FORMAT_CONFIGS, default=list(FORMAT_CONFIGS))
    parser.add_argument("--tokens", nargs="+", type=int, default=[4096, 8192, 16384])
    parser.add_argument("--ep", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--jitter", type=float, default=0.10)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument(
        "--dynamic-sched",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the dynamic tile scheduler selected by TE's fused grouped MLP path.",
    )
    parser.add_argument(
        "--vector-f32",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the grouped-GEMM vectorized-FP32 epilogue mode.",
    )
    parser.add_argument("--mma-tiler-m", type=int, choices=(128, 256), default=256)
    parser.add_argument("--csv-output", type=Path)
    args = parser.parse_args()

    if not 0.0 <= args.jitter < 1.0:
        raise ValueError("--jitter must be in [0, 1)")
    if any(tokens < 4096 for tokens in args.tokens):
        raise ValueError("All --tokens values must be at least 4096")

    test_python = args.cudnn_fe_root.resolve() / "test" / "python"
    sys.path.insert(0, str(test_python))
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.cuda.get_device_properties(0)
    metadata = {
        "gpu": device.name,
        "compute_capability": torch.cuda.get_device_capability(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn_frontend": importlib.metadata.version("nvidia-cudnn-frontend"),
        "cutlass_dsl": importlib.metadata.version("nvidia-cutlass-dsl"),
        "ep": args.ep,
        "formats": args.formats,
        "tokens": args.tokens,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "rounds": args.rounds,
        "dynamic_sched": args.dynamic_sched,
        "vector_f32": args.vector_f32,
        "mma_tiler_m": args.mma_tiler_m,
        "situ_beta1": KIMI_SITU_BETA1,
        "situ_beta2": KIMI_SITU_BETA2,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    print("metadata=" + json.dumps(metadata, sort_keys=True), flush=True)

    rows = []
    for model_name in args.models:
        for format_name in args.formats:
            for tokens in args.tokens:
                rows.extend(_benchmark_case(args, model_name, format_name, tokens))

    if args.csv_output is not None:
        _write_csv(args.csv_output, rows)
        print(f"csv_output={args.csv_output}", flush=True)


if __name__ == "__main__":
    main()
