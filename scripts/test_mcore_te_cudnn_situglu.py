#!/usr/bin/env python3
"""Prove MCore uses TE SiTU-GLU and the real cuDNN fused grouped kernels."""

from __future__ import annotations

import argparse
import functools
import os

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=("mxfp8", "nvfp4"), default="mxfp8")
    args = parser.parse_args()
    os.environ.setdefault("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "1")

    import cudnn
    import transformer_engine.pytorch as te
    from transformer_engine.pytorch.ops.fused import grouped_mlp

    native_situ = getattr(te.ops, "ScaledSiTUGLU", None)
    if native_situ is None:
        raise RuntimeError("TE ScaledSiTUGLU is unavailable; MCore would use its local fallback")

    original_forward = cudnn.grouped_gemm_glu_wrapper_sm100
    original_backward = cudnn.grouped_gemm_dglu_wrapper_sm100
    original_hadamard = getattr(cudnn, "grouped_gemm_glu_hadamard_wrapper_sm100", None)
    calls: list[tuple[str, torch.dtype, torch.dtype | None, str]] = []

    @functools.wraps(original_forward)
    def traced_forward(*args, **kwargs):
        calls.append(
            (
                "forward_regular",
                kwargs["alpha_tensor"].dtype,
                None,
                kwargs.get("act_func", ""),
            )
        )
        return original_forward(*args, **kwargs)

    @functools.wraps(original_backward)
    def traced_backward(*args, **kwargs):
        calls.append(
            (
                "backward",
                kwargs["alpha_tensor"].dtype,
                kwargs["beta_tensor"].dtype,
                kwargs.get("act_func", ""),
            )
        )
        return original_backward(*args, **kwargs)

    if original_hadamard is not None:

        @functools.wraps(original_hadamard)
        def traced_hadamard(*wrapper_args, **kwargs):
            calls.append(
                (
                    "forward_hadamard",
                    kwargs["alpha_tensor"].dtype,
                    None,
                    kwargs.get("act_func", ""),
                )
            )
            return original_hadamard(*wrapper_args, **kwargs)

        cudnn.grouped_gemm_glu_hadamard_wrapper_sm100 = traced_hadamard

    cudnn.grouped_gemm_glu_wrapper_sm100 = traced_forward
    cudnn.grouped_gemm_dglu_wrapper_sm100 = traced_backward

    # The wrapper getters and feature probe are cached. Clear them so the fused
    # operation captures the tracing wrappers instead of an earlier import.
    grouped_mlp._cudnn_frontend_supports_grouped_gemm_situglu.cache_clear()
    fused_cls = te.ops.fused.GroupedMLP_CuTeGEMMGLU
    fused_cls.is_supported.cache_clear()
    fused_cls.grouped_gemm_activation_kernel.cache_clear()
    fused_cls.grouped_gemm_dactivation_kernel.cache_clear()
    fused_cls.grouped_gemm_act_hadamard_kernel.cache_clear()

    if not grouped_mlp._cudnn_frontend_supports_grouped_gemm_situglu():
        raise RuntimeError("cuDNN forward/backward wrappers do not expose SiTU-GLU")
    if not fused_cls.is_supported():
        raise RuntimeError("TE GroupedMLP_CuTeGEMMGLU reports unsupported")

    from tests.unit_tests.fusions.test_cutedsl_situ_glu import (
        test_mcore_moe_glu_forward_backward_uses_expected_backend,
    )

    test_mcore_moe_glu_forward_backward_uses_expected_backend(args.precision, "situglu")
    torch.cuda.synchronize()

    regular_forward_calls = [call for call in calls if call[0] == "forward_regular"]
    hadamard_forward_calls = [call for call in calls if call[0] == "forward_hadamard"]
    backward_calls = [call for call in calls if call[0] == "backward"]
    if not backward_calls:
        raise AssertionError("The real cuDNN grouped dGLU backward wrapper was not called")

    expected_scale_dtype = torch.bfloat16 if args.precision == "mxfp8" else torch.float32
    if args.precision == "mxfp8":
        if not regular_forward_calls:
            raise AssertionError("The real cuDNN grouped GLU forward wrapper was not called")
        assert not hadamard_forward_calls, hadamard_forward_calls
    else:
        if not hadamard_forward_calls:
            raise AssertionError("The real cuDNN grouped GLU-Hadamard wrapper was not called")
        assert not regular_forward_calls, regular_forward_calls

    for _, alpha_dtype, _, act_func in regular_forward_calls + hadamard_forward_calls:
        assert alpha_dtype == expected_scale_dtype, alpha_dtype
        assert act_func == "situglu", act_func
    for _, alpha_dtype, beta_dtype, act_func in backward_calls:
        assert alpha_dtype == expected_scale_dtype, alpha_dtype
        assert beta_dtype == expected_scale_dtype, beta_dtype
        assert act_func == "dsituglu", act_func

    print(f"TE_ACTIVATION={native_situ.__module__}.{native_situ.__name__}")
    print(f"TE_FUSED_OP={fused_cls.__module__}.{fused_cls.__name__}")
    print(f"PRECISION={args.precision}")
    print(f"CUDNN_REGULAR_FORWARD_CALLS={regular_forward_calls}")
    print(f"CUDNN_HADAMARD_FORWARD_CALLS={hadamard_forward_calls}")
    print(f"CUDNN_BACKWARD_CALLS={backward_calls}")
    print("RESULT=PASS")


if __name__ == "__main__":
    main()
