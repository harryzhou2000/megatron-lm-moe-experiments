# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
Minimal test for device-initiated CUTLASS grouped GEMM.

Tests the GEMM kernel directly via the Python wrapper, bypassing GroupedLinear.
Compares results against per-expert PyTorch BF16 matmuls, and optionally against
the cuBLAS-based GroupedTensor grouped GEMM path.

Known constraint: the CUTLASS device-initiated grouped GEMM kernel requires each
non-zero expert's token count (m_splits[i]) to be a multiple of 128, due to the
SM100 MXFP8 block-scaling tile size. This applies to BOTH fprop and wgrad:
  - fprop: the scale-factor pointer for expert i is computed as
    ptr_SFA + m_offset * ((K+127)/128*4), where m_offset = sum(m_splits[:i]).
    A non-128-aligned m_offset produces an incorrect SF pointer.
  - wgrad: gemm_k / 128 in the layout computation would produce a degenerate layout.

Usage:
    python scripts/test_device_init_grouped_gemm.py
    python scripts/test_device_init_grouped_gemm.py --no-cublas   # skip cuBLAS tests
"""

import argparse
import os
import sys
import torch

# TE imports
import transformer_engine  # noqa: F401 — ensures libtransformer_engine.so is loaded
import transformer_engine_torch as tex
from transformer_engine.pytorch.quantization import (
    FP8GlobalStateManager,
    is_mxfp8_available,
)
from transformer_engine.pytorch import autocast, GroupedLinear, quantized_model_init
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
from transformer_engine.common import recipe as te_recipe

# Device-init wrapper
from transformer_engine.pytorch.cpp_extensions.device_init_grouped_gemm import (
    device_init_grouped_gemm,
    get_device_init_grouped_gemm_workspace,
)


def check_hardware():
    """Check if hardware supports MXFP8."""
    avail, reason = is_mxfp8_available(return_reason=True)
    if not avail:
        print(f"SKIP: {reason}")
        sys.exit(0)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"SM: {torch.cuda.get_device_capability()}")
    print(f"cuBLAS: {tex.get_cublasLt_version()}")


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _rmse_check(test, ref, name, tol=0.1):
    """Return (pass, message) based on relative RMSE."""
    diff = test.float() - ref.float()
    rmse = diff.square().mean().sqrt().item()
    val_range = ref.float().max().item() - ref.float().min().item()
    rel = rmse / max(val_range, 1e-6)
    ok = rel < tol
    msg = f"RMSE={rmse:.4f}, range={val_range:.4f}, rel_RMSE={rel:.4f}"
    return ok, msg


# ---------------------------------------------------------------------------
# BF16 reference
# ---------------------------------------------------------------------------


def ref_grouped_matmul_tn(weights, inp, m_splits_list):
    """out[i] = inp_chunk[i] @ W[i].T  (TN layout, fprop/dgrad)."""
    chunks = torch.split(inp, m_splits_list)
    outs = []
    for i, chunk in enumerate(chunks):
        if chunk.numel() == 0:
            outs.append(torch.empty(0, weights[i].size(0), dtype=inp.dtype, device=inp.device))
        else:
            outs.append((chunk.float() @ weights[i].float().T).to(inp.dtype))
    return torch.cat(outs)


def ref_grouped_matmul_nt(inp, grad_out, m_splits_list):
    """dW[i] = grad_chunk[i].T @ inp_chunk[i]  (NT layout, wgrad)."""
    inp_chunks = torch.split(inp, m_splits_list)
    grad_chunks = torch.split(grad_out, m_splits_list)
    N = grad_out.size(-1)
    K = inp.size(-1)
    wgrads = []
    for i in range(len(m_splits_list)):
        if m_splits_list[i] == 0:
            wgrads.append(torch.zeros(N, K, dtype=inp.dtype, device=inp.device))
        else:
            wgrads.append((grad_chunks[i].float().T @ inp_chunks[i].float()).to(inp.dtype))
    return wgrads


# ---------------------------------------------------------------------------
# cuBLAS reference via GroupedLinear (uses TE's cuBLAS grouped GEMM)
# ---------------------------------------------------------------------------


def cublas_grouped_gemm_fprop(weights_bf16, inp_bf16, m_splits_list, K, N):
    """Run fprop through GroupedLinear with list m_splits (cuBLAS path)."""
    num_experts = len(m_splits_list)
    mxfp8_recipe = te_recipe.MXFP8BlockScaling()
    FP8GlobalStateManager.reset()

    with quantized_model_init(enabled=False, recipe=mxfp8_recipe):
        gl = GroupedLinear(
            num_experts, K, N, bias=False,
            params_dtype=torch.bfloat16, device="cuda",
        ).eval()

    with torch.no_grad():
        for i in range(num_experts):
            getattr(gl, f"weight{i}").copy_(weights_bf16[i])

    with autocast(enabled=True, recipe=mxfp8_recipe):
        out = gl(inp_bf16.unsqueeze(1), m_splits_list)

    return out.squeeze(1)


# ---------------------------------------------------------------------------
# FPROP test
# ---------------------------------------------------------------------------


def test_fprop(num_experts, m_splits_list, K, N, use_cublas=True, dtype=torch.bfloat16):
    """Test forward: out = inp @ W.T via device-init grouped GEMM."""
    label = f"fprop: {num_experts}E, m={m_splits_list}, K={K}, N={N}"
    print(f"\n--- {label} ---")

    # Check alignment: all non-zero m_splits must be multiples of 128
    for i, m in enumerate(m_splits_list):
        if m != 0 and m % 128 != 0:
            print(f"  SKIP: m_splits[{i}]={m} is not a multiple of 128 "
                  f"(required by CUTLASS device-init kernel)")
            return None

    total_M = sum(m_splits_list)

    torch.manual_seed(42)
    weights_bf16 = [torch.randn(N, K, dtype=dtype, device="cuda") for _ in range(num_experts)]
    inp_bf16 = torch.randn(total_M, K, dtype=dtype, device="cuda")

    # BF16 reference
    out_ref = ref_grouped_matmul_tn(weights_bf16, inp_bf16, m_splits_list)

    # --- Device-init CUTLASS path ---
    FP8GlobalStateManager.reset()
    q = MXFP8Quantizer(tex.DType.kFloat8E4M3, rowwise=True, columnwise=False)
    weights_mxfp8 = [q(w) for w in weights_bf16]
    q_inp = MXFP8Quantizer(tex.DType.kFloat8E4M3, rowwise=True, columnwise=False)
    inp_mxfp8 = q_inp(inp_bf16)

    out_di = torch.empty(total_M, N, dtype=dtype, device="cuda")
    m_dev = torch.tensor(m_splits_list, dtype=torch.int64, device="cuda")
    ws = get_device_init_grouped_gemm_workspace()
    device_init_grouped_gemm(
        weights_mxfp8, [inp_mxfp8], [out_di], dtype, ws,
        layout="TN", m_splits=m_dev, single_output=True,
    )
    torch.cuda.synchronize()

    # Per-expert device-init vs BF16
    all_pass = True
    offset = 0
    for i, m in enumerate(m_splits_list):
        if m == 0:
            print(f"  [device-init] Expert {i}: m=0 (skipped)")
            offset += m
            continue
        ok, msg = _rmse_check(out_di[offset:offset + m], out_ref[offset:offset + m],
                              f"Expert {i}")
        status = "OK" if ok else "FAIL"
        print(f"  [device-init] Expert {i}: m={m}, {msg} [{status}]")
        if not ok:
            all_pass = False
        offset += m

    ok, msg = _rmse_check(out_di, out_ref, "overall")
    print(f"  [device-init] Overall: {msg} [{'PASS' if ok else 'FAIL'}]")
    if not ok:
        all_pass = False

    # --- cuBLAS path ---
    if use_cublas:
        FP8GlobalStateManager.reset()
        out_cublas = cublas_grouped_gemm_fprop(weights_bf16, inp_bf16, m_splits_list, K, N)
        ok_cb, msg_cb = _rmse_check(out_cublas, out_ref, "cublas_vs_ref")
        print(f"  [cuBLAS]      vs BF16 ref: {msg_cb} [{'OK' if ok_cb else 'FAIL'}]")
        ok_di_cb, msg_di_cb = _rmse_check(out_di, out_cublas, "device_init_vs_cublas")
        print(f"  [di vs cuBLAS] {msg_di_cb} [{'OK' if ok_di_cb else 'FAIL'}]")

    return all_pass


# ---------------------------------------------------------------------------
# WGRAD test
# ---------------------------------------------------------------------------


def test_wgrad(num_experts, m_splits_list, K, N, dtype=torch.bfloat16):
    """Test wgrad: dW[i] = grad_out_chunk[i].T @ inp_chunk[i].

    NOTE: The CUTLASS wgrad kernel requires each expert's token count
    (gemm_k in the NT GEMM) to be a multiple of 128 due to SM100 tile size.
    """
    label = f"wgrad: {num_experts}E, m={m_splits_list}, K={K}, N={N}"
    print(f"\n--- {label} ---")

    # Validate alignment
    for i, m in enumerate(m_splits_list):
        if m != 0 and m % 128 != 0:
            print(f"  SKIP: m_splits[{i}]={m} is not a multiple of 128 "
                  f"(required by CUTLASS wgrad kernel)")
            return None  # skip, not fail

    total_M = sum(m_splits_list)

    torch.manual_seed(123)
    inp_bf16 = torch.randn(total_M, K, dtype=dtype, device="cuda")
    grad_out_bf16 = torch.randn(total_M, N, dtype=dtype, device="cuda")

    # BF16 reference
    wgrad_ref = ref_grouped_matmul_nt(inp_bf16, grad_out_bf16, m_splits_list)

    # Quantize
    FP8GlobalStateManager.reset()
    q_inp = MXFP8Quantizer(tex.DType.kFloat8E4M3, rowwise=False, columnwise=True)
    inp_mxfp8 = q_inp(inp_bf16)
    q_grad = MXFP8Quantizer(tex.DType.kFloat8E4M3, rowwise=False, columnwise=True)
    grad_mxfp8 = q_grad(grad_out_bf16)

    wgrad_list = [torch.empty(N, K, dtype=dtype, device="cuda") for _ in range(num_experts)]
    m_dev = torch.tensor(m_splits_list, dtype=torch.int64, device="cuda")

    ws = get_device_init_grouped_gemm_workspace()
    device_init_grouped_gemm(
        [inp_mxfp8], [grad_mxfp8], wgrad_list, dtype, ws,
        layout="NT", m_splits=m_dev, wgrad=True,
    )
    torch.cuda.synchronize()

    all_pass = True
    for i in range(num_experts):
        if m_splits_list[i] == 0:
            is_zero = wgrad_list[i].abs().max().item() < 1e-6
            status = "OK" if is_zero else "FAIL"
            print(f"  Expert {i}: m=0, max_abs={wgrad_list[i].abs().max().item():.6f} [{status}]")
            if not is_zero:
                all_pass = False
            continue
        ok, msg = _rmse_check(wgrad_list[i], wgrad_ref[i], f"Expert {i}")
        status = "OK" if ok else "FAIL"
        print(f"  Expert {i}: m={m_splits_list[i]}, {msg} [{status}]")
        if not ok:
            all_pass = False

    print(f"  Overall: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cublas", action="store_true", help="Skip cuBLAS comparison")
    args = parser.parse_args()
    use_cublas = not args.no_cublas

    check_hardware()

    results = []

    # --- fprop tests ---
    results.append(("fprop_uniform",
                    test_fprop(4, [128, 128, 128, 128], 768, 3072, use_cublas)))
    results.append(("fprop_varying_128",
                    test_fprop(4, [128, 256, 256, 128], 768, 3072, use_cublas)))
    results.append(("fprop_zero_expert",
                    test_fprop(4, [256, 0, 128, 128], 768, 3072, use_cublas)))
    results.append(("fprop_8_experts",
                    test_fprop(8, [128, 128, 128, 128, 256, 256, 128, 128], 256, 512, use_cublas)))
    results.append(("fprop_single",
                    test_fprop(1, [512], 768, 3072, use_cublas)))
    # This should be skipped (not multiples of 128):
    results.append(("fprop_NOT128",
                    test_fprop(4, [64, 256, 128, 64], 768, 3072, use_cublas)))

    # --- wgrad tests (m_splits must be multiples of 128 or 0) ---
    results.append(("wgrad_uniform",
                    test_wgrad(4, [128, 128, 128, 128], 768, 3072)))
    results.append(("wgrad_varying_128",
                    test_wgrad(4, [128, 256, 128, 256], 768, 3072)))
    results.append(("wgrad_zero_expert",
                    test_wgrad(4, [256, 0, 128, 128], 768, 3072)))
    # This should be skipped (not a multiple of 128):
    results.append(("wgrad_varying_NOT128",
                    test_wgrad(4, [64, 256, 128, 64], 768, 3072)))

    # Summary
    print(f"\n{'='*60}")
    passed = sum(1 for _, r in results if r is True)
    failed = sum(1 for _, r in results if r is False)
    skipped = sum(1 for _, r in results if r is None)
    for name, r in results:
        tag = "PASS" if r is True else ("FAIL" if r is False else "SKIP")
        print(f"  {tag:4s}  {name}")
    print(f"\nResults: {passed} passed, {failed} failed, {skipped} skipped, {len(results)} total")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
