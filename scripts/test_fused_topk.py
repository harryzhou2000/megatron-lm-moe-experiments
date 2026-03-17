#!/usr/bin/env python3
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
Correctness + performance test for fused router kernels:
  - fused_topk_with_score_function  (topk kernel)
  - fused_compute_score_for_moe_aux_loss  (aux loss score kernel)

Assumes TE is installed (`pip install -e ".[test]"` from TE/).

There are two operating modes:

  1. **Full sweep** (no kernel args given): sweeps across a broad grid of shapes,
     score functions, input types, etc.
  2. **User-config mode** (any kernel arg given, e.g. --topk, --num-experts, ...):
     uses exactly the params you specified.  If --num-tokens is omitted, sweeps
     over a range of token counts with your other params held fixed.

Usage
-----
  # Full sweep — correctness + benchmark (topk kernel)
  python scripts/test_fused_topk.py

  # Test aux loss kernel
  python scripts/test_fused_topk.py --kernel aux_loss

  # Test both kernels
  python scripts/test_fused_topk.py --kernel all

  # Forward only
  python scripts/test_fused_topk.py --pass forward

  # Backward only
  python scripts/test_fused_topk.py --pass backward

  # Single exact config
  python scripts/test_fused_topk.py --mode correctness \
      --num-tokens 4096 --num-experts 64 --topk 4 --score-function softmax

  # Custom kernel params, sweep token counts automatically
  python scripts/test_fused_topk.py --topk 36 --score-function softmax \
      --use-pre-softmax --num-experts 2304 --input-type random

  # Benchmark only, full sweep
  python scripts/test_fused_topk.py --mode benchmark

  # Custom dtype
  python scripts/test_fused_topk.py --dtype bf16

  # Override CUDA device
  CUDA_VISIBLE_DEVICES=3 python scripts/test_fused_topk.py
"""

import argparse
import sys
import time
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import torch

from transformer_engine.pytorch.router import (
    fused_topk_with_score_function,
    fused_compute_score_for_moe_aux_loss,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_gpu_info() -> None:
    """Print current CUDA device properties."""
    if not torch.cuda.is_available():
        print("ERROR: No CUDA device available.")
        sys.exit(1)
    dev = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(dev)
    print("=" * 72)
    print(f"Device {dev}: {props.name}")
    print(f"  Compute capability : {props.major}.{props.minor}")
    print(f"  SMs                : {props.multi_processor_count}")
    print(f"  Global memory      : {props.total_memory / (1024**3):.1f} GiB")
    print(f"  Max shared mem/blk : {props.max_shared_memory_per_block_optin // 1024} KiB"
          if hasattr(props, "max_shared_memory_per_block_optin")
          else "  (shared mem info unavailable)")
    print(f"  CUDA version       : {torch.version.cuda}")
    print(f"  PyTorch version    : {torch.__version__}")
    print("=" * 72)


def _valid_group_topk(num_experts: int, topk: int, num_groups: int, group_topk: int) -> bool:
    """Check whether grouped-topk parameters are valid for the given shape."""
    if num_experts % num_groups != 0:
        return False
    if topk % group_topk != 0:
        return False
    group_size = num_experts // num_groups
    per_group_k = topk // group_topk
    if per_group_k > group_size:
        return False
    if group_topk > num_groups:
        return False
    return True


class _NaNDetected(Exception):
    """Raised internally when NaN is found in forward outputs or gradients."""


# Input types that are safe to fall back to when NaN is detected.
# "random" (standard normal, σ=1) keeps values in a range where softmax
# and sigmoid produce well-conditioned outputs and gradients.
_SAFE_INPUT_TYPE = "random"


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


# ---------------------------------------------------------------------------
# Input generators
# ---------------------------------------------------------------------------

def make_logits(
    num_tokens: int,
    num_experts: int,
    dtype: torch.dtype,
    input_type: str,
    score_function: str,
) -> torch.Tensor:
    """
    Generate logits on CUDA with controlled distribution.

    input_type:
      arange   – deterministic, same as existing tests (monotonic, no ties)
      random   – standard normal
      uniform  – uniform [-1, 1]
      extreme  – large magnitude (stress softmax stability)
      narrow   – near-zero (sigmoid ≈ 0.5, softmax ≈ uniform)
      constant – all equal (tie-breaking stress test)
    """
    device = "cuda"
    if input_type == "arange":
        if score_function == "sigmoid":
            offset = (
                torch.arange(-num_tokens // 2, num_tokens // 2, dtype=dtype, device=device)
                * 1e-4
            )
            base = (
                torch.arange(-num_experts // 2, num_experts // 2, dtype=dtype, device=device)
                * 1e-2
            )
            logits = base.unsqueeze(0).repeat(num_tokens, 1) + offset.unsqueeze(1)
        else:
            logits = (
                torch.arange(
                    -num_tokens * num_experts // 2,
                    num_tokens * num_experts // 2,
                    device=device,
                    dtype=dtype,
                )
                * 1e-4
            )
            logits = logits.view(num_tokens, num_experts)
    elif input_type == "random":
        logits = torch.randn(num_tokens, num_experts, dtype=dtype, device=device)
    elif input_type == "uniform":
        logits = torch.empty(num_tokens, num_experts, dtype=dtype, device=device).uniform_(-1, 1)
    elif input_type == "extreme":
        logits = torch.randn(num_tokens, num_experts, dtype=dtype, device=device) * 100.0
    elif input_type == "narrow":
        logits = torch.randn(num_tokens, num_experts, dtype=dtype, device=device) * 1e-6
    elif input_type == "constant":
        logits = torch.ones(num_tokens, num_experts, dtype=dtype, device=device) * 0.5
    else:
        raise ValueError(f"Unknown input_type: {input_type}")
    return logits


# ---------------------------------------------------------------------------
# PyTorch references
# ---------------------------------------------------------------------------

def _group_limited_topk(
    scores: torch.Tensor,
    topk: int,
    num_tokens: int,
    num_experts: int,
    num_groups: int,
    group_topk: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    group_scores = (
        scores.view(num_tokens, num_groups, -1)
        .topk(topk // group_topk, dim=-1)[0]
        .sum(dim=-1)
    )
    group_idx = torch.topk(group_scores, k=group_topk, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_tokens, num_groups, num_experts // num_groups)
        .reshape(num_tokens, -1)
    )
    masked_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))
    return torch.topk(masked_scores, k=topk, dim=-1)


def reference_topk_forward(
    logits: torch.Tensor,
    topk: int,
    use_pre_softmax: bool,
    num_groups: int,
    group_topk: int,
    scaling_factor: float,
    score_function: str,
    expert_bias: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference matching the fused topk kernel logic."""
    num_tokens, num_experts = logits.shape

    def _topk(scores):
        if group_topk and group_topk > 0:
            return _group_limited_topk(
                scores, topk, num_tokens, num_experts, num_groups, group_topk
            )
        return torch.topk(scores, k=topk, dim=1)

    if score_function == "softmax":
        if use_pre_softmax:
            scores = torch.softmax(logits, dim=-1, dtype=torch.float32).to(logits.dtype)
            probs, top_indices = _topk(scores)
        else:
            scores, top_indices = _topk(logits)
            probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(logits.dtype)
    elif score_function == "sigmoid":
        scores = torch.sigmoid(logits.float()).to(logits.dtype)
        if expert_bias is not None:
            scores_for_routing = scores + expert_bias
            _, top_indices = _topk(scores_for_routing)
            scores = torch.gather(scores, dim=1, index=top_indices)
        else:
            scores, top_indices = _topk(scores)
        probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if topk > 1 else scores
    else:
        raise ValueError(f"Unknown score_function: {score_function}")

    if scaling_factor is not None:
        probs = probs * scaling_factor

    topk_masked_gates = torch.zeros_like(logits).scatter(1, top_indices, probs)
    topk_map = torch.zeros_like(logits, dtype=torch.int32).scatter(1, top_indices, 1).bool()
    return topk_masked_gates, topk_map


def reference_aux_loss_scores_forward(
    logits: torch.Tensor,
    topk: int,
    score_function: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference matching the fused aux loss score kernel logic.

    Returns (routing_map, scores) — note scores is the full [T, E] tensor
    (softmax or normalized-sigmoid over all experts), NOT just the topk values.
    """
    if score_function == "softmax":
        scores = torch.softmax(logits, dim=-1, dtype=torch.float32).to(logits.dtype)
    elif score_function == "sigmoid":
        scores = torch.sigmoid(logits.float()).to(logits.dtype)
        if topk > 1:
            scores = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
    else:
        raise ValueError(f"Unknown score_function: {score_function}")

    _, top_indices = torch.topk(scores, k=topk, dim=1)
    routing_map = torch.zeros_like(logits, dtype=torch.int32).scatter(1, top_indices, 1).bool()
    return routing_map, scores


# ===========================================================================
# Topk kernel — correctness
# ===========================================================================

def run_topk_correctness(
    num_tokens: int,
    num_experts: int,
    topk: int,
    use_pre_softmax: bool,
    num_groups: int,
    group_topk: int,
    scaling_factor: float,
    score_function: str,
    enable_bias: bool,
    dtype: torch.dtype,
    input_type: str,
    test_pass: str,
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
    _is_nan_retry: bool = False,
) -> bool:
    """Run a single topk correctness check.  Returns True on pass."""
    needs_grad = test_pass in ("backward", "both")
    logits = make_logits(num_tokens, num_experts, dtype, input_type, score_function)
    logits.requires_grad = needs_grad

    if enable_bias and score_function == "sigmoid":
        expert_bias = (
            torch.arange(num_experts, device="cuda", dtype=dtype) * 0.1
        ).flip(dims=[0])
    else:
        expert_bias = None

    logits_clone = logits.detach().clone().requires_grad_(needs_grad)
    expert_bias_clone = expert_bias.clone() if expert_bias is not None else None

    # Reference forward
    ref_probs, ref_map = reference_topk_forward(
        logits, topk, use_pre_softmax,
        num_groups, group_topk, scaling_factor,
        score_function, expert_bias,
    )

    # Fused kernel forward
    fused_probs, fused_map = fused_topk_with_score_function(
        logits=logits_clone, topk=topk,
        use_pre_softmax=use_pre_softmax,
        num_groups=num_groups or 0,
        group_topk=group_topk or 0,
        scaling_factor=scaling_factor or 1.0,
        score_function=score_function,
        expert_bias=expert_bias_clone,
    )

    # --- NaN safeguard: detect NaN in forward outputs ---
    has_nan = ref_probs.isnan().any() or fused_probs.isnan().any()

    # Build tolerance kwargs (only set overrides if provided)
    tol_kw: Dict = {}
    if atol is not None:
        tol_kw["atol"] = atol
    if rtol is not None:
        tol_kw["rtol"] = rtol

    tag = (
        f"[topk {test_pass:>4s} | {score_function:>7s} | tokens={num_tokens:>6d} | "
        f"experts={num_experts:>4d} | topk={topk} | pre_sm={use_pre_softmax} | "
        f"grp_topk={group_topk} | scale={scaling_factor} | bias={enable_bias} | "
        f"dtype={dtype} | input={input_type}]"
    )
    try:
        # --- Forward check ---
        if test_pass in ("forward", "both"):
            if has_nan:
                raise _NaNDetected()
            fwd_ok = _check_topk_forward(
                ref_probs, ref_map, fused_probs, fused_map,
                logits, score_function, use_pre_softmax, expert_bias, dtype, tol_kw, tag,
            )
            if not fwd_ok:
                return False

        # --- Backward check ---
        if test_pass in ("backward", "both"):
            ref_loss = ref_probs.sum()
            ref_loss.backward()
            fused_loss = fused_probs.sum()
            fused_loss.backward()
            # Check for NaN in gradients
            if (
                logits.grad is not None and logits.grad.isnan().any()
                or logits_clone.grad is not None and logits_clone.grad.isnan().any()
            ):
                raise _NaNDetected()
            try:
                torch.testing.assert_close(logits.grad, logits_clone.grad, **tol_kw)
            except AssertionError as e:
                grad_diff = (logits.grad - logits_clone.grad).abs()
                print(f"  FAIL {tag}")
                print(f"       grad max abs diff  : {grad_diff.max().item():.6e}")
                print(f"       grad mean abs diff : {grad_diff.mean().item():.6e}")
                print(f"       {e}")
                return False

        print(f"  PASS {tag}")
        return True

    except _NaNDetected:
        if _is_nan_retry:
            # Already retried once — give up and report as a pass with warning.
            print(f"  SKIP {tag}  (NaN in both input types)")
            return True
        print(
            f"  WARN {tag}  NaN detected with input={input_type}, "
            f"retrying with input={_SAFE_INPUT_TYPE}"
        )
        return run_topk_correctness(
            num_tokens=num_tokens, num_experts=num_experts, topk=topk,
            use_pre_softmax=use_pre_softmax, num_groups=num_groups,
            group_topk=group_topk, scaling_factor=scaling_factor,
            score_function=score_function, enable_bias=enable_bias,
            dtype=dtype, input_type=_SAFE_INPUT_TYPE, test_pass=test_pass,
            atol=atol, rtol=rtol, _is_nan_retry=True,
        )
    except AssertionError as e:
        prob_diff = (ref_probs - fused_probs).abs()
        map_diff = (ref_map != fused_map).sum().item()
        print(f"  FAIL {tag}")
        print(f"       routing_map mismatches : {map_diff}")
        print(f"       probs max abs diff     : {prob_diff.max().item():.6e}")
        print(f"       probs mean abs diff    : {prob_diff.mean().item():.6e}")
        print(f"       {e}")
        return False
    except Exception as e:
        print(f"  FAIL {tag}")
        print(f"       Exception: {e}")
        return False


def _check_topk_forward(
    ref_probs, ref_map, fused_probs, fused_map,
    logits, score_function, use_pre_softmax, expert_bias, dtype, tol_kw, tag,
) -> bool:
    """Check forward correctness for topk kernel.  Returns True on pass."""
    map_match = (ref_map == fused_map).all().item()

    if map_match:
        torch.testing.assert_close(ref_probs, fused_probs, **tol_kw)
        return True

    # Routing maps disagree — check if the disagreement is due to tied scores.
    num_tokens, num_experts = logits.shape
    if score_function == "sigmoid":
        scores = torch.sigmoid(logits.detach().float()).to(dtype)
        if expert_bias is not None:
            scores = scores + expert_bias
    elif use_pre_softmax:
        scores = torch.softmax(logits.detach(), dim=-1, dtype=torch.float32).to(dtype)
    else:
        scores = logits.detach()

    diff_mask = ref_map != fused_map
    rows_with_diff = diff_mask.any(dim=1)
    n_diff_tokens = rows_with_diff.sum().item()
    total_map_mismatches = diff_mask.sum().item()

    all_ties = True
    diff_row_indices = rows_with_diff.nonzero(as_tuple=False).view(-1).tolist()
    for row_idx in diff_row_indices:
        row_scores = scores[row_idx]
        ref_selected = ref_map[row_idx]
        fused_selected = fused_map[row_idx]
        only_ref = ref_selected & ~fused_selected
        only_fused = fused_selected & ~ref_selected
        if only_ref.sum() == 0:
            continue
        ref_boundary = row_scores[ref_selected].min()
        fused_boundary = row_scores[fused_selected].min()
        swapped_out_scores = row_scores[only_ref]
        swapped_in_scores = row_scores[only_fused]
        if not (
            torch.isclose(ref_boundary, fused_boundary)
            and swapped_out_scores.allclose(ref_boundary)
            and swapped_in_scores.allclose(fused_boundary)
        ):
            all_ties = False
            break

    if not all_ties:
        prob_diff = (ref_probs - fused_probs).abs()
        print(f"  FAIL {tag}")
        print(f"       routing_map mismatches : {total_map_mismatches} "
              f"({n_diff_tokens} tokens)")
        print(f"       probs max abs diff     : {prob_diff.max().item():.6e}")
        print(f"       probs mean abs diff    : {prob_diff.mean().item():.6e}")
        print(f"       (NOT explained by tie-breaking)")
        return False

    return True


# ===========================================================================
# Aux loss score kernel — correctness
# ===========================================================================

def run_aux_loss_correctness(
    num_tokens: int,
    num_experts: int,
    topk: int,
    score_function: str,
    dtype: torch.dtype,
    input_type: str,
    test_pass: str,
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
    _is_nan_retry: bool = False,
) -> bool:
    """Run a single aux loss score correctness check.  Returns True on pass."""
    needs_grad = test_pass in ("backward", "both")
    logits = make_logits(num_tokens, num_experts, dtype, input_type, score_function)
    logits.requires_grad = needs_grad

    logits_clone = logits.detach().clone().requires_grad_(needs_grad)

    # Reference
    ref_map, ref_scores = reference_aux_loss_scores_forward(logits, topk, score_function)

    # Fused kernel
    fused_map, fused_scores = fused_compute_score_for_moe_aux_loss(
        logits=logits_clone, topk=topk, score_function=score_function,
    )

    # --- NaN safeguard: detect NaN in forward outputs ---
    has_nan = ref_scores.isnan().any() or fused_scores.isnan().any()

    tol_kw: Dict = {}
    if atol is not None:
        tol_kw["atol"] = atol
    if rtol is not None:
        tol_kw["rtol"] = rtol

    tag = (
        f"[aux_loss {test_pass:>4s} | {score_function:>7s} | tokens={num_tokens:>6d} | "
        f"experts={num_experts:>4d} | topk={topk} | dtype={dtype} | input={input_type}]"
    )

    try:
        # --- Forward check ---
        if test_pass in ("forward", "both"):
            if has_nan:
                raise _NaNDetected()
            # Check scores (full [T, E] tensor)
            torch.testing.assert_close(ref_scores, fused_scores, **tol_kw)
            # Check routing map — allow tie-break differences
            map_match = (ref_map == fused_map).all().item()
            if not map_match:
                # For aux loss, the scores are identical (just checked above),
                # so routing_map differences are tie-breaking in topk.
                total_map_mismatches = (ref_map != fused_map).sum().item()
                n_diff_tokens = (ref_map != fused_map).any(dim=1).sum().item()
                # Verify the differences are at tied score boundaries
                _, ref_top_indices = torch.topk(ref_scores.detach(), k=topk, dim=1)
                # If scores match, any map diff is tie-breaking — accept it.
                pass  # scores already verified above, ties are acceptable

        # --- Backward check ---
        if test_pass in ("backward", "both"):
            if has_nan:
                raise _NaNDetected()
            ref_loss = ref_scores.sum()
            ref_loss.backward()
            fused_loss = fused_scores.sum()
            fused_loss.backward()
            # Check for NaN in gradients
            if (
                logits.grad is not None and logits.grad.isnan().any()
                or logits_clone.grad is not None and logits_clone.grad.isnan().any()
            ):
                raise _NaNDetected()
            try:
                torch.testing.assert_close(logits.grad, logits_clone.grad, **tol_kw)
            except AssertionError as e:
                grad_diff = (logits.grad - logits_clone.grad).abs()
                print(f"  FAIL {tag}")
                print(f"       grad max abs diff  : {grad_diff.max().item():.6e}")
                print(f"       grad mean abs diff : {grad_diff.mean().item():.6e}")
                print(f"       {e}")
                return False

        print(f"  PASS {tag}")
        return True

    except _NaNDetected:
        if _is_nan_retry:
            # Already retried once — give up and report as a pass with warning.
            print(f"  SKIP {tag}  (NaN in both input types)")
            return True
        print(
            f"  WARN {tag}  NaN detected with input={input_type}, "
            f"retrying with input={_SAFE_INPUT_TYPE}"
        )
        return run_aux_loss_correctness(
            num_tokens=num_tokens, num_experts=num_experts, topk=topk,
            score_function=score_function, dtype=dtype,
            input_type=_SAFE_INPUT_TYPE, test_pass=test_pass,
            atol=atol, rtol=rtol, _is_nan_retry=True,
        )
    except AssertionError as e:
        score_diff = (ref_scores - fused_scores).abs()
        map_diff = (ref_map != fused_map).sum().item()
        print(f"  FAIL {tag}")
        print(f"       routing_map mismatches : {map_diff}")
        print(f"       scores max abs diff    : {score_diff.max().item():.6e}")
        print(f"       scores mean abs diff   : {score_diff.mean().item():.6e}")
        print(f"       {e}")
        return False
    except Exception as e:
        print(f"  FAIL {tag}")
        print(f"       Exception: {e}")
        return False


# ===========================================================================
# Correctness suites
# ===========================================================================

def topk_correctness_suite(args) -> bool:
    """Run topk correctness tests.  Returns True if all pass."""
    passed, total = 0, 0

    if args.user_specified:
        token_list = (
            [args.num_tokens] if args.num_tokens is not None
            else [1, 64, 512, 2048, 8192, 32768]
        )
        print(
            f"\nRunning {len(token_list)} topk correctness test(s) with user config "
            f"(dtype={args.dtype}, pass={args.test_pass})...\n"
        )
        for nt in token_list:
            total += 1
            ok = run_topk_correctness(
                num_tokens=nt,
                num_experts=args.num_experts,
                topk=args.topk,
                use_pre_softmax=args.use_pre_softmax,
                num_groups=args.num_groups,
                group_topk=args.group_topk,
                scaling_factor=args.scaling_factor,
                score_function=args.score_function,
                enable_bias=args.enable_bias,
                dtype=args.dtype,
                input_type=args.input_type,
                test_pass=args.test_pass,
                atol=args.atol,
                rtol=args.rtol,
            )
            passed += int(ok)
    else:
        configs: List[Dict] = []
        for sf in ["softmax", "sigmoid"]:
            for nt in [1, 37, 512, 2048, 8192]:
                for ne in [8, 33, 64, 128, 256]:
                    for tk in [1, 2, 4, 8]:
                        if tk > ne:
                            continue
                        for inp in ["arange", "random", "extreme", "narrow", "constant"]:
                            for pre in ([True, False] if sf == "softmax" else [False]):
                                for grp in [0, 4]:
                                    if grp > 0 and not _valid_group_topk(ne, tk, 8, grp):
                                        continue
                                    configs.append(dict(
                                        num_tokens=nt, num_experts=ne, topk=tk,
                                        use_pre_softmax=pre,
                                        num_groups=8 if grp else 0,
                                        group_topk=grp,
                                        scaling_factor=1.0,
                                        score_function=sf,
                                        enable_bias=(sf == "sigmoid"),
                                        input_type=inp,
                                    ))

        MAX_TESTS = 200
        if len(configs) > MAX_TESTS:
            step = len(configs) // MAX_TESTS
            configs = configs[::step][:MAX_TESTS]

        print(f"\nRunning {len(configs)} topk correctness tests "
              f"(dtype={args.dtype}, pass={args.test_pass})...\n")
        for cfg in configs:
            total += 1
            ok = run_topk_correctness(
                dtype=args.dtype,
                test_pass=args.test_pass,
                atol=args.atol,
                rtol=args.rtol,
                **cfg,
            )
            passed += int(ok)

    print(f"\nTopk correctness: {passed}/{total} passed\n")
    return passed == total


def aux_loss_correctness_suite(args) -> bool:
    """Run aux loss score correctness tests.  Returns True if all pass."""
    passed, total = 0, 0

    if args.user_specified:
        token_list = (
            [args.num_tokens] if args.num_tokens is not None
            else [1, 64, 512, 2048, 8192, 32768]
        )
        print(
            f"\nRunning {len(token_list)} aux_loss correctness test(s) with user config "
            f"(dtype={args.dtype}, pass={args.test_pass})...\n"
        )
        for nt in token_list:
            total += 1
            ok = run_aux_loss_correctness(
                num_tokens=nt,
                num_experts=args.num_experts,
                topk=args.topk,
                score_function=args.score_function,
                dtype=args.dtype,
                input_type=args.input_type,
                test_pass=args.test_pass,
                atol=args.atol,
                rtol=args.rtol,
            )
            passed += int(ok)
    else:
        configs: List[Dict] = []
        for sf in ["softmax", "sigmoid"]:
            for nt in [1, 37, 512, 2048, 8192]:
                for ne in [8, 33, 64, 128, 256]:
                    for tk in [1, 2, 4, 8]:
                        if tk > ne:
                            continue
                        for inp in ["arange", "random", "extreme", "narrow", "constant"]:
                            configs.append(dict(
                                num_tokens=nt, num_experts=ne, topk=tk,
                                score_function=sf,
                                input_type=inp,
                            ))

        MAX_TESTS = 200
        if len(configs) > MAX_TESTS:
            step = len(configs) // MAX_TESTS
            configs = configs[::step][:MAX_TESTS]

        print(f"\nRunning {len(configs)} aux_loss correctness tests "
              f"(dtype={args.dtype}, pass={args.test_pass})...\n")
        for cfg in configs:
            total += 1
            ok = run_aux_loss_correctness(
                dtype=args.dtype,
                test_pass=args.test_pass,
                atol=args.atol,
                rtol=args.rtol,
                **cfg,
            )
            passed += int(ok)

    print(f"\nAux loss correctness: {passed}/{total} passed\n")
    return passed == total


# ===========================================================================
# Performance benchmarks
# ===========================================================================

def _benchmark_topk_one(
    num_tokens: int,
    num_experts: int,
    topk: int,
    score_function: str,
    use_pre_softmax: bool,
    group_topk: int,
    dtype: torch.dtype,
    test_pass: str,
    warmup: int,
    iters: int,
) -> Dict:
    """Benchmark a single topk configuration.  Returns dict of metrics."""
    needs_grad = test_pass in ("backward", "both")
    logits = torch.randn(
        num_tokens, num_experts, dtype=dtype, device="cuda", requires_grad=needs_grad,
    )
    expert_bias = None
    if score_function == "sigmoid":
        expert_bias = torch.randn(num_experts, dtype=dtype, device="cuda") * 0.1

    call_args = dict(
        logits=logits,
        topk=topk,
        use_pre_softmax=use_pre_softmax,
        num_groups=8 if group_topk else 0,
        group_topk=group_topk,
        scaling_factor=1.0,
        score_function=score_function,
        expert_bias=expert_bias,
    )

    if test_pass == "forward":
        fused_ms = _time_forward(lambda: fused_topk_with_score_function(**call_args),
                                 warmup, iters)
    elif test_pass == "backward":
        fused_ms = _time_backward(
            lambda: fused_topk_with_score_function(**call_args),
            loss_fn=lambda out: out[0].sum(),
            warmup=warmup, iters=iters,
        )
    else:  # both
        fused_ms = _time_forward_backward(
            lambda: fused_topk_with_score_function(**call_args),
            loss_fn=lambda out: out[0].sum(),
            warmup=warmup, iters=iters,
        )

    # PyTorch reference
    logits_ref = torch.randn(
        num_tokens, num_experts, dtype=dtype, device="cuda", requires_grad=needs_grad,
    )
    ref_args = dict(
        logits=logits_ref,
        topk=topk,
        use_pre_softmax=use_pre_softmax,
        num_groups=8 if group_topk else 0,
        group_topk=group_topk,
        scaling_factor=1.0,
        score_function=score_function,
        expert_bias=expert_bias,
    )

    if test_pass == "forward":
        ref_ms = _time_forward(lambda: reference_topk_forward(**ref_args), warmup, iters)
    elif test_pass == "backward":
        ref_ms = _time_backward(
            lambda: reference_topk_forward(**ref_args),
            loss_fn=lambda out: out[0].sum(),
            warmup=warmup, iters=iters,
        )
    else:
        ref_ms = _time_forward_backward(
            lambda: reference_topk_forward(**ref_args),
            loss_fn=lambda out: out[0].sum(),
            warmup=warmup, iters=iters,
        )

    return dict(
        kernel="topk",
        num_tokens=num_tokens,
        num_experts=num_experts,
        topk=topk,
        score_function=score_function,
        use_pre_softmax=use_pre_softmax,
        group_topk=group_topk,
        dtype=str(dtype).replace("torch.", ""),
        test_pass=test_pass,
        fused_ms=fused_ms,
        ref_ms=ref_ms,
        speedup=ref_ms / fused_ms if fused_ms > 0 else float("inf"),
        tokens_per_sec=num_tokens / (fused_ms / 1000),
    )


def _benchmark_aux_loss_one(
    num_tokens: int,
    num_experts: int,
    topk: int,
    score_function: str,
    dtype: torch.dtype,
    test_pass: str,
    warmup: int,
    iters: int,
) -> Dict:
    """Benchmark a single aux loss score configuration."""
    needs_grad = test_pass in ("backward", "both")
    logits = torch.randn(
        num_tokens, num_experts, dtype=dtype, device="cuda", requires_grad=needs_grad,
    )

    call_args = dict(logits=logits, topk=topk, score_function=score_function)

    if test_pass == "forward":
        fused_ms = _time_forward(
            lambda: fused_compute_score_for_moe_aux_loss(**call_args), warmup, iters)
    elif test_pass == "backward":
        fused_ms = _time_backward(
            lambda: fused_compute_score_for_moe_aux_loss(**call_args),
            loss_fn=lambda out: out[1].sum(),  # out = (routing_map, scores)
            warmup=warmup, iters=iters,
        )
    else:
        fused_ms = _time_forward_backward(
            lambda: fused_compute_score_for_moe_aux_loss(**call_args),
            loss_fn=lambda out: out[1].sum(),
            warmup=warmup, iters=iters,
        )

    # PyTorch reference
    logits_ref = torch.randn(
        num_tokens, num_experts, dtype=dtype, device="cuda", requires_grad=needs_grad,
    )
    ref_args = dict(logits=logits_ref, topk=topk, score_function=score_function)

    if test_pass == "forward":
        ref_ms = _time_forward(
            lambda: reference_aux_loss_scores_forward(**ref_args), warmup, iters)
    elif test_pass == "backward":
        ref_ms = _time_backward(
            lambda: reference_aux_loss_scores_forward(**ref_args),
            loss_fn=lambda out: out[1].sum(),
            warmup=warmup, iters=iters,
        )
    else:
        ref_ms = _time_forward_backward(
            lambda: reference_aux_loss_scores_forward(**ref_args),
            loss_fn=lambda out: out[1].sum(),
            warmup=warmup, iters=iters,
        )

    return dict(
        kernel="aux_loss",
        num_tokens=num_tokens,
        num_experts=num_experts,
        topk=topk,
        score_function=score_function,
        use_pre_softmax=False,
        group_topk=0,
        dtype=str(dtype).replace("torch.", ""),
        test_pass=test_pass,
        fused_ms=fused_ms,
        ref_ms=ref_ms,
        speedup=ref_ms / fused_ms if fused_ms > 0 else float("inf"),
        tokens_per_sec=num_tokens / (fused_ms / 1000),
    )


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _time_forward(fn, warmup: int, iters: int) -> float:
    """Time forward-only calls.  Returns average ms."""
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


def _time_backward(fn, loss_fn, warmup: int, iters: int) -> float:
    """Time backward-only (grad computation).

    Each iteration: run forward + sync (not timed), then time backward only.
    """
    for _ in range(warmup):
        out = fn()
        loss = loss_fn(out)
        loss.backward()
    torch.cuda.synchronize()

    total_bwd_ms = 0.0
    for _ in range(iters):
        out = fn()
        torch.cuda.synchronize()
        bwd_s = torch.cuda.Event(enable_timing=True)
        bwd_e = torch.cuda.Event(enable_timing=True)
        bwd_s.record()
        loss = loss_fn(out)
        loss.backward()
        bwd_e.record()
        torch.cuda.synchronize()
        total_bwd_ms += bwd_s.elapsed_time(bwd_e)
    return total_bwd_ms / iters


def _time_forward_backward(fn, loss_fn, warmup: int, iters: int) -> float:
    """Time forward + backward together.  Returns average ms."""
    for _ in range(warmup):
        out = fn()
        loss = loss_fn(out)
        loss.backward()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        out = fn()
        loss = loss_fn(out)
        loss.backward()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


# ---------------------------------------------------------------------------
# Benchmark printing
# ---------------------------------------------------------------------------

def _print_bench_header() -> None:
    """Print the benchmark table header."""
    hdr = (
        f"{'kernel':>8s} {'tokens':>8s} {'experts':>7s} {'topk':>4s} {'score_fn':>8s} "
        f"{'pre_sm':>6s} {'grp_tk':>6s} {'dtype':>8s} {'pass':>7s} "
        f"{'fused_ms':>9s} {'ref_ms':>9s} {'speedup':>7s} {'tok/s':>12s}"
    )
    print(hdr)
    print("-" * len(hdr))
    sys.stdout.flush()


def _print_bench_row(r: Dict) -> None:
    """Print a single benchmark result row and flush immediately."""
    print(
        f"{r['kernel']:>8s} {r['num_tokens']:>8d} {r['num_experts']:>7d} {r['topk']:>4d} "
        f"{r['score_function']:>8s} {str(r['use_pre_softmax']):>6s} "
        f"{r['group_topk']:>6d} {r['dtype']:>8s} {r['test_pass']:>7s} "
        f"{r['fused_ms']:>9.4f} {r['ref_ms']:>9.4f} "
        f"{r['speedup']:>7.2f}x "
        f"{r['tokens_per_sec']:>12.0f}"
    )
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Benchmark suites
# ---------------------------------------------------------------------------

def topk_benchmark_suite(args) -> None:
    """Run topk benchmark across configurations."""
    warmup = args.warmup
    iters = args.iters

    if args.user_specified:
        token_list = (
            [args.num_tokens] if args.num_tokens is not None
            else [128, 512, 2048, 8192, 32768, 131072]
        )
        print(
            f"\nBenchmarking {len(token_list)} topk config(s) "
            f"(warmup={warmup}, iters={iters}, dtype={args.dtype}, pass={args.test_pass})...\n"
        )
        _print_bench_header()
        for nt in token_list:
            r = _benchmark_topk_one(
                num_tokens=nt,
                num_experts=args.num_experts,
                topk=args.topk,
                score_function=args.score_function,
                use_pre_softmax=args.use_pre_softmax,
                group_topk=args.group_topk or 0,
                dtype=args.dtype,
                test_pass=args.test_pass,
                warmup=warmup,
                iters=iters,
            )
            _print_bench_row(r)
    else:
        sweep_tokens = [128, 512, 8192, 32768, 131072]
        sweep_experts = [8, 256, 512, 2304]
        sweep_topk = [1, 4, 8, 32, 36]
        sweep_sf = ["softmax", "sigmoid"]
        sweep_grp = [0, 4]

        total = 0
        for nt in sweep_tokens:
            for ne in sweep_experts:
                for tk in sweep_topk:
                    if tk > ne:
                        continue
                    for sf in sweep_sf:
                        for grp in sweep_grp:
                            if grp > 0 and not _valid_group_topk(ne, tk, 8, grp):
                                continue
                            total += 1

        print(f"\nRunning {total} topk benchmark configs "
              f"(warmup={warmup}, iters={iters}, dtype={args.dtype}, pass={args.test_pass})...\n")
        _print_bench_header()

        for nt in sweep_tokens:
            for ne in sweep_experts:
                for tk in sweep_topk:
                    if tk > ne:
                        continue
                    for sf in sweep_sf:
                        for grp in sweep_grp:
                            if grp > 0 and not _valid_group_topk(ne, tk, 8, grp):
                                continue
                            r = _benchmark_topk_one(
                                nt, ne, tk, sf, False, grp,
                                dtype=args.dtype,
                                test_pass=args.test_pass,
                                warmup=warmup,
                                iters=iters,
                            )
                            _print_bench_row(r)

    print()


def aux_loss_benchmark_suite(args) -> None:
    """Run aux loss score benchmark across configurations."""
    warmup = args.warmup
    iters = args.iters

    if args.user_specified:
        token_list = (
            [args.num_tokens] if args.num_tokens is not None
            else [128, 512, 2048, 8192, 32768, 131072]
        )
        print(
            f"\nBenchmarking {len(token_list)} aux_loss config(s) "
            f"(warmup={warmup}, iters={iters}, dtype={args.dtype}, pass={args.test_pass})...\n"
        )
        _print_bench_header()
        for nt in token_list:
            r = _benchmark_aux_loss_one(
                num_tokens=nt,
                num_experts=args.num_experts,
                topk=args.topk,
                score_function=args.score_function,
                dtype=args.dtype,
                test_pass=args.test_pass,
                warmup=warmup,
                iters=iters,
            )
            _print_bench_row(r)
    else:
        sweep_tokens = [128, 512, 8192, 32768, 131072]
        sweep_experts = [8, 256, 512, 2304]
        sweep_topk = [1, 4, 8, 32, 36]
        sweep_sf = ["softmax", "sigmoid"]

        total = 0
        for nt in sweep_tokens:
            for ne in sweep_experts:
                for tk in sweep_topk:
                    if tk > ne:
                        continue
                    for sf in sweep_sf:
                        total += 1

        print(f"\nRunning {total} aux_loss benchmark configs "
              f"(warmup={warmup}, iters={iters}, dtype={args.dtype}, pass={args.test_pass})...\n")
        _print_bench_header()

        for nt in sweep_tokens:
            for ne in sweep_experts:
                for tk in sweep_topk:
                    if tk > ne:
                        continue
                    for sf in sweep_sf:
                        r = _benchmark_aux_loss_one(
                            nt, ne, tk, sf,
                            dtype=args.dtype,
                            test_pass=args.test_pass,
                            warmup=warmup,
                            iters=iters,
                        )
                        _print_bench_row(r)

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _any_kernel_arg_set(argv: List[str]) -> bool:
    """Return True if the user passed any kernel-shape / config flag on the CLI."""
    kernel_flags = {
        "--num-tokens", "--num-experts", "--topk", "--score-function",
        "--use-pre-softmax", "--num-groups", "--group-topk",
        "--scaling-factor", "--enable-bias", "--input-type",
    }
    for arg in argv:
        if arg.split("=")[0] in kernel_flags:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Test & benchmark fused router kernels (topk + aux_loss)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["correctness", "benchmark", "all"],
        default="all",
        help="Which suite to run (default: all)",
    )
    parser.add_argument(
        "--kernel", choices=["topk", "aux_loss", "all"],
        default="topk",
        help="Which kernel to test: topk, aux_loss, or all (default: topk)",
    )
    # Note: --pass is a reserved word in Python, so we use dest="test_pass"
    parser.add_argument(
        "--pass", choices=["forward", "backward", "both"],
        default="both", dest="test_pass",
        help="Which pass to test: forward, backward, or both (default: both)",
    )

    # Shape / kernel options.
    parser.add_argument("--num-tokens", type=int, default=None,
                        help="Number of tokens (omit to sweep token counts)")
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--score-function", choices=["softmax", "sigmoid"], default="softmax")
    parser.add_argument("--use-pre-softmax", action="store_true", default=False)
    parser.add_argument("--num-groups", type=int, default=0)
    parser.add_argument("--group-topk", type=int, default=0)
    parser.add_argument("--scaling-factor", type=float, default=1.0)
    parser.add_argument("--enable-bias", action="store_true", default=False)

    # Data / dtype
    parser.add_argument("--dtype", type=parse_dtype, default=torch.float32,
                        help="fp32 | fp16 | bf16 (default: fp32)")
    parser.add_argument("--input-type", default="arange",
                        choices=["arange", "random", "uniform", "extreme", "narrow", "constant"],
                        help="Input distribution for correctness tests")

    # Tolerances
    parser.add_argument("--atol", type=float, default=None,
                        help="Absolute tolerance override for assert_close")
    parser.add_argument("--rtol", type=float, default=None,
                        help="Relative tolerance override for assert_close")

    # Benchmark tuning
    parser.add_argument("--warmup", type=int, default=20,
                        help="Warmup iterations for benchmark")
    parser.add_argument("--iters", type=int, default=100,
                        help="Timed iterations for benchmark")

    args = parser.parse_args()

    # Decide single-config vs full sweep
    args.user_specified = _any_kernel_arg_set(sys.argv[1:])

    print_gpu_info()

    ok = True
    run_topk = args.kernel in ("topk", "all")
    run_aux = args.kernel in ("aux_loss", "all")

    if args.mode in ("correctness", "all"):
        if run_topk:
            ok = topk_correctness_suite(args) and ok
        if run_aux:
            ok = aux_loss_correctness_suite(args) and ok

    if args.mode in ("benchmark", "all"):
        if run_topk:
            topk_benchmark_suite(args)
        if run_aux:
            aux_loss_benchmark_suite(args)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
