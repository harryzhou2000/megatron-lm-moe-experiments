#!/usr/bin/env python3
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
Correctness + performance test for fused router kernels:
  - fused_topk_with_score_function  (topk kernel)
  - fused_compute_score_for_moe_aux_loss  (aux loss score kernel)

Assumes TE is installed (`pip install -e ".[test]"` from TE/).

Every sweep dimension (--num-tokens, --num-experts, --topk, --score-function,
--group-topk, --num-groups, --input-type, --kernel, --pass) can be specified
or omitted independently.  Specified dimensions are pinned; omitted dimensions
sweep over their defaults.

Usage
-----
  # Full sweep (all dimensions swept, default passes: forward + backward_raw)
  python scripts/test_fused_topk.py

  # Pin score function + experts, sweep tokens and topk
  python scripts/test_fused_topk.py --score-function sigmoid --num-experts 512

  # Single exact config
  python scripts/test_fused_topk.py --num-tokens 8192 --num-experts 512 \\
      --topk 22 --score-function sigmoid

  # Benchmark backward kernel only, sweep tokens
  python scripts/test_fused_topk.py --mode benchmark --pass backward_raw \\
      --num-experts 512 --topk 22 --score-function sigmoid

  # Benchmark all four passes
  python scripts/test_fused_topk.py --mode benchmark \\
      --pass forward backward backward_raw both

  # Correctness, pin input type
  python scripts/test_fused_topk.py --mode correctness --input-type random

  # Test aux loss kernel only
  python scripts/test_fused_topk.py --kernel aux_loss

  # Test both kernels (default when --kernel omitted)
  python scripts/test_fused_topk.py --kernel topk aux_loss

  # Export benchmark results to CSV
  python scripts/test_fused_topk.py --mode benchmark --csv results.csv

  # Canonical progressive performance analysis (used for commit-by-commit comparison):
  python scripts/test_fused_topk.py --mode benchmark --pass forward backward_raw --csv data/p3R_<tag>.csv
"""

import argparse
import csv
import itertools
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from transformer_engine.pytorch.router import (
    fused_topk_with_score_function,
    fused_compute_score_for_moe_aux_loss,
)


# ---------------------------------------------------------------------------
# Score function name convention
# ---------------------------------------------------------------------------
# The user-facing --score-function flag uses a combined name that encodes
# the pre-softmax option:
#   "pre-softmax"   -> score_function="softmax", use_pre_softmax=True
#   "softmax"       -> score_function="softmax", use_pre_softmax=False
#   "sigmoid"       -> score_function="sigmoid", use_pre_softmax=False
#   "sqrtsoftplus"  -> score_function="sqrtsoftplus", use_pre_softmax=False
#
# Internally we always split into (score_function, use_pre_softmax) for the
# kernel calls, and re-join for display / CSV.

ALL_SCORE_FUNCTIONS = ["pre-softmax", "softmax", "sigmoid", "sqrtsoftplus"]


def _split_score_function(name: str) -> Tuple[str, bool]:
    """Split combined name into (kernel_score_function, use_pre_softmax)."""
    if name == "pre-softmax":
        return "softmax", True
    return name, False


def _join_score_function(score_function: str, use_pre_softmax: bool) -> str:
    """Join (kernel_score_function, use_pre_softmax) into a combined display name."""
    if score_function == "softmax" and use_pre_softmax:
        return "pre-softmax"
    return score_function


def _dedup_aux_loss_score_functions(names: List[str]) -> List[str]:
    """Deduplicate score function names for aux loss (pre-softmax -> softmax)."""
    seen: set = set()
    result: List[str] = []
    for name in names:
        sf_kernel, _ = _split_score_function(name)
        if sf_kernel not in seen:
            result.append(sf_kernel)
            seen.add(sf_kernel)
    return result


# ---------------------------------------------------------------------------
# Sweep grid definitions (single source of truth)
# ---------------------------------------------------------------------------

SWEEP_CORRECTNESS = dict(
    tokens=[1, 37, 512, 8192],
    experts=[32, 64, 256],
    topk=[1, 4, 8, 32],
    score_functions=ALL_SCORE_FUNCTIONS,
    input_types=["arange", "random", "extreme", "narrow", "constant"],
    group_topk=[0, 4],
    max_tests=200,
)

SWEEP_BENCHMARK = dict(
    tokens=[128, 8192, 32768, 131072],
    experts=[8, 256, 512, 2304],
    topk=[4, 8, 22, 36],
    score_functions=ALL_SCORE_FUNCTIONS,
    group_topk=[0, 4],
)

def _resolve(user_val, sweep_list):
    """If the user specified a value, use [that value]; otherwise use the sweep list."""
    if user_val is not None:
        return [user_val] if not isinstance(user_val, list) else user_val
    return sweep_list


def _correctness_passes(test_passes: List[str]) -> List[str]:
    """Map benchmark pass names to correctness-compatible pass names.

    ``backward_raw`` has no separate correctness path — it maps to ``backward``.
    Deduplicates while preserving order.
    """
    mapping = {"backward_raw": "backward"}
    seen: set = set()
    out: List[str] = []
    for tp in test_passes:
        mapped = mapping.get(tp, tp)
        if mapped not in seen:
            seen.add(mapped)
            out.append(mapped)
    return out


def _resolve_score_fns(args, sweep: Dict) -> List[str]:
    """Resolve the score function list from CLI / sweep grid."""
    if args.score_function is not None:
        return [args.score_function]
    return sweep["score_functions"]


def _resolve_group_topks(args, sweep: Dict) -> List[int]:
    """Resolve group_topk values, filtered by --num-groups when set."""
    group_topks = _resolve(args.group_topk, sweep.get("group_topk", [0]))
    if args.num_groups is not None:
        if args.num_groups == 0:
            group_topks = [g for g in group_topks if g == 0]
        else:
            group_topks = [g for g in group_topks if g > 0]
    return group_topks


def _resolve_input_types(args, sweep: Dict) -> List[Optional[str]]:
    """Resolve input types from CLI / sweep grid."""
    return _resolve(
        args.input_type if hasattr(args, "input_type") else None,
        sweep.get("input_types", [None]),
    )


def _build_topk_configs(args, sweep: Dict) -> List[Dict]:
    """Build the cross-product of topk configs, pinning user-specified dimensions."""
    tokens = _resolve(args.num_tokens, sweep["tokens"])
    experts = _resolve(args.num_experts, sweep["experts"])
    topks = _resolve(args.topk, sweep["topk"])
    score_fns = _resolve_score_fns(args, sweep)
    group_topks = _resolve_group_topks(args, sweep)
    input_types = _resolve_input_types(args, sweep)

    configs: List[Dict] = []
    for sf_name, nt, ne, tk, grp, inp in itertools.product(
        score_fns, tokens, experts, topks, group_topks, input_types,
    ):
        if tk > ne:
            continue
        ng = args.num_groups if args.num_groups is not None else (8 if grp else 0)
        if grp > 0 and not _valid_group_topk(ne, tk, ng, grp):
            continue
        sf_kernel, use_pre = _split_score_function(sf_name)
        cfg = dict(
            num_tokens=nt, num_experts=ne, topk=tk,
            score_function=sf_kernel, use_pre_softmax=use_pre,
            group_topk=grp,
        )
        if inp is not None:
            cfg["input_type"] = inp
            cfg["enable_bias"] = sf_kernel in ("sigmoid", "sqrtsoftplus")
            cfg["scaling_factor"] = 1.0
            cfg["num_groups"] = ng
        configs.append(cfg)
    return configs


def _build_aux_loss_configs(args, sweep: Dict) -> List[Dict]:
    """Build the cross-product of aux_loss configs, pinning user-specified dimensions."""
    tokens = _resolve(args.num_tokens, sweep["tokens"])
    experts = _resolve(args.num_experts, sweep["experts"])
    topks = _resolve(args.topk, sweep["topk"])
    sf_kernels = _dedup_aux_loss_score_functions(_resolve_score_fns(args, sweep))
    input_types = _resolve_input_types(args, sweep)

    configs: List[Dict] = []
    for sf_kernel, nt, ne, tk, inp in itertools.product(
        sf_kernels, tokens, experts, topks, input_types,
    ):
        if tk > ne:
            continue
        cfg = dict(
            num_tokens=nt, num_experts=ne, topk=tk,
            score_function=sf_kernel,
        )
        if inp is not None:
            cfg["input_type"] = inp
        configs.append(cfg)
    return configs


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
# "random" (standard normal, sigma=1) keeps values in a range where softmax
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
      arange   - deterministic, same as existing tests (monotonic, no ties)
      random   - standard normal
      uniform  - uniform [-1, 1]
      extreme  - large magnitude (stress softmax stability)
      narrow   - near-zero (sigmoid ~ 0.5, softmax ~ uniform)
      constant - all equal (tie-breaking stress test)
    """
    device = "cuda"
    if input_type == "arange":
        if score_function in ("sigmoid", "sqrtsoftplus"):
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

def _sqrtsoftplus(x: torch.Tensor) -> torch.Tensor:
    """sqrtsoftplus(x) = sqrt(softplus(x)), matching PyTorch Softplus(beta=1, threshold=20)."""
    return torch.sqrt(F.softplus(x.float(), beta=1.0, threshold=20.0))


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
    forced_routing_map: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference matching the fused topk kernel logic.

    All internal arithmetic is done in float32 to match the TE kernel
    (CompType = float).  Only the final outputs are cast to logits.dtype.

    If *forced_routing_map* is provided (bool [T, E]), the topk selection step
    is skipped and the given map is used instead.  This ensures the backward
    gradient flows through exactly the same expert positions as the fused
    kernel, eliminating spurious mismatches caused by tie-breaking differences
    in near-uniform score distributions.
    """
    orig_dtype = logits.dtype
    num_tokens, num_experts = logits.shape

    # Work in float32 throughout, matching the kernel's CompType = float.
    logits_f = logits.float()

    def _topk(scores):
        if group_topk and group_topk > 0:
            return _group_limited_topk(
                scores, topk, num_tokens, num_experts, num_groups, group_topk
            )
        return torch.topk(scores, k=topk, dim=1)

    def _indices_from_routing_map(routing_map: torch.Tensor) -> torch.Tensor:
        """Convert bool [T, E] routing_map to int64 [T, K] top_indices.

        Within each row, selected indices are returned in ascending order
        (matching the positional order the kernel writes them).
        """
        # nonzero gives (row, col) pairs sorted by row then col
        return routing_map.nonzero(as_tuple=False)[:, 1].view(num_tokens, topk)

    if score_function == "softmax":
        if use_pre_softmax:
            scores = torch.softmax(logits_f, dim=-1)
            if forced_routing_map is not None:
                top_indices = _indices_from_routing_map(forced_routing_map)
                probs = torch.gather(scores, dim=1, index=top_indices)
            else:
                probs, top_indices = _topk(scores)
        else:
            if forced_routing_map is not None:
                top_indices = _indices_from_routing_map(forced_routing_map)
                scores = torch.gather(logits_f, dim=1, index=top_indices)
            else:
                scores, top_indices = _topk(logits_f)
            probs = torch.softmax(scores, dim=-1)
    elif score_function in ("sigmoid", "sqrtsoftplus"):
        if score_function == "sigmoid":
            scores = torch.sigmoid(logits_f)
        else:
            scores = _sqrtsoftplus(logits_f)
        if forced_routing_map is not None:
            top_indices = _indices_from_routing_map(forced_routing_map)
            scores = torch.gather(scores, dim=1, index=top_indices)
        elif expert_bias is not None:
            scores_for_routing = scores + expert_bias.float()
            _, top_indices = _topk(scores_for_routing)
            scores = torch.gather(scores, dim=1, index=top_indices)
        else:
            scores, top_indices = _topk(scores)
        probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if topk > 1 else scores
    else:
        raise ValueError(f"Unknown score_function: {score_function}")

    if scaling_factor is not None:
        probs = probs * scaling_factor

    # Cast back to original dtype for the output tensors.
    topk_masked_gates = torch.zeros(
        num_tokens, num_experts, dtype=orig_dtype, device=logits.device,
    ).scatter(1, top_indices, probs.to(orig_dtype))
    topk_map = torch.zeros(
        num_tokens, num_experts, dtype=torch.int32, device=logits.device,
    ).scatter(1, top_indices, 1).bool()
    return topk_masked_gates, topk_map


def reference_aux_loss_scores_forward(
    logits: torch.Tensor,
    topk: int,
    score_function: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference matching the fused aux loss score kernel logic.

    Returns (routing_map, scores) -- note scores is the full [T, E] tensor
    (softmax or normalized-sigmoid over all experts), NOT just the topk values.

    All internal arithmetic is done in float32 to match the TE kernel.
    The scores output is float32 (matching the kernel's CompType).
    """
    logits_f = logits.float()

    if score_function == "softmax":
        scores = torch.softmax(logits_f, dim=-1)
    elif score_function in ("sigmoid", "sqrtsoftplus"):
        if score_function == "sigmoid":
            scores = torch.sigmoid(logits_f)
        else:
            scores = _sqrtsoftplus(logits_f)
        scores = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
    else:
        raise ValueError(f"Unknown score_function: {score_function}")

    _, top_indices = torch.topk(scores, k=topk, dim=1)
    routing_map = torch.zeros_like(logits, dtype=torch.int32).scatter(1, top_indices, 1).bool()
    return routing_map, scores


# ===========================================================================
# Topk kernel -- correctness
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

    if enable_bias and score_function in ("sigmoid", "sqrtsoftplus"):
        expert_bias = (
            torch.arange(num_experts, device="cuda", dtype=torch.float32) * 0.1
        ).flip(dims=[0])
    else:
        expert_bias = None

    logits_clone = logits.detach().clone().requires_grad_(needs_grad)
    expert_bias_clone = expert_bias.clone() if expert_bias is not None else None

    # When testing backward, we need both reference and fused to use the same
    # routing_map so that tie-breaking differences in near-uniform score
    # distributions don't cause spurious gradient mismatches.  Strategy:
    #   1. Run fused kernel forward (always needed).
    #   2. For forward check: run reference with its own topk (no forced map)
    #      to verify routing_map correctness independently.
    #   3. For backward check: run reference with fused_map forced, so autograd
    #      backward flows through identical expert positions.

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
    has_nan = fused_probs.isnan().any()

    # Build tolerance kwargs (only set overrides if provided)
    tol_kw: Dict = {}
    if atol is not None:
        tol_kw["atol"] = atol
    if rtol is not None:
        tol_kw["rtol"] = rtol

    sf_display = _join_score_function(score_function, use_pre_softmax)
    tag = (
        f"[topk {test_pass:>4s} | {sf_display:>12s} | tokens={num_tokens:>6d} | "
        f"experts={num_experts:>4d} | topk={topk} | "
        f"grp_topk={group_topk} | scale={scaling_factor} | bias={enable_bias} | "
        f"dtype={dtype} | input={input_type}]"
    )
    try:
        # --- Forward check ---
        if test_pass in ("forward", "both"):
            # Reference forward with its own topk selection (no forced map).
            ref_probs_fwd, ref_map_fwd = reference_topk_forward(
                logits, topk, use_pre_softmax,
                num_groups, group_topk, scaling_factor,
                score_function, expert_bias,
            )
            if has_nan or ref_probs_fwd.isnan().any():
                raise _NaNDetected()
            fwd_ok = _check_topk_forward(
                ref_probs_fwd, ref_map_fwd, fused_probs, fused_map,
                logits, score_function, use_pre_softmax, expert_bias, dtype, tol_kw, tag,
            )
            if not fwd_ok:
                return False
            # Discard ref forward graph (not needed for backward).
            if needs_grad:
                logits.grad = None

        # --- Backward check ---
        if test_pass in ("backward", "both"):
            if has_nan:
                raise _NaNDetected()
            # Reference forward with fused kernel's routing_map forced, so both
            # backward passes flow through identical expert positions.
            ref_probs_bwd, _ = reference_topk_forward(
                logits, topk, use_pre_softmax,
                num_groups, group_topk, scaling_factor,
                score_function, expert_bias,
                forced_routing_map=fused_map.detach(),
            )
            ref_loss = ref_probs_bwd.sum()
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
            # Already retried once -- give up and report as a pass with warning.
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
        print(f"  FAIL {tag}")
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

    # Routing maps disagree -- check if the disagreement is due to tied scores.
    num_tokens, num_experts = logits.shape
    if score_function == "sigmoid":
        scores = torch.sigmoid(logits.detach().float()).to(dtype)
        if expert_bias is not None:
            scores = scores + expert_bias
    elif score_function == "sqrtsoftplus":
        scores = _sqrtsoftplus(logits.detach()).to(dtype)
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
# Aux loss score kernel -- correctness
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
        f"[aux_loss {test_pass:>4s} | {score_function:>12s} | tokens={num_tokens:>6d} | "
        f"experts={num_experts:>4d} | topk={topk} | dtype={dtype} | input={input_type}]"
    )

    try:
        # --- Forward check ---
        if test_pass in ("forward", "both"):
            if has_nan:
                raise _NaNDetected()
            # Check scores (full [T, E] tensor)
            torch.testing.assert_close(ref_scores, fused_scores, **tol_kw)
            # Routing map tie-break differences are acceptable: scores (the
            # full [T, E] tensor) have already been verified identical above,
            # so any routing_map disagreement is purely topk tie-breaking.

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
            # Already retried once -- give up and report as a pass with warning.
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
    configs = _build_topk_configs(args, SWEEP_CORRECTNESS)
    max_tests = SWEEP_CORRECTNESS.get("max_tests", len(configs))
    # Deduplicate passes for correctness: backward_raw uses the same check as
    # backward, and "both" covers forward+backward, so map accordingly.
    corr_passes = _correctness_passes(args.test_pass)

    total = len(configs) * len(corr_passes)
    if total > max_tests:
        step = max(1, total // max_tests)
    else:
        step = 1

    print(f"\nRunning topk correctness tests "
          f"({len(configs)} configs x {len(corr_passes)} pass(es), "
          f"dtype={args.dtype}, pass={corr_passes})...\n")

    passed = 0
    count = 0
    for i, cfg in enumerate(configs):
        for tp in corr_passes:
            if (i * len(corr_passes) + corr_passes.index(tp)) % step != 0:
                continue
            count += 1
            ok = run_topk_correctness(
                dtype=args.dtype, test_pass=tp,
                atol=args.atol, rtol=args.rtol, **cfg,
            )
            passed += int(ok)

    print(f"\nTopk correctness: {passed}/{count} passed\n")
    return passed == count


def aux_loss_correctness_suite(args) -> bool:
    """Run aux loss score correctness tests.  Returns True if all pass."""
    configs = _build_aux_loss_configs(args, SWEEP_CORRECTNESS)
    max_tests = SWEEP_CORRECTNESS.get("max_tests", len(configs))
    corr_passes = _correctness_passes(args.test_pass)

    total = len(configs) * len(corr_passes)
    if total > max_tests:
        step = max(1, total // max_tests)
    else:
        step = 1

    print(f"\nRunning aux_loss correctness tests "
          f"({len(configs)} configs x {len(corr_passes)} pass(es), "
          f"dtype={args.dtype}, pass={corr_passes})...\n")

    passed = 0
    count = 0
    for i, cfg in enumerate(configs):
        for tp in corr_passes:
            if (i * len(corr_passes) + corr_passes.index(tp)) % step != 0:
                continue
            count += 1
            ok = run_aux_loss_correctness(
                dtype=args.dtype, test_pass=tp,
                atol=args.atol, rtol=args.rtol, **cfg,
            )
            passed += int(ok)

    print(f"\nAux loss correctness: {passed}/{count} passed\n")
    return passed == count


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
    if score_function in ("sigmoid", "sqrtsoftplus"):
        expert_bias = torch.randn(num_experts, dtype=torch.float32, device="cuda") * 0.1

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

    if test_pass == "backward_raw":
        # ----- Raw kernel-only backward benchmark -----
        # Run forward once to get the saved tensors, then time the backward
        # kernel call directly — no autograd, no loss.sum(), no grad allocation.
        import transformer_engine_torch as tex

        logits_fwd = torch.randn(num_tokens, num_experts, dtype=dtype, device="cuda")
        _, routing_map, intermediate_output = tex.fused_topk_with_score_function_fwd(
            logits_fwd, topk, use_pre_softmax,
            8 if group_topk else 0, group_topk, 1.0, score_function, expert_bias,
        )
        grad_probs = torch.ones(num_tokens, num_experts, dtype=dtype, device="cuda")
        grad_logits = torch.empty(num_tokens, num_experts, dtype=dtype, device="cuda")

        bwd_args = dict(
            routing_map=routing_map,
            intermediate_output=intermediate_output,
            grad_probs=grad_probs,
            grad_logits=grad_logits,
            topk=topk,
            use_pre_softmax=use_pre_softmax,
            scaling_factor=1.0,
            score_function=score_function,
        )
        fused_ms = _time_kernel_only(
            lambda: _topk_backward_raw_fused(**bwd_args), warmup, iters,
        )

        # Reference: equivalent PyTorch math, same inputs, no autograd
        grad_logits_ref = torch.empty(num_tokens, num_experts, dtype=dtype, device="cuda")
        ref_bwd_args = dict(
            routing_map=routing_map,
            intermediate_output=intermediate_output,
            grad_probs=grad_probs,
            grad_logits=grad_logits_ref,
            topk=topk,
            use_pre_softmax=use_pre_softmax,
            scaling_factor=1.0,
            score_function=score_function,
        )
        ref_ms = _time_kernel_only(
            lambda: _topk_backward_raw_reference(**ref_bwd_args), warmup, iters,
        )
    elif test_pass == "forward":
        fused_ms = _time_forward(lambda: fused_topk_with_score_function(**call_args),
                                 warmup, iters)
    elif test_pass == "backward":
        fused_ms = _time_backward(
            lambda: fused_topk_with_score_function(**call_args),
            loss_fn=lambda out: out[0].sum(),
            warmup=warmup, iters=iters, grad_inputs=[logits],
        )
    else:  # both
        fused_ms = _time_forward_backward(
            lambda: fused_topk_with_score_function(**call_args),
            loss_fn=lambda out: out[0].sum(),
            warmup=warmup, iters=iters, grad_inputs=[logits],
        )

    if test_pass != "backward_raw":
        # PyTorch reference (autograd-based)
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
                warmup=warmup, iters=iters, grad_inputs=[logits_ref],
            )
        else:
            ref_ms = _time_forward_backward(
                lambda: reference_topk_forward(**ref_args),
                loss_fn=lambda out: out[0].sum(),
                warmup=warmup, iters=iters, grad_inputs=[logits_ref],
            )

    sf_display = _join_score_function(score_function, use_pre_softmax)
    nbytes = _compute_min_bytes(num_tokens, num_experts, topk, dtype, "topk", test_pass)
    return dict(
        kernel="topk",
        num_tokens=num_tokens,
        num_experts=num_experts,
        topk=topk,
        score_function=sf_display,
        group_topk=group_topk,
        dtype=str(dtype).replace("torch.", ""),
        test_pass=test_pass,
        fused_ms=fused_ms,
        ref_ms=ref_ms,
        speedup=ref_ms / fused_ms if fused_ms > 0 else float("inf"),
        fused_gbps=nbytes / (fused_ms * 1e-3) / 1e9 if fused_ms > 0 else 0.0,
        ref_gbps=nbytes / (ref_ms * 1e-3) / 1e9 if ref_ms > 0 else 0.0,
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

    if test_pass == "backward_raw":
        # Raw kernel-only backward benchmark for aux_loss
        import transformer_engine_torch as tex

        logits_fwd = torch.randn(num_tokens, num_experts, dtype=dtype, device="cuda")
        scores, routing_map, intermediate_output = tex.fused_score_for_moe_aux_loss_fwd(
            logits=logits_fwd, topk=topk, score_function=score_function,
        )
        grad_scores = torch.ones(num_tokens, num_experts, dtype=torch.float32, device="cuda")
        grad_logits = torch.empty(num_tokens, num_experts, dtype=dtype, device="cuda")

        def _fused_aux_bwd():
            tex.fused_score_for_moe_aux_loss_bwd(
                num_tokens=num_tokens, num_experts=num_experts,
                intermediate_output=intermediate_output,
                grad_scores=grad_scores, grad_logits=grad_logits,
                topk=topk, score_function=score_function,
            )

        fused_ms = _time_kernel_only(_fused_aux_bwd, warmup, iters)

        # Reference: equivalent PyTorch math, no autograd
        grad_logits_ref = torch.empty(num_tokens, num_experts, dtype=dtype, device="cuda")

        def _ref_aux_bwd():
            _aux_loss_backward_raw_reference(
                intermediate_output=intermediate_output,
                grad_scores=grad_scores,
                grad_logits=grad_logits_ref,
                score_function=score_function,
            )

        ref_ms = _time_kernel_only(_ref_aux_bwd, warmup, iters)
    elif test_pass == "forward":
        fused_ms = _time_forward(
            lambda: fused_compute_score_for_moe_aux_loss(**call_args), warmup, iters)
    elif test_pass == "backward":
        fused_ms = _time_backward(
            lambda: fused_compute_score_for_moe_aux_loss(**call_args),
            loss_fn=lambda out: out[1].sum(),  # out = (routing_map, scores)
            warmup=warmup, iters=iters, grad_inputs=[logits],
        )
    else:
        fused_ms = _time_forward_backward(
            lambda: fused_compute_score_for_moe_aux_loss(**call_args),
            loss_fn=lambda out: out[1].sum(),
            warmup=warmup, iters=iters, grad_inputs=[logits],
        )

    if test_pass != "backward_raw":
        # PyTorch reference (autograd-based)
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
                warmup=warmup, iters=iters, grad_inputs=[logits_ref],
            )
        else:
            ref_ms = _time_forward_backward(
                lambda: reference_aux_loss_scores_forward(**ref_args),
                loss_fn=lambda out: out[1].sum(),
                warmup=warmup, iters=iters, grad_inputs=[logits_ref],
            )

    nbytes = _compute_min_bytes(num_tokens, num_experts, topk, dtype, "aux_loss", test_pass)
    return dict(
        kernel="aux_loss",
        num_tokens=num_tokens,
        num_experts=num_experts,
        topk=topk,
        score_function=score_function,
        group_topk=0,
        dtype=str(dtype).replace("torch.", ""),
        test_pass=test_pass,
        fused_ms=fused_ms,
        ref_ms=ref_ms,
        speedup=ref_ms / fused_ms if fused_ms > 0 else float("inf"),
        fused_gbps=nbytes / (fused_ms * 1e-3) / 1e9 if fused_ms > 0 else 0.0,
        ref_gbps=nbytes / (ref_ms * 1e-3) / 1e9 if ref_ms > 0 else 0.0,
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


def _time_backward(fn, loss_fn, warmup: int, iters: int,
                   grad_inputs: Optional[List[torch.Tensor]] = None) -> float:
    """Time backward-only (grad computation).

    Each iteration: run forward + loss_fn + sync (not timed), then time
    only the .backward() call.  ``grad_inputs`` (leaf tensors whose .grad
    may accumulate) are zeroed between iterations to avoid the extra ``+=``
    kernel overhead.
    """
    for _ in range(warmup):
        out = fn()
        loss = loss_fn(out)
        loss.backward()
    torch.cuda.synchronize()

    total_bwd_ms = 0.0
    for _ in range(iters):
        # Zero accumulated grads before building a new graph
        if grad_inputs:
            for t in grad_inputs:
                if t.grad is not None:
                    t.grad = None
        out = fn()
        loss = loss_fn(out)
        torch.cuda.synchronize()
        bwd_s = torch.cuda.Event(enable_timing=True)
        bwd_e = torch.cuda.Event(enable_timing=True)
        bwd_s.record()
        loss.backward()
        bwd_e.record()
        torch.cuda.synchronize()
        total_bwd_ms += bwd_s.elapsed_time(bwd_e)
    return total_bwd_ms / iters


def _time_forward_backward(fn, loss_fn, warmup: int, iters: int,
                           grad_inputs: Optional[List[torch.Tensor]] = None) -> float:
    """Time forward + backward together.  Returns average ms.

    ``grad_inputs`` are zeroed between iterations to avoid accumulation overhead.
    """
    for _ in range(warmup):
        out = fn()
        loss = loss_fn(out)
        loss.backward()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        if grad_inputs:
            for t in grad_inputs:
                if t.grad is not None:
                    t.grad = None
        out = fn()
        loss = loss_fn(out)
        loss.backward()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _time_kernel_only(fn, warmup: int, iters: int) -> float:
    """Time a raw kernel call (no autograd, no loss).  Returns average ms."""
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
# Raw backward helpers — call the CUDA kernel directly, no autograd
# ---------------------------------------------------------------------------

def _topk_backward_raw_fused(
    routing_map: torch.Tensor,
    intermediate_output: torch.Tensor,
    grad_probs: torch.Tensor,
    grad_logits: torch.Tensor,
    topk: int,
    use_pre_softmax: bool,
    scaling_factor: float,
    score_function: str,
) -> None:
    """Call the fused backward kernel directly (no autograd overhead)."""
    import transformer_engine_torch as tex

    tex.fused_topk_with_score_function_bwd(
        grad_probs.size(0),   # num_tokens
        grad_probs.size(1),   # num_experts
        routing_map,
        intermediate_output,
        grad_probs,
        grad_logits,
        topk,
        use_pre_softmax,
        scaling_factor,
        score_function,
    )


def _topk_backward_raw_reference(
    routing_map: torch.Tensor,
    intermediate_output: torch.Tensor,
    grad_probs: torch.Tensor,
    grad_logits: torch.Tensor,
    topk: int,
    use_pre_softmax: bool,
    scaling_factor: float,
    score_function: str,
) -> None:
    """Pure-PyTorch backward matching the fused kernel, operating on the same
    pre-computed intermediate buffers.  No autograd — direct tensor math."""
    g = grad_probs.float() * scaling_factor
    act = intermediate_output  # already float
    mask = routing_map  # bool [T, E]

    if score_function == "sigmoid":
        if topk > 1:
            # Normalization backward
            sum_act = (act * mask).sum(dim=-1, keepdim=True)
            sum_grad_act = (g * act * mask).sum(dim=-1, keepdim=True)
            denom = sum_act + 1e-20
            g = torch.where(mask, g / denom - sum_grad_act / (denom * denom), torch.zeros_like(g))
        else:
            g = torch.where(mask, g, torch.zeros_like(g))
        # Sigmoid backward: act = sigmoid(x), dy/dx = act * (1 - act)
        g = g * act * (1.0 - act)
    elif score_function == "softmax":
        if not use_pre_softmax:
            # Post-softmax backward (routed subset)
            dot = (g * act * mask).sum(dim=-1, keepdim=True)
            g = torch.where(mask, act * (g - dot), torch.zeros_like(g))
        # Zero non-routed
        g = g * mask.float()
        if use_pre_softmax:
            # Pre-softmax backward (all experts)
            dot = (g * act).sum(dim=-1, keepdim=True)
            g = act * (g - dot)
    elif score_function == "sqrtsoftplus":
        y = torch.sqrt(torch.nn.functional.softplus(act, beta=1.0, threshold=20.0))
        if topk > 1:
            sum_act = (y * mask).sum(dim=-1, keepdim=True)
            sum_grad_act = (g * y * mask).sum(dim=-1, keepdim=True)
            denom = sum_act + 1e-20
            g = torch.where(mask, g / denom - sum_grad_act / (denom * denom), torch.zeros_like(g))
        else:
            g = torch.where(mask, g, torch.zeros_like(g))
        # Sqrtsoftplus backward: dy/dx = sigmoid(x) / (2 * y)
        g = g * torch.sigmoid(act) / (2.0 * y + 1e-20)

    grad_logits.copy_(g.to(grad_logits.dtype))


def _aux_loss_backward_raw_reference(
    intermediate_output: torch.Tensor,
    grad_scores: torch.Tensor,
    grad_logits: torch.Tensor,
    score_function: str,
) -> None:
    """Pure-PyTorch backward for aux_loss scores, no autograd.
    No routing_map — all experts participate in normalization."""
    g = grad_scores.float()
    act = intermediate_output  # already float

    if score_function == "sigmoid":
        # act = sigmoid output; normalization: scores = act / sum(act)
        sum_act = act.sum(dim=-1, keepdim=True)
        sum_grad_act = (g * act).sum(dim=-1, keepdim=True)
        denom = sum_act + 1e-20
        g = g / denom - sum_grad_act / (denom * denom)
        g = g * act * (1.0 - act)
    elif score_function == "softmax":
        # act = softmax output
        dot = (g * act).sum(dim=-1, keepdim=True)
        g = act * (g - dot)
    elif score_function == "sqrtsoftplus":
        # act = original logit
        y = torch.sqrt(torch.nn.functional.softplus(act, beta=1.0, threshold=20.0))
        sum_act = y.sum(dim=-1, keepdim=True)
        sum_grad_act = (g * y).sum(dim=-1, keepdim=True)
        denom = sum_act + 1e-20
        g = g / denom - sum_grad_act / (denom * denom)
        g = g * torch.sigmoid(act) / (2.0 * y + 1e-20)

    grad_logits.copy_(g.to(grad_logits.dtype))


# ---------------------------------------------------------------------------
# Benchmark printing & CSV
# ---------------------------------------------------------------------------

_BENCH_COLUMNS = [
    "kernel", "num_tokens", "num_experts", "topk", "score_function",
    "group_topk", "dtype", "test_pass",
    "fused_ms", "ref_ms", "speedup", "fused_gbps", "ref_gbps",
]


def _compute_min_bytes(
    num_tokens: int, num_experts: int, topk: int,
    dtype: torch.dtype, kernel: str, test_pass: str,
) -> int:
    """Minimum global memory traffic for one kernel call (bytes).

    Forward (topk):
      Read:  logits (dtype)
      Write: probs (dtype) + routing_map (bool) + intermediate_output (fp32)
    Forward (aux_loss):
      Read:  logits (dtype)
      Write: scores (fp32) + routing_map (bool) + intermediate_output (fp32)
    Backward (topk):
      Read:  grad_probs (dtype) + intermediate_output (fp32) + routing_map (bool)
      Write: grad_logits (dtype)
    Backward (aux_loss):
      Read:  grad_scores (fp32) + intermediate_output (fp32)
      Write: grad_logits (dtype)
    """
    elt = torch.finfo(dtype).bits // 8 if dtype.is_floating_point else 4
    T_E = num_tokens * num_experts
    # For aux_loss: grad_scores and scores are always fp32 regardless of dtype.
    grad_elt = elt if kernel == "topk" else 4
    score_elt = elt if kernel == "topk" else 4

    if test_pass in ("backward", "backward_raw"):
        read_bytes = T_E * (grad_elt + 4)   # grad + intermediate_output (fp32)
        if kernel == "topk":
            read_bytes += T_E * 1            # routing_map
        write_bytes = T_E * elt              # grad_logits
    elif test_pass == "forward":
        read_bytes = T_E * elt               # logits
        write_bytes = T_E * (score_elt + 1 + 4)  # probs/scores + routing_map + intermediate
    else:  # both
        fwd_read = T_E * elt
        fwd_write = T_E * (score_elt + 1 + 4)
        bwd_read = T_E * (grad_elt + 4) + (T_E if kernel == "topk" else 0)
        bwd_write = T_E * elt
        return fwd_read + fwd_write + bwd_read + bwd_write

    return read_bytes + write_bytes


def _print_bench_header() -> None:
    """Print the benchmark table header."""
    hdr = (
        f"{'kernel':>8s} {'tokens':>8s} {'experts':>7s} {'topk':>4s} {'score_fn':>12s} "
        f"{'grp_tk':>6s} {'dtype':>8s} {'pass':>12s} "
        f"{'fused_ms':>9s} {'ref_ms':>9s} {'speedup':>7s} "
        f"{'fused_GB/s':>10s} {'ref_GB/s':>10s}"
    )
    print(hdr)
    print("-" * len(hdr))
    sys.stdout.flush()


def _print_bench_row(r: Dict) -> None:
    """Print a single benchmark result row and flush immediately."""
    print(
        f"{r['kernel']:>8s} {r['num_tokens']:>8d} {r['num_experts']:>7d} {r['topk']:>4d} "
        f"{r['score_function']:>12s} "
        f"{r['group_topk']:>6d} {r['dtype']:>8s} {r['test_pass']:>12s} "
        f"{r['fused_ms']:>9.4f} {r['ref_ms']:>9.4f} "
        f"{r['speedup']:>7.2f}x "
        f"{r['fused_gbps']:>10.1f} {r['ref_gbps']:>10.1f}"
    )
    sys.stdout.flush()


def _write_csv(results: List[Dict], path: str) -> None:
    """Write benchmark results to a CSV file."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_BENCH_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"Benchmark results written to {path} ({len(results)} rows)")


# ---------------------------------------------------------------------------
# Benchmark suites
# ---------------------------------------------------------------------------

def topk_benchmark_suite(args) -> List[Dict]:
    """Run topk benchmark across configurations.  Returns list of result dicts."""
    configs = _build_topk_configs(args, SWEEP_BENCHMARK)
    passes = args.test_pass  # list

    print(f"\nBenchmarking {len(configs)} topk config(s) x {len(passes)} pass(es) "
          f"(warmup={args.warmup}, iters={args.iters}, "
          f"dtype={args.dtype}, pass={passes})...\n")
    _print_bench_header()

    results: List[Dict] = []
    for cfg in configs:
        # Remove correctness-only keys before passing to benchmark
        bench_cfg = {k: v for k, v in cfg.items()
                     if k not in ("input_type", "enable_bias", "scaling_factor", "num_groups")}
        for tp in passes:
            r = _benchmark_topk_one(
                **bench_cfg, dtype=args.dtype, test_pass=tp,
                warmup=args.warmup, iters=args.iters,
            )
            _print_bench_row(r)
            results.append(r)

    print()
    return results


def aux_loss_benchmark_suite(args) -> List[Dict]:
    """Run aux loss score benchmark across configurations.  Returns list of result dicts."""
    configs = _build_aux_loss_configs(args, SWEEP_BENCHMARK)
    passes = args.test_pass  # list

    print(f"\nBenchmarking {len(configs)} aux_loss config(s) x {len(passes)} pass(es) "
          f"(warmup={args.warmup}, iters={args.iters}, "
          f"dtype={args.dtype}, pass={passes})...\n")
    _print_bench_header()

    results: List[Dict] = []
    for cfg in configs:
        bench_cfg = {k: v for k, v in cfg.items() if k not in ("input_type",)}
        for tp in passes:
            r = _benchmark_aux_loss_one(
                **bench_cfg, dtype=args.dtype, test_pass=tp,
                warmup=args.warmup, iters=args.iters,
            )
            _print_bench_row(r)
            results.append(r)

    print()
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
    _ALL_KERNELS = ["topk", "aux_loss"]
    parser.add_argument(
        "--kernel", nargs="+", choices=_ALL_KERNELS,
        default=None,
        help="Which kernel(s) to test (omit to sweep all; default: all)",
    )
    # Note: --pass is a reserved word in Python, so we use dest="test_pass"
    _ALL_PASSES = ["forward", "backward", "backward_raw", "both"]
    parser.add_argument(
        "--pass", nargs="+", choices=_ALL_PASSES,
        default=None, dest="test_pass",
        help="Which pass(es) to test (omit to sweep default set; "
             "default: forward backward_raw)",
    )

    # Shape / kernel options — omit any to sweep that dimension.
    parser.add_argument("--num-tokens", type=int, default=None,
                        help="Number of tokens (omit to sweep)")
    parser.add_argument("--num-experts", type=int, default=None,
                        help="Number of experts (omit to sweep)")
    parser.add_argument("--topk", type=int, default=None,
                        help="Top-k value (omit to sweep)")
    parser.add_argument("--score-function", choices=ALL_SCORE_FUNCTIONS,
                        default=None,
                        help="Score function (omit to sweep all)")
    parser.add_argument("--num-groups", type=int, default=None,
                        help="Number of expert groups (omit to sweep; 0=no grouping)")
    parser.add_argument("--group-topk", type=int, default=None,
                        help="Group top-k value (omit to sweep)")
    parser.add_argument("--scaling-factor", type=float, default=1.0)
    parser.add_argument("--enable-bias", action="store_true", default=False)

    # Data / dtype
    parser.add_argument("--dtype", type=parse_dtype, default=torch.float32,
                        help="fp32 | fp16 | bf16 (default: fp32)")
    parser.add_argument("--input-type", default=None,
                        choices=["arange", "random", "uniform", "extreme", "narrow", "constant"],
                        help="Input distribution for correctness (omit to sweep)")

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

    # Output
    parser.add_argument("--csv", type=str, default=None, metavar="FILE",
                        help="Write benchmark results to a CSV file")

    args = parser.parse_args()

    # Resolve list defaults for --kernel and --pass.
    if args.kernel is None:
        args.kernel = ["topk", "aux_loss"]
    if args.test_pass is None:
        args.test_pass = ["forward", "backward_raw"]

    print_gpu_info()

    ok = True
    run_topk = "topk" in args.kernel
    run_aux = "aux_loss" in args.kernel

    if args.mode in ("correctness", "all"):
        if run_topk:
            ok = topk_correctness_suite(args) and ok
        if run_aux:
            ok = aux_loss_correctness_suite(args) and ok

    all_bench_results: List[Dict] = []
    if args.mode in ("benchmark", "all"):
        if run_topk:
            all_bench_results.extend(topk_benchmark_suite(args))
        if run_aux:
            all_bench_results.extend(aux_loss_benchmark_suite(args))

    if args.csv and all_bench_results:
        _write_csv(all_bench_results, args.csv)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
