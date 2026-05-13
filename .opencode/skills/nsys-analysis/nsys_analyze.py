#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Analyze nsys SQLite exports for GPU profiling — single-rank and multi-rank.

Reads .sqlite files produced by `nsys export --type sqlite` and generates
summary statistics for CUDA kernels, memory operations, NCCL collectives,
and MoE-specific operations.

Iteration detection works directly from the kernel timeline (no NVTX needed),
which is critical for CUDA-graph-captured workloads where NVTX markers are not
replayed.  The algorithm bins kernel GPU-time into fixed time windows, identifies
contiguous "dense compute" phases, and groups them into training iterations
(forward+backward) versus optimizer/gradient-sync phases.

Usage:
    # Analyze a single rank
    python nsys_analyze.py /path/to/rank0.sqlite

    # Analyze all ranks in a directory
    python nsys_analyze.py /path/to/nsys/dir

    # Restrict to specific iteration (1-indexed)
    python nsys_analyze.py /path/to/rank0.sqlite --iteration 2

    # Restrict to an explicit time window (ms)
    python nsys_analyze.py /path/to/rank0.sqlite --window 2355,2767

    # Output as JSON
    python nsys_analyze.py /path/to/dir --format json --top 10

    # Save report to file
    python nsys_analyze.py /path/to/dir --output report.txt
"""

import argparse
import json
import re
import sqlite3
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Kernel name classification
# ---------------------------------------------------------------------------

MOE_KERNEL_PATTERNS = [
    (r"combine_kernel", "MoE combine"),
    (r"dispatch_kernel", "MoE dispatch"),
    (r"permute_preprocessing_kernel", "MoE permute_preprocess"),
    (r"permute_kernel", "MoE permute"),
    (r"unpermute_kernel", "MoE unpermute"),
    (r"scan\b", "MoE scan"),
    (r"device_sync_kernel", "MoE device_sync"),
    (r"_paged_stash_pop_kernel", "MoE paged_stash_pop"),
    (r"_paged_stash_copy_kernel", "MoE paged_stash_copy"),
    (r"fused_moe_aux_loss", "MoE aux_loss"),
    (r"fused_score_for_moe_aux_loss", "MoE score_aux_loss"),
    (r"fused_topk_with_score_function", "MoE topk"),
    (r"grouped_gemm.*dglu.*dbiasBlockScaled", "MoE grouped_gemm_dglu_dbias"),
    (r"grouped_gemm.*quantBlockScaled", "MoE grouped_gemm_quant"),
    (r"grouped_gemm.*wgradBlockScaled", "MoE grouped_gemm_wgrad"),
    (r"grouped_gemm.*glu.*biasBlockScaled", "MoE grouped_gemm_glu_bias"),
]

NCCL_PATTERN = re.compile(r"^ncclDevKernel_(\w+)")
ATTENTION_PATTERN = re.compile(r"cudnn_generated.*sdpa.*sm\d+.*flash")
GEMM_PATTERN = re.compile(r"cutlass3x_sm|nvjet_sm")
NORM_PATTERN = re.compile(r"ln_tma_|layernorm|rmsnorm", re.IGNORECASE)
QUANT_PATTERN = re.compile(r"quantize_mxfp8|group_quantize_mxfp8")


def classify_kernel(short_name: str, demangled_name: str) -> str:
    """Classify a kernel into a high-level category."""
    for pattern, label in MOE_KERNEL_PATTERNS:
        if re.search(pattern, short_name) or re.search(pattern, demangled_name):
            return label
    if NCCL_PATTERN.match(short_name):
        m = NCCL_PATTERN.match(short_name)
        return f"NCCL {m.group(1)}"
    if ATTENTION_PATTERN.search(demangled_name) or ATTENTION_PATTERN.search(short_name):
        return "Attention (cuDNN Flash)"
    if GEMM_PATTERN.search(short_name) or GEMM_PATTERN.search(demangled_name):
        return "GEMM (CUTLASS/nvjet)"
    if NORM_PATTERN.search(short_name):
        return "LayerNorm/RMSNorm"
    if QUANT_PATTERN.search(short_name):
        return "Quantization (MXFP8)"
    if "elementwise" in short_name.lower():
        return "Elementwise"
    if "reduce_kernel" in short_name:
        return "Reduction"
    if "Optimizer" in short_name or "adam" in short_name.lower() or "multi_tensor" in short_name:
        return "Optimizer"
    return "Other"


# ---------------------------------------------------------------------------
# Iteration detection
# ---------------------------------------------------------------------------

BIN_WIDTH_NS = 10_000_000  # 10 ms bins
DENSE_THRESHOLD_NS = 8_000_000  # >=8ms of GPU time in a 10ms bin = "dense"
MIN_COMPUTE_PHASE_MS = 200  # discard phases shorter than this


@dataclass
class TimePhase:
    """A contiguous time phase (dense compute or sparse gap)."""

    start_ms: float
    end_ms: float
    gpu_time_ms: float
    is_dense: bool
    label: str = ""  # "compute", "optimizer", "gradient_sync", "warmup", etc.

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms


@dataclass
class Iteration:
    """One training iteration: compute phase + optional optimizer phase."""

    index: int  # 1-indexed
    compute: TimePhase
    optimizer: Optional[TimePhase]  # None for the last iteration if profile ends early

    @property
    def start_ms(self) -> float:
        return self.compute.start_ms

    @property
    def end_ms(self) -> float:
        if self.optimizer:
            return self.optimizer.end_ms
        return self.compute.end_ms

    @property
    def compute_duration_ms(self) -> float:
        return self.compute.duration_ms


def detect_phases(conn: sqlite3.Connection) -> list[TimePhase]:
    """Detect dense compute phases from kernel timestamps.

    Bins kernel GPU time into BIN_WIDTH_NS windows and identifies contiguous
    runs of dense (high GPU utilization) bins.
    """
    query = f"""
        WITH bins AS (
            SELECT CAST(start / {BIN_WIDTH_NS} AS INT) AS bin_idx,
                   SUM(end - start) AS total_ns
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            GROUP BY CAST(start / {BIN_WIDTH_NS} AS INT)
        ),
        classified AS (
            SELECT bin_idx, total_ns,
                   CASE WHEN total_ns >= {DENSE_THRESHOLD_NS} THEN 1 ELSE 0 END AS is_dense
            FROM bins
        ),
        groups AS (
            SELECT bin_idx, total_ns, is_dense,
                   bin_idx - ROW_NUMBER() OVER (
                       PARTITION BY is_dense ORDER BY bin_idx
                   ) AS grp
            FROM classified
        )
        SELECT is_dense,
               MIN(bin_idx) * {BIN_WIDTH_NS // 1_000_000} AS start_ms,
               (MAX(bin_idx) + 1) * {BIN_WIDTH_NS // 1_000_000} AS end_ms,
               SUM(total_ns) AS gpu_time_ns
        FROM groups
        GROUP BY is_dense, grp
        ORDER BY start_ms
    """

    phases = []
    for row in conn.execute(query):
        phases.append(
            TimePhase(
                start_ms=row[1],
                end_ms=row[2],
                gpu_time_ms=row[3] / 1e6,
                is_dense=bool(row[0]),
            )
        )
    return phases


def detect_iterations(conn: sqlite3.Connection) -> list[Iteration]:
    """Detect training iterations from dense compute phases.

    The algorithm:
    1. Find all contiguous dense-compute phases (high GPU util per 10ms bin)
    2. Filter to phases longer than MIN_COMPUTE_PHASE_MS
    3. Pair each compute phase with the following sparse (optimizer/sync) phase
    """
    phases = detect_phases(conn)
    if not phases:
        return []

    # Filter to significant dense phases
    compute_phases = [
        p for p in phases if p.is_dense and p.duration_ms >= MIN_COMPUTE_PHASE_MS
    ]

    if not compute_phases:
        return []

    # Pair each compute phase with the next sparse phase
    iterations = []
    for i, cp in enumerate(compute_phases):
        # Find the sparse phase immediately following this compute phase
        opt_phase = None
        for p in phases:
            if not p.is_dense and p.start_ms >= cp.end_ms:
                # Only pair if it's before the next compute phase
                next_cp = compute_phases[i + 1] if i + 1 < len(compute_phases) else None
                if next_cp is None or p.end_ms <= next_cp.start_ms:
                    opt_phase = p
                    opt_phase.label = "optimizer/sync"
                break

        cp.label = "compute (fwd+bwd)"
        iterations.append(
            Iteration(index=i + 1, compute=cp, optimizer=opt_phase)
        )

    return iterations


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class KernelStats:
    name: str
    category: str
    invocations: int
    total_ns: int
    avg_ns: float
    min_ns: int
    max_ns: int
    total_grid_blocks: int = 0

    @property
    def total_ms(self) -> float:
        return self.total_ns / 1e6

    @property
    def avg_us(self) -> float:
        return self.avg_ns / 1e3

    @property
    def min_us(self) -> float:
        return self.min_ns / 1e3

    @property
    def max_us(self) -> float:
        return self.max_ns / 1e3


@dataclass
class CategoryStats:
    category: str
    invocations: int = 0
    total_ns: int = 0
    kernel_count: int = 0

    @property
    def total_ms(self) -> float:
        return self.total_ns / 1e6


@dataclass
class MemcpyStats:
    kind: str
    invocations: int
    total_ns: int
    total_bytes: int

    @property
    def total_ms(self) -> float:
        return self.total_ns / 1e6

    @property
    def total_mb(self) -> float:
        return self.total_bytes / 1e6

    @property
    def bandwidth_gbps(self) -> float:
        if self.total_ns == 0:
            return 0.0
        return (self.total_bytes / 1e9) / (self.total_ns / 1e9)


@dataclass
class GPUInfo:
    name: str
    sm_count: int
    total_memory_bytes: int
    clock_rate_hz: int
    compute_major: int
    compute_minor: int


@dataclass
class RankReport:
    rank: int
    gpu_info: list[GPUInfo]
    profile_duration_s: float
    total_kernel_time_ms: float
    num_kernels: int
    kernel_stats: list[KernelStats]
    category_stats: list[CategoryStats]
    memcpy_stats: list[MemcpyStats]
    nccl_summary: list[dict]
    iterations: list[Iteration]
    # Time window that was analyzed (None = full profile)
    window_start_ms: Optional[float] = None
    window_end_ms: Optional[float] = None


# ---------------------------------------------------------------------------
# Single-rank analysis
# ---------------------------------------------------------------------------


def _build_time_filter(
    window_start_ms: Optional[float], window_end_ms: Optional[float]
) -> tuple[str, list]:
    """Build a SQL WHERE clause fragment for time filtering."""
    if window_start_ms is not None and window_end_ms is not None:
        return (
            " AND k.start >= ? AND k.start < ?",
            [int(window_start_ms * 1e6), int(window_end_ms * 1e6)],
        )
    return ("", [])


def analyze_rank(
    sqlite_path: Path,
    top_n: int = 30,
    window_start_ms: Optional[float] = None,
    window_end_ms: Optional[float] = None,
) -> RankReport:
    """Analyze a single rank's .sqlite file and return a RankReport."""
    rank = extract_rank(sqlite_path.name)
    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row

    # GPU info
    gpu_info = []
    try:
        for row in conn.execute("SELECT * FROM TARGET_INFO_GPU"):
            gpu_info.append(
                GPUInfo(
                    name=row["name"] or "unknown",
                    sm_count=row["smCount"] or 0,
                    total_memory_bytes=row["totalMemory"] or 0,
                    clock_rate_hz=row["clockRate"] or 0,
                    compute_major=row["computeMajor"] or 0,
                    compute_minor=row["computeMinor"] or 0,
                )
            )
    except sqlite3.OperationalError:
        pass

    # Detect iterations (always on full profile)
    iterations = detect_iterations(conn)

    # Build time filter
    time_clause, time_params = _build_time_filter(window_start_ms, window_end_ms)

    # Profile time range from kernel timestamps (within window)
    row = conn.execute(
        "SELECT MIN(k.start) as t0, MAX(k.end) as t1, COUNT(*) as n, "
        "SUM(k.end - k.start) as total "
        f"FROM CUPTI_ACTIVITY_KIND_KERNEL k WHERE 1=1 {time_clause}",
        time_params,
    ).fetchone()
    profile_duration_s = (row["t1"] - row["t0"]) / 1e9 if row["t1"] else 0.0
    total_kernel_time_ms = row["total"] / 1e6 if row["total"] else 0.0
    num_kernels = row["n"] or 0

    # Per-kernel stats (by shortName)
    kernel_stats = []
    query = f"""
        SELECT s.value AS short_name,
               d.value AS demangled_name,
               COUNT(*) AS invocations,
               SUM(k.end - k.start) AS total_ns,
               AVG(k.end - k.start) AS avg_ns,
               MIN(k.end - k.start) AS min_ns,
               MAX(k.end - k.start) AS max_ns,
               SUM(k.gridX * k.gridY * k.gridZ) AS total_blocks
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.shortName = s.id
        JOIN StringIds d ON k.demangledName = d.id
        WHERE 1=1 {time_clause}
        GROUP BY s.value
        ORDER BY total_ns DESC
    """
    for row in conn.execute(query, time_params):
        cat = classify_kernel(row["short_name"], row["demangled_name"])
        kernel_stats.append(
            KernelStats(
                name=row["short_name"],
                category=cat,
                invocations=row["invocations"],
                total_ns=row["total_ns"],
                avg_ns=row["avg_ns"],
                min_ns=row["min_ns"],
                max_ns=row["max_ns"],
                total_grid_blocks=row["total_blocks"] or 0,
            )
        )

    # Category aggregation
    cat_map: dict[str, CategoryStats] = {}
    for ks in kernel_stats:
        if ks.category not in cat_map:
            cat_map[ks.category] = CategoryStats(category=ks.category)
        cs = cat_map[ks.category]
        cs.invocations += ks.invocations
        cs.total_ns += ks.total_ns
        cs.kernel_count += 1
    category_stats = sorted(cat_map.values(), key=lambda c: c.total_ns, reverse=True)

    # Memcpy stats
    memcpy_stats = []
    try:
        mc_clause = ""
        mc_params: list = []
        if window_start_ms is not None and window_end_ms is not None:
            mc_clause = " AND m.start >= ? AND m.start < ?"
            mc_params = [int(window_start_ms * 1e6), int(window_end_ms * 1e6)]
        query = f"""
            SELECT e.label AS kind,
                   COUNT(*) AS invocations,
                   SUM(m.end - m.start) AS total_ns,
                   SUM(m.bytes) AS total_bytes
            FROM CUPTI_ACTIVITY_KIND_MEMCPY m
            JOIN ENUM_CUDA_MEMCPY_OPER e ON m.copyKind = e.id
            WHERE 1=1 {mc_clause}
            GROUP BY e.label
            ORDER BY total_ns DESC
        """
        for row in conn.execute(query, mc_params):
            memcpy_stats.append(
                MemcpyStats(
                    kind=row["kind"],
                    invocations=row["invocations"],
                    total_ns=row["total_ns"],
                    total_bytes=row["total_bytes"],
                )
            )
    except sqlite3.OperationalError:
        pass

    # NCCL summary (from kernel names)
    nccl_summary = []
    for ks in kernel_stats:
        m = NCCL_PATTERN.match(ks.name)
        if m:
            nccl_summary.append(
                {
                    "operation": m.group(1),
                    "kernel": ks.name,
                    "invocations": ks.invocations,
                    "total_ms": round(ks.total_ms, 3),
                    "avg_us": round(ks.avg_us, 2),
                    "max_us": round(ks.max_us, 2),
                }
            )

    conn.close()

    return RankReport(
        rank=rank,
        gpu_info=gpu_info,
        profile_duration_s=round(profile_duration_s, 3),
        total_kernel_time_ms=round(total_kernel_time_ms, 3),
        num_kernels=num_kernels,
        kernel_stats=kernel_stats[:top_n],
        category_stats=category_stats,
        memcpy_stats=memcpy_stats,
        nccl_summary=nccl_summary,
        iterations=iterations,
        window_start_ms=window_start_ms,
        window_end_ms=window_end_ms,
    )


# ---------------------------------------------------------------------------
# Multi-rank aggregation
# ---------------------------------------------------------------------------


@dataclass
class MultiRankSummary:
    num_ranks: int
    gpu_info: list[GPUInfo]
    profile_durations_s: list[float]
    category_by_rank: dict[str, dict[int, CategoryStats]]
    kernel_by_rank: dict[str, dict[int, KernelStats]]
    nccl_by_rank: dict[str, dict[int, dict]]
    total_kernel_time_by_rank: dict[int, float]
    # Iteration info (from rank 0 or first available)
    iterations: list[Iteration]


def aggregate_ranks(reports: list[RankReport]) -> MultiRankSummary:
    """Aggregate multiple rank reports into a multi-rank summary."""
    category_by_rank: dict[str, dict[int, CategoryStats]] = defaultdict(dict)
    kernel_by_rank: dict[str, dict[int, KernelStats]] = defaultdict(dict)
    nccl_by_rank: dict[str, dict[int, dict]] = defaultdict(dict)
    total_kernel_time_by_rank: dict[int, float] = {}

    for rpt in reports:
        total_kernel_time_by_rank[rpt.rank] = rpt.total_kernel_time_ms
        for cs in rpt.category_stats:
            category_by_rank[cs.category][rpt.rank] = cs
        for ks in rpt.kernel_stats:
            kernel_by_rank[ks.name][rpt.rank] = ks
        for nccl in rpt.nccl_summary:
            nccl_by_rank[nccl["kernel"]][rpt.rank] = nccl

    ref_iters = reports[0].iterations if reports else []

    return MultiRankSummary(
        num_ranks=len(reports),
        gpu_info=reports[0].gpu_info if reports else [],
        profile_durations_s=[r.profile_duration_s for r in reports],
        category_by_rank=dict(category_by_rank),
        kernel_by_rank=dict(kernel_by_rank),
        nccl_by_rank=dict(nccl_by_rank),
        total_kernel_time_by_rank=total_kernel_time_by_rank,
        iterations=ref_iters,
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def fmt_table(headers: list[str], rows: list[list], col_widths: Optional[list[int]] = None) -> str:
    """Format a simple text table."""
    if not rows:
        return "(no data)\n"

    if col_widths is None:
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))

    col_widths = [min(w, 60) for w in col_widths]

    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    lines = [fmt.format(*[str(h)[:w] for h, w in zip(headers, col_widths)])]
    lines.append(fmt.format(*["-" * w for w in col_widths]))
    for row in rows:
        lines.append(fmt.format(*[str(c)[:w] for c, w in zip(row, col_widths)]))
    return "\n".join(lines) + "\n"


def truncate_name(name: str, max_len: int = 55) -> str:
    """Truncate a kernel name, keeping the most informative parts."""
    if len(name) <= max_len:
        return name
    return name[: max_len - 3] + "..."


# ---------------------------------------------------------------------------
# Text report formatting
# ---------------------------------------------------------------------------


def format_iterations(iterations: list[Iteration]) -> str:
    """Format the iteration timeline."""
    lines = []
    lines.append(f"{'─' * 80}")
    lines.append("  DETECTED ITERATIONS (from kernel timeline density)")
    lines.append(f"{'─' * 80}")

    if not iterations:
        lines.append("  (no iterations detected)")
        return "\n".join(lines)

    headers = ["Iter", "Compute (ms)", "Duration (ms)", "GPU Time (ms)", "Optimizer (ms)"]
    rows = []
    for it in iterations:
        opt_dur = f"{it.optimizer.duration_ms:.1f}" if it.optimizer else "—"
        rows.append([
            it.index,
            f"{it.compute.start_ms:.1f} – {it.compute.end_ms:.1f}",
            f"{it.compute.duration_ms:.1f}",
            f"{it.compute.gpu_time_ms:.1f}",
            opt_dur,
        ])
    lines.append(fmt_table(headers, rows))

    # Summary
    durations = [it.compute.duration_ms for it in iterations]
    gpu_times = [it.compute.gpu_time_ms for it in iterations]
    lines.append(
        f"  {len(iterations)} iteration(s) detected  |  "
        f"Compute phase: {statistics.mean(durations):.1f}ms mean "
        f"(std {statistics.stdev(durations):.1f}ms)"
        if len(durations) > 1
        else f"  {len(iterations)} iteration(s) detected  |  "
        f"Compute phase: {durations[0]:.1f}ms"
    )
    lines.append(
        f"  GPU time per iteration: {statistics.mean(gpu_times):.1f}ms mean"
    )

    return "\n".join(lines)


def format_single_rank_report(rpt: RankReport, top_n: int = 30) -> str:
    """Format a single-rank report as a text string."""
    lines = []
    lines.append(f"{'=' * 80}")
    if rpt.window_start_ms is not None:
        lines.append(
            f"  NSYS Profile Report — Rank {rpt.rank}  "
            f"[window: {rpt.window_start_ms:.1f} – {rpt.window_end_ms:.1f} ms]"
        )
    else:
        lines.append(f"  NSYS Profile Report — Rank {rpt.rank}")
    lines.append(f"{'=' * 80}")

    # GPU info
    if rpt.gpu_info:
        g = rpt.gpu_info[0]
        lines.append(
            f"\n  GPU: {g.name}  |  SMs: {g.sm_count}  |  "
            f"Memory: {g.total_memory_bytes / 1e9:.1f} GB  |  "
            f"Compute: {g.compute_major}.{g.compute_minor}"
        )

    lines.append(f"  Profile duration: {rpt.profile_duration_s:.3f} s")
    lines.append(
        f"  Total kernel time: {rpt.total_kernel_time_ms:.2f} ms "
        f"({rpt.num_kernels} kernels)"
    )

    # Iteration timeline
    if rpt.iterations:
        lines.append("")
        lines.append(format_iterations(rpt.iterations))

    # Category breakdown
    lines.append(f"\n{'─' * 80}")
    lines.append("  KERNEL TIME BY CATEGORY")
    lines.append(f"{'─' * 80}")
    headers = ["Category", "Time (ms)", "% Total", "Invocations", "# Kernels"]
    rows = []
    for cs in rpt.category_stats:
        pct = (
            cs.total_ms / rpt.total_kernel_time_ms * 100
            if rpt.total_kernel_time_ms
            else 0
        )
        rows.append([
            cs.category,
            f"{cs.total_ms:.2f}",
            f"{pct:.1f}%",
            cs.invocations,
            cs.kernel_count,
        ])
    lines.append(fmt_table(headers, rows))

    # Top kernels
    lines.append(f"{'─' * 80}")
    lines.append(f"  TOP {top_n} KERNELS BY TOTAL TIME")
    lines.append(f"{'─' * 80}")
    headers = [
        "Kernel",
        "Category",
        "Invoc",
        "Total(ms)",
        "Avg(us)",
        "Min(us)",
        "Max(us)",
    ]
    rows = []
    for ks in rpt.kernel_stats[:top_n]:
        rows.append([
            truncate_name(ks.name),
            ks.category,
            ks.invocations,
            f"{ks.total_ms:.2f}",
            f"{ks.avg_us:.1f}",
            f"{ks.min_us:.1f}",
            f"{ks.max_us:.1f}",
        ])
    lines.append(fmt_table(headers, rows))

    # NCCL
    if rpt.nccl_summary:
        lines.append(f"{'─' * 80}")
        lines.append("  NCCL COLLECTIVES")
        lines.append(f"{'─' * 80}")
        headers = ["Operation", "Invocations", "Total(ms)", "Avg(us)", "Max(us)"]
        rows = []
        for n in rpt.nccl_summary:
            rows.append([
                n["operation"],
                n["invocations"],
                f"{n['total_ms']:.2f}",
                f"{n['avg_us']:.1f}",
                f"{n['max_us']:.1f}",
            ])
        lines.append(fmt_table(headers, rows))

    # Memcpy
    if rpt.memcpy_stats:
        lines.append(f"{'─' * 80}")
        lines.append("  MEMORY COPIES")
        lines.append(f"{'─' * 80}")
        headers = ["Kind", "Invocations", "Total(ms)", "Total(MB)", "BW(GB/s)"]
        rows = []
        for mc in rpt.memcpy_stats:
            rows.append([
                mc.kind,
                mc.invocations,
                f"{mc.total_ms:.2f}",
                f"{mc.total_mb:.1f}",
                f"{mc.bandwidth_gbps:.1f}",
            ])
        lines.append(fmt_table(headers, rows))

    return "\n".join(lines)


def format_multi_rank_report(summary: MultiRankSummary, top_n: int = 20) -> str:
    """Format a multi-rank summary as a text string."""
    lines = []
    lines.append(f"{'=' * 100}")
    lines.append(f"  NSYS Multi-Rank Summary — {summary.num_ranks} ranks")
    lines.append(f"{'=' * 100}")

    if summary.gpu_info:
        g = summary.gpu_info[0]
        lines.append(
            f"\n  GPU: {g.name}  |  SMs: {g.sm_count}  |  "
            f"Memory: {g.total_memory_bytes / 1e9:.1f} GB  |  "
            f"Compute: {g.compute_major}.{g.compute_minor}"
        )

    durations = summary.profile_durations_s
    lines.append(
        f"  Profile duration: min={min(durations):.3f}s  max={max(durations):.3f}s  "
        f"mean={sum(durations)/len(durations):.3f}s"
    )

    kernel_times = list(summary.total_kernel_time_by_rank.values())
    lines.append(
        f"  Total kernel time: min={min(kernel_times):.1f}ms  "
        f"max={max(kernel_times):.1f}ms  "
        f"mean={sum(kernel_times)/len(kernel_times):.1f}ms"
    )

    # Iteration timeline (from reference rank)
    if summary.iterations:
        lines.append("")
        lines.append(format_iterations(summary.iterations))

    # Category breakdown
    lines.append(f"\n{'─' * 100}")
    lines.append("  KERNEL TIME BY CATEGORY (aggregated across ranks)")
    lines.append(f"{'─' * 100}")
    headers = ["Category", "Mean(ms)", "Min(ms)", "Max(ms)", "Std(ms)", "Mean Invoc"]
    rows = []
    for cat, rank_map in sorted(
        summary.category_by_rank.items(),
        key=lambda x: -sum(cs.total_ms for cs in x[1].values()) / len(x[1]),
    ):
        times = [cs.total_ms for cs in rank_map.values()]
        invocs = [cs.invocations for cs in rank_map.values()]
        mean_t = statistics.mean(times)
        std_t = statistics.stdev(times) if len(times) > 1 else 0.0
        rows.append([
            cat,
            f"{mean_t:.2f}",
            f"{min(times):.2f}",
            f"{max(times):.2f}",
            f"{std_t:.2f}",
            f"{statistics.mean(invocs):.0f}",
        ])
    lines.append(fmt_table(headers, rows))

    # Top kernels
    lines.append(f"{'─' * 100}")
    lines.append(f"  TOP {top_n} KERNELS BY MEAN TIME ACROSS RANKS")
    lines.append(f"{'─' * 100}")

    kernel_means = []
    for name, rank_map in summary.kernel_by_rank.items():
        times = [ks.total_ms for ks in rank_map.values()]
        mean_t = statistics.mean(times)
        std_t = statistics.stdev(times) if len(times) > 1 else 0.0
        cat = next(iter(rank_map.values())).category
        kernel_means.append(
            (name, cat, mean_t, min(times), max(times), std_t, len(rank_map))
        )

    kernel_means.sort(key=lambda x: -x[2])
    headers = [
        "Kernel",
        "Category",
        "Mean(ms)",
        "Min(ms)",
        "Max(ms)",
        "Std(ms)",
        "#Ranks",
    ]
    rows = []
    for name, cat, mean_t, min_t, max_t, std_t, n_ranks in kernel_means[:top_n]:
        rows.append([
            truncate_name(name),
            cat,
            f"{mean_t:.2f}",
            f"{min_t:.2f}",
            f"{max_t:.2f}",
            f"{std_t:.2f}",
            n_ranks,
        ])
    lines.append(fmt_table(headers, rows))

    # NCCL
    if summary.nccl_by_rank:
        lines.append(f"{'─' * 100}")
        lines.append("  NCCL COLLECTIVES ACROSS RANKS")
        lines.append(f"{'─' * 100}")
        headers = ["Operation", "Mean(ms)", "Min(ms)", "Max(ms)", "Std(ms)", "#Ranks"]
        rows = []
        for kernel, rank_map in sorted(
            summary.nccl_by_rank.items(),
            key=lambda x: -statistics.mean(
                [n["total_ms"] for n in x[1].values()]
            ),
        ):
            times = [n["total_ms"] for n in rank_map.values()]
            mean_t = statistics.mean(times)
            std_t = statistics.stdev(times) if len(times) > 1 else 0.0
            m = NCCL_PATTERN.match(kernel)
            op = m.group(1) if m else kernel
            rows.append([
                op,
                f"{mean_t:.2f}",
                f"{min(times):.2f}",
                f"{max(times):.2f}",
                f"{std_t:.2f}",
                len(rank_map),
            ])
        lines.append(fmt_table(headers, rows))

    # Load imbalance
    lines.append(f"{'─' * 100}")
    lines.append(
        "  LOAD IMBALANCE (categories with highest coefficient of variation)"
    )
    lines.append(f"{'─' * 100}")
    imbalances = []
    for cat, rank_map in summary.category_by_rank.items():
        times = [cs.total_ms for cs in rank_map.values()]
        mean_t = statistics.mean(times)
        if mean_t < 1.0:
            continue
        std_t = statistics.stdev(times) if len(times) > 1 else 0.0
        cv = std_t / mean_t if mean_t > 0 else 0.0
        imbalances.append((cat, cv, mean_t, std_t, min(times), max(times)))
    imbalances.sort(key=lambda x: -x[1])
    headers = ["Category", "CV", "Mean(ms)", "Std(ms)", "Min(ms)", "Max(ms)"]
    rows = []
    for cat, cv, mean_t, std_t, min_t, max_t in imbalances[:15]:
        rows.append([
            cat,
            f"{cv:.3f}",
            f"{mean_t:.2f}",
            f"{std_t:.2f}",
            f"{min_t:.2f}",
            f"{max_t:.2f}",
        ])
    lines.append(fmt_table(headers, rows))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extract_rank(filename: str) -> int:
    """Extract rank number from filename."""
    m = re.search(r"-rank(\d+)-", filename)
    return int(m.group(1)) if m else -1


def find_sqlite_files(path: Path) -> list[Path]:
    """Find .sqlite files in a path."""
    if path.is_file() and path.suffix == ".sqlite":
        return [path]
    elif path.is_dir():
        files = sorted(
            path.glob("*.sqlite"), key=lambda f: extract_rank(f.name)
        )
        if not files:
            print(f"No .sqlite files found in {path}", file=sys.stderr)
            print(
                "Run nsys_convert.py first to convert .nsys-rep files.",
                file=sys.stderr,
            )
            sys.exit(1)
        return files
    else:
        print(f"Path does not exist: {path}", file=sys.stderr)
        sys.exit(1)


def resolve_window(
    args_iteration: Optional[int],
    args_window: Optional[str],
    iterations: list[Iteration],
) -> tuple[Optional[float], Optional[float]]:
    """Resolve --iteration or --window into (start_ms, end_ms)."""
    if args_window:
        parts = args_window.split(",")
        return float(parts[0]), float(parts[1])
    if args_iteration is not None:
        idx = args_iteration
        if idx < 1 or idx > len(iterations):
            print(
                f"Iteration {idx} out of range (have {len(iterations)})",
                file=sys.stderr,
            )
            sys.exit(1)
        it = iterations[idx - 1]
        return it.compute.start_ms, it.compute.end_ms
    return None, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Analyze nsys SQLite exports")
    parser.add_argument(
        "path",
        type=Path,
        help="Path to a .sqlite file or directory containing .sqlite files",
    )
    parser.add_argument(
        "--top", type=int, default=30, help="Number of top kernels to show"
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument("--output", "-o", type=Path, help="Write output to file")
    parser.add_argument(
        "--ranks",
        type=str,
        help="Comma-separated ranks to analyze (e.g., '0,1,2' or '0-7')",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        help="Restrict to Nth compute iteration (1-indexed). "
        "Detects iterations from kernel timeline density.",
    )
    parser.add_argument(
        "--window",
        type=str,
        help="Restrict to explicit time window in ms (e.g., '2355,2767'). "
        "Overrides --iteration.",
    )
    args = parser.parse_args()

    sqlite_files = find_sqlite_files(args.path)

    # Filter ranks if specified
    if args.ranks:
        selected_ranks: set[int] = set()
        for part in args.ranks.split(","):
            if "-" in part:
                lo, hi = part.split("-")
                selected_ranks.update(range(int(lo), int(hi) + 1))
            else:
                selected_ranks.add(int(part))
        sqlite_files = [
            f for f in sqlite_files if extract_rank(f.name) in selected_ranks
        ]

    print(f"Analyzing {len(sqlite_files)} .sqlite file(s)...")

    # First pass: detect iterations from first file
    first_conn = sqlite3.connect(str(sqlite_files[0]))
    first_conn.row_factory = sqlite3.Row
    ref_iterations = detect_iterations(first_conn)
    first_conn.close()

    # Resolve time window
    win_start, win_end = resolve_window(
        args.iteration, args.window, ref_iterations
    )
    if win_start is not None:
        print(f"  Time window: {win_start:.1f} – {win_end:.1f} ms")

    reports = []
    for i, f in enumerate(sqlite_files):
        rank = extract_rank(f.name)
        print(f"  [{i+1}/{len(sqlite_files)}] rank {rank}...", end="", flush=True)
        rpt = analyze_rank(
            f, top_n=args.top, window_start_ms=win_start, window_end_ms=win_end
        )
        reports.append(rpt)
        print(f" {rpt.total_kernel_time_ms:.1f}ms kernel time")

    reports.sort(key=lambda r: r.rank)

    if args.format == "json":
        data: dict = {
            "num_ranks": len(reports),
            "iterations": [
                {
                    "index": it.index,
                    "compute_start_ms": it.compute.start_ms,
                    "compute_end_ms": it.compute.end_ms,
                    "compute_duration_ms": it.compute.duration_ms,
                    "compute_gpu_time_ms": it.compute.gpu_time_ms,
                }
                for it in ref_iterations
            ],
            "window": (
                {"start_ms": win_start, "end_ms": win_end}
                if win_start is not None
                else None
            ),
            "ranks": [],
        }
        for rpt in reports:
            rank_data = {
                "rank": rpt.rank,
                "profile_duration_s": rpt.profile_duration_s,
                "total_kernel_time_ms": rpt.total_kernel_time_ms,
                "num_kernels": rpt.num_kernels,
                "category_stats": [
                    {
                        "category": cs.category,
                        "total_ms": cs.total_ms,
                        "invocations": cs.invocations,
                    }
                    for cs in rpt.category_stats
                ],
                "top_kernels": [
                    {
                        "name": ks.name,
                        "category": ks.category,
                        "invocations": ks.invocations,
                        "total_ms": ks.total_ms,
                        "avg_us": ks.avg_us,
                    }
                    for ks in rpt.kernel_stats
                ],
                "nccl_summary": rpt.nccl_summary,
                "memcpy_stats": [
                    {
                        "kind": mc.kind,
                        "total_ms": mc.total_ms,
                        "total_mb": mc.total_mb,
                    }
                    for mc in rpt.memcpy_stats
                ],
            }
            data["ranks"].append(rank_data)

        output = json.dumps(data, indent=2)
    else:
        parts = []
        if len(reports) == 1:
            parts.append(format_single_rank_report(reports[0], args.top))
        else:
            summary = aggregate_ranks(reports)
            parts.append(format_multi_rank_report(summary, args.top))
            parts.append(f"\n\n{'#' * 100}")
            parts.append("  DETAILED REPORT FOR RANK 0 (reference)")
            parts.append(f"{'#' * 100}\n")
            rank0 = next((r for r in reports if r.rank == 0), reports[0])
            parts.append(format_single_rank_report(rank0, args.top))

        output = "\n".join(parts)

    if args.output:
        args.output.write_text(output)
        print(f"\nReport written to {args.output}")
    else:
        print()
        print(output)


if __name__ == "__main__":
    main()
