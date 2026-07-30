#!/usr/bin/env python3
"""Render presentation-specific Sparser MoE benchmark plots."""

from __future__ import annotations

import glob
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "assets" / "generated"

GREEN = "#76B900"
DARK_GREEN = "#355E0B"
TEAL = "#1DBBA4"
BLUE = "#0074DF"
PURPLE = "#952FC6"
ORANGE = "#EF9100"
CHARCOAL = "#313131"
GRAY = "#757575"
GRID = "#D9D9D9"
LIGHT = "#F7F7F7"


def configure_style() -> None:
    """Configure a slide-friendly NVIDIA-light plotting style."""
    managed_dir = Path("/Library/Fonts/Managed")
    if managed_dir.exists():
        for candidate in managed_dir.glob("NVIDIASans-*.ttf"):
            font_manager.fontManager.addfont(candidate)
    if managed_dir.exists() and any(managed_dir.glob("NVIDIASans-*.ttf")):
        plt.rcParams["font.family"] = "NVIDIA Sans"
    else:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["Helvetica Neue", "Arial", "DejaVu Sans"]
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.edgecolor": GRAY,
            "axes.labelcolor": CHARCOAL,
            "xtick.color": CHARCOAL,
            "ytick.color": CHARCOAL,
            "text.color": CHARCOAL,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    OUT.mkdir(parents=True, exist_ok=True)


def clean_axes(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.set_axisbelow(True)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def add_bar_labels(
    ax: plt.Axes,
    bars,
    *,
    fmt: str = "{:.1f}",
    pad_fraction: float = 0.018,
    color: str = CHARCOAL,
) -> None:
    ymax = ax.get_ylim()[1]
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + ymax * pad_fraction,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=10.5,
            color=color,
        )


def router_pr_progression() -> None:
    """Plot matched 2304/36 trace-back speedup across router PR checkpoints."""
    data = pd.read_csv(DATA / "router_trace_back.csv")
    query = data[
        (data["num_tokens"] == 65536)
        & (data["num_experts"] == 2304)
        & (data["topk"] == 36)
        & (data["kernel"] == "topk")
        & (data["score_function"] == "softmax")
        & (data["group_topk"] == 0)
    ]
    checkpoints = ["trace-back", "trace-back-2821", "trace-back-3012", "trace-back-3129"]
    labels = ["Before radix", "#2821\nRadix", "#3012\nKernel family", "#3129\nDense output"]
    forward = []
    backward = []
    for checkpoint in checkpoints:
        rows = query[query["checkpoint"] == checkpoint]
        forward.append(float(rows[rows["test_pass"] == "forward"]["fused_ms"].iloc[0]))
        backward.append(float(rows[rows["test_pass"] == "backward_raw"]["fused_ms"].iloc[0]))
    forward_speedup = forward[0] / np.array(forward)
    backward_speedup = backward[0] / np.array(backward)

    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(10.8, 4.4))
    bars_fwd = ax.bar(x - width / 2, forward_speedup, width, color=GREEN, label="Forward")
    bars_bwd = ax.bar(x + width / 2, backward_speedup, width, color=DARK_GREEN, label="Backward")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Speedup vs. pre-radix checkpoint")
    ax.set_ylim(0, 27)
    clean_axes(ax)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    add_bar_labels(ax, bars_fwd, fmt="{:.1f}×")
    add_bar_labels(ax, bars_bwd, fmt="{:.1f}×")
    fig.tight_layout()
    save(fig, "router_pr_progression")


def p3r_stages() -> None:
    """Plot the incremental P3R bandwidth story for the target router shape."""
    stage_files = sorted(glob.glob(str(DATA / "router_fix_p3R_*.csv")))
    stage_labels = [
        "P2\nbaseline",
        "Fused\nloops",
        "Async\nload",
        "Packed\nradix",
        "Static\nscore",
        "Full\nradix",
        "Merged\nhead",
    ]
    records = []
    for stage_index, path in enumerate(stage_files):
        frame = pd.read_csv(path)
        frame["stage"] = stage_index
        records.append(frame)
    data = pd.concat(records, ignore_index=True)
    query = data[
        (data["num_tokens"] == 8192)
        & (data["num_experts"] == 2304)
        & (data["topk"] == 36)
        & (data["score_function"] == "softmax")
        & (data["group_topk"] == 0)
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    for ax, test_pass, title in zip(
        axes, ["forward", "backward_raw"], ["Forward", "Backward"]
    ):
        for kernel, color, marker, label in [
            ("topk", GREEN, "o", "Top-k"),
            ("aux_loss", BLUE, "s", "Aux-loss"),
        ]:
            rows = (
                query[(query["kernel"] == kernel) & (query["test_pass"] == test_pass)]
                .sort_values("stage")
                .set_index("stage")
            )
            ax.plot(
                range(len(stage_labels)),
                rows.loc[range(len(stage_labels)), "fused_gbps"],
                color=color,
                marker=marker,
                linewidth=2.4,
                markersize=6,
                label=label,
            )
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks(range(len(stage_labels)), stage_labels)
        ax.set_ylabel("Effective GB/s")
        clean_axes(ax)
        ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(w_pad=2)
    save(fig, "router_p3r_stages")


def router_roof_reference() -> None:
    """Compare final target-shape effective bandwidth with the raw B300 HBM ceiling."""
    data = pd.read_csv(DATA / "router_fix_p3R_6_9cfb651a_head.csv")
    query = data[
        (data["num_tokens"] == 8192)
        & (data["num_experts"] == 2304)
        & (data["topk"] == 36)
        & (data["score_function"] == "softmax")
        & (data["group_topk"] == 0)
    ]
    cases = [
        ("Top-k FWD", "topk", "forward"),
        ("Top-k BWD", "topk", "backward_raw"),
        ("Aux FWD", "aux_loss", "forward"),
        ("Aux BWD", "aux_loss", "backward_raw"),
    ]
    values = [
        float(
            query[(query["kernel"] == kernel) & (query["test_pass"] == test_pass)][
                "fused_gbps"
            ].iloc[0]
        )
        for _, kernel, test_pass in cases
    ]
    labels = [name for name, _, _ in cases]
    percentages = np.array(values) / 8000 * 100

    fig, ax = plt.subplots(figsize=(10.4, 4.0))
    bars = ax.bar(labels, percentages, color=[GREEN, DARK_GREEN, BLUE, PURPLE], width=0.64)
    ax.axhline(100, color=ORANGE, linewidth=2, linestyle="--", label="B300 raw HBM: 8 TB/s")
    ax.set_ylabel("Effective bandwidth / 8 TB/s")
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    clean_axes(ax)
    ax.legend(frameon=False, loc="upper left")
    for bar, value, pct in zip(bars, values, percentages):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{value:.0f} GB/s\n({pct:.0f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    fig.tight_layout()
    save(fig, "router_roof_reference")


def hybrid_ep_microbench() -> None:
    """Plot matched before/after effective payload throughput for HybridEP kernels."""
    data = pd.read_csv(DATA / "hybrid_ep_microbench.csv")
    x = np.arange(len(data))
    width = 0.34
    fig, ax = plt.subplots(figsize=(10.8, 4.4))
    before = ax.bar(
        x - width / 2,
        data["before_gbs"],
        width,
        color=GREEN,
        label="Before",
    )
    after = ax.bar(
        x + width / 2,
        data["after_gbs"],
        width,
        color=DARK_GREEN,
        label="After",
    )
    ax.set_xticks(
        x,
        [
            "Permute",
            "Unpermute",
            "Dispatch\n(kernel only)",
            "Combine\n(kernel only)",
        ],
    )
    ax.set_ylabel("Effective bandwidth (GB/s, higher is better)")
    ax.set_ylim(0, 740)
    clean_axes(ax)
    for bars in (before, after):
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 7,
                f"{value:.0f}",
                ha="center",
                va="bottom",
                fontsize=9.2,
                fontweight="bold",
                color=CHARCOAL,
            )
    for idx, row in data.iterrows():
        gain = row["after_gbs"] / row["before_gbs"]
        ax.text(
            idx,
            max(row["before_gbs"], row["after_gbs"]) + 31,
            f"{gain:.2f}×",
            ha="center",
            va="bottom",
            fontsize=9.6,
            fontweight="bold",
            color=DARK_GREEN,
        )
    ax.legend(frameon=False, ncol=2, loc="upper left")
    fig.tight_layout()
    save(fig, "hybrid_ep_microbench")


def full_model_stages() -> None:
    """Plot the staged OCI-AGA GB300 full-model sweep."""
    data = pd.read_csv(DATA / "full_model_stages.csv")
    stages = list(dict.fromkeys(data["stage"]))
    models = list(dict.fromkeys(data["model"]))
    x = np.arange(len(stages))
    width = 0.34
    fig, ax = plt.subplots(figsize=(11.2, 4.5))
    bars = []
    for idx, (model, color) in enumerate(zip(models, [GREEN, DARK_GREEN])):
        values = (
            data[data["model"] == model].set_index("stage").loc[stages, "median_tflops"]
        )
        current = ax.bar(
            x + (idx - 0.5) * width, values, width, color=color, label=model
        )
        bars.append(current)
    ax.set_xticks(
        x,
        ["No tune /\nno radix", "HybridEP\ntuned", "Router\n#2821", "Router\n#3012", "Full sparse\nstack"],
    )
    ax.set_ylabel("Median TFLOP/s/GPU")
    ax.set_ylim(0, 500)
    clean_axes(ax)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    for group in bars:
        add_bar_labels(ax, group, fmt="{:.0f}", pad_fraction=0.012)
    fig.tight_layout()
    save(fig, "full_model_stages")


def moe_only_matrix() -> None:
    """Plot the canonical 160-iteration MoE-only matrix."""
    data = pd.read_csv(DATA / "moe_only_matrix.csv")
    y = np.arange(len(data))
    height = 0.34
    fig, ax = plt.subplots(figsize=(11.2, 5.0))
    no_tune = ax.barh(
        y + height / 2, data["no_tune"], height, color=GREEN, label="No tune / no radix"
    )
    optimized = ax.barh(
        y - height / 2, data["optimized"], height, color=DARK_GREEN, label="Optimized stack"
    )
    ax.set_yticks(y, data["model"])
    ax.invert_yaxis()
    ax.set_xlabel("Median TFLOP/s/GPU")
    ax.set_xlim(0, 1770)
    clean_axes(ax, grid_axis="x")
    ax.legend(frameon=False, ncol=2, loc="lower right")
    for bars in [no_tune, optimized]:
        for bar in bars:
            value = bar.get_width()
            ax.text(
                value + 18,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.0f}",
                va="center",
                ha="left",
                fontsize=10,
            )
    for idx, row in data.iterrows():
        gain = row["optimized"] / row["no_tune"]
        ax.text(
            max(row["optimized"], row["no_tune"]) + 105,
            idx,
            f"{gain:.2f}×",
            va="center",
            ha="left",
            fontsize=10.5,
            fontweight="bold",
            color=DARK_GREEN,
        )
    fig.tight_layout()
    save(fig, "moe_only_matrix")


def main() -> None:
    configure_style()
    router_pr_progression()
    p3r_stages()
    router_roof_reference()
    hybrid_ep_microbench()
    full_model_stages()
    moe_only_matrix()
    print(f"Wrote plots to {OUT}")


if __name__ == "__main__":
    main()
