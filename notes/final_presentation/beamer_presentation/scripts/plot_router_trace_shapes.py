#!/usr/bin/env python3
"""Render complete 0730 fused-router trace comparisons for the Beamer deck."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CHECKPOINTS = (
    ("Pre-large-top-k", "trace-back.csv"),
    ("PR #2821", "trace-back-2821.csv"),
    ("PR #3012", "trace-back-3012.csv"),
    ("PR #3129", "trace-back-3129-sparse.csv"),
)
SHAPES = ("256/8", "384/6", "512/10", "512/22", "896/16", "2304/16", "2304/36")
TOKENS = (4096, 16384, 65536, 262144)
PLOTS = (
    ("topk", "forward"),
    ("topk", "backward_raw"),
    ("aux_loss", "forward"),
    ("aux_loss", "backward_raw"),
)
SCORE_FUNCTIONS = ("softmax", "sigmoid")
# Distinct, colorblind-safe checkpoint colors: baseline, radix, secondary
# optimizations, and dense int16 output respectively.
COLORS = ("#0072B2", "#E69F00", "#CC79A7", "#009E73")


def _with_metadata(data: pd.DataFrame, label: str) -> pd.DataFrame:
    data = data.copy()
    data["checkpoint"] = label
    data["shape"] = data["num_experts"].astype(str) + "/" + data["topk"].astype(str)
    return data


def load_measurements(data_dir: Path) -> pd.DataFrame:
    records = []
    for label, filename in CHECKPOINTS:
        data = _with_metadata(pd.read_csv(data_dir / filename), label)
        if label == "PR #3129":
            dense = _with_metadata(
                pd.read_csv(data_dir / "trace-back-3129-dense-int16-topk-forward.csv"),
                "PR #3129 int16 output",
            )
            data = data[~((data["kernel"] == "topk") & (data["test_pass"] == "forward"))]
            data = pd.concat((data, dense), ignore_index=True)
        records.append(data)
    return pd.concat(records, ignore_index=True)


def plot_one(
    data: pd.DataFrame,
    kernel: str,
    test_pass: str,
    score_function: str,
    tokens_subset: tuple[int, ...],
    output: Path,
) -> None:
    # Two panels per slide preserve the complete token sweep while giving the
    # legend, axes, and seven-shape labels enough projected size.
    fig, axes = plt.subplots(1, len(tokens_subset), figsize=(13.0, 4.2), sharey=True)
    if len(tokens_subset) == 1:
        axes = (axes,)
    subset = data[
        (data["kernel"] == kernel)
        & (data["test_pass"] == test_pass)
        & (data["score_function"] == score_function)
    ]
    ymax = 1.08 * max(subset["fused_gbps"].max(), subset["ref_gbps"].max())
    x = list(range(len(SHAPES)))

    for col, tokens in enumerate(tokens_subset):
        ax = axes[col]
        panel = subset[subset["num_tokens"] == tokens]
        reference = panel.groupby("shape")["ref_gbps"].mean()
        ax.plot(
            x,
            [reference.get(shape, float("nan")) for shape in SHAPES],
            color="#202020",
            linestyle="--",
            linewidth=1.15,
            marker="x",
            markersize=4,
            label="unfused reference" if col == 0 else None,
        )
        for (label, _), color in zip(CHECKPOINTS, COLORS):
            checkpoint = "PR #3129 int16 output" if label == "PR #3129" and kernel == "topk" and test_pass == "forward" else label
            series = panel[panel["checkpoint"] == checkpoint].set_index("shape")
            ax.plot(
                x,
                [series["fused_gbps"].get(shape, float("nan")) for shape in SHAPES],
                color=color,
                linewidth=1.55,
                marker="o",
                markersize=3.6,
                label=checkpoint if col == 0 else None,
            )
        ax.set_title(f"T={tokens}", fontsize=10)
        ax.set_xticks(x, SHAPES, rotation=35, ha="right", fontsize=8.5)
        ax.set_ylim(0, ymax)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        if col == 0:
            ax.set_ylabel("Effective GB/s", fontsize=9)
        ax.tick_params(axis="y", labelsize=8.5)

    axes[0].legend(loc="upper left", fontsize=7.5, frameon=False)
    fig.subplots_adjust(left=0.060, right=0.997, bottom=0.25, top=0.91, wspace=0.20)
    fig.savefig(output, dpi=220, facecolor="white")
    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    data_dir = repo_root / "data" / "trace-back-comparisons"
    output_dir = Path(__file__).resolve().parents[1] / "assets" / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_measurements(data_dir)
    for kernel, test_pass in PLOTS:
        for score_function in SCORE_FUNCTIONS:
            for suffix, tokens_subset in (("low_tokens", TOKENS[:2]), ("high_tokens", TOKENS[2:])):
                output = output_dir / f"trace_{kernel}_{test_pass}_{score_function}_{suffix}.pdf"
                plot_one(data, kernel, test_pass, score_function, tokens_subset, output)
                print(output)


if __name__ == "__main__":
    main()
