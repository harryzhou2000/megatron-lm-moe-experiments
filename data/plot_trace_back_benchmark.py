#!/usr/bin/env python3
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Combine and plot the fused-router trace checkpoint benchmarks."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

CHECKPOINTS = (
    ("trace-back", "trace-back.csv"),
    ("trace-back-2821", "trace-back-2821.csv"),
    ("trace-back-3012", "trace-back-3012.csv"),
    ("trace-back-3129", "trace-back-3129-sparse.csv"),
)
SHAPES = ("256/8", "384/6", "512/10", "512/22", "896/16", "2304/16", "2304/36")
TOKENS = (4096, 16384, 65536, 262144)
SCORE_FUNCTIONS = ("softmax", "sigmoid")
PLOTS = (
    ("topk", "forward", "topk_forward.png"),
    ("topk", "backward_raw", "topk_backward_raw.png"),
    ("aux_loss", "forward", "aux_loss_forward.png"),
    ("aux_loss", "backward_raw", "aux_loss_backward_raw.png"),
)
COLORS = ("tab:blue", "tab:orange", "tab:green", "tab:red")
P3R_RADIX_THRESHOLD = 10


def _add_labels(data: pd.DataFrame, checkpoint: str) -> pd.DataFrame:
    data = data.copy()
    data["checkpoint"] = checkpoint
    data["shape"] = data["num_experts"].astype(str) + "/" + data["topk"].astype(str)
    return data


def _load_checkpoints(data_dir: Path) -> pd.DataFrame:
    all_data = []
    for checkpoint, filename in CHECKPOINTS:
        data = _add_labels(pd.read_csv(data_dir / filename), checkpoint)
        if checkpoint == "trace-back-3129":
            dense = _add_labels(
                pd.read_csv(data_dir / "trace-back-3129-dense-int16-topk-forward.csv"),
                checkpoint,
            )
            data = data[
                ~((data["kernel"] == "topk") & (data["test_pass"] == "forward"))
            ]
            data = pd.concat((data, dense), ignore_index=True)
            data.to_csv(data_dir / "trace-back-3129.csv", index=False)
        all_data.append(data)
    return pd.concat(all_data, ignore_index=True)


def _plot(data: pd.DataFrame, kernel: str, test_pass: str, output_path: Path) -> None:
    fig, axes = plt.subplots(
        len(SCORE_FUNCTIONS),
        len(TOKENS),
        figsize=(4.5 * len(TOKENS), 3.5 * len(SCORE_FUNCTIONS)),
        sharey="row",
    )
    x_positions = range(len(SHAPES))

    y_max_by_score_function = {}
    for score_function in SCORE_FUNCTIONS:
        row_data = data[
            (data["kernel"] == kernel)
            & (data["test_pass"] == test_pass)
            & (data["score_function"] == score_function)
        ]
        y_max_by_score_function[score_function] = 1.05 * max(
            row_data["fused_gbps"].max(), row_data["ref_gbps"].max()
        )

    for row, score_function in enumerate(SCORE_FUNCTIONS):
        for col, num_tokens in enumerate(TOKENS):
            ax = axes[row, col]
            subset = data[
                (data["kernel"] == kernel)
                & (data["test_pass"] == test_pass)
                & (data["score_function"] == score_function)
                & (data["num_tokens"] == num_tokens)
            ]
            reference = subset.groupby("shape", as_index=True)["ref_gbps"].mean()
            reference_values = [
                reference.loc[shape] if shape in reference.index else float("nan")
                for shape in SHAPES
            ]
            ax.plot(
                x_positions,
                reference_values,
                color="black",
                linestyle="--",
                marker="x",
                label="unfused ref" if row == 0 and col == 0 else None,
            )
            for (checkpoint, _), color in zip(CHECKPOINTS, COLORS):
                series = subset[subset["checkpoint"] == checkpoint].set_index("shape")
                y_values = [
                    (
                        series.loc[shape, "fused_gbps"]
                        if shape in series.index
                        else float("nan")
                    )
                    for shape in SHAPES
                ]
                label = (
                    "3129 dense int16"
                    if checkpoint == "trace-back-3129"
                    and test_pass == "forward"
                    and kernel == "topk"
                    else checkpoint
                )
                ax.plot(x_positions, y_values, marker="o", color=color, label=label)
            ax.set_title(f"{score_function}, T={num_tokens}")
            ax.set_xticks(x_positions, SHAPES, rotation=45, ha="right")
            ax.set_ylim(0, y_max_by_score_function[score_function])
            ax.grid(axis="y", alpha=0.3)
            if col == 0:
                ax.set_ylabel("Effective GB/s")
            if row == 0 and col == 0:
                ax.legend(fontsize="small")

    fig.suptitle(
        f"Fused router trace checkpoints: {kernel} {test_pass} "
        f"(p3R radix threshold={P3R_RADIX_THRESHOLD})",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    output_dir = Path(__file__).parent / "trace-back-comparisons"
    output_dir.mkdir(exist_ok=True)

    data = _load_checkpoints(output_dir)
    data.to_csv(output_dir / "router_benchmark_combined.csv", index=False)
    for kernel, test_pass, filename in PLOTS:
        _plot(data, kernel, test_pass, output_dir / filename)
    print(f"Wrote combined CSV and plots to {output_dir}")


if __name__ == "__main__":
    main()
