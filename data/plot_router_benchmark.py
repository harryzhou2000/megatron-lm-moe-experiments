# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Plotting script for router benchmark results.

Generates 4 plots (2 kernels x forward/backward) comparing fused router
implementations against unfused reference. Automatically discovers all
router_fix_p*.csv files in the data directory.

Each plot is faceted by score function and sequence length.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Available colors and markers for automatic assignment
COLORS = [
    "tab:blue",
    "tab:orange",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
]
MARKERS = ["s", "o", "^", "D", "v", "<", ">", "p", "h", "*"]


def discover_csv_files(data_dir: Path, pattern: str = "router_fix_p*.csv") -> list[Path]:
    """Discover CSV files matching the pattern.

    Args:
        data_dir: Directory to search for CSV files.
        pattern: Glob pattern for CSV files.

    Returns:
        Sorted list of matching CSV file paths.
    """
    files = sorted(data_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found in {data_dir}")
    return files


def load_data(csv_files: list[Path]) -> dict[str, pd.DataFrame]:
    """Load benchmark CSV files.

    Args:
        csv_files: List of CSV file paths to load.

    Returns:
        Dictionary mapping filename stem to DataFrame.
    """
    data = {}
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        data[csv_file.stem] = df
    return data


def order_series_names(names: list[str], base_name: str) -> list[str]:
    """Order series names: non-suffixed first, then +suffix names ordered by suffix chain.

    Args:
        names: List of series names.
        base_name: Base name for detecting + suffixes.

    Returns:
        Ordered list of names.
    """
    # Separate into base+suffix names and others
    suffix_names = []
    other_names = []
    for name in names:
        if name.startswith(base_name + "+"):
            suffix_names.append(name)
        else:
            other_names.append(name)

    # Sort suffix names by length (shorter = fewer suffixes = earlier)
    suffix_names.sort(key=len)

    # Others first (sorted), then suffix names
    return sorted(other_names) + suffix_names


def get_display_names(names: list[str], base_name: str) -> dict[str, str]:
    """Create incremental display names for series.

    Args:
        names: List of series names (already ordered).
        base_name: Base name for detecting + suffixes.

    Returns:
        Dictionary mapping original name to display name.
    """
    display_map = {}
    prev_name = base_name

    for name in names:
        if name == base_name:
            display_map[name] = base_name
        elif name.startswith(prev_name + "+"):
            # Extract the incremental part
            incremental = name[len(prev_name):]
            display_map[name] = incremental
            prev_name = name
        elif name.startswith(base_name + "+"):
            # Not directly chained, show relative to base
            display_map[name] = name[len(base_name):]
            prev_name = name
        else:
            # Not a suffix name, keep as is
            display_map[name] = name

    return display_map


def prepare_data(
    data: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Prepare and filter data for plotting.

    Filters out 8-expert case and computes unfused_ref as average of ref_gbps.

    Args:
        data: Dictionary mapping filename stem to DataFrame.

    Returns:
        Tuple of (filtered_data dict, unfused_ref_df).
    """
    filtered_data = {}

    for name, df in data.items():
        # Filter out 8-expert case
        filtered = df[df["num_experts"] != 8].copy()
        # Create x-axis label combining num_experts and topk
        filtered["x_label"] = (
            filtered["num_experts"].astype(str) + "/" + filtered["topk"].astype(str)
        )
        filtered_data[name] = filtered

    # Compute unfused_ref as average of ref_gbps from all CSVs
    combined = pd.concat(filtered_data.values(), ignore_index=True)
    unfused_ref = (
        combined.groupby(
            ["kernel", "num_tokens", "num_experts", "topk", "score_function", "test_pass"]
        )["ref_gbps"]
        .mean()
        .reset_index()
    )
    unfused_ref["x_label"] = (
        unfused_ref["num_experts"].astype(str) + "/" + unfused_ref["topk"].astype(str)
    )

    return filtered_data, unfused_ref


def get_style_cycler(n_series: int) -> list[tuple[str, str]]:
    """Generate color and marker combinations for n series.

    Each series gets a unique color and marker combination, cycling through
    both lists in parallel (not as a cartesian product).

    Args:
        n_series: Number of data series to style.

    Returns:
        List of (color, marker) tuples.
    """
    styles = []
    for i in range(n_series):
        color = COLORS[i % len(COLORS)]
        marker = MARKERS[i % len(MARKERS)]
        styles.append((color, marker))
    return styles


def create_plot(
    kernel: str,
    test_pass: str,
    filtered_data: dict[str, pd.DataFrame],
    unfused_ref: pd.DataFrame,
    output_dir: Path,
    base_name: str = "router_fix_p2",
    score_functions_filter: list[str] | None = None,
) -> None:
    """Create a single faceted plot for a kernel/test_pass combination.

    Args:
        kernel: Kernel name ('topk' or 'aux_loss').
        test_pass: Test pass ('forward' or 'backward_raw').
        filtered_data: Dictionary mapping name to filtered DataFrame.
        unfused_ref: DataFrame with averaged ref_gbps.
        output_dir: Directory to save plots.
        base_name: Base name for shortening legend labels.
        score_functions_filter: List of score functions to include (None = all).
    """
    # Filter data for this kernel and test_pass
    series_data = {}
    for name, df in filtered_data.items():
        filtered = df[(df["kernel"] == kernel) & (df["test_pass"] == test_pass)]
        if not filtered.empty:
            series_data[name] = filtered

    ref_data = unfused_ref[
        (unfused_ref["kernel"] == kernel) & (unfused_ref["test_pass"] == test_pass)
    ]

    if not series_data:
        print(f"No data for kernel={kernel}, test_pass={test_pass}")
        return

    # Order series names: others first, then +suffix names
    ordered_names = order_series_names(list(series_data.keys()), base_name)

    # Get display names (incremental)
    display_names = get_display_names(ordered_names, base_name)
    display_names["unfused_ref"] = "unfused_ref"

    # Get unique score functions and sequence lengths from first available dataset
    first_df = next(iter(series_data.values()))
    score_functions = sorted(first_df["score_function"].unique())
    if score_functions_filter is not None:
        score_functions = [sf for sf in score_functions if sf in score_functions_filter]
    seq_lengths = sorted(first_df["num_tokens"].unique())

    # Create figure with subplots (score_functions x seq_lengths)
    n_rows = len(score_functions)
    n_cols = len(seq_lengths)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))

    # Handle case where there's only one row or column
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Define x-tick order (num_experts/topk combinations)
    x_tick_order = [
        "256/8",
        "256/22",
        "256/36",
        "512/8",
        "512/22",
        "512/36",
        "2304/8",
        "2304/22",
        "2304/36",
    ]

    # Get styles for all series (ordered data files + unfused_ref)
    all_series = ordered_names + ["unfused_ref"]
    styles = get_style_cycler(len(all_series))
    style_map = {name: styles[i] for i, name in enumerate(all_series)}

    for row_idx, score_func in enumerate(score_functions):
        for col_idx, seq_len in enumerate(seq_lengths):
            ax = axes[row_idx, col_idx]

            # Collect all x_labels present across all series
            all_x_labels = set()
            subplot_data = {}
            for name in ordered_names:
                df = series_data[name]
                subset = df[
                    (df["score_function"] == score_func) & (df["num_tokens"] == seq_len)
                ]
                subplot_data[name] = subset
                all_x_labels.update(subset["x_label"].tolist())

            ref_subplot = ref_data[
                (ref_data["score_function"] == score_func) & (ref_data["num_tokens"] == seq_len)
            ]
            all_x_labels.update(ref_subplot["x_label"].tolist())

            # Sort x_labels by predefined order
            x_labels_present = sorted(
                all_x_labels,
                key=lambda x: x_tick_order.index(x) if x in x_tick_order else 999,
            )
            x_positions = {label: i for i, label in enumerate(x_labels_present)}

            # Plot each data series
            def plot_line(data, y_col, label, color, marker):
                if data.empty:
                    return
                # Sort by x_label order
                data_sorted = data.sort_values(
                    "x_label",
                    key=lambda s: s.map(
                        lambda x: x_tick_order.index(x) if x in x_tick_order else 999
                    ),
                )
                x_vals = [x_positions[x] for x in data_sorted["x_label"]]
                y_vals = data_sorted[y_col].values
                ax.plot(
                    x_vals,
                    y_vals,
                    marker=marker,
                    linestyle="-",
                    label=label,
                    color=color,
                    markersize=6,
                    markerfacecolor="none",
                    markeredgewidth=1.5,
                )

            # Plot fused_gbps for each input file (in order)
            for name in ordered_names:
                subset = subplot_data[name]
                color, marker = style_map[name]
                plot_line(subset, "fused_gbps", display_names[name], color, marker)

            # Plot unfused_ref
            color, marker = style_map["unfused_ref"]
            plot_line(ref_subplot, "ref_gbps", display_names["unfused_ref"], color, marker)

            # Set labels and title
            ax.set_title(f"{score_func}, seq_len={seq_len}")
            ax.set_xlabel("num_experts/topk")
            ax.set_ylabel("GB/s BW")
            ax.set_xticks(range(len(x_labels_present)))
            ax.set_xticklabels(x_labels_present, rotation=45, ha="right")
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.3)

    # Set main title
    pass_label = "fprop" if test_pass == "forward" else "bprop"
    fig.suptitle(f"{kernel} kernel - {pass_label}", fontsize=14, fontweight="bold", y=0.995)

    # Add legend to the right of the figure (vertical)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="center right", fontsize=9,
        frameon=True, bbox_to_anchor=(1.0, 0.5)
    )

    plt.tight_layout(rect=[0, 0, 0.88, 0.97])

    # Save figure
    output_path = output_dir / f"{kernel}_{pass_label}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Plot router benchmark results from CSV files."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing CSV files (default: script directory)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="router_fix_*.csv",
        help="Glob pattern for CSV files (default: router_fix_*.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save plots (default: same as data-dir)",
    )
    parser.add_argument(
        "--base-name",
        type=str,
        default="router_fix_p2",
        help="Base name for shortening legend labels (default: router_fix_p2)",
    )
    parser.add_argument(
        "--score-functions",
        type=str,
        nargs="+",
        default=["softmax", "sigmoid"],
        help="Score functions to include (default: softmax sigmoid)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir or Path(__file__).parent
    output_dir = args.output_dir or data_dir

    # Discover and load CSV files
    csv_files = discover_csv_files(data_dir, args.pattern)
    print(f"Found {len(csv_files)} CSV files: {[f.name for f in csv_files]}")

    data = load_data(csv_files)

    # Prepare data
    filtered_data, unfused_ref = prepare_data(data)

    # Create plots for each kernel and test_pass combination
    kernels = ["topk", "aux_loss"]
    test_passes = ["forward", "backward_raw"]

    for kernel in kernels:
        for test_pass in test_passes:
            create_plot(
                kernel, test_pass, filtered_data, unfused_ref, output_dir,
                args.base_name, args.score_functions
            )

    print("Done!")


if __name__ == "__main__":
    main()
