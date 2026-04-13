# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Print BW improvements of router variants over a baseline."""

import argparse
from pathlib import Path

import pandas as pd


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Print BW improvements over a baseline."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing CSV files (default: script directory)",
    )
    parser.add_argument(
        "--base-name",
        type=str,
        default="router_fix_p2",
        help="Base name for baseline and pattern matching (default: router_fix_p2)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=8192,
        help="Sequence length to filter (default: 8192)",
    )
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        default=["512/22", "2304/36"],
        help="num_experts/topk configs to show (default: 512/22 2304/36)",
    )
    parser.add_argument(
        "--score-function",
        type=str,
        default="softmax",
        help="Score function to filter (default: softmax)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir or Path(__file__).parent
    base_name = args.base_name
    seq_len = args.seq_len
    score_function = args.score_function
    configs = [tuple(map(int, c.split("/"))) for c in args.configs]

    # Load all base_name* files
    files = sorted(data_dir.glob(f"{base_name}*.csv"))
    if not files:
        print(f"No {base_name}*.csv files found in {data_dir}")
        return

    data = {f.stem: pd.read_csv(f) for f in files}

    # Check baseline exists
    if base_name not in data:
        print(f"Baseline {base_name}.csv not found")
        return

    baseline = data[base_name]
    kernels = ["topk", "aux_loss"]
    test_passes = ["forward", "backward_raw"]

    # Get column names (other variants), ordered by suffix chain length
    other_names = [n for n in sorted(data.keys(), key=len) if n != base_name]

    # Create incremental display names
    # e.g., "router_fix_p2+fuseloop+asyncload" -> "+asyncload" (relative to previous)
    display_names = []
    prev_name = base_name
    for name in other_names:
        if name.startswith(prev_name + "+"):
            # Extract the incremental part
            incremental = name[len(prev_name):]
            display_names.append(incremental)
        else:
            # Fall back to full name if not incremental
            display_names.append(name.replace(base_name, "") or name)
        prev_name = name

    # Print header
    header = f"| kernel | pass | config | {base_name} | " + " | ".join(display_names) + " |"
    sep_parts = ["--------|------|--------|---------------"] + ["----------"] * len(other_names)
    sep = "|" + "|".join(sep_parts) + "|"

    print(f"## BW Improvements over {base_name} (seq_len={seq_len}, score_function={score_function})\n")
    print(header)
    print(sep)

    for kernel in kernels:
        for test_pass in test_passes:
            for num_experts, topk in configs:
                pass_label = "fprop" if test_pass == "forward" else "bprop"
                config = f"{num_experts}/{topk}"

                # Get baseline value
                base_mask = (
                    (baseline["kernel"] == kernel)
                    & (baseline["test_pass"] == test_pass)
                    & (baseline["num_tokens"] == seq_len)
                    & (baseline["num_experts"] == num_experts)
                    & (baseline["topk"] == topk)
                    & (baseline["score_function"] == score_function)
                )
                base_val = baseline.loc[base_mask, "fused_gbps"].values
                if len(base_val) == 0:
                    continue
                base_val = base_val[0]

                row = f"| {kernel} | {pass_label} | {config} | {base_val:.1f} |"

                # Get other variant values
                for name in other_names:
                    df = data[name]
                    mask = (
                        (df["kernel"] == kernel)
                        & (df["test_pass"] == test_pass)
                        & (df["num_tokens"] == seq_len)
                        & (df["num_experts"] == num_experts)
                        & (df["topk"] == topk)
                        & (df["score_function"] == score_function)
                    )
                    val = df.loc[mask, "fused_gbps"].values
                    if len(val) > 0:
                        val = val[0]
                        improvement = (val - base_val) / base_val * 100
                        row += f" {val:.1f} ({improvement:+.1f}%) |"
                    else:
                        row += " N/A |"

                print(row)


if __name__ == "__main__":
    main()
