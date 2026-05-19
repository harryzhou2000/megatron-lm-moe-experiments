# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Print BW improvements of router variants over a baseline.

Usage (for p3R progressive results):
  python data/print_bw_improvements.py --pattern "router_fix_p3R_*.csv"
  python data/print_bw_improvements.py --pattern "router_fix_p3R_*.csv" --score-function sigmoid
  python data/print_bw_improvements.py --pattern "router_fix_p3R_*.csv" --seq-len 131072
"""

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
        "--pattern",
        type=str,
        default="router_fix_p3R_*.csv",
        help="Glob pattern for CSV files (default: router_fix_p3R_*.csv)",
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
    seq_len = args.seq_len
    score_function = args.score_function
    configs = [tuple(map(int, c.split("/"))) for c in args.configs]

    # Load all matching CSV files, sorted by name (which encodes order via _N_ prefix)
    files = sorted(data_dir.glob(args.pattern))
    if not files:
        print(f"No files matching '{args.pattern}' found in {data_dir}")
        return

    data = {f.stem: pd.read_csv(f) for f in files}
    ordered_names = list(data.keys())

    if len(ordered_names) < 2:
        print("Need at least 2 CSV files (baseline + variant)")
        return

    # First file is baseline
    base_name = ordered_names[0]
    baseline = data[base_name]
    other_names = ordered_names[1:]

    # Create display names from the filename suffix after the commit hash
    # e.g. "router_fix_p3R_1_6d6005a7_fuseloop" -> "+fuseloop"
    display_names = []
    for name in other_names:
        parts = name.split("_")
        # Last part is the human-readable tag
        tag = parts[-1] if len(parts) > 4 else name
        display_names.append(f"+{tag}")

    # Also get a short baseline display name
    base_parts = base_name.split("_")
    base_display = base_parts[-1] if len(base_parts) > 4 else base_name

    # Print header
    header = f"| kernel | pass | config | {base_display} | " + " | ".join(display_names) + " |"
    sep_parts = ["--------|------|--------|----------"] + ["----------"] * len(other_names)
    sep = "|" + "|".join(sep_parts) + "|"

    print(f"## BW Improvements over {base_display} "
          f"(seq_len={seq_len}, score_function={score_function})\n")
    print(header)
    print(sep)

    kernels = ["topk", "aux_loss"]
    test_passes = ["forward", "backward_raw"]

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
