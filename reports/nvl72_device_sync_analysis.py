#!/usr/bin/env python3
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
Analysis of device_sync_kernel overhead in NVL72 EP72 Qwen3 training profile.

Profile: mcore-benchmarking-v0.16-dev-Head128-hepS-AG1-noCG-no1f1b-profile-SM32-Comb_36_4_1_18
Config: TP1 PP1 EP72, 72x B300, CUDA graphs OFF, 1F1B overlap OFF

Produces per-rank timing CSVs and a summary report.
"""

import sqlite3
import os
import glob
import csv
import json
import sys

NSYS_DIR = os.path.expanduser(
    "~/recv/output_oci-hsg/output/"
    "mcore-benchmarking-v0.16-dev-Head128-hepS-AG1-noCG-no1f1b-profile-SM32-Comb_36_4_1_18/"
    "Qwen3-Next-80B-A3B_E72-TP1PP1EP72VPP1-MBS2GBS576/nsys"
)
REPORT_DIR = os.path.expanduser("~/projects/moe/reports")

BACKWARD_INDICATORS = frozenset({
    "rmsnorm_bwd_tuned_kernel",
    "rmsnorm_bwd_finalize_tuned_kernel",
    "setGroupedGemmWgradArguments",
})


def find_db_files():
    db_files = {}
    for f in glob.glob(os.path.join(NSYS_DIR, "*.sqlite")):
        parts = os.path.basename(f).split("-rank")
        if len(parts) >= 2:
            rank = int(parts[1].split("-")[0])
            db_files[rank] = f
    return db_files


def load_kernels(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT k.start, k.end, s.value
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.shortName = s.id
        ORDER BY k.start
    """)
    rows = cur.fetchall()
    conn.close()
    return rows


def is_backward(all_kernels, idx, window=50):
    lo = max(0, idx - window)
    hi = min(len(all_kernels), idx + window)
    return bool({all_kernels[k][2] for k in range(lo, hi)} & BACKWARD_INDICATORS)


def med(lst):
    if not lst:
        return 0
    s = sorted(lst)
    return s[len(s) // 2]


def percentile(lst, p):
    if not lst:
        return 0
    s = sorted(lst)
    idx = int(len(s) * p)
    return s[min(idx, len(s) - 1)]


def analyze_rank(rank, all_kernels):
    """Extract per-barrier-segment timing for one rank."""
    # Classify each device_sync
    syncs = []  # (type, start, end, dur_us)
    for i, (s, e, n) in enumerate(all_kernels):
        if n != "device_sync_kernel":
            continue
        dur_us = (e - s) / 1000

        # Find prev/next non-sync kernel
        prev_name = None
        for j in range(i - 1, max(i - 5, 0), -1):
            if all_kernels[j][2] not in (
                "device_sync_kernel",
                "update_expected_value_kernel",
                "pad_tokens_per_expert_kernel",
            ):
                prev_name = all_kernels[j][2]
                break

        next_name = None
        for j in range(i + 1, min(i + 5, len(all_kernels))):
            if all_kernels[j][2] != "device_sync_kernel":
                next_name = all_kernels[j][2]
                break

        bwd = is_backward(all_kernels, i)

        if next_name == "dispatch_kernel" and dur_us > 500:
            syncs.append(("pre_dispatch", s, e, dur_us, bwd))
        elif next_name == "combine_kernel" and dur_us > 500:
            syncs.append(("pre_combine", s, e, dur_us, bwd))
        elif prev_name == "dispatch_kernel" and dur_us < 300:
            syncs.append(("post_dispatch", s, e, dur_us, bwd))
        elif prev_name == "combine_kernel" and dur_us < 1000:
            syncs.append(("post_combine", s, e, dur_us, bwd))

    # Compute inter-barrier segments
    # Segment A: post_dispatch_END -> pre_combine_START (permute + expert + unpermute)
    # Segment B: post_combine_END -> next pre_dispatch_START (inter-layer bwd compute)
    seg_a = []
    seg_b = []

    bwd_syncs = [(t, s, e, d) for t, s, e, d, bwd in syncs if bwd]

    for idx in range(len(bwd_syncs)):
        typ, start, end, dur = bwd_syncs[idx]
        if typ == "post_dispatch":
            # find next pre_combine
            for idx2 in range(idx + 1, len(bwd_syncs)):
                if bwd_syncs[idx2][0] == "pre_combine":
                    gap = (bwd_syncs[idx2][1] - end) / 1000
                    if 0 < gap < 50000:
                        seg_a.append(gap)
                    break
        elif typ == "post_combine":
            for idx2 in range(idx + 1, len(bwd_syncs)):
                if bwd_syncs[idx2][0] == "pre_dispatch":
                    gap = (bwd_syncs[idx2][1] - end) / 1000
                    if 0 < gap < 50000:
                        seg_b.append(gap)
                    break

    # Per-type stats
    stats = {}
    for phase in ("forward", "backward"):
        for stype in ("pre_dispatch", "post_dispatch", "pre_combine", "post_combine"):
            durs = [
                d
                for t, s, e, d, bwd in syncs
                if t == stype and bwd == (phase == "backward") and d < 50000
            ]
            if durs:
                stats[f"{phase}_{stype}"] = {
                    "n": len(durs),
                    "med": med(durs),
                    "avg": sum(durs) / len(durs),
                    "p5": percentile(durs, 0.05),
                    "p95": percentile(durs, 0.95),
                    "max": max(durs),
                }

    stats["seg_a"] = {
        "n": len(seg_a),
        "med": med(seg_a),
        "avg": sum(seg_a) / len(seg_a) if seg_a else 0,
        "p95": percentile(seg_a, 0.95),
        "max": max(seg_a) if seg_a else 0,
    }
    stats["seg_b"] = {
        "n": len(seg_b),
        "med": med(seg_b),
        "avg": sum(seg_b) / len(seg_b) if seg_b else 0,
        "p95": percentile(seg_b, 0.95),
        "max": max(seg_b) if seg_b else 0,
    }

    return stats


def main():
    db_files = find_db_files()
    print(f"Found {len(db_files)} rank sqlite files")

    all_stats = {}
    for rank in sorted(db_files.keys()):
        kernels = load_kernels(db_files[rank])
        all_stats[rank] = analyze_rank(rank, kernels)
        if rank % 10 == 0:
            print(f"  Processed rank {rank}")

    # --- Write CSV: per-rank summary ---
    csv_path = os.path.join(REPORT_DIR, "nvl72_device_sync_per_rank.csv")
    fields = [
        "rank",
        "bwd_pre_dispatch_med",
        "bwd_pre_dispatch_p95",
        "bwd_post_dispatch_med",
        "bwd_pre_combine_med",
        "bwd_pre_combine_p95",
        "bwd_post_combine_med",
        "fwd_pre_dispatch_med",
        "fwd_pre_combine_med",
        "seg_a_med",
        "seg_a_p95",
        "seg_a_max",
        "seg_b_med",
        "seg_b_p95",
        "seg_b_max",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rank in sorted(all_stats.keys()):
            s = all_stats[rank]
            row = {"rank": rank}
            for key in (
                "backward_pre_dispatch",
                "backward_post_dispatch",
                "backward_pre_combine",
                "backward_post_combine",
                "forward_pre_dispatch",
                "forward_pre_combine",
            ):
                prefix = key.replace("backward_", "bwd_").replace("forward_", "fwd_")
                if key in s:
                    row[f"{prefix}_med"] = f"{s[key]['med']:.0f}"
                    if f"{prefix}_p95" in fields:
                        row[f"{prefix}_p95"] = f"{s[key]['p95']:.0f}"
                else:
                    row[f"{prefix}_med"] = ""
                    if f"{prefix}_p95" in fields:
                        row[f"{prefix}_p95"] = ""

            for seg in ("seg_a", "seg_b"):
                if seg in s:
                    row[f"{seg}_med"] = f"{s[seg]['med']:.0f}"
                    row[f"{seg}_p95"] = f"{s[seg]['p95']:.0f}"
                    row[f"{seg}_max"] = f"{s[seg]['max']:.0f}"
            writer.writerow(row)

    print(f"\nWrote {csv_path}")

    # --- Write JSON: full stats ---
    json_path = os.path.join(REPORT_DIR, "nvl72_device_sync_full.json")
    with open(json_path, "w") as f:
        json.dump(all_stats, f, indent=2, default=str)
    print(f"Wrote {json_path}")

    # --- Print summary ---
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for metric, label in [
        ("backward_pre_dispatch", "BWD pre-dispatch sync"),
        ("backward_post_dispatch", "BWD post-dispatch sync"),
        ("backward_pre_combine", "BWD pre-combine sync"),
        ("backward_post_combine", "BWD post-combine sync"),
        ("forward_pre_dispatch", "FWD pre-dispatch sync"),
        ("forward_pre_combine", "FWD pre-combine sync"),
    ]:
        meds_list = [
            all_stats[r][metric]["med"]
            for r in all_stats
            if metric in all_stats[r]
        ]
        if meds_list:
            print(
                f"  {label:30s}: median_of_medians={med(meds_list):8.0f} us, "
                f"spread={max(meds_list)-min(meds_list):.0f} us"
            )

    for seg, label in [
        ("seg_a", "Seg A (postDisp→preComb)"),
        ("seg_b", "Seg B (postComb→preDisp)"),
    ]:
        meds_list = [all_stats[r][seg]["med"] for r in all_stats if seg in all_stats[r]]
        p95s_list = [all_stats[r][seg]["p95"] for r in all_stats if seg in all_stats[r]]
        if meds_list:
            print(
                f"  {label:30s}: median spread={max(meds_list)-min(meds_list):.0f} us, "
                f"p95 spread={max(p95s_list)-min(p95s_list):.0f} us"
            )


if __name__ == "__main__":
    main()
