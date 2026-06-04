#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Parse repeated Hybrid-EP benchmark logs into compact tables."""

import argparse
import re
from pathlib import Path


PATTERNS = {
    "dispatch API": r"^dispatch \(BF16(?:, probs=True)?\):.* t: ([0-9.]+) us",
    "combine API": r"^combine \((?:w/ probs|probs=True)\):.* t: ([0-9.]+) us",
    "dispatch no-prob API": r"^dispatch (?:no-prob \(BF16\)|\(BF16, probs=False\)):.* t: ([0-9.]+) us",
    "combine no-prob API": r"^combine \((?:no probs|probs=False)\):.* t: ([0-9.]+) us",
    "dispatch+permute API": r"^dispatch\+permute \(BF16\):.* t: ([0-9.]+) us",
    "combine+unpermute API": r"^combine\+unpermute:.* t: ([0-9.]+) us",
    "fused dispatch+permute API": r"^fused dispatch\+permute \(BF16\):.* t: ([0-9.]+) us",
    "fused combine+unpermute API": r"^fused combine\+unpermute:.* t: ([0-9.]+) us",
    "dispatch kernel": r"^dispatch kernel \(BF16(?:, probs=True)?\).* avg_t=([0-9.]+) us",
    "combine kernel": r"^combine kernel \((?:w/ probs|probs=True)\).* avg_t=([0-9.]+) us",
    "dispatch no-prob kernel": r"^dispatch (?:no-prob kernel \(BF16\)|kernel \(BF16, probs=False\)).* avg_t=([0-9.]+) us",
    "combine no-prob kernel": r"^combine kernel \((?:no probs|probs=False)\).* avg_t=([0-9.]+) us",
    "fused dispatch+permute kernel": r"^fused dispatch\+permute kernel \(BF16\).* avg_t=([0-9.]+) us",
    "fused combine+unpermute kernel": r"^fused combine\+unpermute kernel.* avg_t=([0-9.]+) us",
    "dispatch kernel in dispatch+permute": r"^\s*dispatch_kernel \(in dispatch\+permute\):\s+avg=([0-9.]+) us",
    "permute kernel": r"^\s*permute_kernel:\s+avg=([0-9.]+) us",
    "unpermute kernel": r"^\s*unpermute_kernel:\s+avg=([0-9.]+) us",
    "combine kernel in combine+unpermute": r"^\s*combine_kernel \(in combine\+unpermute\):\s+avg=([0-9.]+) us",
}

PREPROC_PATTERN = re.compile(r"^\s*(scan\+permute-preprocessing [^:]+):\s+avg=([0-9.]+) us")


def summarize(values: list[float]) -> str:
    if not values:
        return "--"
    avg = sum(values) / len(values)
    return f"{avg:.1f} [{min(values):.1f}, {max(values):.1f}]"


def parse_branch(logdir: Path, name: str) -> dict[str, list[float]]:
    results: dict[str, list[float]] = {key: [] for key in PATTERNS}
    for perf_path in sorted(logdir.glob(f"{name}.perf.run*.txt")):
        seen = set()
        for line in perf_path.read_text().splitlines():
            for key, pattern in PATTERNS.items():
                if key in seen:
                    continue
                match = re.search(pattern, line)
                if match:
                    results[key].append(float(match.group(1)))
                    seen.add(key)
    for preproc_path in sorted(logdir.glob(f"{name}.preproc.run*.txt")):
        for line in preproc_path.read_text().splitlines():
            match = PREPROC_PATTERN.search(line)
            if match:
                results.setdefault(match.group(1), []).append(float(match.group(2)))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entries", nargs="+", help="name=/path/to/logdir")
    args = parser.parse_args()

    parsed = {}
    for entry in args.entries:
        name, path = entry.split("=", 1)
        parsed[name] = parse_branch(Path(path), name)

    keys = []
    for branch_results in parsed.values():
        for key, values in branch_results.items():
            if values and key not in keys:
                keys.append(key)

    print("metric," + ",".join(parsed))
    for key in keys:
        print(key + "," + ",".join(summarize(parsed[name].get(key, [])) for name in parsed))


if __name__ == "__main__":
    main()
