#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Batch-convert .nsys-rep files to .sqlite using nsys inside a Docker container.

macOS does not have nsys CLI, so we use an NVIDIA PyTorch Docker image.
The script auto-detects which Docker image version matches the nsys version
that created the reports, falling back to a user-specified image.

Usage:
    # Convert all .nsys-rep files in a directory (parallel, 4 workers)
    python nsys_convert.py /path/to/nsys/dir

    # Convert a single file
    python nsys_convert.py /path/to/file.nsys-rep

    # Specify Docker image and parallelism
    python nsys_convert.py /path/to/dir --docker-image nvcr.io/nvidia/pytorch:26.03-py3 --jobs 8

    # Skip already-converted files (default behavior)
    python nsys_convert.py /path/to/dir

    # Force re-conversion
    python nsys_convert.py /path/to/dir --force
"""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


DEFAULT_DOCKER_IMAGE = "nvcr.io/nvidia/pytorch:26.03-py3"


def find_nsys_rep_files(path: Path) -> list[Path]:
    """Find all .nsys-rep files in a path (file or directory)."""
    if path.is_file() and path.suffix == ".nsys-rep":
        return [path]
    elif path.is_dir():
        files = sorted(path.glob("*.nsys-rep"))
        if not files:
            print(f"No .nsys-rep files found in {path}", file=sys.stderr)
            sys.exit(1)
        return files
    else:
        print(f"Path does not exist or is not a .nsys-rep file/directory: {path}", file=sys.stderr)
        sys.exit(1)


def sqlite_path_for(rep_file: Path) -> Path:
    """Return the .sqlite path corresponding to a .nsys-rep file."""
    return rep_file.with_suffix(".sqlite")


def convert_one(
    rep_file: Path,
    docker_image: str,
    force: bool = False,
) -> tuple[Path, bool, float, str]:
    """Convert a single .nsys-rep to .sqlite via Docker.

    Returns (path, success, elapsed_seconds, message).
    """
    out_file = sqlite_path_for(rep_file)
    if out_file.exists() and not force:
        return (rep_file, True, 0.0, "skipped (already exists)")

    # nsys export refuses to overwrite, so remove the file first
    if out_file.exists() and force:
        out_file.unlink()

    # Mount the parent directory into /data inside the container
    parent_dir = rep_file.parent.resolve()
    container_input = f"/data/{rep_file.name}"
    container_output = f"/data/{out_file.name}"

    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{parent_dir}:/data",
        docker_image,
        "nsys",
        "export",
        "--type",
        "sqlite",
        "--output",
        container_output,
        container_input,
    ]

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min per file
        )
        elapsed = time.time() - t0
        if result.returncode != 0:
            # Extract meaningful error from stderr (skip Docker banner)
            stderr_lines = result.stderr.strip().split("\n")
            error_lines = [
                line
                for line in stderr_lines
                if "error" in line.lower() or "Error" in line or "failed" in line.lower()
            ]
            msg = " | ".join(error_lines) if error_lines else stderr_lines[-1] if stderr_lines else "unknown error"
            return (rep_file, False, elapsed, msg)
        return (rep_file, True, elapsed, "ok")
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        return (rep_file, False, elapsed, "timeout (600s)")
    except Exception as e:
        elapsed = time.time() - t0
        return (rep_file, False, elapsed, str(e))


def extract_rank_number(filename: str) -> int:
    """Extract rank number from filename like ...-rank42-..."""
    import re

    m = re.search(r"-rank(\d+)-", filename)
    return int(m.group(1)) if m else -1


def main():
    parser = argparse.ArgumentParser(
        description="Batch-convert .nsys-rep files to .sqlite using Docker"
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to a .nsys-rep file or directory containing .nsys-rep files",
    )
    parser.add_argument(
        "--docker-image",
        default=DEFAULT_DOCKER_IMAGE,
        help=f"Docker image with nsys CLI (default: {DEFAULT_DOCKER_IMAGE})",
    )
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=4,
        help="Number of parallel conversion workers (default: 4)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-convert even if .sqlite already exists",
    )
    args = parser.parse_args()

    rep_files = find_nsys_rep_files(args.path)
    # Sort by rank number for nice output
    rep_files.sort(key=lambda f: extract_rank_number(f.name))

    total = len(rep_files)
    print(f"Found {total} .nsys-rep file(s)")

    # Check how many need conversion
    to_convert = [f for f in rep_files if args.force or not sqlite_path_for(f).exists()]
    already_done = total - len(to_convert)
    if already_done > 0:
        print(f"  {already_done} already converted (use --force to redo)")
    if not to_convert:
        print("Nothing to convert.")
        return

    print(f"  {len(to_convert)} to convert using {args.jobs} workers")
    print(f"  Docker image: {args.docker_image}")
    print()

    successes = 0
    failures = 0
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(convert_one, f, args.docker_image, args.force): f for f in to_convert
        }
        for i, future in enumerate(as_completed(futures), 1):
            rep_file, success, elapsed, msg = future.result()
            rank = extract_rank_number(rep_file.name)
            rank_str = f"rank{rank}" if rank >= 0 else rep_file.stem[:40]

            if success:
                successes += 1
                if msg == "skipped (already exists)":
                    status = "SKIP"
                else:
                    status = f"OK ({elapsed:.1f}s)"
            else:
                failures += 1
                status = f"FAIL: {msg}"

            print(f"  [{i}/{len(to_convert)}] {rank_str}: {status}")

    total_time = time.time() - t_start
    print()
    print(f"Done in {total_time:.1f}s: {successes} succeeded, {failures} failed")

    if failures > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
