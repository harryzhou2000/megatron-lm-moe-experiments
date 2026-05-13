#!/usr/bin/env python3
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Select a SLURM partition and submit a salloc job on computelab.

Queries sinfo/squeue to find partitions matching a regex, picks the best
node based on GPU availability, and runs salloc.

Modes
-----
- **select** (default): Pick the single best partition/node and submit one salloc.
- **--all**: Submit salloc on every matching partition simultaneously (first wins).

Usage
-----
    # Select best b300 partition, request 8 GPUs:
    python salloc_select.py b300 --gpus 8

    # Submit on ALL matching b300 partitions at once:
    python salloc_select.py b300 --gpus 8 --all

    # Dry-run — show what would be submitted:
    python salloc_select.py b300 --gpus 8 --dry-run

    # Custom time limit:
    python salloc_select.py b300 --gpus 8 --time 2:00:00
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class NodeInfo:
    """Per-node information aggregated from sinfo."""

    name: str
    partition: str
    state: str  # idle, mixed, allocated, reserved, down, drained, etc.
    gpus_total: int
    gpus_used: int

    @property
    def gpus_free(self) -> int:
        return self.gpus_total - self.gpus_used


@dataclass
class JobInfo:
    """A running job from squeue."""

    job_id: str
    partition: str
    node: str
    state: str
    elapsed_sec: int
    time_limit_sec: int
    gpus: int
    user: str


@dataclass
class Candidate:
    """A partition+node candidate for salloc submission."""

    partition: str
    node: str
    gpus_free: int
    gpus_total: int
    state: str
    # Seconds until the longest-running job on this node finishes (estimated).
    # Lower = sooner availability.  None if node is idle with enough GPUs.
    soonest_free_sec: int | None = None
    # For display
    jobs: list[JobInfo] = field(default_factory=list)


# ---------------------------------------------------------------------------
# SLURM queries
# ---------------------------------------------------------------------------

def run_cmd(cmd: list[str], *, host: str | None = None) -> str:
    """Run a command locally or via ssh on *host*."""
    if host:
        # Pass the entire command as a single shell string so SSH doesn't
        # split on special characters (|, ;, etc.) in sinfo --Format.
        remote_cmd = " ".join(shlex.quote(c) for c in cmd)
        cmd = ["ssh", host, remote_cmd]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        print(f"ERROR running: {' '.join(cmd)}", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)
    return result.stdout


def parse_gres_count(gres_str: str) -> int:
    """Extract GPU count from a GRES string like 'gpu:b300:8(S:0-1)'."""
    m = re.search(r"gpu:[^:]*:(\d+)", gres_str)
    return int(m.group(1)) if m else 0


def parse_elapsed(elapsed: str) -> int:
    """Parse SLURM elapsed time (D-HH:MM:SS or HH:MM:SS or MM:SS) to seconds."""
    days = 0
    if "-" in elapsed:
        day_part, elapsed = elapsed.split("-", 1)
        days = int(day_part)
    parts = elapsed.split(":")
    if len(parts) == 3:
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
    elif len(parts) == 2:
        h, m, s = 0, int(parts[0]), int(parts[1])
    else:
        h, m, s = 0, 0, int(parts[0])
    return days * 86400 + h * 3600 + m * 60 + s


def parse_time_limit(tlimit: str) -> int:
    """Parse SLURM time limit (D-HH:MM:SS, HH:MM:SS, or UNLIMITED) to seconds."""
    if tlimit.upper() in ("UNLIMITED", "INVALID"):
        return 7 * 86400  # treat as 7 days
    return parse_elapsed(tlimit)


def query_nodes(pattern: re.Pattern, host: str | None) -> list[NodeInfo]:
    """Query sinfo for per-node data, filtered by partition regex."""
    raw = run_cmd(
        [
            "sinfo", "-N",
            "--Format=NodeHost:|,Partition:|,StateLong:|,Gres:|,GresUsed:|",
            "--noheader",
        ],
        host=host,
    )
    nodes: list[NodeInfo] = []
    for line in raw.strip().splitlines():
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if len(parts) < 5:
            continue
        node_name, partition, state, gres, gres_used = (
            parts[0], parts[1], parts[2], parts[3], parts[4],
        )
        if not pattern.search(partition):
            continue
        gpus_total = parse_gres_count(gres)
        gpus_used = parse_gres_count(gres_used)
        # Normalize state — strip suffixes like *, $, ~, #, @, etc.
        state_clean = re.sub(r"[*$~#@!%+\-]", "", state).lower()
        nodes.append(NodeInfo(
            name=node_name,
            partition=partition,
            state=state_clean,
            gpus_total=gpus_total,
            gpus_used=gpus_used,
        ))
    return nodes


def query_jobs(partitions: set[str], host: str | None) -> list[JobInfo]:
    """Query squeue for running jobs on the given partitions."""
    raw = run_cmd(
        [
            "squeue",
            "--states=RUNNING",
            "--format='%i|%P|%N|%T|%M|%l|%b|%u'",
            "--noheader",
        ],
        host=host,
    )
    jobs: list[JobInfo] = []
    for line in raw.strip().splitlines():
        line = line.strip().strip("'")
        parts = line.split("|")
        if len(parts) < 8:
            continue
        job_id, partition, node, state, elapsed, tlimit, gres_req, user = (
            parts[0].strip(), parts[1].strip(), parts[2].strip(),
            parts[3].strip(), parts[4].strip(), parts[5].strip(),
            parts[6].strip(), parts[7].strip(),
        )
        if partition not in partitions:
            continue
        if state != "RUNNING":
            continue
        gpus = parse_gres_count(gres_req)
        jobs.append(JobInfo(
            job_id=job_id,
            partition=partition,
            node=node,
            state=state,
            elapsed_sec=parse_elapsed(elapsed),
            time_limit_sec=parse_time_limit(tlimit),
            gpus=max(gpus, 0),
            user=user,
        ))
    return jobs


# ---------------------------------------------------------------------------
# Selection logic
# ---------------------------------------------------------------------------

def _fmt_duration(sec: int) -> str:
    """Format seconds as H:MM:SS."""
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h}:{m:02d}:{s:02d}"


def _estimate_wait(
    node: NodeInfo,
    node_job_list: list[JobInfo],
    gpus_needed: int,
) -> int | None:
    """Estimate seconds until *gpus_needed* GPUs are free on *node*.

    Walks through running jobs in order of soonest-finishing.  As each job
    finishes, its GPUs are freed.  Returns the remaining-time of the job
    whose completion pushes free GPUs >= *gpus_needed*, or ``None`` if that
    can never happen (e.g. all jobs report N/A GPUs).

    Jobs with unknown GPU count (N/A → parsed as 0) are skipped in the
    accumulation since we can't credit them.  To compensate, unaccounted
    GPUs (gpus_used − Σ known job GPUs) are attributed to the **longest-
    running** unknown-GPU job so the estimate stays conservative.
    """
    if node.gpus_free >= gpus_needed:
        return 0

    gpus_deficit = gpus_needed - node.gpus_free

    # Split jobs into known-GPU and unknown-GPU buckets
    known_jobs = [(j, j.gpus) for j in node_job_list if j.gpus > 0]
    unknown_jobs = [j for j in node_job_list if j.gpus <= 0]

    # Attribute unaccounted GPUs to unknown jobs.
    # sinfo gpus_used is ground truth; known jobs may not sum to it.
    known_gpu_sum = sum(g for _, g in known_jobs)
    unaccounted = max(0, node.gpus_used - known_gpu_sum)

    # Distribute unaccounted GPUs evenly across unknown jobs (ceiling),
    # or give them all to the one finishing soonest if only one.
    augmented: list[tuple[JobInfo, int]] = list(known_jobs)
    if unknown_jobs and unaccounted > 0:
        per_job = max(1, unaccounted // len(unknown_jobs))
        remainder = unaccounted
        for uj in unknown_jobs:
            assign = min(per_job, remainder)
            if assign > 0:
                augmented.append((uj, assign))
                remainder -= assign
        # Any leftover to the last one
        if remainder > 0 and augmented:
            last_j, last_g = augmented[-1]
            augmented[-1] = (last_j, last_g + remainder)

    if not augmented:
        return None

    # Sort by remaining time ascending (soonest to finish first)
    augmented.sort(key=lambda pair: pair[0].time_limit_sec - pair[0].elapsed_sec)

    freed = 0
    for j, gpu_count in augmented:
        remaining = max(0, j.time_limit_sec - j.elapsed_sec)
        freed += gpu_count
        if freed >= gpus_deficit:
            return remaining

    return None


def select_best(
    nodes: list[NodeInfo],
    jobs: list[JobInfo],
    gpus_needed: int,
) -> Candidate | None:
    """Pick the best partition+node for *gpus_needed* GPUs.

    Priority:
    1. Idle nodes with enough free GPUs (prefer most free GPUs).
    2. Mixed nodes with enough free GPUs (prefer most free GPUs).
    3. Nodes where enough jobs finish soonest to free *gpus_needed* GPUs.
       For each node we walk jobs by ascending remaining time, accumulating
       freed GPUs, and record the wait until the deficit is covered.
    """
    # Build a map: node_name -> list of jobs
    node_jobs: dict[str, list[JobInfo]] = {}
    for j in jobs:
        node_jobs.setdefault(j.node, []).append(j)

    # Deduplicate nodes (same node may appear under multiple partitions in
    # sinfo -N output; we prefer the first partition seen).
    seen_nodes: dict[str, NodeInfo] = {}
    for n in nodes:
        if n.name not in seen_nodes:
            seen_nodes[n.name] = n

    # Tier 1 & 2: nodes with enough free GPUs right now
    ready: list[Candidate] = []
    for n in seen_nodes.values():
        if n.state not in ("idle", "mixed"):
            continue
        if n.gpus_free < gpus_needed:
            continue
        ready.append(Candidate(
            partition=n.partition,
            node=n.name,
            gpus_free=n.gpus_free,
            gpus_total=n.gpus_total,
            state=n.state,
            soonest_free_sec=None,
            jobs=node_jobs.get(n.name, []),
        ))

    if ready:
        # Prefer idle over mixed, then most free GPUs
        ready.sort(key=lambda c: (0 if c.state == "idle" else 1, -c.gpus_free))
        return ready[0]

    # Tier 3: no immediately-available node — estimate wait per node
    upcoming: list[Candidate] = []
    for n in seen_nodes.values():
        if n.state in ("down", "drained", "draining", "reserved", "inval"):
            continue
        node_job_list = node_jobs.get(n.name, [])
        if not node_job_list:
            continue
        wait = _estimate_wait(n, node_job_list, gpus_needed)
        if wait is None:
            continue
        upcoming.append(Candidate(
            partition=n.partition,
            node=n.name,
            gpus_free=n.gpus_free,
            gpus_total=n.gpus_total,
            state=n.state,
            soonest_free_sec=wait,
            jobs=node_job_list,
        ))

    if upcoming:
        # Sort by shortest estimated wait
        upcoming.sort(key=lambda c: c.soonest_free_sec or 0)
        return upcoming[0]

    return None


# ---------------------------------------------------------------------------
# Printing / display
# ---------------------------------------------------------------------------

def print_node_table(
    nodes: list[NodeInfo],
    node_jobs: dict[str, list[JobInfo]],
    gpus_needed: int,
) -> None:
    """Print a human-readable table of matching nodes."""
    # Deduplicate by node name (keep first partition)
    seen: dict[str, NodeInfo] = {}
    for n in nodes:
        if n.name not in seen:
            seen[n.name] = n

    print(
        f"\n{'Node':<25} {'Partition':<55} {'State':<12}"
        f" {'GPUs':>9} {'Avail In':>10} {'Longest':>10}"
    )
    print("-" * 123)
    for n in sorted(seen.values(), key=lambda x: (x.state, x.name)):
        gpu_str = f"{n.gpus_free}/{n.gpus_total}"
        longest = ""
        avail_in = ""
        njobs = node_jobs.get(n.name, [])
        if njobs:
            max_elapsed = max(j.elapsed_sec for j in njobs)
            longest = _fmt_duration(max_elapsed)
        if n.gpus_free >= gpus_needed:
            avail_in = "now"
        elif njobs:
            wait = _estimate_wait(n, njobs, gpus_needed)
            avail_in = _fmt_duration(wait) if wait is not None else "?"
        print(
            f"{n.name:<25} {n.partition:<55} {n.state:<12}"
            f" {gpu_str:>9} {avail_in:>10} {longest:>10}"
        )
    print()


# ---------------------------------------------------------------------------
# salloc command builders
# ---------------------------------------------------------------------------

def build_salloc_cmd(
    partition: str,
    gpus: int,
    time_limit: str,
    extra_args: list[str],
) -> list[str]:
    """Build an salloc command list."""
    cmd = ["salloc", "-p", partition, "--gpus", str(gpus), "-N1", "-t", time_limit]
    cmd.extend(extra_args)
    return cmd


def do_select_mode(
    nodes: list[NodeInfo],
    jobs: list[JobInfo],
    gpus: int,
    time_limit: str,
    extra_args: list[str],
    *,
    dry_run: bool,
    host: str | None,
) -> None:
    """Mode 1: select the single best partition and submit."""
    node_jobs: dict[str, list[JobInfo]] = {}
    for j in jobs:
        node_jobs.setdefault(j.node, []).append(j)

    print_node_table(nodes, node_jobs, gpus)

    candidate = select_best(nodes, jobs, gpus)
    if candidate is None:
        print("ERROR: No suitable node found matching the filter.", file=sys.stderr)
        sys.exit(1)

    reason = ""
    if candidate.state == "idle":
        reason = f"idle, {candidate.gpus_free} free GPUs"
    elif candidate.gpus_free >= gpus:
        reason = f"mixed, {candidate.gpus_free} free GPUs"
    elif candidate.soonest_free_sec is not None:
        reason = f"soonest job finishes in ~{_fmt_duration(candidate.soonest_free_sec)}"

    print(f"==> Selected: {candidate.node} on partition {candidate.partition}")
    print(f"    Reason:   {reason}")
    print(f"    GPUs:     {candidate.gpus_free}/{candidate.gpus_total} free")

    cmd = build_salloc_cmd(candidate.partition, gpus, time_limit, extra_args)
    cmd_str = " ".join(cmd)
    print(f"\n==> Command: {cmd_str}\n")

    if dry_run:
        print("[dry-run] Not submitting.")
        return

    if host:
        # Run salloc via ssh with a PTY so the interactive session works
        os.execvp("ssh", ["ssh", "-t", host, cmd_str])
    else:
        os.execvp(cmd[0], cmd)


def do_all_mode(
    nodes: list[NodeInfo],
    jobs: list[JobInfo],
    gpus: int,
    time_limit: str,
    extra_args: list[str],
    *,
    dry_run: bool,
    host: str | None,
) -> None:
    """Mode 2: submit salloc on ALL matching partitions (first one wins).

    We launch a background salloc for each unique partition. The first to
    allocate gets kept; the rest are cancelled.
    """
    node_jobs: dict[str, list[JobInfo]] = {}
    for j in jobs:
        node_jobs.setdefault(j.node, []).append(j)

    print_node_table(nodes, node_jobs, gpus)

    # Collect unique partitions with at least one usable (non-down) node
    usable_states = {"idle", "mixed", "allocated", "completing"}
    partitions: list[str] = []
    seen_parts: set[str] = set()
    for n in nodes:
        if n.partition not in seen_parts and n.state in usable_states:
            partitions.append(n.partition)
            seen_parts.add(n.partition)

    if not partitions:
        print("ERROR: No usable partitions found.", file=sys.stderr)
        sys.exit(1)

    print(f"==> Submitting salloc on {len(partitions)} partition(s):\n")
    for p in partitions:
        cmd = build_salloc_cmd(p, gpus, time_limit, extra_args)
        print(f"    {' '.join(cmd)}")
    print()

    if dry_run:
        print("[dry-run] Not submitting.")
        return

    # Build a shell script that launches all salloc jobs in parallel.
    # Each salloc runs in background; we wait for the first to succeed,
    # then kill the rest.
    trap_and_wait = _build_race_script(partitions, gpus, time_limit, extra_args)

    if host:
        os.execvp("ssh", ["ssh", "-t", host, "bash", "-c", repr(trap_and_wait)])
    else:
        os.execvp("bash", ["bash", "-c", trap_and_wait])


def _build_race_script(
    partitions: list[str],
    gpus: int,
    time_limit: str,
    extra_args: list[str],
) -> str:
    """Build a bash script that races salloc across partitions."""
    lines = [
        "#!/bin/bash",
        "set -m",  # enable job control
        'PIDS=""',
        'WINNER_PID=""',
        "",
        "cleanup() {",
        '  for pid in $PIDS; do',
        '    if [ "$pid" != "$WINNER_PID" ]; then',
        "      kill $pid 2>/dev/null",
        '      scancel -u $USER --name=__salloc_race_$pid 2>/dev/null',
        "    fi",
        "  done",
        "}",
        "trap cleanup EXIT",
        "",
    ]

    extra = " ".join(extra_args) if extra_args else ""
    for i, part in enumerate(partitions):
        cmd = f"salloc -p '{part}' --gpus {gpus} -N1 -t {time_limit} {extra}".strip()
        lines.append(f"# Partition {i + 1}: {part}")
        lines.append(f"{cmd} &")
        lines.append(f"PIDS=\"$PIDS $!\"")
        lines.append("")

    lines.extend([
        "echo '==> Waiting for first allocation...'",
        "wait -n -p WINNER_PID",
        'echo "==> Allocated via PID $WINNER_PID"',
        "cleanup",
        "wait",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select a SLURM partition and submit a salloc job.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "pattern",
        help="Regex to filter partition names (e.g. 'b300', 'b.00', 'b200@ts4').",
    )
    parser.add_argument(
        "--gpus", "-g",
        type=int,
        default=8,
        help="Number of GPUs to request (default: 8).",
    )
    parser.add_argument(
        "--time", "-t",
        default="4:00:00",
        dest="time_limit",
        help="Time limit for the salloc job (default: 4:00:00).",
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        dest="all_mode",
        help="Submit salloc on ALL matching partitions simultaneously.",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be done without actually submitting.",
    )
    parser.add_argument(
        "--host",
        default=None,
        help=(
            "SSH host to run sinfo/squeue/salloc on (e.g. 'computelab'). "
            "If not set, commands run locally (useful when already on the login node)."
        ),
    )
    parser.add_argument(
        "extra",
        nargs="*",
        help="Extra arguments passed to salloc (e.g. --exclusive).",
    )

    args = parser.parse_args()

    try:
        pattern = re.compile(args.pattern, re.IGNORECASE)
    except re.error as e:
        print(f"ERROR: Invalid regex pattern '{args.pattern}': {e}", file=sys.stderr)
        sys.exit(1)

    # Query cluster state
    print(f"==> Querying cluster for partitions matching /{args.pattern}/i ...")
    all_nodes = query_nodes(pattern, args.host)
    if not all_nodes:
        print(f"ERROR: No partitions matching '{args.pattern}' found.", file=sys.stderr)
        sys.exit(1)

    # Drop unusable nodes early to avoid false selections and noisy output:
    #  - Nodes whose total GPUs < requested (a 1-GPU node can't give 8).
    #  - Nodes in a dead-end state (down, drained, draining, inval).
    _dead_states = {"down", "drained", "draining", "inval"}
    nodes = [
        n for n in all_nodes
        if n.gpus_total >= args.gpus and n.state not in _dead_states
    ]
    if not nodes:
        total_parts = len({n.partition for n in all_nodes})
        print(
            f"ERROR: {len(all_nodes)} node(s) across {total_parts} partition(s) matched "
            f"'{args.pattern}', but none are usable with >= {args.gpus} GPUs.",
            file=sys.stderr,
        )
        sys.exit(1)

    partitions = {n.partition for n in nodes}
    skipped = len(all_nodes) - len(nodes)
    skip_msg = f" (skipped {skipped} unusable)" if skipped else ""
    print(
        f"==> Found {len(nodes)} node entries across {len(partitions)} partition(s).{skip_msg}"
    )

    jobs = query_jobs(partitions, args.host)
    print(f"==> Found {len(jobs)} running jobs on matching partitions.")

    if args.all_mode:
        do_all_mode(
            nodes, jobs, args.gpus, args.time_limit, args.extra,
            dry_run=args.dry_run, host=args.host,
        )
    else:
        do_select_mode(
            nodes, jobs, args.gpus, args.time_limit, args.extra,
            dry_run=args.dry_run, host=args.host,
        )


if __name__ == "__main__":
    main()
