# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
Demonstrates the necessity of ``torch.Tensor.record_stream`` and explicit
event synchronisation when a tensor produced on one CUDA stream is consumed
on another.

Reference
---------
Based on the pattern from PyTorch's own ``test_record_stream`` in
``test/test_cuda.py`` (see ``pytorch/pytorch`` on GitHub).

Strategy
--------
Two explicit (non-default) CUDA streams are used:

*  **producer_stream** — fills ``tmp`` from a pre-populated ``source`` buffer,
   then (after a short sleep) overwrites ``tmp`` with a poison value.
*  **consumer_stream** — reads ``tmp`` into ``result`` after a deliberately
   long ``torch.cuda._sleep`` delay.

After a fork point (the consumer waits for the producer's fill via
``wait_event``), both streams run **concurrently**:

*  consumer_stream: long sleep (200 ms) → ``result.copy_(tmp)``
*  producer_stream: short sleep (10 ms) → ``tmp.copy_(poison)``

Because the producer's sleep is much shorter, the overwrite lands **before**
the consumer reads ``tmp`` — corrupting the result.

Two independent protection mechanisms are tested:

1.  **record_stream** — ``tmp.record_stream(consumer_stream)`` tells the
    caching allocator that ``tmp``'s memory block is in use on
    ``consumer_stream``.  This is an **allocator-level** hint; it does **not**
    add kernel-ordering dependencies.
2.  **event_sync** — ``producer_stream.wait_event(consumer_done)`` before the
    overwrite.  This is a **kernel-ordering** dependency that forces the
    producer to wait for the consumer to finish reading.

This yields **four** test cases per mode:

+------------------+------------+-----------------------------------------------+
| record_stream    | event_sync | Expected result                               |
+==================+============+===============================================+
| No               | No         | Corruption — no protection at all             |
+------------------+------------+-----------------------------------------------+
| Yes              | No         | Corruption — allocator hint alone cannot      |
|                  |            | order kernels                                 |
+------------------+------------+-----------------------------------------------+
| No               | Yes        | Correct — event sync orders the kernels       |
+------------------+------------+-----------------------------------------------+
| Yes              | Yes        | Correct — both protections active             |
+------------------+------------+-----------------------------------------------+

Modes
-----
*  **Eager** (default) — ``enqueue_work`` is called directly, twice.
*  **CUDA graph** (``--cuda-graph``) — ``enqueue_work`` is captured into a
   ``torch.cuda.CUDAGraph`` and replayed twice.

Usage::

    # Eager mode — all 4 cases
    python scripts/test_record_stream.py

    # CUDA-graph mode — all 4 cases
    python scripts/test_record_stream.py --cuda-graph

    # Filter to specific combination
    python scripts/test_record_stream.py --use-record-stream --event-sync
    python scripts/test_record_stream.py --cuda-graph --no-record-stream --no-event-sync
"""

import argparse
import sys
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_cycles_per_ms() -> float:
    """Calibrate ``torch.cuda._sleep`` so we can request a wall-clock delay."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    torch.cuda._sleep(1_000_000)
    end.record()
    end.synchronize()
    return 1_000_000 / start.elapsed_time(end)


# ---------------------------------------------------------------------------
# Core test logic
# ---------------------------------------------------------------------------

def run_test(
    use_record_stream: bool,
    use_event_sync: bool,
    use_cuda_graph: bool,
) -> bool:
    """Run a single test case.

    Parameters
    ----------
    use_record_stream : bool
        Call ``tmp.record_stream(consumer_stream)``.
    use_event_sync : bool
        Insert ``producer_stream.wait_event(consumer_done)`` before the
        producer overwrites ``tmp``.
    use_cuda_graph : bool
        If True, capture ``enqueue_work`` into a CUDA graph and replay it;
        otherwise call it directly.

    Returns
    -------
    data_correct : bool
    """
    cycles_per_ms = _get_cycles_per_ms()

    numel = (1 << 20) * 1024
    host_tensor = torch.arange(numel, dtype=torch.float32).pin_memory()

    producer_stream = torch.cuda.Stream()
    consumer_stream = torch.cuda.Stream()

    # --- Pre-allocate all static tensors ----------------------------------
    with torch.cuda.stream(producer_stream):
        source = torch.empty(numel, device="cuda", dtype=torch.float32)

        poison = torch.full((numel,), -1.0, device="cuda", dtype=torch.float32)

    with torch.cuda.stream(consumer_stream):
        result = torch.empty(numel, device="cuda", dtype=torch.float32)

    torch.cuda.synchronize()

    # Populate source with the host data once (constant across iterations).
    source.copy_(host_tensor.cuda())

    # --- Multi-stream work (shared by eager and graph paths) --------------
    def enqueue_work():
        """Enqueue the full producer → consumer → overwrite pipeline.

        After the fork:
        *  consumer_stream sleeps 200 ms then reads tmp into result.
        *  producer_stream sleeps  10 ms then overwrites tmp with poison.
        The producer finishes first → overwrites tmp before consumer reads it.

        When *use_event_sync* is True the producer waits for the consumer
        to finish reading before overwriting, preventing corruption.
        """
        # 1. Producer fills tmp from source.
        with torch.cuda.stream(producer_stream):
            tmp = torch.empty(numel, device="cuda", dtype=torch.float32)
            tmp.copy_(source)
            prod_done = producer_stream.record_event()

        # 2. Fork: consumer waits for producer's fill to complete.
        consumer_stream.wait_event(prod_done)

        # record_stream: tell the allocator that tmp is also in use on
        # consumer_stream.  Placed here — after tmp is produced and before
        # the consumer reads it — matching real-world usage.

        # 3. Consumer: long sleep then read.
        with torch.cuda.stream(consumer_stream):
            torch.cuda._sleep(int(200 * cycles_per_ms))
            result.copy_(tmp)
            consumer_done = consumer_stream.record_event()

        if use_record_stream:
            tmp.record_stream(consumer_stream)
        tmp.untyped_storage().resize_(0)

        # 4. Producer overwrites tmp with poison.
        with torch.cuda.stream(producer_stream):
            if use_event_sync:
                producer_stream.wait_event(consumer_done)
            else:
                torch.cuda._sleep(int(10 * cycles_per_ms))
            tmp = torch.empty(numel, device="cuda", dtype=torch.float32)
            tmp.copy_(poison)

        # 5. Join consumer back to producer (required for graph capture;
        #    harmless in eager mode).
        producer_stream.wait_stream(consumer_stream)

    # --- Execute ----------------------------------------------------------
    if use_cuda_graph:
        # Warm-up (required before capture).
        enqueue_work()
        torch.cuda.synchronize()

        # Capture.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(producer_stream):
            graph.capture_begin()
        enqueue_work()
        with torch.cuda.stream(producer_stream):
            graph.capture_end()
        torch.cuda.synchronize()

        # Replay.
        data_correct = True
        for _i in range(2):
            graph.replay()
            torch.cuda.synchronize()
            if not torch.equal(result, source):
                data_correct = False
    else:
        # Eager: run directly.
        data_correct = True
        for _i in range(2):
            enqueue_work()
            torch.cuda.synchronize()
            if not torch.equal(result, source):
                data_correct = False

    return data_correct


# ---------------------------------------------------------------------------
# Test matrix
# ---------------------------------------------------------------------------

_CASES = [
    # (record_stream, event_sync, label, expect_correct)
    (False, False, "no record_stream, no event_sync", False),
    (True,  False, "record_stream,    no event_sync", False),
    (False, True,  "no record_stream, event_sync",    True),
    (True,  True,  "record_stream,    event_sync",    True),
]


def run_all_cases(
    use_cuda_graph: bool,
    record_stream_filter: Optional[bool] = None,
    event_sync_filter: Optional[bool] = None,
) -> bool:
    """Run up to 4 cases.  Returns True if all behave as expected."""
    all_ok = True

    for use_rs, use_es, label, expect_correct in _CASES:
        if record_stream_filter is not None and use_rs != record_stream_filter:
            continue
        if event_sync_filter is not None and use_es != event_sync_filter:
            continue

        print(f"  Case: {label}")
        data_correct = run_test(
            use_record_stream=use_rs,
            use_event_sync=use_es,
            use_cuda_graph=use_cuda_graph,
        )
        status = "correct" if data_correct else "CORRUPTED"
        expected = "correct" if expect_correct else "CORRUPTED"
        match = (data_correct == expect_correct)
        verdict = "PASS" if match else "FAIL"

        print(f"    Data:     {status}")
        print(f"    Expected: {expected}")
        print(f"    Verdict:  {verdict}")

        if not match:
            all_ok = False

    return all_ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demonstrate record_stream and event_sync for cross-stream tensor safety.",
    )
    rs_group = parser.add_mutually_exclusive_group()
    rs_group.add_argument(
        "--use-record-stream",
        action="store_true",
        default=False,
        help="Filter: only cases with record_stream enabled.",
    )
    rs_group.add_argument(
        "--no-record-stream",
        action="store_true",
        default=False,
        help="Filter: only cases with record_stream disabled.",
    )
    es_group = parser.add_mutually_exclusive_group()
    es_group.add_argument(
        "--event-sync",
        action="store_true",
        default=False,
        help="Filter: only cases with event sync enabled.",
    )
    es_group.add_argument(
        "--no-event-sync",
        action="store_true",
        default=False,
        help="Filter: only cases with event sync disabled.",
    )
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Capture multi-stream work in a CUDA graph and replay it.",
    )
    return parser.parse_args()


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA is not available — nothing to test.")
        sys.exit(1)

    # Warm up the CUDA context.
    torch.cuda.synchronize()
    _ = torch.empty(1, device="cuda")

    args = _parse_args()

    rs_filter: Optional[bool] = None
    if args.use_record_stream:
        rs_filter = True
    elif args.no_record_stream:
        rs_filter = False

    es_filter: Optional[bool] = None
    if args.event_sync:
        es_filter = True
    elif args.no_event_sync:
        es_filter = False

    mode = "CUDA GRAPH" if args.cuda_graph else "EAGER"
    print("=" * 70)
    print(f"{mode} MODE — record_stream x event_sync matrix")
    print("=" * 70)

    ok = run_all_cases(
        use_cuda_graph=args.cuda_graph,
        record_stream_filter=rs_filter,
        event_sync_filter=es_filter,
    )

    if ok:
        print("  All cases behaved as expected.\n")
    else:
        print("  Some cases did NOT behave as expected!\n")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
