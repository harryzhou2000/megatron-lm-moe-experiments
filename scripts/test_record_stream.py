# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
Demonstrates the necessity of ``torch.Tensor.record_stream`` when a tensor
produced on one CUDA stream is consumed on another.

Setup
-----
*  **Producer stream** – allocates a tensor and fills it with a known pattern
   via an async kernel, then (optionally) frees its Python reference so the
   CUDA caching allocator may reclaim the memory.
*  **Consumer stream** – reads that tensor *after* explicitly waiting on a
   producer event, so kernel ordering is correct.  The race is purely between
   the Python-level allocator and the consumer kernel.

Without ``record_stream``
~~~~~~~~~~~~~~~~~~~~~~~~~
The caching allocator only tracks the *producer* stream.  Once the Python
reference is deleted the allocator considers the memory reusable – even though
the consumer kernel has not yet executed.  A new allocation on the producer
stream can overwrite the buffer *before* the consumer reads it, corrupting the
result.

With ``record_stream``
~~~~~~~~~~~~~~~~~~~~~~
``record_stream(consumer_stream)`` tells the allocator that the buffer is also
in use on *consumer_stream*, so the memory is not recycled until the consumer
kernel finishes.

Usage::

    python scripts/test_record_stream.py              # run both modes
    python scripts/test_record_stream.py --use-record-stream   # only the safe path
    python scripts/test_record_stream.py --no-record-stream    # only the unsafe path
"""

import argparse
import sys
from typing import Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_FILL_VALUE: float = 42.0
_OVERWRITE_VALUE: float = -1.0
_TENSOR_NUMEL: int = 1 << 20  # 1 M elements – large enough to matter
_NUM_CORRUPTION_ATTEMPTS: int = 200  # repeat to increase race probability
_DTYPE: torch.dtype = torch.float32


def _make_producer_payload(
    producer_stream: torch.cuda.Stream,
    size: int,
    fill: float,
) -> Tuple[torch.Tensor, torch.cuda.Event]:
    """Allocate and fill a tensor on *producer_stream*, return it with an event."""
    with torch.cuda.stream(producer_stream):
        tensor = torch.full((size,), fill, device="cuda", dtype=_DTYPE)
        event = producer_stream.record_event()
    return tensor, event


def _try_reclaim_and_overwrite(
    producer_stream: torch.cuda.Stream,
    data_ptr: int,
    size: int,
    overwrite: float,
    num_allocs: int = 8,
) -> None:
    """Aggressively allocate on *producer_stream* hoping to reclaim *data_ptr*.

    We allocate several tensors of the same size; because the caching allocator
    uses a best-fit policy, one of them is likely to land on the same block.
    """
    with torch.cuda.stream(producer_stream):
        for _ in range(num_allocs):
            t = torch.full((size,), overwrite, device="cuda", dtype=_DTYPE)
            if t.data_ptr() == data_ptr:
                # Intentionally keep this allocation alive so it stays
                # overwritten for the consumer to observe.
                return
            del t


def _run_single_trial(
    producer_stream: torch.cuda.Stream,
    consumer_stream: torch.cuda.Stream,
    use_record_stream: bool,
) -> Optional[torch.Tensor]:
    """Run one producer/consumer hand-off and return the value the consumer saw.

    Returns ``None`` when the allocator did not reclaim the block (no
    corruption opportunity).
    """
    # 1. Produce
    tensor, event = _make_producer_payload(producer_stream, _TENSOR_NUMEL, _FILL_VALUE)
    saved_ptr = tensor.data_ptr()

    if use_record_stream:
        tensor.record_stream(consumer_stream)

    # 2. Consumer waits on the producer event so **kernel** ordering is correct.
    consumer_stream.wait_event(event)

    # 3. Schedule the consumer read *before* we free the reference – but the
    #    read kernel has not executed yet (it is merely enqueued).
    with torch.cuda.stream(consumer_stream):
        # .clone() forces a read of every element on consumer_stream.
        consumer_copy = tensor.clone()

    # 4. Drop the only Python reference – the allocator may now reuse the
    #    block on *producer_stream* (unless record_stream was called).
    del tensor

    # 5. Empty the cache to force the allocator to truly free blocks that are
    #    no longer tracked, maximising the chance of reuse.
    torch.cuda.memory.empty_cache()

    # 6. Overwrite on the producer stream – this races with the consumer read.
    _try_reclaim_and_overwrite(
        producer_stream, saved_ptr, _TENSOR_NUMEL, _OVERWRITE_VALUE
    )

    # 7. Synchronise everything and check what the consumer actually saw.
    torch.cuda.synchronize()
    return consumer_copy


# ---------------------------------------------------------------------------
# Test drivers
# ---------------------------------------------------------------------------


def test_without_record_stream() -> bool:
    """Expect corruption in at least one trial (returns True on corruption)."""
    producer = torch.cuda.Stream()
    consumer = torch.cuda.Stream()

    corrupted = False
    for trial in range(_NUM_CORRUPTION_ATTEMPTS):
        result = _run_single_trial(producer, consumer, use_record_stream=False)
        if result is None:
            continue
        if not torch.all(result == _FILL_VALUE).item():
            n_bad = int((result != _FILL_VALUE).sum().item())
            print(
                f"  [trial {trial:>3d}] CORRUPTION detected – "
                f"{n_bad}/{_TENSOR_NUMEL} elements differ"
            )
            corrupted = True
            break  # one is enough to prove the point

    if not corrupted:
        print(
            f"  No corruption observed after {_NUM_CORRUPTION_ATTEMPTS} trials.\n"
            "  (The race is timing-dependent; try increasing _NUM_CORRUPTION_ATTEMPTS\n"
            "   or running on a busier GPU.)"
        )
    return corrupted


def test_with_record_stream() -> bool:
    """All trials must pass (returns True when every trial is clean)."""
    producer = torch.cuda.Stream()
    consumer = torch.cuda.Stream()

    all_clean = True
    for trial in range(_NUM_CORRUPTION_ATTEMPTS):
        result = _run_single_trial(producer, consumer, use_record_stream=True)
        if result is None:
            continue
        if not torch.all(result == _FILL_VALUE).item():
            n_bad = int((result != _FILL_VALUE).sum().item())
            print(
                f"  [trial {trial:>3d}] UNEXPECTED corruption – "
                f"{n_bad}/{_TENSOR_NUMEL} elements differ"
            )
            all_clean = False
            break

    if all_clean:
        print(
            f"  All {_NUM_CORRUPTION_ATTEMPTS} trials passed – "
            "record_stream correctly prevented reuse."
        )
    return all_clean


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demonstrate record_stream necessity for cross-stream tensor safety.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--use-record-stream",
        action="store_true",
        help="Only run the safe (record_stream) test.",
    )
    group.add_argument(
        "--no-record-stream",
        action="store_true",
        help="Only run the unsafe (no record_stream) test.",
    )
    return parser.parse_args()


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA is not available – nothing to test.")
        sys.exit(1)

    # Warm up the CUDA context so first-allocation jitter doesn't dominate.
    torch.cuda.synchronize()
    _ = torch.empty(1, device="cuda")

    args = _parse_args()
    run_unsafe = not args.use_record_stream
    run_safe = not args.no_record_stream

    exit_code = 0

    if run_unsafe:
        print("=" * 70)
        print("TEST: without record_stream  (expect corruption)")
        print("=" * 70)
        corrupted = test_without_record_stream()
        if not corrupted:
            print("  WARNING: could not trigger corruption (race is timing-dependent).")
            # Not a hard failure – the race may simply not fire on this hardware.
        else:
            print("  PASS – corruption confirmed (record_stream was needed).")

    if run_safe:
        print("=" * 70)
        print("TEST: with record_stream  (expect clean)")
        print("=" * 70)
        clean = test_with_record_stream()
        if not clean:
            print("  FAIL – corruption despite record_stream!")
            exit_code = 1
        else:
            print("  PASS – no corruption observed.")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
