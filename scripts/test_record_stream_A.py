import torch

# Ensure we are using a GPU
if not torch.cuda.is_available():
    raise RuntimeError("This script requires a CUDA-capable GPU.")


# 1. Setup two concurrent streams
stream_1 = torch.cuda.Stream()
stream_2 = torch.cuda.Stream()
def _get_cycles_per_ms() -> float:
    """Calibrate ``torch.cuda._sleep`` so we can request a wall-clock delay.

    ``_sleep(N)`` spins the GPU for *N* clock cycles.  This helper measures
    how many cycles correspond to one millisecond on the current GPU.
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    torch.cuda._sleep(1_000_000)
    end.record()
    end.synchronize()
    return 1_000_000 / start.elapsed_time(end)


def run_race_condition(fix_issue=False):
    # Use a large enough tensor to ensure the allocator behavior is visible
    size = (1024, 1024, 1024)

    with torch.cuda.stream(stream_1):
        # Create a tensor on stream_1
        tensor_a = torch.full(size, 1.0, device="cuda")

    sleeps = int(_get_cycles_per_ms() * 1000)

    # We want stream_2 to process tensor_a
    with torch.cuda.stream(stream_2):
        # Problem: stream_2 is slower or starts later than stream_1's cleanup.
        # record_stream tells the allocator: "Don't reuse this memory
        # until all work currently queued on THIS stream (stream_2) is done."
        torch.cuda._sleep(sleeps)
        output = tensor_a * 2.0
        if fix_issue:
            tensor_a.record_stream(stream_2)

        # Async operation on stream_2 using tensor_a

    # 2. Trigger the Race:
    # Delete the CPU reference to tensor_a.
    # From the CPU's perspective, tensor_a is no longer needed.
    tensor_a.untyped_storage().resize_(0)

    # 3. Immediately allocate new data on stream_1
    with torch.cuda.stream(stream_1):
        # The allocator sees tensor_a's memory as "free" because stream_1 is finished.
        # It may overwrite that memory with tensor_c.
        tensor_c = torch.full(size, 99.0, device="cuda")

    # Wait for everything to finish
    torch.cuda.synchronize()

    return output


# --- Execution ---

print("Running WITHOUT fix (Expect potential corruption/99.0 values):")
corrupted_out = run_race_condition(fix_issue=False)
print(f"Max value: {corrupted_out.max().item()} (Expected 2.0)")

print("\nRunning WITH fix (record_stream):")
correct_out = run_race_condition(fix_issue=True)
print(f"Max value: {correct_out.max().item()} (Expected 2.0)")
