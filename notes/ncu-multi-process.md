# NCU Multi-Process Profiling for Collective Kernels

## Problem

Profiling collective CUDA kernels (e.g., DeepEP's `combine_kernel`, `dispatch_kernel`) that
communicate across multiple GPUs via NVLink is non-trivial because:

1. **Kernel replay fails** — ncu tries to save/restore all accessible GPU memory (275 GB HBM +
   IPC-mapped remote memory) which exceeds limits. Fails with "Failed to save memory for replay".
2. **Range replay fails** — NCCL uses `cuThreadExchangeStreamCaptureMode` during the profiled
   range which is unsupported during capture.

## Solution: Application Replay with TCP Communicator

All ranks run under their own `ncu` instance with `--replay-mode application`. Instead of
saving/restoring GPU memory for kernel replay, ncu re-launches the entire Python process for
each metric collection pass. The TCP communicator coordinates all ncu instances so they start
each pass together, and `--lockstep-kernel-launch` ensures the collective kernel executes on
all ranks simultaneously during every replay pass.

### How Application Replay Works Mechanically

1. ncu launches the Python process and lets it run until `cudaProfilerStart()`
2. The script executes `dispatch_with_permute` → launches `dispatch_kernel`
3. The script calls `cudaProfilerStop()` — ncu collects metrics for pass 1
4. `cudaProfilerStop()` returns, the process continues to completion and exits
5. ncu **re-launches the entire Python process from scratch** for pass 2
6. The whole script runs again: import, init_dist, create buffer, warmup, Start, dispatch, Stop
7. ncu collects a different metric group this time
8. Repeat until all metric groups are collected (~13 passes for the default set)

This is why connect/disconnect messages appear repeatedly — each pass is a fresh process.

### Key Flags

```
ncu \
  --replay-mode application \               # Re-run entire process per metric pass
  --communicator tcp \                      # Use TCP for inter-ncu coordination
  --communicator-tcp-num-peers 8 \          # Total number of ranks
  --communicator-tcp-hostname 127.0.0.1 \   # Rendezvous host (localhost for single-node)
  --lockstep-kernel-launch \                # Synchronize kernel replay across all ranks
  --profile-from-start off \                # Only profile after cudaProfilerStart()
  --kernel-name <regex> \                   # Filter to specific kernel
  --launch-count 1 \                        # Profile 1 kernel launch
  --metrics <metrics> \                     # Comma-separated metric list
  -o <output>_rank<N> -f \                  # Per-rank output file
  python script.py --local-rank <N> ...
```

### Architecture

```
launcher (parent process)
 ├── ncu --replay-mode application --communicator tcp ... python script.py --local-rank 0 --ncu-child
 ├── ncu --replay-mode application --communicator tcp ... python script.py --local-rank 1 --ncu-child
 ├── ...
 └── ncu --replay-mode application --communicator tcp ... python script.py --local-rank 7 --ncu-child
```

Each rank is a separate OS process. The launcher spawns all 8 `ncu` instances and waits.
ncu's TCP communicator handles the rendezvous — no MPI or torchrun needed.

### Critical: No NCCL Collectives in NCU Mode

With `--lockstep-kernel-launch`, ncu synchronizes ALL kernel launches across ranks — not
just the profiled kernel. NCCL collectives (`dist.barrier()`, `dist.all_reduce()`, etc.)
launch internal kernels whose count and order are non-deterministic across ranks. This
causes the lockstep mechanism to deadlock.

Rules for NCU-mode code paths:
- **Skip correctness tests** that use NCCL collectives before the profiled region
- **Replace `dist.barrier()` with `torch.cuda.synchronize()`** around the profiler bracket
- Do NOT call `dist.barrier()` or `dist.destroy_process_group()` — the process will exit
  naturally after `cudaProfilerStop()` returns and the script completes

## DeepEP Integration

### Available Test Scripts

| Script                         | Profiles                  | Status  |
|-------------------------------|---------------------------|---------|
| `tests/test_hybrid_ep.py`     | Non-direct dispatch/combine | Working |
| `tests/test_hybrid_ep_direct.py` | Direct-permute dispatch  | Working |

Both scripts support `--ncu-profile`, `--ncu-kernel`, `--ncu-metrics` arguments.

### Test Script Usage

```bash
cd /path/to/DeepEP

# Step 1: Normal run to warm JIT cache (creates .so files in ~/.deepep/hybrid_ep/jit/)
MASTER_PORT=29570 HIDDEN_DIM=512 NUM_TOKENS_PER_RANK=8192 \
NUM_LOCAL_EXPERTS=32 TOPK=8 NUM_SMS_DISPATCH=24 NUM_SMS_COMBINE=24 \
python tests/test_hybrid_ep.py --num-processes 8

# Step 2: NCU profiling (auto-launches all ranks under ncu)
mkdir -p ncu_results && \
MASTER_PORT=29571 HIDDEN_DIM=512 NUM_TOKENS_PER_RANK=8192 \
NUM_LOCAL_EXPERTS=32 TOPK=8 NUM_SMS_DISPATCH=24 NUM_SMS_COMBINE=24 \
python tests/test_hybrid_ep.py --num-processes 8 \
  --ncu-profile ncu_results/ncu_nondirect_k8 \
  --ncu-kernel dispatch_kernel
```

### Direct-Permute Profiling

```bash
# Step 1: Warm JIT cache
MASTER_PORT=29580 HIDDEN_DIM=512 NUM_TOKENS_PER_RANK=8192 \
NUM_LOCAL_EXPERTS=32 TOPK=8 NUM_SMS_DISPATCH=24 NUM_SMS_COMBINE=24 \
python tests/test_hybrid_ep_direct.py --num-processes 8

# Step 2: NCU profiling
mkdir -p ncu_results && \
MASTER_PORT=29581 HIDDEN_DIM=512 NUM_TOKENS_PER_RANK=8192 \
NUM_LOCAL_EXPERTS=32 TOPK=8 NUM_SMS_DISPATCH=24 NUM_SMS_COMBINE=24 \
python tests/test_hybrid_ep_direct.py --num-processes 8 \
  --ncu-profile ncu_results/ncu_direct_k8
```

### What `--ncu-profile` Does

1. Launches 8 processes, each under its own `ncu` with `--replay-mode application`
   and `--communicator tcp`
2. Each rank runs with `--local-rank <N>` and `--ncu-child` flags
3. Sets env: `DEEP_EP_NCU_MODE=1`, `WORLD_SIZE=1`, `RANK=0`, `NCCL_TIMEOUT=1800`
4. `DEEP_EP_NCU_MODE=1` triggers:
   - `load_cached_kernels=True` — loads pre-compiled .so files from ANY proc-* directory
     (cross-PID cache) to avoid JIT compilation under ncu (gcc crashes when ncu injects
     into nvcc subprocess)
   - Skips correctness test (avoids non-deterministic NCCL collective kernels)
   - Warmup dispatches + `cudaProfilerStart/Stop` bracketed profiled iteration
5. ncu re-launches the process ~13 times (one per metric group)
6. Produces per-rank reports: `<output>_rank0.ncu-rep`, `<output>_rank1.ncu-rep`, etc.

### Custom Metrics

```bash
# Override default metrics with --ncu-metrics
python tests/test_hybrid_ep.py --num-processes 8 \
  --ncu-profile ncu_results/output \
  --ncu-kernel dispatch_kernel \
  --ncu-metrics "gpu__time_duration.sum,nvlrx__bytes.sum,nvltx__bytes.sum"
```

## Extracting Results

```bash
# CSV export (inside container)
ncu --import /path/to/output_rank2.ncu-rep --csv

# Specific metric values
ncu --import /path/to/output_rank2.ncu-rep --csv | grep "Metric Value"
```

## Known Issues

- **~13 passes** for the full default metric set (stalls + NVLink + DRAM + L1 + SM throughput).
  Each pass re-launches the entire Python process. Total time ~5-10 min.
- **JIT cache must be warm** before ncu run. If a kernel config is missing from cache,
  nvcc compilation under ncu will crash (ncu injects into all child processes including
  gcc/nvcc).
- **NCCL heartbeat warnings** may appear during shutdown ("Failed to check the should dump
  flag on TCPStore"). These are harmless — they occur because the process exits before NCCL's
  heartbeat monitor can cleanly disconnect from the TCPStore.

## Process Group Init Under NCU

`tests/utils.py:init_dist()` treats `WORLD_SIZE` env as `num_nodes` and `RANK` as `node_rank`:
```
world_size = num_nodes * num_local_ranks
global_rank = node_rank * num_local_ranks + local_rank
```

For single-node 8-GPU: set `WORLD_SIZE=1`, `RANK=0`. Each rank passes its own
`local_rank` via `--local-rank`. The total world size is `1 * 8 = 8`.

Do NOT set `WORLD_SIZE=8` — that would create a 64-process group (8 nodes × 8 local).
