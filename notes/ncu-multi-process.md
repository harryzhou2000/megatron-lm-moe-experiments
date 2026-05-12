# NCU Multi-Process Profiling for Collective Kernels

## Problem

Profiling collective CUDA kernels (e.g., DeepEP's `combine_kernel`, `dispatch_kernel`) that
communicate across multiple GPUs via NVLink is non-trivial because:

1. **Kernel replay fails** with single-rank profiling — ncu tries to save/restore all accessible
   GPU memory (275 GB HBM + IPC-mapped remote memory) which exceeds limits.
2. **Application replay fails** with multi-process — kernel launch counts differ between passes
   due to non-deterministic NCCL barrier timing.
3. **Range replay fails** — NCCL uses `cuThreadExchangeStreamCaptureMode` during the profiled
   range which is unsupported during capture.

## Solution: Collective NCU with TCP Communicator

All ranks run under their own `ncu` instance. The instances coordinate kernel replay via a TCP
rendezvous, ensuring all ranks replay the collective kernel simultaneously (so the NVLink
communication works correctly during replay).

### Key Flags

```
ncu \
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
 ├── ncu --communicator tcp ... python script.py --local-rank 0 --ncu-child
 ├── ncu --communicator tcp ... python script.py --local-rank 1 --ncu-child
 ├── ...
 └── ncu --communicator tcp ... python script.py --local-rank 7 --ncu-child
```

Each rank is a separate OS process. The launcher spawns all 8 `ncu` instances and waits.
ncu's TCP communicator handles the rendezvous — no MPI or torchrun needed.

## DeepEP Integration

### Test Script Usage

```bash
cd /path/to/DeepEP

# Step 1: Normal run to warm JIT cache (creates .so files in ~/.deepep/hybrid_ep/jit/)
NUM_SMS_DISPATCH=32 NUM_SMS_COMBINE=32 HIDDEN_DIM=512 NUM_TOKENS_PER_RANK=8192 \
NUM_LOCAL_EXPERTS=32 TOPK=36 \
NUM_OF_STAGES_G2S_COMBINE_API=64 NUM_OF_STAGES_S2G_COMBINE_API=8 \
NUM_TOKENS_COMBINE_REDUCE_BATCH_COMBINE_API=16 NUM_OF_TOKENS_PER_GROUP_COMBINE_API=2 \
python tests/test_hybrid_ep.py --num-processes 8

# Step 2: NCU profiling (auto-launches all ranks under ncu)
NUM_SMS_DISPATCH=32 NUM_SMS_COMBINE=32 HIDDEN_DIM=512 NUM_TOKENS_PER_RANK=8192 \
NUM_LOCAL_EXPERTS=32 TOPK=36 \
NUM_OF_STAGES_G2S_COMBINE_API=64 NUM_OF_STAGES_S2G_COMBINE_API=8 \
NUM_TOKENS_COMBINE_REDUCE_BATCH_COMBINE_API=16 NUM_OF_TOKENS_PER_GROUP_COMBINE_API=2 \
python tests/test_hybrid_ep.py --num-processes 8 \
  --ncu-profile /path/to/output \
  --ncu-kernel combine_kernel
```

### What `--ncu-profile` Does

1. Launches 8 processes, each under its own `ncu` with `--communicator tcp`
2. Each rank runs with `--local-rank <N>` and `--ncu-child` flags
3. Sets env: `DEEP_EP_NCU_MODE=1`, `WORLD_SIZE=1`, `RANK=0`, `NCCL_TIMEOUT=1800`
4. `DEEP_EP_NCU_MODE=1` triggers:
   - `load_cached_kernels=True` — loads pre-compiled .so files from ANY proc-* directory
     (cross-PID cache) to avoid JIT compilation under ncu (gcc crashes when ncu injects
     into nvcc subprocess)
   - Correctness test followed by cudaProfilerStart/Stop bracketed combine iteration
5. Produces per-rank reports: `<output>_rank0.ncu-rep`, `<output>_rank1.ncu-rep`, etc.

### Custom Metrics

```bash
# Override default metrics with --ncu-metrics
python tests/test_hybrid_ep.py --num-processes 8 \
  --ncu-profile /path/to/output \
  --ncu-kernel dispatch_kernel \
  --ncu-metrics "gpu__time_duration.sum,nvlrx__bytes.sum,nvltx__bytes.sum"
```

### Profile dispatch_kernel Instead

```bash
# Change --ncu-kernel to target dispatch
python tests/test_hybrid_ep.py --num-processes 8 \
  --ncu-profile /path/to/output \
  --ncu-kernel dispatch_kernel
```

Note: the cudaProfilerStart/Stop brackets currently wrap `combine_with_unpermute`.
To profile dispatch, modify the ncu mode block in `test_main()` to bracket
`dispatch_with_permute` instead.

## Extracting Results

```bash
# CSV export (inside container)
ncu --import /path/to/output_rank2.ncu-rep --csv

# Specific metric values
ncu --import /path/to/output_rank2.ncu-rep --csv | grep "Metric Value"
```

## Known Issues

- **1 rank may fail** (~rank 0 or rank 1) with "Failed to save memory for replay" due to
  275GB HBM + IPC memory exceeding ncu's save capacity. The other 7 ranks complete
  successfully. This is a known ncu limitation on large-memory GPUs.
- **13 passes** for the full default metric set (stalls + NVLink + DRAM + L1). Takes ~3-5 min.
- **JIT cache must be warm** before ncu run. If a kernel config is missing from cache,
  nvcc compilation under ncu will crash (ncu injects into all child processes including
  gcc/nvcc).

## Process Group Init Under NCU

`tests/utils.py:init_dist()` treats `WORLD_SIZE` env as `num_nodes` and `RANK` as `node_rank`:
```
world_size = num_nodes * num_local_ranks
global_rank = node_rank * num_local_ranks + local_rank
```

For single-node 8-GPU: set `WORLD_SIZE=1`, `RANK=0`. Each rank passes its own
`local_rank` via `--local-rank`. The total world size is `1 * 8 = 8`.

Do NOT set `WORLD_SIZE=8` — that would create a 64-process group (8 nodes × 8 local).
