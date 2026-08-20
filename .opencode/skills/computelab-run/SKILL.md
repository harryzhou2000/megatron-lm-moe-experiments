---
name: computelab-run
description: Run commands on remote GPU compute nodes via the computelab SLURM cluster. Use when the user wants to run tests, benchmarks, or any command on the remote GPU machine, or when they say "run on compute", "run remotely", "test on GPU", "run on computelab", or similar phrases.
---

# Computelab Run

Run commands on remote GPU compute nodes through a 3-hop SSH chain: local Mac → computelab (SLURM login) → compute node → enroot container with GPU access.

## Environment

### Access Chain

```
local (Mac)  ──ssh──►  computelab (SLURM login)  ──ssh──►  compute node (GPUs)
                                                            └── enroot container (PyTorch + venv)
```

### Hardware

- **Compute node**: 8× NVIDIA B300 SXM6 AC (275 GB HBM each), NVLink interconnect
- **CUDA**: 13.1, Driver 590.48.01
- **Container**: NVIDIA PyTorch 26.06, Python 3.12
- **Enroot container name**: `test_container_2606`
- **Python environment**: `test_container_2606` uses system Python/pip; older
  containers may provide `/workspace/venv/bin/activate`
- **Dependency versions**: verify inside the selected container before testing; the
  experimental container may be overwritten.

### Key Paths

| Location | Local (Mac) | Remote (computelab + compute node) |
|---|---|---|
| Project root | `~/projects/moe/` | `~/projects/moe/` (computelab home) |
| DeepEP | `~/projects/moe/DeepEP/` | `/home/scratch.hhanyu_gpu/projects/moe/DeepEP/` (inside container) |
| TE | `~/projects/moe/TE/` | `/home/scratch.hhanyu_gpu/projects/moe/TE/` |
| MLM | `~/projects/moe/MLM/` | `/home/scratch.hhanyu_gpu/projects/moe/MLM/` |
| Scripts | `~/projects/moe/scripts/` | `~/projects/moe/scripts/` |
| Helper script | `~/projects/moe/scripts/run_on_compute.sh` | `~/projects/moe/scripts/run_on_compute.sh` |
| Container launcher | — | `~/scratch/enroot_test1.sh` (on computelab) |
| DeepEP JIT cache | — | `~/.deepep/hybrid_ep/jit/` (inside container) |
| Rsync exclude | `~/.rsync-exclude` | — |

### Compiler Cache

Use the persistent user cache at `~/scratch/.ccache`. The login shell and
`scripts/run_on_compute.sh` configure:

```bash
export CCACHE_DIR="$HOME/scratch/.ccache"
export CCACHE_BASEDIR=/home/scratch.hhanyu_gpu/projects/moe
export CCACHE_NOHASHDIR=1
export CCACHE_COMPILERCHECK=content
```

`CCACHE_BASEDIR` and `CCACHE_NOHASHDIR` let identical sources in worktrees below
the project root share entries. `CCACHE_COMPILERCHECK=content` also permits reuse
when an otherwise identical compiler binary has a different timestamp. A changed
shared header, compiler flag, CUDA architecture list, or preprocessor definition
still produces a legitimate miss. Keep complete build logs for later inspection.

## Workflow

### Step 1: Discover or Request an Allocation

```bash
ssh computelab "squeue -u \$USER -h -o '%i %N %t %j'"
```

If a suitable allocation is already running, use its assigned node. Do not request a
second allocation.

Before requesting a new allocation, inspect Blackwell nodes and their partitions, then
choose one concrete partition. Select the **partition**, not a node; let SLURM place the
job on a node in that partition.

```bash
ssh computelab "sinfo -N -h -o '%N %P %T %G' | rg -i 'b200|b300|blackwell'"
ssh computelab "squeue -u \$USER -h -o '%i %P %N %t %j'"
```

Submit exactly one four-hour batch allocation holder whose only payload is
`sleep infinity`:

```bash
ssh computelab "sbatch --parsable \
    -p <selected-blackwell-partition> \
    --gpus 8 -N1 -t 4:00:00 -J codex-moe-test \
    --wrap='sleep infinity'"
```

Do not use `scripts/salloc_select.py`, do not use interactive `salloc`, do not race
partitions, and do not submit duplicate allocation requests. Record the returned job ID
and poll that single job until it is running:

```bash
ssh computelab "squeue -j <job-id> -h -o '%i %N %t %j'"
```

If it remains pending longer than expected, keep waiting and report its pending reason;
do not submit another job. If it enters a terminal failure state, report the failure and
request direction before submitting a replacement.

After it starts, read the node that SLURM assigned to the partition-scoped job. Use that
resolved node only for workload execution (never as an allocation constraint), then release
only that agent-owned allocation holder:

```bash
ssh computelab "bash ~/projects/moe/scripts/run_on_compute.sh \
    -n <node> -c test_container_2606 '<command>'"
ssh computelab "scancel <job-id>"
```

Do not cancel allocations that were not created for the current task.

### Step 2: Sync Code (if needed)

Always sync before running if local code has changed. Use per-submodule rsync with the exclude file:

```bash
# Sync DeepEP (trailing slashes are IMPORTANT)
rsync ~/projects/moe/DeepEP/ computelab:~/projects/moe/DeepEP/ -auPv \
    --exclude-from=$(realpath ~/.rsync-exclude)

# Sync TE
rsync ~/projects/moe/TE/ computelab:~/projects/moe/TE/ -auPv \
    --exclude-from=$(realpath ~/.rsync-exclude)

# Sync MLM
rsync ~/projects/moe/MLM/ computelab:~/projects/moe/MLM/ -auPv \
    --exclude-from=$(realpath ~/.rsync-exclude)

# Sync scripts
rsync ~/projects/moe/scripts/ computelab:~/projects/moe/scripts/ -auPv \
    --exclude-from=$(realpath ~/.rsync-exclude)
```

After syncing, optionally verify: `ssh computelab "cd ~/projects/moe/DeepEP && git diff --stat HEAD"`

### Step 3: Run Commands on the Compute Node

**Option A — Helper script (recommended for single commands):**

```bash
ssh computelab "bash ~/projects/moe/scripts/run_on_compute.sh \
    -n <allocated-node> -c test_container_2606 '<command>'"
```

The helper script automatically:
1. Uses the node assigned to the recorded SLURM job (pass it with `-n`)
2. SSHs to it
3. Launches the enroot container with mounts
4. Activates `/workspace/venv` when the selected container provides it
5. Runs the command

**Option B — Manual 3-hop (for interactive sessions or debugging):**

```bash
# From local:
ssh computelab
# On computelab:
squeue -u $USER -h -o "%N"    # e.g. umb-b300-dp-184
ssh umb-b300-dp-184
# On compute node:
bash ~/scratch/enroot_test1.sh
# Inside container:
source /workspace/venv/bin/activate
cd /home/scratch.hhanyu_gpu/projects/moe/DeepEP
```

### Step 4: Check GPU Status

```bash
ssh computelab "ssh <allocated-node> 'nvidia-smi'"
```

## Common Tasks

### Rebuild Transformer Engine after code changes

The experimental `test_container_2606` TE installation may be overwritten. Build
both Blackwell targets used by the current B200/B300 allocations and retain complete
unfiltered logs:

```bash
ssh computelab "bash ~/projects/moe/scripts/run_on_compute.sh \
    -n <allocated-node> -c test_container_2606 \
    'cd /home/scratch.hhanyu_gpu/projects/moe/TE && \
     export PATH=/home/hhanyu/.pixi.x86_64/bin:\$PATH && \
     NVTE_BUILD_THREADS_PER_JOB=4 \
     NVTE_CUDA_ARCHS=\"100;103a;\" \
     NVTE_USE_CCACHE=1 \
     /usr/bin/python3 -m pip install --no-build-isolation -e \".[test]\" --verbose \
     > logs/te_build_\$(date +%Y%m%d_%H%M%S).log 2>&1'"
```

Use `/usr/bin/python3 -m pip` for the 26.06 container so Pixi's `PATH` addition for
ccache does not accidentally select a different Python environment. Do not pipe
build or test output through `tail`, `grep`, or other online filters.
Inspect the persistent log only after the command finishes. The fallback ccache
executable is `/home/hhanyu/.pixi.x86_64/bin/ccache`.

### Rebuild DeepEP after code changes

DeepEP has two compilation stages:
1. **Static build** (`pip install`): Compiles pybind, executor, permute kernels. Uses setuptools + CMake.
2. **JIT compilation** (at runtime): Compiles the main dispatch/combine kernels from `hybrid_ep_backend.cuh`. Cached in `~/.deepep/hybrid_ep/jit/`.

**Fast rebuild** (ccache + single arch):

```bash
ssh computelab "bash ~/projects/moe/scripts/run_on_compute.sh \
    -n <allocated-node> -c test_container_2606 \
    'cd /home/scratch.hhanyu_gpu/projects/moe/DeepEP && \
     PYTORCH_NVCC=\"ccache nvcc\" NVCC_APPEND_FLAGS=\"--threads 8\" \
     TORCH_CUDA_ARCH_LIST=\"10.3\" pip install --no-build-isolation . -v 2>&1 | tail -5'"
```

Build flags explained:
- `PYTORCH_NVCC="ccache nvcc"` — Use ccache for nvcc (ccache at `/home/hhanyu/.pixi.x86_64/bin/ccache`, added to PATH by `run_on_compute.sh`)
- `NVCC_APPEND_FLAGS="--threads 8"` — Parallel nvcc compilation threads
- `TORCH_CUDA_ARCH_LIST="10.3"` — Build only for B300 (sm_103a). Omitting this builds for many architectures and is much slower.
- `--no-build-isolation` — Reuses existing build environment, faster incremental builds
- `-v` — Verbose output to see compilation progress

**When to clear the JIT cache** (`rm -rf ~/.deepep/hybrid_ep/jit/`):
- Changed `hybrid_ep_backend.cuh` (the main kernel file — JIT-compiled at runtime)
- Changed template parameters or kernel signatures
- Changed `config.cuh` SMEM layout or pipeline configuration

**When JIT cache clearing is NOT needed**:
- Changed only Python code (`deep_ep/*.py`, `tests/*.py`)
- Changed only statically-compiled C++ (`executor.cu`, `permute.cu`, `hybrid_ep.cu`, `pybind_*.cu`)
- Changed only `compiler.cu` / `compiler.cuh` (the JIT compiler itself — needs static rebuild but not JIT cache clear)

**Full rebuild with JIT cache clear**:

```bash
ssh computelab "bash ~/projects/moe/scripts/run_on_compute.sh \
    -n <allocated-node> -c test_container_2606 \
    'cd /home/scratch.hhanyu_gpu/projects/moe/DeepEP && \
     rm -rf ~/.deepep/hybrid_ep/jit/ && \
     PYTORCH_NVCC=\"ccache nvcc\" NVCC_APPEND_FLAGS=\"--threads 8\" \
     TORCH_CUDA_ARCH_LIST=\"10.3\" pip install --no-build-isolation . -v 2>&1 | tail -5'"
```

**Typical build times** (with ccache warm, single arch):
- Static build (no changes): ~5s
- Static build (C++ changes): ~15-30s
- JIT compilation (first run after cache clear): ~30-120s per unique kernel config

### Run DeepEP hybrid-ep test

```bash
ssh computelab "bash ~/projects/moe/scripts/run_on_compute.sh \
    -n <allocated-node> -c test_container_2606 \
    'cd /home/scratch.hhanyu_gpu/projects/moe/DeepEP && \
     NUM_SMS_DISPATCH=24 NUM_SMS_COMBINE=24 HIDDEN_DIM=512 \
     NUM_TOKENS_PER_RANK=8192 NUM_LOCAL_EXPERTS=32 TOPK=36 \
     python tests/test_hybrid_ep.py --num-processes 8'"
```

Key environment variables for `test_hybrid_ep.py`:

| Variable | Default | Description |
|---|---|---|
| `HIDDEN_DIM` | 7168 | Token hidden dimension |
| `NUM_LOCAL_EXPERTS` | 1 | Experts per rank |
| `NUM_TOKENS_PER_RANK` | 4096 | Tokens per GPU |
| `MAX_NUM_OF_TOKENS_PER_RANK` | `NUM_TOKENS_PER_RANK` | Max tokens buffer size |
| `TOPK` | 8 | Top-k routing |
| `NUM_SMS_DISPATCH` | 24 (single-node) | SMs for dispatch kernel |
| `NUM_SMS_COMBINE` | 24 (single-node) | SMs for combine kernel |
| `USE_MNNVL` | auto-detect | 1 to force MNNVL fabric memory |
| `NUM_OF_STAGES_DISPATCH_API` | 10 | Dispatch SMEM pipeline stages |
| `NUM_OF_IN_FLIGHT_S2G_DISPATCH_API` | 8 | Dispatch S2G in-flight TMA groups |

### Run HybridEP with full-iteration CUDA graphs

HybridEP graph capture requires the sync-free dispatch path. Before capture:

1. Give HybridEP a fixed `num_permuted_tokens` upper bound.
2. Ensure `dispatch_with_permute(..., non_blocking=True)` is selected.
3. Warm up the exact dispatch/combine specialization before entering
   `torch.cuda.graph`.
4. Keep graph-owned dispatch handles alive through replay.

In the MCore MoE perf frontend, use the frontend-specific static-capacity knob:

```text
--moe-perf-full-iter-cuda-graph
--moe-perf-expert-rank-capacity-factor <factor>
```

Add `--moe-perf-paged-stash` when testing the overflow-recovery path. A value such
as `1.2` is an expected-load budget with 20% slack; it is not a mathematical
no-drop maximum. Check the persisted log for both
`paged_stash_overflow=False` and `paged_stash_host_spill=False`.

Do not attempt capture with a dynamic HybridEP receive size. If
`num_permuted_tokens=None`, MCore selects `non_blocking=False`, and HybridEP uses
host-visible metadata to derive the output size. CUDA stream capture will be
invalidated. The final error may surface at a later `cudaGetLastError()` after
the dispatch launch; treat that line as the observer, not automatically the
root cause.

For a strict no-drop bound with equal static input sizes, budget for the maximum
routes that the TPxEP group can send to one rank. This consumes more memory than
an expected-load factor. Record the chosen contract in the test log.

### Full sync-rebuild-test cycle

```bash
# 1. Sync code
rsync ~/projects/moe/DeepEP/ computelab:~/projects/moe/DeepEP/ -aPv \
    --exclude-from=$(realpath ~/.rsync-exclude)

# 2. Rebuild + clear JIT + test (all in one, with complete persistent logs)
ssh computelab "bash ~/projects/moe/scripts/run_on_compute.sh \
    -n <allocated-node> -c test_container_2606 \
    'cd /home/scratch.hhanyu_gpu/projects/moe/DeepEP && \
     rm -rf ~/.deepep/hybrid_ep/jit/ && \
     PYTORCH_NVCC=\"ccache nvcc\" NVCC_APPEND_FLAGS=\"--threads 8\" \
     TORCH_CUDA_ARCH_LIST=\"10.3\" pip install --no-build-isolation . -v \
       > logs/deepep_build_\$(date +%Y%m%d_%H%M%S).log 2>&1 && \
     NUM_SMS_DISPATCH=24 NUM_SMS_COMBINE=24 HIDDEN_DIM=512 \
     NUM_TOKENS_PER_RANK=8192 NUM_LOCAL_EXPERTS=32 TOPK=36 \
     python tests/test_hybrid_ep.py --num-processes 8 \
       > logs/hybrid_ep_test_\$(date +%Y%m%d_%H%M%S).log 2>&1'"
```

## Timeouts

- SSH to computelab: ~2s
- SSH to compute node: ~1s
- Enroot container launch: ~5s (prints PyTorch banner)
- `pip install` (DeepEP, ccache warm, single arch): ~5-30s
- `pip install` (DeepEP, cold build, all archs): ~5-10 min
- JIT kernel compilation: ~30-120s per unique kernel config (first run only)
- `test_hybrid_ep.py --num-processes 8`: ~2-5 min total

Set Bash tool timeout to at least 600000ms (10 min) for test runs.

## Troubleshooting

- **"No active SLURM job found"**: No interactive job running. User needs to start one on computelab.
- **Container banner but no output**: The enroot container prints a long PyTorch banner on launch. Wait for it.
- **JIT compilation errors**: Clear JIT cache and rebuild: `rm -rf ~/.deepep/hybrid_ep/jit/ && pip install -e .`
- **`cudaErrorIllegalAddress`**: Usually an asynchronous error from a previously launched kernel. Add `CUDA_LAUNCH_BLOCKING=1` to the command to pinpoint the faulting kernel.
- **Stale code**: Verify rsync worked: `ssh computelab "cd ~/projects/moe/DeepEP && git diff --stat HEAD"` — should match local `git diff --stat HEAD`.
