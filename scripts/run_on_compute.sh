#!/bin/bash
# run_on_compute.sh — Run a command inside the enroot container on a compute node.
#
# Usage (from computelab):
#   bash ~/projects/moe/scripts/run_on_compute.sh 'your command here'
#   bash ~/projects/moe/scripts/run_on_compute.sh -n umb-b300-dp-184 'your command here'
#
# Usage (from local Mac):
#   ssh computelab "bash ~/projects/moe/scripts/run_on_compute.sh 'your command here'"
#
# Options:
#   -n NODE   Specify the compute node manually (skip squeue discovery)
#
# NOTE: Compilation (pip install, nvcc, etc.) must run on the compute node
#       inside the container — not on the computelab login node.

set -euo pipefail

NODE=""
while getopts "n:" opt; do
    case $opt in
        n) NODE="$OPTARG" ;;
        *) echo "Usage: $0 [-n NODE] 'command'" >&2; exit 1 ;;
    esac
done
shift $((OPTIND - 1))

CMD="${1:?Usage: $0 [-n NODE] 'command to run inside container'}"

# Discover the active compute node if not specified
if [ -z "$NODE" ]; then
    NODE=$(squeue -u "$USER" -h -o "%N" | head -1)
    if [ -z "$NODE" ]; then
        echo "ERROR: No active SLURM job found. Use -n NODE to specify manually." >&2
        exit 1
    fi
    echo "==> Discovered active node: $NODE"
else
    echo "==> Using specified node: $NODE"
fi

# SSH to the node and run inside the enroot container.
# Mounts match ~/scratch/enroot_test1.sh:
#   --mount /home/scratch.hhanyu_gpu:/home/scratch.hhanyu_gpu  (scratch storage)
#   --mount $HOME:$HOME                                        (home dir with .bashrc, .local/, etc.)
#
# Environment setup inside the container:
#   1. Add pixi bin to PATH (ccache lives here)
#   2. Set CCACHE_DIR for build caching
#   3. Activate the Python venv (adds venv bin to PATH for pip, ninja, python)
ssh "$NODE" "enroot start -w \
    --mount /home/scratch.hhanyu_gpu:/home/scratch.hhanyu_gpu \
    --mount $HOME:$HOME \
    test_container_2602 \
    bash -c 'export PATH=$HOME/.pixi.x86_64/bin:$HOME/.local/bin:/workspace/venv/bin:\$PATH && \
             export CCACHE_DIR=$HOME/scratch/.ccache && \
             source /workspace/venv/bin/activate && \
             $CMD'"
