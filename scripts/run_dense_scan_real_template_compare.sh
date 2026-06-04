#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

set -euo pipefail

BASE=${BASE:-/home/scratch.hhanyu_gpu/projects/moe}
DEEPEP_MAIN=${DEEPEP_MAIN:-$BASE/DeepEP}
SCRIPTS_DIR=${SCRIPTS_DIR:-$HOME/projects/moe/scripts}
LOG_ROOT=${LOG_ROOT:-$BASE/bench_logs}
LOGDIR=${LOGDIR:-$LOG_ROOT/dense_scan_real_template_compare_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$LOGDIR"

BUILD_ENV=(
  PYTORCH_NVCC="ccache nvcc"
  NVCC_APPEND_FLAGS="--threads 8"
  TORCH_CUDA_ARCH_LIST=10.3
)

BENCH_ARGS=(
  --num-processes 1
  --hidden-dim 512
  --num-tokens 12288
  --max-num-tokens 12288
  --num-local-experts 32
  --num-total-experts 2304
  --topk 36
  --fake-ranks-per-node 72
  --pad-multiple 256
  --num-sms-dispatch 32
  --num-sms-combine 32
  --num-sms-preprocessing 108
  --num-warmups 20
  --num-tests 50
)

prepare_baseline() {
  cd "$DEEPEP_MAIN"
  git fetch origin
  local wt="$BASE/DeepEP_dense_scan_before"
  git worktree remove -f "$wt" >/dev/null 2>&1 || true
  git worktree add -B dense_scan_before "$wt" origin/hhanyu/hybrid-ep-sparse-opt-2 >&2
  echo "$wt"
}

run_one() {
  local name=$1
  local path=$2
  cd "$path"
  {
    echo "name=$name"
    echo "path=$path"
    git status --short --branch || true
    git rev-parse --short HEAD || true
  } | tee "$LOGDIR/$name.meta.txt"

  rm -rf ~/.deepep/hybrid_ep/jit/ >/dev/null 2>&1 || true
  env "${BUILD_ENV[@]}" python -m pip install --no-build-isolation . -v \
    > "$LOGDIR/$name.build.txt" 2>&1

  NUM_OF_THREADS_PER_BLOCK_PREPROCESSING_API=256 \
    python "$SCRIPTS_DIR/bench_hybrid_ep_dense_preprocess.py" \
    --deepep-path "$path" \
    "${BENCH_ARGS[@]}" \
    > "$LOGDIR/$name.preprocess.real_template.txt" 2>&1
  grep -E "dense scan|Traceback|Error|RuntimeError|AssertionError" \
    "$LOGDIR/$name.preprocess.real_template.txt" | tail -10
}

before_path=$(prepare_baseline)
run_one before "$before_path"
run_one after "$DEEPEP_MAIN"

echo "LOGDIR=$LOGDIR"
