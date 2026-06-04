#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

set -euo pipefail

BASE=${BASE:-/home/scratch.hhanyu_gpu/projects/moe}
DEEPEP_MAIN=${DEEPEP_MAIN:-$BASE/DeepEP}
SCRIPTS_DIR=${SCRIPTS_DIR:-$HOME/projects/moe/scripts}
LOG_ROOT=${LOG_ROOT:-$BASE/bench_logs}
LOGDIR=${LOGDIR:-$LOG_ROOT/dense_scan_compare_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$LOGDIR"

BUILD_ENV=(
  PYTORCH_NVCC="ccache nvcc"
  NVCC_APPEND_FLAGS="--threads 8"
  TORCH_CUDA_ARCH_LIST=10.3
)

CORRECTNESS_ENV=(
  HIDDEN_DIM=512
  NUM_TOKENS_PER_RANK=8192
  MAX_NUM_OF_TOKENS_PER_RANK=8192
  NUM_LOCAL_EXPERTS=32
  TOPK=36
  NUM_SMS_DISPATCH=32
  NUM_SMS_COMBINE=32
  NUM_OF_STAGES_G2S_COMBINE_API=64
  NUM_OF_STAGES_S2G_COMBINE_API=8
  NUM_TOKENS_COMBINE_REDUCE_BATCH_COMBINE_API=16
  NUM_OF_TOKENS_PER_GROUP_COMBINE_API=2
)

PREPROCESS_COMMON_ARGS=(
  --num-processes 1
  --hidden-dim 512
  --num-tokens 12288
  --max-num-tokens 12288
  --num-local-experts 32
  --pad-multiple 32
  --num-sms-dispatch 32
  --num-sms-combine 32
  --num-sms-preprocessing 108
  --num-warmups 50
  --num-tests 100
)

PREPROCESS_CASES=(
  "e2304_k36:2304:36"
  "e128_k8:128:8"
  "e384_k6:384:6"
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

  env "${CORRECTNESS_ENV[@]}" python tests/test_hybrid_ep.py --num-processes 8 \
    > "$LOGDIR/$name.correctness.txt" 2>&1

  for cfg in "${PREPROCESS_CASES[@]}"; do
    local label=${cfg%%:*}
    local rest=${cfg#*:}
    local total_experts=${rest%%:*}
    local topk=${rest#*:}
    rm -rf ~/.deepep/hybrid_ep/jit/ >/dev/null 2>&1 || true
    python "$SCRIPTS_DIR/bench_hybrid_ep_dense_preprocess.py" \
      --deepep-path "$path" \
      --num-total-experts "$total_experts" \
      --topk "$topk" \
      "${PREPROCESS_COMMON_ARGS[@]}" \
      > "$LOGDIR/$name.preprocess.$label.txt" 2>&1
  done
}

before_path=$(prepare_baseline)
run_one before "$before_path"
run_one after "$DEEPEP_MAIN"

echo "LOGDIR=$LOGDIR"
