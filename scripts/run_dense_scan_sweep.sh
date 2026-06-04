#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

set -euo pipefail

BASE=${BASE:-/home/scratch.hhanyu_gpu/projects/moe}
DEEPEP_PATH=${DEEPEP_PATH:-$BASE/DeepEP}
SCRIPTS_DIR=${SCRIPTS_DIR:-$HOME/projects/moe/scripts}
LOG_ROOT=${LOG_ROOT:-$BASE/bench_logs}
LOGDIR=${LOGDIR:-$LOG_ROOT/dense_scan_sweep_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$LOGDIR"

BUILD=${BUILD:-1}
WARMUPS=${WARMUPS:-20}
TESTS=${TESTS:-50}

BUILD_ENV=(
  PYTORCH_NVCC="ccache nvcc"
  NVCC_APPEND_FLAGS="--threads 8"
  TORCH_CUDA_ARCH_LIST=10.3
)

COMMON_ARGS=(
  --deepep-path "$DEEPEP_PATH"
  --num-processes 1
  --hidden-dim 512
  --num-tokens 12288
  --max-num-tokens 12288
  --num-local-experts 32
  --pad-multiple 32
  --num-sms-dispatch 32
  --num-sms-combine 32
  --num-warmups "$WARMUPS"
  --num-tests "$TESTS"
)

CASES=(
  "e2304_k36:2304:36"
  "e128_k8:128:8"
  "e384_k6:384:6"
)

THREADS=(128 256 384 512)
BLOCKS=(6 8 16 32 64 108 128)

cd "$DEEPEP_PATH"
{
  echo "path=$DEEPEP_PATH"
  git status --short --branch || true
  git rev-parse --short HEAD || true
  echo "warmups=$WARMUPS tests=$TESTS"
} | tee "$LOGDIR/meta.txt"

if [[ "$BUILD" == "1" ]]; then
  rm -rf ~/.deepep/hybrid_ep/jit/ >/dev/null 2>&1 || true
  env "${BUILD_ENV[@]}" python -m pip install --no-build-isolation . -v \
    > "$LOGDIR/build.txt" 2>&1
fi

for cfg in "${CASES[@]}"; do
  label=${cfg%%:*}
  rest=${cfg#*:}
  total_experts=${rest%%:*}
  topk=${rest#*:}
  for threads in "${THREADS[@]}"; do
    for blocks in "${BLOCKS[@]}"; do
      out="$LOGDIR/${label}.t${threads}.b${blocks}.txt"
      NUM_OF_THREADS_PER_BLOCK_PREPROCESSING_API=$threads \
        python "$SCRIPTS_DIR/bench_hybrid_ep_dense_preprocess.py" \
        "${COMMON_ARGS[@]}" \
        --num-total-experts "$total_experts" \
        --topk "$topk" \
        --num-sms-preprocessing "$blocks" \
        > "$out" 2>&1
      grep -E "dense scan|Traceback|Error|RuntimeError|AssertionError" "$out" | tail -5
    done
  done
done | tee "$LOGDIR/summary.txt"

echo "LOGDIR=$LOGDIR"
