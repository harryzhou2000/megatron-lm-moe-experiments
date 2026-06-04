#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

set -euo pipefail

BASE=${BASE:-/home/scratch.hhanyu_gpu/projects/moe}
DEEPEP_MAIN=${DEEPEP_MAIN:-$BASE/DeepEP}
SCRIPTS_DIR=${SCRIPTS_DIR:-$HOME/projects/moe/scripts}
LOG_ROOT=${LOG_ROOT:-$BASE/bench_logs}
LOGDIR=${LOGDIR:-$LOG_ROOT/hybrid_ep_compare_$(date +%Y%m%d_%H%M%S)}
REPEATS=${REPEATS:-3}
METADATA_WARMUPS=${METADATA_WARMUPS:-50}
METADATA_TESTS=${METADATA_TESTS:-100}

mkdir -p "$LOGDIR"

COMMON_ENV=(
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

BUILD_ENV=(
  PYTORCH_NVCC="ccache nvcc"
  NVCC_APPEND_FLAGS="--threads 8"
  TORCH_CUDA_ARCH_LIST=10.3
)

BRANCHES=(
  "opt:hhanyu/hybrid-ep-sparse-opt"
  "opt1:origin/hhanyu/hybrid-ep-sparse-opt-1"
  "opt2:origin/hhanyu/hybrid-ep-sparse-opt-2"
)

if [[ $# -gt 0 ]]; then
  case "$1" in
    opt) BRANCHES=("opt:hhanyu/hybrid-ep-sparse-opt") ;;
    opt1) BRANCHES=("opt1:origin/hhanyu/hybrid-ep-sparse-opt-1") ;;
    opt2) BRANCHES=("opt2:origin/hhanyu/hybrid-ep-sparse-opt-2") ;;
    *:*) BRANCHES=("$1") ;;
    *)
      echo "Usage: $0 [opt|opt1|opt2|name:git-ref]" >&2
      exit 2
      ;;
  esac
fi

cd "$DEEPEP_MAIN"
git fetch origin

for item in "${BRANCHES[@]}"; do
  name=${item%%:*}
  ref=${item#*:}
  wt="$BASE/DeepEP_bench_$name"

  git worktree remove -f "$wt" >/dev/null 2>&1 || true
  git worktree add -B "bench_$name" "$wt" "$ref"

  cd "$wt"
  commit=$(git rev-parse --short HEAD)
  {
    echo "branch=$name"
    echo "ref=$ref"
    echo "commit=$commit"
  } | tee "$LOGDIR/$name.meta.txt"

  rm -rf ~/.deepep/hybrid_ep/jit/ >/dev/null 2>&1 || true

  env "${BUILD_ENV[@]}" python -m pip install --no-build-isolation . -v \
    > "$LOGDIR/$name.build.txt" 2>&1

  case "$name" in
    opt) preproc_mode=standalone ;;
    opt1) preproc_mode=both ;;
    opt2) preproc_mode=fused ;;
    *) preproc_mode=both ;;
  esac

  for run in $(seq 1 "$REPEATS"); do
    env "${COMMON_ENV[@]}" python tests/test_hybrid_ep.py --num-processes 8 \
      > "$LOGDIR/$name.perf.run$run.txt" 2>&1

    env "${COMMON_ENV[@]}" python "$SCRIPTS_DIR/bench_hybrid_ep_metadata.py" \
      --deepep-path "$wt" --num-processes 8 --mode "$preproc_mode" \
      --num-warmups "$METADATA_WARMUPS" --num-tests "$METADATA_TESTS" \
      > "$LOGDIR/$name.preproc.run$run.txt" 2>&1
  done

  echo "DONE $name" | tee -a "$LOGDIR/$name.meta.txt"
done

echo "LOGDIR=$LOGDIR"
