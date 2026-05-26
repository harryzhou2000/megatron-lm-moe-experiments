#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

set -euo pipefail

BASE=${BASE:-/home/scratch.hhanyu_gpu/projects/moe}
SCRIPTS_DIR=${SCRIPTS_DIR:-$HOME/projects/moe/scripts}
SRC=${SRC:-$SCRIPTS_DIR/isolated_scan_bench.cu}
OUTDIR=${OUTDIR:-$BASE/bench_logs/isolated_scan_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUTDIR"

compile_one() {
  local name=$1
  local deepep=$2
  local threads=$3
  local blocks=$4
  local experts=$5
  local topk=$6
  local exe="$OUTDIR/scan_bench_${name}_t${threads}_b${blocks}_e${experts}_k${topk}"
  local build_log="$OUTDIR/${name}.build.t${threads}.b${blocks}.e${experts}.k${topk}.txt"
  if ! timeout 120s nvcc -std=c++17 -O3 --threads 8 -arch=sm_103a \
    -Xptxas -O2 \
    -DSCAN_THREADS="$threads" \
    -DSCAN_BLOCKS="$blocks" \
    -DSCAN_LOCAL_EXPERTS="$experts" \
    -DSCAN_TOPK="$topk" \
    -DSCAN_PERMUTE_FUSION="${SCAN_PERMUTE_FUSION:-1}" \
    -I"$deepep/csrc/hybrid_ep/backend" \
    -I"$deepep/csrc/hybrid_ep" \
    "$SRC" -o "$exe" \
    > "$build_log" 2>&1; then
    echo ""
    return 1
  fi
  echo "$exe"
}

run_one() {
  local name=$1
  local deepep=$2
  echo "=== $name ===" | tee -a "$OUTDIR/summary.txt"
  for case in ${SCAN_CASES:-32/36}; do
    local experts=${case%%/*}
    local topk=${case#*/}
    for cfg in ${SCAN_CONFIGS:-256/108 256/64 512/108 512/64}; do
      local threads=${cfg%%/*}
      local blocks=${cfg#*/}
      local exe
      if exe=$(compile_one "$name" "$deepep" "$threads" "$blocks" "$experts" "$topk") && [[ -n "$exe" ]]; then
        "$exe" 10 30 | tee -a "$OUTDIR/summary.txt"
      else
        echo "isolated scan<${threads},${blocks},256,12288,64,72,1,${experts},${topk}>: SKIP compile_failed_or_timeout" | tee -a "$OUTDIR/summary.txt"
      fi
    done
  done
}

BEFORE=${BEFORE:-$BASE/DeepEP_dense_scan_before}
AFTER=${AFTER:-$BASE/DeepEP}

case "${SCAN_VERSION:-both}" in
  before) run_one before "$BEFORE" ;;
  after) run_one after "$AFTER" ;;
  both)
    run_one before "$BEFORE"
    run_one after "$AFTER"
    ;;
  *)
    echo "Invalid SCAN_VERSION=${SCAN_VERSION}; expected before, after, or both" >&2
    exit 2
    ;;
esac

echo "OUTDIR=$OUTDIR"
