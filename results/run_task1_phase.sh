#!/usr/bin/env bash
set -euo pipefail
RW="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "$RW/setup.sh"
OUT="$RW/results/runs/task1_phase_kmajor_true"
mkdir -p "$OUT"
cd "$RW/snnMNIST"
SNN_LINALG_OPTS='skip-zeros=true k-major=true' make -f Makefile generate NO_TRAIN=1 TEST_VECTORS=1 BUILD_NO_TRAIN=1
python3 "$RW/results/tools/inject_phase_markers.py" build/network.ll build/network_phase.ll
SNN_LINALG_OPTS='skip-zeros=true k-major=true' \
  NETWORK_LL=build/network_phase.ll PHASE_PROFILE=1 \
  make -f Makefile build-nogen NO_TRAIN=1
cp build/sim.elf "$OUT/"
GEM5_OUTDIR="$OUT/gem5" RUN_TIMESTAMP=task1_phase "$RW/run.sh" "$OUT/sim.elf" baremetal
rg "PHASE_CYCLES|FINAL_CLASS" "$OUT/gem5/program.out" | tee "$OUT/phase_summary.txt"
python3 "$RW/results/extract_roi_stats.py" "$OUT/gem5/stats.txt" -o "$OUT/roi_metrics.json"
