#!/usr/bin/env bash
# Step 0: reproduce layout A/B with TEST_VECTORS=1 (drawing 0 only in ROI).
set -euo pipefail

RESEARCH_WORK="$(cd "$(dirname "$0")/../.." && pwd)"
# shellcheck source=/dev/null
source "$RESEARCH_WORK/setup.sh"

RUN_ID="${RUN_ID:-step0-$(date +%Y%m%d-%H%M%S)}"
OUT="$RESEARCH_WORK/results/step0/$RUN_ID"
mkdir -p "$OUT"

record_repo() {
  local name=$1 dir=$2
  if [[ -d "$dir/.git" ]]; then
    git -C "$dir" rev-parse HEAD > "$OUT/${name}.commit"
    git -C "$dir" status --short > "$OUT/${name}.status" || true
  else
    echo "no-git" > "$OUT/${name}.commit"
  fi
}

record_repo snnMNIST "$RESEARCH_WORK/snnMNIST"
record_repo snn-mlir "$RESEARCH_WORK/snn-mlir"
record_repo gem5 "$RESEARCH_WORK/gem5"

build_and_run() {
  local tag=$1
  local linalg_opts=$2
  local sub="$OUT/$tag"
  mkdir -p "$sub"

  echo "=== build $tag: $linalg_opts ==="
  (
    cd "$RESEARCH_WORK/snnMNIST"
    SNN_MLIR_ROOT="$SNN_MLIR_ROOT" SNN_OPT="$SNN_OPT" \
      MLIR_DIR="$MLIR_DIR" LLVM_BIN="$LLVM_BIN" \
      MLIR_OPT_BIN="$MLIR_OPT_BIN" MLIR_TRANSLATE_BIN="$MLIR_TRANSLATE_BIN" \
      SNN_LINALG_OPTS="$linalg_opts" \
      make -f Makefile build NO_TRAIN=1 TEST_VECTORS=1 BUILD_NO_TRAIN=1
  )

  cp "$RESEARCH_WORK/snnMNIST/build/sim.elf" "$sub/sim.elf"
  cp "$RESEARCH_WORK/snnMNIST/build/network.mlir" "$sub/network.mlir"
  cp "$RESEARCH_WORK/snnMNIST/build/network.ll" "$sub/network.ll"
  cp "$RESEARCH_WORK/snnMNIST/build/input.h" "$sub/input.h"
  echo "$linalg_opts" > "$sub/snn_linalg_opts.txt"

  echo "=== gem5 $tag ==="
  GEM5_OUTDIR="$sub/gem5" RUN_TIMESTAMP="$tag" \
    "$RESEARCH_WORK/run.sh" "$sub/sim.elf" baremetal 2>&1 | tee "$sub/gem5_console.log"

  python3 "$RESEARCH_WORK/results/extract_roi_stats.py" \
    "$sub/gem5/stats.txt" -o "$sub/roi_metrics.json"

  rg "FINAL_CLASS" "$sub/gem5/program.out" > "$sub/final_classes.txt" || true
}

build_and_run "kmajor_false" "skip-zeros=true k-major=false"
build_and_run "kmajor_true" "skip-zeros=true k-major=true"

python3 - <<'PY' "$OUT"
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
rows = []
for tag in ("kmajor_false", "kmajor_true"):
    p = out / tag / "roi_metrics.json"
    if p.exists():
        m = json.loads(p.read_text())
        rows.append((tag, m))
print("tag\tsimSeconds_ms\tipc\tnumCycles\tsimInsts")
for tag, m in rows:
    print(f"{tag}\t{m.get('simSeconds',0)*1000:.6f}\t{m.get('ipc')}\t{m.get('numCycles')}\t{m.get('simInsts')}")
PY

echo "Wrote $OUT"
