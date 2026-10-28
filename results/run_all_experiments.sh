#!/usr/bin/env bash
# Long-running gem5 matrix (~30+ runs). Logs to results/experiment.log
set -euo pipefail
RW="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "$RW/setup.sh"
cd "$RW"
LOG="$RW/results/experiment.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== experiment batch $(date -Is) ==="
python3 results/run_experiments.py
python3 results/make_tables.py
python3 results/generate_summary.py
bash results/run_task1_phase.sh || echo "task1 phase failed (see log)"
python3 results/generate_summary.py
echo "=== done $(date -Is) ==="
