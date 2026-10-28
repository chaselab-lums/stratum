#!/usr/bin/env bash
set -euo pipefail
RW="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "$RW/setup.sh"
cd "$RW"
# Wait until no other Research-Work gem5 (e.g. Task 1 phase) before starting N dense sims
while pgrep -f 'gem5.opt.*Research-Work' >/dev/null 2>&1; do
  echo "[dense-parallel] waiting for other gem5 $(date -Is)"
  sleep 20
done
exec python3 results/run_dense_parallel.py
