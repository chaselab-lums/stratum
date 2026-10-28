#!/usr/bin/env bash
RW="$(cd "$(dirname "$0")/.." && pwd)"
csv="$RW/results/metrics.csv"
runs=$(find "$RW/results/runs" -name roi_metrics.json 2>/dev/null | wc -l)
lines=$(($(wc -l <"$csv") - 1))
echo "metrics rows: $lines  roi_metrics.json: $runs"
pgrep -af 'gem5.opt.*Research-Work' | head -1 || echo "gem5: idle"
pgrep -af 'run_experiments.py' | head -1 || echo "runner: idle"
pgrep -af 'run_experiments\.py|run_all_experiments' | head -1 || echo "experiments: idle"
echo "log tail:"
tail -3 "$RW/results/experiment.log" 2>/dev/null
