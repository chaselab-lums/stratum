#!/usr/bin/env bash

set -eo pipefail

RESEARCH_WORK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_ROOT="${RESULTS_ROOT:-$RESEARCH_WORK/gem5/work/results}"

remove_if_exists() {
	local path="$1"
	if [[ -e "$path" ]]; then
		rm -rf "$path"
	fi
}

# Keep one canonical session per mode. Remove older ad-hoc smoke/layout runs.
remove_if_exists "$RESULTS_ROOT/profiles/stats/layout-check"
remove_if_exists "$RESULTS_ROOT/profiles/stats/smoke-session"
remove_if_exists "$RESULTS_ROOT/profiles/o3pipe/smoke-session"
remove_if_exists "$RESULTS_ROOT/profiles/dashboard-smoke-session.html"

# Keep only the most recent baremetal run by default.
if [[ -d "$RESULTS_ROOT/runs/baremetal" ]]; then
	mapfile -t run_dirs < <(find "$RESULTS_ROOT/runs/baremetal" -mindepth 1 -maxdepth 1 -type d | sort)
	if (( ${#run_dirs[@]} > 1 )); then
		for ((i = 0; i < ${#run_dirs[@]} - 1; i++)); do
			rm -rf "${run_dirs[$i]}"
		done
	fi
fi

echo "Cleaned old result sessions under: $RESULTS_ROOT"