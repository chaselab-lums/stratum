#!/usr/bin/env bash

set -eo pipefail

RESEARCH_WORK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-all}"
ELF="${2:-$RESEARCH_WORK/snnMNIST/build/sim.elf}"

# shellcheck source=setup.sh
source "$RESEARCH_WORK/setup.sh"

RESULTS_ROOT="${RESULTS_ROOT:-$RESEARCH_WORK/gem5/work/results}"
PROFILE_ROOT="${PROFILE_ROOT:-$RESULTS_ROOT/profiles}"
REFERENCE_CPU_TYPE="${REFERENCE_CPU_TYPE:-RiscvO3CPU}"
# Same wide-BOOM sizing as ./run.sh. GEM5_BOOM=0 keeps the stock O3 widths.
export GEM5_BOOM="${GEM5_BOOM:-1}"

# profile.json names one config file per mode.
profile_json() {
	python3 -c '
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
index = json.loads((root / "profile.json").read_text())
entry = index[sys.argv[2]]
path = Path(entry)
if not path.is_absolute():
    path = root / path
print(path)
' "$RESEARCH_WORK" "$1"
}

profile_get() {
	python3 -c '
import json, sys
from pathlib import Path
doc = json.loads(Path(sys.argv[1]).read_text())
value = doc.get(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "")
if isinstance(value, bool):
    print("true" if value else "false")
else:
    print(value)
' "$1" "$2" "${3:-}"
}

PROFILE_NAME="${PROFILE_NAME:-}"
if [[ -z "$PROFILE_NAME" ]]; then
	case "$MODE" in
	stats|o3pipe)
		PROFILE_NAME="$(profile_get "$(profile_json "$MODE")" name "")"
		;;
	esac
fi
if [[ -z "$PROFILE_NAME" ]]; then
	case "$MODE" in
	stats)
		PROFILE_NAME="inference-stats"
		;;
	o3pipe)
		PROFILE_NAME="pipeline-detail"
		;;
	opt-compare)
		PROFILE_NAME="opt-compare"
		;;
	all)
		PROFILE_NAME="main"
		;;
	sweep)
		PROFILE_NAME="sweep"
		;;
	*)
		PROFILE_NAME="$(date +%Y%m%d-%H%M%S)"
		;;
	esac
fi
OUTDIR="${GEM5_OUTDIR:-$PROFILE_ROOT/$MODE/$PROFILE_NAME}"
PROFILE_MAXINSTS="${PROFILE_MAXINSTS:-}"
SKIP_WORKLOAD_BUILD="${SKIP_WORKLOAD_BUILD:-0}"

if [[ "$MODE" != "sweep" ]]; then
	mkdir -p "$OUTDIR"
fi

ensure_build() {
	if [[ "$SKIP_WORKLOAD_BUILD" == "1" ]]; then
		return
	fi
	make -C "$RESEARCH_WORK/snnMNIST" build ENABLE_GEM5_M5OPS=1 BUILD_NO_TRAIN=1
}

run_profile() {
	local cpu_type="$1"
	local global_args="$2"
	local workload_args="$3"
	local enable_caches="$4"

	GEM5_CPU_TYPE="$cpu_type" \
	GEM5_OUTDIR="$OUTDIR" \
	GEM5_ENABLE_CACHES="$enable_caches" \
	GEM5_GLOBAL_ARGS="$global_args" \
	GEM5_WORKLOAD_ARGS="$workload_args" \
	SKIP_BOTTLENECK_HTML=1 \
	"$RESEARCH_WORK/run.sh" "$ELF" baremetal
}

case "$MODE" in
	stats)
		STATS_JSON="$(profile_json stats)"
		STATS_CPU="$(profile_get "$STATS_JSON" cpu "$REFERENCE_CPU_TYPE")"
		STATS_CACHES="$(profile_get "$STATS_JSON" caches true)"
		STATS_ARGS="$(profile_get "$STATS_JSON" args "")"
		STATS_MAXINSTS="$(profile_get "$STATS_JSON" max_insts "")"
		if [[ -n "${PROFILE_MAXINSTS:-}" ]]; then
			STATS_MAXINSTS="$PROFILE_MAXINSTS"
		fi
		STATS_CACHE_FLAG=1
		if [[ "$STATS_CACHES" == "false" ]]; then
			STATS_CACHE_FLAG=0
		fi
		ensure_build
		run_profile "${GEM5_CPU_TYPE:-$STATS_CPU}" "" "${STATS_MAXINSTS:+-I $STATS_MAXINSTS} $STATS_ARGS" "$STATS_CACHE_FLAG"
		python3 "$RESEARCH_WORK/scripts/gem5_full_report.py" "$PROFILE_ROOT"
		echo "ROI stats written to: $OUTDIR/stats.txt"
		echo "Report: $PROFILE_ROOT/report.html"
		;;
	o3pipe)
		O3_JSON="$(profile_json o3pipe)"
		O3_CPU="$(profile_get "$O3_JSON" cpu "$REFERENCE_CPU_TYPE")"
		O3_FLAGS="$(profile_get "$O3_JSON" debug_flags O3PipeView)"
		O3_FILE="$(profile_get "$O3_JSON" debug_file trace.out)"
		O3_INTO="$(profile_get "$O3_JSON" cycles_into_roi 2000)"
		O3_WINDOW="$(profile_get "$O3_JSON" window_cycles 400)"
		O3_STATS_REL="$(profile_get "$O3_JSON" stats "")"
		ensure_build
		if [[ -n "$O3_STATS_REL" ]]; then
			STATS_FILE="$RESEARCH_WORK/$O3_STATS_REL"
		else
			STATS_FILE="$PROFILE_ROOT/stats/inference-stats/stats.txt"
		fi
		if [[ ! -f "$STATS_FILE" ]]; then
			echo "No stats session yet. Running stats so the trace can start inside the kernel."
			SKIP_WORKLOAD_BUILD=1 PROFILE_NAME="inference-stats" "$0" stats "$ELF"
			STATS_FILE="$PROFILE_ROOT/stats/inference-stats/stats.txt"
		fi
		WINDOW_TEXT="$(python3 "$RESEARCH_WORK/scripts/o3_debug_window.py" --cycles-into "$O3_INTO" --window "$O3_WINDOW" "$STATS_FILE")"
		mapfile -t WINDOW_LINES <<< "$WINDOW_TEXT"
		echo "O3 trace window: ${WINDOW_LINES[0]} ${WINDOW_LINES[1]}"
		SKIP_BOTTLENECK_HTML=1 run_profile "${GEM5_CPU_TYPE:-$O3_CPU}" \
			"--debug-flags=${O3_FLAGS} --debug-file=${O3_FILE} ${WINDOW_LINES[0]}" \
			"${WINDOW_LINES[1]} ${PROFILE_MAXINSTS:+-I $PROFILE_MAXINSTS}" 1
		python3 "$RESEARCH_WORK/gem5/util/o3-pipeview.py" -o "$OUTDIR/pipeview.out" "$OUTDIR/trace.out"
		python3 "$RESEARCH_WORK/scripts/gem5_full_report.py" "$PROFILE_ROOT"
		echo "O3 trace written to: $OUTDIR/trace.out"
		echo "Report: $PROFILE_ROOT/report.html"
		;;
	sweep)
		STATS_FILE="$PROFILE_ROOT/stats/inference-stats/stats.txt"
		if [[ ! -f "$STATS_FILE" ]]; then
			echo "No stats session yet. Running stats so the sweep can stop after one timestep."
			SKIP_WORKLOAD_BUILD=1 PROFILE_NAME="inference-stats" "$0" stats "$ELF"
		fi
		# Stop when the first stats region ends, so each knob is one image.
		ROI_END="$(python3 "$RESEARCH_WORK/scripts/o3_debug_window.py" --roi-end "$STATS_FILE")"
		# The stats file may be the 4MiB run, which finishes the image near 8e9
		# ticks. A 512KiB L2 needed about 1.6e10 ticks for that same image.
		ROI_TICK="${ROI_END#--abs-max-tick }"
		PROFILE_JSON="${PROFILE_JSON:-$(profile_json sweep)}"
		MIN_TICK="$(profile_get "$PROFILE_JSON" min_tick 18000000000)"
		if [[ "$ROI_TICK" =~ ^[0-9]+$ && "$MIN_TICK" =~ ^[0-9]+$ && "$ROI_TICK" -lt "$MIN_TICK" ]]; then
			ROI_END="--abs-max-tick $MIN_TICK"
		fi
		echo "Sweep stops at: $ROI_END"
		if [[ ! -f "$PROFILE_JSON" ]]; then
			echo "ERROR: sweep config not found: $PROFILE_JSON" >&2
			exit 1
		fi
		echo "Sweep points from: $PROFILE_JSON"
		SWEEP_ROOT="$PROFILE_ROOT/sweep"
		run_sweep_point() {
			local name="$1"
			local knob="$2"
			local workload_args="$3"
			local point_dir="$SWEEP_ROOT/$name"
			mkdir -p "$point_dir"
			printf '%s\n' "$knob" > "$point_dir/knob.txt"
			GEM5_CPU_TYPE="${REFERENCE_CPU_TYPE}" \
			GEM5_OUTDIR="$point_dir" \
			GEM5_ENABLE_CACHES=1 \
			GEM5_WORKLOAD_ARGS="$ROI_END $workload_args" \
			"$RESEARCH_WORK/run.sh" "$ELF" baremetal
		}
		while IFS=$'\t' read -r name label args; do
			point_dir="$SWEEP_ROOT/$name"
			if [[ -f "$point_dir/stats.txt" && -f "$point_dir/knob.txt" ]]; then
				existing="$(<"$point_dir/knob.txt")"
				if [[ "$existing" == "$label" ]]; then
					echo "Reusing sweep point: $name ($label)"
					continue
				fi
			fi
			echo "Sweep point: $name ($label)"
			run_sweep_point "$name" "$label" "$args"
		done < <(python3 -c '
import json, sys
from pathlib import Path
doc = json.loads(Path(sys.argv[1]).read_text())
for point in doc["points"]:
    print(point["name"] + "\t" + point["label"] + "\t" + point.get("args", ""))
' "$PROFILE_JSON")
		python3 "$RESEARCH_WORK/scripts/gem5_full_report.py" "$PROFILE_ROOT"
		;;
	all)
		ensure_build
		SKIP_WORKLOAD_BUILD=1 PROFILE_NAME="inference-stats" "$0" stats "$ELF"
		SKIP_WORKLOAD_BUILD=1 PROFILE_NAME="pipeline-detail" "$0" o3pipe "$ELF"
		"$0" sweep "$ELF"
		python3 "$RESEARCH_WORK/scripts/gem5_full_report.py" "$PROFILE_ROOT"
		echo ""
		echo "Report: $PROFILE_ROOT/report.html"
		;;
	opt-compare)
		python3 "$RESEARCH_WORK/scripts/snn_opt_compare_report.py" "$(profile_json opt-compare)"
		;;
	*)
		echo "Usage: $0 [all|stats|o3pipe|opt-compare|sweep] [path/to/sim.elf]" >&2
		exit 1
		;;
esac

xdg-open "$PROFILE_ROOT/report.html" >/dev/null 2>&1 || true