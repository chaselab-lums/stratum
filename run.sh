#!/usr/bin/env bash
# Run snnMNIST on gem5.
#
# Usage:
#   ./run.sh [path/to/sim.elf] [baremetal|se]
set -eo pipefail

RESEARCH_WORK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=setup.sh
source "$RESEARCH_WORK/setup.sh"

ELF="${1:-$RESEARCH_WORK/snnMNIST/build/sim.elf}"
MODE="${2:-${GEM5_MODE:-baremetal}}"
GEM5_BIN="${GEM5_BIN:-$RESEARCH_WORK/gem5/build/RISCV/gem5.opt}"
GEM5_SCRIPT="${GEM5_SCRIPT:-$RESEARCH_WORK/gem5/work/main.py}"
GEM5_BAREMETAL_SCRIPT="${GEM5_BAREMETAL_SCRIPT:-$RESEARCH_WORK/gem5/configs/example/riscv/fs_linux.py}"
GEM5_CPU_TYPE="${GEM5_CPU_TYPE:-RiscvO3CPU}"
# 1: RiscvO3CPU sized to the wide BOOM pipeline figure.
# 0: stock gem5 O3 widths (the ./profile.sh reference).
GEM5_BOOM="${GEM5_BOOM:-1}"
GEM5_L2_SIZE="${GEM5_L2_SIZE:-4MiB}"
GEM5_BOOM_WIDTH="${GEM5_BOOM_WIDTH:-4}"
GEM5_MEM_SIZE="${GEM5_MEM_SIZE:-8GiB}"
RESULTS_ROOT="${RESULTS_ROOT:-$RESEARCH_WORK/gem5/work/results}"
RUN_ROOT="${RUN_ROOT:-$RESULTS_ROOT/runs}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
GEM5_ENABLE_CACHES="${GEM5_ENABLE_CACHES:-1}"
GEM5_GLOBAL_ARGS_RAW_VALUE="${GEM5_GLOBAL_ARGS_RAW:-${GEM5_GLOBAL_ARGS:-}}"
GEM5_WORKLOAD_ARGS_RAW_VALUE="${GEM5_WORKLOAD_ARGS_RAW:-${GEM5_WORKLOAD_ARGS:-}}"

GEM5_GLOBAL_ARGS=()
if [[ -n "$GEM5_GLOBAL_ARGS_RAW_VALUE" ]]; then
	read -r -a GEM5_GLOBAL_ARGS <<< "$GEM5_GLOBAL_ARGS_RAW_VALUE"
fi

GEM5_WORKLOAD_ARGS=()
if [[ -n "$GEM5_WORKLOAD_ARGS_RAW_VALUE" ]]; then
	read -r -a GEM5_WORKLOAD_ARGS <<< "$GEM5_WORKLOAD_ARGS_RAW_VALUE"
fi

GEM5_CACHE_ARGS=()
if [[ "$GEM5_ENABLE_CACHES" == "1" ]]; then
	GEM5_CACHE_ARGS+=(--caches)
fi

# Figure: fetch 8, decode/commit 4, issue 8, ROB 128, one 32-entry issue
# queue, 128 int and fp physical registers, LDQ/STQ 32, 32KiB 8-way L1s,
# 8 L1D MSHRs. The figure's L2 is 512KiB. fc1 is 2.3MB. A 4MiB L2 holds it;
# 8MiB measured the same rate, so L2 stays 4MiB. fetchWidth, issueWidth,
# wbWidth, and the load/store queues already match, so they stay at the gem5
# defaults. fetchBufferSize stays 64 bytes (it is a byte count, not an entry
# count).
BOOM_ARGS=()
if [[ "$MODE" == "baremetal" && "$GEM5_BOOM" == "1" ]]; then
	_rob=128
	_iq=32
	_preg=128
	case "$GEM5_BOOM_WIDTH" in
	2)
		_decode=2; _rename=2; _dispatch=2; _commit=2; _squash=2
		_rob=64; _iq=16; _preg=64
		;;
	8)
		_decode=8; _rename=8; _dispatch=8; _commit=8; _squash=8
		_rob=192; _iq=64; _preg=192
		;;
	*)
		_decode=4; _rename=4; _dispatch=4; _commit=4; _squash=4
		;;
	esac
	BOOM_ARGS+=(
		--param system.cpu[0].decodeWidth="${_decode}"
		--param system.cpu[0].renameWidth="${_rename}"
		--param system.cpu[0].dispatchWidth="${_dispatch}"
		--param system.cpu[0].commitWidth="${_commit}"
		--param system.cpu[0].squashWidth="${_squash}"
		--param system.cpu[0].numROBEntries="${_rob}"
		--param system.cpu[0].numIQEntries="${_iq}"
		--param system.cpu[0].numPhysIntRegs="${_preg}"
		--param system.cpu[0].numPhysFloatRegs="${_preg}"
		--l1i_size 32KiB
		--l1i_assoc 8
		--l1d_size 32KiB
		--l1d_assoc 8
		--l2cache
		--l2_size "$GEM5_L2_SIZE"
		--l2_assoc 8
		--param system.cpu[0].dcache.mshrs=8
		--param system.cpu[0].dcache.tgts_per_mshr=8
	)
fi

if [[ ! -x "$GEM5_BIN" ]]; then
	echo "ERROR: gem5 binary not found or not executable: $GEM5_BIN" >&2
	echo "Build it with: cd $RESEARCH_WORK/gem5 && scons build/RISCV/gem5.opt" >&2
	exit 1
fi

if [[ ! -f "$ELF" ]]; then
	echo "ERROR: ELF not found: $ELF" >&2
	echo "Build first: cd $RESEARCH_WORK/snnMNIST && make build" >&2
	exit 1
fi

ELF="$(readlink -f "$ELF")"

if [[ -z "${GEM5_OUTDIR:-}" ]]; then
	case "$MODE" in
	baremetal)
		GEM5_OUTDIR="$RUN_ROOT/baremetal/$RUN_TIMESTAMP"
		;;
	se)
		GEM5_OUTDIR="$RUN_ROOT/se/$RUN_TIMESTAMP"
		;;
	esac
fi

mkdir -p "$GEM5_OUTDIR"

case "$MODE" in
	baremetal)
		echo "gem5 bare-metal run"
		echo "  gem5=$GEM5_BIN"
		echo "  script=$GEM5_BAREMETAL_SCRIPT"
		echo "  elf=$ELF"
		echo "  cpu=$GEM5_CPU_TYPE"
		echo "  boom=$GEM5_BOOM"
		echo "  outdir=$GEM5_OUTDIR"
		echo ""

		cd "$RESEARCH_WORK"
		"$GEM5_BIN" \
			"${GEM5_GLOBAL_ARGS[@]}" \
			-d "$GEM5_OUTDIR" \
			"$GEM5_BAREMETAL_SCRIPT" \
			--bare-metal \
			--semihosting \
			--kernel "$ELF" \
			--cpu-type "$GEM5_CPU_TYPE" \
			--num-cpus 1 \
			--mem-size "$GEM5_MEM_SIZE" \
			"${GEM5_CACHE_ARGS[@]}" \
			--riscv-32bits \
			"${BOOM_ARGS[@]}" \
			"${GEM5_WORKLOAD_ARGS[@]}" \
			2>&1 | tee "$GEM5_OUTDIR/program.out"
		;;
	se)
		echo "gem5 SE run"
		echo "  gem5=$GEM5_BIN"
		echo "  script=$GEM5_SCRIPT"
		echo "  elf=$ELF"
		echo "  outdir=$GEM5_OUTDIR"
		echo ""

		cd "$RESEARCH_WORK/gem5/work"
		"$GEM5_BIN" "${GEM5_GLOBAL_ARGS[@]}" -d "$GEM5_OUTDIR" "$GEM5_SCRIPT" "$ELF" "${GEM5_WORKLOAD_ARGS[@]}" \
			2>&1 | tee "$GEM5_OUTDIR/program.out"
		;;
	*)
		echo "ERROR: unsupported mode '$MODE' (expected 'baremetal' or 'se')" >&2
		exit 1
		;;
esac

if [[ "${SKIP_BOTTLENECK_HTML:-1}" != "1" ]]; then
	python3 "$RESEARCH_WORK/scripts/gem5_bottleneck_html.py" "$GEM5_OUTDIR"
fi
