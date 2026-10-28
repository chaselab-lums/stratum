#!/usr/bin/env python3
"""Parse gem5 stats.txt into a flat metrics dict for CSV export."""
from __future__ import annotations

import re
from pathlib import Path

PATTERNS: dict[str, str] = {
    "simSeconds": r"^simSeconds\s+([0-9.eE+-]+)",
    "simTicks": r"^simTicks\s+([0-9.eE+-]+)",
    "simInsts": r"^simInsts\s+([0-9.eE+-]+)",
    "simOps": r"^simOps\s+([0-9.eE+-]+)",
    "hostSeconds": r"^hostSeconds\s+([0-9.eE+-]+)",
    "numCycles": r"^system\.cpu\.numCycles\s+([0-9.eE+-]+)",
    "ipc": r"^system\.cpu\.ipc\s+([0-9.eE+-]+)",
    "cpi": r"^system\.cpu\.cpi\s+([0-9.eE+-]+)",
    "l1d_accesses": r"^system\.cpu\.dcache\.overallAccesses::total\s+([0-9.eE+-]+)",
    "l1d_misses": r"^system\.cpu\.dcache\.overallMisses::total\s+([0-9.eE+-]+)",
    "l1d_miss_rate": r"^system\.cpu\.dcache\.overallMissRate::total\s+([0-9.eE+-]+)",
    "l1i_accesses": r"^system\.cpu\.icache\.overallAccesses::total\s+([0-9.eE+-]+)",
    "l1i_misses": r"^system\.cpu\.icache\.overallMisses::total\s+([0-9.eE+-]+)",
    "l1i_miss_rate": r"^system\.cpu\.icache\.overallMissRate::total\s+([0-9.eE+-]+)",
    "l2_accesses": r"^system\.l2\.overallAccesses::total\s+([0-9.eE+-]+)",
    "l2_misses": r"^system\.l2\.overallMisses::total\s+([0-9.eE+-]+)",
    "l2_miss_rate": r"^system\.l2\.overallMissRate::total\s+([0-9.eE+-]+)",
    "loads": r"^system\.cpu\.commitStats0\.numLoadInsts\s+([0-9.eE+-]+)",
    "stores": r"^system\.cpu\.commitStats0\.numStoreInsts\s+([0-9.eE+-]+)",
    "branches": r"^system\.cpu\.commitStats0\.numBranches\s+([0-9.eE+-]+)",
    "fp_ops": r"^system\.cpu\.commitStats0\.numFpInsts\s+([0-9.eE+-]+)",
    "branch_mispredicts": r"^system\.cpu\.branchPred\.condIncorrect\s+([0-9.eE+-]+)",
    "fetch_icache_stall_cycles": r"^system\.cpu\.fetchStats0\.icacheStallCycles\s+([0-9.eE+-]+)",
    "decode_blocked_cycles": r"^system\.cpu\.decode\.blockedCycles\s+([0-9.eE+-]+)",
    "rename_block_cycles": r"^system\.cpu\.rename\.blockCycles\s+([0-9.eE+-]+)",
    "rename_ROB_full_events": r"^system\.cpu\.rename\.ROBFullEvents\s+([0-9.eE+-]+)",
    "rename_IQ_full_events": r"^system\.cpu\.rename\.IQFullEvents\s+([0-9.eE+-]+)",
    "rename_LQ_full_events": r"^system\.cpu\.rename\.LQFullEvents\s+([0-9.eE+-]+)",
    "rename_SQ_full_events": r"^system\.cpu\.rename\.SQFullEvents\s+([0-9.eE+-]+)",
    "iew_block_cycles": r"^system\.cpu\.iew\.blockCycles\s+([0-9.eE+-]+)",
    "iew_iq_full_events": r"^system\.cpu\.iew\.iqFullEvents\s+([0-9.eE+-]+)",
    "iew_lsq_full_events": r"^system\.cpu\.iew\.lsqFullEvents\s+([0-9.eE+-]+)",
    "dcache_blocked_no_targets_cycles": r"^system\.cpu\.dcache\.blockedCycles::no_targets\s+([0-9.eE+-]+)",
    "sys_clk_period_ticks": r"^system\.clk_domain\.clock\s+([0-9.eE+-]+)",
    "cpu_clk_period_ticks": r"^system\.cpu_clk_domain\.clock\s+([0-9.eE+-]+)",
    "sim_freq_hz": r"^simFreq\s+([0-9.eE+-]+)",
}

LEGACY_L2_MISS = r"^system\.l2\.overall_miss_rate::total\s+([0-9.eE+-]+)"

# CSV columns populated from roi_metrics.json (after parse_stats)
METRIC_CSV_KEYS = [
    "sim_seconds_ms",
    "num_cycles",
    "ipc",
    "sim_insts",
    "l1d_accesses",
    "l1d_misses",
    "l1d_miss_rate",
    "l1i_accesses",
    "l1i_misses",
    "l1i_miss_rate",
    "l2_accesses",
    "l2_misses",
    "l2_miss_rate",
    "loads",
    "stores",
    "fetch_icache_stall_cycles",
    "decode_blocked_cycles",
    "rename_block_cycles",
    "rename_ROB_full_events",
    "rename_IQ_full_events",
    "rename_LQ_full_events",
    "rename_SQ_full_events",
    "iew_block_cycles",
    "iew_iq_full_events",
    "iew_lsq_full_events",
    "dcache_blocked_no_targets_cycles",
    "cpu_freq_hz",
]


def roi_stats_text(path: Path) -> str:
    text = path.read_text(errors="replace")
    parts = text.split("---------- Begin Simulation Statistics ----------")
    if len(parts) < 2:
        return text
    return parts[1].split("---------- End Simulation Statistics ----------")[0]


def parse_stats(path: Path) -> dict[str, float]:
    text = roi_stats_text(path)
    out: dict[str, float] = {}
    for key, pat in PATTERNS.items():
        m = re.search(pat, text, re.MULTILINE)
        if m:
            out[key] = float(m.group(1))
    if "l2_miss_rate" not in out:
        m = re.search(LEGACY_L2_MISS, text, re.MULTILINE)
        if m:
            out["l2_miss_rate"] = float(m.group(1))
    if out.get("sim_freq_hz") and out.get("cpu_clk_period_ticks"):
        out["cpu_freq_hz"] = out["sim_freq_hz"] / out["cpu_clk_period_ticks"]
    if out.get("sys_clk_period_ticks"):
        out["clk_period_ticks"] = out["sys_clk_period_ticks"]
    return out


def metrics_for_csv(m: dict[str, float]) -> dict[str, str]:
    """Flat string dict for CSV columns derived from parse_stats output."""
    row: dict[str, str] = {}
    if "simSeconds" in m:
        row["sim_seconds_ms"] = f"{m['simSeconds'] * 1000:.6f}"
    for k in ("numCycles", "simInsts"):
        if k in m:
            row[{"numCycles": "num_cycles", "simInsts": "sim_insts"}[k]] = str(int(m[k]))
    if "ipc" in m:
        row["ipc"] = f"{m['ipc']:.6f}"
    for k in METRIC_CSV_KEYS:
        if k in ("sim_seconds_ms", "num_cycles", "ipc", "sim_insts"):
            continue
        src = k
        if k == "cpu_freq_hz" and k in m:
            row[k] = f"{m[k]:.0f}"
        elif src in m:
            if "rate" in k:
                row[k] = f"{m[src]:.6f}"
            else:
                row[k] = str(int(m[src]))
    return row


def parse_final_class(program_out: Path, vec: int = 0) -> int | None:
    if not program_out.exists():
        return None
    pat = re.compile(rf"FINAL_CLASS\[{vec}\]\s+(\d+)")
    last = None
    for line in program_out.read_text(errors="replace").splitlines():
        m = pat.search(line)
        if m:
            last = int(m.group(1))
    return last


def parse_phase_cycles(program_out: Path) -> dict[str, int] | None:
    if not program_out.exists():
        return None
    pat = re.compile(
        r"PHASE_CYCLES vec=\d+ scan=(\d+) gather=(\d+) lif1=(\d+) fc2=(\d+) other=(\d+)"
    )
    last = None
    for line in program_out.read_text(errors="replace").splitlines():
        m = pat.search(line)
        if m:
            last = {
                "scan": int(m.group(1)),
                "gather": int(m.group(2)),
                "lif1": int(m.group(3)),
                "fc2": int(m.group(4)),
                "other": int(m.group(5)),
            }
    return last
