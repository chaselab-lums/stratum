#!/usr/bin/env python3
"""Print gem5 flags that trace a few hundred cycles inside the first full ROI.

The window starts 2000 cycles after the ROI begins, so the trace is the inner
kernel rather than reset, and it ends 400 cycles later.
"""

from __future__ import annotations

import sys
from pathlib import Path

from gem5_bottleneck_html import get, parse_stats, steady_sections

CYCLES_INTO_ROI = 2000
WINDOW_CYCLES = 400


def main() -> int:
    roi_end = False
    cycles_into = CYCLES_INTO_ROI
    window_cycles = WINDOW_CYCLES
    stats_path: Path | None = None
    args = sys.argv[1:]
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--roi-end":
            roi_end = True
        elif arg == "--cycles-into" and index + 1 < len(args):
            index += 1
            cycles_into = int(args[index])
        elif arg == "--window" and index + 1 < len(args):
            index += 1
            window_cycles = int(args[index])
        elif arg.startswith("-"):
            print(
                "Usage: o3_debug_window.py [--roi-end] [--cycles-into N] [--window N] /path/to/stats.txt",
                file=sys.stderr,
            )
            return 1
        else:
            stats_path = Path(arg)
        index += 1
    if stats_path is None:
        print(
            "Usage: o3_debug_window.py [--roi-end] [--cycles-into N] [--window N] /path/to/stats.txt",
            file=sys.stderr,
        )
        return 1
    if not stats_path.exists():
        print(f"ERROR: stats file not found: {stats_path}", file=sys.stderr)
        return 1
    steady = steady_sections(parse_stats(stats_path))
    if not steady:
        print(f"ERROR: no retired instructions in {stats_path}", file=sys.stderr)
        return 1
    section = steady[0]
    sim_ticks = get(section, "simTicks")
    final_tick = get(section, "finalTick")
    cycles = get(section, "system.cpu.numCycles")
    if sim_ticks <= 0 or cycles <= 0 or final_tick <= sim_ticks:
        print("ERROR: stats are missing simTicks, finalTick, or numCycles", file=sys.stderr)
        return 1
    if roi_end:
        # Stop just after the first timestep so a sweep does not resimulate all 25.
        print(f"--abs-max-tick {int(final_tick)}")
        return 0
    period = sim_ticks / cycles
    roi_start = final_tick - sim_ticks
    start = int(roi_start + cycles_into * period)
    end = int(start + window_cycles * period)
    # Line 1 is a gem5.opt flag. Line 2 is a config-script flag.
    print(f"--debug-start={start} --debug-end={end}")
    print(f"--abs-max-tick {end}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
