#!/usr/bin/env python3

from __future__ import annotations

import csv
import sys
from pathlib import Path


def read_rows(csv_path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            rows.append(
                {
                    "roi_index": float(row["roi_index"]),
                    "sim_insts": float(row["sim_insts"]),
                    "sim_ticks": float(row["sim_ticks"]),
                    "cpi": float(row["cpi"]),
                    "ipc": float(row["ipc"]),
                    "loads": float(row["loads"]),
                    "stores": float(row["stores"]),
                    "dcache_miss_rate": float(row["dcache_miss_rate"]),
                }
            )
    return rows


def format_table(rows: list[dict[str, float]]) -> str:
    lines = [
        "| ROI | Instructions | Ticks | CPI | IPC | Loads | Stores | DCache Miss Rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {roi_index:.0f} | {sim_insts:.0f} | {sim_ticks:.0f} | {cpi:.4f} | {ipc:.4f} | {loads:.0f} | {stores:.0f} | {dcache_miss_rate:.6f} |".format(
                **row
            )
        )
    return "\n".join(lines)


def main() -> int:
    if len(sys.argv) not in (2, 3):
        print("Usage: roi_hotspots.py /path/to/roi_sections.csv [top_n]", file=sys.stderr)
        return 1

    csv_path = Path(sys.argv[1])
    top_n = int(sys.argv[2]) if len(sys.argv) == 3 else 5
    rows = read_rows(csv_path)
    hottest = sorted(rows, key=lambda row: (row["sim_insts"], row["sim_ticks"], row["cpi"]), reverse=True)[:top_n]
    print(format_table(hottest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())