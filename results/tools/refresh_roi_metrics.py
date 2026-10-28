#!/usr/bin/env python3
"""Re-parse gem5 stats.txt into roi_metrics.json for all runs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "results" / "lib"))
from gem5_metrics import parse_stats  # noqa: E402

RUNS = ROOT / "results" / "runs"


def main() -> None:
    n = 0
    for stats in sorted(RUNS.glob("*/gem5/stats.txt")):
        out = stats.parent.parent / "roi_metrics.json"
        m = parse_stats(stats)
        out.write_text(json.dumps(m, indent=2) + "\n")
        n += 1
    print(f"refreshed {n} roi_metrics.json files")


if __name__ == "__main__":
    main()
