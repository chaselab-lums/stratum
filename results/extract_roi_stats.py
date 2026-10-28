#!/usr/bin/env python3
"""Extract ROI metrics from a gem5 stats.txt (post m5_dump_reset_stats window)."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

def parse_stats(path: Path) -> dict[str, float]:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
    from gem5_metrics import parse_stats as _parse  # noqa: E402

    return _parse(path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("stats_txt", type=Path)
    p.add_argument("-o", "--json", type=Path)
    args = p.parse_args()
    metrics = parse_stats(args.stats_txt)
    if not metrics.get("simSeconds"):
        print(f"ERROR: no simSeconds in {args.stats_txt}", file=sys.stderr)
        sys.exit(1)
    line = json.dumps(metrics, indent=2)
    if args.json:
        args.json.write_text(line + "\n")
    print(line)


if __name__ == "__main__":
    main()
