#!/usr/bin/env python3
"""Sanity-check synthetic input.h + spike fragment before gem5."""
from __future__ import annotations

import re
import sys
from pathlib import Path

N_STEPS = 300


def verify(build_dir: Path, mean_nnz: float) -> int:
    input_h = build_dir / "input.h"
    text = input_h.read_text(errors="replace")
    m = re.search(r'#include "([^"]+)"', text)
    if not m:
        raise RuntimeError("synthetic verify: no fragment include in input.h")
    frag = build_dir / m.group(1)
    if not frag.exists():
        raise RuntimeError(f"synthetic verify: missing {frag}")
    spikes = sum(1 for line in frag.read_text().splitlines() if line.strip().startswith("["))
    avg = spikes / N_STEPS
    lo = max(0.0, mean_nnz * 0.4)
    hi = mean_nnz * 2.5 + 5.0
    if avg < lo or avg > hi:
        raise RuntimeError(
            f"synthetic verify: mean nnz {avg:.2f} outside [{lo:.1f},{hi:.1f}] for target {mean_nnz}"
        )
    return spikes


if __name__ == "__main__":
    verify(Path(sys.argv[1]), float(sys.argv[2]))
    print("ok")
