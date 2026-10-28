#!/usr/bin/env python3
"""Insert calls to @snn_phase_marker at fc1/fc2 region boundaries in network.ll."""
from __future__ import annotations

import re
import sys
from pathlib import Path

MARKERS: list[tuple[str, int]] = [
    (r"^46:\s+; preds = %30", 1),
    (r"^68:\s+; preds = %47", 2),
    (r"^88:\s+; preds = %70", 3),
]

DECL = "declare void @snn_phase_marker(i32)\n"


def inject(ll_path: Path, out_path: Path) -> None:
    lines = ll_path.read_text(errors="replace").splitlines()
    out: list[str] = []
    inserted_decl = False
    for line in lines:
        if not inserted_decl and line.startswith("define void @snn_forward_step"):
            out.append(DECL.strip())
            inserted_decl = True
        out.append(line)
        for pat, phase_id in MARKERS:
            if re.match(pat, line.strip()):
                out.append(f"  call void @snn_phase_marker(i32 {phase_id})")
                break
    if not inserted_decl:
        out.insert(4, DECL.strip())
    for i, line in enumerate(out):
        if line.startswith("define void @snn_forward_step"):
            j = i + 1
            while j < len(out) and "{" not in out[j]:
                j += 1
            if j < len(out):
                out.insert(j + 1, "  call void @snn_phase_marker(i32 0)")
            break
    out_path.write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    inject(Path(sys.argv[1]), Path(sys.argv[2]))
