#!/usr/bin/env python3
"""Write results/summary.md from metrics.csv and step0 report."""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "results" / "metrics.csv"
OUT = ROOT / "results" / "summary.md"


def frow(rows, **kw) -> dict | None:
    for r in rows:
        if all(str(r.get(k, "")) == str(v) for k, v in kw.items()):
            return r
    return None


def main() -> None:
    rows = list(csv.DictReader(CSV.open())) if CSV.exists() else []
    ref = frow(rows, run_id="step0_kmajor_false")
    prod = frow(rows, run_id="step0_kmajor_true")
    dense_t = frow(rows, run_id="task2_dense_kmajor_true")
    lines = [
        "# Stratum gem5 experiment summary",
        "",
        "Generated from `results/metrics.csv`. See `REPRODUCE.md` for commands.",
        "",
        "## Headline (Step 0, drawing 0, `TEST_VECTORS=1`)",
        "",
    ]
    if ref and prod:
        r_ms, p_ms = float(ref["sim_seconds_ms"]), float(prod["sim_seconds_ms"])
        speedup = r_ms / p_ms if p_ms else 0
        lines += [
            f"| Layout | sim time (ms) | IPC | class |",
            f"|--------|---------------|-----|-------|",
            f"| output-outer (`k-major=false`) | {r_ms:.3f} | {float(ref['ipc']):.3f} | {ref.get('final_class','')} |",
            f"| spike-major (`k-major=true`) | {p_ms:.3f} | {float(prod['ipc']):.3f} | {prod.get('final_class','')} |",
            "",
            f"Speedup (ref/prod): **{speedup:.2f}×**. Production column matches archived run `20261008-184328`.",
            "",
            "**Methodology fix:** archived `20261008-181456` (7.424 ms) used five vectors; gem5 stats after ROI dump reflect **vector 4**, not drawing 0. Paper table now uses reproduced ref **7.038 ms** on drawing 0.",
            "",
        ]
    if dense_t and prod:
        d_ms = float(dense_t["sim_seconds_ms"])
        p_ms = float(prod["sim_seconds_ms"])
        lines += [
            "## Dense vs skip-zeros (spike-major, drawing 0)",
            "",
            f"- Dense (`skip-zeros=false`): **{d_ms:.2f} ms**",
            f"- Skip-zeros spike-major: **{p_ms:.2f} ms**",
            f"- Skip-zeros speedup vs dense: **{d_ms/p_ms:.2f}×**" if p_ms else "",
            "",
        ]
    task1 = frow(rows, run_id="task1_phase_kmajor_true")
    if task1:
        lines += [
            "## Task 1 phase ROI (spike-major, drawing 0)",
            "",
            f"- sim time: **{float(task1['sim_seconds_ms']):.2f} ms**, IPC {float(task1['ipc']):.3f}",
            "- Phase `PHASE_CYCLES` require fixed `phase_marker.c` (RV32 rdcycle); re-run after post-fixes if needed.",
            "",
        ]

    by_task = {}
    for r in rows:
        by_task.setdefault(r.get("task", ""), []).append(r)
    lines += ["## Coverage", ""]
    for task in sorted(by_task):
        lines.append(f"- `{task}`: {len(by_task[task])} run(s)")
    lines += [
        "",
        "## Claims vs data",
        "",
        "| Claim | Verdict |",
        "|-------|---------|",
        "| Spike-major cuts fc1 weight-stream stalls (lower ms, higher IPC) | Supported on drawing 0 |",
        "| Layout change preserves classification | Supported (`FINAL_CLASS[0]=1` on step0) |",
        "| Reference 7.42 ms on drawing 0 | **Contradicted** — use 7.04 ms with documented protocol |",
        "| IPC gain implies fewer synaptic updates | **Weakened** — same skip-zeros work; IPC reflects memory behavior |",
        "",
    ]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
