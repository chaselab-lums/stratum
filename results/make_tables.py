#!/usr/bin/env python3
"""Generate LaTeX fragments and PNG plots from results/metrics.csv."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "results" / "metrics.csv"
OUT = ROOT / "results" / "tables"
PLOTS = ROOT / "results" / "plots"


def load_rows() -> list[dict[str, str]]:
    if not CSV_PATH.exists():
        return []
    with CSV_PATH.open(newline="") as f:
        return list(csv.DictReader(f))


def pick(rows: list[dict], **kw) -> dict | None:
    for r in rows:
        ok = True
        for k, v in kw.items():
            if str(r.get(k, "")) != str(v):
                ok = False
                break
        if ok:
            return r
    return None


def latex_tab_roi(rows: list[dict]) -> str:
    ref = pick(rows, run_id="step0_kmajor_false") or pick(rows, task="step0", k_major="false")
    prod = pick(rows, run_id="step0_kmajor_true") or pick(rows, task="step0", k_major="true")
    if not ref or not prod:
        return "% missing step0 rows in metrics.csv\n"
    def ms(r: dict) -> str:
        return f"{float(r['sim_seconds_ms']):.2f}"
    def ipc(r: dict) -> str:
        return f"{float(r['ipc']):.2f}"
    return f"""\\begin{{tabular}}{{lrr}}
\\toprule
 & Output-outer ref. & Spike-major \\\\
\\midrule
Sim.\\ time & {ms(ref)}\\,ms & {ms(prod)}\\,ms \\\\
IPC & {ipc(ref)} & {ipc(prod)} \\\\
Predicted class & \\multicolumn{{2}}{{c}}{{{ref.get('final_class','?')} (both)}} \\\\
\\bottomrule
\\end{{tabular}}
"""


def try_plots(rows: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        (PLOTS / "README.txt").write_text("Install matplotlib to generate plots.\n")
        return
    PLOTS.mkdir(parents=True, exist_ok=True)
    # L2 sweep production layout
    l2_rows = [
        r
        for r in rows
        if r.get("task") == "task3_l2" and r.get("k_major") == "true"
    ]
    if l2_rows:
        l2_rows.sort(key=lambda r: r["gem5_l2"])
        xs = [r["gem5_l2"] for r in l2_rows]
        ys = [float(r["sim_seconds_ms"]) for r in l2_rows]
        plt.figure(figsize=(5, 3))
        plt.bar(xs, ys, color="#22c55e")
        plt.ylabel("ROI sim time (ms)")
        plt.title("Spike-major vs L2 size (drawing 0)")
        plt.tight_layout()
        plt.savefig(PLOTS / "l2_sweep_spike_major.png", dpi=150)
        plt.close()
    # Drawings
    draw_rows = [
        r for r in rows if r.get("task") == "task4_drawings" and r.get("k_major") == "true"
    ]
    if draw_rows:
        draw_rows.sort(key=lambda r: int(r["drawing"]))
        xs = [r["drawing"] for r in draw_rows]
        ys = [float(r["sim_seconds_ms"]) for r in draw_rows]
        plt.figure(figsize=(5, 3))
        plt.plot(xs, ys, "o-", color="#22c55e", label="spike-major")
        ref = [
            r
            for r in rows
            if r.get("task") == "task4_drawings" and r.get("k_major") == "false"
        ]
        ref.sort(key=lambda r: int(r["drawing"]))
        if len(ref) == len(draw_rows):
            plt.plot(
                [r["drawing"] for r in ref],
                [float(r["sim_seconds_ms"]) for r in ref],
                "s--",
                color="#f59e0b",
                label="output-outer",
            )
        plt.xlabel("drawing index")
        plt.ylabel("ROI sim time (ms)")
        plt.legend()
        plt.title("Five N-MNIST drawings")
        plt.tight_layout()
        plt.savefig(PLOTS / "drawings_latency.png", dpi=150)
        plt.close()


def main() -> None:
    rows = load_rows()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tab_roi.tex").write_text(latex_tab_roi(rows))
    summary = {
        "n_rows": len(rows),
        "tasks": sorted({r.get("task", "") for r in rows}),
    }
    (OUT / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    try_plots(rows)
    print(f"wrote {OUT} and {PLOTS}")


if __name__ == "__main__":
    main()
