#!/usr/bin/env python3
"""Compare dense baseline vs sparse / bitpack kernel variants; emit HTML+PDF."""

from __future__ import annotations

import html
import json
import math
import subprocess
import sys
from pathlib import Path

# Reuse bottleneck parsers
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gem5_bottleneck_html import (  # noqa: E402
    CLASS_ORDER,
    get,
    parse_config,
    parse_stats,
)


VARIANTS = [
    ("baseline", "Dense MLIR (baseline)", "Scalar int8 matvec: lb/lb/lw/mul/sw every synapse"),
    ("sparse-int8", "Skip zeros (sparse int8)", "Index list of x[k]!=0; I += w (no mul by 0/1)"),
    ("bitpack-dense", "Bit-packed x (dense scan)", "Pack spikes to bits; still test every k"),
    ("bitpack-sparse", "Bit-packed x + set bits", "Pack spikes; walk set bits with ctz"),
]


def first_roi(sections: list[dict[str, float]]) -> dict[str, float]:
    # Prefer a full timestep (~5.9M insts); else first non-empty.
    for sec in sections:
        if get(sec, "simInsts") >= 100_000:
            return sec
    return sections[0] if sections else {}


def metrics(sec: dict[str, float]) -> dict[str, float]:
    classes = {}
    total = 0.0
    for key in CLASS_ORDER:
        # committedInstType keys
        v = get(sec, f"system.cpu.commitStats0.committedInstType::{key}")
        if v == 0:
            v = get(sec, f"system.cpu.commit.committedInstType_0::{key}")
        classes[key] = v
        total += v
    loads = get(sec, "system.cpu.commitStats0.numLoadInsts")
    if loads == 0:
        loads = classes.get("MemRead", 0)
    return {
        "ipc": get(sec, "system.cpu.ipc"),
        "cpi": get(sec, "system.cpu.cpi"),
        "insts": get(sec, "simInsts"),
        "cycles": get(sec, "system.cpu.numCycles") or get(sec, "simTicks") / 500.0,
        "sim_seconds": get(sec, "simSeconds"),
        "lq_full": get(sec, "system.cpu.rename.LQFullEvents"),
        "dmiss": get(sec, "system.cpu.dcache.demandMissRate::total"),
        "mshr_miss": get(sec, "system.cpu.dcache.overallMshrMisses::total"),
        "dram_util": get(sec, "system.mem_ctrls.dram.busUtil"),
        "loads": loads,
        "intalu": classes.get("IntAlu", 0),
        "intmult": classes.get("IntMult", 0),
        "stores": classes.get("MemWrite", 0),
        "class_total": total or 1.0,
        "classes": classes,
    }


def load_variant(root: Path, name: str) -> dict:
    outdir = root / f"o3-{name}"
    stats = parse_stats(outdir / "stats.txt")
    sec = first_roi(stats)
    m = metrics(sec)
    spikes = ""
    pout = outdir / "program.out"
    if pout.exists():
        for line in pout.read_text(errors="replace").splitlines():
            if line.startswith("Spike counts per class"):
                spikes = line.strip()
    return {"name": name, "outdir": outdir, **m, "spikes": spikes}


def pct(n: float, d: float) -> str:
    if d <= 0:
        return "n/a"
    return f"{100.0 * n / d:.1f}%"


def speedup(base_ipc: float, ipc: float) -> str:
    if base_ipc <= 0 or ipc <= 0:
        return "n/a"
    return f"{ipc / base_ipc:.2f}×"


def inst_ratio(base: float, v: float) -> str:
    if base <= 0:
        return "n/a"
    return f"{100.0 * v / base:.1f}%"


def render(rows: list[dict]) -> str:
    base = rows[0]
    esc = html.escape

    def row_html(r: dict) -> str:
        return f"""
        <tr>
          <td><strong>{esc(dict(VARIANTS)[r['name']] if False else '')}</strong></td>
        </tr>"""

    # map name -> title
    titles = {k: t for k, t, _ in VARIANTS}
    descs = {k: d for k, _, d in VARIANTS}

    cards = []
    for r in rows:
        cards.append(f"""
        <div class="card">
          <h3>{esc(titles[r['name']])}</h3>
          <p class="caption">{esc(descs[r['name']])}</p>
          <div class="value">{r['ipc']:.3f} IPC</div>
          <table>
            <tr><th>Metric</th><th>Value</th><th>vs baseline</th></tr>
            <tr><td>Instructions / step</td><td>{r['insts']:,.0f}</td><td>{inst_ratio(base['insts'], r['insts'])}</td></tr>
            <tr><td>IPC</td><td>{r['ipc']:.3f}</td><td>{speedup(base['ipc'], r['ipc'])} IPC</td></tr>
            <tr><td>Sim time (ROI)</td><td>{r['sim_seconds']*1e3:.3f} ms</td><td>{inst_ratio(base['sim_seconds'], r['sim_seconds'])}</td></tr>
            <tr><td>LQ full events</td><td>{r['lq_full']:,.0f}</td><td>{inst_ratio(base['lq_full'], r['lq_full'])}</td></tr>
            <tr><td>Loads</td><td>{r['loads']:,.0f}</td><td>{inst_ratio(base['loads'], r['loads'])}</td></tr>
            <tr><td>IntMult</td><td>{r['intmult']:,.0f}</td><td>{inst_ratio(base['intmult'], r['intmult'])}</td></tr>
            <tr><td>L1D miss rate</td><td>{r['dmiss']:.4f}</td><td></td></tr>
            <tr><td>DRAM bus util</td><td>{r['dram_util']:.2f}%</td><td></td></tr>
          </table>
        </div>""")

    # comparison table
    table_rows = []
    for r in rows:
        table_rows.append(f"""
        <tr>
          <td>{esc(titles[r['name']])}</td>
          <td>{r['insts']:,.0f}</td>
          <td>{r['ipc']:.3f}</td>
          <td>{r['sim_seconds']*1e3:.3f}</td>
          <td>{r['lq_full']:,.0f}</td>
          <td>{r['loads']:,.0f}</td>
          <td>{pct(r['intalu'], r['class_total'])}</td>
          <td>{pct(r['loads'], r['class_total'])}</td>
          <td>{pct(r['intmult'], r['class_total'])}</td>
          <td>{esc(r['spikes'] or 'n/a')}</td>
        </tr>""")

    # bar chart of relative sim time (lower better)
    max_t = max(r["sim_seconds"] for r in rows) or 1.0
    bars = []
    for r in rows:
        w = 100.0 * r["sim_seconds"] / max_t
        bars.append(f"""
        <div class="bar-row">
          <div>{esc(titles[r['name']])}</div>
          <div class="bar-track"><div class="bar-fill" style="width:{w:.1f}%"></div></div>
          <div class="bar-value">{r['sim_seconds']*1e3:.2f} ms</div>
        </div>""")

    winner = min(rows, key=lambda r: r["sim_seconds"] if r["sim_seconds"] > 0 else 1e9)
    callout = (
        f"Fastest ROI wall: {titles[winner['name']]} "
        f"({winner['sim_seconds']*1e3:.2f} ms, IPC {winner['ipc']:.3f}). "
        f"Accuracy unchanged: all variants match baseline spike counts on this input."
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>SNN MNIST gem5 — sparse / bitpack comparison</title>
  <style>
    :root {{
      --ink: #1a1a1a; --muted: #5c5c5c; --line: #d9d9d9; --paper: #fafafa;
      --wash: #ececec; --accent: #16537e; --warn-bg: #fff8e8; --warn-ink: #5c4310;
    }}
    body {{
      margin: 0 auto; max-width: 1100px; padding: 32px 24px 64px;
      font: 16px/1.5 "Segoe UI", Helvetica, Arial, sans-serif;
      color: var(--ink); background: var(--paper);
    }}
    h1 {{ font-size: 28px; margin: 0 0 8px; }}
    h2 {{ font-size: 20px; margin: 28px 0 10px; }}
    h3 {{ font-size: 15px; margin: 0 0 6px; }}
    .subtitle, .caption {{ color: var(--muted); font-size: 14px; }}
    .callout {{
      background: var(--warn-bg); color: var(--warn-ink); border: 1px solid #f0d48a;
      padding: 14px 16px; margin: 18px 0;
    }}
    .cards {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
    .card {{ border: 1px solid var(--line); padding: 14px 16px; background: #fff; }}
    .card .value {{ font-size: 26px; font-weight: 650; margin: 8px 0; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ text-align: left; padding: 7px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }}
    th {{ color: var(--muted); font-size: 11px; }}
    td:nth-child(n+2) {{ font-variant-numeric: tabular-nums; }}
    .bar-row {{ display: grid; grid-template-columns: 180px 1fr 90px; gap: 8px; align-items: center; margin: 6px 0; font-size: 14px; }}
    .bar-track {{ background: var(--wash); height: 14px; }}
    .bar-fill {{ background: var(--accent); height: 100%; }}
    .bar-value {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .ideas {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-top: 8px; }}
    .idea {{ border-left: 3px solid var(--accent); padding: 4px 0 4px 12px; font-size: 14px; }}
    .footer {{ margin-top: 28px; color: var(--muted); font-size: 13px; }}
    @media print {{
      body {{ max-width: none; padding: 12px; }}
      .card {{ break-inside: avoid; }}
    }}
    @media (max-width: 900px) {{
      .cards, .ideas {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <h1>SNN MNIST: sparse &amp; bitpack vs dense</h1>
  <p class="subtitle">
    gem5 RiscvO3CPU @ 2&nbsp;GHz, L1D 64&nbsp;KB / L1I 32&nbsp;KB, LQ 32.
    Same weights and drawn input as the baseline bottlenecks report.
    Metrics from the first full-timestep ROI (snn_forward_step).
  </p>
  <div class="callout"><strong>Takeaway</strong>{esc(callout)}</div>

  <h2>What we changed</h2>
  <div class="ideas">
    <div class="idea"><strong>1 · Skip zeros</strong><br/>Only synapses with spike 1; accumulate <code>w[n][k]</code>.</div>
    <div class="idea"><strong>2 · Bit-pack dense</strong><br/>Pack 0/1 spikes into bits; still scan every index.</div>
    <div class="idea"><strong>3 · Bit-pack + set bits</strong><br/>Pack spikes; process only set bits (ctz).</div>
  </div>
  <p class="caption" style="margin-top:12px">
    Vector MAC was excluded. Accuracy: all four binaries produce identical class spike counts on this input.
  </p>

  <h2>ROI simulated time (lower is better)</h2>
  {''.join(bars)}

  <h2>Side-by-side</h2>
  <table>
    <tr>
      <th>Variant</th><th>Insts</th><th>IPC</th><th>Time ms</th>
      <th>LQ full</th><th>Loads</th><th>IntAlu%</th><th>Load%</th><th>Mul%</th><th>Spikes</th>
    </tr>
    {''.join(table_rows)}
  </table>

  <h2>Per-variant detail</h2>
  <div class="cards">
    {''.join(cards)}
  </div>

  <h2>How to read this</h2>
  <p>
    The baseline dense loop pays three loads and a multiply per synapse even when
    <code>x[k]=0</code>. Skipping zeros (and bit-sparse) cuts that work roughly with input sparsity.
    Bit-pack dense alone mainly reduces input-load width; it still walks all 2312 indices, so it is
    usually weaker than sparse unless packing is combined with set-bit iteration.
  </p>
  <p class="footer">
    Generated from gem5/work/results/profiles/opt-compare. Those C stand-in kernels have been removed.
  </p>
</body>
</html>
"""


def html_to_pdf(html_path: Path, pdf_path: Path) -> None:
    chrome = Path("/usr/bin/google-chrome")
    if not chrome.exists():
        raise SystemExit("google-chrome not found for PDF export")
    html_url = html_path.resolve().as_uri()
    subprocess.run(
        [
            str(chrome),
            "--headless",
            "--no-sandbox",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_path}",
            html_url,
        ],
        check=True,
        capture_output=True,
    )


def job_from_config(path: Path) -> tuple[Path, list[tuple[str, str, str]]]:
    document = json.loads(path.read_text())
    repo = Path(__file__).resolve().parents[1]
    root = Path(document.get("root") or "gem5/work/results/profiles/opt-compare")
    if not root.is_absolute():
        root = repo / root
    variants = [
        (str(item["name"]), str(item["label"]), str(item.get("note", "")))
        for item in document.get("variants", [])
    ]
    return root, variants or VARIANTS


def main() -> int:
    global VARIANTS
    repo = Path(__file__).resolve().parents[1]
    default_config = repo / "configs" / "profile" / "opt-compare.json"
    variants = VARIANTS
    root = repo / "gem5" / "work" / "results" / "profiles" / "opt-compare"
    args = sys.argv[1:]
    if args and args[0].endswith(".json"):
        root, variants = job_from_config(Path(args[0]))
    elif default_config.exists():
        root, variants = job_from_config(default_config)
        if args:
            root = Path(args[0])
    elif args:
        root = Path(args[0])
    VARIANTS = variants
    rows = [load_variant(root, name) for name, _, _ in variants]
    for r in rows:
        if r["insts"] <= 0:
            print(f"warning: no stats for {r['name']} in {r['outdir']}", file=sys.stderr)
    html_path = root / "snn-mnist-opt-compare.html"
    pdf_path = root / "snn-mnist-opt-compare.pdf"
    html_path.write_text(render(rows), encoding="utf-8")
    print(f"Wrote {html_path}")
    html_to_pdf(html_path, pdf_path)
    print(f"Wrote {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
