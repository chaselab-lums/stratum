#!/usr/bin/env python3
"""Compare gem5 sweep points and plot them on one roofline."""

from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

from gem5_bottleneck_html import (
    clock_ghz,
    flop_count,
    fu_count,
    get,
    ghz_label,
    parse_config,
    parse_stats,
    rate_label,
    render_roofline,
    steady_sections,
)

SPIKE_LINE = re.compile(r"^(?:\d+ \d+ \[|Spike counts per class:|FINAL_CLASS\[)")
REGION_LINE = re.compile(r"^0 \d+ \[")
PALETTE = [
    "#16537e",
    "#0f766e",
    "#b45309",
    "#c2410c",
    "#6d28d9",
    "#1d4e89",
    "#9a3412",
    "#3f6212",
    "#0369a1",
    "#0e7490",
    "#a16207",
    "#be123c",
    "#4338ca",
    "#047857",
    "#7c3aed",
    "#b91c1c",
]


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def spike_lines(program_out: Path) -> list[str]:
    if not program_out.exists():
        return []
    lines = []
    for raw in program_out.read_text(errors="replace").splitlines():
        line = raw.strip()
        if SPIKE_LINE.match(line):
            lines.append(line)
    return lines


def sweep_document(path: Path) -> dict:
    document = json.loads(path.read_text())
    entry = document.get("sweep")
    if "points" not in document and isinstance(entry, str):
        sweep_path = Path(entry)
        if not sweep_path.is_absolute():
            sweep_path = path.parent / sweep_path
        document = json.loads(sweep_path.read_text())
    return document


def load_profile(path: Path | None) -> tuple[list[dict[str, str]], list[dict[str, object]], dict[str, str]]:
    if path is None or not path.exists():
        return [], [], {}
    document = sweep_document(path)
    meta = {
        "title": str(document.get("title") or "Sweep"),
        "description": str(document.get("description") or ""),
    }
    charts = list(document.get("charts") or [])
    return list(document.get("points", [])), charts, meta


def dram_bytes(section: dict[str, float]) -> float:
    value = get(section, "system.mem_ctrls.requestorReadBytes::cpu.data")
    if value <= 0:
        value = get(section, "system.mem_ctrls.dram.bytesRead::cpu.data")
    return value


def point_record(outdir: Path, meta: dict[str, str], color: str) -> dict[str, object] | None:
    if not (outdir / "stats.txt").exists():
        return None
    sections = steady_sections(parse_stats(outdir / "stats.txt"))
    if not sections:
        return None
    section = sections[0]
    config = parse_config(outdir / "config.ini")
    flops, flop_kind = flop_count(section)
    seconds = get(section, "simSeconds")
    traffic = dram_bytes(section)
    achieved = flops / seconds if seconds > 0 else 0.0
    intensity = flops / traffic if traffic > 0 else 0.0
    knob_path = outdir / "knob.txt"
    label = meta.get("label") or (knob_path.read_text(errors="replace").strip() if knob_path.exists() else outdir.name)
    record: dict[str, object] = {
        "name": outdir.name,
        "label": label,
        "group": meta.get("group", ""),
        "flops": flops,
        "flop_kind": flop_kind,
        "seconds": seconds,
        "dram_bytes": traffic,
        "achieved": achieved,
        "intensity": intensity,
        "l2_misses": get(section, "system.l2.overallMisses::total"),
        "ipc": get(section, "system.cpu.ipc"),
        "insts": get(section, "simInsts"),
        "peak_bw": get(section, "system.mem_ctrls.dram.peakBW"),
        "ghz": clock_ghz(config, section),
        "flop_units": fu_count(config, "FloatAdd") + fu_count(config, "FloatMult"),
        "clock": ghz_label(clock_ghz(config, section)),
        "color": color,
        "config": config,
    }
    for key, value in meta.items():
        if key.endswith("_label") and value:
            record[key] = value
    return record


def bar_chart(title: str, y_label: str, rows: list[tuple[str, float, str]]) -> str:
    if not rows:
        return ""
    width, height = 860, 300
    left, right, top, bottom = 78, 20, 36, 56
    inner_w = width - left - right
    inner_h = height - top - bottom
    peak = max(value for _label, value, _color in rows) or 1.0
    y_max = peak * 1.18
    slot = inner_w / len(rows)
    bar_w = min(56, slot * 0.46)
    parts = [
        f"<svg viewBox='0 0 {width} {height}' width='100%' role='img' aria-label='{esc(title)}'>",
        f"<text x='{left}' y='20' font-size='14' fill='#1b1b1b'>{esc(title)}</text>",
        f"<text x='16' y='{(top + height - bottom) / 2:.0f}' font-size='12' fill='#444' text-anchor='middle' transform='rotate(-90 16 {(top + height - bottom) / 2:.0f})'>{esc(y_label)}</text>",
    ]
    for frac in (0.0, 0.5, 1.0):
        y = height - bottom - inner_h * (frac / 1.18 if frac else 0.0)
        if frac == 1.0:
            y = height - bottom - inner_h * (peak / y_max)
            tick = peak
        elif frac == 0.5:
            tick = peak / 2
            y = height - bottom - inner_h * (tick / y_max)
        else:
            tick = 0.0
            y = height - bottom
        parts.append(
            f"<line x1='{left}' y1='{y:.1f}' x2='{width - right}' y2='{y:.1f}' stroke='#ececec'/>"
        )
        parts.append(
            f"<text x='{left - 8}' y='{y + 4:.1f}' font-size='11' text-anchor='end' fill='#666'>{tick:.0f}</text>"
        )
    parts.extend([
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{height - bottom}' stroke='#222'/>",
        f"<line x1='{left}' y1='{height - bottom}' x2='{width - right}' y2='{height - bottom}' stroke='#222'/>",
    ])
    for index, (label, value, color) in enumerate(rows):
        cx = left + slot * (index + 0.5)
        bar_h = inner_h * (value / y_max) if y_max else 0.0
        y = height - bottom - bar_h
        parts.append(
            f"<rect x='{cx - bar_w / 2:.1f}' y='{y:.1f}' width='{bar_w:.1f}' height='{bar_h:.1f}' fill='{color}'/>"
        )
        parts.append(
            f"<text x='{cx:.1f}' y='{y - 6:.1f}' font-size='11' text-anchor='middle' fill='#1b1b1b'>{value:.1f}</text>"
        )
        parts.append(
            f"<text x='{cx:.1f}' y='{height - bottom + 18:.1f}' font-size='11' text-anchor='middle' fill='#444'>{esc(label)}</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


def chart_block(chart: dict[str, object], points: list[dict[str, object]]) -> str:
    field = str(chart.get("field") or "")
    order = [str(item) for item in chart.get("order") or []]
    rows: list[tuple[str, float, str]] = []
    for point in points:
        label = str(point.get(field) or "")
        if not label:
            continue
        rows.append((label, float(point["achieved"]) / 1e6, str(point["color"])))
    if order:
        rank = {name: index for index, name in enumerate(order)}
        rows.sort(key=lambda row: rank.get(row[0], len(order)))
    if not rows:
        return ""
    heading = str(chart.get("heading") or chart.get("title") or field)
    caption = str(chart.get("caption") or "")
    title = str(chart.get("title") or heading)
    return (
        f"<h2>{esc(heading)}</h2>"
        + (f"<p>{esc(caption)}</p>" if caption else "")
        + bar_chart(title, "MFLOP/s", rows)
    )


def render(
    points: list[dict[str, object]],
    spike_note: str,
    spike_ok: bool,
    charts: list[dict[str, object]],
    page: dict[str, str],
) -> str:
    headers = [
        "L2",
        "IPC",
        "Achieved",
        "Intensity",
        "L2 misses",
        "DRAM read",
    ]
    body = []
    for point in points:
        cells = [
            str(point["label"]),
            f"{float(point['ipc']):.3f}",
            rate_label(float(point["achieved"])),
            f"{float(point['intensity']):.3f} FLOP/byte",
            f"{float(point['l2_misses']):,.0f}",
            f"{float(point['dram_bytes']) / 1024:.0f} KB",
        ]
        body.append("<tr>" + "".join(f"<td>{esc(cell)}</td>" for cell in cells) + "</tr>")

    baseline = next((point for point in points if point["name"] == "baseline"), points[0])
    markers = [
        {
            "label": str(point["label"]),
            "intensity": float(point["intensity"]),
            "achieved": float(point["achieved"]),
            "color": str(point["color"]),
        }
        for point in points
        if float(point["intensity"]) > 0 and float(point["achieved"]) > 0
    ]
    roofline = ""
    if markers and float(baseline["dram_bytes"]) > 0 and float(baseline["seconds"]) > 0:
        kind = str(baseline["flop_kind"])
        if kind == "float":
            source = (
                "Work is retired FloatAdd and FloatMult, one FLOP each, plus FloatMultAcc as two FLOPs. "
                "The flat roof is one result per cycle on each pipelined FP add unit and FP multiply unit."
            )
        else:
            source = "Work counts each retired integer multiply-accumulate as two FLOPs."
        roofline = render_roofline(
            flops=float(baseline["flops"]),
            dram_bytes=float(baseline["dram_bytes"]),
            seconds=float(baseline["seconds"]),
            peak_bw_MBps=float(baseline["peak_bw"]),
            flop_units=float(baseline["flop_units"]),
            ghz=float(baseline["ghz"] or 0.0),
            source=source,
            markers=markers,
        )

    if not charts:
        charts = [
            {
                "field": "size_label",
                "heading": "Achieved rate by L2 size",
                "title": "L2 size, 8-way",
                "caption": "8-way L2.",
            },
            {
                "field": "assoc_label",
                "heading": "Achieved rate by L2 associativity",
                "title": "L2 associativity, 4 MiB",
                "caption": "4 MiB L2.",
                "order": ["2-way", "4-way", "8-way", "16-way"],
            },
        ]
    sections = "".join(chart_block(chart, points) for chart in charts)

    tone = "#e8f5e9" if spike_ok else "#fff6dd"
    clock = str(baseline.get("clock", ""))
    title = page.get("title") or "Sweep"
    description = page.get("description") or (
        "Each point is the first stats region of snn_forward_step on gem5 RiscvO3CPU"
        + (f" at {clock}" if clock else "")
        + ". The roofs are the baseline machine."
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{esc(title)}</title>
  <style>
    html {{ color-scheme: light; }}
    html, body {{ background: #fff; }}
    body {{ margin: 0 auto; max-width: 1100px; padding: 32px 24px 64px; font: 16px/1.5 "Segoe UI", Helvetica, Arial, sans-serif; color: #1b1b1b; }}
    h1 {{ font-size: 28px; margin: 0 0 8px; }}
    h2 {{ font-size: 20px; margin: 28px 0 8px; }}
    p {{ color: #444; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid #e2e2e2; vertical-align: top; }}
    th {{ font-size: 12px; color: #5c5c5c; }}
    td:nth-child(n+2) {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .note {{ background: {tone}; border: 1px solid #e2e2e2; padding: 12px 14px; }}
    code {{ background: #f3f3f3; padding: 0.1rem 0.3rem; }}
  </style>
</head>
<body>
  <h1>{esc(title)}</h1>
  <p>{esc(description)} Each point is the first image on gem5 RiscvO3CPU{(' at ' + esc(clock)) if clock else ''}. The roofs are the baseline machine.</p>
  <div class="note">{esc(spike_note)}</div>
  {roofline}
  {sections}
  <h2>Counters</h2>
  <table>
    <thead><tr>{''.join(f'<th>{esc(header)}</th>' for header in headers)}</tr></thead>
    <tbody>{''.join(body)}</tbody>
  </table>
  <p>The point list is <code>configs/profile/sweep.json</code>.</p>
</body>
</html>
"""


def assemble(sweep_root: Path, profile_path: Path | None) -> tuple[str, str, bool]:
    catalog, charts, page = load_profile(profile_path)
    by_name = {point["name"]: point for point in catalog}
    order = [point["name"] for point in catalog]

    if catalog:
        dirs = [sweep_root / name for name in order if (sweep_root / name / "stats.txt").exists()]
    else:
        dirs = [path for path in sweep_root.iterdir() if path.is_dir() and (path / "stats.txt").exists()]
        dirs.sort(key=lambda path: path.name)

    points: list[dict[str, object]] = []
    spikes: dict[str, list[str]] = {}
    for index, outdir in enumerate(dirs):
        meta = by_name.get(outdir.name, {"label": outdir.name, "group": ""})
        record = point_record(outdir, meta, PALETTE[index % len(PALETTE)])
        if record is None:
            continue
        points.append(record)
        lines = spike_lines(outdir / "program.out")
        region = [line for line in lines if REGION_LINE.match(line)]
        spikes[outdir.name] = region or lines
    if not points:
        note = f"No sweep stats under {sweep_root}."
        return "", note, False

    baseline_spikes = spikes.get("baseline", [])
    mismatched = []
    missing = []
    for name, lines in spikes.items():
        if not lines:
            missing.append(name)
        elif baseline_spikes and lines != baseline_spikes:
            mismatched.append(name)
    if not baseline_spikes:
        spike_ok = False
        spike_note = "Baseline program.out has no spike lines, so the functional check could not run."
    elif missing or mismatched:
        spike_ok = False
        parts = []
        if missing:
            parts.append("no spike lines in " + ", ".join(missing))
        if mismatched:
            parts.append("spike output differs in " + ", ".join(mismatched))
        spike_note = "Functional check failed: " + "; ".join(parts) + "."
    else:
        spike_ok = True
        spike_note = (
            f"Functional check passed. All {len(spikes)} configs printed the same "
            f"{len(baseline_spikes)} frames of the first image."
        )

    return render(points, spike_note, spike_ok, charts, page), spike_note, spike_ok


def main() -> int:
    if len(sys.argv) not in (2, 3):
        print("Usage: gem5_sweep_report.py /path/to/profiles/sweep [profile.json]", file=sys.stderr)
        return 1
    sweep_root = Path(sys.argv[1])
    profile_path = Path(sys.argv[2]) if len(sys.argv) == 3 else None
    _page, spike_note, spike_ok = assemble(sweep_root, profile_path)
    print(spike_note)
    return 0 if spike_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
