#!/usr/bin/env python3

from __future__ import annotations

import math
import sys
from pathlib import Path


def write_text(path: Path, content: str) -> None:
    path.write_text(content)


def parse_stats_sections(stats_path: Path) -> list[dict[str, float]]:
    sections: list[dict[str, float]] = []
    current: dict[str, float] | None = None

    for raw_line in stats_path.read_text(errors="replace").splitlines():
        line = raw_line.strip()
        if line == "---------- Begin Simulation Statistics ----------":
            current = {}
            sections.append(current)
            continue
        if not line or current is None or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 2:
            continue
        key = parts[0]
        try:
            current[key] = float(parts[1])
        except ValueError:
            continue

    return sections


def svg_line_plot(series: list[float], title: str, y_label: str, output_path: Path) -> None:
    width = 900
    height = 360
    left = 70
    right = 20
    top = 35
    bottom = 50
    inner_width = width - left - right
    inner_height = height - top - bottom

    if not series:
        write_text(output_path, "<svg xmlns='http://www.w3.org/2000/svg' width='900' height='120'><text x='20' y='60'>No data</text></svg>\n")
        return

    y_max = max(series)
    y_min = min(series)
    if math.isclose(y_max, y_min):
        y_min = 0.0
        y_max = y_max if y_max != 0 else 1.0

    def x_pos(index: int) -> float:
        if len(series) == 1:
            return left + inner_width / 2
        return left + inner_width * index / (len(series) - 1)

    def y_pos(value: float) -> float:
        return top + inner_height * (1 - ((value - y_min) / (y_max - y_min)))

    polyline = " ".join(f"{x_pos(i):.2f},{y_pos(value):.2f}" for i, value in enumerate(series))
    points = "\n".join(
        f"<circle cx='{x_pos(i):.2f}' cy='{y_pos(value):.2f}' r='3' fill='#16537e' />"
        for i, value in enumerate(series)
    )
    x_ticks = "\n".join(
        f"<text x='{x_pos(i):.2f}' y='{height - 20}' font-size='11' text-anchor='middle'>{i}</text>"
        for i in range(len(series))
    )

    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>
<rect width='100%' height='100%' fill='white' />
<text x='{width / 2:.0f}' y='20' font-size='18' text-anchor='middle'>{title}</text>
<line x1='{left}' y1='{top}' x2='{left}' y2='{height - bottom}' stroke='black' />
<line x1='{left}' y1='{height - bottom}' x2='{width - right}' y2='{height - bottom}' stroke='black' />
<text x='20' y='{height / 2:.0f}' transform='rotate(-90 20 {height / 2:.0f})' font-size='12'>{y_label}</text>
<polyline fill='none' stroke='#16537e' stroke-width='2' points='{polyline}' />
{points}
{x_ticks}
<text x='{left}' y='{top + 12}' font-size='11'>{y_max:.4g}</text>
<text x='{left}' y='{height - bottom}' font-size='11'>{y_min:.4g}</text>
</svg>
"""
    write_text(output_path, svg)


def svg_multi_line_plot(
    series_map: list[tuple[str, list[float], str]],
    title: str,
    y_label: str,
    output_path: Path,
) -> None:
    width = 900
    height = 360
    left = 70
    right = 20
    top = 35
    bottom = 50
    inner_width = width - left - right
    inner_height = height - top - bottom

    non_empty = [series for _, series, _ in series_map if series]
    if not non_empty:
        write_text(output_path, "<svg xmlns='http://www.w3.org/2000/svg' width='900' height='120'><text x='20' y='60'>No data</text></svg>\n")
        return

    flattened = [value for series in non_empty for value in series]
    y_max = max(flattened)
    y_min = min(flattened)
    if math.isclose(y_max, y_min):
        y_min = 0.0
        y_max = y_max if y_max != 0 else 1.0

    point_count = max(len(series) for series in non_empty)

    def x_pos(index: int, length: int) -> float:
        if length <= 1:
            return left + inner_width / 2
        return left + inner_width * index / (length - 1)

    def y_pos(value: float) -> float:
        return top + inner_height * (1 - ((value - y_min) / (y_max - y_min)))

    polylines = []
    points = []
    legends = []
    for legend_index, (name, series, color) in enumerate(series_map):
        if not series:
            continue
        polyline = " ".join(
            f"{x_pos(i, len(series)):.2f},{y_pos(value):.2f}"
            for i, value in enumerate(series)
        )
        polylines.append(
            f"<polyline fill='none' stroke='{color}' stroke-width='2' points='{polyline}' />"
        )
        points.extend(
            f"<circle cx='{x_pos(i, len(series)):.2f}' cy='{y_pos(value):.2f}' r='2.5' fill='{color}' />"
            for i, value in enumerate(series)
        )
        legend_y = top + 18 * legend_index
        legends.append(f"<line x1='{width - 200}' y1='{legend_y}' x2='{width - 180}' y2='{legend_y}' stroke='{color}' stroke-width='2' />")
        legends.append(f"<text x='{width - 170}' y='{legend_y + 4}' font-size='11'>{name}</text>")

    x_ticks = "\n".join(
        f"<text x='{x_pos(i, point_count):.2f}' y='{height - 20}' font-size='11' text-anchor='middle'>{i}</text>"
        for i in range(point_count)
    )

    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>
<rect width='100%' height='100%' fill='white' />
<text x='{width / 2:.0f}' y='20' font-size='18' text-anchor='middle'>{title}</text>
<line x1='{left}' y1='{top}' x2='{left}' y2='{height - bottom}' stroke='black' />
<line x1='{left}' y1='{height - bottom}' x2='{width - right}' y2='{height - bottom}' stroke='black' />
<text x='20' y='{height / 2:.0f}' transform='rotate(-90 20 {height / 2:.0f})' font-size='12'>{y_label}</text>
{' '.join(polylines)}
{' '.join(points)}
{' '.join(legends)}
{x_ticks}
<text x='{left}' y='{top + 12}' font-size='11'>{y_max:.4g}</text>
<text x='{left}' y='{height - bottom}' font-size='11'>{y_min:.4g}</text>
</svg>
"""
    write_text(output_path, svg)


def write_roi_csv(outdir: Path, roi_sections: list[dict[str, float]]) -> None:
    header = [
        "roi_index",
        "sim_insts",
        "sim_ticks",
        "cpi",
        "ipc",
        "loads",
        "stores",
        "dcache_miss_rate",
    ]
    rows = [",".join(header)]
    for index, section in enumerate(roi_sections):
        rows.append(
            ",".join(
                [
                    str(index),
                    f"{section.get('simInsts', 0.0):.0f}",
                    f"{section.get('simTicks', 0.0):.0f}",
                    f"{section.get('system.cpu.cpi', 0.0):.6f}",
                    f"{section.get('system.cpu.ipc', 0.0):.6f}",
                    f"{section.get('system.cpu.commitStats0.numLoadInsts', 0.0):.0f}",
                    f"{section.get('system.cpu.commitStats0.numStoreInsts', 0.0):.0f}",
                    f"{section.get('system.cpu.dcache.overallMissRate::total', 0.0):.6f}",
                ]
            )
        )
    write_text(outdir / "roi_sections.csv", "\n".join(rows) + "\n")


def summarize_roi_sections(roi_sections: list[dict[str, float]]) -> list[dict[str, float]]:
    summaries = []
    for index, section in enumerate(roi_sections):
        summaries.append(
            {
                "roi_index": index,
                "sim_insts": section.get("simInsts", 0.0),
                "sim_ticks": section.get("simTicks", 0.0),
                "cpi": section.get("system.cpu.cpi", 0.0),
                "ipc": section.get("system.cpu.ipc", 0.0),
                "loads": section.get("system.cpu.commitStats0.numLoadInsts", 0.0),
                "stores": section.get("system.cpu.commitStats0.numStoreInsts", 0.0),
                "dcache_miss_rate": section.get(
                    "system.cpu.dcache.overallMissRate::total", 0.0
                ),
            }
        )
    return summaries


def markdown_roi_table(roi_summary: list[dict[str, float]]) -> str:
    header = (
        "| ROI | Instructions | Ticks | CPI | IPC | Loads | Stores | DCache Miss Rate |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    )
    rows = []
    for row in roi_summary:
        rows.append(
            "| {roi_index} | {sim_insts:.0f} | {sim_ticks:.0f} | {cpi:.4f} | {ipc:.4f} | {loads:.0f} | {stores:.0f} | {dcache_miss_rate:.6f} |".format(
                **row
            )
        )
    return header + "\n".join(rows)


def top_hotspots(roi_summary: list[dict[str, float]], limit: int = 5) -> list[dict[str, float]]:
    return sorted(
        roi_summary,
        key=lambda row: (row["sim_insts"], row["sim_ticks"], row["cpi"]),
        reverse=True,
    )[:limit]


def write_stats_report(input_dir: Path, output_dir: Path) -> None:
    stats_path = input_dir / "stats.txt"
    sections = parse_stats_sections(stats_path)
    roi_sections = [section for section in sections if section.get("simInsts", 0.0) > 0.0]
    roi_summary = summarize_roi_sections(roi_sections)

    sim_insts = [section.get("simInsts", 0.0) for section in roi_sections]
    cpi = [section.get("system.cpu.cpi", 0.0) for section in roi_sections]
    loads = [section.get("system.cpu.commitStats0.numLoadInsts", 0.0) for section in roi_sections]
    stores = [section.get("system.cpu.commitStats0.numStoreInsts", 0.0) for section in roi_sections]
    miss_rate = [section.get("system.cpu.dcache.overallMissRate::total", 0.0) for section in roi_sections]

    svg_line_plot(sim_insts, "ROI Instructions Per Step", "Instructions", output_dir / "roi_siminsts.svg")
    svg_line_plot(cpi, "ROI CPI Per Step", "CPI", output_dir / "roi_cpi.svg")
    svg_line_plot(miss_rate, "ROI DCache Miss Rate Per Step", "Miss Rate", output_dir / "roi_dcache_miss_rate.svg")
    svg_multi_line_plot(
        [
            ("Loads", loads, "#16537e"),
            ("Stores", stores, "#cc4c02"),
        ],
        "ROI Load/Store Mix Per Step",
        "Instructions",
        output_dir / "roi_load_store_mix.svg",
    )
    write_roi_csv(output_dir, roi_sections)

    total_steps = len(roi_sections)
    total_insts = sum(sim_insts)
    avg_cpi = sum(cpi) / total_steps if total_steps else 0.0
    avg_miss_rate = sum(miss_rate) / total_steps if total_steps else 0.0
    load_share = (sum(loads) / total_insts) if total_insts else 0.0
    store_share = (sum(stores) / total_insts) if total_insts else 0.0
    hottest = top_hotspots(roi_summary)

    hottest_lines = [
        "| ROI | Instructions | Ticks | CPI | DCache Miss Rate |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in hottest:
        hottest_lines.append(
            "| {roi_index} | {sim_insts:.0f} | {sim_ticks:.0f} | {cpi:.4f} | {dcache_miss_rate:.6f} |".format(
                **row
            )
        )

    report = f"""# gem5 Stats Report

- Output directory: {output_dir}
- ROI sections captured: {total_steps}
- Total ROI instructions: {total_insts:.0f}
- Average ROI CPI: {avg_cpi:.4f}
- Average ROI dcache miss rate: {avg_miss_rate:.6f}
- Load share of retired instructions: {load_share:.2%}
- Store share of retired instructions: {store_share:.2%}

## Files

- stats: {stats_path.name}
- instructions plot: roi_siminsts.svg
- cpi plot: roi_cpi.svg
- dcache miss rate plot: roi_dcache_miss_rate.svg
- load/store plot: roi_load_store_mix.svg
- ROI CSV: roi_sections.csv

## Hottest ROI Steps

{chr(10).join(hottest_lines)}

## Per-Step Summary

{markdown_roi_table(roi_summary)}

## Interpretation

- `simInsts` tells you the per-step dynamic work for `snn_forward_step`.
- `system.cpu.cpi` is the fastest way to see whether a step became more memory-bound or pipeline-bound.
- `system.cpu.commitStats0.numLoadInsts` and `system.cpu.commitStats0.numStoreInsts` show the memory intensity of each step.
- `system.cpu.dcache.overallMissRate::total` is the key cache-health metric for this workload.
"""
    write_text(output_dir / "report.md", report)


def write_o3_report(input_dir: Path, output_dir: Path) -> None:
    trace_path = input_dir / "trace.out"
    pipeview_path = input_dir / "pipeview.out"
    trace_lines = sum(1 for _ in trace_path.open(errors="replace")) if trace_path.exists() else 0
    pipe_preview = []
    if pipeview_path.exists():
        with pipeview_path.open(errors="replace") as pipe_file:
            for _ in range(20):
                line = pipe_file.readline()
                if not line:
                    break
                pipe_preview.append(line.rstrip())

    report = "# gem5 O3 Pipe Report\n\n"
    report += f"- Output directory: {output_dir}\n"
    report += f"- Trace lines: {trace_lines}\n"
    report += f"- Trace file: {trace_path.name}\n"
    report += f"- Pipeview file: {pipeview_path.name}\n\n"
    report += "## Notes\n\n"
    report += "- Use `pipeview.out` to inspect fetch, decode, rename, issue, execute, and commit timing for individual instructions.\n"
    report += "- The trace window is a few hundred cycles inside the first ROI, not the reset sequence. `profile.sh o3pipe` places it from the stats session.\n\n"
    report += "## Pipeview Preview\n\n```text\n"
    report += "\n".join(pipe_preview)
    report += "\n```\n"
    write_text(output_dir / "report.md", report)


def main() -> int:
    if len(sys.argv) not in (3, 4):
        print("Usage: gem5_profile_report.py [stats|o3pipe] /path/to/input_dir [/path/to/output_dir]", file=sys.stderr)
        return 1

    mode = sys.argv[1]
    input_dir = Path(sys.argv[2])
    output_dir = Path(sys.argv[3]) if len(sys.argv) == 4 else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if mode == "stats":
        write_stats_report(input_dir, output_dir)
    elif mode == "o3pipe":
        write_o3_report(input_dir, output_dir)
    else:
        print(f"Unsupported mode: {mode}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())