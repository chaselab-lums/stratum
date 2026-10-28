#!/usr/bin/env python3

from __future__ import annotations

import html
import os
import sys
from pathlib import Path


def rel_link(from_dir: Path, target: Path) -> str:
    return os.path.relpath(target, start=from_dir).replace(os.sep, "/")


def link_if_exists(base: Path, dashboard_dir: Path, label: str) -> str:
    if base.exists():
        href = rel_link(dashboard_dir, base)
        return f"<li><a href='{html.escape(href)}'>{html.escape(label)}</a></li>"
    return f"<li>{html.escape(label)}: missing</li>"


def image_if_exists(base: Path, dashboard_dir: Path, label: str) -> str:
    if base.exists():
        href = rel_link(dashboard_dir, base)
        return (
            f"<figure><img src='{html.escape(href)}' alt='{html.escape(label)}' />"
            f"<figcaption>{html.escape(label)}</figcaption></figure>"
        )
    return f"<p>{html.escape(label)}: missing</p>"


def read_report_excerpt(report_path: Path) -> str:
    if not report_path.exists():
        return "<p>Report not generated yet.</p>"

    lines = report_path.read_text(errors="replace").splitlines()
    excerpt = []
    for line in lines[2:8]:
        if line.startswith("- "):
            excerpt.append(f"<li>{html.escape(line[2:])}</li>")
    if not excerpt:
        return ""
    return "<div class='summary'><strong>Summary</strong><ul>" + "".join(excerpt) + "</ul></div>"


def build_section(
    title: str,
    report_dir: Path,
    dashboard_dir: Path,
    extras: list[tuple[Path, str]],
    images: list[tuple[str, str]] | None = None,
    note: str | None = None,
) -> str:
    report_path = report_dir / "report.md"
    items = [link_if_exists(report_dir / "report.md", dashboard_dir, "report.md")]
    for path, label in extras:
        items.append(link_if_exists(path, dashboard_dir, label))
    gallery = ""
    if images:
        gallery = "<div class='gallery'>" + "".join(
            image_if_exists(report_dir / filename, dashboard_dir, label)
            for filename, label in images
        ) + "</div>"
    note_html = f"<p class='note'>{html.escape(note)}</p>" if note else ""
    summary_html = read_report_excerpt(report_path)
    return (
        f"<section><h2>{html.escape(title)}</h2>"
        f"<p>Directory: <code>{html.escape(str(report_dir))}</code></p>"
        f"{note_html}{summary_html}<ul>{''.join(items)}</ul>{gallery}</section>"
    )


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: profile_dashboard.py /path/to/profiles dashboard-name", file=sys.stderr)
        return 1

    profile_root = Path(sys.argv[1])
    dashboard_name = sys.argv[2]

    stats_dir = profile_root / "stats" / "inference-stats"
    o3_dir = profile_root / "o3pipe" / "pipeline-detail"
    sweep_dir = profile_root / "sweep"
    dashboard_file = profile_root / f"dashboard-{dashboard_name}.html"
    dashboard_dir = dashboard_file.parent

    html_doc = """<!DOCTYPE html>
<html lang='en'>
<head>
  <meta charset='utf-8'>
  <title>gem5 Profile Dashboard</title>
  <style>
    body { font-family: Georgia, serif; margin: 2rem auto; max-width: 960px; line-height: 1.5; color: #1b1b1b; }
    h1, h2 { font-family: 'Trebuchet MS', sans-serif; }
    code { background: #f3f3f3; padding: 0.1rem 0.3rem; }
    section { border-top: 1px solid #d7d7d7; padding-top: 1rem; margin-top: 1rem; }
    a { color: #16537e; }
        .gallery { display: grid; grid-template-columns: 1fr; gap: 1rem; margin-top: 1rem; }
        figure { margin: 0; padding: 0.75rem; background: #fafafa; border: 1px solid #e2e2e2; }
        img { max-width: 100%; height: auto; display: block; background: white; }
        figcaption { font-size: 0.95rem; margin-top: 0.5rem; color: #444; }
        .note { color: #5c3b00; background: #fff5d6; border: 1px solid #f0d48a; padding: 0.75rem; }
        .summary { background: #f7f7f7; border: 1px solid #e0e0e0; padding: 0.75rem; margin-bottom: 1rem; }
        .summary ul { margin: 0.5rem 0 0 1.25rem; }
  </style>
</head>
<body>
"""
    html_doc += f"<h1>gem5 Profile Dashboard</h1><p>Dashboard name: <code>{html.escape(dashboard_name)}</code></p>"
    html_doc += build_section(
        "Stats",
        stats_dir,
        dashboard_dir,
        [
            (stats_dir / "roi_siminsts.svg", "ROI instructions plot"),
            (stats_dir / "roi_cpi.svg", "ROI CPI plot"),
            (stats_dir / "roi_dcache_miss_rate.svg", "ROI dcache miss-rate plot"),
            (stats_dir / "roi_load_store_mix.svg", "ROI load/store plot"),
            (stats_dir / "roi_sections.csv", "ROI CSV"),
            (stats_dir / "bottleneck.html", "bottleneck report"),
        ],
        [
            ("roi_siminsts.svg", "ROI instructions per step"),
            ("roi_cpi.svg", "ROI CPI per step"),
            ("roi_dcache_miss_rate.svg", "ROI dcache miss rate per step"),
            ("roi_load_store_mix.svg", "ROI load and store mix per step"),
        ],
		None,
    )
    html_doc += build_section(
        "O3 Pipe",
        o3_dir,
        dashboard_dir,
        [
            (o3_dir / "trace.out", "O3 raw trace"),
            (o3_dir / "pipeview.out", "O3 pipeview output"),
        ],
		None,
		"Trace window is a few hundred cycles inside the first ROI, not the reset sequence.",
    )
    html_doc += build_section(
        "Config sweep",
        sweep_dir,
        dashboard_dir,
        [
            (sweep_dir / "comparison.html", "sweep comparison"),
        ],
        None,
        "One timestep, one knob at a time: load queue, L1D size, miss buffers, and L2.",
    )
    html_doc += "</body></html>\n"
    dashboard_file.write_text(html_doc)
    print(dashboard_file.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())