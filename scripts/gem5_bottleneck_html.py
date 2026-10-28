#!/usr/bin/env python3
"""Write a self-contained bottleneck.html from a gem5 output directory."""

from __future__ import annotations

import html
import math
import sys
from pathlib import Path

FC1_MACS = 2312 * 256
FC2_MACS = 256 * 10
MACS_PER_STEP = FC1_MACS + FC2_MACS

CLASS_LABELS = {
    "IntAlu": "IntAlu",
    "IntMult": "IntMult",
    "IntDiv": "IntDiv",
    "MemRead": "Loads",
    "MemWrite": "Stores",
    "FloatAdd": "FloatAdd",
    "FloatMult": "FloatMult",
}
CLASS_COLORS = {
    "IntAlu": "#16537e",
    "MemRead": "#c2410c",
    "MemWrite": "#b45309",
    "IntMult": "#6d28d9",
}
CLASS_ROLES = {
    "IntAlu": "Address math, add, branch",
    "MemRead": "Weight, input, accumulator",
    "MemWrite": "Accumulator store",
    "IntMult": "Integer multiply",
}
CLASS_ORDER = ["IntAlu", "MemRead", "MemWrite", "IntMult"]


def parse_stats(stats_path: Path) -> list[dict[str, float]]:
    sections: list[dict[str, float]] = []
    current: dict[str, float] | None = None
    for raw_line in stats_path.read_text(errors="replace").splitlines():
        line = raw_line.strip()
        if line == "---------- Begin Simulation Statistics ----------":
            current = {}
            sections.append(current)
            continue
        if current is None or not line or line.startswith("#") or line.startswith("-"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            current[parts[0]] = float(parts[1])
        except ValueError:
            continue
    return sections


def parse_config(config_path: Path) -> dict[str, dict[str, str]]:
    sections: dict[str, dict[str, str]] = {}
    current: dict[str, str] | None = None
    if not config_path.exists():
        return sections
    for raw_line in config_path.read_text(errors="replace").splitlines():
        line = raw_line.strip()
        if line.startswith("[") and line.endswith("]"):
            current = {}
            sections[line[1:-1]] = current
            continue
        if current is None or "=" not in line:
            continue
        key, value = line.split("=", 1)
        current[key] = value
    return sections


def get(section: dict[str, float], key: str, default: float = 0.0) -> float:
    value = section.get(key, default)
    if value is None or math.isnan(value) or math.isinf(value):
        return default
    return value


def has(section: dict[str, float], key: str) -> bool:
    return key in section and not math.isnan(section[key])


def comma(value: float) -> str:
    return f"{value:,.0f}"


def compact(value: float) -> str:
    absolute = abs(value)
    if absolute >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if absolute >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if absolute >= 10_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,.0f}"


def pct(part: float, whole: float) -> str:
    if whole <= 0:
        return "0%"
    return f"{100.0 * part / whole:.1f}%"


def bytes_label(num_bytes: float) -> str:
    if num_bytes >= 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.1f} MB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.0f} KB"
    return f"{num_bytes:.0f} B"


def cache_size_label(raw: str | None) -> str:
    if not raw:
        return "unknown"
    try:
        return bytes_label(float(raw))
    except ValueError:
        return raw


def clock_ghz(config: dict[str, dict[str, str]], section: dict[str, float]) -> float | None:
    clock = config.get("system.cpu_clk_domain", {}).get("clock")
    sim_freq = get(section, "simFreq", 1e12)
    if not clock:
        cycles = get(section, "system.cpu.numCycles")
        ticks = get(section, "simTicks")
        if cycles <= 0 or ticks <= 0:
            return None
        return sim_freq / (ticks / cycles) / 1e9
    try:
        period = float(clock)
    except ValueError:
        return None
    if period <= 0:
        return None
    return sim_freq / period / 1e9


def ghz_label(ghz: float | None) -> str:
    if ghz is None:
        return "unknown clock"
    if abs(ghz - round(ghz)) < 0.05:
        return f"{round(ghz):.0f} GHz"
    return f"{ghz:.2f} GHz"


def int_mult_fu(config: dict[str, dict[str, str]]) -> tuple[str | None, str | None]:
    for name, sec in config.items():
        if sec.get("opClass") != "IntMult":
            continue
        parent = name.rsplit(".", 1)[0]
        return config.get(parent, {}).get("count"), sec.get("opLat")
    return None, None


def fu_count(config: dict[str, dict[str, str]], op_class: str) -> float:
    for name, sec in config.items():
        if sec.get("opClass") != op_class:
            continue
        parent = name.rsplit(".", 1)[0]
        raw = config.get(parent, {}).get("count")
        try:
            return float(raw or 0)
        except ValueError:
            return 0.0
    return 0.0


def flop_count(section: dict[str, float]) -> tuple[float, str]:
    """Retired floating-point arithmetic. A fused multiply-add is two FLOPs."""
    classes = dict(committed_classes(section))
    adds = classes.get("FloatAdd", 0.0)
    mults = classes.get("FloatMult", 0.0)
    divs = classes.get("FloatDiv", 0.0) + classes.get("FloatSqrt", 0.0)
    fmas = classes.get("FloatMultAcc", 0.0)
    flops = adds + mults + divs + 2 * fmas
    if flops > 0:
        return flops, "float"
    # An integer multiply-accumulate is a multiply plus the add into the accumulator.
    return 2 * classes.get("IntMult", 0.0), "int"


def committed_classes(section: dict[str, float]) -> list[tuple[str, float]]:
    prefix = "system.cpu.commitStats0.committedInstType::"
    found: list[tuple[str, float]] = []
    for key, value in section.items():
        if not key.startswith(prefix) or key.endswith("::total"):
            continue
        name = key[len(prefix) :]
        if value > 0:
            found.append((name, value))
    order = {name: index for index, name in enumerate(CLASS_ORDER)}
    found.sort(key=lambda item: (order.get(item[0], 100), -item[1]))
    return found


def steady_sections(sections: list[dict[str, float]]) -> list[dict[str, float]]:
    positive = [section for section in sections if get(section, "simInsts") > 0]
    if not positive:
        return []
    peak = max(get(section, "simInsts") for section in positive)
    steady = [section for section in positive if get(section, "simInsts") >= 0.99 * peak]
    return steady or positive


def color_for(name: str, index: int) -> str:
    if name in CLASS_COLORS:
        return CLASS_COLORS[name]
    palette = ["#0f766e", "#64748b", "#9f1239", "#365314"]
    return palette[index % len(palette)]


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def pie_svg(slices: list[tuple[str, float, str]]) -> str:
    total = sum(value for _, value, _ in slices)
    if total <= 0:
        return ""
    cx, cy, radius, width = 110, 110, 68, 28
    circumference = 2 * math.pi * radius
    offset = 0.0
    rings: list[str] = []
    for _label, value, color in slices:
        length = circumference * value / total
        gap = circumference - length
        rings.append(
            "<circle cx='{cx}' cy='{cy}' r='{r}' fill='none' stroke='{color}' "
            "stroke-width='{width}' stroke-dasharray='{length:.2f} {gap:.2f}' "
            "stroke-dashoffset='{offset:.2f}' transform='rotate(-90 {cx} {cy})'/>".format(
                cx=cx,
                cy=cy,
                r=radius,
                width=width,
                color=color,
                length=length,
                gap=gap,
                offset=-offset,
            )
        )
        offset += length
    legend = []
    for label, value, color in slices:
        legend.append(
            "<div class='legend-row'><span class='swatch' style='background:{color}'></span>"
            "<span>{label}</span><span class='legend-value'>{share}</span></div>".format(
                color=color,
                label=esc(label),
                share=pct(value, total),
            )
        )
    return (
        "<div class='pie-wrap'>"
        f"<svg viewBox='0 0 220 220' width='220' height='220' role='img' aria-label='Instruction mix'>"
        f"{''.join(rings)}"
        f"<text x='110' y='106' text-anchor='middle' font-size='18' font-weight='600'>{esc(compact(total))}</text>"
        "<text x='110' y='126' text-anchor='middle' font-size='11' fill='#666'>retired</text>"
        "</svg>"
        f"<div class='legend'>{''.join(legend)}</div></div>"
    )


def hbar(rows: list[tuple[str, float]], unit: str = "") -> str:
    peak = max((value for _, value in rows), default=0.0)
    if peak <= 0:
        return ""
    body = []
    for label, value in rows:
        width = 100.0 * value / peak
        shown = f"{value:.1f}{unit}" if unit == "%" else comma(value)
        body.append(
            "<div class='bar-row'>"
            f"<div class='bar-label'>{esc(label)}</div>"
            "<div class='bar-track'><div class='bar-fill' style='width:{width:.4f}%'></div></div>"
            f"<div class='bar-value'>{esc(shown)}</div></div>".format(width=width)
        )
    return "".join(body)


def line_svg(values: list[float], y_label: str) -> str:
    if len(values) < 2:
        return ""
    width, height = 860, 240
    left, right, top, bottom = 56, 16, 16, 36
    inner_w = width - left - right
    inner_h = height - top - bottom
    y_min = min(values)
    y_max = max(values)
    if y_max - y_min < 0.01:
        pad = 0.002
        y_min -= pad
        y_max += pad
    else:
        pad = (y_max - y_min) * 0.1
        y_min -= pad
        y_max += pad

    def x_pos(index: int) -> float:
        return left + inner_w * index / (len(values) - 1)

    def y_pos(value: float) -> float:
        return top + inner_h * (1 - (value - y_min) / (y_max - y_min))

    points = " ".join(f"{x_pos(i):.1f},{y_pos(value):.1f}" for i, value in enumerate(values))
    dots = "".join(
        f"<circle cx='{x_pos(i):.1f}' cy='{y_pos(value):.1f}' r='3' fill='#16537e'/>"
        for i, value in enumerate(values)
    )
    step = 1 if len(values) <= 16 else max(1, len(values) // 12)
    ticks = "".join(
        f"<text x='{x_pos(i):.1f}' y='{height - 12}' font-size='11' text-anchor='middle' fill='#444'>{i}</text>"
        for i in range(0, len(values), step)
    )
    return (
        f"<svg viewBox='0 0 {width} {height}' width='100%' role='img' aria-label='{esc(y_label)}'>"
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{height - bottom}' stroke='#222'/>"
        f"<line x1='{left}' y1='{height - bottom}' x2='{width - right}' y2='{height - bottom}' stroke='#222'/>"
        f"<text x='14' y='{height / 2:.0f}' font-size='11' fill='#444' transform='rotate(-90 14 {height / 2:.0f})'>{esc(y_label)}</text>"
        f"<text x='{left}' y='{top + 12}' font-size='11' fill='#444'>{y_max:.4f}</text>"
        f"<text x='{left}' y='{height - bottom}' font-size='11' fill='#444'>{y_min:.4f}</text>"
        f"<polyline fill='none' stroke='#16537e' stroke-width='2' points='{points}'/>"
        f"{dots}{ticks}</svg>"
    )


def stat_card(value: str, label: str) -> str:
    return f"<div class='stat'><div class='value'>{esc(value)}</div><div class='label'>{esc(label)}</div></div>"


def table(headers: list[str], rows: list[list[str]], emphasize: set[int] | None = None) -> str:
    emphasize = emphasize or set()
    head = "".join(f"<th>{esc(header)}</th>" for header in headers)
    body = []
    for index, row in enumerate(rows):
        klass = " class='emphasis'" if index in emphasize else ""
        cells = "".join(f"<td>{esc(cell)}</td>" for cell in row)
        body.append(f"<tr{klass}>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def network_note(multiplies: float) -> str:
    if multiplies <= 0 or multiplies % MACS_PER_STEP != 0:
        return ""
    steps = int(multiplies // MACS_PER_STEP)
    if steps == 1:
        return " That multiply count is fc1 (2312×256) plus fc2 (256×10)."
    return (
        f" That multiply count is {steps} timesteps of fc1 (2312×256) plus fc2 (256×10)."
    )


def rate_label(value: float) -> str:
    """Compact rate. Decade ticks must stay short enough to fit the axis margin."""
    if value >= 1e9:
        return f"{value / 1e9:.3g} GFLOP/s"
    if value >= 1e6:
        return f"{value / 1e6:.3g} MFLOP/s"
    if value >= 1e3:
        return f"{value / 1e3:.3g} kFLOP/s"
    return f"{value:.3g} FLOP/s"


def _log_span(value: float, low: float, high: float, length: float) -> float:
    return length * (math.log10(value) - math.log10(low)) / (math.log10(high) - math.log10(low))


def render_roofline(
    flops: float,
    dram_bytes: float,
    seconds: float,
    peak_bw_MBps: float,
    flop_units: float,
    ghz: float,
    source: str,
    markers: list[dict[str, object]] | None = None,
) -> str:
    if flops <= 0 or dram_bytes <= 0 or seconds <= 0 or peak_bw_MBps <= 0:
        return ""
    achieved = flops / seconds
    intensity = flops / dram_bytes
    peak_Bps = peak_bw_MBps * 1e6
    bandwidth_roof = peak_Bps * intensity
    compute_roof = flop_units * ghz * 1e9 if flop_units > 0 and ghz > 0 else 0.0
    ridge = compute_roof / peak_Bps if compute_roof > 0 else 0.0

    x_min, x_max = 0.01, 100.0
    y_candidates = [achieved, bandwidth_roof, peak_Bps * x_min]
    if compute_roof > 0:
        y_candidates.append(compute_roof)
    for marker in markers or []:
        marker_rate = float(marker["achieved"])
        if marker_rate > 0:
            y_candidates.append(marker_rate)
    y_min = min(y_candidates) / 4
    y_max = max(y_candidates) * 4

    width, height = 860, 340
    # "10 GFLOP/s" is wider than the old 78px margin, which clipped it to "0.00".
    left, right, top, bottom = 118, 24, 16, 46
    inner_w = width - left - right
    inner_h = height - top - bottom

    def x_pos(x_value: float) -> float:
        return left + _log_span(x_value, x_min, x_max, inner_w)

    def y_pos(y_value: float) -> float:
        return top + inner_h - _log_span(y_value, y_min, y_max, inner_h)

    roof_xs = [x_min * (10 ** (i / 40)) for i in range(0, 161)]
    roof_pts = []
    for x_value in roof_xs:
        y_value = peak_Bps * x_value
        if compute_roof > 0:
            y_value = min(y_value, compute_roof)
        if y_min <= y_value <= y_max and x_min <= x_value <= x_max:
            roof_pts.append(f"{x_pos(x_value):.1f},{y_pos(y_value):.1f}")
    decades_x = [0.01, 0.1, 1, 10, 100]
    decades_y = [10 ** exp for exp in range(math.floor(math.log10(y_min)), math.ceil(math.log10(y_max)) + 1)]
    grid = []
    x_labels = []
    for tick in decades_x:
        x = x_pos(tick)
        grid.append(f"<line x1='{x:.1f}' y1='{top}' x2='{x:.1f}' y2='{height - bottom}' stroke='#e6e6e6'/>")
        x_labels.append(f"<text x='{x:.1f}' y='{height - 16}' font-size='11' text-anchor='middle' fill='#444'>{tick:g}</text>")
    y_labels = []
    for tick in decades_y:
        if not (y_min <= tick <= y_max):
            continue
        y = y_pos(tick)
        grid.append(f"<line x1='{left}' y1='{y:.1f}' x2='{width - right}' y2='{y:.1f}' stroke='#e6e6e6'/>")
        y_labels.append(f"<text x='{left - 8}' y='{y + 4:.1f}' font-size='11' text-anchor='end' fill='#444'>{rate_label(tick)}</text>")

    point = ""
    legend = ""
    drawn: list[tuple[float, float]] = []
    if markers:
        dots = []
        legend_items = []
        placed: dict[tuple[int, int], int] = {}
        for marker in markers:
            mx = float(marker["intensity"])
            my = float(marker["achieved"])
            if not (x_min <= mx <= x_max and y_min <= my <= y_max):
                continue
            color = str(marker["color"])
            label = str(marker["label"])
            cx, cy = x_pos(mx), y_pos(my)
            key = (round(cx), round(cy))
            slot = placed.get(key, 0)
            placed[key] = slot + 1
            if slot:
                angle = (slot - 1) * (2 * math.pi / 6)
                cx += 11 * math.cos(angle)
                cy += 11 * math.sin(angle)
            drawn.append((mx, my))
            dots.append(
                f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='5.5' fill='{color}' stroke='#fff' stroke-width='1.5'>"
                f"<title>{esc(label)}</title></circle>"
            )
            legend_items.append(
                "<span style='display:inline-flex;align-items:center;gap:6px;margin:0 14px 6px 0'>"
                f"<i style='display:inline-block;width:10px;height:10px;border-radius:50%;background:{color}'></i>"
                f"{esc(label)}</span>"
            )
        point = "".join(dots)
        legend = f"<p class='caption'>{''.join(legend_items)}</p>"
    elif x_min <= intensity <= x_max and y_min <= achieved <= y_max:
        point = (
            f"<circle cx='{x_pos(intensity):.1f}' cy='{y_pos(achieved):.1f}' r='5' fill='#c2410c'/>"
            f"<text x='{x_pos(intensity) + 8:.1f}' y='{y_pos(achieved) - 8:.1f}' font-size='12' fill='#c2410c'>this region</text>"
        )

    if markers and drawn:
        lo_i = min(item[0] for item in drawn)
        hi_i = max(item[0] for item in drawn)
        lo_r = min(item[1] for item in drawn)
        hi_r = max(item[1] for item in drawn)
        if ridge > 0 and hi_i < ridge:
            place = f"All of them sit on the bandwidth side of the ridge ({ridge:.2f} FLOP/byte)."
        elif ridge > 0 and lo_i >= ridge:
            place = f"All of them sit on the compute side of the ridge ({ridge:.2f} FLOP/byte)."
        elif ridge > 0:
            place = f"The ridge is {ridge:.2f} FLOP/byte."
        else:
            place = ""
        side = (
            f"Each dot is one swept configuration. They span {lo_i:.3f}–{hi_i:.3f} FLOP/byte "
            f"and {rate_label(lo_r)}–{rate_label(hi_r)}. "
            f"{place} The table is the baseline point. "
            "Dots that share a coordinate are drawn a few pixels apart."
        )
    elif ridge > 0 and intensity >= ridge:
        side = (
            f"At {intensity:.2f} FLOP/byte the point is on the compute side of the ridge "
            f"({ridge:.2f} FLOP/byte). It still reaches only {pct(achieved, compute_roof)} of the "
            "floating-point roof, because each floating-point operation is wrapped in scalar "
            "integer instructions and load-queue stalls."
        )
    elif ridge > 0:
        side = (
            f"At {intensity:.2f} FLOP/byte the point is on the bandwidth side of the ridge "
            f"({ridge:.2f} FLOP/byte)."
        )
    else:
        side = f"Arithmetic intensity is {intensity:.2f} FLOP/byte."

    rows = [
        ["Arithmetic intensity", f"{intensity:.2f} FLOP/byte"],
        ["Achieved throughput", rate_label(achieved)],
        ["DRAM roof at this intensity", f"{rate_label(bandwidth_roof)} ({pct(achieved, bandwidth_roof)} of roof)"],
    ]
    if compute_roof > 0:
        rows.append(
            ["Floating-point roof", f"{rate_label(compute_roof)} ({pct(achieved, compute_roof)} of roof)"]
        )
    if ridge > 0:
        rows.append(["Ridge", f"{ridge:.2f} FLOP/byte"])

    svg = (
        f"<svg viewBox='0 0 {width} {height}' width='100%' role='img' aria-label='Roofline'>"
        f"{''.join(grid)}"
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{height - bottom}' stroke='#222'/>"
        f"<line x1='{left}' y1='{height - bottom}' x2='{width - right}' y2='{height - bottom}' stroke='#222'/>"
        f"<polyline fill='none' stroke='#16537e' stroke-width='2' points='{' '.join(roof_pts)}'/>"
        f"{point}{''.join(x_labels)}{''.join(y_labels)}"
        f"<text x='14' y='{(top + height - bottom) / 2:.0f}' font-size='12' fill='#444' text-anchor='middle' transform='rotate(-90 14 {(top + height - bottom) / 2:.0f})'>Achieved rate</text>"
        f"<text x='{(left + width - right) / 2:.0f}' y='{height - 2}' font-size='12' text-anchor='middle' fill='#444'>Arithmetic intensity (FLOP/byte)</text>"
        "</svg>"
    )
    return (
        "<h2>Roofline</h2>"
        "<p class='caption'>DRAM roof is peak bandwidth times intensity. "
        f"{esc(source)} "
        "Both axes are logarithmic.</p>"
        f"{svg}"
        f"{legend}"
        f"{table(['Metric', 'Value'], rows)}"
        f"<p class='caption'>{esc(side)}</p>"
    )


def build_html(outdir: Path, sections: list[dict[str, float]], config: dict[str, dict[str, str]]) -> str:
    positive = [section for section in sections if get(section, "simInsts") > 0]
    steady = steady_sections(sections)
    if not steady:
        return (
            "<!DOCTYPE html><html><body><p>No retired instructions in "
            f"{esc(outdir / 'stats.txt')}.</p></body></html>\n"
        )

    rep = steady[0]
    cpu = config.get("system.cpu", {})
    cpu_type = cpu.get("type", "CPU")
    ghz = clock_ghz(config, rep)
    l1d = cache_size_label(config.get("system.cpu.dcache", {}).get("size"))
    l1i = cache_size_label(config.get("system.cpu.icache", {}).get("size"))
    l1d_bytes = float(config.get("system.cpu.dcache", {}).get("size") or 0)
    mshrs = config.get("system.cpu.dcache", {}).get("mshrs", "?")
    lq_entries = cpu.get("LQEntries", "?")
    sq_entries = cpu.get("SQEntries", "?")
    iq_entries = cpu.get("numIQEntries", "?")
    rob_entries = cpu.get("numROBEntries", "?")
    has_l2 = any(name == "system.l2" or name.startswith("system.l2.") for name in config)
    mul_count, mul_lat = int_mult_fu(config)
    o3 = has(rep, "system.cpu.rename.LQFullEvents") or cpu_type == "BaseO3CPU"

    insts = get(rep, "simInsts")
    ipc = get(rep, "system.cpu.ipc")
    cpi = get(rep, "system.cpu.cpi")
    classes = committed_classes(rep)
    class_map = dict(classes)
    alu = class_map.get("IntAlu", 0.0)
    loads = class_map.get("MemRead", 0.0)
    stores = class_map.get("MemWrite", 0.0)
    mults = class_map.get("IntMult", 0.0)
    dram_util = get(rep, "system.mem_ctrls.dram.busUtil") if has(rep, "system.mem_ctrls.dram.busUtil") else None
    dram_bytes = get(rep, "system.mem_ctrls.requestorReadBytes::cpu.data")
    if dram_bytes <= 0:
        dram_bytes = get(rep, "system.mem_ctrls.dram.bytesRead::cpu.data")
    icache_misses = get(rep, "system.cpu.icache.overallMisses::total")
    dcache_miss_rate = get(rep, "system.cpu.dcache.overallMissRate::total")
    dcache_accesses = get(rep, "system.cpu.dcache.demandAccesses::total")
    dcache_hits = get(rep, "system.cpu.dcache.overallHits::total")
    mshr_hits = get(rep, "system.cpu.dcache.demandMshrHits::total")
    mshr_misses = get(rep, "system.cpu.dcache.demandMshrMisses::total")
    forwarded = get(rep, "system.cpu.lsq0.forwLoads")
    load_to_use_fast = get(rep, "system.cpu.lsq0.loadToUse::0-9")
    load_to_use_total = get(rep, "system.cpu.lsq0.loadToUse::total")
    mispredicts = get(rep, "system.cpu.commit.branchMispredicts")
    predicted = get(rep, "system.cpu.fetch.predictedBranches")
    fu_mult_busy = get(rep, "system.cpu.statFuBusy::IntMult")
    row_hit = get(rep, "system.mem_ctrls.dram.pageHitRate")
    peak_bw = get(rep, "system.mem_ctrls.dram.peakBW")
    avg_rd_bw = get(rep, "system.mem_ctrls.dram.avgRdBW")

    lq_full = get(rep, "system.cpu.rename.LQFullEvents")
    iq_full = get(rep, "system.cpu.rename.IQFullEvents")
    sq_full = get(rep, "system.cpu.rename.SQFullEvents")
    rob_full = get(rep, "system.cpu.rename.ROBFullEvents")
    rename_events = lq_full + iq_full + sq_full + rob_full

    rename_idle = get(rep, "system.cpu.rename.idleCycles")
    rename_run = get(rep, "system.cpu.rename.runCycles")
    rename_block = get(rep, "system.cpu.rename.blockCycles")
    rename_unblock = get(rep, "system.cpu.rename.unblockCycles")
    rename_total = rename_idle + rename_run + rename_block + rename_unblock
    decode_blocked = get(rep, "system.cpu.decode.blockedCycles")
    decode_total = sum(
        get(rep, key)
        for key in (
            "system.cpu.decode.idleCycles",
            "system.cpu.decode.blockedCycles",
            "system.cpu.decode.runCycles",
            "system.cpu.decode.unblockCycles",
            "system.cpu.decode.squashCycles",
        )
    )
    fetch_zero = get(rep, "system.cpu.fetch.nisnDist::0")
    fetch_total = get(rep, "system.cpu.fetch.nisnDist::total")

    steady_seconds = sum(get(section, "simSeconds") for section in steady)
    total_insts = sum(get(section, "simInsts") for section in positive)
    smaller = len(positive) - len(steady)
    scalar_mac = mults > 0 and alu / mults >= 3 and abs(stores - mults) / mults < 0.05
    lq_bound = o3 and rename_events > 0 and lq_full / rename_events >= 0.8
    dram_idle = dram_util is not None and dram_util < 15 and cpu_type == "BaseO3CPU"

    if lq_bound and dram_idle:
        callout_title = f"The core is waiting on a {lq_entries}-entry load queue"
        alu_per_mul = alu / mults if mults else 0
        callout_body = (
            f"The kernel is a scalar int8 MAC loop. About {alu_per_mul:.0f} integer ALU "
            f"instructions retire for every multiply, weight traffic does not fit in the {l1d} "
            f"data cache, and rename stalls when the load queue fills. DRAM is {dram_util:.1f}% busy, "
            f"and the instruction cache misses {icache_misses:.0f} times in this region."
        )
    elif o3 and rename_events > 0:
        dominant = max(
            (("load queue", lq_full), ("issue queue", iq_full), ("store queue", sq_full), ("ROB", rob_full)),
            key=lambda item: item[1],
        )
        callout_title = f"Rename stalls most often on the {dominant[0]}"
        callout_body = (
            f"{comma(dominant[1])} {dominant[0]} full events in this region, "
            f"out of {comma(rename_events)} structural rename stalls. IPC is {ipc:.2f}."
        )
    elif not o3:
        callout_title = "Instruction mix is in this run; pipeline stalls are not"
        callout_body = (
            f"This run used {cpu_type}. Load-queue, rename, and issue counters come from "
            "the out-of-order CPU. Re-run with GEM5_CPU_TYPE=RiscvO3CPU, which profile.sh "
            "already does, to fill in the pipeline section. The instruction mix and cache "
            "numbers below are from this run."
        )
    else:
        callout_title = f"IPC is {ipc:.2f} on {cpu_type}"
        callout_body = "Pipeline full-event counters were not present in this stats dump."

    sim_time = f"{steady_seconds * 1000:.1f} ms" if steady_seconds < 1 else f"{steady_seconds:.2f} s"
    sim_label = f"Simulated time, {len(steady)} full regions"
    dram_timed = dram_util is not None and (o3 or dram_util > 0)
    dram_value = f"{dram_util:.1f}%" if dram_timed else "n/a"
    dram_label = "DRAM bus utilization" if dram_timed else "DRAM bus not timed on this CPU"

    slices = [
        (f"{CLASS_LABELS.get(name, name)} {pct(value, insts)}", value, color_for(name, index))
        for index, (name, value) in enumerate(classes)
    ]
    mix_rows = []
    emphasize: set[int] = set()
    for index, (name, value) in enumerate(classes):
        role = CLASS_ROLES.get(name, "")
        if name == "IntMult":
            role = "One int8×int8 MAC" if scalar_mac else role
        mix_rows.append([CLASS_LABELS.get(name, name), comma(value), pct(value, insts), role])
        if name == "MemRead":
            emphasize.add(index)

    mix_note = network_note(mults)
    if fu_mult_busy > 0 and mults > 0 and mul_count and mul_lat:
        mix_note += (
            f" {mul_count} pipelined integer multipliers, latency {mul_lat}. "
            f"The multiply unit was busy on {comma(fu_mult_busy)} issues, {pct(fu_mult_busy, mults)} of multiplies."
        )

    pipeline_html = ""
    if o3 and rename_total > 0:
        pipeline_html = f"""
        <h2>Where the pipeline actually stops</h2>
        <p class='caption'>Source: gem5 rename and decode for the representative region.
        Sampled rename cycles are {comma(rename_total)}. IQ is {esc(iq_entries)} entries,
        ROB is {esc(rob_entries)}, load queue and store queue are {esc(lq_entries)}.</p>
        <div class='split'>
          <div>
            <h3>Rename full events</h3>
            {hbar([("Load queue", lq_full), ("Issue queue", iq_full), ("Store queue", sq_full), ("ROB", rob_full)])}
            <p class='caption'>Full events in this region. Load-queue full is {pct(lq_full, rename_events)} of structural rename stalls.</p>
          </div>
          <div>
            <h3>Rename cycle split</h3>
            {hbar([
                ("Idle", 100 * rename_idle / rename_total),
                ("Running", 100 * rename_run / rename_total),
                ("Blocked", 100 * rename_block / rename_total),
                ("Unblocking", 100 * rename_unblock / rename_total),
            ], unit="%")}
            <p class='caption'>Blocked rename is the load queue. Decode is blocked for {pct(decode_blocked, decode_total)} of its cycles behind that backpressure. Fetch issues nothing on {pct(fetch_zero, fetch_total)} of cycles. I-cache misses: {comma(icache_misses)}.</p>
          </div>
        </div>
        """

    true_dram_rate = mshr_misses / dcache_accesses if dcache_accesses else 0.0
    memory_rows: list[list[str]] = []
    memory_emphasis: set[int] = set()
    if forwarded > 0:
        memory_rows.append(["Loads forwarded from a store", comma(forwarded), "Accumulator reload hits the store queue"])
    if dcache_hits > 0:
        memory_rows.append(["L1D hits", comma(dcache_hits), f"Lines that stayed in the {l1d} data cache"])
    if mshr_hits > 0:
        memory_rows.append(
            ["L1D misses that hit an in-flight MSHR", comma(mshr_hits), f"Weight stream piles onto {mshrs} MSHRs"]
        )
        memory_emphasis.add(len(memory_rows) - 1)
    if mshr_misses > 0:
        memory_rows.append(
            ["MSHR misses to DRAM", comma(mshr_misses), f"{bytes_label(dram_bytes)} read from DRAM in this region"]
        )
        memory_emphasis.add(len(memory_rows) - 1)
    if load_to_use_total > 0:
        memory_rows.append(
            ["Load-to-use, 0–9 cycles", pct(load_to_use_fast, load_to_use_total) + " of loads", "Short-latency loads and forwards"]
        )
    if icache_misses >= 0 and has(rep, "system.cpu.icache.overallMisses::total"):
        memory_rows.append(["I-cache misses", comma(icache_misses), f"Instruction cache is {l1i}"])
    if predicted > 0:
        memory_rows.append(
            ["Branch mispredicts", comma(mispredicts), f"{pct(mispredicts, predicted)} of predicted branches"]
        )

    memory_note = ""
    if peak_bw > 0 and avg_rd_bw > 0:
        memory_note = (
            f"DRAM peak bandwidth in this config is {peak_bw / 1000:.1f} GB/s. "
            f"Average read bandwidth is {avg_rd_bw:.0f} MB/s."
        )
        if dram_bytes > l1d_bytes > 0 and dram_idle:
            memory_note += (
                " The weight matrix is re-read from DRAM because it does not fit in L1, "
                "and the core runs out of load-queue entries before it runs out of memory bandwidth."
            )
        if abs(dram_bytes - FC1_MACS) / FC1_MACS < 0.05:
            memory_note += " The DRAM read is about the size of the fc1 weight matrix (592 KB)."

    cpi_values = [get(section, "system.cpu.cpi") for section in steady if has(section, "system.cpu.cpi")]
    chart = line_svg(cpi_values, "CPI")
    chart_html = ""
    if chart:
        omitted = ""
        if smaller:
            omitted = f" {smaller} smaller region{'s' if smaller != 1 else ''} omitted (epilogue or a partial dump)."
        same_count = len({int(get(section, "simInsts")) for section in steady}) == 1
        retired = comma(insts) if same_count else "nearly the same number of"
        chart_html = f"""
        <h2>Full regions are flat</h2>
        <p class='caption'>CPI for {len(steady)} regions that retire about {compact(insts)} instructions each.
        Every one of those regions retires {retired} instructions.{omitted}</p>
        {chart}
        <p class='caption'>Y-axis is cycles per instruction. X-axis is the region index. A flat line means the same kernel ran again on the next input frame.</p>
        """

    cards = []
    if scalar_mac:
        cards.append(
            (
                "Instruction overhead",
                "Half of retirement is integer ALU around a single multiply. A packed MAC or wider loads changes the instruction mix directly.",
            )
        )
    if lq_bound:
        cards.append(
            (
                "Load queue",
                f"{comma(lq_full)} load-queue-full events versus {comma(rob_full)} ROB-full events. Widening the {lq_entries}-entry LQ, or cutting loads per MAC, is the pipeline knob.",
            )
        )
    if dram_bytes > l1d_bytes > 0:
        cards.append(
            (
                "Weight reuse",
                f"This region reads {bytes_label(dram_bytes)} from DRAM against a {l1d} L1 and {mshrs} MSHRs. A larger L1 or a scratchpad removes that stream.",
            )
        )
    if not cards and classes:
        cards.append(("Instruction mix", f"The largest retired class is {CLASS_LABELS.get(classes[0][0], classes[0][0])} at {pct(classes[0][1], insts)}."))

    cards_html = ""
    if cards:
        pieces = []
        for index, (title, body) in enumerate(cards[:3], start=1):
            pieces.append(
                f"<article class='card'><h3><span>{index}</span> {esc(title)}</h3><p>{esc(body)}</p></article>"
            )
        cards_html = "<h2>How to read this for the next experiment</h2><div class='cards'>" + "".join(pieces) + "</div>"

    widths = ", ".join(
        f"{name} {cpu[key]}"
        for name, key in (
            ("fetch", "fetchWidth"),
            ("decode", "decodeWidth"),
            ("rename", "renameWidth"),
            ("issue", "issueWidth"),
            ("commit", "commitWidth"),
        )
        if key in cpu
    )
    mul_footer = ""
    if mul_count and mul_lat:
        mul_footer = f", {mul_count}× IntMult latency {mul_lat}"
    l2_footer = "no L2" if not has_l2 else "L2 present"
    footer = (
        f"Model: gem5 {cpu_type}"
        + (f", {widths}" if widths else "")
        + (f", LQ {lq_entries}, SQ {sq_entries}, IQ {iq_entries}, ROB {rob_entries}" if o3 else "")
        + f"{mul_footer}. {l2_footer}. "
        f"Instruction total across {len(positive)} dumped regions is {comma(total_insts)}. "
        f"Source: {outdir}/stats.txt."
    )

    l2_phrase = "no L2" if not has_l2 else "with an L2"
    if cpu_type == "BaseO3CPU":
        model_note = (
            f"IPC {ipc:.2f} is for this gem5 RiscvO3CPU "
            f"(LQ {lq_entries}, L1D {l1d}, {l2_phrase}). It is not CVA6 and not a measurement from silicon."
        )
    else:
        model_note = (
            f"These counters are for gem5 {cpu_type}. They are not a measurement from silicon."
        )
    flops, flop_kind = flop_count(rep)
    if flop_kind == "float":
        flop_units = fu_count(config, "FloatAdd") + fu_count(config, "FloatMult")
        flop_source = (
            "Work is retired FloatAdd and FloatMult, one FLOP each, plus FloatMultAcc as two FLOPs. "
            "The flat roof is one result per cycle on each pipelined FP add unit and FP multiply unit. "
            "DRAM bytes and simSeconds are for this region."
        )
    else:
        flop_units = 2 * (float(mul_count) if mul_count else 0.0)
        flop_source = (
            "Work counts each retired integer multiply-accumulate as two FLOPs. "
            "The flat roof is two FLOPs per cycle per pipelined integer multiplier. "
            "DRAM bytes and simSeconds are for this region."
        )
    roofline_html = render_roofline(
        flops=flops,
        dram_bytes=dram_bytes,
        seconds=get(rep, "simSeconds"),
        peak_bw_MBps=peak_bw,
        flop_units=flop_units,
        ghz=ghz or 0.0,
        source=flop_source,
    )
    subtitle = (
        f"{cpu_type}, one core at {ghz_label(ghz)}, L1D {l1d} / L1I {l1i}, {l2_phrase}. "
        f"Stats dump has {len(positive)} regions with retired instructions"
        f"{f', {len(steady)} of them full' if smaller else ''}. "
        "Charts use the first full region."
    )

    usage = ""
    if classes:
        usage = (
            "<div class='usage' aria-hidden='true'>"
            + "".join(
                "<span style='width:{width:.2f}%;background:{color}'></span>".format(
                    width=100.0 * value / insts,
                    color=color_for(name, index),
                )
                for index, (name, value) in enumerate(classes)
            )
            + "</div>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SNN MNIST gem5 bottlenecks</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #1b1b1b;
      --muted: #5c5c5c;
      --line: #e2e2e2;
      --paper: #fff;
      --wash: #f7f7f7;
      --warn-bg: #fff6dd;
      --warn-ink: #6b4e00;
      --accent: #16537e;
    }}
    body {{
      margin: 0 auto;
      max-width: 1040px;
      padding: 32px 24px 64px;
      font: 16px/1.5 "Segoe UI", Helvetica, Arial, sans-serif;
      color: var(--ink);
      background: var(--paper);
    }}
    h1 {{ font-size: 28px; line-height: 1.2; margin: 0 0 8px; }}
    h2 {{ font-size: 20px; margin: 32px 0 8px; }}
    h3 {{ font-size: 15px; margin: 0 0 8px; }}
    p {{ margin: 0 0 12px; }}
    .subtitle, .caption {{ color: var(--muted); font-size: 14px; }}
    .callout {{
      background: var(--warn-bg);
      color: var(--warn-ink);
      border: 1px solid #f0d48a;
      padding: 14px 16px;
      margin: 20px 0;
    }}
    .callout strong {{ display: block; margin-bottom: 4px; }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
      margin: 8px 0 8px;
    }}
    .stat {{ border: 1px solid var(--line); padding: 14px 16px; }}
    .stat .value {{ font-size: 28px; font-weight: 650; letter-spacing: -0.03em; }}
    .stat .label {{ color: var(--muted); font-size: 13px; }}
    .split, .mix {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 24px;
      align-items: start;
    }}
    .pie-wrap {{ display: flex; gap: 16px; align-items: center; }}
    .legend-row {{ display: grid; grid-template-columns: 12px 1fr auto; gap: 8px; align-items: center; font-size: 14px; margin: 4px 0; }}
    .swatch {{ width: 12px; height: 12px; display: inline-block; }}
    .legend-value {{ color: var(--muted); }}
    .usage {{ display: flex; height: 14px; width: 100%; background: var(--wash); margin: 8px 0 16px; }}
    .usage span {{ display: block; height: 100%; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }}
    th {{ font-size: 12px; color: var(--muted); font-weight: 600; }}
    td:nth-child(2), td:nth-child(3) {{ text-align: right; font-variant-numeric: tabular-nums; }}
    tr.emphasis td:first-child {{ border-left: 3px solid var(--accent); }}
    .bar-row {{ display: grid; grid-template-columns: 110px 1fr 88px; gap: 8px; align-items: center; margin: 6px 0; font-size: 14px; }}
    .bar-track {{ background: var(--wash); height: 14px; }}
    .bar-fill {{ background: var(--accent); height: 100%; }}
    .bar-value {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .mini-stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 12px 0; }}
    .cards {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }}
    .card {{ border: 1px solid var(--line); padding: 14px 16px; }}
    .card h3 span {{ color: var(--muted); margin-right: 6px; }}
    .model {{ font-size: 14px; color: var(--ink); border-left: 3px solid var(--accent); padding: 4px 0 4px 12px; margin: 12px 0 16px; }}
    .footer {{ margin-top: 28px; color: var(--muted); font-size: 13px; }}
    @media (max-width: 800px) {{
      .stats, .split, .mix, .mini-stats, .cards {{ grid-template-columns: 1fr; }}
      .pie-wrap {{ flex-direction: column; align-items: flex-start; }}
    }}
  </style>
</head>
<body>
  <h1>SNN MNIST inference on gem5</h1>
  <p class="subtitle">{esc(subtitle)}</p>
  <div class="callout"><strong>{esc(callout_title)}</strong>{esc(callout_body)}</div>
  <p class="model">{esc(model_note)}</p>
  <div class="stats">
    {stat_card(f"{ipc:.2f}" if ipc else "n/a", "IPC in the representative region")}
    {stat_card(compact(insts), "Instructions in that region")}
    {stat_card(sim_time, sim_label)}
    {stat_card(dram_value, dram_label)}
  </div>

  {roofline_html}

  <h2>What one region retires</h2>
  <p class="caption">Source: gem5 commitStats0. {comma(insts)} committed instructions. CPI {cpi:.4f}.</p>
  <div class="mix">
    <div>{pie_svg(slices)}</div>
    <div>
      {usage}
      {table(["Class", "Count", "Share", "Role in the MAC"], mix_rows, emphasize)}
      <p class="caption">{esc(mix_note.strip())}</p>
    </div>
  </div>

  {pipeline_html}

  <h2>Memory path</h2>
  <div class="mini-stats">
    {stat_card(f"{100 * dcache_miss_rate:.2f}%" if dcache_accesses else "n/a", "L1D tag miss rate")}
    {stat_card(f"{100 * true_dram_rate:.2f}%" if dcache_accesses and mshr_misses else "n/a", "Accesses that reach DRAM")}
    {stat_card(bytes_label(dram_bytes) if dram_bytes else "n/a", "DRAM data read in this region")}
    {stat_card(f"{row_hit:.0f}%" if row_hit else "n/a", "DRAM row-hit rate")}
  </div>
  {table(["Stage", "This region", "What it means"], memory_rows, memory_emphasis) if memory_rows else ""}
  <p class="caption">{esc(memory_note)}</p>

  {chart_html}
  {cards_html}
  <p class="footer">{esc(footer)}</p>
</body>
</html>
"""


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: gem5_bottleneck_html.py /path/to/gem5-outdir", file=sys.stderr)
        return 1
    outdir = Path(sys.argv[1])
    stats_path = outdir / "stats.txt"
    if not stats_path.exists():
        print(f"ERROR: stats.txt not found in {outdir}", file=sys.stderr)
        return 1
    sections = parse_stats(stats_path)
    config = parse_config(outdir / "config.ini")
    report = outdir / "bottleneck.html"
    report.write_text(build_html(outdir, sections, config))
    print(report.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
