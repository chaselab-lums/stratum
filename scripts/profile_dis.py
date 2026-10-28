#!/usr/bin/env python3
"""
Profile RISC-V ELF disassembly statically and/or with a CVA6 execution trace.

Static mode (dis only): instruction counts per function in the .text image.
Dynamic mode (dis + trace): retired-instruction counts from iti.traces or
trace_rvfi_hart_00.dasm, with load/store and arithmetic breakdown per function.

Usage:
  profile_dis.py sim.elf.dis --plots-dir build/profile_plots
  profile_dis.py sim.elf.dis --save-stats build/profile_stats.json --func snn_forward_step
  profile_dis.py sim.elf.dis --trace iti.traces   # optional: dynamic sim trace
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import DefaultDict, Dict, Iterator, List, Optional, Tuple

RE_LABEL = re.compile(r"^([0-9a-f]+)\s+<([^>]+)>:\s*$")
RE_INSN = re.compile(r"^\s*([0-9a-f]+):\s+[0-9a-f]+\s+(\S+)")
RE_RVFI = re.compile(r"core\s+\d+:\s+0x([0-9a-f]+)\s+\(0x([0-9a-f]+)\)")

LOAD_MNEMS = frozenset(
    {
        "lb",
        "lbu",
        "lh",
        "lhu",
        "lw",
        "lwu",
        "ld",
        "c.lw",
        "c.lwsp",
        "c.ld",
        "c.ldsp",
        "flw",
        "fld",
        "lr.w",
        "lr.d",
    }
)
STORE_MNEMS = frozenset(
    {
        "sb",
        "sh",
        "sw",
        "sd",
        "c.sw",
        "c.swsp",
        "c.sd",
        "c.sdsp",
        "fsw",
        "fsd",
        "sc.w",
        "sc.d",
        "amoadd.w",
        "amoadd.d",
        "amoxor.w",
        "amoxor.d",
        "amoor.w",
        "amoor.d",
        "amoand.w",
        "amoand.d",
        "amomin.w",
        "amomin.d",
        "amomax.w",
        "amomax.d",
    }
)
BRANCH_MNEMS = frozenset(
    {
        "beq",
        "bne",
        "blt",
        "bge",
        "bltu",
        "bgeu",
        "bltz",
        "bgez",
        "bgtz",
        "blez",
        "bgt",
        "ble",
        "c.beqz",
        "c.bnez",
        "jal",
        "jalr",
        "j",
        "c.j",
        "c.jal",
        "c.jr",
        "c.jalr",
        "ret",
    }
)
ARITH_MNEMS = frozenset(
    {
        "add",
        "sub",
        "addi",
        "addw",
        "subw",
        "addiw",
        "mul",
        "mulh",
        "mulhsu",
        "mulhu",
        "div",
        "divu",
        "rem",
        "remu",
        "and",
        "or",
        "xor",
        "andi",
        "ori",
        "xori",
        "sll",
        "srl",
        "sra",
        "slli",
        "srli",
        "srai",
        "sllw",
        "srlw",
        "sraiw",
        "slliw",
        "srliw",
        "slt",
        "slti",
        "sltu",
        "sltiu",
        "lui",
        "auipc",
        "c.add",
        "c.addi",
        "c.addiw",
        "c.sub",
        "c.and",
        "c.or",
        "c.xor",
        "c.mv",
        "c.li",
        "c.lui",
        "c.slli",
        "c.srli",
        "c.srai",
        "c.addi16sp",
        "c.addi4spn",
        "fmadd.s",
        "fmsub.s",
        "fnmsub.s",
        "fnmadd.s",
        "fadd.s",
        "fsub.s",
        "fmul.s",
        "fdiv.s",
        "fsqrt.s",
        "fmin.s",
        "fmax.s",
        "fadd.d",
        "fsub.d",
        "fmul.d",
        "fdiv.d",
        "fsqrt.d",
        "fmin.d",
        "fmax.d",
    }
)
PSEUDO_ARITH = frozenset({"li", "mv", "neg", "not", "sext.w", "seqz", "snez"})

# Contiguous offsets within snn_forward_step (no gaps → no lost "other" bucket).
SNN_FORWARD_STEP_REGIONS: List[Tuple[str, int, int]] = [
    ("prologue / init", 0x000, 0x04C),
    ("fc1 outer loop", 0x04C, 0x066),
    ("fc1 matmul inner", 0x066, 0x088),
    ("lif hidden + fc2 setup", 0x088, 0x130),
    ("fc2 matmul inner", 0x130, 0x152),
    ("bias accumulate", 0x152, 0x1A2),
    ("lif output (10n)", 0x1A2, 0x1CE),
    ("epilogue", 0x1CE, 0x200),
]

CAT_COLORS = {
    "loads": "#4C78A8",
    "stores": "#F58518",
    "arith": "#54A24B",
    "branches": "#E45756",
    "other": "#B279A2",
}


@dataclass
class InstrInfo:
    mnemonic: str
    category: str  # load | store | arith | branch | other


@dataclass
class FuncStats:
    total: int = 0
    loads: int = 0
    stores: int = 0
    arithmetic: int = 0
    branches: int = 0
    other: int = 0
    mnemonics: Dict[str, int] = field(default_factory=dict)

    def add(self, cat: str, mnemonic: str = "") -> None:
        self.total += 1
        if cat == "load":
            self.loads += 1
        elif cat == "store":
            self.stores += 1
        elif cat == "arith":
            self.arithmetic += 1
        elif cat == "branch":
            self.branches += 1
        else:
            self.other += 1
        if mnemonic:
            self.mnemonics[mnemonic] = self.mnemonics.get(mnemonic, 0) + 1


@dataclass
class ProfileResult:
    functions: Dict[str, FuncStats]
    regions: Dict[str, FuncStats] = field(default_factory=dict)
    region_func: str = ""
    retired: int = 0
    unknown: int = 0


@dataclass
class Disassembly:
    symbols: List[Tuple[int, str]] = field(default_factory=list)
    sym_starts: List[int] = field(default_factory=list)
    named_sym_starts: List[int] = field(default_factory=list)
    named_sym_names: List[str] = field(default_factory=list)
    pc_insn: Dict[int, InstrInfo] = field(default_factory=dict)
    text_start: int = 0
    text_end: int = 0

    @classmethod
    def parse(cls, path: Path) -> "Disassembly":
        dis = cls()
        current_label: Optional[str] = None
        for raw in path.read_text(errors="replace").splitlines():
            line = raw.rstrip()
            m_label = RE_LABEL.match(line)
            if m_label:
                addr = int(m_label.group(1), 16)
                current_label = m_label.group(2)
                dis.symbols.append((addr, current_label))
                continue
            m_insn = RE_INSN.match(line)
            if not m_insn:
                continue
            pc = int(m_insn.group(1), 16)
            mnem = m_insn.group(2).lower()
            dis.pc_insn[pc] = InstrInfo(mnem, classify_mnemonic(mnem))
            if dis.text_start == 0 or pc < dis.text_start:
                dis.text_start = pc
            dis.text_end = max(dis.text_end, pc)

        dis.symbols.sort(key=lambda x: x[0])
        dis.sym_starts = [a for a, _ in dis.symbols]
        for addr, name in dis.symbols:
            base = normalize_name(name)
            if base.startswith("."):
                continue
            dis.named_sym_starts.append(addr)
            dis.named_sym_names.append(base)
        return dis

    def func_for_pc(self, pc: int) -> str:
        if not self.sym_starts:
            return f"0x{pc:08x}"
        idx = bisect.bisect_right(self.sym_starts, pc) - 1
        if idx < 0:
            return f"0x{pc:08x}"
        name = normalize_name(self.symbols[idx][1])
        if not name.startswith("."):
            return name
        nidx = bisect.bisect_right(self.named_sym_starts, pc) - 1
        if nidx >= 0:
            return self.named_sym_names[nidx]
        return name

    def func_entry(self, name: str) -> Optional[int]:
        for addr, sym in self.symbols:
            if normalize_name(sym) == name:
                return addr
        return None

    def static_stats(self) -> ProfileResult:
        stats: DefaultDict[str, FuncStats] = defaultdict(FuncStats)
        regions: DefaultDict[str, FuncStats] = defaultdict(FuncStats)
        region_func = "snn_forward_step"
        region_entry = self.func_entry(region_func)
        count = 0
        for pc, info in sorted(self.pc_insn.items()):
            fn = self.func_for_pc(pc)
            stats[fn].add(info.category, info.mnemonic)
            count += 1
            if region_entry is not None and fn == region_func:
                rname = region_for_pc(fn, region_entry, pc) or "other"
                regions[rname].add(info.category, info.mnemonic)
        return ProfileResult(
            functions=dict(stats),
            regions=dict(regions),
            region_func=region_func if regions else "",
            retired=count,
            unknown=0,
        )


def normalize_name(name: str) -> str:
    return name.split("+", 1)[0]


def classify_mnemonic(mnem: str) -> str:
    if mnem in LOAD_MNEMS:
        return "load"
    if mnem in STORE_MNEMS:
        return "store"
    if mnem in BRANCH_MNEMS:
        return "branch"
    if mnem in ARITH_MNEMS or mnem in PSEUDO_ARITH:
        return "arith"
    if mnem in {"nop", "ebreak", "ecall", "fence", "fence.i", "csrrw", "csrrs", "csrrc", "csrrwi", "csrrsi", "csrrci", "unimp"}:
        return "other"
    if mnem.startswith("f") and mnem not in LOAD_MNEMS and mnem not in STORE_MNEMS:
        return "arith"
    return "other"


def iter_trace_pcs(path: Path, fmt: str) -> Iterator[int]:
    with path.open("r", errors="replace") as fh:
        if fmt == "iti":
            first = True
            for line in fh:
                if first:
                    first = False
                    continue
                parts = line.strip().split(",")
                if len(parts) < 5:
                    continue
                yield int(parts[4], 16)
        elif fmt == "rvfi":
            for line in fh:
                m = RE_RVFI.match(line.strip())
                if m:
                    yield int(m.group(1), 16)
        else:
            raise ValueError(f"unknown trace format: {fmt}")


def region_for_pc(func: str, entry: int, pc: int) -> Optional[str]:
    if func != "snn_forward_step":
        return None
    off = pc - entry
    for name, lo, hi in SNN_FORWARD_STEP_REGIONS:
        if lo <= off < hi:
            return name
    return "other"


def dynamic_stats(
    dis: Disassembly,
    trace_path: Path,
    fmt: str,
    *,
    pc_min: Optional[int] = None,
    pc_max: Optional[int] = None,
    progress_every: int = 5_000_000,
) -> ProfileResult:
    stats: DefaultDict[str, FuncStats] = defaultdict(FuncStats)
    regions: DefaultDict[str, FuncStats] = defaultdict(FuncStats)
    region_func = "snn_forward_step"
    region_entry = dis.func_entry(region_func)
    total = 0
    unknown = 0
    for pc in iter_trace_pcs(trace_path, fmt):
        if pc_min is not None and pc < pc_min:
            continue
        if pc_max is not None and pc > pc_max:
            continue
        info = dis.pc_insn.get(pc)
        if info is None:
            unknown += 1
            continue
        fn = dis.func_for_pc(pc)
        stats[fn].add(info.category, info.mnemonic)
        if region_entry is not None and fn == region_func:
            rname = region_for_pc(fn, region_entry, pc) or "other"
            regions[rname].add(info.category, info.mnemonic)
        total += 1
        if progress_every and total % progress_every == 0:
            print(f"... processed {total:,} retired instructions", file=sys.stderr)
    return ProfileResult(
        functions=dict(stats),
        regions=dict(regions),
        region_func=region_func,
        retired=total,
        unknown=unknown,
    )


def print_table(
    stats: Dict[str, FuncStats],
    *,
    title: str,
    top: int,
    sort_key: str,
) -> None:
    rows = list(stats.items())
    key_fn = {
        "total": lambda kv: kv[1].total,
        "loads": lambda kv: kv[1].loads,
        "stores": lambda kv: kv[1].stores,
        "arithmetic": lambda kv: kv[1].arithmetic,
    }[sort_key]
    rows.sort(key=key_fn, reverse=True)

    print(title)
    print(
        f"{'function':<28} {'total':>12} {'loads':>10} {'stores':>10} "
        f"{'arith':>10} {'branch':>10} {'other':>8}"
    )
    print("-" * 92)
    shown = 0
    for name, s in rows:
        if s.total == 0:
            continue
        print(
            f"{name:<28} {s.total:>12,} {s.loads:>10,} {s.stores:>10,} "
            f"{s.arithmetic:>10,} {s.branches:>10,} {s.other:>8,}"
        )
        shown += 1
        if top and shown >= top:
            break
    print()


def sorted_rows(
    stats: Dict[str, FuncStats],
    sort_key: str = "total",
    top: int = 0,
) -> List[Tuple[str, FuncStats]]:
    key_fn = {
        "total": lambda kv: kv[1].total,
        "loads": lambda kv: kv[1].loads,
        "stores": lambda kv: kv[1].stores,
        "arithmetic": lambda kv: kv[1].arithmetic,
    }[sort_key]
    rows = [(n, s) for n, s in stats.items() if s.total > 0]
    rows.sort(key=key_fn, reverse=True)
    if top:
        rows = rows[:top]
    return rows


def _parse_region_stats(raw_regions: dict) -> Dict[str, FuncStats]:
    if not raw_regions:
        return {}
    sample = next(iter(raw_regions.values()))
    if isinstance(sample, int):
        return {name: FuncStats(total=cnt) for name, cnt in raw_regions.items()}
    return {
        name: FuncStats(**{**vals, "mnemonics": vals.get("mnemonics", {})})
        for name, vals in raw_regions.items()
    }


def save_stats_json(result: ProfileResult, path: Path) -> None:
    payload = {
        "functions": {name: asdict(s) for name, s in result.functions.items()},
        "regions": {name: asdict(s) for name, s in result.regions.items()},
        "region_func": result.region_func,
        "retired": result.retired,
        "unknown": result.unknown,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def load_stats_json(path: Path) -> ProfileResult:
    raw = json.loads(path.read_text())
    if "functions" in raw:
        funcs = {
            name: FuncStats(**{**vals, "mnemonics": vals.get("mnemonics", {})})
            for name, vals in raw["functions"].items()
        }
        return ProfileResult(
            functions=funcs,
            regions=_parse_region_stats(raw.get("regions", {})),
            region_func=raw.get("region_func", ""),
            retired=raw.get("retired", 0),
            unknown=raw.get("unknown", 0),
        )
    # legacy: flat function map
    funcs = {
        name: FuncStats(**{**vals, "mnemonics": vals.get("mnemonics", {})})
        for name, vals in raw.items()
    }
    return ProfileResult(functions=funcs)


def _fmt_millions(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _plot_segments(
    result: ProfileResult,
    top: int,
) -> List[Tuple[str, FuncStats]]:
    """Regions inside snn_forward_step + all other functions."""
    stats = result.functions
    rows = sorted_rows(stats, "total", top=0)
    if not rows:
        return []

    segments: List[Tuple[str, FuncStats]] = []
    if result.regions and result.region_func:
        for rname, _, _ in SNN_FORWARD_STEP_REGIONS:
            s = result.regions.get(rname, FuncStats())
            segments.append((f"{result.region_func} :: {rname}", s))
        other = result.regions.get("other", FuncStats())
        if other.total:
            segments.append((f"{result.region_func} :: unmapped", other))
        for name, s in rows:
            if name != result.region_func and s.total > 0:
                segments.append((name, s))
    else:
        show = rows[:top] if top else rows
        segments = list(show)

    segments = [(n, s) for n, s in segments if s.total > 0]
    segments.sort(key=lambda kv: kv[1].total, reverse=True)
    return segments


def print_region_table(profile: ProfileResult) -> None:
    if not profile.regions or profile.region_func not in profile.functions:
        return
    hot_total = profile.functions[profile.region_func].total
    print(f"=== {profile.region_func} regions ===")
    for rname, _, _ in SNN_FORWARD_STEP_REGIONS:
        rs = profile.regions.get(rname, FuncStats())
        pct = 100.0 * rs.total / hot_total if hot_total else 0
        print(
            f"  {rname:<26} {rs.total:>6,}  "
            f"loads {rs.loads:>4,}  stores {rs.stores:>4,}  "
            f"arith {rs.arithmetic:>4,}  ({pct:5.1f}%)"
        )
    extra = profile.regions.get("other", FuncStats())
    if extra.total:
        pct = 100.0 * extra.total / hot_total if hot_total else 0
        print(
            f"  {'unmapped':<26} {extra.total:>6,}  "
            f"loads {extra.loads:>4,}  stores {extra.stores:>4,}  "
            f"arith {extra.arithmetic:>4,}  ({pct:5.1f}%)"
        )
    print()


# Metrics drawn as separate bars per row (only categories with any data globally).
PLOT_METRICS = ("loads", "stores", "arithmetic", "branches")


def write_plots(
    result: ProfileResult,
    out_dir: Path,
    *,
    title_prefix: str,
    top: int,
    detail_funcs: List[str],
) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.png"):
        old.unlink()

    segments = _plot_segments(result, top)
    if not segments:
        return []

    labels = [name for name, _ in segments]
    metric_vals = {
        "loads": np.array([s.loads for _, s in segments], dtype=np.int64),
        "stores": np.array([s.stores for _, s in segments], dtype=np.int64),
        "arithmetic": np.array([s.arithmetic for _, s in segments], dtype=np.int64),
        "branches": np.array([s.branches for _, s in segments], dtype=np.int64),
    }
    others = np.array([s.other for _, s in segments], dtype=np.int64)
    totals = np.array([s.total for _, s in segments], dtype=np.int64)

    # Only plot metric columns that have data somewhere in the chart.
    active_metrics = [m for m in PLOT_METRICS if metric_vals[m].sum() > 0]
    if others.sum() > 0:
        metric_vals["other"] = others
        active_metrics.append("other")

    n_metrics = len(active_metrics)
    y_base = np.arange(len(labels))
    bar_h = min(0.16, 0.72 / max(n_metrics, 1))
    offsets = np.linspace(-(n_metrics - 1) * bar_h / 2, (n_metrics - 1) * bar_h / 2, n_metrics)

    fig_h = max(6.0, 0.48 * len(labels))
    fig, ax = plt.subplots(figsize=(14, fig_h))

    legend_seen: set[str] = set()
    for mi, mname in enumerate(active_metrics):
        vals = metric_vals[mname]
        color = CAT_COLORS.get(mname, "#999999")
        y_pos = y_base + offsets[mi]
        label = mname if mname not in legend_seen else None
        ax.barh(y_pos, vals, height=bar_h, color=color, label=label)
        if label:
            legend_seen.add(mname)
        # Explicit "0" where a metric is absent for this row.
        for i, v in enumerate(vals):
            if v == 0:
                ax.text(
                    0.5,
                    y_pos[i],
                    "0",
                    va="center",
                    ha="left",
                    fontsize=7,
                    color="#666666",
                )
            elif v > 0 and totals[i] > 0 and v / totals[i] >= 0.08:
                ax.text(
                    v * 0.5,
                    y_pos[i],
                    _fmt_millions(int(v)),
                    va="center",
                    ha="center",
                    fontsize=6,
                    color="white" if mname != "arithmetic" else "black",
                )

    ax.set_yticks(y_base)
    display_labels = [
        f"{name}  ({_fmt_millions(int(tot))} total)" for name, tot in zip(labels, totals)
    ]
    ax.set_yticklabels(display_labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Instruction count (log scale)")
    ax.set_title(
        f"{title_prefix}: loads / stores / arithmetic / branches",
        fontsize=12,
    )
    ax.set_xscale("symlog", linthresh=100)
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _p: _fmt_millions(int(x)) if x >= 1 else "0")
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", which="both", linestyle=":", alpha=0.35)

    fig.tight_layout()
    out_path = out_dir / "profile_bar.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [out_path]


def print_func_detail(
    name: str,
    dis: Disassembly,
    stats: FuncStats,
    *,
    dynamic: bool = False,
) -> None:
    print(f"=== {name} ===")
    print(f"  total instructions : {stats.total:,}")
    print(f"  loads              : {stats.loads:,}")
    print(f"  stores             : {stats.stores:,}")
    print(f"  load+store         : {stats.loads + stats.stores:,}")
    print(f"  arithmetic         : {stats.arithmetic:,}")
    print(f"  branches           : {stats.branches:,}")
    print(f"  other              : {stats.other:,}")
    if stats.total:
        pct = 100.0 * stats.arithmetic / stats.total
        mem = 100.0 * (stats.loads + stats.stores) / stats.total
        print(f"  arithmetic %       : {pct:.1f}%")
        print(f"  memory ops %       : {mem:.1f}%")

    # Top static PCs inside this function (by mnemonic variety, for context)
    pcs = [
        (pc, info)
        for pc, info in dis.pc_insn.items()
        if dis.func_for_pc(pc) == name
    ]
    pcs.sort()
    if pcs:
        print(f"  static size        : {len(pcs)} instructions in .dis")
        if dynamic and stats.total and len(pcs):
            print(f"  avg dynamic/static : {stats.total / len(pcs):.1f}x")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description="Profile RISC-V disassembly and/or sim trace")
    ap.add_argument("dis", type=Path, help="objdump -d output (.dis)")
    ap.add_argument("--trace", type=Path, help="iti.traces or trace_rvfi_hart_00.dasm")
    ap.add_argument(
        "--trace-format",
        choices=("auto", "iti", "rvfi"),
        default="auto",
        help="trace format (default: auto from filename)",
    )
    ap.add_argument("--top", type=int, default=25, help="rows to show (0 = all)")
    ap.add_argument(
        "--sort",
        choices=("total", "loads", "stores", "arithmetic"),
        default="total",
        help="sort key for ranking functions",
    )
    ap.add_argument("--func", action="append", default=[], help="detailed stats for function(s)")
    ap.add_argument(
        "--app-only",
        action="store_true",
        help="only count PCs in .text range from the .dis file",
    )
    ap.add_argument("--no-progress", action="store_true", help="disable trace progress on stderr")
    ap.add_argument(
        "--plots-dir",
        type=Path,
        help="write profile_bar.png to this directory (requires matplotlib)",
    )
    ap.add_argument(
        "--plot-top",
        type=int,
        default=10,
        help="functions to include in charts (default: 10)",
    )
    ap.add_argument(
        "--save-stats",
        type=Path,
        help="save profile JSON for fast replotting with --plots-from-stats",
    )
    ap.add_argument(
        "--plots-from-stats",
        type=Path,
        help="skip trace scan; load stats JSON and only generate plots",
    )
    args = ap.parse_args()

    if args.plots_from_stats:
        if not args.plots_from_stats.is_file():
            print(f"error: stats file not found: {args.plots_from_stats}", file=sys.stderr)
            return 1
        if not args.plots_dir:
            print("error: --plots-from-stats requires --plots-dir", file=sys.stderr)
            return 1
        profile = load_stats_json(args.plots_from_stats)
        paths = write_plots(
            profile,
            args.plots_dir,
            title_prefix="Dynamic profile",
            top=args.plot_top,
            detail_funcs=args.func,
        )
        print(f"Wrote {len(paths)} plot(s) to {args.plots_dir}:")
        for p in paths:
            print(f"  {p}")
        return 0

    if not args.dis.is_file():
        print(f"error: dis file not found: {args.dis}", file=sys.stderr)
        return 1

    dis = Disassembly.parse(args.dis)
    print(f"Parsed {args.dis}: {len(dis.pc_insn)} static instructions, "
          f"{len(dis.named_sym_names)} named functions")
    print(f".text range: 0x{dis.text_start:08x} – 0x{dis.text_end:08x}\n")

    if args.trace:
        if not args.trace.is_file():
            print(f"error: trace file not found: {args.trace}", file=sys.stderr)
            return 1
        fmt = args.trace_format
        if fmt == "auto":
            name = args.trace.name
            if "iti" in name:
                fmt = "iti"
            elif "rvfi" in name or name.endswith(".dasm"):
                fmt = "rvfi"
            else:
                fmt = "iti"
        pc_min = dis.text_start if args.app_only else None
        pc_max = dis.text_end if args.app_only else None
        print(f"Dynamic profile from {args.trace} ({fmt})")
        if args.app_only:
            print(f"  filtering to app .text: 0x{pc_min:08x} – 0x{pc_max:08x}")
        profile = dynamic_stats(
            dis,
            args.trace,
            fmt,
            pc_min=pc_min,
            pc_max=pc_max,
            progress_every=0 if args.no_progress else 5_000_000,
        )
        print(f"  retired in range : {profile.retired:,}")
        print(f"  unknown PCs      : {profile.unknown:,} (boot/trampoline or stale trace)\n")
        table_title = "Dynamic profile (retired instructions from simulation):"
        plot_prefix = "Dynamic profile"
        is_dynamic = True
    else:
        profile = dis.static_stats()
        print(f"Static profile from {args.dis} ({profile.retired:,} instructions in .text)\n")
        table_title = "Profile from .dis (static instruction counts per function):"
        plot_prefix = "Static profile (.dis)"
        is_dynamic = False

    print_table(
        profile.functions,
        title=table_title,
        top=args.top,
        sort_key=args.sort,
    )
    print_region_table(profile)
    for fn in args.func:
        if fn in profile.functions:
            print_func_detail(fn, dis, profile.functions[fn], dynamic=is_dynamic)
        else:
            print(f"warning: no stats for {fn!r}", file=sys.stderr)

    if args.save_stats:
        save_stats_json(profile, args.save_stats)
        print(f"Saved stats to {args.save_stats}")

    if args.plots_dir:
        paths = write_plots(
            profile,
            args.plots_dir,
            title_prefix=plot_prefix,
            top=args.plot_top,
            detail_funcs=args.func,
        )
        print(f"Wrote {len(paths)} plot(s) to {args.plots_dir}:")
        for p in paths:
            print(f"  {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
