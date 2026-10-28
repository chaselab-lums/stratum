#!/usr/bin/env python3
"""
Run gem5 experiment matrix for Stratum paper (Tasks 2–6).
Appends one row per run to results/metrics.csv and stores artifacts under results/runs/<id>/.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "results" / "lib"))
from gem5_metrics import (  # noqa: E402
    METRIC_CSV_KEYS,
    metrics_for_csv,
    parse_final_class,
    parse_phase_cycles,
    parse_stats,
)

CSV_PATH = ROOT / "results" / "metrics.csv"
RUNS_DIR = ROOT / "results" / "runs"
TOOLS = ROOT / "results" / "tools"

CSV_FIELDS = [
    "run_id",
    "task",
    "timestamp_utc",
    "snn_linalg_opts",
    "drawing",
    "gem5_l2",
    "gem5_width",
    "skip_zeros",
    "k_major",
    *METRIC_CSV_KEYS,
    "final_class",
    "phase_scan",
    "phase_gather",
    "phase_lif1",
    "phase_fc2",
    "phase_other",
    "artifact_dir",
]


@dataclass
class Exp:
    run_id: str
    task: str
    linalg_opts: str
    drawing: int = 0
    gem5_l2: str = "4MiB"
    gem5_width: int = 4
    synthetic_nnz: float | None = None
    phase_profile: bool = False
    extra_env: dict[str, str] = field(default_factory=dict)


def parse_opts(opts: str) -> tuple[bool | None, bool | None]:
    skip = None
    km = None
    for part in opts.split():
        if part.startswith("skip-zeros="):
            skip = part.split("=", 1)[1] == "true"
        if part.startswith("k-major="):
            km = part.split("=", 1)[1] == "true"
    return skip, km


def load_csv() -> list[dict[str, str]]:
    if not CSV_PATH.exists():
        return []
    with CSV_PATH.open(newline="") as f:
        return list(csv.DictReader(f))


def append_row(row: dict[str, str]) -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = load_csv()
    rid = row.get("run_id", "")
    rows = [r for r in rows if r.get("run_id") != rid]
    rows.append({k: row.get(k, "") for k in CSV_FIELDS})
    with CSV_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def infer_run_meta(run_id: str) -> tuple[str, str, str, str]:
    """Return (task, drawing, gem5_l2, gem5_width)."""
    task = "unknown"
    if run_id.startswith("step0"):
        task = "step0"
    elif run_id.startswith("task2") or run_id.startswith("followup2_dense"):
        task = "task2_dense"
    elif "followup2_l2" in run_id or "task3_l2" in run_id:
        task = "task3_l2"
    elif "task3_width" in run_id:
        task = "task3_width"
    elif run_id.startswith("task4"):
        task = "task4_drawings"
    elif run_id.startswith("task5") or run_id.startswith("followup2_synthetic"):
        task = "task5_synthetic"
    elif run_id.startswith("task6"):
        task = "task6_determinism"
    elif run_id.startswith("task1") or run_id.startswith("followup2_phase"):
        task = "task1_phase"
    drawing = "0"
    if "_draw" in run_id:
        import re as _re

        dm = _re.search(r"draw(\d+)", run_id)
        if dm:
            drawing = dm.group(1)
    l2 = "4MiB"
    for prefix in ("followup2_l2_", "task3_l2_"):
        if prefix in run_id:
            l2 = run_id.split(prefix)[1].split("_")[0]
            break
    width = "4"
    if "task3_width_" in run_id:
        width = run_id.split("task3_width_")[1].split("_")[0]
    return task, drawing, l2, width


def row_from_metrics(
    run_id: str,
    task: str,
    opts: str,
    drawing: str,
    l2: str,
    width: str,
    m: dict,
    artifact_dir: str,
    program_out: Path,
    timestamp_utc: str = "",
) -> dict[str, str]:
    skip, km = parse_opts(opts)
    fc = parse_final_class(program_out, 0)
    phase = parse_phase_cycles(program_out)
    row: dict[str, str] = {
        "run_id": run_id,
        "task": task,
        "timestamp_utc": timestamp_utc,
        "snn_linalg_opts": opts,
        "drawing": drawing,
        "gem5_l2": l2,
        "gem5_width": width,
        "skip_zeros": "" if skip is None else str(skip).lower(),
        "k_major": "" if km is None else str(km).lower(),
        "final_class": "" if fc is None else str(fc),
        "artifact_dir": artifact_dir,
        "phase_scan": "",
        "phase_gather": "",
        "phase_lif1": "",
        "phase_fc2": "",
        "phase_other": "",
    }
    row.update(metrics_for_csv(m))
    if phase:
        row["phase_scan"] = str(phase["scan"])
        row["phase_gather"] = str(phase["gather"])
        row["phase_lif1"] = str(phase["lif1"])
        row["phase_fc2"] = str(phase["fc2"])
        row["phase_other"] = str(phase["other"])
    return row


def rebuild_csv_from_runs() -> None:
    """Rebuild metrics.csv from every results/runs/*/roi_metrics.json."""
    rows: list[dict[str, str]] = []
    if not RUNS_DIR.is_dir():
        return
    for metrics_json in sorted(RUNS_DIR.glob("*/roi_metrics.json")):
        run_id = metrics_json.parent.name
        m = json.loads(metrics_json.read_text())
        opts_path = metrics_json.parent / "snn_linalg_opts.txt"
        opts = opts_path.read_text().strip() if opts_path.exists() else ""
        task, drawing, l2, width = infer_run_meta(run_id)
        rows.append(
            row_from_metrics(
                run_id,
                task,
                opts,
                drawing,
                l2,
                width,
                m,
                str(metrics_json.parent.relative_to(ROOT)),
                metrics_json.parent / "gem5" / "program.out",
            )
        )
    with CSV_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def record_commits(out: Path) -> None:
    for name, sub in (
        ("snnMNIST", ROOT / "snnMNIST"),
        ("snn-mlir", ROOT / "snn-mlir"),
        ("gem5", ROOT / "gem5"),
    ):
        d = out / "repos"
        d.mkdir(parents=True, exist_ok=True)
        if (sub / ".git").is_dir():
            subprocess.run(
                ["git", "-C", str(sub), "rev-parse", "HEAD"],
                check=True,
                stdout=open(d / f"{name}.commit", "w"),
            )


def shell(cmd: list[str], env: dict[str, str] | None = None, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env, cwd=cwd)


def build(exp: Exp, out: Path) -> Path:
    snn = ROOT / "snnMNIST"
    build_dir = snn / "build"
    env = os.environ.copy()
    env.update(exp.extra_env)
    env["SNN_LINALG_OPTS"] = exp.linalg_opts

    custom_input = exp.synthetic_nnz is not None or exp.drawing != 0

    make_env = env.copy()
    if custom_input:
        make_env["SNN_SKIP_INPUT_REWRITE"] = "1"

    if exp.synthetic_nnz is not None:
        shell(
            [
                sys.executable,
                str(TOOLS / "generate_synthetic_input.py"),
                str(build_dir),
                str(exp.synthetic_nnz),
                "--seed",
                str(100 + int(exp.synthetic_nnz * 10)),
                "--tag",
                exp.run_id.replace("/", "_"),
            ]
        )
        shell(
            [
                sys.executable,
                str(TOOLS / "verify_synthetic_input.py"),
                str(build_dir),
                str(exp.synthetic_nnz),
            ]
        )
    elif exp.drawing != 0:
        shell(
            [sys.executable, str(TOOLS / "single_vector_input.py"), str(build_dir), str(exp.drawing)]
        )
        make_env["SNN_SKIP_INPUT_REWRITE"] = "1"
    else:
        # drawing 0: restore input_0.h (Task 5 synthetic leaves wrong input.h in build/)
        shell([sys.executable, str(TOOLS / "single_vector_input.py"), str(build_dir), "0"])
        make_env["SNN_SKIP_INPUT_REWRITE"] = "1"

    for k in (
        "SNN_MLIR_ROOT",
        "SNN_OPT",
        "MLIR_DIR",
        "LLVM_BIN",
        "MLIR_OPT_BIN",
        "MLIR_TRANSLATE_BIN",
    ):
        make_env.setdefault(k, os.environ.get(k, ""))

    if exp.phase_profile:
        shell(
            [
                "make",
                "-f",
                "Makefile",
                "generate",
                "NO_TRAIN=1",
                "TEST_VECTORS=1",
                "BUILD_NO_TRAIN=1",
            ],
            env=make_env,
            cwd=snn,
        )
        phase_ll = snn / "build" / "network_phase.ll"
        shell(
            [
                sys.executable,
                str(TOOLS / "inject_phase_markers.py"),
                str(snn / "build" / "network.ll"),
                str(phase_ll),
            ]
        )
        make_env["NETWORK_LL"] = str(phase_ll)
        make_env["PHASE_PROFILE"] = "1"
        shell(
            [
                "make",
                "-f",
                "Makefile",
                "build-nogen",
                "NO_TRAIN=1",
            ],
            env=make_env,
            cwd=snn,
        )
    else:
        shell(
            [
                "make",
                "-f",
                "Makefile",
                "build",
                "NO_TRAIN=1",
                "TEST_VECTORS=1",
                "BUILD_NO_TRAIN=1",
            ],
            env=make_env,
            cwd=snn,
        )
    elf = snn / "build" / "sim.elf"
    out.mkdir(parents=True, exist_ok=True)
    for name in ("sim.elf", "network.mlir", "network.ll", "input.h"):
        subprocess.run(["cp", str(snn / "build" / name), str(out / name)], check=True)
    (out / "snn_linalg_opts.txt").write_text(exp.linalg_opts + "\n")
    return elf


def gem5(elf: Path, out: Path, exp: Exp) -> Path:
    gem5_dir = out / "gem5"
    gem5_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["GEM5_OUTDIR"] = str(gem5_dir)
    env["RUN_TIMESTAMP"] = exp.run_id
    env["GEM5_L2_SIZE"] = exp.gem5_l2
    env["GEM5_BOOM_WIDTH"] = str(exp.gem5_width)
    env["GEM5_BOOM"] = "1"
    log = out / "gem5_console.log"
    with log.open("w") as lf:
        subprocess.run(
            [str(ROOT / "run.sh"), str(elf), "baremetal"],
            check=True,
            env=env,
            stdout=lf,
            stderr=subprocess.STDOUT,
        )
    return gem5_dir / "stats.txt"


def run_one(exp: Exp, force: bool = False) -> dict[str, str]:
    out = RUNS_DIR / exp.run_id
    metrics_json = out / "roi_metrics.json"
    print(f"[run] {exp.run_id} task={exp.task} opts={exp.linalg_opts}", flush=True)
    if metrics_json.exists() and not force:
        print(f"skip existing {exp.run_id}", flush=True)
        m = json.loads(metrics_json.read_text())
    else:
        record_commits(out)
        elf = build(exp, out)
        stats = gem5(elf, out, exp)
        m = parse_stats(stats)
        metrics_json.write_text(json.dumps(m, indent=2) + "\n")
        subprocess.run(
            ["rg", "FINAL_CLASS", str(out / "gem5" / "program.out")],
            stdout=open(out / "final_classes.txt", "w"),
            stderr=subprocess.DEVNULL,
        )

    fc = parse_final_class(out / "gem5" / "program.out", 0)
    if exp.drawing == 0 and exp.synthetic_nnz is None and fc != 1:
        raise RuntimeError(f"{exp.run_id}: expected FINAL_CLASS[0]=1, got {fc}")
    row = row_from_metrics(
        exp.run_id,
        exp.task,
        exp.linalg_opts,
        str(exp.drawing),
        exp.gem5_l2,
        str(exp.gem5_width),
        m,
        str(out.relative_to(ROOT)),
        out / "gem5" / "program.out",
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    append_row(row)
    return row


def import_step0() -> None:
    base = ROOT / "results" / "step0" / "step0-repro-20261011"
    mapping = {
        "kmajor_false": ("step0_kmajor_false", "skip-zeros=true k-major=false"),
        "kmajor_true": ("step0_kmajor_true", "skip-zeros=true k-major=true"),
    }
    existing = {r["run_id"] for r in load_csv()}
    for tag, (run_id, opts) in mapping.items():
        if run_id in existing:
            continue
        src = base / tag
        if not (src / "roi_metrics.json").exists():
            continue
        dst = RUNS_DIR / run_id
        dst.mkdir(parents=True, exist_ok=True)
        for f in ("roi_metrics.json", "final_classes.txt", "snn_linalg_opts.txt"):
            subprocess.run(["cp", str(src / f), str(dst / f)], check=False)
        subprocess.run(["cp", "-r", str(src / "gem5"), str(dst / "gem5")], check=False)
        m = json.loads((dst / "roi_metrics.json").read_text())
        append_row(
            row_from_metrics(
                run_id,
                "step0",
                opts,
                "0",
                "4MiB",
                "4",
                m,
                str(dst.relative_to(ROOT)),
                dst / "gem5" / "program.out",
                "2026-10-11T00:00:00Z",
            )
        )


def experiment_list() -> list[Exp]:
    exps: list[Exp] = []
    dense: list[Exp] = []
    layouts = [
        ("ref", "skip-zeros=true k-major=false"),
        ("prod", "skip-zeros=true k-major=true"),
    ]
    # Task 3: L2 sweep
    for l2 in ("1MiB", "2MiB", "8MiB", "16MiB"):
        for tag, opts in layouts:
            exps.append(Exp(f"task3_l2_{l2}_{tag}", "task3_l2", opts, gem5_l2=l2))
    # Task 3b: core width
    for w in (2, 8):
        for tag, opts in layouts:
            exps.append(Exp(f"task3_width_{w}_{tag}", "task3_width", opts, gem5_width=w))
    # Task 4: five drawings
    for d in range(5):
        for tag, opts in layouts:
            exps.append(Exp(f"task4_draw{d}_{tag}", "task4_drawings", opts, drawing=d))
    # Task 5: synthetic nnz (production layout)
    for nnz in (1.0, 2.7, 5.0, 10.0, 20.0):
        exps.append(
            Exp(
                f"task5_synthetic_nnz{nnz}",
                "task5_synthetic",
                "skip-zeros=true k-major=true",
                synthetic_nnz=nnz,
            )
        )
    # Task 6: determinism (repeat production drawing 0)
    exps.append(
        Exp("task6_determinism_rerun", "task6_determinism", "skip-zeros=true k-major=true")
    )
    # Task 2 dense last (each run can take 1–2+ host hours on O3)
    dense.append(
        Exp("task2_dense_kmajor_true", "task2_dense", "skip-zeros=false k-major=true")
    )
    dense.append(
        Exp("task2_dense_kmajor_false", "task2_dense", "skip-zeros=false k-major=false")
    )
    return exps + dense


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", action="append", help="run_id prefix filter")
    ap.add_argument("--import-step0-only", action="store_true")
    ap.add_argument("--rebuild-csv", action="store_true")
    ap.add_argument("--skip-dense", action="store_true", help="skip skip-zeros=false runs")
    args = ap.parse_args()

    import_step0()
    if args.rebuild_csv:
        rebuild_csv_from_runs()
        print(f"rebuilt {CSV_PATH}")
        return
    if args.import_step0_only:
        return

    for exp in experiment_list():
        if args.skip_dense and "skip-zeros=false" in exp.linalg_opts:
            continue
        if args.only and not any(exp.run_id.startswith(p) for p in args.only):
            continue
        try:
            run_one(exp, force=args.force)
        except subprocess.CalledProcessError as e:
            print(f"FAILED {exp.run_id}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
