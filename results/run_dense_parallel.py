#!/usr/bin/env python3
"""Build both dense configs sequentially, then run gem5 for each in parallel."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "results"))
sys.path.insert(0, str(ROOT / "results" / "lib"))

import run_experiments as rex  # noqa: E402
from gem5_metrics import parse_stats  # noqa: E402

DENSE = [
    rex.Exp("task2_dense_kmajor_true", "task2_dense", "skip-zeros=false k-major=true"),
    rex.Exp("task2_dense_kmajor_false", "task2_dense", "skip-zeros=false k-major=false"),
]


def gem5_only(exp: rex.Exp) -> None:
    out = rex.RUNS_DIR / exp.run_id
    if (out / "roi_metrics.json").exists():
        print(f"skip complete {exp.run_id}", flush=True)
        rex.run_one(exp, force=False)
        return
    elf = out / "sim.elf"
    if not elf.exists():
        raise SystemExit(f"missing {elf} — build failed?")
    print(f"gem5 start {exp.run_id}", flush=True)
    stats = rex.gem5(elf, out, exp)
    m = parse_stats(stats)
    (out / "roi_metrics.json").write_text(json.dumps(m, indent=2) + "\n")
    subprocess.run(
        ["rg", "FINAL_CLASS", str(out / "gem5" / "program.out")],
        stdout=open(out / "final_classes.txt", "w"),
        stderr=subprocess.DEVNULL,
    )
    rex.run_one(exp, force=False)
    print(f"gem5 done {exp.run_id}", flush=True)


def main() -> None:
    max_p = int(os.environ.get("GEM5_PARALLEL", "2"))
    pending = [e for e in DENSE if not (rex.RUNS_DIR / e.run_id / "roi_metrics.json").exists()]
    if not pending:
        print("all dense runs already complete")
        rex.rebuild_csv_from_runs()
        return

    for exp in pending:
        out = rex.RUNS_DIR / exp.run_id
        if (out / "sim.elf").exists() and (out / "roi_metrics.json").exists():
            continue
        print(f"build {exp.run_id}", flush=True)
        rex.record_commits(out)
        rex.build(exp, out)

    still = [e for e in DENSE if not (rex.RUNS_DIR / e.run_id / "roi_metrics.json").exists()]
    if max_p <= 1 or len(still) == 1:
        for exp in still:
            gem5_only(exp)
    else:
        workers = min(max_p, len(still))
        print(f"parallel gem5 workers={workers}", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(gem5_only, e) for e in still]
            for f in as_completed(futs):
                f.result()

    rex.rebuild_csv_from_runs()
    print("dense parallel batch finished", flush=True)


if __name__ == "__main__":
    main()
