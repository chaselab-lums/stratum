#!/usr/bin/env python3
"""Follow-up gem5 runs (one lever per run, drawing 0 unless noted)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "results"))
from run_experiments import Exp, run_one  # noqa: E402

NNZ_VALUES = (1.0, 2.7, 5.0, 10.0, 20.0)


def followup_experiments() -> list[Exp]:
    exps: list[Exp] = []
    # 1–2: dense baseline re-run (drawing 0 input restored in build())
    for tag, opts in (
        ("kmajor_false", "skip-zeros=false k-major=false"),
        ("kmajor_true", "skip-zeros=false k-major=true"),
    ):
        exps.append(Exp(f"followup2_dense_{tag}", "task2_dense", opts))
    # 3: reference layout phase profile
    exps.append(
        Exp(
            "followup2_phase_ref_kmajor_false",
            "task1_phase",
            "skip-zeros=true k-major=false",
            phase_profile=True,
        )
    )
    # 4: L2 256 KiB / 512 KiB both layouts
    for l2 in ("256KiB", "512KiB"):
        for tag, opts in (
            ("ref", "skip-zeros=true k-major=false"),
            ("prod", "skip-zeros=true k-major=true"),
        ):
            exps.append(Exp(f"followup2_l2_{l2}_{tag}", "task3_l2", opts, gem5_l2=l2))
    # 5: density sweep ref + dense
    for nnz in NNZ_VALUES:
        exps.append(
            Exp(
                f"followup2_synthetic_ref_nnz{nnz}",
                "task5_synthetic",
                "skip-zeros=true k-major=false",
                synthetic_nnz=nnz,
            )
        )
        exps.append(
            Exp(
                f"followup2_synthetic_dense_nnz{nnz}",
                "task5_synthetic",
                "skip-zeros=false k-major=false",
                synthetic_nnz=nnz,
            )
        )
    return exps


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", action="append", help="run_id prefix filter")
    ap.add_argument(
        "--skip-dense",
        action="store_true",
        help="skip followup2_dense_* drawing-0 baselines only (not synthetic dense sweep)",
    )
    args = ap.parse_args()

    failed: list[str] = []
    for exp in followup_experiments():
        if args.skip_dense and exp.run_id.startswith("followup2_dense_"):
            continue
        if args.only and not any(exp.run_id.startswith(p) for p in args.only):
            continue
        try:
            run_one(exp, force=args.force)
        except Exception as e:
            print(f"FAILED {exp.run_id}: {e}", file=sys.stderr)
            failed.append(exp.run_id)

    if failed:
        print("failures:", ", ".join(failed), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
