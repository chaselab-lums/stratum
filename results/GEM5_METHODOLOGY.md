# gem5 methodology notes

## CPU clock: use **2 GHz**, not 1 GHz

`run.sh` → `fs_linux.py` with gem5 defaults:

| Domain | Stat | Period (ticks) | Frequency |
|--------|------|----------------|-----------|
| **CPU** | `system.cpu_clk_domain.clock` | **500** | **2 GHz** (`--cpu-clock` default `2GHz` in `configs/common/Options.py`) |
| System | `system.clk_domain.clock` | 1000 | 1 GHz |

`simFreq = 1e12` ticks/s. ROI **`simSeconds`** is tick time / `simFreq`. **`system.cpu.numCycles`** advances at the **CPU** clock, so:

```text
numCycles / simSeconds ≈ 2e9  (e.g. 8,846,746 / 0.004423 s)
```

**Paper:** state simulated **core clock = 2 GHz**. Do not infer 1 GHz from `system.clk_domain.clock` alone.

## L2 miss rate in CSV (fixed)

Parser previously looked for `overall_miss_rate` (snake_case). gem5 25 prints **`system.l2.overallMissRate::total`** (~0.846 on 4 MiB prod ROI; ~0.253 on 1 MiB ref).

Re-export after fix:

```bash
python3 results/run_experiments.py --rebuild-csv
python3 results/generate_summary.py
```

(`generate_summary.py` writes `results/summary.md` from the CSV.)

Until then, read miss rates from local `results/runs/*/gem5/stats.txt` after a run.

## Phase markers (Task 1)

Spike-major ROI **without** markers: **4.423 ms**. With `snn_phase_marker` injected: **4.923 ms** (~**+11.3%**). Stage cycle shares from `PHASE_CYCLES` are **approximate** (marker + extra calls), not micro-arch ground truth.

Reference layout phase profile: **not run yet**.

## Pending measurements (paper status table)

| Item | In export? |
|------|------------|
| Dense baseline (`skip-zeros=false`) | Pending (long O3 runs) |
| Phase profile, reference layout | Pending |
| L2 **256 KiB**, **512 KiB** | Not in sweep (used 1/2/4/8/16 MiB) |
| Hand-written C kernels (Task 7) | Not measured |
| Dense + reference curves in synthetic nnz sweep | Not measured (spike-major only) |
| Dense vs skip-zeros speedup | Analytical only (~591k vs ~3k loop iter/step) |

## Reference list

Verify all citations against your `.bib` / DOIs; do not trust model memory.
