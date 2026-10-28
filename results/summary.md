# Stratum gem5 experiment summary

Generated from `results/metrics.csv`. See `REPRODUCE.md` for commands.

## Headline (Step 0, drawing 0, `TEST_VECTORS=1`)

| Layout | sim time (ms) | IPC | class |
|--------|---------------|-----|-------|
| output-outer (`k-major=false`) | 7.038 | 1.157 | 1 |
| spike-major (`k-major=true`) | 4.423 | 1.692 | 1 |

Speedup (ref/prod): **1.59×**. Production column matches archived run `20261008-184328`.

**Methodology fix:** archived `20261008-181456` (7.424 ms) used five vectors; gem5 stats after ROI dump reflect **vector 4**, not drawing 0. Paper table now uses reproduced ref **7.038 ms** on drawing 0.

## Task 1 phase ROI (spike-major, drawing 0)

- sim time: **4.92 ms**, IPC 1.524
- Phase `PHASE_CYCLES` require fixed `phase_marker.c` (RV32 rdcycle); re-run after post-fixes if needed.

## Coverage

- `step0`: 2 run(s)
- `task1_phase`: 1 run(s)
- `task3_l2`: 8 run(s)
- `task3_width`: 4 run(s)
- `task4_drawings`: 10 run(s)
- `task5_synthetic`: 5 run(s)
- `task6_determinism`: 1 run(s)

## Claims vs data

| Claim | Verdict |
|-------|---------|
| Spike-major cuts fc1 weight-stream stalls (lower ms, higher IPC) | Supported on drawing 0 |
| Layout change preserves classification | Supported (`FINAL_CLASS[0]=1` on step0) |
| Reference 7.42 ms on drawing 0 | **Contradicted** — use 7.04 ms with documented protocol |
| IPC gain implies fewer synaptic updates | **Weakened** — same skip-zeros work; IPC reflects memory behavior |

