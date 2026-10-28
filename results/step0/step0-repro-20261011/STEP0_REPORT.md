# Step 0 — layout reproduction (2026-10-11)

## Protocol (this rerun)

- **Input:** `snnMNIST/build/input.h`, `NUM_TEST_VECTORS=1` (drawing 0 only).
- **ROI:** `snn_gem5_roi_begin/end` around 300 forward steps per vector (`main.c`).
- **gem5:** `./run.sh <elf> baremetal` with `GEM5_BOOM=1` (same BOOM-class params as Oct 8 runs).
- **Build:** `make build NO_TRAIN=1 TEST_VECTORS=1` with `SNN_LINALG_OPTS` set per column.

### Reference layout (`k-major=false`)

```bash
cd snnMNIST
SNN_LINALG_OPTS='skip-zeros=true k-major=false' \
  make build NO_TRAIN=1 TEST_VECTORS=1
```

`snn-opt` pass: `--convert-snn-to-linalg=skip-zeros=true k-major=false`  
(see `snnMNIST/lower_cpu_linux.sh`; default was `k-major=true` only before this env var).

### Production layout (`k-major=true`)

```bash
SNN_LINALG_OPTS='skip-zeros=true k-major=true' make build NO_TRAIN=1 TEST_VECTORS=1
```

## Measured ROI (this run)

| Layout | simSeconds (ms) | IPC | numCycles | simInsts | FINAL_CLASS[0] |
|--------|-----------------|-----|-----------|----------|----------------|
| k-major=false | **7.038** | **1.157306** | 14,076,427 | 16,290,734 | 1 |
| k-major=true  | **4.423** | **1.692167** | 8,846,746 | 14,970,175 | 1 |

Artifacts: `kmajor_false/`, `kmajor_true/` (ELF, MLIR, LL, gem5 `stats.txt`, commit files).

## Archived runs (paper cites)

| Run ID | simSeconds (ms) | IPC | Notes from `program.out` |
|--------|-----------------|-----|---------------------------|
| 20261008-181456 | **7.424** | **1.097201** | **5** test vectors; ROI stats = **last** vector (index 4), not drawing 0 |
| 20261008-184328 | **4.423** | **1.692167** | **1** test vector; matches this Step 0 production column **exactly** |

## Reproduction verdict

| Metric | k-major=true | k-major=false |
|--------|--------------|---------------|
| vs paper rounded value | **Matches** 4.42 ms / 1.69 IPC | **Does not match** 7.42 ms / 1.10 IPC on drawing 0 |
| vs archived stats.txt | **Exact** match to 184328 | **Differs** from 181456 (−0.386 ms, +0.060 IPC) |

**Do not treat the published reference column as reproduced on drawing 0 until the Oct 8 ELF and `NUM_TEST_VECTORS` are recovered or the reference is re-run under the same protocol as production.**

## “Not linked today” vs 7.42 ms column

- The **7.42 ms** number is real gem5 output from run **20261008-181456** with a **`sim.elf` built with output-outer gather** (not the current default `lower_cpu_linux.sh`, which only ever implied `k-major=true`).
- The canvas line “Reference IR / not linked today” means **Makefile does not ship a second production ELF** — not that the reference binary was never built. Oct 8 reference was built by running `snn-opt` with **`k-major=false`** (manual or one-off), then `make build` / link.
- **Fix for paper/canvas:** state that reference numbers come from an **explicitly linked** `skip-zeros=true k-major=false` binary; document `SNN_LINALG_OPTS` (now in `lower_cpu_linux.sh`).

## Clock / cycles

- `simFreq` = 1e12 ticks/s (gem5 global tick).
- **CPU clock:** `system.cpu_clk_domain.clock` = **500** ticks → **2 GHz** (default `--cpu-clock=2GHz` in `fs_linux.py`).
- **System clock:** `system.clk_domain.clock` = 1000 ticks → 1 GHz (memory/interconnect domain; not core cycle rate).
- Example (k-major=true): 4.423 ms sim time, **8,846,746** `numCycles` → **8.846746e6 / 0.004423 ≈ 2.0 GHz** effective core frequency.

## Git commits (Step 0 run)

See `snnMNIST.commit`, `snn-mlir.commit`, `gem5.commit` in this directory.
