# Reproducing Stratum gem5 experiments

All commands assume a Linux host with RISC-V GCC, LLVM 18 (`llc`, `mlir-opt`), built `snn-opt`, and gem5 under `Research-Work/gem5`.

## Environment

```bash
source /home/wajahatriaz/Research-Work/setup.sh
cd /home/wajahatriaz/Research-Work
```

## Step 0 — layout A/B (drawing 0, `TEST_VECTORS=1`)

```bash
bash results/step0/run_reproduce.sh
```

Artifacts: `results/step0/step0-repro-<timestamp>/`.

Reference (output-outer):

```bash
cd snnMNIST
SNN_LINALG_OPTS='skip-zeros=true k-major=false' make build NO_TRAIN=1 TEST_VECTORS=1
cd ..
GEM5_OUTDIR=/tmp/gem5-ref run.sh snnMNIST/build/sim.elf baremetal
python3 results/extract_roi_stats.py /tmp/gem5-ref/stats.txt
```

Production (spike-major):

```bash
cd snnMNIST
SNN_LINALG_OPTS='skip-zeros=true k-major=true' make build NO_TRAIN=1 TEST_VECTORS=1
```

Oct 11 2026 reproduction (`step0-repro-20261011`): **7.038 ms / IPC 1.157** (ref) vs **4.423 ms / IPC 1.692** (prod). Archived Oct 8 run `20261008-181456` used **five** test vectors; its `stats.txt` reflects the **last** vector, not drawing 0.

## Full experiment matrix (Tasks 2–6)

```bash
python3 results/run_experiments.py          # skip completed runs
python3 results/run_experiments.py --force  # re-run everything
python3 results/make_tables.py
```

CSV: `results/metrics.csv`. Per-run trees: `results/runs/<run_id>/`.

### gem5 knobs

| Variable | Default | Notes |
|----------|---------|--------|
| `GEM5_BOOM` | `1` | BOOM-class O3 bare-metal |
| `GEM5_L2_SIZE` | `4MiB` | L2 sweep |
| `GEM5_BOOM_WIDTH` | `4` | 2 / 4 / 8-wide |

### Single drawing *k* (0–4)

```bash
python3 results/tools/single_vector_input.py snnMNIST/build 3
cd snnMNIST && make -f Makefile build NO_TRAIN=1 TEST_VECTORS=1
```

### Synthetic input density

```bash
python3 results/tools/generate_synthetic_input.py snnMNIST/build 2.7 --seed 42
cd snnMNIST && SNN_LINALG_OPTS='skip-zeros=true k-major=true' make build NO_TRAIN=1 TEST_VECTORS=1
```

## Phase attribution (Task 1)

```bash
cd snnMNIST
SNN_LINALG_OPTS='skip-zeros=true k-major=true' make -f Makefile generate NO_TRAIN=1 TEST_VECTORS=1
python3 ../results/tools/inject_phase_markers.py build/network.ll build/network_phase.ll
PHASE_PROFILE=1 make -f Makefile build NO_TRAIN=1 TEST_VECTORS=1 NETWORK_LL=build/network_phase.ll
```

Parse `PHASE_CYCLES` lines from `gem5/.../program.out`.

## Commits recorded at Step 0

See `results/step0/step0-repro-20261011/*.commit` for pinned `snnMNIST`, `snn-mlir`, and `gem5` SHAs.
