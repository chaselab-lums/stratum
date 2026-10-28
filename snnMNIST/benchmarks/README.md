# Hand-written fc1 micro-benchmarks (Task 7)

Compare LLVM lowerings against minimal C loops with the same dimensions
(`INPUT_SIZE=2312`, `HIDDEN=256`) and mean nnz $\approx 2.7$ on drawing~0.

Build (requires same RISC-V flags as `snnMNIST/Makefile`):

```bash
source ../../setup.sh
make -C benchmarks
../../run.sh benchmarks/fc1_bench.elf baremetal
```

Kernels: `ref_output_outer.c` (stride-$K$ gather), `ref_spike_major.c` (spike-outer),
`ref_dense.c` (full matvec). ROI markers match `include/gem5_profile.h`.
