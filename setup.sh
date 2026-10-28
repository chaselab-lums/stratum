#!/usr/bin/env bash
# Environment for snn-mlir + snnMNIST builds in Research-Work.
# Usage: source /home/wajahatriaz/Research-Work/setup.sh

RESEARCH_WORK="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

# MLIR 18 install (mlir-opt, mlir-translate, MLIRConfig.cmake)
MLIR_ROOT="$RESEARCH_WORK/snn-mlir/.local/mlir18/usr/lib/llvm-18"
export MLIR_DIR="$MLIR_ROOT/lib/cmake/mlir"
export MLIR_OPT_BIN="$MLIR_ROOT/bin/mlir-opt"
export MLIR_TRANSLATE_BIN="$MLIR_ROOT/bin/mlir-translate"

# llc + ld.lld (system LLVM 18; mlir18 package does not ship these)
export LLVM_BIN="/usr/bin"

export SNN_MLIR_ROOT="$RESEARCH_WORK/snn-mlir"
export SNN_OPT="$SNN_MLIR_ROOT/build/bin/snn-opt"

# snnMNIST Python venv
export PYTHON="$RESEARCH_WORK/snnMNIST/.venv/bin/python"

# RISC-V GCC
export RISCV="${RISCV:-${HOME}/RISCV/riscv}"
export PATH="${PATH}:${RISCV}/bin"

if [[ ! -f "$MLIR_DIR/MLIRConfig.cmake" ]]; then
  echo "WARNING: MLIRConfig.cmake not found at MLIR_DIR=$MLIR_DIR"
  echo "Expected MLIR at snn-mlir/.local/mlir18 (symlinked from scratch install)."
fi

if [[ ! -x "$SNN_OPT" ]]; then
  echo "WARNING: snn-opt not found at $SNN_OPT"
  echo "Copy or build it: cp ~/scratch/snn-mlir/build/bin/snn-opt $SNN_OPT"
fi

echo "Research-Work environment loaded:"
echo "  SNN_MLIR_ROOT=$SNN_MLIR_ROOT"
echo "  SNN_OPT=$SNN_OPT"
echo "  MLIR_DIR=$MLIR_DIR"
echo "  LLVM_BIN=$LLVM_BIN (llc, ld.lld)"
echo "  PYTHON=$PYTHON"
echo "  RISCV=$RISCV"
