#!/usr/bin/env bash
# Copyright 2026 N Vision Systems And Technologies SL
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SNN dialect -> LLVM IR (.ll) on RISC-V Linux.
# Uses three path variables (override as env vars if needed):
#   SNN_MLIR_ROOT  path to snn-mlir checkout
#   MLIR_DIR       LLVM CMake mlir directory
#   LLVM_BIN       LLVM bin directory containing mlir-opt/mlir-translate

set -euo pipefail

: "${SNN_MLIR_ROOT:?SNN_MLIR_ROOT is required}"
: "${SNN_OPT:?SNN_OPT is required}"

: "${MLIR_DIR:?MLIR_DIR is required}"

: "${LLVM_BIN:?LLVM_BIN is required}"
: "${MLIR_OPT_BIN:?MLIR_OPT_BIN is required}"
: "${MLIR_TRANSLATE_BIN:?MLIR_TRANSLATE_BIN is required}"

if [ ! -x "$MLIR_OPT_BIN" ] || [ ! -x "$MLIR_TRANSLATE_BIN" ]; then
    echo "Error: mlir-opt/mlir-translate not found or not executable"
    echo "Check MLIR_OPT_BIN and MLIR_TRANSLATE_BIN"
    exit 1
fi

INPUT="${1:?Usage: snnModel/lower_cpu_linux.sh <input.mlir>}"
OUT_LL="$(dirname "$INPUT")/$(basename "$INPUT" .mlir).ll"

# RV32 target: keep memref/loop index width at 32 bits (matches main.c ABI).
INDEX_BW=32

# Lowering flags (override for experiments):
#   SNN_LINALG_OPTS='skip-zeros=true k-major=false'  # output-outer reference
#   SNN_LINALG_OPTS='skip-zeros=true k-major=true'   # production spike-major (default)
#   SNN_LINALG_OPTS='skip-zeros=false'               # dense matvec baseline
SNN_LINALG_OPTS="${SNN_LINALG_OPTS:-skip-zeros=true k-major=true}"
"$SNN_OPT" "$INPUT" --convert-snn-to-linalg="${SNN_LINALG_OPTS}" \
    | sed 's/^module {/module attributes { llvm.target_triple = "riscv32-unknown-linux-gnu" } {/' \
    | "$MLIR_OPT_BIN" \
        -convert-linalg-to-affine-loops \
        -affine-loop-invariant-code-motion \
        -affine-scalrep \
        -cse \
        -canonicalize \
        -lower-affine \
        -convert-scf-to-cf \
        -convert-func-to-llvm="index-bitwidth=${INDEX_BW}" \
        -convert-cf-to-llvm="index-bitwidth=${INDEX_BW}" \
        -finalize-memref-to-llvm="index-bitwidth=${INDEX_BW}" \
        -convert-arith-to-llvm="index-bitwidth=${INDEX_BW}" \
        -convert-index-to-llvm="index-bitwidth=${INDEX_BW}" \
        -reconcile-unrealized-casts \
        -canonicalize \
        -llvm-request-c-wrappers \
    | "$MLIR_TRANSLATE_BIN" --mlir-to-llvmir \
    > "$OUT_LL"

echo "LLVM IR written to: $OUT_LL"