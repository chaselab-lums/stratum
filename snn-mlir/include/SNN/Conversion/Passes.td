// Copyright 2026 N Vision Systems And Technologies SL
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===- Passes.td - SNN conversion passes -------------------*- tablegen -*-===//

#ifndef SNN_CONVERSION_PASSES_TD
#define SNN_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

def ConvertSNNToLinalg : Pass<"convert-snn-to-linalg", "mlir::ModuleOp"> {
  let summary = "Convert SNN dialect ops to Linalg/Arith for CPU execution";
  let description = [{
    Lowers the SNN dialect (`snn.linear`, `snn.rescale`, `snn.cubalif`,
    `snn.lif`, `snn.li`, `snn.cubali`) to standard `linalg`/`arith`/`memref`
    operations that any MLIR CPU backend can consume. Dispatches on operand
    element types: float operands lower to floating-point arithmetic, integer
    operands lower to Q12 fixed-point arithmetic.

    With `skip-zeros`, a statically shaped `snn.linear` accumulates a weight
    only when the input spike equals 1. An integer synapse adds `sext(weight)`
    inside the full reduction loop. A float synapse first records the indices
    where the spike is 1.0, then adds the weight at those indices only, which
    is the same product and the same per-output addition order as the dense
    matvec. With `k-major=true` (default), a constant float matrix is rewritten
    from `[N, K]` to `[K, N]` and the gather nest is spike-outer. With
    `k-major=false`, the matrix stays `[N, K]` and the nest is output-outer.
  }];
  let constructor = "snn::createConvertSNNToLinalgPass()";
  let options = [
    Option<"skipZeros", "skip-zeros", "bool", /*default=*/"false",
           "Gate snn.linear on a spike equal to 1 and omit the multiply">,
    Option<"kMajor", "k-major", "bool", /*default=*/"true",
           "Transpose constant float weights to [K, N] and spike-outer loops">
  ];
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
  ];
}

#endif // SNN_CONVERSION_PASSES_TD
