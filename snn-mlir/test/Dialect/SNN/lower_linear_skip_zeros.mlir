// Copyright 2026 N Vision Systems And Technologies SL
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Test: integer snn.linear with skip-zeros omits the multiply.
// RUN: %snn-opt --convert-snn-to-linalg=skip-zeros=true %s | %FileCheck %s

// CHECK-LABEL: func.func @linear_quantized_skip_zeros
// CHECK:         linalg.fill
// CHECK:         scf.for
// CHECK:         arith.cmpi eq
// CHECK:         scf.if
// CHECK:         arith.extsi
// CHECK:         arith.addi
// CHECK-NOT:     arith.muli
// CHECK-NOT:     snn.linear
func.func @linear_quantized_skip_zeros(
    %input:   memref<32xi8>,
    %weights: memref<64x32xi8>,
    %out:     memref<64xi32>
) {
  snn.linear ins(%input, %weights) out(%out) {w_scale = 7 : i64}
      : memref<32xi8>, memref<64x32xi8> -> memref<64xi32>
  return
}

// Float spikes are stored as 0.0 or 1.0. Live indices are gathered once, then
// each output adds the weight only at those indices.
// CHECK-LABEL: func.func @linear_float_skip_zeros
// CHECK:         linalg.fill
// CHECK:         memref.alloca
// CHECK:         scf.for
// CHECK:         arith.cmpf oeq
// CHECK:         scf.if
// CHECK:         memref.store
// CHECK:         scf.for
// CHECK:         arith.addf
// CHECK-NOT:     arith.mulf
// CHECK-NOT:     linalg.matvec
// CHECK-NOT:     snn.linear
func.func @linear_float_skip_zeros(
    %input:   memref<32xf32>,
    %weights: memref<64x32xf32>,
    %out:     memref<64xf32>
) {
  snn.linear ins(%input, %weights) out(%out)
      : memref<32xf32>, memref<64x32xf32> -> memref<64xf32>
  return
}

// CHECK-LABEL: func.func @linear_float_skip_zeros_bias
// CHECK:         scf.if
// CHECK:         linalg.generic
// CHECK:         arith.addf
// CHECK-NOT:     arith.mulf
// CHECK-NOT:     linalg.matvec
// CHECK-NOT:     snn.linear
func.func @linear_float_skip_zeros_bias(
    %input:   memref<32xf32>,
    %weights: memref<64x32xf32>,
    %bias:    memref<64xf32>,
    %out:     memref<64xf32>
) {
  snn.linear ins(%input, %weights) bias(%bias : memref<64xf32>) out(%out)
      : memref<32xf32>, memref<64x32xf32> -> memref<64xf32>
  return
}

// Bias stays a dense elementwise add after the gated accumulation.
// CHECK-LABEL: func.func @linear_quantized_skip_zeros_bias
// CHECK:         scf.if
// CHECK:         linalg.generic
// CHECK:         arith.addi
// CHECK-NOT:     arith.muli
// CHECK-NOT:     snn.linear
func.func @linear_quantized_skip_zeros_bias(
    %input:   memref<32xi8>,
    %weights: memref<64x32xi8>,
    %bias:    memref<64xi32>,
    %out:     memref<64xi32>
) {
  snn.linear ins(%input, %weights) bias(%bias : memref<64xi32>) out(%out)
      {w_scale = 7 : i64}
      : memref<32xi8>, memref<64x32xi8> -> memref<64xi32>
  return
}

// A constant [2, 4] matrix is stored [4, 2]. Row k of the new matrix is column
// k of the old one, so the two weights of one spike are adjacent.
// CHECK: memref.global "private" constant @w_km_src : memref<4x2xf32> = dense<{{\[}}[1.000000e+00, 5.000000e+00], [2.000000e+00, 6.000000e+00], [3.000000e+00, 7.000000e+00], [4.000000e+00, 8.000000e+00]]>
// CHECK-NOT: memref<2x4xf32>
// CHECK-LABEL: func.func @linear_float_skip_zeros_kmajor
// CHECK: memref.get_global @w_km_src : memref<4x2xf32>
// CHECK: arith.addf
// CHECK-NOT: arith.mulf
// CHECK-NOT: linalg.matvec
// CHECK-NOT: snn.linear
memref.global "private" constant @w_km_src : memref<2x4xf32> = dense<[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]>
func.func @linear_float_skip_zeros_kmajor(
    %input: memref<4xf32>,
    %out: memref<2xf32>
) {
  %w = memref.get_global @w_km_src : memref<2x4xf32>
  snn.linear ins(%input, %w) out(%out)
      : memref<4xf32>, memref<2x4xf32> -> memref<2xf32>
  return
}
