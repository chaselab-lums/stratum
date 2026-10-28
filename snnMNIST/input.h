// Copyright 2026 N Vision Systems And Technologies SL
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Legacy compatibility shim.
//
// Canonical runtime inference input is now generated/managed at:
//   snnModel/build/input.h
//
// Typical workflow:
//   1) make draw-input    (optional, create user-drawn input)
//   2) make generate
//
// This file intentionally forwards to build/input.h to avoid ambiguity.

#ifndef LEGACY_INPUT_SHIM_H
#define LEGACY_INPUT_SHIM_H

#if defined(__has_include)
#  if __has_include("build/input.h")
#    include "build/input.h"
#  else
#    error "build/input.h not found. Run 'make generate' (or 'make draw-input' then 'make generate')."
#  endif
#else
#  error "Compiler does not support __has_include. Include build/input.h directly."
#endif

#endif // LEGACY_INPUT_SHIM_H
