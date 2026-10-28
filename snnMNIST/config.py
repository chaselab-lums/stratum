# Copyright 2026 N Vision Systems And Technologies SL
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Shared configuration constants for the snnModel example.

Each constant below is intentionally module-level so run.py, model.py,
train.py, and export_nir.py use one source of truth.
"""

# Runtime / model defaults ---------------------------------------------------

# Number of simulation timesteps for per-sample sequence evaluation.
N_STEPS_DEFAULT = 25

# Initial scalar weight for the 1x1 linear layer before training.
FC1_INIT_WEIGHT = 1.0

# SNNTorch Leaky neuron decay factor used by model.py float training/inference.
LIF_BETA = 0.95

# Membrane threshold for spike generation.
LIF_THRESHOLD = 2.0

# LIF membrane time constant used by NIR export and float-equation parity checks.
LIF_TAU = 20.0

# LIF resistance parameter chosen so exported decay matches LIF_BETA:
# decay = 1 - 1 / LIF_R  =>  LIF_R = 1 / (1 - LIF_BETA)
LIF_R = 1.0 / (1.0 - LIF_BETA)

# Membrane reset value applied immediately after a spike event.
LIF_V_RESET = 0.0

# Quantization ---------------------------------------------------------------

# Fixed-point fractional-bit scale (Q12) for quantized membrane dynamics.
D_SCALE = 12

# Training defaults ----------------------------------------------------------

# Default mini-batch size for train.py DataLoader.
BATCH_SIZE_DEFAULT = 64

# Default number of training epochs.
EPOCHS_DEFAULT = 120

# Default total number of synthetic training samples.
NUM_SAMPLES_DEFAULT = 2048

# Bernoulli spike probability for class-0 (sparse) synthetic sequences.
CLASS0_RATE = 0.12

# Bernoulli spike probability for class-1 (dense) synthetic sequences.
CLASS1_RATE = 0.55

# Fraction of dataset reserved for validation split.
VAL_SPLIT = 0.2
