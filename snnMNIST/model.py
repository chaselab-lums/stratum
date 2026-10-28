# Copyright 2026 N Vision Systems And Technologies SL
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Minimal single-neuron SNN used by snnModel.

Topology:
    x_t [B, 1] --Linear(1x1)--> I_t [B, 1] --LIF--> s_t [B, 1]

Forward returns a stacked spike tensor over time:
    S = stack_t(s_t) with shape [T, B, 1]
"""

import torch
import torch.nn as nn
import snntorch as snn

from config import FC1_INIT_WEIGHT, LIF_BETA, LIF_THRESHOLD


class Network(nn.Module):
    def __init__(self, beta=LIF_BETA, threshold=LIF_THRESHOLD):
        super().__init__()

        # Minimal 1->1 topology used by export_nir.py and input.h.
        # If you widen this layer, also update NIR shapes and INPUT_SIZE/OUTPUT_SIZE assumptions.
        self.fc1 = nn.Linear(1, 1, bias=False)
        # Higher threshold helps validate temporal integration with repeated spikes.
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold)

        with torch.no_grad():
            # Deterministic baseline weight for predictable spike behavior.
            self.fc1.weight.fill_(FC1_INIT_WEIGHT)

    def forward(self, x, n_steps=25):
        """Run temporal simulation and return spike tensor [T, B, 1].

        Supported input forms:
        - Static input: x shape [B, 1], reused every timestep.
        - Sequence input: x shape [T, B, 1], one value per timestep.
        """
        # Keep hidden state across timesteps; reset once per sample/sequence.
        mem1 = self.lif1.init_leaky()

        spk_rec = []

        if x.dim() == 3:
            # x is already a timestep sequence [T, B, 1].
            for t in range(x.shape[0]):
                cur1 = self.fc1(x[t]).squeeze()
                spk1, mem1 = self.lif1(cur1, mem1)
                spk_rec.append(spk1)
        else:
            for _ in range(n_steps):
                # squeeze() keeps scalar-like shape expected by single-neuron LIF path.
                cur1 = self.fc1(x).squeeze()
                spk1, mem1 = self.lif1(cur1, mem1)
                spk_rec.append(spk1)

        return torch.stack(spk_rec)