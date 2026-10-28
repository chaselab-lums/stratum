# Copyright 2026 N Vision Systems And Technologies SL
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path

import nir
import numpy as np
import torch

from config import FC1_INIT_WEIGHT, LIF_R, LIF_TAU, LIF_THRESHOLD, LIF_V_RESET


HERE = Path(__file__).parent


def _load_fc1_weight(weights_path: Path) -> np.ndarray:
    payload = torch.load(weights_path, map_location="cpu")

    if isinstance(payload, dict):
        if "fc1.weight" not in payload:
            raise KeyError(f"Expected key 'fc1.weight' in {weights_path}")
        weight_tensor = payload["fc1.weight"]
    elif isinstance(payload, torch.Tensor):
        weight_tensor = payload
    else:
        raise TypeError(f"Unsupported weights format in {weights_path}: {type(payload)!r}")

    weight = weight_tensor.detach().cpu().numpy().astype(np.float32)
    if weight.shape != (1, 1):
        raise ValueError(f"Expected fc1.weight shape (1, 1), got {weight.shape}")
    return weight


def export_network_nir(output_path: Path | None = None, weights_path: Path | None = None) -> Path:
    # Default deterministic weight if no trained weights are provided.
    fc1_weight = np.array([[FC1_INIT_WEIGHT]], dtype=np.float32)
    if weights_path is not None:
        fc1_weight = _load_fc1_weight(weights_path)

    # Direct NIR construction is used here for a stable minimal graph.
    # Node names are referenced in generated debug/state prints, so keep them intentional.
    nir_model = nir.NIRGraph(
        nodes={
            # Shape [1] means one scalar input feature per timestep.
            "input": nir.Input(input_type={"input": np.array([1])}),
            "fc1": nir.Linear(weight=fc1_weight),
            "lif1": nir.LIF(
                # Single-neuron LIF parameters; arrays must match neuron count.
                tau=np.array([LIF_TAU], dtype=np.float32),
                r=np.array([LIF_R], dtype=np.float32),
                v_leak=np.array([0.0], dtype=np.float32),
                v_threshold=np.array([LIF_THRESHOLD], dtype=np.float32),
                v_reset=np.array([LIF_V_RESET], dtype=np.float32),
            ),
            "output": nir.Output(output_type={"output": np.array([1])}),
        },
        edges=[("input", "fc1"), ("fc1", "lif1"), ("lif1", "output")],
    )
    # Let NIR infer/check tensor compatibility before writing the file.
    nir_model.infer_types()

    if output_path is None:
        output_path = HERE / "network.nir"

    nir.write(output_path, nir_model)
    return output_path


def main():
    default_weights = HERE / "weights.pt"
    output_path = export_network_nir(weights_path=default_weights if default_weights.exists() else None)
    print(f"Created {output_path.name}")


if __name__ == "__main__":
    main()