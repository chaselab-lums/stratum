#!/usr/bin/env python3
# Copyright 2026 N Vision Systems And Technologies SL
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Load a trained SNN, test drawn inputs, and emit the snn-mlir C runtime.

Training stays in train.py. This script calls it only when a new checkpoint is requested.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import nir
import numpy as np
import torch
import torch.nn as nn
import tonic
from snntorch import utils

import _codegen
from train import build_mlir_network

try:
    import snn_mlir  # pyright: ignore[reportMissingImports]
except ImportError:
    snn_mlir = None  # type: ignore[assignment]


HERE = Path(__file__).parent
USER_DRAWN_INPUT_MARKER = "USER_DRAWN_INPUT_HEADER"


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _beta_to_tau_r(beta: float) -> tuple[float, float]:
    tau = 1.0
    r = 1.0 / max(1.0 - beta, 1e-6)
    return tau, r


def export_mlir_model_to_nir(net: nn.Module, beta: float, input_size: int, output_nir: Path) -> Path:
    if not isinstance(net, nn.Sequential) or len(net) < 4:
        raise ValueError("Expected nn.Sequential MLIR network")

    fc1 = net[0]
    fc2 = net[2]
    if not isinstance(fc1, nn.Linear) or not isinstance(fc2, nn.Linear):
        raise TypeError("Expected Linear layers at positions 0 and 2")

    w1 = fc1.weight.detach().cpu().numpy().astype(np.float32)
    w2 = fc2.weight.detach().cpu().numpy().astype(np.float32)
    b2 = fc2.bias.detach().cpu().numpy().astype(np.float32)

    hidden_size = w1.shape[0]
    out_size = w2.shape[0]
    tau, r = _beta_to_tau_r(beta)

    graph = nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([input_size], dtype=np.int64)}),
            "fc1": nir.Linear(weight=w1),
            "lif1": nir.LIF(
                tau=np.full((hidden_size,), tau, dtype=np.float32),
                r=np.full((hidden_size,), r, dtype=np.float32),
                v_leak=np.zeros((hidden_size,), dtype=np.float32),
                v_threshold=np.ones((hidden_size,), dtype=np.float32),
                v_reset=np.zeros((hidden_size,), dtype=np.float32),
            ),
            "fc2": nir.Affine(weight=w2, bias=b2),
            "lif2": nir.LIF(
                tau=np.full((out_size,), tau, dtype=np.float32),
                r=np.full((out_size,), r, dtype=np.float32),
                v_leak=np.zeros((out_size,), dtype=np.float32),
                v_threshold=np.ones((out_size,), dtype=np.float32),
                v_reset=np.zeros((out_size,), dtype=np.float32),
            ),
            "output": nir.Output(output_type={"output": np.array([out_size], dtype=np.int64)}),
        },
        edges=[
            ("input", "fc1"),
            ("fc1", "lif1"),
            ("lif1", "fc2"),
            ("fc2", "lif2"),
            ("lif2", "output"),
        ],
    )
    graph.infer_types()
    output_nir.parent.mkdir(parents=True, exist_ok=True)
    nir.write(output_nir, graph)
    return output_nir


def _read_input_header(
    input_file: Path,
    *,
    n_steps: int,
    input_size: int,
    num_vectors: int,
    device: torch.device,
) -> torch.Tensor:
    seq = torch.zeros((num_vectors, n_steps, input_size), dtype=torch.float32, device=device)
    if not input_file.exists():
        return seq

    text = input_file.read_text()
    clean = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
    clean = re.sub(r"/\*.*?\*/", "", clean, flags=re.DOTALL)

    include_pattern = re.compile(r"#include\s+\"(input_(\d+)\.h)\"")
    include_hits = include_pattern.findall(text)
    if include_hits:
        for file_name, vec_s in include_hits:
            vec = int(vec_s)
            if not (0 <= vec < num_vectors):
                continue
            vec_path = input_file.parent / file_name
            if not vec_path.exists():
                continue
            vec_text = vec_path.read_text()
            vec_text = re.sub(r"//.*?$", "", vec_text, flags=re.MULTILINE)
            vec_text = re.sub(r"/\*.*?\*/", "", vec_text, flags=re.DOTALL)
            for t_s, i_s, v_s in re.findall(r"\[(\d+)\]\[(\d+)\]\s*=\s*(-?\d+)", vec_text):
                t = int(t_s)
                i = int(i_s)
                v = int(v_s)
                if v != 0 and 0 <= t < n_steps and 0 <= i < input_size:
                    seq[vec, t, i] = 1.0
        return seq

    pattern_3d = re.compile(r"\[(\d+)\]\[(\d+)\]\[(\d+)\]\s*=\s*(-?\d+)")
    pattern_2d = re.compile(r"\[(\d+)\]\[(\d+)\]\s*=\s*(-?\d+)")

    found_3d = False
    for vec_s, t_s, i_s, v_s in pattern_3d.findall(clean):
        found_3d = True
        vec = int(vec_s)
        t = int(t_s)
        i = int(i_s)
        v = int(v_s)
        if v != 0 and 0 <= vec < num_vectors and 0 <= t < n_steps and 0 <= i < input_size:
            seq[vec, t, i] = 1.0

    if not found_3d:
        for t_s, i_s, v_s in pattern_2d.findall(clean):
            t = int(t_s)
            i = int(i_s)
            v = int(v_s)
            if v != 0 and 0 <= t < n_steps and 0 <= i < input_size:
                seq[0, t, i] = 1.0
    return seq


def _is_user_drawn_input_header(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        head = "\n".join(path.read_text().splitlines()[:8])
    except OSError:
        return False
    return USER_DRAWN_INPUT_MARKER in head


def _input_grid_width(input_size: int) -> int:
    side = int(round((input_size // 2) ** 0.5))
    if 2 * side * side != input_size:
        raise ValueError(f"INPUT_SIZE {input_size} is not 2*{side}x{side}")
    return side


def _print_input_spike_grid(step_values, *, grid_w: int, n_steps: int) -> None:
    """Print one 34x34 grid with all timesteps merged (. none, + ON, - OFF, * both)."""
    plane = grid_w * grid_w
    on = step_values[:, :plane].any(dim=0).reshape(grid_w, grid_w)
    off = step_values[:, plane : 2 * plane].any(dim=0).reshape(grid_w, grid_w)
    active_steps = sum(1 for t in range(n_steps) if step_values[t].any())
    print("Test input — all timesteps merged (.=none  +=ON  -=OFF  *=both):")
    print(f"(events in {active_steps}/{n_steps} timesteps, {grid_w}x{grid_w} grid)")
    for y in range(grid_w):
        chars = []
        for x in range(grid_w):
            on_sp = bool(on[y, x])
            off_sp = bool(off[y, x])
            if on_sp and off_sp:
                chars.append("*")
            elif on_sp:
                chars.append("+")
            elif off_sp:
                chars.append("-")
            else:
                chars.append(".")
        print(f"{y:2d} | {''.join(chars)}")


def _drawn_header_path(output_dir: Path) -> Path:
    build_input = output_dir / "input.h"
    if build_input.exists():
        return build_input
    return HERE / "input.h"


def _drawn_vector_count(path: Path) -> int:
    """How many vectors ``input.h`` currently lists."""
    if not path.exists():
        return 0
    text = path.read_text()
    for line in text.splitlines()[:16]:
        line = line.strip()
        if line.startswith("#define NUM_TEST_VECTORS"):
            return int(line.split()[-1])
    indexes = [int(value) for value in re.findall(r'#include\s+"input_(\d+)\.h"', text)]
    return len(indexes)


def _saved_recording_count(out_dir: Path) -> int:
    """Highest input_N.h present on disk (draw-input saves up to five)."""
    indexes = []
    for part in out_dir.glob("input_*.h"):
        stem = part.stem.removeprefix("input_")
        if stem.isdigit():
            indexes.append(int(stem))
    return max(indexes) + 1 if indexes else 0


def _rewrite_input_header(path: Path, count: int, n_steps: int | None = None) -> None:
    """Point input.h at the first ``count`` recordings (input_0.h, ...)."""
    if count < 1:
        raise ValueError("count must be >= 1")
    if n_steps is None:
        n_steps = _user_drawn_input_steps(path)
    if n_steps is None:
        raise ValueError(f"could not read INPUT_N_STEPS from {path}")
    includes = []
    for index in range(count):
        part = path.parent / f"input_{index}.h"
        if not part.exists():
            raise SystemExit(
                f"Need input_0.h .. input_{count - 1}.h under {path.parent}; "
                f"missing {part.name}. Run make draw-input or lower --test-vectors."
            )
        includes.append("    {")
        includes.append(f'#include "input_{index}.h"')
        includes.append("    },")
    master = [
        "// Auto-generated from draw_input_app.py for snn-mlir runtime",
        f"// {USER_DRAWN_INPUT_MARKER}",
        f"#define INPUT_N_STEPS {n_steps}",
        "#ifndef INPUT_H",
        "#define INPUT_H",
        f"#define NUM_TEST_VECTORS {count}",
        "",
        "#include <stdint.h>",
        "",
        "int8_t L0_input[NUM_TEST_VECTORS][N_STEPS][INPUT_SIZE] = {",
        *includes,
        "};",
        "",
        "#endif // INPUT_H",
        "",
    ]
    path.write_text("\n".join(master))


def _limit_test_vectors(header: Path, limit: int | None) -> int:
    """Return the number of vectors in ``header`` after applying ``limit``."""
    _require_drawn_header(header)
    listed = _drawn_vector_count(header)
    on_disk = _saved_recording_count(header.parent)
    available = max(listed, on_disk)
    if limit is None:
        return listed
    if limit < 1:
        raise SystemExit("--test-vectors must be >= 1")
    if limit > available:
        raise SystemExit(
            f"--test-vectors={limit} but only {available} recording(s) under "
            f"{header.parent}. Run make draw-input."
        )
    if limit != listed:
        if os.environ.get("SNN_SKIP_INPUT_REWRITE") == "1":
            if limit > listed:
                raise SystemExit(
                    f"--test-vectors={limit} but input.h lists {listed} vector(s) "
                    f"and SNN_SKIP_INPUT_REWRITE=1"
                )
            return limit
        steps = _user_drawn_input_steps(header)
        _rewrite_input_header(header, limit, n_steps=steps)
        print(f"Using {limit} test vector(s) in {header} (was {listed})")
    return limit


def _require_drawn_header(path: Path) -> int:
    if not _is_user_drawn_input_header(path):
        raise SystemExit(
            "Test images come from the drawing app.\n"
            "Run: make draw-input\n"
            f"Expected the five recordings in {path}"
        )
    count = _drawn_vector_count(path)
    if count < 1:
        raise SystemExit(f"{path} has no recordings. Run: make draw-input")
    return count


def _user_drawn_input_steps(path: Path) -> int | None:
    if not path.exists():
        return None
    for line in path.read_text().splitlines()[:12]:
        line = line.strip()
        if line.startswith("#define INPUT_N_STEPS"):
            return int(line.split()[-1])
    return None


def _n_steps_from_drawing(output_dir: Path, n_steps: int) -> int:
    header = _drawn_header_path(output_dir)
    if not _is_user_drawn_input_header(header):
        return n_steps
    stored = _user_drawn_input_steps(header)
    if stored is None or stored == n_steps:
        return n_steps
    print(f"Using INPUT_N_STEPS={stored} from {header}")
    return stored


def _load_trained_net(weights_path: Path, beta: float, hidden_size: int, device: torch.device) -> tuple[nn.Module, int]:
    if not weights_path.exists():
        raise SystemExit(f"Weights not found: {weights_path}")
    input_size = int(np.prod(tonic.datasets.NMNIST.sensor_size))
    net = build_mlir_network(input_size=input_size, hidden_size=hidden_size, beta=beta).to(device)
    state = torch.load(weights_path, map_location="cpu")
    net.load_state_dict(state, strict=False)
    net.eval()
    return net, input_size


@torch.no_grad()
def run_python_test(
    net: nn.Module,
    *,
    input_file: Path,
    n_steps: int,
    input_size: int,
    num_vectors: int,
    device: torch.device,
) -> None:
    net.eval()
    utils.reset(net)

    x_seq = _read_input_header(
        input_file,
        n_steps=n_steps,
        input_size=input_size,
        num_vectors=num_vectors,
        device=device,
    )

    for vec in range(num_vectors):
        utils.reset(net)
        spike_counts = torch.zeros((10,), dtype=torch.int32)
        print(f"\n=== Test vector {vec} ===")
        grid_w = _input_grid_width(input_size)
        _print_input_spike_grid(x_seq[vec].cpu(), grid_w=grid_w, n_steps=n_steps)
        print("\nPer-step class spikes:")

        for t in range(n_steps):
            spk_out, _mem_out = net(x_seq[vec, t].unsqueeze(0))
            step_spikes = (spk_out[0] != 0).to(torch.int32).cpu()
            spike_counts += step_spikes
            print(f"{t:02d}  {step_spikes.tolist()}")

        pred = int(torch.argmax(spike_counts).item())
        print("\nSpike counts per class:", spike_counts.tolist())
        print(f"FINAL_CLASS[{vec}]", pred)


def _train(
    *,
    weights_path: Path,
    epochs: int | None,
    batch_size: int | None,
    num_iters: int | None,
    num_samples: int | None,
) -> None:
    cmd = [
        sys.executable,
        str(HERE / "train.py"),
        "--arch",
        "mlir",
        "--save",
        str(weights_path),
    ]
    if epochs is not None:
        cmd.extend(["--epochs", str(epochs)])
    if batch_size is not None:
        cmd.extend(["--batch-size", str(batch_size)])
    if num_iters is not None:
        cmd.extend(["--num-iters", str(num_iters)])
    if num_samples is not None:
        cmd.extend(["--num-samples", str(num_samples)])

    print(f"Training via: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _generate(
    *,
    net: nn.Module,
    beta: float,
    input_size: int,
    n_steps: int,
    quantize: bool,
    out_dir: Path,
    test_vectors: int | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if snn_mlir is None:
        raise SystemExit("snn_mlir import failed. Activate the snn-mlir environment and re-run.")

    nir_path = export_mlir_model_to_nir(
        net=net,
        beta=beta,
        input_size=input_size,
        output_nir=out_dir / "network.nir",
    )

    input_header = out_dir / "input.h"
    _limit_test_vectors(input_header, test_vectors)
    stored_steps = _user_drawn_input_steps(input_header)
    if stored_steps is not None and stored_steps != n_steps:
        raise SystemExit(
            f"User-drawn input has INPUT_N_STEPS={stored_steps} but "
            f"--n-steps={n_steps}. "
            f"Re-run with N_STEPS={stored_steps} or regenerate via draw-input."
        )
    print(f"Reusing user-drawn input header: {input_header}")

    mlir_path = out_dir / "network.mlir"
    snn_mlir_api: Any = snn_mlir
    snn_mlir_api.export(nir_path, mlir_path, quantize=quantize)
    _codegen.export(
        nir_path,
        out_dir,
        quantize=quantize,
        n_steps=n_steps,
        index_bits=32,
        input_file=None,
    )

    print("Generated snn-mlir artifacts:")
    print(f"  {nir_path}")
    print(f"  {mlir_path}")
    print(f"  {out_dir / 'main.c'}")
    print(f"  {out_dir / 'snn_data.h'}")
    print(f"  {input_header}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--n-steps", type=int, default=25)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--weights", type=Path, default=HERE / "weights.pt")
    parser.add_argument("--output-dir", type=Path, default=HERE / "build")
    parser.add_argument("--python-test", action="store_true")
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--train-epochs", type=int, default=None)
    parser.add_argument("--train-num-iters", type=int, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--train-num-samples", type=int, default=None)
    parser.add_argument(
        "--test-vectors",
        type=int,
        default=None,
        help="Use only the first N drawn recordings in build/input.h (default: all listed)",
    )
    args = parser.parse_args()

    if not args.no_train:
        _train(
            weights_path=args.weights,
            epochs=args.train_epochs,
            batch_size=args.train_batch_size,
            num_iters=args.train_num_iters,
            num_samples=args.train_num_samples,
        )

    n_steps = _n_steps_from_drawing(args.output_dir, args.n_steps)
    device = _resolve_device()
    net, input_size = _load_trained_net(args.weights, args.beta, args.hidden_size, device)

    if args.python_test:
        input_header = _drawn_header_path(args.output_dir)
        num_vectors = _require_drawn_header(input_header)
        run_python_test(
            net,
            input_file=input_header,
            n_steps=n_steps,
            input_size=input_size,
            num_vectors=num_vectors,
            device=device,
        )
        return

    _generate(
        net=net,
        beta=args.beta,
        input_size=input_size,
        n_steps=n_steps,
        quantize=args.quantize,
        out_dir=args.output_dir,
        test_vectors=args.test_vectors,
    )

    mode = "quantized" if args.quantize else "float32"
    print(f"Done ({mode}). Files in {args.output_dir}:")
    for path in sorted(args.output_dir.iterdir()):
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
