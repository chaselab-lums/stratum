#!/usr/bin/env python3
"""Synthetic N-MNIST-like spike trains with controlled mean nnz per frame."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

INPUT_SIZE = 2312
N_STEPS = 300


def write_vector_fragment(path: Path, mean_nnz: float, seed: int) -> None:
    rng = random.Random(seed)
    lines: list[str] = []
    for step in range(N_STEPS):
        k = max(0, int(round(rng.gauss(mean_nnz, max(0.3, mean_nnz * 0.15)))))
        k = min(k, INPUT_SIZE)
        for j in rng.sample(range(INPUT_SIZE), k):
            lines.append(f"[{step}][{j}] = 1,")
    path.write_text("\n".join(lines) + "\n")


def write_input_h(build_dir: Path, fragment_name: str, n_steps: int) -> None:
    master = [
        "// USER_DRAWN_INPUT_HEADER",
        "// synthetic test vector",
        f"#define INPUT_N_STEPS {n_steps}",
        "#ifndef INPUT_H",
        "#define INPUT_H",
        "#define NUM_TEST_VECTORS 1",
        "",
        "#include <stdint.h>",
        "",
        "int8_t L0_input[NUM_TEST_VECTORS][N_STEPS][INPUT_SIZE] = {",
        "    {",
        f'#include "{fragment_name}"',
        "    },",
        "};",
        "",
        "#endif // INPUT_H",
        "",
    ]
    (build_dir / "input.h").write_text("\n".join(master))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("build_dir", type=Path)
    p.add_argument("mean_nnz", type=float)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", default=None)
    args = p.parse_args()
    tag = args.tag or f"nnz{args.mean_nnz:.1f}".replace(".", "p")
    args.build_dir.mkdir(parents=True, exist_ok=True)
    frag = f"synthetic_{tag}.h"
    write_vector_fragment(args.build_dir / frag, args.mean_nnz, args.seed)
    write_input_h(args.build_dir, frag, N_STEPS)
    print(f"synthetic input tag={tag} mean_nnz={args.mean_nnz}")


if __name__ == "__main__":
    main()
