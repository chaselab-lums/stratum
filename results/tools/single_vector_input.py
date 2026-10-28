#!/usr/bin/env python3
"""Build build/input.h that contains exactly one drawing (input_k.h)."""
from __future__ import annotations

import argparse
import re
from pathlib import Path


def read_n_steps(path: Path) -> int:
    for line in path.read_text().splitlines()[:32]:
        if line.strip().startswith("#define INPUT_N_STEPS"):
            return int(line.split()[-1])
    m = re.search(r"INPUT_N_STEPS\s+(\d+)", path.read_text())
    if m:
        return int(m.group(1))
    raise SystemExit(f"cannot read INPUT_N_STEPS from {path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("build_dir", type=Path)
    p.add_argument("index", type=int, help="0..4 drawing index")
    args = p.parse_args()
    idx = args.index
    part = args.build_dir / f"input_{idx}.h"
    if not part.exists():
        raise SystemExit(f"missing {part}")
    n_steps = read_n_steps(args.build_dir / "input.h") if (args.build_dir / "input.h").exists() else read_n_steps(part)
    text = [
        "// USER_DRAWN_INPUT_HEADER",
        "// single-vector header for gem5 ROI (one drawing)",
        f"#define INPUT_N_STEPS {n_steps}",
        "#ifndef INPUT_H",
        "#define INPUT_H",
        "#define NUM_TEST_VECTORS 1",
        "",
        "#include <stdint.h>",
        "",
        "int8_t L0_input[NUM_TEST_VECTORS][N_STEPS][INPUT_SIZE] = {",
        "    {",
        f'#include "input_{idx}.h"',
        "    },",
        "};",
        "",
        "#endif // INPUT_H",
        "",
    ]
    out = args.build_dir / "input.h"
    out.write_text("\n".join(text))
    print(f"wrote {out} for drawing {idx}")


if __name__ == "__main__":
    main()
