#!/usr/bin/env python3
# Copyright 2026 N Vision Systems And Technologies SL
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Simple draw app to create user-provided input headers for snnModel.

Draw a 34x34 digit on the N-MNIST sensor and save five recordings:
- build/input.h
- build/input_0.h through build/input_4.h

The generated input.h contains a marker so run.py reuses it on make generate.
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

try:
    import tkinter as _tk
    from tkinter import messagebox as _messagebox
except ModuleNotFoundError:  # pragma: no cover - depends on system packages
    _tk = None  # type: ignore[assignment]
    _messagebox = None  # type: ignore[assignment]

tk: Any = _tk
messagebox: Any = _messagebox


HERE = Path(__file__).parent
OUTPUT_DIR_DEFAULT = HERE / "build"

GRID = 34
CELL = 14
TARGET_W = GRID
TARGET_H = GRID
INPUT_SIZE = 2 * TARGET_W * TARGET_H
MARKER = "USER_DRAWN_INPUT_HEADER"
RECORDING_COUNT = 5
# Training bins each N-MNIST recording with a 1 ms window. A recording is
# three 100 ms saccades, so the runtime has to see about 300 frames.
NMNIST_STEPS = 300
# Centroid drift measured on the test set: about (+3, 0), then (-1, -3).
_SACCADE_CORNERS = ((0.0, 0.0), (3.0, 0.0), (2.0, -3.0), (0.0, 0.0))


def _round_half_up(value: float) -> int:
    return int(value + 0.5) if value >= 0.0 else int(value - 0.5)


def _saccade_position(step: int, n_steps: int, amp: int) -> tuple[float, float]:
    """Continuous position along the three N-MNIST saccades."""
    if n_steps <= 1 or amp <= 0:
        return 0.0, 0.0
    scale = amp / 3.0
    phase = (step / float(n_steps)) * 3.0
    seg = min(int(phase), 2)
    frac = phase - float(seg)
    x0, y0 = _SACCADE_CORNERS[seg]
    x1, y1 = _SACCADE_CORNERS[seg + 1]
    return ((1.0 - frac) * x0 + frac * x1) * scale, ((1.0 - frac) * y0 + frac * y1) * scale


def _occupancy_events(prev: list[list[int]], cur: list[list[int]]) -> list[int]:
    events: list[int] = []
    for y in range(TARGET_H):
        for x in range(TARGET_W):
            if cur[y][x] == 1 and prev[y][x] == 0:
                events.append(y * TARGET_W + x)
            elif cur[y][x] == 0 and prev[y][x] == 1:
                events.append((TARGET_W * TARGET_H) + (y * TARGET_W + x))
    return events


def parse_steps(text: str, n_steps: int) -> list[int]:
    """Parse step specification like: 0,1,2 or 0-4,8,10-12."""
    raw = text.strip()
    if not raw:
        return [0]

    out: set[int] = set()
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        if "-" in p:
            a_s, b_s = p.split("-", 1)
            a = int(a_s)
            b = int(b_s)
            if a > b:
                a, b = b, a
            for t in range(a, b + 1):
                if 0 <= t < n_steps:
                    out.add(t)
        else:
            t = int(p)
            if 0 <= t < n_steps:
                out.add(t)

    if not out:
        return [0]
    return sorted(out)


class DrawApp:
    def __init__(self, root: Any) -> None:
        self.root = root
        self.root.title("N-MNIST Draw 34x34 -> input.h")

        self.pixels = [[0 for _ in range(GRID)] for _ in range(GRID)]

        self.canvas = tk.Canvas(root, width=GRID * CELL, height=GRID * CELL, bg="white")
        self.canvas.grid(row=0, column=0, columnspan=4, padx=8, pady=8)
        self.canvas.bind("<Button-1>", self.on_draw)
        self.canvas.bind("<B1-Motion>", self.on_draw)

        self.step_count_var = tk.StringVar(value=str(NMNIST_STEPS))
        self.step_spec_var = tk.StringVar(value=f"0-{NMNIST_STEPS - 1}")
        self.saccade_amp_var = tk.StringVar(value="3")
        self.noise_ppm_var = tk.StringVar(value="0")
        self.seed_var = tk.StringVar(value="0")
        self.out_dir_var = tk.StringVar(value=str(OUTPUT_DIR_DEFAULT))
        self.max_events_per_step_var = tk.StringVar(value="160")

        tk.Label(root, text="N_STEPS").grid(row=1, column=0, sticky="w", padx=8)
        tk.Entry(root, textvariable=self.step_count_var, width=8).grid(row=1, column=1, sticky="w")

        tk.Label(root, text="Emit steps").grid(row=2, column=0, sticky="w", padx=8)
        tk.Entry(root, textvariable=self.step_spec_var, width=24).grid(row=2, column=1, columnspan=3, sticky="w")

        tk.Label(root, text="Saccade amp").grid(row=3, column=0, sticky="w", padx=8)
        tk.Entry(root, textvariable=self.saccade_amp_var, width=8).grid(row=3, column=1, sticky="w")

        tk.Label(root, text="Noise ppm").grid(row=3, column=2, sticky="w", padx=8)
        tk.Entry(root, textvariable=self.noise_ppm_var, width=8).grid(row=3, column=3, sticky="w")

        tk.Label(root, text="Seed").grid(row=4, column=0, sticky="w", padx=8)
        tk.Entry(root, textvariable=self.seed_var, width=8).grid(row=4, column=1, sticky="w")

        tk.Label(root, text="Max evt/step").grid(row=4, column=2, sticky="w", padx=8)
        tk.Entry(root, textvariable=self.max_events_per_step_var, width=8).grid(row=4, column=3, sticky="w")

        tk.Label(root, text="Output dir").grid(row=5, column=0, sticky="w", padx=8)
        tk.Entry(root, textvariable=self.out_dir_var, width=40).grid(row=5, column=1, columnspan=3, sticky="w")

        tk.Button(root, text="Clear", command=self.clear).grid(row=6, column=0, padx=8, pady=8)
        tk.Button(root, text="Save", command=self.save).grid(row=6, column=1, padx=8, pady=8)
        tk.Button(root, text="Quit", command=root.destroy).grid(row=6, column=2, padx=8, pady=8)

        self.count_var = tk.StringVar(value=self._count_label(self._saved_count()))
        tk.Label(root, textvariable=self.count_var).grid(row=7, column=0, columnspan=4, sticky="w", padx=8, pady=(0, 8))

        self.redraw()

    def _saved_count(self) -> int:
        out_dir = Path(self.out_dir_var.get().strip()).expanduser()
        header = out_dir / "input.h"
        if not header.exists():
            return 0
        text = header.read_text(errors="replace")
        if MARKER not in text:
            return 0
        indexes = [int(value) for value in re.findall(r'#include\s+"input_(\d+)\.h"', text)]
        if not indexes:
            return 0
        return max(indexes) + 1

    def _count_label(self, count: int) -> str:
        return f"Recordings saved: {count} of {RECORDING_COUNT}"

    def clear(self) -> None:
        for y in range(GRID):
            for x in range(GRID):
                self.pixels[y][x] = 0
        self.redraw()

    def on_draw(self, event: Any) -> None:
        x = int(event.x // CELL)
        y = int(event.y // CELL)
        for yy in range(y - 1, y + 2):
            for xx in range(x - 1, x + 2):
                if 0 <= xx < GRID and 0 <= yy < GRID:
                    self.pixels[yy][xx] = 1
        self.redraw()

    def redraw(self) -> None:
        self.canvas.delete("all")
        for y in range(GRID):
            for x in range(GRID):
                x0 = x * CELL
                y0 = y * CELL
                x1 = x0 + CELL
                y1 = y0 + CELL
                fill = "black" if self.pixels[y][x] else "white"
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline="#d0d0d0")

    def _base_frame(self) -> list[list[int]]:
        return [row[:] for row in self.pixels]

    def _shifted_frame(self, frame: list[list[int]], dx: int, dy: int) -> list[list[int]]:
        shifted = [[0 for _ in range(TARGET_W)] for _ in range(TARGET_H)]
        for y in range(TARGET_H):
            yy = y - dy
            if yy < 0 or yy >= TARGET_H:
                continue
            for x in range(TARGET_W):
                xx = x - dx
                if xx < 0 or xx >= TARGET_W:
                    continue
                shifted[y][x] = frame[yy][xx]
        return shifted

    def _vector_lines(
        self,
        steps: list[int],
        n_steps: int,
        saccade_amp: int,
        noise_ppm: int,
        max_events_per_step: int,
        seed: int,
    ) -> list[str]:
        step_set = set(steps)
        base = self._base_frame()
        lines: list[str] = []
        rng = random.Random(seed)
        # The digit is already on the sensor when the recording starts.
        # Later frames only contain the brightness changes caused by the saccade,
        # spread across the milliseconds of each one-pixel move.
        fx, fy = _saccade_position(0, n_steps, saccade_amp)
        prev = self._shifted_frame(base, _round_half_up(fx), _round_half_up(fy))
        assigned: list[set[int]] = [set() for _ in range(n_steps)]
        last = 0
        for t in range(1, n_steps):
            fx, fy = _saccade_position(t, n_steps, saccade_amp)
            cur = self._shifted_frame(base, _round_half_up(fx), _round_half_up(fy))
            events = _occupancy_events(prev, cur)
            if events:
                span = max(t - last, 1)
                for k, idx in enumerate(events):
                    slot = last + (k * span) // len(events)
                    if slot >= t:
                        slot = t - 1
                    if slot in step_set:
                        assigned[slot].add(idx)
                prev = cur
                last = t

        for t in range(n_steps):
            if t not in step_set:
                continue
            step_events = assigned[t]
            if noise_ppm > 0:
                for i in range(INPUT_SIZE):
                    if rng.randrange(1_000_000) < noise_ppm:
                        step_events.add(i)
            indices = sorted(step_events)
            if max_events_per_step > 0 and len(indices) > max_events_per_step:
                indices = sorted(rng.sample(indices, max_events_per_step))
            for idx in indices:
                lines.append(f"[{t}][{idx}] = 1,")

        return lines

    def _settings(self) -> tuple[int, list[int], int, int, int, int] | None:
        try:
            n_steps = int(self.step_count_var.get().strip())
            if n_steps <= 0:
                raise ValueError("N_STEPS must be > 0")
            steps = parse_steps(self.step_spec_var.get(), n_steps=n_steps)
            saccade_amp = int(self.saccade_amp_var.get().strip())
            if saccade_amp < 0:
                raise ValueError("Saccade amp must be >= 0")
            noise_ppm = int(self.noise_ppm_var.get().strip())
            if noise_ppm < 0:
                raise ValueError("Noise ppm must be >= 0")
            max_events_per_step = int(self.max_events_per_step_var.get().strip())
            if max_events_per_step < 0:
                raise ValueError("Max evt/step must be >= 0")
            seed = int(self.seed_var.get().strip())
        except Exception as exc:
            messagebox.showerror("Invalid input", str(exc))
            return None
        return n_steps, steps, saccade_amp, noise_ppm, max_events_per_step, seed

    def _write_vector(self, out_dir: Path, index: int, settings: tuple[int, list[int], int, int, int, int]) -> None:
        n_steps, steps, saccade_amp, noise_ppm, max_events_per_step, seed = settings
        vec_lines = self._vector_lines(
            steps,
            n_steps=n_steps,
            saccade_amp=saccade_amp,
            noise_ppm=noise_ppm,
            max_events_per_step=max_events_per_step,
            seed=seed,
        )
        (out_dir / f"input_{index}.h").write_text("\n".join(vec_lines) + "\n")

    def _write_master(self, out_dir: Path, n_steps: int, count: int) -> None:
        includes = []
        for index in range(count):
            includes.append("    {")
            includes.append(f'#include "input_{index}.h"')
            includes.append("    },")
        master = [
            "// Auto-generated from draw_input_app.py for snn-mlir runtime",
            f"// {MARKER}",
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
        (out_dir / "input.h").write_text("\n".join(master))
        for path in out_dir.glob("input_*.h"):
            stem = path.stem.removeprefix("input_")
            if stem.isdigit() and int(stem) >= count:
                path.unlink()

    def save(self) -> None:
        settings = self._settings()
        if settings is None:
            return
        out_dir = Path(self.out_dir_var.get().strip()).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        saved = self._saved_count()
        index = 0 if saved >= RECORDING_COUNT else saved
        self._write_vector(out_dir, index, settings)
        count = index + 1
        self._write_master(out_dir, settings[0], count)
        self.count_var.set(self._count_label(count))
        self.clear()
        if count == RECORDING_COUNT:
            detail = "All 5 recordings are saved. Run: make python-test NO_TRAIN=1"
        elif index == 0 and saved >= RECORDING_COUNT:
            detail = "Started a new set. Draw the next digit and click Save."
        else:
            detail = "Draw the next digit and click Save."
        messagebox.showinfo(
            "Saved",
            f"Saved recording {index + 1} of {RECORDING_COUNT}.\n"
            f"{out_dir / f'input_{index}.h'}\n\n"
            f"{detail}",
        )


def main() -> None:
    if tk is None:
        raise SystemExit(
            "tkinter is not available in this Python environment.\n"
            "Install it on Linux with:\n"
            "  sudo apt-get update && sudo apt-get install -y python3-tk\n"
            "Then re-run:\n"
            "  ./draw_input_app.py\n"
            "or:\n"
            "  make draw-input"
        )
    root = tk.Tk()
    DrawApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
