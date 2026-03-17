#!/usr/bin/env python3
"""Create a quick-look figure for one snapshot of a FASE 1 run."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_suffix(step: int) -> str:
    return f"t{step:06d}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    run_dir = Path(args.run_dir).resolve()
    suffix = build_suffix(args.step)
    output_path = (
        Path(args.output).resolve()
        if args.output is not None
        else run_dir / f"snapshot_{suffix}.png"
    )

    file_map = {
        "ux": run_dir / f"ux_{suffix}.csv",
        "uy": run_dir / f"uy_{suffix}.csv",
        "speed": run_dir / f"speed_{suffix}.csv",
        "vorticity": run_dir / f"vorticity_{suffix}.csv",
        "mask": run_dir / f"mask_{suffix}.csv",
    }

    missing = [path for path in file_map.values() if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing snapshot files: {missing_text}")

    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    panels = [
        ("ux", "ux", "coolwarm"),
        ("uy", "uy", "coolwarm"),
        ("speed", "speed", "viridis"),
        ("vorticity", "vorticity", "RdBu_r"),
        ("mask", "mask", "gray"),
    ]

    for axis, (title, key, cmap) in zip(axes.flat, panels):
        data = np.loadtxt(file_map[key], delimiter=",")
        image = axis.imshow(data, origin="lower", cmap=cmap, aspect="auto")
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        figure.colorbar(image, ax=axis, shrink=0.8)

    axes.flat[-1].axis("off")
    figure.suptitle(f"FASE 1 snapshot {suffix}")
    figure.savefig(output_path, dpi=150)
    print(f"[VIS] wrote {output_path}")


if __name__ == "__main__":
    main()
