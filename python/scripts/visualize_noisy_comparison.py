#!/usr/bin/env python3
"""Create a clean-vs-noisy comparison figure for one noisy dataset run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "python" / "src"))

    from fluid_denoise.phase3_noisy_dataset import save_noisy_comparison_figure

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--variable", default="ux")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output_path = save_noisy_comparison_figure(
        args.run_dir,
        step=args.step,
        variable=args.variable,
        output_path=args.output,
    )
    print(f"[VIS] wrote {output_path}")


if __name__ == "__main__":
    main()
