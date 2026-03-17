#!/usr/bin/env python3
"""Validate one noisy dataset run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "python" / "src"))

    from fluid_denoise.phase3_noisy_dataset import validate_noisy_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    summary = validate_noisy_dataset(args.run_dir)
    print(
        f"[NOISY] valid dataset: {summary.experiment_id} "
        f"({summary.num_snapshots} snapshots, nx={summary.nx}, ny={summary.ny})"
    )


if __name__ == "__main__":
    main()
