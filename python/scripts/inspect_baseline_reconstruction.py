#!/usr/bin/env python3
"""Inspect one saved baseline reconstruction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "python" / "src"))

    from fluid_denoise.phase5_baselines import summarize_baseline_reconstruction

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--as-json", action="store_true")
    args = parser.parse_args()

    summary = summarize_baseline_reconstruction(args.run_dir)
    if args.as_json:
        print(
            json.dumps(
                {
                    "reconstruction_id": summary.reconstruction_id,
                    "model_name": summary.model_name,
                    "source_experiment_id": summary.source_experiment_id,
                    "variables": list(summary.variables),
                    "num_snapshots": summary.num_snapshots,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    print(f"reconstruction_id: {summary.reconstruction_id}")
    print(f"model_name: {summary.model_name}")
    print(f"source_experiment_id: {summary.source_experiment_id}")
    print(f"variables: {', '.join(summary.variables)}")
    print(f"num_snapshots: {summary.num_snapshots}")


if __name__ == "__main__":
    main()
