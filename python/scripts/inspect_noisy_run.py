#!/usr/bin/env python3
"""Inspect and summarize one noisy dataset run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "python" / "src"))

    from fluid_denoise.phase3_noisy_dataset import summarize_noisy_run

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--as-json", action="store_true")
    args = parser.parse_args()

    summary = summarize_noisy_run(args.run_dir)
    if args.as_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    print(f"experiment_id: {summary['experiment_id']}")
    print(
        f"dimensions: nx={summary['dimensions']['nx']} ny={summary['dimensions']['ny']}"
    )
    print(f"num_snapshots: {summary['num_snapshots']}")
    print(f"source_experiment_id: {summary['source_experiment_id']}")
    print(f"noise_kinds: {', '.join(summary['noise_kinds'])}")
    print(
        "final_corrupted_cells: "
        f"ux={summary['final_corrupted_cells']['ux']} "
        f"uy={summary['final_corrupted_cells']['uy']}"
    )


if __name__ == "__main__":
    main()
