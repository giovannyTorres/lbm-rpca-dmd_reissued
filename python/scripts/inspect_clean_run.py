#!/usr/bin/env python3
"""Inspect and summarize one clean dataset run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "python" / "src"))

    from fluid_denoise.phase2_clean_dataset import summarize_clean_run

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--as-json", action="store_true")
    args = parser.parse_args()

    summary = summarize_clean_run(args.run_dir)
    if args.as_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    print(f"experiment_id: {summary['experiment_id']}")
    print(
        f"dimensions: nx={summary['dimensions']['nx']} ny={summary['dimensions']['ny']}"
    )
    print(f"num_snapshots: {summary['num_snapshots']}")
    print(f"variables: {', '.join(summary['variables'])}")
    params = summary["parameters"]
    print(
        "parameters: "
        f"reynolds={params['reynolds']} "
        f"u_in={params['u_in']} "
        f"obstacle_cx={params['obstacle_cx']} "
        f"obstacle_cy={params['obstacle_cy']} "
        f"obstacle_r={params['obstacle_r']} "
        f"tau={params['tau']}"
    )


if __name__ == "__main__":
    main()
