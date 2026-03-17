#!/usr/bin/env python3
"""Run one baseline reconstruction from a JSON config file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "python" / "src"))

    from fluid_denoise.phase5_baselines import run_baseline_from_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline_phase5_example.json")
    args = parser.parse_args()

    config_path = (repo_root / args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    output_dir = run_baseline_from_config(config_path)
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    print(f"[BASELINE] wrote reconstruction to {output_dir}")
    if metadata["warnings"]:
        print("[BASELINE] warnings:")
        for warning in metadata["warnings"]:
            print(f"  - {warning}")


if __name__ == "__main__":
    main()
