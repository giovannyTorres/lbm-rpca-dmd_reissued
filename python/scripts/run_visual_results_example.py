#!/usr/bin/env python3
"""Generate the FASE 7 visual results pipeline from a JSON config file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "python" / "src"))

    from fluid_denoise.phase7_visual_results import generate_visual_results_from_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/visual_results_phase7_example.json")
    args = parser.parse_args()

    config_path = (repo_root / args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    summary = generate_visual_results_from_config(config_path)
    print(f"[VISUALS] benchmark_id={summary.benchmark_id}")
    print(f"[VISUALS] selected_experiments={summary.num_selected_experiments}")
    print(f"[VISUALS] figures={summary.num_figures} tables={summary.num_tables}")
    print(f"[VISUALS] output_root={summary.output_root}")
    print(f"[VISUALS] catalog_json={summary.catalog_json_path}")
    print(f"[VISUALS] catalog_csv={summary.catalog_csv_path}")
    print(f"[VISUALS] catalog_md={summary.catalog_markdown_path}")


if __name__ == "__main__":
    main()
