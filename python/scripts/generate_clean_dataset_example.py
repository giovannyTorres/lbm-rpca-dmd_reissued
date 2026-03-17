#!/usr/bin/env python3
"""Generate a small clean dataset example from the current solver output format."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "python" / "src"))

    from fluid_denoise.phase1_runtime import (
        configure_and_build,
        find_solver_binary,
        resolve_path,
        run_command,
    )
    from fluid_denoise.phase1_validation import validate_run_outputs
    from fluid_denoise.phase2_clean_dataset import convert_raw_run_to_clean_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lbm_cylinder_clean_example.json")
    parser.add_argument("--build-dir", default="cpp/lbm_core/build_phase2_example")
    parser.add_argument("--raw-run-dir", default=None)
    parser.add_argument("--clean-root", default="data/clean")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config_path = resolve_path(repo_root, args.config).resolve()
    clean_root = resolve_path(repo_root, args.clean_root)

    if args.raw_run_dir is not None:
        raw_run_dir = resolve_path(repo_root, args.raw_run_dir).resolve()
    else:
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        source_dir = repo_root / "cpp" / "lbm_core"
        build_dir = resolve_path(repo_root, args.build_dir)
        configure_and_build(source_dir, build_dir, build_type="Release")
        solver_binary = find_solver_binary(build_dir, build_type="Release")
        run_command([str(solver_binary), "--config", str(config_path)], cwd=build_dir)
        raw_run_dir = resolve_path(repo_root, config["output_root"]) / config["run_id"]

    validate_run_outputs(raw_run_dir)
    clean_run_dir = convert_raw_run_to_clean_dataset(
        raw_run_dir,
        clean_root=clean_root,
        overwrite=args.overwrite,
    )
    print(f"[CLEAN] wrote clean dataset to {clean_run_dir}")


if __name__ == "__main__":
    main()
