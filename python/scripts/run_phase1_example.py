#!/usr/bin/env python3
"""Configure, build and run the FASE 1 example."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "python" / "src"))

    from fluid_denoise.phase1_runtime import (  # pylint: disable=import-error
        configure_and_build,
        find_solver_binary,
        resolve_path,
        run_command,
    )
    from fluid_denoise.phase1_validation import (  # pylint: disable=import-error
        validate_run_outputs,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lbm_cylinder_base.json")
    parser.add_argument("--build-dir", default="cpp/lbm_core/build")
    parser.add_argument("--build-type", default="Release")
    args = parser.parse_args()

    source_dir = repo_root / "cpp" / "lbm_core"
    build_dir = resolve_path(repo_root, args.build_dir)
    config_path = resolve_path(repo_root, args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    configure_and_build(source_dir, build_dir, build_type=args.build_type)
    solver_binary = find_solver_binary(build_dir, build_type=args.build_type)
    run_command([str(solver_binary), "--config", str(config_path)], cwd=build_dir)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_dir = resolve_path(repo_root, config["output_root"]) / config["run_id"]
    summary = validate_run_outputs(run_dir)
    print(
        f"[VALIDATION] checked {len(summary.steps)} snapshot steps in {summary.run_dir}"
    )


if __name__ == "__main__":
    main()
