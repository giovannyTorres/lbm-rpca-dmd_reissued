#!/usr/bin/env python3
"""Generate a noisy dataset example from a clean run or from the current solver output."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _default_noise_specs() -> list[dict[str, Any]]:
    return [
        {
            "kind": "gaussian",
            "intensity": 0.15,
            "seed": 11,
            "channels": ["ux", "uy"],
            "params": {"scale_mode": "channel_std"},
        },
        {
            "kind": "piv_outliers",
            "intensity": 0.04,
            "seed": 19,
            "channels": ["ux", "uy"],
            "params": {"outlier_scale": 4.0, "mode": "replace"},
        },
    ]


def _load_noise_specs(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return _default_noise_specs()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Noise config must be a JSON array of noise stages")
    return [dict(item) for item in payload]


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
    from fluid_denoise.phase3_noisy_dataset import (
        create_noisy_dataset,
        save_noisy_comparison_figure,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lbm_cylinder_clean_example.json")
    parser.add_argument("--build-dir", default="cpp/lbm_core/build_phase3_example")
    parser.add_argument("--raw-run-dir", default=None)
    parser.add_argument("--clean-run-dir", default=None)
    parser.add_argument("--clean-root", default="data/clean")
    parser.add_argument("--noisy-root", default="data/noisy")
    parser.add_argument("--noise-config", default=None)
    parser.add_argument("--figure-variable", default="ux")
    parser.add_argument("--figure-step", type=int, default=None)
    parser.add_argument("--skip-figure", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config_path = resolve_path(repo_root, args.config).resolve()
    clean_root = resolve_path(repo_root, args.clean_root)
    noisy_root = resolve_path(repo_root, args.noisy_root)
    noise_config_path = (
        resolve_path(repo_root, args.noise_config).resolve()
        if args.noise_config is not None
        else None
    )
    noise_specs = _load_noise_specs(noise_config_path)

    if args.clean_run_dir is not None:
        clean_run_dir = resolve_path(repo_root, args.clean_run_dir).resolve()
    else:
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

    noisy_run_dir = create_noisy_dataset(
        clean_run_dir,
        noise_specs=noise_specs,
        noisy_root=noisy_root,
        overwrite=args.overwrite,
    )
    print(f"[NOISY] wrote noisy dataset to {noisy_run_dir}")

    if not args.skip_figure:
        figure_path = save_noisy_comparison_figure(
            noisy_run_dir,
            step=args.figure_step,
            variable=args.figure_variable,
        )
        print(f"[NOISY] wrote comparison figure to {figure_path}")


if __name__ == "__main__":
    main()
