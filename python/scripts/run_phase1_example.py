#!/usr/bin/env python3
"""Compila y ejecuta el caso base de FASE 1."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    print(f"[RUN] {' '.join(cmd)} (cwd={cwd})")
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lbm_cylinder_base.cfg")
    parser.add_argument("--build-dir", default="cpp/lbm_core/build")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    build_dir = repo_root / args.build_dir
    build_dir.mkdir(parents=True, exist_ok=True)

    run(["cmake", ".."], cwd=build_dir)
    run(["cmake", "--build", ".", "-j"], cwd=build_dir)
    run(["./lbm_sim", "--config", str(repo_root / args.config)], cwd=build_dir)


if __name__ == "__main__":
    main()
