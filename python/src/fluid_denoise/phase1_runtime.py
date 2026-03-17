from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def ensure_cmake_available() -> str:
    cmake_path = shutil.which("cmake")
    if cmake_path is None:
        raise RuntimeError(
            "cmake was not found in PATH. Install CMake and a C++17 compiler before "
            "running FASE 1."
        )
    return cmake_path


def resolve_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return repo_root / path


def run_command(command: list[str], cwd: Path) -> None:
    print(f"[RUN] {' '.join(command)} (cwd={cwd})")
    subprocess.run(command, cwd=cwd, check=True)


def configure_and_build(source_dir: Path, build_dir: Path, build_type: str = "Release") -> None:
    ensure_cmake_available()
    build_dir.mkdir(parents=True, exist_ok=True)
    run_command(["cmake", "-S", str(source_dir), "-B", str(build_dir)], cwd=source_dir)
    run_command(
        ["cmake", "--build", str(build_dir), "--config", build_type, "--parallel"],
        cwd=source_dir,
    )


def find_solver_binary(build_dir: Path, build_type: str = "Release") -> Path:
    executable_name = "lbm_sim.exe" if os.name == "nt" else "lbm_sim"
    candidates = [
        build_dir / executable_name,
        build_dir / build_type / executable_name,
        build_dir / "Debug" / executable_name,
        build_dir / "Release" / executable_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find built solver binary in {build_dir}. Checked: {candidates}"
    )
