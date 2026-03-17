from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def ensure_cmake_available() -> str:
    """Return CMake executable path or raise a user-facing runtime error."""
    cmake_path = shutil.which("cmake")
    if cmake_path is None:
        raise RuntimeError(
            "cmake was not found in PATH. Install CMake and a C++17 compiler before "
            "running FASE 1."
        )
    return cmake_path


def resolve_path(repo_root: Path, raw_path: str | Path) -> Path:
    """Resolve a path against the repository root when it is not absolute."""
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def run_command(command: list[str], cwd: Path) -> None:
    """Execute a command and stream output, raising on non-zero exit code."""
    LOGGER.info("[RUN] %s (cwd=%s)", " ".join(command), cwd)
    subprocess.run(command, cwd=cwd, check=True)


def configure_and_build(source_dir: Path, build_dir: Path, build_type: str = "Release") -> None:
    """Configure and build the C++ solver with CMake."""
    ensure_cmake_available()
    build_dir.mkdir(parents=True, exist_ok=True)
    run_command(["cmake", "-S", str(source_dir), "-B", str(build_dir)], cwd=source_dir)
    run_command(
        ["cmake", "--build", str(build_dir), "--config", build_type, "--parallel"],
        cwd=source_dir,
    )


def find_solver_binary(build_dir: Path, build_type: str = "Release") -> Path:
    """Locate the generated solver binary across single/multi-config generators."""
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
