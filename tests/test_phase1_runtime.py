from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python" / "src"))


class Phase1RuntimeHelpersTest(unittest.TestCase):
    def test_ensure_cmake_available_raises_when_missing(self) -> None:
        from fluid_denoise.phase1_runtime import ensure_cmake_available

        with mock.patch("shutil.which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "cmake was not found"):
                ensure_cmake_available()

    def test_find_solver_binary_prefers_existing_candidate(self) -> None:
        from fluid_denoise.phase1_runtime import find_solver_binary

        with tempfile.TemporaryDirectory(prefix="lbm_runtime_") as tmp:
            build_dir = Path(tmp)
            binary_dir = build_dir / "Release"
            binary_dir.mkdir(parents=True, exist_ok=True)
            binary_path = binary_dir / "lbm_sim.exe"
            binary_path.write_text("", encoding="utf-8")

            with mock.patch("os.name", "nt"):
                resolved = find_solver_binary(build_dir, build_type="Release")

            self.assertEqual(resolved, binary_path)
