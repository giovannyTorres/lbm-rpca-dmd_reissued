from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python" / "src"))


def read_csv_matrix(path: Path) -> list[list[float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    return [[float(value) for value in row] for row in rows]


class Phase1SmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if shutil.which("cmake") is None:
            raise unittest.SkipTest("cmake is not available in PATH")

    def test_phase1_generates_snapshots(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        tmpdir = Path(tempfile.mkdtemp(prefix="lbm_phase1_"))

        try:
            run_id = "smoke_run"
            nx = 80
            ny = 40
            output_root = tmpdir / "raw"
            config_path = tmpdir / "smoke.json"
            build_dir = tmpdir / "build"

            config_path.write_text(
                json.dumps(
                    {
                        "nx": nx,
                        "ny": ny,
                        "reynolds": 120.0,
                        "u_in": 0.05,
                        "iterations": 30,
                        "save_stride": 10,
                        "obstacle_cx": 20,
                        "obstacle_cy": 20,
                        "obstacle_r": 5,
                        "output_root": str(output_root),
                        "run_id": run_id,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            subprocess.run(
                [
                    sys.executable,
                    str(repo / "python" / "scripts" / "run_phase1_example.py"),
                    "--config",
                    str(config_path),
                    "--build-dir",
                    str(build_dir),
                ],
                cwd=repo,
                check=True,
            )

            run_dir = output_root / run_id
            self.assertTrue(run_dir.exists())
            self.assertTrue((run_dir / "manifest.json").exists())

            from fluid_denoise.phase1_validation import validate_run_outputs

            summary = validate_run_outputs(run_dir)
            self.assertEqual(summary.nx, nx)
            self.assertEqual(summary.ny, ny)
            self.assertEqual(summary.steps[0], 0)
            self.assertEqual(summary.steps[-1], 30)

            expected_files = [
                "ux_t000000.csv",
                "uy_t000000.csv",
                "speed_t000000.csv",
                "vorticity_t000000.csv",
                "mask_t000000.csv",
                "ux_t000010.csv",
                "speed_t000030.csv",
                "vorticity_t000030.csv",
                "mask_t000030.csv",
            ]
            for filename in expected_files:
                self.assertTrue((run_dir / filename).exists(), filename)

            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["nx"], nx)
            self.assertEqual(manifest["ny"], ny)
            self.assertGreater(manifest["tau"], 0.5)

            ux = read_csv_matrix(run_dir / "ux_t000030.csv")
            uy = read_csv_matrix(run_dir / "uy_t000030.csv")
            speed = read_csv_matrix(run_dir / "speed_t000030.csv")
            vorticity = read_csv_matrix(run_dir / "vorticity_t000030.csv")
            mask = read_csv_matrix(run_dir / "mask_t000030.csv")

            for field in [ux, uy, speed, vorticity, mask]:
                self.assertEqual(len(field), ny)
                self.assertTrue(all(len(row) == nx for row in field))

            for field in [ux, uy, speed, vorticity]:
                self.assertTrue(
                    all(math.isfinite(value) for row in field for value in row)
                )

            mask_values = {int(value) for row in mask for value in row}
            self.assertLessEqual(mask_values, {0, 1})

            sample_points = [(0, 0), (ny // 2, nx // 2), (ny - 1, nx - 1)]
            for row_index, col_index in sample_points:
                expected_speed = math.sqrt(
                    ux[row_index][col_index] ** 2 + uy[row_index][col_index] ** 2
                )
                self.assertAlmostEqual(
                    speed[row_index][col_index], expected_speed, places=6
                )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
