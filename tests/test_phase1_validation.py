from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python" / "src"))


def write_csv(path: Path, rows: list[list[float]]) -> None:
    path.write_text(
        "\n".join(",".join(str(value) for value in row) for row in rows) + "\n",
        encoding="utf-8",
    )


class Phase1ValidationTest(unittest.TestCase):
    def _write_minimal_run(self, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "nx": 2,
            "ny": 2,
            "iterations": 10,
            "save_stride": 10,
        }
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

        ux = [[0.1, 0.0], [0.0, 0.1]]
        uy = [[0.0, 0.0], [0.0, 0.0]]
        speed = [[0.1, 0.0], [0.0, 0.1]]
        vorticity = [[0.0, 0.0], [0.0, 0.0]]
        mask = [[0, 1], [0, 0]]

        for step in (0, 10):
            suffix = f"t{step:06d}"
            write_csv(run_dir / f"ux_{suffix}.csv", ux)
            write_csv(run_dir / f"uy_{suffix}.csv", uy)
            write_csv(run_dir / f"speed_{suffix}.csv", speed)
            write_csv(run_dir / f"vorticity_{suffix}.csv", vorticity)
            write_csv(run_dir / f"mask_{suffix}.csv", mask)

    def test_validate_run_outputs_accepts_valid_minimal_run(self) -> None:
        from fluid_denoise.phase1_validation import validate_run_outputs

        with tempfile.TemporaryDirectory(prefix="lbm_validation_") as tmp:
            run_dir = Path(tmp) / "run"
            self._write_minimal_run(run_dir)

            summary = validate_run_outputs(run_dir)

            self.assertEqual(summary.nx, 2)
            self.assertEqual(summary.ny, 2)
            self.assertEqual(summary.steps, (0, 10))

    def test_validate_run_outputs_rejects_speed_inconsistency(self) -> None:
        from fluid_denoise.phase1_validation import validate_run_outputs

        with tempfile.TemporaryDirectory(prefix="lbm_validation_") as tmp:
            run_dir = Path(tmp) / "run"
            self._write_minimal_run(run_dir)
            write_csv(run_dir / "speed_t000010.csv", [[0.9, 0.0], [0.0, 0.1]])

            with self.assertRaisesRegex(ValueError, "Speed mismatch"):
                validate_run_outputs(run_dir)

    def test_validate_run_outputs_rejects_missing_snapshot_family(self) -> None:
        from fluid_denoise.phase1_validation import validate_run_outputs

        with tempfile.TemporaryDirectory(prefix="lbm_validation_") as tmp:
            run_dir = Path(tmp) / "run"
            self._write_minimal_run(run_dir)
            (run_dir / "mask_t000010.csv").unlink()

            with self.assertRaisesRegex(ValueError, "Mismatched snapshot steps"):
                validate_run_outputs(run_dir)
