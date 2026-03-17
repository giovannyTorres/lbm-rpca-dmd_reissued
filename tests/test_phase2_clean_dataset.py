from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python" / "src"))


def write_csv(path: Path, rows: list[list[float]]) -> None:
    path.write_text(
        "\n".join(",".join(str(value) for value in row) for row in rows) + "\n",
        encoding="utf-8",
    )


class Phase2CleanDatasetTest(unittest.TestCase):
    def _write_raw_run(self, root: Path, *, varying_mask: bool = False) -> tuple[Path, Path]:
        raw_run_dir = root / "raw" / "sample_run"
        raw_run_dir.mkdir(parents=True, exist_ok=True)

        config_path = root / "sample_config.json"
        config_payload = {
            "nx": 3,
            "ny": 2,
            "reynolds": 100.0,
            "u_in": 0.04,
            "iterations": 10,
            "save_stride": 10,
            "obstacle_cx": 1,
            "obstacle_cy": 1,
            "obstacle_r": 1,
            "output_root": str(root / "raw"),
            "run_id": "sample_run",
            "seed": 7,
        }
        config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

        manifest_payload = {
            "config_path": str(config_path),
            "nx": 3,
            "ny": 2,
            "reynolds": 100.0,
            "u_in": 0.04,
            "iterations": 10,
            "save_stride": 10,
            "obstacle_cx": 1,
            "obstacle_cy": 1,
            "obstacle_r": 1,
            "output_root": str(root / "raw"),
            "run_id": "sample_run",
            "nu": 0.0008,
            "tau": 0.5024,
        }
        (raw_run_dir / "manifest.json").write_text(
            json.dumps(manifest_payload, indent=2),
            encoding="utf-8",
        )

        ux0 = [[0.04, 0.03, 0.02], [0.04, 0.00, 0.01]]
        uy0 = [[0.00, 0.00, 0.00], [0.00, 0.00, 0.00]]
        speed0 = [[0.04, 0.03, 0.02], [0.04, 0.00, 0.01]]
        vort0 = [[0.00, 0.01, 0.02], [0.00, -0.01, -0.02]]
        mask0 = [[0, 0, 0], [0, 1, 0]]

        ux1 = [[0.04, 0.025, 0.015], [0.04, 0.00, 0.005]]
        uy1 = [[0.00, 0.001, 0.000], [0.00, 0.000, -0.001]]
        speed1 = (
            np.sqrt(np.asarray(ux1, dtype=np.float64) ** 2 + np.asarray(uy1, dtype=np.float64) ** 2)
            .round(12)
            .tolist()
        )
        vort1 = [[0.00, 0.02, 0.03], [0.00, -0.02, -0.03]]
        mask1 = [[0, 0, 0], [0, 0 if varying_mask else 1, 0]]

        snapshot_payloads = {
            0: (ux0, uy0, speed0, vort0, mask0),
            10: (ux1, uy1, speed1, vort1, mask1),
        }
        for step, (ux, uy, speed, vorticity, mask) in snapshot_payloads.items():
            suffix = f"t{step:06d}"
            write_csv(raw_run_dir / f"ux_{suffix}.csv", ux)
            write_csv(raw_run_dir / f"uy_{suffix}.csv", uy)
            write_csv(raw_run_dir / f"speed_{suffix}.csv", speed)
            write_csv(raw_run_dir / f"vorticity_{suffix}.csv", vorticity)
            write_csv(raw_run_dir / f"mask_{suffix}.csv", mask)

        return raw_run_dir, config_path

    def test_convert_load_and_validate_clean_dataset(self) -> None:
        from fluid_denoise.phase2_clean_dataset import (
            convert_raw_run_to_clean_dataset,
            load_clean_run,
            summarize_clean_run,
            validate_clean_dataset,
        )

        with tempfile.TemporaryDirectory(prefix="phase2_clean_") as tmp:
            root = Path(tmp)
            raw_run_dir, _ = self._write_raw_run(root)

            clean_run_dir = convert_raw_run_to_clean_dataset(
                raw_run_dir,
                clean_root=root / "clean",
            )

            summary = validate_clean_dataset(clean_run_dir)
            self.assertEqual(summary.nx, 3)
            self.assertEqual(summary.ny, 2)
            self.assertEqual(summary.num_snapshots, 2)
            self.assertIn("mask", summary.variables)

            clean_run = load_clean_run(clean_run_dir)
            self.assertEqual(clean_run.ux.shape, (2, 2, 3))
            self.assertEqual(clean_run.mask.shape, (2, 3))
            self.assertEqual(clean_run.metadata["source"]["seed"], 7)

            inspect_summary = summarize_clean_run(clean_run_dir)
            self.assertEqual(inspect_summary["dimensions"]["nx"], 3)
            self.assertEqual(inspect_summary["num_snapshots"], 2)

    def test_build_experiment_id_contains_traceability_tokens(self) -> None:
        from fluid_denoise.phase2_clean_dataset import build_experiment_id

        manifest = {
            "reynolds": 100.0,
            "u_in": 0.04,
            "nx": 96,
            "ny": 48,
            "obstacle_r": 6,
            "run_id": "Phase2 Example Run",
        }
        config = {"seed": 0}

        experiment_id = build_experiment_id(manifest, config)

        self.assertIn("re0100", experiment_id)
        self.assertIn("seed-0", experiment_id)
        self.assertIn("phase2-example-run", experiment_id)

    def test_convert_rejects_non_static_mask(self) -> None:
        from fluid_denoise.phase2_clean_dataset import convert_raw_run_to_clean_dataset

        with tempfile.TemporaryDirectory(prefix="phase2_clean_") as tmp:
            root = Path(tmp)
            raw_run_dir, _ = self._write_raw_run(root, varying_mask=True)

            with self.assertRaisesRegex(ValueError, "Mask changed across snapshots"):
                convert_raw_run_to_clean_dataset(raw_run_dir, clean_root=root / "clean")
