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


class Phase3NoisyDatasetTest(unittest.TestCase):
    def _write_raw_run(self, root: Path) -> Path:
        raw_run_dir = root / "raw" / "sample_run"
        raw_run_dir.mkdir(parents=True, exist_ok=True)

        config_path = root / "sample_config.json"
        config_payload = {
            "nx": 5,
            "ny": 4,
            "reynolds": 100.0,
            "u_in": 0.04,
            "iterations": 10,
            "save_stride": 10,
            "obstacle_cx": 2,
            "obstacle_cy": 2,
            "obstacle_r": 1,
            "output_root": str(root / "raw"),
            "run_id": "sample_run",
            "seed": 7,
        }
        config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

        manifest_payload = {
            "config_path": str(config_path),
            "nx": 5,
            "ny": 4,
            "reynolds": 100.0,
            "u_in": 0.04,
            "iterations": 10,
            "save_stride": 10,
            "obstacle_cx": 2,
            "obstacle_cy": 2,
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

        ux0 = [
            [0.04, 0.03, 0.02, 0.02, 0.01],
            [0.04, 0.03, 0.00, 0.02, 0.01],
            [0.04, 0.03, 0.00, 0.02, 0.01],
            [0.04, 0.03, 0.02, 0.02, 0.01],
        ]
        uy0 = [
            [0.00, 0.00, 0.00, 0.00, 0.00],
            [0.00, 0.01, 0.00, -0.01, 0.00],
            [0.00, 0.01, 0.00, -0.01, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00],
        ]
        speed0 = (
            np.sqrt(np.asarray(ux0, dtype=np.float64) ** 2 + np.asarray(uy0, dtype=np.float64) ** 2)
            .round(12)
            .tolist()
        )
        vort0 = [
            [0.00, 0.01, 0.02, 0.01, 0.00],
            [0.00, 0.01, 0.00, -0.01, 0.00],
            [0.00, 0.01, 0.00, -0.01, 0.00],
            [0.00, 0.01, 0.02, 0.01, 0.00],
        ]
        mask = [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ]

        ux1 = [
            [0.04, 0.028, 0.018, 0.018, 0.009],
            [0.04, 0.026, 0.00, 0.016, 0.008],
            [0.04, 0.024, 0.00, 0.014, 0.007],
            [0.04, 0.022, 0.016, 0.012, 0.006],
        ]
        uy1 = [
            [0.00, 0.001, 0.000, -0.001, 0.000],
            [0.00, 0.010, 0.000, -0.010, 0.000],
            [0.00, 0.009, 0.000, -0.009, 0.000],
            [0.00, 0.001, 0.000, -0.001, 0.000],
        ]
        speed1 = (
            np.sqrt(np.asarray(ux1, dtype=np.float64) ** 2 + np.asarray(uy1, dtype=np.float64) ** 2)
            .round(12)
            .tolist()
        )
        vort1 = [
            [0.00, 0.02, 0.03, 0.02, 0.00],
            [0.00, 0.02, 0.00, -0.02, 0.00],
            [0.00, 0.02, 0.00, -0.02, 0.00],
            [0.00, 0.02, 0.03, 0.02, 0.00],
        ]

        for step, ux, uy, speed, vorticity in [
            (0, ux0, uy0, speed0, vort0),
            (10, ux1, uy1, speed1, vort1),
        ]:
            suffix = f"t{step:06d}"
            write_csv(raw_run_dir / f"ux_{suffix}.csv", ux)
            write_csv(raw_run_dir / f"uy_{suffix}.csv", uy)
            write_csv(raw_run_dir / f"speed_{suffix}.csv", speed)
            write_csv(raw_run_dir / f"vorticity_{suffix}.csv", vorticity)
            write_csv(raw_run_dir / f"mask_{suffix}.csv", mask)

        return raw_run_dir

    def _build_clean_run(self, root: Path) -> Path:
        from fluid_denoise.phase2_clean_dataset import convert_raw_run_to_clean_dataset

        raw_run_dir = self._write_raw_run(root)
        return convert_raw_run_to_clean_dataset(raw_run_dir, clean_root=root / "clean")

    def test_create_validate_and_visualize_noisy_dataset(self) -> None:
        from fluid_denoise.phase2_clean_dataset import load_clean_run
        from fluid_denoise.phase3_noisy_dataset import (
            create_noisy_dataset,
            load_noisy_run,
            save_noisy_comparison_figure,
            summarize_noisy_run,
            validate_noisy_dataset,
        )

        noise_specs = [
            {
                "kind": "gaussian",
                "intensity": 0.2,
                "seed": 11,
                "channels": ["ux"],
            },
            {
                "kind": "missing_blocks",
                "intensity": 0.25,
                "seed": 13,
                "channels": ["uy"],
                "params": {
                    "fill_value": 0.0,
                    "min_block_height": 1,
                    "max_block_height": 2,
                    "min_block_width": 1,
                    "max_block_width": 2,
                    "persistent": True,
                },
            },
            {
                "kind": "piv_outliers",
                "intensity": 0.15,
                "seed": 17,
                "channels": ["ux", "uy"],
                "params": {"outlier_scale": 3.5, "mode": "replace"},
            },
        ]

        with tempfile.TemporaryDirectory(prefix="phase3_noisy_") as tmp:
            root = Path(tmp)
            clean_run_dir = self._build_clean_run(root)
            noisy_run_dir = create_noisy_dataset(
                clean_run_dir,
                noise_specs=noise_specs,
                noisy_root=root / "noisy",
            )

            summary = validate_noisy_dataset(noisy_run_dir)
            self.assertEqual(summary.num_snapshots, 2)
            self.assertEqual(summary.noise_kinds, ("gaussian", "missing_blocks", "piv_outliers"))

            clean_run = load_clean_run(clean_run_dir)
            noisy_run = load_noisy_run(noisy_run_dir)
            self.assertEqual(noisy_run.ux.shape, (2, 4, 5))
            self.assertEqual(noisy_run.corruption_ux_mask.shape, (2, 4, 5))
            self.assertEqual(
                noisy_run.metadata["source"]["source_experiment_id"],
                clean_run.metadata["experiment_id"],
            )

            obstacle = clean_run.mask == 1
            self.assertTrue(np.array_equal(noisy_run.mask, clean_run.mask))
            self.assertTrue(np.all(noisy_run.corruption_ux_mask[:, obstacle] == 0))
            self.assertTrue(np.all(noisy_run.corruption_uy_mask[:, obstacle] == 0))
            self.assertTrue(np.allclose(noisy_run.ux[:, obstacle], clean_run.ux[:, obstacle]))
            self.assertTrue(np.allclose(noisy_run.uy[:, obstacle], clean_run.uy[:, obstacle]))

            inspect_summary = summarize_noisy_run(noisy_run_dir)
            self.assertEqual(inspect_summary["dimensions"]["nx"], 5)
            self.assertGreater(inspect_summary["final_corrupted_cells"]["ux"], 0)

            figure_path = save_noisy_comparison_figure(noisy_run_dir, step=10, variable="ux")
            self.assertTrue(figure_path.exists())

    def test_noisy_dataset_can_corrupt_obstacle_when_requested(self) -> None:
        from fluid_denoise.phase2_clean_dataset import load_clean_run
        from fluid_denoise.phase3_noisy_dataset import create_noisy_dataset, load_noisy_run

        noise_specs = [
            {
                "kind": "salt_and_pepper",
                "intensity": 1.0,
                "seed": 23,
                "channels": ["ux"],
                "include_obstacle": True,
                "params": {
                    "salt_probability": 1.0,
                    "salt_value": 1.0,
                    "pepper_value": -1.0,
                },
            }
        ]

        with tempfile.TemporaryDirectory(prefix="phase3_noisy_") as tmp:
            root = Path(tmp)
            clean_run_dir = self._build_clean_run(root)
            noisy_run_dir = create_noisy_dataset(
                clean_run_dir,
                noise_specs=noise_specs,
                noisy_root=root / "noisy",
            )

            clean_run = load_clean_run(clean_run_dir)
            noisy_run = load_noisy_run(noisy_run_dir)
            obstacle = clean_run.mask == 1

            self.assertTrue(np.all(noisy_run.corruption_ux_mask[:, obstacle] == 1))
            self.assertTrue(np.all(noisy_run.corruption_uy_mask == 0))
            self.assertFalse(np.allclose(noisy_run.ux[:, obstacle], clean_run.ux[:, obstacle]))


if __name__ == "__main__":
    unittest.main()
