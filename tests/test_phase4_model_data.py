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


class Phase4ModelDataTest(unittest.TestCase):
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

    def _build_clean_and_noisy(self, root: Path) -> tuple[Path, Path]:
        from fluid_denoise.phase2_clean_dataset import convert_raw_run_to_clean_dataset
        from fluid_denoise.phase3_noisy_dataset import create_noisy_dataset

        raw_run_dir = self._write_raw_run(root)
        clean_run_dir = convert_raw_run_to_clean_dataset(raw_run_dir, clean_root=root / "clean")
        noisy_run_dir = create_noisy_dataset(
            clean_run_dir,
            noise_specs=[
                {
                    "kind": "gaussian",
                    "intensity": 0.2,
                    "seed": 11,
                    "channels": ["ux", "uy"],
                }
            ],
            noisy_root=root / "noisy",
        )
        return clean_run_dir, noisy_run_dir

    def test_snapshot_batch_pack_unpack_and_alignment(self) -> None:
        from fluid_denoise.phase4_model_data import (
            align_model_runs,
            load_model_run,
            pack_snapshot_batch,
            stack_run_variables,
            unpack_snapshot_batch,
        )

        with tempfile.TemporaryDirectory(prefix="phase4_model_") as tmp:
            root = Path(tmp)
            clean_run_dir, noisy_run_dir = self._build_clean_and_noisy(root)

            pair = align_model_runs(noisy_run_dir, clean_run_dir)
            self.assertEqual(pair.observed.dataset_kind, "noisy")
            self.assertEqual(pair.reference.dataset_kind, "clean")

            clean_run = load_model_run(clean_run_dir)
            stacked, selected_steps = stack_run_variables(clean_run, variables=["ux", "uy"])
            batch = pack_snapshot_batch(
                clean_run,
                variables=["ux", "uy"],
                include_mask_channel=True,
                mask_policy="keep",
            )

            self.assertEqual(selected_steps.tolist(), [0, 10])
            self.assertEqual(batch.data.shape, (2, 3, 4, 5))
            self.assertTrue(np.all(batch.data[:, 2] == clean_run.mask))

            restored = unpack_snapshot_batch(batch)
            self.assertTrue(np.allclose(restored, stacked))

    def test_temporal_window_fill_obstacle_and_matrix_normalization(self) -> None:
        from fluid_denoise.phase4_model_data import (
            NormalizationSpec,
            invert_normalization,
            load_model_run,
            pack_space_time_matrix,
            pack_temporal_windows,
            unpack_space_time_matrix,
            unpack_temporal_windows,
        )

        with tempfile.TemporaryDirectory(prefix="phase4_model_") as tmp:
            root = Path(tmp)
            clean_run_dir, noisy_run_dir = self._build_clean_and_noisy(root)
            clean_run = load_model_run(clean_run_dir)
            noisy_run = load_model_run(noisy_run_dir)

            windows = pack_temporal_windows(
                clean_run,
                variables=["ux", "uy"],
                window_size=2,
                mask_policy="fill_obstacle",
                obstacle_fill_value=-99.0,
            )
            restored_windows = unpack_temporal_windows(windows)
            obstacle = clean_run.mask == 1
            obstacle_flat = obstacle.reshape(-1)

            self.assertEqual(windows.data.shape, (1, 2, 2, 4, 5))
            self.assertTrue(
                np.all(
                    restored_windows.reshape(1, 2, 2, -1)[..., obstacle_flat] == -99.0
                )
            )

            raw_matrix = pack_space_time_matrix(
                noisy_run,
                variables=["ux", "uy"],
                mask_policy="flatten_fluid",
                normalization=None,
            )
            normalized_matrix = pack_space_time_matrix(
                noisy_run,
                variables=["ux", "uy"],
                mask_policy="flatten_fluid",
                normalization=NormalizationSpec(mode="standardize", scope="per_feature"),
            )

            self.assertEqual(raw_matrix.data.shape[1], 2)
            self.assertEqual(raw_matrix.data.shape[0], 2 * int(np.count_nonzero(clean_run.mask == 0)))
            self.assertIsNotNone(normalized_matrix.normalization)

            restored_raw = invert_normalization(
                normalized_matrix.data,
                normalized_matrix.normalization,
            )
            self.assertTrue(np.allclose(restored_raw, raw_matrix.data))

            unpacked_matrix = unpack_space_time_matrix(raw_matrix)
            self.assertEqual(unpacked_matrix.shape, (2, 2, 4, 5))
            self.assertTrue(
                np.all(unpacked_matrix.reshape(2, 2, -1)[..., obstacle_flat] == 0.0)
            )

    def test_dimension_validations_raise_clear_errors(self) -> None:
        from fluid_denoise.phase4_model_data import load_model_run, pack_temporal_windows

        with tempfile.TemporaryDirectory(prefix="phase4_model_") as tmp:
            root = Path(tmp)
            clean_run_dir, _ = self._build_clean_and_noisy(root)
            clean_run = load_model_run(clean_run_dir)

            with self.assertRaisesRegex(ValueError, "window_size cannot exceed"):
                pack_temporal_windows(
                    clean_run,
                    variables=["ux"],
                    window_size=3,
                )

            with self.assertRaisesRegex(ValueError, "Snapshot index out of bounds"):
                pack_temporal_windows(
                    clean_run,
                    variables=["ux"],
                    window_size=1,
                    snapshot_indices=[0, 10],
                )


if __name__ == "__main__":
    unittest.main()
