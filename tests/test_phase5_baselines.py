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


class Phase5BaselineTest(unittest.TestCase):
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
                },
                {
                    "kind": "piv_outliers",
                    "intensity": 0.1,
                    "seed": 17,
                    "channels": ["ux", "uy"],
                    "params": {"outlier_scale": 3.0, "mode": "replace"},
                },
            ],
            noisy_root=root / "noisy",
        )
        return clean_run_dir, noisy_run_dir

    def test_matrix_baselines_fit_reconstruct_and_keep_interface(self) -> None:
        from fluid_denoise.phase5_baselines import create_baseline_model, prepare_baseline_input

        with tempfile.TemporaryDirectory(prefix="phase5_baseline_") as tmp:
            root = Path(tmp)
            _, noisy_run_dir = self._build_clean_and_noisy(root)
            prepared = prepare_baseline_input(
                noisy_run_dir,
                input_kind="space_time_matrix",
                variables=["ux", "uy"],
                mask_policy="flatten_fluid",
            )

            for name, params in [
                ("rpca_ialm", {"max_iter": 50, "tol": 1e-5}),
                ("dmd", {"rank": 2}),
                ("truncated_svd", {"rank": 2}),
            ]:
                model = create_baseline_model(name, params)
                model.fit(prepared)
                reconstructed = model.reconstruct()

                self.assertEqual(model.get_name(), name)
                self.assertEqual(model.get_params(), params)
                self.assertEqual(reconstructed.shape, prepared.data.shape)
                self.assertTrue(np.all(np.isfinite(reconstructed)))

    def test_spatial_baselines_emit_honest_warnings(self) -> None:
        from fluid_denoise.phase5_baselines import create_baseline_model, prepare_baseline_input

        with tempfile.TemporaryDirectory(prefix="phase5_baseline_") as tmp:
            root = Path(tmp)
            _, noisy_run_dir = self._build_clean_and_noisy(root)
            prepared = prepare_baseline_input(
                noisy_run_dir,
                input_kind="snapshot_batch",
                variables=["ux", "uy"],
                mask_policy="keep",
            )

            median_model = create_baseline_model("median_filter", {"kernel_size": 3})
            median_model.fit(prepared)
            median_reconstruction = median_model.reconstruct()
            self.assertEqual(median_reconstruction.shape, prepared.data.shape)
            self.assertTrue(any("mask_policy='keep'" in warning for warning in median_model.get_warnings()))

            wiener_model = create_baseline_model("wiener_filter", {"kernel_size": 3})
            wiener_model.fit(prepared)
            wiener_reconstruction = wiener_model.reconstruct()
            self.assertEqual(wiener_reconstruction.shape, prepared.data.shape)
            self.assertTrue(any("locally stationary additive noise" in warning for warning in wiener_model.get_warnings()))

    def test_pipeline_saves_reconstruction_and_metadata(self) -> None:
        from fluid_denoise.phase5_baselines import run_baseline_pipeline, summarize_baseline_reconstruction

        with tempfile.TemporaryDirectory(prefix="phase5_baseline_") as tmp:
            root = Path(tmp)
            clean_run_dir, noisy_run_dir = self._build_clean_and_noisy(root)
            config = {
                "source": {
                    "run_dir": str(noisy_run_dir),
                    "reference_run_dir": str(clean_run_dir),
                },
                "input": {
                    "kind": "space_time_matrix",
                    "variables": ["ux", "uy"],
                    "mask_policy": "flatten_fluid",
                },
                "model": {
                    "name": "truncated_svd",
                    "params": {"rank": 2},
                },
                "output": {
                    "root": str(root / "processed" / "baselines"),
                    "overwrite": True,
                },
            }

            output_dir = run_baseline_pipeline(config)
            self.assertTrue((output_dir / "metadata.json").exists())
            self.assertTrue((output_dir / "reconstruction.npz").exists())

            summary = summarize_baseline_reconstruction(output_dir)
            self.assertEqual(summary.model_name, "truncated_svd")
            self.assertEqual(summary.variables, ("ux", "uy"))
            self.assertEqual(summary.num_snapshots, 2)

            metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["source"]["dataset_kind"], "noisy")
            self.assertTrue(metadata["output"]["reference_available"])

            with np.load(output_dir / "reconstruction.npz") as payload:
                self.assertIn("observed", payload.files)
                self.assertIn("reconstructed", payload.files)
                self.assertIn("reference", payload.files)
                self.assertEqual(payload["reconstructed"].shape, (2, 2, 4, 5))


if __name__ == "__main__":
    unittest.main()
