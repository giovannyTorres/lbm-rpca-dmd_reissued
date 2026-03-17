from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python" / "src"))


def write_csv(path: Path, rows: list[list[float]]) -> None:
    path.write_text(
        "\n".join(",".join(str(value) for value in row) for row in rows) + "\n",
        encoding="utf-8",
    )


class Phase6BenchmarkTest(unittest.TestCase):
    def _build_config(self, root: Path) -> dict[str, object]:
        base_config_path = root / "configs" / "base_solver.json"
        base_config_path.parent.mkdir(parents=True, exist_ok=True)
        base_config = {
            "nx": 96,
            "ny": 48,
            "reynolds": 100.0,
            "u_in": 0.04,
            "iterations": 10,
            "save_stride": 10,
            "obstacle_cx": 24,
            "obstacle_cy": 24,
            "obstacle_r": 6,
            "output_root": str(root / "raw"),
            "run_id": "unused_base_run_id",
            "seed": 0,
        }
        base_config_path.write_text(json.dumps(base_config, indent=2), encoding="utf-8")

        return {
            "benchmark": {
                "benchmark_id": "phase6_test_benchmark",
                "metrics_root": str(root / "results" / "metrics"),
                "tables_root": str(root / "results" / "tables"),
                "export_formats": ["csv", "parquet"],
                "resume": True,
            },
            "solver": {
                "base_config_path": str(base_config_path),
                "build_dir": str(root / "build_unused"),
                "build_type": "Release",
                "resolutions": [
                    {"label": "half", "nx": 48, "ny": 24},
                    {"label": "full", "nx": 96, "ny": 48},
                ],
            },
            "noise": [
                {
                    "label": "gaussian_low",
                    "noise_specs": [
                        {
                            "kind": "gaussian",
                            "intensity": 0.1,
                            "seed": 11,
                            "channels": ["ux", "uy"],
                        }
                    ],
                },
                {
                    "label": "gaussian_high",
                    "noise_specs": [
                        {
                            "kind": "gaussian",
                            "intensity": 0.2,
                            "seed": 13,
                            "channels": ["ux", "uy"],
                        }
                    ],
                },
            ],
            "models": [
                {
                    "name": "truncated_svd",
                    "input": {
                        "kind": "space_time_matrix",
                        "variables": ["ux", "uy"],
                        "mask_policy": "flatten_fluid",
                    },
                    "param_grid": {"rank": [1, 2]},
                },
                {
                    "name": "median_filter",
                    "input": {
                        "kind": "snapshot_batch",
                        "variables": ["ux", "uy"],
                        "mask_policy": "fill_obstacle",
                        "obstacle_fill_value": 0.0,
                    },
                    "params": {"kernel_size": 3},
                },
            ],
            "execution": {
                "reuse_existing": True,
                "clean_root": str(root / "clean"),
                "noisy_root": str(root / "noisy"),
            },
        }

    def _write_raw_run_for_case(self, case) -> None:
        raw_run_dir = case.raw_run_dir
        raw_run_dir.mkdir(parents=True, exist_ok=True)
        ny = case.ny
        nx = case.nx
        cy = case.scaled_geometry["obstacle_cy"]
        cx = case.scaled_geometry["obstacle_cx"]
        radius = case.scaled_geometry["obstacle_r"]

        yy, xx = np.indices((ny, nx), dtype=np.float64)
        mask = (((xx - cx) ** 2 + (yy - cy) ** 2) <= float(radius**2)).astype(np.uint8)

        manifest = {
            "config_path": str(case.generated_solver_config_path),
            "nx": nx,
            "ny": ny,
            "reynolds": float(case.solver_config["reynolds"]),
            "u_in": float(case.solver_config["u_in"]),
            "iterations": int(case.solver_config["iterations"]),
            "save_stride": int(case.solver_config["save_stride"]),
            "obstacle_cx": int(case.scaled_geometry["obstacle_cx"]),
            "obstacle_cy": int(case.scaled_geometry["obstacle_cy"]),
            "obstacle_r": int(case.scaled_geometry["obstacle_r"]),
            "output_root": str(case.raw_run_dir.parent),
            "run_id": str(case.solver_config["run_id"]),
            "nu": 0.0008,
            "tau": 0.5024,
        }
        (raw_run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        for step, phase in [(0, 0.0), (10, 0.25)]:
            ux = 0.04 - 0.0003 * xx + 0.0002 * yy + phase * 0.002
            uy = 0.002 * np.sin((xx + phase) / max(1.0, float(nx))) - 0.001 * np.cos((yy + phase) / max(1.0, float(ny)))
            ux = np.where(mask == 1, 0.0, ux)
            uy = np.where(mask == 1, 0.0, uy)
            speed = np.sqrt(ux * ux + uy * uy)
            vorticity = np.gradient(uy, axis=1) - np.gradient(ux, axis=0)

            suffix = f"t{step:06d}"
            write_csv(raw_run_dir / f"ux_{suffix}.csv", ux.round(12).tolist())
            write_csv(raw_run_dir / f"uy_{suffix}.csv", uy.round(12).tolist())
            write_csv(raw_run_dir / f"speed_{suffix}.csv", speed.round(12).tolist())
            write_csv(raw_run_dir / f"vorticity_{suffix}.csv", vorticity.round(12).tolist())
            write_csv(raw_run_dir / f"mask_{suffix}.csv", mask.tolist())

    def test_expand_benchmark_cases_scales_geometry_and_produces_unique_ids(self) -> None:
        from fluid_denoise.phase6_benchmark import expand_benchmark_cases

        with tempfile.TemporaryDirectory(prefix="phase6_benchmark_") as tmp:
            root = Path(tmp)
            config = self._build_config(root)
            cases = expand_benchmark_cases(config, config_base_dir=root)

            self.assertEqual(len(cases), 12)
            self.assertEqual(len({case.experiment_id for case in cases}), 12)

            half_case = next(case for case in cases if case.resolution_label == "half")
            self.assertEqual(half_case.scaled_geometry["obstacle_cx"], 12)
            self.assertEqual(half_case.scaled_geometry["obstacle_cy"], 12)
            self.assertEqual(half_case.scaled_geometry["obstacle_r"], 3)

    def test_compute_benchmark_metrics_masks_obstacle_and_handles_missing_velocity(self) -> None:
        from fluid_denoise.phase6_benchmark import compute_benchmark_metrics

        mask = np.asarray([[0, 1], [0, 0]], dtype=np.uint8)
        reference = np.asarray(
            [[[[1.0, 9.0], [3.0, 5.0]], [[0.5, 2.0], [1.5, 2.5]]]],
            dtype=np.float64,
        )
        reconstructed = np.asarray(
            [[[[2.0, 100.0], [4.0, 6.0]], [[1.5, -50.0], [2.5, 3.5]]]],
            dtype=np.float64,
        )

        metrics, warnings = compute_benchmark_metrics(
            reconstructed,
            reference,
            variables=["ux", "uy"],
            mask=mask,
        )

        self.assertAlmostEqual(metrics["rmse"], 1.0)
        self.assertAlmostEqual(metrics["mae"], 1.0)
        self.assertGreater(metrics["relative_l2_error"], 0.0)
        self.assertIsNotNone(metrics["vorticity_rmse"])
        self.assertEqual(warnings, [])

        speed_reference = np.asarray([[[[1.0, 5.0], [2.0, 4.0]]]], dtype=np.float64)
        speed_reconstructed = speed_reference + 0.5
        metrics_missing, warnings_missing = compute_benchmark_metrics(
            speed_reconstructed,
            speed_reference,
            variables=["speed"],
            mask=mask,
        )
        self.assertIsNone(metrics_missing["vorticity_rmse"])
        self.assertTrue(any("Physical metrics require both ux and uy" in warning for warning in warnings_missing))

    def test_run_benchmark_writes_outputs_and_resume_skips_completed_cases(self) -> None:
        from fluid_denoise.phase6_benchmark import expand_benchmark_cases, run_benchmark

        with tempfile.TemporaryDirectory(prefix="phase6_benchmark_") as tmp:
            root = Path(tmp)
            config = self._build_config(root)
            config["solver"]["resolutions"] = [{"label": "full", "nx": 96, "ny": 48}]
            cases = expand_benchmark_cases(config, config_base_dir=root)

            for case in {case.raw_run_dir: case for case in cases}.values():
                self._write_raw_run_for_case(case)

            summary = run_benchmark(config)
            self.assertEqual(summary.num_cases, 6)
            self.assertEqual(summary.completed, 6)
            self.assertEqual(summary.failed, 0)
            self.assertEqual(summary.skipped, 0)
            self.assertTrue(summary.summary_csv_path.exists())
            self.assertTrue(summary.summary_parquet_path.exists())

            ledger_lines = summary.ledger_path.read_text(encoding="utf-8").splitlines()
            completed_entries = [json.loads(line) for line in ledger_lines if json.loads(line)["status"] == "completed"]
            self.assertEqual(len(completed_entries), 6)

            second_summary = run_benchmark(config)
            self.assertEqual(second_summary.completed, 0)
            self.assertEqual(second_summary.failed, 0)
            self.assertEqual(second_summary.skipped, 6)

            rows = summary.summary_csv_path.read_text(encoding="utf-8").splitlines()
            self.assertGreaterEqual(len(rows), 7)

    def test_runner_records_failed_case_and_parquet_dependency_error_is_clear(self) -> None:
        from fluid_denoise.phase6_benchmark import expand_benchmark_cases, load_benchmark_config, run_benchmark

        with tempfile.TemporaryDirectory(prefix="phase6_benchmark_") as tmp:
            root = Path(tmp)
            yaml_path = root / "benchmark.yaml"
            yaml_path.write_text("benchmark_id: not_supported\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "supports JSON configs only"):
                load_benchmark_config(yaml_path)

            config = self._build_config(root)
            config["solver"]["resolutions"] = [{"label": "full", "nx": 96, "ny": 48}]
            config["models"] = [
                {
                    "name": "median_filter",
                    "input": {
                        "kind": "space_time_matrix",
                        "variables": ["ux", "uy"],
                    },
                    "params": {"kernel_size": 3},
                }
            ]
            cases = expand_benchmark_cases(config, config_base_dir=root)
            for case in {case.raw_run_dir: case for case in cases}.values():
                self._write_raw_run_for_case(case)

            with self.assertRaisesRegex(RuntimeError, "finished without completed cases"):
                run_benchmark(config)

            ledger_path = root / "results" / "metrics" / "phase6_test_benchmark" / "ledger.jsonl"
            ledger_entries = [
                json.loads(line)
                for line in ledger_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            failed_entries = [entry for entry in ledger_entries if entry["status"] == "failed"]
            self.assertEqual(len(failed_entries), 2)
            self.assertTrue(any("Median filter requires a snapshot_batch input" in entry["error"] for entry in failed_entries))

            valid_config = self._build_config(root)
            valid_config["solver"]["resolutions"] = [{"label": "full", "nx": 96, "ny": 48}]
            valid_cases = expand_benchmark_cases(valid_config, config_base_dir=root)
            for case in {case.raw_run_dir: case for case in valid_cases}.values():
                self._write_raw_run_for_case(case)
            valid_config["benchmark"]["benchmark_id"] = "phase6_no_parquet"

            with patch("fluid_denoise.phase6_benchmark._ensure_parquet_supported", side_effect=RuntimeError("Parquet export was requested, but pandas+pyarrow are not available in this environment.")):
                with self.assertRaisesRegex(RuntimeError, "Parquet export was requested"):
                    run_benchmark(valid_config)


if __name__ == "__main__":
    unittest.main()
