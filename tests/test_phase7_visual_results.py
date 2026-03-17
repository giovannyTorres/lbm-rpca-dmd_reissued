from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python" / "src"))


SUMMARY_COLUMNS = (
    "benchmark_id",
    "experiment_id",
    "resolution_label",
    "nx",
    "ny",
    "obstacle_cx",
    "obstacle_cy",
    "obstacle_r",
    "noise_label",
    "noise_specs_json",
    "model_name",
    "model_params_json",
    "model_input_kind",
    "model_input_json",
    "variables",
    "rmse",
    "mae",
    "relative_l2_error",
    "psnr",
    "vorticity_rmse",
    "kinetic_energy_relative_l2",
    "divergence_residual_rms",
    "reconstruction_time_sec",
    "total_time_sec",
    "estimated_memory_bytes",
    "warnings",
    "raw_run_dir",
    "clean_run_dir",
    "noisy_run_dir",
    "reconstruction_dir",
)


class Phase7VisualResultsTest(unittest.TestCase):
    def _write_summary_csv(self, path: Path, rows: list[dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(SUMMARY_COLUMNS))
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _synthetic_velocity_fields(
        self,
        *,
        nx: int,
        ny: int,
        noise_intensity: float,
        reconstruction_scale: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        steps = np.asarray([0, 10, 20], dtype=np.int32)
        yy, xx = np.indices((ny, nx), dtype=np.float64)
        cx = nx // 4
        cy = ny // 2
        radius = max(1, min(nx, ny) // 10)
        mask = (((xx - cx) ** 2 + (yy - cy) ** 2) <= float(radius**2)).astype(np.uint8)

        reference = np.zeros((steps.shape[0], 2, ny, nx), dtype=np.float64)
        observed = np.zeros_like(reference)
        reconstructed = np.zeros_like(reference)
        base_pattern = np.sin(xx / max(1.0, float(nx))) + 0.5 * np.cos(yy / max(1.0, float(ny)))
        correction_pattern = np.cos((xx + yy) / max(1.0, float(nx + ny)))

        for step_index, step in enumerate(steps):
            phase = step_index * 0.2
            ux = 0.04 + 0.003 * np.sin((xx + phase) / max(1.0, float(nx))) + 0.001 * yy / max(1.0, float(ny))
            uy = -0.01 + 0.002 * np.cos((yy + phase) / max(1.0, float(ny))) + 0.0015 * xx / max(1.0, float(nx))
            reference[step_index, 0] = np.where(mask == 1, 0.0, ux)
            reference[step_index, 1] = np.where(mask == 1, 0.0, uy)
            observed[step_index, 0] = np.where(mask == 1, 0.0, ux + noise_intensity * base_pattern)
            observed[step_index, 1] = np.where(mask == 1, 0.0, uy - 0.7 * noise_intensity * base_pattern)
            reconstructed[step_index, 0] = np.where(
                mask == 1,
                0.0,
                ux + reconstruction_scale * noise_intensity * correction_pattern,
            )
            reconstructed[step_index, 1] = np.where(
                mask == 1,
                0.0,
                uy - 0.5 * reconstruction_scale * noise_intensity * correction_pattern,
            )
        return steps, mask, observed, reconstructed, reference

    def _create_synthetic_benchmark(self, root: Path) -> tuple[str, dict[str, object]]:
        from fluid_denoise.phase6_benchmark import compute_benchmark_metrics

        benchmark_id = "phase7_test_benchmark"
        metrics_root = root / "results" / "metrics"
        tables_root = root / "results" / "tables"
        experiments_root = metrics_root / benchmark_id / "experiments"
        summary_rows: list[dict[str, object]] = []

        resolutions = [
            ("coarse", 48, 24),
            ("fine", 96, 48),
        ]
        noise_cases = [
            ("gaussian_low", 0.08),
            ("gaussian_high", 0.16),
        ]
        models = [
            ("truncated_svd", {"rank": 2}, 0.45, "space_time_matrix"),
            ("median_filter", {"kernel_size": 3}, 0.7, "snapshot_batch"),
        ]

        for resolution_label, nx, ny in resolutions:
            for noise_label, noise_intensity in noise_cases:
                for model_name, params, reconstruction_scale, input_kind in models:
                    experiment_id = f"{benchmark_id}__{resolution_label}__{noise_label}__{model_name}"
                    experiment_dir = experiments_root / experiment_id
                    experiment_dir.mkdir(parents=True, exist_ok=True)

                    steps, mask, observed, reconstructed, reference = self._synthetic_velocity_fields(
                        nx=nx,
                        ny=ny,
                        noise_intensity=noise_intensity,
                        reconstruction_scale=reconstruction_scale,
                    )
                    metrics, warnings = compute_benchmark_metrics(
                        reconstructed,
                        reference,
                        variables=["ux", "uy"],
                        mask=mask,
                    )

                    np.savez_compressed(
                        experiment_dir / "reconstruction.npz",
                        steps=steps,
                        mask=mask,
                        observed=observed,
                        reconstructed=reconstructed,
                        reference=reference,
                    )
                    metadata = {
                        "schema_version": "phase5.baseline_reconstruction.v1",
                        "reconstruction_id": experiment_id,
                        "model": {"name": model_name, "params": params, "fit_metadata": {}},
                        "input": {
                            "kind": input_kind,
                            "variables": ["ux", "uy"],
                            "num_snapshots": int(steps.shape[0]),
                            "steps": [int(step) for step in steps.tolist()],
                        },
                        "source": {
                            "run_dir": str((root / "data" / "noisy" / experiment_id).resolve()),
                            "experiment_id": experiment_id,
                            "dataset_kind": "noisy",
                            "schema_version": "phase3.noisy.v1",
                        },
                    }
                    (experiment_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                    (experiment_dir / "benchmark_result.json").write_text(
                        json.dumps({"experiment_id": experiment_id, "metrics": metrics}, indent=2),
                        encoding="utf-8",
                    )

                    reconstruction_time = 0.3 + (0.2 if model_name == "truncated_svd" else 0.1) + noise_intensity
                    total_time = reconstruction_time + 0.2
                    summary_rows.append(
                        {
                            "benchmark_id": benchmark_id,
                            "experiment_id": experiment_id,
                            "resolution_label": resolution_label,
                            "nx": nx,
                            "ny": ny,
                            "obstacle_cx": nx // 4,
                            "obstacle_cy": ny // 2,
                            "obstacle_r": max(1, min(nx, ny) // 10),
                            "noise_label": noise_label,
                            "noise_specs_json": json.dumps(
                                [{"kind": "gaussian", "intensity": noise_intensity, "seed": 11}],
                                sort_keys=True,
                            ),
                            "model_name": model_name,
                            "model_params_json": json.dumps(params, sort_keys=True),
                            "model_input_kind": input_kind,
                            "model_input_json": json.dumps(
                                {"kind": input_kind, "variables": ["ux", "uy"]},
                                sort_keys=True,
                            ),
                            "variables": "ux,uy",
                            "rmse": metrics["rmse"],
                            "mae": metrics["mae"],
                            "relative_l2_error": metrics["relative_l2_error"],
                            "psnr": metrics["psnr"],
                            "vorticity_rmse": metrics["vorticity_rmse"],
                            "kinetic_energy_relative_l2": metrics["kinetic_energy_relative_l2"],
                            "divergence_residual_rms": metrics["divergence_residual_rms"],
                            "reconstruction_time_sec": reconstruction_time,
                            "total_time_sec": total_time,
                            "estimated_memory_bytes": int(observed.nbytes + reconstructed.nbytes + reference.nbytes),
                            "warnings": json.dumps(warnings, sort_keys=True),
                            "raw_run_dir": str((root / "data" / "raw" / experiment_id).resolve()),
                            "clean_run_dir": str((root / "data" / "clean" / experiment_id).resolve()),
                            "noisy_run_dir": str((root / "data" / "noisy" / experiment_id).resolve()),
                            "reconstruction_dir": str(experiment_dir.resolve()),
                        }
                    )

        self._write_summary_csv(tables_root / benchmark_id / "summary.csv", summary_rows)
        config = {
            "source": {
                "benchmark_id": benchmark_id,
                "metrics_root": str(metrics_root),
                "tables_root": str(tables_root),
            },
            "output": {
                "root": str(root / "results" / "visuals"),
                "exploratory_formats": ["png"],
                "final_formats": ["png"],
            },
            "selection": {
                "max_comparison_cases": 2,
                "comparison_rank_metric": "rmse",
            },
            "figures": {
                "variable": "ux",
                "step_policy": "middle",
                "summary_metric": "rmse",
                "time_series_metric": "rmse",
                "time_bar_metric": "reconstruction_time_sec",
            },
            "tables": {
                "sort_by": "rmse",
                "ascending": True,
            },
        }
        return benchmark_id, config

    def test_generate_visual_results_creates_catalog_figures_and_tables(self) -> None:
        from fluid_denoise.phase7_visual_results import generate_visual_results

        with tempfile.TemporaryDirectory(prefix="phase7_visuals_") as tmp:
            root = Path(tmp)
            benchmark_id, config = self._create_synthetic_benchmark(root)
            summary = generate_visual_results(config)

            self.assertEqual(summary.benchmark_id, benchmark_id)
            self.assertEqual(summary.num_selected_experiments, 2)
            self.assertGreater(summary.num_figures, 0)
            self.assertGreater(summary.num_tables, 0)
            self.assertTrue(summary.catalog_json_path.exists())
            self.assertTrue(summary.catalog_csv_path.exists())
            self.assertTrue(summary.catalog_markdown_path.exists())

            self.assertTrue(any((summary.exploratory_dir / "comparisons").glob("*.png")))
            self.assertTrue(any((summary.exploratory_dir / "error_maps").glob("*.png")))
            self.assertTrue(any((summary.exploratory_dir / "error_series").glob("*.png")))
            self.assertTrue(any((summary.exploratory_dir / "aggregate").glob("*.png")))
            self.assertTrue((summary.exploratory_dir / "tables" / "experiment-summary.csv").exists())
            self.assertTrue((summary.exploratory_dir / "tables" / "experiment-summary.tex").exists())
            self.assertTrue(any((summary.thesis_ready_dir / "figures" / "aggregate").glob("*.png")))
            self.assertTrue((summary.thesis_ready_dir / "tables" / "model-aggregate.tex").exists())

            catalog_payload = json.loads(summary.catalog_json_path.read_text(encoding="utf-8"))
            categories = {entry["category"] for entry in catalog_payload["artifacts"]}
            roles = {entry["role"] for entry in catalog_payload["artifacts"]}
            self.assertTrue({"comparison", "error_map", "error_series", "aggregate", "table"}.issubset(categories))
            self.assertTrue({"exploratory", "final", "thesis_ready"}.issubset(roles))

    def test_config_loader_rejects_non_json_and_missing_summary_is_clear(self) -> None:
        from fluid_denoise.phase7_visual_results import generate_visual_results, load_visual_results_config

        with tempfile.TemporaryDirectory(prefix="phase7_visuals_") as tmp:
            root = Path(tmp)
            yaml_path = root / "visuals.yaml"
            yaml_path.write_text("source:\n  benchmark_id: invalid\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "supports JSON configs only"):
                load_visual_results_config(yaml_path)

            metrics_benchmark_root = root / "results" / "metrics" / "missing_benchmark"
            (metrics_benchmark_root / "experiments").mkdir(parents=True, exist_ok=True)
            tables_benchmark_root = root / "results" / "tables" / "missing_benchmark"
            tables_benchmark_root.mkdir(parents=True, exist_ok=True)

            config = {
                "source": {
                    "benchmark_id": "missing_benchmark",
                    "metrics_root": str(root / "results" / "metrics"),
                    "tables_root": str(root / "results" / "tables"),
                },
                "output": {
                    "root": str(root / "results" / "visuals"),
                },
            }
            with self.assertRaisesRegex(RuntimeError, "experiments folder is empty"):
                generate_visual_results(config)

            # add one fake experiment folder so validation reaches summary.csv checks
            (metrics_benchmark_root / "experiments" / "dummy_case").mkdir(parents=True, exist_ok=True)
            with self.assertRaisesRegex(FileNotFoundError, "Benchmark summary CSV not found"):
                generate_visual_results(config)

    def test_generate_visual_results_fails_when_summary_references_missing_or_inconsistent_artifacts(self) -> None:
        from fluid_denoise.phase7_visual_results import generate_visual_results

        with tempfile.TemporaryDirectory(prefix="phase7_visuals_") as tmp:
            root = Path(tmp)
            benchmark_id, config = self._create_synthetic_benchmark(root)

            summary_path = root / "results" / "tables" / benchmark_id / "summary.csv"
            rows = list(csv.DictReader(summary_path.open("r", encoding="utf-8", newline="")))
            first = dict(rows[0])
            first["reconstruction_dir"] = str((root / "results" / "metrics" / benchmark_id / "experiments" / "wrong_dir").resolve())
            self._write_summary_csv(summary_path, [first])

            with self.assertRaisesRegex(ValueError, "does not match the expected FASE 6 path"):
                generate_visual_results(config)


if __name__ == "__main__":
    unittest.main()
