from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python" / "src"))


class PipelineRunnerTest(unittest.TestCase):
    def test_run_pipeline_mode_minimal_writes_trace(self) -> None:
        from fluid_denoise.pipeline_runner import run_pipeline_mode

        with tempfile.TemporaryDirectory(prefix="lbm_pipeline_trace_") as tmp:
            metrics_root = Path(tmp) / "metrics"
            ledger_path = metrics_root / "ledger.jsonl"
            summary_csv_path = metrics_root / "summary.csv"

            benchmark_summary = mock.Mock()
            benchmark_summary.benchmark_id = "bench"
            benchmark_summary.num_cases = 1
            benchmark_summary.completed = 1
            benchmark_summary.failed = 0
            benchmark_summary.skipped = 0
            benchmark_summary.metrics_root = metrics_root
            benchmark_summary.ledger_path = ledger_path
            benchmark_summary.summary_csv_path = summary_csv_path
            benchmark_summary.summary_parquet_path = None

            with mock.patch(
                "fluid_denoise.pipeline_runner.run_benchmark_from_config",
                return_value=benchmark_summary,
            ) as benchmark_mock:
                summary = run_pipeline_mode("minimal")

            benchmark_mock.assert_called_once()
            self.assertEqual(summary.mode, "minimal")
            self.assertTrue(summary.trace_path.exists())
            trace_text = summary.trace_path.read_text(encoding="utf-8")
            self.assertIn('"mode": "minimal"', trace_text)
            self.assertIn('"benchmark_id": "bench"', trace_text)

    def test_full_mode_invokes_visual_generation(self) -> None:
        from fluid_denoise.pipeline_runner import run_pipeline_mode

        benchmark_summary = mock.Mock()
        benchmark_summary.benchmark_id = "phase6_example"
        benchmark_summary.num_cases = 2
        benchmark_summary.completed = 2
        benchmark_summary.failed = 0
        benchmark_summary.skipped = 0
        benchmark_summary.metrics_root = REPO_ROOT / "results" / "metrics" / "phase6_example"
        benchmark_summary.ledger_path = benchmark_summary.metrics_root / "ledger.jsonl"
        benchmark_summary.summary_csv_path = benchmark_summary.metrics_root / "summary.csv"
        benchmark_summary.summary_parquet_path = None

        visual_summary = mock.Mock()
        visual_summary.benchmark_id = "phase6_example"
        visual_summary.num_selected_experiments = 2
        visual_summary.num_figures = 3
        visual_summary.num_tables = 2
        visual_summary.output_root = REPO_ROOT / "results" / "visuals" / "phase6_example"
        visual_summary.catalog_json_path = visual_summary.output_root / "catalog.json"
        visual_summary.catalog_csv_path = visual_summary.output_root / "catalog.csv"
        visual_summary.catalog_markdown_path = visual_summary.output_root / "catalog.md"

        with mock.patch(
            "fluid_denoise.pipeline_runner.run_benchmark_from_config",
            return_value=benchmark_summary,
        ), mock.patch(
            "fluid_denoise.pipeline_runner.generate_visual_results_from_config",
            return_value=visual_summary,
        ) as visual_mock:
            summary = run_pipeline_mode("full")

        visual_mock.assert_called_once()
        self.assertIsNotNone(summary.visual_summary)


if __name__ == "__main__":
    unittest.main()
