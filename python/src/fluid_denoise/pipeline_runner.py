from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

REPO_ROOT = Path(__file__).resolve().parents[3]
LOGGER = logging.getLogger(__name__)

PipelineMode = Literal["minimal", "light", "full"]


@dataclass(frozen=True)
class PipelineConfig:
    mode: PipelineMode
    benchmark_config_path: Path
    visual_config_path: Path
    run_visuals: bool


@dataclass(frozen=True)
class PipelineExecutionSummary:
    mode: PipelineMode
    benchmark_summary: Any
    visual_summary: Any | None
    trace_path: Path


def _resolve_repo_path(raw_path: str | Path) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


def _config_for_mode(mode: PipelineMode) -> PipelineConfig:
    mode_map: dict[PipelineMode, PipelineConfig] = {
        "minimal": PipelineConfig(
            mode="minimal",
            benchmark_config_path=_resolve_repo_path("configs/benchmark_phase6_smoke.json"),
            visual_config_path=_resolve_repo_path("configs/visual_results_phase7_smoke.json"),
            run_visuals=False,
        ),
        "light": PipelineConfig(
            mode="light",
            benchmark_config_path=_resolve_repo_path("configs/benchmark_phase6_smoke.json"),
            visual_config_path=_resolve_repo_path("configs/visual_results_phase7_smoke.json"),
            run_visuals=True,
        ),
        "full": PipelineConfig(
            mode="full",
            benchmark_config_path=_resolve_repo_path("configs/benchmark_phase6_example.json"),
            visual_config_path=_resolve_repo_path("configs/visual_results_phase7_example.json"),
            run_visuals=True,
        ),
    }
    return mode_map[mode]


def _write_pipeline_trace(
    *,
    config: PipelineConfig,
    benchmark_summary: Any,
    visual_summary: Any | None,
) -> Path:
    trace_path = benchmark_summary.metrics_root / "pipeline_trace.json"
    trace_payload = {
        "mode": config.mode,
        "benchmark_config_path": str(config.benchmark_config_path),
        "visual_config_path": str(config.visual_config_path),
        "run_visuals": config.run_visuals,
        "benchmark": {
            "benchmark_id": benchmark_summary.benchmark_id,
            "num_cases": benchmark_summary.num_cases,
            "completed": benchmark_summary.completed,
            "failed": benchmark_summary.failed,
            "skipped": benchmark_summary.skipped,
            "metrics_root": str(benchmark_summary.metrics_root),
            "ledger_path": str(benchmark_summary.ledger_path),
            "summary_csv_path": str(benchmark_summary.summary_csv_path),
            "summary_parquet_path": (
                str(benchmark_summary.summary_parquet_path)
                if benchmark_summary.summary_parquet_path is not None
                else None
            ),
        },
        "visual_results": (
            {
                "benchmark_id": visual_summary.benchmark_id,
                "num_selected_experiments": visual_summary.num_selected_experiments,
                "num_figures": visual_summary.num_figures,
                "num_tables": visual_summary.num_tables,
                "output_root": str(visual_summary.output_root),
                "catalog_json_path": str(visual_summary.catalog_json_path),
                "catalog_csv_path": str(visual_summary.catalog_csv_path),
                "catalog_markdown_path": str(visual_summary.catalog_markdown_path),
            }
            if visual_summary is not None
            else None
        ),
    }
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(json.dumps(trace_payload, indent=2, sort_keys=True), encoding="utf-8")
    return trace_path


def run_benchmark_from_config(config_path: Path | str) -> Any:
    from fluid_denoise.phase6_benchmark import run_benchmark_from_config as run_benchmark

    return run_benchmark(config_path)


def generate_visual_results_from_config(config_path: Path | str) -> Any:
    from fluid_denoise.phase7_visual_results import (
        generate_visual_results_from_config as run_visuals,
    )

    return run_visuals(config_path)


def run_pipeline_mode(mode: PipelineMode) -> PipelineExecutionSummary:
    config = _config_for_mode(mode)
    if not config.benchmark_config_path.exists():
        raise FileNotFoundError(f"Benchmark config does not exist: {config.benchmark_config_path}")

    LOGGER.info("Running benchmark with config: %s", config.benchmark_config_path)
    benchmark_summary = run_benchmark_from_config(config.benchmark_config_path)

    visual_summary: Any | None = None
    if config.run_visuals:
        if not config.visual_config_path.exists():
            raise FileNotFoundError(f"Visual config does not exist: {config.visual_config_path}")
        LOGGER.info("Running visual results with config: %s", config.visual_config_path)
        visual_summary = generate_visual_results_from_config(config.visual_config_path)

    trace_path = _write_pipeline_trace(
        config=config,
        benchmark_summary=benchmark_summary,
        visual_summary=visual_summary,
    )
    LOGGER.info("Wrote pipeline trace to: %s", trace_path)

    return PipelineExecutionSummary(
        mode=mode,
        benchmark_summary=benchmark_summary,
        visual_summary=visual_summary,
        trace_path=trace_path,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run thesis pipeline profiles")
    parser.add_argument(
        "--mode",
        choices=["minimal", "light", "full"],
        default="minimal",
        help="minimal=benchmark smoke only, light=smoke + visuals, full=example benchmark + visuals",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args()

    summary = run_pipeline_mode(args.mode)
    print(
        "[PIPELINE] "
        f"mode={summary.mode} "
        f"benchmark_id={summary.benchmark_summary.benchmark_id} "
        f"completed={summary.benchmark_summary.completed}/{summary.benchmark_summary.num_cases} "
        f"failed={summary.benchmark_summary.failed} "
        f"skipped={summary.benchmark_summary.skipped}"
    )
    print(f"[PIPELINE] trace={summary.trace_path}")


__all__ = [
    "PipelineConfig",
    "PipelineExecutionSummary",
    "PipelineMode",
    "build_arg_parser",
    "run_pipeline_mode",
]


if __name__ == "__main__":
    main()
