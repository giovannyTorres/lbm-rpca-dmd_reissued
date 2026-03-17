#!/usr/bin/env python3
"""Run one FASE 6 benchmark from a JSON config file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "python" / "src"))

    from fluid_denoise.phase6_benchmark import run_benchmark_from_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark_phase6_example.json")
    args = parser.parse_args()

    config_path = (repo_root / args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    summary = run_benchmark_from_config(config_path)
    print(f"[BENCHMARK] benchmark_id={summary.benchmark_id}")
    print(
        f"[BENCHMARK] completed={summary.completed} failed={summary.failed} skipped={summary.skipped} "
        f"of total_cases={summary.num_cases}"
    )
    print(f"[BENCHMARK] ledger={summary.ledger_path}")
    print(f"[BENCHMARK] summary_csv={summary.summary_csv_path}")
    if summary.summary_parquet_path is not None:
        print(f"[BENCHMARK] summary_parquet={summary.summary_parquet_path}")


if __name__ == "__main__":
    main()
