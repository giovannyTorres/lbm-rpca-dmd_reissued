#!/usr/bin/env python3
"""Run the minimal benchmark-only smoke pipeline for local validation."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "python" / "src"))

    from fluid_denoise.pipeline_runner import run_pipeline_mode

    summary = run_pipeline_mode("minimal")
    print(
        "[PIPELINE_TEST] "
        f"benchmark_id={summary.benchmark_summary.benchmark_id} "
        f"completed={summary.benchmark_summary.completed}/{summary.benchmark_summary.num_cases} "
        f"failed={summary.benchmark_summary.failed} "
        f"trace={summary.trace_path}"
    )


if __name__ == "__main__":
    main()
