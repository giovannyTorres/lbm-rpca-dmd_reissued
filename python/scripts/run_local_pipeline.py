#!/usr/bin/env python3
"""Run local thesis pipeline profiles (minimal, light, full)."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "python" / "src"))

    from fluid_denoise.pipeline_runner import main as pipeline_main

    pipeline_main()


if __name__ == "__main__":
    main()
