#!/usr/bin/env python3
"""Reproducible research campaign; set BLAS thread budgets before imports."""
import os
for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python" / "src"))
from fluid_denoise.research.runner import main

if __name__ == "__main__":
    main()
