from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class Phase1SmokeTest(unittest.TestCase):
    def test_phase1_generates_snapshots(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        tmpdir = Path(tempfile.mkdtemp(prefix="lbm_phase1_"))
        try:
            run_id = "smoke_run"
            out_root = tmpdir / "raw"
            cfg = tmpdir / "smoke.cfg"
            cfg.write_text(
                "\n".join(
                    [
                        "nx = 80",
                        "ny = 40",
                        "reynolds = 120.0",
                        "u_in = 0.05",
                        "iterations = 30",
                        "save_stride = 10",
                        "seed = 123",
                        "obstacle_cx = 20",
                        "obstacle_cy = 20",
                        "obstacle_r = 5",
                        f"output_root = {out_root}",
                        f"run_id = {run_id}",
                    ]
                )
            )

            build_dir = repo / "cpp/lbm_core/build_test"
            build_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(["cmake", ".."], cwd=build_dir, check=True)
            subprocess.run(["cmake", "--build", ".", "-j"], cwd=build_dir, check=True)
            subprocess.run(["./lbm_sim", "--config", str(cfg)], cwd=build_dir, check=True)

            run_dir = out_root / run_id
            self.assertTrue((run_dir / "manifest.txt").exists())
            self.assertTrue((run_dir / "ux_t000000.csv").exists())
            self.assertTrue((run_dir / "ux_t000010.csv").exists())
            self.assertTrue((run_dir / "vort_t000030.csv").exists())

            sample = (run_dir / "umag_t000030.csv").read_text().splitlines()[0]
            self.assertIn(",", sample)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
