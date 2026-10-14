"""Phase-4 vector alignment and explicit physical-declaration contracts."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python" / "src"))

from fluid_denoise.phase4_model_data import ModelRun, ModelRunPair, pack_space_time_matrix
from fluid_denoise.research.bridge import model_pair_to_research, model_run_to_research
from fluid_denoise.research.metrics import physical_metrics


class Phase4ResearchBridgeTests(unittest.TestCase):
    def setUp(self):
        y, x = np.mgrid[:6, :7]
        ux = np.stack([1 + .1 * x + .2 * t for t in range(3)])
        uy = np.stack([.3 * y - .1 * t for t in range(3)])
        mask = np.zeros((6, 7), dtype=np.uint8)
        mask[2, 3] = 1
        arrays = {"ux": ux, "uy": uy, "speed": np.hypot(ux, uy),
                  "vorticity": np.zeros_like(ux)}
        self.run = ModelRun(Path("declared_phase4_run"), "clean", "phase2_clean_v1", "fixture",
                            {}, np.array([0, 5, 10]), mask, arrays, {})
        self.declarations = dict(times=[0, .5, 1], dx=.1, dy=.2,
                                 units="lattice", boundary="open", periodic_axes=[])

    def test_pair_preserves_components_mask_steps_and_phase4_feature_order(self):
        observed = ModelRun(self.run.run_dir, "noisy", self.run.schema_version, "observed", {},
                            self.run.steps, self.run.mask,
                            {**self.run.arrays, "ux": self.run.arrays["ux"] + .01}, {})
        pair = model_pair_to_research(ModelRunPair(observed, self.run), **self.declarations)
        data = pair.reference
        self.assertEqual(data["velocity"].shape, (3, 2, 6, 7))
        np.testing.assert_array_equal(data["steps"], [0, 5, 10])
        np.testing.assert_array_equal(data["mask"], self.run.mask.astype(bool))
        np.testing.assert_allclose(pair.observed["velocity"][:, 0], self.run.arrays["ux"] + .01)
        matrix = pack_space_time_matrix(self.run, variables=["ux", "uy"])
        np.testing.assert_allclose(data["velocity"][:, :, ~data["mask"]].reshape(3, -1).T,
                                   matrix.data)
        self.assertFalse(data["metadata"]["dns_certified"])
        self.assertIsNone(data["density"])
        self.assertEqual(data["metadata"]["density_status"], "unknown")

    def test_unknown_density_keeps_kinematics_and_withholds_physical_mass(self):
        data = model_run_to_research(self.run, **self.declarations)
        metrics = physical_metrics(data, data["velocity"])
        self.assertEqual(metrics["velocity_relative_l2"], 0)
        self.assertEqual(metrics["enstrophy_relative_l2"], 0)
        self.assertFalse(metrics["density_metrics_available"])
        self.assertFalse(metrics["constant_density_mass_trivial"])
        for name in ("mass_relative_l2", "kinetic_energy_relative_l2", "continuity_rms",
                     "boundary_flux_rms", "mass_balance_rms"):
            self.assertIsNone(metrics[name])
        self.assertIsNone(metrics["series"]["reference_mass"])
        json.dumps(metrics, allow_nan=False)

    def test_supplied_densities_are_independent_and_not_estimated(self):
        pair = model_pair_to_research(ModelRunPair(self.run, self.run),
                                      reference_density=2, **self.declarations)
        self.assertIsNone(pair.observed["density"])
        np.testing.assert_allclose(pair.reference["density"], 2)
        metrics = physical_metrics(pair.reference, pair.reference["velocity"])
        self.assertTrue(metrics["density_metrics_available"])
        self.assertTrue(metrics["candidate_density_assumed"])
        self.assertEqual(metrics["mass_relative_l2"], 0)
        observed_metrics = physical_metrics(pair.reference, pair.observed)
        self.assertFalse(observed_metrics["density_metrics_available"])
        self.assertFalse(observed_metrics["candidate_density_assumed"])
        self.assertIsNone(observed_metrics["mass_relative_l2"])

    def test_missing_or_inconsistent_declarations_are_rejected(self):
        for changes in ({"times": [0, 0, 1]}, {"dx": 0}, {"units": ""},
                        {"boundary": "unknown"}, {"periodic_axes": ["z"]},
                        {"boundary": "periodic", "periodic_axes": ["x"]}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                model_run_to_research(self.run, **{**self.declarations, **changes})
        with self.assertRaisesRegex(ValueError, "density units"):
            model_run_to_research(self.run, density=1,
                                 **{**self.declarations,
                                    "units": {"length": "m", "time": "s", "velocity": "m/s"}})
        with self.assertRaises(ValueError):
            model_run_to_research(self.run, density=-1, **self.declarations)

    def test_pair_rejects_time_misalignment_before_conversion(self):
        shifted = ModelRun(self.run.run_dir, "noisy", self.run.schema_version, "shifted", {},
                           self.run.steps + 1, self.run.mask, self.run.arrays, {})
        with self.assertRaisesRegex(ValueError, "steps must agree"):
            model_pair_to_research(ModelRunPair(shifted, self.run), **self.declarations)


if __name__ == "__main__":
    unittest.main()
