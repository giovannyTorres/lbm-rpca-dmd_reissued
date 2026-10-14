"""Independent analytical and refinement checks for non-Newtonian references."""
import unittest
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python" / "src"))

import numpy as np

from fluid_denoise.research.rheology import (
    channel_convergence_study,
    herschel_bulkley_channel,
    herschel_bulkley_profile,
    newtonian_startup_profile,
    oldroyd_b_channel,
    reference_rheology_bank,
)


class HerschelBulkleyReferenceTests(unittest.TestCase):
    def test_newtonian_limit_and_signed_pressure(self):
        y = np.linspace(-2, 2, 33)
        for g in (1.5, -1.5, 0):
            actual = herschel_bulkley_profile(y, half_height=2, pressure_gradient=g,
                                             consistency=.7, flow_index=1, yield_stress=0)
            expected = g / (2 * .7) * (4 - y * y)
            np.testing.assert_allclose(actual, expected, atol=1e-14)

    def test_bingham_plug_and_yielded_region(self):
        y = np.array([-1, -.7, -.5, -.25, 0, .25, .5, .7, 1])
        actual = herschel_bulkley_profile(y, pressure_gradient=2, consistency=.5,
                                         flow_index=1, yield_stress=1)
        # Plug edge is H_p=tau_y/G=.5; above it the Bingham integral is quadratic.
        expected = ((2 * 1 - 1) ** 2 - np.maximum(2 * np.abs(y) - 1, 0) ** 2) / 2
        np.testing.assert_allclose(actual, expected)
        self.assertEqual(actual[4], actual[3])
        self.assertEqual(actual[4], actual[5])
        stopped = herschel_bulkley_profile(y, pressure_gradient=.9, yield_stress=1)
        np.testing.assert_array_equal(stopped, 0)

    def test_flow_rate_and_constitutive_law_for_all_indices(self):
        for n in (.6, 1., 1.4):
            data = herschel_bulkley_channel(ny=1001, nx=2, flow_index=n,
                                           pressure_gradient=1.7, consistency=.8,
                                           yield_stress=.3, times=[0, 1])
            y = np.array(data["metadata"]["y"])
            # Use trapezoidal quadrature as an independent observable check.
            u = data["velocity"][0, 0, :, 0]
            q = float(np.sum((u[:-1] + u[1:]) * .5 * data["dy"]))
            exact = data["metadata"]["analytical_flow_rate_per_width"]
            self.assertLess(abs(q - exact) / abs(exact), 3e-6)
            self.assertEqual(data["velocity"].shape, (2, 2, 1001, 2))
            self.assertFalse(data["mask"].any())
            np.testing.assert_allclose(data["stress_xy"][0, :, 0], -1.7 * y)
            self.assertLess(data["metadata"]["constitutive_residual_max_pa"], 1e-12)
            self.assertEqual(data["boundary"], "channel")

    def test_invalid_inputs_fail(self):
        for kwargs in ({"flow_index": 0}, {"consistency": -1}, {"yield_stress": -1},
                       {"ny": 2}, {"nx": 1}, {"times": [1, 0]}, {"pressure_gradient": np.nan}):
            with self.assertRaises(ValueError):
                herschel_bulkley_channel(**kwargs)


class OldroydBReferenceTests(unittest.TestCase):
    def test_steady_poiseuille_velocity_and_normal_stress(self):
        data = oldroyd_b_channel(ny=33, nx=3, eta_s=.3, eta_p=.7,
                                relaxation_time=.6, steps=100, initial="steady")
        y = np.array(data["metadata"]["y"])
        expected_u = .5 * (1 - y * y)
        np.testing.assert_allclose(data["velocity"][:, 0, :, 0],
                                   np.tile(expected_u, (len(data["times"]), 1)), atol=1e-13)
        face_y = .5 * (y[:-1] + y[1:])
        expected_tau_xy = -.7 * face_y
        expected_tau_xx = 2 * .6 * .7 * face_y ** 2
        np.testing.assert_allclose(data["stress_xy_faces"][-1], expected_tau_xy, atol=1e-12)
        np.testing.assert_allclose(data["stress_xx_faces"][-1], expected_tau_xx, atol=1e-12)
        np.testing.assert_allclose(data["stress_xy"][-1, :, 0], -y, atol=1e-12)
        self.assertLess(data["metadata"]["momentum_residual_max_pa_per_m"], 1e-10)

    def test_newtonian_startup_matches_independent_fourier_solution(self):
        data = oldroyd_b_channel(ny=65, nx=2, relaxation_time=0,
                                dt=.0005, steps=400, snapshot_stride=400)
        y = np.array(data["metadata"]["y"])
        exact = newtonian_startup_profile(y, .2)
        error = np.linalg.norm(data["velocity"][-1, 0, :, 0] - exact) / np.linalg.norm(exact)
        self.assertLess(error, .001)
        np.testing.assert_array_equal(newtonian_startup_profile(y, 0), 0)

    def test_zero_polymer_is_newtonian_for_any_relaxation_time(self):
        parameters = dict(ny=33, nx=2, eta_s=.8, eta_p=0,
                          dt=.002, steps=100, snapshot_stride=20)
        a = oldroyd_b_channel(relaxation_time=.9, **parameters)
        b = oldroyd_b_channel(relaxation_time=0, **parameters)
        np.testing.assert_allclose(a["velocity"], b["velocity"], atol=1e-14)
        np.testing.assert_array_equal(a["polymer_stress"], 0)

    def test_elastic_startup_has_memory_and_relaxes_to_exact_steady_state(self):
        data = oldroyd_b_channel(ny=33, nx=2, dt=.005, steps=1600, snapshot_stride=100)
        y = np.array(data["metadata"]["y"])
        np.testing.assert_allclose(data["velocity"][-1, 0, :, 0], .5 * (1-y*y), atol=2e-6)
        gradient = np.diff(data["velocity"][-1, 0, :, 0]) / data["dy"]
        np.testing.assert_allclose(data["stress_xy_faces"][-1], .5 * gradient, atol=1e-6)
        np.testing.assert_allclose(data["stress_xx_faces"][-1], .5 * gradient ** 2, atol=4e-6)
        early_gradient = np.diff(data["velocity"][1, 0, :, 0]) / data["dy"]
        self.assertGreater(np.max(np.abs(data["stress_xy_faces"][1] - .5 * early_gradient)), .02)
        self.assertGreaterEqual(data["metadata"]["min_conformation_determinant_faces"], 1-1e-12)
        self.assertLess(data["metadata"]["constitutive_xy_residual_max_pa"], 1e-10)
        self.assertLess(data["metadata"]["constitutive_xx_residual_max_pa"], 1e-10)

    def test_spatial_and_temporal_refinement_have_decreasing_changes(self):
        for mode in ("space", "time"):
            report = channel_convergence_study(refinement=mode, end_time=.5, levels=3)
            coarse, medium, fine = report["levels"]
            self.assertGreater(coarse["relative_l2_change_vs_finest"],
                               2 * medium["relative_l2_change_vs_finest"])
            self.assertNotIn("relative_l2_change_vs_finest", fine)
        report = channel_convergence_study(relaxation_time=0, end_time=.2, levels=3)
        errors = [row["relative_l2_error_vs_analytical"] for row in report["levels"]]
        self.assertGreater(errors[0], 1.8 * errors[1])
        self.assertGreater(errors[1], 1.8 * errors[2])

    def test_oscillatory_forcing_and_final_snapshot_contract(self):
        data = oldroyd_b_channel(ny=17, nx=2, steps=31, snapshot_stride=7,
                                pressure_gradient=lambda t: np.sin(2*np.pi*t))
        self.assertEqual(data["times"][-1], .31)
        self.assertEqual(len(data["times"]), 6)
        self.assertGreaterEqual(data["metadata"]["min_conformation_determinant_faces"], 1-1e-12)
        np.testing.assert_array_equal(data["velocity"][:, :, [0, -1]], 0)
        np.testing.assert_array_equal(data["velocity"][:, 1], 0)

    def test_invalid_parameters_fail(self):
        for kwargs in ({"dt": 0}, {"steps": 0}, {"eta_s": 0, "eta_p": 0},
                       {"relaxation_time": -1}, {"initial": "wake"},
                       {"pressure_gradient": lambda _: np.inf}, {"snapshot_stride": 0}):
            with self.assertRaises(ValueError):
                oldroyd_b_channel(**kwargs)

    def test_finite_inputs_that_overflow_stress_are_rejected(self):
        # A finite velocity must not disguise nonfinite normal stress/residuals.
        with self.assertRaises(FloatingPointError):
            oldroyd_b_channel(ny=5, nx=3, steps=2, initial="steady",
                              pressure_gradient=1e152, relaxation_time=1e8)

    def test_callable_forcing_refinement_does_not_use_constant_force_series(self):
        report = channel_convergence_study(
            relaxation_time=0, coarse_ny=9, coarse_dt=.02, levels=2,
            end_time=.1, pressure_gradient=lambda t: np.sin(t),
        )
        self.assertNotIn("relative_l2_error_vs_analytical", report["levels"][0])

    def test_bank_distinguishes_exact_stationary_and_developing_flows(self):
        bank = reference_rheology_bank(ny=17, nx=2, frames=7)
        self.assertEqual(len(bank), 8)
        for data in bank.values():
            self.assertEqual(data["velocity"].shape, (7, 2, 17, 2))
        np.testing.assert_array_equal(bank["hb_unyielded"]["velocity"], 0)
        self.assertGreater(np.max(bank["oldroyd_b_startup_high_wi"]["stress_xx_faces"]), 0)


if __name__ == "__main__":
    unittest.main()
