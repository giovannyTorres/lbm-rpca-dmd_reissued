"""Analytic conservation and accounting checks for the research extension."""
import json
import unittest
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python" / "src"))

import numpy as np

from fluid_denoise.research.cost import (measure_call, operation_proxies, pareto_front,
                                    quality_gates, summarize_profiles)
from fluid_denoise.research.metrics import physical_metrics, spatial_derivative


def taylor_green(n=32):
    axis = np.arange(n) * (2 * np.pi / n)
    x, y = np.meshgrid(axis, axis)
    times = np.linspace(0, 0.3, 4)
    amplitude = np.exp(-0.2 * times)
    u = amplitude[:, None, None] * np.sin(x) * np.cos(y)
    v = -amplitude[:, None, None] * np.cos(x) * np.sin(y)
    velocity = np.stack([u, v], axis=1)
    return {"velocity": velocity, "times": times, "dx": 2 * np.pi / n,
            "dy": 2 * np.pi / n, "boundary": "periodic", "density": 1.0}, x, y


class PhysicalMetricsTests(unittest.TestCase):
    def test_taylor_green_exact_spectral_derivatives(self):
        data, x, y = taylor_green()
        u, v = data["velocity"][0]
        divergence = spatial_derivative(u, data["dx"], -1, True) + spatial_derivative(v, data["dy"], -2, True)
        vorticity = spatial_derivative(v, data["dx"], -1, True) - spatial_derivative(u, data["dy"], -2, True)
        np.testing.assert_allclose(divergence, 0, atol=1e-14)
        np.testing.assert_allclose(vorticity, 2 * np.sin(x) * np.sin(y), atol=1e-14)
        metrics = physical_metrics(data, data["velocity"])
        self.assertLess(metrics["divergence_rms"], 1e-14)
        self.assertEqual(metrics["velocity_relative_l2"], 0.0)
        self.assertEqual(metrics["vorticity_relative_l2"], 0.0)
        self.assertEqual(metrics["vorticity_correlation"], 1.0)
        self.assertAlmostEqual(metrics["series"]["reference_kinetic_energy"][0], np.pi**2)
        self.assertAlmostEqual(metrics["series"]["reference_enstrophy"][0], 2 * np.pi**2)
        self.assertAlmostEqual(metrics["series"]["reference_mass"][0], 4 * np.pi**2)
        self.assertLess(metrics["mass_balance_rms"], 1e-12)
        json.dumps(metrics, allow_nan=False)

    def test_irrotational_artifact_is_visible_in_divergence(self):
        data, x, _ = taylor_green()
        corrupt = data["velocity"].copy()
        corrupt[:, 0] += 0.2 * np.sin(x)
        metrics = physical_metrics(data, corrupt)
        self.assertAlmostEqual(metrics["extra_divergence_rms"], 0.2 / np.sqrt(2))
        self.assertLess(metrics["vorticity_rmse"], 1e-14)
        self.assertGreater(metrics["velocity_relative_l2"], 0)
        self.assertTrue(metrics["constant_density_mass_trivial"])
        self.assertEqual(metrics["mass_relative_l2"], 0)
        self.assertAlmostEqual(metrics["extra_continuity_rms"], 0.2 / np.sqrt(2))
        self.assertGreater(metrics["extra_divergence_normalized"], 0)

    def test_conservation_can_pass_while_energy_is_wrong(self):
        data, _, _ = taylor_green()
        metrics = physical_metrics(data, 0.9 * data["velocity"])
        self.assertAlmostEqual(metrics["velocity_relative_l2"], 0.1)
        self.assertAlmostEqual(metrics["vorticity_relative_l2"], 0.1)
        self.assertAlmostEqual(metrics["kinetic_energy_relative_l2"], 0.19)
        self.assertAlmostEqual(metrics["enstrophy_relative_l2"], 0.19)
        self.assertLess(metrics["extra_divergence_normalized"], 1e-14)
        self.assertAlmostEqual(metrics["vorticity_correlation"], 1)
        self.assertFalse(quality_gates(metrics, {"kinetic_energy_relative_l2": 0.15})["passed"])

    def test_uniform_hb_slip_fails_wall_gate_despite_passing_bulk_gates(self):
        from fluid_denoise.research.rheology import herschel_bulkley_channel
        data = herschel_bulkley_channel(ny=33, nx=8, times=np.array([0.0, 0.05, 0.1]))
        bulk_gates = {"velocity_relative_l2": 0.15, "vorticity_relative_l2": 0.25,
                      "extra_divergence_normalized": 0.1, "kinetic_energy_relative_l2": 0.15}
        clean = physical_metrics(data, data["velocity"])
        self.assertTrue(clean["wall_gate_applicable"])
        self.assertEqual(clean["wall_velocity_relative_rms"], 0)
        self.assertTrue(quality_gates(clean, {**bulk_gates, "wall_velocity_relative_rms": 0.02})["passed"])
        slipped = data["velocity"].copy()
        slipped[:, 0] += 0.005
        metrics = physical_metrics(data, slipped)
        self.assertTrue(quality_gates(metrics, bulk_gates)["passed"])
        self.assertGreater(metrics["wall_velocity_relative_rms"], 0.02)
        self.assertFalse(quality_gates(metrics, {**bulk_gates, "wall_velocity_relative_rms": 0.02})["passed"])

    def test_uniform_wall_normal_penetration_cancels_signed_flux(self):
        from fluid_denoise.research.rheology import herschel_bulkley_channel
        data = herschel_bulkley_channel(ny=33, nx=8, times=np.array([0.0, 0.05, 0.1]))
        penetrating = data["velocity"].copy()
        penetrating[:, 1] += 0.005
        metrics = physical_metrics(data, penetrating)
        self.assertAlmostEqual(metrics["boundary_flux_rms"], 0)
        self.assertAlmostEqual(metrics["extra_divergence_rms"], 0)
        self.assertAlmostEqual(metrics["wall_normal_velocity_rms"], 0.005)
        self.assertGreater(metrics["wall_normal_velocity_relative_rms"], 0.02)
        self.assertFalse(quality_gates(metrics, {"wall_velocity_relative_rms": 0.02})["passed"])

    def test_moving_wall_error_and_wall_gate_applicability(self):
        data, _, _ = taylor_green()
        periodic = physical_metrics(data, data["velocity"])
        self.assertFalse(periodic["wall_gate_applicable"])
        self.assertIsNone(periodic["wall_velocity_relative_rms"])
        velocity = np.zeros((3, 2, 9, 8))
        velocity[:, 0] = np.linspace(0, 1, 9)[None, :, None]
        moving = {"velocity": velocity, "times": [0, 1, 2], "dx": 1 / 8,
                  "dy": 1 / 8, "boundary": "channel", "density": 1.0}
        metrics = physical_metrics(moving, velocity)
        self.assertGreater(metrics["wall_velocity_relative_rms"], 0)
        self.assertEqual(metrics["wall_velocity_error_relative_rms"], 0)
        self.assertEqual(metrics["wall_normal_velocity_relative_rms"], 0)
        self.assertTrue(quality_gates(metrics, {"wall_velocity_error_relative_rms": 0.02})["passed"])

    def test_channel_nonperiodic_y_flux_and_quadrature(self):
        nx, ny, length, height = 12, 9, 3.0, 2.0
        y = np.linspace(0, height, ny)
        velocity = np.zeros((3, 2, ny, nx))
        velocity[:, 0] = (1 - (y - 1)**2)[None, :, None]
        data = {"velocity": velocity, "times": np.arange(3.0), "dx": length / nx,
                "dy": height / (ny - 1), "boundary": "channel", "density": 2.0}
        cand = velocity.copy()
        cand[:, 1] = 0.3 * y[None, :, None]
        metrics = physical_metrics(data, cand)
        self.assertEqual(metrics["periodic_axes"], ["x"])
        self.assertAlmostEqual(metrics["fluid_area"], length * height)
        self.assertAlmostEqual(metrics["divergence_rms"], 0.3)
        self.assertAlmostEqual(metrics["boundary_flux_rms"], 2 * 0.3 * height * length)
        self.assertAlmostEqual(metrics["mass_balance_rms"], 2 * 0.3 * height * length)
        self.assertAlmostEqual(metrics["continuity_rms"], 2 * 0.3)
        self.assertLess(metrics["vorticity_rmse"], 1e-14)
        self.assertEqual(metrics["derivative_valid_nodes"], (ny - 2) * nx)

    def test_open_linear_flow_closes_divergence_theorem(self):
        nx, ny, lx, ly = 11, 8, 2.0, 1.0
        x = np.linspace(0, lx, nx)
        velocity = np.zeros((3, 2, ny, nx))
        data = {"velocity": velocity, "times": [0, 1, 2], "dx": lx / (nx - 1),
                "dy": ly / (ny - 1), "boundary": "open", "density": 1.0}
        cand = velocity.copy()
        cand[:, 0] = x[None, None, :]
        metrics = physical_metrics(data, cand)
        self.assertAlmostEqual(metrics["divergence_rms"], 1)
        self.assertAlmostEqual(metrics["boundary_flux_rms"], lx * ly)
        self.assertAlmostEqual(metrics["fluid_area"], lx * ly)
        self.assertIsNone(metrics["velocity_relative_l2"])
        json.dumps(metrics, allow_nan=False)

    def test_varying_density_balances_outward_flux(self):
        nx, ny = 7, 6
        times = np.array([0.0, 0.2, 0.4])
        density = 2.0 - times[:, None, None] + np.zeros((3, ny, nx))
        y = np.linspace(0, 1, ny)
        velocity = np.zeros((3, 2, ny, nx))
        velocity[:, 1] = y[None, :, None] / density
        data = {"velocity": velocity, "times": times, "dx": 2 / nx, "dy": 1 / (ny - 1),
                "boundary": "channel", "density": density}
        metrics = physical_metrics(data, velocity, density)
        self.assertLess(metrics["continuity_rms"], 1e-14)
        self.assertLess(metrics["mass_balance_rms"], 1e-14)
        self.assertFalse(metrics["constant_density_mass_trivial"])
        self.assertFalse(metrics["candidate_density_assumed"])

    def test_mask_halo_does_not_use_solid_values(self):
        data, _, _ = taylor_green(24)
        mask = np.zeros((24, 24), dtype=bool)
        mask[10:13, 10:13] = True
        data["mask"] = mask
        corrupt = data["velocity"].copy()
        corrupt[:, :, mask] = np.nan
        metrics = physical_metrics(data, corrupt)
        self.assertEqual(metrics["velocity_relative_l2"], 0)
        self.assertEqual(metrics["extra_divergence_rms"], 0)
        self.assertLess(metrics["derivative_valid_nodes"], np.sum(~mask))
        self.assertEqual(metrics["derivative_method"], "finite_difference_mask_halo")

    def test_shape_time_spacing_and_json_units(self):
        data, _, _ = taylor_green()
        with self.assertRaises(ValueError):
            physical_metrics(data, data["velocity"][:, :, :-1])
        with self.assertRaises(ValueError):
            physical_metrics({**data, "dx": 0}, data["velocity"])
        with self.assertRaises(ValueError):
            physical_metrics({**data, "times": [0, 0, 1, 2]}, data["velocity"])
        with self.assertRaises(ValueError):
            physical_metrics(data, data["velocity"], -1)
        one = {**data, "times": data["times"][:1], "velocity": data["velocity"][:1]}
        metrics = physical_metrics(one, one["velocity"])
        self.assertIsNone(metrics["continuity_rms"])
        self.assertEqual(metrics["units"]["vorticity_rmse"], "1/time")
        json.dumps(metrics, allow_nan=False)


class CostTests(unittest.TestCase):
    def test_measurement_and_robust_summary(self):
        value, profile = measure_call(lambda: np.arange(128).sum())
        self.assertEqual(value, sum(range(128)))
        self.assertGreaterEqual(profile["rss_peak_bytes"], profile["rss_peak_before_bytes"])
        self.assertGreaterEqual(profile["wall_seconds"], 0)
        self.assertGreater(profile["python_traced_peak_bytes"], 0)
        profiles = [{"wall_seconds": float(i), "cpu_seconds": 2.0 * i,
                     "rss_peak_bytes": 100.0 * i, "python_traced_peak_bytes": 10.0 * i}
                    for i in [1, 2, 3, 4, 100]]
        summary = summarize_profiles(profiles)
        self.assertEqual(summary["wall_seconds_median"], 3)
        self.assertEqual(summary["wall_seconds_iqr"], 2)
        self.assertEqual(summary["rss_peak_bytes_median"], 300)
        json.dumps(summary, allow_nan=False)

    def test_pareto_front_and_gates(self):
        records = [{"wall_seconds_median": time, "rss_peak_bytes_median": mem, "velocity_relative_l2": error}
                   for time, mem, error in [(1, 10, .1), (2, 20, .2), (3, 5, .02), (1, 10, .1)]]
        records.append({"wall_seconds_median": 1, "rss_peak_bytes_median": 1, "velocity_relative_l2": None})
        self.assertEqual(pareto_front(records), [0, 2, 3])
        self.assertTrue(quality_gates({"velocity_relative_l2": .05}, {"velocity_relative_l2": .1})["passed"])
        self.assertFalse(quality_gates({}, {"velocity_relative_l2": .1})["passed"])
        self.assertFalse(quality_gates({}, {})["passed"])

    def test_operation_proxies_are_explicit_diagnostics(self):
        proxy = operation_proxies((10, 4), svd_calls=3, fft_shapes=[(8, 8)])
        self.assertEqual(proxy["svd_full_thin_work_proxy"], 480)
        self.assertEqual(proxy["fft_nlog2n_work_proxy"], 384)
        with self.assertRaises(ValueError):
            operation_proxies((10, 0))


if __name__ == "__main__":
    unittest.main()
