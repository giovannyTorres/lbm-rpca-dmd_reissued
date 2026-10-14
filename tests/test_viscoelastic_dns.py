from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

from fluid_denoise.research.dns import periodic_diagnostics, simulate_periodic_ns
from fluid_denoise.research.viscoelastic_dns import (
    integrate_uniform_shear,
    simulate_viscoelastic_periodic,
    uniform_shear_reference,
    validation_report,
)


class ViscoelasticDNSTest(unittest.TestCase):
    def test_coupled_periodic_fields_are_finite_admissible_and_incompressible(self):
        for model in ("oldroyd_b", "fene_p"):
            with self.subTest(model=model):
                run = simulate_viscoelastic_periodic(model=model, nx=16, dt=.01,
                                                      n_steps=10, sample_every=2, amplitude=.3)
                self.assertEqual(run["velocity"].shape, (6, 2, 16, 16))
                self.assertEqual(run["conformation"].shape, (6, 3, 3, 16, 16))
                self.assertEqual(run["stress"].shape, run["conformation"].shape)
                self.assertEqual(run["constitutive_rhs"].shape, run["conformation"].shape)
                self.assertEqual(run["mask"].dtype, np.dtype(bool))
                self.assertFalse(run["mask"].any())
                self.assertEqual(run["boundary"], "periodic")
                for name in ("velocity", "conformation", "stress", "constitutive_rhs"):
                    self.assertTrue(np.all(np.isfinite(run[name])))
                eigenvalues = np.linalg.eigvalsh(np.moveaxis(run["conformation"], (1, 2), (-2, -1)))
                self.assertGreater(eigenvalues.min(), 0.)
                if model == "fene_p":
                    self.assertLess(np.trace(run["conformation"], axis1=1, axis2=2).max(), 50.)
                divergence = periodic_diagnostics(run["velocity"], run["dx"], run["dy"])["divergence_rms"]
                self.assertLess(max(divergence), 1e-12)
                self.assertIsNotNone(run["metadata"]["constitutive_residual_rms"])
                self.assertIsNotNone(run["metadata"]["constitutive_pde_residual_rms"])
                self.assertEqual(len(run["metadata"]["constitutive_spatial_projection_defect_rms"]), 6)

    def test_zero_polymer_matches_independent_vorticity_solver(self):
        settings = dict(nx=24, case="interacting_vortices", amplitude=.4, dt=.005,
                        n_steps=20, sample_every=5)
        polymer = simulate_viscoelastic_periodic(viscosity_s=.02, viscosity_p=0., **settings)
        newtonian = simulate_periodic_ns(viscosity=.02, **settings)
        np.testing.assert_allclose(polymer["velocity"], newtonian["velocity"], atol=2e-14, rtol=2e-13)
        np.testing.assert_array_equal(polymer["stress"], 0.)

    def test_polymer_memory_changes_velocity_and_transports_conformation(self):
        options = dict(nx=24, case="interacting_vortices", amplitude=.5, dt=.005,
                       n_steps=40, sample_every=10, viscosity_s=.02)
        coupled = simulate_viscoelastic_periodic(viscosity_p=.08, relaxation_time=.4, **options)
        uncoupled = simulate_viscoelastic_periodic(viscosity_p=0., relaxation_time=.4, **options)
        self.assertGreater(np.linalg.norm(coupled["velocity"][-1] - uncoupled["velocity"][-1]), 1e-3)
        self.assertGreater(np.linalg.norm(coupled["conformation"][-1] - coupled["conformation"][0]), .1)
        # Normal stresses and off-diagonal shear stress both emerge from C=I.
        self.assertGreater(np.max(np.abs(coupled["stress"][-1, 0, 1])), 1e-3)
        self.assertGreater(np.max(np.abs(coupled["stress"][-1, 0, 0])), 1e-3)

    def test_uniform_shear_source_ode_has_fourth_order_accuracy(self):
        errors = []
        for dt in (.04, .02, .01):
            numerical = integrate_uniform_shear(dt=dt, n_steps=round(.4 / dt),
                                                shear_rate=1.4, relaxation_time=.3)
            reference = uniform_shear_reference(numerical["times"], shear_rate=1.4, relaxation_time=.3)
            errors.append(float(np.linalg.norm(numerical["conformation"][-1] - reference[-1])))
        self.assertGreater(errors[0] / errors[1], 12.)
        self.assertGreater(errors[1] / errors[2], 12.)
        self.assertLess(errors[-1], 1e-7)

    def test_fene_equilibrium_and_finite_extension_are_actual_tensor_model(self):
        equilibrium = integrate_uniform_shear(model="fene_p", shear_rate=0., dt=.01,
                                              n_steps=20, extensibility_squared=8.)
        np.testing.assert_array_equal(equilibrium["conformation"], np.broadcast_to(np.eye(3), (21, 3, 3)))
        fene = integrate_uniform_shear(model="fene_p", shear_rate=1., relaxation_time=.5,
                                      dt=.005, n_steps=100, extensibility_squared=8.)
        self.assertLess(np.trace(fene["conformation"], axis1=1, axis2=2).max(), 8.)
        self.assertGreater(np.linalg.eigvalsh(fene["conformation"]).min(), 0.)
        self.assertLess(fene["conformation"][-1, 2, 2], 1.)

    def test_invalid_constitutive_initial_state_is_rejected(self):
        for matrix in (np.diag([-1., 1., 1.]), np.diag([2., 2., 2.])):
            field = np.broadcast_to(matrix[:, :, None, None], (3, 3, 8, 8)).copy()
            with self.assertRaises(FloatingPointError):
                simulate_viscoelastic_periodic(model="fene_p", nx=8, n_steps=0,
                                               extensibility_squared=5., initial_conformation=field)
        with self.assertRaises(ValueError):
            simulate_viscoelastic_periodic(extensibility_squared=3.)

    def test_finite_extreme_input_cannot_export_nonfinite_energy_or_stress(self):
        with self.assertRaises(FloatingPointError):
            simulate_viscoelastic_periodic(nx=8, amplitude=1e200, n_steps=0)

    def test_trace_barrier_trial_is_rejected_without_clipping(self):
        # A deliberately nearly extended input exercises the rejection path.
        # This numerical admissibility test is not high-Wi flow validation.
        field = np.broadcast_to(np.diag([3.8, .1, .1])[:, :, None, None], (3, 3, 8, 8)).copy()
        run = simulate_viscoelastic_periodic(model="fene_p", nx=8, dt=.1, n_steps=1,
                                            sample_every=1, initial_conformation=field,
                                            extensibility_squared=4.01, relaxation_time=10.)
        self.assertGreater(run["metadata"]["rejected_steps"], 0)
        self.assertLess(run["metadata"]["stage_max_conformation_trace"], 4.01)
        self.assertGreater(run["metadata"]["stage_min_conformation_eigenvalue"], 0.)
        self.assertEqual(run["times"][-1], .1)
        np.testing.assert_allclose(run["conformation"][0], field, atol=1e-15)

    def test_small_relaxation_limit_moves_towards_total_newtonian_viscosity(self):
        options = dict(nx=16, amplitude=.1, dt=.0025, n_steps=40, sample_every=40)
        newtonian = simulate_periodic_ns(viscosity=.04, **options)
        for model in ("oldroyd_b", "fene_p"):
            with self.subTest(model=model):
                errors = []
                for lam in (.02, .01, .005):
                    run = simulate_viscoelastic_periodic(model=model, viscosity_s=.02, viscosity_p=.02,
                                                        relaxation_time=lam, **options)
                    errors.append(np.linalg.norm(run["velocity"][-1] - newtonian["velocity"][-1]))
                self.assertGreater(errors[0] / errors[1], 1.8)
                self.assertGreater(errors[1] / errors[2], 1.8)

    def test_polymer_exchange_and_total_free_energy_dissipate(self):
        for model in ("oldroyd_b", "fene_p"):
            run = simulate_viscoelastic_periodic(model=model, nx=24, amplitude=.5,
                                                dt=.01, n_steps=20, sample_every=2,
                                                viscosity_s=.02, viscosity_p=.04)
            d = run["metadata"]["diagnostics"]
            total = np.asarray(d["kinetic_energy"]) + d["polymer_free_energy"]
            self.assertLessEqual(np.max(np.diff(total)), 1e-11)
            self.assertLess(max(abs(v) for v in d["kinetic_exchange_residual"]), 1e-12)
            self.assertLess(max(abs(v) for v in d["total_energy_rate_residual"]), 1e-8)
            self.assertGreaterEqual(min(d["polymer_relaxation_dissipation"]), -1e-14)

    def test_constitutive_snapshot_residual_refines_in_time(self):
        errors = []
        for dt in (.01, .005):
            run = simulate_viscoelastic_periodic(nx=16, dt=dt, n_steps=round(.1 / dt),
                                                sample_every=1, amplitude=.3)
            errors.append(max(run["metadata"]["constitutive_residual_rms"][1:-1]))
        self.assertGreater(errors[0] / errors[1], 3.5)

    def test_forcing_rk_orders_and_optional_two_dimensional_tensor(self):
        for order in (2, 3, 4):
            run = simulate_viscoelastic_periodic(nx=12, rk_order=order, tensor_dimension=2,
                                                forcing="kolmogorov", forcing_amplitude=.1,
                                                dt=.005, n_steps=2, sample_every=1)
            self.assertEqual(run["conformation"].shape[1:3], (2, 2))
            self.assertTrue(np.all(np.isfinite(run["stress"])))
        with self.assertRaises(ValueError):
            simulate_viscoelastic_periodic(forcing_amplitude=1.)

    def test_explicit_stress_diffusion_is_reported_and_has_positive_dissipation(self):
        run = simulate_viscoelastic_periodic(nx=16, dt=.005, n_steps=10,
                                            sample_every=2, stress_diffusivity=.01)
        self.assertTrue(run["metadata"]["stress_diffusion_is_artificial"])
        self.assertEqual(run["metadata"]["stress_diffusivity"], .01)
        self.assertGreater(run["metadata"]["diagnostics"]["polymer_diffusion_dissipation"][-1], 0.)

    def test_report_is_serializable_and_contains_three_distinct_refinements(self):
        report = validation_report(grids=(8, 12, 16), reference_nx=24, nx=16,
                                   time_steps=(.02, .01, .005), reference_dt=.0025,
                                   duration=.04, stress_diffusivity=.001)
        json.dumps(report, allow_nan=False)
        self.assertEqual(len(report["grid_refinement"]), 3)
        self.assertEqual(len(report["temporal_refinement"]), 3)
        self.assertEqual(len(report["stress_diffusion_sensitivity"]), 3)
        self.assertFalse(report["dns_certified"])
        self.assertTrue(report["finite_fields"])


if __name__ == "__main__":
    unittest.main()
