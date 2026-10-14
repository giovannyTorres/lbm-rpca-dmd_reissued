"""Constitutive, momentum and energy checks for periodic viscoplastic flow."""
import json

import numpy as np
import pytest
from scipy.signal import resample

from fluid_denoise.research.dns import taylor_green_reference
from fluid_denoise.research.viscoplastic_dns import (
    apparent_viscosity, simulate_viscoplastic_periodic, validation_report,
)


def _strain(velocity, dx, dy):
    """Measure divergence and symmetric strain from resolved physical fields."""
    ny, nx = velocity.shape[-2:]
    kx, ky = np.meshgrid(2 * np.pi * np.fft.fftfreq(nx, dx),
                          2 * np.pi * np.fft.fftfreq(ny, dy))
    velocity_hat = np.fft.fft2(velocity, axes=(-2, -1))
    du_dx, dv_dx = np.moveaxis(np.fft.ifft2(1j * kx * velocity_hat,
                                         axes=(-2, -1)).real, -3, 0)
    du_dy, dv_dy = np.moveaxis(np.fft.ifft2(1j * ky * velocity_hat,
                                         axes=(-2, -1)).real, -3, 0)
    dxy = .5 * (du_dy + dv_dx)
    gamma = np.sqrt(2 * du_dx ** 2 + 2 * dv_dy ** 2 + 4 * dxy ** 2)
    return du_dx + dv_dy, gamma, np.stack((du_dx, dxy, dv_dy), axis=-3)


@pytest.mark.parametrize("flow_index", [.6, 1., 1.4])
def test_papanastasiou_is_continuous_at_zero_without_a_viscosity_cap(flow_index):
    k, epsilon, tau_y, m, solvent = .07, 1e-3, .13, 17., .011
    gamma = np.array([0., 1e-14, 1e-10, 1e-7])
    eta = apparent_viscosity(gamma, consistency=k, flow_index=flow_index,
                            yield_stress=tau_y, regularization_m=m,
                            shear_epsilon=epsilon, solvent_viscosity=solvent)
    zero_limit = solvent + k * epsilon ** (flow_index - 1) + tau_y * m
    assert eta[0] == pytest.approx(zero_limit, rel=2e-15)
    np.testing.assert_allclose(eta[1:3], zero_limit, rtol=1e-9, atol=0)
    # The Taylor expansion tests the uncapped continuous yield term without
    # subtracting two almost equal exponential values in the reference.
    yield_eta = tau_y * (m - .5 * m ** 2 * gamma + m ** 3 * gamma ** 2 / 6)
    expected = solvent + k * (gamma ** 2 + epsilon ** 2) ** ((flow_index - 1) / 2) + yield_eta
    np.testing.assert_allclose(eta, expected, rtol=3e-15, atol=0)
    assert eta[0] * gamma[0] == 0.  # Finite viscosity does not create stress at rest.
    assert apparent_viscosity(0., consistency=k, flow_index=flow_index,
                              yield_stress=tau_y, regularization_m=m,
                              shear_epsilon=epsilon, solvent_viscosity=solvent) == pytest.approx(zero_limit)


def test_newtonian_taylor_green_velocity_pressure_and_density_scaling():
    density, dynamic_viscosity = 2., .1 + .05
    run = simulate_viscoplastic_periodic(
        case="taylor_green", nx=24, ny=16, length_x=4 * np.pi,
        consistency=.1, solvent_viscosity=.05, density=density,
        yield_stress=0., flow_index=1., amplitude=.7,
        dt=.01, n_steps=40, sample_every=7,
    )
    exact = taylor_green_reference(
        nx=24, ny=16, length_x=4 * np.pi, times=run["times"],
        viscosity=dynamic_viscosity / density, amplitude=.7,
    )
    np.testing.assert_allclose(run["velocity"], exact["velocity"], atol=2e-12, rtol=0)
    np.testing.assert_allclose(run["pressure"], density * exact["pressure"], atol=2e-12, rtol=0)
    np.testing.assert_allclose(run["kinetic_energy"], density * exact["kinetic_energy_exact"], atol=2e-13)
    np.testing.assert_array_equal(run["times"], .01 * np.array([0, 7, 14, 21, 28, 35, 40]))
    np.testing.assert_array_equal(run["density"], density)
    np.testing.assert_allclose(run["apparent_viscosity"], dynamic_viscosity, atol=0, rtol=0)
    assert np.max(np.abs(run["energy_budget_residual"])) < 2e-12


def test_newtonian_periodic_shear_has_exact_decay_and_dissipation():
    density, eta, amplitude, ky = 1.7, .08 + .03, .9, .5
    run = simulate_viscoplastic_periodic(
        case="periodic_shear", nx=16, ny=24, length_y=4 * np.pi,
        consistency=.08, solvent_viscosity=.03, density=density,
        yield_stress=0., flow_index=1., amplitude=amplitude,
        dt=.02, n_steps=25, sample_every=6,
    )
    decay = np.exp(-eta / density * ky ** 2 * run["times"])
    exact_velocity = decay[:, None, None, None] * run["velocity"][0]
    np.testing.assert_allclose(run["velocity"], exact_velocity, atol=2e-12, rtol=0)
    np.testing.assert_allclose(run["pressure"], 0., atol=2e-14, rtol=0)
    np.testing.assert_allclose(run["kinetic_energy"], density * amplitude ** 2 / 4 * decay ** 2, atol=2e-13)
    np.testing.assert_allclose(run["dissipation_rate"], eta * amplitude ** 2 * ky ** 2 / 2 * decay ** 2, atol=2e-13)


@pytest.mark.parametrize("flow_index", [.6, 1., 1.4])
def test_nonlinear_shear_obeys_full_variable_viscosity_momentum(flow_index):
    # With u=A*sin(ky*y), v=0, advection and pressure gradients vanish.
    # d(eta(gamma)*u_y)/dy = (eta + gamma*eta')*u_yy. This independent
    # analytical tangent catches replacing div(2*eta*D) by eta*laplacian(u).
    ny, dense_ny, ky = 32, 4096, .5
    amplitude, density, epsilon = .8, 1.3, .02
    k, tau_y, m, solvent = .035, .03, 5., .007
    phase = 2 * np.pi * np.arange(dense_ny) / dense_ny
    gamma = np.abs(amplitude * ky * np.cos(phase))
    tangent = (solvent
               + k * (gamma ** 2 + epsilon ** 2) ** ((flow_index - 3) / 2)
               * (epsilon ** 2 + flow_index * gamma ** 2)
               + tau_y * m * np.exp(-m * gamma))
    exact_force = -amplitude * ky ** 2 * np.sin(phase) * tangent / density
    coefficients = np.fft.fft(exact_force) / dense_ny
    modes = np.rint(np.fft.fftfreq(ny) * ny).astype(int)
    retained = np.where(np.abs(modes) < ny / 3, coefficients[modes % dense_ny], 0.)
    expected_galerkin_force = np.fft.ifft(ny * retained).real
    h = 1e-6
    run = simulate_viscoplastic_periodic(
        case="periodic_shear", nx=8, ny=ny, length_y=4 * np.pi,
        consistency=k, flow_index=flow_index, yield_stress=tau_y,
        regularization_m=m, shear_epsilon=epsilon,
        solvent_viscosity=solvent, density=density, amplitude=amplitude,
        quadrature_factor=8., dt=h, n_steps=1, sample_every=1,
    )
    measured_force = (run["velocity"][1, 0, :, 0] - run["velocity"][0, 0, :, 0]) / h
    np.testing.assert_allclose(measured_force, expected_galerkin_force, atol=3e-7, rtol=1e-5)
    np.testing.assert_allclose(run["velocity"][:, 1], 0., atol=2e-14, rtol=0)
    np.testing.assert_allclose(run["pressure"], 0., atol=2e-14, rtol=0)
    # Spatially varying rheology generates an initially absent third harmonic.
    final_spectrum = np.fft.fft(run["velocity"][-1, 0, :, 0]) / ny
    assert abs(final_spectrum[3]) > 1e-9


@pytest.fixture(scope="module")
def nonlinear_runs():
    settings = dict(
        case="nonlinear_modes", nx=24, consistency=.035,
        regularization_m=10., shear_epsilon=1e-3, solvent_viscosity=.005,
        density=1.7, dt=.004, n_steps=25, sample_every=7, amplitude=.6,
    )
    return {
        "newtonian": simulate_viscoplastic_periodic(**settings, flow_index=1., yield_stress=0.),
        **{n: simulate_viscoplastic_periodic(**settings, flow_index=n, yield_stress=.04)
           for n in (.6, 1., 1.4)},
    }


@pytest.mark.parametrize("flow_index", [.6, 1., 1.4])
def test_nonlinear_snapshots_resolve_constitutive_fields_and_are_not_rescaled_newtonian(flow_index, nonlinear_runs):
    run = nonlinear_runs[flow_index]
    assert run["velocity"].shape == (5, 2, 24, 24)
    assert run["deviatoric_stress"].shape == (5, 3, 24, 24)
    assert np.all(np.isfinite(run["velocity"]))
    assert np.ptp(run["apparent_viscosity"][-1]) > .1
    divergence, gamma, strain = _strain(run["velocity"], run["dx"], run["dy"])
    assert np.sqrt(np.mean(divergence ** 2)) < 2e-14
    np.testing.assert_allclose(run["shear_rate"], gamma, atol=3e-14, rtol=1e-13)
    np.testing.assert_allclose(run["deviatoric_stress"],
                               2 * run["apparent_viscosity"][:, None] * strain, atol=2e-14, rtol=2e-13)
    stress = run["deviatoric_stress"]
    measured_invariant = np.sqrt((stress[:, 0] ** 2 + 2 * stress[:, 1] ** 2 + stress[:, 2] ** 2) / 2)
    np.testing.assert_allclose(run["stress_invariant"], measured_invariant, atol=2e-14)
    np.testing.assert_array_equal(run["approximate_yielded"], run["shear_rate"] > 1e-3)
    np.testing.assert_array_equal(run["over_yield_stress"], run["stress_invariant"] > .04)
    initial, final = run["velocity"][[0, -1]]
    for reference in (initial, nonlinear_runs["newtonian"]["velocity"][-1]):
        best_scalar = np.sum(final * reference) / np.sum(reference ** 2)
        residual = np.linalg.norm(final - best_scalar * reference) / np.linalg.norm(final)
        assert residual > .003
    assert run["metadata"]["viscosity_cap"] is None
    json.dumps(run["metadata"], allow_nan=False)


@pytest.mark.parametrize("flow_index", [.6, 1., 1.4])
def test_nonlinear_energy_loss_matches_nonnegative_padded_grid_dissipation(flow_index, nonlinear_runs):
    run = nonlinear_runs[flow_index]
    metadata = run["metadata"]
    density = metadata["constant_density"]
    measured_energy = .5 * density * np.mean(np.sum(run["velocity"] ** 2, axis=1), axis=(-2, -1))
    np.testing.assert_allclose(run["kinetic_energy"], measured_energy, atol=1e-15)
    py, px = metadata["quadrature_shape"]
    padded = resample(resample(run["velocity"], py, axis=-2), px, axis=-1)
    _, gamma, _ = _strain(padded, metadata["domain"][0] / px, metadata["domain"][1] / py)
    yield_eta = np.full_like(gamma, metadata["yield_stress"] * metadata["regularization_m"])
    positive = gamma > 0
    yield_eta[positive] = (metadata["yield_stress"]
                           * -np.expm1(-metadata["regularization_m"] * gamma[positive]) / gamma[positive])
    eta = (metadata["solvent_viscosity"] + metadata["consistency"]
           * (gamma ** 2 + metadata["shear_epsilon"] ** 2) ** ((flow_index - 1) / 2) + yield_eta)
    measured_dissipation = np.mean(eta * gamma ** 2, axis=(-2, -1))
    np.testing.assert_allclose(run["dissipation_rate"], measured_dissipation, rtol=3e-14, atol=1e-15)
    assert np.all(run["dissipation_rate"] >= 0)
    assert np.all(np.diff(run["kinetic_energy"]) < 0)
    assert np.all(np.diff(run["cumulative_dissipation"]) > 0)
    np.testing.assert_allclose(run["energy_budget_residual"],
                               measured_energy - measured_energy[0] + run["cumulative_dissipation"], atol=1e-15)
    assert np.max(np.abs(run["energy_budget_residual"])) / measured_energy[0] < 1e-9


def test_uniform_mean_velocity_has_no_stress_or_energy_loss():
    initial = np.broadcast_to(np.array([.3, -.2])[:, None, None], (2, 16, 16)).copy()
    run = simulate_viscoplastic_periodic(
        nx=16, initial_velocity=initial, flow_index=.6, yield_stress=.2,
        regularization_m=50., shear_epsilon=1e-3, dt=.01, n_steps=3, sample_every=2,
    )
    np.testing.assert_allclose(run["velocity"], np.broadcast_to(initial, run["velocity"].shape), atol=2e-15)
    np.testing.assert_allclose(run["deviatoric_stress"], 0., atol=1e-14)
    np.testing.assert_allclose(run["dissipation_rate"], 0., atol=1e-27)
    np.testing.assert_allclose(run["energy_budget_residual"], 0., atol=1e-15)
    assert not run["approximate_yielded"].any()


def test_pure_yield_regularization_deforms_below_yield_and_classifier_does_not_change_flow():
    settings = dict(case="periodic_shear", nx=24, consistency=0., flow_index=1.,
                    yield_stress=.03, regularization_m=8., amplitude=.2,
                    dt=.004, n_steps=25, sample_every=10)
    run = simulate_viscoplastic_periodic(**settings, yield_shear_threshold=.5)
    differently_classified = simulate_viscoplastic_periodic(**settings, yield_shear_threshold=0.)
    assert not run["approximate_yielded"].any()
    assert not run["over_yield_stress"].any()
    assert np.max(run["stress_invariant"]) < settings["yield_stress"]
    assert np.linalg.norm(run["velocity"][-1] - run["velocity"][0]) > .01
    assert run["kinetic_energy"][-1] < run["kinetic_energy"][0]
    assert np.all(run["dissipation_rate"] > 0)
    # Classification is a diagnostic; it must never freeze momentum updates.
    np.testing.assert_array_equal(run["velocity"], differently_classified["velocity"])
    np.testing.assert_array_equal(run["kinetic_energy"], differently_classified["kinetic_energy"])
    assert differently_classified["approximate_yielded"].any()


def test_advection_cfl_subdivides_steps_without_changing_inviscid_shear():
    run = simulate_viscoplastic_periodic(
        case="periodic_shear", nx=16, consistency=0., yield_stress=0.,
        solvent_viscosity=0., amplitude=4., cfl=.12, dt=.15, n_steps=2, sample_every=2,
    )
    metadata = run["metadata"]
    np.testing.assert_array_equal(run["times"], [0., .3])
    assert metadata["internal_steps"] > 2
    assert metadata["internal_dt_max"] < .15
    assert metadata["stability_limited_steps"] > 0
    assert metadata["max_advective_cfl"] <= .12 * (1 + 1e-12)
    assert metadata["max_advective_cfl"] > .119
    assert metadata["max_diffusive_number"] == 0.
    np.testing.assert_allclose(run["velocity"][-1], run["velocity"][0], atol=1e-14)


def _diffusion_limited_settings():
    return dict(case="nonlinear_modes", nx=16, flow_index=.6, shear_epsilon=1e-3,
                consistency=.06, yield_stress=.05, solvent_viscosity=.01,
                density=.8, dt=.15, n_steps=3, sample_every=2, amplitude=.8)


def test_diffusion_limit_and_stage_rejections_preserve_samples_and_energy():
    run = simulate_viscoplastic_periodic(**_diffusion_limited_settings())
    metadata = run["metadata"]
    np.testing.assert_array_equal(run["times"], .15 * np.array([0, 2, 3]))
    assert metadata["internal_steps"] > 3
    assert metadata["rejected_steps"] > 0
    assert 0 < metadata["internal_dt_min"] <= metadata["internal_dt_max"] < .15
    assert metadata["max_advective_cfl"] <= metadata["cfl_limit"] * (1 + 1e-12)
    assert metadata["max_diffusive_number"] <= metadata["diffusion_cfl_limit"] * (1 + 1e-12)
    assert np.all(np.diff(run["kinetic_energy"]) <= 0)
    assert np.max(np.abs(run["energy_budget_residual"])) / run["kinetic_energy"][0] < 1e-7


def test_internal_step_budget_fails_before_returning_a_truncated_run():
    with pytest.raises(RuntimeError, match="max_internal_steps"):
        simulate_viscoplastic_periodic(**_diffusion_limited_settings(), max_internal_steps=1)


def test_stage_rejection_budget_fails_explicitly():
    with pytest.raises(FloatingPointError, match="rejection budget exhausted"):
        simulate_viscoplastic_periodic(**_diffusion_limited_settings(), max_step_rejections=0)


def test_zero_duration_has_one_complete_snapshot_and_no_internal_step():
    run = simulate_viscoplastic_periodic(nx=8, n_steps=0, initial_velocity=np.zeros((2, 8, 8)))
    np.testing.assert_array_equal(run["times"], [0.])
    np.testing.assert_array_equal(run["kinetic_energy"], [0.])
    np.testing.assert_array_equal(run["dissipation_rate"], [0.])
    assert run["velocity"].shape == (1, 2, 8, 8)
    assert run["metadata"]["internal_steps"] == 0
    assert run["metadata"]["internal_dt_min"] is None
    assert run["metadata"]["internal_dt_max"] is None
    json.dumps(run["metadata"], allow_nan=False)


def test_validation_report_runs_nonlinear_models_and_separates_sensitivity_families():
    report = validation_report(nx=12, grids=(8, 12), reference_nx=16,
                               duration=.01, dt=.005,
                               epsilon_values=(1e-2, 1e-3), reference_epsilon=1e-4)
    assert report["dns_certified"] is False
    assert [row["flow_index"] for row in report["nonlinear_flow_indices"]] == [.6, 1., 1.4]
    for row in report["nonlinear_flow_indices"]:
        assert row["yield_stress"] > 0
        assert row["all_dataset_fields_finite"]
        assert row["velocity_shape"] == [row["snapshots"], 2, 12, 12]
        assert row["nonrescaling_relative_change"] > 0
        assert row["minimum_dissipation_rate"] >= 0
    assert max(row["relative_velocity_error"] for row in report["newtonian_limits"]) < 1e-10
    assert max(row["initial_relative_error"] for row in report["grid_refinement"]["rows"]) < 1e-13
    assert report["regularization_refinement"]["fixed_nx"] == 12
    assert report["power_law_epsilon_sensitivity"]["fixed_regularization_m"] == 8.
    epsilon_rows = report["power_law_epsilon_sensitivity"]["rows"]
    assert epsilon_rows[1]["zero_shear_apparent_viscosity"] > epsilon_rows[0]["zero_shear_apparent_viscosity"]
    assert report["pure_yield_regularized"]["consistency"] == 0.
    assert report["pure_yield_regularized"]["final_energy"] < report["pure_yield_regularized"]["initial_energy"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("settings", [
    {"density": 0.}, {"solvent_viscosity": -.1}, {"flow_index": 0.},
    {"regularization_m": 0.}, {"shear_epsilon": 0.}, {"consistency": np.inf},
    {"yield_stress": -.1}, {"diffusion_cfl": 0.}, {"quadrature_factor": 1.},
    {"max_internal_steps": 0}, {"max_step_rejections": -1},
    {"n_steps": True}, {"sample_every": 0}, {"yield_shear_threshold": -.1},
    {"initial_velocity": np.zeros((2, 4, 4))},
    {"initial_velocity": np.full((2, 8, 8), np.nan)},
])
def test_invalid_physical_parameters_and_budgets_fail(settings):
    with pytest.raises(ValueError):
        simulate_viscoplastic_periodic(**{"nx": 8, "n_steps": 0, **settings})


def test_invalid_shear_rate_is_rejected():
    with pytest.raises(ValueError, match="shear_rate"):
        apparent_viscosity([0., -1.])
    with pytest.raises(ValueError, match="shear_rate"):
        apparent_viscosity([np.nan])
