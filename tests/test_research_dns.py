"""Physical and numerical checks for the periodic Navier--Stokes benchmarks."""
import json

import numpy as np
import pytest

from fluid_denoise.research.dns import (
    periodic_diagnostics, simulate_periodic_ns, taylor_green_reference,
    validate_periodic_resolution, validate_periodic_time_refinement,
    validate_taylor_green,
)


def test_exact_reference_shape_energy_and_pressure():
    times = np.array([0., .4, 1.])
    result = taylor_green_reference(nx=24, ny=16, times=times, viscosity=.1,
                                    amplitude=.7, length_x=4 * np.pi)
    assert result["velocity"].shape == (3, 2, 16, 24)
    assert result["pressure"].shape == (3, 16, 24)
    measured = .5 * np.mean(np.sum(result["velocity"] ** 2, axis=1), axis=(-2, -1))
    np.testing.assert_allclose(measured, result["kinetic_energy_exact"], atol=1e-15)
    np.testing.assert_allclose(result["pressure"].mean(axis=(-2, -1)), 0., atol=1e-15)
    assert max(periodic_diagnostics(result["velocity"], result["dx"], result["dy"])["divergence_rms"]) < 1e-13
    json.dumps(result["metadata"], allow_nan=False)


def test_exact_velocity_and_pressure_satisfy_momentum_equation():
    reference = taylor_green_reference(nx=24, ny=24, times=[.3], viscosity=.2,
                                       amplitude=.7, length_x=4 * np.pi)
    u, v = reference["velocity"][0]
    pressure = reference["pressure"][0]
    kx, ky = np.meshgrid(2 * np.pi * np.fft.fftfreq(24, reference["dx"]),
                          2 * np.pi * np.fft.fftfreq(24, reference["dy"]))

    def derivative(field, wave):
        return np.fft.ifft2(1j * wave * np.fft.fft2(field)).real

    decay_rate = -.2 * (.5 ** 2 + 1. ** 2)
    for velocity in (u, v):
        laplacian = np.fft.ifft2(-(kx ** 2 + ky ** 2) * np.fft.fft2(velocity)).real
        pressure_gradient = derivative(pressure, kx if velocity is u else ky)
        residual = decay_rate * velocity + u * derivative(velocity, kx) + v * derivative(velocity, ky) + pressure_gradient - .2 * laplacian
        assert np.max(np.abs(residual)) < 2e-14


def test_taylor_green_rk4_matches_exact_rectangular_domain():
    run = simulate_periodic_ns(nx=24, ny=16, n_steps=40, sample_every=7,
                              viscosity=.1, dt=.01, length_x=4 * np.pi)
    exact = taylor_green_reference(nx=24, ny=16, times=run["times"],
                                  viscosity=.1, length_x=4 * np.pi)
    np.testing.assert_allclose(run["velocity"], exact["velocity"], atol=1e-12)
    np.testing.assert_array_equal(run["times"], .01 * np.array([0, 7, 14, 21, 28, 35, 40]))
    assert not run["mask"].any()
    assert run["boundary"] == "periodic"
    assert "not certified" in run["metadata"]["dns_validation"]
    json.dumps(run["metadata"], allow_nan=False)


def test_rk4_temporal_convergence_and_spatial_representation():
    validation = validate_taylor_green()
    errors = [row["relative_velocity_error"] for row in validation["temporal"]]
    assert errors[0] > errors[1] > errors[2] > 0
    assert all(3.8 < row["observed_order"] < 4.3 for row in validation["temporal"][1:])
    assert max(row["relative_velocity_error"] for row in validation["spatial_representation"]) < 1e-8
    assert max(row["max_divergence_rms"] for row in validation["spatial_representation"]) < 1e-14
    assert validation["dns_certified"] is False


@pytest.mark.parametrize("case", ["interacting_vortices", "decaying_turbulence"])
def test_nonlinear_cases_are_finite_incompressible_and_dissipative(case):
    run = simulate_periodic_ns(case=case, nx=24, n_steps=30, dt=.01,
                              sample_every=5, viscosity=.05, seed=17)
    diagnostic = periodic_diagnostics(run["velocity"], run["dx"], run["dy"])
    assert np.all(np.isfinite(run["velocity"]))
    assert max(diagnostic["divergence_rms"]) < 2e-14
    assert np.max(np.diff(diagnostic["kinetic_energy"])) < 1e-10
    assert 0 <= diagnostic["high_mode_energy_fraction"][-1] <= 1
    np.testing.assert_allclose(np.sqrt(np.mean(np.sum(run["velocity"][0] ** 2, axis=0))), 1.)


def test_adaptive_limits_hit_exact_requested_sample_times():
    run = simulate_periodic_ns(case="interacting_vortices", nx=24, dt=.2,
                              n_steps=2, sample_every=1, viscosity=.5, amplitude=2.)
    np.testing.assert_allclose(run["times"], [0., .2, .4], atol=0)
    assert run["metadata"]["internal_steps"] > 2
    assert run["metadata"]["internal_dt_max"] < .2
    assert run["metadata"]["max_advective_cfl"] <= .35 * (1 + 1e-14)
    assert np.all(np.isfinite(run["velocity"]))


def test_zero_duration_and_zero_custom_vorticity():
    result = simulate_periodic_ns(nx=8, n_steps=0, initial_vorticity=np.ones((8, 8)))
    np.testing.assert_array_equal(result["velocity"], 0)
    np.testing.assert_array_equal(result["times"], [0.])
    assert result["metadata"]["removed_vorticity_mean"] == 1.
    assert result["metadata"]["internal_dt_min"] is None
    assert periodic_diagnostics(result["velocity"], result["dx"], result["dy"])["high_mode_energy_fraction"] == [0.]


def test_seeded_turbulence_is_reproducible():
    first = simulate_periodic_ns(case="decaying_turbulence", nx=32, n_steps=2, seed=99)
    second = simulate_periodic_ns(case="decaying_turbulence", nx=32, n_steps=2, seed=99)
    different = simulate_periodic_ns(case="decaying_turbulence", nx=32, n_steps=2, seed=98)
    np.testing.assert_array_equal(first["velocity"], second["velocity"])
    assert np.linalg.norm(first["velocity"] - different["velocity"]) > 1


def test_nonlinear_grid_refinement_reduces_error():
    validation = validate_periodic_resolution(grids=(16, 24, 32), reference_nx=48,
                                               duration=.08, dt=.004)
    errors = [row["final_relative_error"] for row in validation["grid_refinement"]]
    assert errors[0] > errors[1] > errors[2]
    assert validation["dns_certified"] is False
    json.dumps(validation, allow_nan=False)


def test_nonlinear_temporal_refinement_reduces_error():
    validation = validate_periodic_time_refinement(duration=.1,
                                                   time_steps=(.02, .01, .005),
                                                   reference_dt=.0025, amplitude=.5)
    rows = validation["temporal_refinement"]
    assert rows[0]["relative_final_error"] > rows[1]["relative_final_error"] > rows[2]["relative_final_error"]
    assert all(3.7 < row["observed_order"] < 4.3 for row in rows[1:])
    assert not any(row["nominal_dt_capped"] for row in rows)
    json.dumps(validation, allow_nan=False)


def test_finite_but_overflowing_initial_state_fails_loudly():
    with pytest.raises(FloatingPointError):
        simulate_periodic_ns(nx=8, n_steps=0, initial_vorticity=np.full((8, 8), 1e308))


def test_custom_initial_vorticity_rejects_nonfinite_and_wrong_shapes():
    with pytest.raises(ValueError):
        simulate_periodic_ns(nx=8, initial_vorticity=np.full((8, 8), np.nan))
    with pytest.raises(ValueError):
        simulate_periodic_ns(nx=8, initial_vorticity=np.zeros((4, 4)))


def test_timestep_budget_failure_is_explicit():
    with pytest.raises(RuntimeError, match="max_internal_steps"):
        simulate_periodic_ns(nx=16, dt=.01, n_steps=10, max_internal_steps=1)


@pytest.mark.parametrize("settings", [
    {"nx": 4}, {"viscosity": -1}, {"dt": 0}, {"cfl": 1},
    {"sample_every": 0}, {"n_steps": -1}, {"amplitude": np.inf},
])
def test_invalid_parameters_fail(settings):
    with pytest.raises(ValueError):
        simulate_periodic_ns(**settings)
