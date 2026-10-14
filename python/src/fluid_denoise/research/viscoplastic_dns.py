"""Periodic generalized-Newtonian momentum solver, without elasticity.

The incompressible velocity is advanced, rather than rescaling a Newtonian
trajectory. Viscous momentum transport uses the divergence of the complete
spatially variable stress. The regularized model has finite zero-shear
viscosity and no exactly unyielded region. Parameters and fields use declared
nondimensional units; see ``docs/research/VISCOPLASTIC_2D.md``.
"""
from __future__ import annotations

import numpy as np

from .dns import _grid, _initial_vorticity, _velocity, periodic_diagnostics, taylor_green_reference


VISCOPLASTIC_CASES = ("taylor_green", "periodic_shear", "nonlinear_modes",
                     "interacting_vortices", "decaying_turbulence")


def apparent_viscosity(shear_rate, consistency: float = .02, flow_index: float = 1.,
                       yield_stress: float = .02, regularization_m: float = 10.,
                       shear_epsilon: float = 1e-3, solvent_viscosity: float = 0.) -> np.ndarray:
    """Return K*(gamma²+epsilon²)^((n-1)/2)+tau_y*(1-exp(-m*gamma))/gamma.

    The continuous value of the yield term at gamma=0 is tau_y*m. ``expm1``
    avoids cancellation near zero. No viscosity cap or clipping is applied.
    ``consistency`` and ``solvent_viscosity`` are dynamic, not kinematic.
    """
    for name, value in (("consistency", consistency), ("yield_stress", yield_stress),
                        ("solvent_viscosity", solvent_viscosity)):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    for name, value in (("flow_index", flow_index), ("regularization_m", regularization_m),
                        ("shear_epsilon", shear_epsilon)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    gamma = np.asarray(shear_rate, dtype=float)
    if not np.all(np.isfinite(gamma)) or np.any(gamma < 0):
        raise ValueError("shear_rate must be finite and nonnegative")
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        eta = solvent_viscosity + consistency * np.hypot(gamma, shear_epsilon) ** (flow_index - 1.)
        if yield_stress:
            regularized = np.full_like(gamma, regularization_m)
            np.divide(-np.expm1(-regularization_m * gamma), gamma,
                      out=regularized, where=gamma > 0)
            eta = eta + yield_stress * regularized
        if not np.all(np.isfinite(eta)):
            raise FloatingPointError("nonfinite apparent viscosity; check constitutive parameters")
    return eta


def _transfer_hat(source: np.ndarray, target_y: int, target_x: int) -> np.ndarray:
    """Transfer resolved Fourier coefficients, preserving physical amplitudes.

    No Nyquist ambiguity occurs for solver states because the strict two-thirds
    cutoff removes the source Nyquist modes. Cropping also filters those modes.
    """
    source_y, source_x = source.shape[-2:]
    sy = np.rint(np.fft.fftfreq(source_y) * source_y).astype(int)
    sx = np.rint(np.fft.fftfreq(source_x) * source_x).astype(int)
    target = np.zeros(source.shape[:-2] + (target_y, target_x), dtype=complex)
    if target_y >= source_y and target_x >= source_x:
        target[..., sy[:, None] % target_y, sx[None, :] % target_x] = source
    elif target_y <= source_y and target_x <= source_x:
        ty = np.rint(np.fft.fftfreq(target_y) * target_y).astype(int)
        tx = np.rint(np.fft.fftfreq(target_x) * target_x).astype(int)
        target = source[..., ty[:, None] % source_y, tx[None, :] % source_x].copy()
    else:
        raise ValueError("Fourier transfer must pad or restrict both grid dimensions")
    return target * (target_x * target_y / (source_x * source_y))


def _project(state: np.ndarray, grid: dict) -> np.ndarray:
    state = state * grid["dealias"]
    longitudinal = (grid["kx"] * state[0] + grid["ky"] * state[1]) * grid["inverse_k2"]
    value = state - np.stack((grid["kx"] * longitudinal, grid["ky"] * longitudinal))
    # Enforce Hermitian symmetry and retain any conserved mean velocity.
    return np.fft.fft2(np.fft.ifft2(value, axes=(-2, -1)).real, axes=(-2, -1)) * grid["dealias"]


def _constitutive_fields(state: np.ndarray, grid: dict, rheology: dict,
                         yield_shear_threshold: float) -> dict:
    derivatives = np.stack((np.fft.ifft2(1j * grid["kx"] * state, axes=(-2, -1)).real,
                            np.fft.ifft2(1j * grid["ky"] * state, axes=(-2, -1)).real))
    dxx, dyy = derivatives[0, 0], derivatives[1, 1]
    dxy = .5 * (derivatives[1, 0] + derivatives[0, 1])
    shear = np.sqrt(2 * (dxx * dxx + dyy * dyy + 2 * dxy * dxy))
    eta = apparent_viscosity(shear, **rheology)
    stress = 2 * eta[None] * np.stack((dxx, dxy, dyy))
    return {"shear_rate": shear, "apparent_viscosity": eta,
            "deviatoric_stress": stress, "stress_invariant": eta * shear,
            "approximate_yielded": shear > yield_shear_threshold,
            "over_yield_stress": eta * shear > rheology["yield_stress"],
            "derivatives": derivatives}


def _initial_state(case: str, grid: dict, amplitude: float, seed: int,
                   initial_velocity: np.ndarray | None) -> tuple[np.ndarray, dict]:
    if initial_velocity is not None:
        initial = np.asarray(initial_velocity, dtype=float)
        if initial.shape != (2, grid["ny"], grid["nx"]) or not np.all(np.isfinite(initial)):
            raise ValueError("initial_velocity must be finite with shape (2, ny, nx)")
        return _project(np.fft.fft2(initial, axes=(-2, -1)), grid), {
            "initial_condition": "user velocity, Fourier filtered and incompressibly projected"}
    x, y = grid["xy"]
    kx, ky = 2 * np.pi / grid["length_x"], 2 * np.pi / grid["length_y"]
    if case == "periodic_shear":
        velocity = np.stack((amplitude * np.sin(ky * y), np.zeros_like(x)))
        description = {"initial_condition": "u=A sin(2*pi*y/Ly), v=0; exact Newtonian periodic shear"}
    elif case == "nonlinear_modes":
        psi = (np.sin(kx * x) * np.sin(ky * y)
               + .35 * np.cos(2 * kx * x) * np.sin(ky * y)
               + .22 * np.sin(kx * x) * np.cos(2 * ky * y))
        psi_hat = np.fft.fft2(psi) * grid["dealias"]
        velocity = np.stack((np.fft.ifft2(1j * grid["ky"] * psi_hat).real,
                             np.fft.ifft2(-1j * grid["kx"] * psi_hat).real))
        velocity *= amplitude / np.sqrt(np.mean(np.sum(velocity ** 2, axis=0)))
        description = {"initial_condition": "three streamfunction modes with unequal Laplacian eigenvalues",
                       "initial_speed_rms": float(amplitude)}
    else:
        omega, description = _initial_vorticity(case, grid, amplitude, seed)
        velocity = _velocity(omega, grid)
    return _project(np.fft.fft2(velocity, axes=(-2, -1)), grid), description


def simulate_viscoplastic_periodic(
    case: str = "interacting_vortices", nx: int = 32, ny: int | None = None,
    consistency: float = .02, flow_index: float = 1., yield_stress: float = .02,
    regularization_m: float = 10., shear_epsilon: float = 1e-3,
    solvent_viscosity: float = 0., density: float = 1.,
    dt: float = .002, n_steps: int = 100, sample_every: int = 10,
    amplitude: float = 1., seed: int = 0,
    length_x: float = 2 * np.pi, length_y: float = 2 * np.pi,
    cfl: float = .35, diffusion_cfl: float = .4, quadrature_factor: float = 1.5,
    yield_shear_threshold: float = 1e-3,
    initial_velocity: np.ndarray | None = None,
    max_internal_steps: int = 1_000_000, max_step_rejections: int = 20,
) -> dict:
    """Solve du/dt + u.grad(u) = -grad(p)/rho + div(2*eta_app*D)/rho.

    FFT projection imposes incompressibility at each RK4 stage. Products and
    constitutive stress are evaluated on a padded grid, then Galerkin filtered.
    Outputs include t=0 and exactly dt*n_steps. Internal steps respect both
    advective CFL and the maximum apparent-viscosity/tangent diffusion bound.
    Stages that exceed these limits or become nonfinite are rejected and halved;
    rejection and integration budgets prevent an endless or underflowing run.
    The classifier is a user-declared shear threshold, not exact yielding.
    """
    grid = _grid(nx, ny, length_x, length_y)
    if case not in VISCOPLASTIC_CASES:
        raise ValueError(f"case must be one of {VISCOPLASTIC_CASES}")
    rheology = dict(consistency=consistency, flow_index=flow_index, yield_stress=yield_stress,
                    regularization_m=regularization_m, shear_epsilon=shear_epsilon,
                    solvent_viscosity=solvent_viscosity)
    apparent_viscosity(np.array([0.]), **rheology)
    if not np.isfinite(density) or density <= 0 or not np.isfinite(amplitude) or amplitude <= 0:
        raise ValueError("density and amplitude must be finite and positive")
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(cfl) or not 0 < cfl <= .8:
        raise ValueError("dt must be positive and cfl in (0, .8]")
    if not np.isfinite(diffusion_cfl) or not 0 < diffusion_cfl <= 1.:
        raise ValueError("diffusion_cfl must be in (0, 1]")
    if not np.isfinite(quadrature_factor) or quadrature_factor < 1.5:
        raise ValueError("quadrature_factor must be finite and >= 1.5")
    if not np.isfinite(yield_shear_threshold) or yield_shear_threshold < 0:
        raise ValueError("yield_shear_threshold must be finite and nonnegative")
    for name, value, minimum in (("n_steps", n_steps, 0), ("sample_every", sample_every, 1),
                                ("max_internal_steps", max_internal_steps, 1),
                                ("max_step_rejections", max_step_rejections, 0)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    padded = _grid(int(np.ceil(grid["nx"] * quadrature_factor)),
                   int(np.ceil(grid["ny"] * quadrature_factor)), length_x, length_y)
    max_k2 = float(np.max(grid["k2"][grid["dealias"]]))
    samples = list(range(0, n_steps + 1, sample_every))
    if samples[-1] != n_steps:
        samples.append(n_steps)
    times = dt * np.asarray(samples, dtype=float)
    if not np.all(np.isfinite(times)):
        raise ValueError("requested duration must be finite")

    def evaluate(state: np.ndarray) -> tuple[np.ndarray, float, float, float, np.ndarray]:
        padded_state = _transfer_hat(state, padded["ny"], padded["nx"])
        velocity = np.fft.ifft2(padded_state, axes=(-2, -1)).real
        fields = _constitutive_fields(padded_state, padded, rheology, yield_shear_threshold)
        derivatives, stress = fields["derivatives"], fields["deviatoric_stress"]
        advective = velocity[0] * derivatives[0] + velocity[1] * derivatives[1]
        stress_hat = np.fft.fft2(stress, axes=(-2, -1))
        stress_div = np.stack((1j * (padded["kx"] * stress_hat[0] + padded["ky"] * stress_hat[1]),
                               1j * (padded["kx"] * stress_hat[1] + padded["ky"] * stress_hat[2])))
        force = -np.fft.fft2(advective, axes=(-2, -1)) + stress_div / density
        force = _transfer_hat(force, grid["ny"], grid["nx"]) * grid["dealias"]
        pressure_hat = (-1j * density * (grid["kx"] * force[0] + grid["ky"] * force[1])
                        * grid["inverse_k2"])
        projected = _project(force, grid)
        advection_rate = float(np.max(np.abs(velocity[0]) / grid["dx"]
                                      + np.abs(velocity[1]) / grid["dy"]))
        eta_max = float(np.max(fields["apparent_viscosity"]))
        diffusion_rate = max(1., flow_index) * eta_max * max_k2 / density
        dissipation = float(np.mean(fields["apparent_viscosity"] * fields["shear_rate"] ** 2))
        if not (np.all(np.isfinite(projected)) and np.isfinite(dissipation)
                and np.isfinite(diffusion_rate) and np.isfinite(advection_rate)):
            raise FloatingPointError("nonfinite viscoplastic RK4 stage")
        return projected, advection_rate, diffusion_rate, dissipation, pressure_hat

    def energy(state: np.ndarray) -> float:
        velocity = np.fft.ifft2(state, axes=(-2, -1)).real
        return float(.5 * density * np.mean(np.sum(velocity * velocity, axis=0)))

    with np.errstate(over="raise", divide="raise", invalid="raise"):
        state, initial_metadata = _initial_state(case, grid, amplitude, seed, initial_velocity)
        if not np.all(np.isfinite(state)):
            raise FloatingPointError("nonfinite initial Fourier velocity")
        initial_energy = energy(state)
        cumulative_dissipation = 0.
        records: dict[str, list] = {key: [] for key in (
            "velocity", "pressure", "shear_rate", "apparent_viscosity", "deviatoric_stress",
            "stress_invariant", "approximate_yielded", "over_yield_stress", "kinetic_energy",
            "dissipation_rate", "cumulative_dissipation", "energy_budget_residual")}

        def record() -> None:
            fields = _constitutive_fields(state, grid, rheology, yield_shear_threshold)
            evaluation = evaluate(state)
            records["velocity"].append(np.fft.ifft2(state, axes=(-2, -1)).real)
            records["pressure"].append(np.fft.ifft2(evaluation[4]).real)
            for key in ("shear_rate", "apparent_viscosity", "deviatoric_stress", "stress_invariant",
                        "approximate_yielded", "over_yield_stress"):
                records[key].append(fields[key])
            kinetic = energy(state)
            records["kinetic_energy"].append(kinetic)
            records["dissipation_rate"].append(evaluation[3])
            records["cumulative_dissipation"].append(cumulative_dissipation)
            records["energy_budget_residual"].append(kinetic - initial_energy + cumulative_dissipation)

        record()
        current_time, accepted, rejected = 0., 0, 0
        actual_dt_min, actual_dt_max = np.inf, 0.
        max_advective_cfl, max_diffusive_number, max_eta = 0., 0., 0.
        stability_limited = 0
        for target in times[1:]:
            while current_time < target:
                if accepted >= max_internal_steps:
                    raise RuntimeError("max_internal_steps reached before requested final time")
                first = evaluate(state)
                adv_dt = cfl / first[1] if first[1] > 0 else np.inf
                diffusion_dt = diffusion_cfl / first[2] if first[2] > 0 else np.inf
                nominal = min(dt, float(target - current_time))
                step_dt = min(nominal, adv_dt, diffusion_dt)
                if step_dt < nominal * (1 - 1e-12):
                    stability_limited += 1
                old_energy = energy(state)
                rejection_count = 0
                while True:
                    if not np.isfinite(step_dt) or step_dt <= 0 or current_time + step_dt == current_time:
                        raise FloatingPointError("viscoplastic timestep underflow or nonfinite timestep")
                    try:
                        second = evaluate(_project(state + .5 * step_dt * first[0], grid))
                        third = evaluate(_project(state + .5 * step_dt * second[0], grid))
                        fourth = evaluate(_project(state + step_dt * third[0], grid))
                        stages = (first, second, third, fourth)
                        stage_adv = max(stage[1] for stage in stages)
                        stage_diff = max(stage[2] for stage in stages)
                        candidate = _project(state + step_dt / 6 * (
                            first[0] + 2 * second[0] + 2 * third[0] + fourth[0]), grid)
                        if not np.all(np.isfinite(candidate)):
                            raise FloatingPointError("nonfinite candidate viscoplastic state")
                        # A finite stage can still leave the explicit stability
                        # domain; reject before accepting that momentum update.
                        if (step_dt * stage_adv > cfl * (1 + 1e-12)
                                or step_dt * stage_diff > diffusion_cfl * (1 + 1e-12)
                                or energy(candidate) > old_energy + 1e-12 * max(old_energy, 1e-30)):
                            raise FloatingPointError("stage stability or dissipative-energy acceptance limit")
                    except FloatingPointError as error:
                        rejected += 1
                        if rejection_count >= max_step_rejections:
                            raise FloatingPointError("viscoplastic step rejection budget exhausted") from error
                        rejection_count += 1
                        step_dt *= .5
                        continue
                    break
                state = candidate
                cumulative_dissipation += step_dt / 6 * (first[3] + 2 * second[3] + 2 * third[3] + fourth[3])
                current_time = min(float(target), current_time + step_dt)
                if target - current_time <= 8 * np.finfo(float).eps * max(1., abs(target)):
                    current_time = float(target)
                accepted += 1
                actual_dt_min, actual_dt_max = min(actual_dt_min, step_dt), max(actual_dt_max, step_dt)
                max_advective_cfl = max(max_advective_cfl, step_dt * stage_adv)
                max_diffusive_number = max(max_diffusive_number, step_dt * stage_diff)
                if max_k2:
                    max_eta = max(max_eta, stage_diff * density / (max(1., flow_index) * max_k2))
            record()
    result = {key: np.asarray(values) for key, values in records.items()}
    result.update(times=times, dx=grid["dx"], dy=grid["dy"],
                  density=np.full((len(times), grid["ny"], grid["nx"]), density),
                  mask=np.zeros((grid["ny"], grid["nx"]), dtype=bool), boundary="periodic")
    result["metadata"] = {
        "case": case, "solver": "2D incompressible generalized-Newtonian momentum Fourier projection RK4",
        "constitutive_model": "Papanastasiou-regularized Herschel-Bulkley" if flow_index != 1 else "Papanastasiou-regularized Bingham",
        **{name: float(value) for name, value in rheology.items()},
        "constant_density": float(density), "amplitude": float(amplitude), "seed": int(seed),
        "domain": [float(length_x), float(length_y)], "units": "nondimensional",
        "dimensionless_scaling": {"velocity": "U", "length": "L", "time": "L/U",
            "density": "rho_ref", "stress": "rho_ref*U^2", "shear_rate": "U/L",
            "apparent_viscosity": "rho_ref*U*L", "consistency": "rho_ref*U^2/(U/L)^n",
            "regularization_m": "L/U", "shear_epsilon": "U/L"},
        "nominal_dt": float(dt), "nominal_steps": int(n_steps), "sample_every": int(sample_every),
        "internal_steps": int(accepted), "rejected_steps": int(rejected),
        "internal_dt_min": float(actual_dt_min) if accepted else None,
        "internal_dt_max": float(actual_dt_max) if accepted else None,
        "stability_limited_steps": int(stability_limited),
        "max_advective_cfl": float(max_advective_cfl), "cfl_limit": float(cfl),
        "max_diffusive_number": float(max_diffusive_number), "diffusion_cfl_limit": float(diffusion_cfl),
        "diffusion_stability_limit": "dt*max(1,n)*max(eta_app)/rho*max_retained_k_squared <= diffusion_cfl",
        "maximum_stage_viscosity": float(max_eta), "viscosity_cap": None,
        "dealiasing": "strict two-thirds state cutoff; padded nonlinear/constitutive quadrature",
        "quadrature_factor": float(quadrature_factor), "quadrature_shape": [padded["ny"], padded["nx"]],
        "constitutive_aliasing_limit": "nonpolynomial stress requires grid and quadrature refinement; 3/2 padding is not exact dealiasing",
        "stress_components": ["xx", "xy", "yy"], "shear_rate_definition": "sqrt(2*D:D)",
        "stress_invariant_definition": "sqrt(tau:tau/2)=eta_app*shear_rate",
        "yield_estimate": {"kind": "approximate shear threshold, not exact unyielded regions",
                           "shear_threshold": float(yield_shear_threshold),
                           "approximate_yielded_definition": "shear_rate > shear_threshold",
                           "over_yield_stress_definition": "stress_invariant > yield_stress"},
        "energy_definition": "rho/2*mean(|u|^2)",
        "dissipation_definition": "padded-grid mean(eta_app*shear_rate^2), nonnegative",
        "energy_budget_definition": "E(t)-E(0)+RK4_stage_integral(dissipation)",
        "physical_scope": "inelastic generalized-Newtonian periodic flow; no walls, cylinder, elasticity or chaos certification",
        "dns_validation": "not certified; nonlinear grid, time, constitutive quadrature and regularization refinement required",
        **initial_metadata,
    }
    return result


def _relative_error(value: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(value - reference) / max(np.linalg.norm(reference), np.finfo(float).tiny))


def _run_summary(run: dict) -> dict:
    diagnostics = periodic_diagnostics(run["velocity"], run["dx"], run["dy"])
    initial, final = run["velocity"][[0, -1]]
    rescaling = float(np.sum(initial * final) / max(np.sum(initial ** 2), np.finfo(float).tiny))
    return {"initial_energy": float(run["kinetic_energy"][0]),
            "final_energy": float(run["kinetic_energy"][-1]),
            "minimum_dissipation_rate": float(np.min(run["dissipation_rate"])),
            "max_energy_increase": float(max(0., np.max(np.diff(run["kinetic_energy"]), initial=0.))),
            "relative_energy_budget_residual": float(np.max(np.abs(run["energy_budget_residual"])) / max(run["kinetic_energy"][0], np.finfo(float).tiny)),
            "max_divergence_rms": float(max(diagnostics["divergence_rms"])),
            "nonrescaling_relative_change": _relative_error(final, rescaling * initial),
            "final_shear_rate_range": [float(np.min(run["shear_rate"][-1])), float(np.max(run["shear_rate"][-1]))],
            "final_viscosity_range": [float(np.min(run["apparent_viscosity"][-1])), float(np.max(run["apparent_viscosity"][-1]))],
            "approximate_yielded_fraction": float(np.mean(run["approximate_yielded"][-1])),
            "snapshots": int(len(run["times"])), "velocity_shape": list(run["velocity"].shape),
            "all_dataset_fields_finite": bool(all(np.all(np.isfinite(value)) for value in run.values()
                                                  if isinstance(value, np.ndarray))),
            "internal_steps": run["metadata"]["internal_steps"],
            "actual_dt_max": run["metadata"]["internal_dt_max"],
            "nominal_dt_capped": bool(run["metadata"]["stability_limited_steps"] or run["metadata"]["rejected_steps"])}


def validation_report(
    nx: int = 24, duration: float = .08, dt: float = .004,
    grids=(16, 32, 48), reference_nx: int = 64,
    regularization_values=(4., 8., 16.), reference_m: float = 32.,
    epsilon_values=(1e-2, 1e-3, 1e-4), reference_epsilon: float = 1e-5,
    consistency: float = .025, yield_stress: float = .015,
    shear_epsilon: float = 1e-3, amplitude: float = .6,
) -> dict:
    """JSON-ready verification; refinement families have different meanings.

    Regularization refinement compares different constitutive models on one
    fixed grid. Grid refinement holds m fixed and uses a finer same-solver
    numerical reference. Neither reference is analytical truth or DNS proof.
    Actual n=.6,1,1.4 momentum runs have nonzero yield stress. Duration and
    nominal dt must be positive; rounded steps land exactly on duration.
    """
    if not np.isfinite(duration) or duration <= 0 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("duration and dt must be finite and positive")
    if not grids or any(g >= reference_nx for g in grids) or any(a >= b for a, b in zip(grids, grids[1:])):
        raise ValueError("grids must increase strictly and stay below reference_nx")
    if (not regularization_values or any(m <= 0 or not np.isfinite(m) for m in regularization_values)
            or any(a >= b for a, b in zip(regularization_values, regularization_values[1:]))
            or not np.isfinite(reference_m) or reference_m <= regularization_values[-1]):
        raise ValueError("regularization_values must increase positively below reference_m")
    if (not epsilon_values or any(e <= 0 or not np.isfinite(e) for e in epsilon_values)
            or any(a <= b for a, b in zip(epsilon_values, epsilon_values[1:]))
            or not np.isfinite(reference_epsilon) or not 0 < reference_epsilon < epsilon_values[-1]):
        raise ValueError("epsilon_values must decrease positively above reference_epsilon")
    count = max(1, int(np.ceil(duration / dt)))
    actual_dt = duration / count
    common = dict(dt=actual_dt, n_steps=count, sample_every=max(1, count // 4),
                  amplitude=amplitude, shear_epsilon=shear_epsilon)
    limits = []
    for case in ("taylor_green", "periodic_shear"):
        run = simulate_viscoplastic_periodic(case=case, nx=nx, consistency=consistency,
                                            flow_index=1., yield_stress=0., **common)
        if case == "taylor_green":
            exact = taylor_green_reference(nx=nx, times=run["times"], viscosity=consistency,
                                            amplitude=amplitude)["velocity"]
        else:
            exact = run["velocity"][0][None] * np.exp(-consistency * run["times"])[:, None, None, None]
        limits.append({"case": case, "relative_velocity_error": _relative_error(run["velocity"], exact),
                       **_run_summary(run)})
    flows = []
    for n in (.6, 1., 1.4):
        run = simulate_viscoplastic_periodic(case="nonlinear_modes", nx=nx, consistency=consistency,
                                            flow_index=n, yield_stress=yield_stress,
                                            regularization_m=8., **common)
        flows.append({"flow_index": n, "yield_stress": float(yield_stress), **_run_summary(run)})
    nonlinear_options = dict(case="nonlinear_modes", consistency=consistency, flow_index=.6,
                             yield_stress=yield_stress, regularization_m=8., **common)
    reference = simulate_viscoplastic_periodic(nx=reference_nx, **nonlinear_options)
    grid_rows = []
    for tested_nx in grids:
        run = simulate_viscoplastic_periodic(nx=tested_nx, **nonlinear_options)
        tested_grid = _grid(tested_nx, tested_nx, 2 * np.pi, 2 * np.pi)
        restricted = np.fft.ifft2(_transfer_hat(np.fft.fft2(reference["velocity"], axes=(-2, -1)),
                                               tested_nx, tested_nx) * tested_grid["dealias"], axes=(-2, -1)).real
        grid_rows.append({"nx": int(tested_nx), "initial_relative_error": _relative_error(run["velocity"][0], restricted[0]),
                          "final_relative_error": _relative_error(run["velocity"][-1], restricted[-1]), **_run_summary(run)})
    fixed = dict(nonlinear_options, nx=nx)
    fine_time = simulate_viscoplastic_periodic(**{**fixed, "dt": actual_dt / 4,
                                                "n_steps": count * 4, "sample_every": count * 4})
    time_rows = []
    for factor in (1, 2):
        run = simulate_viscoplastic_periodic(**{**fixed, "dt": actual_dt / factor,
                                               "n_steps": count * factor, "sample_every": count * factor})
        time_rows.append({"requested_dt": float(actual_dt / factor),
                          "relative_final_error": _relative_error(run["velocity"][-1], fine_time["velocity"][-1]),
                          **_run_summary(run)})
    for previous, row in zip(time_rows, time_rows[1:]):
        row["observed_order"] = (float(np.log(previous["relative_final_error"] / row["relative_final_error"]) / np.log(2.))
                                 if not previous["nominal_dt_capped"] and not row["nominal_dt_capped"]
                                 and min(previous["relative_final_error"], row["relative_final_error"]) > 1e-14 else None)
    m_reference = simulate_viscoplastic_periodic(**{**fixed, "regularization_m": reference_m})
    m_rows = []
    for m in regularization_values:
        run = simulate_viscoplastic_periodic(**{**fixed, "regularization_m": m})
        m_rows.append({"regularization_m": float(m),
                       "relative_final_error_to_larger_m": _relative_error(run["velocity"][-1], m_reference["velocity"][-1]),
                       **_run_summary(run)})
    epsilon_reference = simulate_viscoplastic_periodic(**{**fixed, "shear_epsilon": reference_epsilon})
    epsilon_rows = []
    for epsilon in epsilon_values:
        run = simulate_viscoplastic_periodic(**{**fixed, "shear_epsilon": epsilon})
        epsilon_rows.append({"shear_epsilon": float(epsilon),
            "relative_final_error_to_smaller_epsilon": _relative_error(run["velocity"][-1], epsilon_reference["velocity"][-1]),
            "zero_shear_apparent_viscosity": float(apparent_viscosity(0., consistency=consistency,
                flow_index=.6, yield_stress=yield_stress, regularization_m=8., shear_epsilon=epsilon)),
            **_run_summary(run)})
    pure_yield = simulate_viscoplastic_periodic(**{**fixed, "case": "periodic_shear",
                                                "consistency": 0., "flow_index": 1.})
    padded_reference = simulate_viscoplastic_periodic(**{**fixed, "quadrature_factor": 3.})
    quadrature_rows = []
    for padding in (1.5, 2.):
        run = simulate_viscoplastic_periodic(**{**fixed, "quadrature_factor": padding})
        quadrature_rows.append({"quadrature_factor": padding,
                                "relative_final_error_to_3x_quadrature": _relative_error(run["velocity"][-1], padded_reference["velocity"][-1])})
    return {"solver": "viscoplastic_periodic_2d", "duration": float(duration), "nx": int(nx),
            "nominal_dt": float(actual_dt), "newtonian_limits": limits, "nonlinear_flow_indices": flows,
            "grid_refinement": {"fixed_regularization_m": 8., "reference_nx": int(reference_nx), "rows": grid_rows,
                                "reference_kind": "finer same-solver numerical solution, restricted to retained modes"},
            "time_refinement": {"fixed_nx": int(nx), "reference_dt": float(actual_dt / 4), "rows": time_rows},
            "regularization_refinement": {"fixed_nx": int(nx), "reference_m": float(reference_m), "rows": m_rows,
                "interpretation": "different finite-viscosity constitutive models, not spatial convergence or an exact ideal-yield solution"},
            "power_law_epsilon_sensitivity": {"fixed_nx": int(nx), "fixed_regularization_m": 8.,
                "reference_epsilon": float(reference_epsilon), "rows": epsilon_rows,
                "interpretation": "power-law low-shear smoothing at fixed Papanastasiou m; neither grid nor m convergence"},
            "pure_yield_regularized": {"consistency": 0., "flow_index": 1.,
                "yield_stress": float(yield_stress), **_run_summary(pure_yield),
                "interpretation": "positive low-shear viscosity permits deformation; no exact unyielded classification"},
            "constitutive_quadrature_refinement": {"fixed_nx": int(nx), "reference_factor": 3., "rows": quadrature_rows},
            "yield_estimates": "regularized flow has no exact unyielded regions; reported classifier is a shear threshold",
            "dns_certified": False,
            "limitations": "finite same-solver references; no cylinder wake, elasticity, ideal-yield or chaos claim"}
