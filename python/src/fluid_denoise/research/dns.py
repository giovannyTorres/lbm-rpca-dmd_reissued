"""Periodic incompressible Navier--Stokes benchmarks and exact references.

Fields use ``(time, component, y, x)`` on endpoint-exclusive periodic grids.
The numerical solver advances scalar vorticity with a Fourier pseudospectral
discretization, strict 2/3 dealiasing and explicit RK4. It resolves Navier--Stokes
without a turbulence model, but a run is *not* certified DNS merely because it
uses this formulation: nonlinear spatial and temporal refinement is required.
All quantities are nondimensional; density is the constant unit density.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import resample


PERIODIC_CASES = ("taylor_green", "interacting_vortices", "decaying_turbulence")


def _grid(nx: int, ny: int | None, length_x: float, length_y: float) -> dict:
    ny = nx if ny is None else ny
    if any(not isinstance(n, (int, np.integer)) or n < 8 for n in (nx, ny)):
        raise ValueError("nx and ny must be integers >= 8")
    if any(not np.isfinite(v) or v <= 0 for v in (length_x, length_y)):
        raise ValueError("domain lengths must be finite and positive")
    mx = np.rint(np.fft.fftfreq(nx) * nx)
    my = np.rint(np.fft.fftfreq(ny) * ny)
    kx, ky = np.meshgrid(2 * np.pi * mx / length_x, 2 * np.pi * my / length_y)
    k2 = kx * kx + ky * ky
    inverse_k2 = np.zeros_like(k2)
    np.divide(1., k2, out=inverse_k2, where=k2 > 0)
    return {
        "nx": int(nx), "ny": int(ny), "dx": float(length_x / nx),
        "dy": float(length_y / ny), "length_x": float(length_x),
        "length_y": float(length_y), "kx": kx, "ky": ky, "k2": k2,
        "inverse_k2": inverse_k2,
        "dealias": (np.abs(mx)[None, :] < nx / 3) & (np.abs(my)[:, None] < ny / 3),
        "xy": np.meshgrid(np.arange(nx) * length_x / nx, np.arange(ny) * length_y / ny),
    }


def _velocity(omega_hat: np.ndarray, grid: dict) -> np.ndarray:
    # omega = dv/dx - du/dy = -laplacian(psi); u=psi_y, v=-psi_x.
    psi_hat = omega_hat * grid["inverse_k2"]
    return np.stack((np.fft.ifft2(1j * grid["ky"] * psi_hat).real,
                     np.fft.ifft2(-1j * grid["kx"] * psi_hat).real))


def _result(velocity: np.ndarray, times: np.ndarray, grid: dict, metadata: dict) -> dict:
    return {
        "velocity": np.asarray(velocity), "times": np.asarray(times, dtype=float),
        "dx": grid["dx"], "dy": grid["dy"],
        "density": np.ones((len(times), grid["ny"], grid["nx"])),
        "mask": np.zeros((grid["ny"], grid["nx"]), dtype=bool),
        "boundary": "periodic", "metadata": metadata,
    }


def taylor_green_reference(
    nx: int = 32, ny: int | None = None, times=None, viscosity: float = .01,
    amplitude: float = 1., length_x: float = 2 * np.pi,
    length_y: float = 2 * np.pi,
) -> dict:
    """Exact decaying Taylor--Green velocity and pressure on a rectangular torus.

    ``u=A sin(kx*x) cos(ky*y) exp[-nu*(kx²+ky²)*t]`` and
    ``v=-A*kx/ky cos(kx*x) sin(ky*y) exp[-nu*(kx²+ky²)*t]``.
    Here kx=2*pi/Lx and ky=2*pi/Ly. Pressure has zero spatial mean.
    This one-mode solution checks signs, diffusion and temporal integration;
    it does not test nonlinear dealiasing or turbulent spatial resolution.
    """
    grid = _grid(nx, ny, length_x, length_y)
    if not np.isfinite(viscosity) or viscosity < 0 or not np.isfinite(amplitude):
        raise ValueError("viscosity must be nonnegative and amplitude finite")
    times = np.asarray([0.] if times is None else times, dtype=float)
    if times.ndim != 1 or len(times) == 0 or not np.all(np.isfinite(times)) or np.any(times < 0):
        raise ValueError("times must be a nonempty finite nonnegative 1D array")
    x, y = grid["xy"]
    kx, ky = 2 * np.pi / length_x, 2 * np.pi / length_y
    decay = np.exp(-viscosity * (kx * kx + ky * ky) * times)
    initial = np.stack((amplitude * np.sin(kx * x) * np.cos(ky * y),
                        -amplitude * kx / ky * np.cos(kx * x) * np.sin(ky * y)))
    result = _result(decay[:, None, None, None] * initial, times, grid, {
        "case": "taylor_green", "solver": "analytical", "viscosity": float(viscosity),
        "amplitude": float(amplitude), "domain": [float(length_x), float(length_y)],
        "units": "nondimensional", "reference": "exact incompressible Navier-Stokes solution",
        "initial_condition": "one streamfunction mode: A/ky sin(kx*x) sin(ky*y)",
        "spatial_validation_scope": "single-mode representation only; no nonlinear DNS certification",
    })
    pressure0 = amplitude ** 2 / 4 * (np.cos(2 * kx * x) + (kx / ky) ** 2 * np.cos(2 * ky * y))
    result["pressure"] = decay[:, None, None] ** 2 * pressure0
    result["kinetic_energy_exact"] = amplitude ** 2 / 8 * (1 + (kx / ky) ** 2) * decay ** 2
    return result


def _initial_vorticity(case: str, grid: dict, amplitude: float, seed: int) -> tuple[np.ndarray, dict]:
    x, y = grid["xy"]
    lx, ly = grid["length_x"], grid["length_y"]
    if case == "taylor_green":
        kx, ky = 2 * np.pi / lx, 2 * np.pi / ly
        omega = amplitude * (kx * kx + ky * ky) / ky * np.sin(kx * x) * np.sin(ky * y)
        description = {"initial_condition": "one streamfunction mode: A/ky sin(kx*x) sin(ky*y)"}
    elif case == "interacting_vortices":
        width = .065 * min(lx, ly)
        centers = ((.30, .35, 1.), (.55, .45, 1.), (.35, .70, -1.), (.70, .65, -1.))
        omega = np.zeros_like(x)
        # Periodized Gaussians (image sum) are smooth through each periodic seam.
        for cx, cy, sign in centers:
            for image_x in (-1, 0, 1):
                for image_y in (-1, 0, 1):
                    radius2 = (x - (cx + image_x) * lx) ** 2 + (y - (cy + image_y) * ly) ** 2
                    omega += sign * np.exp(-radius2 / (2 * width ** 2))
        description = {"initial_condition": "four periodized Gaussian vorticity cores; two positive, two negative",
                       "centers_fractional": [list(c) for c in centers], "core_width": float(width)}
    else:
        # Fixed physical Fourier coefficients, independent of grid size, let
        # resolution studies use the same initial condition. The grid filter
        # removes unresolved modes; grids >= 28 retain every initial mode.
        rng = np.random.default_rng(seed)
        omega = np.zeros_like(x)
        for mx in range(0, 9):
            for my in range(-8, 9):
                if mx == 0 and my <= 0:
                    continue
                radius = np.hypot(mx, my)
                if radius > 8:
                    continue
                coefficient = np.exp(-.5 * ((radius - 4.) / 1.5) ** 2) * rng.standard_normal()
                phase = rng.uniform(0., 2 * np.pi)
                omega += coefficient * np.cos(2 * np.pi * (mx * x / lx + my * y / ly) + phase)
        description = {"initial_condition": "seeded Fourier vorticity band, peak integer wavenumber 4, cutoff 8",
                       "initial_spectrum_peak": 4, "initial_spectrum_cutoff": 8, "seed": int(seed)}
    removed_mean = float(omega.mean())
    omega_hat = np.fft.fft2(omega - removed_mean) * grid["dealias"]
    if case != "taylor_green":
        speed_rms = np.sqrt(np.mean(np.sum(_velocity(omega_hat, grid) ** 2, axis=0)))
        if speed_rms == 0:
            raise ValueError("initial velocity has zero RMS")
        omega_hat *= amplitude / speed_rms
        description["initial_speed_rms"] = float(amplitude)
    description["removed_vorticity_mean"] = removed_mean
    return omega_hat, description


def simulate_periodic_ns(
    case: str = "taylor_green", nx: int = 32, ny: int | None = None,
    viscosity: float = .01, dt: float = .005, n_steps: int = 100,
    sample_every: int = 5, amplitude: float = 1., seed: int = 0,
    length_x: float = 2 * np.pi, length_y: float = 2 * np.pi,
    cfl: float = .35, initial_vorticity: np.ndarray | None = None,
    max_internal_steps: int = 1_000_000,
) -> dict:
    """Advance periodic 2D incompressible Navier--Stokes without a closure.

    Nominal duration is ``dt*n_steps``. Outputs occur at nominal sampled times,
    including zero and the exact final time. Internal timesteps are capped by
    ``dt``, advective CFL and the RK4 diffusion bound ``nu*dt*kmax² <= 2``.
    Means of supplied vorticity are removed because a periodic velocity has
    zero integrated curl; mean velocity is zero. Nonfinite stages fail loudly.
    ``amplitude`` sets Taylor--Green u amplitude, or initial speed RMS for the
    nonlinear cases. No cylinder, wall or constitutive physics is implied.
    """
    grid = _grid(nx, ny, length_x, length_y)
    if case not in PERIODIC_CASES:
        raise ValueError(f"case must be one of {PERIODIC_CASES}")
    if not np.isfinite(viscosity) or viscosity < 0:
        raise ValueError("viscosity must be finite and nonnegative")
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(cfl) or not 0 < cfl <= .8:
        raise ValueError("dt must be positive and cfl in (0, .8]")
    if not np.isfinite(amplitude) or amplitude <= 0:
        raise ValueError("amplitude must be finite and positive")
    if (not isinstance(n_steps, (int, np.integer)) or n_steps < 0
            or not isinstance(sample_every, (int, np.integer)) or sample_every < 1
            or not isinstance(max_internal_steps, (int, np.integer)) or max_internal_steps < 1):
        raise ValueError("n_steps >= 0, sample_every >= 1 and max_internal_steps >= 1 must be integers")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        if initial_vorticity is None:
            omega, initial_metadata = _initial_vorticity(case, grid, amplitude, seed)
        else:
            initial = np.asarray(initial_vorticity, dtype=float)
            if initial.shape != (grid["ny"], grid["nx"]) or not np.all(np.isfinite(initial)):
                raise ValueError("initial_vorticity must be finite with shape (ny, nx)")
            removed_mean = float(initial.mean())
            omega = np.fft.fft2(initial - removed_mean) * grid["dealias"]
            initial_metadata = {"initial_condition": "user supplied vorticity, projected to resolved periodic modes",
                                "removed_vorticity_mean": removed_mean}
        if not np.all(np.isfinite(omega)):
            raise FloatingPointError("nonfinite initial Fourier vorticity")
    sample_indices = list(range(0, n_steps + 1, sample_every))
    if sample_indices[-1] != n_steps:
        sample_indices.append(n_steps)
    times = dt * np.asarray(sample_indices, dtype=float)
    if not np.all(np.isfinite(times)):
        raise ValueError("requested total duration is not finite")
    snapshots = [_velocity(omega, grid)]
    if not np.all(np.isfinite(snapshots[0])):
        raise FloatingPointError("nonfinite initial velocity")
    max_k2 = float(np.max(grid["k2"][grid["dealias"]]))
    diffusion_dt = 2. / (viscosity * max_k2) if viscosity > 0 else np.inf
    step_count, actual_dt_min, actual_dt_max, max_cfl = 0, np.inf, 0., 0.
    stability_limited_steps = 0

    def rhs(state: np.ndarray) -> np.ndarray:
        velocity = _velocity(state, grid)
        wx = np.fft.ifft2(1j * grid["kx"] * state).real
        wy = np.fft.ifft2(1j * grid["ky"] * state).real
        advection = np.fft.fft2(velocity[0] * wx + velocity[1] * wy) * grid["dealias"]
        value = -advection - viscosity * grid["k2"] * state
        if not np.all(np.isfinite(value)):
            raise FloatingPointError("nonfinite Navier-Stokes RK4 stage; refine timestep or initial condition")
        return value

    current_time = 0.
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for target_time in times[1:]:
            while current_time < target_time:
                velocity = _velocity(omega, grid)
                advective_rate = float(np.max(np.abs(velocity[0]) / grid["dx"]
                                              + np.abs(velocity[1]) / grid["dy"]))
                cfl_dt = cfl / advective_rate if advective_rate > 0 else np.inf
                nominal_remaining_dt = min(dt, float(target_time - current_time))
                stability_dt = min(cfl_dt, diffusion_dt)
                if stability_dt < nominal_remaining_dt * (1 - 1e-12):
                    stability_limited_steps += 1
                step_dt = min(nominal_remaining_dt, stability_dt)
                if not np.isfinite(step_dt) or step_dt <= 0 or current_time + step_dt == current_time:
                    raise FloatingPointError("Navier-Stokes timestep underflow or nonfinite timestep")
                if step_count >= max_internal_steps:
                    raise RuntimeError("max_internal_steps reached before requested final time")
                k1 = rhs(omega)
                k2 = rhs(omega + .5 * step_dt * k1)
                k3 = rhs(omega + .5 * step_dt * k2)
                k4 = rhs(omega + step_dt * k3)
                omega = (omega + step_dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)) * grid["dealias"]
                if not np.all(np.isfinite(omega)):
                    raise FloatingPointError("nonfinite Navier-Stokes state")
                omega[0, 0] = 0.
                current_time = min(float(target_time), current_time + step_dt)
                # Avoid a spurious last substep from ordinary addition roundoff.
                if target_time - current_time <= 8 * np.finfo(float).eps * max(1., abs(target_time)):
                    current_time = float(target_time)
                step_count += 1
                actual_dt_min, actual_dt_max = min(actual_dt_min, step_dt), max(actual_dt_max, step_dt)
                max_cfl = max(max_cfl, step_dt * advective_rate)
            snapshots.append(_velocity(omega, grid))
    velocity = np.asarray(snapshots)
    energy = .5 * np.mean(np.sum(velocity ** 2, axis=1), axis=(-2, -1))
    metadata = {
        "case": case, "solver": "2D incompressible Navier-Stokes vorticity Fourier RK4",
        "viscosity": float(viscosity), "amplitude": float(amplitude), "seed": int(seed),
        "domain": [float(length_x), float(length_y)], "units": "nondimensional",
        "nominal_dt": float(dt), "nominal_steps": int(n_steps),
        "sample_every": int(sample_every), "internal_steps": int(step_count),
        "internal_dt_min": float(actual_dt_min) if step_count else None,
        "internal_dt_max": float(actual_dt_max) if step_count else None,
        "stability_limited_steps": int(stability_limited_steps),
        "max_advective_cfl": float(max_cfl), "cfl_limit": float(cfl),
        "diffusion_stability_limit": "nu * internal_dt * max_retained_k_squared <= 2",
        "dealiasing": "strict two-thirds component-wise Fourier cutoff",
        "kinetic_energy": energy.tolist(), "dns_validation": "not certified; nonlinear grid and time refinement required",
        **initial_metadata,
    }
    return _result(velocity, times, grid, metadata)


def periodic_diagnostics(velocity: np.ndarray, dx: float, dy: float) -> dict:
    """Return spectral divergence, kinetic energy and resolved high-mode energy.

    Tail energy is the fraction in the outer 20% of the retained radial
    wavenumber range. A small fraction alone cannot establish DNS convergence.
    """
    velocity = np.asarray(velocity, dtype=float)
    if velocity.ndim != 4 or velocity.shape[1] != 2 or not np.all(np.isfinite(velocity)):
        raise ValueError("velocity must be finite with shape (T, 2, Y, X)")
    grid = _grid(velocity.shape[-1], velocity.shape[-2], dx * velocity.shape[-1], dy * velocity.shape[-2])
    transformed = np.fft.fft2(velocity, axes=(-2, -1))
    divergence = np.fft.ifft2(1j * grid["kx"] * transformed[:, 0]
                              + 1j * grid["ky"] * transformed[:, 1]).real
    mode_energy = np.sum(np.abs(transformed) ** 2, axis=1)
    max_retained_radius = np.sqrt(np.max(grid["k2"][grid["dealias"]]))
    tail = (np.sqrt(grid["k2"]) >= .8 * max_retained_radius) & grid["dealias"]
    total_energy = np.sum(mode_energy, axis=(-2, -1))
    tail_energy = np.sum(mode_energy[:, tail], axis=1)
    fractions = np.divide(tail_energy, total_energy, out=np.zeros_like(total_energy), where=total_energy > 0)
    return {
        "divergence_rms": np.sqrt(np.mean(divergence ** 2, axis=(-2, -1))).tolist(),
        "kinetic_energy": (.5 * np.mean(np.sum(velocity ** 2, axis=1), axis=(-2, -1))).tolist(),
        "high_mode_energy_fraction": fractions.tolist(),
    }


def validate_taylor_green(
    time_steps=(.2, .1, .05), grids=(8, 12, 16), duration: float = 1.,
    viscosity: float = .2, amplitude: float = .1,
) -> dict:
    """Check RK4 temporal accuracy, spatial representation and energy decay.

    Times must divide duration exactly. Grid tests are representation checks
    because the exact solution consists of one Fourier mode. They cannot be
    interpreted as nonlinear Navier--Stokes spatial convergence evidence.
    """
    if not np.isfinite(duration) or duration <= 0 or len(time_steps) == 0 or len(grids) == 0:
        raise ValueError("positive duration and nonempty time_steps/grids required")
    if not np.isfinite(amplitude) or amplitude <= 0:
        raise ValueError("validation amplitude must be finite and positive")
    if np.any(np.diff(np.asarray(time_steps, dtype=float)) >= 0):
        raise ValueError("time_steps must be strictly decreasing")
    temporal, spatial = [], []
    for requested_dt in time_steps:
        if not np.isfinite(requested_dt) or requested_dt <= 0:
            raise ValueError("time_steps must be finite and positive")
        count = int(round(duration / requested_dt))
        if count < 1 or not np.isclose(count * requested_dt, duration, rtol=1e-12, atol=1e-14):
            raise ValueError("each time step must divide duration")
        run = simulate_periodic_ns(nx=16, dt=requested_dt, n_steps=count, sample_every=count,
                                   viscosity=viscosity, amplitude=amplitude)
        exact = taylor_green_reference(nx=16, times=run["times"], viscosity=viscosity, amplitude=amplitude)
        final = run["velocity"][-1]
        error = float(np.linalg.norm(final - exact["velocity"][-1]) / np.linalg.norm(exact["velocity"][-1]))
        energy = .5 * np.mean(np.sum(final ** 2, axis=0))
        expected_energy = exact["kinetic_energy_exact"][-1]
        row = {"dt": float(requested_dt), "relative_velocity_error": error,
               "relative_energy_error": float(abs(energy - expected_energy) / expected_energy),
               "actual_dt_max": run["metadata"]["internal_dt_max"],
               "nominal_dt_capped": bool(run["metadata"]["stability_limited_steps"]),
               "observed_order": None}
        if (temporal and not row["nominal_dt_capped"] and not temporal[-1]["nominal_dt_capped"]
                and error > 10 * np.finfo(float).eps and temporal[-1]["relative_velocity_error"] > 10 * np.finfo(float).eps):
            row["observed_order"] = float(np.log(temporal[-1]["relative_velocity_error"] / error)
                                           / np.log(temporal[-1]["dt"] / requested_dt))
        temporal.append(row)
    spatial_dt = min(time_steps)
    count = int(round(duration / spatial_dt))
    for nx in grids:
        run = simulate_periodic_ns(nx=nx, dt=spatial_dt, n_steps=count, sample_every=count,
                                   viscosity=viscosity, amplitude=amplitude)
        exact = taylor_green_reference(nx=nx, times=run["times"], viscosity=viscosity, amplitude=amplitude)
        error = float(np.linalg.norm(run["velocity"][-1] - exact["velocity"][-1]) / np.linalg.norm(exact["velocity"][-1]))
        spatial.append({"nx": int(nx), "relative_velocity_error": error,
                        "max_divergence_rms": max(periodic_diagnostics(run["velocity"], run["dx"], run["dy"])["divergence_rms"])})
    return {
        "temporal": temporal, "spatial_representation": spatial,
        "scope": "exact Taylor-Green velocity and energy; single-mode spatial representation only",
        "dns_certified": False,
    }


def validate_periodic_resolution(
    case: str = "interacting_vortices", grids=(16, 24, 32), reference_nx: int = 48,
    duration: float = .2, dt: float = .002, viscosity: float = .01,
    amplitude: float = 1., seed: int = 0,
) -> dict:
    """Compare nonlinear grid refinements against a finer numerical reference.

    The reference is this solver on ``reference_nx``, not analytical truth.
    It is spectrally restricted to each tested grid before comparison.
    Initial-state differences and final errors are both reported. Repeat with
    finer reference and halved dt before claiming resolved nonlinear DNS.
    """
    if case == "taylor_green" or case not in PERIODIC_CASES:
        raise ValueError("nonlinear interacting_vortices or decaying_turbulence case required")
    if not np.isfinite(duration) or duration <= 0 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("duration and dt must be finite and positive")
    if len(grids) == 0 or any(nx >= reference_nx for nx in grids):
        raise ValueError("reference_nx must exceed every tested grid")
    steps = max(1, int(np.ceil(duration / dt)))
    actual_dt = duration / steps
    options = dict(case=case, viscosity=viscosity, dt=actual_dt, n_steps=steps,
                   sample_every=steps, amplitude=amplitude, seed=seed)
    reference = simulate_periodic_ns(nx=reference_nx, **options)
    reference_diagnostics = periodic_diagnostics(reference["velocity"], reference["dx"], reference["dy"])
    rows = []
    for nx in grids:
        run = simulate_periodic_ns(nx=nx, **options)
        restricted = resample(resample(reference["velocity"], nx, axis=-1), nx, axis=-2)
        # Retain precisely the coarse solver's active modes, preventing an
        # unresolved reference tail from dominating the coefficient comparison.
        grid = _grid(nx, nx, 2 * np.pi, 2 * np.pi)
        restricted = np.fft.ifft2(np.fft.fft2(restricted, axes=(-2, -1)) * grid["dealias"], axes=(-2, -1)).real
        errors = [float(np.linalg.norm(run["velocity"][i] - restricted[i]) / np.linalg.norm(restricted[i])) for i in (0, -1)]
        diagnostics = periodic_diagnostics(run["velocity"], run["dx"], run["dy"])
        rows.append({"nx": int(nx), "initial_relative_error": errors[0], "final_relative_error": errors[1],
                     "initial_kinetic_energy": diagnostics["kinetic_energy"][0],
                     "final_kinetic_energy": diagnostics["kinetic_energy"][-1],
                     "final_high_mode_energy_fraction": diagnostics["high_mode_energy_fraction"][-1],
                     "max_divergence_rms": max(diagnostics["divergence_rms"]),
                     "internal_dt_max": run["metadata"]["internal_dt_max"],
                     "stability_limited_steps": run["metadata"]["stability_limited_steps"]})
    return {"case": case, "grid_refinement": rows, "reference_nx": int(reference_nx),
            "reference_diagnostics": {
                "initial_kinetic_energy": reference_diagnostics["kinetic_energy"][0],
                "final_kinetic_energy": reference_diagnostics["kinetic_energy"][-1],
                "final_high_mode_energy_fraction": reference_diagnostics["high_mode_energy_fraction"][-1],
                "max_divergence_rms": max(reference_diagnostics["divergence_rms"]),
                "internal_dt_max": reference["metadata"]["internal_dt_max"],
                "stability_limited_steps": reference["metadata"]["stability_limited_steps"],
            },
            "duration": float(duration), "dt": float(actual_dt), "dns_certified": False,
            "reference_kind": "finer same-solver numerical solution, spectrally restricted to tested modes",
            "limitation": "repeat with finer reference and smaller timestep; no automatic DNS certification"}


def validate_periodic_time_refinement(
    case: str = "interacting_vortices", time_steps=(.01, .005, .0025),
    reference_dt: float = .00125, nx: int = 32, duration: float = .2,
    viscosity: float = .01, amplitude: float = 1., seed: int = 0,
) -> dict:
    """Test nonlinear RK4 time refinement against a finer same-grid run.

    Each requested dt must divide duration and exceed reference_dt. If adaptive
    stability caps override it, the output flags that nominal refinements are
    not distinct integration steps. The grid remains fixed and this test
    provides no evidence about spatial resolution.
    """
    if case == "taylor_green" or case not in PERIODIC_CASES:
        raise ValueError("a nonlinear case is required")
    if not np.isfinite(duration) or duration <= 0 or not np.isfinite(reference_dt) or reference_dt <= 0:
        raise ValueError("duration and reference_dt must be finite and positive")
    requested = np.asarray(time_steps, dtype=float)
    if (requested.ndim != 1 or len(requested) == 0 or not np.all(np.isfinite(requested))
            or np.any(requested <= reference_dt) or np.any(np.diff(requested) >= 0)):
        raise ValueError("time_steps must decrease strictly and all exceed reference_dt")

    def run_with_dt(step_dt):
        count = int(round(duration / step_dt))
        if count < 1 or not np.isclose(count * step_dt, duration, rtol=1e-12, atol=1e-14):
            raise ValueError("every timestep, including reference_dt, must divide duration")
        return simulate_periodic_ns(case=case, nx=nx, dt=float(step_dt), n_steps=count,
                                    sample_every=count, viscosity=viscosity,
                                    amplitude=amplitude, seed=seed)

    reference = run_with_dt(reference_dt)
    reference_final = reference["velocity"][-1]
    reference_norm = np.linalg.norm(reference_final)
    rows = []
    for step_dt in requested:
        run = run_with_dt(step_dt)
        error = float(np.linalg.norm(run["velocity"][-1] - reference_final) / reference_norm)
        maximum_actual = run["metadata"]["internal_dt_max"]
        capped = bool(run["metadata"]["stability_limited_steps"])
        row = {"dt": float(step_dt), "actual_dt_max": maximum_actual,
               "nominal_dt_capped": capped, "relative_final_error": error,
               "observed_order": None}
        if (rows and not capped and not rows[-1]["nominal_dt_capped"]
                and error > 10 * np.finfo(float).eps and rows[-1]["relative_final_error"] > 10 * np.finfo(float).eps):
            row["observed_order"] = float(np.log(rows[-1]["relative_final_error"] / error)
                                           / np.log(rows[-1]["dt"] / step_dt))
        rows.append(row)
    return {"case": case, "nx": int(nx), "duration": float(duration),
            "reference_dt": float(reference_dt), "temporal_refinement": rows,
            "reference_kind": "same-grid same-solver numerical solution with finer timestep",
            "dns_certified": False,
            "limitation": "finer timestep reference is numerical; fixed-grid study does not certify spatial resolution"}
