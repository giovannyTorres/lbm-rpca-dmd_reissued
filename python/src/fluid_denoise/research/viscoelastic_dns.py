"""Planar periodic incompressible Oldroyd-B and FENE-P reference simulations.

Velocity lives in two spatial dimensions; the default conformation is the full
3x3 tensor of a three-dimensional polymer in planar flow. No constitutive
information is inferred from a variable-viscosity Newtonian calculation.
Fourier collocation, explicit RK2/3/4 and a Leray projection are used. These
are reference simulations, not an unconditional claim of resolved DNS.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import resample

from .dns import _grid, _initial_vorticity, _velocity, periodic_diagnostics, simulate_periodic_ns


MODELS = ("oldroyd_b", "fene_p")


class _InvalidConformation(FloatingPointError):
    """A trial stage must be rejected without altering its conformation."""


def _physical(state: np.ndarray) -> np.ndarray:
    return np.fft.ifft2(state, axes=(-2, -1)).real


def _fourier(field: np.ndarray, grid: dict) -> np.ndarray:
    return np.fft.fft2(field, axes=(-2, -1)) * grid["dealias"]


def _project(velocity_hat: np.ndarray, grid: dict) -> np.ndarray:
    value = velocity_hat.copy()
    longitudinal = (grid["kx"] * value[0] + grid["ky"] * value[1]) * grid["inverse_k2"]
    value[0] -= grid["kx"] * longitudinal
    value[1] -= grid["ky"] * longitudinal
    return value * grid["dealias"]


def _gradient(field_hat: np.ndarray, grid: dict) -> np.ndarray:
    return np.stack((_physical(1j * grid["kx"] * field_hat),
                     _physical(1j * grid["ky"] * field_hat)), axis=0)


def _check_conformation(c: np.ndarray, model: str, l2: float,
                        eigenvalue_floor: float) -> tuple[float, float]:
    if not np.all(np.isfinite(c)):
        raise _InvalidConformation("nonfinite conformation")
    if np.max(np.abs(c - np.swapaxes(c, 0, 1))) > 1e-10 * max(1., float(np.max(np.abs(c)))):
        raise _InvalidConformation("conformation lost symmetry")
    eigenvalues = np.linalg.eigvalsh(np.moveaxis(c, (0, 1), (-2, -1)))
    minimum = float(eigenvalues.min())
    maximum_trace = float(np.trace(c, axis1=0, axis2=1).max())
    if minimum <= eigenvalue_floor:
        raise _InvalidConformation("conformation is not strictly positive definite")
    if model == "fene_p" and maximum_trace >= l2:
        raise _InvalidConformation("FENE-P trace reached the finite-extensibility barrier")
    return minimum, maximum_trace


def _factor(c: np.ndarray, model: str, l2: float) -> np.ndarray:
    if model == "oldroyd_b":
        return np.ones(c.shape[-2:])
    # Normalization makes C=I an equilibrium in dimension d, including Czz.
    return (l2 - c.shape[0]) / (l2 - np.trace(c, axis1=0, axis2=1))


def _source(c: np.ndarray, g: np.ndarray, model: str, l2: float,
            relaxation_time: float) -> np.ndarray:
    identity = np.eye(c.shape[0])[:, :, None, None]
    stretching = (np.einsum("ik...,kj...->ij...", g, c)
                  + np.einsum("ik...,jk...->ij...", c, g))
    return stretching - (_factor(c, model, l2)[None, None] * c - identity) / relaxation_time


def _rhs(u_hat: np.ndarray, c_hat: np.ndarray, grid: dict, *, model: str,
         viscosity_s: float, viscosity_p: float, relaxation_time: float,
         l2: float, kappa: float, forcing_hat: np.ndarray,
         eigenvalue_floor: float) -> tuple[np.ndarray, np.ndarray]:
    u, c = _physical(u_hat), _physical(c_hat)
    _check_conformation(c, model, l2, eigenvalue_floor)
    gradients_u, gradients_c = _gradient(u_hat, grid), _gradient(c_hat, grid)
    dimension = c.shape[0]
    g = np.zeros_like(c)
    # G_ij = partial_j u_i, including zeros outside the planar velocity block.
    g[:2, 0], g[:2, 1] = gradients_u[0], gradients_u[1]
    identity = np.eye(dimension)[:, :, None, None]
    tau = viscosity_p / relaxation_time * (_factor(c, model, l2)[None, None] * c - identity)
    tau_hat = _fourier(tau, grid)
    stress_divergence = 1j * (grid["kx"] * tau_hat[:2, 0] + grid["ky"] * tau_hat[:2, 1])
    u_advection = u[0] * gradients_u[0] + u[1] * gradients_u[1]
    du = _project(-_fourier(u_advection, grid) + stress_divergence + forcing_hat, grid)
    du -= viscosity_s * grid["k2"] * u_hat
    c_advection = u[0] * gradients_c[0] + u[1] * gradients_c[1]
    dc = _fourier(-c_advection + _source(c, g, model, l2, relaxation_time), grid)
    dc -= kappa * grid["k2"] * c_hat
    if not np.all(np.isfinite(du)) or not np.all(np.isfinite(dc)):
        raise FloatingPointError("nonfinite viscoelastic right-hand side")
    return du, dc


def _rk_step(u: np.ndarray, c: np.ndarray, step_dt: float, rhs, order: int) -> tuple[np.ndarray, np.ndarray]:
    a, b = rhs(u, c)
    if order == 2:
        d, e = rhs(u + .5 * step_dt * a, c + .5 * step_dt * b)
        return u + step_dt * d, c + step_dt * e
    if order == 3:
        d, e = rhs(u + step_dt * a, c + step_dt * b)
        um, cm = .75 * u + .25 * (u + step_dt * (a + d)), .75 * c + .25 * (c + step_dt * (b + e))
        f, h = rhs(um, cm)
        return u / 3 + 2 / 3 * (um + step_dt * f), c / 3 + 2 / 3 * (cm + step_dt * h)
    d, e = rhs(u + .5 * step_dt * a, c + .5 * step_dt * b)
    f, h = rhs(u + .5 * step_dt * d, c + .5 * step_dt * e)
    j, k = rhs(u + step_dt * f, c + step_dt * h)
    return (u + step_dt / 6 * (a + 2 * d + 2 * f + j),
            c + step_dt / 6 * (b + 2 * e + 2 * h + k))


def _forcing(grid: dict, kind: str | None, amplitude: float, mode: int) -> np.ndarray:
    x, y = grid["xy"]
    field = np.zeros((2, grid["ny"], grid["nx"]))
    if kind == "kolmogorov":
        field[0] = amplitude * np.sin(mode * 2 * np.pi * y / grid["length_y"])
    elif kind == "cellular":
        kx, ky = mode * 2 * np.pi / grid["length_x"], mode * 2 * np.pi / grid["length_y"]
        field[0] = amplitude * np.sin(kx * x) * np.cos(ky * y)
        field[1] = -amplitude * kx / ky * np.cos(kx * x) * np.sin(ky * y)
    elif kind is not None:
        raise ValueError("forcing must be None, 'kolmogorov' or 'cellular'")
    if kind is None and amplitude != 0:
        raise ValueError("nonzero forcing_amplitude requires a forcing kind")
    if kind is not None and (mode >= grid["nx"] / 3 or mode >= grid["ny"] / 3):
        raise ValueError("forcing_mode must lie within the retained Fourier band")
    return _project(_fourier(field, grid), grid)


def _energetics(u_hat: np.ndarray, c_hat: np.ndarray, grid: dict, rhs,
                 viscosity_s: float, viscosity_p: float, relaxation_time: float,
                 model: str, l2: float, kappa: float, force: np.ndarray) -> dict:
    """Instantaneous discrete rates and continuum exchange/dissipation terms.

    The free-energy identity can have a spatial quadrature/projection defect;
    its residual is measured, not assumed zero. It is a refinement diagnostic.
    """
    u, c = _physical(u_hat), _physical(c_hat)
    matrices = np.moveaxis(c, (0, 1), (-2, -1))
    c_inverse = np.moveaxis(np.linalg.inv(matrices), (-2, -1), (0, 1))
    trace = np.trace(c, axis1=0, axis2=1)
    trace_inverse = np.trace(c_inverse, axis1=0, axis2=1)
    logdet = np.linalg.slogdet(matrices)[1]
    dimension = c.shape[0]
    f = _factor(c, model, l2)
    potential = (trace - logdet - dimension if model == "oldroyd_b" else
                 -(l2 - dimension) * np.log((l2 - trace) / (l2 - dimension)) - logdet)
    grad_u, grad_c = _gradient(u_hat, grid), _gradient(c_hat, grid)
    tau = viscosity_p / relaxation_time * (f[None, None] * c - np.eye(dimension)[:, :, None, None])
    work = float(np.mean(np.sum(tau[:2, :2] * np.moveaxis(grad_u, 0, 1), axis=(0, 1))))
    solvent = float(viscosity_s * np.mean(np.sum(grad_u ** 2, axis=(0, 1))))
    polymer = float(viscosity_p / (2 * relaxation_time ** 2)
                    * np.mean(f ** 2 * trace - 2 * f * dimension + trace_inverse))
    diffusion = 0.
    if kappa:
        diffusion_integrand = np.zeros_like(trace)
        for derivative in grad_c:
            product = np.einsum("ik...,kj...->ij...", c_inverse, derivative)
            diffusion_integrand += np.einsum("ij...,ji...->...", product, product)
            if model == "fene_p":
                diffusion_integrand += ((l2 - dimension) / (l2 - trace) ** 2
                                        * np.trace(derivative, axis1=0, axis2=1) ** 2)
        diffusion = float(viscosity_p * kappa / (2 * relaxation_time) * np.mean(diffusion_integrand))
    du, dc = rhs(u_hat, c_hat)
    kinetic_rate = float(np.mean(np.sum(u * _physical(du), axis=0)))
    energy_gradient = f[None, None] * np.eye(dimension)[:, :, None, None] - c_inverse
    polymer_rate = float(viscosity_p / (2 * relaxation_time)
                         * np.mean(np.sum(energy_gradient * _physical(dc), axis=(0, 1))))
    force_work = float(np.mean(np.sum(u * force, axis=0)))
    values = {"kinetic_energy": float(.5 * np.mean(np.sum(u ** 2, axis=0))),
            "polymer_free_energy": float(viscosity_p / (2 * relaxation_time) * np.mean(potential)),
            "polymer_stress_work": work, "solvent_dissipation": solvent,
            "polymer_relaxation_dissipation": polymer, "polymer_diffusion_dissipation": diffusion,
            "forcing_work": force_work, "kinetic_energy_rate": kinetic_rate,
            "polymer_free_energy_rate": polymer_rate,
            "kinetic_exchange_residual": kinetic_rate + solvent + work - force_work,
            "total_energy_rate_residual": kinetic_rate + polymer_rate + solvent + polymer + diffusion - force_work}
    if not all(np.isfinite(value) for value in values.values()):
        raise FloatingPointError("nonfinite viscoelastic energy or exchange diagnostic")
    return values


def simulate_viscoelastic_periodic(
    model: str = "oldroyd_b", nx: int = 32, ny: int | None = None,
    viscosity_s: float = .01, viscosity_p: float = .01,
    relaxation_time: float = .2, dt: float = .002, n_steps: int = 100,
    sample_every: int = 5, case: str = "taylor_green", amplitude: float = 1.,
    seed: int = 0, length_x: float = 2 * np.pi, length_y: float = 2 * np.pi,
    cfl: float = .25, extensibility_squared: float = 50., stress_diffusivity: float = 0.,
    forcing: str | None = None, forcing_amplitude: float = 0., forcing_mode: int = 1,
    tensor_dimension: int = 3, rk_order: int = 4, eigenvalue_floor: float = 1e-12,
    initial_conformation: np.ndarray | None = None, max_internal_steps: int = 1_000_000,
    max_step_rejections: int = 40,
) -> dict:
    """Integrate Navier--Stokes with upper-convected polymer memory on a torus.

    Unit density makes ``viscosity_s``/``viscosity_p`` nondimensional kinematic
    viscosities. ``extensibility_squared`` is L², not L. Polymer stress is
    ``viscosity_p / relaxation_time * (f*C-I)``. Default C=I; optional explicit
    ``stress_diffusivity`` adds kappa*laplacian(C) to the constitutive equation.
    There is no diffusion otherwise. All RK stages and accepted states must
    have positive eigenvalues and, for FENE-P, trace(C)<L². Invalid trials are
    rejected and their timestep halved; no clipping or eigenvalue repair occurs.

    Outputs include both endpoints at nominal sampled times. Conformation and
    stress have shape (T,d,d,Y,X), d=3 by default. Constitutive RHS snapshots
    and an independent finite-difference-in-time residual are returned too.
    """
    grid = _grid(nx, ny, length_x, length_y)
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}")
    if case not in ("taylor_green", "interacting_vortices", "decaying_turbulence"):
        raise ValueError("unknown periodic initial case")
    for name, value in (("viscosity_s", viscosity_s), ("viscosity_p", viscosity_p),
                        ("stress_diffusivity", stress_diffusivity)):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    for name, value in (("relaxation_time", relaxation_time), ("dt", dt), ("amplitude", amplitude)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(cfl) or not 0 < cfl <= .5:
        raise ValueError("cfl must lie in (0,.5]")
    if tensor_dimension not in (2, 3) or rk_order not in (2, 3, 4):
        raise ValueError("tensor_dimension must be 2 or 3 and rk_order 2, 3 or 4")
    if not np.isfinite(extensibility_squared) or extensibility_squared <= tensor_dimension:
        raise ValueError("extensibility_squared must exceed tensor_dimension")
    if not np.isfinite(eigenvalue_floor) or not 0 <= eigenvalue_floor < 1:
        raise ValueError("eigenvalue_floor must lie in [0,1)")
    if not np.isfinite(forcing_amplitude):
        raise ValueError("forcing_amplitude must be finite")
    for name, value, lower in (("n_steps", n_steps, 0), ("sample_every", sample_every, 1),
                               ("forcing_mode", forcing_mode, 1), ("max_internal_steps", max_internal_steps, 1),
                               ("max_step_rejections", max_step_rejections, 1)):
        if not isinstance(value, (int, np.integer)) or value < lower:
            raise ValueError(f"{name} must be an integer >= {lower}")
    identity = np.broadcast_to(np.eye(tensor_dimension)[:, :, None, None],
                               (tensor_dimension, tensor_dimension, grid["ny"], grid["nx"])).copy()
    initial = identity if initial_conformation is None else np.asarray(initial_conformation, dtype=float)
    if initial.shape != identity.shape:
        raise ValueError("initial_conformation must have shape (d,d,ny,nx)")
    _check_conformation(initial, model, extensibility_squared, eigenvalue_floor)
    c_hat = _fourier(initial, grid)
    # A supplied field is filtered, and the filtered field must also be valid.
    _check_conformation(_physical(c_hat), model, extensibility_squared, eigenvalue_floor)
    omega_hat, initial_metadata = _initial_vorticity(case, grid, amplitude, seed)
    u_hat = _project(_fourier(_velocity(omega_hat, grid), grid), grid)
    force_hat = _forcing(grid, forcing, forcing_amplitude, forcing_mode)
    rhs_options = dict(model=model, viscosity_s=viscosity_s, viscosity_p=viscosity_p,
                       relaxation_time=relaxation_time, l2=extensibility_squared,
                       kappa=stress_diffusivity, forcing_hat=force_hat, eigenvalue_floor=eigenvalue_floor)

    def rhs(u, c):
        return _rhs(u, c, grid, **rhs_options)

    sample_indices = list(range(0, n_steps + 1, sample_every))
    if sample_indices[-1] != n_steps:
        sample_indices.append(n_steps)
    times = dt * np.asarray(sample_indices, dtype=float)
    if not np.all(np.isfinite(times)):
        raise ValueError("requested duration must be finite")
    velocity, conformation, stress, constitutive_rhs, diagnostics = [], [], [], [], []
    unfiltered_constitutive_rhs, constitutive_projection_defect = [], []
    stage_minimum, stage_trace_maximum = np.inf, -np.inf

    def record():
        c, u = _physical(c_hat), _physical(u_hat)
        minimum, maximum_trace = _check_conformation(c, model, extensibility_squared, eigenvalue_floor)
        tau = viscosity_p / relaxation_time * (_factor(c, model, extensibility_squared)[None, None] * c - identity)
        if not np.all(np.isfinite(tau)):
            raise FloatingPointError("nonfinite polymer stress")
        energy = _energetics(u_hat, c_hat, grid, rhs, viscosity_s, viscosity_p,
                            relaxation_time, model, extensibility_squared, stress_diffusivity, _physical(force_hat))
        energy.update(min_conformation_eigenvalue=minimum, max_conformation_trace=maximum_trace,
                      stress_rms=float(np.sqrt(np.mean(tau ** 2))))
        velocity.append(u)
        conformation.append(c)
        stress.append(tau)
        discrete_rhs = _physical(rhs(u_hat, c_hat)[1])
        c_grad, u_grad = _gradient(c_hat, grid), _gradient(u_hat, grid)
        g = np.zeros_like(c)
        g[:2, 0], g[:2, 1] = u_grad[0], u_grad[1]
        physical_rhs = (-u[0] * c_grad[0] - u[1] * c_grad[1]
                        + _source(c, g, model, extensibility_squared, relaxation_time)
                        - stress_diffusivity * _physical(grid["k2"] * c_hat))
        if not np.all(np.isfinite(physical_rhs)):
            raise FloatingPointError("nonfinite unfiltered constitutive PDE right-hand side")
        constitutive_rhs.append(discrete_rhs)
        unfiltered_constitutive_rhs.append(physical_rhs)
        constitutive_projection_defect.append(float(np.sqrt(np.mean((physical_rhs - discrete_rhs) ** 2))))
        diagnostics.append(energy)

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        record()
    max_k2 = float(np.max(grid["k2"][grid["dealias"]]))
    diffusion_coefficient = max(viscosity_s, stress_diffusivity)
    diffusion_dt = .9 / (diffusion_coefficient * max_k2) if diffusion_coefficient else np.inf
    current_time, steps, rejected, capped = 0., 0, 0, 0
    dt_min, dt_max, maximum_cfl = np.inf, 0., 0.
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for target in times[1:]:
            while current_time < target:
                if steps >= max_internal_steps:
                    raise RuntimeError("max_internal_steps reached before final time")
                u, c = _physical(u_hat), _physical(c_hat)
                factor_max = float(_factor(c, model, extensibility_squared).max())
                advection_rate = float(np.max(np.abs(u[0]) / grid["dx"] + np.abs(u[1]) / grid["dy"]))
                advection_dt = cfl / advection_rate if advection_rate else np.inf
                relaxation_dt = .2 * relaxation_time / max(1., factor_max)
                stretching_rate = float(np.max(np.sum(np.abs(_gradient(u_hat, grid)), axis=(0, 1))))
                stretching_dt = .2 / stretching_rate if stretching_rate else np.inf
                eigen_max = float(np.linalg.eigvalsh(np.moveaxis(c, (0, 1), (-2, -1))).max())
                elastic_speed = np.sqrt(viscosity_p * factor_max * eigen_max / relaxation_time)
                elastic_dt = cfl * min(grid["dx"], grid["dy"]) / elastic_speed if elastic_speed else np.inf
                nominal = min(dt, float(target - current_time))
                step_dt = min(nominal, advection_dt, diffusion_dt, relaxation_dt, stretching_dt, elastic_dt)
                if step_dt < nominal * (1 - 1e-12):
                    capped += 1
                for attempt in range(max_step_rejections + 1):
                    if not np.isfinite(step_dt) or step_dt <= 0 or current_time + step_dt == current_time:
                        raise FloatingPointError("viscoelastic timestep underflow")
                    try:
                        def checked_rhs(uh, ch):
                            nonlocal stage_minimum, stage_trace_maximum
                            low, high = _check_conformation(_physical(ch), model, extensibility_squared, eigenvalue_floor)
                            stage_minimum, stage_trace_maximum = min(stage_minimum, low), max(stage_trace_maximum, high)
                            return rhs(uh, ch)

                        candidate_u, candidate_c = _rk_step(u_hat, c_hat, step_dt, checked_rhs, rk_order)
                        candidate_u = _project(candidate_u, grid)
                        candidate_c *= grid["dealias"]
                        low, high = _check_conformation(_physical(candidate_c), model, extensibility_squared, eigenvalue_floor)
                        if not np.all(np.isfinite(candidate_u)):
                            raise FloatingPointError("nonfinite trial velocity")
                        stage_minimum, stage_trace_maximum = min(stage_minimum, low), max(stage_trace_maximum, high)
                        break
                    except FloatingPointError as error:
                        rejected += 1
                        if attempt == max_step_rejections:
                            raise FloatingPointError("conformation/stability trial rejection limit reached; refine grid or parameters") from error
                        step_dt *= .5
                u_hat, c_hat = candidate_u, candidate_c
                steps += 1
                dt_min, dt_max = min(dt_min, step_dt), max(dt_max, step_dt)
                maximum_cfl = max(maximum_cfl, step_dt * advection_rate)
                current_time = min(float(target), current_time + step_dt)
                if target - current_time <= 8 * np.finfo(float).eps * max(1., abs(target)):
                    current_time = float(target)
            record()
    conformation_array, rhs_array = np.asarray(conformation), np.asarray(constitutive_rhs)
    finite_difference_residual = None
    pde_residual = None
    if len(times) >= 3:
        derivative = np.gradient(conformation_array, times, axis=0, edge_order=2)
        finite_difference_residual = np.sqrt(np.mean((derivative - rhs_array) ** 2, axis=(1, 2, 3, 4))).tolist()
        pde_residual = np.sqrt(np.mean((derivative - np.asarray(unfiltered_constitutive_rhs)) ** 2, axis=(1, 2, 3, 4))).tolist()
    diagnostic_columns = {name: [row[name] for row in diagnostics] for name in diagnostics[0]}
    characteristic_length = (float(initial_metadata["core_width"]) if case == "interacting_vortices" else
                             min(length_x, length_y) / (2 * np.pi * (4 if case == "decaying_turbulence" else 1)))
    initial_rms_speed = float(np.sqrt(np.mean(np.sum(velocity[0] ** 2, axis=0))))
    metadata = {"model": model, "case": case,
                "solver": f"2D periodic projected Navier-Stokes + upper-convected conformation Fourier RK{rk_order}",
                "units": "nondimensional, unit density", "domain": [float(length_x), float(length_y)],
                "boundary": "periodic",
                "viscosity_s": float(viscosity_s), "viscosity_p": float(viscosity_p),
                "amplitude": float(amplitude), "seed": int(seed),
                "relaxation_time": float(relaxation_time), "tensor_dimension": int(tensor_dimension),
                "characteristic_length": float(characteristic_length), "initial_rms_speed": initial_rms_speed,
                "weissenberg_reference": float(relaxation_time * initial_rms_speed / characteristic_length),
                "reynolds_reference": float(initial_rms_speed * characteristic_length / (viscosity_s + viscosity_p)) if viscosity_s + viscosity_p > 0 else None,
                "dimensionless_definition": "Wi=lambda*U_rms_initial/ell; Re=U_rms_initial*ell/(nu_s+nu_p); ell=Gaussian core width, TG inverse fundamental wavenumber, or inverse spectrum-peak wavenumber",
                "extensibility_squared": float(extensibility_squared) if model == "fene_p" else None,
                "stress_diffusivity": float(stress_diffusivity),
                "stress_diffusion_is_artificial": bool(stress_diffusivity),
                "forcing": forcing, "forcing_amplitude": float(forcing_amplitude), "forcing_mode": int(forcing_mode),
                "nominal_dt": float(dt), "nominal_steps": int(n_steps), "sample_every": int(sample_every),
                "internal_steps": int(steps), "internal_dt_min": float(dt_min) if steps else None,
                "internal_dt_max": float(dt_max) if steps else None, "stability_limited_steps": int(capped),
                "rejected_steps": int(rejected), "max_advective_cfl": float(maximum_cfl), "cfl_limit": float(cfl),
                "stage_min_conformation_eigenvalue": float(stage_minimum) if steps else diagnostic_columns["min_conformation_eigenvalue"][0],
                "stage_max_conformation_trace": float(stage_trace_maximum) if steps else diagnostic_columns["max_conformation_trace"][0],
                "positivity_policy": "reject and halve timestep; no conformation clipping",
                "dealiasing": "strict component-wise two-thirds cutoff; quadratic products dealiased; rational FENE factor filtered on collocation grid",
                "constitutive_residual_rms": finite_difference_residual,
                "constitutive_pde_residual_rms": pde_residual,
                "constitutive_spatial_projection_defect_rms": constitutive_projection_defect,
                "constitutive_residual_method": "second-order snapshot time difference minus stored projected model RHS; absent with fewer than three snapshots",
                "constitutive_pde_residual_method": "second-order snapshot time difference minus unfiltered physical constitutive RHS on collocation grid; includes temporal and spectral projection errors",
                "diagnostics": diagnostic_columns, "dns_certified": False,
                "validation_scope": "finite moderate-parameter periodic reference; refinement needed, no cylinder/wall/high-Wi claim",
                **initial_metadata}
    return {"velocity": np.asarray(velocity), "times": times, "dx": grid["dx"], "dy": grid["dy"],
            "density": np.ones((len(times), grid["ny"], grid["nx"])),
            "mask": np.zeros((grid["ny"], grid["nx"]), dtype=bool), "boundary": "periodic",
            "conformation": conformation_array, "stress": np.asarray(stress),
            "constitutive_rhs": rhs_array, "metadata": metadata}


def uniform_shear_reference(times, shear_rate: float = 1., relaxation_time: float = .2,
                            tensor_dimension: int = 3) -> np.ndarray:
    """Exact homogeneous Oldroyd-B C(t), initially I, in imposed uniform shear.

    This tests the constitutive source ODE; uniform shear velocity is not a
    periodic DNS solution on the torus.
    """
    t = np.asarray(times, dtype=float)
    if t.ndim != 1 or not np.all(np.isfinite(t)) or np.any(t < 0):
        raise ValueError("times must be a finite nonnegative 1D array")
    if not np.isfinite(shear_rate) or not np.isfinite(relaxation_time) or relaxation_time <= 0:
        raise ValueError("finite shear_rate and positive relaxation_time required")
    if tensor_dimension not in (2, 3):
        raise ValueError("tensor_dimension must be 2 or 3")
    result = np.broadcast_to(np.eye(tensor_dimension), (len(t), tensor_dimension, tensor_dimension)).copy()
    q, wi = t / relaxation_time, shear_rate * relaxation_time
    result[:, 0, 1] = result[:, 1, 0] = wi * (-np.expm1(-q))
    result[:, 0, 0] += 2 * wi ** 2 * (-np.expm1(-q) - q * np.exp(-q))
    return result


def integrate_uniform_shear(model: str = "oldroyd_b", shear_rate: float = 1.,
                            relaxation_time: float = .2, dt: float = .005,
                            n_steps: int = 100, extensibility_squared: float = 50.,
                            tensor_dimension: int = 3) -> dict:
    """RK4 source-only constitutive test with explicit stage admissibility."""
    if model not in MODELS or tensor_dimension not in (2, 3):
        raise ValueError("valid model and tensor_dimension required")
    if (not np.isfinite(dt) or dt <= 0 or not np.isfinite(relaxation_time) or relaxation_time <= 0
            or not np.isfinite(shear_rate) or not np.isfinite(extensibility_squared)
            or extensibility_squared <= tensor_dimension):
        raise ValueError("invalid source-ODE parameter")
    if not isinstance(n_steps, (int, np.integer)) or n_steps < 0:
        raise ValueError("n_steps must be a nonnegative integer")
    c = np.eye(tensor_dimension)[:, :, None, None]
    g = np.zeros_like(c)
    g[0, 1] = shear_rate
    values = [c[:, :, 0, 0].copy()]

    def source(value):
        _check_conformation(value, model, extensibility_squared, 0.)
        return _source(value, g, model, extensibility_squared, relaxation_time)

    for _ in range(n_steps):
        a = source(c)
        b = source(c + .5 * dt * a)
        d = source(c + .5 * dt * b)
        e = source(c + dt * d)
        c = c + dt / 6 * (a + 2 * b + 2 * d + e)
        _check_conformation(c, model, extensibility_squared, 0.)
        values.append(c[:, :, 0, 0].copy())
    return {"times": dt * np.arange(n_steps + 1), "conformation": np.asarray(values),
            "scope": "spatially homogeneous source ODE, not a periodic velocity simulation"}


def _relative(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(float(np.linalg.norm(b)), np.finfo(float).tiny))


def _restrict(field: np.ndarray, nx: int, grid: dict) -> np.ndarray:
    value = resample(resample(field, nx, axis=-1), nx, axis=-2)
    return _physical(_fourier(value, grid))


def validation_report(
    model: str = "oldroyd_b", grids=(16, 24, 32), reference_nx: int = 48,
    time_steps=(.02, .01, .005), reference_dt: float = .0025,
    duration: float = .1, nx: int = 32, case: str = "interacting_vortices",
    viscosity_s: float = .02, viscosity_p: float = .02, relaxation_time: float = .25,
    amplitude: float = .3, extensibility_squared: float = 30.,
    stress_diffusivity: float = 0.,
) -> dict:
    """JSON-ready checks, >=3 grid/time refinements and honest numerical refs.

    No report is a global DNS certification. Newtonian-limit checks use
    both models and decreasing relaxation time. With kappa>0 a matched
    kappa=0/half/full sensitivity comparison is mandatory and returned.
    """
    if model not in MODELS:
        raise ValueError("unknown model")
    g = np.asarray(grids)
    t = np.asarray(time_steps, dtype=float)
    if (g.ndim != 1 or len(g) < 3 or any(not isinstance(v, (int, np.integer)) for v in grids)
            or np.any(g < 8) or np.any(np.diff(g) <= 0) or reference_nx <= max(g)):
        raise ValueError("at least three increasing grids and a finer reference_nx required")
    if (t.ndim != 1 or len(t) < 3 or np.any(~np.isfinite(t)) or np.any(np.diff(t) >= 0)
            or np.any(t <= reference_dt) or not np.isfinite(reference_dt) or reference_dt <= 0
            or not np.isfinite(duration) or duration <= 0):
        raise ValueError("at least three decreasing timesteps > positive reference_dt required")

    def count(step_dt):
        n = int(round(duration / step_dt))
        if n < 1 or not np.isclose(n * step_dt, duration, rtol=1e-12, atol=1e-14):
            raise ValueError("duration must be divisible by every timestep")
        return n

    for step_dt in (*t, reference_dt):
        count(step_dt)
    options = dict(model=model, case=case, viscosity_s=viscosity_s, viscosity_p=viscosity_p,
                   relaxation_time=relaxation_time, amplitude=amplitude,
                   extensibility_squared=extensibility_squared, stress_diffusivity=stress_diffusivity)

    def run(size, step_dt, **overrides):
        n = count(step_dt)
        return simulate_viscoelastic_periodic(nx=int(size), dt=float(step_dt), n_steps=n,
                                              sample_every=n, **(options | overrides))

    reference = run(reference_nx, reference_dt)
    spatial = []
    for size in g:
        coarse = run(size, reference_dt)
        grid = _grid(int(size), int(size), 2 * np.pi, 2 * np.pi)
        velocity_ref = _restrict(reference["velocity"], int(size), grid)
        c_ref = _restrict(reference["conformation"], int(size), grid)
        stress_ref = _restrict(reference["stress"], int(size), grid)
        identity = np.eye(c_ref.shape[1])[:, :, None, None]
        spatial.append({"nx": int(size), "initial_velocity_relative_error": _relative(coarse["velocity"][0], velocity_ref[0]),
                        "final_velocity_relative_error": _relative(coarse["velocity"][-1], velocity_ref[-1]),
                        "final_conformation_relative_error": _relative(coarse["conformation"][-1], c_ref[-1]),
                        "final_deformation_relative_error": _relative(coarse["conformation"][-1] - identity, c_ref[-1] - identity),
                        "final_stress_relative_error": _relative(coarse["stress"][-1], stress_ref[-1]),
                        "final_conformation_rms_error": float(np.sqrt(np.mean((coarse["conformation"][-1] - c_ref[-1]) ** 2))),
                        "final_energy_rate_residual": coarse["metadata"]["diagnostics"]["total_energy_rate_residual"][-1]})
    temporal_reference = run(nx, reference_dt)
    temporal = []
    for step_dt in t:
        candidate = run(nx, step_dt)
        error = _relative(candidate["velocity"][-1], temporal_reference["velocity"][-1])
        row = {"dt": float(step_dt), "actual_dt_max": candidate["metadata"]["internal_dt_max"],
               "nominal_dt_capped": bool(candidate["metadata"]["stability_limited_steps"]),
               "velocity_relative_error": error,
               "conformation_relative_error": _relative(candidate["conformation"][-1], temporal_reference["conformation"][-1]),
               "stress_relative_error": _relative(candidate["stress"][-1], temporal_reference["stress"][-1]),
               "observed_velocity_order": None}
        if (temporal and not row["nominal_dt_capped"] and not temporal[-1]["nominal_dt_capped"]
                and error > 1e-14 and temporal[-1]["velocity_relative_error"] > 1e-14):
            row["observed_velocity_order"] = float(np.log(temporal[-1]["velocity_relative_error"] / error)
                                                    / np.log(temporal[-1]["dt"] / step_dt))
        temporal.append(row)
    # Exact source ODE and zero-polymer coupling checks are distinct tests.
    shear = integrate_uniform_shear(dt=.0025, n_steps=40, relaxation_time=.25)
    shear_error = _relative(shear["conformation"], uniform_shear_reference(shear["times"], relaxation_time=.25))
    zero = run(16, reference_dt, viscosity_p=0.)
    newtonian = simulate_periodic_ns(nx=16, case=case, viscosity=viscosity_s,
                                    amplitude=amplitude, dt=reference_dt, n_steps=count(reference_dt), sample_every=count(reference_dt))
    zero_error = _relative(zero["velocity"][-1], newtonian["velocity"][-1])
    # Weak TG avoids a spatially under-resolved small-lambda limit comparison.
    limit_viscosity_s, limit_viscosity_p = .02, .02
    newtonian_limit = simulate_periodic_ns(nx=16, case="taylor_green", viscosity=.04,
                                          amplitude=.1, dt=reference_dt, n_steps=count(reference_dt), sample_every=count(reference_dt))
    relaxation_limits = {}
    for limit_model in MODELS:
        relaxation_rows = []
        for lam in (.02, .01, .005):
            limit = simulate_viscoelastic_periodic(nx=16, model=limit_model, case="taylor_green",
                                                  viscosity_s=limit_viscosity_s, viscosity_p=limit_viscosity_p,
                                                  relaxation_time=lam, amplitude=.1, dt=reference_dt,
                                                  n_steps=count(reference_dt), sample_every=count(reference_dt),
                                                  extensibility_squared=extensibility_squared)
            relaxation_rows.append({"relaxation_time": lam,
                                    "relative_velocity_error": _relative(limit["velocity"][-1], newtonian_limit["velocity"][-1]),
                                    "actual_dt_max": limit["metadata"]["internal_dt_max"]})
        relaxation_limits[limit_model] = relaxation_rows
    diffusion_sensitivity = None
    if stress_diffusivity > 0:
        sensitivity_runs = [run(nx, reference_dt, stress_diffusivity=k)
                            for k in (0., stress_diffusivity / 2, stress_diffusivity)]
        diffusion_sensitivity = [{"stress_diffusivity": k,
                                  "relative_velocity_change_vs_zero": _relative(value["velocity"][-1], sensitivity_runs[0]["velocity"][-1]),
                                  "relative_conformation_change_vs_zero": _relative(value["conformation"][-1], sensitivity_runs[0]["conformation"][-1])}
                                 for k, value in zip((0., stress_diffusivity / 2, stress_diffusivity), sensitivity_runs)]
    diagnostics = reference["metadata"]["diagnostics"]
    return {"model": model, "case": case, "duration": float(duration), "parameters": options,
            "grid_refinement": spatial, "reference_nx": int(reference_nx),
            "temporal_refinement": temporal, "reference_dt": float(reference_dt),
            "temporal_nx": int(nx), "grid_reference_dt": float(reference_dt),
            "temporal_reference_dt": float(reference_dt),
            "numerical_reference": "same solver, finer grid/time; spatial comparison restricted to each coarse active band",
            "uniform_shear_oldroyd_b_relative_error": shear_error,
            "zero_polymer_newtonian_relative_error": zero_error,
            "oldroyd_b_small_relaxation_limit": relaxation_limits["oldroyd_b"],
            "fene_p_small_relaxation_limit": relaxation_limits["fene_p"],
            "small_relaxation_limit_scope": "weak Taylor-Green for both models, C(0)=I includes initial stress layer; fixed viscosities .02+.02",
            "finite_fields": bool(all(np.all(np.isfinite(reference[name])) for name in ("velocity", "conformation", "stress", "constitutive_rhs"))),
            "minimum_conformation_eigenvalue": min(diagnostics["min_conformation_eigenvalue"]),
            "maximum_conformation_trace": max(diagnostics["max_conformation_trace"]),
            "max_divergence_rms": max(periodic_diagnostics(reference["velocity"], reference["dx"], reference["dy"])["divergence_rms"]),
            "max_kinetic_exchange_residual": max(abs(v) for v in diagnostics["kinetic_exchange_residual"]),
            "max_total_energy_rate_residual": max(abs(v) for v in diagnostics["total_energy_rate_residual"]),
            "stress_diffusion_sensitivity": diffusion_sensitivity, "dns_certified": False,
            "limitations": ["same-solver numerical references require further refinement before a DNS claim",
                            "FENE rational constitutive factor is filtered; residual aliasing assessed by spatial refinement",
                            "positive grid-point conformation is checked; positivity between collocation points is not certified",
                            "no high-Weissenberg, chaotic flow, walls or cylinder wake validation"]}
