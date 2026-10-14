"""Verified planar reference problems for non-Newtonian reconstruction studies.

Herschel--Bulkley profiles are analytical. Oldroyd-B startup is a spatially
resolved, fully developed 1-D reduction, not a 2-D obstacle/wake DNS. SI units
are used unless the caller deliberately supplies consistent nondimensional
parameters. The streamwise direction is periodic; channel walls are y nodes.
"""
from __future__ import annotations

from collections.abc import Callable
import math

import numpy as np


def _positive(name: str, value: float, *, allow_zero: bool = False) -> float:
    value = float(value)
    if not np.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        suffix = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {suffix}")
    return value


def _integer(name: str, value: int, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _require_finite(name: str, *values) -> None:
    if any(not np.all(np.isfinite(value)) for value in values):
        raise FloatingPointError(f"nonfinite {name}; rescale parameters or refine the calculation")


def _grid(ny: int, nx: int, half_height: float, length: float):
    ny, nx = _integer("ny", ny, 3), _integer("nx", nx, 2)
    half_height, length = _positive("half_height", half_height), _positive("length", length)
    y = np.linspace(-half_height, half_height, ny)
    return y, length / nx, 2 * half_height / (ny - 1)


def _times(times: np.ndarray | None) -> np.ndarray:
    value = np.asarray([0.0] if times is None else times, dtype=float)
    if value.ndim != 1 or value.size == 0 or not np.all(np.isfinite(value)):
        raise ValueError("times must be a nonempty, finite 1-D sequence")
    if np.any(value < 0) or np.any(np.diff(value) <= 0):
        raise ValueError("times must be nonnegative and strictly increasing")
    return value


def _velocity(u: np.ndarray, nx: int) -> np.ndarray:
    value = np.zeros((len(u), 2, u.shape[1], nx), dtype=float)
    value[:, 0] = u[:, :, None]
    return value


def _base_metadata(y: np.ndarray, density: float) -> dict:
    return {
        "grid_convention": "periodic_x_nodal_y", "periodic_axes": ["x"],
        "wall_indices": [0, len(y) - 1], "y": y.tolist(),
        "units": {"length": "m", "time": "s", "velocity": "m/s",
                  "density": "kg/m^3", "stress": "Pa", "pressure_gradient": "Pa/m"},
        "density": density, "streamwise_homogeneous": True,
        "scope": "fully developed planar channel reference, no obstacle or wake",
    }


def herschel_bulkley_profile(
    y: np.ndarray, *, half_height: float = 1.0, pressure_gradient: float = 1.0,
    consistency: float = 1.0, flow_index: float = 0.6, yield_stress: float = 0.2,
) -> np.ndarray:
    """Exact no-slip pressure-driven profile, including an unyielded plug.

    ``pressure_gradient`` is the signed driving force ``-dp/dx`` (Pa/m),
    ``consistency`` has units Pa s**n, and ``yield_stress`` is in Pa. Below the
    wall yield threshold ``abs(G)*H <= tau_y``, velocity is identically zero.
    With n=1 and tau_y=0 this is Newtonian Poiseuille flow, viscosity K.
    """
    h = _positive("half_height", half_height)
    k = _positive("consistency", consistency)
    n = _positive("flow_index", flow_index)
    tau_y = _positive("yield_stress", yield_stress, allow_zero=True)
    gradient = float(pressure_gradient)
    yy = np.asarray(y, dtype=float)
    if not np.isfinite(gradient) or not np.all(np.isfinite(yy)):
        raise ValueError("pressure_gradient and y must be finite")
    if np.any(np.abs(yy) > h + 1e-12 * h):
        raise ValueError("y must be inside [-half_height, half_height]")
    g = abs(gradient)
    if g == 0 or g * h <= tau_y:
        return np.zeros_like(yy)
    exponent = 1 + 1 / n
    return (np.sign(gradient) * n / (n + 1) / g / k ** (1 / n)
            * ((g * h - tau_y) ** exponent
               - np.maximum(g * np.abs(yy) - tau_y, 0) ** exponent))


def herschel_bulkley_channel(
    *, ny: int = 65, nx: int = 16, half_height: float = 1.0, length: float = 2.0,
    pressure_gradient: float = 1.0, consistency: float = 1.0,
    flow_index: float = 0.6, yield_stress: float = 0.2,
    times: np.ndarray | None = None, density: float = 1.0,
) -> dict:
    """Return analytical HB/Bingham snapshots under the research data contract.

    This stationary bank tests spatial denoising and physical observables;
    identical clean snapshots do not establish temporal modal complexity.
    """
    y, dx, dy = _grid(ny, nx, half_height, length)
    rho = _positive("density", density)
    tt = _times(times)
    profile = herschel_bulkley_profile(
        y, half_height=half_height, pressure_gradient=pressure_gradient,
        consistency=consistency, flow_index=flow_index, yield_stress=yield_stress,
    )
    g, k, n, tau_y = abs(float(pressure_gradient)), float(consistency), float(flow_index), float(yield_stress)
    shear_stress = -float(pressure_gradient) * y
    shear_rate = np.sign(shear_stress) * (np.maximum(np.abs(shear_stress) - tau_y, 0) / k) ** (1 / n)
    plug = np.abs(shear_stress) <= tau_y
    # At zero shear, the HB relation is the yield inequality, not tau=0.
    constitutive = np.where(
        plug, np.maximum(np.abs(shear_stress) - tau_y, 0),
        shear_stress - np.sign(shear_rate) * (tau_y + k * np.abs(shear_rate) ** n),
    )
    exponent = 1 / n
    wall_excess = max(g * half_height - tau_y, 0)
    if g > 0 and wall_excess > 0:
        # Integral Q = 2 integral_0^H y*((G*y-tau_y)/K)**(1/n) dy.
        flow_rate = (2 * np.sign(pressure_gradient) / (g * g * k ** exponent)
                     * (wall_excess ** (exponent + 2) / (exponent + 2)
                        + tau_y * wall_excess ** (exponent + 1) / (exponent + 1)))
    else:
        flow_rate = 0.0
    metadata = _base_metadata(y, rho)
    metadata.update(
        case="herschel_bulkley_channel", reference="analytical steady solution",
        rheology="Bingham" if n == 1 and tau_y > 0 else "Herschel-Bulkley",
        parameters={"half_height": half_height, "pressure_gradient": pressure_gradient,
                    "consistency": consistency, "flow_index": flow_index,
                    "yield_stress": yield_stress},
        consistency_units=f"Pa s^{n:g}", plug_half_width=min(half_height, tau_y / g) if g else half_height,
        analytical_flow_rate_per_width=flow_rate,
        constitutive_residual_max_pa=float(np.max(np.abs(constitutive))),
        limitation="stationary clean snapshots; not a vortex-shedding or transient benchmark",
    )
    shape = (len(tt), ny, nx)
    return {
        "velocity": _velocity(np.tile(profile, (len(tt), 1)), nx),
        "times": tt, "dx": dx, "dy": dy, "density": rho,
        "mask": np.zeros((ny, nx), dtype=bool), "boundary": "channel", "metadata": metadata,
        "stress_xy": np.broadcast_to(shear_stress[None, :, None], shape).copy(),
        "shear_rate": np.broadcast_to(shear_rate[None, :, None], shape).copy(),
        "plug_mask": np.broadcast_to(plug[:, None], (ny, nx)).copy(),
        "constitutive_residual": np.broadcast_to(constitutive[None, :, None], shape).copy(),
    }


def newtonian_startup_profile(
    y: np.ndarray, time: float, *, half_height: float = 1.0,
    pressure_gradient: float = 1.0, viscosity: float = 1.0,
    density: float = 1.0, modes: int = 256,
) -> np.ndarray:
    """Fourier-series solution of sudden pressure-driven Newtonian startup."""
    h, eta, rho = (_positive("half_height", half_height),
                   _positive("viscosity", viscosity), _positive("density", density))
    time = _positive("time", time, allow_zero=True)
    modes = _integer("modes", modes, 1)
    yy = np.asarray(y, dtype=float)
    if not np.all(np.isfinite(yy)) or np.any(np.abs(yy) > h + 1e-12 * h):
        raise ValueError("y must be finite and inside the channel")
    g = float(pressure_gradient)
    if not np.isfinite(g):
        raise ValueError("pressure_gradient must be finite")
    if time == 0:
        return np.zeros_like(yy)
    odd = 2 * np.arange(modes) + 1
    wave = odd * np.pi / (2 * h)
    coefficients = ((-1.0) ** np.arange(modes) / odd ** 3
                    * np.exp(-eta / rho * wave ** 2 * time))
    transient = (16 * g * h * h / (eta * np.pi ** 3)
                 * np.sum(coefficients[:, None] * np.cos(wave[:, None] * yy.ravel()), axis=0))
    return (g / (2 * eta) * (h * h - yy.ravel() ** 2) - transient).reshape(yy.shape)


def _face_to_nodes(value: np.ndarray) -> np.ndarray:
    """Second-order interpolation, with linear extrapolation at wall nodes."""
    result = np.empty(value.shape[:-1] + (value.shape[-1] + 1,))
    result[..., 1:-1] = (value[..., :-1] + value[..., 1:]) * 0.5
    result[..., 0] = 1.5 * value[..., 0] - 0.5 * value[..., 1]
    result[..., -1] = 1.5 * value[..., -1] - 0.5 * value[..., -2]
    return result


def _tridiagonal_factor(diagonal: float, off_diagonal: float, count: int):
    inverse = np.empty(count)
    upper = np.empty(max(0, count - 1))
    inverse[0] = 1 / diagonal
    if count > 1:
        upper[0] = off_diagonal * inverse[0]
    for i in range(1, count):
        inverse[i] = 1 / (diagonal - off_diagonal * upper[i - 1])
        if i < count - 1:
            upper[i] = off_diagonal * inverse[i]
    return inverse, upper


def _tridiagonal_solve(rhs, off_diagonal, inverse, upper):
    solution = np.empty_like(rhs)
    solution[0] = rhs[0] * inverse[0]
    for i in range(1, len(rhs)):
        solution[i] = (rhs[i] - off_diagonal * solution[i - 1]) * inverse[i]
    for i in range(len(rhs) - 2, -1, -1):
        solution[i] -= upper[i] * solution[i + 1]
    return solution


@np.errstate(over="raise", invalid="raise", divide="raise")
def oldroyd_b_channel(
    *, ny: int = 65, nx: int = 16, half_height: float = 1.0, length: float = 2.0,
    density: float = 1.0, eta_s: float = 0.5, eta_p: float = 0.5,
    relaxation_time: float = 0.5, dt: float = 0.01, steps: int = 300,
    snapshot_stride: int = 5,
    pressure_gradient: float | Callable[[float], float] = 1.0,
    initial: str = "rest",
) -> dict:
    """Solve actual coupled velocity/polymer-stress planar Oldroyd-B startup.

    rho*u_t = eta_s*u_yy + (tau_xy)_y + G(t)
    lambda*(tau_xy)_t + tau_xy = eta_p*u_y
    lambda*(tau_xx)_t + tau_xx = 2*lambda*u_y*tau_xy; tau_yy=0.

    No-slip velocity lives at y nodes and polymer stress at cell faces.
    Coupled backward Euler is first order in time and second order in space;
    tau_xy is eliminated before one tridiagonal velocity solve. lambda=0 is
    handled exactly as Newtonian viscosity eta_s+eta_p. Residuals are measured
    at every internal step (not by differencing sparsely saved snapshots).
    """
    y, dx, dy = _grid(ny, nx, half_height, length)
    rho = _positive("density", density)
    solvent = _positive("eta_s", eta_s, allow_zero=True)
    polymer = _positive("eta_p", eta_p, allow_zero=True)
    if solvent + polymer == 0:
        raise ValueError("eta_s + eta_p must be positive")
    lam = _positive("relaxation_time", relaxation_time, allow_zero=True)
    dt = _positive("dt", dt)
    steps = _integer("steps", steps, 1)
    stride = _integer("snapshot_stride", snapshot_stride, 1)
    if initial not in {"rest", "steady"}:
        raise ValueError("initial must be 'rest' or 'steady'")
    force = pressure_gradient if callable(pressure_gradient) else lambda _t: float(pressure_gradient)

    def get_force(t):
        g = float(force(t))
        if not np.isfinite(g):
            raise ValueError("pressure_gradient must return finite values")
        return g

    g0 = get_force(0.0)
    u = np.zeros(ny)
    if initial == "steady":
        u = g0 * (half_height ** 2 - y ** 2) / (2 * (solvent + polymer))
    tau = polymer * np.diff(u) / dy
    tau_xx = 2 * lam * polymer * (np.diff(u) / dy) ** 2
    _require_finite("initial velocity/stress", u, tau, tau_xx)
    _require_finite("relaxation denominator", lam + dt)
    a, b = lam / (lam + dt), polymer * dt / (lam + dt)
    effective = solvent + b
    off_diagonal = -effective / dy ** 2
    diagonal = rho / dt - 2 * off_diagonal
    _require_finite("discrete system coefficients", a, b, effective, off_diagonal, diagonal)
    factors = _tridiagonal_factor(diagonal, off_diagonal, ny - 2)
    _require_finite("tridiagonal factorization", *factors)
    stored_steps = sorted(set([0, steps] + list(range(stride, steps + 1, stride))))
    stored_set = set(stored_steps)
    u_history, tau_history, xx_history = [u.copy()], [tau.copy()], [tau_xx.copy()]
    xy_residuals, xx_residuals, momentum_residuals = [np.zeros(ny - 1)], [np.zeros(ny - 1)], [np.zeros(ny - 2)]
    forces = [g0]
    max_xy = max_xx = max_momentum = 0.0
    min_det = math.inf

    def conformation_det(txy, txx):
        if polymer == 0 or lam == 0:
            return 1.0
        determinants = 1 + lam / polymer * txx - (lam / polymer * txy) ** 2
        _require_finite("conformation determinant", determinants)
        minimum = float(np.min(determinants))
        if minimum <= 0:
            raise FloatingPointError("non-positive Oldroyd-B conformation; calculation is inadmissible")
        return minimum

    min_det = min(min_det, conformation_det(tau, tau_xx))
    for step in range(1, steps + 1):
        g = get_force(step * dt)
        previous_u, previous_tau, previous_xx = u.copy(), tau.copy(), tau_xx.copy()
        rhs = rho / dt * previous_u[1:-1] + a * np.diff(previous_tau) / dy + g
        u[1:-1] = _tridiagonal_solve(rhs, off_diagonal, *factors)
        gradient = np.diff(u) / dy
        tau = a * previous_tau + b * gradient
        tau_xx = a * previous_xx + 2 * a * dt * gradient * tau
        _require_finite(f"velocity/stress at step {step}", u, tau, tau_xx)
        rxy = lam * (tau - previous_tau) / dt + tau - polymer * gradient
        rxx = lam * (tau_xx - previous_xx) / dt + tau_xx - 2 * lam * gradient * tau
        rmomentum = (rho * (u[1:-1] - previous_u[1:-1]) / dt
                     - solvent * np.diff(u, n=2) / dy ** 2 - np.diff(tau) / dy - g)
        _require_finite(f"equation residuals at step {step}", rxy, rxx, rmomentum)
        max_xy, max_xx = max(max_xy, float(np.max(np.abs(rxy)))), max(max_xx, float(np.max(np.abs(rxx))))
        max_momentum = max(max_momentum, float(np.max(np.abs(rmomentum))))
        min_det = min(min_det, conformation_det(tau, tau_xx))
        if step in stored_set:
            u_history.append(u.copy())
            tau_history.append(tau.copy())
            xx_history.append(tau_xx.copy())
            xy_residuals.append(rxy.copy())
            xx_residuals.append(rxx.copy())
            momentum_residuals.append(rmomentum.copy())
            forces.append(g)
    uu, txy, txx = np.asarray(u_history), np.asarray(tau_history), np.asarray(xx_history)
    saved_t = np.asarray(stored_steps) * dt
    polymer_tensor = np.zeros((len(saved_t), 2, 2, ny, nx))
    polymer_tensor[:, 0, 0] = _face_to_nodes(txx)[:, :, None]
    polymer_tensor[:, 0, 1] = polymer_tensor[:, 1, 0] = _face_to_nodes(txy)[:, :, None]
    total_shear = _face_to_nodes(solvent * np.diff(uu, axis=1) / dy + txy)
    _require_finite("exported stress/velocity", uu, polymer_tensor, total_shear)
    metadata = _base_metadata(y, rho)
    metadata.update(
        case="oldroyd_b_channel", rheology="Oldroyd-B",
        reference="numerical fully developed 1-D upper-convected Oldroyd-B reduction",
        parameters={"half_height": half_height, "eta_s": solvent, "eta_p": polymer,
                    "relaxation_time": lam, "dt": dt, "steps": steps, "initial": initial,
                    "pressure_gradient": "callable" if callable(pressure_gradient) else float(pressure_gradient)},
        numerical_method="coupled backward Euler; staggered second-order centered y differences",
        temporal_order=1, spatial_order=2, stress_location="faces; tensors interpolated to y nodes",
        constitutive_xy_residual_max_pa=max_xy, constitutive_xx_residual_max_pa=max_xx,
        momentum_residual_max_pa_per_m=max_momentum,
        min_conformation_determinant_faces=min_det,
        beta=solvent / (solvent + polymer),
        limitation="no streamwise variations, obstacle, vortex shedding, elastic instability, or chaos",
    )
    if not callable(pressure_gradient):
        scale_u = abs(float(pressure_gradient)) * half_height ** 2 / (2 * (solvent + polymer))
        metadata["Re_half_height_centerline"] = rho * scale_u * half_height / (solvent + polymer)
        metadata["Wi_half_height_centerline"] = lam * scale_u / half_height
    return {
        "velocity": _velocity(uu, nx), "times": saved_t, "dx": dx, "dy": dy,
        "density": rho, "mask": np.zeros((ny, nx), dtype=bool),
        "boundary": "channel", "metadata": metadata,
        "polymer_stress": polymer_tensor,
        "stress_xy": np.broadcast_to(total_shear[:, :, None], (len(saved_t), ny, nx)).copy(),
        "stress_xy_faces": txy, "stress_xx_faces": txx,
        "constitutive_residual": np.asarray(xy_residuals),
        "normal_constitutive_residual": np.asarray(xx_residuals),
        "momentum_residual": np.asarray(momentum_residuals),
        "pressure_gradient": np.asarray(forces),
    }


def channel_convergence_study(
    *, relaxation_time: float = 0.5, eta_s: float = 0.5, eta_p: float = 0.5,
    end_time: float = 1.0, coarse_ny: int = 33, coarse_dt: float = 0.02,
    levels: int = 3, refinement: str = "combined", **parameters,
) -> dict:
    """Compute refinement evidence, never equating solver residual to accuracy.

    Space-only and time-only studies can isolate their error contributions.
    Combined refinement records changes but cannot estimate either order in
    isolation. Last/fine run is a comparison reference, not an exact truth.
    For Newtonian limits, additionally compare the final profile to the
    independent Fourier-series analytical solution.
    """
    coarse_ny = _integer("coarse_ny", coarse_ny, 3)
    levels = _integer("levels", levels, 2)
    end_time, coarse_dt = _positive("end_time", end_time), _positive("coarse_dt", coarse_dt)
    if refinement not in {"space", "time", "combined"}:
        raise ValueError("refinement must be 'space', 'time', or 'combined'")
    for reserved in ("ny", "nx", "steps", "dt", "snapshot_stride", "initial"):
        if reserved in parameters:
            raise ValueError(f"{reserved} is controlled by the convergence study")
    runs, records = [], []
    base_steps = max(1, int(math.ceil(end_time / coarse_dt)))
    for level in range(levels):
        scale = 2 ** level
        ny = (coarse_ny - 1) * scale + 1 if refinement != "time" else coarse_ny
        steps = base_steps * scale if refinement != "space" else base_steps
        data = oldroyd_b_channel(
            ny=ny, nx=2, dt=end_time / steps, steps=steps, snapshot_stride=steps,
            relaxation_time=relaxation_time, eta_s=eta_s, eta_p=eta_p,
            initial="rest", **parameters,
        )
        runs.append(data)
        row = {"ny": ny, "dt": end_time / steps, "steps": steps,
               "constitutive_residual_max_pa": data["metadata"]["constitutive_xy_residual_max_pa"],
               "momentum_residual_max_pa_per_m": data["metadata"]["momentum_residual_max_pa_per_m"]}
        if (relaxation_time == 0 or eta_p == 0) and not callable(parameters.get("pressure_gradient", 1.0)):
            reference = newtonian_startup_profile(
                np.asarray(data["metadata"]["y"]), end_time,
                half_height=float(parameters.get("half_height", 1.0)),
                pressure_gradient=float(parameters.get("pressure_gradient", 1.0)),
                viscosity=eta_s + eta_p, density=float(parameters.get("density", 1.0)),
            )
            difference = data["velocity"][-1, 0, :, 0] - reference
            norm = np.linalg.norm(reference)
            row["relative_l2_error_vs_analytical"] = float(np.linalg.norm(difference) / norm) if norm else float(np.linalg.norm(difference))
        records.append(row)
    fine = runs[-1]
    for i, data in enumerate(runs[:-1]):
        sampling = (fine["velocity"].shape[2] - 1) // (data["velocity"].shape[2] - 1)
        reference = fine["velocity"][-1, 0, ::sampling, 0]
        difference = data["velocity"][-1, 0, :, 0] - reference
        norm = np.linalg.norm(reference)
        records[i]["relative_l2_change_vs_finest"] = float(np.linalg.norm(difference) / norm) if norm else float(np.linalg.norm(difference))
        stress_reference = fine["polymer_stress"][-1, :, :, ::sampling, 0]
        stress_difference = data["polymer_stress"][-1, :, :, :, 0] - stress_reference
        stress_norm = np.linalg.norm(stress_reference)
        records[i]["relative_l2_stress_change_vs_finest"] = float(np.linalg.norm(stress_difference) / stress_norm) if stress_norm else float(np.linalg.norm(stress_difference))
    return {"refinement": refinement, "end_time": end_time, "levels": records,
            "interpretation": "independent spatial/time studies needed; finest numerical run is not exact truth"}


def reference_rheology_bank(*, ny: int = 65, nx: int = 16, frames: int = 31) -> dict[str, dict]:
    """Small reproducible bank for reconstruction experiments and notebooks."""
    frames = _integer("frames", frames, 2)
    times = np.linspace(0, 3, frames)
    bank = {
        "hb_shear_thinning": herschel_bulkley_channel(ny=ny, nx=nx, flow_index=.6, times=times),
        "bingham": herschel_bulkley_channel(ny=ny, nx=nx, flow_index=1, times=times),
        "hb_shear_thickening": herschel_bulkley_channel(ny=ny, nx=nx, flow_index=1.4, times=times),
        "newtonian_analytic": herschel_bulkley_channel(ny=ny, nx=nx, flow_index=1, yield_stress=0, times=times),
        "hb_unyielded": herschel_bulkley_channel(ny=ny, nx=nx, yield_stress=1.2, times=times),
    }
    # Exactly frames snapshots including t=0; 10 implicit steps per interval.
    steps = (frames - 1) * 10
    for name, lam in (("oldroyd_b_startup_low_wi", .1), ("oldroyd_b_startup_high_wi", 1.0)):
        bank[name] = oldroyd_b_channel(ny=ny, nx=nx, relaxation_time=lam,
                                     dt=3 / steps, steps=steps, snapshot_stride=10)
    bank["oldroyd_b_oscillatory"] = oldroyd_b_channel(
        ny=ny, nx=nx, relaxation_time=.5, dt=3 / steps, steps=steps,
        snapshot_stride=10, pressure_gradient=lambda t: np.sin(2 * np.pi * t),
    )
    return bank


def rheology_validation_report() -> dict:
    """Recalculate resolution evidence matched to the 3-second pilot channel.

    Returns only JSON-compatible scalar evidence. The same 31 saved times are
    compared at all resolutions, for velocity and interpolated polymer stress.
    The finest Oldroyd-B run is numerical; the Newtonian reference is analytical.
    """
    parameters = {"density": 1.0, "eta_s": .5, "eta_p": .5,
                  "relaxation_time": .5, "pressure_gradient": 1.0,
                  "half_height": 1.0, "length": 2.0}
    runs, levels = [], []
    for scale in (1, 2, 4):
        ny, dt, steps, stride = 32 * scale + 1, .01 / scale, 300 * scale, 10 * scale
        data = oldroyd_b_channel(ny=ny, nx=8, dt=dt, steps=steps,
                                snapshot_stride=stride, **parameters)
        runs.append(data)
        limits = dict(parameters)
        limits["relaxation_time"] = 0.0
        newtonian = oldroyd_b_channel(ny=ny, nx=8, dt=dt, steps=steps,
                                     snapshot_stride=stride, **limits)
        y = np.asarray(data["metadata"]["y"])
        exact = np.asarray([newtonian_startup_profile(y, time) for time in data["times"]])
        numerical = newtonian["velocity"][:, 0, :, 0]
        levels.append({
            "ny": ny, "nx": 8, "dx": data["dx"], "dy": data["dy"],
            "dt": dt, "steps": steps, "snapshot_stride": stride, "snapshots": len(data["times"]),
            "newtonian_limit_relative_l2_velocity_error_all_saved_times": float(np.linalg.norm(numerical-exact)/np.linalg.norm(exact)),
            "newtonian_limit_relative_l2_velocity_error_final": float(np.linalg.norm(numerical[-1]-exact[-1])/np.linalg.norm(exact[-1])),
            "constitutive_xy_residual_max_pa": data["metadata"]["constitutive_xy_residual_max_pa"],
            "constitutive_xx_residual_max_pa": data["metadata"]["constitutive_xx_residual_max_pa"],
            "momentum_residual_max_pa_per_m": data["metadata"]["momentum_residual_max_pa_per_m"],
            "min_conformation_determinant_faces": data["metadata"]["min_conformation_determinant_faces"],
            "no_slip_velocity_max_m_per_s": float(np.max(np.abs(data["velocity"][:, :, [0, -1]]))),
        })
    finest = runs[-1]
    for data, row in zip(runs[:-1], levels[:-1]):
        sampling = (finest["velocity"].shape[2] - 1) // (data["velocity"].shape[2] - 1)
        velocity_reference = finest["velocity"][:, 0, ::sampling, 0]
        difference = data["velocity"][:, 0, :, 0] - velocity_reference
        stress_reference = finest["polymer_stress"][:, :, :, ::sampling, 0]
        stress_difference = data["polymer_stress"][:, :, :, :, 0] - stress_reference
        row.update(
            oldroyd_b_relative_l2_velocity_change_vs_finest_all_saved_times=float(np.linalg.norm(difference)/np.linalg.norm(velocity_reference)),
            oldroyd_b_relative_l2_velocity_change_vs_finest_final=float(np.linalg.norm(difference[-1])/np.linalg.norm(velocity_reference[-1])),
            oldroyd_b_relative_l2_polymer_stress_change_vs_finest_all_saved_times=float(np.linalg.norm(stress_difference)/np.linalg.norm(stress_reference)),
            oldroyd_b_relative_l2_polymer_stress_change_vs_finest_final=float(np.linalg.norm(stress_difference[-1])/np.linalg.norm(stress_reference[-1])),
        )
    hb = []
    for n in (.6, 1., 1.4):
        for ny in (33, 65, 129):
            data = herschel_bulkley_channel(ny=ny, nx=8, flow_index=n)
            u = data["velocity"][0, 0, :, 0]
            numerical_q = float(np.sum((u[1:]+u[:-1])*.5*data["dy"]))
            q = float(data["metadata"]["analytical_flow_rate_per_width"])
            hb.append({"flow_index": n, "ny": ny, "nx": 8,
                       "dx": data["dx"], "dy": data["dy"],
                       "analytical_flow_rate_per_width_m2_per_s": q,
                       "numerical_trapezoidal_flow_rate_per_width_m2_per_s": numerical_q,
                       "relative_flow_rate_quadrature_error": abs(numerical_q-q)/abs(q),
                       "constitutive_residual_max_pa": data["metadata"]["constitutive_residual_max_pa"]})
    return {
        "schema_version": 1, "generator": "fluid_denoise.research.rheology.rheology_validation_report",
        "scope": "matched fully developed 1-D channel pilot; not external wake DNS validation",
        "oldroyd_b_parameters": parameters, "duration_s": 3.0,
        "saved_times_s": runs[0]["times"].tolist(), "oldroyd_b_and_newtonian_levels": levels,
        "herschel_bulkley_parameters": {"half_height": 1.0, "pressure_gradient": 1.0,
                                      "consistency": 1.0, "yield_stress": .2},
        "herschel_bulkley_quadrature_levels": hb,
        "interpretation": "combined space/time refinement; finest Oldroyd-B run is numerical, not exact. Stress comparisons use nodal interpolation. HB field is exact; Q error is quadrature error.",
    }
