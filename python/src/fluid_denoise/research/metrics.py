"""Physical diagnostics for collocated two-dimensional velocity snapshots.

The reference is a dataset with velocity shaped (time, component, y, x),
times, dx, dy, boundary, and an optional obstacle mask (True = solid).
These diagnostics measure reconstruction fidelity; they do not certify DNS
resolution, replace a solver's discrete conservation test, or infer drag.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def _relative_l2(reference: np.ndarray, candidate: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(reference.ravel()))
    numerator = float(np.linalg.norm((candidate - reference).ravel()))
    return numerator / denominator if denominator > 0.0 else (0.0 if numerator == 0.0 else None)


def _rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(value))))


def _correlation(reference: np.ndarray, candidate: np.ndarray) -> float | None:
    a, b = reference.ravel(), candidate.ravel()
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return 1.0 if np.array_equal(a, b) else None
    return float(np.corrcoef(a, b)[0, 1])


def spatial_derivative(field: np.ndarray, spacing: float, axis: int,
                       periodic: bool) -> np.ndarray:
    """Spectral derivative on periodic axes, second-order differences otherwise.

    Periodic grids exclude a duplicated endpoint. Nyquist derivatives are zero
    for real even-length input, as required by a real trigonometric interpolant.
    """
    values = np.asarray(field, dtype=float)
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("spacing must be finite and positive")
    axis %= values.ndim
    if values.shape[axis] < 3:
        raise ValueError("spatial axes must have at least three points")
    if not periodic:
        return np.gradient(values, spacing, axis=axis, edge_order=2)
    wave_numbers = 2 * np.pi * np.fft.fftfreq(values.shape[axis], d=spacing)
    if values.shape[axis] % 2 == 0:
        wave_numbers[values.shape[axis] // 2] = 0.0
    shape = [1] * values.ndim
    shape[axis] = len(wave_numbers)
    return np.fft.ifft(1j * wave_numbers.reshape(shape)
                       * np.fft.fft(values, axis=axis), axis=axis).real


def _density(value: Any, shape: tuple[int, ...]) -> np.ndarray:
    try:
        density = np.broadcast_to(np.asarray(value, dtype=float), shape)
    except ValueError as error:
        raise ValueError("density must broadcast to (time, y, x)") from error
    return density


def _integration_weights(n: int, periodic: bool, spacing: float) -> np.ndarray:
    weights = np.full(n, spacing, dtype=float)
    if not periodic:
        weights[[0, -1]] *= 0.5
    return weights


def _valid_derivatives(mask: np.ndarray, px: bool, py: bool) -> np.ndarray:
    """Remove one-cell solid halos and nonperiodic exterior boundary nodes."""
    valid = ~mask.copy()
    # A spectral derivative is nonlocal. With obstacles, we instead use local
    # finite differences below, for which a one-cell halo is sufficient.
    for axis, periodic in [(0, py), (1, px)]:
        for offset in (-1, 1):
            shifted = np.roll(mask, offset, axis=axis)
            if not periodic:
                edge = [slice(None)] * 2
                edge[axis] = 0 if offset == 1 else -1
                shifted[tuple(edge)] = False
            valid &= ~shifted
        if not periodic:
            edge = [slice(None)] * 2
            edge[axis] = [0, -1]
            valid[tuple(edge)] = False
    return valid


def physical_metrics(reference: Mapping[str, Any],
                     candidate_velocity: np.ndarray | Mapping[str, Any],
                     candidate_density: Any = None) -> dict[str, Any]:
    """Compare physical fields, conservation residuals, and integral quantities.

    All outputs are JSON-compatible. Undefined relative errors/correlations are
    None (never NaN/Infinity). Units use the physical units supplied by the
    dataset; no conversion from lattice units is presumed. Channel defaults to
    periodic x and nodal y; open defaults to nodal x/y. ``metadata.periodic_axes``
    can override these choices. Scalar/2-D density broadcasts across time.
    Missing reference density remains unknown: kinematic diagnostics are
    available, but mass, energy, mass flux and continuity are returned as None.
    """
    ref = np.asarray(reference["velocity"], dtype=float)
    candidate_mapping = isinstance(candidate_velocity, Mapping)
    if candidate_mapping:
        if candidate_density is None:
            candidate_density = candidate_velocity.get("density")
        candidate_velocity = candidate_velocity["velocity"]
    cand = np.asarray(candidate_velocity, dtype=float)
    if ref.ndim != 4 or ref.shape[0] < 1 or ref.shape[1] != 2 or min(ref.shape[-2:]) < 3:
        raise ValueError("velocity must have shape (time, 2, y>=3, x>=3)")
    if cand.shape != ref.shape:
        raise ValueError("candidate and reference velocity shapes must agree")
    times = np.asarray(reference["times"], dtype=float)
    if times.shape != (ref.shape[0],) or not np.all(np.isfinite(times)):
        raise ValueError("times must be a finite vector matching snapshots")
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ValueError("times must be strictly increasing")
    dx, dy = float(reference["dx"]), float(reference["dy"])
    if not all(np.isfinite(s) and s > 0 for s in (dx, dy)):
        raise ValueError("dx and dy must be positive and finite")
    boundary = reference.get("boundary")
    if boundary not in {"periodic", "channel", "open"}:
        raise ValueError("boundary must be periodic, channel, or open")
    metadata = reference.get("metadata", {})
    periodic_axes = metadata.get("periodic_axes", ["x", "y"] if boundary == "periodic"
                                 else ["x"] if boundary == "channel" else [])
    if any(axis not in {"x", "y"} for axis in periodic_axes):
        raise ValueError("periodic_axes may contain only x and y")
    px, py = "x" in periodic_axes, "y" in periodic_axes
    mask = np.asarray(reference.get("mask", np.zeros(ref.shape[-2:], bool)), dtype=bool)
    if mask.shape != ref.shape[-2:]:
        raise ValueError("mask must have shape (y, x), True for obstacles")
    fluid = ~mask
    if not np.any(fluid):
        raise ValueError("at least one fluid node is required")
    if not np.all(np.isfinite(ref[:, :, fluid])) or not np.all(np.isfinite(cand[:, :, fluid])):
        raise ValueError("velocity must be finite at fluid nodes")
    # A field mapping explicitly represents the candidate dataset. Missing
    # density in that mapping stays unknown; velocity-only ndarray fits keep
    # the legacy, explicitly reported reference-density assumption.
    density_available = (reference.get("density") is not None
                         and not (candidate_mapping and candidate_density is None))
    density_shape = ref.shape[:1] + ref.shape[-2:]
    density_assumed = density_available and candidate_density is None
    # Zero placeholders permit the common derivative/integration code to run;
    # no density-dependent result is exposed unless density was supplied.
    rho_ref = (_density(reference["density"], density_shape) if density_available
               else np.zeros(density_shape))
    rho_cand = (_density(rho_ref if density_assumed else candidate_density, density_shape)
                if density_available else np.zeros(density_shape))
    if density_available:
        for rho in (rho_ref, rho_cand):
            if not np.all(np.isfinite(rho[:, fluid])) or np.any(rho[:, fluid] <= 0):
                raise ValueError("density must be positive and finite at fluid nodes")
    elif candidate_density is not None:
        supplied_density = _density(candidate_density, density_shape)
        if (not np.isfinite(supplied_density[:, fluid]).all()
                or np.any(supplied_density[:, fluid] <= 0)):
            raise ValueError("density must be positive and finite at fluid nodes")
    warnings: list[str] = []
    if not density_available:
        warnings.append("Reference or candidate density is unknown; mass, kinetic energy, mass flux and continuity metrics are unavailable. No density is inferred.")
    if density_assumed:
        warnings.append("Candidate density is copied from the reference; mass fidelity is conditional on this assumption.")
    constant_density = bool(density_available and np.all(rho_ref[:, fluid] == rho_ref[:, fluid][0, 0])
                            and np.all(rho_cand[:, fluid] == rho_cand[:, fluid][0, 0]))
    if constant_density:
        warnings.append("Constant-density total mass is trivial for velocity-only reconstruction; use divergence and flux residuals.")
    if np.any(mask):
        warnings.append("Obstacle grid uses finite differences and a one-cell derivative halo; integral area is a grid-mask approximation. Solid-wall flux is assumed zero and is not separately integrated.")
    ref = np.where(mask[None, None], 0.0, ref)
    cand = np.where(mask[None, None], 0.0, cand)
    rho_ref = np.where(mask[None], 0.0, rho_ref)
    rho_cand = np.where(mask[None], 0.0, rho_cand)
    valid = _valid_derivatives(mask, px, py)
    if not np.any(valid):
        raise ValueError("no fluid interior remains after derivative halo exclusion")

    def deriv(a: np.ndarray, axis: str) -> np.ndarray:
        spacing, index, periodic = (dx, -1, px) if axis == "x" else (dy, -2, py)
        if not np.any(mask):
            return spatial_derivative(a, spacing, index, periodic)
        if periodic:
            return (np.roll(a, -1, axis=index) - np.roll(a, 1, axis=index)) / (2 * spacing)
        return spatial_derivative(a, spacing, index, False)

    def flow_diagnostics(velocity: np.ndarray, rho: np.ndarray) -> tuple[np.ndarray, ...]:
        u, v = velocity[:, 0], velocity[:, 1]
        divergence = deriv(u, "x") + deriv(v, "y")
        vorticity = deriv(v, "x") - deriv(u, "y")
        mass_divergence = deriv(rho * u, "x") + deriv(rho * v, "y")
        if len(times) > 1:
            density_rate = np.gradient(rho, times, axis=0, edge_order=2 if len(times) > 2 else 1)
            continuity = density_rate + mass_divergence
        else:
            continuity = None
        return divergence, vorticity, continuity

    div_ref, vort_ref, continuity_ref = flow_diagnostics(ref, rho_ref)
    div_cand, vort_cand, continuity_cand = flow_diagnostics(cand, rho_cand)
    wx = _integration_weights(ref.shape[-1], px, dx)
    wy = _integration_weights(ref.shape[-2], py, dy)
    weights = wy[:, None] * wx[None, :] * fluid
    area = float(weights.sum())

    def integrals(velocity: np.ndarray, rho: np.ndarray, vorticity: np.ndarray):
        mass = np.sum(rho * weights, axis=(-2, -1))
        energy = 0.5 * np.sum(rho * np.sum(velocity**2, axis=1) * weights, axis=(-2, -1))
        # Derivative-valid enstrophy avoids the halo of a masked solid wall.
        enstrophy = 0.5 * np.sum(vorticity**2 * weights * valid, axis=(-2, -1))
        flux = np.zeros(len(times))
        if not px:
            flux += np.sum((rho[:, :, -1] * velocity[:, 0, :, -1]
                            - rho[:, :, 0] * velocity[:, 0, :, 0]) * wy, axis=-1)
        if not py:
            flux += np.sum((rho[:, -1, :] * velocity[:, 1, -1, :]
                            - rho[:, 0, :] * velocity[:, 1, 0, :]) * wx, axis=-1)
        mass_rate = (np.gradient(mass, times, edge_order=2 if len(times) > 2 else 1)
                     if len(times) > 1 else None)
        return mass, energy, enstrophy, flux, mass_rate

    mass_ref, energy_ref, ens_ref, flux_ref, mass_rate_ref = integrals(ref, rho_ref, vort_ref)
    mass_cand, energy_cand, ens_cand, flux_cand, mass_rate_cand = integrals(cand, rho_cand, vort_cand)
    div_ref, div_cand = div_ref[:, valid], div_cand[:, valid]
    vort_ref_valid, vort_cand_valid = vort_ref[:, valid], vort_cand[:, valid]
    reference_vorticity_rms = _rms(vort_ref_valid)
    reference_speed_rms = float(np.sqrt(np.mean(np.sum(ref[:, :, fluid]**2, axis=1))))
    reference_density_mean = float(np.mean(rho_ref[:, fluid]))
    flux_scale = reference_density_mean * reference_speed_rms * np.sqrt(area)
    extra_divergence = _rms(div_cand - div_ref)
    result: dict[str, Any] = {
        "velocity_relative_l2": _relative_l2(ref[:, :, fluid], cand[:, :, fluid]),
        "velocity_rmse": _rms(cand[:, :, fluid] - ref[:, :, fluid]),
        "vorticity_relative_l2": _relative_l2(vort_ref_valid, vort_cand_valid),
        "vorticity_rmse": _rms(vort_cand_valid - vort_ref_valid),
        "vorticity_correlation": _correlation(vort_ref_valid, vort_cand_valid),
        "reference_vorticity_rms": reference_vorticity_rms,
        "reference_speed_rms": reference_speed_rms,
        "reference_divergence_rms": _rms(div_ref),
        "divergence_rms": _rms(div_cand),
        "extra_divergence_rms": extra_divergence,
        "extra_divergence_normalized": extra_divergence / reference_vorticity_rms if reference_vorticity_rms > 0 else None,
        "mass_relative_l2": _relative_l2(mass_ref, mass_cand),
        "kinetic_energy_relative_l2": _relative_l2(energy_ref, energy_cand),
        "enstrophy_relative_l2": _relative_l2(ens_ref, ens_cand),
        "reference_boundary_flux_rms": _rms(flux_ref),
        "boundary_flux_rms": _rms(flux_cand),
        "extra_boundary_flux_rms": _rms(flux_cand - flux_ref),
        "boundary_flux_normalized": _rms(flux_cand) / flux_scale if flux_scale > 0 else None,
        "extra_boundary_flux_normalized": _rms(flux_cand - flux_ref) / flux_scale if flux_scale > 0 else None,
        "boundary_flux_scale": float(flux_scale),
        "normalization": {"extra_divergence_normalized": "extra divergence RMS / reference vorticity RMS; undefined for zero vorticity",
                          "boundary_flux_normalized": "flux RMS / (reference mean density * reference speed RMS * sqrt(fluid area))",
                          "mass_balance_normalized": "mass balance RMS / same flux scale"},
        "fluid_area": area,
        "derivative_valid_nodes": int(valid.sum()),
        "candidate_density_assumed": density_assumed,
        "density_metrics_available": density_available,
        "constant_density_mass_trivial": constant_density,
        "derivative_method": "finite_difference_mask_halo" if np.any(mask) else "spectral_periodic_finite_difference_nonperiodic",
        "periodic_axes": list(periodic_axes),
        "quadrature": "tensor-product periodic rectangles / nonperiodic trapezoids",
        "units": {"velocity_rmse": "velocity", "vorticity_rmse": "1/time", "divergence_rms": "1/time",
                  "continuity_rms": "density/time", "mass": "density*length^2 (per unit depth)",
                  "boundary_flux": "density*length^2/time (per unit depth)",
                  "kinetic_energy": "density*velocity^2*length^2 (per unit depth)",
                  "enstrophy": "length^2/time^2", "relative_l2": "dimensionless"},
        "series": {"times": times.tolist(), "reference_mass": mass_ref.tolist(), "mass": mass_cand.tolist(),
                   "reference_kinetic_energy": energy_ref.tolist(), "kinetic_energy": energy_cand.tolist(),
                   "reference_enstrophy": ens_ref.tolist(), "enstrophy": ens_cand.tolist(),
                   "reference_boundary_flux": flux_ref.tolist(), "boundary_flux": flux_cand.tolist()},
        "warnings": warnings,
    }
    # A signed net flux can cancel simultaneous inflow and outflow through two
    # walls. Inspect boundary velocities separately, including tangential slip.
    wall_keys = ("wall_velocity_relative_rms", "wall_normal_velocity_relative_rms",
                 "wall_velocity_error_relative_rms", "wall_normal_velocity_error_relative_rms",
                 "reference_wall_velocity_relative_rms", "wall_velocity_rms",
                 "wall_normal_velocity_rms", "wall_velocity_error_rms",
                 "wall_normal_velocity_max_abs")
    result.update({key: None for key in wall_keys})
    result["wall_gate_applicable"] = False
    if boundary == "channel" and not py:
        wall_indices = metadata.get("wall_indices", [0, -1])
        if not isinstance(wall_indices, (list, tuple, np.ndarray)) or len(wall_indices) == 0:
            raise ValueError("wall_indices must be a nonempty list of y-node indices")
        if any(isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer))
               or not -ref.shape[-2] <= index < ref.shape[-2] for index in wall_indices):
            raise ValueError("wall_indices must be valid integer y-node indices")
        indices = np.unique(np.asarray(wall_indices, dtype=int) % ref.shape[-2])
        wall_fluid = fluid[indices]
        if np.any(wall_fluid):
            wall_ref = ref[:, :, indices, :][:, :, wall_fluid]
            wall_cand = cand[:, :, indices, :][:, :, wall_fluid]
            wall_error = wall_cand - wall_ref
            wall_speed = float(np.sqrt(np.mean(np.sum(wall_cand**2, axis=1))))
            wall_error_speed = float(np.sqrt(np.mean(np.sum(wall_error**2, axis=1))))
            wall_reference_speed = float(np.sqrt(np.mean(np.sum(wall_ref**2, axis=1))))
            wall_normal_speed = _rms(wall_cand[:, 1])
            wall_normal_error = _rms(wall_error[:, 1])

            def normalize_wall(value: float) -> float | None:
                return value / reference_speed_rms if reference_speed_rms > 0 else (0.0 if value == 0 else None)

            result.update(wall_gate_applicable=True, wall_indices=indices.tolist(),
                          wall_velocity_rms=wall_speed, wall_velocity_error_rms=wall_error_speed,
                          wall_normal_velocity_rms=wall_normal_speed,
                          wall_normal_velocity_max_abs=float(np.max(np.abs(wall_cand[:, 1]))),
                          wall_velocity_relative_rms=normalize_wall(wall_speed),
                          wall_velocity_error_relative_rms=normalize_wall(wall_error_speed),
                          wall_normal_velocity_relative_rms=normalize_wall(wall_normal_speed),
                          wall_normal_velocity_error_relative_rms=normalize_wall(wall_normal_error),
                          reference_wall_velocity_relative_rms=normalize_wall(wall_reference_speed))
            result["normalization"]["wall_relative_rms"] = "wall vector speed RMS (or normal/error RMS) / reference bulk speed RMS; exact zero/zero is reported as zero"
            warnings.append("Channel wall metrics sample declared horizontal y-node walls; absolute wall velocity tests stationary no-slip, and wall error tests agreement with a moving-wall reference. Net flux alone can hide wall penetration.")
        else:
            warnings.append("Declared channel wall nodes are all masked solid; no collocated fluid wall velocity is available, so the wall gate is not applicable.")
    if len(times) > 1:
        c_ref, c_cand = continuity_ref[:, valid], continuity_cand[:, valid]
        balance_ref, balance_cand = mass_rate_ref + flux_ref, mass_rate_cand + flux_cand
        result.update(reference_continuity_rms=_rms(c_ref), continuity_rms=_rms(c_cand),
                      extra_continuity_rms=_rms(c_cand - c_ref),
                      reference_mass_balance_rms=_rms(balance_ref), mass_balance_rms=_rms(balance_cand),
                      extra_mass_balance_rms=_rms(balance_cand - balance_ref),
                      mass_balance_normalized=_rms(balance_cand) / flux_scale if flux_scale > 0 else None,
                      extra_mass_balance_normalized=_rms(balance_cand - balance_ref) / flux_scale if flux_scale > 0 else None)
        result["series"].update(reference_mass_balance=balance_ref.tolist(), mass_balance=balance_cand.tolist())
    else:
        result.update(reference_continuity_rms=None, continuity_rms=None, extra_continuity_rms=None,
                      reference_mass_balance_rms=None, mass_balance_rms=None, extra_mass_balance_rms=None,
                      mass_balance_normalized=None, extra_mass_balance_normalized=None)
        warnings.append("One snapshot cannot determine temporal continuity or total mass balance.")
    if not density_available:
        for key in list(result):
            if ("mass" in key or "continuity" in key or "boundary_flux" in key
                    or key == "kinetic_energy_relative_l2"):
                if key != "constant_density_mass_trivial":
                    result[key] = None
        for key in list(result["series"]):
            if "mass" in key or "kinetic_energy" in key or "boundary_flux" in key:
                result["series"][key] = None
    warnings.append("Reference residuals describe clean-data discretization; extra residuals isolate reconstruction changes, not a Navier-Stokes solver certification.")
    return result
