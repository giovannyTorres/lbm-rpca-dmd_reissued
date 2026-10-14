"""Geometry-only incompressibility priors and pressure-free NS diagnostics.

Projection is an estimator postprocess, not a declaration that a field solves
Navier--Stokes. No reference velocity is used. Nonperiodic boundaries and solids
require a different Poisson boundary problem and are rejected here.
"""
from __future__ import annotations

import numpy as np


def _wavenumbers(ny, nx, dx, dy):
    if min(ny, nx) < 3 or not all(np.isfinite(x) and x > 0 for x in (dx, dy)):
        raise ValueError("positive spacing and at least three nodes per axis required")
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    # On real endpoint-exclusive grids a Nyquist first derivative has no real
    # representation. Match the real FFT derivative used by the metric code.
    if nx % 2 == 0:
        kx[nx // 2] = 0
    if ny % 2 == 0:
        ky[ny // 2] = 0
    kx, ky = np.meshgrid(kx, ky)
    return kx, ky, kx*kx + ky*ky


def helmholtz_project(velocity, dx, dy, *, dealias=False):
    """Orthogonal projection onto the discrete periodic solenoidal subspace.

    ``dealias=True`` also truncates each axis to strict 2/3 resolved modes. This
    supplies an explicit bandwidth prior rather than silently smoothing fields.
    The unfiltered projection preserves mean velocity and discrete curl.
    """
    value = np.asarray(velocity, dtype=float)
    if value.ndim != 4 or value.shape[1] != 2 or not np.isfinite(value).all():
        raise ValueError("finite velocity (T,2,Y,X) required")
    ny, nx = value.shape[-2:]
    kx, ky, k2 = _wavenumbers(ny, nx, dx, dy)
    transformed = np.fft.fft2(value, axes=(-2, -1))
    dot = kx*transformed[:, 0] + ky*transformed[:, 1]
    pressure_gradient_factor = np.zeros_like(dot)
    np.divide(dot, k2, out=pressure_gradient_factor, where=k2 > 0)
    transformed[:, 0] -= kx*pressure_gradient_factor
    transformed[:, 1] -= ky*pressure_gradient_factor
    if dealias:
        mx = np.rint(np.fft.fftfreq(nx)*nx)
        my = np.rint(np.fft.fftfreq(ny)*ny)
        resolved = (np.abs(mx)[None, :] < nx/3) & (np.abs(my)[:, None] < ny/3)
        transformed *= resolved
    result = np.fft.ifft2(transformed, axes=(-2, -1)).real
    if not np.isfinite(result).all():
        raise FloatingPointError("projection produced nonfinite velocity")
    denominator = max(float(np.linalg.norm(value)), np.finfo(float).tiny)
    return result, {
        "postprocess": "helmholtz_dealias" if dealias else "helmholtz",
        "relative_change": float(np.linalg.norm(result-value)/denominator),
        "mean_velocity_change_max": float(np.max(np.abs(result.mean(axis=(-2,-1))-value.mean(axis=(-2,-1))))),
        "fft_calls": 2, "fft_points": int(value.shape[0]*2*ny*nx),
        "boundary": "periodic", "reference_velocity_used": False,
        "nyquist_policy": "real compatible first derivative: Nyquist wave number zero; strict 2/3 truncation when requested",
        "curl_preserved": not dealias,
        "kinetic_energy_nonincreasing_unit_density": True,
        "qualification": "incompressibility projection does not certify dynamics or stress reconstruction",
    }


def apply_postprocessing(dataset, velocity, kind="none"):
    if kind in (None, "none"):
        return np.asarray(velocity), {"postprocess": "none", "reference_velocity_used": False}
    if kind not in ("helmholtz", "helmholtz_dealias"):
        raise ValueError("unknown postprocessing")
    mask = np.asarray(dataset.get("mask", np.zeros(np.asarray(velocity).shape[-2:], dtype=bool)))
    if dataset["boundary"] != "periodic" or np.any(mask):
        raise ValueError("periodic Helmholtz projection cannot enforce channel or obstacle boundaries")
    return helmholtz_project(velocity, dataset["dx"], dataset["dy"], dealias=kind.endswith("dealias"))


def vorticity_residual_metrics(reference, candidate_velocity):
    """Matched sampled-time vorticity equation residual, only Newtonian torus.

    Temporal differences include sampling truncation. For a non-Newtonian field
    velocity alone cannot supply the polymer/yield stress curl contribution.
    Such cases are explicitly unsupported, rather than omitting that force.
    """
    metadata = reference.get("metadata", {})
    model = str(metadata.get("model", metadata.get("constitutive_model", "newtonian"))).lower()
    unsupported = {"vorticity_equation_applicable": False,
                   "vorticity_equation_residual_rms": None,
                   "reference_vorticity_equation_residual_rms": None,
                   "extra_vorticity_equation_residual_rms": None,
                   "extra_vorticity_equation_residual_normalized": None}
    mask = np.asarray(reference.get("mask", np.zeros(np.asarray(candidate_velocity).shape[-2:], dtype=bool)))
    if reference.get("boundary") != "periodic" or np.any(mask) or model != "newtonian" or "viscosity" not in metadata:
        return {**unsupported, "vorticity_equation_reason": "Requires periodic Newtonian velocity with viscosity and known forcing; non-Newtonian stress not inferred"}
    times = np.asarray(reference["times"], dtype=float)
    if len(times) < 3:
        return {**unsupported, "vorticity_equation_reason": "At least three snapshots required for sampled time derivatives"}
    if np.any(np.diff(times) <= 0):
        raise ValueError("times must increase strictly")
    ref = np.asarray(reference["velocity"], dtype=float)
    candidate = np.asarray(candidate_velocity, dtype=float)
    if candidate.shape != ref.shape or not np.isfinite(candidate).all():
        raise ValueError("finite candidate with reference shape required")
    viscosity = float(metadata["viscosity"])
    if not np.isfinite(viscosity) or viscosity < 0:
        raise ValueError("viscosity must be finite and nonnegative")
    kx, ky, k2 = _wavenumbers(*ref.shape[-2:], reference["dx"], reference["dy"])
    # A real Nyquist first derivative vanishes at sampled nodes, but its
    # second derivative does not. Do not reuse the projection's k2 here.
    ny, nx = ref.shape[-2:]
    raw_x, raw_y = np.meshgrid(2*np.pi*np.fft.fftfreq(nx,d=reference["dx"]),
                               2*np.pi*np.fft.fftfreq(ny,d=reference["dy"]))
    laplacian_k2 = raw_x**2+raw_y**2
    forcing_curl = reference.get("forcing_curl")
    if forcing_curl is None and metadata.get("forcing") not in (None, "none", 0):
        return {**unsupported, "vorticity_equation_reason": "Forcing present but its curl is unavailable"}
    forcing_curl = 0. if forcing_curl is None else np.asarray(forcing_curl)

    def derivative(a, wave):
        return np.fft.ifft2(1j*wave*np.fft.fft2(a), axes=(-2,-1)).real

    def residual(velocity):
        u, v = velocity[:, 0], velocity[:, 1]
        omega = derivative(v, kx)-derivative(u, ky)
        temporal = np.gradient(omega, times, axis=0, edge_order=2)
        convection = u*derivative(omega, kx)+v*derivative(omega, ky)
        diffusion = np.fft.ifft2(-laplacian_k2*np.fft.fft2(omega), axes=(-2,-1)).real
        return temporal+convection-viscosity*diffusion-forcing_curl, omega

    clean_residual, omega = residual(ref)
    reconstructed_residual, _ = residual(candidate)
    rms = lambda array: float(np.sqrt(np.mean(array**2)))
    length = float(np.sqrt(ref.shape[-2]*reference["dy"]*ref.shape[-1]*reference["dx"]))
    speed = rms(np.sqrt(np.sum(ref**2,axis=1)))
    scale = rms(omega)*(speed/length + viscosity/length**2)
    difference = rms(reconstructed_residual-clean_residual)
    values = {"vorticity_equation_applicable": True,
              "vorticity_equation_residual_rms": rms(reconstructed_residual),
              "reference_vorticity_equation_residual_rms": rms(clean_residual),
              "extra_vorticity_equation_residual_rms": difference,
              "extra_vorticity_equation_residual_normalized": difference/scale if scale > 0 else None,
              "vorticity_equation_scale": scale,
              "vorticity_equation_definition": "omega_t+u*omega_x+v*omega_y-nu*laplacian(omega)-curl(force)",
              "vorticity_equation_time_operator": "second-order nonuniform sampled-time gradient, with one-sided endpoint stencils",
              "vorticity_equation_scope": "pressure-free Newtonian torus; includes finite snapshot-spacing error"}
    for key, val in values.items():
        if isinstance(val, float) and not np.isfinite(val):
            raise FloatingPointError(f"Nonfinite physical residual {key}")
    return values
