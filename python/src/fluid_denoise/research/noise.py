"""Controlled contamination for research velocity fields.

All models have the *same realised RMS* on fluid entries, so the sparse model
does not receive a different noise budget from the dense Gaussian model.
Snapshots use (time, component, y, x); solid values are never altered.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


NOISE_KINDS = (
    "gaussian", "sparse_impulsive", "mixed", "spatial_correlated",
    "temporal_correlated", "column_outliers",
    "student_t", "heteroscedastic", "sensor_drift",
)


def _rms(values):
    scale = float(np.max(np.abs(values)))
    return scale * float(np.sqrt(np.mean((values / scale) ** 2))) if scale else 0.0


def add_noise(
    velocity: np.ndarray,
    kind: str,
    intensity: float = 0.1,
    seed: int = 0,
    obstacle_mask: np.ndarray | None = None,
    **params,
) -> tuple[np.ndarray, dict]:
    """Add deterministic, RMS-matched noise without accessing a clean estimator.

    ``intensity`` is noise RMS / input fluid velocity RMS, including both
    components. An all-zero input receives zero noise under this relative
    convention. ``sparse_impulsive`` uses ``fraction`` (default .05);
    ``column_outliers`` uses the same fraction of entire time snapshots.
    ``mixed`` divides expected noise energy between Gaussian and impulses with
    ``dense_fraction`` (default .5). Spatial Gaussian filtering uses
    ``correlation_length`` in grid cells (default 1.5); temporal noise is a
    stationary AR(1) process with ``correlation`` (default .8).
    """
    value = np.asarray(velocity, dtype=np.float64)
    if value.ndim != 4 or value.shape[1] != 2 or any(n == 0 for n in value.shape):
        raise ValueError("velocity must have shape (T, 2, Y, X), without empty axes")
    if not np.all(np.isfinite(value)):
        raise ValueError("velocity must contain finite values")
    if not np.isfinite(intensity) or intensity < 0:
        raise ValueError("intensity must be finite and nonnegative")
    aliases = {"sparse": "sparse_impulsive", "impulsive": "sparse_impulsive",
               "spatial": "spatial_correlated", "temporal": "temporal_correlated"}
    kind = aliases.get(kind, kind)
    if kind not in NOISE_KINDS:
        raise ValueError(f"unknown noise kind {kind!r}; expected {NOISE_KINDS}")
    solids = np.zeros(value.shape[-2:], dtype=bool) if obstacle_mask is None else np.asarray(obstacle_mask, dtype=bool)
    if solids.shape != value.shape[-2:]:
        raise ValueError("obstacle_mask must have shape (Y, X)")
    fluid = np.broadcast_to(~solids, value.shape)
    if not np.any(fluid):
        raise ValueError("obstacle_mask leaves no fluid entries")
    rng = np.random.default_rng(seed)
    signal_rms = _rms(value[fluid])
    target_rms = float(intensity * signal_rms)
    if not np.isfinite(target_rms):
        raise ValueError("requested noise RMS overflowed; rescale the input")
    fraction = float(params.get("fraction", .05))
    settings: dict = {}

    def impulses() -> np.ndarray:
        if not 0 < fraction <= 1:
            raise ValueError("fraction must lie in (0, 1]")
        selected = (rng.random(value.shape) < fraction) & fluid
        if not np.any(selected):
            indices = np.flatnonzero(fluid)
            selected.flat[int(rng.choice(indices))] = True
        return rng.standard_normal(value.shape) * selected / np.sqrt(fraction)

    if kind == "gaussian":
        noise = rng.standard_normal(value.shape)
    elif kind == "sparse_impulsive":
        noise = impulses()
        settings["fraction"] = fraction
    elif kind == "mixed":
        dense_fraction = float(params.get("dense_fraction", .5))
        if not 0 <= dense_fraction <= 1:
            raise ValueError("dense_fraction must lie in [0, 1]")
        noise = (np.sqrt(dense_fraction) * rng.standard_normal(value.shape)
                 + np.sqrt(1 - dense_fraction) * impulses())
        settings.update(fraction=fraction, dense_fraction=dense_fraction)
    elif kind == "spatial_correlated":
        length = float(params.get("correlation_length", 1.5))
        if not np.isfinite(length) or length < 0:
            raise ValueError("correlation_length must be finite and nonnegative")
        noise = gaussian_filter(rng.standard_normal(value.shape), sigma=(0, 0, length, length), mode="wrap")
        settings["correlation_length"] = length
    elif kind == "temporal_correlated":
        correlation = float(params.get("correlation", .8))
        if not np.isfinite(correlation) or not -1 < correlation < 1:
            raise ValueError("correlation must lie in (-1, 1)")
        noise = rng.standard_normal(value.shape)
        for t in range(1, value.shape[0]):
            noise[t] = correlation * noise[t - 1] + np.sqrt(1 - correlation ** 2) * noise[t]
        settings["correlation"] = correlation
    elif kind == "student_t":
        degrees = float(params.get("degrees_of_freedom", 3.))
        if not np.isfinite(degrees) or degrees <= 2:
            raise ValueError("Student-t requires degrees_of_freedom > 2 for finite variance")
        noise = rng.standard_t(degrees, size=value.shape) / np.sqrt(degrees/(degrees-2))
        settings["degrees_of_freedom"] = degrees
    elif kind == "heteroscedastic":
        floor = float(params.get("noise_floor_fraction", .05))
        if not np.isfinite(floor) or floor < 0:
            raise ValueError("noise_floor_fraction must be finite and nonnegative")
        noise = rng.standard_normal(value.shape) * (np.abs(value)+floor*signal_rms)
        settings.update(noise_floor_fraction=floor, observation_scale="abs(local component velocity) + floor*fluid RMS")
    elif kind == "sensor_drift":
        # A coherent affine-in-time bias can itself be low rank. Algorithms
        # cannot automatically distinguish this from genuine coherent motion.
        slope = rng.standard_normal((1,2,1,1))
        offset = rng.standard_normal((1,2,1,1))
        trend = np.linspace(-1,1,value.shape[0])[:,None,None,None]
        noise = np.broadcast_to(offset+slope*trend,value.shape).copy()
        settings["bias_model"] = "spatially uniform component offsets and linear temporal drift; not zero-mean white noise"
    else:
        if not 0 < fraction <= 1:
            raise ValueError("fraction must lie in (0, 1]")
        count = max(1, int(np.ceil(fraction * value.shape[0])))
        selected_times = np.sort(rng.choice(value.shape[0], size=count, replace=False))
        noise = np.zeros_like(value)
        noise[selected_times] = rng.standard_normal(noise[selected_times].shape)
        settings.update(fraction=fraction, contaminated_snapshots=selected_times.tolist())
    noise[~fluid] = 0.0
    raw_rms = _rms(noise[fluid])
    if target_rms == 0:
        noise.fill(0)
    elif raw_rms > 0:
        noise *= target_rms / raw_rms
    else:
        raise RuntimeError("generated noise has zero norm at positive target RMS")
    with np.errstate(over="raise", invalid="raise"):
        noisy = value + noise
    if not np.all(np.isfinite(noisy)):
        raise RuntimeError("noise generation produced nonfinite data; rescale the input")
    realised_rms = _rms(noise[fluid])
    return noisy, {
        "kind": kind, "seed": int(seed), "intensity_requested": float(intensity),
        "signal_rms": signal_rms, "noise_rms": realised_rms,
        "realized_relative_rms": realised_rms / signal_rms if signal_rms else 0.0,
        "nonzero_fraction": float(np.count_nonzero(noise[fluid]) / np.count_nonzero(fluid)),
        "normalization": "realized fluid-entry RMS matched across noise kinds",
        "settings": settings,
    }


# Explicit synonym for callers that prefer the velocity-oriented name.
corrupt_velocity = add_noise
