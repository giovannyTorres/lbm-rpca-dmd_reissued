"""Chronological forecasting with train-only estimators and noisy validation.

Unlike ``algorithms.reconstruct``, these estimators retain dynamics and evaluate
them at genuinely unobserved times. Clean future fields enter only the final
physical evaluation. The validation objective is observed-data MSE: it is an
unbiased clean-risk proxy only for independent, zero-mean validation noise.
Temporal correlation and outlier contamination weaken that interpretation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import copy
import hashlib
import json
import time
from typing import Any

import numpy as np
from scipy.optimize import least_squares


FORECAST_METHODS = ("persistence", "pod_ar", "dmd", "tls_dmd", "optimized_dmd")


class ForecastFailure(RuntimeError):
    """A numerical failure, never a zero-filled successful prediction."""

    def __init__(self, reason: str, diagnostics: dict | None = None):
        super().__init__(reason)
        self.diagnostics = {**(diagnostics or {}), "status": "failed", "converged": False,
                            "reason": reason}


def _times(times, n=None):
    t = np.asarray(times, dtype=float)
    if t.ndim != 1 or len(t) == 0 or (n is not None and len(t) != n):
        raise ValueError("times must be a nonempty vector matching snapshots")
    if not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError("times must be finite and strictly increasing")
    return t


def chronological_split(times, train_fraction=.6, validation_fraction=.2):
    """Contiguous index arrays; rounding remainder belongs to the test block."""
    t = _times(times)
    fractions = np.asarray([train_fraction, validation_fraction], dtype=float)
    if not np.all(np.isfinite(fractions)) or np.any(fractions <= 0) or fractions.sum() >= 1:
        raise ValueError("positive train/validation fractions must sum to less than one")
    train_end = int(np.floor(len(t) * train_fraction))
    val_end = train_end + int(np.floor(len(t) * validation_fraction))
    if train_end < 3 or val_end <= train_end or val_end >= len(t):
        raise ValueError("split needs at least three training and one validation/test snapshots")
    return {"train": np.arange(train_end), "validation": np.arange(train_end, val_end),
            "test": np.arange(val_end, len(t))}


def _digest(values):
    values = np.ascontiguousarray(values)
    h = hashlib.sha256()
    h.update(str(values.shape).encode())
    h.update(str(values.dtype).encode())
    h.update(values.tobytes())
    return h.hexdigest()


def _finite(value, reason):
    if not np.all(np.isfinite(value)):
        raise ForecastFailure(reason)
    return value


def _exponentials(rates, elapsed):
    # There is deliberately no spectral-radius projection or exponent clipping.
    try:
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            return _finite(np.exp(rates[:, None] * elapsed[None, :]),
                           "forecast exponential evolution is nonfinite")
    except FloatingPointError as exc:
        raise ForecastFailure("forecast exponential evolution overflowed") from exc


def _truncate(matrix, rank, rtol):
    u, s, vh = np.linalg.svd(matrix, full_matrices=False)
    effective = min(rank, int(np.count_nonzero(s > (s[0] * rtol if len(s) else 0))))
    if effective == 0:
        raise ForecastFailure("training matrix has numerical rank zero")
    return u[:, :effective], s[:effective], vh[:effective], effective


def _uniform_dt(t):
    dt = float(np.mean(np.diff(t)))
    if not np.allclose(np.diff(t), dt, rtol=1e-6, atol=1e-12):
        raise ValueError("discrete DMD/POD autoregression needs uniform times; use optimized_dmd")
    return dt


@dataclass
class ForecastModel:
    """Inspectable fitted dynamics in full feature coordinates.

    Exponential models use ``modes @ (amplitudes * exp(rates * elapsed))``.
    POD autoregression uses its learned affine reduced operator from the last
    training state; persistence repeats the last training observation.
    """
    method: str
    training_times: np.ndarray
    diagnostics: dict[str, Any]
    last_snapshot: np.ndarray
    modes: np.ndarray | None = None
    rates: np.ndarray | None = None
    amplitudes: np.ndarray | None = None
    basis: np.ndarray | None = None
    mean: np.ndarray | None = None
    operator: np.ndarray | None = None
    last_state: np.ndarray | None = None
    dt: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def predict(self, target_times):
        t = _times(target_times)
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                if self.method == "persistence" or self.diagnostics.get("zero_input"):
                    output = np.repeat(self.last_snapshot[:, None], len(t), axis=1)
                elif self.method == "pod_ar":
                    steps = (t - self.training_times[-1]) / self.dt
                    if np.any(steps < -1e-8) or not np.allclose(steps, np.rint(steps), atol=1e-7, rtol=1e-7):
                        raise ValueError("POD autoregression predicts nonnegative integer steps after its training anchor")
                    # Sorted targets let us advance once, including all omitted steps.
                    state = self.last_state.copy()
                    predictions = []
                    current_step = 0
                    for step in np.rint(steps).astype(np.int64):
                        gap = int(step) - current_step
                        if gap:
                            state = np.linalg.matrix_power(self.operator, gap) @ state
                        predictions.append(self.mean + self.basis @ state[:-1])
                        current_step = int(step)
                    output = np.column_stack(predictions)
                else:
                    evolution = _exponentials(self.rates, t - self.training_times[0])
                    output = (self.modes @ (self.amplitudes[:, None] * evolution)).real
                return _finite(output, "forecast produced nonfinite full-feature fields")
        except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
            raise ForecastFailure(str(exc), self.diagnostics) from exc


def _preprocess(matrix, times, preprocessing, parameters):
    aliases = {"pcp": "rpca", "stable": "stable_rpca"}
    preprocessing = aliases.get(preprocessing, preprocessing)
    if preprocessing == "none":
        return matrix.copy(), {"method": "none", "converged": True, "fit_scope": "train_only"}
    if preprocessing not in ("rpca", "stable_rpca"):
        raise ValueError("preprocessing must be none, rpca/pcp, or stable_rpca")
    from .algorithms import reconstruct
    low, diag = reconstruct(matrix, preprocessing, rank=1, times=times, **parameters)
    return low, {**diag, "fit_scope": "train_only", "rank_budget_ignored_by_convex_preprocessor": True}


def fit_forecast(training_matrix, times, method, rank, **params):
    """Fit once using only supplied features x training-time observations.

    Parameters include ``preprocessing``, ``preprocessing_parameters``, a
    nonnegative POD ``ridge`` (default 1e-6, relative to reduced X norm squared),
    DMD ``amplitude_fit`` (global/first), and optimized-DMD solver bounds and
    ``max_nfev``/``n_starts``. Bounds constrain fitting, never future evolution.
    """
    original = np.asarray(training_matrix)
    if original.ndim != 2 or min(original.shape) == 0 or np.iscomplexobj(original):
        raise ValueError("training_matrix must be a nonempty real features x time matrix")
    matrix = np.asarray(original, dtype=float).copy()
    if not np.all(np.isfinite(matrix)):
        raise ValueError("training_matrix must be finite")
    t = _times(times, matrix.shape[1]).copy()
    if len(t) < 2:
        raise ValueError("forecast fitting requires at least two training snapshots")
    if isinstance(rank, (bool, np.bool_)) or int(rank) != rank or rank < 1:
        raise ValueError("rank must be a positive integer")
    rank = int(rank)
    method = {"exact_dmd": "dmd", "pod": "pod_ar", "persistent": "persistence"}.get(method, method)
    if method not in FORECAST_METHODS:
        raise ValueError(f"unknown forecast method {method!r}; expected {FORECAST_METHODS}")
    preprocessing = params.get("preprocessing", "none")
    preprocessing_parameters = dict(params.get("preprocessing_parameters", {}))
    rtol = float(params.get("svd_rtol", max(matrix.shape) * np.finfo(float).eps))
    if not np.isfinite(rtol) or rtol <= 0:
        raise ValueError("svd_rtol must be finite and positive")
    base = {"method": method, "rank_requested": rank, "shape": list(matrix.shape),
            "fit_scope": "train_only", "training_observations_sha256": _digest(matrix),
            "training_times_sha256": _digest(t), "converged": True, "status": "ok"}
    try:
        m, prediag = _preprocess(matrix, t, preprocessing, preprocessing_parameters)
        base["preprocessing"] = prediag
        base["converged"] = bool(prediag.get("converged", False))
        base["status"] = "ok" if base["converged"] else "not_converged"
        model = ForecastModel(method, t, base, m[:, -1].copy(),
                              metadata={"training_only": True, "future_fields_used": False,
                                        "spectral_clipping": False, "evaluation_postprocessing": "none"})
        with np.errstate(over="raise", invalid="raise"):
            training_norm = float(np.linalg.norm(m))
        _finite(training_norm, "training matrix norm overflowed; rescale observations")
        if training_norm == 0 or method == "persistence":
            base.update(effective_rank=0 if training_norm == 0 else None,
                        zero_input=bool(training_norm == 0), iterations=0,
                        dynamics="constant last training observation")
            return model
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            if method == "pod_ar":
                model.dt = _uniform_dt(t)
                model.mean = np.mean(m, axis=1)
                centered = m - model.mean[:, None]
                if np.linalg.norm(centered) == 0:
                    base.update(effective_rank=0, zero_input=True, iterations=0,
                                dynamics="constant training series")
                    return model
                model.basis, _, _, r = _truncate(centered, rank, rtol)
                coordinates = model.basis.T @ centered
                x, y = coordinates[:, :-1], coordinates[:, 1:]
                design = np.vstack((x, np.ones(x.shape[1])))
                ridge = float(params.get("ridge", 1e-6))
                if not np.isfinite(ridge) or ridge < 0:
                    raise ValueError("ridge must be finite and nonnegative")
                penalty = ridge * float(np.linalg.norm(x) ** 2) / max(r, 1)
                augmented = np.column_stack((design, np.diag(np.r_[np.full(r, np.sqrt(penalty)), 0.])))
                targets = np.column_stack((y, np.zeros((r, r + 1))))
                fitted = np.linalg.lstsq(augmented.T, targets.T, rcond=None)[0].T
                model.operator = np.vstack((fitted, np.r_[np.zeros(r), 1.]))
                model.last_state = np.r_[coordinates[:, -1], 1.]
                eigs = np.linalg.eigvals(fitted[:, :r])
                base.update(effective_rank=r, iterations=1, ridge_relative=ridge,
                            ridge_absolute=penalty, spectral_radius=float(np.max(np.abs(eigs))),
                            dt=model.dt, dynamics="train-centered POD, affine reduced AR(1), train-only ridge")
            elif method in ("dmd", "tls_dmd"):
                model.dt = _uniform_dt(t)
                x, y = m[:, :-1], m[:, 1:]
                if method == "tls_dmd":
                    _, _, vh, _ = _truncate(np.vstack((x, y)), rank, rtol)
                    projection = vh.conj().T @ vh
                    x, y = x @ projection, y @ projection
                u, s, vh, r = _truncate(x, rank, rtol)
                reduced = (u.T @ y @ vh.T) / s[None, :]
                eigenvalues, vectors = np.linalg.eig(reduced)
                if np.any(np.abs(eigenvalues) == 0):
                    raise ForecastFailure("zero DMD eigenvalue has no finite continuous exponential rate", base)
                model.modes = ((y @ vh.T) / s[None, :]) @ vectors
                model.rates = np.log(eigenvalues.astype(complex)) / model.dt
                powers = _exponentials(model.rates, t - t[0])
                strategy = params.get("amplitude_fit", "global")
                if strategy == "first":
                    model.amplitudes = np.linalg.lstsq(model.modes, m[:, 0], rcond=None)[0]
                elif strategy == "global":
                    gram = (model.modes.conj().T @ model.modes) * (powers.conj() @ powers.T)
                    rhs = np.sum((model.modes.conj().T @ m) * powers.conj(), axis=1)
                    model.amplitudes = np.linalg.lstsq(gram, rhs, rcond=None)[0]
                else:
                    raise ValueError("amplitude_fit must be global or first")
                base.update(effective_rank=r, iterations=1, dt=model.dt, amplitude_fit=strategy,
                            eigenvalues_real=eigenvalues.real.tolist(), eigenvalues_imag=eigenvalues.imag.tolist(),
                            rates_real=model.rates.real.tolist(), rates_imag=model.rates.imag.tolist(),
                            spectral_radius=float(np.max(np.abs(eigenvalues))),
                            dynamics="exact full-feature DMD modes with unclipped continuous exponential evolution")
            else:
                if len(t) < 3:
                    raise ValueError("optimized_dmd needs at least three training snapshots")
                u, _, _, r = _truncate(m, min(rank, len(t) - 1), rtol)
                elapsed = t - t[0]
                reduced = (u.T @ m).T.astype(complex)
                initial_op = np.linalg.lstsq(reduced[:-1], reduced[1:], rcond=None)[0].T
                eigs = np.linalg.eigvals(initial_op)
                average_dt = float(np.mean(np.diff(t)))
                initial = (np.log(np.maximum(np.abs(eigs), 1e-12)) + 1j * np.angle(eigs)) / average_dt
                rate_limit = float(params.get("rate_limit", 40.)) / elapsed[-1]
                if not np.isfinite(rate_limit) or rate_limit <= 0:
                    raise ValueError("rate_limit must be finite and positive")
                frequency_limit = float(np.pi / np.min(np.diff(t)))
                lower = np.r_[np.full(r, -rate_limit), np.full(r, -frequency_limit)]
                upper = -lower
                initial_parameters = np.clip(np.r_[initial.real, initial.imag], lower + 1e-10, upper - 1e-10)
                max_nfev, n_starts = params.get("max_nfev", 100), params.get("n_starts", 2)
                if any(isinstance(v, bool) or int(v) != v or v < 1 for v in (max_nfev, n_starts)) or n_starts > 8:
                    raise ValueError("max_nfev/n_starts must be positive integers, n_starts <= 8")
                rng = np.random.default_rng(params.get("seed", 0))
                evaluations = 0

                def solve(parameters):
                    rates = parameters[:r] + 1j * parameters[r:]
                    exponentials = _exponentials(rates, elapsed).T
                    coefficients = np.linalg.lstsq(exponentials, reduced, rcond=None)[0]
                    return coefficients, exponentials @ coefficients - reduced

                def objective(parameters):
                    nonlocal evaluations
                    evaluations += 1
                    _, residual = solve(parameters)
                    return np.r_[residual.real.ravel(), residual.imag.ravel()]

                outcomes = []
                for start in range(int(n_starts)):
                    init = initial_parameters.copy()
                    if start:
                        init += rng.normal(size=2 * r) * np.r_[np.full(r, .05 / elapsed[-1]), np.full(r, .2 / average_dt)]
                        init = np.clip(init, lower + 1e-10, upper - 1e-10)
                    outcomes.append(least_squares(objective, init, bounds=(lower, upper), max_nfev=int(max_nfev),
                                                  ftol=1e-9, xtol=1e-9, gtol=1e-9))
                best = min(outcomes, key=lambda result: result.cost)
                coefficients, _ = solve(best.x)
                model.modes = u.astype(complex) @ coefficients.T
                model.amplitudes = np.ones(r, dtype=complex)
                model.rates = best.x[:r] + 1j * best.x[r:]
                base["converged"] = bool(base["converged"] and best.success)
                base.update(effective_rank=r, iterations=int(best.nfev),
                            status="ok" if base["converged"] else "not_converged",
                            optimizer_message=str(best.message), optimizer_cost=float(best.cost),
                            optimizer_optimality=float(best.optimality), residual_evaluations=evaluations,
                            n_starts=int(n_starts), rates_real=model.rates.real.tolist(),
                            rates_imag=model.rates.imag.tolist(), rate_bounds=[-rate_limit, rate_limit],
                            frequency_bounds=[-frequency_limit, frequency_limit],
                            active_bound_count=int(np.count_nonzero(best.active_mask)),
                            dynamics="train-only POD variable-projection exponential fit; bounds apply only during fit")
                _finite(model.modes, "optimized DMD produced nonfinite modes")
            return model
    except ForecastFailure:
        raise
    except (np.linalg.LinAlgError, FloatingPointError, OverflowError) as exc:
        raise ForecastFailure(str(exc), base) from exc


def make_forecasting_noise(dataset, *, kind, intensity, seed, train_fraction=.6,
                           validation_fraction=.2, **params):
    """Generate contamination causally, with scale calibrated on training only.

    Random innovations/support never depend on clean values. Unlike full-window
    RMS matching, changing a clean future cannot rescale earlier observations.
    Temporal AR noise evolves forward. Column supports are Bernoulli, with a
    training-only fallback if no training column is initially selected.
    """
    clean = np.asarray(dataset["velocity"], dtype=float)
    if clean.ndim != 4 or clean.shape[1] != 2 or not np.all(np.isfinite(clean)):
        raise ValueError("clean velocity must be finite (time,2,y,x)")
    split = chronological_split(dataset["times"], train_fraction, validation_fraction)
    intensity = float(intensity)
    if not np.isfinite(intensity) or intensity < 0:
        raise ValueError("intensity must be finite and nonnegative")
    kind = {"temporal": "temporal_correlated", "sparse": "sparse_impulsive"}.get(kind, kind)
    if kind not in ("gaussian", "mixed", "temporal_correlated", "column_outliers", "sparse_impulsive"):
        raise ValueError("forecast noise supports gaussian, mixed, temporal_correlated, column_outliers, sparse_impulsive")
    mask = np.asarray(dataset.get("mask", np.zeros(clean.shape[-2:], bool)), dtype=bool)
    if mask.shape != clean.shape[-2:] or not np.any(~mask):
        raise ValueError("invalid obstacle mask")
    rng = np.random.default_rng(seed)
    fraction = float(params.get("fraction", .1))
    settings = {}

    def impulses():
        if not 0 < fraction <= 1:
            raise ValueError("fraction must be in (0,1]")
        return rng.standard_normal(clean.shape) * (rng.random(clean.shape) < fraction) / np.sqrt(fraction)

    if kind == "gaussian":
        noise = rng.standard_normal(clean.shape)
    elif kind == "sparse_impulsive":
        noise = impulses()
        settings["fraction"] = fraction
    elif kind == "mixed":
        dense_fraction = float(params.get("dense_fraction", .5))
        if not 0 <= dense_fraction <= 1:
            raise ValueError("dense_fraction must be in [0,1]")
        noise = np.sqrt(dense_fraction) * rng.standard_normal(clean.shape) + np.sqrt(1 - dense_fraction) * impulses()
        settings.update(fraction=fraction, dense_fraction=dense_fraction)
    elif kind == "temporal_correlated":
        correlation = float(params.get("correlation", .8))
        if not np.isfinite(correlation) or not -1 < correlation < 1:
            raise ValueError("correlation must be in (-1,1)")
        noise = rng.standard_normal(clean.shape)
        for index in range(1, len(clean)):
            noise[index] = correlation * noise[index - 1] + np.sqrt(1 - correlation ** 2) * noise[index]
        settings["correlation"] = correlation
    else:
        if not 0 < fraction <= 1:
            raise ValueError("fraction must be in (0,1]")
        selected = rng.random(len(clean)) < fraction
        if not np.any(selected[split["train"]]):
            selected[int(rng.choice(split["train"]))] = True
        noise = rng.standard_normal(clean.shape) * selected[:, None, None, None] / np.sqrt(fraction)
        settings.update(fraction=fraction, contaminated_snapshots=np.flatnonzero(selected).tolist(),
                        support_rule="Bernoulli columns; ensure one training column only")
    noise[:, :, mask] = 0
    fluid = ~mask
    train = split["train"]
    signal_rms = float(np.sqrt(np.mean(clean[train][:, :, fluid] ** 2)))
    raw_rms = float(np.sqrt(np.mean(noise[train][:, :, fluid] ** 2)))
    target = intensity * signal_rms
    if raw_rms == 0 and target > 0:
        raise ValueError("training noise realization has zero RMS; choose another seed/fraction")
    noise *= target / raw_rms if raw_rms else 0.
    noisy = _finite(clean + noise, "noise addition produced nonfinite fields")
    return noisy, {"kind": kind, "seed": int(seed), "intensity_requested": intensity,
                   "training_signal_rms": signal_rms, "training_noise_rms": target,
                   "realized_relative_rms": intensity if signal_rms else 0.,
                   "normalization": "realized training fluid-entry RMS only; future blocks are not renormalized",
                   "settings": settings, "noise_sha256": _digest(noise),
                   "training_noise_sha256": _digest(noise[train])}


def _subset_dataset(dataset, indices):
    result = dict(dataset)
    count = len(dataset["times"])
    # Only fields with an explicit time axis are sliced. Spatial masks stay fixed.
    for key in ("velocity", "times", "density", "pressure", "stress", "conformation", "forcing_curl"):
        value = dataset.get(key)
        if isinstance(value, np.ndarray) and value.ndim and value.shape[0] == count:
            if key == "density" and value.ndim == 2:
                continue
            result[key] = value[indices]
    result["velocity"] = np.asarray(dataset["velocity"])[indices]
    result["times"] = np.asarray(dataset["times"])[indices]
    return result


def _scalars(metrics):
    return {key: value for key, value in metrics.items()
            if value is None or isinstance(value, (str, bool, int, float))}


def evaluate_forecasting(dataset, noisy, config, *, return_forecasts=False):
    """Select on noisy validation, freeze selection, then evaluate clean test.

    Candidates are declared before fitting. No refit on validation takes place.
    All candidates and the persistence baseline predict identical test times;
    clean test scores are reported only after selection and cannot change it.
    """
    from .metrics import physical_metrics
    from .physics import apply_postprocessing, vorticity_residual_metrics
    times = _times(dataset["times"])
    shape = np.asarray(dataset["velocity"]).shape
    observations = np.asarray(noisy, dtype=float)
    if observations.shape != shape or len(shape) != 4 or shape[1] != 2 or len(times) != shape[0]:
        raise ValueError("noisy/reference velocity must have matching (time,2,y,x) shape")
    split_config = config.get("split", {})
    split = chronological_split(times, split_config.get("train_fraction", .6), split_config.get("validation_fraction", .2))
    fluid = ~np.asarray(dataset.get("mask", np.zeros(shape[-2:], bool)), dtype=bool)
    if fluid.shape != shape[-2:] or not np.any(fluid):
        raise ValueError("mask must retain fluid nodes")
    train, validation, test = (split[key] for key in ("train", "validation", "test"))
    # Crucially, test observations are neither validated nor transformed here.
    training = observations[train][:, :, fluid].reshape(len(train), -1).T.copy()
    val_observed = observations[validation][:, :, fluid].reshape(len(validation), -1).T.copy()
    if not np.all(np.isfinite(training)) or not np.all(np.isfinite(val_observed)):
        raise ValueError("training and validation observations must be finite")
    declared = config.get("candidates")
    if declared is None:
        declared = [{"method": method, "rank": rank}
                    for method in ("pod_ar", "dmd", "tls_dmd", "optimized_dmd") for rank in (2, 4, 8)]
    if not isinstance(declared, list) or not declared:
        raise ValueError("candidates must be a nonempty predeclared list")
    declared = [{"id": "persistence", "method": "persistence", "rank": 1}, *declared]
    identifiers = [spec.get("id", f"candidate_{index:03d}") for index, spec in enumerate(declared)]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("candidate ids must be unique (persistence is reserved)")
    records, models = [], {}
    preprocessed = {}
    fitted_models = {}
    postprocessing_profile = {}
    parameters = dict(config.get("algorithm_parameters", {}))
    known_noise = dict(config.get("known_noise", {}))
    for index, spec in enumerate(declared):
        identifier = identifiers[index]
        method, rank = spec["method"], spec.get("rank", 4)
        merged = {**parameters, **spec.get("parameters", {})}
        preprocessing = spec.get("preprocessing", "none")
        postprocess = spec.get("postprocess", "none")
        preparams = {**config.get("preprocessing_parameters", {}), **spec.get("preprocessing_parameters", {})}
        # This is declared injection information, not a clean-derived noise estimate.
        if preprocessing in ("stable_rpca", "stable") and "intensity" in known_noise:
            preparams.setdefault("noise_intensity", known_noise["intensity"])
        record = {"id": identifier, "method": method, "rank_requested": rank,
                  "preprocessing": preprocessing, "postprocess": postprocess, "parameters": merged,
                  "preprocessing_parameters": preparams, "eligible": False}
        try:
            key = json.dumps([preprocessing, preparams], sort_keys=True, allow_nan=False)
            if key not in preprocessed:
                try:
                    preprocessed[key] = _preprocess(training, times[train], preprocessing, preparams)
                except Exception as exc:
                    preprocessed[key] = exc
            cached = preprocessed[key]
            if isinstance(cached, Exception):
                raise cached
            low, prediag = cached
            fit_key = json.dumps([key, method, rank, merged], sort_keys=True, allow_nan=False)
            if fit_key not in fitted_models:
                try:
                    fitted_models[fit_key] = fit_forecast(low, times[train], method, rank, **merged)
                except Exception as exc:
                    fitted_models[fit_key] = exc
            fitted = fitted_models[fit_key]
            if isinstance(fitted, Exception):
                raise fitted
            model = copy.deepcopy(fitted)
            model.diagnostics["preprocessing"] = prediag
            model.diagnostics["training_observations_sha256"] = _digest(training)
            converged = bool(model.diagnostics["converged"] and prediag.get("converged", False))
            model.diagnostics.update(converged=converged, status="ok" if converged else "not_converged")
            model.metadata.update(evaluation_postprocessing=postprocess,
                                  projection_applies_to="validation_and_test_predictions",
                                  projection_domain="periodic" if postprocess != "none" else None,
                                  bandwidth_prior="strict two-thirds Fourier truncation" if postprocess == "helmholtz_dealias" else None,
                                  reference_velocity_used=False)
            model.diagnostics["constraints"] = model.metadata
            predicted = model.predict(times[validation])
            validation_velocity = np.zeros((len(validation), *shape[1:]), dtype=float)
            validation_velocity[:, :, fluid] = predicted.T.reshape(len(validation), 2, int(fluid.sum()))
            started = time.perf_counter()
            validation_velocity, projection_diagnostics = apply_postprocessing(dataset, validation_velocity, postprocess)
            postprocessing_profile[identifier] = {"validation_elapsed_seconds": time.perf_counter() - started}
            model.diagnostics["validation_postprocessing"] = projection_diagnostics
            predicted = validation_velocity[:, :, fluid].reshape(len(validation), -1).T
            error = predicted - val_observed
            with np.errstate(over="raise", invalid="raise"):
                score = float(np.mean(error ** 2))
                relative = float(np.linalg.norm(error) / np.linalg.norm(val_observed)) if np.linalg.norm(val_observed) else None
            _finite(score, "validation error is nonfinite")
            eligible = converged or not config.get("require_convergence", True)
            record.update(status=model.diagnostics["status"], validation_observed_mse=score,
                          validation_observed_relative_l2=relative, eligible=eligible,
                          diagnostics=model.diagnostics)
            models[identifier] = model
        except Exception as exc:
            record.update(status="failed", failure=str(exc), diagnostics=getattr(exc, "diagnostics", {}),
                          validation_observed_mse=None, validation_observed_relative_l2=None)
        records.append(record)
    eligible = [record for record in records if record["eligible"]]
    if not eligible:
        raise ForecastFailure("no candidate produced an eligible validation forecast", {"candidates": records})
    # Stable declaration order breaks exact ties; clean fields have not been read.
    selected = min(eligible, key=lambda record: record["validation_observed_mse"])
    selection = {"selected_id": selected["id"], "criterion": "minimum noisy validation observed MSE",
                 "validation_observed_mse": selected["validation_observed_mse"],
                 "tie_breaker": "predeclared candidate order", "refit_after_selection": False,
                 "clean_fields_used_for_selection": False, "test_observations_used": False,
                 "candidates": records}
    reference = _subset_dataset(dataset, test)
    test_metrics, test_failures, fingerprints, returned_predictions = {}, {}, {}, {}
    for record in records:
        identifier = record["id"]
        if identifier not in models:
            continue
        try:
            prediction = models[identifier].predict(times[test])
            velocity = np.zeros((len(test), *shape[1:]), dtype=float)
            velocity[:, :, fluid] = prediction.T.reshape(len(test), 2, int(fluid.sum()))
            started = time.perf_counter()
            velocity, projection_diagnostics = apply_postprocessing(dataset, velocity, record["postprocess"])
            postprocessing_profile.setdefault(identifier, {})["test_elapsed_seconds"] = time.perf_counter() - started
            postprocessing_profile[identifier]["test_diagnostics"] = projection_diagnostics
            metrics = physical_metrics(reference, velocity)
            metrics.update(vorticity_residual_metrics(reference, velocity))
            test_metrics[identifier] = _scalars(metrics)
            fingerprints[identifier] = _digest(velocity)
            if return_forecasts and identifier in (selected["id"], "persistence"):
                returned_predictions[identifier] = velocity.copy()
        except Exception as exc:
            test_failures[identifier] = {"reason": str(exc), "diagnostics": getattr(exc, "diagnostics", {})}
    selected_metrics = test_metrics.get(selected["id"])
    baseline_metrics = test_metrics.get("persistence")
    improvements = {}
    for metric in ("velocity_relative_l2", "vorticity_relative_l2"):
        selected_error = selected_metrics.get(metric) if selected_metrics else None
        baseline_error = baseline_metrics.get(metric) if baseline_metrics else None
        improvements[metric] = (baseline_error - selected_error) if selected_error is not None and baseline_error is not None else None
    selection_input = {"training_sha256": _digest(training), "validation_sha256": _digest(val_observed),
                       "times_sha256": _digest(times), "candidates": declared,
                       "known_noise": known_noise, "algorithm_parameters": parameters,
                       "preprocessing_parameters": config.get("preprocessing_parameters", {}),
                       "require_convergence": config.get("require_convergence", True)}
    result = {"evaluation": "chronological_out_of_sample_forecasting", "status": "ok" if selected_metrics else "selected_test_failed",
            "selection": selection,
            "split": {key: {"indices": values.tolist(), "times": times[values].tolist(), "count": len(values)}
                      for key, values in split.items()},
            "test_times": times[test].tolist(), "selected_test_metrics": selected_metrics,
            "persistence_test_metrics": baseline_metrics, "test_metrics": test_metrics,
            "test_failures": test_failures, "improvement_over_persistence": improvements,
            "forecast_sha256": fingerprints, "postprocessing_profile": postprocessing_profile,
            "provenance": {"selection_inputs_sha256": hashlib.sha256(json.dumps(selection_input, sort_keys=True, allow_nan=False).encode()).hexdigest(),
                           "training_observations_sha256": _digest(training),
                           "validation_observations_sha256": _digest(val_observed),
                           "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest(),
                           "known_noise": known_noise, "preprocessing_scope": "train only, cached by declared parameters",
                           "density_scope": "clean reference density is retained for velocity-only physical metrics",
                           "selection_interpretation": "observed MSE equals clean risk plus constant noise variance in expectation only under independent zero-mean validation noise; temporal correlation/outliers may bias selection",
                           "uncertainty_scope": "variation across declared noise seeds conditional on these deterministic flows; no population/generalization confidence claim"}}
    if return_forecasts:
        result["test_predictions"] = {"selected": returned_predictions.get(selected["id"]),
                                      "persistence": returned_predictions.get("persistence")}
    return result
