"""Inspectable low-rank estimators for the extended thesis experiment.

Input is features x time. Estimators only access that observed matrix, never a
clean reference. ``rank`` is a user-specified budget for POD/DMD, while the
convex RPCA models select rank through their regularisation parameters.

Numerical failures raise ``ReconstructionFailure`` carrying JSON diagnostics.
No NaN/Inf is replaced by zero, and convergence is distinct from reconstruction
error. The FLOP counter is a declared dominant-operation *proxy*, not measured
hardware instructions; actual SVD, least-squares and eigensolver calls are also
reported separately.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


METHODS = ("identity", "pod", "rpca", "stable_rpca", "outlier_pursuit",
           "dmd", "tls_dmd", "fb_dmd", "optimized_dmd", "rpca_dmd", "stable_rpca_dmd")


class ReconstructionFailure(RuntimeError):
    """A failed estimator, with machine-readable details for a benchmark row."""

    def __init__(self, reason: str, diagnostics: dict):
        super().__init__(reason)
        self.diagnostics = {**diagnostics, "status": "failed", "converged": False, "reason": reason}


@dataclass
class _Work:
    svd_calls: int = 0
    lstsq_calls: int = 0
    eig_calls: int = 0
    flops: float = 0.0

    def svd(self, x):
        m, n = x.shape
        k = min(m, n)
        self.svd_calls += 1
        self.flops += 4.0 * m * n * k + (8.0 / 3.0) * k ** 3
        return np.linalg.svd(x, full_matrices=False)

    def lstsq(self, x, y):
        m, n = x.shape
        self.lstsq_calls += 1
        self.flops += 4.0 * m * n * min(m, n) + (8.0 / 3.0) * min(m, n) ** 3
        return np.linalg.lstsq(x, y, rcond=None)[0]

    def mm(self, a, b):
        self.flops += 2.0 * a.shape[0] * a.shape[1] * b.shape[1]
        return a @ b

    def eig(self, a):
        self.eig_calls += 1
        self.flops += 25.0 * a.shape[0] ** 3
        return np.linalg.eig(a)

    def diagnostics(self):
        return {
            "svd_calls": self.svd_calls, "lstsq_calls": self.lstsq_calls,
            "eigendecomposition_calls": self.eig_calls,
            "flops_proxy": float(self.flops),
            "flops_proxy_definition": (
                "dominant real arithmetic: thin SVD/lstsq 4*m*n*k+8*k^3/3; "
                "matmul 2*m*n*p; eig 25*r^3; fbDMD solve 2*r^3/3. Excludes elementwise, norms, "
                "optimizer overhead, complex-arithmetic factors; not exact FLOPs"
            ),
        }


def _soft(x, threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


def _svt(x, threshold, work):
    u, s, vh = work.svd(x)
    s = np.maximum(s - threshold, 0)
    r = int(np.count_nonzero(s > 0))
    return work.mm(u[:, :r] * s[:r], vh[:r]) if r else np.zeros_like(x), r


def _column_soft(x, threshold):
    norms = np.linalg.norm(x, axis=0)
    scales = np.zeros_like(norms)
    keep = norms > threshold
    scales[keep] = 1 - threshold / norms[keep]
    return x * scales[None, :]


def _positive(params, key, default):
    value = float(params.get(key, default))
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{key} must be finite and positive")
    return value


def _nonnegative(params, key, default):
    value = float(params.get(key, default))
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{key} must be finite and nonnegative")
    return value


def _count(params, key, default):
    value = params.get(key, default)
    if int(value) != value or value < 1:
        raise ValueError(f"{key} must be a positive integer")
    return int(value)


def _pcp(m, params, work, columnwise=False):
    """IALM for min ||L||_* + lam ||S||_1 (or column l2,1), M=L+S."""
    if columnwise:
        # Both nuclear norm and the sum of column Euclidean norms scale by
        # sqrt(k) if every feature is replicated k times. Therefore lambda
        # must not change under feature replication.
        gamma = float(params.get("gamma", .1))
        if not np.isfinite(gamma) or not 0 < gamma < 1:
            raise ValueError("gamma must be a column-outlier fraction upper bound in (0,1)")
        lam_default = 3 / (7 * np.sqrt(gamma * m.shape[1]))
        lambda_rule = "Xu et al. Theorem 1 parameter rule: 3/(7*sqrt(gamma*snapshot_count)); gamma default=.1 predeclared; theorem assumptions and true outlier bound are not verified, so no recovery guarantee"
    else:
        gamma = None
        lam_default = 1 / np.sqrt(max(m.shape))
        lambda_rule = "PCP heuristic 1/sqrt(max(shape))"
    lam = _positive(params, "lam", lam_default)
    if "lam" in params:
        lambda_rule = "explicit lam supplied by caller"
    tol = _positive(params, "tol", 1e-6)
    max_iter = _count(params, "max_iter", 300)
    _, singular_values, _ = work.svd(m)
    spectral = float(singular_values[0])
    dual_sparse = float(np.max(np.linalg.norm(m, axis=0))) if columnwise else float(np.max(np.abs(m)))
    y = m / max(spectral, dual_sparse / lam)
    mu = _positive(params, "mu", 1.25 / spectral)
    mu_bar = mu * 1e7
    rho = _positive(params, "rho", 1.5)
    if rho <= 1:
        raise ValueError("rho must exceed 1")
    low = np.zeros_like(m)
    sparse = np.zeros_like(m)
    norm = np.linalg.norm(m)
    for iteration in range(1, max_iter + 1):
        previous_sparse = sparse
        low, r = _svt(m - sparse + y / mu, 1 / mu, work)
        sparse = (_column_soft if columnwise else _soft)(m - low + y / mu, lam / mu)
        residual = m - low - sparse
        primal = float(np.linalg.norm(residual) / norm)
        y += mu * residual
        dual = float(mu * np.linalg.norm(sparse - previous_sparse) / max(np.linalg.norm(y), np.finfo(float).tiny))
        satisfied = primal <= tol and (not columnwise or dual <= tol)
        if satisfied:
            break
        if not columnwise:
            mu = min(mu * rho, mu_bar)
    sparse_cost = float(np.sum(np.linalg.norm(sparse, axis=0))) if columnwise else float(np.sum(np.abs(sparse)))
    return low, {
        "iterations": iteration, "effective_rank": r, "converged": bool(satisfied),
        "status": "ok" if satisfied else "not_converged", "primal_residual": primal,
        "dual_relative_residual": dual,
        "convergence_criterion": "relative_primal_and_dual_residuals" if columnwise else "primal_feasibility",
        "optimality_certified": False,
        "solver": "constant_mu_ADMM" if columnwise else "inexact_augmented_Lagrangian_increasing_mu",
        "decomposition_residual": primal, "lam": lam, "tol": tol,
        "lambda_rule": lambda_rule,
        "sparse_cost": sparse_cost, "sparse_fraction": float(np.count_nonzero(sparse) / sparse.size),
        "sparse_norm": float(np.linalg.norm(sparse)),
        "objective_definition": "min ||L||_* + lam*||C||_2,1 subject to M=L+C" if columnwise else "min ||L||_* + lam*||S||_1 subject to M=L+S",
        "rank_selection": "nuclear norm proximal threshold on observed data; requested rank not imposed",
        **({"gamma_upper_bound": gamma, "recovery_scope": "column subspace and outlier support; clean values within completely corrupted columns are not identifiable", "reference": "https://arxiv.org/abs/1010.4237"} if columnwise else {}),
    }


def _stable(m, params, work):
    """Exact alternating proximals for penalized stable RPCA.

    Solve .5||M-L-S||_F^2 + alpha||L||_* + beta||S||_1. The residual is
    explicitly dense and NOT forced to zero. This is a penalized SPCP model;
    it is not the constrained delta-ball program from Zhou et al.
    """
    tol = _positive(params, "tol", 1e-6)
    max_iter = _count(params, "max_iter", 300)
    _, singular_values, _ = work.svd(m)
    sigma_estimate = None
    if "noise_sigma" in params:
        sigma_estimate = _nonnegative(params, "noise_sigma", 1.0)
        penalty_rule = "supplied noise_sigma; alpha=sigma*(sqrt(m)+sqrt(n)), beta=sigma*sqrt(2*log(m*n))"
    elif "noise_intensity" in params:
        intensity = _nonnegative(params, "noise_intensity", .2)
        observed_rms = float(np.sqrt(np.mean(m ** 2)))
        sigma_estimate = intensity * observed_rms / np.sqrt(1 + intensity ** 2)
        penalty_rule = "known relative injection intensity and observed RMS: sigma=intensity*observed_RMS/sqrt(1+intensity^2); alpha=sigma*(sqrt(m)+sqrt(n)), beta=sigma*sqrt(2*log(m*n)); independence approximation"
    else:
        penalty_rule = "alpha=.03*observed spectral norm; beta=alpha/sqrt(max(shape)); use train/validation to tune"
    alpha_default = .03 * float(singular_values[0]) if sigma_estimate is None else sigma_estimate * (np.sqrt(m.shape[0]) + np.sqrt(m.shape[1]))
    alpha = _nonnegative(params, "alpha", alpha_default)
    beta_default = alpha / np.sqrt(max(m.shape)) if sigma_estimate is None else sigma_estimate * np.sqrt(2 * np.log(max(m.size, 2)))
    beta = _nonnegative(params, "beta", beta_default)
    low = np.zeros_like(m)
    sparse = np.zeros_like(m)
    norm = np.linalg.norm(m)
    change = float("inf")
    for iteration in range(1, max_iter + 1):
        previous_low, previous_sparse = low, sparse
        low, r = _svt(m - sparse, alpha, work)
        sparse = _soft(m - low, beta)
        change = float(np.sqrt(np.linalg.norm(low - previous_low) ** 2 + np.linalg.norm(sparse - previous_sparse) ** 2) / norm)
        if change <= tol:
            break
    residual = m - low - sparse
    # Nuclear norm recovered from a counted factorisation for the final objective.
    _, final_s, _ = work.svd(low)
    objective = .5 * np.linalg.norm(residual) ** 2 + alpha * np.sum(final_s) + beta * np.sum(np.abs(sparse))
    return low, {
        "iterations": iteration, "effective_rank": r, "converged": bool(change <= tol),
        "status": "ok" if change <= tol else "not_converged", "iterate_relative_change": change,
        "decomposition_residual": float(np.linalg.norm(residual) / norm),
        "dense_residual_norm": float(np.linalg.norm(residual)), "sparse_norm": float(np.linalg.norm(sparse)),
        "sparse_fraction": float(np.count_nonzero(sparse) / sparse.size),
        "alpha": alpha, "beta": beta, "tol": tol, "objective": float(objective),
        "objective_definition": "0.5*||M-L-S||_F^2 + alpha*||L||_* + beta*||S||_1",
        "rank_selection": "nuclear norm proximal threshold on observed data; requested rank not imposed",
        "default_penalties": penalty_rule, "noise_sigma_estimate": sigma_estimate,
    }


def _truncate(x, rank, work, params):
    u, s, vh = work.svd(x)
    rtol = _positive(params, "svd_rtol", max(x.shape) * np.finfo(float).eps)
    r = min(rank, int(np.count_nonzero(s > (s[0] * rtol if len(s) else 0))))
    if r == 0:
        raise np.linalg.LinAlgError("observed snapshot matrix has numerical rank zero")
    return u[:, :r], s[:r], vh[:r], r


def _times(times, n):
    t = np.arange(n, dtype=float) if times is None else np.asarray(times, dtype=float)
    if t.shape != (n,) or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError("times must be finite, strictly increasing, and match the snapshot count")
    return t - t[0]


def _amplitudes(modes, powers, m, work, strategy):
    if strategy == "first":
        return work.lstsq(modes, m[:, 0])
    if strategy != "global":
        raise ValueError("amplitude_fit must be 'global' or 'first'")
    gram = work.mm(modes.conj().T, modes) * work.mm(powers.conj(), powers.T)
    projected = work.mm(modes.conj().T, m)
    rhs = np.sum(projected * powers.conj(), axis=1)
    # SVD least squares handles a rank-deficient Gram matrix without inversion.
    return work.lstsq(gram, rhs)


def _dmd(m, method, rank, times, params, work):
    if m.shape[1] < 2:
        raise ValueError("DMD needs at least two snapshots")
    t = _times(times, m.shape[1])
    dt = float(np.mean(np.diff(t)))
    if not np.allclose(np.diff(t), dt, rtol=1e-6, atol=1e-12):
        raise ValueError("discrete DMD requires uniform times; use optimized_dmd for irregular times")
    x, y = m[:, :-1], m[:, 1:]
    rank = min(rank, min(x.shape))
    if method == "tls_dmd":
        _, _, vh, r = _truncate(np.vstack((x, y)), rank, work, params)
        projection = work.mm(vh.conj().T, vh)
        x, y = work.mm(x, projection), work.mm(y, projection)
    u, s, vh, r = _truncate(x, rank, work, params)
    operator = work.mm(work.mm(u.conj().T, y), vh.conj().T) / s[None, :]
    if method == "fb_dmd":
        xr, yr = work.mm(u.conj().T, x), work.mm(u.conj().T, y)
        backward = work.lstsq(yr.T, xr.T).T
        _, backward_s, _ = work.svd(backward)
        if backward_s[-1] <= backward_s[0] * 1e-12:
            raise np.linalg.LinAlgError("backward reduced operator is singular/ill-conditioned")
        corrected = np.linalg.solve(backward.T, operator.T).T
        work.flops += (2.0 / 3.0) * backward.shape[0] ** 3
        # A principal matrix square root loses negative discrete eigenvalues.
        # Resolve +/- roots in each eigendirection using the forward operator;
        # ties are genuinely ambiguous and must produce an explicit failure.
        squared_eigenvalues, root_vectors = work.eig(corrected)
        _, root_s, _ = work.svd(root_vectors)
        if root_s[-1] <= root_s[0] * 1e-10:
            raise np.linalg.LinAlgError("fbDMD square-root eigenbasis is ill-conditioned")
        forward_vectors = work.mm(operator, root_vectors)
        rayleigh = np.sum(root_vectors.conj() * forward_vectors, axis=0) / np.sum(np.abs(root_vectors) ** 2, axis=0)
        roots = np.sqrt(squared_eigenvalues.astype(complex))
        plus_distance = np.abs(roots - rayleigh)
        minus_distance = np.abs(-roots - rayleigh)
        margin = np.abs(plus_distance - minus_distance)
        if np.any(margin <= 1e-8 * np.maximum(1., np.abs(rayleigh))):
            raise np.linalg.LinAlgError("fbDMD square-root branch is ambiguous relative to forward dynamics")
        roots = np.where(minus_distance < plus_distance, -roots, roots)
        left = root_vectors * roots[None, :]
        operator = np.linalg.solve(root_vectors.T, left.T).T
        work.flops += (2.0 / 3.0) * root_vectors.shape[0] ** 3
        root_residual = float(np.linalg.norm(work.mm(operator, operator) - corrected) / max(np.linalg.norm(corrected), np.finfo(float).tiny))
        if root_residual > 1e-7:
            raise np.linalg.LinAlgError("fbDMD selected matrix square root failed its residual check")
    eigenvalues, vectors = work.eig(operator)
    modes = work.mm(u, vectors) if method == "fb_dmd" else work.mm(work.mm(y, vh.conj().T) / s[None, :], vectors)
    powers = eigenvalues[:, None] ** np.arange(m.shape[1])[None, :]
    if not np.all(np.isfinite(powers)):
        raise FloatingPointError("DMD exponential evolution overflowed")
    strategy = params.get("amplitude_fit", "global")
    amplitudes = _amplitudes(modes, powers, m, work, strategy)
    reconstructed = work.mm(modes, amplitudes[:, None] * powers)
    return reconstructed.real, {
        "iterations": 1, "effective_rank": r, "converged": True, "status": "ok",
        "eigenvalues_real": eigenvalues.real.tolist(), "eigenvalues_imag": eigenvalues.imag.tolist(),
        "spectral_radius": float(np.max(np.abs(eigenvalues))), "dt": dt,
        "amplitude_fit": strategy, "imaginary_relative_norm": float(np.linalg.norm(reconstructed.imag) / max(np.linalg.norm(m), np.finfo(float).tiny)),
        "rank_selection": "requested rank clipped to observed numerical rank",
        "operator_definition": "common observed POD coordinates, sqrt(A_forward*inv(A_backward)) branch nearest forward Rayleigh quotient; ambiguous branches fail" if method == "fb_dmd" else method,
        **({"square_root_relative_residual": root_residual, "minimum_branch_margin": float(np.min(margin))} if method == "fb_dmd" else {}),
    }


def _optimized(m, rank, times, params, work):
    """Variable projection exp(t*omega) fit in the observed POD subspace.

    This implements the separable exponential fitting formulation of optimized
    DMD, with SciPy finite-difference trust-region least squares. It is not a
    call to BOP-DMD and provides no bootstrap uncertainty estimate.
    """
    if m.shape[1] < 3:
        raise ValueError("optimized_dmd needs at least three snapshots")
    t = _times(times, m.shape[1])
    u, _, _, r = _truncate(m, min(rank, m.shape[1] - 1), work, params)
    y = work.mm(u.T, m).T.astype(complex)
    initial_operator = work.lstsq(y[:-1], y[1:]).T
    eigs, _ = work.eig(initial_operator)
    average_dt = float(np.mean(np.diff(t)))
    initial = (np.log(np.maximum(np.abs(eigs), 1e-12)) + 1j * np.angle(eigs)) / average_dt
    rate_limit = _positive(params, "rate_limit", 40.0) / t[-1]
    frequency_limit = float(np.pi / np.min(np.diff(t)))
    lower = np.r_[np.full(r, -rate_limit), np.full(r, -frequency_limit)]
    upper = -lower
    initial_parameters = np.clip(np.r_[initial.real, initial.imag], lower + 1e-8, upper - 1e-8)
    max_nfev = _count(params, "max_nfev", 150)
    starts = _count(params, "n_starts", 2)
    if starts > 8:
        raise ValueError("n_starts is limited to 8 to keep the benchmark bounded")
    rng = np.random.default_rng(params.get("seed", 0))
    total_evaluations = 0

    def fit(parameters):
        omega = parameters[:r] + 1j * parameters[r:]
        basis = np.exp(t[:, None] * omega[None, :])
        coefficients = work.lstsq(basis, y)
        return basis, coefficients, work.mm(basis, coefficients) - y

    def objective(parameters):
        nonlocal total_evaluations
        total_evaluations += 1
        _, _, residual = fit(parameters)
        return np.r_[residual.real.ravel(), residual.imag.ravel()]

    outcomes = []
    for start in range(starts):
        init = initial_parameters.copy()
        if start:
            init += rng.normal(size=2 * r) * np.r_[np.full(r, .05 / t[-1]), np.full(r, .2 / average_dt)]
            init = np.clip(init, lower + 1e-8, upper - 1e-8)
        result = least_squares(objective, init, bounds=(lower, upper), max_nfev=max_nfev,
                               ftol=1e-9, xtol=1e-9, gtol=1e-9)
        outcomes.append(result)
    best = min(outcomes, key=lambda result: result.cost)
    basis, coefficients, _ = fit(best.x)
    reconstruction = work.mm(u.astype(complex), work.mm(basis, coefficients).T)
    omega = best.x[:r] + 1j * best.x[r:]
    return reconstruction.real, {
        "iterations": int(best.nfev), "effective_rank": r, "converged": bool(best.success),
        "status": "ok" if best.success else "not_converged", "optimizer_message": str(best.message),
        "optimizer_status": int(best.status), "optimizer_cost": float(best.cost),
        "optimizer_optimality": float(best.optimality), "residual_evaluations": total_evaluations,
        "n_starts": starts, "starts_converged": int(sum(bool(out.success) for out in outcomes)),
        "rates_real": omega.real.tolist(), "rates_imag": omega.imag.tolist(),
        "rate_bounds": [-rate_limit, rate_limit], "frequency_bounds": [-frequency_limit, frequency_limit],
        "active_bound_count": int(np.count_nonzero(best.active_mask)),
        "imaginary_relative_norm": float(np.linalg.norm(reconstruction.imag) / max(np.linalg.norm(m), np.finfo(float).tiny)),
        "rank_selection": "requested rank clipped to observed POD numerical rank",
        "objective_definition": "min_omega ||Y - exp(t*omega)*B(omega)||_F^2; Y=observed POD coordinates; B=least-squares variable projection",
    }


def reconstruct(
    matrix: np.ndarray,
    method: str,
    rank: int,
    times: np.ndarray | None = None,
    **params,
) -> tuple[np.ndarray, dict]:
    """Reconstruct a finite real observed matrix and return JSON diagnostics.

    ``rpca`` is equality-constrained PCP; ``stable_rpca`` is the penalized
    low-rank+sparse+dense model documented above; ``outlier_pursuit`` uses
    column sparsity. Its predeclared default ``gamma=.1`` gives
    lam=3/(7*sqrt(gamma*snapshot_count)) from Xu et al. (2010). Supply gamma
    as an upper bound on the outlier-column fraction; the true fraction and
    theorem's incoherence/small-outlier-fraction assumptions are not checked,
    so this parameter rule supplies no automatic recovery guarantee. This model
    targets the clean column subspace and outlier support; it cannot identify
    missing clean values within fully corrupted columns.
    ``dmd``/``tls_dmd`` use exact modes; ``fb_dmd`` uses a
    common POD basis and chooses each square-root sign nearest the forward
    Rayleigh quotient, rejecting ambiguous/ill-conditioned roots. DMD amplitudes fit
    all observed snapshots by default (set ``amplitude_fit='first'`` for the
    first-snapshot convention). ``rpca_dmd``/``stable_rpca_dmd`` sequentially
    apply their named decomposition then DMD to its low-rank component. Set
    ``modal_method`` to dmd/tls_dmd/fb_dmd/optimized_dmd for a hybrid variant;
    optional ``modal_params`` override shared parameters for that stage.
    For stable RPCA, ``noise_sigma`` or ``noise_intensity`` optionally sets a
    predeclared noise-scale penalty rule; intensity estimates sigma from the
    observed RMS under an independent additive-noise approximation. Explicit
    alpha/beta override those defaults. Use held-out noisy data to tune
    rank/penalties.
    """
    original = np.asarray(matrix)
    if original.ndim != 2 or any(d == 0 for d in original.shape) or np.iscomplexobj(original):
        raise ValueError("matrix must be a nonempty real (features, time) matrix")
    m = np.asarray(original, dtype=np.float64)
    if not np.all(np.isfinite(m)):
        raise ValueError("matrix must contain finite values")
    if isinstance(rank, bool) or int(rank) != rank or rank < 1:
        raise ValueError("rank must be a positive integer")
    rank = int(rank)
    aliases = {"svd": "pod", "truncated_svd": "pod", "pcp": "rpca",
               "exact_dmd": "dmd", "tlsdmd": "tls_dmd", "fbdmd": "fb_dmd",
               "optdmd": "optimized_dmd", "stable_rpca_penalized": "stable_rpca"}
    method = aliases.get(method, method)
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected {METHODS}")
    if times is not None:
        _times(times, m.shape[1])
    base = {"method": method, "shape": list(m.shape), "rank_requested": rank}
    work = _Work()
    with np.errstate(over="ignore", invalid="ignore"):
        norm = float(np.linalg.norm(m))
    if not np.isfinite(norm):
        raise ReconstructionFailure("observed matrix norm overflowed; rescale the input", {**base, **work.diagnostics()})
    if norm == 0:
        return np.zeros_like(m), {
            **base, "iterations": 0, "effective_rank": 0, "residual": 0.0,
            "converged": True, "status": "ok", "zero_input": True, **work.diagnostics(),
        }
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            if method == "identity":
                output = m.copy()
                diagnostics = {"iterations": 0, "effective_rank": None, "converged": True, "status": "ok",
                               "rank_selection": "identity does not estimate matrix rank"}
            elif method == "pod":
                u, s, vh, r = _truncate(m, rank, work, params)
                output = work.mm(u * s, vh)
                diagnostics = {"iterations": 1, "effective_rank": r, "converged": True, "status": "ok",
                               "rank_selection": "requested rank clipped to observed numerical rank"}
            elif method in ("rpca", "outlier_pursuit"):
                output, diagnostics = _pcp(m, params, work, columnwise=method == "outlier_pursuit")
            elif method == "stable_rpca":
                output, diagnostics = _stable(m, params, work)
            elif method in ("rpca_dmd", "stable_rpca_dmd"):
                low, preprocessing = (_pcp if method == "rpca_dmd" else _stable)(m, params, work)
                preprocessing = {**preprocessing, **work.diagnostics()}
                modal_method = params.get("modal_method", "dmd")
                if modal_method not in ("dmd", "tls_dmd", "fb_dmd", "optimized_dmd"):
                    raise ValueError("hybrid modal_method must be dmd, tls_dmd, fb_dmd or optimized_dmd")
                modal_params = {**params, **params.get("modal_params", {})}
                modal_work = _Work()
                try:
                    if np.linalg.norm(low) == 0:
                        output = np.zeros_like(low)
                        modal = {"iterations": 0, "effective_rank": 0, "status": "ok", "converged": True, "zero_input": True}
                    elif modal_method == "optimized_dmd":
                        output, modal = _optimized(low, rank, times, modal_params, modal_work)
                    else:
                        output, modal = _dmd(low, modal_method, rank, times, modal_params, modal_work)
                finally:
                    work.svd_calls += modal_work.svd_calls
                    work.lstsq_calls += modal_work.lstsq_calls
                    work.eig_calls += modal_work.eig_calls
                    work.flops += modal_work.flops
                modal = {"method": modal_method, **modal, **modal_work.diagnostics()}
                converged = preprocessing["converged"] and modal["converged"]
                diagnostics = {"iterations": preprocessing["iterations"] + modal["iterations"],
                               "effective_rank": modal["effective_rank"], "converged": converged,
                               "status": "ok" if converged else "not_converged",
                               "preprocessing": preprocessing, "modal": modal,
                               "rank_selection": "penalized observed-data preprocessing, then requested modal rank"}
            elif method == "optimized_dmd":
                output, diagnostics = _optimized(m, rank, times, params, work)
            else:
                output, diagnostics = _dmd(m, method, rank, times, params, work)
            if not np.all(np.isfinite(output)):
                raise FloatingPointError("estimator produced nonfinite reconstruction")
            residual = float(np.linalg.norm(m - output) / norm)
    except (np.linalg.LinAlgError, FloatingPointError, OverflowError) as error:
        raise ReconstructionFailure(str(error), {**base, **work.diagnostics()}) from error
    return output, {**base, **diagnostics, "residual": residual, **work.diagnostics()}
