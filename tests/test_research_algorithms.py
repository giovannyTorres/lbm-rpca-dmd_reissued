"""Scientific checks against known dynamics and planted contamination."""
import json

import numpy as np
import pytest

from fluid_denoise.research.algorithms import METHODS, ReconstructionFailure, reconstruct
from fluid_denoise.research.noise import NOISE_KINDS, add_noise


def _linear_dynamics():
    # Two independent decay rates, sampled with physical dt rather than index.
    times = np.arange(40) * .1
    states = np.vstack((np.exp(-.3 * times), 2 * np.exp(-.8 * times)))
    mixing = np.array([[1., .1], [.2, 1.], [-.4, .5], [.3, -.2]])
    return mixing @ states, times


def _planted_lowrank():
    rng = np.random.default_rng(22)
    left, _ = np.linalg.qr(rng.normal(size=(50, 3)))
    right, _ = np.linalg.qr(rng.normal(size=(40, 3)))
    return (left * np.array([20, 13, 8])) @ right.T, rng


@pytest.mark.parametrize("method", ["dmd", "tls_dmd", "fb_dmd", "optimized_dmd"])
def test_exact_linear_system_and_rates(method):
    clean, times = _linear_dynamics()
    result, diagnostics = reconstruct(clean, method, 2, times)
    assert np.linalg.norm(result - clean) / np.linalg.norm(clean) < 1e-7
    assert diagnostics["converged"]
    assert diagnostics["effective_rank"] == 2
    assert diagnostics["svd_calls"] >= 1
    assert diagnostics["flops_proxy"] > 0
    if method == "optimized_dmd":
        np.testing.assert_allclose(sorted(diagnostics["rates_real"]), [-.8, -.3], atol=1e-6)
    else:
        np.testing.assert_allclose(sorted(diagnostics["eigenvalues_real"]), sorted(np.exp(np.array([-.8, -.3]) * .1)), atol=1e-7)
    json.dumps(diagnostics, allow_nan=False)


def test_dmd_conjugate_oscillatory_modes():
    t = np.arange(35) * .2
    clean = np.vstack([np.cos(1.3 * t), np.sin(1.3 * t)]) * np.exp(-.05 * t)
    for method in ["dmd", "tls_dmd", "fb_dmd", "optimized_dmd"]:
        result, diagnostics = reconstruct(clean, method, 2, t, n_starts=1)
        assert np.linalg.norm(result - clean) / np.linalg.norm(clean) < 1e-7
        assert diagnostics["imaginary_relative_norm"] < 1e-7


def test_forward_backward_dmd_preserves_negative_discrete_eigenvalue():
    state = (-.8) ** np.arange(20)
    clean = np.vstack([state, 2 * state])
    result, diagnostics = reconstruct(clean, "fb_dmd", 1)
    np.testing.assert_allclose(result, clean, atol=1e-10)
    assert diagnostics["eigenvalues_real"] == pytest.approx([-.8])
    assert diagnostics["square_root_relative_residual"] < 1e-10


def test_forward_backward_ambiguous_square_root_explicitly_fails():
    t = np.arange(20)
    clean = np.vstack([np.cos(np.pi * t / 2), np.sin(np.pi * t / 2)]) * .9 ** t
    with pytest.raises(ReconstructionFailure, match="ambiguous"):
        reconstruct(clean, "fb_dmd", 2)


@pytest.mark.parametrize("method", METHODS)
def test_zero_matrix_is_finite_and_requires_no_factorization(method):
    result, diagnostics = reconstruct(np.zeros((8, 5)), method, 2)
    assert np.all(result == 0)
    assert diagnostics["effective_rank"] == 0
    assert diagnostics["svd_calls"] == 0
    assert diagnostics["residual"] == 0
    assert diagnostics["converged"]
    json.dumps(diagnostics, allow_nan=False)


def test_pcp_recovers_incoherent_lowrank_from_entrywise_outliers():
    clean, rng = _planted_lowrank()
    outliers = (rng.random(clean.shape) < .04) * rng.normal(0, 7, clean.shape)
    noisy = clean + outliers
    estimate, diagnostics = reconstruct(noisy, "rpca", 1)
    assert np.linalg.norm(estimate - clean) / np.linalg.norm(clean) < 2e-5
    assert diagnostics["converged"]
    assert diagnostics["effective_rank"] == 3  # penalty, not the requested rank
    assert diagnostics["primal_residual"] <= diagnostics["tol"]
    assert diagnostics["svd_calls"] == diagnostics["iterations"] + 1


def test_dense_noise_pod_and_penalized_stable_rpca_reduce_error():
    clean, rng = _planted_lowrank()
    sigma = .06
    noisy = clean + rng.normal(0, sigma, clean.shape)
    input_error = np.linalg.norm(noisy - clean)
    pod, _ = reconstruct(noisy, "pod", 3)
    stable, diagnostics = reconstruct(
        noisy, "stable_rpca", 3,
        alpha=sigma * (np.sqrt(50) + np.sqrt(40)),
        beta=sigma * np.sqrt(2 * np.log(clean.size)), max_iter=1000,
    )
    assert np.linalg.norm(pod - clean) < .6 * input_error
    assert np.linalg.norm(stable - clean) < input_error
    assert diagnostics["converged"]
    assert diagnostics["dense_residual_norm"] > 0
    assert "0.5*" in diagnostics["objective_definition"]


def test_column_outlier_decomposition_and_explicit_nonconvergence():
    clean, rng = _planted_lowrank()
    noisy = clean.copy()
    noisy[:, [2, 9, 25]] += rng.normal(0, 6, (50, 3))
    result, diagnostics = reconstruct(noisy, "outlier_pursuit", 3, lam=.6)
    assert np.linalg.norm(result - clean) < .2 * np.linalg.norm(noisy - clean)
    assert diagnostics["converged"]
    assert "2,1" in diagnostics["objective_definition"]
    _, unfinished = reconstruct(noisy, "rpca", 3, max_iter=1, tol=1e-12)
    assert not unfinished["converged"]
    assert unfinished["status"] == "not_converged"


def test_column_penalty_and_decomposition_invariant_to_repeated_features():
    clean, rng = _planted_lowrank()
    noisy = clean.copy()
    noisy[:, [2, 9, 25]] += rng.normal(0, 6, (50, 3))
    base, first = reconstruct(noisy, "outlier_pursuit", 3)
    repeated = np.tile(noisy, (3, 1))
    duplicated, second = reconstruct(repeated, "outlier_pursuit", 3)
    assert first["lam"] == second["lam"] == pytest.approx(3 / (7 * np.sqrt(.1 * noisy.shape[1])))
    np.testing.assert_allclose(duplicated, np.tile(base, (3, 1)), atol=1e-8, rtol=1e-8)
    assert first["effective_rank"] == second["effective_rank"]
    assert first["iterations"] == second["iterations"]
    _, theoretical = reconstruct(noisy, "outlier_pursuit", 3, gamma=.1)
    assert theoretical["lam"] == pytest.approx(3 / (7 * np.sqrt(.1 * noisy.shape[1])))
    assert "not identifiable" in theoretical["recovery_scope"]


def test_column_model_does_not_collapse_clean_tall_rank_one_matrix():
    clean = np.ones((200, 10))
    result, diagnostics = reconstruct(clean, "outlier_pursuit", 1)
    np.testing.assert_allclose(result, clean, atol=1e-6)
    assert diagnostics["effective_rank"] == 1
    assert diagnostics["converged"]
    assert diagnostics["gamma_upper_bound"] == .1
    assert "no recovery guarantee" in diagnostics["lambda_rule"]
    # The infeasible old default has a known optimal L=0. Constant-mu ADMM
    # must reach that optimum instead of stopping on an early feasible split.
    collapsed, other = reconstruct(clean, "outlier_pursuit", 1, lam=1 / np.sqrt(200), max_iter=500)
    assert other["converged"]
    assert np.linalg.norm(collapsed) < 1e-7
    assert other["convergence_criterion"] == "relative_primal_and_dual_residuals"


def test_irregular_sampling_only_supported_by_continuous_fit():
    t = np.cumsum(np.random.default_rng(7).uniform(.05, .15, 35))
    t -= t[0]
    clean = np.vstack([np.exp(-.3 * t), np.exp(-.8 * t)])
    with pytest.raises(ValueError, match="uniform"):
        reconstruct(clean, "dmd", 2, t)
    result, diagnostics = reconstruct(clean, "optimized_dmd", 2, t, n_starts=2)
    assert np.linalg.norm(result - clean) / np.linalg.norm(clean) < 1e-6
    assert diagnostics["converged"]


@pytest.mark.parametrize("method", ["rpca_dmd", "stable_rpca_dmd"])
def test_hybrid_recovers_dynamics_and_aggregates_work(method):
    rng = np.random.default_rng(44)
    spatial, _ = np.linalg.qr(rng.normal(size=(50, 2)))
    times = np.arange(45) * .1
    clean = spatial @ np.vstack([np.cos(times), np.sin(times)])
    noisy = clean + (rng.random(clean.shape) < .025) * rng.normal(0, 2, clean.shape)
    result, diagnostics = reconstruct(noisy, method, 2, times, max_iter=500)
    assert np.linalg.norm(result - clean) / np.linalg.norm(clean) < .1
    assert diagnostics["converged"]
    pre, modal = diagnostics["preprocessing"], diagnostics["modal"]
    assert diagnostics["svd_calls"] == pre["svd_calls"] + modal["svd_calls"]
    assert diagnostics["flops_proxy"] == pytest.approx(pre["flops_proxy"] + modal["flops_proxy"])
    json.dumps(diagnostics, allow_nan=False)


def test_noise_informed_penalties_only_use_observed_scale():
    clean, rng = _planted_lowrank()
    noisy = clean + rng.normal(0, .1, clean.shape)
    _, diagnostics = reconstruct(noisy, "stable_rpca", 3, noise_intensity=.2)
    expected_sigma = .2 * np.sqrt(np.mean(noisy ** 2)) / np.sqrt(1 + .2 ** 2)
    assert diagnostics["noise_sigma_estimate"] == pytest.approx(expected_sigma)
    assert diagnostics["alpha"] == pytest.approx(expected_sigma * (np.sqrt(50) + np.sqrt(40)))
    unregularized, zero_diagnostics = reconstruct(noisy, "stable_rpca", 3, noise_intensity=0)
    np.testing.assert_allclose(unregularized, noisy, atol=1e-12)
    assert zero_diagnostics["noise_sigma_estimate"] == 0


def test_invalid_data_and_overflow_are_not_silently_sanitized():
    with pytest.raises(ValueError, match="finite"):
        reconstruct(np.array([[1, np.nan], [2, 3]]), "dmd", 1)
    with pytest.raises(ValueError, match="positive integer"):
        reconstruct(np.ones((3, 4)), "pod", 0)
    # Huge finite data should either remain finite or explicitly fail. A
    # failure must carry a valid benchmark diagnostic, not a zero replacement.
    with pytest.raises(ReconstructionFailure) as caught:
        reconstruct(np.full((3, 4), 1e200), "pod", 1)
    assert caught.value.diagnostics["status"] == "failed"
    assert not caught.value.diagnostics["converged"]
    json.dumps(caught.value.diagnostics, allow_nan=False)


@pytest.mark.parametrize("kind", NOISE_KINDS)
def test_noise_reproducibility_equal_rms_and_solids(kind):
    rng = np.random.default_rng(5)
    velocity = rng.normal(size=(40, 2, 9, 11))
    mask = np.zeros((9, 11), dtype=bool)
    mask[3:6, 4:7] = True
    first, diagnostics = add_noise(velocity, kind, .2, seed=27, obstacle_mask=mask)
    second, second_diagnostics = add_noise(velocity, kind, .2, seed=27, obstacle_mask=mask)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[..., mask], velocity[..., mask])
    assert diagnostics == second_diagnostics
    assert diagnostics["realized_relative_rms"] == pytest.approx(.2)
    different, _ = add_noise(velocity, kind, .2, seed=28, obstacle_mask=mask)
    assert not np.array_equal(first, different)
    json.dumps(diagnostics, allow_nan=False)


def test_noise_models_have_distinct_support_and_temporal_correlation():
    velocity = np.ones((100, 2, 15, 15))
    sparse, sparse_d = add_noise(velocity, "sparse_impulsive", .1, fraction=.03)
    columns, column_d = add_noise(velocity, "column_outliers", .1, fraction=.03)
    temporal, _ = add_noise(velocity, "temporal_correlated", .1, correlation=.85)
    gaussian, _ = add_noise(velocity, "gaussian", .1)
    assert .02 < sparse_d["nonzero_fraction"] < .04
    affected = np.flatnonzero(np.any(columns != velocity, axis=(1, 2, 3)))
    assert affected.tolist() == column_d["settings"]["contaminated_snapshots"]
    assert len(affected) == 3
    ar_noise, independent = temporal - velocity, gaussian - velocity
    assert np.corrcoef(ar_noise[:-1].ravel(), ar_noise[1:].ravel())[0, 1] > .8
    assert abs(np.corrcoef(independent[:-1].ravel(), independent[1:].ravel())[0, 1]) < .05


def test_zero_noise_budget_and_invalid_noise_parameters():
    result, diagnostics = add_noise(np.zeros((3, 2, 4, 4)), "gaussian", .2)
    assert np.all(result == 0)
    assert diagnostics["noise_rms"] == 0
    with pytest.raises(ValueError, match="correlation"):
        add_noise(np.ones((3, 2, 4, 4)), "temporal_correlated", correlation=1)
    with pytest.raises(ValueError, match="fraction"):
        add_noise(np.ones((3, 2, 4, 4)), "sparse_impulsive", fraction=0)
