from __future__ import annotations

import math
from typing import Any

import numpy as np


def soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def singular_value_threshold(matrix: np.ndarray, threshold: float) -> np.ndarray:
    u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
    shrunk = np.maximum(singular_values - threshold, 0.0)
    rank = int(np.count_nonzero(shrunk > 0.0))
    if rank == 0:
        return np.zeros_like(matrix)
    return (u[:, :rank] * shrunk[:rank]) @ vh[:rank, :]


def rpca_ialm(
    matrix: np.ndarray,
    *,
    lam: float | None = None,
    mu: float | None = None,
    rho: float = 1.5,
    max_iter: int = 500,
    tol: float = 1e-7,
) -> dict[str, Any]:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("RPCA-IALM expects a 2D matrix input")

    fro_norm = float(np.linalg.norm(matrix, ord="fro"))
    if fro_norm == 0.0:
        zeros = np.zeros_like(matrix)
        return {
            "low_rank": zeros,
            "sparse": zeros,
            "iterations": 0,
            "converged": True,
            "residual": 0.0,
            "effective_lambda": 0.0 if lam is None else float(lam),
            "effective_mu": 0.0 if mu is None else float(mu),
        }

    if lam is None:
        lam = 1.0 / math.sqrt(float(max(matrix.shape)))
    if lam <= 0.0 or not math.isfinite(lam):
        raise ValueError("RPCA lambda must be finite and positive")

    spectral_norm = float(np.linalg.norm(matrix, ord=2))
    inf_norm = float(np.max(np.abs(matrix)))
    dual_norm = max(spectral_norm, inf_norm / lam, 1e-12)
    y = matrix / dual_norm

    if mu is None:
        mu = 1.25 / max(spectral_norm, 1e-12)
    if mu <= 0.0 or not math.isfinite(mu):
        raise ValueError("RPCA mu must be finite and positive")

    mu_bar = mu * 1e7
    low_rank = np.zeros_like(matrix)
    sparse = np.zeros_like(matrix)
    residual = float("inf")
    converged = False
    iterations = 0

    for iterations in range(1, max_iter + 1):
        low_rank = singular_value_threshold(matrix - sparse + (1.0 / mu) * y, 1.0 / mu)
        sparse = soft_threshold(matrix - low_rank + (1.0 / mu) * y, lam / mu)
        residual_matrix = matrix - low_rank - sparse
        residual = float(np.linalg.norm(residual_matrix, ord="fro") / fro_norm)
        if residual < tol:
            converged = True
            break
        y = y + mu * residual_matrix
        mu = min(mu * rho, mu_bar)

    return {
        "low_rank": np.asarray(low_rank, dtype=np.float64),
        "sparse": np.asarray(sparse, dtype=np.float64),
        "iterations": iterations,
        "converged": converged,
        "residual": residual,
        "effective_lambda": float(lam),
        "effective_mu": float(mu),
    }


def truncated_svd_reconstruct(
    matrix: np.ndarray,
    *,
    rank: int,
) -> dict[str, Any]:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("Truncated SVD expects a 2D matrix input")
    if rank <= 0:
        raise ValueError("Truncated SVD rank must be positive")

    u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
    effective_rank = min(rank, int(singular_values.shape[0]))
    reconstructed = (
        (u[:, :effective_rank] * singular_values[:effective_rank]) @ vh[:effective_rank, :]
    )
    discarded_energy = float(np.sum(singular_values[effective_rank:] ** 2))
    total_energy = float(np.sum(singular_values**2))
    retained_energy_ratio = 1.0 if total_energy == 0.0 else 1.0 - discarded_energy / total_energy
    return {
        "reconstruction": np.asarray(reconstructed, dtype=np.float64),
        "effective_rank": int(effective_rank),
        "retained_energy_ratio": float(retained_energy_ratio),
        "singular_values": np.asarray(singular_values, dtype=np.float64),
    }


def dmd_reconstruct(
    matrix: np.ndarray,
    *,
    rank: int | None = None,
    exact: bool = True,
) -> dict[str, Any]:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("DMD expects a 2D matrix input")
    if matrix.shape[1] < 2:
        raise ValueError("DMD requires at least two temporal snapshots")

    x1 = matrix[:, :-1]
    x2 = matrix[:, 1:]
    u, singular_values, vh = np.linalg.svd(x1, full_matrices=False)

    max_rank = int(np.count_nonzero(singular_values > 1e-12))
    if max_rank == 0:
        return {
            "reconstruction": np.zeros_like(matrix),
            "effective_rank": 0,
            "eigenvalues": np.asarray([], dtype=np.complex128),
            "amplitudes": np.asarray([], dtype=np.complex128),
        }

    if rank is None:
        effective_rank = max_rank
    else:
        if rank <= 0:
            raise ValueError("DMD rank must be positive")
        effective_rank = min(rank, max_rank)

    u_r = u[:, :effective_rank]
    s_r = singular_values[:effective_rank]
    vh_r = vh[:effective_rank, :]

    a_tilde = u_r.T @ x2 @ vh_r.T @ np.diag(1.0 / s_r)
    eigenvalues, w = np.linalg.eig(a_tilde)

    if exact:
        modes = x2 @ vh_r.T @ np.diag(1.0 / s_r) @ w
    else:
        modes = u_r @ w

    amplitudes, *_ = np.linalg.lstsq(modes, matrix[:, 0], rcond=None)
    time_indices = np.arange(matrix.shape[1], dtype=np.float64)
    dynamics = np.zeros((effective_rank, matrix.shape[1]), dtype=np.complex128)
    for mode_index in range(effective_rank):
        dynamics[mode_index, :] = amplitudes[mode_index] * (eigenvalues[mode_index] ** time_indices)

    reconstruction = np.real(modes @ dynamics)
    return {
        "reconstruction": np.asarray(reconstruction, dtype=np.float64),
        "effective_rank": int(effective_rank),
        "eigenvalues": np.asarray(eigenvalues),
        "amplitudes": np.asarray(amplitudes),
    }


def _normalize_kernel_size(kernel_size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(kernel_size, int):
        kernel_y = kernel_size
        kernel_x = kernel_size
    else:
        kernel_y, kernel_x = kernel_size
    if kernel_y <= 0 or kernel_x <= 0:
        raise ValueError("Kernel size must be positive")
    if kernel_y % 2 == 0 or kernel_x % 2 == 0:
        raise ValueError("Kernel size must be odd along both axes")
    return int(kernel_y), int(kernel_x)


def median_filter_2d(
    image: np.ndarray,
    *,
    kernel_size: int | tuple[int, int] = 3,
    padding_mode: str = "reflect",
) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError("Median filter expects a 2D image")
    kernel_y, kernel_x = _normalize_kernel_size(kernel_size)
    pad_y = kernel_y // 2
    pad_x = kernel_x // 2
    padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode=padding_mode)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kernel_y, kernel_x))
    return np.median(windows, axis=(-2, -1))


def _local_mean_and_variance(
    image: np.ndarray,
    *,
    kernel_size: tuple[int, int],
    padding_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    kernel_y, kernel_x = kernel_size
    pad_y = kernel_y // 2
    pad_x = kernel_x // 2
    padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode=padding_mode)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kernel_y, kernel_x))
    local_mean = np.mean(windows, axis=(-2, -1))
    local_var = np.var(windows, axis=(-2, -1))
    return np.asarray(local_mean, dtype=np.float64), np.asarray(local_var, dtype=np.float64)


def wiener_filter_2d(
    image: np.ndarray,
    *,
    kernel_size: int | tuple[int, int] = 3,
    noise_power: float | None = None,
    padding_mode: str = "reflect",
) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError("Wiener filter expects a 2D image")
    kernel = _normalize_kernel_size(kernel_size)
    local_mean, local_var = _local_mean_and_variance(
        image,
        kernel_size=kernel,
        padding_mode=padding_mode,
    )
    if noise_power is None:
        noise_power = float(np.mean(local_var))
    if noise_power < 0.0 or not math.isfinite(noise_power):
        raise ValueError("noise_power must be finite and non-negative")

    centered = image - local_mean
    gain = np.maximum(local_var - noise_power, 0.0) / np.maximum(local_var, 1e-12)
    return np.asarray(local_mean + gain * centered, dtype=np.float64)
