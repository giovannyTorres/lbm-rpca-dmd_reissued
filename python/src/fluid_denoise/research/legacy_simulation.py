"""Isolated legacy D2Q9 cylinder benchmark retained for comparisons.

This solver is separate from the phase-1 C++ solver. Its periodic transverse
streaming and simplified open boundaries have no cylinder DNS certification.
"""
from __future__ import annotations

import numpy as np

# D2Q9 model constants
C = np.array(
    [
        [0, 0],
        [1, 0],
        [0, 1],
        [-1, 0],
        [0, -1],
        [1, 1],
        [-1, 1],
        [-1, -1],
        [1, -1],
    ],
    dtype=np.int32,
)
W = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36], dtype=np.float64)
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)


def _equilibrium(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    cu = np.zeros((9, *rho.shape), dtype=np.float64)
    usq = ux**2 + uy**2
    for i in range(9):
        ci_dot_u = C[i, 0] * ux + C[i, 1] * uy
        cu[i] = 1 + 3 * ci_dot_u + 4.5 * ci_dot_u**2 - 1.5 * usq
    return W[:, None, None] * rho[None, :, :] * cu


def cylinder_mask(nx: int, ny: int, radius_ratio: float = 0.1) -> np.ndarray:
    radius = max(3, int(radius_ratio * ny))
    cx = int(nx * 0.25)
    cy = ny // 2
    y, x = np.ogrid[:ny, :nx]
    return (x - cx) ** 2 + (y - cy) ** 2 <= radius**2


def simulate_von_karman(
    nx: int,
    ny: int,
    n_steps: int = 900,
    sample_every: int = 15,
    warmup: int = 250,
    u_in: float = 0.08,
    reynolds: float = 140.0,
    return_fields: bool = False,
) -> tuple[np.ndarray, np.ndarray] | dict:
    """Simulate 2D flow past a cylinder and return sampled speed frames and obstacle mask.

    Returns:
        speed_frames: (n_frames, ny, nx)
        obstacle: (ny, nx) bool
    """
    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be greater than 0.")
    if n_steps <= 0:
        raise ValueError("n_steps must be greater than 0.")
    if sample_every <= 0:
        raise ValueError("sample_every must be greater than 0.")
    if warmup < 0:
        raise ValueError("warmup cannot be negative.")

    if not 0 < u_in < 0.2 or reynolds <= 0:
        raise ValueError("Require 0 < u_in < 0.2 and positive Reynolds number.")
    obstacle = cylinder_mask(nx, ny)
    nu = u_in * (2 * np.sqrt(obstacle.sum() / np.pi)) / reynolds
    tau = 3 * nu + 0.5
    omega = 1.0 / tau

    rho = np.ones((ny, nx), dtype=np.float64)
    ux = np.full((ny, nx), u_in, dtype=np.float64)
    uy = np.zeros((ny, nx), dtype=np.float64)
    ux[obstacle] = 0.0

    f = _equilibrium(rho, ux, uy)
    frames = []
    velocity_frames, density_frames, times = [], [], []

    for step in range(n_steps):
        # Streaming
        for i in range(9):
            f[i] = np.roll(f[i], shift=C[i, 0], axis=1)
            f[i] = np.roll(f[i], shift=C[i, 1], axis=0)

        # Bounce-back at obstacle
        bounced = f[:, obstacle].copy()
        f[:, obstacle] = bounced[OPP]

        # Inlet: enforce velocity profile (constant)
        ux[:, 0] = u_in
        uy[:, 0] = 0.0
        rho[:, 0] = (
            f[0, :, 0]
            + f[2, :, 0]
            + f[4, :, 0]
            + 2.0 * (f[3, :, 0] + f[6, :, 0] + f[7, :, 0])
        ) / (1 - ux[:, 0])
        feq_in = _equilibrium(rho[:, :1], ux[:, :1], uy[:, :1])
        f[:, :, 0] = feq_in[:, :, 0]

        # Outlet: simple copy
        f[:, :, -1] = f[:, :, -2]

        # Macroscopic quantities
        rho = np.sum(f, axis=0)
        safe_rho = np.maximum(rho, 1e-12)
        ux = (f[1] + f[5] + f[8] - (f[3] + f[6] + f[7])) / safe_rho
        uy = (f[2] + f[5] + f[6] - (f[4] + f[7] + f[8])) / safe_rho
        ux[obstacle] = 0.0
        uy[obstacle] = 0.0

        # Collision
        feq = _equilibrium(rho, ux, uy)
        f += -omega * (f - feq)

        if step >= warmup and (step - warmup) % sample_every == 0:
            if not (np.isfinite(f).all() and np.isfinite(rho).all() and (rho > 0).all()):
                raise FloatingPointError(f"LBM unstable at step {step}; refine grid/parameters.")
            if return_fields:
                velocity_frames.append(np.stack((ux, uy)))
                density_frames.append(rho.copy())
                times.append(float(step + 1))
            speed = np.sqrt(ux**2 + uy**2)
            speed[obstacle] = 0.0
            frames.append(speed.astype(np.float32))

    if not frames:
        raise ValueError("No frames were sampled. Increase n_steps or reduce warmup/sample_every.")

    if return_fields:
        return {
            "velocity": np.stack(velocity_frames), "density": np.stack(density_frames),
            "times": np.asarray(times), "dx": 1.0, "dy": 1.0,
            "mask": obstacle, "boundary": "open",
            "metadata": {
                "solver": "legacy_D2Q9_BGK", "case": "cylinder", "units": "lattice",
                "reynolds": reynolds, "u_in": u_in, "viscosity": float(nu),
                "tau": float(tau), "grid_convention": "nodal",
                "periodic_axes": ["y"], "reference_status": "unvalidated_legacy_LBM",
                "limitations": "Periodic transverse streaming, equilibrium inlet and copy outlet; no cylinder DNS convergence demonstrated.",
            },
        }
    return np.stack(frames, axis=0), obstacle


def simulate_von_karman_fields(**kwargs) -> dict:
    """Legacy cylinder benchmark retaining u, v, rho in lattice units.

    This preserves the legacy boundaries; it is not a validated cylinder DNS.
    """
    kwargs.pop("return_fields", None)
    return simulate_von_karman(**kwargs, return_fields=True)
