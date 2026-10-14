#!/usr/bin/env python3
"""Independent adversarial checks, separate from the reconstruction campaign.

Run from any directory. The JSON records exact inputs, criteria and source
hashes. These checks verify numerical contracts and physical identities;
they do not certify DNS resolution or general algorithm superiority.
"""
from __future__ import annotations

import os
for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[name] = "1"

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))

import numpy as np

from fluid_denoise.research.algorithms import ReconstructionFailure, reconstruct
from fluid_denoise.research.cost import quality_gates
from fluid_denoise.research.dns import simulate_periodic_ns
from fluid_denoise.research.metrics import physical_metrics, spatial_derivative
from fluid_denoise.research.rheology import herschel_bulkley_channel, oldroyd_b_channel
from fluid_denoise.research.runner import assert_finite


def relative_error(candidate, reference):
    return float(np.linalg.norm(candidate - reference) / np.linalg.norm(reference))


def flow_quantities(run):
    velocity, dx, dy = run["velocity"], run["dx"], run["dy"]
    omega = (spatial_derivative(velocity[:, 1], dx, -1, True)
             - spatial_derivative(velocity[:, 0], dy, -2, True))
    energy = .5 * np.mean(np.sum(velocity ** 2, axis=1), axis=(-2, -1))
    enstrophy = .5 * np.mean(omega ** 2, axis=(-2, -1))
    return omega, energy, enstrophy


def algorithm_checks():
    clean = np.ones((300, 20))
    output, diagnostics = reconstruct(clean, "outlier_pursuit", 1, max_iter=1000)
    default_error = relative_error(output, clean)
    # Norm inequalities imply the exact optimum is L=0 for this old lambda.
    old_lambda = float(1 / np.sqrt(300))
    old_output, old_diagnostics = reconstruct(
        clean, "outlier_pursuit", 1, lam=old_lambda, max_iter=1000)
    old_norm_ratio = float(np.linalg.norm(old_output) / np.linalg.norm(clean))
    times = np.arange(20, dtype=float)
    negative = np.vstack([(-.8) ** times, 2 * (-.8) ** times])
    output, negative_diagnostics = reconstruct(negative, "fb_dmd", 1, times)
    negative_error = relative_error(output, negative)
    rotations = []
    for theta in (.6, 2.1):
        observed = np.vstack([np.cos(theta * times), np.sin(theta * times)]) * .9 ** times
        output, diag = reconstruct(observed, "fb_dmd", 2, times)
        rotations.append({"theta_radians_per_sample": theta,
                          "relative_error": relative_error(output, observed),
                          "minimum_branch_margin": diag["minimum_branch_margin"]})
    ambiguous = np.vstack([np.cos(np.pi / 2 * times), np.sin(np.pi / 2 * times)]) * .9 ** times
    ambiguous_rejected, ambiguous_reason = False, None
    try:
        reconstruct(ambiguous, "fb_dmd", 2, times)
    except ReconstructionFailure as error:
        ambiguous_rejected, ambiguous_reason = True, str(error)
    passed = (default_error < 1e-10 and old_norm_ratio < 1e-10
              and diagnostics["converged"] and old_diagnostics["converged"]
              and negative_error < 1e-10
              and all(row["relative_error"] < 1e-10 for row in rotations)
              and ambiguous_rejected)
    return {"passed": bool(passed), "outlier_pursuit_tall_clean": {
        "shape": [300, 20], "default_lambda": diagnostics["lam"],
        "default_relative_error": default_error,
        "default_gamma_upper_bound": diagnostics["gamma_upper_bound"],
        "old_lambda": old_lambda, "old_lambda_optimum": "zero low-rank component",
        "old_lambda_lowrank_norm_ratio": old_norm_ratio,
        "old_lambda_primal_residual": old_diagnostics["primal_residual"],
        "old_lambda_dual_relative_residual": old_diagnostics["dual_relative_residual"],
        "norm_inequality": "lambda < 1/sqrt(snapshot_count) implies unique optimum L=0 because ||L||_* >= ||L||_F >= ||L||_2,1/sqrt(n)"},
        "fb_dmd": {"negative_eigenvalue": -.8, "negative_eigenvalue_relative_error": negative_error,
                   "negative_eigenvalue_selected": negative_diagnostics["eigenvalues_real"],
                   "rotations": rotations, "ambiguous_theta": float(np.pi / 2),
                   "ambiguous_branch_rejected": ambiguous_rejected,
                   "ambiguous_branch_reason": ambiguous_reason},
        "criteria": "relative reconstruction error and zero-optimum norm ratio < 1e-10; ambiguous root must fail"}


def wall_checks():
    reference = herschel_bulkley_channel(ny=33, nx=8, times=[0, .05, .1])
    base = {"velocity_relative_l2": .15, "vorticity_relative_l2": .25,
            "extra_divergence_normalized": .1, "kinetic_energy_relative_l2": .15}
    wall = {"wall_velocity_error_relative_rms": .02,
            "wall_normal_velocity_relative_rms": .02}
    rows = []
    for component, label in ((0, "tangential_slip"), (1, "equal_wall_penetration")):
        candidate = reference["velocity"].copy()
        candidate[:, component] += .005
        metrics = physical_metrics(reference, candidate)
        base_gate = quality_gates(metrics, base)
        complete_gate = quality_gates(metrics, {**base, **wall})
        rows.append({"artifact": label, "constant_velocity_added": .005,
                     "component": component, "base_gates_pass": base_gate["passed"],
                     "channel_wall_gates_pass": complete_gate["passed"],
                     "metrics": {key: metrics[key] for key in (*base, *wall,
                                 "extra_boundary_flux_rms", "wall_normal_velocity_max_abs")}})
    return {"passed": all(row["base_gates_pass"] and not row["channel_wall_gates_pass"] for row in rows),
            "reference": "stationary Herschel-Bulkley channel; declared no-slip walls",
            "parameters": {"ny": 33, "nx": 8, "times": [0, .05, .1]},
            "base_thresholds": base, "additional_channel_thresholds": wall, "artifacts": rows,
            "interpretation": "Uniform wall-normal penetration can cancel net boundary flux; tangential slip does not change curl or divergence."}


def finite_checks():
    assert_finite({"undefined": None, "array": np.ones(3), "nested": [2., "ok"]})
    probes = {"scalar_nan": {"x": float("nan")},
              "array_inf": {"nested": [np.array([1., np.inf])]},
              "numpy_scalar_nan": np.float64(np.nan)}
    outcomes = []
    for label, value in probes.items():
        rejected, reason = False, None
        try:
            assert_finite(value)
        except FloatingPointError as error:
            rejected, reason = True, str(error)
        outcomes.append({"probe": label, "rejected": rejected, "reason": reason})
    overflow_parameters = {"ny": 5, "nx": 3, "steps": 2, "initial": "steady",
                           "pressure_gradient": 1e152, "relaxation_time": 1e8}
    overflow_rejected, overflow_reason = False, None
    try:
        oldroyd_b_channel(**overflow_parameters)
    except (FloatingPointError, OverflowError) as error:
        overflow_rejected, overflow_reason = True, str(error)
    normal = oldroyd_b_channel(ny=33, nx=3, eta_s=.3, eta_p=.7,
                              relaxation_time=.6, steps=20, initial="steady")
    assert_finite(normal)
    y = np.array(normal["metadata"]["y"])
    face_y = .5 * (y[:-1] + y[1:])
    expected_normal_stress = 2 * .6 * .7 * face_y ** 2
    normal_error = float(np.max(np.abs(normal["stress_xx_faces"][-1] - expected_normal_stress)))
    return {"passed": (all(row["rejected"] for row in outcomes) and overflow_rejected
                       and normal_error < 1e-10),
            "runner_recursive_guard": {"legitimate_none_accepted": True, "probes": outcomes},
            "rheology_overflow": {"parameters": overflow_parameters,
                                   "rejected": overflow_rejected, "reason": overflow_reason},
            "rheology_normal_control": {"initial": "steady", "eta_s": .3, "eta_p": .7,
                                         "relaxation_time": .6,
                                         "normal_stress_max_absolute_error": normal_error,
                                         "minimum_conformation_determinant": normal["metadata"]["min_conformation_determinant_faces"]},
            "criteria": "NaN/Inf and overflowing Oldroyd-B state explicitly rejected; normal finite reference remains accurate"}


def dns_checks():
    n, spacing = 32, 2 * np.pi / 32
    x, y = np.meshgrid(np.arange(n) * spacing, np.arange(n) * spacing)
    omega0 = np.sin(x) * np.sin(y) + .35 * np.cos(2 * x) * np.sin(3 * y) + .1 * np.sin(5 * x + 2 * y)
    inviscid = simulate_periodic_ns(nx=n, dt=.001, n_steps=400, sample_every=20,
                                    viscosity=0., initial_vorticity=omega0)
    _, energy, enstrophy = flow_quantities(inviscid)
    energy_drift = float((energy[-1] - energy[0]) / energy[0])
    enstrophy_drift = float((enstrophy[-1] - enstrophy[0]) / enstrophy[0])
    nu, dt = .02, .0001
    viscous = simulate_periodic_ns(nx=n, dt=dt, n_steps=2, sample_every=1,
                                   viscosity=nu, initial_vorticity=omega0)
    omega, energy, enstrophy = flow_quantities(viscous)
    wx = spatial_derivative(omega, spacing, -1, True)
    wy = spatial_derivative(omega, spacing, -2, True)
    actual_energy_rate = float((energy[-1] - energy[0]) / (2 * dt))
    actual_enstrophy_rate = float((enstrophy[-1] - enstrophy[0]) / (2 * dt))
    expected_energy_rate = float(-nu * np.mean(omega[1] ** 2))
    expected_enstrophy_rate = float(-nu * np.mean(wx[1] ** 2 + wy[1] ** 2))
    a, b = .6, .03
    omega0 = 2 * a * np.sin(x) * np.sin(y) + 13 * b * np.cos(2 * x) * np.sin(3 * y)
    u = a * np.sin(x) * np.cos(y) + 3 * b * np.cos(2 * x) * np.cos(3 * y)
    v = -a * np.cos(x) * np.sin(y) + 2 * b * np.sin(2 * x) * np.sin(3 * y)
    wx = 2 * a * np.cos(x) * np.sin(y) - 26 * b * np.sin(2 * x) * np.sin(3 * y)
    wy = 2 * a * np.sin(x) * np.cos(y) + 39 * b * np.cos(2 * x) * np.cos(3 * y)
    expected_rhs = (-u * wx - v * wy
                    - nu * (4 * a * np.sin(x) * np.sin(y) + 169 * b * np.cos(2 * x) * np.sin(3 * y)))
    rhs_rows = []
    for dt in (.001, .0005, .00025):
        run = simulate_periodic_ns(nx=n, dt=dt, n_steps=2, sample_every=1,
                                   viscosity=nu, initial_vorticity=omega0)
        omega, _, _ = flow_quantities(run)
        measured_rhs = (-3 * omega[0] + 4 * omega[1] - omega[2]) / (2 * dt)
        error = relative_error(measured_rhs, expected_rhs)
        order = float(np.log2(rhs_rows[-1]["relative_error"] / error)) if rhs_rows else None
        rhs_rows.append({"dt": dt, "relative_error": error, "observed_stencil_order": order})
    passed = (abs(energy_drift) < 1e-10 and abs(enstrophy_drift) < 1e-10
              and abs(actual_energy_rate - expected_energy_rate) < 1e-10
              and abs(actual_enstrophy_rate - expected_enstrophy_rate) < 1e-9
              and rhs_rows[-1]["relative_error"] < 1e-7
              and all(1.9 < row["observed_stencil_order"] < 2.1 for row in rhs_rows[1:]))
    return {"passed": bool(passed), "nx": n, "domain": [float(2 * np.pi)] * 2,
            "inviscid_invariants": {"initial_vorticity": "sin(x)sin(y)+.35cos(2x)sin(3y)+.1sin(5x+2y)",
                                     "dt": .001, "duration": .4,
                                     "relative_energy_drift": energy_drift,
                                     "relative_enstrophy_drift": enstrophy_drift},
            "viscous_dissipation_identities": {"viscosity": nu, "dt": .0001,
                "energy_rate_measured": actual_energy_rate,
                "energy_rate_exact_identity": expected_energy_rate,
                "energy_identity": "dE/dt=-nu*mean(omega^2)",
                "enstrophy_rate_measured": actual_enstrophy_rate,
                "enstrophy_rate_exact_identity": expected_enstrophy_rate,
                "enstrophy_identity": "dZ/dt=-nu*mean(omega_x^2+omega_y^2)"},
            "nonlinear_rhs": {"initial_streamfunction": ".6sin(x)sin(y)+.03cos(2x)sin(3y)",
                              "expected_rhs": "-u*omega_x-v*omega_y+nu*laplacian(omega), evaluated by independent trigonometric formulas",
                              "numerical_rhs": "(-3omega(0)+4omega(dt)-omega(2dt))/(2dt)",
                              "viscosity": nu, "rows": rhs_rows,
                              "expected_order": "2 from the derivative stencil; not the RK4 integration order"},
            "criteria": "inviscid drift <1e-10; energy identity error <1e-10; enstrophy identity error <1e-9; final RHS relative error <1e-7 and stencil order 1.9..2.1",
            "dns_certified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/research/solver_validation/adversarial_review.json")
    arguments = parser.parse_args()
    paths = ["python/scripts/verify_adversarial_physics.py", "python/src/fluid_denoise/research/algorithms.py",
             "python/src/fluid_denoise/research/dns.py", "python/src/fluid_denoise/research/metrics.py",
             "python/src/fluid_denoise/research/rheology.py", "python/src/fluid_denoise/research/runner.py",
             "configs/research/pilot.json"]
    checks = {"algorithms": algorithm_checks(), "wall_adversaries": wall_checks(),
              "finite_guards": finite_checks(), "dns_physical_identities": dns_checks()}
    result = {"schema_version": 1, "generated_utc": datetime.now(timezone.utc).isoformat(),
              "all_checks_passed": all(check["passed"] for check in checks.values()),
              "scope": "independent adversarial review; not campaign results or DNS certification",
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "source_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths},
              "numpy_version": np.__version__, "checks": checks}
    assert_finite(result)
    output = Path(arguments.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"all_checks_passed": result["all_checks_passed"], "output": str(output),
                      "check_groups": {name: check["passed"] for name, check in checks.items()}}, ensure_ascii=False))
    return 0 if result["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
