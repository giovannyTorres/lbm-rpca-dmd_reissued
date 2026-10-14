"""Controlled reconstruction experiments, with an isolated process per fit.

This measures in-window reconstruction, not forecasting. Clean fields are only
used to create contamination and evaluate estimates; algorithms receive the
observed matrix, fixed rank/parameters and time coordinates. No clean-reference
hyperparameter optimisation is performed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def assert_finite(value, path="result"):
    """None may encode an undefined metric; NaN/Inf may never encode success."""
    if isinstance(value, dict):
        for key, item in value.items():
            assert_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_finite(item, f"{path}[{index}]")
    elif isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise FloatingPointError(f"Nonfinite array at {path}")
    elif isinstance(value, (float, complex, np.floating, np.complexfloating)) and not np.isfinite(value):
        raise FloatingPointError(f"Nonfinite number at {path}")


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def make_dataset(spec):
    from .dns import simulate_periodic_ns, taylor_green_reference
    from .rheology import herschel_bulkley_channel, oldroyd_b_channel
    from .legacy_simulation import simulate_von_karman_fields
    from .viscoelastic_dns import simulate_viscoelastic_periodic
    from .viscoplastic_dns import simulate_viscoplastic_periodic
    functions = {
        "spectral_ns": simulate_periodic_ns, "taylor_green_exact": taylor_green_reference,
        "herschel_bulkley": herschel_bulkley_channel, "oldroyd_b": oldroyd_b_channel,
        "lbm_cylinder": simulate_von_karman_fields,
        "viscoelastic_periodic": simulate_viscoelastic_periodic,
        "viscoplastic_periodic": simulate_viscoplastic_periodic,
    }
    parameters = dict(spec.get("parameters", {}))
    if "times" in parameters:
        parameters["times"] = np.asarray(parameters["times"], dtype=float)
    return functions[spec["solver"]](**parameters)


def dataset_digest(dataset):
    digest = hashlib.sha256()
    for key in sorted(dataset):
        value = dataset[key]
        digest.update(key.encode())
        if isinstance(value, np.ndarray):
            digest.update(str(value.shape).encode())
            digest.update(str(value.dtype).encode())
            digest.update(np.ascontiguousarray(value).tobytes())
        else:
            digest.update(json.dumps(json_safe(value), sort_keys=True).encode())
    return digest.hexdigest()


def _fit_worker(connection, dataset, noisy, method, rank, parameters, output_path):
    try:
        from .algorithms import reconstruct, ReconstructionFailure
        from .cost import measure_call
        from .metrics import physical_metrics
        from .physics import apply_postprocessing, vorticity_residual_metrics
        shape = noisy.shape
        # Spatial solids are excluded from factorisation as well as evaluation.
        fluid = ~np.asarray(dataset["mask"], dtype=bool)
        matrix = noisy[:, :, fluid].reshape(shape[0], -1).T.copy()
        parameters = dict(parameters)
        postprocess = parameters.pop("postprocess", "none")
        fallback = parameters.pop("fallback_method", None)
        def fit():
            try:
                estimate, diagnostics = reconstruct(matrix, method, rank, times=dataset["times"], **parameters)
            except ReconstructionFailure as primary_error:
                if fallback is None:
                    raise
                estimate, diagnostics = reconstruct(matrix, fallback, rank, times=dataset["times"], **parameters)
                diagnostics["primary_failure"] = primary_error.diagnostics
                diagnostics["fallback_method"] = fallback
                diagnostics["fallback_used"] = True
                for key in ("svd_calls", "lstsq_calls", "flops_proxy"):
                    diagnostics[key] = diagnostics.get(key, 0)+primary_error.diagnostics.get(key, 0)
                diagnostics["status"] = "ok_with_declared_fallback" if diagnostics["converged"] else "not_converged"
            velocity = np.zeros_like(noisy)
            velocity[:, :, fluid] = estimate.T.reshape(shape[0], 2, int(fluid.sum()))
            velocity, postprocess_diagnostics = apply_postprocessing(dataset, velocity, postprocess)
            diagnostics["postprocessing"] = postprocess_diagnostics
            diagnostics["fft_calls"] = postprocess_diagnostics.get("fft_calls", 0)
            return velocity, diagnostics
        (velocity, diagnostics), profile = measure_call(fit)
        metrics = physical_metrics(dataset, velocity)
        metrics.update(vorticity_residual_metrics(dataset, velocity))
        if output_path:
            np.savez_compressed(output_path, velocity=velocity, times=dataset["times"])
        result = {"diagnostics": diagnostics, "profile": profile, "metrics": metrics}
        assert_finite(result)
        connection.send(json_safe(result))
    except Exception as exc:
        connection.send({"failure": str(exc), "diagnostics": getattr(exc, "diagnostics", {}),
                         "traceback": traceback.format_exc()})
    finally:
        connection.close()


def isolated_fit(dataset, noisy, method, rank, parameters, timeout, output_path=None):
    # fork gives every fit a fresh ru_maxrss and identical loaded-library state.
    # Linux is the reference environment. spawn works on platforms without fork.
    context = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_fit_worker, args=(child, dataset, noisy, method, rank, parameters, output_path))
    process.start()
    child.close()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if parent.poll(min(.2, max(0, deadline - time.monotonic()))):
            try:
                result = parent.recv()
            except EOFError:
                result = {"failure": f"worker exited with code {process.exitcode}"}
            process.join(5)
            if process.is_alive():
                process.terminate()
                process.join()
            parent.close()
            return result
        if not process.is_alive():
            process.join()
            parent.close()
            return {"failure": f"worker exited with code {process.exitcode}"}
    process.terminate()
    process.join()
    parent.close()
    return {"failure": f"timeout after {timeout} seconds"}


def _scalar_metrics(metrics):
    return {key: value for key, value in metrics.items()
            if value is None or isinstance(value, (str, bool, int, float))}


def run_campaign(config, root, case_filter=None):
    from .algorithms import METHODS
    from .cost import environment_metadata, summarize_profiles, quality_gates, pareto_front
    from .metrics import physical_metrics
    from .noise import add_noise
    repetitions = int(config["repetitions"])
    if repetitions < 1 or len(config["seeds"]) < 1:
        raise ValueError("Require repetitions >= 1 and at least one seed")
    destination = root / "results" / "research" / config["campaign_id"]
    destination.mkdir(parents=True, exist_ok=True)
    arrays = destination / "datasets"
    arrays.mkdir(exist_ok=True)
    reconstructions = destination / "reconstructions"
    reconstructions.mkdir(exist_ok=True)
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    provenance = {"config": config, "config_sha256": config_hash,
                  "environment": environment_metadata(), "evaluation": "in_sample_reconstruction",
                  "timing_scope": "reconstruct, vector reshape and declared physics postprocessing; excludes generation, I/O, physics evaluation and worker launch",
                  "rss_scope": "absolute per-fit child process peak; includes runtime and inherited dataset",
                  "physical_density_scope": "Velocity-only reconstructions; supplied clean density is retained, not estimated",
                  "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()}
    provenance["source_files_sha256"] = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((root / "python" / "src" / "fluid_denoise" / "research").glob("*.py"))
    }
    import scipy
    provenance["scipy_version"] = scipy.__version__
    write_json(destination / "provenance.json", provenance)
    rows, datasets = [], []
    ledger = destination / "ledger.jsonl"
    ledger.write_text("")
    for spec in config["cases"]:
        if case_filter and spec["id"] not in case_filter:
            continue
        print(f"Generating {spec['id']}", flush=True)
        dataset = make_dataset(spec)
        assert_finite(dataset, "dataset")
        clean = np.asarray(dataset["velocity"], dtype=np.float64)
        if not np.isfinite(clean).all() or clean.shape[0] < 2:
            raise ValueError("Invalid clean dataset")
        clean_path = arrays / f"{spec['id']}_clean.npz"
        np.savez_compressed(clean_path, **{k: v for k, v in dataset.items() if isinstance(v, np.ndarray)})
        manifest = {"case": spec, "shape": list(clean.shape), "dataset_sha256": dataset_digest(dataset),
                    "metadata": dataset["metadata"], "dx": dataset["dx"], "dy": dataset["dy"],
                    "boundary": dataset["boundary"], "reference_metrics": physical_metrics(dataset, clean)}
        datasets.append(manifest)
        write_json(arrays / f"{spec['id']}_metadata.json", manifest)
        for noise in config["noise_models"]:
            for seed in config["seeds"]:
                noisy, noise_metadata = add_noise(clean, seed=seed, obstacle_mask=dataset["mask"], **noise)
                for method_spec in config["methods"]:
                    method = method_spec["method"]
                    if method not in METHODS:
                        raise ValueError(f"Unknown method {method}")
                    rank = int(method_spec.get("rank", config.get("rank", 4)))
                    label = method_spec.get("label", method)
                    parameters = {"noise_intensity": noise["intensity"], **config.get("algorithm_parameters", {}),
                                  **method_spec.get("parameters", {}), "postprocess": method_spec.get("postprocess", "none")}
                    identifier = f"{spec['id']}__{noise['kind']}__s{seed}__{label}__r{rank}"
                    fits = []
                    for repetition in range(repetitions):
                        save = str(reconstructions / (identifier + ".npz")) if repetition == 0 and config.get("save_reconstructions", False) else None
                        fit = isolated_fit(dataset, noisy, method, rank, parameters, float(config.get("fit_timeout_seconds", 120)), save)
                        fit.update(repetition=repetition)
                        fits.append(fit)
                    successful = [f for f in fits if "failure" not in f]
                    record = {"id": identifier, "case": spec["id"], "noise": noise_metadata,
                              "method": label, "base_method": method, "postprocess": parameters["postprocess"],
                              "rank_requested": rank, "seed": seed, "fits": fits}
                    row = {"id": identifier, "case": spec["id"], "noise": noise["kind"], "seed": seed,
                           "method": label, "base_method": method, "postprocess": parameters["postprocess"],
                           "rank_requested": rank, "noise_rms_relative": noise_metadata["realized_relative_rms"],
                           "successful_repetitions": len(successful), "failed_repetitions": repetitions - len(successful)}
                    if successful:
                        profile = summarize_profiles([f["profile"] for f in successful])
                        metrics = successful[0]["metrics"]
                        diagnostics = successful[0]["diagnostics"]
                        row.update(profile)
                        row.update(_scalar_metrics(metrics))
                        row.update({k: diagnostics.get(k) for k in (
                            "status", "converged", "effective_rank", "iterations", "svd_calls", "lstsq_calls", "fft_calls", "flops_proxy")})
                        row["all_repetitions_converged"] = all(f["diagnostics"].get("converged", False) for f in successful) and len(successful) == repetitions
                        thresholds = dict(config["quality_thresholds"])
                        if dataset["boundary"] == "channel":
                            thresholds.update(config.get("channel_quality_thresholds", {}))
                        gate = quality_gates(metrics, thresholds)
                        row["quality_pass"] = gate["passed"] and row["all_repetitions_converged"]
                        record["quality_gate"] = gate
                    else:
                        row.update(status="failed", converged=False, quality_pass=False)
                    with ledger.open("a") as handle:
                        # Keep the independently evaluated scalars for every
                        # repetition, but only one identical time-series copy.
                        first_metrics = next((i for i,f in enumerate(record["fits"]) if "metrics" in f), None)
                        for fit_index, fit in enumerate(record["fits"]):
                            if fit_index != first_metrics and "metrics" in fit:
                                fit["metrics"].pop("series", None)
                                fit["metric_series_reference"] = f"fits[{first_metrics}].metrics.series"
                        handle.write(json.dumps(json_safe(record), ensure_ascii=False, allow_nan=False) + "\n")
                    rows.append(row)
                    if len(rows) % 12 == 0:
                        print(f"Completed {len(rows)} experiment conditions", flush=True)
    write_json(destination / "dataset_manifests.json", datasets)
    # Different flows/noise models are different optimisation problems. Compare
    # Pareto membership only inside matched case/noise/seed groups.
    for row in rows:
        row["pareto_member"] = False
        row["feasible_pareto_member"] = False
    groups = {(r["case"], r["noise"], r["seed"]) for r in rows}
    for group in groups:
        eligible = [r for r in rows if (r["case"], r["noise"], r["seed"]) == group and r.get("all_repetitions_converged")]
        for index in pareto_front(eligible):
            eligible[index]["pareto_member"] = True
        feasible = [r for r in eligible if r.get("quality_pass")]
        for index in pareto_front(feasible):
            feasible[index]["feasible_pareto_member"] = True
    keys = sorted({k for row in rows for k in row})
    with (destination / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(json_safe(rows))
    write_json(destination / "summary.json", rows)
    expected = sum(not case_filter or s["id"] in case_filter for s in config["cases"]) * len(config["noise_models"]) * len(config["seeds"]) * len(config["methods"])
    write_json(destination / "completion.json", {"conditions": len(rows), "expected_conditions": expected,
               "complete": len(rows) == expected, "fits": len(rows) * repetitions,
               "failed_fits": sum(r["failed_repetitions"] for r in rows),
               "not_converged_conditions": sum(not r.get("all_repetitions_converged", False) for r in rows),
               "quality_pass_conditions": sum(r.get("quality_pass", False) for r in rows),
               "config_sha256": config_hash})
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/research/pilot.json")
    parser.add_argument("--cases", nargs="+")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    config = json.loads((root / args.config).read_text())
    print(run_campaign(config, root, args.cases), flush=True)
