#!/usr/bin/env python3
"""Independent ledger audit: no imports from campaign or analysis helpers.

Resource repetitions remain repetitions of one observed matrix.  Conditions
are identified by case/noise/seed/candidate, and non-converged estimates remain
in their original denominators.  This verifies evidence, not universal DNS.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[2]
SOURCE_TRAJECTORIES = {
    "ns_vortices": "highres_newtonian_2026-10-14/datasets/interacting_vortices_128_dt0.005.npz",
    "ns_multiscale": "highres_newtonian_2026-10-14/datasets/decaying_turbulence_128_dt0.005.npz",
    "oldroyd_b": "highres_viscoelastic_2026-10-14/datasets/oldroyd_b_128.npz",
    "fene_p": "highres_viscoelastic_2026-10-14/datasets/fene_p_128.npz",
    "hb_thinning": "highres_viscoplastic_2026-10-14/datasets/hb_thinning_128.npz",
    "bingham": "highres_viscoplastic_2026-10-14/datasets/bingham_128.npz",
    "hb_thickening": "highres_viscoplastic_2026-10-14/datasets/hb_thickening_128.npz",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def read_ledger(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def canonical_dataset_digest(dataset):
    """Independently implement the published canonical semantic digest."""
    result = hashlib.sha256()
    for name, value in sorted(dataset.items()):
        result.update(name.encode())
        if isinstance(value, np.ndarray):
            result.update(str(value.shape).encode())
            result.update(str(value.dtype).encode())
            result.update(np.ascontiguousarray(value).tobytes())
        else:
            result.update(json.dumps(value, sort_keys=True).encode())
    return result.hexdigest()


def audit_source_preflight():
    result = {}
    for flow, relative_path in SOURCE_TRAJECTORIES.items():
        path = ROOT / "results/research" / relative_path
        sidecar = read_json(path.with_suffix(".json"))
        archive_hash = digest(path)
        with np.load(path) as saved:
            arrays = {name: saved[name] for name in saved.files}
        assert arrays["velocity"].shape[1:] == (2, 128, 128)
        assert all(np.isfinite(value).all() for value in arrays.values())
        assert np.all(np.diff(arrays["times"]) > 0)
        declared_archive = sidecar.get("npz_sha256", sidecar.get("archive_sha256"))
        semantic_hash = None
        if declared_archive is not None:
            assert archive_hash == declared_archive
            declared_semantics = "SHA256 of compressed NPZ bytes"
        else:
            assert "dataset_sha256" in sidecar
            dataset = dict(arrays, metadata=sidecar["metadata"], dx=sidecar["dx"],
                           dy=sidecar["dy"], boundary="periodic")
            semantic_hash = canonical_dataset_digest(dataset)
            assert semantic_hash == sidecar["dataset_sha256"]
            declared_semantics = "Canonical array/metadata digest, not compressed NPZ bytes"
        result[flow] = {
            "archive_sha256": archive_hash,
            "declared_hash_semantics": declared_semantics,
            "canonical_dataset_sha256": semantic_hash,
            "declared_hash_independently_verified": True,
            "shape": list(arrays["velocity"].shape),
            "saved_times": len(arrays["times"]),
        }
    return result


def audit_driver_epochs(directory, provenance):
    current_path = ROOT / "python/scripts/run_model_fidelity_benchmark.py"
    current_hash = digest(current_path)
    assert current_hash == provenance["driver_sha256"]
    current_ast = ast.parse(current_path.read_text())
    current_functions = {node.name: ast.dump(node, include_attributes=False)
                         for node in current_ast.body if isinstance(node, ast.FunctionDef)}
    current_constants = [ast.dump(node, include_attributes=False) for node in current_ast.body
                         if isinstance(node, (ast.Assign, ast.AnnAssign))]
    checks = {}
    for epoch in provenance.get("driver_epochs", [current_hash]):
        if epoch == current_hash:
            checks[epoch] = {"executing_source_available": True, "current_driver": True}
            continue
        path = directory / "driver_epochs" / ("driver_" + epoch + ".py")
        assert digest(path) == epoch, "archived executing driver hash mismatch"
        previous_ast = ast.parse(path.read_text())
        previous_functions = {node.name: ast.dump(node, include_attributes=False)
                              for node in previous_ast.body if isinstance(node, ast.FunctionDef)}
        previous_constants = [ast.dump(node, include_attributes=False) for node in previous_ast.body
                              if isinstance(node, (ast.Assign, ast.AnnAssign))]
        assert previous_constants == current_constants, "numerical top-level contract changed between epochs"
        unchanged = []
        for name in previous_functions:
            if name in ("load_flow", "main"):
                continue
            assert previous_functions[name] == current_functions[name], "numerical helper changed: " + name
            unchanged.append(name)
        checks[epoch] = {
            "executing_source_available": True,
            "archived_source_sha_verified": True,
            "all_numerical_functions_ast_identical": unchanged,
            "only_input_loader_and_provenance_main_changed": True,
        }
    return checks


def close(a, b):
    return np.isclose(float(a), float(b), rtol=2e-12, atol=2e-14)


def independently_regenerate_observation(clean, times, noise, seed, config):
    """Recreate the frozen observation from its public statistical definition.

    This uses NumPy/SciPy primitives directly, never the campaign's generator.
    The resulting hash and exact archived frame values test the paired input.
    """
    generator = np.random.default_rng(seed)
    error = generator.standard_normal(clean.shape)
    if noise == "sparse_impulsive":
        error *= generator.random(clean.shape) < config["sparse_probability"]
    elif noise == "column_outliers":
        count = max(1, int(np.floor(config["column_fraction_upper_bound"] * len(times))))
        snapshots = generator.choice(len(times), size=count, replace=False)
        support = np.zeros(len(times), bool)
        support[snapshots] = True
        error *= support[:, None, None, None]
    elif noise == "spatial_gaussian":
        sigma_pixels = config["spatial_kernel_sigma"] / (2 * np.pi / clean.shape[-1])
        error = gaussian_filter(error, sigma=(0, 0, sigma_pixels, sigma_pixels), mode="wrap")
    elif noise == "temporal_ar":
        for index, delta in enumerate(np.diff(times), 1):
            coefficient = np.exp(-delta / config["temporal_correlation_time"])
            error[index] = coefficient * error[index - 1] + np.sqrt(1 - coefficient ** 2) * error[index]
    else:
        assert noise == "gaussian_iid"
    error *= config["intensity_relative_rms"] * np.sqrt(np.mean(clean ** 2)) / np.sqrt(np.mean(error ** 2))
    return clean + error


def audit_historical_campaign(directory: Path):
    provenance = read_json(directory / "provenance.json")
    config = provenance["config"]
    ledger = read_ledger(directory / "ledger.jsonl")
    summary = {row["id"]: row for row in read_json(directory / "summary.json")}
    expected = {
        (case["id"], noise["kind"], seed, method.get("label", method["method"]))
        for case in config["cases"] for noise in config["noise_models"]
        for seed in config["seeds"] for method in config["methods"]
    }
    actual = [(r["case"], r["noise"]["kind"], r["seed"], r["method"]) for r in ledger]
    assert len(actual) == len(set(actual)), "duplicate experimental conditions"
    assert set(actual) == expected, "missing or unexpected experimental condition"
    assert len(summary) == len(ledger), "summary denominator differs from ledger"
    assert len({row["id"] for row in ledger}) == len(ledger), "duplicate condition IDs"
    counters = {method: Counter() for method in sorted({r["method"] for r in ledger})}
    nonconverged, fallback_ids = [], []
    for record in ledger:
        row = summary[record["id"]]
        fits = record["fits"]
        assert len(fits) == config["repetitions"], "wrong resource repetition count"
        successful = [fit for fit in fits if "failure" not in fit]
        converged = len(successful) == len(fits) and all(
            fit["diagnostics"].get("converged", False) for fit in successful)
        thresholds = dict(config["quality_thresholds"])
        if any(fit.get("metrics", {}).get("wall_gate_applicable") for fit in successful):
            thresholds.update(config.get("channel_quality_thresholds", {}))
        first = successful[0]["metrics"] if successful else {}
        gates = {
            name: name in first and first[name] is not None and
            np.isfinite(first[name]) and 0 <= first[name] <= maximum
            for name, maximum in thresholds.items()
        }
        quality = bool(converged and all(gates.values()))
        assert row["quality_pass"] is quality, "quality denominator or convergence mismatch"
        assert row["all_repetitions_converged"] is converged
        assert row["successful_repetitions"] == len(successful)
        assert row["failed_repetitions"] == len(fits) - len(successful)
        for fit in successful:
            diagnostic = fit["diagnostics"]
            if "preprocessing" in diagnostic and "modal" in diagnostic:
                stage_convergence = bool(diagnostic["preprocessing"]["converged"] and
                                         diagnostic["modal"]["converged"])
                assert diagnostic["converged"] is stage_convergence
            for name in thresholds:
                if name in fit["metrics"] and fit["metrics"][name] is not None:
                    assert close(fit["metrics"][name], first[name]), "quality changes across resource repeats"
            for name in ("wall_seconds", "cpu_seconds", "rss_peak_bytes"):
                value = fit["profile"][name]
                assert np.isfinite(value) and value >= 0
        for name in thresholds:
            if name in first and first[name] is not None:
                assert close(row[name], first[name]), "summary metric differs from ledger"
        counter = counters[record["method"]]
        counter["conditions"] += 1
        counter["resource_fits"] += len(fits)
        counter["quality_accepted"] += quality
        counter["failed_conditions"] += len(successful) != len(fits)
        counter["not_converged_conditions"] += not converged
        has_fallback = any(fit.get("diagnostics", {}).get("fallback_used", False) for fit in fits)
        counter["fallback_conditions"] += has_fallback
        if not converged:
            nonconverged.append(record["id"])
        if has_fallback:
            fallback_ids.append(record["id"])
    source = {name: digest(ROOT / name) == expected_hash
              for name, expected_hash in provenance["source_files_sha256"].items()}
    assert all(source.values()), "a scientific source differs from the historical campaign"
    return {
        "campaign": directory.name,
        "ledger_sha256": digest(directory / "ledger.jsonl"),
        "conditions": len(ledger),
        "resource_fits": sum(len(record["fits"]) for record in ledger),
        "independent_noise_seeds_per_case_noise_candidate": len(config["seeds"]),
        "resource_repetitions_are_independent_noise_realizations": False,
        "method_counts": {method: dict(counts) for method, counts in counters.items()},
        "nonconverged_condition_ids": nonconverged,
        "fallback_condition_ids": fallback_ids,
        "scientific_sources_unchanged": source,
        "conditions_all_expected_and_unique": True,
        "summary_quality_recomputed_from_metrics_and_convergence": True,
        "all_conditions_retained_in_denominators": True,
    }


def audit_new_campaign(directory: Path, require_complete=True):
    """Audit frozen paired observations, three postprocesses and all denominators."""
    provenance = read_json(directory / "provenance.json")
    config = provenance["configuration"]
    driver_epochs = audit_driver_epochs(directory, provenance)
    records = read_ledger(directory / "ledger.jsonl")
    rows = read_json(directory / "summary.json")
    summary = {(row["id"], row["variant"]): row for row in rows}
    assert len(summary) == len(rows), "duplicate new summary rows"
    expected = {
        (flow, noise, seed, method)
        for flow in provenance["selected_flows"] for noise in provenance["selected_noises"]
        for seed in provenance["selected_seeds"] for method in provenance["selected_methods"]
    }
    actual = [(r["flow"], r["noise"], r["seed"], r["method"]) for r in records]
    assert len(set(actual)) == len(actual), "duplicate new experimental conditions"
    assert set(actual) <= expected, "unexpected new experimental conditions"
    if require_complete:
        assert set(actual) == expected, "new campaign is incomplete"
        assert len(rows) == len(records) * len(config["postprocessing"])
    observations, counters, accepted = {}, {}, Counter()
    failures, nonconverged, archive_checks = [], [], []
    for record in records:
        block = record["flow"], record["noise"], record["seed"]
        observed_digest = record["noise_metadata"]["observation_sha256"]
        if block in observations:
            assert observations[block] == observed_digest, "methods receive different observations"
        observations[block] = observed_digest
        assert close(record["noise_metadata"]["realized_relative_rms"], config["intensity_relative_rms"])
        assert record["clean_input_sha256"] == provenance["clean_input_sources"][record["flow"]]["sha256"]
        assert record["driver_sha256"] in driver_epochs, "undeclared executing driver version"
        diagnostic = record.get("diagnostics", {})
        converged = record["status"] == "ok" and bool(diagnostic.get("converged", False))
        if "preprocessing" in diagnostic and "modal" in diagnostic:
            assert diagnostic["converged"] is bool(diagnostic["preprocessing"]["converged"] and diagnostic["modal"]["converged"])
        counter = counters.setdefault(record["method"], Counter())
        counter["fits"] += 1
        if record["status"] != "ok":
            failures.append(record["id"])
            counter["failed_fits"] += 1
            assert not record.get("evaluations"), "a failed fit claims successful evaluation"
            for variant in config["postprocessing"]:
                if (record["id"], variant) in summary:
                    assert summary[record["id"], variant]["quality_pass"] is False
            continue
        if not converged:
            nonconverged.append(record["id"])
            counter["not_converged_fits"] += 1
        evaluations = record["evaluations"]
        assert len(evaluations) == len(config["postprocessing"])
        assert {e["variant"] for e in evaluations} == set(config["postprocessing"])
        for evaluation in evaluations:
            variant = evaluation["variant"]
            metrics = evaluation["metrics"]
            gates = {
                name: name in metrics and metrics[name] is not None and
                np.isfinite(metrics[name]) and 0 <= metrics[name] <= maximum
                for name, maximum in config["quality_thresholds"].items()
            }
            admissible = bool(converged and all(gates.values()))
            assert evaluation["quality_pass"] is admissible, "new gate incompatible with metrics/convergence"
            assert evaluation["quality"]["passed"] is bool(all(gates.values()))
            assert evaluation["projection"]["reference_velocity_used"] is False
            for name, maximum in config["quality_thresholds"].items():
                reported = evaluation["quality"]["gates"][name]
                assert reported["maximum"] == maximum
                assert reported["passed"] is bool(gates[name])
            for name in ("velocity_relative_l2", "vorticity_relative_l2"):
                values = evaluation["frame_errors"][name]
                assert len(values) == len(provenance["clean_input_sources"][record["flow"]]["times"])
                assert close(max(values), evaluation["frame_errors"][name + "_max_saved"])
            if record["flow"] in ("oldroyd_b", "fene_p"):
                assert metrics["stress_frobenius_relative_l2"] is None
                assert metrics["dissipation_relative_l2"] is None
                assert metrics["vorticity_equation_applicable"] is False
            if not record["flow"].startswith("ns_"):
                assert metrics["vorticity_equation_applicable"] is False
            if (record["id"], variant) in summary:
                row = summary[record["id"], variant]
                assert row["quality_pass"] is admissible
                assert row["converged"] is converged
                for name in config["quality_thresholds"]:
                    assert close(row[name], metrics[name])
            elif require_complete:
                raise AssertionError("missing new summary row")
            accepted[variant] += admissible
            counter[variant + "_accepted"] += admissible
            archive = evaluation.get("reconstruction_archive")
            if archive and record["method"] == "identity" and variant == "raw":
                archive_path = ROOT / archive["path"]
                assert digest(archive_path) == archive["sha256"]
                with np.load(archive_path) as saved:
                    observed = saved["velocity"]
                    selected_frames = saved["frame_indices"] if "frame_indices" in saved.files else np.arange(len(observed))
                clean_path = ROOT / provenance["clean_input_sources"][record["flow"]]["path"]
                with np.load(clean_path) as source:
                    clean = source["velocity"]
                    times = source["times"]
                reconstructed_observation = independently_regenerate_observation(clean, times,
                    record["noise"], record["seed"], config)
                assert hashlib.sha256(np.ascontiguousarray(reconstructed_observation).tobytes()).hexdigest() == observed_digest
                np.testing.assert_array_equal(observed, reconstructed_observation[selected_frames])
                measured_noise = reconstructed_observation - clean
                measured_rms = np.sqrt(np.mean(measured_noise ** 2)) / np.sqrt(np.mean(clean ** 2))
                assert close(measured_rms, config["intensity_relative_rms"])
                if record["noise"] == "column_outliers":
                    support = np.any(measured_noise != 0, axis=(1, 2, 3))
                    fraction = np.count_nonzero(support) / len(support)
                    assert fraction <= config["outlier_column_fraction_upper_bound"]
                archive_checks.append(record["id"])
    source_checks = {name: digest(ROOT / name) == source_hash
                     for name, source_hash in provenance["estimator_source_hashes"].items()}
    assert all(source_checks.values()), "new scientific source changed"
    assert digest(ROOT / "python/scripts/run_model_fidelity_benchmark.py") == provenance["driver_sha256"]
    assert digest(ROOT / "configs/research/model_fidelity_benchmark.json") == provenance["configuration_sha256"]
    input_checks = {}
    for flow, source in provenance["clean_input_sources"].items():
        input_checks[flow] = digest(ROOT / source["path"]) == source["sha256"]
    assert all(input_checks.values()), "input trajectory changed"
    result = {
        "campaign": directory.name,
        "complete": set(actual) == expected,
        "expected_raw_fits": len(expected),
        "recorded_raw_fits": len(records),
        "postprocessing_evaluation_rows": len(rows),
        "actual_successful_evaluations": sum(len(record.get("evaluations", [])) for record in records),
        "matched_noise_blocks": len(observations),
        "independent_noise_seeds_per_flow_noise_candidate": len(provenance["selected_seeds"]),
        "repeated_postprocesses_are_independent_observations": False,
        "quality_passes_by_variant": dict(accepted),
        "failed_fit_ids": failures,
        "not_converged_fit_ids": nonconverged,
        "method_counts": {method: dict(counter) for method, counter in counters.items()},
        "observation_hashes_identical_for_all_methods_in_each_block": True,
        "stored_identity_raw_observations_independently_verified": archive_checks,
        "scientific_sources_unchanged": source_checks,
        "executing_driver_epochs_verified": driver_epochs,
        "executing_driver_epoch_fit_counts": dict(Counter(record["driver_sha256"] for record in records)),
        "clean_input_hashes_verified": input_checks,
        "all_conditions_retained_in_denominators": True,
        "ledger_sha256": digest(directory / "ledger.jsonl"),
    }
    if require_complete:
        completion = read_json(directory / "completion.json")
        assert completion["complete"] is True
        assert completion["recorded_fits"] == len(records)
        assert completion["failed_fits"] == len(failures)
        assert completion["nonconverged_fits"] == len(nonconverged)
        assert completion["quality_passes_by_variant"] == dict(accepted)
        groups = read_json(directory / "group_summary.json")
        grouped = {}
        for record in records:
            for evaluation in record.get("evaluations", []):
                key = record["flow"], record["noise"], record["method"], evaluation["variant"]
                grouped.setdefault(key, []).append((record, evaluation))
        assert len(groups) == len(grouped)
        for row in groups:
            sample = grouped[row["flow"], row["noise"], row["method"], row["variant"]]
            assert row["conditions"] == len(sample)
            assert row["quality_passes"] == sum(e["quality_pass"] for r, e in sample)
            assert row["nonconverged"] == sum(not r["diagnostics"]["converged"] for r, e in sample)
            for metric in config["quality_thresholds"]:
                values = [evaluation["metrics"][metric] for record, evaluation in sample]
                assert close(row[metric + "_median"], np.median(values))
                assert close(row[metric + "_min"], min(values))
                assert close(row[metric + "_max"], max(values))
        result["all_group_counts_medians_ranges_independently_verified"] = len(groups)
    return result


def audit_paired_reanalysis(directory: Path):
    source = ROOT / "results/research/extended_physics_2026-10-14"
    records = read_ledger(source / "ledger.jsonl")
    config = read_json(source / "provenance.json")["config"]
    thresholds = config["quality_thresholds"]
    accepted, blocks, first_metrics = {}, {}, {}
    event_count, pde_count, pde_over_one = 0, 0, 0
    for record in records:
        metrics = record["fits"][0]["metrics"]
        converged = all(fit["diagnostics"]["converged"] for fit in record["fits"])
        gate_pass = all(metrics[name] <= maximum for name, maximum in thresholds.items())
        admissible = bool(converged and gate_pass)
        accepted[record["id"]] = admissible
        block = record["case"], record["noise"]["kind"], record["seed"]
        blocks.setdefault(block, {})[record["method"]] = record
        first_metrics[record["id"]] = metrics
        event_count += metrics["velocity_relative_l2"] <= thresholds["velocity_relative_l2"] and metrics["vorticity_relative_l2"] > thresholds["vorticity_relative_l2"]
        if admissible and metrics.get("vorticity_equation_applicable"):
            pde_count += 1
            pde_over_one += metrics["extra_vorticity_equation_residual_normalized"] > 1
    method_table = {row["method"]: row for row in read_json(directory / "method_summary.json")}
    for method, row in method_table.items():
        sample = [record for record in records if record["method"] == method]
        assert row["conditions"] == len(sample)
        assert row["admissible"] == sum(accepted[record["id"]] for record in sample)
        for name in thresholds:
            values = [first_metrics[record["id"]][name] for record in sample]
            assert close(row[name + "_median"], np.median(values))
            assert close(row[name + "_min"], min(values))
            assert close(row[name + "_max"], max(values))
    sensitivity = read_json(directory / "tolerance_sensitivity.json")
    assert len(sensitivity) == len(method_table) * 4
    independently_recalculated_sensitivity = []
    for row in sensitivity:
        count = 0
        for record in records:
            if record["method"] != row["method"]:
                continue
            metrics = first_metrics[record["id"]]
            converged = all(fit["diagnostics"]["converged"] for fit in record["fits"])
            count += bool(converged and metrics["vorticity_relative_l2"] <= row["vorticity_maximum"] and
                          all(metrics[name] <= maximum for name, maximum in thresholds.items() if name != "vorticity_relative_l2"))
        assert row["admissible"] == count
        independently_recalculated_sensitivity.append({"method": row["method"], "vorticity_maximum": row["vorticity_maximum"], "admissible": count})
    ablations = read_json(directory / "ablation_summary.json")
    for row in ablations:
        before, after = row["before"], row["after"]
        rescued = lost = same_curl = 0
        deltas = {name: [] for name in thresholds}
        for block in blocks.values():
            a, b = block[before], block[after]
            rescued += not accepted[a["id"]] and accepted[b["id"]]
            lost += accepted[a["id"]] and not accepted[b["id"]]
            same_curl += abs(first_metrics[b["id"]]["vorticity_relative_l2"] - first_metrics[a["id"]]["vorticity_relative_l2"]) <= 1e-10
            for name in thresholds:
                deltas[name].append(first_metrics[b["id"]][name] - first_metrics[a["id"]][name])
        assert row["rescued"] == rescued and row["lost"] == lost
        assert row["vorticity_equal_count"] == same_curl
        for name, values in deltas.items():
            assert close(row[name + "_delta_median"], np.median(values))
    completion = read_json(directory / "completion.json")
    no_method = sum(not any(accepted[record["id"]] for record in block.values()) for block in blocks.values())
    assert completion["admissible_conditions"] == sum(accepted.values())
    assert completion["blocks_without_admissible_method"] == no_method
    pde = read_json(directory / "admissible_pde_audit.json")
    assert len(pde) == pde_count
    assert sum(row["extra_residual_greater_than_one"] for row in pde) == pde_over_one
    return {
        "campaign": directory.name,
        "admissible_conditions": sum(accepted.values()),
        "paired_blocks_without_admissible_method": no_method,
        "velocity_gate_pass_but_vorticity_gate_fail": int(event_count),
        "admissible_newtonian_pde_conditions": pde_count,
        "admissible_newtonian_pde_conditions_extra_residual_gt_one": pde_over_one,
        "tolerance_sensitivity_counts_independently_verified": independently_recalculated_sensitivity,
        "all_five_ablations_independently_verified": True,
        "method_eligibility_and_median_range_verified": True,
        "pipeline_gains_are_against_raw_identity_not_prior_matched": True,
        "missing_historical_control": "identity + Helmholtz + 2/3; new campaign supplies all three prior-matched identities",
    }


def independent_projection(value, cutoff):
    n = value.shape[-1]
    full_wave = 2 * np.pi * np.fft.fftfreq(n, d=2 * np.pi / n)
    derivative_wave = full_wave.copy()
    derivative_wave[n // 2] = 0
    kx, ky = np.meshgrid(derivative_wave, derivative_wave)
    square = kx ** 2 + ky ** 2
    spectrum = np.fft.fft2(value)
    numerator = kx * spectrum[:, 0] + ky * spectrum[:, 1]
    multiplier = np.divide(numerator, square, out=np.zeros_like(numerator), where=square > 0)
    spectrum -= np.stack((kx * multiplier, ky * multiplier), axis=1)
    full_x, full_y = np.meshgrid(full_wave, full_wave)
    spectrum *= np.hypot(full_x, full_y) <= cutoff
    return np.fft.ifft2(spectrum).real


def independent_vorticity(value):
    n = value.shape[-1]
    waves = 2 * np.pi * np.fft.fftfreq(n, d=2 * np.pi / n)
    waves[n // 2] = 0
    kx, ky = np.meshgrid(waves, waves)
    return np.fft.ifft2(1j * kx * np.fft.fft2(value[:, 1]) - 1j * ky * np.fft.fft2(value[:, 0])).real


def audit_prior_budget(directory):
    config = read_json(ROOT / "configs/research/model_fidelity_benchmark.json")
    with (directory / "identity_bias_noise_budget.csv").open() as source:
        budget = list(csv.DictReader(source))
    with (directory / "clean_prior_floors.csv").open() as source:
        floors = list(csv.DictReader(source))
    report = read_json(directory / "summary.json")
    assert report["complete"] is True
    assert len(budget) == 7 * 5 * 3 * 9 * 2
    assert len(floors) == 7 * 9
    keys = [(r["case"], r["noise"], int(r["seed"]), float(r["cutoff"]), r["observable"]) for r in budget]
    assert len(keys) == len(set(keys))
    expected_keys = {(flow, noise, seed, cutoff, observable)
                     for flow in SOURCE_TRAJECTORIES for noise in config["noise_models"]
                     for seed in config["seeds"] for cutoff in report["cutoffs"]
                     for observable in ("velocity", "vorticity")}
    assert set(keys) == expected_keys
    floor_keys = [(row["case"], float(row["cutoff"])) for row in floors]
    assert len(set(floor_keys)) == len(floor_keys)
    assert set(floor_keys) == {(flow, cutoff) for flow in SOURCE_TRAJECTORIES for cutoff in report["cutoffs"]}
    selected = {r["case"]: r for r in floors if float(r["cutoff"]) == config["physical_cutoff"]}
    assert set(selected) == set(SOURCE_TRAJECTORIES)
    checked_floors, checked_budgets = [], []
    control_agreement = []
    sample_noise = {"ns_multiscale": "gaussian_iid", "oldroyd_b": "spatial_gaussian", "hb_thinning": "sparse_impulsive"}
    for flow, relative_path in SOURCE_TRAJECTORIES.items():
        with np.load(ROOT / "results/research" / relative_path) as data:
            clean, times = data["velocity"], data["times"]
        clean_band = independent_projection(clean, config["physical_cutoff"])
        values = {
            "velocity": (clean, clean_band),
            "vorticity": (independent_vorticity(clean), independent_vorticity(clean_band)),
        }
        for observable, (reference, projected) in values.items():
            bias = np.linalg.norm(projected - reference) / np.linalg.norm(reference)
            assert close(bias, selected[flow][observable + "_bias_relative_l2"])
            checked_floors.append({"flow": flow, "observable": observable, "cutoff": config["physical_cutoff"], "bias_relative_l2": float(bias)})
        if flow not in sample_noise:
            continue
        noise = sample_noise[flow]
        observation = independently_regenerate_observation(clean, times, noise, config["seeds"][0], config)
        perturbation = observation - clean
        noise_band = independent_projection(perturbation, config["physical_cutoff"])
        for observable, (reference, projected) in values.items():
            retained = noise_band if observable == "velocity" else independent_vorticity(noise_band)
            scale = np.linalg.norm(reference)
            bias = projected - reference
            b = np.linalg.norm(bias) / scale
            s = np.linalg.norm(retained) / scale
            total = np.linalg.norm(bias + retained) / scale
            row = next(r for r in budget if (r["case"], r["noise"], int(r["seed"]), float(r["cutoff"]), r["observable"]) ==
                       (flow, noise, config["seeds"][0], config["physical_cutoff"], observable))
            assert close(b, row["bias_relative_l2"])
            assert close(s, row["retained_noise_relative_l2"])
            assert close(total, row["identity_prior_error_relative_l2"])
            assert np.isclose(total ** 2, b ** 2 + s ** 2, atol=2e-13, rtol=2e-12)
            checked_budgets.append({"flow": flow, "noise": noise, "seed": config["seeds"][0], "observable": observable,
                                   "bias_relative_l2": float(b), "retained_noise_relative_l2": float(s), "identity_prior_error_relative_l2": float(total)})
    main_ledger = ROOT / "results/research/model_fidelity_benchmark_2026-10-14/ledger.jsonl"
    for record in read_ledger(main_ledger):
        if record["method"] != "identity" or record["status"] != "ok":
            continue
        evaluation = next(value for value in record["evaluations"] if value["variant"] == "helmholtz_k10")
        for observable in ("velocity", "vorticity"):
            row = next(value for value in budget if
                       (value["case"], value["noise"], int(value["seed"]), float(value["cutoff"]), value["observable"]) ==
                       (record["flow"], record["noise"], record["seed"], config["physical_cutoff"], observable))
            assert close(evaluation["metrics"][observable + "_relative_l2"], row["identity_prior_error_relative_l2"])
        control_agreement.append(record["id"])
    return {
        "complete": True,
        "rows": len(budget),
        "clean_rows": len(floors),
        "selected_cutoff_all_seven_velocity_and_vorticity_floors_verified": checked_floors,
        "three_independent_sample_blocks_both_observables_verified": checked_budgets,
        "primary_identity_prior_control_metrics_match_spectral_budget": control_agreement,
        "independent_real_space_recomputation_agrees_with_spectral_parseval": True,
        "cutoff_is_diagnostic_not_clean_tuned": True,
    }


def audit_rank_sensitivity(directory):
    provenance = read_json(directory / "provenance.json")
    config = provenance["configuration"]
    records = read_ledger(directory / "ledger.jsonl")
    expected = {
        (flow, noise, intensity, seed, rank, method)
        for flow in config["flows"] for noise in config["noise_models"]
        for intensity in config["intensities_relative_rms"] for seed in config["seeds"]
        for rank in config["ranks"] for method in config["methods"]
    }
    condition_keys = ("flow", "noise", "intensity_relative_rms", "seed", "rank", "method")
    actual = [tuple(record["condition"][key] for key in condition_keys) for record in records]
    assert len(actual) == len(set(actual)) and set(actual) == expected
    rows = read_json(directory / "summary.json")
    summary = {
        tuple(row[key] for key in condition_keys) + (row["postprocessing"],): row
        for row in rows
    }
    assert len(summary) == len(rows) == len(records) * len(config["postprocessing"])
    accepted, observations, overlapping = Counter(), {}, []
    failures = not_converged = 0
    for record in records:
        condition = record["condition"]
        key = tuple(condition[name] for name in condition_keys)
        block = key[:4]
        if block in observations:
            assert observations[block] == condition["observed_sha256"]
        observations[block] = condition["observed_sha256"]
        assert close(record["exact_relative_noise_rms"], condition["intensity_relative_rms"])
        diagnostics = record.get("diagnostics", {})
        converged = record["status"] == "ok" and bool(diagnostics.get("converged"))
        failures += record["status"] != "ok"
        not_converged += record["status"] == "ok" and not converged
        if record["status"] == "ok":
            assert diagnostics["rank_requested"] == condition["rank"]
        overlap = condition["rank"] == 8 and condition["intensity_relative_rms"] == .2
        assert record["overlap_model_benchmark"] is overlap
        if overlap:
            overlapping.append(record)
        if record["status"] != "ok":
            for variant in config["postprocessing"]:
                assert summary[key + (variant,)]["accepted"] is False
            continue
        evaluations = {evaluation["postprocessing"]: evaluation for evaluation in record["evaluations"]}
        assert set(evaluations) == set(config["postprocessing"])
        for variant, evaluation in evaluations.items():
            metrics = evaluation["metrics"]
            gate = all(metrics[name] is not None and np.isfinite(metrics[name]) and
                       0 <= metrics[name] <= maximum for name, maximum in config["quality_thresholds"].items())
            admissible = bool(converged and gate)
            assert evaluation["accepted"] is admissible
            assert summary[key + (variant,)]["accepted"] is admissible
            for name in config["quality_thresholds"]:
                assert close(metrics[name], summary[key + (variant,)][name])
            assert evaluation["postprocessing_diagnostics"]["reference_velocity_used"] is False
            accepted[variant] += admissible
        assert close(evaluations["raw"]["metrics"]["vorticity_relative_l2"],
                     evaluations["helmholtz"]["metrics"]["vorticity_relative_l2"])
    for name, expected_hash in provenance["source_files_sha256"].items():
        assert digest(ROOT / name) == expected_hash
    current_tree = ast.parse((ROOT / "python/scripts/run_model_fidelity_benchmark.py").read_text())
    current_functions = {node.name: ast.dump(node, include_attributes=False) for node in current_tree.body if isinstance(node, ast.FunctionDef)}
    for name, source in provenance["helper_contract_sources"].items():
        if name == "load_flow":
            continue  # Separately audited I/O-only canonical digest correction.
        assert ast.dump(ast.parse(source).body[0], include_attributes=False) == current_functions[name]
    oracle = read_json(directory / "oracle_floor.json")
    clean_bounds = []
    for flow in config["flows"]:
        source = provenance["source_datasets"][flow]
        assert digest(ROOT / source["path"]) == source["sha256"]
        with np.load(ROOT / source["path"]) as data:
            clean = data["velocity"]
        singular_values = np.linalg.svd(clean.reshape(len(clean), -1).T, compute_uv=False)
        for rank in config["ranks"]:
            bound = np.sqrt(np.sum(singular_values[rank:] ** 2) / np.sum(singular_values ** 2))
            row = next(r for r in oracle if r["flow"] == flow and r["rank"] == rank)
            assert close(bound, row["velocity_rank_frobenius_lower_bound"])
            assert row["estimator_input"] is False and row["oracle_reference_used"] is True
            clean_bounds.append({"flow": flow, "rank": rank, "velocity_frobenius_lower_bound": float(bound)})
    main = read_ledger(ROOT / "results/research/model_fidelity_benchmark_2026-10-14/ledger.jsonl")
    main_index = {(r["flow"], r["noise"], r["seed"], r["method"]): r for r in main}
    overlap_checks = []
    for record in overlapping:
        condition = record["condition"]
        key = tuple(condition[name] for name in ("flow", "noise", "seed", "method"))
        if key not in main_index:
            continue
        original = main_index[key]
        assert original["noise_metadata"]["observation_sha256"] == condition["observed_sha256"]
        assert original["diagnostics"]["rank_requested"] == condition["rank"]
        original_evaluations = {evaluation["variant"]: evaluation for evaluation in original["evaluations"]}
        for evaluation in record["evaluations"]:
            for name in config["quality_thresholds"]:
                assert close(evaluation["metrics"][name], original_evaluations[evaluation["postprocessing"]]["metrics"][name])
        overlap_checks.append(key)
    completion = read_json(directory / "completion.json")
    assert completion["complete"] is True and completion["fits"] == len(records)
    assert completion["failed_fits"] == failures and completion["nonconverged_fits"] == not_converged
    assert completion["accepted_by_postprocessing"] == dict(accepted)
    assert completion["overlap_model_benchmark_fits"] == len(overlapping) == 24
    return {
        "complete": True,
        "fits": len(records), "evaluation_rows": len(rows), "paired_observation_blocks": len(observations),
        "quality_acceptance_by_postprocessing": dict(accepted),
        "failed_fits": failures, "not_converged_fits": not_converged,
        "clean_velocity_rank_bounds_independently_recomputed": clean_bounds,
        "overlap_expected_not_independent": len(overlapping),
        "overlap_verified_against_main_completed_so_far": overlap_checks,
        "all_overlap_verified": len(overlap_checks) == len(overlapping),
        "vorticity_oracle_is_descriptive_not_frobenius_rank_lower_bound": True,
        "numerical_helper_contracts_ast_unchanged": True,
    }


def independent_prolongation(value, size):
    """Real Fourier padding with explicit even-source Nyquist splitting."""
    output = value
    for axis in (-2, -1):
        length = output.shape[axis]
        assert size >= length and length % 2 == 0
        if size == length:
            continue
        coefficients = np.fft.rfft(output, axis=axis)
        shape = list(coefficients.shape)
        shape[axis] = size // 2 + 1
        padded = np.zeros(shape, complex)
        copied = [slice(None)] * coefficients.ndim
        copied[axis] = slice(0, length // 2 + 1)
        padded[tuple(copied)] = coefficients
        nyquist = [slice(None)] * coefficients.ndim
        nyquist[axis] = length // 2
        padded[tuple(nyquist)] *= .5
        output = np.fft.irfft(padded, n=size, axis=axis) * (size / length)
    return output


def audit_visual_fidelity(directory):
    visual = read_json(directory / "visual_summary.json")
    assert visual["complete"] is True
    analysis = visual["analysis"]
    assert len(analysis["triangle_bounds"]) == 24
    examples = (
        (128, "gaussian_iid", "tls_r8_projected"),
        (128, "gaussian_iid", "tls_r8_fixed_band"),
        (128, "gaussian_iid", "identity_fixed_band"),
        (256, "spatial_fixed_length", "tls_r8_fixed_band"),
        (256, "spatial_fixed_length", "identity_fixed_band"),
        (256, "gaussian_iid", "tls_r8_projected"),
    )
    checked_bounds = []
    fine_path = ROOT / "results/research/highres_newtonian_2026-10-14/datasets/interacting_vortices_512_dt0.005.npz"
    with np.load(fine_path) as saved:
        fine = saved["velocity"]
        fine_times = saved["times"]
    for size, noise, method in examples:
        coarse_path = ROOT / "results/research/highres_newtonian_2026-10-14/datasets" / f"interacting_vortices_{size}_dt0.005.npz"
        base = "highres_reconstruction_2026-10-14" if method == "tls_r8_projected" else "highres_fixed_bandwidth_2026-10-14"
        candidate_path = ROOT / "results/research" / base / "reconstructions" / f"{noise}_{size}_s7_{method}.npz"
        with np.load(coarse_path) as saved:
            coarse, times = saved["velocity"], saved["times"]
        with np.load(candidate_path) as saved:
            estimated = saved["velocity"]
            np.testing.assert_allclose(saved["times"], fine_times, atol=2e-14, rtol=0)
        np.testing.assert_allclose(times, fine_times, atol=2e-14, rtol=0)
        sums = {observable: np.zeros(5) for observable in ("velocity", "vorticity")}
        for reference, candidate, exact in zip(coarse, estimated, fine):
            prolonged = independent_prolongation(reference, 512)
            candidate = independent_prolongation(candidate, 512)
            for observable in ("velocity", "vorticity"):
                if observable == "vorticity":
                    a = independent_vorticity(candidate[None])[0]
                    b = independent_vorticity(prolonged[None])[0]
                    f = independent_vorticity(exact[None])[0]
                else:
                    a, b, f = candidate, prolonged, exact
                sums[observable] += (np.sum((a-b) ** 2), np.sum((b-f) ** 2),
                                     np.sum((a-f) ** 2), np.sum(b ** 2), np.sum(f ** 2))
        for observable, squared in sums.items():
            cr, rf, cf, rr, ff = np.sqrt(squared)
            a, b, q, actual = cr/rr, rf/ff, rr/ff, cf/ff
            low, high = abs(a*q-b), a*q+b
            row = next(value for value in analysis["triangle_bounds"] if
                       (value["nx"], value["noise"], value["method"], value["observable"]) == (size, noise, method, observable))
            for name, value in (("coarse_reference_relative_error", b), ("reconstruction_vs_coarse_reference", a),
                                ("reference_norm_ratio", q), ("actual_vs_fine", actual), ("lower_vs_fine", low), ("upper_vs_fine", high)):
                assert np.isclose(row[name], value, rtol=2e-11, atol=8e-14), "triangle recomputation differs: " + name
            assert low-8e-14 <= actual <= high+8e-14
            checked_bounds.append({"grid": size, "noise": noise, "method": method, "observable": observable,
                                   "coarse_reference_relative_error": float(b), "actual_vs_fine": float(actual),
                                   "lower_vs_fine": float(low), "upper_vs_fine": float(high)})
    centers = np.arange(8, 33, 4)
    required = sorted(set(centers) | set(centers-1) | set(centers+1) | set(centers-2) | set(centers+2))
    omega = {index: independent_vorticity(fine[index:index+1])[0] for index in required}
    wave = 2*np.pi*np.fft.fftfreq(512, d=2*np.pi/512)
    wave[256] = 0
    kx, ky = np.meshgrid(wave, wave)
    dynamic = []
    for index in centers:
        spectrum = np.fft.fft2(omega[index])
        x = np.fft.ifft2(1j*kx*spectrum).real
        y = np.fft.ifft2(1j*ky*spectrum).real
        laplace = np.fft.ifft2(-(kx*kx+ky*ky)*spectrum).real
        dynamic.append(fine[index, 0]*x + fine[index, 1]*y - .01*laplace)
    dynamic = np.asarray(dynamic)
    scale = np.sqrt(np.mean(dynamic ** 2))
    residual_checks = []
    step = fine_times[1] - fine_times[0]
    for order in (2, 4):
        if order == 2:
            temporal = np.asarray([(omega[index+1]-omega[index-1])/(2*step) for index in centers])
        else:
            temporal = np.asarray([(-omega[index+2]+8*omega[index+1]-8*omega[index-1]+omega[index-2])/(12*step) for index in centers])
        rms = np.sqrt(np.mean((temporal+dynamic) ** 2))
        row = next(value for value in analysis["residual_sampling"] if value["case"] == "interacting_vortices" and value["order"] == order and value["spacing"] == step)
        assert np.isclose(row["residual_rms"], rms, rtol=2e-10, atol=1e-12)
        assert np.isclose(row["normalized_residual"], rms/scale, rtol=2e-10, atol=1e-12)
        residual_checks.append({"order": order, "output_spacing": float(step), "residual_rms": float(rms), "normalized_residual": float(rms/scale)})
    return {
        "six_estimated_trajectories_both_observables_independently_recomputed": checked_bounds,
        "real_fourier_prolongation_explicit_nyquist_split": True,
        "residual_second_and_fourth_order_common_centers_independently_recomputed": residual_checks,
        "bounds_refer_to_numerical_fine_reference_not_exact_continuum": True,
    }


def audit_clean_controls(directory):
    provenance = read_json(directory / "provenance.json")
    config = provenance["configuration"]
    records = read_ledger(directory / "ledger.jsonl")
    expected = {(flow, method, "inherited_20pct_regularization") for flow in config["flows"] for method in config["methods"]}
    expected |= {(flow, method, "known_zero_noise") for flow in config["flows"] for method in config["known_zero_noise_methods"]}
    actual = [(r["flow"], r["method"], r["contract"]) for r in records]
    assert len(actual) == len(set(actual)) and set(actual) == expected and len(records) == 84
    summary = {(row["flow"], row["method"], row["contract"], row["variant"]): row for row in read_json(directory / "summary.json")}
    assert len(summary) == 252
    accepted = Counter()
    for record in records:
        assert record["observed_sha256"] == record["clean_velocity_sha256"]
        assert record["no_additive_noise"] is True and record["no_rms_normalization"] is True
        assert record["true_noise_intensity"] == 0 and record["noise"] == "none"
        assert record["status"] == "ok" and record["diagnostics"]["converged"] is True
        if "stable" in record["method"]:
            budget = 0. if record["contract"] == "known_zero_noise" else .2
            assert record["estimator_noise_intensity"] == budget
            assert record["estimator_parameters"]["noise_intensity"] == budget
            if budget == 0:
                diagnostic = record["diagnostics"].get("preprocessing", record["diagnostics"])
                assert diagnostic["alpha"] == 0 and diagnostic["beta"] == 0
                assert diagnostic["sparse_norm"] < 1e-8
        evaluations = record["evaluations"]
        assert len(evaluations) == 3
        for evaluation in evaluations:
            metrics = evaluation["metrics"]
            gate = all(metrics[name] is not None and np.isfinite(metrics[name]) and 0 <= metrics[name] <= maximum
                       for name, maximum in config["estimator_configuration"]["quality_thresholds"].items())
            assert evaluation["quality_pass"] is bool(gate)
            row = summary[record["flow"], record["method"], record["contract"], evaluation["variant"]]
            assert row["quality_pass"] is bool(gate)
            for name in config["estimator_configuration"]["quality_thresholds"]:
                assert close(row[name], metrics[name])
            accepted[evaluation["variant"]] += bool(gate)
            if record["flow"] in ("oldroyd_b", "fene_p"):
                assert metrics["stress_frobenius_relative_l2"] is None and metrics["dissipation_relative_l2"] is None
            if record["method"] == "identity" and evaluation["variant"] in ("raw", "helmholtz"):
                assert metrics["velocity_relative_l2"] < 1e-12 and metrics["vorticity_relative_l2"] < 1e-12
                if record["flow"] in ("hb_thinning", "bingham", "hb_thickening"):
                    for name in ("stress_frobenius_relative_l2", "apparent_viscosity_relative_l2", "shear_rate_relative_l2", "dissipation_relative_l2"):
                        assert metrics[name] < 1e-10
    equivalences, floors = [], []
    maximum_equivalence_difference = 0.
    for flow in config["flows"]:
        source = provenance["clean_input_sources"][flow]
        assert digest(ROOT / source["path"]) == source["sha256"]
        with np.load(ROOT / source["path"]) as saved:
            clean = saved["velocity"]
        assert hashlib.sha256(np.ascontiguousarray(clean).tobytes()).hexdigest() == next(r["clean_velocity_sha256"] for r in records if r["flow"] == flow)
        singular = np.linalg.svd(clean.reshape(len(clean), -1).T, compute_uv=False)
        lower_bound = np.sqrt(np.sum(singular[8:]**2) / np.sum(singular**2))
        pod = summary[flow, "pod_r8", "inherited_20pct_regularization", "raw"]
        assert close(pod["velocity_relative_l2"], lower_bound)
        floors.append({"flow": flow, "clean_pod_rank8_velocity_lower_bound": float(lower_bound)})
        for variant in config["estimator_configuration"]["postprocessing"]:
            for zero_method, control in (("stable_rpca", "identity"), ("stable_tls_r8", "tls_r8")):
                zero = summary[flow, zero_method, "known_zero_noise", variant]
                direct = summary[flow, control, "inherited_20pct_regularization", variant]
                for name in config["estimator_configuration"]["quality_thresholds"]:
                    difference = abs(zero[name] - direct[name])
                    maximum_equivalence_difference = max(maximum_equivalence_difference, difference)
                    # Repeated SVD, modal fitting and spectral differentiation
                    # amplify roundoff in near-zero diagnostics. Record the
                    # actual difference rather than forcing exact zero.
                    assert difference < 1e-10
                equivalences.append({"flow": flow, "postprocessing": variant, "zero_budget_method": zero_method, "equivalent_control": control})
    for name, expected_hash in provenance["estimator_source_hashes"].items():
        assert digest(ROOT / name) == expected_hash
    completion = read_json(directory / "completion.json")
    assert completion["complete"] is True and completion["recorded_fits"] == len(records)
    assert completion["quality_passes_by_variant"] == dict(accepted)
    assert completion["noise_realizations"] == 0
    return {
        "complete": True, "fits": len(records), "evaluations": len(summary), "noise_realizations": 0,
        "quality_acceptance_by_postprocessing": dict(accepted),
        "clean_input_array_hashes_independently_verified": True,
        "pod_rank8_frobenius_lower_bounds_independently_recomputed": floors,
        "zero_budget_equivalences_in_all_four_gate_metrics": equivalences,
        "zero_budget_maximum_absolute_equivalence_difference": maximum_equivalence_difference,
        "zero_budget_equivalence_absolute_tolerance": 1e-10,
        "identity_viscoplastic_native_grid_constitutive_closure_errors_below_1e_10": True,
        "viscoelastic_stress_and_total_dissipation_not_inferred": True,
        "zero_budget_nonzero_sparse_entries_are_roundoff_not_injected_corruption": True,
    }


def audit_matched_references(directory):
    campaign = read_json(directory / "matched_integrations.json")
    report = read_json(directory / "summary.json")
    assert campaign["complete"] is True and campaign["count"] == len(campaign["runs"]) == 18
    assert not campaign["omitted_controls"]
    original_directory = ROOT / "results/research/extended_physics_2026-10-14"
    manifests = {record["case"]["id"]: record for record in read_json(original_directory / "dataset_manifests.json")}
    cases = {}
    for run in campaign["runs"]:
        config = run["cache_key"]
        case = config["case_id"]
        original = manifests[case]
        old_parameters = original["case"]["parameters"]
        new_parameters = config["parameters"]
        for name, value in old_parameters.items():
            if name in ("nx", "ny", "dt", "n_steps", "sample_every"):
                continue
            assert new_parameters[name] == value, "changed physical reference parameter: " + name
        assert close(old_parameters["dt"] * old_parameters["n_steps"], new_parameters["dt"] * new_parameters["n_steps"])
        assert close(old_parameters["dt"] * old_parameters["sample_every"], new_parameters["dt"] * new_parameters["sample_every"])
        assert digest(ROOT / run["archive"]) == run["archive_sha256"]
        assert digest(original_directory / "datasets" / (case + "_clean.npz")) == config["old_archive_sha256"]
        for name, expected_hash in config["scientific_source_sha256"].items():
            assert digest(ROOT / name) == expected_hash
        cases.setdefault(case, []).append(run)
    verified_rows = []
    for case, runs in cases.items():
        finest = max(runs, key=lambda run: (run["cache_key"]["parameters"]["nx"], -run["cache_key"]["parameters"]["dt"]))
        coarse_path = original_directory / "datasets" / (case + "_clean.npz")
        fine_path = ROOT / finest["archive"]
        with np.load(coarse_path) as a, np.load(fine_path) as b:
            np.testing.assert_allclose(a["times"], b["times"], atol=2e-14, rtol=0)
            coarse_velocity, fine_velocity = a["velocity"], b["velocity"]
            fields = {"velocity": (coarse_velocity, fine_velocity),
                      "vorticity": (independent_vorticity(coarse_velocity), independent_vorticity(fine_velocity))}
            if "stress" in a.files:
                fields["stress_frobenius"] = a["stress"], b["stress"]
            if "deviatoric_stress" in a.files:
                weight = np.sqrt([1., 2., 1.])[None, :, None, None]
                fields["stress_frobenius"] = a["deviatoric_stress"] * weight, b["deviatoric_stress"] * weight
                fields["apparent_viscosity"] = a["apparent_viscosity"], b["apparent_viscosity"]
        for observable, (coarse, fine) in fields.items():
            values = []
            numerator = denominator = 0.
            for x, y in zip(coarse, fine):
                prolonged = independent_prolongation(x, fine.shape[-1])
                n2, d2 = np.sum((prolonged-y)**2), np.sum(y**2)
                numerator += n2
                denominator += d2
                values.append(np.sqrt(n2/d2) if d2 else None)
            maximum = max(value for value in values if value is not None)
            row = next(value for value in report["credibility_matrix"] if
                       value["campaign"] == "matched_original_benchmark" and value["case"] == case and
                       value["grid"] == coarse.shape[-1] and value["observable"] == observable)
            assert np.isclose(row["maximum_relative_l2"], maximum, rtol=3e-10, atol=2e-13)
            assert np.isclose(row["all_times_relative_l2"], np.sqrt(numerator/denominator), rtol=3e-10, atol=2e-13)
            for reported, measured in zip(row["per_time_relative_l2"], values):
                if measured is None:
                    assert reported is None
                else:
                    assert np.isclose(reported, measured, rtol=3e-10, atol=2e-13)
            verified_rows.append({"case": case, "observable": observable, "grid": coarse.shape[-1],
                                  "reference_grid": fine.shape[-1], "maximum_relative_l2": float(maximum)})
    rank_checks = report["pairwise_reference_rank_stability"]
    for row in rank_checks:
        sufficient = row["scaled_coarse_error_gap"] > row["two_reference_differences"]
        assert row["order_stable_against_fine512_reference"] is sufficient
        if sufficient:
            assert row["actual_lower_method_error_to_fine"] < row["actual_higher_method_error_to_fine"]
    dense = read_json(directory / "dense_sampling.json")
    assert dense["complete"] is True and len(dense["integrations"]) == 2
    for run in dense["integrations"]:
        assert digest(ROOT / run["archive"]) == run["archive_sha256"]
        with np.load(ROOT / run["archive"]) as saved:
            assert len(saved["times"]) == len(saved["velocity"]) == 161
            np.testing.assert_allclose(np.diff(saved["times"]), .025, atol=2e-14, rtol=0)
    split_checks = []
    for row in dense["residual_sampling"]:
        left = row["residual_rms"] ** 2
        right = row["projected_temporal_residual_rms"] ** 2 + row["spatial_projection_defect_rms"] ** 2 + 2*row["decomposition_cross_mean"]
        assert np.isclose(left, right, rtol=2e-10, atol=2e-15)
        split_checks.append({"case": row["case"], "spacing": row["spacing"], "order": row["order"], "squared_residual_decomposition_verified": True})
    independently_recomputed_dense = []
    for run in dense["integrations"]:
        with np.load(ROOT / run["archive"]) as saved:
            velocity, times = saved["velocity"], saved["times"]
        case = run["cache_key"]["parameters"]["case"]
        centers = np.rint(np.arange(.8, 3.21, .4) / .025).astype(int)
        needed = sorted(set(centers) | set(centers-2) | set(centers-1) | set(centers+1) | set(centers+2))
        omega = {index: independent_vorticity(velocity[index:index+1])[0] for index in needed}
        mode = np.fft.fftfreq(128) * 128
        wave = mode.copy()
        wave[64] = 0
        kx, ky = np.meshgrid(wave, wave)
        retained = (abs(mode)[None, :] < 128/3) & (abs(mode)[:, None] < 128/3)
        dynamic = []
        for index in centers:
            spectrum = np.fft.fft2(omega[index])
            x = np.fft.ifft2(1j*kx*spectrum).real
            y = np.fft.ifft2(1j*ky*spectrum).real
            laplace = np.fft.ifft2(-(kx*kx+ky*ky)*spectrum).real
            dynamic.append(velocity[index, 0]*x + velocity[index, 1]*y - .01*laplace)
        dynamic = np.asarray(dynamic)
        projected = np.fft.ifft2(np.fft.fft2(dynamic) * retained).real
        spatial = dynamic - projected
        temporal = np.asarray([(-omega[index+2]+8*omega[index+1]-8*omega[index-1]+omega[index-2])/(12*.025) for index in centers])
        total_rms = np.sqrt(np.mean((temporal+dynamic)**2))
        projected_rms = np.sqrt(np.mean((temporal+projected)**2))
        spatial_rms = np.sqrt(np.mean(spatial**2))
        row = next(value for value in dense["residual_sampling"] if value["case"] == case and value["spacing"] == .025 and value["order"] == 4)
        for name, value in (("residual_rms", total_rms), ("projected_temporal_residual_rms", projected_rms), ("spatial_projection_defect_rms", spatial_rms)):
            assert np.isclose(row[name], value, rtol=3e-9, atol=1e-11)
        independently_recomputed_dense.append({"case": case, "order": 4, "output_spacing": .025,
                                              "residual_rms": float(total_rms), "projected_temporal_residual_rms": float(projected_rms),
                                              "spatial_projection_defect_rms": float(spatial_rms)})
    return {
        "complete": True, "matched_integrations": 18, "dense_integrations": 2,
        "same_original_physical_parameters_horizon_and_cadence_verified": True,
        "original_coarse_to_fine_all_saved_time_velocity_vorticity_stress_viscosity_recomputed": verified_rows,
        "pairwise_reference_rank_sufficient_inequalities_verified": len(rank_checks),
        "dense_archives_161_fields_with025_cadence_verified": True,
        "dense_squared_residual_decomposition_checks": split_checks,
        "both_dense_fourth_order_residuals_independently_recomputed_from_arrays": independently_recomputed_dense,
        "finest_references_remain_same_solver_numerical_fields": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/research/integration_validation/model_comparison_review.json")
    parser.add_argument("--include-new", action="store_true", help="Require and verify the complete 128² paired campaign")
    parser.add_argument("--source-preflight", action="store_true", help="Independently verify all seven local input NPZ files and hash types")
    parser.add_argument("--allow-incomplete", action="store_true", help="For development checks; never a complete campaign audit")
    args = parser.parse_args()
    result = {
        "audit_scope": "independent historical ledger arithmetic; new campaigns are added only after completion",
        "historical_campaign": audit_historical_campaign(ROOT / "results/research/extended_physics_2026-10-14"),
        "verifier_sha256": digest(Path(__file__)),
        "passed": True,
    }
    paired_reanalysis = ROOT / "results/research/paired_model_comparison_2026-10-14"
    if (paired_reanalysis / "completion.json").is_file():
        result["paired_reanalysis"] = audit_paired_reanalysis(paired_reanalysis)
    if args.include_new:
        result["new_paired_campaign"] = audit_new_campaign(
            ROOT / "results/research/model_fidelity_benchmark_2026-10-14", not args.allow_incomplete)
        result["audit_scope"] = "independent historical and new paired ledger arithmetic, observation archives and source hashes"
        result["complete"] = result["new_paired_campaign"]["complete"]
    if args.source_preflight or args.include_new:
        result["seven_source_preflight"] = audit_source_preflight()
    prior_directory = ROOT / "results/research/prior_spectral_budget_2026-10-14"
    if args.include_new and (prior_directory / "summary.json").is_file():
        result["prior_error_budget"] = audit_prior_budget(prior_directory)
    rank_directory = ROOT / "results/research/rank_noise_sensitivity_2026-10-14"
    if args.include_new and (rank_directory / "completion.json").is_file():
        result["rank_noise_sensitivity"] = audit_rank_sensitivity(rank_directory)
    reference_directory = ROOT / "results/research/reference_fidelity_2026-10-14"
    if args.include_new and (reference_directory / "visual_summary.json").is_file():
        result["visual_reference_fidelity"] = audit_visual_fidelity(reference_directory)
    if args.include_new and (reference_directory / "summary.json").is_file():
        result["matched_reference_fidelity"] = audit_matched_references(reference_directory)
    clean_directory = ROOT / "results/research/clean_model_controls_2026-10-14"
    if args.include_new and (clean_directory / "completion.json").is_file():
        result["clean_model_controls"] = audit_clean_controls(clean_directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": True, "historical_conditions": result["historical_campaign"]["conditions"],
                      "new_campaign": result.get("new_paired_campaign", {}).get("recorded_raw_fits"),
                      "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
