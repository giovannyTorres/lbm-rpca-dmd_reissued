from __future__ import annotations

import csv
import hashlib
import importlib
import itertools
import json
import math
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

from fluid_denoise.phase1_runtime import (
    configure_and_build,
    find_solver_binary,
    run_command,
)
from fluid_denoise.phase1_validation import validate_run_outputs
from fluid_denoise.phase2_clean_dataset import (
    build_experiment_id as build_clean_experiment_id,
    convert_raw_run_to_clean_dataset,
    validate_clean_dataset,
)
from fluid_denoise.phase3_noisy_dataset import (
    build_noisy_experiment_id,
    create_noisy_dataset,
    validate_noisy_dataset,
)
from fluid_denoise.phase4_model_data import align_model_runs, stack_run_variables
from fluid_denoise.phase5_baselines import (
    create_baseline_model,
    materialize_reconstruction_grid,
    prepare_baseline_input,
    save_baseline_reconstruction,
)


SCHEMA_VERSION = "phase6.benchmark.v1"
REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ExpandedBenchmarkCase:
    benchmark_id: str
    experiment_id: str
    resolution_label: str
    nx: int
    ny: int
    scaled_geometry: dict[str, int]
    generated_solver_config_path: Path
    solver_config: dict[str, Any]
    raw_run_dir: Path
    clean_run_dir: Path
    noisy_run_dir: Path
    noise_label: str
    noise_specs: list[dict[str, Any]]
    model_name: str
    model_params: dict[str, Any]
    model_input: dict[str, Any]
    experiment_dir: Path


@dataclass(frozen=True)
class BenchmarkRunSummary:
    benchmark_id: str
    num_cases: int
    completed: int
    failed: int
    skipped: int
    metrics_root: Path
    ledger_path: Path
    summary_csv_path: Path
    summary_parquet_path: Path | None


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_token(value: str) -> str:
    import re

    sanitized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return sanitized or "unknown"


def _stable_hash(payload: Any, *, length: int = 10) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:length]


def _resolve_path(base_dir: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_benchmark_config(config_path: Path | str) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    if config_path.suffix.lower() != ".json":
        raise ValueError(
            "FASE 6 currently supports JSON configs only. Use a .json file for the benchmark config."
        )
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Benchmark config must be a JSON object")
    return payload


def _validate_benchmark_config(config_payload: Mapping[str, Any]) -> None:
    required_sections = ("benchmark", "solver", "noise", "models")
    for section in required_sections:
        if section not in config_payload:
            raise ValueError(f"Benchmark config is missing required section: {section}")

    benchmark_config = config_payload["benchmark"]
    solver_config = config_payload["solver"]
    noise_cases = config_payload["noise"]
    models = config_payload["models"]

    if not isinstance(benchmark_config, Mapping):
        raise ValueError("benchmark section must be an object")
    if not isinstance(solver_config, Mapping):
        raise ValueError("solver section must be an object")
    if not isinstance(noise_cases, Sequence) or isinstance(noise_cases, (str, bytes)):
        raise ValueError("noise section must be an array of noise cases")
    if not noise_cases:
        raise ValueError("noise section cannot be empty")
    if not isinstance(models, Sequence) or isinstance(models, (str, bytes)):
        raise ValueError("models section must be an array of models")
    if not models:
        raise ValueError("models section cannot be empty")

    if not str(benchmark_config.get("benchmark_id", "")).strip():
        raise ValueError("benchmark.benchmark_id is required")
    if not str(solver_config.get("base_config_path", "")).strip():
        raise ValueError("solver.base_config_path is required")
    resolutions = solver_config.get("resolutions")
    if not isinstance(resolutions, Sequence) or isinstance(resolutions, (str, bytes)) or not resolutions:
        raise ValueError("solver.resolutions must be a non-empty array")


def _load_base_solver_config(
    config_payload: Mapping[str, Any],
    *,
    config_base_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    solver_config = config_payload["solver"]
    base_config_path = _resolve_path(config_base_dir, str(solver_config["base_config_path"]))
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base solver config not found: {base_config_path}")
    payload = json.loads(base_config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Base solver config must be a JSON object")
    return base_config_path, payload


def _normalize_noise_case(noise_case: Mapping[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    label = str(noise_case.get("label", "")).strip() or f"noise-{_stable_hash(dict(noise_case), length=6)}"
    raw_specs = noise_case.get("noise_specs")
    if not isinstance(raw_specs, Sequence) or isinstance(raw_specs, (str, bytes)) or not raw_specs:
        raise ValueError(f"Noise case '{label}' must define a non-empty noise_specs array")
    return label, [dict(spec) for spec in raw_specs]


def _expand_param_grid(model_config: Mapping[str, Any]) -> list[dict[str, Any]]:
    base_params = dict(model_config.get("params", {}))
    raw_grid = dict(model_config.get("param_grid", {}))
    if not raw_grid:
        return [base_params]

    keys = sorted(raw_grid)
    value_lists: list[list[Any]] = []
    for key in keys:
        raw_value = raw_grid[key]
        if isinstance(raw_value, Sequence) and not isinstance(raw_value, (str, bytes)):
            values = list(raw_value)
        else:
            values = [raw_value]
        if not values:
            raise ValueError(f"Model param_grid entry '{key}' cannot be empty")
        value_lists.append(values)

    expanded: list[dict[str, Any]] = []
    for combination in itertools.product(*value_lists):
        params = dict(base_params)
        for key, value in zip(keys, combination):
            params[key] = value
        expanded.append(params)
    return expanded


def _scale_geometry(base_config: Mapping[str, Any], *, nx: int, ny: int) -> dict[str, int]:
    base_nx = int(base_config["nx"])
    base_ny = int(base_config["ny"])
    if base_nx <= 0 or base_ny <= 0:
        raise ValueError("Base solver config must have positive nx and ny")

    scale_x = float(nx) / float(base_nx)
    scale_y = float(ny) / float(base_ny)
    radius_scale = min(scale_x, scale_y)

    obstacle_cx = int(round(float(base_config["obstacle_cx"]) * scale_x))
    obstacle_cy = int(round(float(base_config["obstacle_cy"]) * scale_y))
    obstacle_r = int(round(float(base_config["obstacle_r"]) * radius_scale))

    obstacle_cx = int(np.clip(obstacle_cx, 0, max(0, nx - 1)))
    obstacle_cy = int(np.clip(obstacle_cy, 0, max(0, ny - 1)))
    if float(base_config["obstacle_r"]) > 0.0:
        obstacle_r = max(1, obstacle_r)
    obstacle_r = int(np.clip(obstacle_r, 0, max(0, min(nx, ny) // 2)))

    return {
        "obstacle_cx": obstacle_cx,
        "obstacle_cy": obstacle_cy,
        "obstacle_r": obstacle_r,
    }


def _build_solver_run_id(
    benchmark_id: str,
    resolution_label: str,
    effective_solver_config: Mapping[str, Any],
) -> str:
    payload = {
        "benchmark_id": benchmark_id,
        "resolution_label": resolution_label,
        "nx": effective_solver_config["nx"],
        "ny": effective_solver_config["ny"],
        "obstacle_cx": effective_solver_config["obstacle_cx"],
        "obstacle_cy": effective_solver_config["obstacle_cy"],
        "obstacle_r": effective_solver_config["obstacle_r"],
        "reynolds": effective_solver_config["reynolds"],
        "u_in": effective_solver_config["u_in"],
        "seed": effective_solver_config.get("seed"),
    }
    digest = _stable_hash(payload)
    return (
        f"{_sanitize_token(benchmark_id)[:20]}__"
        f"{_sanitize_token(resolution_label)[:18]}__"
        f"h-{digest}"
    )


def _build_case_experiment_id(
    *,
    benchmark_id: str,
    resolution_label: str,
    noise_label: str,
    model_name: str,
    model_params: Mapping[str, Any],
    model_input: Mapping[str, Any],
    noise_specs: Sequence[Mapping[str, Any]],
    nx: int,
    ny: int,
) -> str:
    digest = _stable_hash(
        {
            "benchmark_id": benchmark_id,
            "resolution_label": resolution_label,
            "noise_label": noise_label,
            "noise_specs": list(noise_specs),
            "model_name": model_name,
            "model_params": dict(model_params),
            "model_input": dict(model_input),
            "nx": nx,
            "ny": ny,
        }
    )
    return (
        f"{_sanitize_token(benchmark_id)[:16]}__"
        f"{_sanitize_token(resolution_label)[:14]}__"
        f"{_sanitize_token(noise_label)[:14]}__"
        f"{_sanitize_token(model_name)[:14]}__"
        f"h-{digest}"
    )


def expand_benchmark_cases(
    config_payload: Mapping[str, Any],
    *,
    config_base_dir: Path | None = None,
) -> list[ExpandedBenchmarkCase]:
    _validate_benchmark_config(config_payload)
    base_dir = REPO_ROOT if config_base_dir is None else Path(config_base_dir).resolve()
    benchmark_config = config_payload["benchmark"]
    solver_section = config_payload["solver"]
    benchmark_id = str(benchmark_config["benchmark_id"]).strip()

    _, base_solver_config = _load_base_solver_config(config_payload, config_base_dir=base_dir)

    raw_output_root = _resolve_path(
        base_dir,
        str(solver_section.get("output_root", base_solver_config.get("output_root", "data/raw"))),
    )
    clean_root = _resolve_path(
        base_dir,
        str(config_payload.get("execution", {}).get("clean_root", "data/clean")),
    )
    noisy_root = _resolve_path(
        base_dir,
        str(config_payload.get("execution", {}).get("noisy_root", "data/noisy")),
    )
    metrics_root = _resolve_path(
        base_dir,
        str(benchmark_config.get("metrics_root", "results/metrics")),
    ) / benchmark_id
    generated_config_root = metrics_root / "generated_solver_configs"
    experiments_root = metrics_root / "experiments"

    expanded_cases: list[ExpandedBenchmarkCase] = []
    for resolution_entry in solver_section["resolutions"]:
        if not isinstance(resolution_entry, Mapping):
            raise ValueError("Each resolution entry must be an object")
        nx = int(resolution_entry["nx"])
        ny = int(resolution_entry["ny"])
        resolution_label = (
            str(resolution_entry.get("label", "")).strip() or f"nx{nx}_ny{ny}"
        )

        effective_solver_config = dict(base_solver_config)
        effective_solver_config["nx"] = nx
        effective_solver_config["ny"] = ny
        effective_solver_config.update(_scale_geometry(base_solver_config, nx=nx, ny=ny))
        effective_solver_config["output_root"] = str(raw_output_root)
        effective_solver_config.update(dict(resolution_entry.get("overrides", {})))
        solver_run_id = _build_solver_run_id(benchmark_id, resolution_label, effective_solver_config)
        effective_solver_config["run_id"] = solver_run_id

        generated_solver_config_path = (
            generated_config_root / f"{_sanitize_token(resolution_label)[:24]}__{solver_run_id}.json"
        )
        raw_run_dir = raw_output_root / solver_run_id

        clean_experiment_id = build_clean_experiment_id(
            dict(effective_solver_config),
            dict(effective_solver_config),
        )
        clean_run_dir = clean_root / clean_experiment_id

        for noise_case in config_payload["noise"]:
            if not isinstance(noise_case, Mapping):
                raise ValueError("Each noise case must be an object")
            noise_label, noise_specs = _normalize_noise_case(noise_case)
            noisy_experiment_id = build_noisy_experiment_id(
                {"experiment_id": clean_experiment_id},
                noise_specs,
            )
            noisy_run_dir = noisy_root / noisy_experiment_id

            for model_config in config_payload["models"]:
                if not isinstance(model_config, Mapping):
                    raise ValueError("Each model entry must be an object")
                model_name = str(model_config.get("name", "")).strip().lower()
                if not model_name:
                    raise ValueError("Every model entry must define name")
                model_input = dict(model_config.get("input", {}))
                for model_params in _expand_param_grid(model_config):
                    experiment_id = _build_case_experiment_id(
                        benchmark_id=benchmark_id,
                        resolution_label=resolution_label,
                        noise_label=noise_label,
                        model_name=model_name,
                        model_params=model_params,
                        model_input=model_input,
                        noise_specs=noise_specs,
                        nx=nx,
                        ny=ny,
                    )
                    expanded_cases.append(
                        ExpandedBenchmarkCase(
                            benchmark_id=benchmark_id,
                            experiment_id=experiment_id,
                            resolution_label=resolution_label,
                            nx=nx,
                            ny=ny,
                            scaled_geometry={
                                "obstacle_cx": int(effective_solver_config["obstacle_cx"]),
                                "obstacle_cy": int(effective_solver_config["obstacle_cy"]),
                                "obstacle_r": int(effective_solver_config["obstacle_r"]),
                            },
                            generated_solver_config_path=generated_solver_config_path,
                            solver_config=dict(effective_solver_config),
                            raw_run_dir=raw_run_dir,
                            clean_run_dir=clean_run_dir,
                            noisy_run_dir=noisy_run_dir,
                            noise_label=noise_label,
                            noise_specs=[dict(spec) for spec in noise_specs],
                            model_name=model_name,
                            model_params=dict(model_params),
                            model_input=model_input,
                            experiment_dir=experiments_root / experiment_id,
                        )
                    )

    if not expanded_cases:
        raise ValueError("Benchmark expansion produced no cases")
    return expanded_cases


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(subvalue) for key, subvalue in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(dict(payload)), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _append_ledger_record(ledger_path: Path, record: Mapping[str, Any]) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_ready(dict(record)), sort_keys=True))
        handle.write("\n")


def _load_ledger_entries(ledger_path: Path) -> list[dict[str, Any]]:
    if not ledger_path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        entries.append(json.loads(stripped))
    return entries


def _has_completed_artifacts(case: ExpandedBenchmarkCase) -> bool:
    return (
        (case.experiment_dir / "reconstruction.npz").exists()
        and (case.experiment_dir / "metadata.json").exists()
        and (case.experiment_dir / "benchmark_result.json").exists()
    )


def _ensure_generated_solver_config(case: ExpandedBenchmarkCase) -> None:
    _write_json(case.generated_solver_config_path, case.solver_config)


def _ensure_solver_binary(
    *,
    build_state: dict[str, Any],
    build_dir: Path,
    build_type: str,
) -> Path:
    binary = build_state.get("solver_binary")
    if isinstance(binary, Path) and binary.exists():
        return binary

    source_dir = REPO_ROOT / "cpp" / "lbm_core"
    configure_and_build(source_dir, build_dir, build_type=build_type)
    solver_binary = find_solver_binary(build_dir, build_type=build_type)
    build_state["solver_binary"] = solver_binary
    return solver_binary


def _ensure_raw_run(
    case: ExpandedBenchmarkCase,
    *,
    build_state: dict[str, Any],
    build_dir: Path,
    build_type: str,
    reuse_existing: bool,
) -> Path:
    _ensure_generated_solver_config(case)
    if reuse_existing and case.raw_run_dir.exists():
        try:
            validate_run_outputs(case.raw_run_dir)
            return case.raw_run_dir
        except Exception:
            pass

    solver_binary = _ensure_solver_binary(
        build_state=build_state,
        build_dir=build_dir,
        build_type=build_type,
    )
    run_command([str(solver_binary), "--config", str(case.generated_solver_config_path)], cwd=build_dir)
    validate_run_outputs(case.raw_run_dir)
    return case.raw_run_dir


def _ensure_clean_run(case: ExpandedBenchmarkCase, *, raw_run_dir: Path, reuse_existing: bool) -> Path:
    if reuse_existing and case.clean_run_dir.exists():
        try:
            validate_clean_dataset(case.clean_run_dir)
            return case.clean_run_dir
        except Exception:
            pass

    return convert_raw_run_to_clean_dataset(
        raw_run_dir,
        clean_root=case.clean_run_dir.parent,
        experiment_id=case.clean_run_dir.name,
        overwrite=True,
    )


def _ensure_noisy_run(case: ExpandedBenchmarkCase, *, clean_run_dir: Path, reuse_existing: bool) -> Path:
    if reuse_existing and case.noisy_run_dir.exists():
        try:
            validate_noisy_dataset(case.noisy_run_dir)
            return case.noisy_run_dir
        except Exception:
            pass

    return create_noisy_dataset(
        clean_run_dir,
        noise_specs=case.noise_specs,
        noisy_root=case.noisy_run_dir.parent,
        experiment_id=case.noisy_run_dir.name,
        overwrite=True,
    )


def _ensure_parquet_supported() -> None:
    try:
        importlib.import_module("pandas")
        importlib.import_module("pyarrow")
    except Exception as exc:
        raise RuntimeError(
            "Parquet export was requested, but pandas+pyarrow are not available in this environment."
        ) from exc


def _export_summary_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    csv_path: Path,
    parquet_path: Path | None,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_rows = [dict(row) for row in rows]

    if ordered_rows:
        preferred_columns = [
            "benchmark_id",
            "experiment_id",
            "resolution_label",
            "nx",
            "ny",
            "noise_label",
            "model_name",
            "model_input_kind",
            "variables",
            "rmse",
            "mae",
            "relative_l2_error",
            "psnr",
            "vorticity_rmse",
            "kinetic_energy_relative_l2",
            "divergence_residual_rms",
            "reconstruction_time_sec",
            "total_time_sec",
            "estimated_memory_bytes",
            "warnings",
            "raw_run_dir",
            "clean_run_dir",
            "noisy_run_dir",
            "reconstruction_dir",
        ]
        remaining_columns = sorted(
            {
                key
                for row in ordered_rows
                for key in row
                if key not in preferred_columns
            }
        )
        fieldnames = [column for column in preferred_columns if any(column in row for row in ordered_rows)]
        fieldnames.extend(remaining_columns)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(ordered_rows)
    else:
        csv_path.write_text("", encoding="utf-8")

    if parquet_path is not None:
        _ensure_parquet_supported()
        import pandas as pd

        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(ordered_rows).to_parquet(parquet_path, index=False)


def _flatten_fluid_values(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    flat = np.asarray(array, dtype=np.float64).reshape(array.shape[:-2] + (mask.size,))
    fluid_flat = (mask.reshape(-1) == 0)
    return flat[..., fluid_flat].reshape(-1)


def _compute_vorticity_from_velocity(ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    dux_dy = np.gradient(ux, axis=1)
    duy_dx = np.gradient(uy, axis=2)
    return np.asarray(duy_dx - dux_dy, dtype=np.float64)


def _compute_divergence_from_velocity(ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    dux_dx = np.gradient(ux, axis=2)
    duy_dy = np.gradient(uy, axis=1)
    return np.asarray(dux_dx + duy_dy, dtype=np.float64)


def _estimate_memory_bytes(*arrays: np.ndarray | None) -> int:
    total = 0
    for array in arrays:
        if array is None:
            continue
        total += int(np.asarray(array).nbytes)
    return total


def compute_benchmark_metrics(
    reconstructed_grid: np.ndarray,
    reference_grid: np.ndarray,
    *,
    variables: Sequence[str],
    mask: np.ndarray,
    eps: float = 1e-12,
) -> tuple[dict[str, float | None], list[str]]:
    reconstructed = np.asarray(reconstructed_grid, dtype=np.float64)
    reference = np.asarray(reference_grid, dtype=np.float64)
    obstacle_mask = np.asarray(mask)
    normalized_variables = tuple(str(variable).strip().lower() for variable in variables)

    if reconstructed.shape != reference.shape:
        raise ValueError("Reconstructed and reference grids must have the same shape")
    if reconstructed.ndim != 4:
        raise ValueError("Benchmark metrics expect tensors shaped as (T, C, ny, nx)")
    if obstacle_mask.shape != reconstructed.shape[-2:]:
        raise ValueError("Mask shape does not match the reconstructed grid")
    if reconstructed.shape[1] != len(normalized_variables):
        raise ValueError("Variable list length does not match the channel dimension")
    if not np.all(np.isfinite(reconstructed)) or not np.all(np.isfinite(reference)):
        raise ValueError("Metrics require finite reconstructed and reference values")

    warnings: list[str] = []
    recon_values = _flatten_fluid_values(reconstructed, obstacle_mask)
    ref_values = _flatten_fluid_values(reference, obstacle_mask)
    if recon_values.size == 0:
        raise ValueError("No fluid cells are available for metric evaluation")

    error = recon_values - ref_values
    mse = float(np.mean(error * error))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(error)))
    ref_norm = float(np.linalg.norm(ref_values, ord=2))
    relative_l2 = float(np.linalg.norm(error, ord=2) / max(ref_norm, eps))
    dynamic_range = float(np.max(ref_values) - np.min(ref_values))
    psnr: float | None
    if dynamic_range <= eps or mse <= eps:
        psnr = None
    else:
        psnr = float(20.0 * math.log10(dynamic_range / math.sqrt(mse)))

    metrics: dict[str, float | None] = {
        "rmse": rmse,
        "mae": mae,
        "relative_l2_error": relative_l2,
        "psnr": psnr,
        "vorticity_rmse": None,
        "kinetic_energy_relative_l2": None,
        "divergence_residual_rms": None,
    }

    if not {"ux", "uy"}.issubset(normalized_variables):
        warnings.append(
            "Physical metrics require both ux and uy in the reconstructed variables; storing nulls for vorticity, kinetic energy, and divergence."
        )
        return metrics, warnings

    channel_index = {name: idx for idx, name in enumerate(normalized_variables)}
    ux_reference = reference[:, channel_index["ux"], :, :]
    uy_reference = reference[:, channel_index["uy"], :, :]
    ux_reconstructed = reconstructed[:, channel_index["ux"], :, :]
    uy_reconstructed = reconstructed[:, channel_index["uy"], :, :]

    reference_vorticity = _compute_vorticity_from_velocity(ux_reference, uy_reference)
    reconstructed_vorticity = _compute_vorticity_from_velocity(ux_reconstructed, uy_reconstructed)
    vorticity_error = _flatten_fluid_values(reconstructed_vorticity - reference_vorticity, obstacle_mask)
    metrics["vorticity_rmse"] = float(np.sqrt(np.mean(vorticity_error * vorticity_error)))

    reference_energy = 0.5 * (ux_reference * ux_reference + uy_reference * uy_reference)
    reconstructed_energy = 0.5 * (ux_reconstructed * ux_reconstructed + uy_reconstructed * uy_reconstructed)
    energy_error = _flatten_fluid_values(reconstructed_energy - reference_energy, obstacle_mask)
    energy_reference = _flatten_fluid_values(reference_energy, obstacle_mask)
    metrics["kinetic_energy_relative_l2"] = float(
        np.linalg.norm(energy_error, ord=2) / max(float(np.linalg.norm(energy_reference, ord=2)), eps)
    )

    divergence = _compute_divergence_from_velocity(ux_reconstructed, uy_reconstructed)
    divergence_values = _flatten_fluid_values(divergence, obstacle_mask)
    metrics["divergence_residual_rms"] = float(np.sqrt(np.mean(divergence_values * divergence_values)))
    return metrics, warnings


def _case_to_payload(case: ExpandedBenchmarkCase) -> dict[str, Any]:
    payload = asdict(case)
    return _json_ready(payload)


def _build_summary_row(
    *,
    case: ExpandedBenchmarkCase,
    metrics: Mapping[str, float | None],
    warnings: Sequence[str],
    reconstruction_time_sec: float,
    total_time_sec: float,
    estimated_memory_bytes: int,
) -> dict[str, Any]:
    return {
        "benchmark_id": case.benchmark_id,
        "experiment_id": case.experiment_id,
        "resolution_label": case.resolution_label,
        "nx": case.nx,
        "ny": case.ny,
        "obstacle_cx": case.scaled_geometry["obstacle_cx"],
        "obstacle_cy": case.scaled_geometry["obstacle_cy"],
        "obstacle_r": case.scaled_geometry["obstacle_r"],
        "noise_label": case.noise_label,
        "noise_specs_json": json.dumps(case.noise_specs, sort_keys=True),
        "model_name": case.model_name,
        "model_params_json": json.dumps(case.model_params, sort_keys=True),
        "model_input_kind": str(case.model_input.get("kind", "space_time_matrix")),
        "model_input_json": json.dumps(case.model_input, sort_keys=True),
        "variables": ",".join(str(value) for value in case.model_input.get("variables", ["ux"])),
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "relative_l2_error": metrics["relative_l2_error"],
        "psnr": metrics["psnr"],
        "vorticity_rmse": metrics["vorticity_rmse"],
        "kinetic_energy_relative_l2": metrics["kinetic_energy_relative_l2"],
        "divergence_residual_rms": metrics["divergence_residual_rms"],
        "reconstruction_time_sec": reconstruction_time_sec,
        "total_time_sec": total_time_sec,
        "estimated_memory_bytes": estimated_memory_bytes,
        "warnings": json.dumps(list(warnings), sort_keys=True),
        "raw_run_dir": str(case.raw_run_dir.resolve()),
        "clean_run_dir": str(case.clean_run_dir.resolve()),
        "noisy_run_dir": str(case.noisy_run_dir.resolve()),
        "reconstruction_dir": str(case.experiment_dir.resolve()),
    }


def _write_benchmark_result(
    case: ExpandedBenchmarkCase,
    *,
    summary_row: Mapping[str, Any],
    metrics: Mapping[str, float | None],
    warnings: Sequence[str],
) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_id": case.benchmark_id,
        "experiment_id": case.experiment_id,
        "case": _case_to_payload(case),
        "metrics": dict(metrics),
        "warnings": list(warnings),
        "summary_row": dict(summary_row),
    }
    _write_json(case.experiment_dir / "benchmark_result.json", payload)


def _run_model_reconstruction(
    case: ExpandedBenchmarkCase,
    *,
    noisy_run_dir: Path,
    clean_run_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], float, int]:
    prepared_input = prepare_baseline_input(
        noisy_run_dir,
        input_kind=str(case.model_input.get("kind", "space_time_matrix")),
        variables=case.model_input.get("variables", ["ux"]),
        steps=case.model_input.get("steps"),
        snapshot_indices=case.model_input.get("snapshot_indices"),
        mask_policy=case.model_input.get("mask_policy"),
        obstacle_fill_value=float(case.model_input.get("obstacle_fill_value", 0.0)),
        include_mask_channel=bool(case.model_input.get("include_mask_channel", False)),
        normalization=case.model_input.get("normalization"),
    )

    model = create_baseline_model(case.model_name, case.model_params)
    fit_start = perf_counter()
    model.fit(prepared_input)
    reconstructed_data = model.reconstruct()
    reconstruction_time_sec = perf_counter() - fit_start
    reconstructed_grid = materialize_reconstruction_grid(prepared_input, reconstructed_data)

    obstacle = prepared_input.run.mask == 1
    reconstructed_flat = reconstructed_grid.reshape(reconstructed_grid.shape[0], reconstructed_grid.shape[1], -1)
    observed_flat = prepared_input.observed_grid.reshape(
        prepared_input.observed_grid.shape[0],
        prepared_input.observed_grid.shape[1],
        -1,
    )
    obstacle_flat = obstacle.reshape(-1)
    reconstructed_flat[:, :, obstacle_flat] = observed_flat[:, :, obstacle_flat]
    reconstructed_grid = reconstructed_flat.reshape(reconstructed_grid.shape)

    aligned = align_model_runs(noisy_run_dir, clean_run_dir)
    reference_grid, _ = stack_run_variables(
        aligned.reference,
        variables=prepared_input.variables,
        steps=prepared_input.steps.tolist(),
    )
    save_baseline_reconstruction(
        case.experiment_dir,
        prepared_input=prepared_input,
        model=model,
        reconstructed_grid=reconstructed_grid,
        config_payload={"phase6_case": _case_to_payload(case)},
        reference_grid=reference_grid,
    )

    estimated_memory_bytes = _estimate_memory_bytes(
        prepared_input.data,
        prepared_input.observed_grid,
        reconstructed_data,
        reconstructed_grid,
        reference_grid,
    )
    return (
        np.asarray(reconstructed_grid, dtype=np.float64),
        np.asarray(reference_grid, dtype=np.float64),
        np.asarray(prepared_input.run.mask),
        list(model.get_warnings()),
        float(reconstruction_time_sec),
        estimated_memory_bytes,
    )


def _collect_completed_rows(ledger_entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    latest_completed: dict[str, dict[str, Any]] = {}
    for entry in ledger_entries:
        if entry.get("status") == "completed" and "summary_row" in entry:
            latest_completed[str(entry["experiment_id"])] = dict(entry["summary_row"])
    return list(latest_completed.values())


def run_benchmark(
    config_payload: Mapping[str, Any],
    *,
    config_path: Path | str | None = None,
) -> BenchmarkRunSummary:
    _validate_benchmark_config(config_payload)
    config_base_dir = REPO_ROOT if config_path is None else Path(config_path).resolve().parent
    benchmark_config = config_payload["benchmark"]
    solver_section = config_payload["solver"]
    execution = dict(config_payload.get("execution", {}))
    benchmark_id = str(benchmark_config["benchmark_id"]).strip()

    metrics_root = _resolve_path(
        config_base_dir,
        str(benchmark_config.get("metrics_root", "results/metrics")),
    ) / benchmark_id
    tables_root = _resolve_path(
        config_base_dir,
        str(benchmark_config.get("tables_root", "results/tables")),
    ) / benchmark_id
    ledger_path = metrics_root / "ledger.jsonl"
    summary_csv_path = tables_root / "summary.csv"
    raw_export_formats = benchmark_config.get("export_formats", ["csv"])
    if isinstance(raw_export_formats, Sequence) and not isinstance(raw_export_formats, (str, bytes)):
        export_formats = [str(value).strip().lower() for value in raw_export_formats]
    else:
        export_formats = [str(raw_export_formats).strip().lower()]
    summary_parquet_path = tables_root / "summary.parquet" if "parquet" in export_formats else None

    metrics_root.mkdir(parents=True, exist_ok=True)
    tables_root.mkdir(parents=True, exist_ok=True)
    _write_json(metrics_root / "benchmark_config_snapshot.json", dict(config_payload))

    cases = expand_benchmark_cases(config_payload, config_base_dir=config_base_dir)
    ledger_entries = _load_ledger_entries(ledger_path)
    latest_status = {str(entry["experiment_id"]): dict(entry) for entry in ledger_entries if "experiment_id" in entry}

    resume = bool(benchmark_config.get("resume", True))
    reuse_existing = bool(execution.get("reuse_existing", True))
    build_dir = _resolve_path(
        config_base_dir,
        str(solver_section.get("build_dir", "cpp/lbm_core/build_phase6")),
    )
    build_type = str(solver_section.get("build_type", "Release"))
    build_state: dict[str, Any] = {}

    completed_count = 0
    failed_count = 0
    skipped_count = 0

    for case in cases:
        previous_entry = latest_status.get(case.experiment_id)
        if (
            resume
            and previous_entry is not None
            and previous_entry.get("status") == "completed"
            and _has_completed_artifacts(case)
        ):
            skipped_record = {
                "schema_version": SCHEMA_VERSION,
                "timestamp": _iso_now(),
                "benchmark_id": benchmark_id,
                "experiment_id": case.experiment_id,
                "status": "skipped",
                "reason": "already_completed",
                "case": _case_to_payload(case),
            }
            _append_ledger_record(ledger_path, skipped_record)
            latest_status[case.experiment_id] = skipped_record
            skipped_count += 1
            continue

        case_start = perf_counter()
        started_record = {
            "schema_version": SCHEMA_VERSION,
            "timestamp": _iso_now(),
            "benchmark_id": benchmark_id,
            "experiment_id": case.experiment_id,
            "status": "started",
            "case": _case_to_payload(case),
        }
        _append_ledger_record(ledger_path, started_record)

        try:
            raw_run_dir = _ensure_raw_run(
                case,
                build_state=build_state,
                build_dir=build_dir,
                build_type=build_type,
                reuse_existing=reuse_existing,
            )
            clean_run_dir = _ensure_clean_run(case, raw_run_dir=raw_run_dir, reuse_existing=reuse_existing)
            noisy_run_dir = _ensure_noisy_run(case, clean_run_dir=clean_run_dir, reuse_existing=reuse_existing)
            reconstructed_grid, reference_grid, mask, model_warnings, reconstruction_time_sec, estimated_memory_bytes = _run_model_reconstruction(
                case,
                noisy_run_dir=noisy_run_dir,
                clean_run_dir=clean_run_dir,
            )
            metrics, metric_warnings = compute_benchmark_metrics(
                reconstructed_grid,
                reference_grid,
                variables=case.model_input.get("variables", ["ux"]),
                mask=mask,
            )
            total_time_sec = perf_counter() - case_start
            warnings = list(model_warnings) + list(metric_warnings)
            summary_row = _build_summary_row(
                case=case,
                metrics=metrics,
                warnings=warnings,
                reconstruction_time_sec=reconstruction_time_sec,
                total_time_sec=total_time_sec,
                estimated_memory_bytes=estimated_memory_bytes,
            )
            _write_benchmark_result(case, summary_row=summary_row, metrics=metrics, warnings=warnings)

            completed_record = {
                "schema_version": SCHEMA_VERSION,
                "timestamp": _iso_now(),
                "benchmark_id": benchmark_id,
                "experiment_id": case.experiment_id,
                "status": "completed",
                "case": _case_to_payload(case),
                "paths": {
                    "raw_run_dir": raw_run_dir,
                    "clean_run_dir": clean_run_dir,
                    "noisy_run_dir": noisy_run_dir,
                    "reconstruction_dir": case.experiment_dir,
                },
                "warnings": warnings,
                "summary_row": summary_row,
            }
            _append_ledger_record(ledger_path, completed_record)
            latest_status[case.experiment_id] = completed_record
            completed_count += 1
        except Exception as exc:
            failed_record = {
                "schema_version": SCHEMA_VERSION,
                "timestamp": _iso_now(),
                "benchmark_id": benchmark_id,
                "experiment_id": case.experiment_id,
                "status": "failed",
                "case": _case_to_payload(case),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "total_time_sec": perf_counter() - case_start,
            }
            _append_ledger_record(ledger_path, failed_record)
            latest_status[case.experiment_id] = failed_record
            failed_count += 1

    final_entries = _load_ledger_entries(ledger_path)
    summary_rows = _collect_completed_rows(final_entries)
    _export_summary_rows(
        summary_rows,
        csv_path=summary_csv_path,
        parquet_path=summary_parquet_path,
    )
    return BenchmarkRunSummary(
        benchmark_id=benchmark_id,
        num_cases=len(cases),
        completed=completed_count,
        failed=failed_count,
        skipped=skipped_count,
        metrics_root=metrics_root,
        ledger_path=ledger_path,
        summary_csv_path=summary_csv_path,
        summary_parquet_path=summary_parquet_path,
    )


def run_benchmark_from_config(config_path: Path | str) -> BenchmarkRunSummary:
    config = load_benchmark_config(config_path)
    return run_benchmark(config, config_path=config_path)


__all__ = [
    "BenchmarkRunSummary",
    "ExpandedBenchmarkCase",
    "compute_benchmark_metrics",
    "expand_benchmark_cases",
    "load_benchmark_config",
    "run_benchmark",
    "run_benchmark_from_config",
]
