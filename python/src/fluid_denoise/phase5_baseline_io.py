from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from fluid_denoise.phase4_model_data import align_model_runs, stack_run_variables
from fluid_denoise.phase5_baseline_wrappers import (
    BaselineModel,
    PreparedBaselineInput,
    create_baseline_model,
    materialize_reconstruction_grid,
    prepare_baseline_input,
)


SCHEMA_VERSION = "phase5.baseline_reconstruction.v1"


@dataclass(frozen=True)
class BaselineReconstructionSummary:
    run_dir: Path
    reconstruction_id: str
    model_name: str
    source_experiment_id: str
    variables: tuple[str, ...]
    num_snapshots: int


def _sanitize_token(value: str) -> str:
    sanitized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return sanitized or "unknown"


def load_baseline_config(config_path: Path | str) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Baseline config must be a JSON object")
    return payload


def build_reconstruction_id(
    prepared_input: PreparedBaselineInput,
    model: BaselineModel,
) -> str:
    variable_token = "-".join(prepared_input.variables)
    digest = hashlib.sha1(
        (
            f"{prepared_input.run.experiment_id}|{model.get_name()}|"
            f"{prepared_input.input_kind}|{variable_token}"
        ).encode("utf-8")
    ).hexdigest()[:10]
    return (
        f"{_sanitize_token(prepared_input.run.experiment_id)[:48]}__"
        f"{_sanitize_token(model.get_name())}__"
        f"{_sanitize_token(prepared_input.input_kind)}__"
        f"vars-{_sanitize_token(variable_token)[:24]}__"
        f"h-{digest}"
    )


def _build_metadata(
    *,
    reconstruction_id: str,
    prepared_input: PreparedBaselineInput,
    model: BaselineModel,
    config_payload: Mapping[str, Any],
    warnings: list[str],
    reconstructed_grid: np.ndarray,
    reference_grid: np.ndarray | None,
) -> dict[str, Any]:
    source = prepared_input.run
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "reconstruction_id": reconstruction_id,
        "model": {
            "name": model.get_name(),
            "params": model.get_params(),
            "fit_metadata": model.get_fit_metadata(),
        },
        "input": {
            "kind": prepared_input.input_kind,
            "variables": list(prepared_input.variables),
            "num_snapshots": int(prepared_input.steps.shape[0]),
            "steps": [int(step) for step in prepared_input.steps.tolist()],
        },
        "source": {
            "run_dir": str(source.run_dir.resolve()),
            "experiment_id": source.experiment_id,
            "dataset_kind": source.dataset_kind,
            "schema_version": source.schema_version,
        },
        "output": {
            "reconstructed_shape": list(reconstructed_grid.shape),
            "reference_available": reference_grid is not None,
        },
        "warnings": warnings,
        "traceability": {
            "config": dict(config_payload),
            "source_metadata": source.metadata,
        },
    }
    if reference_grid is not None:
        metadata["reference"] = {
            "shape": list(reference_grid.shape),
        }
    return metadata


def save_baseline_reconstruction(
    output_dir: Path | str,
    *,
    prepared_input: PreparedBaselineInput,
    model: BaselineModel,
    reconstructed_grid: np.ndarray,
    config_payload: Mapping[str, Any],
    reference_grid: np.ndarray | None = None,
) -> Path:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings = list(model.get_warnings())
    metadata = _build_metadata(
        reconstruction_id=output_dir.name,
        prepared_input=prepared_input,
        model=model,
        config_payload=config_payload,
        warnings=warnings,
        reconstructed_grid=reconstructed_grid,
        reference_grid=reference_grid,
    )
    arrays = {
        "steps": np.asarray(prepared_input.steps, dtype=np.int32),
        "mask": np.asarray(prepared_input.run.mask, dtype=np.uint8),
        "observed": np.asarray(prepared_input.observed_grid, dtype=np.float64),
        "reconstructed": np.asarray(reconstructed_grid, dtype=np.float64),
    }
    if reference_grid is not None:
        arrays["reference"] = np.asarray(reference_grid, dtype=np.float64)
    np.savez_compressed(output_dir / "reconstruction.npz", **arrays)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_dir


def run_baseline_pipeline(
    config_payload: Mapping[str, Any],
    *,
    config_path: Path | str | None = None,
) -> Path:
    source_config = dict(config_payload.get("source", {}))
    model_config = dict(config_payload.get("model", {}))
    input_config = dict(config_payload.get("input", {}))
    output_config = dict(config_payload.get("output", {}))

    run_dir = source_config.get("run_dir")
    if run_dir is None:
        raise ValueError("Baseline config requires source.run_dir")

    model_name = str(model_config.get("name", "")).strip().lower()
    if not model_name:
        raise ValueError("Baseline config requires model.name")
    model = create_baseline_model(model_name, params=dict(model_config.get("params", {})))

    prepared_input = prepare_baseline_input(
        run_dir,
        input_kind=str(input_config.get("kind", "space_time_matrix")),
        variables=input_config.get("variables", ["ux"]),
        steps=input_config.get("steps"),
        snapshot_indices=input_config.get("snapshot_indices"),
        mask_policy=input_config.get("mask_policy"),
        obstacle_fill_value=float(input_config.get("obstacle_fill_value", 0.0)),
        include_mask_channel=bool(input_config.get("include_mask_channel", False)),
        normalization=input_config.get("normalization"),
    )

    model.fit(prepared_input)
    reconstructed_data = model.reconstruct()
    reconstructed_grid = materialize_reconstruction_grid(prepared_input, reconstructed_data)

    obstacle = prepared_input.run.mask == 1
    reconstructed_grid_flat = reconstructed_grid.reshape(
        reconstructed_grid.shape[0],
        reconstructed_grid.shape[1],
        -1,
    )
    observed_grid_flat = prepared_input.observed_grid.reshape(
        prepared_input.observed_grid.shape[0],
        prepared_input.observed_grid.shape[1],
        -1,
    )
    obstacle_flat = obstacle.reshape(-1)
    reconstructed_grid_flat[:, :, obstacle_flat] = observed_grid_flat[:, :, obstacle_flat]
    reconstructed_grid = reconstructed_grid_flat.reshape(reconstructed_grid.shape)

    reference_grid = None
    reference_run_dir = source_config.get("reference_run_dir")
    if reference_run_dir is not None:
        pair = align_model_runs(run_dir, reference_run_dir)
        reference_stacked, _ = stack_run_variables(
            pair.reference,
            variables=prepared_input.variables,
            steps=prepared_input.steps.tolist(),
        )
        reference_grid = np.asarray(reference_stacked, dtype=np.float64)

    output_root = Path(output_config.get("root", "data/processed/baselines")).resolve()
    reconstruction_id = str(output_config.get("reconstruction_id", "")).strip()
    if not reconstruction_id:
        reconstruction_id = build_reconstruction_id(prepared_input, model)
    output_dir = output_root / reconstruction_id

    overwrite = bool(output_config.get("overwrite", False))
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Baseline reconstruction already exists: {output_dir}. Use overwrite=true to replace it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    effective_config = dict(config_payload)
    if config_path is not None:
        effective_config["config_path"] = str(Path(config_path).resolve())
    return save_baseline_reconstruction(
        output_dir,
        prepared_input=prepared_input,
        model=model,
        reconstructed_grid=reconstructed_grid,
        config_payload=effective_config,
        reference_grid=reference_grid,
    )


def run_baseline_from_config(config_path: Path | str) -> Path:
    config = load_baseline_config(config_path)
    return run_baseline_pipeline(config, config_path=config_path)


def summarize_baseline_reconstruction(run_dir: Path | str) -> BaselineReconstructionSummary:
    run_dir = Path(run_dir).resolve()
    metadata_path = run_dir / "metadata.json"
    reconstruction_path = run_dir / "reconstruction.npz"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
    if not reconstruction_path.exists():
        raise FileNotFoundError(f"Missing reconstruction file: {reconstruction_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(reconstruction_path) as payload:
        observed = np.asarray(payload["observed"])
    return BaselineReconstructionSummary(
        run_dir=run_dir,
        reconstruction_id=str(metadata["reconstruction_id"]),
        model_name=str(metadata["model"]["name"]),
        source_experiment_id=str(metadata["source"]["experiment_id"]),
        variables=tuple(str(value) for value in metadata["input"]["variables"]),
        num_snapshots=int(observed.shape[0]),
    )
