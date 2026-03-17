from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fluid_denoise.phase6_benchmark import compute_benchmark_metrics


SCHEMA_VERSION = "phase7.visual_results.v1"
REPO_ROOT = Path(__file__).resolve().parents[3]
FLOAT_COLUMNS = {
    "rmse",
    "mae",
    "relative_l2_error",
    "psnr",
    "vorticity_rmse",
    "kinetic_energy_relative_l2",
    "divergence_residual_rms",
    "reconstruction_time_sec",
    "total_time_sec",
}
INT_COLUMNS = {
    "nx",
    "ny",
    "obstacle_cx",
    "obstacle_cy",
    "obstacle_r",
    "estimated_memory_bytes",
}
JSON_COLUMNS = {
    "noise_specs_json": "noise_specs",
    "model_params_json": "model_params",
    "model_input_json": "model_input",
    "warnings": "warnings_list",
}
LOWER_IS_BETTER_METRICS = {
    "rmse",
    "mae",
    "relative_l2_error",
    "vorticity_rmse",
    "kinetic_energy_relative_l2",
    "divergence_residual_rms",
    "reconstruction_time_sec",
    "total_time_sec",
}
SUPPORTED_SERIES_METRICS = {"rmse", "mae", "relative_l2_error", "psnr"}
DEFAULT_EXPERIMENT_TABLE_COLUMNS = (
    "experiment_id",
    "resolution_label",
    "nx",
    "ny",
    "noise_label",
    "model_variant",
    "rmse",
    "mae",
    "relative_l2_error",
    "vorticity_rmse",
    "reconstruction_time_sec",
)
DEFAULT_AGGREGATE_TABLE_COLUMNS = (
    "model_variant",
    "num_cases",
    "mean_rmse",
    "mean_mae",
    "mean_relative_l2_error",
    "mean_vorticity_rmse",
    "mean_reconstruction_time_sec",
)


@dataclass(frozen=True)
class ReconstructionBundle:
    run_dir: Path
    experiment_id: str
    variables: tuple[str, ...]
    steps: np.ndarray
    mask: np.ndarray
    observed: np.ndarray
    reconstructed: np.ndarray
    reference: np.ndarray | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class VisualResultsSummary:
    benchmark_id: str
    output_root: Path
    exploratory_dir: Path
    final_dir: Path
    thesis_ready_dir: Path
    catalog_json_path: Path
    catalog_csv_path: Path
    catalog_markdown_path: Path
    num_figures: int
    num_tables: int
    num_selected_experiments: int


def _resolve_path(base_dir: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _sanitize_token(value: str) -> str:
    import re

    sanitized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return sanitized or "unknown"


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_ready(item) for item in value]
    return value


def _load_json_or_empty(raw_value: str, *, fallback: Any) -> Any:
    if not raw_value:
        return fallback
    return json.loads(raw_value)


def _parse_summary_value(key: str, raw_value: str) -> Any:
    if key in INT_COLUMNS:
        return None if raw_value == "" else int(float(raw_value))
    if key in FLOAT_COLUMNS:
        return None if raw_value == "" else float(raw_value)
    return raw_value


def _format_param_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _format_model_variant(model_name: str, params: Mapping[str, Any]) -> str:
    if not params:
        return model_name
    parts = [f"{key}={_format_param_value(params[key])}" for key in sorted(params)]
    return f"{model_name}({', '.join(parts)})"


def _load_summary_rows(summary_csv_path: Path) -> list[dict[str, Any]]:
    if not summary_csv_path.exists():
        raise FileNotFoundError(f"Benchmark summary CSV not found: {summary_csv_path}")

    rows: list[dict[str, Any]] = []
    with summary_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            row = {key: _parse_summary_value(key, value) for key, value in raw_row.items()}
            for column, target_key in JSON_COLUMNS.items():
                if column in {"noise_specs_json", "warnings"}:
                    fallback: Any = []
                else:
                    fallback = {}
                row[target_key] = _load_json_or_empty(
                    str(raw_row.get(column, "")),
                    fallback=fallback,
                )
            row["noise_specs"] = list(row["noise_specs"])
            row["model_variant"] = _format_model_variant(
                str(row["model_name"]),
                dict(row["model_params"]),
            )
            row["resolution_display"] = f"{row['resolution_label']} ({row['nx']}x{row['ny']})"
            rows.append(row)

    if not rows:
        raise ValueError(f"Benchmark summary CSV is empty: {summary_csv_path}")
    return rows


def load_visual_results_config(config_path: Path | str) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    if config_path.suffix.lower() != ".json":
        raise ValueError(
            "FASE 7 currently supports JSON configs only. Use a .json file for the visual results config."
        )
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Visual results config must be a JSON object")
    return payload


def _validate_visual_results_config(config_payload: Mapping[str, Any]) -> None:
    if "source" not in config_payload or "output" not in config_payload:
        raise ValueError("Visual results config requires 'source' and 'output' sections")
    source = config_payload["source"]
    output = config_payload["output"]
    if not isinstance(source, Mapping):
        raise ValueError("source section must be an object")
    if not isinstance(output, Mapping):
        raise ValueError("output section must be an object")
    if not str(source.get("benchmark_id", "")).strip():
        raise ValueError("source.benchmark_id is required")


def _resolve_source_paths(
    config_payload: Mapping[str, Any],
    *,
    config_base_dir: Path,
) -> tuple[str, Path, Path, Path]:
    source = config_payload["source"]
    benchmark_id = str(source["benchmark_id"]).strip()
    metrics_root = _resolve_path(config_base_dir, str(source.get("metrics_root", "results/metrics")))
    tables_root = _resolve_path(config_base_dir, str(source.get("tables_root", "results/tables")))
    benchmark_metrics_root = metrics_root / benchmark_id
    benchmark_tables_root = tables_root / benchmark_id
    summary_csv_path = benchmark_tables_root / "summary.csv"
    return benchmark_id, benchmark_metrics_root, benchmark_tables_root, summary_csv_path


def _validate_phase6_sources(
    *,
    benchmark_id: str,
    benchmark_metrics_root: Path,
    benchmark_tables_root: Path,
    summary_csv_path: Path,
) -> None:
    if not benchmark_metrics_root.exists():
        raise FileNotFoundError(
            "FASE 7 cannot run because the benchmark metrics folder does not exist: "
            f"{benchmark_metrics_root}"
        )
    experiments_root = benchmark_metrics_root / "experiments"
    if not experiments_root.exists():
        raise FileNotFoundError(
            "FASE 7 cannot run because FASE 6 experiments are missing: "
            f"{experiments_root}"
        )
    if not any(path.is_dir() for path in experiments_root.iterdir()):
        raise RuntimeError(
            "FASE 7 cannot run because FASE 6 experiments folder is empty: "
            f"{experiments_root}"
        )

    if not benchmark_tables_root.exists():
        raise FileNotFoundError(
            "FASE 7 cannot run because the benchmark tables folder does not exist: "
            f"{benchmark_tables_root}"
        )
    if not summary_csv_path.exists():
        if list(benchmark_tables_root.glob("summary.*")):
            raise ValueError(
                "FASE 7 expects summary.csv but found a different extension in "
                f"{benchmark_tables_root}. Configure FASE 6 export_formats to include CSV."
            )
        raise FileNotFoundError(f"Benchmark summary CSV not found: {summary_csv_path}")


def _validate_summary_row_artifacts(
    rows: Sequence[Mapping[str, Any]],
    *,
    benchmark_metrics_root: Path,
) -> None:
    experiments_root = benchmark_metrics_root / "experiments"
    for row in rows:
        experiment_id = str(row.get("experiment_id", "")).strip()
        reconstruction_dir_raw = str(row.get("reconstruction_dir", "")).strip()
        if not experiment_id:
            raise ValueError("Invalid summary row: missing experiment_id")
        if not reconstruction_dir_raw:
            raise ValueError(f"Invalid summary row for experiment '{experiment_id}': missing reconstruction_dir")

        reconstruction_dir = Path(reconstruction_dir_raw).resolve()
        expected_dir = (experiments_root / experiment_id).resolve()
        if reconstruction_dir != expected_dir:
            raise ValueError(
                "Summary row reconstruction_dir does not match the expected FASE 6 path for "
                f"experiment '{experiment_id}'. Expected {expected_dir}, found {reconstruction_dir}."
            )

        npz_path = reconstruction_dir / "reconstruction.npz"
        metadata_path = reconstruction_dir / "metadata.json"
        if npz_path.suffix.lower() != ".npz":
            raise ValueError(f"Unexpected reconstruction file extension: {npz_path}")
        if metadata_path.suffix.lower() != ".json":
            raise ValueError(f"Unexpected metadata file extension: {metadata_path}")
        if not npz_path.exists() or not metadata_path.exists():
            missing = [str(path) for path in (npz_path, metadata_path) if not path.exists()]
            raise FileNotFoundError(
                "FASE 7 found summary rows that reference missing FASE 6 outputs: "
                + ", ".join(missing)
            )

def _resolve_output_paths(
    config_payload: Mapping[str, Any],
    *,
    config_base_dir: Path,
    benchmark_id: str,
) -> dict[str, Path]:
    output = config_payload["output"]
    output_root = _resolve_path(config_base_dir, str(output.get("root", "results/visuals"))) / benchmark_id
    exploratory_dir = output_root / "exploratory"
    final_dir = output_root / "final"
    thesis_ready_dir = output_root / "thesis_ready"
    paths = {
        "output_root": output_root,
        "exploratory_dir": exploratory_dir,
        "final_dir": final_dir,
        "thesis_ready_dir": thesis_ready_dir,
        "config_snapshot_path": output_root / "visual_results_config_snapshot.json",
        "catalog_json_path": output_root / "catalog.json",
        "catalog_csv_path": output_root / "catalog.csv",
        "catalog_markdown_path": output_root / "catalog.md",
    }
    for path in (
        exploratory_dir / "comparisons",
        exploratory_dir / "error_maps",
        exploratory_dir / "error_series",
        exploratory_dir / "aggregate",
        exploratory_dir / "tables",
        final_dir / "comparisons",
        final_dir / "error_maps",
        final_dir / "error_series",
        final_dir / "aggregate",
        final_dir / "tables",
        thesis_ready_dir / "figures" / "comparisons",
        thesis_ready_dir / "figures" / "error_maps",
        thesis_ready_dir / "figures" / "error_series",
        thesis_ready_dir / "figures" / "aggregate",
        thesis_ready_dir / "tables",
    ):
        path.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    return paths


def _metric_sort_value(metric_name: str, value: float | None) -> float:
    if value is None or not math.isfinite(value):
        return math.inf if metric_name in LOWER_IS_BETTER_METRICS else -math.inf
    if metric_name in LOWER_IS_BETTER_METRICS:
        return float(value)
    return float(-value)


def _select_comparison_rows(
    rows: Sequence[Mapping[str, Any]],
    selection_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    explicit_ids = [str(value) for value in selection_config.get("comparison_experiment_ids", [])]
    if explicit_ids:
        row_map = {str(row["experiment_id"]): dict(row) for row in rows}
        selected = [row_map[experiment_id] for experiment_id in explicit_ids if experiment_id in row_map]
        if selected:
            return selected

    max_cases = max(1, int(selection_config.get("max_comparison_cases", 4)))
    rank_metric = str(selection_config.get("comparison_rank_metric", "rmse")).strip()
    sorted_rows = sorted(
        (dict(row) for row in rows),
        key=lambda row: (_metric_sort_value(rank_metric, row.get(rank_metric)), str(row["experiment_id"])),
    )
    selected: list[dict[str, Any]] = []
    seen_models: set[str] = set()
    for row in sorted_rows:
        model_variant = str(row["model_variant"])
        if model_variant in seen_models:
            continue
        selected.append(row)
        seen_models.add(model_variant)
        if len(selected) >= max_cases:
            return selected

    for row in sorted_rows:
        if any(str(existing["experiment_id"]) == str(row["experiment_id"]) for existing in selected):
            continue
        selected.append(row)
        if len(selected) >= max_cases:
            break
    return selected


def _register_artifact(
    catalog_entries: list[dict[str, Any]],
    *,
    artifact_type: str,
    role: str,
    category: str,
    benchmark_id: str,
    path: Path,
    title: str,
    experiment_ids: Sequence[str] = (),
    variable: str | None = None,
    step: int | None = None,
    metric: str | None = None,
) -> None:
    catalog_entries.append(
        {
            "artifact_id": path.stem,
            "artifact_type": artifact_type,
            "role": role,
            "category": category,
            "benchmark_id": benchmark_id,
            "path": str(path.resolve()),
            "title": title,
            "experiment_ids": list(experiment_ids),
            "variable": variable,
            "step": step,
            "metric": metric,
        }
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_catalog(
    output_paths: Mapping[str, Path],
    *,
    catalog_entries: Sequence[Mapping[str, Any]],
) -> None:
    json_path = output_paths["catalog_json_path"]
    csv_path = output_paths["catalog_csv_path"]
    markdown_path = output_paths["catalog_markdown_path"]

    _write_json(
        json_path,
        {
            "schema_version": SCHEMA_VERSION,
            "artifacts": list(catalog_entries),
        },
    )

    csv_columns = (
        "artifact_id",
        "artifact_type",
        "role",
        "category",
        "benchmark_id",
        "path",
        "title",
        "experiment_ids",
        "variable",
        "step",
        "metric",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_columns))
        writer.writeheader()
        for raw_entry in catalog_entries:
            entry = dict(raw_entry)
            entry["experiment_ids"] = ",".join(str(value) for value in entry.get("experiment_ids", []))
            writer.writerow(entry)

    lines = [
        "# Figure Catalog",
        "",
        "| artifact_id | role | category | experiments | path |",
        "| --- | --- | --- | --- | --- |",
    ]
    for entry in catalog_entries:
        experiment_ids = ", ".join(str(value) for value in entry.get("experiment_ids", [])) or "-"
        lines.append(
            f"| {entry['artifact_id']} | {entry['role']} | {entry['category']} | {experiment_ids} | {entry['path']} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_reconstruction_bundle(reconstruction_dir: Path | str) -> ReconstructionBundle:
    run_dir = Path(reconstruction_dir).resolve()
    metadata_path = run_dir / "metadata.json"
    reconstruction_path = run_dir / "reconstruction.npz"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing reconstruction metadata file: {metadata_path}")
    if not reconstruction_path.exists():
        raise FileNotFoundError(f"Missing reconstruction array file: {reconstruction_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    variables = tuple(str(value).strip().lower() for value in metadata["input"]["variables"])
    with np.load(reconstruction_path) as payload:
        reference = np.asarray(payload["reference"], dtype=np.float64) if "reference" in payload.files else None
        bundle = ReconstructionBundle(
            run_dir=run_dir,
            experiment_id=str(metadata.get("reconstruction_id", run_dir.name)),
            variables=variables,
            steps=np.asarray(payload["steps"], dtype=np.int32),
            mask=np.asarray(payload["mask"], dtype=np.uint8),
            observed=np.asarray(payload["observed"], dtype=np.float64),
            reconstructed=np.asarray(payload["reconstructed"], dtype=np.float64),
            reference=reference,
            metadata=metadata,
        )
    if bundle.observed.shape != bundle.reconstructed.shape:
        raise ValueError(f"Observed and reconstructed arrays do not match in {run_dir}")
    if bundle.reference is not None and bundle.reference.shape != bundle.reconstructed.shape:
        raise ValueError(f"Reference and reconstructed arrays do not match in {run_dir}")
    if bundle.reconstructed.ndim != 4:
        raise ValueError(f"Expected reconstruction tensors with shape (T, C, ny, nx) in {run_dir}")
    return bundle


def _resolve_variable(bundle: ReconstructionBundle, requested_variable: str) -> tuple[str, int]:
    normalized = requested_variable.strip().lower()
    if normalized in bundle.variables:
        return normalized, int(bundle.variables.index(normalized))
    return bundle.variables[0], 0


def _select_step_index(
    steps: np.ndarray,
    *,
    requested_step: int | None,
    step_policy: str,
) -> int:
    if requested_step is not None:
        indices = np.flatnonzero(np.asarray(steps) == int(requested_step))
        if indices.size == 0:
            raise ValueError(f"Requested step {requested_step} is not present in the reconstruction bundle")
        return int(indices[0])

    normalized_policy = step_policy.strip().lower()
    if normalized_policy == "first":
        return 0
    if normalized_policy == "last":
        return int(len(steps) - 1)
    if normalized_policy == "middle":
        return int(len(steps) // 2)
    raise ValueError(f"Unsupported step_policy: {step_policy}")


def _fluid_values(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.asarray(np.asarray(array, dtype=np.float64)[..., mask == 0], dtype=np.float64)


def _masked_grid(array_2d: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.where(mask == 1, np.nan, np.asarray(array_2d, dtype=np.float64))


def _limits_from_arrays(
    arrays: Sequence[np.ndarray],
    *,
    mask: np.ndarray,
    symmetric: bool,
    nonnegative: bool = False,
) -> tuple[float, float]:
    values = [np.ravel(_fluid_values(array, mask)) for array in arrays]
    merged = np.concatenate(values) if values else np.asarray([], dtype=np.float64)
    if merged.size == 0:
        return (0.0, 1.0)
    if nonnegative:
        vmax = float(np.max(merged))
        return (0.0, max(vmax, 1e-12))
    if symmetric:
        bound = float(np.max(np.abs(merged)))
        return (-max(bound, 1e-12), max(bound, 1e-12))
    minimum = float(np.min(merged))
    maximum = float(np.max(merged))
    if math.isclose(minimum, maximum):
        delta = max(abs(minimum), 1.0) * 1e-6
        return (minimum - delta, maximum + delta)
    return (minimum, maximum)


def _choose_cmap(variable: str, *, error_map: bool = False) -> str:
    if error_map:
        return "magma"
    if variable == "speed":
        return "viridis"
    return "coolwarm"


def _build_base_filename(
    prefix: str,
    benchmark_id: str,
    *,
    experiment_id: str | None = None,
    variable: str | None = None,
    step: int | None = None,
    metric: str | None = None,
) -> str:
    tokens = [prefix, _sanitize_token(benchmark_id)]
    if experiment_id is not None:
        tokens.append(_sanitize_token(experiment_id))
    if variable is not None:
        tokens.append(_sanitize_token(variable))
    if step is not None:
        tokens.append(f"t{int(step):06d}")
    if metric is not None:
        tokens.append(_sanitize_token(metric))
    return "__".join(tokens)


def _role_formats(config_payload: Mapping[str, Any], role: str) -> tuple[str, ...]:
    output = config_payload["output"]
    if role == "exploratory":
        raw_formats = output.get("exploratory_formats", ["png"])
    else:
        raw_formats = output.get("final_formats", ["png", "pdf"])
    if isinstance(raw_formats, Sequence) and not isinstance(raw_formats, (str, bytes)):
        return tuple(str(value).strip().lower() for value in raw_formats)
    return (str(raw_formats).strip().lower(),)


def _style_for_role(role: str) -> dict[str, Any]:
    if role == "final":
        return {"dpi": 180, "comparison_figsize": (12.0, 3.8), "aggregate_figsize": (10.0, 5.0)}
    return {"dpi": 140, "comparison_figsize": (10.5, 3.4), "aggregate_figsize": (9.5, 4.6)}


def _save_figure(
    fig: plt.Figure,
    base_path: Path,
    *,
    formats: Sequence[str],
) -> list[Path]:
    written_paths: list[Path] = []
    for fmt in formats:
        output_path = base_path.with_suffix(f".{fmt}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=fig.dpi, bbox_inches="tight")
        written_paths.append(output_path)
    plt.close(fig)
    return written_paths


def _copy_to_thesis_ready(source_path: Path, destination_path: Path) -> Path:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)
    return destination_path


def _per_step_metric_series(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    variables: Sequence[str],
    mask: np.ndarray,
    metric_name: str,
) -> np.ndarray:
    normalized_metric = metric_name.strip().lower()
    if normalized_metric not in SUPPORTED_SERIES_METRICS:
        raise ValueError(f"Unsupported time_series_metric: {metric_name}")

    values: list[float] = []
    for step_index in range(candidate.shape[0]):
        metrics, _ = compute_benchmark_metrics(
            candidate[step_index : step_index + 1],
            reference[step_index : step_index + 1],
            variables=variables,
            mask=mask,
        )
        metric_value = metrics[normalized_metric]
        values.append(np.nan if metric_value is None else float(metric_value))
    return np.asarray(values, dtype=np.float64)


def _group_mean(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_keys: Sequence[str],
    value_key: str,
) -> dict[tuple[Any, ...], float]:
    grouped: dict[tuple[Any, ...], list[float]] = {}
    for row in rows:
        value = row.get(value_key)
        if value is None or not math.isfinite(float(value)):
            continue
        grouped.setdefault(tuple(row[key] for key in group_keys), []).append(float(value))
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def _plot_triptych(
    *,
    bundle: ReconstructionBundle,
    row: Mapping[str, Any],
    benchmark_id: str,
    variable: str,
    step_index: int,
    role: str,
    output_dir: Path,
    formats: Sequence[str],
) -> list[Path]:
    if bundle.reference is None:
        return []

    _, channel_index = _resolve_variable(bundle, variable)
    step = int(bundle.steps[step_index])
    reference = bundle.reference[step_index, channel_index]
    observed = bundle.observed[step_index, channel_index]
    reconstructed = bundle.reconstructed[step_index, channel_index]
    vmin, vmax = _limits_from_arrays(
        [reference, observed, reconstructed],
        mask=bundle.mask,
        symmetric=variable in {"ux", "uy", "vorticity"},
    )

    style = _style_for_role(role)
    fig, axes = plt.subplots(1, 3, figsize=style["comparison_figsize"], dpi=style["dpi"], constrained_layout=True)
    images = []
    for axis, title, field in zip(
        axes,
        ("Clean", "Noisy", "Reconstructed"),
        (reference, observed, reconstructed),
    ):
        image = axis.imshow(
            _masked_grid(field, bundle.mask),
            origin="lower",
            cmap=_choose_cmap(variable),
            vmin=vmin,
            vmax=vmax,
        )
        images.append(image)
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
    fig.colorbar(images[-1], ax=axes.ravel().tolist(), shrink=0.88, label=variable)
    fig.suptitle(f"{variable} comparison | {row['experiment_id']} | step={step}")
    base_path = output_dir / _build_base_filename(
        "comparison",
        benchmark_id,
        experiment_id=str(row["experiment_id"]),
        variable=variable,
        step=step,
    )
    return _save_figure(fig, base_path, formats=formats)


def _plot_error_maps(
    *,
    bundle: ReconstructionBundle,
    row: Mapping[str, Any],
    benchmark_id: str,
    variable: str,
    step_index: int,
    role: str,
    output_dir: Path,
    formats: Sequence[str],
) -> list[Path]:
    if bundle.reference is None:
        return []

    _, channel_index = _resolve_variable(bundle, variable)
    step = int(bundle.steps[step_index])
    reference = bundle.reference[step_index, channel_index]
    noisy_error = np.abs(bundle.observed[step_index, channel_index] - reference)
    reconstructed_error = np.abs(bundle.reconstructed[step_index, channel_index] - reference)
    vmin, vmax = _limits_from_arrays(
        [noisy_error, reconstructed_error],
        mask=bundle.mask,
        symmetric=False,
        nonnegative=True,
    )

    style = _style_for_role(role)
    fig, axes = plt.subplots(1, 2, figsize=style["comparison_figsize"], dpi=style["dpi"], constrained_layout=True)
    images = []
    for axis, title, field in zip(
        axes,
        ("|Noisy - Clean|", "|Reconstructed - Clean|"),
        (noisy_error, reconstructed_error),
    ):
        image = axis.imshow(
            _masked_grid(field, bundle.mask),
            origin="lower",
            cmap=_choose_cmap(variable, error_map=True),
            vmin=vmin,
            vmax=vmax,
        )
        images.append(image)
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
    fig.colorbar(images[-1], ax=axes.ravel().tolist(), shrink=0.88, label=f"|error| {variable}")
    fig.suptitle(f"Error maps | {row['experiment_id']} | {variable} | step={step}")
    base_path = output_dir / _build_base_filename(
        "error-map",
        benchmark_id,
        experiment_id=str(row["experiment_id"]),
        variable=variable,
        step=step,
    )
    return _save_figure(fig, base_path, formats=formats)


def _plot_error_series(
    *,
    bundle: ReconstructionBundle,
    row: Mapping[str, Any],
    benchmark_id: str,
    role: str,
    output_dir: Path,
    formats: Sequence[str],
    metric_name: str,
) -> list[Path]:
    if bundle.reference is None:
        return []

    noisy_series = _per_step_metric_series(
        bundle.observed,
        bundle.reference,
        variables=bundle.variables,
        mask=bundle.mask,
        metric_name=metric_name,
    )
    reconstructed_series = _per_step_metric_series(
        bundle.reconstructed,
        bundle.reference,
        variables=bundle.variables,
        mask=bundle.mask,
        metric_name=metric_name,
    )

    style = _style_for_role(role)
    fig, axis = plt.subplots(1, 1, figsize=style["aggregate_figsize"], dpi=style["dpi"], constrained_layout=True)
    axis.plot(bundle.steps, noisy_series, marker="o", linewidth=1.8, label="Noisy")
    axis.plot(bundle.steps, reconstructed_series, marker="s", linewidth=1.8, label="Reconstructed")
    axis.set_xlabel("time step")
    axis.set_ylabel(metric_name)
    axis.set_title(f"Temporal error series | {row['experiment_id']}")
    axis.grid(True, alpha=0.25)
    merged = np.concatenate([noisy_series, reconstructed_series])
    if merged.size and np.all(np.isnan(merged) | (merged >= 0.0)):
        axis.set_ylim(bottom=0.0)
    axis.legend()
    base_path = output_dir / _build_base_filename(
        "error-series",
        benchmark_id,
        experiment_id=str(row["experiment_id"]),
        metric=metric_name,
    )
    return _save_figure(fig, base_path, formats=formats)


def _plot_metric_by_resolution(
    *,
    rows: Sequence[Mapping[str, Any]],
    benchmark_id: str,
    role: str,
    output_dir: Path,
    formats: Sequence[str],
    metric_name: str,
) -> list[Path]:
    resolution_rows = {str(row["resolution_label"]): (int(row["nx"]), int(row["ny"])) for row in rows}
    resolution_order = sorted(
        resolution_rows,
        key=lambda label: (resolution_rows[label][0] * resolution_rows[label][1], label),
    )
    noise_labels = sorted({str(row["noise_label"]) for row in rows})
    model_variants = sorted({str(row["model_variant"]) for row in rows})
    grouped = _group_mean(
        rows,
        group_keys=("noise_label", "model_variant", "resolution_label"),
        value_key=metric_name,
    )

    style = _style_for_role(role)
    fig, axes = plt.subplots(
        len(noise_labels),
        1,
        figsize=(style["aggregate_figsize"][0], max(3.0, 2.8 * len(noise_labels))),
        dpi=style["dpi"],
        squeeze=False,
        constrained_layout=True,
    )
    x_positions = np.arange(len(resolution_order), dtype=np.float64)
    for axis, noise_label in zip(axes[:, 0], noise_labels):
        for model_variant in model_variants:
            values = [
                grouped.get((noise_label, model_variant, resolution_label), np.nan)
                for resolution_label in resolution_order
            ]
            axis.plot(x_positions, values, marker="o", linewidth=1.8, label=model_variant)
        axis.set_title(f"{metric_name} by resolution | noise={noise_label}")
        axis.set_ylabel(metric_name)
        axis.grid(True, alpha=0.25)
        if metric_name in LOWER_IS_BETTER_METRICS:
            axis.set_ylim(bottom=0.0)
        axis.set_xticks(x_positions)
        axis.set_xticklabels([f"{label}\n{resolution_rows[label][0]}x{resolution_rows[label][1]}" for label in resolution_order])
    axes[-1, 0].set_xlabel("resolution")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncols=max(1, min(3, len(labels))))
    base_path = output_dir / _build_base_filename("aggregate-resolution", benchmark_id, metric=metric_name)
    return _save_figure(fig, base_path, formats=formats)


def _plot_metric_by_noise_case(
    *,
    rows: Sequence[Mapping[str, Any]],
    benchmark_id: str,
    role: str,
    output_dir: Path,
    formats: Sequence[str],
    metric_name: str,
) -> list[Path]:
    resolution_labels = sorted({str(row["resolution_label"]) for row in rows})
    noise_labels = sorted({str(row["noise_label"]) for row in rows})
    model_variants = sorted({str(row["model_variant"]) for row in rows})
    grouped = _group_mean(
        rows,
        group_keys=("resolution_label", "model_variant", "noise_label"),
        value_key=metric_name,
    )

    style = _style_for_role(role)
    fig, axes = plt.subplots(
        len(resolution_labels),
        1,
        figsize=(style["aggregate_figsize"][0], max(3.0, 2.8 * len(resolution_labels))),
        dpi=style["dpi"],
        squeeze=False,
        constrained_layout=True,
    )
    x_positions = np.arange(len(noise_labels), dtype=np.float64)
    for axis, resolution_label in zip(axes[:, 0], resolution_labels):
        for model_variant in model_variants:
            values = [grouped.get((resolution_label, model_variant, noise_label), np.nan) for noise_label in noise_labels]
            axis.plot(x_positions, values, marker="o", linewidth=1.8, label=model_variant)
        axis.set_title(f"{metric_name} by noise case | resolution={resolution_label}")
        axis.set_ylabel(metric_name)
        axis.grid(True, alpha=0.25)
        if metric_name in LOWER_IS_BETTER_METRICS:
            axis.set_ylim(bottom=0.0)
        axis.set_xticks(x_positions)
        axis.set_xticklabels(noise_labels, rotation=20, ha="right")
    axes[-1, 0].set_xlabel("noise case")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncols=max(1, min(3, len(labels))))
    base_path = output_dir / _build_base_filename("aggregate-noise", benchmark_id, metric=metric_name)
    return _save_figure(fig, base_path, formats=formats)


def _plot_runtime_bars(
    *,
    rows: Sequence[Mapping[str, Any]],
    benchmark_id: str,
    role: str,
    output_dir: Path,
    formats: Sequence[str],
    metric_name: str,
) -> list[Path]:
    grouped = _group_mean(rows, group_keys=("model_variant",), value_key=metric_name)
    if not grouped:
        return []
    ordered_items = sorted(grouped.items(), key=lambda item: item[1])
    labels = [item[0][0] for item in ordered_items]
    values = [item[1] for item in ordered_items]

    style = _style_for_role(role)
    height = max(3.0, 0.5 * len(labels) + 1.4)
    fig, axis = plt.subplots(1, 1, figsize=(style["aggregate_figsize"][0], height), dpi=style["dpi"], constrained_layout=True)
    y_positions = np.arange(len(labels), dtype=np.float64)
    axis.barh(y_positions, values)
    axis.set_yticks(y_positions)
    axis.set_yticklabels(labels)
    axis.set_xlabel(metric_name)
    axis.set_title(f"Compute time bars | mean {metric_name} by model")
    axis.grid(True, axis="x", alpha=0.25)
    axis.set_xlim(left=0.0)
    base_path = output_dir / _build_base_filename("aggregate-runtime", benchmark_id, metric=metric_name)
    return _save_figure(fig, base_path, formats=formats)


def _plot_performance_heatmaps(
    *,
    rows: Sequence[Mapping[str, Any]],
    benchmark_id: str,
    role: str,
    output_dir: Path,
    formats: Sequence[str],
    metric_name: str,
) -> list[Path]:
    resolution_labels = sorted({str(row["resolution_label"]) for row in rows})
    noise_labels = sorted({str(row["noise_label"]) for row in rows})
    model_variants = sorted({str(row["model_variant"]) for row in rows})
    grouped = _group_mean(
        rows,
        group_keys=("resolution_label", "noise_label", "model_variant"),
        value_key=metric_name,
    )

    style = _style_for_role(role)
    fig, axes = plt.subplots(
        len(resolution_labels),
        1,
        figsize=(style["aggregate_figsize"][0] + 1.0, max(3.4, 2.8 * len(resolution_labels))),
        dpi=style["dpi"],
        squeeze=False,
        constrained_layout=True,
    )
    for axis, resolution_label in zip(axes[:, 0], resolution_labels):
        matrix = np.full((len(noise_labels), len(model_variants)), np.nan, dtype=np.float64)
        for noise_index, noise_label in enumerate(noise_labels):
            for model_index, model_variant in enumerate(model_variants):
                matrix[noise_index, model_index] = grouped.get((resolution_label, noise_label, model_variant), np.nan)
        finite_values = matrix[np.isfinite(matrix)]
        vmin = float(np.min(finite_values)) if finite_values.size else 0.0
        vmax = float(np.max(finite_values)) if finite_values.size else 1.0
        if math.isclose(vmin, vmax):
            vmax = vmin + max(abs(vmin), 1.0) * 1e-6
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color="#e0e0e0")
        image = axis.imshow(matrix, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(f"{metric_name} heatmap | resolution={resolution_label}")
        axis.set_xticks(np.arange(len(model_variants)))
        axis.set_xticklabels(model_variants, rotation=20, ha="right")
        axis.set_yticks(np.arange(len(noise_labels)))
        axis.set_yticklabels(noise_labels)
        for noise_index in range(len(noise_labels)):
            for model_index in range(len(model_variants)):
                value = matrix[noise_index, model_index]
                text = "-" if not np.isfinite(value) else f"{value:.3g}"
                color = "black" if not np.isfinite(value) else "white"
                axis.text(model_index, noise_index, text, ha="center", va="center", color=color)
        fig.colorbar(image, ax=axis, shrink=0.86, label=metric_name)
    base_path = output_dir / _build_base_filename("aggregate-heatmap", benchmark_id, metric=metric_name)
    return _save_figure(fig, base_path, formats=formats)


def _stringify_table_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.6g}"
    return str(value)


def _write_csv_table(
    path: Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for raw_row in rows:
            writer.writerow({column: _stringify_table_value(raw_row.get(column)) for column in columns})
    return path


def _escape_latex(text: str) -> str:
    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _write_latex_table(
    path: Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
    caption: str,
    label: str,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    column_spec = "l" * len(columns)
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{_escape_latex(caption)}}}",
        f"\\label{{{_escape_latex(label)}}}",
        f"\\begin{{tabular}}{{{column_spec}}}",
        "\\hline",
        " & ".join(_escape_latex(column) for column in columns) + " \\\\",
        "\\hline",
    ]
    for raw_row in rows:
        values = [_escape_latex(_stringify_table_value(raw_row.get(column))) for column in columns]
        lines.append(" & ".join(values) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _build_experiment_summary_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    sort_by: str,
    ascending: bool,
) -> list[dict[str, Any]]:
    prepared = [dict(row) for row in rows]
    prepared.sort(
        key=lambda row: (
            math.inf if row.get(sort_by) is None else float(row[sort_by]),
            str(row["experiment_id"]),
        ),
        reverse=not ascending,
    )
    return prepared


def _build_model_aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["model_variant"]), []).append(row)

    aggregate_rows: list[dict[str, Any]] = []
    for model_variant, model_rows in sorted(grouped.items()):
        rmse_values = [row["rmse"] for row in model_rows if row.get("rmse") is not None]
        mae_values = [row["mae"] for row in model_rows if row.get("mae") is not None]
        rel_l2_values = [row["relative_l2_error"] for row in model_rows if row.get("relative_l2_error") is not None]
        vort_values = [row["vorticity_rmse"] for row in model_rows if row.get("vorticity_rmse") is not None]
        time_values = [row["reconstruction_time_sec"] for row in model_rows if row.get("reconstruction_time_sec") is not None]
        aggregate_rows.append(
            {
                "model_variant": model_variant,
                "num_cases": len(model_rows),
                "mean_rmse": float(np.mean(rmse_values)) if rmse_values else None,
                "mean_mae": float(np.mean(mae_values)) if mae_values else None,
                "mean_relative_l2_error": float(np.mean(rel_l2_values)) if rel_l2_values else None,
                "mean_vorticity_rmse": float(np.mean(vort_values)) if vort_values else None,
                "mean_reconstruction_time_sec": float(np.mean(time_values)) if time_values else None,
            }
        )
    return aggregate_rows


def _generate_tables(
    *,
    rows: Sequence[Mapping[str, Any]],
    benchmark_id: str,
    role: str,
    output_dir: Path,
    thesis_ready_dir: Path,
    catalog_entries: list[dict[str, Any]],
    experiment_columns: Sequence[str],
    aggregate_columns: Sequence[str],
    sort_by: str,
    ascending: bool,
) -> int:
    experiment_rows = _build_experiment_summary_rows(rows, sort_by=sort_by, ascending=ascending)
    aggregate_rows = _build_model_aggregate_rows(rows)
    specs = (
        (
            "experiment-summary",
            experiment_rows,
            experiment_columns,
            f"Benchmark experiment summary for {benchmark_id}",
            f"tab:{_sanitize_token(benchmark_id)}:experiment_summary",
        ),
        (
            "model-aggregate",
            aggregate_rows,
            aggregate_columns,
            f"Model aggregate summary for {benchmark_id}",
            f"tab:{_sanitize_token(benchmark_id)}:model_aggregate",
        ),
    )

    count = 0
    for suffix, table_rows, columns, caption, label in specs:
        csv_path = output_dir / f"{suffix}.csv"
        tex_path = output_dir / f"{suffix}.tex"
        _write_csv_table(csv_path, rows=table_rows, columns=columns)
        _write_latex_table(tex_path, rows=table_rows, columns=columns, caption=caption, label=label)
        _register_artifact(
            catalog_entries,
            artifact_type="table",
            role=role,
            category="table",
            benchmark_id=benchmark_id,
            path=csv_path,
            title=csv_path.stem,
        )
        _register_artifact(
            catalog_entries,
            artifact_type="table",
            role=role,
            category="table",
            benchmark_id=benchmark_id,
            path=tex_path,
            title=tex_path.stem,
        )
        count += 2
        if role == "final":
            thesis_csv = _copy_to_thesis_ready(csv_path, thesis_ready_dir / "tables" / csv_path.name)
            thesis_tex = _copy_to_thesis_ready(tex_path, thesis_ready_dir / "tables" / tex_path.name)
            _register_artifact(
                catalog_entries,
                artifact_type="table",
                role="thesis_ready",
                category="table",
                benchmark_id=benchmark_id,
                path=thesis_csv,
                title=thesis_csv.stem,
            )
            _register_artifact(
                catalog_entries,
                artifact_type="table",
                role="thesis_ready",
                category="table",
                benchmark_id=benchmark_id,
                path=thesis_tex,
                title=thesis_tex.stem,
            )
            count += 2
    return count


def _generate_selected_experiment_figures(
    *,
    selected_rows: Sequence[Mapping[str, Any]],
    benchmark_id: str,
    role: str,
    config_payload: Mapping[str, Any],
    output_paths: Mapping[str, Path],
    catalog_entries: list[dict[str, Any]],
) -> int:
    figure_config = dict(config_payload.get("figures", {}))
    variable = str(figure_config.get("variable", "ux")).strip().lower()
    requested_step = figure_config.get("step")
    step_policy = str(figure_config.get("step_policy", "middle"))
    time_series_metric = str(figure_config.get("time_series_metric", "rmse")).strip().lower()
    formats = _role_formats(config_payload, role)

    if role == "exploratory":
        comparison_dir = output_paths["exploratory_dir"] / "comparisons"
        error_map_dir = output_paths["exploratory_dir"] / "error_maps"
        error_series_dir = output_paths["exploratory_dir"] / "error_series"
    else:
        comparison_dir = output_paths["final_dir"] / "comparisons"
        error_map_dir = output_paths["final_dir"] / "error_maps"
        error_series_dir = output_paths["final_dir"] / "error_series"

    count = 0
    for row in selected_rows:
        bundle = _load_reconstruction_bundle(Path(str(row["reconstruction_dir"])))
        resolved_variable, _ = _resolve_variable(bundle, variable)
        step_index = _select_step_index(
            bundle.steps,
            requested_step=int(requested_step) if requested_step is not None else None,
            step_policy=step_policy,
        )
        figure_groups = (
            (
                "comparison",
                _plot_triptych(
                    bundle=bundle,
                    row=row,
                    benchmark_id=benchmark_id,
                    variable=resolved_variable,
                    step_index=step_index,
                    role=role,
                    output_dir=comparison_dir,
                    formats=formats,
                ),
                resolved_variable,
                int(bundle.steps[step_index]),
                None,
            ),
            (
                "error_map",
                _plot_error_maps(
                    bundle=bundle,
                    row=row,
                    benchmark_id=benchmark_id,
                    variable=resolved_variable,
                    step_index=step_index,
                    role=role,
                    output_dir=error_map_dir,
                    formats=formats,
                ),
                resolved_variable,
                int(bundle.steps[step_index]),
                None,
            ),
            (
                "error_series",
                _plot_error_series(
                    bundle=bundle,
                    row=row,
                    benchmark_id=benchmark_id,
                    role=role,
                    output_dir=error_series_dir,
                    formats=formats,
                    metric_name=time_series_metric,
                ),
                None,
                None,
                time_series_metric,
            ),
        )

        for category, written_paths, category_variable, category_step, category_metric in figure_groups:
            for written_path in written_paths:
                _register_artifact(
                    catalog_entries,
                    artifact_type="figure",
                    role=role,
                    category=category,
                    benchmark_id=benchmark_id,
                    path=written_path,
                    title=written_path.stem,
                    experiment_ids=[str(row["experiment_id"])],
                    variable=category_variable,
                    step=category_step,
                    metric=category_metric,
                )
                count += 1
                if role == "final":
                    if category == "comparison":
                        thesis_subdir = output_paths["thesis_ready_dir"] / "figures" / "comparisons"
                    elif category == "error_map":
                        thesis_subdir = output_paths["thesis_ready_dir"] / "figures" / "error_maps"
                    else:
                        thesis_subdir = output_paths["thesis_ready_dir"] / "figures" / "error_series"
                    thesis_path = _copy_to_thesis_ready(written_path, thesis_subdir / written_path.name)
                    _register_artifact(
                        catalog_entries,
                        artifact_type="figure",
                        role="thesis_ready",
                        category=category,
                        benchmark_id=benchmark_id,
                        path=thesis_path,
                        title=thesis_path.stem,
                        experiment_ids=[str(row["experiment_id"])],
                        variable=category_variable,
                        step=category_step,
                        metric=category_metric,
                    )
                    count += 1
    return count


def _generate_aggregate_figures(
    *,
    rows: Sequence[Mapping[str, Any]],
    benchmark_id: str,
    role: str,
    config_payload: Mapping[str, Any],
    output_paths: Mapping[str, Path],
    catalog_entries: list[dict[str, Any]],
) -> int:
    figure_config = dict(config_payload.get("figures", {}))
    summary_metric = str(figure_config.get("summary_metric", "rmse")).strip().lower()
    time_bar_metric = str(figure_config.get("time_bar_metric", "reconstruction_time_sec")).strip().lower()
    formats = _role_formats(config_payload, role)
    aggregate_dir = output_paths["exploratory_dir"] / "aggregate" if role == "exploratory" else output_paths["final_dir"] / "aggregate"

    figure_specs = (
        ("aggregate", _plot_metric_by_resolution(rows=rows, benchmark_id=benchmark_id, role=role, output_dir=aggregate_dir, formats=formats, metric_name=summary_metric), summary_metric),
        ("aggregate", _plot_metric_by_noise_case(rows=rows, benchmark_id=benchmark_id, role=role, output_dir=aggregate_dir, formats=formats, metric_name=summary_metric), summary_metric),
        ("aggregate", _plot_runtime_bars(rows=rows, benchmark_id=benchmark_id, role=role, output_dir=aggregate_dir, formats=formats, metric_name=time_bar_metric), time_bar_metric),
        ("aggregate", _plot_performance_heatmaps(rows=rows, benchmark_id=benchmark_id, role=role, output_dir=aggregate_dir, formats=formats, metric_name=summary_metric), summary_metric),
    )

    count = 0
    for category, written_paths, metric_name in figure_specs:
        for written_path in written_paths:
            _register_artifact(
                catalog_entries,
                artifact_type="figure",
                role=role,
                category=category,
                benchmark_id=benchmark_id,
                path=written_path,
                title=written_path.stem,
                metric=metric_name,
            )
            count += 1
            if role == "final":
                thesis_path = _copy_to_thesis_ready(
                    written_path,
                    output_paths["thesis_ready_dir"] / "figures" / "aggregate" / written_path.name,
                )
                _register_artifact(
                    catalog_entries,
                    artifact_type="figure",
                    role="thesis_ready",
                    category=category,
                    benchmark_id=benchmark_id,
                    path=thesis_path,
                    title=thesis_path.stem,
                    metric=metric_name,
                )
                count += 1
    return count


def generate_visual_results(
    config_payload: Mapping[str, Any],
    *,
    config_path: Path | str | None = None,
) -> VisualResultsSummary:
    _validate_visual_results_config(config_payload)
    config_base_dir = REPO_ROOT if config_path is None else Path(config_path).resolve().parent
    benchmark_id, benchmark_metrics_root, benchmark_tables_root, summary_csv_path = _resolve_source_paths(
        config_payload,
        config_base_dir=config_base_dir,
    )
    _validate_phase6_sources(
        benchmark_id=benchmark_id,
        benchmark_metrics_root=benchmark_metrics_root,
        benchmark_tables_root=benchmark_tables_root,
        summary_csv_path=summary_csv_path,
    )

    output_paths = _resolve_output_paths(config_payload, config_base_dir=config_base_dir, benchmark_id=benchmark_id)
    _write_json(output_paths["config_snapshot_path"], dict(config_payload))

    rows = _load_summary_rows(summary_csv_path)
    _validate_summary_row_artifacts(rows, benchmark_metrics_root=benchmark_metrics_root)
    selected_rows = _select_comparison_rows(rows, dict(config_payload.get("selection", {})))
    catalog_entries: list[dict[str, Any]] = []

    figure_count = 0
    table_count = 0
    figure_count += _generate_selected_experiment_figures(
        selected_rows=selected_rows,
        benchmark_id=benchmark_id,
        role="exploratory",
        config_payload=config_payload,
        output_paths=output_paths,
        catalog_entries=catalog_entries,
    )
    figure_count += _generate_selected_experiment_figures(
        selected_rows=selected_rows,
        benchmark_id=benchmark_id,
        role="final",
        config_payload=config_payload,
        output_paths=output_paths,
        catalog_entries=catalog_entries,
    )
    figure_count += _generate_aggregate_figures(
        rows=rows,
        benchmark_id=benchmark_id,
        role="exploratory",
        config_payload=config_payload,
        output_paths=output_paths,
        catalog_entries=catalog_entries,
    )
    figure_count += _generate_aggregate_figures(
        rows=rows,
        benchmark_id=benchmark_id,
        role="final",
        config_payload=config_payload,
        output_paths=output_paths,
        catalog_entries=catalog_entries,
    )

    table_config = dict(config_payload.get("tables", {}))
    experiment_columns = tuple(table_config.get("experiment_columns", DEFAULT_EXPERIMENT_TABLE_COLUMNS))
    aggregate_columns = tuple(table_config.get("aggregate_columns", DEFAULT_AGGREGATE_TABLE_COLUMNS))
    sort_by = str(table_config.get("sort_by", "rmse")).strip()
    ascending = bool(table_config.get("ascending", True))
    table_count += _generate_tables(
        rows=rows,
        benchmark_id=benchmark_id,
        role="exploratory",
        output_dir=output_paths["exploratory_dir"] / "tables",
        thesis_ready_dir=output_paths["thesis_ready_dir"],
        catalog_entries=catalog_entries,
        experiment_columns=experiment_columns,
        aggregate_columns=aggregate_columns,
        sort_by=sort_by,
        ascending=ascending,
    )
    table_count += _generate_tables(
        rows=rows,
        benchmark_id=benchmark_id,
        role="final",
        output_dir=output_paths["final_dir"] / "tables",
        thesis_ready_dir=output_paths["thesis_ready_dir"],
        catalog_entries=catalog_entries,
        experiment_columns=experiment_columns,
        aggregate_columns=aggregate_columns,
        sort_by=sort_by,
        ascending=ascending,
    )

    _write_catalog(output_paths, catalog_entries=catalog_entries)
    return VisualResultsSummary(
        benchmark_id=benchmark_id,
        output_root=output_paths["output_root"],
        exploratory_dir=output_paths["exploratory_dir"],
        final_dir=output_paths["final_dir"],
        thesis_ready_dir=output_paths["thesis_ready_dir"],
        catalog_json_path=output_paths["catalog_json_path"],
        catalog_csv_path=output_paths["catalog_csv_path"],
        catalog_markdown_path=output_paths["catalog_markdown_path"],
        num_figures=figure_count,
        num_tables=table_count,
        num_selected_experiments=len(selected_rows),
    )


def generate_visual_results_from_config(config_path: Path | str) -> VisualResultsSummary:
    config = load_visual_results_config(config_path)
    return generate_visual_results(config, config_path=config_path)


__all__ = [
    "ReconstructionBundle",
    "SCHEMA_VERSION",
    "VisualResultsSummary",
    "generate_visual_results",
    "generate_visual_results_from_config",
    "load_visual_results_config",
]
