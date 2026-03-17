from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from fluid_denoise.phase2_clean_dataset import (
    SCHEMA_VERSION as CLEAN_SCHEMA_VERSION,
    load_clean_run,
    validate_clean_dataset,
)
from fluid_denoise.phase3_noisy_dataset import (
    SCHEMA_VERSION as NOISY_SCHEMA_VERSION,
    load_noisy_run,
    validate_noisy_dataset,
)


MODEL_DYNAMIC_VARIABLES = ("ux", "uy", "speed", "vorticity")
MASK_POLICIES = ("keep", "fill_obstacle", "flatten_fluid")
NORMALIZATION_MODES = ("none", "standardize", "minmax", "maxabs")
NORMALIZATION_SCOPES = ("global", "per_channel", "per_feature")


@dataclass(frozen=True)
class ModelRun:
    run_dir: Path
    dataset_kind: str
    schema_version: str
    experiment_id: str
    metadata: dict[str, Any]
    steps: np.ndarray
    mask: np.ndarray
    arrays: dict[str, np.ndarray]
    diagnostics: dict[str, np.ndarray]

    @property
    def dimensions(self) -> tuple[int, int]:
        return (int(self.mask.shape[0]), int(self.mask.shape[1]))

    @property
    def num_snapshots(self) -> int:
        return int(self.steps.shape[0])


@dataclass(frozen=True)
class ModelRunPair:
    observed: ModelRun
    reference: ModelRun


@dataclass(frozen=True)
class NormalizationSpec:
    mode: str = "none"
    scope: str = "global"
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        mode = self.mode.strip().lower()
        scope = self.scope.strip().lower()
        if mode not in NORMALIZATION_MODES:
            raise ValueError(f"Unsupported normalization mode: {self.mode}")
        if scope not in NORMALIZATION_SCOPES:
            raise ValueError(f"Unsupported normalization scope: {self.scope}")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError("Normalization epsilon must be finite and positive")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "scope", scope)


@dataclass(frozen=True)
class NormalizationStats:
    mode: str
    scope: str
    offset: np.ndarray
    scale: np.ndarray
    epsilon: float
    channel_axis: int | None
    feature_axis: int | None


@dataclass(frozen=True)
class PackingContext:
    mask_policy: str
    spatial_shape: tuple[int, int]
    obstacle_fill_value: float
    fluid_flat_indices: np.ndarray | None
    include_mask_channel: bool
    variable_channel_count: int
    layout: str


@dataclass(frozen=True)
class SnapshotBatch:
    run_dir: Path
    dataset_kind: str
    experiment_id: str
    variables: tuple[str, ...]
    steps: np.ndarray
    data: np.ndarray
    mask: np.ndarray
    context: PackingContext
    normalization: NormalizationStats | None


@dataclass(frozen=True)
class TemporalWindowBatch:
    run_dir: Path
    dataset_kind: str
    experiment_id: str
    variables: tuple[str, ...]
    window_steps: np.ndarray
    data: np.ndarray
    mask: np.ndarray
    context: PackingContext
    normalization: NormalizationStats | None


@dataclass(frozen=True)
class SpaceTimeMatrix:
    run_dir: Path
    dataset_kind: str
    experiment_id: str
    variables: tuple[str, ...]
    steps: np.ndarray
    data: np.ndarray
    mask: np.ndarray
    context: PackingContext
    normalization: NormalizationStats | None
    feature_layout: str


def _normalize_variables(variables: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(variable.strip().lower() for variable in variables))
    if not normalized:
        raise ValueError("At least one variable is required")
    if any(variable not in MODEL_DYNAMIC_VARIABLES for variable in normalized):
        raise ValueError(f"Variables must be a subset of {MODEL_DYNAMIC_VARIABLES}")
    return normalized


def _normalize_mask_policy(mask_policy: str) -> str:
    normalized = mask_policy.strip().lower()
    if normalized not in MASK_POLICIES:
        raise ValueError(f"Unsupported mask policy: {mask_policy}")
    return normalized


def _normalize_step_indices(
    run: ModelRun,
    *,
    snapshot_indices: Sequence[int] | None = None,
    steps: Sequence[int] | None = None,
) -> np.ndarray:
    if snapshot_indices is not None and steps is not None:
        raise ValueError("Use either snapshot_indices or steps, not both")

    if snapshot_indices is not None:
        indices = np.asarray(snapshot_indices, dtype=np.int64)
    elif steps is not None:
        requested_steps = [int(step) for step in steps]
        index_map = {int(step): idx for idx, step in enumerate(run.steps.tolist())}
        missing = [step for step in requested_steps if step not in index_map]
        if missing:
            raise ValueError(f"Requested steps were not found in the run: {missing}")
        indices = np.asarray([index_map[step] for step in requested_steps], dtype=np.int64)
    else:
        indices = np.arange(run.num_snapshots, dtype=np.int64)

    if indices.ndim != 1:
        raise ValueError("Snapshot selection must be 1D")
    if indices.size == 0:
        raise ValueError("Snapshot selection cannot be empty")
    if np.any(indices < 0) or np.any(indices >= run.num_snapshots):
        raise ValueError("Snapshot index out of bounds")
    return indices


def _build_packing_context(
    *,
    mask_policy: str,
    mask: np.ndarray,
    obstacle_fill_value: float,
    include_mask_channel: bool,
    variable_channel_count: int,
    layout: str,
) -> PackingContext:
    fluid_flat_indices = None
    if mask_policy == "flatten_fluid":
        fluid_flat_indices = np.flatnonzero(mask.reshape(-1) == 0).astype(np.int64)
        if fluid_flat_indices.size == 0:
            raise ValueError("Mask contains no fluid cells to flatten")
    return PackingContext(
        mask_policy=mask_policy,
        spatial_shape=(int(mask.shape[0]), int(mask.shape[1])),
        obstacle_fill_value=float(obstacle_fill_value),
        fluid_flat_indices=fluid_flat_indices,
        include_mask_channel=include_mask_channel,
        variable_channel_count=variable_channel_count,
        layout=layout,
    )


def load_model_run(run_dir: Path | str) -> ModelRun:
    run_dir = Path(run_dir).resolve()
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    schema_version = str(metadata.get("schema_version"))
    if schema_version == CLEAN_SCHEMA_VERSION:
        validate_clean_dataset(run_dir)
        clean_run = load_clean_run(run_dir)
        return ModelRun(
            run_dir=clean_run.run_dir,
            dataset_kind="clean",
            schema_version=schema_version,
            experiment_id=str(clean_run.metadata["experiment_id"]),
            metadata=clean_run.metadata,
            steps=np.asarray(clean_run.steps),
            mask=np.asarray(clean_run.mask),
            arrays={
                variable: np.asarray(getattr(clean_run, variable))
                for variable in MODEL_DYNAMIC_VARIABLES
            },
            diagnostics={},
        )

    if schema_version == NOISY_SCHEMA_VERSION:
        validate_noisy_dataset(run_dir)
        noisy_run = load_noisy_run(run_dir)
        return ModelRun(
            run_dir=noisy_run.run_dir,
            dataset_kind="noisy",
            schema_version=schema_version,
            experiment_id=str(noisy_run.metadata["experiment_id"]),
            metadata=noisy_run.metadata,
            steps=np.asarray(noisy_run.steps),
            mask=np.asarray(noisy_run.mask),
            arrays={
                variable: np.asarray(getattr(noisy_run, variable))
                for variable in MODEL_DYNAMIC_VARIABLES
            },
            diagnostics={
                "corruption_ux_mask": np.asarray(noisy_run.corruption_ux_mask),
                "corruption_uy_mask": np.asarray(noisy_run.corruption_uy_mask),
            },
        )

    raise ValueError(f"Unsupported dataset schema version for model data: {schema_version}")


def align_model_runs(
    observed_run_dir: Path | str,
    reference_run_dir: Path | str,
) -> ModelRunPair:
    observed = load_model_run(observed_run_dir)
    reference = load_model_run(reference_run_dir)

    if observed.dimensions != reference.dimensions:
        raise ValueError("Observed and reference runs must have the same spatial dimensions")
    if not np.array_equal(observed.mask, reference.mask):
        raise ValueError("Observed and reference runs must share the same obstacle mask")
    if not np.array_equal(observed.steps, reference.steps):
        raise ValueError("Observed and reference runs must share the same snapshot steps")

    return ModelRunPair(observed=observed, reference=reference)


def stack_run_variables(
    run: ModelRun,
    *,
    variables: Sequence[str],
    snapshot_indices: Sequence[int] | None = None,
    steps: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    normalized_variables = _normalize_variables(variables)
    indices = _normalize_step_indices(run, snapshot_indices=snapshot_indices, steps=steps)
    stacked = np.stack([run.arrays[variable][indices] for variable in normalized_variables], axis=1)
    return np.asarray(stacked, dtype=np.float64), np.asarray(run.steps[indices], dtype=np.int32)


def fit_normalization(
    data: np.ndarray,
    spec: NormalizationSpec,
    *,
    channel_axis: int | None = None,
    feature_axis: int | None = None,
) -> NormalizationStats:
    if data.size == 0:
        raise ValueError("Cannot fit normalization on an empty array")

    if spec.scope == "global":
        reduction_axes = tuple(range(data.ndim))
    elif spec.scope == "per_channel":
        if channel_axis is None:
            raise ValueError("channel_axis is required for per_channel normalization")
        normalized_axis = channel_axis % data.ndim
        reduction_axes = tuple(axis for axis in range(data.ndim) if axis != normalized_axis)
        channel_axis = normalized_axis
    else:
        if feature_axis is None:
            raise ValueError("feature_axis is required for per_feature normalization")
        normalized_axis = feature_axis % data.ndim
        reduction_axes = tuple(axis for axis in range(data.ndim) if axis != normalized_axis)
        feature_axis = normalized_axis

    if spec.mode == "none":
        offset = np.zeros([1] * data.ndim, dtype=np.float64)
        scale = np.ones([1] * data.ndim, dtype=np.float64)
    elif spec.mode == "standardize":
        offset = np.mean(data, axis=reduction_axes, keepdims=True, dtype=np.float64)
        scale = np.std(data, axis=reduction_axes, keepdims=True, dtype=np.float64)
    elif spec.mode == "minmax":
        offset = np.min(data, axis=reduction_axes, keepdims=True)
        scale = np.max(data, axis=reduction_axes, keepdims=True) - offset
    else:
        offset = np.zeros([1] * data.ndim, dtype=np.float64)
        scale = np.max(np.abs(data), axis=reduction_axes, keepdims=True)

    scale = np.maximum(np.asarray(scale, dtype=np.float64), spec.epsilon)
    offset = np.asarray(offset, dtype=np.float64)
    return NormalizationStats(
        mode=spec.mode,
        scope=spec.scope,
        offset=offset,
        scale=scale,
        epsilon=spec.epsilon,
        channel_axis=channel_axis,
        feature_axis=feature_axis,
    )


def apply_normalization(data: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    return np.asarray((data - stats.offset) / stats.scale, dtype=np.float64)


def invert_normalization(data: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    return np.asarray(data * stats.scale + stats.offset, dtype=np.float64)


def _normalize_array(
    data: np.ndarray,
    spec: NormalizationSpec | None,
    *,
    channel_axis: int | None = None,
    feature_axis: int | None = None,
) -> tuple[np.ndarray, NormalizationStats | None]:
    if spec is None or spec.mode == "none":
        return np.asarray(data, dtype=np.float64), None
    stats = fit_normalization(
        np.asarray(data, dtype=np.float64),
        spec,
        channel_axis=channel_axis,
        feature_axis=feature_axis,
    )
    return apply_normalization(data, stats), stats


def _append_mask_channel(
    data: np.ndarray,
    mask: np.ndarray,
    *,
    channel_axis: int,
) -> np.ndarray:
    if data.shape[-2:] != mask.shape:
        raise ValueError("Grid data shape does not match the obstacle mask shape")
    expanded_mask = mask.astype(np.float64)
    for _ in range(data.ndim - 2):
        expanded_mask = np.expand_dims(expanded_mask, axis=0)
    target_shape = list(data.shape)
    target_shape[channel_axis] = 1
    expanded_mask = np.broadcast_to(expanded_mask, target_shape)
    return np.concatenate([data, expanded_mask], axis=channel_axis)


def _apply_grid_mask_policy(
    data: np.ndarray,
    mask: np.ndarray,
    *,
    mask_policy: str,
    obstacle_fill_value: float,
) -> np.ndarray:
    if data.shape[-2:] != mask.shape:
        raise ValueError("Grid data shape does not match the obstacle mask shape")
    if mask_policy == "keep":
        return np.asarray(data, dtype=np.float64).copy()
    if mask_policy == "fill_obstacle":
        output = np.asarray(data, dtype=np.float64).copy()
        output[..., mask == 1] = obstacle_fill_value
        return output
    raise ValueError(f"Grid mask policy not supported here: {mask_policy}")


def _flatten_spatial_grid(data: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if data.shape[-2:] != mask.shape:
        raise ValueError("Grid data shape does not match the obstacle mask shape")
    flat = data.reshape(data.shape[:-2] + (mask.size,))
    fluid_flat_indices = np.flatnonzero(mask.reshape(-1) == 0).astype(np.int64)
    if fluid_flat_indices.size == 0:
        raise ValueError("Mask contains no fluid cells to flatten")
    return np.asarray(flat[..., fluid_flat_indices], dtype=np.float64), fluid_flat_indices


def _restore_spatial_grid(
    data: np.ndarray,
    *,
    context: PackingContext,
) -> np.ndarray:
    ny, nx = context.spatial_shape
    if context.mask_policy == "flatten_fluid":
        assert context.fluid_flat_indices is not None
        restored = np.full(
            data.shape[:-1] + (ny * nx,),
            fill_value=context.obstacle_fill_value,
            dtype=np.float64,
        )
        restored[..., context.fluid_flat_indices] = data
        return restored.reshape(data.shape[:-1] + (ny, nx))
    if data.shape[-2:] != (ny, nx):
        raise ValueError("Packed grid does not match the stored spatial shape")
    return np.asarray(data, dtype=np.float64)


def _strip_mask_channel(data: np.ndarray, *, context: PackingContext, channel_axis: int) -> np.ndarray:
    if not context.include_mask_channel:
        return data
    slicer = [slice(None)] * data.ndim
    slicer[channel_axis] = slice(0, context.variable_channel_count)
    return np.asarray(data[tuple(slicer)], dtype=np.float64)


def pack_snapshot_batch(
    run: ModelRun,
    *,
    variables: Sequence[str],
    snapshot_indices: Sequence[int] | None = None,
    steps: Sequence[int] | None = None,
    mask_policy: str = "keep",
    obstacle_fill_value: float = 0.0,
    include_mask_channel: bool = False,
    normalization: NormalizationSpec | None = None,
) -> SnapshotBatch:
    normalized_policy = _normalize_mask_policy(mask_policy)
    stacked, selected_steps = stack_run_variables(
        run,
        variables=variables,
        snapshot_indices=snapshot_indices,
        steps=steps,
    )

    if normalized_policy == "flatten_fluid":
        if include_mask_channel:
            raise ValueError("include_mask_channel is not supported with flatten_fluid")
        packed, fluid_flat_indices = _flatten_spatial_grid(stacked, run.mask)
        normalized_data, normalization_stats = _normalize_array(
            packed,
            normalization,
            channel_axis=1,
            feature_axis=2,
        )
        context = PackingContext(
            mask_policy=normalized_policy,
            spatial_shape=run.dimensions,
            obstacle_fill_value=float(obstacle_fill_value),
            fluid_flat_indices=fluid_flat_indices,
            include_mask_channel=False,
            variable_channel_count=packed.shape[1],
            layout="batch_channel_feature",
        )
        return SnapshotBatch(
            run_dir=run.run_dir,
            dataset_kind=run.dataset_kind,
            experiment_id=run.experiment_id,
            variables=_normalize_variables(variables),
            steps=selected_steps,
            data=normalized_data,
            mask=np.asarray(run.mask),
            context=context,
            normalization=normalization_stats,
        )

    grid = _apply_grid_mask_policy(
        stacked,
        run.mask,
        mask_policy=normalized_policy,
        obstacle_fill_value=obstacle_fill_value,
    )
    normalized_grid, normalization_stats = _normalize_array(
        grid,
        normalization,
        channel_axis=1,
        feature_axis=None,
    )
    if include_mask_channel:
        normalized_grid = _append_mask_channel(normalized_grid, run.mask, channel_axis=1)
    context = _build_packing_context(
        mask_policy=normalized_policy,
        mask=run.mask,
        obstacle_fill_value=obstacle_fill_value,
        include_mask_channel=include_mask_channel,
        variable_channel_count=stacked.shape[1],
        layout="batch_channel_y_x",
    )
    return SnapshotBatch(
        run_dir=run.run_dir,
        dataset_kind=run.dataset_kind,
        experiment_id=run.experiment_id,
        variables=_normalize_variables(variables),
        steps=selected_steps,
        data=normalized_grid,
        mask=np.asarray(run.mask),
        context=context,
        normalization=normalization_stats,
    )


def unpack_snapshot_batch(batch: SnapshotBatch) -> np.ndarray:
    if batch.context.layout == "batch_channel_feature":
        restored = _restore_spatial_grid(batch.data, context=batch.context)
        return np.asarray(restored, dtype=np.float64)
    if batch.context.layout == "batch_channel_y_x":
        stripped = _strip_mask_channel(batch.data, context=batch.context, channel_axis=1)
        return _restore_spatial_grid(stripped, context=batch.context)
    raise ValueError(f"Unsupported snapshot batch layout: {batch.context.layout}")


def pack_temporal_windows(
    run: ModelRun,
    *,
    variables: Sequence[str],
    window_size: int,
    stride: int = 1,
    snapshot_indices: Sequence[int] | None = None,
    steps: Sequence[int] | None = None,
    mask_policy: str = "keep",
    obstacle_fill_value: float = 0.0,
    include_mask_channel: bool = False,
    normalization: NormalizationSpec | None = None,
) -> TemporalWindowBatch:
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")

    normalized_policy = _normalize_mask_policy(mask_policy)
    stacked, selected_steps = stack_run_variables(
        run,
        variables=variables,
        snapshot_indices=snapshot_indices,
        steps=steps,
    )
    if stacked.shape[0] < window_size:
        raise ValueError("window_size cannot exceed the number of selected snapshots")

    start_indices = np.arange(0, stacked.shape[0] - window_size + 1, stride, dtype=np.int64)
    windows = np.stack(
        [stacked[start : start + window_size] for start in start_indices],
        axis=0,
    )
    window_steps = np.stack(
        [selected_steps[start : start + window_size] for start in start_indices],
        axis=0,
    )

    if normalized_policy == "flatten_fluid":
        if include_mask_channel:
            raise ValueError("include_mask_channel is not supported with flatten_fluid")
        packed, fluid_flat_indices = _flatten_spatial_grid(windows, run.mask)
        normalized_data, normalization_stats = _normalize_array(
            packed,
            normalization,
            channel_axis=2,
            feature_axis=3,
        )
        context = PackingContext(
            mask_policy=normalized_policy,
            spatial_shape=run.dimensions,
            obstacle_fill_value=float(obstacle_fill_value),
            fluid_flat_indices=fluid_flat_indices,
            include_mask_channel=False,
            variable_channel_count=windows.shape[2],
            layout="batch_time_channel_feature",
        )
        return TemporalWindowBatch(
            run_dir=run.run_dir,
            dataset_kind=run.dataset_kind,
            experiment_id=run.experiment_id,
            variables=_normalize_variables(variables),
            window_steps=window_steps,
            data=normalized_data,
            mask=np.asarray(run.mask),
            context=context,
            normalization=normalization_stats,
        )

    grid = _apply_grid_mask_policy(
        windows,
        run.mask,
        mask_policy=normalized_policy,
        obstacle_fill_value=obstacle_fill_value,
    )
    normalized_grid, normalization_stats = _normalize_array(
        grid,
        normalization,
        channel_axis=2,
        feature_axis=None,
    )
    if include_mask_channel:
        normalized_grid = _append_mask_channel(normalized_grid, run.mask, channel_axis=2)
    context = _build_packing_context(
        mask_policy=normalized_policy,
        mask=run.mask,
        obstacle_fill_value=obstacle_fill_value,
        include_mask_channel=include_mask_channel,
        variable_channel_count=windows.shape[2],
        layout="batch_time_channel_y_x",
    )
    return TemporalWindowBatch(
        run_dir=run.run_dir,
        dataset_kind=run.dataset_kind,
        experiment_id=run.experiment_id,
        variables=_normalize_variables(variables),
        window_steps=window_steps,
        data=normalized_grid,
        mask=np.asarray(run.mask),
        context=context,
        normalization=normalization_stats,
    )


def unpack_temporal_windows(batch: TemporalWindowBatch) -> np.ndarray:
    if batch.context.layout == "batch_time_channel_feature":
        return _restore_spatial_grid(batch.data, context=batch.context)
    if batch.context.layout == "batch_time_channel_y_x":
        stripped = _strip_mask_channel(batch.data, context=batch.context, channel_axis=2)
        return _restore_spatial_grid(stripped, context=batch.context)
    raise ValueError(f"Unsupported temporal window layout: {batch.context.layout}")


def pack_space_time_matrix(
    run: ModelRun,
    *,
    variables: Sequence[str],
    snapshot_indices: Sequence[int] | None = None,
    steps: Sequence[int] | None = None,
    mask_policy: str = "flatten_fluid",
    obstacle_fill_value: float = 0.0,
    normalization: NormalizationSpec | None = None,
) -> SpaceTimeMatrix:
    normalized_policy = _normalize_mask_policy(mask_policy)
    stacked, selected_steps = stack_run_variables(
        run,
        variables=variables,
        snapshot_indices=snapshot_indices,
        steps=steps,
    )

    if normalized_policy == "flatten_fluid":
        packed_grid, fluid_flat_indices = _flatten_spatial_grid(stacked, run.mask)
        matrix = packed_grid.transpose(1, 2, 0).reshape(
            packed_grid.shape[1] * packed_grid.shape[2],
            packed_grid.shape[0],
        )
        context = PackingContext(
            mask_policy=normalized_policy,
            spatial_shape=run.dimensions,
            obstacle_fill_value=float(obstacle_fill_value),
            fluid_flat_indices=fluid_flat_indices,
            include_mask_channel=False,
            variable_channel_count=stacked.shape[1],
            layout="feature_time",
        )
    else:
        grid = _apply_grid_mask_policy(
            stacked,
            run.mask,
            mask_policy=normalized_policy,
            obstacle_fill_value=obstacle_fill_value,
        )
        ny, nx = run.dimensions
        matrix = grid.transpose(1, 2, 3, 0).reshape(stacked.shape[1] * ny * nx, stacked.shape[0])
        context = _build_packing_context(
            mask_policy=normalized_policy,
            mask=run.mask,
            obstacle_fill_value=obstacle_fill_value,
            include_mask_channel=False,
            variable_channel_count=stacked.shape[1],
            layout="feature_time",
        )

    normalized_matrix, normalization_stats = _normalize_array(
        matrix,
        normalization,
        channel_axis=None,
        feature_axis=0,
    )
    return SpaceTimeMatrix(
        run_dir=run.run_dir,
        dataset_kind=run.dataset_kind,
        experiment_id=run.experiment_id,
        variables=_normalize_variables(variables),
        steps=selected_steps,
        data=normalized_matrix,
        mask=np.asarray(run.mask),
        context=context,
        normalization=normalization_stats,
        feature_layout="variable_major_then_row_major_spatial",
    )


def unpack_space_time_matrix(matrix: SpaceTimeMatrix) -> np.ndarray:
    num_variables = len(matrix.variables)
    num_snapshots = int(matrix.steps.shape[0])
    ny, nx = matrix.context.spatial_shape

    if matrix.context.mask_policy == "flatten_fluid":
        assert matrix.context.fluid_flat_indices is not None
        num_fluid = int(matrix.context.fluid_flat_indices.shape[0])
        expected_features = num_variables * num_fluid
        if matrix.data.shape != (expected_features, num_snapshots):
            raise ValueError("Space-time matrix shape is inconsistent with the stored mask context")
        packed = matrix.data.reshape(num_variables, num_fluid, num_snapshots).transpose(2, 0, 1)
        return _restore_spatial_grid(packed, context=matrix.context)

    expected_features = num_variables * ny * nx
    if matrix.data.shape != (expected_features, num_snapshots):
        raise ValueError("Space-time matrix shape is inconsistent with the stored spatial shape")
    restored = matrix.data.reshape(num_variables, ny, nx, num_snapshots).transpose(3, 0, 1, 2)
    return np.asarray(restored, dtype=np.float64)


def summarize_model_run(run_dir: Path | str) -> dict[str, Any]:
    run = load_model_run(run_dir)
    return {
        "experiment_id": run.experiment_id,
        "dataset_kind": run.dataset_kind,
        "schema_version": run.schema_version,
        "dimensions": {"ny": run.dimensions[0], "nx": run.dimensions[1]},
        "num_snapshots": run.num_snapshots,
        "variables": list(MODEL_DYNAMIC_VARIABLES),
        "diagnostics": sorted(run.diagnostics.keys()),
    }
