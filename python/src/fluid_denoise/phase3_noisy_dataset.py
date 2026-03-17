from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from fluid_denoise.phase2_clean_dataset import (
    CleanRun,
    load_clean_run,
    validate_clean_dataset,
)


SCHEMA_VERSION = "phase3.noisy.v1"
DYNAMIC_VARIABLES = ("ux", "uy", "speed", "vorticity")
STATIC_VARIABLES = ("mask",)
DIAGNOSTIC_VARIABLES = ("corruption_ux_mask", "corruption_uy_mask")
TARGET_CHANNELS = ("ux", "uy")
NOISE_KINDS = (
    "salt_and_pepper",
    "gaussian",
    "speckle",
    "spatial_dropout",
    "missing_blocks",
    "piv_outliers",
)


@dataclass(frozen=True)
class NoiseSpec:
    kind: str
    intensity: float
    seed: int | None
    channels: tuple[str, ...] = ("ux", "uy")
    include_obstacle: bool = False
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        kind = self.kind.strip().lower().replace("-", "_")
        if kind not in NOISE_KINDS:
            raise ValueError(f"Unsupported noise kind: {self.kind}")
        if not math.isfinite(self.intensity) or self.intensity < 0.0:
            raise ValueError("Noise intensity must be a finite non-negative number")
        normalized_channels = tuple(
            dict.fromkeys(channel.strip().lower() for channel in self.channels)
        )
        if not normalized_channels:
            raise ValueError("Noise channels cannot be empty")
        if any(channel not in TARGET_CHANNELS for channel in normalized_channels):
            raise ValueError(f"Noise channels must be a subset of {TARGET_CHANNELS}")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "channels", normalized_channels)
        object.__setattr__(self, "params", dict(self.params))


@dataclass(frozen=True)
class NoisyRun:
    run_dir: Path
    metadata: dict[str, Any]
    steps: np.ndarray
    ux: np.ndarray
    uy: np.ndarray
    speed: np.ndarray
    vorticity: np.ndarray
    mask: np.ndarray
    corruption_ux_mask: np.ndarray
    corruption_uy_mask: np.ndarray

    @property
    def variables(self) -> tuple[str, ...]:
        return DYNAMIC_VARIABLES + STATIC_VARIABLES + DIAGNOSTIC_VARIABLES

    @property
    def dimensions(self) -> tuple[int, int]:
        return (int(self.metadata["grid"]["ny"]), int(self.metadata["grid"]["nx"]))

    @property
    def num_snapshots(self) -> int:
        return int(self.steps.shape[0])


@dataclass(frozen=True)
class NoisyDatasetSummary:
    run_dir: Path
    experiment_id: str
    nx: int
    ny: int
    num_snapshots: int
    noise_kinds: tuple[str, ...]
    steps: tuple[int, ...]


def _sanitize_token(value: str) -> str:
    sanitized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return sanitized or "unknown"


def _normalize_noise_specs(
    noise_specs: Sequence[NoiseSpec | Mapping[str, Any]],
) -> list[NoiseSpec]:
    normalized: list[NoiseSpec] = []
    for spec in noise_specs:
        if isinstance(spec, NoiseSpec):
            normalized.append(spec)
            continue
        params = dict(spec.get("params", {}))
        normalized.append(
            NoiseSpec(
                kind=str(spec["kind"]),
                intensity=float(spec["intensity"]),
                seed=spec.get("seed"),
                channels=tuple(spec.get("channels", TARGET_CHANNELS)),
                include_obstacle=bool(spec.get("include_obstacle", False)),
                params=params,
            )
        )
    if not normalized:
        raise ValueError("At least one noise specification is required")
    return normalized


def build_noisy_experiment_id(
    clean_metadata: Mapping[str, Any],
    noise_specs: Sequence[NoiseSpec | Mapping[str, Any]],
) -> str:
    normalized = _normalize_noise_specs(noise_specs)
    source_experiment_id = _sanitize_token(str(clean_metadata["experiment_id"]))
    stage_tokens = []
    for spec in normalized:
        intensity_token = int(round(spec.intensity * 1000.0))
        seed_token = "none" if spec.seed is None else str(spec.seed)
        stage_tokens.append(f"{spec.kind}-i{intensity_token:04d}-s{seed_token}")
    return f"{source_experiment_id}__noisy__{'__'.join(stage_tokens)}"


def _resolve_source_scale(
    array: np.ndarray,
    eligible_mask_2d: np.ndarray,
    *,
    scale_mode: str,
    explicit_scale: float | None,
) -> float:
    if explicit_scale is not None:
        if explicit_scale < 0.0 or not math.isfinite(explicit_scale):
            raise ValueError("Explicit scale must be finite and non-negative")
        return float(explicit_scale)

    values = array[:, eligible_mask_2d]
    if values.size == 0:
        return 0.0

    scale_mode = scale_mode.strip().lower()
    if scale_mode == "channel_std":
        scale = float(np.std(values))
    elif scale_mode == "channel_range":
        scale = float(np.max(values) - np.min(values))
    elif scale_mode == "channel_rms":
        scale = float(np.sqrt(np.mean(values * values)))
    elif scale_mode == "absolute":
        scale = 1.0
    else:
        raise ValueError(f"Unsupported scale_mode: {scale_mode}")

    if not math.isfinite(scale):
        raise ValueError("Computed scale is not finite")
    if scale == 0.0 and scale_mode != "absolute":
        fallback = float(np.max(np.abs(values))) if values.size else 0.0
        return fallback if fallback > 0.0 else 1.0
    return scale


def _selection_fraction(intensity: float) -> float:
    if intensity > 1.0:
        raise ValueError("Selection-based noise intensity must be within [0, 1]")
    return intensity


def _eligible_mask_3d(mask: np.ndarray, time_len: int, include_obstacle: bool) -> np.ndarray:
    spatial = np.ones_like(mask, dtype=bool) if include_obstacle else mask == 0
    return np.broadcast_to(spatial, (time_len,) + spatial.shape).copy()


def _sample_fraction_mask(
    rng: np.random.Generator,
    eligible_mask: np.ndarray,
    fraction: float,
) -> np.ndarray:
    if fraction <= 0.0:
        return np.zeros_like(eligible_mask, dtype=bool)
    return eligible_mask & (rng.random(eligible_mask.shape) < fraction)


def _generate_spatial_dropout_mask(
    rng: np.random.Generator,
    eligible_mask_2d: np.ndarray,
    time_len: int,
    fraction: float,
    *,
    radius_y: int,
    radius_x: int,
    persistent: bool,
) -> np.ndarray:
    if fraction <= 0.0:
        return np.zeros((time_len,) + eligible_mask_2d.shape, dtype=bool)

    indices = np.argwhere(eligible_mask_2d)
    if indices.size == 0:
        return np.zeros((time_len,) + eligible_mask_2d.shape, dtype=bool)

    def build_single_mask() -> np.ndarray:
        target = int(math.ceil(fraction * int(np.count_nonzero(eligible_mask_2d))))
        selected = np.zeros_like(eligible_mask_2d, dtype=bool)
        attempts = 0
        max_attempts = max(target * 10, 50)
        while np.count_nonzero(selected) < target and attempts < max_attempts:
            cy, cx = indices[int(rng.integers(len(indices)))]
            y0 = max(0, cy - radius_y)
            y1 = min(eligible_mask_2d.shape[0], cy + radius_y + 1)
            x0 = max(0, cx - radius_x)
            x1 = min(eligible_mask_2d.shape[1], cx + radius_x + 1)
            selected[y0:y1, x0:x1] = True
            selected &= eligible_mask_2d
            attempts += 1
            if np.count_nonzero(selected) == np.count_nonzero(eligible_mask_2d):
                break
        return selected

    if persistent:
        single = build_single_mask()
        return np.broadcast_to(single, (time_len,) + single.shape).copy()
    return np.stack([build_single_mask() for _ in range(time_len)], axis=0)


def _generate_missing_blocks_mask(
    rng: np.random.Generator,
    eligible_mask_2d: np.ndarray,
    time_len: int,
    fraction: float,
    *,
    min_block_height: int,
    max_block_height: int,
    min_block_width: int,
    max_block_width: int,
    persistent: bool,
) -> np.ndarray:
    if fraction <= 0.0:
        return np.zeros((time_len,) + eligible_mask_2d.shape, dtype=bool)

    if min_block_height <= 0 or min_block_width <= 0:
        raise ValueError("Block dimensions must be positive integers")
    if max_block_height < min_block_height or max_block_width < min_block_width:
        raise ValueError("Maximum block dimensions must be >= minimum dimensions")

    ny, nx = eligible_mask_2d.shape

    def build_single_mask() -> np.ndarray:
        target = int(math.ceil(fraction * int(np.count_nonzero(eligible_mask_2d))))
        selected = np.zeros_like(eligible_mask_2d, dtype=bool)
        attempts = 0
        max_attempts = max(target * 8, 50)
        while np.count_nonzero(selected) < target and attempts < max_attempts:
            block_height = int(rng.integers(min_block_height, max_block_height + 1))
            block_width = int(rng.integers(min_block_width, max_block_width + 1))
            top = int(rng.integers(0, max(1, ny - block_height + 1)))
            left = int(rng.integers(0, max(1, nx - block_width + 1)))
            selected[top : top + block_height, left : left + block_width] = True
            selected &= eligible_mask_2d
            attempts += 1
            if np.count_nonzero(selected) == np.count_nonzero(eligible_mask_2d):
                break
        return selected

    if persistent:
        single = build_single_mask()
        return np.broadcast_to(single, (time_len,) + single.shape).copy()
    return np.stack([build_single_mask() for _ in range(time_len)], axis=0)


def _compute_vorticity(ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    dux_dy = np.gradient(ux, axis=1)
    duy_dx = np.gradient(uy, axis=2)
    return np.asarray(duy_dx - dux_dy, dtype=np.float64)


def _channel_stats(array: np.ndarray, eligible_mask_2d: np.ndarray) -> dict[str, float]:
    values = array[:, eligible_mask_2d]
    if values.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def _apply_noise_stage(
    arrays: dict[str, np.ndarray],
    *,
    spec: NoiseSpec,
    obstacle_mask: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    rng = np.random.default_rng(spec.seed)
    time_len = arrays["ux"].shape[0]
    eligible_mask = _eligible_mask_3d(obstacle_mask, time_len, spec.include_obstacle)
    eligible_mask_2d = eligible_mask[0]
    channel_masks = {
        "ux": np.zeros_like(arrays["ux"], dtype=bool),
        "uy": np.zeros_like(arrays["uy"], dtype=bool),
    }
    per_channel: dict[str, Any] = {}

    if spec.kind == "gaussian":
        scale_mode = str(spec.params.get("scale_mode", "channel_std"))
        explicit_scale = spec.params.get("scale")
        for channel in spec.channels:
            if spec.intensity == 0.0:
                per_channel[channel] = {
                    "sigma": 0.0,
                    "modified_count": 0,
                    "realized_fraction_of_eligible": 0.0,
                }
                continue
            scale = _resolve_source_scale(
                arrays[channel],
                eligible_mask_2d,
                scale_mode=scale_mode,
                explicit_scale=float(explicit_scale) if explicit_scale is not None else None,
            )
            sigma = spec.intensity * scale
            noise = rng.normal(0.0, sigma, size=arrays[channel].shape)
            arrays[channel] = np.where(eligible_mask, arrays[channel] + noise, arrays[channel])
            channel_masks[channel] = eligible_mask.copy()
            per_channel[channel] = {
                "sigma": float(sigma),
                "modified_count": int(np.count_nonzero(channel_masks[channel])),
                "realized_fraction_of_eligible": 1.0 if np.count_nonzero(eligible_mask) else 0.0,
            }

    elif spec.kind == "speckle":
        factor_scale = float(spec.params.get("factor_scale", 1.0))
        for channel in spec.channels:
            if spec.intensity == 0.0:
                per_channel[channel] = {
                    "factor_sigma": 0.0,
                    "modified_count": 0,
                    "realized_fraction_of_eligible": 0.0,
                }
                continue
            factor_noise = rng.normal(
                0.0,
                spec.intensity * factor_scale,
                size=arrays[channel].shape,
            )
            arrays[channel] = np.where(
                eligible_mask,
                arrays[channel] + arrays[channel] * factor_noise,
                arrays[channel],
            )
            channel_masks[channel] = eligible_mask.copy()
            per_channel[channel] = {
                "factor_sigma": float(spec.intensity * factor_scale),
                "modified_count": int(np.count_nonzero(channel_masks[channel])),
                "realized_fraction_of_eligible": 1.0 if np.count_nonzero(eligible_mask) else 0.0,
            }

    elif spec.kind == "salt_and_pepper":
        fraction = _selection_fraction(spec.intensity)
        linked_channels = bool(spec.params.get("linked_channels", True))
        salt_probability = float(spec.params.get("salt_probability", 0.5))
        if not 0.0 <= salt_probability <= 1.0:
            raise ValueError("salt_probability must be within [0, 1]")
        shared_selection = _sample_fraction_mask(rng, eligible_mask, fraction) if linked_channels else None
        for channel in spec.channels:
            selection = (
                shared_selection.copy()
                if shared_selection is not None
                else _sample_fraction_mask(rng, eligible_mask, fraction)
            )
            values = arrays[channel][:, eligible_mask_2d]
            min_value = float(np.min(values)) if values.size else 0.0
            max_value = float(np.max(values)) if values.size else 0.0
            pepper_value = float(spec.params.get("pepper_value", min_value))
            salt_value = float(spec.params.get("salt_value", max_value))
            salt_mask = selection & (rng.random(arrays[channel].shape) < salt_probability)
            replacement = np.where(salt_mask, salt_value, pepper_value)
            arrays[channel] = np.where(selection, replacement, arrays[channel])
            channel_masks[channel] = selection
            eligible_count = int(np.count_nonzero(eligible_mask))
            per_channel[channel] = {
                "salt_value": salt_value,
                "pepper_value": pepper_value,
                "modified_count": int(np.count_nonzero(selection)),
                "realized_fraction_of_eligible": (
                    float(np.count_nonzero(selection)) / float(eligible_count)
                    if eligible_count
                    else 0.0
                ),
            }

    elif spec.kind == "spatial_dropout":
        fraction = _selection_fraction(spec.intensity)
        radius_y = int(spec.params.get("radius_y", spec.params.get("radius", 1)))
        radius_x = int(spec.params.get("radius_x", spec.params.get("radius", 1)))
        fill_value = float(spec.params.get("fill_value", 0.0))
        persistent = bool(spec.params.get("persistent", False))
        linked_channels = bool(spec.params.get("linked_channels", True))
        shared_selection = _generate_spatial_dropout_mask(
            rng,
            eligible_mask_2d,
            time_len,
            fraction,
            radius_y=radius_y,
            radius_x=radius_x,
            persistent=persistent,
        )
        for channel in spec.channels:
            selection = (
                shared_selection.copy()
                if linked_channels
                else _generate_spatial_dropout_mask(
                    rng,
                    eligible_mask_2d,
                    time_len,
                    fraction,
                    radius_y=radius_y,
                    radius_x=radius_x,
                    persistent=persistent,
                )
            )
            arrays[channel] = np.where(selection, fill_value, arrays[channel])
            channel_masks[channel] = selection
            eligible_count = int(np.count_nonzero(eligible_mask))
            per_channel[channel] = {
                "fill_value": fill_value,
                "radius_y": radius_y,
                "radius_x": radius_x,
                "persistent": persistent,
                "modified_count": int(np.count_nonzero(selection)),
                "realized_fraction_of_eligible": (
                    float(np.count_nonzero(selection)) / float(eligible_count)
                    if eligible_count
                    else 0.0
                ),
            }

    elif spec.kind == "missing_blocks":
        fraction = _selection_fraction(spec.intensity)
        fill_value = float(spec.params.get("fill_value", 0.0))
        persistent = bool(spec.params.get("persistent", True))
        linked_channels = bool(spec.params.get("linked_channels", True))
        min_block_height = int(spec.params.get("min_block_height", 2))
        max_block_height = int(spec.params.get("max_block_height", 6))
        min_block_width = int(spec.params.get("min_block_width", 2))
        max_block_width = int(spec.params.get("max_block_width", 6))
        shared_selection = _generate_missing_blocks_mask(
            rng,
            eligible_mask_2d,
            time_len,
            fraction,
            min_block_height=min_block_height,
            max_block_height=max_block_height,
            min_block_width=min_block_width,
            max_block_width=max_block_width,
            persistent=persistent,
        )
        for channel in spec.channels:
            selection = (
                shared_selection.copy()
                if linked_channels
                else _generate_missing_blocks_mask(
                    rng,
                    eligible_mask_2d,
                    time_len,
                    fraction,
                    min_block_height=min_block_height,
                    max_block_height=max_block_height,
                    min_block_width=min_block_width,
                    max_block_width=max_block_width,
                    persistent=persistent,
                )
            )
            arrays[channel] = np.where(selection, fill_value, arrays[channel])
            channel_masks[channel] = selection
            eligible_count = int(np.count_nonzero(eligible_mask))
            per_channel[channel] = {
                "fill_value": fill_value,
                "persistent": persistent,
                "min_block_height": min_block_height,
                "max_block_height": max_block_height,
                "min_block_width": min_block_width,
                "max_block_width": max_block_width,
                "modified_count": int(np.count_nonzero(selection)),
                "realized_fraction_of_eligible": (
                    float(np.count_nonzero(selection)) / float(eligible_count)
                    if eligible_count
                    else 0.0
                ),
            }

    elif spec.kind == "piv_outliers":
        fraction = _selection_fraction(spec.intensity)
        linked_channels = bool(spec.params.get("linked_channels", True))
        outlier_scale = float(spec.params.get("outlier_scale", 5.0))
        mode = str(spec.params.get("mode", "replace")).strip().lower()
        if mode not in {"replace", "add"}:
            raise ValueError("piv_outliers mode must be 'replace' or 'add'")
        scale_mode = str(spec.params.get("scale_mode", "channel_std"))
        shared_selection = _sample_fraction_mask(rng, eligible_mask, fraction) if linked_channels else None
        for channel in spec.channels:
            selection = (
                shared_selection.copy()
                if shared_selection is not None
                else _sample_fraction_mask(rng, eligible_mask, fraction)
            )
            reference_scale = _resolve_source_scale(
                arrays[channel],
                eligible_mask_2d,
                scale_mode=scale_mode,
                explicit_scale=None,
            )
            outlier_values = rng.normal(
                0.0,
                outlier_scale * reference_scale,
                size=arrays[channel].shape,
            )
            replacement = outlier_values if mode == "replace" else arrays[channel] + outlier_values
            arrays[channel] = np.where(selection, replacement, arrays[channel])
            channel_masks[channel] = selection
            eligible_count = int(np.count_nonzero(eligible_mask))
            per_channel[channel] = {
                "mode": mode,
                "outlier_scale": outlier_scale,
                "reference_scale": float(reference_scale),
                "modified_count": int(np.count_nonzero(selection)),
                "realized_fraction_of_eligible": (
                    float(np.count_nonzero(selection)) / float(eligible_count)
                    if eligible_count
                    else 0.0
                ),
            }

    else:
        raise ValueError(f"Unsupported noise kind: {spec.kind}")

    stage_summary = {
        "kind": spec.kind,
        "intensity": float(spec.intensity),
        "seed": spec.seed,
        "channels": list(spec.channels),
        "include_obstacle": spec.include_obstacle,
        "params": dict(spec.params),
        "per_channel": per_channel,
    }
    return arrays, channel_masks, stage_summary


def _build_metadata(
    *,
    experiment_id: str,
    clean_run: CleanRun,
    noise_specs: Sequence[NoiseSpec],
    stage_summaries: Sequence[dict[str, Any]],
    corruption_masks: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    source_metadata = clean_run.metadata
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "dataset_kind": "noisy_lbm_run",
        "storage_format": {
            "recommended": "npz",
            "npz_file": "fields.npz",
            "metadata_file": "metadata.json",
            "compression": "np.savez_compressed",
        },
        "experiment_id": experiment_id,
        "source": {
            "clean_run_dir": str(clean_run.run_dir.resolve()),
            "clean_metadata_path": str((clean_run.run_dir / "metadata.json").resolve()),
            "clean_fields_path": str((clean_run.run_dir / "fields.npz").resolve()),
            "source_experiment_id": source_metadata["experiment_id"],
            "source_schema_version": source_metadata["schema_version"],
        },
        "grid": dict(source_metadata["grid"]),
        "snapshots": dict(source_metadata["snapshots"]),
        "variables": {
            "dynamic": list(DYNAMIC_VARIABLES),
            "static": list(STATIC_VARIABLES),
            "diagnostic": list(DIAGNOSTIC_VARIABLES),
        },
        "parameters": dict(source_metadata["parameters"]),
        "corruption": {
            "target_channels": list(TARGET_CHANNELS),
            "num_stages": len(noise_specs),
            "pipeline": list(stage_summaries),
            "protect_obstacle_by_default": True,
            "obstacle_touched": bool(
                np.any(corruption_masks["ux"][:, clean_run.mask == 1])
                or np.any(corruption_masks["uy"][:, clean_run.mask == 1])
            ),
            "final_corrupted_cells": {
                "ux": int(np.count_nonzero(corruption_masks["ux"])),
                "uy": int(np.count_nonzero(corruption_masks["uy"])),
            },
        },
        "traceability": {
            "source_clean_metadata": source_metadata,
            "noise_specs": [
                {
                    "kind": spec.kind,
                    "intensity": float(spec.intensity),
                    "seed": spec.seed,
                    "channels": list(spec.channels),
                    "include_obstacle": spec.include_obstacle,
                    "params": dict(spec.params),
                }
                for spec in noise_specs
            ],
        },
    }
    return metadata


def create_noisy_dataset(
    clean_run_dir: Path | str,
    *,
    noise_specs: Sequence[NoiseSpec | Mapping[str, Any]],
    noisy_root: Path | str = "data/noisy",
    experiment_id: str | None = None,
    overwrite: bool = False,
) -> Path:
    clean_run_dir = Path(clean_run_dir).resolve()
    validate_clean_dataset(clean_run_dir)
    clean_run = load_clean_run(clean_run_dir)
    normalized_specs = _normalize_noise_specs(noise_specs)

    if experiment_id is None:
        experiment_id = build_noisy_experiment_id(clean_run.metadata, normalized_specs)

    noisy_root = Path(noisy_root).resolve()
    noisy_run_dir = noisy_root / experiment_id
    if noisy_run_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Noisy dataset already exists: {noisy_run_dir}. Use overwrite=True to replace it."
        )
    noisy_run_dir.mkdir(parents=True, exist_ok=True)

    arrays = {
        "ux": np.asarray(clean_run.ux, dtype=np.float64).copy(),
        "uy": np.asarray(clean_run.uy, dtype=np.float64).copy(),
    }
    corruption_masks = {
        "ux": np.zeros_like(clean_run.ux, dtype=bool),
        "uy": np.zeros_like(clean_run.uy, dtype=bool),
    }
    stage_summaries: list[dict[str, Any]] = []

    for stage_index, spec in enumerate(normalized_specs):
        before_stats = {
            channel: _channel_stats(arrays[channel], clean_run.mask == 0)
            for channel in TARGET_CHANNELS
        }
        arrays, stage_masks, stage_summary = _apply_noise_stage(
            arrays,
            spec=spec,
            obstacle_mask=clean_run.mask,
        )
        for channel in TARGET_CHANNELS:
            corruption_masks[channel] |= stage_masks[channel]
        after_stats = {
            channel: _channel_stats(arrays[channel], clean_run.mask == 0)
            for channel in TARGET_CHANNELS
        }
        stage_summary["stage_index"] = stage_index
        stage_summary["before"] = before_stats
        stage_summary["after"] = after_stats
        stage_summaries.append(stage_summary)

    speed = np.sqrt(arrays["ux"] * arrays["ux"] + arrays["uy"] * arrays["uy"])
    vorticity = _compute_vorticity(arrays["ux"], arrays["uy"])

    if not any(spec.include_obstacle for spec in normalized_specs):
        obstacle = clean_run.mask == 1
        arrays["ux"][:, obstacle] = clean_run.ux[:, obstacle]
        arrays["uy"][:, obstacle] = clean_run.uy[:, obstacle]
        speed[:, obstacle] = clean_run.speed[:, obstacle]
        vorticity[:, obstacle] = clean_run.vorticity[:, obstacle]

    metadata = _build_metadata(
        experiment_id=experiment_id,
        clean_run=clean_run,
        noise_specs=normalized_specs,
        stage_summaries=stage_summaries,
        corruption_masks=corruption_masks,
    )

    np.savez_compressed(
        noisy_run_dir / "fields.npz",
        steps=np.asarray(clean_run.steps, dtype=np.int32),
        ux=np.asarray(arrays["ux"], dtype=np.float64),
        uy=np.asarray(arrays["uy"], dtype=np.float64),
        speed=np.asarray(speed, dtype=np.float64),
        vorticity=np.asarray(vorticity, dtype=np.float64),
        mask=np.asarray(clean_run.mask, dtype=np.uint8),
        corruption_ux_mask=np.asarray(corruption_masks["ux"], dtype=np.uint8),
        corruption_uy_mask=np.asarray(corruption_masks["uy"], dtype=np.uint8),
    )
    (noisy_run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    validate_noisy_dataset(noisy_run_dir)
    return noisy_run_dir


def load_noisy_run(run_dir: Path | str) -> NoisyRun:
    run_dir = Path(run_dir).resolve()
    metadata_path = run_dir / "metadata.json"
    fields_path = run_dir / "fields.npz"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
    if not fields_path.exists():
        raise FileNotFoundError(f"Missing NPZ file: {fields_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(fields_path) as data:
        return NoisyRun(
            run_dir=run_dir,
            metadata=metadata,
            steps=np.asarray(data["steps"]),
            ux=np.asarray(data["ux"]),
            uy=np.asarray(data["uy"]),
            speed=np.asarray(data["speed"]),
            vorticity=np.asarray(data["vorticity"]),
            mask=np.asarray(data["mask"]),
            corruption_ux_mask=np.asarray(data["corruption_ux_mask"]),
            corruption_uy_mask=np.asarray(data["corruption_uy_mask"]),
        )


def validate_noisy_dataset(run_dir: Path | str, *, speed_tolerance: float = 1e-6) -> NoisyDatasetSummary:
    run = load_noisy_run(run_dir)
    metadata = run.metadata

    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported noisy dataset schema version")

    nx = int(metadata["grid"]["nx"])
    ny = int(metadata["grid"]["ny"])
    expected_steps = list(metadata["snapshots"]["steps"])
    expected_num_snapshots = int(metadata["snapshots"]["num_snapshots"])

    if metadata["variables"]["dynamic"] != list(DYNAMIC_VARIABLES):
        raise ValueError("Unexpected dynamic variable declaration in noisy metadata")
    if metadata["variables"]["static"] != list(STATIC_VARIABLES):
        raise ValueError("Unexpected static variable declaration in noisy metadata")
    if metadata["variables"]["diagnostic"] != list(DIAGNOSTIC_VARIABLES):
        raise ValueError("Unexpected diagnostic variable declaration in noisy metadata")

    if run.steps.ndim != 1:
        raise ValueError("steps must be a 1D array")
    if list(run.steps.tolist()) != expected_steps:
        raise ValueError("steps array does not match metadata")
    if run.steps.shape[0] != expected_num_snapshots:
        raise ValueError("Number of noisy snapshots does not match metadata")

    expected_dynamic_shape = (expected_num_snapshots, ny, nx)
    for name, array in {
        "ux": run.ux,
        "uy": run.uy,
        "speed": run.speed,
        "vorticity": run.vorticity,
    }.items():
        if array.shape != expected_dynamic_shape:
            raise ValueError(f"{name} has unexpected shape: {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains non-finite values")

    if run.mask.shape != (ny, nx):
        raise ValueError(f"mask has unexpected shape: {run.mask.shape}")
    if not np.all(np.isin(run.mask, [0, 1])):
        raise ValueError("mask contains values other than 0 or 1")

    for name, array in {
        "corruption_ux_mask": run.corruption_ux_mask,
        "corruption_uy_mask": run.corruption_uy_mask,
    }.items():
        if array.shape != expected_dynamic_shape:
            raise ValueError(f"{name} has unexpected shape: {array.shape}")
        if not np.all(np.isin(array, [0, 1])):
            raise ValueError(f"{name} must be binary")

    expected_speed = np.sqrt(run.ux * run.ux + run.uy * run.uy)
    if not np.allclose(run.speed, expected_speed, atol=speed_tolerance, rtol=0.0):
        raise ValueError("speed array is inconsistent with noisy ux and uy")

    source_clean_dir = Path(metadata["source"]["clean_run_dir"]).resolve()
    validate_clean_dataset(source_clean_dir)
    clean_run = load_clean_run(source_clean_dir)

    if list(clean_run.steps.tolist()) != expected_steps:
        raise ValueError("Noisy dataset steps do not match source clean dataset")
    if clean_run.mask.shape != run.mask.shape or not np.array_equal(clean_run.mask, run.mask):
        raise ValueError("Noisy dataset mask does not match source clean dataset")

    obstacle_touched_from_masks = bool(
        np.any(run.corruption_ux_mask[:, run.mask == 1])
        or np.any(run.corruption_uy_mask[:, run.mask == 1])
    )
    obstacle_touched = bool(metadata["corruption"]["obstacle_touched"])
    any_stage_allows_obstacle = any(
        bool(stage["include_obstacle"]) for stage in metadata["corruption"]["pipeline"]
    )
    if obstacle_touched != obstacle_touched_from_masks:
        raise ValueError("Obstacle corruption flag does not match corruption masks")
    if not any_stage_allows_obstacle:
        if obstacle_touched_from_masks:
            raise ValueError("Obstacle cells were corrupted without explicit permission")
        obstacle = run.mask == 1
        if not np.allclose(run.ux[:, obstacle], clean_run.ux[:, obstacle], atol=0.0, rtol=0.0):
            raise ValueError("ux changed inside obstacle despite protected obstacle policy")
        if not np.allclose(run.uy[:, obstacle], clean_run.uy[:, obstacle], atol=0.0, rtol=0.0):
            raise ValueError("uy changed inside obstacle despite protected obstacle policy")

    final_corrupted_cells = metadata["corruption"]["final_corrupted_cells"]
    if int(final_corrupted_cells["ux"]) != int(np.count_nonzero(run.corruption_ux_mask)):
        raise ValueError("corruption_ux_mask count does not match metadata")
    if int(final_corrupted_cells["uy"]) != int(np.count_nonzero(run.corruption_uy_mask)):
        raise ValueError("corruption_uy_mask count does not match metadata")

    return NoisyDatasetSummary(
        run_dir=run.run_dir,
        experiment_id=str(metadata["experiment_id"]),
        nx=nx,
        ny=ny,
        num_snapshots=expected_num_snapshots,
        noise_kinds=tuple(str(stage["kind"]) for stage in metadata["corruption"]["pipeline"]),
        steps=tuple(int(step) for step in run.steps.tolist()),
    )


def summarize_noisy_run(run_dir: Path | str) -> dict[str, Any]:
    summary = validate_noisy_dataset(run_dir)
    metadata = json.loads((Path(run_dir).resolve() / "metadata.json").read_text(encoding="utf-8"))
    return {
        "experiment_id": summary.experiment_id,
        "dimensions": {"nx": summary.nx, "ny": summary.ny},
        "num_snapshots": summary.num_snapshots,
        "noise_kinds": list(summary.noise_kinds),
        "steps": list(summary.steps),
        "source_experiment_id": metadata["source"]["source_experiment_id"],
        "final_corrupted_cells": metadata["corruption"]["final_corrupted_cells"],
    }


def save_noisy_comparison_figure(
    noisy_run_dir: Path | str,
    *,
    step: int | None = None,
    variable: str = "ux",
    output_path: Path | str | None = None,
) -> Path:
    variable = variable.strip().lower()
    if variable not in DYNAMIC_VARIABLES:
        raise ValueError(f"variable must be one of {DYNAMIC_VARIABLES}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    noisy_run = load_noisy_run(noisy_run_dir)
    clean_run = load_clean_run(Path(noisy_run.metadata["source"]["clean_run_dir"]))

    if step is None:
        step = int(noisy_run.steps[-1])
    step_matches = np.where(noisy_run.steps == step)[0]
    if step_matches.size == 0:
        raise ValueError(f"Requested step {step} was not found in the noisy dataset")
    index = int(step_matches[0])

    clean_array = getattr(clean_run, variable)[index]
    noisy_array = getattr(noisy_run, variable)[index]
    difference = noisy_array - clean_array

    if variable == "ux":
        corruption_mask = noisy_run.corruption_ux_mask[index]
        cmap = "coolwarm"
    elif variable == "uy":
        corruption_mask = noisy_run.corruption_uy_mask[index]
        cmap = "coolwarm"
    elif variable == "speed":
        corruption_mask = np.logical_or(
            noisy_run.corruption_ux_mask[index],
            noisy_run.corruption_uy_mask[index],
        )
        cmap = "viridis"
    else:
        corruption_mask = np.logical_or(
            noisy_run.corruption_ux_mask[index],
            noisy_run.corruption_uy_mask[index],
        )
        cmap = "RdBu_r"

    if output_path is None:
        output = Path(noisy_run_dir).resolve() / f"compare_{variable}_t{step:06d}.png"
    else:
        output = Path(output_path).resolve()

    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    panels = [
        (clean_array, f"clean {variable}", cmap),
        (noisy_array, f"noisy {variable}", cmap),
        (difference, f"delta {variable}", "RdBu_r"),
        (corruption_mask.astype(np.uint8), "corruption mask", "gray_r"),
    ]

    for axis, (panel_data, title, panel_cmap) in zip(axes.flat, panels):
        image = axis.imshow(panel_data, origin="lower", cmap=panel_cmap, aspect="auto")
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        figure.colorbar(image, ax=axis, shrink=0.8)

    figure.suptitle(f"FASE 3 comparison at step t{step:06d}")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return output
