from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fluid_denoise.phase1_validation import validate_run_outputs


SCHEMA_VERSION = "phase2.clean.v1"
DYNAMIC_VARIABLES = ("ux", "uy", "speed", "vorticity")
STATIC_VARIABLES = ("mask",)
ALL_VARIABLES = DYNAMIC_VARIABLES + STATIC_VARIABLES


@dataclass(frozen=True)
class CleanRun:
    run_dir: Path
    metadata: dict[str, Any]
    steps: np.ndarray
    ux: np.ndarray
    uy: np.ndarray
    speed: np.ndarray
    vorticity: np.ndarray
    mask: np.ndarray

    @property
    def variables(self) -> tuple[str, ...]:
        return ALL_VARIABLES

    @property
    def dimensions(self) -> tuple[int, int]:
        return (int(self.metadata["grid"]["ny"]), int(self.metadata["grid"]["nx"]))

    @property
    def num_snapshots(self) -> int:
        return int(self.steps.shape[0])


@dataclass(frozen=True)
class CleanDatasetSummary:
    run_dir: Path
    experiment_id: str
    nx: int
    ny: int
    num_snapshots: int
    variables: tuple[str, ...]
    steps: tuple[int, ...]


def _read_csv_matrix(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"Empty CSV file: {path}")
    return np.asarray([[float(value) for value in row] for row in rows], dtype=np.float64)


def _sanitize_token(value: str) -> str:
    sanitized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return sanitized or "unknown"


def _normalize_seed(config_payload: dict[str, Any]) -> tuple[Any, str]:
    if "seed" not in config_payload:
        return None, "deterministic_without_seed"
    return config_payload["seed"], "configured_seed"


def build_experiment_id(
    source_manifest: dict[str, Any],
    source_config: dict[str, Any],
) -> str:
    reynolds = int(round(float(source_manifest["reynolds"])))
    u_in = int(round(float(source_manifest["u_in"]) * 10000.0))
    nx = int(source_manifest["nx"])
    ny = int(source_manifest["ny"])
    obstacle_r = int(source_manifest["obstacle_r"])
    run_id = _sanitize_token(str(source_manifest["run_id"]))
    seed, _ = _normalize_seed(source_config)
    seed_token = f"seed-{seed}" if seed is not None else "seed-none"
    return (
        f"clean_lbm2d_d2q9bgk_cylinder_re{reynolds:04d}_"
        f"uin{u_in:04d}_nx{nx:04d}_ny{ny:04d}_r{obstacle_r:03d}_"
        f"{seed_token}_{run_id}"
    )


def _build_metadata(
    *,
    experiment_id: str,
    source_run_dir: Path,
    source_manifest: dict[str, Any],
    source_config: dict[str, Any],
    steps: list[int],
) -> dict[str, Any]:
    seed, seed_mode = _normalize_seed(source_config)
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_kind": "clean_lbm_run",
        "storage_format": {
            "recommended": "npz",
            "npz_file": "fields.npz",
            "metadata_file": "metadata.json",
            "compression": "np.savez_compressed",
        },
        "experiment_id": experiment_id,
        "source": {
            "run_dir": str(source_run_dir.resolve()),
            "manifest_path": str((source_run_dir / "manifest.json").resolve()),
            "config_path": source_manifest.get("config_path"),
            "run_id": source_manifest["run_id"],
            "seed": seed,
            "seed_mode": seed_mode,
        },
        "grid": {
            "nx": int(source_manifest["nx"]),
            "ny": int(source_manifest["ny"]),
        },
        "snapshots": {
            "num_snapshots": len(steps),
            "steps": steps,
            "first_step": steps[0],
            "last_step": steps[-1],
            "save_stride": int(source_manifest["save_stride"]),
            "iterations": int(source_manifest["iterations"]),
        },
        "variables": {
            "dynamic": list(DYNAMIC_VARIABLES),
            "static": list(STATIC_VARIABLES),
        },
        "parameters": {
            "reynolds": float(source_manifest["reynolds"]),
            "u_in": float(source_manifest["u_in"]),
            "obstacle_cx": int(source_manifest["obstacle_cx"]),
            "obstacle_cy": int(source_manifest["obstacle_cy"]),
            "obstacle_r": int(source_manifest["obstacle_r"]),
            "nu": float(source_manifest["nu"]),
            "tau": float(source_manifest["tau"]),
        },
        "traceability": {
            "source_manifest": source_manifest,
            "source_config": source_config,
        },
    }


def convert_raw_run_to_clean_dataset(
    raw_run_dir: Path | str,
    *,
    clean_root: Path | str = "data/clean",
    experiment_id: str | None = None,
    overwrite: bool = False,
) -> Path:
    raw_run_dir = Path(raw_run_dir).resolve()
    summary = validate_run_outputs(raw_run_dir)
    manifest_path = raw_run_dir / "manifest.json"
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    config_path = source_manifest.get("config_path")
    source_config: dict[str, Any] = {}
    if config_path:
        config_candidate = Path(config_path)
        if config_candidate.exists():
            source_config = json.loads(config_candidate.read_text(encoding="utf-8"))

    if experiment_id is None:
        experiment_id = build_experiment_id(source_manifest, source_config)

    clean_root = Path(clean_root).resolve()
    clean_run_dir = clean_root / experiment_id
    if clean_run_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Clean dataset already exists: {clean_run_dir}. Use overwrite=True to replace it."
        )
    clean_run_dir.mkdir(parents=True, exist_ok=True)

    steps = list(summary.steps)
    stacked_fields: dict[str, list[np.ndarray]] = {name: [] for name in DYNAMIC_VARIABLES}
    reference_mask: np.ndarray | None = None

    for step in steps:
        suffix = f"t{step:06d}"
        ux = _read_csv_matrix(raw_run_dir / f"ux_{suffix}.csv")
        uy = _read_csv_matrix(raw_run_dir / f"uy_{suffix}.csv")
        speed = _read_csv_matrix(raw_run_dir / f"speed_{suffix}.csv")
        vorticity = _read_csv_matrix(raw_run_dir / f"vorticity_{suffix}.csv")
        mask = _read_csv_matrix(raw_run_dir / f"mask_{suffix}.csv").astype(np.uint8)

        if reference_mask is None:
            reference_mask = mask
        elif not np.array_equal(reference_mask, mask):
            raise ValueError("Mask changed across snapshots; expected a static obstacle mask")

        stacked_fields["ux"].append(ux)
        stacked_fields["uy"].append(uy)
        stacked_fields["speed"].append(speed)
        stacked_fields["vorticity"].append(vorticity)

    assert reference_mask is not None

    arrays = {
        "steps": np.asarray(steps, dtype=np.int32),
        "ux": np.stack(stacked_fields["ux"], axis=0),
        "uy": np.stack(stacked_fields["uy"], axis=0),
        "speed": np.stack(stacked_fields["speed"], axis=0),
        "vorticity": np.stack(stacked_fields["vorticity"], axis=0),
        "mask": reference_mask,
    }

    metadata = _build_metadata(
        experiment_id=experiment_id,
        source_run_dir=raw_run_dir,
        source_manifest=source_manifest,
        source_config=source_config,
        steps=steps,
    )

    np.savez_compressed(clean_run_dir / "fields.npz", **arrays)
    (clean_run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    validate_clean_dataset(clean_run_dir)
    return clean_run_dir


def load_clean_run(run_dir: Path | str) -> CleanRun:
    run_dir = Path(run_dir).resolve()
    metadata_path = run_dir / "metadata.json"
    fields_path = run_dir / "fields.npz"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
    if not fields_path.exists():
        raise FileNotFoundError(f"Missing NPZ file: {fields_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(fields_path) as data:
        run = CleanRun(
            run_dir=run_dir,
            metadata=metadata,
            steps=np.asarray(data["steps"]),
            ux=np.asarray(data["ux"]),
            uy=np.asarray(data["uy"]),
            speed=np.asarray(data["speed"]),
            vorticity=np.asarray(data["vorticity"]),
            mask=np.asarray(data["mask"]),
        )
    return run


def validate_clean_dataset(run_dir: Path | str, *, speed_tolerance: float = 1e-6) -> CleanDatasetSummary:
    run = load_clean_run(run_dir)
    metadata = run.metadata

    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported clean dataset schema version")

    nx = int(metadata["grid"]["nx"])
    ny = int(metadata["grid"]["ny"])
    expected_steps = list(metadata["snapshots"]["steps"])
    expected_num_snapshots = int(metadata["snapshots"]["num_snapshots"])
    if metadata["variables"]["dynamic"] != list(DYNAMIC_VARIABLES):
        raise ValueError("Unexpected dynamic variable declaration in metadata")
    if metadata["variables"]["static"] != list(STATIC_VARIABLES):
        raise ValueError("Unexpected static variable declaration in metadata")

    if run.steps.ndim != 1:
        raise ValueError("steps must be a 1D array")
    if list(run.steps.tolist()) != expected_steps:
        raise ValueError("steps array does not match metadata")
    if run.steps.shape[0] != expected_num_snapshots:
        raise ValueError("Number of snapshots does not match metadata")

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

    expected_speed = np.sqrt(run.ux * run.ux + run.uy * run.uy)
    if not np.allclose(run.speed, expected_speed, atol=speed_tolerance, rtol=0.0):
        raise ValueError("speed array is inconsistent with ux and uy")

    source = metadata["source"]
    traceability = metadata["traceability"]
    if traceability["source_manifest"]["run_id"] != source["run_id"]:
        raise ValueError("source run_id mismatch inside metadata")

    return CleanDatasetSummary(
        run_dir=run.run_dir,
        experiment_id=str(metadata["experiment_id"]),
        nx=nx,
        ny=ny,
        num_snapshots=expected_num_snapshots,
        variables=ALL_VARIABLES,
        steps=tuple(int(step) for step in run.steps.tolist()),
    )


def summarize_clean_run(run_dir: Path | str) -> dict[str, Any]:
    summary = validate_clean_dataset(run_dir)
    metadata = json.loads((Path(run_dir).resolve() / "metadata.json").read_text(encoding="utf-8"))
    return {
        "experiment_id": summary.experiment_id,
        "dimensions": {
            "nx": summary.nx,
            "ny": summary.ny,
        },
        "num_snapshots": summary.num_snapshots,
        "variables": list(summary.variables),
        "parameters": metadata["parameters"],
        "steps": list(summary.steps),
    }
