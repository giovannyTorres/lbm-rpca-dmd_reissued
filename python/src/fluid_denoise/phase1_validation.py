from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path


SNAPSHOT_FIELDS = ("ux", "uy", "speed", "vorticity", "mask")
SNAPSHOT_PATTERN = re.compile(
    r"^(ux|uy|speed|vorticity|mask)_t(?P<step>\d{6})\.csv$"
)


@dataclass(frozen=True)
class ValidationSummary:
    run_dir: Path
    steps: tuple[int, ...]
    nx: int
    ny: int


def _read_csv_matrix(path: Path) -> list[list[float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"Empty CSV file: {path}")
    return [[float(value) for value in row] for row in rows]


def _collect_steps(run_dir: Path) -> dict[str, set[int]]:
    step_map = {field: set() for field in SNAPSHOT_FIELDS}
    for path in run_dir.glob("*.csv"):
        match = SNAPSHOT_PATTERN.match(path.name)
        if not match:
            continue
        field = match.group(1)
        step_map[field].add(int(match.group("step")))
    return step_map


def validate_run_outputs(run_dir: Path | str, *, speed_tolerance: float = 1e-6) -> ValidationSummary:
    run_dir = Path(run_dir).resolve()
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    nx = int(manifest["nx"])
    ny = int(manifest["ny"])
    iterations = int(manifest["iterations"])
    save_stride = int(manifest["save_stride"])

    step_map = _collect_steps(run_dir)
    reference_steps = step_map["ux"]
    if not reference_steps:
        raise ValueError(f"No ux snapshots found in {run_dir}")

    for field in SNAPSHOT_FIELDS[1:]:
        if step_map[field] != reference_steps:
            raise ValueError(f"Mismatched snapshot steps for field '{field}'")

    if 0 not in reference_steps:
        raise ValueError("Missing snapshot at step 0")
    if iterations not in reference_steps:
        raise ValueError("Missing final snapshot")

    expected_count = len(range(0, iterations, save_stride)) + 1
    if len(reference_steps) < expected_count:
        raise ValueError("Fewer snapshots than expected from save_stride and iterations")

    for step in sorted(reference_steps):
        suffix = f"t{step:06d}"
        ux = _read_csv_matrix(run_dir / f"ux_{suffix}.csv")
        uy = _read_csv_matrix(run_dir / f"uy_{suffix}.csv")
        speed = _read_csv_matrix(run_dir / f"speed_{suffix}.csv")
        vorticity = _read_csv_matrix(run_dir / f"vorticity_{suffix}.csv")
        mask = _read_csv_matrix(run_dir / f"mask_{suffix}.csv")

        for field_name, matrix in {
            "ux": ux,
            "uy": uy,
            "speed": speed,
            "vorticity": vorticity,
            "mask": mask,
        }.items():
            if len(matrix) != ny or any(len(row) != nx for row in matrix):
                raise ValueError(f"Unexpected shape for {field_name} at step {step}")

        for field_name, matrix in {
            "ux": ux,
            "uy": uy,
            "speed": speed,
            "vorticity": vorticity,
        }.items():
            if not all(math.isfinite(value) for row in matrix for value in row):
                raise ValueError(f"Non-finite values in {field_name} at step {step}")

        mask_values = {int(value) for row in mask for value in row}
        if not mask_values.issubset({0, 1}):
            raise ValueError(f"Mask contains values other than 0/1 at step {step}")

        for row_index in range(ny):
            for col_index in range(nx):
                expected_speed = math.sqrt(
                    ux[row_index][col_index] ** 2 + uy[row_index][col_index] ** 2
                )
                if abs(speed[row_index][col_index] - expected_speed) > speed_tolerance:
                    raise ValueError(
                        f"Speed mismatch at step {step}, row {row_index}, col {col_index}"
                    )

    return ValidationSummary(
        run_dir=run_dir,
        steps=tuple(sorted(reference_steps)),
        nx=nx,
        ny=ny,
    )
