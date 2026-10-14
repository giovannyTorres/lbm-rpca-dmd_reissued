"""Explicit conversion from the phase-4 contract to research vector fields.

Phase 4 stores velocity components and integer solver steps, but does not
declare physical spacing, time units, boundary topology or density. Callers
must supply these declarations; a conversion does not validate the solver.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from fluid_denoise.phase4_model_data import ModelRun, ModelRunPair, stack_run_variables


@dataclass(frozen=True)
class ResearchRunPair:
    observed: dict[str, Any]
    reference: dict[str, Any]


def _declared_units(units: str | Mapping[str, str]) -> str | dict[str, str]:
    if isinstance(units, str):
        if not units.strip():
            raise ValueError("units must explicitly name the supplied unit system")
        return units.strip()
    if not isinstance(units, Mapping) or not {"length", "time", "velocity"} <= units.keys():
        raise ValueError("units mapping must declare length, time and velocity")
    if any(not isinstance(value, str) or not value.strip() for value in units.values()):
        raise ValueError("unit declarations must be nonempty strings")
    return dict(units)


def model_run_to_research(
    run: ModelRun,
    *,
    times: Sequence[float],
    dx: float,
    dy: float,
    units: str | Mapping[str, str],
    boundary: str,
    periodic_axes: Sequence[str],
    density: Any = None,
) -> dict[str, Any]:
    """Return velocity ``(T, 2, Y, X)`` without inferring missing physics.

    ``times`` must express saved times in the declared units; integer steps
    are retained separately for provenance. A periodic axis declares an
    endpoint-exclusive grid, while other axes use nodal quadrature. Density
    may be a positive scalar or broadcastable field; omitted density remains
    unknown and density-dependent research metrics are unavailable.
    """
    unit_system = _declared_units(units)
    if boundary not in {"periodic", "channel", "open"}:
        raise ValueError("boundary must be explicitly periodic, channel or open")
    if isinstance(periodic_axes, str):
        raise ValueError("periodic_axes must be a sequence containing x and/or y")
    axes = tuple(periodic_axes)
    if len(set(axes)) != len(axes) or any(axis not in {"x", "y"} for axis in axes):
        raise ValueError("periodic_axes must uniquely declare x and/or y")
    if boundary == "periodic" and set(axes) != {"x", "y"}:
        raise ValueError("periodic boundary requires both periodic axes")
    if not all(np.isfinite(value) and value > 0 for value in (dx, dy)):
        raise ValueError("dx and dy must be explicitly positive and finite")
    velocity, steps = stack_run_variables(run, variables=("ux", "uy"))
    mask = np.asarray(run.mask)
    if mask.ndim != 2 or not np.all(np.isin(mask, (0, 1))):
        raise ValueError("phase-4 mask must contain zero for fluid and one for solid")
    mask = mask.astype(bool)
    if velocity.shape != (run.num_snapshots, 2, *mask.shape):
        raise ValueError("phase-4 velocity channels do not agree with the mask")
    if min(mask.shape) < 3 or not (~mask).any():
        raise ValueError("research grids need at least three nodes per axis and fluid cells")
    if not np.isfinite(velocity[:, :, ~mask]).all():
        raise ValueError("velocity components must be finite at fluid nodes")
    time_values = np.asarray(times, dtype=float)
    if time_values.shape != (run.num_snapshots,) or not np.isfinite(time_values).all():
        raise ValueError("times must be a finite vector matching saved snapshots")
    if len(time_values) > 1 and np.any(np.diff(time_values) <= 0):
        raise ValueError("times must be strictly increasing")
    density_values = None
    if density is not None:
        if isinstance(unit_system, dict) and "density" not in unit_system:
            raise ValueError("density units must be declared when density is supplied")
        try:
            density_values = np.broadcast_to(
                np.asarray(density, dtype=float), (run.num_snapshots, *mask.shape)
            ).copy()
        except ValueError as error:
            raise ValueError("density must broadcast to (time, y, x)") from error
        if not np.isfinite(density_values[:, ~mask]).all() or np.any(density_values[:, ~mask] <= 0):
            raise ValueError("density must be positive and finite at fluid nodes")
    return {
        "velocity": velocity.copy(), "times": time_values.copy(), "steps": steps.copy(),
        "mask": mask.copy(), "density": density_values, "dx": float(dx), "dy": float(dy),
        "boundary": boundary,
        "metadata": {
            "units": unit_system, "periodic_axes": list(axes),
            "grid_convention": {
                axis: "endpoint_exclusive" if axis in axes else "nodal"
                for axis in ("x", "y")
            },
            "density_status": "unknown" if density is None else "caller_supplied",
            "reference_status": "phase4_export_not_dns_certified", "dns_certified": False,
            "source": {
                "schema_version": run.schema_version, "dataset_kind": run.dataset_kind,
                "experiment_id": run.experiment_id, "run_dir": str(run.run_dir),
                "solver_steps": steps.tolist(),
            },
            "conversion": "explicit caller declarations; no lattice-to-SI conversion inferred",
        },
    }


def model_pair_to_research(
    pair: ModelRunPair,
    *,
    times: Sequence[float],
    dx: float,
    dy: float,
    units: str | Mapping[str, str],
    boundary: str,
    periodic_axes: Sequence[str],
    reference_density: Any = None,
    observed_density: Any = None,
) -> ResearchRunPair:
    """Preserve phase-4 alignment and convert both vector-component runs.

    Densities are independently declared. Reference density is never silently
    copied into the observed run, and speed is never used to invent components.
    """
    if pair.observed.dimensions != pair.reference.dimensions:
        raise ValueError("observed and reference grids must agree")
    if not np.array_equal(pair.observed.mask, pair.reference.mask):
        raise ValueError("observed and reference masks must agree")
    if not np.array_equal(pair.observed.steps, pair.reference.steps):
        raise ValueError("observed and reference steps must agree")
    declarations = dict(times=times, dx=dx, dy=dy, units=units,
                        boundary=boundary, periodic_axes=periodic_axes)
    return ResearchRunPair(
        observed=model_run_to_research(pair.observed, density=observed_density, **declarations),
        reference=model_run_to_research(pair.reference, density=reference_density, **declarations),
    )
