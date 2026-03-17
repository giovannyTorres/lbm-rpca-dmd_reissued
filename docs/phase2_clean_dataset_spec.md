# FASE 2 Clean Dataset Specification

## Recommended persistence format

Recommended format: `NPZ` with `np.savez_compressed`, plus a human-readable `metadata.json` sidecar.

Why `NPZ` now:

- Native and simple for Python and NumPy workflows.
- Easy to version, inspect, and move without introducing HDF5-specific runtime complexity.
- A good fit for the current scope: single-run clean datasets with dense arrays and modest size.
- Keeps the clean dataset separated from raw CSV while remaining directly traceable to the original run.

`HDF5` can be reconsidered later if the project grows into larger multi-run collections or needs partial I/O at scale.

## Clean dataset layout

Each clean run lives under:

```text
data/clean/<experiment_id>/
  fields.npz
  metadata.json
```

## Naming convention

Directory naming convention:

```text
clean_lbm2d_d2q9bgk_cylinder_re<Re>_uin<u_in>_nx<nx>_ny<ny>_r<obstacle_r>_seed-<seed|none>_<run_id>
```

This keeps the dataset identifiable from its main physical and grid parameters while preserving traceability to the original solver `run_id`.

## Metadata schema

`metadata.json` stores:

- `schema_version`
- `dataset_kind`
- `storage_format`
- `experiment_id`
- `source`
  - raw run directory
  - raw manifest path
  - config path
  - source `run_id`
  - seed and seed mode
- `grid`
- `snapshots`
- `variables`
- `parameters`
- `traceability`
  - full source manifest
  - full source config

This makes every clean dataset traceable back to the exact simulation configuration. If a seed is absent, metadata records that explicitly as deterministic-without-seed.

## Arrays inside `fields.npz`

- `steps`: shape `(T,)`
- `ux`: shape `(T, ny, nx)`
- `uy`: shape `(T, ny, nx)`
- `speed`: shape `(T, ny, nx)`
- `vorticity`: shape `(T, ny, nx)`
- `mask`: shape `(ny, nx)`

`mask` is stored once because it is static for the run.

## Validation rules

The clean dataset validator checks:

- required files exist
- schema version matches
- metadata and arrays agree on dimensions and snapshot count
- all dynamic arrays are finite
- `mask` is binary
- `speed == sqrt(ux^2 + uy^2)` within tolerance
- traceability fields are internally consistent
