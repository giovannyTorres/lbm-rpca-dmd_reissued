# FASE 3 Noisy Dataset Specification

## Layout

Each noisy run lives under:

```text
data/noisy/<experiment_id>/
  fields.npz
  metadata.json
```

## Stored arrays

- `steps`: shape `(T,)`
- `ux`: shape `(T, ny, nx)`
- `uy`: shape `(T, ny, nx)`
- `speed`: shape `(T, ny, nx)`, recomputed from noisy `ux` and `uy`
- `vorticity`: shape `(T, ny, nx)`, recomputed from noisy `ux` and `uy`
- `mask`: shape `(ny, nx)`
- `corruption_ux_mask`: shape `(T, ny, nx)`
- `corruption_uy_mask`: shape `(T, ny, nx)`

`corruption_*_mask` stores the final union of the cells touched by the configured corruption pipeline.

## Supported noise stages

- `salt_and_pepper`
- `gaussian`
- `speckle`
- `spatial_dropout`
- `missing_blocks`
- `piv_outliers`
- combinations through ordered pipelines of multiple stages

Every stage records:

- `kind`
- `intensity`
- `seed`
- `channels`
- `include_obstacle`
- `params`
- per-channel realized counts and summary statistics

## Obstacle policy

By default, corruption is applied only on fluid cells where `mask == 0`.
Obstacle cells remain untouched unless `include_obstacle=true` is set on a stage.

## Validation rules

The noisy dataset validator checks:

- schema version and required files
- metadata/array shape consistency
- finite dynamic arrays
- binary `mask` and corruption masks
- `speed == sqrt(ux^2 + uy^2)` within tolerance
- traceability back to the source clean dataset
- obstacle protection unless a stage explicitly enables obstacle corruption
