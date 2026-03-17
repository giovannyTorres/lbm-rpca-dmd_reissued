# FASE 4 Model Data Contract

## Processing units

FASE 4 does not force a single universal representation.
Instead, it exposes three explicit units depending on the algorithm:

- `SnapshotBatch`: one sample per saved time step.
  Typical shape:
  - grid mode: `(N, C, ny, nx)`
  - flattened fluid mode: `(N, C, F)`
- `TemporalWindowBatch`: one sample per contiguous temporal window.
  Typical shape:
  - grid mode: `(N, W, C, ny, nx)`
  - flattened fluid mode: `(N, W, C, F)`
- `SpaceTimeMatrix`: matrix form for methods such as RPCA and DMD.
  Typical shape:
  - `(features, T)`
  - feature order: variable-major, then row-major spatial order

## Mask handling

Obstacle handling is always explicit through `mask_policy`:

- `keep`: preserve the full grid as stored
- `fill_obstacle`: preserve the full grid but overwrite obstacle cells with a configured fill value
- `flatten_fluid`: keep only fluid cells where `mask == 0`

For grid representations, `include_mask_channel=True` can append the obstacle mask as one extra channel.

## Normalization

Normalization is optional and configurable through `NormalizationSpec`:

- modes: `none`, `standardize`, `minmax`, `maxabs`
- scopes: `global`, `per_channel`, `per_feature`

The module exposes both forward and inverse transforms so the packed representation can be normalized and later mapped back to the original scale.

## Main API

- `load_model_run(...)`
- `align_model_runs(...)`
- `stack_run_variables(...)`
- `pack_snapshot_batch(...)` / `unpack_snapshot_batch(...)`
- `pack_temporal_windows(...)` / `unpack_temporal_windows(...)`
- `pack_space_time_matrix(...)` / `unpack_space_time_matrix(...)`
- `fit_normalization(...)`
- `apply_normalization(...)`
- `invert_normalization(...)`

This keeps the interface compatible with:

- RPCA and DMD through `SpaceTimeMatrix`
- classical spatial filters through grid snapshots/windows
- future paired clean/noisy workflows through `align_model_runs(...)`
