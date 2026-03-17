# FASE 5 Baseline Models

## Scope

FASE 5 introduces classical baseline methods behind one shared interface:

- `fit(...)`
- `reconstruct(...)`
- `get_params()`
- `get_name()`

No deep learning is included in this stage.

## Implemented methods

- `rpca_ialm`
  - expects `space_time_matrix`
  - returns the low-rank component as reconstruction
- `dmd`
  - expects `space_time_matrix`
  - reconstructs a low-rank dynamical approximation from sequential snapshots
- `truncated_svd`
  - expects `space_time_matrix`
  - returns a rank-truncated low-rank approximation
- `median_filter`
  - expects grid `snapshot_batch`
  - applies a spatial median filter independently per snapshot and channel
- `wiener_filter`
  - expects grid `snapshot_batch`
  - applies a local adaptive Wiener filter independently per snapshot and channel

## Separation of responsibilities

- `phase5_baseline_impl.py`
  - numerical implementations of the algorithms
- `phase5_baseline_wrappers.py`
  - common interface, warnings, and mapping to FASE 4 packed inputs
- `phase5_baseline_io.py`
  - config loading, pipeline execution, and persistence of reconstructed outputs
- `phase5_baselines.py`
  - public facade for the stage

## Saved outputs

Each reconstruction is stored under:

```text
data/processed/baselines/<reconstruction_id>/
  reconstruction.npz
  metadata.json
```

`reconstruction.npz` stores:

- `steps`
- `mask`
- `observed`
- `reconstructed`
- `reference` when provided in the config

## Honest limitations

- RPCA, DMD, and truncated SVD work on matrix representations and assume useful low-rank structure.
- DMD needs temporal ordering and can become numerically weak with very few snapshots.
- Median and Wiener filters are local spatial baselines; they do not exploit long-range temporal structure.
- Wiener filtering assumes locally stationary additive noise, which may not hold for block corruption or strong outliers.
- None of these methods automatically imply physically valid reconstructions.

## Minimal usage

Edit `configs/baseline_phase5_example.json` so `source.run_dir` points to an existing noisy dataset, then run:

```bash
python python/scripts/run_baseline_example.py --config configs/baseline_phase5_example.json
```
