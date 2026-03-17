# FASE 6 Benchmark Runner

## Purpose

FASE 6 adds a sequential benchmark runner that expands a JSON sweep, reuses the artifacts from FASE 1-5, computes comparable metrics, and writes a resumable experiment ledger.

## Public API

- `load_benchmark_config(...)`
- `expand_benchmark_cases(...)`
- `compute_benchmark_metrics(...)`
- `run_benchmark(...)`
- `run_benchmark_from_config(...)`

## Sweep dimensions

The benchmark expands the Cartesian product of:

- `solver.resolutions`
- `noise` cases
- `models`
- `models[*].param_grid`

Resolution changes scale `obstacle_cx`, `obstacle_cy`, and `obstacle_r` proportionally from the base solver config.

## Persistence

Per benchmark:

```text
results/metrics/<benchmark_id>/
  benchmark_config_snapshot.json
  ledger.jsonl
  generated_solver_configs/
  experiments/<experiment_id>/
    reconstruction.npz
    metadata.json
    benchmark_result.json

results/tables/<benchmark_id>/
  summary.csv
  summary.parquet   # optional
```

`ledger.jsonl` is append-only and supports resuming interrupted runs.

## Metrics

Computed on fluid cells only:

- `rmse`
- `mae`
- `relative_l2_error`
- `psnr`
- `vorticity_rmse`
- `kinetic_energy_relative_l2`
- `divergence_residual_rms`
- `reconstruction_time_sec`
- `total_time_sec`
- `estimated_memory_bytes`

Physical metrics are stored as `null` when the reconstruction does not include both `ux` and `uy`.

## Important limits

- v1 is JSON-only; YAML is intentionally deferred.
- The runner is sequential by design.
- Memory is approximate and only counts visible NumPy buffers inside the Python runner.
- PSNR is stored as `null` when the reference dynamic range or the error degenerates numerically.

## Minimal usage

```bash
python python/scripts/run_benchmark_example.py --config configs/benchmark_phase6_example.json
```
