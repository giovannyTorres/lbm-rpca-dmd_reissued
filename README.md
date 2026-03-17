# lbm-rpca-dmd_reissued

FASE 1 now contains a minimal viable LBM solver for 2D D2Q9 BGK flow around a cylinder.

FASE 2 adds a clean dataset standardization layer on top of the solver outputs.

FASE 3 adds a configurable and reproducible artificial noise module that derives noisy datasets from the clean datasets, without introducing denoising models yet.

FASE 4 adds the model-facing data contract and preprocessing layer, with explicit packing rules for snapshots, temporal windows, and space-time matrices.

FASE 5 adds classical baseline denoising models behind a shared interface and saves reconstructed outputs with metadata for later comparison.

FASE 6 adds a sequential experiment runner and benchmark table exporter on top of the earlier phases.

FASE 7 adds an automatic visual results pipeline that turns benchmark outputs into figures, tables, and a traceable catalog for thesis-ready assets.

## What is included

- Modular C++ solver in `cpp/lbm_core`
- JSON experiment config in `configs/lbm_cylinder_base.json`
- Python runner in `python/scripts/run_phase1_example.py`
- Snapshot visualizer in `python/scripts/visualize_phase1_snapshot.py`
- Smoke test in `tests/test_phase1_smoke.py`
- Output validator in `python/src/fluid_denoise/phase1_validation.py`
- Solver notes in `docs/phase1_solver_assumptions_limits_risks.md`
- Clean dataset tools in `python/src/fluid_denoise/phase2_clean_dataset.py`
- Clean dataset spec in `docs/phase2_clean_dataset_spec.md`
- Noisy dataset tools in `python/src/fluid_denoise/phase3_noisy_dataset.py`
- Noisy dataset spec in `docs/phase3_noisy_dataset_spec.md`
- Model data contract tools in `python/src/fluid_denoise/phase4_model_data.py`
- Model data contract doc in `docs/phase4_model_data_contract.md`
- Baseline model tools in `python/src/fluid_denoise/phase5_baselines.py`
- Baseline model doc in `docs/phase5_baseline_models.md`
- Benchmark runner tools in `python/src/fluid_denoise/phase6_benchmark.py`
- Benchmark runner doc in `docs/phase6_benchmark_runner.md`
- Visual results tools in `python/src/fluid_denoise/phase7_visual_results.py`
- Visual results doc in `docs/phase7_visual_results_pipeline.md`

## Prerequisites

- Python 3.10+
- CMake 3.16+
- A C++17 compiler available to CMake
- `numpy` and `matplotlib` for visualization

The runner and smoke test perform a clear preflight check for `cmake` in `PATH`.
The runner also validates the generated output directory automatically after each run.

## Build and run

Build and run the example case:

```bash
python python/scripts/run_phase1_example.py --config configs/lbm_cylinder_base.json
```

Manual CMake flow:

```bash
cmake -S cpp/lbm_core -B cpp/lbm_core/build
cmake --build cpp/lbm_core/build --config Release --parallel
cpp/lbm_core/build/lbm_sim --config configs/lbm_cylinder_base.json
```

On Windows with a multi-config generator, the binary may be placed under `cpp/lbm_core/build/Release/lbm_sim.exe`.

## Configuration

The solver accepts a flat JSON file through:

```text
lbm_sim --config <config.json>
```

Supported keys:

- `nx`, `ny`
- `reynolds`, `u_in`
- `iterations`, `save_stride`
- `obstacle_cx`, `obstacle_cy`, `obstacle_r`
- `output_root`, `run_id`

Example:

```json
{
  "nx": 220,
  "ny": 80,
  "reynolds": 150.0,
  "u_in": 0.06,
  "iterations": 1200,
  "save_stride": 200,
  "obstacle_cx": 55,
  "obstacle_cy": 40,
  "obstacle_r": 8,
  "output_root": "data/raw",
  "run_id": "phase1_cylinder_re150"
}
```

## Outputs

Each run writes to `data/raw/<run_id>/` and produces:

- `manifest.json`
- `ux_tXXXXXX.csv`
- `uy_tXXXXXX.csv`
- `speed_tXXXXXX.csv`
- `vorticity_tXXXXXX.csv`
- `mask_tXXXXXX.csv`

Snapshots are written at `t=0`, every `save_stride`, and at the final iteration. Each CSV has shape `ny x nx`.

## Visualization

Generate a basic PNG summary for one saved step:

```bash
python python/scripts/visualize_phase1_snapshot.py --run-dir data/raw/phase1_cylinder_re150 --step 1200
```

This writes `snapshot_t001200.png` inside the run directory by default.

## Test

Run the smoke test:

```bash
python -m unittest tests.test_phase1_smoke
```

The test builds the solver, runs a short simulation, and checks that the expected files exist and contain valid numeric data.

## Solver Safety Notes

Short assumptions, limits, risks, and safe starter parameters are documented in `docs/phase1_solver_assumptions_limits_risks.md`.

## Clean Dataset

Recommended clean dataset format: compressed `NPZ` plus `metadata.json`.
The detailed specification, metadata schema, naming convention, and validation rules are documented in `docs/phase2_clean_dataset_spec.md`.

Generate a small clean dataset example:

```bash
python python/scripts/generate_clean_dataset_example.py
```

Inspect a clean dataset:

```bash
python python/scripts/inspect_clean_run.py --run-dir data/clean/<experiment_id>
```

Validate a clean dataset:

```bash
python python/scripts/validate_clean_dataset.py --run-dir data/clean/<experiment_id>
```

## Noisy Dataset

Recommended noisy dataset format: compressed `NPZ` plus `metadata.json`, aligned with the clean dataset layout and extended with corruption masks and pipeline metadata.
The detailed specification is documented in `docs/phase3_noisy_dataset_spec.md`.

Generate a noisy dataset example:

```bash
python python/scripts/generate_noisy_dataset_example.py
```

Inspect a noisy dataset:

```bash
python python/scripts/inspect_noisy_run.py --run-dir data/noisy/<experiment_id>
```

Validate a noisy dataset:

```bash
python python/scripts/validate_noisy_dataset.py --run-dir data/noisy/<experiment_id>
```

Create a clean-vs-noisy comparison figure:

```bash
python python/scripts/visualize_noisy_comparison.py --run-dir data/noisy/<experiment_id> --variable ux --step 1200
```

## Model Data Contract

The model-facing preprocessing API is documented in `docs/phase4_model_data_contract.md`.
It defines three compatible processing units instead of a single forced representation:

- `SnapshotBatch` for frame-wise methods
- `TemporalWindowBatch` for local temporal methods
- `SpaceTimeMatrix` for RPCA, DMD, and related matrix-based methods

## Baseline Models

FASE 5 adds the following baseline methods with a common interface:

- `rpca_ialm`
- `dmd`
- `truncated_svd`
- `median_filter`
- `wiener_filter`

Run a baseline reconstruction from a config file:

```bash
python python/scripts/run_baseline_example.py --config configs/baseline_phase5_example.json
```

Inspect a saved reconstruction:

```bash
python python/scripts/inspect_baseline_reconstruction.py --run-dir data/processed/baselines/<reconstruction_id>
```

## Benchmark Runner

FASE 6 adds a JSON-configured benchmark runner with resumable execution, experiment ledgering, and CSV/Parquet summary export.
The benchmark interface is documented in `docs/phase6_benchmark_runner.md`.

Run the example benchmark:

```bash
python python/scripts/run_benchmark_example.py --config configs/benchmark_phase6_example.json
```

## Visual Results Pipeline

FASE 7 consumes a completed benchmark and produces:

- per-experiment clean / noisy / reconstructed figures
- error maps and temporal error series
- aggregate comparisons by resolution and noise case
- compute-time bar charts and performance heatmaps
- summary tables in CSV and LaTeX
- a figure catalog plus a `thesis_ready/` asset folder

Run the example visual pipeline after FASE 6:

```bash
python python/scripts/run_visual_results_example.py --config configs/visual_results_phase7_example.json
```
