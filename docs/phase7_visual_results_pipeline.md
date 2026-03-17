# FASE 7 Visual Results Pipeline

## Purpose

FASE 7 turns a completed FASE 6 benchmark into reproducible figures and tables ready for technical analysis and thesis insertion.

The pipeline reads:

- `results/tables/<benchmark_id>/summary.csv`
- `results/metrics/<benchmark_id>/experiments/<experiment_id>/reconstruction.npz`
- `results/metrics/<benchmark_id>/experiments/<experiment_id>/metadata.json`

## Public API

- `load_visual_results_config(...)`
- `generate_visual_results(...)`
- `generate_visual_results_from_config(...)`

## Generated outputs

Per benchmark, the pipeline writes:

```text
results/visuals/<benchmark_id>/
  visual_results_config_snapshot.json
  catalog.json
  catalog.csv
  catalog.md
  exploratory/
    comparisons/
    error_maps/
    error_series/
    aggregate/
    tables/
  final/
    comparisons/
    error_maps/
    error_series/
    aggregate/
    tables/
  thesis_ready/
    figures/
      comparisons/
      error_maps/
      error_series/
      aggregate/
    tables/
```

## Figures produced automatically

- Clean / noisy / reconstructed comparisons
- Error maps
- Temporal error series
- Metric comparisons by resolution
- Metric comparisons by noise case
- Compute-time bar charts
- Performance heatmaps

## Tables produced automatically

- `experiment-summary.csv`
- `experiment-summary.tex`
- `model-aggregate.csv`
- `model-aggregate.tex`

## Design notes

- Filenames always include the benchmark id and, for per-experiment figures, the experiment id and time step.
- Exploratory and final outputs are stored separately.
- `thesis_ready/` receives copies of the final figures and tables.
- Comparison figures use a shared color scale across clean, noisy, and reconstructed panels.
- Error maps use a shared non-negative scale across noisy and reconstructed errors.
- The noise comparison plot uses `noise_label` categories directly instead of forcing a fake universal scalar intensity for multi-stage noise cases.

## Honest limits

- The pipeline only uses completed benchmark rows found in `summary.csv`.
- If a reconstruction bundle does not contain `reference`, the per-experiment comparison figures are skipped.
- LaTeX tables are plain `tabular` outputs by design; they do not enforce thesis-specific style packages.

## Minimal usage

Run FASE 6 first, then:

```bash
python python/scripts/run_visual_results_example.py --config configs/visual_results_phase7_example.json
```
