# Ejecutar investigación de fase 8

La fase 8 integra reconstrucción, pronóstico cronológico y solvers reológicos 2D en `fluid_denoise.research`. Las fases 1–7 mantienen su numeración y contratos. La guía de [fase 8](../phase8_research.md) explica el puente explícito desde `ModelRunPair`, que exige tiempos, malla, unidades y fronteras sin inferir densidad ni certificación DNS.

## Instalación y comprobación

Desde la raíz del repositorio, con Python 3.10 o posterior:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install pytest
python -m unittest discover -s tests
PYTHONPATH=python/src python -m pytest tests/test_research*.py
```

Las dependencias incluyen NumPy, SciPy, pandas y matplotlib; pyarrow habilita Parquet. CMake y un compilador C++17 permiten ejecutar el solver de fases 1–7. La verificación consolidada registró 212 tests y 28 subtests sin omisiones; el pipeline completo de fases 1–7 produjo 12 experimentos y 80 figuras.

## Pipeline integrado

```bash
python python/scripts/run_phase8_pipeline.py --mode smoke
python python/scripts/run_phase8_pipeline.py --mode full --reuse-completed
```

`smoke` ejecuta `configs/research/extended_smoke.json` y `forecast_smoke.json`; `full` ejecuta `extended_physics.json` y `forecasting_campaign.json`. El pipeline corre reconstrucción, análisis de esa campaña y pronóstico. `--reuse-completed` sólo reutiliza una campaña completa cuyos hashes de configuración y fuentes coinciden; el análisis de reconstrucción se vuelve a generar. Sin reutilización, la campaña vuelve a ejecutarse.

Las salidas mínimas son `results/research/extended_smoke/` y `forecast_smoke/`. Las completas son `extended_physics_2026-10-14/` y `forecasting_extended_2026-10-14/`. La corrida mínima tiene una repetición de recursos y no estima su variabilidad. Un mismo `campaign_id` sustituye artefactos; usar otro identificador en una copia de la configuración para conservar una ejecución distinta. No escribir campañas nuevas dentro de `baseline_2026-10-14`.

## Comandos por tarea

```bash
python python/scripts/run_research.py --config configs/research/extended_physics.json
python python/scripts/analyze_research.py --campaign extended_physics_2026-10-14
python python/scripts/run_forecasting.py --config configs/research/forecasting_campaign.json
python python/scripts/verify_adversarial_physics.py --output results/research/solver_validation_local/adversarial_review.json
```

El análisis escribe tablas, figuras, `discussion.md` y `results_generated.tex` dentro de la campaña. Sólo la opción `--update-docs` reemplaza `docs/research/RESULTS.md` y `docs/research/results_generated.tex`; los informes integrados se mantienen en `EXTENDED_RESULTS.md` y `extended_generated.tex`.

La comparación reológica y el resumen integrado se generan aparte del pipeline:

```bash
python python/scripts/run_rheology_comparison.py
python python/scripts/summarize_extended_research.py
```

El primer comando produce los seis controles reológicos comunes y su GIF; el segundo deriva el resumen, la figura de pronóstico y la discusión integrada de las campañas completas. Este último reemplaza `docs/research/EXTENDED_RESULTS.md` y `extended_generated.tex`.

## Resultados completos y procedencia

La reconstrucción ejecutó **2808 condiciones y 8424 ajustes**: 8 casos × 9 ruidos × 13 métodos/variantes × 3 semillas × 3 repeticiones de costo. Registró 0 ajustes fallidos, 6 condiciones no convergidas y 430 aceptadas. Hay algún estimador admisible en 150/216 grupos caso/ruido/semilla; 66 no tienen ninguno. Ruido temporal y deriva de sensor no tienen condiciones aceptadas.

El pronóstico completó **60 escenarios** —5 flujos × 4 ruidos × 3 semillas— con 29 opciones predeclaradas. La selección usa sólo validación ruidosa después de ajuste y limpieza sobre entrenamiento; la referencia limpia futura sólo evalúa. Las comparaciones consolidadas emplean `abs(Δerror_L2_relativo)>1e-10`: mejora persistencia cruda en 48/60 y persistencia filtrada en 43/60, con empates en el resto. La vorticidad futura permanece difícil en casos no lineales. Los conteos de artefactos crudos se conservan junto con la regla de consolidación.

Cada campaña registra `ledger.jsonl`, `summary.csv/json`, `completion.json`, `provenance.json`, manifiestos y configuración. Los hashes identifican las fuentes, parámetros y entorno de aquella ejecución. Los NPZ limpios se generan localmente; `save_reconstructions=true` conserva estimaciones. Las matrices y la procedencia permiten comprobar qué entrada se evaluó.

El baseline previo se conserva con procedencia propia en `results/research/baseline_2026-10-14/{pilot_2026-10-14,solver_validation}/`. Sus cifras no se atribuyen a las campañas nuevas. Véanse [discusión del baseline](../../results/research/baseline_2026-10-14/pilot_2026-10-14/discussion.md), [RESULTS.md](RESULTS.md) para la nueva reconstrucción y [EXTENDED_RESULTS.md](EXTENDED_RESULTS.md) para el análisis integrado.

## Costo e interpretación física

`run_research.py` fija un hilo BLAS antes de importar NumPy y cada ajuste usa un proceso aislado. CPU/wall incluyen `tracemalloc`, pero excluyen simulación, lectura, evaluación física y lanzamiento de proceso. RSS es el máximo absoluto del proceso con intérprete, bibliotecas y entradas. Los proxies de trabajo no son FLOPs medidos ni energía eléctrica. Las repeticiones de recursos no aumentan la muestra de calidad; las semillas contaminan el mismo flujo determinista.

Los nuevos Oldroyd-B/FENE-P 2D evolucionan conformación con realimentación de esfuerzo; HB/Bingham 2D resuelven un esfuerzo variable regularizado. Los perfiles analíticos y controles de canal siguen disponibles por separado. El solver, la proyección solenoidal, las restricciones físicas y los umbrales de aceptación no certifican DNS universal, alto Wi, estelas de cilindro ni caos. La reconstrucción y el pronóstico son tareas distintas. PIV real, 3D, GPU/online, EVP externo y métodos avanzados adicionales requieren campañas propias.

## Documentos y artefactos

- [Resultados integrados](EXTENDED_RESULTS.md), [pronóstico](FORECASTING.md), [restricciones físicas](PHYSICS_CONSTRAINTS.md) y [segunda revisión](SECOND_REVIEW.md).
- [Oldroyd-B/FENE-P 2D](VISCOELASTIC_2D.md) y [HB/Bingham 2D](VISCOPLASTIC_2D.md).
- [Resumen consolidado y tolerancia de comparación](../../results/research/extended_summary.json).
- [Tabla de reconstrucción](../../results/research/extended_physics_2026-10-14/summary.csv), [tabla de pronóstico](../../results/research/forecasting_extended_2026-10-14/summary.csv) y [proveniencia del pronóstico](../../results/research/forecasting_extended_2026-10-14/provenance.json).
- [Pronóstico en PDF](../../results/research/forecasting_extended_2026-10-14/figures/holdout_comparison.pdf), [GIF reológico](../../results/research/rheology_matched_2026-10-14/vortices_rheology.gif) y [campos finales en PDF](../../results/research/rheology_matched_2026-10-14/final_vorticity.pdf).
- [Documento científico PDF](../../results/research/document/extension.pdf) y [fuente LaTeX](extension.tex).

Para compilar con una distribución LaTeX y `latexmk`, desde la raíz:

```bash
mkdir -p results/research/document
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=results/research/document docs/research/extension.tex
```

El documento incorpora los archivos generados y las figuras identificadas en ellos. Comprobar su campaña y procedencia antes de atribuir cifras a otra ejecución.
