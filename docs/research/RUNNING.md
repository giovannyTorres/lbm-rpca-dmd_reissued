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

Las dependencias incluyen NumPy, SciPy, pandas y matplotlib; pyarrow habilita Parquet. CMake y un compilador C++17 permiten ejecutar el solver de fases 1–7. La suite completa, incluida la ampliación de mallas, registra 214 tests y 32 subtests sin omisiones (`PYTHONPATH=python/src python -m pytest tests`). La verificación previa de fases 1–7 produjo 12 experimentos y 80 figuras.

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

## Campañas de mallas ampliadas

Estas campañas se ejecutan con los scripts siguientes. El modo `full` del pipeline anterior conserva su banco original y no ejecuta automáticamente esta ampliación. Desde la raíz, después de instalar las dependencias:

```bash
python python/scripts/run_highres_newtonian.py --reuse
python python/scripts/run_highres_viscoelastic.py --workers 2
python python/scripts/run_highres_viscoplastic.py
python results/research/highres_newtonian_2026-10-14/render_publication.py
python results/research/highres_viscoplastic_2026-10-14/derive_stress_frobenius.py
python results/research/highres_viscoplastic_2026-10-14/verify_data_integrity.py --require-complete
python results/research/highres_viscoplastic_2026-10-14/render_publication.py
python python/scripts/run_highres_reconstruction_visuals.py --reuse-completed
python python/scripts/run_highres_fixed_bandwidth.py
python python/scripts/export_highres_visual_samples.py
python python/scripts/draw_research_diagrams.py
python python/scripts/build_highres_chapter.py
```

Navier–Stokes integra 64²/128²/256²/512² hasta T=4 con 41 estados. Oldroyd-B/FENE-P integra 64²/128²/256² hasta T=2.4 con 25 estados, dos controles temporales y un control newtoniano. HB/Bingham integra 64²/128² hasta T=2 y 256² sólo hasta T=0.1; el control corto no valida el horizonte largo. Las campañas registran 45 integraciones/controles y 84 condiciones de reconstrucción adicionales. La ejecución completa requiere memoria para los tensores y varias horas de cálculo, según el equipo; los tiempos publicados proceden de un equipo compartido.

`--reuse` verifica configuración y fuentes de los NPZ newtonianos; el script viscoelástico comprueba su caché por defecto y `--analyze-only` vuelve a generar análisis de los datos existentes. El script viscoplástico verifica su caché de configuraciones. `--reuse-completed` del script de reconstrucción exige hashes coincidentes de fuentes y entradas. El seguimiento de ancho fijo vuelve a ejecutar sus 24 condiciones y se distingue del resultado original 0/60: obtiene 18/24, con un control que aplica el mismo filtrado sin DMD. No constituye una validación ciega en otro flujo.

Los NPZ completos se generan localmente y están excluidos de la rama principal. Las figuras PDF/PNG, las cinco animaciones, las tablas y la procedencia están incluidas. La [rama de muestras](https://github.com/giovannyTorres/lbm-rpca-dmd_reissued/tree/data/high-resolution-visuals-2026-10-14/results/research/highres_visual_samples_2026-10-14) contiene ocho NPZ con cinco estados exactos por trayectoria, sin interpolación. El [manifiesto](../../results/research/highres_visual_samples_2026-10-14/manifest.json) conserva los hashes de origen y de selección. Estas muestras no sustituyen los 41/25/21 estados usados en los máximos de error.

Para compilar solamente el documento desde un clon nuevo, ejecutar directamente el comando LaTeX del final de esta guía: los archivos generados `.tex` y todas las figuras ya están incluidos. `build_highres_chapter.py`, incluso con `--check-only`, requiere las historias NPZ completas regeneradas localmente para verificar el cierre de las campañas. Las muestras de cinco cuadros no satisfacen ese requisito. `export_highres_visual_samples.py` también requiere las historias originales. Véanse [resultados ampliados](HIGHRES_RESULTS.md) y [revisión independiente](HIGHRES_REVIEW.md).

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
