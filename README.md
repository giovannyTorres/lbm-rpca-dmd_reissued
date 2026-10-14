# lbm-rpca-dmd_reissued

Guía operativa para correr el proyecto localmente, de punta a punta, sin asumir conocimiento previo.

La ampliación de investigación incorpora solvers no newtonianos 2D, nueve tipos de ruido, métricas físicas y de recursos, y pronóstico cronológico. Consulta el [documento científico de 76 páginas](results/research/document/extension.pdf), los [resultados ejecutados](docs/research/EXTENDED_RESULTS.md) y la [guía de fase 8](docs/phase8_research.md).

La nueva campaña de [mallas ampliadas y evolución física](docs/research/HIGHRES_RESULTS.md) integra Navier–Stokes hasta **512²** y Oldroyd-B/FENE-P hasta **256²**. HB/Bingham evoluciona hasta **128² durante T=2**, con referencias **256² solamente hasta T=0.1**. El documento conserva cinco capítulos e incluye **37 figuras**, seis diagramas explicativos y enlaces a **cinco animaciones**. Los mapas provienen de campos calculados; las mallas finas y los controles temporales permiten evaluar su resolución.

Se ejecutaron 45 integraciones/controles de simulación y 84 condiciones de reconstrucción adicionales. La primera comparación acepta 0/60 condiciones; el seguimiento exploratorio de ancho físico fijo acepta 18/24 y conserva su control de filtrado sin DMD. La [revisión independiente](docs/research/HIGHRES_REVIEW.md) y la [guía de reproducción](docs/research/RUNNING.md#campañas-de-mallas-ampliadas) explican alcances, fuentes y comandos. La suite completa registra **214 pruebas y 32 subcasos**.

Para inspeccionar campos sin regenerar todas las historias, descarga la [rama de muestras visuales](https://github.com/giovannyTorres/lbm-rpca-dmd_reissued/tree/data/high-resolution-visuals-2026-10-14/results/research/highres_visual_samples_2026-10-14): ocho NPZ con cinco estados exactos por trayectoria, [metadatos y hashes](results/research/highres_visual_samples_2026-10-14/manifest.json). Estas selecciones permiten reproducir imágenes, pero los máximos sobre todos los tiempos requieren las historias completas.

## 1) Propósito del proyecto

Este repositorio implementa un pipeline modular para experimentos de reconstrucción de flujo 2D:

1. **FASE 1**: simulación LBM en C++ (`cpp/lbm_core`).
2. **FASE 2**: conversión de snapshots CSV a dataset limpio (`data/clean`).
3. **FASE 3**: inyección de ruido y generación de dataset ruidoso (`data/noisy`).
4. **FASE 4**: contrato de datos para entrenamiento/benchmark.
5. **FASE 5**: ejecución de baselines de reconstrucción.
6. **FASE 6**: benchmark reproducible con métricas/tablas.
7. **FASE 7**: figuras y tablas finales para análisis.
8. **FASE 8**: solvers reológicos 2D, reconstrucción con fidelidad física/costo y pronóstico cronológico.

## 2) Estructura del repositorio

```text
.
├── cpp/lbm_core/                      # Solver LBM (CMake + C++17, binario: lbm_sim)
├── python/src/fluid_denoise/          # Fases 1–7 y paquete research de FASE 8
├── python/scripts/                    # Scripts CLI listos para ejecutar
├── configs/                           # Configuraciones JSON de ejemplo/smoke
├── tests/                             # Tests unitarios e integración ligera
├── data/
│   ├── raw/                           # Salida cruda del solver (CSV + manifest)
│   ├── clean/                         # Dataset limpio (NPZ + metadata)
│   └── noisy/                         # Dataset con ruido (NPZ + metadata)
├── results/
│   ├── metrics/                       # Ledger y resultados por experimento de benchmark
│   ├── tables/                        # summary.csv / summary.parquet
│   ├── visuals/                       # Figuras y catálogos de FASE 7
│   └── research/                      # Campañas, baseline, procedencia y documento de FASE 8
└── README.md
```

## 3) Prerrequisitos

Instalar antes de correr cualquier script:

- **Python 3.10+**
- **CMake 3.16+**
- **Compilador C++17** (GCC/Clang/MSVC)

Validación rápida de herramientas:

```bash
python3 --version
cmake --version
```

## 4) Instalación paso a paso

Desde la raíz de tu clon del repositorio:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Dependencias reales en `requirements.txt`:

- `numpy>=1.24`
- `matplotlib>=3.7`
- `scipy>=1.10`
- `pandas>=2.0`
- `pyarrow>=12.0`

## 5) Creación del entorno virtual

Si ya clonaste el repo y solo quieres recrear entorno:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Para desactivar:

```bash
deactivate
```

## 6) Instalación de dependencias

Con entorno activo:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Para exportar Parquet en FASE 6 (`summary.parquet`) se utilizan pandas y pyarrow, incluidos en los requisitos:

```bash
python -m pip install pandas pyarrow
```

## 7) Compilación (si aplica)

### Opción A (recomendada): compilación automática por scripts

Scripts como `python/scripts/run_phase1_example.py`, `python/scripts/generate_clean_dataset_example.py` y `python/scripts/run_test_pipeline.py` llaman internamente a CMake y compilan `lbm_sim`.

### Opción B (manual): compilar tú mismo

```bash
cmake -S cpp/lbm_core -B cpp/lbm_core/build_manual
cmake --build cpp/lbm_core/build_manual --config Release --parallel
```

Binario esperado (Linux/macOS):

- `cpp/lbm_core/build_manual/lbm_sim`

En generadores multi-config también puede quedar en:

- `cpp/lbm_core/build_manual/Release/lbm_sim`

## 8) Ejecución mínima de prueba (smoke)

Esta es la corrida mínima recomendada para validar instalación completa:

```bash
source .venv/bin/activate
python python/scripts/run_test_pipeline.py
```

Este script ejecuta `run_pipeline_mode("minimal")`, que usa:

- `configs/benchmark_phase6_smoke.json`

Salida clave esperada:

- `results/metrics/phase6_smoke/ledger.jsonl`
- `results/metrics/phase6_smoke/pipeline_trace.json`
- `results/tables/phase6_smoke/summary.csv`

## 9) Ejecución ligera

Modo ligero = benchmark smoke + visuales smoke:

```bash
source .venv/bin/activate
python python/scripts/run_local_pipeline.py --mode light
```

Configs usadas por el modo `light`:

- `configs/benchmark_phase6_smoke.json`
- `configs/visual_results_phase7_smoke.json`

Salidas esperadas:

- `results/metrics/phase6_smoke/pipeline_trace.json`
- `results/visuals/phase6_smoke/catalog.json`
- `results/visuals/phase6_smoke/catalog.csv`
- `results/visuals/phase6_smoke/catalog.md`

## 10) Ejecución completa

Modo completo = benchmark ejemplo + visuales ejemplo:

```bash
source .venv/bin/activate
python python/scripts/run_local_pipeline.py --mode full
```

Configs usadas por el modo `full`:

- `configs/benchmark_phase6_example.json`
- `configs/visual_results_phase7_example.json`

Salidas esperadas:

- `results/metrics/phase6_example/ledger.jsonl`
- `results/tables/phase6_example/summary.csv`
- `results/visuals/phase6_example/catalog.json`

## 11) Rutas de salida esperadas (por fase)

### FASE 1 (solver)

Al correr:

```bash
python python/scripts/run_phase1_example.py --config configs/lbm_cylinder_base.json
```

se escribe en:

- `data/raw/phase1_cylinder_re150/manifest.json`
- `data/raw/phase1_cylinder_re150/ux_t000000.csv`
- `data/raw/phase1_cylinder_re150/uy_t000000.csv`
- `data/raw/phase1_cylinder_re150/speed_t000000.csv`
- `data/raw/phase1_cylinder_re150/vorticity_t000000.csv`
- `data/raw/phase1_cylinder_re150/mask_t000000.csv`

### FASE 2 (dataset limpio)

```bash
python python/scripts/generate_clean_dataset_example.py --config configs/lbm_cylinder_clean_example.json
```

genera un experimento en:

- `data/clean/<experiment_id>/fields.npz`
- `data/clean/<experiment_id>/metadata.json`

### FASE 3 (dataset ruidoso)

```bash
python python/scripts/generate_noisy_dataset_example.py --config configs/lbm_cylinder_clean_example.json
```

genera en:

- `data/noisy/<experiment_id>/fields.npz`
- `data/noisy/<experiment_id>/metadata.json`
- `data/noisy/<experiment_id>/figures/noisy_comparison_*.png` (si no pasas `--skip-figure`)

### FASE 5 (baseline individual)

`configs/baseline_phase5_example.json` trae placeholders (`data/noisy/<experiment_id>`). Debes reemplazar `<experiment_id>` por un id real generado en FASE 2/3 antes de correr:

```bash
python python/scripts/run_baseline_example.py --config configs/baseline_phase5_example.json
```

Salida en:

- `data/processed/baselines/<...>/reconstruction.npz`
- `data/processed/baselines/<...>/metadata.json`

### FASE 6 (benchmark)

```bash
python python/scripts/run_benchmark_example.py --config configs/benchmark_phase6_example.json
```

Salida en:

- `results/metrics/<benchmark_id>/ledger.jsonl`
- `results/metrics/<benchmark_id>/experiments/<experiment_id>/reconstruction.npz`
- `results/metrics/<benchmark_id>/experiments/<experiment_id>/metadata.json`
- `results/metrics/<benchmark_id>/experiments/<experiment_id>/benchmark_result.json`
- `results/tables/<benchmark_id>/summary.csv`

### FASE 7 (visuales)

```bash
python python/scripts/run_visual_results_example.py --config configs/visual_results_phase7_example.json
```

Salida en:

- `results/visuals/<benchmark_id>/catalog.json`
- `results/visuals/<benchmark_id>/catalog.csv`
- `results/visuals/<benchmark_id>/catalog.md`
- `results/visuals/<benchmark_id>/exploratory/...`
- `results/visuals/<benchmark_id>/final/...`
- `results/visuals/<benchmark_id>/thesis_ready/...`

## 12) Validación de resultados

### Validar que FASE 1 está correcta

```bash
python python/scripts/run_phase1_example.py --config configs/lbm_cylinder_base.json
```

El script corre validación automática (`validate_run_outputs`) y muestra:

- `[VALIDATION] checked ... snapshot steps in ...`

### Validar datasets explícitamente

```bash
python python/scripts/validate_clean_dataset.py --run-dir data/clean/<experiment_id>
python python/scripts/validate_noisy_dataset.py --run-dir data/noisy/<experiment_id>
```

### Validar con tests

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

## 13) Errores comunes

1. **`RuntimeError: cmake was not found in PATH`**
   - Causa: CMake no instalado o no visible en `PATH`.
   - Acción: instala CMake y verifica con `cmake --version`.

2. **`ModuleNotFoundError: numpy` / `matplotlib`**
   - Causa: dependencias no instaladas en el entorno activo.
   - Acción: activar `.venv` y correr `python -m pip install -r requirements.txt`.

3. **`FileNotFoundError: Config file not found`**
   - Causa: ruta de `--config` incorrecta.
   - Acción: usar rutas relativas a la raíz, por ejemplo `configs/benchmark_phase6_smoke.json`.

4. **FASE 5 falla con `<experiment_id>` literal**
   - Causa: no reemplazaste placeholders en `configs/baseline_phase5_example.json`.
   - Acción: editar ese archivo con rutas reales de `data/clean/...` y `data/noisy/...`.

5. **No se generan visuales en FASE 7**
   - Causa: no existe `results/tables/<benchmark_id>/summary.csv` del benchmark correspondiente.
   - Acción: correr primero FASE 6 para el mismo `benchmark_id`.

## 14) Notas de limitaciones

- El pipeline de benchmark (FASE 6) es **secuencial**; no hay paralelismo interno.
- Exportación a Parquet es opcional; sus dependencias pandas y pyarrow están incluidas en los requisitos.
- `configs/baseline_phase5_example.json` no es ejecutable tal cual por los placeholders `<experiment_id>`.
- `benchmark_phase6_example.json` y `--mode full` pueden tardar mucho más que smoke/light.
- En este repo no hay empaquetado Python formal (`pyproject.toml`); se ejecuta por scripts directos en `python/scripts/`.


## 15) FASE 8: investigación integrada

Con el entorno preparado, ejecutar reconstrucción y pronóstico:

```bash
python -m pip install pytest
python -m unittest discover -s tests
PYTHONPATH=python/src python -m pytest tests/test_research*.py
python python/scripts/run_phase8_pipeline.py --mode smoke
python python/scripts/run_phase8_pipeline.py --mode full --reuse-completed
```

`smoke` usa `extended_smoke.json` y `forecast_smoke.json`; `full` usa `extended_physics.json` y `forecasting_campaign.json`. La reutilización exige finalización y hashes coincidentes de configuración y fuentes. La fase 8 conserva la numeración de las anteriores; un mismo identificador de campaña sustituye sus artefactos si se vuelve a ejecutar.

La campaña completa de reconstrucción tiene **2808 condiciones y 8424 ajustes**: 8 casos, 9 ruidos, 13 métodos/variantes, 3 semillas y 3 repeticiones de recursos. Hubo 0 ajustes fallidos, 6 condiciones no convergidas y 430 aceptadas; 150/216 grupos tienen algún estimador admisible y 66 ninguno. Ruido temporal y deriva de sensor no alcanzan los umbrales.

El pronóstico completó **60 escenarios** con 29 opciones, limpieza/ajuste sólo sobre entrenamiento y selección mediante validación ruidosa. Con tolerancia absoluta `1e-10` en la diferencia de error L2 relativo futuro de velocidad, mejora persistencia cruda en 48/60 escenarios y persistencia filtrada en 43/60; los restantes empatan. La vorticidad futura en flujos no lineales sigue presentando errores grandes. Se conserva la evidencia cruda junto con la comparación consolidada.

Oldroyd-B/FENE-P 2D evolucionan conformación y realimentan momento; HB/Bingham 2D resuelven esfuerzo variable regularizado. Son casos periódicos con parámetros y verificaciones declarados; no certifican DNS universal, alto Wi, estelas de cilindro ni caos. PIV real, 3D, GPU/online, EVP externo y otros métodos avanzados siguen como trabajo posterior.

La validación consolidada registró **212 tests y 28 subtests sin omisiones**, y la ejecución completa de fases 1–7 produjo **12 experimentos y 80 figuras**. El baseline previo queda preservado en `results/research/baseline_2026-10-14/` con sus hashes, configuración y entorno; no se atribuye a la campaña nueva.

- [Guía de fase 8](docs/phase8_research.md), [comandos detallados](docs/research/RUNNING.md) y [protocolo científico](docs/research/RESEARCH_PROTOCOL.md).
- [Resultados integrados](docs/research/EXTENDED_RESULTS.md), [pronóstico](docs/research/FORECASTING.md) y [segunda revisión](docs/research/SECOND_REVIEW.md).
- [Oldroyd-B/FENE-P 2D](docs/research/VISCOELASTIC_2D.md), [HB/Bingham 2D](docs/research/VISCOPLASTIC_2D.md) y [restricciones físicas](docs/research/PHYSICS_CONSTRAINTS.md).
- [Resumen consolidado y tolerancia de comparación](results/research/extended_summary.json).
- [Tabla de reconstrucción](results/research/extended_physics_2026-10-14/summary.csv) y [tabla de pronóstico](results/research/forecasting_extended_2026-10-14/summary.csv).
- [GIF reológico](results/research/rheology_matched_2026-10-14/vortices_rheology.gif), [campos reológicos PDF](results/research/rheology_matched_2026-10-14/final_vorticity.pdf) y [pronóstico PDF](results/research/forecasting_extended_2026-10-14/figures/holdout_comparison.pdf).
- [Documento científico PDF](results/research/document/extension.pdf) y [fuente LaTeX](docs/research/extension.tex).
