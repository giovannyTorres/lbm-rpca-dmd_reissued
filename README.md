# lbm-rpca-dmd_reissued

Guía operativa para correr el proyecto localmente, de punta a punta, sin asumir conocimiento previo.

## 1) Propósito del proyecto

Este repositorio implementa un pipeline modular para experimentos de reconstrucción de flujo 2D:

1. **FASE 1**: simulación LBM en C++ (`cpp/lbm_core`).
2. **FASE 2**: conversión de snapshots CSV a dataset limpio (`data/clean`).
3. **FASE 3**: inyección de ruido y generación de dataset ruidoso (`data/noisy`).
4. **FASE 4**: contrato de datos para entrenamiento/benchmark.
5. **FASE 5**: ejecución de baselines de reconstrucción.
6. **FASE 6**: benchmark reproducible con métricas/tablas.
7. **FASE 7**: figuras y tablas finales para análisis.

## 2) Estructura del repositorio

```text
.
├── cpp/lbm_core/                      # Solver LBM (CMake + C++17, binario: lbm_sim)
├── python/src/fluid_denoise/          # Implementación de fases 1–7
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
│   └── visuals/                       # Figuras y catálogos de FASE 7
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

Desde la raíz del repo (`/workspace/lbm-rpca-dmd_reissued`):

```bash
cd /workspace/lbm-rpca-dmd_reissued
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Dependencias reales en `requirements.txt`:

- `numpy>=1.24`
- `matplotlib>=3.7`

## 5) Creación del entorno virtual

Si ya clonaste el repo y solo quieres recrear entorno:

```bash
cd /workspace/lbm-rpca-dmd_reissued
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

Opcional para exportar Parquet en FASE 6 (`summary.parquet`):

```bash
python -m pip install pandas pyarrow
```

## 7) Compilación (si aplica)

### Opción A (recomendada): compilación automática por scripts

Scripts como `python/scripts/run_phase1_example.py`, `python/scripts/generate_clean_dataset_example.py` y `python/scripts/run_test_pipeline.py` llaman internamente a CMake y compilan `lbm_sim`.

### Opción B (manual): compilar tú mismo

```bash
cd /workspace/lbm-rpca-dmd_reissued
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
cd /workspace/lbm-rpca-dmd_reissued
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
cd /workspace/lbm-rpca-dmd_reissued
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
cd /workspace/lbm-rpca-dmd_reissued
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
- Exportación a Parquet es opcional y depende de instalar `pandas` + `pyarrow`.
- `configs/baseline_phase5_example.json` no es ejecutable tal cual por los placeholders `<experiment_id>`.
- `benchmark_phase6_example.json` y `--mode full` pueden tardar mucho más que smoke/light.
- En este repo no hay empaquetado Python formal (`pyproject.toml`); se ejecuta por scripts directos en `python/scripts/`.
