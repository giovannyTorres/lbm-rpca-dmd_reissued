# lbm-rpca-dmd_reissued

Pipeline modular para tesis de licenciatura: simulación LBM 2D, estandarización de datasets, inyección de ruido, reconstrucción con baselines, benchmark y generación de figuras/tablas.

## Estado actual (resumen honesto)

- **Funcional**: FASE 1 (solver + validación), FASE 2 (dataset limpio), FASE 3 (dataset ruidoso), FASE 4 (contrato de datos), FASE 5 (baselines), FASE 6 (benchmark), FASE 7 (visuales).
- **Parcial/incompleto**: integración de entorno Python (faltaba guía robusta e instalación de dependencias), trazabilidad de ejecución end-to-end en un solo entry point.
- **Mejorado en esta iteración**:
  - runner de pipeline con perfiles (`minimal`, `light`, `full`),
  - configs smoke dedicadas,
  - `pipeline_trace.json` para trazabilidad de corrida,
  - README operativo paso a paso,
  - test unitario para runner de pipeline.

---

## Requisitos del sistema

- Linux/macOS/WSL (Windows también funciona, pero verificar rutas de binario en CMake multi-config).
- Python recomendado: **3.10+**.
- CMake: **3.16+**.
- Compilador C++17 (GCC/Clang/MSVC).

## Dependencias Python

Dependencias mínimas para ejecutar tests y pipeline base:

```bash
pip install -r requirements.txt
```

`requirements.txt` incluye:
- `numpy`
- `matplotlib`

> Nota: export a Parquet en FASE 6 requiere `pandas` + `pyarrow` opcionalmente.

---

## Setup local recomendado

1. Clonar repo.
2. Crear y activar entorno virtual.
3. Instalar dependencias.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

No hay compilación manual obligatoria: los scripts de pipeline construyen el solver automáticamente cuando hace falta.

---

## Estructura general del proyecto

- `cpp/lbm_core/`: solver LBM (C++).
- `python/src/fluid_denoise/`: módulos por fase.
- `python/scripts/`: entry points CLI.
- `configs/`: configuraciones JSON reproducibles.
- `tests/`: pruebas unitarias y de integración ligera.
- `docs/`: especificaciones y contratos por fase.
- `data/`: salidas de datasets (`raw`, `clean`, `noisy`).
- `results/`: métricas, tablas y visuales.

---

## Scripts principales (qué hace cada uno)

- `python/scripts/run_phase1_example.py`: corre solver FASE 1.
- `python/scripts/generate_clean_dataset_example.py`: genera dataset limpio FASE 2.
- `python/scripts/generate_noisy_dataset_example.py`: genera dataset ruidoso FASE 3.
- `python/scripts/run_baseline_example.py`: reconstrucción baseline FASE 5.
- `python/scripts/run_benchmark_example.py`: benchmark FASE 6.
- `python/scripts/run_visual_results_example.py`: visuales FASE 7.
- `python/scripts/run_local_pipeline.py`: **nuevo** runner unificado por perfil.
- `python/scripts/run_test_pipeline.py`: **nuevo** smoke end-to-end mínimo.

---

## Cómo correr

## 1) Prueba mínima (smoke)

```bash
python python/scripts/run_test_pipeline.py
```

Qué ejecuta:
- benchmark smoke (`configs/benchmark_phase6_smoke.json`),
- 1 resolución, 1 caso de ruido, 1 baseline,
- escribe trazabilidad en `results/metrics/phase6_smoke/pipeline_trace.json`.

## 2) Pipeline ligero

```bash
python python/scripts/run_local_pipeline.py --mode light
```

Qué ejecuta:
- benchmark smoke,
- visuales smoke (`configs/visual_results_phase7_smoke.json`),
- deja outputs en `results/metrics/phase6_smoke` y `results/visuals/phase6_smoke`.

## 3) Pipeline completo (config de ejemplo)

```bash
python python/scripts/run_local_pipeline.py --mode full
```

Qué ejecuta:
- benchmark de ejemplo (`configs/benchmark_phase6_example.json`),
- visuales de ejemplo (`configs/visual_results_phase7_example.json`).

---

## Archivos que deberían generarse

- `data/raw/<run_id>/...csv + manifest.json`
- `data/clean/<experiment_id>/fields.npz + metadata.json`
- `data/noisy/<experiment_id>/fields.npz + metadata.json`
- `results/metrics/<benchmark_id>/ledger.jsonl`
- `results/metrics/<benchmark_id>/summary.csv`
- `results/metrics/<benchmark_id>/pipeline_trace.json` (**nuevo**)
- `results/visuals/<benchmark_id>/...` (si se ejecuta FASE 7)

---

## Cómo verificar que funcionó

1. Revisar que no haya fallos en consola (`failed=0` en resumen).
2. Confirmar existencia de `summary.csv` y `pipeline_trace.json`.
3. En modo con visuales, confirmar catálogo (`catalog.json/csv/md`).
4. Validar datasets puntualmente con:

```bash
python python/scripts/validate_clean_dataset.py --run-dir data/clean/<experiment_id>
python python/scripts/validate_noisy_dataset.py --run-dir data/noisy/<experiment_id>
```

---

## Tests

Ejecutar suite:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

Si faltan dependencias, algunos tests fallarán por import error (por ejemplo `numpy` no instalado).

---

## Errores comunes y depuración

- **`cmake was not found in PATH`**: instalar CMake y verificar PATH.
- **`ModuleNotFoundError: numpy`**: instalar `requirements.txt`.
- **No aparece `lbm_sim`**: revisar build dir y generador de CMake.
- **Noisy/Clean validation falla**: revisar integridad de archivos y steps.
- **Visuales no salen**: asegurar que benchmark previo terminó y existe `summary.csv`.

---

## Limitaciones pendientes

- Algunas configuraciones reales de tesis pueden ser costosas en CPU/tiempo.
- Export Parquet no está activado por defecto (depende de `pandas+pyarrow`).
- El repositorio aún no incluye empaquetado Python formal (`pyproject.toml`) ni CI completa.

---

## Trazabilidad recomendada para experimentos

Para cada corrida guardar y versionar:
- config JSON utilizada,
- seed,
- dataset de entrada,
- tipo/intensidad de ruido,
- algoritmo y parámetros,
- artefactos de salida,
- `pipeline_trace.json` y `ledger.jsonl`.

Con esto cada resultado queda auditable y reproducible.
