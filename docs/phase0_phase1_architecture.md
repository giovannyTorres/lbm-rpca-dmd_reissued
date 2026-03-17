# FASE 0 + Arranque de FASE 1: Arquitectura y plan técnico

## 1) Objetivo de FASE 0
Diseñar una base de proyecto **reproducible, modular y extensible** para una tesis de mecánica de fluidos + ciencia de datos, dejando listo el arranque de FASE 1 (solver LBM 2D D2Q9 BGK) sin adelantar fases posteriores.

## 2) Arquitectura concreta del repositorio

```text
/cpp/lbm_core                  # Núcleo de simulación de alto rendimiento
  CMakeLists.txt
  include/
  src/
/python/src/fluid_denoise      # Librería Python para orquestación y análisis
/python/scripts                # Scripts ejecutables (pipeline, utilidades)
/configs                       # Configs versionadas de experimentos
/data/raw                      # Salidas directas de simulación
/data/clean                    # Dataset limpio consolidado (FASE 2)
/data/noisy                    # Dataset contaminado (FASE 3)
/data/processed                # Datos intermedios/transformados
/results/metrics               # Métricas numéricas/físicas
/results/figures               # Figuras listas para tesis
/results/tables                # Tablas para tesis y benchmark
/notebooks                     # Exploración controlada
/tests                         # Pruebas automáticas
/docs                          # Diseño, decisiones, bitácora técnica
/dashboard                     # App JS de exploración (FASE 8)
/thesis                        # Documento LaTeX (FASE 7)
```

## 3) Decisiones técnicas y justificación

1. **Separación C++/Python por responsabilidad**
   - C++ ejecuta el solver LBM (rendimiento y control numérico).
   - Python coordina corridas, validaciones y análisis.

2. **Experimentos definidos por configuración**
   - Se usa `configs/*.cfg` con parámetros explícitos (sin hardcode).
   - Permite trazabilidad y repetibilidad por archivo.

3. **Semilla fija obligatoria**
   - `seed` existe desde FASE 1 para continuidad de reproducibilidad, aunque el caso base sea mayormente determinista.

4. **Salida estructurada por `run_id`**
   - Cada corrida genera carpeta propia con `manifest.txt` + snapshots.
   - Evita sobreescritura y facilita auditoría.

5. **Formato inicial simple (CSV) para arranque de FASE 1**
   - Permite inspección rápida y depuración temprana.
   - En FASE 2 se formalizará persistencia (NPY/NPZ/HDF5) y metadatos enriquecidos.

## 4) Entregables de FASE 0

- Estructura base del repositorio creada.
- Documento de arquitectura y decisiones (`docs/phase0_phase1_architecture.md`).
- Configuración de ejemplo para correr FASE 1 (`configs/lbm_cylinder_base.cfg`).

## 5) Criterios de validación de FASE 0

- El repositorio contiene todas las carpetas objetivo.
- Existe al menos un flujo de ejecución documentado (build + run).
- Existe al menos una configuración reproducible versionada.

---

## 6) Objetivo de FASE 1 (arranque)

Implementar un solver LBM 2D D2Q9 con BGK para flujo alrededor de cilindro y exportar snapshots de:
- `ux`, `uy`
- magnitud de velocidad
- vorticidad
- máscara del obstáculo

## 7) Entregables de FASE 1 (arranque)

- Solver C++ compilable (`cpp/lbm_core`).
- CLI para cargar `--config`.
- Script Python para compilar + ejecutar (`python/scripts/run_phase1_example.py`).
- Prueba mínima de humo (`tests/test_phase1_smoke.py`).

## 8) Criterios de validación de FASE 1

1. Compila sin errores con CMake.
2. Ejecuta una corrida corta con configuración de ejemplo.
3. Genera snapshots según `save_stride`.
4. Los archivos de salida existen y contienen valores numéricos finitos.
5. Se guarda un `manifest.txt` por corrida con parámetros efectivos.

## 9) Riesgos conocidos (actuales)

- Condiciones de frontera simplificadas (inlet/outlet) pueden requerir refinamiento para estudios de alta fidelidad.
- Exportación CSV puede ser costosa para corridas grandes (se migrará en FASE 2).
- Validaciones físicas actuales son básicas; se ampliarán en FASE 5.
