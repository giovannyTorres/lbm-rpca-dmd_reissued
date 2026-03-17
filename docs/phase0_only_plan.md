# FASE 0 (solo diseño): propuesta de arquitectura y definición de FASE 1

## 1) Árbol completo del repositorio propuesto

```text
.
├── cpp/
│   └── lbm_core/
│       ├── include/
│       ├── src/
│       ├── CMakeLists.txt                  # TODO (FASE 1)
│       └── README.md                       # alcance y build del core
├── python/
│   ├── src/
│   │   └── fluid_denoise/
│   │       ├── __init__.py                 # TODO (FASE 1)
│   │       ├── io/                          
│   │       ├── noise/                        
│   │       ├── models/                       
│   │       ├── metrics/                      
│   │       ├── viz/                          
│   │       └── orchestration/               
│   └── scripts/
│       ├── run_lbm.py                       # TODO (FASE 1)
│       └── inspect_dataset.py               # TODO (FASE 2)
├── configs/
│   ├── experiments/
│   │   └── phase1/                          # configs de simulación
│   └── schema/
│       └── experiment.schema.yaml           # validación de configuración
├── data/
│   ├── raw/
│   ├── clean/
│   ├── noisy/
│   └── processed/
├── results/
│   ├── metrics/
│   ├── figures/
│   └── tables/
├── notebooks/
├── tests/
│   ├── cpp/
│   └── python/
├── docs/
│   ├── phase0_only_plan.md
│   ├── architecture_decisions.md            # ADRs breves
│   └── reproducibility_checklist.md         # TODO (FASE 9)
├── dashboard/
├── thesis/
├── README.md
├── .gitignore
└── Makefile                                 # atajos: setup, run, test, lint
```

## 2) Decisiones técnicas clave (justificación breve y rigurosa)

1. **Núcleo numérico en C++ y orquestación en Python**  
   C++ para rendimiento y control fino del solver LBM; Python para experimentación rápida, automatización y análisis reproducible.

2. **Arquitectura por capas (simulación → datos → modelos → métricas → visualización)**  
   Reduce acoplamiento, facilita pruebas unitarias y permite sustituir componentes (ej. cambiar formato de persistencia sin tocar solver).

3. **Experimentos 100% definidos por configuración versionada**  
   Parámetros en `configs/` (sin hardcode): mejora trazabilidad, repetibilidad y auditoría de resultados.

4. **Identidad de corrida (`run_id`) + metadatos de ejecución**  
   Cada corrida se guarda en ruta aislada con manifest de parámetros efectivos, semilla y versiones para reproducibilidad.

5. **Estructura de datos separada por estado (raw/clean/noisy/processed)**  
   Hace explícita la transformación de datos y evita mezclar etapas, reduciendo riesgo de contaminación experimental.

6. **Pruebas tempranas mínimas y crecientes**  
   Iniciar con smoke/integración en FASE 1 y extender a validaciones numéricas y de pipeline en fases posteriores.

7. **Simplicidad primero (sin DL en baseline inicial)**  
   Mantiene foco metodológico de tesis de licenciatura: baselines interpretables y comparación rigurosa antes de complejidad adicional.

## 3) Entregables exactos de FASE 1

1. **Solver LBM 2D D2Q9 BGK en `cpp/lbm_core`** para caso de flujo alrededor de cilindro.
2. **Interfaz de ejecución por configuración** (`nx`, `ny`, `Re`, `u_in`, `iterations`, `save_stride`, `seed`, geometría).
3. **Exportación de snapshots limpios** de `ux`, `uy`, `|u|`, vorticidad y máscara de obstáculo.
4. **Manifiesto por corrida** con parámetros efectivos y metadatos mínimos de ejecución.
5. **Script ejecutable reproducible** (build + run) desde `python/scripts`.
6. **Pruebas mínimas automáticas** (smoke + checks básicos de artefactos generados).
7. **Documentación de uso** en README con comandos exactos.

## 4) Criterios de validación de FASE 1

1. **Compilación reproducible**: el solver compila en entorno limpio con comandos documentados.
2. **Ejecución reproducible por configuración**: misma config + semilla produce misma estructura de artefactos.
3. **Integridad de salida**: se generan snapshots según `save_stride` y archivos esperados por campo.
4. **Sanidad numérica mínima verificable**: arrays sin `NaN/Inf` y dimensiones coherentes con la malla configurada.
5. **Trazabilidad mínima**: manifiesto contiene config efectiva, semilla, identificador de corrida y versión de ejecución.
6. **Prueba de humo en CI/local**: al menos un caso corto que falle si no se genera salida o si falta algún artefacto crítico.

> Nota de alcance: en esta entrega no se implementa FASE 1; solo se define su contrato de entrega y validación.
