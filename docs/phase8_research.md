# FASE 8 Research: física, estimadores y costo

La fase 8 incorpora un banco de investigación en `fluid_denoise.research` sin renombrar las fases 1–7. Añade Navier–Stokes periódico, solvers reológicos 2D, contaminación pareada, estimadores, restricciones físicas y recursos medidos. Se ejecutaron reconstrucción dentro de la ventana y pronóstico cronológico fuera de muestra. PIV real, estelas externas, alto Wi, caos de cilindro y DNS certificada requieren evidencia adicional.

## Componentes

| Módulo | Responsabilidad |
|---|---|
| `fluid_denoise.research.dns` | Referencia Taylor–Green, integración NS 2D periódica y refinamiento espacial/temporal. El nombre del módulo no certifica DNS. |
| `fluid_denoise.research.rheology` | Perfiles exactos HB/Bingham y arranque Oldroyd-B de canal con esfuerzos y controles constitutivos. |
| `fluid_denoise.research.noise` | Nueve modelos: gaussiano, impulsivo, mixto, correlacionado espacial/temporal, columnas, Student-t, heteroscedástico y deriva; intensidad realizada registrada. |
| `fluid_denoise.research.viscoelastic_dns` | Oldroyd-B/FENE-P 2D periódicos; conformación dinámica y realimentación del esfuerzo en momento. |
| `fluid_denoise.research.viscoplastic_dns` | HB/Bingham 2D regularizados con esfuerzo variable acoplado a momento. |
| `fluid_denoise.research.physics` | Proyección de Helmholtz y corte de banda predeclarados para geometrías periódicas compatibles. |
| `fluid_denoise.research.forecasting` y `forecast_runner` | Ajuste sólo con entrenamiento, selección mediante validación ruidosa y evaluación futura separada. |
| `fluid_denoise.research.algorithms` | Identidad, POD/SVD, PCP, RPCA estable penalizada, Outlier Pursuit, variantes DMD y cadenas de limpieza. |
| `fluid_denoise.research.metrics` | Errores vectoriales, vorticidad, divergencia, fronteras e integrales físicas según los datos disponibles. |
| `fluid_denoise.research.cost` | CPU/wall, RSS, memoria trazada, repeticiones, proxies de trabajo y Pareto condicionado a calidad. |
| `fluid_denoise.research.runner` | Campañas reproducibles y un proceso aislado por ajuste; registra fallos, convergencia y procedencia. |
| `fluid_denoise.research.bridge` | Conversión explícita de `ModelRun`/`ModelRunPair` de fase 4 a campos vectoriales de investigación. |

La CLI integrada es `python/scripts/run_phase8_pipeline.py`. Las CLI por tarea incluyen `run_research.py`, `run_forecasting.py`, `analyze_research.py` y `verify_adversarial_physics.py` en `python/scripts/`. Las configuraciones se almacenan en `configs/research/`. El repositorio se resuelve desde `Path(__file__).resolve().parents[2]` en scripts y `parents[4]` en módulos de investigación: los artefactos pertenecen a la raíz del proyecto, no a `python/`.

## Preparación y ejecución mínima

Desde la raíz, con Python 3.10 o posterior:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install pytest
python -m unittest discover -s tests
PYTHONPATH=python/src python -m pytest tests/test_research*.py
python python/scripts/run_phase8_pipeline.py --mode smoke
python python/scripts/run_phase8_pipeline.py --mode full --reuse-completed
python python/scripts/verify_adversarial_physics.py --output results/research/solver_validation_local/adversarial_review.json
```

NumPy, SciPy 1.10 o posterior, pandas y matplotlib cubren generación, ajuste y análisis. El comando `unittest` comprueba los contratos existentes; pytest ejecuta la cobertura específica de investigación. La corrida mínima tiene una repetición de recursos y no caracteriza su variabilidad. `smoke` usa `extended_smoke.json` y `forecast_smoke.json`; `full` usa `extended_physics.json` y `forecasting_campaign.json`. `--reuse-completed` exige campaña completa, configuración coincidente y hashes de fuentes vigentes. [RUNNING.md](research/RUNNING.md) detalla los comandos por tarea y el análisis:

```bash
python python/scripts/analyze_research.py --campaign extended_physics_2026-10-14
```

El análisis requiere una campaña completa y escribe `discussion.md` y `results_generated.tex` dentro de su carpeta. La opción explícita `--update-docs` reemplaza `docs/research/RESULTS.md` y `docs/research/results_generated.tex`. Un mismo `campaign_id` sustituye sus artefactos; usar identificadores nuevos para comparar o conservar campañas.

## Puente desde el contrato de fase 4

`ModelRunPair` contiene observación y referencia alineadas por malla, máscara y pasos. Esos pasos son índices del solver y no definen por sí solos tiempos físicos, unidades o topología. `model_pair_to_research` requiere explícitamente:

- `times`: vector finito, estrictamente creciente, con un tiempo por snapshot en las unidades declaradas.
- `dx`, `dy`: espaciamientos positivos y finitos.
- `units`: nombre explícito del sistema, por ejemplo `"lattice"`, o diccionario con `length`, `time` y `velocity`; si se suministra densidad al usar diccionario, también `density`.
- `boundary`: `"periodic"`, `"channel"` o `"open"`, según la formulación y las fronteras reales del solver.
- `periodic_axes`: lista explícita de ejes, como `[]`, `["x"]`, `["y"]` o `["x", "y"]`; `boundary="periodic"` exige ambos ejes.

Una malla periódica excluye su extremo duplicado; un eje no periódico es nodal. Revisar la configuración y el manifiesto del solver antes de declarar la frontera. El puente no infiere una frontera para el cilindro LBM, ni convierte unidades lattice a SI.

Ejemplo de uso con declaraciones revisadas para el experimento y guardadas en un JSON propio:

```python
import json
from pathlib import Path
from fluid_denoise.phase4_model_data import align_model_runs
from fluid_denoise.research.bridge import model_pair_to_research
from fluid_denoise.research.metrics import physical_metrics

pair = align_model_runs(observed_run_dir, reference_run_dir)
physics = json.loads(Path("physics_declarations.json").read_text())
research_pair = model_pair_to_research(
    pair,
    times=physics["times"], dx=physics["dx"], dy=physics["dy"],
    units=physics["units"], boundary=physics["boundary"],
    periodic_axes=physics["periodic_axes"],
)
metrics = physical_metrics(research_pair.reference, research_pair.observed)
```

Ejecutar un script con estos imports mediante `PYTHONPATH=python/src python <script.py>`. Las rutas `observed_run_dir` y `reference_run_dir` son las de los datasets de fases 2 y 3 del experimento, y el JSON debe declarar la física real de ese caso.

El resultado es `ResearchRunPair(observed, reference)`, con velocidad `(T, 2, Y, X)` obtenida de `ux` y `uy`; no se inventan componentes a partir de `speed`. Se conserva máscara (`True` sólido), pasos y procedencia. El puente valida la correspondencia entre canales, máscara y tiempos.

La densidad es opcional y se declara por separado mediante `reference_density` y `observed_density`. Sin ellas queda `density=None` y `metadata.density_status="unknown"`; no se supone `rho=1` ni se copia la densidad de referencia a la observación. La evaluación mantiene diagnósticos cinemáticos y devuelve `null` para masa, energía, flujo másico y continuidad dependientes de densidad hasta suministrarla para ambos campos. Una densidad suministrada debe ser positiva, finita y compatible con `(T, Y, X)`; su unidad y procedencia se deben justificar. La conversión conserva `dns_certified=False` y no valida el solver de origen.

## Baseline y campañas nuevas

La evidencia de partida de investigación se conserva en:

```text
results/research/baseline_2026-10-14/
  pilot_2026-10-14/
  solver_validation/
```

[RESULTS.md](research/RESULTS.md) describe las 1584 condiciones y 4752 ajustes de ese baseline preservado. Los hashes de aquellos artefactos identifican las fuentes, configuración y entorno del baseline; no certifican la integración actual. Las campañas nuevas escriben `results/research/<campaign_id>/` y registran evidencia propia. El runner produce ledger, resumen, manifiestos, procedencia y estado de finalización; el análisis conserva fallos y distingue convergencia de aceptación física.

El baseline verifica velocidad y vorticidad en 31 snapshots frente a una referencia numérica `128²` para dos casos periódicos. Sus tolerancias no certifican todos los observables ni otros regímenes. Los perfiles HB/Bingham son referencias analíticas estacionarias; Oldroyd-B es un canal paralelo transitorio. El cilindro de referencia y el puente desde fase 4 conservan su estado de DNS no certificada. [DNS_VALIDATION.md](research/DNS_VALIDATION.md), [NON_NEWTONIAN.md](research/NON_NEWTONIAN.md) y [SCIENTIFIC_REVIEW.md](research/SCIENTIFIC_REVIEW.md) detallan estas limitaciones.

## Campañas integradas y entregables

La reconstrucción `extended_physics_2026-10-14` completó **2808 condiciones y 8424 ajustes**: 8 casos, 9 ruidos, 13 métodos/variantes, 3 semillas y 3 repeticiones de costo. Hubo 0 ajustes fallidos, 6 condiciones no convergidas y 430 aceptadas; 150/216 grupos caso/ruido/semilla tienen algún estimador admisible y 66 ninguno. Ruido temporal y deriva de sensor no alcanzan los umbrales.

El pronóstico `forecasting_extended_2026-10-14` completó **60 escenarios**, con 5 flujos, 4 ruidos, 3 semillas y 29 opciones predeclaradas. Utiliza 60% entrenamiento, 20% validación y el resto test; limpieza y ajuste usan sólo entrenamiento, y la elección usa MSE de validación ruidosa. Con tolerancia absoluta `1e-10` en la diferencia de error L2 relativo futuro de velocidad, mejora persistencia cruda en 48/60 escenarios y persistencia filtrada en 43/60; los restantes empatan. La vorticidad futura conserva errores grandes en casos no lineales. Los artefactos crudos y el resumen consolidado conservan la regla de comparación y su procedencia.

La verificación consolidada registró **212 tests y 28 subtests sin omisiones**. Las fases 1–7 también se ejecutaron completas, con **12 experimentos y 80 figuras**. Estas comprobaciones no certifican DNS ni todos los regímenes físicos.

- [Resultados integrados](research/EXTENDED_RESULTS.md), [pronóstico](research/FORECASTING.md), [restricciones físicas](research/PHYSICS_CONSTRAINTS.md) y [segunda revisión](research/SECOND_REVIEW.md).
- [Viscoelasticidad 2D](research/VISCOELASTIC_2D.md) y [viscoplasticidad 2D](research/VISCOPLASTIC_2D.md).
- [Resumen consolidado y tolerancia de comparación](../results/research/extended_summary.json).
- [Tabla de reconstrucción](../results/research/extended_physics_2026-10-14/summary.csv) y [tabla de pronóstico](../results/research/forecasting_extended_2026-10-14/summary.csv).
- [PDF de pronóstico](../results/research/forecasting_extended_2026-10-14/figures/holdout_comparison.pdf), [GIF reológico](../results/research/rheology_matched_2026-10-14/vortices_rheology.gif) y [PDF de campos finales](../results/research/rheology_matched_2026-10-14/final_vorticity.pdf).
- [Documento científico PDF](../results/research/document/extension.pdf) y [fuente LaTeX](research/extension.tex).

Oldroyd-B/FENE-P evolucionan conformación y realimentan momento; HB/Bingham resuelven esfuerzo variable regularizado. Son flujos periódicos 2D con parámetros declarados, no evidencia de alto Wi, estelas de cilindro o caos. Los controles analíticos de canal permanecen separados. BOP/HODMD/piDMD/SPOD/ResDMD, EVP externo, 3D, PIV real, GPU/online, aprendizaje profundo y datos faltantes siguen como trabajo posterior.

## Lectura científica de una campaña

Comparar métodos sobre la misma entrada, ruido y semilla. La referencia limpia se usa para generar contaminación y evaluar; no selecciona rango o hiperparámetros del ajuste. Las semillas son réplicas de contaminación y las repeticiones de recursos no aumentan la muestra de calidad. Los proxies de trabajo no son contadores hardware ni energía; RSS incluye intérprete, dependencias y datos, y `tracemalloc` añade sobrecosto.

La aceptación combina umbrales predeclarados y convergencia. En canales incluye pared tangencial y penetración normal: un flujo neto pequeño no prueba impermeabilidad local. La frontera Pareto exploratoria y la físicamente admisible son resultados distintos. Una métrica ausente no supera un gate. El [protocolo](research/RESEARCH_PROTOCOL.md) especifica los supuestos, diseños futuros y referencias públicas; [extension.tex](research/extension.tex) reúne el documento científico compilable.
