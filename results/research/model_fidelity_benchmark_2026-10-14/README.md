# Comparación de modelos y fidelidad de reconstrucción

El objeto de esta campaña es comparar **qué conserva cada modelo de una simulación observada con ruido**. El tamaño de malla y la complejidad constitutiva son contexto de la prueba; no sustituyen la comparación de los estimadores.

El estado definitivo, incluidos los fallos, se encuentra en `completion.json`. El ledger primario tiene un ajuste por condición; sus tres evaluaciones reutilizan exactamente la misma reconstrucción cruda.

## Diseño de la comparación

| Dimensión | Diseño |
|---|---|
| Referencias | Siete trayectorias calculadas en 128²: dos Navier–Stokes, Oldroyd-B, FENE-P y tres fluidos viscoplásticos |
| Ventana | NS: t = 0…4, 41 estados; viscoelásticos: 0…2.4, 25; viscoplásticos: 0…2, 21 |
| Ruido | Gaussiano IID; impulsos en 1% de entradas; gaussiano suavizado espacialmente con σ de kernel = 0.10; AR temporal con τ = 0.30; columnas contaminadas |
| Intensidad | Exactamente 20% de RMS global de la velocidad limpia, realizado por muestra |
| Semillas nuevas | 101, 307, 911 |
| Estimadores | Identidad, POD, DMD, TLS-DMD, PCP, RPCA estable, Outlier Pursuit, PCP→TLS, estable→TLS y optimized DMD |
| Rango | 8 para métodos modales; PCP/estable/OP seleccionan su rango mediante la penalización, sin imponer rango 8 |
| Evaluaciones | Cruda; proyección Helmholtz H; H más corte físico isotrópico k≤10 |
| Total primario |7 × 5 × 3 × 10 = 1050 ajustes; 3150 evaluaciones de esos ajustes |

Las tres semillas describen variabilidad de ruido sobre cada trayectoria fija. Las siete trayectorias son casos físicos heterogéneos, no siete réplicas estadísticas intercambiables. Una mediana global no establece un modelo universalmente superior.

La selección de columnas usa `floor(.10*T)` sin reemplazo:4/41,2/25 y2/21 estados, dejando la fracción realizada por debajo de γ = 0.10. No existe una fracción exactamente 10% con esas cantidades de estados. La amplitud se normaliza al mismo 20% RMS de toda la ventana, por lo que la perturbación local en las columnas contaminadas es mayor.

El sigma espacial corresponde al **kernel gaussiano de suavizado**, no a la desviación estándar de su autocorrelación. El AR usa `exp(-Δt/.30)` y empieza en distribución estacionaria. Igualar RMS no iguala el riesgo de reconstrucción: diferenciación espacial, persistencia temporal y estructura de dispersión son distintos.

## Separación de información y controles

`estimate_observation(observed,times,method,seed,config)` recibe solamente la observación y tiempos. `postprocess_field` recibe solamente la reconstrucción y geometría. La referencia limpia interviene en la amplitud de inyección y en la evaluación, nunca en el ajuste, proyección o elección de parámetros. Se usan los mismos datos y el mismo posprocesado para la identidad y los modelos; así se distingue la mejora temporal del beneficio de filtrar el espacio.

El banco reutiliza familias de flujo y el prior k≤10 estudiados antes, aunque las semillas son nuevas. Es una **comparación exploratoria**, no una confirmación ciega independiente. No se ajustan hiperparámetros con la referencia limpia. El rango fijo tampoco demuestra que cada método fue optimizado; la sensibilidad de rangos/intensidades tiene una campaña separada.

Los métodos iterativos usan máximo 600 iteraciones y tolerancia 1e-5. Optimized DMD usa 150 evaluaciones nominales por inicio y dos inicios; sus evaluaciones residuales reales y límites se registran. Stable RPCA recibe intensidad relativa = 0.2 y estima su escala de ruido a partir de RMS observado: la regla supone independencia y no ofrece garantía con ruido correlacionado. Outlier Pursuit usa dispersión por columnas y γ = 0.10; identificar un subespacio no equivale a recuperar valores limpios dentro de columnas enteramente contaminadas.

## Qué significa fidelidad

Se comparan velocidad, vorticidad, correlación de vorticidad, divergencia, energía y enstrofía contra **la referencia discreta 128²**, conservando todos sus modos en la evaluación. Las series muestran todo tiempo guardado; el máximo guardado no es un máximo continuo. La fidelidad a una referencia calculada y la fidelidad de esa referencia al problema continuo se presentan por separado en el documento.

Los criterios operativos son e(u) ≤ 15%, e(ω) ≤ 25%, divergencia adicional/RMS(ω) ≤ 10% y e(energía) ≤ 15%, además de convergencia de todas las etapas. No certifican una solución exacta de PDE. Los fallos y ajustes sin converger permanecen en el denominador de aceptación; las tablas identifican también medianas de toda salida finita y medianas restringidas a convergentes.

Se declara ρ = 1 constante. La masa total es entonces trivial en reconstrucción de velocidad, y copiar la densidad de referencia no prueba conservación aprendida. El diagnóstico de la ecuación de vorticidad se aplica sólo a los dos casos newtonianos y conserva el defecto temporal de las instantáneas guardadas. No se omite la fuerza polimérica para afirmar una ecuación newtoniana en casos no newtonianos.

Para los viscoplásticos se calculan a partir del gradiente reconstruido la viscosidad aparente, el esfuerzo instantáneo regularizado y disipación. Esto es un diagnóstico de cierre constitutivo usando sus parámetros conocidos, no una nueva estimación de los parámetros ni un modelo de memoria. El esfuerzo simétrico planar usa norma Frobenius con pesos (1, 2, 1) para (xx, xy, yy). En Oldroyd-B/FENE-P, con esta reconstrucción de velocidad solamente, **no se infiere la conformación ni el esfuerzo polimérico con memoria**.

## Recursos y estados numéricos

Cada ajuste usa un hijo nuevo y un hilo BLAS; se ejecutan hasta dos hijos a la vez. Se separan CPU, tiempo transcurrido, RSS absoluto, memoria visible a Python, número de SVD/least-squares/eigen, iteraciones y un proxy declarado de trabajo dominante. Las mediciones tienen una repetición por condición en un host compartido: no representan intervalos de confianza ni un orden exacto de rendimiento hardware.

El RSS del ajuste incluye entorno y datos heredados. El RSS de los posprocesados es el máximo acumulado durante la vida del mismo hijo y puede heredar picos del ajuste y evaluaciones anteriores; **no es memoria incremental atribuible a la proyección**. CPU y tiempo del ajuste/posprocesados se miden por separado; la evaluación física queda fuera de esos tiempos.

`aggregate_comparison.json` incluye CPU/proxy de trabajo por aceptación operativa, sumando también los intentos perfilados que no fueron aceptados. El cociente es indefinido con cero aceptaciones y se marca como incompleto si hay intentos sin perfil. Cada variante reutiliza un único ajuste: sumar los recursos de las tres variantes como si fueran tres ajustes duplicaría el trabajo.

## Evidencia y reproducción

| Archivo | Función |
|---|---|
|`ledger.jsonl`|Diagnósticos de cada ajuste, perfiles separados, métricas físicas y errores en cada tiempo guardado |
|`summary.json/csv`|Una fila por ajuste/variante, incluidos fallos |
|`group_summary.json`|Resultados por flujo, ruido, método y variante; mediana/mínimo/máximo de semillas |
|`aggregate_comparison.json/csv`|Comparación por ruido y global, con denominadores y medianas finitas/convergentes separadas |
|`pipeline_resource_summary.json`|Recursos por variante: CPU de ajuste + posproceso, trabajo matricial y FFT contados por separado |
|`operational_pareto.json`|Frentes de coste/fidelidad sólo entre métodos con criterios operativos cumplidos en la misma observación |
|`paired_prior_contributions.json`|Cambios pareados al añadir H, corte físico y modelo frente a identidad igualmente filtrada |
|`input_preflight.json`|Verificación de las siete entradas antes de continuar los ajustes |
|`provenance.json`|Configuración, entorno, fuentes, archivos limpios y épocas del driver |
|`integrity_validation.json`|Verificación de controles, aceptación, fuentes, observaciones pareadas y muestras |
|`figures/`|Diagramas cuantitativos y mapas calculados con escalas comunes y saturación declarada |

Desde la raíz del repositorio, teniendo disponibles las siete simulaciones completas:

```bash
python python/scripts/run_model_fidelity_benchmark.py --workers 2
python results/research/model_fidelity_benchmark_2026-10-14/validate_model_comparison.py
python results/research/model_fidelity_benchmark_2026-10-14/analyze_model_comparison.py
```

La reanudación verifica la configuración numérica, hashes de módulos científicos, referencias y observaciones. NS/VE declaran SHA de bytes del archivo; el histórico VP declara un hash canónico de todos los campos, geometría y metadatos. Ambos se verifican correctamente y se conserva por separado SHA del archivo VP. Una corrección exclusivamente de esta lectura de hashes queda registrada mediante épocas del driver, conservando los ajustes NS/VE anteriores y su SHA de ejecución original.

Los NPZ de reconstrucción guardan solamente tres estados calculados (0/medio/final) de semilla 101, con sus índices exactos. No interpolan ni representan una trayectoria completa; el ledger mantiene errores en todos los tiempos. Los mapas usan la misma escala de vorticidad de referencia y anotan la fracción saturada. Las muestras NPZ están ignoradas por Git; las figuras, tablas, código y procedencia son los artefactos publicados.

El piloto inicial y el arranque editorial parcial se conservan como auditoría separada. No se suman a 1050 ni se presentan como realizaciones independientes adicionales.
