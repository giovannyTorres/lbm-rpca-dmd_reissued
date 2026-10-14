# Sensibilidad de rango, intensidad y fidelidad física

Esta campaña contrasta POD y TLS-DMD para comprobar cuánto dependen las conclusiones de fijar un único rango y una única intensidad de ruido. El diseño se fijó antes de ejecutar sus resultados: **360 ajustes**, sobre **36 observaciones diferentes**, y **1080 evaluaciones** de los mismos ajustes con tres tratamientos espaciales. Las tres semillas son nuevas realizaciones de ruido sobre dos trayectorias ya estudiadas; no prueban generalización a flujos nuevos.

Los campos son Navier–Stokes multiescala (128², T=4, 41 estados) y Oldroyd-B (128², T=2.4, 25 estados). No se agrupan como una comparación física común de materiales: tienen trayectorias y horizontes distintos. Se compara el comportamiento de los reconstruidores dentro de cada campo.

Se usan ruido gaussiano IID y ruido impulsivo sobre el 1% de las **entradas de la matriz**, normalizados a RMS global 5%, 20% y 40% de la señal limpia; rangos 2, 4, 8, 12 y 16; semillas 101, 307 y 911. La amplitud sintética usa la referencia limpia para definir el experimento, pero el estimador recibe exclusivamente observación y tiempos. Igual RMS no implica igual dificultad entre distribuciones.

Cada estimación se evalúa sin tratamiento (Raw), con proyección de Helmholtz (H) y con H más corte isotrópico físico k≤10. El corte 10 se hereda de una exploración previa como supuesto de suavidad; no es un parámetro elegido mediante el mínimo de error limpio de esta campaña. Los alias de implementación `pod_r8` y `tls_r8` identifican las familias del benchmark; el rango efectivo de esta campaña lo determina la columna `rank`.

## Cómo leer los resultados

Los errores se calculan sobre el campo completo y toda la ventana temporal, incluyendo todos los modos limpios. Cada celda muestra **mediana [mínimo, máximo] de tres semillas**, en porcentaje. No es un intervalo de confianza. La aceptación exige convergencia y los cuatro límites declarados: error de velocidad ≤15%, vorticidad ≤25%, divergencia extra normalizada ≤10% y energía cinética ≤15%. Cumplirlos es una tolerancia de reconstrucción, no certificación DNS ni solución exacta de las ecuaciones.

Completitud: 360 ajustes; 0 fallos; 0 ajustes sin convergencia. Evaluaciones aceptadas: Raw 0/360; H 12/360; H+k≤10 324/360. Los tres tratamientos reutilizan cada ajuste y no constituyen tres experimentos independientes.

La tabla siguiente muestra el contraste fijado de RMS 20% para rangos 2, 8 y 16. El resto de rangos e intensidades permanece en las tablas completas. H conserva exactamente la vorticidad discreta: su beneficio de divergencia o velocidad no se interpreta como recuperación de gradientes.

### NS multiescala — Gaussiano IID

| Modelo | Rango | Error ω Raw [%] | Error ω H+k≤10 [%] | Error u H+k≤10 [%] | Aceptadas H+k≤10 |
|---|---:|---:|---:|---:|---:|
| POD | 2 | 62.93 [62.80, 62.95] | 40.77 [40.77, 40.78] | 29.95 [29.95, 29.95] | 0/3 |
| POD | 8 | 96.64 [96.47, 96.93] | 11.30 [11.29, 11.30] | 3.40 [3.39, 3.40] | 3/3 |
| POD | 16 | 137.89 [137.53, 137.97] | 11.22 [11.22, 11.22] | 3.36 [3.36, 3.36] | 3/3 |
| TLS-DMD | 2 | 63.83 [63.74, 63.93] | 42.86 [42.84, 42.86] | 31.42 [31.41, 31.42] | 0/3 |
| TLS-DMD | 8 | 104.25 [103.72, 104.36] | 11.70 [11.68, 11.73] | 3.80 [3.78, 3.83] | 3/3 |
| TLS-DMD | 16 | 121.85 [120.71, 123.17] | 11.87 [11.79, 11.89] | 4.14 [4.06, 4.18] | 3/3 |

### NS multiescala — Impulsivo: 1% de entradas

| Modelo | Rango | Error ω Raw [%] | Error ω H+k≤10 [%] | Error u H+k≤10 [%] | Aceptadas H+k≤10 |
|---|---:|---:|---:|---:|---:|
| POD | 2 | 62.84 [62.68, 63.05] | 40.77 [40.76, 40.78] | 29.95 [29.95, 29.95] | 0/3 |
| POD | 8 | 97.26 [96.85, 97.87] | 11.37 [11.35, 11.46] | 3.46 [3.44, 3.52] | 3/3 |
| POD | 16 | 141.26 [140.27, 141.51] | 11.34 [11.31, 11.37] | 3.44 [3.42, 3.46] | 3/3 |
| TLS-DMD | 2 | 63.82 [63.51, 63.98] | 42.84 [42.84, 42.86] | 31.41 [31.41, 31.41] | 0/3 |
| TLS-DMD | 8 | 101.97 [101.82, 102.49] | 11.73 [11.66, 11.80] | 3.79 [3.76, 3.85] | 3/3 |
| TLS-DMD | 16 | 111.04 [110.19, 120.80] | 11.67 [11.46, 12.76] | 3.84 [3.67, 4.93] | 3/3 |

### Oldroyd-B — Gaussiano IID

| Modelo | Rango | Error ω Raw [%] | Error ω H+k≤10 [%] | Error u H+k≤10 [%] | Aceptadas H+k≤10 |
|---|---:|---:|---:|---:|---:|
| POD | 2 | 147.14 [146.43, 147.32] | 19.25 [19.24, 19.26] | 9.30 [9.30, 9.30] | 3/3 |
| POD | 8 | 292.48 [292.34, 292.92] | 6.15 [6.11, 6.16] | 1.29 [1.29, 1.29] | 3/3 |
| POD | 16 | 413.43 [412.63, 413.76] | 8.25 [8.15, 8.35] | 1.67 [1.65, 1.67] | 3/3 |
| TLS-DMD | 2 | 145.47 [144.91, 145.73] | 20.95 [20.94, 20.96] | 10.18 [10.17, 10.18] | 3/3 |
| TLS-DMD | 8 | 249.04 [248.72, 249.75] | 7.25 [7.15, 7.31] | 2.49 [2.41, 2.50] | 3/3 |
| TLS-DMD | 16 | 283.65 [274.20, 295.14] | 9.32 [9.32, 9.83] | 3.69 [3.48, 3.86] | 3/3 |

### Oldroyd-B — Impulsivo: 1% de entradas

| Modelo | Rango | Error ω Raw [%] | Error ω H+k≤10 [%] | Error u H+k≤10 [%] | Aceptadas H+k≤10 |
|---|---:|---:|---:|---:|---:|
| POD | 2 | 145.39 [145.21, 145.61] | 19.26 [19.25, 19.27] | 9.30 [9.30, 9.30] | 3/3 |
| POD | 8 | 298.58 [296.52, 300.38] | 6.66 [6.37, 6.85] | 1.55 [1.42, 1.62] | 3/3 |
| POD | 16 | 418.81 [417.01, 421.53] | 8.38 [8.17, 8.49] | 1.74 [1.68, 1.75] | 3/3 |
| TLS-DMD | 2 | 143.66 [142.99, 144.21] | 20.96 [20.90, 21.01] | 10.18 [10.15, 10.20] | 3/3 |
| TLS-DMD | 8 | 220.53 [218.64, 236.36] | 7.12 [7.06, 7.24] | 2.34 [2.29, 2.45] | 3/3 |
| TLS-DMD | 16 | 248.58 [241.90, 259.32] | 7.83 [7.49, 9.65] | 2.80 [2.59, 3.85] | 3/3 |

## Compresión matricial de la señal limpia y sesgo del filtro

El oráculo POD se calcula **por separado** sobre la matriz limpia para diagnosticar cuánto puede aproximarse con rango reducido. Esta compresión matricial no describe la compresibilidad del fluido. Sólo su error de velocidad en norma Frobenius es una cota inferior entre matrices de rango r. Su error de vorticidad es descriptivo: el SVD no minimiza el error de derivadas. El oráculo no entra en los estimadores ruidosos ni selecciona un rango.

| Campo | Rango | Cota limpia de error u [%] | Error ω de POD limpio [%] |
|---|---:|---:|---:|
| NS multiescala | 2 | 29.89327 | 40.31644 |
| NS multiescala | 4 | 8.99295 | 17.03935 |
| NS multiescala | 8 | 1.41107 | 4.21031 |
| NS multiescala | 12 | 0.32392 | 1.25633 |
| NS multiescala | 16 | 0.07647 | 0.37720 |
| Oldroyd-B | 2 | 9.28263 | 19.03657 |
| Oldroyd-B | 4 | 1.31184 | 4.08975 |
| Oldroyd-B | 8 | 0.08052 | 0.56342 |
| Oldroyd-B | 12 | 0.01169 | 0.12062 |
| Oldroyd-B | 16 | 0.00216 | 0.03689 |

Aplicar el corte k≤10 directamente a la referencia limpia de NS multiescala produce 3.068% de error de velocidad y **10.837% de error de vorticidad**. Por tanto, la mejora al filtrar ruido tiene un coste físico explícito y dependiente del campo.

Aplicar el corte k≤10 directamente a la referencia limpia de Oldroyd-B produce 0.167% de error de velocidad y **1.448% de error de vorticidad**. Por tanto, la mejora al filtrar ruido tiene un coste físico explícito y dependiente del campo.

## Figuras y datos

- [Velocidad frente a rango e intensidad](figures/rank_velocity.pdf).
- [Vorticidad frente a rango e intensidad](figures/rank_vorticity.pdf).
- [Separación de algoritmo, proyección y corte](figures/rank_postprocessing.pdf).
- [Aceptación de cada rango/intensidad](figures/acceptance_rank_noise.pdf).
- [Oráculo limpio y sesgo del prior](figures/clean_oracle_floor.pdf).

Las fuentes inspectables son `ledger.jsonl` (cada ajuste, diagnósticos, perfiles y métricas completas), `summary.csv` (1080 evaluaciones), `aggregates.json` (mediana/mínimo/máximo), `oracle_floor.csv` (diagnóstico limpio), `diagnostic_comparisons.json` y `rank_cost_diagnostics.csv` (360 filas de coste por ajuste real). `analysis_verification.json` verifica hashes científicos y de entrada, las 36 observaciones regeneradas y la conservación de vorticidad bajo H.

La medida de coste separa ajuste y postprocesado de la evaluación. CPU, tiempo de pared, RSS absoluto y memoria trazada se registran con un proceso nuevo y un hilo BLAS por ajuste. El RSS de un postprocesado conserva el pico histórico de ese proceso; no es memoria incremental del filtro. Los proxies aritméticos no son FLOPs medidos. Tres semillas son observaciones diferentes, no tres repeticiones de coste sobre una misma matriz; los tiempos de un equipo compartido son descriptivos.

Los 24 ajustes de rango 8/RMS 20% coinciden con condiciones del benchmark principal si se verifican observación SHA256 y parámetros. Estado de esa comprobación: **24/24** disponibles y verificados en `overlap_verification.json`. Se conservan como análisis de sensibilidad, sin sumarlos como evidencia independiente.

## Reproducción

Desde la raíz del repositorio, con las dependencias instaladas y los dos archivos completos de simulación generados por las campañas de mallas ampliadas:

```bash
python python/scripts/run_rank_noise_sensitivity.py --resume
python results/research/rank_noise_sensitivity_2026-10-14/analysis.py
# Cuando también termine el benchmark principal:
python results/research/rank_noise_sensitivity_2026-10-14/analysis.py --require-main-overlap
```

La caché local valida configuración, fuentes, entradas y condición; no se publica por duplicar el ledger. Los NPZ de reconstrucción se guardan sólo para semilla 101, rango 8 y RMS 20%, con H+k≤10, y figuran en `selected_reconstructions_manifest.json`. Los archivos de muestras visuales de cinco estados del repositorio no sustituyen las historias completas necesarias para recalcular las métricas. `provenance.json` conserva los contratos de los helpers y sus hashes al inicio: ajustes posteriores de lectura de otros materiales no se presentan como un módulo idéntico.
