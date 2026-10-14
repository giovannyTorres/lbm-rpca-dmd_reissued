# Protocolo de ampliación de la investigación LBM–RPCA–DMD

Documento científico de la fase 8. Las fuentes sustentan métodos y antecedentes; los resultados del baseline de 14 de octubre de 2026 y las campañas nuevas se identifican por separado. La bibliografía reutilizable está en [references.bib](references.bib).

## Introducción

DMD en un wake viscoelástico ya aparece en Hamid, Sasmal y Chhabra (2022, [R18]); DMD en EVP aparece en Raffi y colaboradores (2024, [R19]). Corrochano y colaboradores publicaron POD/HODMD en wakes EVP en 2026 ([R20]). Por ello, una hipótesis nueva y comprobable sería: **la selección conjunta de limpieza y modelo dinámico depende del tipo de contaminación, el régimen reológico y la fidelidad física exigida; la mejora en error de velocidad puede coexistir con degradación de vorticidad o conservación**. Comparar esas dependencias con costo verificable y evaluación fuera de muestra tiene mayor alcance que añadir un fluido al catálogo.


El protocolo se organiza en Introducción, Marco teórico, Desarrollo experimental, Resultados y discusión, y Conclusiones y líneas de trabajo futuro. La fase 8 añade el paquete `fluid_denoise.research` a la arquitectura existente; las fases 1–7 mantienen sus contratos y numeración. La guía de integración está en [phase8_research.md](../phase8_research.md).

Una comparación entre RPCA–IALM y DMD clásico bajo ruido impulsivo sólo permite conclusiones sobre esa contaminación y esas formulaciones. El siguiente mapa distingue los métodos implementados de PIV, campos 3D, online/GPU, aprendizaje profundo y otras campañas que requieren evidencia propia.

| Pendiente o petición | Módulo / artefacto de ampliación | Alcance y criterio científico |
|---|---|---|
| Costo más robusto que tiempo | `python/src/fluid_denoise/research/cost.py` | Vector CPU/wall/RSS, dimensiones, estimación de trabajo SVD, repeticiones, Pareto y calidad exigida. |
| Conservación y vorticidad | `python/src/fluid_denoise/research/metrics.py` | Error u/ω, divergencia, materia/flujo según datos, energía/enstrofía; fronteras explícitas. |
| Ruido que no favorezca PCP | `python/src/fluid_denoise/research/noise.py` | Nueve modelos ejecutados: `gaussian`, `sparse_impulsive`, `mixed`, `spatial_correlated`, `temporal_correlated`, `column_outliers`, `student_t`, `heteroscedastic` y `sensor_drift`; realizaciones pareadas por método. Observaciones faltantes y ruido de proceso requieren otra campaña. |
| Variantes robustas / otros algoritmos | `python/src/fluid_denoise/research/algorithms.py` | SVD/POD, PCP, RPCA estable penalizada, Outlier Pursuit, exact/TLS/fb/optimized DMD y cadenas de limpieza; registrar formulación exacta. BOP/HODMD/piDMD son extensiones posteriores. |
| DNS más cercana a NS | `python/src/fluid_denoise/research/dns.py` | Navier–Stokes 2D periódico, pruebas Taylor–Green y campos interacting/decaying; referencia analítica separada de integración numérica. |
| Referencias reológicas | `fluid_denoise.research.rheology` | Canal HB/Bingham exacto y startup Oldroyd-B 1D; controles constitutivos separados de los solvers 2D. |
| Viscoelasticidad 2D | `fluid_denoise.research.viscoelastic_dns` | Oldroyd-B/FENE-P periódicos con evolución de conformación y realimentación de esfuerzo en momento; parámetros moderados verificados. |
| Viscoplasticidad 2D | `fluid_denoise.research.viscoplastic_dns` | HB/Bingham regularizados con esfuerzo variable acoplado a momento; declarar regularización y refinamiento. |
| Pronóstico cronológico | `fluid_denoise.research.forecasting` y `forecast_runner` | Ajuste y limpieza sólo con entrenamiento; selección con validación ruidosa y evaluación futura separada. |
| Restricciones físicas | `fluid_denoise.research.physics` | Proyección de Helmholtz y corte de banda predeclarado en dominios periódicos sin sólidos; no certifican la ecuación de movimiento. |
| RPCA→DMD y reproducibilidad | Runner de campaña + configuración | Casos, semillas, rangos fijos, repeticiones, fallos, tablas/figuras y manifiestos de procedencia. |
| Wakes Oldroyd-B/FENE-P/EVP | `docs/research/NON_NEWTONIAN.md` y manifiestos externos | Re=100, fronteras/constitutiva explícitas, log-conformation y validación; campaña HPC pendiente. |
| PIV, 3D, online/GPU, deep learning, industrial | Líneas futuras de este protocolo | No ejecutadas por añadir un manifiesto. Necesitan datos/métodos/validación propios. |

Hipótesis a contrastar: H1, la jerarquía de algoritmos cambia con ruido denso/correlacionado; H2, minimizar L2 de velocidad no garantiza preservar vorticidad o continuidad; H3, el menor tiempo puede incumplir la fidelidad física exigida; H4, elasticidad y cedencia alteran rango/modalidad y transferibilidad. Son hipótesis, no conclusiones.


## Marco teórico

### Costo computacional condicionado a calidad

No existe un único número independiente del hardware y del objetivo científico. Usar dos niveles:

1. **Vector de recursos:** wall time, tiempo CPU del proceso, memoria residente pico, energía sólo si un contador real está disponible, bytes de datos y modelo, número de iteraciones y de factorizaciones, fracasos y convergencia. Las asignaciones NumPy o `tracemalloc` no sustituyen la RSS de bibliotecas nativas. Medir cada ensayo en un proceso nuevo permite atribuir su pico de memoria. Una estimación de complejidad o FLOPs debe denominarse estimación, nunca contador hardware.
2. **Costo a calidad fijada:** elegir el menor recurso usado por una configuración que supera simultáneamente criterios predeclarados de error de velocidad, vorticidad y conservación, en datos de validación. Una configuración que no los alcanza cuenta como fracaso; no se elimina para mejorar su promedio. Mostrar frontera de Pareto costo–error y perfiles de desempeño [R1].

Conservar el tiempo, pero reportar mediana, dispersión e intervalos entre réplicas; fijar precisión, hilos BLAS, versiones y máquina. Los principios de Roofline [R2] motivan registrar cómputo y movimiento de datos: dos SVD con igual orden asintótico pueden estar limitadas por factores distintos. El conteo de operaciones no predice por sí solo el tiempo en CPU/GPU.

Para tamaños m×n y rango r, declarar el algoritmo de SVD: densa, truncada o aleatoria; registrar pasadas, sobre-muestreo y semillas [R3]. No comparar una implementación optimizada con otra deliberadamente densa sin separar el efecto de implementación. Amortizar el entrenamiento únicamente declarando el número real de reconstrucciones/pronósticos. El costo debe incluir RPCA más DMD, ajuste de hiperparámetros y predicción, distinguiendo esos componentes.

### Métricas físicas y referencia limpia

La norma con pesos de cuadratura ||a||_W² = Σ_i w_i |a_i|² es una opción metodológica para mallas generales. **Implementación actual:** `metrics.py` calcula errores de velocidad/vorticidad con `np.linalg.norm`, sin pesos, sobre nodos de fluido/válidos; usa pesos de cuadratura para masa, energía, enstrofía y flujos. La campaña conserva esa definición nodal y no afirma haber implementado el error ponderado. Aplicar la misma malla y operador derivativo a reconstrucción y referencia limpia. No diferenciar atravesando una máscara sólida; usar estencils válidos o campos del solver.

| Métrica | Definición recomendada | Cuidado |
|---|---|---|
| Error de velocidad | ||u_hat−u_ref||_2 / ||u_ref||_2 | Separar error instantáneo, promedio temporal y pronóstico. Denominador casi nulo necesita escala fija documentada. |
| Vorticidad | ω = ∂_x v−∂_y u; error ||ω_hat−ω_ref||_2/||ω_ref||_2 | La diferenciación amplifica ruido; fijar resolución, operador y filtro. En 3D usar curl vectorial. |
| Continuidad local | RMS(∇·u_hat) y RMS(∇·u_hat−∇·u_ref)/RMS(ω_ref) en la implementación actual | En incomprensibles de densidad constante, M=ρV es constante por definición y no demuestra conservación de la reconstrucción. |
| Flujo neto | |Σ_f ρ_f u_f·n_f A_f| / Q_scale | Requiere integración orientada de todas las fronteras. Periodicidad se verifica con caras emparejadas; no tratar fronteras periódicas como inlet/outlet. |
| Balance de materia | |dM/dt + Σ_f ρ_f u_f·n_f A_f| / (M_scale/T_scale), M=Σ_iρ_iV_i | Para LBM compresible usar ρ y u consistentes. Para incomprensibles reportar divergencia y flujo. |
| Energía cinética | E=½Σ_iρ_i |u_i|² V_i; error de E(t) | En flujo abierto o forzado no exigir decaimiento monótono. |
| Enstrofía | Z=½Σ_i |ω_i|² V_i; error de Z(t) | Es sensible a pérdida de estructuras pequeñas; complementa L2 de velocidad. |
| Circulación (línea futura) | Γ_C=∮_C u·dl o integral de ω en una región | No dividir por Γ_ref≈0; usar escala UL. Elegir contornos fuera del sólido. |
| Frecuencia de shedding (línea futura) | St=fD/U y error contra referencia | Ventana suficientemente larga, espectro e incertidumbre. Un pico de velocidad no sustituye lift si no se dispone de fuerzas. |
| Residuo de momento (línea futura) | ∂_t u+u·∇u+∇p/ρ−ν_sΔu−∇·τ_p/ρ−f | Sólo si existen p, esfuerzos, condiciones de contorno y fuerza. No omitir presión/esfuerzos y llamar al resultado residuo Navier–Stokes. |

Adicionales por reología: error de esfuerzos poliméricos, positividad del tensor de conformación, traza y energía elástica apropiada al modelo; área de regiones cedidas/no cedidas para viscoplásticos con definición y tolerancia explícitas. Una regularización de Papanastasiou aproxima la transición, no produce una superficie de cedencia exacta [R16].

Una proyección de Helmholtz periódica puede imponer divergencia casi nula y constituye un baseline físico; no demuestra que la dinámica, vorticidad o energía se hayan reconstruido. En dominios con cilindro requiere un problema de presión y condiciones de contorno adecuados. La formulación piDMD [R12] permite imponer estructuras justificadas, pero **incompresibilidad no implica operador DMD unitario**: un flujo viscoso disipa energía.

### Contaminación, identificabilidad y variantes de modelos

Modelo de observación separado de física: X_obs=X_clean+E. PCP [R4] supone estructura de bajo rango más corrupción dispersa; no es un denoiser universal. SPCP [R5] admite además perturbaciones densas acotadas; su parámetro de ruido debe calibrarse. Outlier Pursuit [R6] usa corrupción de columnas: una columna corresponde a un snapshot completo si las columnas son tiempo. No garantiza recuperar cada entrada de esas columnas, sino subespacio y detección bajo sus hipótesis.

La reconstrucción ejecutada utiliza nueve modelos: gaussiano denso, impulsivo disperso, mixto, correlacionado espacial, correlacionado temporal AR(1), outliers de snapshots completos, Student-t, heteroscedástico y deriva de sensor. Igualan la RMS realizada sobre fluido dentro de la ventana reconstruida. El pronóstico calibra la intensidad sólo con entrenamiento y no renormaliza los bloques futuros. Observaciones faltantes y ruido de proceso permanecen pendientes. El catálogo distingue sus supuestos:

- Gaussiano aditivo blanco a varios SNR, sin outliers; baseline POD/SVD de rango fijo. El umbral de Gavish–Donoho [R7] es una propuesta posterior y no está implementado en el piloto. Su optimalidad es asintótica bajo ruido blanco y estructura baja en rango, no vale sin más para ruido correlacionado.
- Gaussiano temporal AR(1) y espacial filtrado con escala de correlación especificada. Esto puede producir estructuras coherentes confundibles con señal de bajo rango. Normalizar la energía de contaminación después del filtrado; no identificarlo con ruido blanco.
- Heteroscedástico, proporcional a magnitud local con piso de ruido; Student-t de varianza finita (p.ej. ν=3); sesgo de sensor/drift; bloques o snapshots completos corruptos; combinaciones denso+disperso.
- Ruido de observación y perturbación de condiciones iniciales/forzado son experimentos diferentes. Lo segundo exige regenerar soluciones físicas y cambia el sistema; no se simula sumando ruido al dataset final.
- Sensores faltantes deben representarse con máscara explícita. Imputar ceros y tratarlos como dato observado sesga RPCA/DMD; introducir métodos de completion sólo con diseño específico.

Matriz mínima de métodos:

| Función | Baselines / variantes | Hipótesis y riesgo |
|---|---|---|
| Limpieza | Identidad; SVD/POD; PCP; SPCP; Outlier Pursuit | Presupuestos explícitos: rango fijado para POD/DMD y regularización para PCP/SPCP; registrar rango efectivo y falsas eliminaciones de vórtices intermitentes. |
| Dinámica | Exact DMD [R8]; TLS-DMD y forward–backward [R9] | TLS corrige errores en ambas matrices de snapshots bajo supuestos apropiados; puede ser inestable a rango/condicionamiento. fbDMD introduce inversión y raíz matricial, con ambigüedades y mal condicionamiento. |
| Ajuste global | Optimized DMD [R10]; BOP-DMD [R11] | Optimized DMD resuelve un problema no lineal y depende de inicialización; bagging da variabilidad de modelos, no cobertura garantizada. Para series correlacionadas, usar bloques temporales o explicar las hipótesis del muestreo. |
| Alternativas | piDMD [R12]; HODMD [R13]; Subspace DMD [R14]; robust optimized DMD [R15] | HODMD introduce retardos y más memoria; Subspace DMD distingue ruido de observación/proceso con hipótesis estocásticas; pérdida robusta de DMD evita limpieza previa pero cambia el ajuste. |
| Análisis modal | SPOD [R24] | Útil para estructuras coherentes dependientes de frecuencia de procesos estacionarios; no es directamente un pronosticador de snapshots. |

La implementación local `stable_rpca` resuelve la formulación **penalizada** ½||M−L−S||_F² + α||L||_* + β||S||_1 mediante proximales alternados, con residuo denso explícito. No es el programa con restricción ||M−L−S||_F≤δ de Zhou et al. [R5]; su teoría no se transfiere automáticamente a cada α/β o caso de ruido. Si se usa la intensidad conocida del ruido sintético para estimar σ, debe declararse como información del experimento; no es una medición disponible de forma automática en PIV real. El rango fijado en configuración se aplica a POD/DMD; los modelos convexos seleccionan rango efectivo mediante regularización y deben reportarlo. La comparación no afirma rangos idénticos entre todas las familias.

No confundir implementar DMD y luego filtrar autovalores con implementar optimized/TLS/BOP-DMD. Reutilizar paquetes científicos versionados y verificar qué formulación implementan; contrastar casos pequeños con un sistema lineal de autovalores conocidos. HODMD y ResDMD [R25] son líneas distintas: la última ayuda a estudiar contaminación espectral/consistencia Koopman, sin garantizar fidelidad completa del campo.

## Desarrollo experimental

### Jerarquía de referencias y simulaciones

**Nivel A: referencias exactas.** Taylor–Green 2D periódico (no equipararlo al problema turbulento 3D original [R21]), Poiseuille/Couette newtonianos, flujo de canal Herschel–Bulkley/Bingham con plug y pruebas de relajación/arranque Oldroyd-B. Si los campos se evalúan directamente de la fórmula, el dataset es una referencia analítica, no una DNS ejecutada. La solución manufacturada [R22] verifica orden del código introduciendo la fuerza exacta; no valida por sí sola un fenómeno natural.

**Nivel B: Navier–Stokes newtoniano numérico.** Solver pseudoespectral de vorticidad–función de corriente con dealiasing y malla periódica, o proyección presión–velocidad tipo [R23]. Primero verificar Taylor–Green con refinamiento de malla/paso; luego double-shear-layer, Kolmogorov forzado y vórtices interactuantes. La solución analítica de Kolmogorov laminar sirve de prueba; un régimen complejo necesita resolución temporal/espacial y convergencia. Mantener LBM como comparación y declarar error de compresibilidad/Mach. Un campo dibujado a partir de vórtices superpuestos no equivale a resolver Navier–Stokes.

**Nivel C: wake de cilindro.** Reproducir la geometría y las condiciones exactas del benchmark Schäfer–Turek [R26] antes de comparar drag/lift/caída de presión/Strouhal. Los valores de un cilindro no confinado no son referencias intercambiables con un canal confinado. Para viscoelásticos considerar Oldroyd-B y FENE-P con tensor de conformación y estabilización logarítmica [R17]. Para viscoplásticos considerar Herschel–Bulkley/Papanastasiou [R16] y, después, EVP con Saramito–Herschel–Bulkley [R27].

La variación de viscosidad sola representa un modelo generalizado newtoniano, **no elasticidad**. Oldroyd-B requiere memoria de esfuerzos; FENE-P agrega extensibilidad finita; EVP requiere ley constitutiva con elasticidad y cedencia. Un modelo de viscosidad regularizada Bingham no es Saramito. La campaña de cylinder EVP de [R20] reporta recursos HPC muy grandes; no atribuir fidelidad comparable a un prototipo corto en CPU. Los manifiestos de casos y la preparación de integración externa son entregables reproducibles, pero deben etiquetarse pendientes hasta ejecutar y verificar el solver.

Validación por niveles: (a) residuo/orden analítico; (b) refinamiento ≥3 mallas y pasos adecuados; (c) conservación y sensibilidad a dominio/fronteras; (d) comparación de observables con referencia; (e) régimen dinámico. No llamar caótico a un campo sólo por apariencia; usar espectros, retorno, estadísticos y, cuando viable, indicadores dinámicos. Una simulación 2D no respalda afirmaciones de turbulencia 3D ni una descripción completa de sus inestabilidades.


### Campaña reproducible de esta ampliación

La campaña inicial usa **tres semillas de contaminación y tres repeticiones de costo por configuración**. Las semillas comparan robustez; las repeticiones comparan variabilidad de recursos sobre la misma entrada. No son nueve realizaciones independientes de ruido. Los rangos y los hiperparámetros se fijan en configuración: **la referencia limpia sólo se usa para evaluar y no para seleccionar el mejor rango o ajuste**. El baseline `baseline_2026-10-14/pilot_2026-10-14` usa NS Taylor–Green 24×24, interacting 48×48, decaying 64×64, cuatro casos reológicos 33×8 nodos y un cilindro LBM de referencia 80×40. Los hashes en su procedencia corresponden a esa ejecución y no verifican por sí solos el paquete integrado de fase 8. Cada campaña nueva registra sus propias fuentes, configuración y entorno. Una campaña posterior puede calibrar con una partición de validación separada, con presupuesto igual por método.

Se ejecutaron dos tareas distintas. La reconstrucción usa el intervalo observado; RPCA y DMD ajustados a esa ventana no prueban pronóstico. La campaña de pronóstico sí utiliza separación cronológica 60% entrenamiento, 20% validación y el resto test: limpieza y ajuste sólo sobre entrenamiento, selección entre 29 opciones con MSE de validación ruidosa y evaluación limpia futura después de congelar la elección. No se reajusta con validación ni se limpia toda la serie antes de dividir. La interpretación de MSE ruidoso como riesgo limpio requiere independencia y media cero; correlación temporal y outliers pueden sesgarla. Véase [FORECASTING.md](FORECASTING.md).

Las simulaciones y perfiles deben informar campos u/v/ρ/p/τ/C disponibles, malla, paso, fronteras, unidades y escalas, máscara, parámetros Re/Wi/Bn/n/β, precisión y solver/commit. Dejar vacíos explícitos los campos no disponibles; no sustituir esfuerzos elásticos por una viscosidad efectiva.

1. Verificar Taylor–Green antes de usar el solver NS en bancos no analíticos. Conservar el error de velocidad/vorticidad contra solución exacta y resultados de refinamiento.
2. Verificar el perfil Herschel–Bulkley contra la ley constitutiva, el balance de presión/esfuerzo y la posición de plug. Verificar startup Oldroyd-B contra el límite estacionario, relajación y refinamiento. El perfil evaluado analíticamente se etiqueta `analytical_reference`, no `dns`.
3. Inyectar ruido con realizaciones pareadas por método; registrar SNR y energía realmente obtenidos, además de los parámetros de distribución/correlación.
4. Ejecutar algoritmos con rangos/configuraciones fijas; registrar convergencia, excepciones y configuraciones inviables. Si PCP no converge, conservar ese resultado como fallo, sin clasificarlo como reconstrucción satisfactoria.
5. Medir cada réplica de costo en un proceso aislado para atribuir RSS y CPU. Registrar tamaño de entrada, precisión, hilos, SVD/rango y entorno. Etiquetar explícitamente `estimated_flops` o `work_proxy`; no llamarlo FLOPs medidos ni costo de energía.
6. Evaluar con los mismos operadores y fronteras en referencia/reconstrucción; registrar campos físicos y costo por ejecución en tablas crudas. Definir gates de calidad antes de ver el resultado; mostrar todos los casos y Pareto, sin ocultar fallos.
7. Producción HPC posterior: refinar wakes newtonianos y reológicos, calibrar contra benchmarks y documentar convergencia antes de generar bancos para reconstrucción. Mantener manifiesto de estado `planned` hasta ejecutar/verificar.

El runner calcula dos pertenencias Pareto dentro de cargas equivalentes (mismo caso, entrada, precisión y ruido): exploratoria entre configuraciones convergentes y físicamente admisible entre aquellas que también satisfacen los gates declarados; no interpretar un algoritmo de una malla pequeña como dominante sobre uno de otra malla mayor. Reportar también el total de configuraciones exitosas y sus errores para evitar selección por conveniencia.



## Resultados y discusión

La campaña integrada `extended_physics_2026-10-14` completó **2808 condiciones y 8424 ajustes**: 8 casos × 9 ruidos × 13 métodos/variantes × 3 semillas, con 3 repeticiones de costo. Registró 0 ajustes fallidos, 6 condiciones no convergidas y 430 aceptadas. De 216 grupos caso/ruido/semilla, 150 tienen algún estimador admisible y 66 ninguno; ruido temporal y deriva de sensor no producen condiciones aceptadas. El pronóstico completó 60 escenarios —5 flujos × 4 ruidos × 3 semillas— con 29 opciones predeclaradas por escenario. Los resultados se inspeccionan en [EXTENDED_RESULTS.md](EXTENDED_RESULTS.md) y sus tablas enlazadas.

Las comparaciones consolidadas de error futuro de velocidad usan una diferencia absoluta mayor de `1e-10` en error L2 relativo: 48/60 escenarios mejoran persistencia cruda y 12 empatan; 43/60 mejoran persistencia filtrada y 17 empatan. Ninguno es materialmente peor con esa tolerancia. Los conteos directos del artefacto crudo pueden reflejar diferencias de redondeo y se conservan; la regla de comparación y el resumen consolidado identifican esta distinción. La vorticidad de pronóstico en casos no lineales conserva errores grandes: esos conteos no demuestran pronóstico físicamente fiable de todos los campos.

El baseline previo queda preservado en `results/research/baseline_2026-10-14/`, con sus propias fuentes, configuración, hashes y entorno, y se presenta en [RESULTS.md](RESULTS.md). No se atribuye retrospectivamente a la campaña integrada. Cada ejecución debe acompañarse de CSV/JSON, configuración, fecha, procedencia, validación y figuras.

Una lectura defendible del runner debe responder:

- ¿Qué algoritmo mejora u y cuál mejora ω frente a identidad/SVD en cada distribución y SNR? Una mejora visual aislada no prueba robustez.
- ¿Se preservan continuidad, flujo, energía y enstrofía? En incomprensibles ρV constante no acredita que u sea físicamente válida.
- ¿La preferencia entre métodos cambia al imponer calidad física mínima? Mostrar recursos, incertidumbre, dimensiones y fallos, evitando un score compuesto arbitrario que mezcle unidades.
- ¿Qué ocurre en casos fuera del rango/SNR del ajuste? La ejecución in-sample sólo justifica reconstrucción, no generalización ni forecast.
- ¿Qué dinámica y esfuerzo conservan los flujos periódicos 2D HB/Bingham y Oldroyd-B/FENE-P al refinar? No extrapolar sus resultados a estelas de cilindro, alto Wi, caos elástico o transición 3D.

Plantilla de tabla por caso: `case_id`, solver/reference_kind y validación, ruido/parámetros/SNR, semilla, algoritmo/cadena, rango, tolerancia, convergencia, error_u, error_omega, continuidad/flux/materia, error_E, error_Z, CPU_s, wall_s, peak_RSS, work_dimensions y `estimated_flops`. Usar valores nulos con causa documentada para métricas que exigen información ausente. Tres semillas ofrecen una exploración inicial; no sostienen generalizaciones estadísticas amplias sobre todos los regímenes.

Para conclusiones sobre física, comparar contra la referencia limpia validada del mismo solver/malla y, cuando exista, contra la solución exacta. Separar error del solver, ruido observado, error de reconstrucción y sesgo del operador derivativo. Si una referencia aún no está validada, sus resultados se reportan como exploratorios.



## Conclusiones y líneas de trabajo futuro

La ampliación transforma la comparación bajo ruido impulsivo en un programa verificable de **algoritmo × contaminación × física × costo**. Su conclusión metodológica es que una comparación bajo una sola contaminación no implica superioridad universal de RPCA. Cada afirmación sobre una campaña nueva requiere sus resultados e incertidumbres después de ejecutar.

La contribución defendible combina banco con procedencia y validación, comparación justa ante ruido que incumple esparsidad, métricas diferenciales/de conservación y costo condicionado a calidad. No reclama primera aplicación de DMD a viscoelasticidad o EVP, pues [R18]–[R20] ya la documentan.

### Avance por etapas y criterios de aceptación

1. **Reproducibilidad local ejecutada:** métricas/costo/ruido/algoritmos, NS periódico, controles de canal y solvers reológicos 2D; campañas completas y procedencia. La verificación consolidada registró 212 tests y 28 subtests sin omisiones, y la ejecución de fases 1–7 produjo 12 experimentos y 80 figuras.
2. **Pronóstico ejecutado y generalización pendiente:** 60 escenarios cronológicos con selección ruidosa; transferencia a SNR/Re/Wi/Bn no usados, calibración de incertidumbre y pronósticos físicamente fiables requieren evidencia adicional. Separar denoising offline de inferencia causal.
3. **Wakes no newtonianos:** benchmark newtoniano, HB/Papanastasiou, Oldroyd-B/FENE-P y Saramito-HB EVP, en ese orden; log-conformation, resolución de capas de esfuerzo y sensibilidad a fronteras y regularización. Integrar solver externo sólo con versión/commit, configuración y pruebas.
4. **Escala/solvers:** 3D y GPU/online tras comprobar conservación y precisión; comparar SVD truncada/aleatoria y variantes incrementales. Un objetivo de reconstrucción sub-segundo requiere una medición propia y no se deduce de usar GPU.
5. **Datos reales y otros modelos:** mediciones PIV con máscaras, ruido/calibración y revisión experimental; SPOD/HODMD/ResDMD, pérdidas robustas y deep learning con conjuntos de entrenamiento/test independientes. Una red entrenada en la misma realización no constituye evidencia de superioridad general.
6. **Integración de uso:** scripts y esquemas estables para adquisición–limpieza–análisis–reporte después de validar etapas anteriores. Los entregables de esta fase son código, configuraciones, documentación y artefactos científicos inspeccionables.

La escala HPC del antecedente EVP [R20] hace necesario presupuestar una campaña distinta para wakes resueltos. Un manifiesto de diez casos no equivale a diez DNS producidas; una calle de vórtices sintética no resuelve la ecuación constitutiva. Esta distinción debe acompañar el documento científico y cualquier presentación de resultados.


## Artefactos para revisión

- [Resultados integrados](EXTENDED_RESULTS.md), [pronóstico](FORECASTING.md), [restricciones físicas](PHYSICS_CONSTRAINTS.md) y [segunda revisión](SECOND_REVIEW.md).
- [Oldroyd-B/FENE-P 2D](VISCOELASTIC_2D.md) y [HB/Bingham 2D](VISCOPLASTIC_2D.md).
- [Resumen consolidado y tolerancia de comparación](../../results/research/extended_summary.json).
- [Reconstrucción: tabla completa](../../results/research/extended_physics_2026-10-14/summary.csv) y [pronóstico: tabla completa](../../results/research/forecasting_extended_2026-10-14/summary.csv).
- [Comparación reológica animada](../../results/research/rheology_matched_2026-10-14/vortices_rheology.gif) y [campos finales en PDF](../../results/research/rheology_matched_2026-10-14/final_vorticity.pdf).
- [Documento científico PDF](../../results/research/document/extension.pdf) y [fuente LaTeX](extension.tex).

Los solvers periódicos no certifican DNS universal, alto Wi ni caos alrededor de cilindros. Estelas externas EVP, 3D, PIV real, GPU/online, aprendizaje profundo, BOP/HODMD/piDMD/SPOD/ResDMD y completion siguen como trabajos posteriores.

## Fuentes bibliográficas públicas

Las páginas editoriales, manuscritos y preprints enlazados permiten consultar las fuentes primarias y sus metadatos. Algunos enlaces editoriales sólo ofrecen resúmenes o metadatos. La bibliografía no implica revisión exhaustiva del texto completo de cada fuente ni ejecución de los métodos citados.

### Tabla de fuentes primarias y claves de bibliografía

| Clave en references.bib | Fuente | DOI / documento primario |
|---|---|---|
| `dolan2002profiles` | Benchmarking optimization software with performance profiles | [10.1007/s101070100263](https://doi.org/10.1007/s101070100263) |
| `williams2009roofline` | Roofline: an insightful visual performance model for multicore architectures | [10.1145/1498765.1498785](https://doi.org/10.1145/1498765.1498785) |
| `halko2011randomness` | Finding Structure with Randomness: Probabilistic Algorithms for Constructing Approximate Matrix Decompositions | [10.1137/090771806](https://doi.org/10.1137/090771806) |
| `candes2011rpca` | Robust principal component analysis? | [10.1145/1970392.1970395](https://doi.org/10.1145/1970392.1970395) |
| `zhou2010stable` | Stable Principal Component Pursuit | [10.1109/ISIT.2010.5513535](https://doi.org/10.1109/ISIT.2010.5513535) |
| `gavish2014threshold` | The Optimal Hard Threshold for Singular Values is 4/√3 | [10.1109/TIT.2014.2323359](https://doi.org/10.1109/TIT.2014.2323359) |
| `schmid2010dmd` | Dynamic mode decomposition of numerical and experimental data | [10.1017/S0022112010001217](https://doi.org/10.1017/S0022112010001217) |
| `dawson2016noise` | Characterizing and correcting for the effect of sensor noise in the dynamic mode decomposition | [10.1007/s00348-016-2127-7](https://doi.org/10.1007/s00348-016-2127-7) |
| `askham2018optimized` | Variable Projection Methods for an Optimized Dynamic Mode Decomposition | [10.1137/M1124176](https://doi.org/10.1137/M1124176) |
| `sashidhar2022bagging` | Bagging, optimized dynamic mode decomposition for robust, stable forecasting with spatial and temporal uncertainty quantification | [10.1098/rsta.2021.0199](https://doi.org/10.1098/rsta.2021.0199) |
| `baddoo2023physics` | Physics-informed dynamic mode decomposition | [10.1098/rspa.2022.0576](https://doi.org/10.1098/rspa.2022.0576) |
| `leclainche2017hodmd` | Higher Order Dynamic Mode Decomposition | [10.1137/15M1054924](https://doi.org/10.1137/15M1054924) |
| `takeishi2017subspace` | Subspace dynamic mode decomposition for stochastic Koopman analysis | [10.1103/PhysRevE.96.033310](https://doi.org/10.1103/PhysRevE.96.033310) |
| `askham2022robust` | Robust and Scalable Methods for the Dynamic Mode Decomposition | [10.1137/21M1417405](https://doi.org/10.1137/21M1417405) |
| `papanastasiou1987yield` | Flows of Materials with Yield | [10.1122/1.549926](https://doi.org/10.1122/1.549926) |
| `fattal2004log` | Constitutive laws for the matrix-logarithm of the conformation tensor | [10.1016/j.jnnfm.2004.08.008](https://doi.org/10.1016/j.jnnfm.2004.08.008) |
| `fattal2005time` | Time-dependent simulation of viscoelastic flows at high Weissenberg number using the log-conformation representation | [10.1016/j.jnnfm.2004.12.003](https://doi.org/10.1016/j.jnnfm.2004.12.003) |
| `hamid2022wake` | Dynamic mode decomposition analysis and fluid-mechanical aspects of viscoelastic fluid flows past a cylinder in laminar vortex shedding regime | [10.1063/5.0122103](https://doi.org/10.1063/5.0122103) |
| `raffi2024evp` | Elastoviscoplastic flows past a cylinder: Fluid-mechanical aspects and dynamic mode decomposition analysis | [10.1063/5.0224004](https://doi.org/10.1063/5.0224004) |
| `corrochano2026evp` | The coherent structures of EVP fluid flow past a circular cylinder | [10.1007/s00162-026-00775-3](https://doi.org/10.1007/s00162-026-00775-3) |
| `taylorgreen1937` | Mechanism of the production of small eddies from large ones | [10.1098/rspa.1937.0036](https://doi.org/10.1098/rspa.1937.0036) |
| `roache2002mms` | Code Verification by the Method of Manufactured Solutions | [10.1115/1.1436090](https://doi.org/10.1115/1.1436090) |
| `chorin1968ns` | Numerical solution of the Navier-Stokes equations | [10.1090/S0025-5718-1968-0242392-2](https://doi.org/10.1090/S0025-5718-1968-0242392-2) |
| `orszag1972dns` | Numerical Simulation of Three-Dimensional Homogeneous Isotropic Turbulence | [10.1103/PhysRevLett.28.76](https://doi.org/10.1103/PhysRevLett.28.76) |
| `towne2018spod` | Spectral proper orthogonal decomposition and its relationship to dynamic mode decomposition and resolvent analysis | [10.1017/jfm.2018.283](https://doi.org/10.1017/jfm.2018.283) |
| `colbrook2023resdmd` | Residual dynamic mode decomposition: robust and verified Koopmanism | [10.1017/jfm.2022.1052](https://doi.org/10.1017/jfm.2022.1052) |
| `saramito2009hb` | A new elastoviscoplastic model based on the Herschel-Bulkley viscoplastic model | [10.1016/j.jnnfm.2008.12.001](https://doi.org/10.1016/j.jnnfm.2008.12.001) |
| `saramito2007evp` | A new constitutive equation for elastoviscoplastic fluid flows | [10.1016/j.jnnfm.2007.04.004](https://doi.org/10.1016/j.jnnfm.2007.04.004) |
| `oldroyd1950` | On the formulation of rheological equations of state | [10.1098/rspa.1950.0035](https://doi.org/10.1098/rspa.1950.0035) |
| `herschel1926bulkley` | Konsistenzmessungen von Gummi-Benzollösungen | [10.1007/BF01432034](https://doi.org/10.1007/BF01432034) |
| `bird1980fenep` | Polymer solution rheology based on a finitely extensible bead—spring chain model | [10.1016/0377-0257(80)85007-5](https://doi.org/10.1016/0377-0257(80)85007-5) |
| `pimenta2017rheotool` | Stabilization of an open-source finite-volume solver for viscoelastic fluid flows | [10.1016/j.jnnfm.2016.12.002](https://doi.org/10.1016/j.jnnfm.2016.12.002) |
| `xu2010outlier` | Robust PCA via Outlier Pursuit | [Fuente primaria](https://papers.nips.cc/paper_files/paper/2010/hash/fe8c15fed5f808006ce95eddb7366e35-Abstract.html) |
| `schaefer1996cylinder` | Benchmark Computations of Laminar Flow Around a Cylinder | [Fuente primaria](https://www.mathematik.tu-dortmund.de/lsiii/cms/papers/SchaeferTurek1996.pdf) |
| `basiliskLogConformation` | The log-conformation method for some viscoelastic constitutive models | [Fuente primaria](https://basilisk.fr/src/log-conform.h) |
| `rheotoolCode` | rheoTool: tools for simulation of rheological flows in OpenFOAM | [Fuente primaria](https://github.com/fppimenta/rheoTool) |

### Alcance de cada referencia

- **[R1] Dolan, E. D.; Moré, J. J. (2002).** *Benchmarking optimization software with performance profiles*. Mathematical Programming 91, 201–213. DOI: [10.1007/s101070100263](https://doi.org/10.1007/s101070100263). [Preprint de autores](https://arxiv.org/abs/cs/0102001). Sustenta perfiles de desempeño y tratamiento comparativo de conjuntos de problemas.
- **[R2] Williams, S.; Waterman, A.; Patterson, D. (2009).** *Roofline: an insightful visual performance model for multicore architectures*. Communications of the ACM 52(4), 65–76. DOI: [10.1145/1498765.1498785](https://doi.org/10.1145/1498765.1498785). [Repositorio institucional](https://escholarship.org/uc/item/78h8v7mr). Explica el vínculo entre capacidad de cómputo, ancho de banda e intensidad aritmética.
- **[R3] Halko, N.; Martinsson, P.-G.; Tropp, J. A. (2011).** *Finding Structure with Randomness: Probabilistic Algorithms for Constructing Approximate Matrix Decompositions*. SIAM Review 53(2), 217–288. DOI: [10.1137/090771806](https://doi.org/10.1137/090771806). Fundamenta alternativas de bajo rango y costo/pasadas sobre datos; no implica que cualquier SVD aleatoria tenga igual precisión.
- **[R4] Candès, E. J.; Li, X.; Ma, Y.; Wright, J. (2011).** *Robust principal component analysis?* Journal of the ACM 58(3), artículo 11. DOI: [10.1145/1970392.1970395](https://doi.org/10.1145/1970392.1970395). [Preprint](https://arxiv.org/abs/0912.3599). PCP recupera bajo rango más disperso bajo hipótesis de identificabilidad/incoherencia y soporte; no universalidad bajo ruido denso.
- **[R5] Zhou, Z.; Li, X.; Wright, J.; Candès, E.; Ma, Y. (2010).** *Stable Principal Component Pursuit*. ISIT 2010. DOI: [10.1109/ISIT.2010.5513535](https://doi.org/10.1109/ISIT.2010.5513535). [Preprint](https://arxiv.org/abs/1001.2363). Relaja PCP para perturbación densa acotada y errores dispersos.
- **[R6] Xu, H.; Caramanis, C.; Sanghavi, S. (2010).** *Robust PCA via Outlier Pursuit*. NeurIPS 23. [Artículo primario](https://papers.nips.cc/paper_files/paper/2010/hash/fe8c15fed5f808006ce95eddb7366e35-Abstract.html). [Preprint](https://arxiv.org/abs/1010.4237). Corrupción de columnas, distinta de dispersión entrada a entrada.
- **[R7] Gavish, M.; Donoho, D. L. (2014).** *The Optimal Hard Threshold for Singular Values is 4/√3*. IEEE Transactions on Information Theory 60(8), 5040–5053. DOI: [10.1109/TIT.2014.2323359](https://doi.org/10.1109/TIT.2014.2323359). [Preprint](https://arxiv.org/abs/1305.5870). Baseline principiado para matrices bajas en rango y ruido blanco; fórmula 4/√3 corresponde al caso cuadrado, no a toda matriz rectangular.
- **[R8] Schmid, P. J. (2010).** *Dynamic mode decomposition of numerical and experimental data*. JFM 656, 5–28. DOI: [10.1017/S0022112010001217](https://doi.org/10.1017/S0022112010001217). Base metodológica de DMD.
- **[R9] Dawson, S. T. M.; Hemati, M. S.; Williams, M. O.; Rowley, C. W. (2016).** *Characterizing and correcting for the effect of sensor noise in the dynamic mode decomposition*. Experiments in Fluids 57, 42. DOI: [10.1007/s00348-016-2127-7](https://doi.org/10.1007/s00348-016-2127-7). [Preprint](https://arxiv.org/abs/1507.02264). Analiza sesgo por ruido y correcciones directas, forward–backward y TLS.
- **[R10] Askham, T.; Kutz, J. N. (2018).** *Variable Projection Methods for an Optimized Dynamic Mode Decomposition*. SIAM J. Applied Dynamical Systems 17(1), 380–416. DOI: [10.1137/M1124176](https://doi.org/10.1137/M1124176). [Preprint](https://arxiv.org/abs/1704.02343). Ajuste no lineal global por proyección variable.
- **[R11] Sashidhar, D.; Kutz, J. N. (2022).** *Bagging, optimized dynamic mode decomposition for robust, stable forecasting with spatial and temporal uncertainty quantification*. Philosophical Transactions A 380, 20210199. DOI: [10.1098/rsta.2021.0199](https://doi.org/10.1098/rsta.2021.0199). [Registro](https://pubmed.ncbi.nlm.nih.gov/35719072/). Ensemble de optimized DMD; la incertidumbre debe calibrarse para el experimento específico.
- **[R12] Baddoo, P. J.; Herrmann, B.; McKeon, B. J.; Kutz, J. N.; Brunton, S. L. (2023).** *Physics-informed dynamic mode decomposition*. Proceedings A 479, 20220576. DOI: [10.1098/rspa.2022.0576](https://doi.org/10.1098/rspa.2022.0576). [Preprint](https://arxiv.org/abs/2112.04307). Impone estructuras del operador justificadas por física.
- **[R13] Le Clainche, S.; Vega, J. M. (2017).** *Higher Order Dynamic Mode Decomposition*. SIAM J. Applied Dynamical Systems 16(2), 882–925. DOI: [10.1137/15M1054924](https://doi.org/10.1137/15M1054924). Retardos para dinámicas periódicas/cuasi-periódicas y transitorios con complejidad temporal mayor que espacial.
- **[R14] Takeishi, N.; Kawahara, Y.; Yairi, T. (2017).** *Subspace dynamic mode decomposition for stochastic Koopman analysis*. Physical Review E 96, 033310. DOI: [10.1103/PhysRevE.96.033310](https://doi.org/10.1103/PhysRevE.96.033310). [Preprint](https://arxiv.org/abs/1705.04908). Estima espectros estocásticos bajo hipótesis explícitas de convergencia.
- **[R15] Askham, T.; Zheng, P.; Aravkin, A.; Kutz, J. N. (2022).** *Robust and Scalable Methods for the Dynamic Mode Decomposition*. SIAM J. Applied Dynamical Systems 21(1), 60–79. DOI: [10.1137/21M1417405](https://doi.org/10.1137/21M1417405). [Preprint](https://arxiv.org/abs/1712.01883). Pérdidas/reguladores robustos dentro de optimized DMD, alternativa científica a RPCA+DMD.
- **[R16] Papanastasiou, T. C. (1987).** *Flows of Materials with Yield*. Journal of Rheology 31(5), 385–404. DOI: [10.1122/1.549926](https://doi.org/10.1122/1.549926). Regularización de materiales con esfuerzo de cedencia; estudiar dependencia del parámetro de regularización.
- **[R17] Fattal, R.; Kupferman, R. (2004).** *Constitutive laws for the matrix-logarithm of the conformation tensor*. J. Non-Newtonian Fluid Mechanics 123, 281–285. DOI: [10.1016/j.jnnfm.2004.08.008](https://doi.org/10.1016/j.jnnfm.2004.08.008). Y (2005): *Time-dependent simulation of viscoelastic flows at high Weissenberg number using the log-conformation representation*. DOI: [10.1016/j.jnnfm.2004.12.003](https://doi.org/10.1016/j.jnnfm.2004.12.003). Estabilización constitutiva, sin eliminar necesidad de resolución y convergencia.
- **[R18] Hamid, F.; Sasmal, C.; Chhabra, R. P. (2022).** *Dynamic mode decomposition analysis and fluid-mechanical aspects of viscoelastic fluid flows past a cylinder in laminar vortex shedding regime*. Physics of Fluids 34, 103114. DOI: [10.1063/5.0122103](https://doi.org/10.1063/5.0122103). Oldroyd-B, Re=100, Wi=0–2, β=0.5–0.85; antecedente directo, no propuesta inédita.
- **[R19] Raffi, S.; Chauhan, A.; Hamid, F.; Sasmal, C. (2024).** *Elastoviscoplastic flows past a cylinder: Fluid-mechanical aspects and dynamic mode decomposition analysis*. Physics of Fluids 36, 083118. DOI: [10.1063/5.0224004](https://doi.org/10.1063/5.0224004). DMD en flujo EVP; enlace editorial con resumen y metadatos; consultar el texto completo antes de reproducir la formulación.
- **[R20] Corrochano, A.; Iqbal, K. T.; Parvar, S.; Le Clainche, S.; Tammisola, O. (2026).** *The coherent structures of EVP fluid flow past a circular cylinder*. Theoretical and Computational Fluid Dynamics 40, artículo 5. DOI: [10.1007/s00162-026-00775-3](https://doi.org/10.1007/s00162-026-00775-3). [Preprint 2025, texto primario](https://arxiv.org/html/2508.03362v1). Saramito–Herschel–Bulkley, Re=100, POD/HODMD; compara regímenes periódicos y complejos. El preprint reporta malla D/64, 512 núcleos y 192 h por caso: sirve para dimensionar la campaña, no para afirmar que se reprodujo.
- **[R21] Taylor, G. I.; Green, A. E. (1937).** *Mechanism of the production of small eddies from large ones*. Proceedings A 158, 499–521. DOI: [10.1098/rspa.1937.0036](https://doi.org/10.1098/rspa.1937.0036). [Texto primario](https://www.ams.jhu.edu/~eyink/Turbulence/classics/TaylorGreen37.pdf). Referencia histórica; el TGV analítico 2D y el desarrollo turbulento 3D son casos distintos.
- **[R22] Roache, P. J. (2002).** *Code Verification by the Method of Manufactured Solutions*. Journal of Fluids Engineering 124(1), 4–10. DOI: [10.1115/1.1436090](https://doi.org/10.1115/1.1436090). Verificación matemática, distinta de validación física experimental.
- **[R23] Chorin, A. J. (1968).** *Numerical solution of the Navier–Stokes equations*. Mathematics of Computation 22(104), 745–762. DOI: [10.1090/S0025-5718-1968-0242392-2](https://doi.org/10.1090/S0025-5718-1968-0242392-2). Método de presión/proyección. Orszag, S. A.; Patterson, G. S. (1972), *Numerical Simulation of Three-Dimensional Homogeneous Isotropic Turbulence*. DOI: [10.1103/PhysRevLett.28.76](https://doi.org/10.1103/PhysRevLett.28.76), fundamento de DNS espectral.
- **[R24] Towne, A.; Schmidt, O. T.; Colonius, T. (2018).** *Spectral proper orthogonal decomposition and its relationship to dynamic mode decomposition and resolvent analysis*. JFM 847, 821–867. DOI: [10.1017/jfm.2018.283](https://doi.org/10.1017/jfm.2018.283). [Preprint](https://arxiv.org/abs/1708.04393). Alternativa modal dependiente de frecuencia para procesos estacionarios.
- **[R25] Colbrook, M. J.; Ayton, L. J.; Szőke, M. (2023).** *Residual dynamic mode decomposition: robust and verified Koopmanism*. JFM 955, A21. DOI: [10.1017/jfm.2022.1052](https://doi.org/10.1017/jfm.2022.1052). [Texto primario](https://www.damtp.cam.ac.uk/user/mjc249/pdfs/residual-dynamic-mode-decomposition-robust-and-verified-koopmanism.pdf). Validación espectral y contaminación espectral; no equivale automáticamente a robustez de reconstrucción bajo cualquier ruido.
- **[R26] Schäfer, M.; Turek, S.; con Durst, F.; Krause, E.; Rannacher, R. (1996).** *Benchmark Computations of Laminar Flow Around a Cylinder*. Notes on Numerical Fluid Mechanics 52, 547–566. [Artículo primario del autor](https://www.mathematik.tu-dortmund.de/lsiii/cms/papers/SchaeferTurek1996.pdf). Especificación y observables del benchmark; conservar geometría y fronteras.
- **[R27] Saramito, P. (2009).** *A new elastoviscoplastic model based on the Herschel–Bulkley viscoplastic model*. J. Non-Newtonian Fluid Mechanics 158, 154–161. DOI: [10.1016/j.jnnfm.2008.12.001](https://doi.org/10.1016/j.jnnfm.2008.12.001). [Texto primario](https://web.mit.edu/nnf/education/Summer2009/Saramito2.pdf). Constitución EVP tridimensional con potencia n>0, base de la extensión física más avanzada.

- **[R28] Oldroyd, J. G. (1950).** *On the formulation of rheological equations of state*. Proceedings A 200, 523–541. DOI: [10.1098/rspa.1950.0035](https://doi.org/10.1098/rspa.1950.0035). Base de constitución viscoelástica objetiva y memoria de esfuerzos.
- **[R29] Herschel, W. H.; Bulkley, R. (1926).** *Konsistenzmessungen von Gummi-Benzollösungen*. Kolloid-Zeitschrift 39, 291–300. DOI: [10.1007/BF01432034](https://doi.org/10.1007/BF01432034). Modelo con esfuerzo de cedencia y ley de potencia.
- **[R30] Bird, R. B.; Dotson, P. J.; Johnson, N. L. (1980).** *Polymer solution rheology based on a finitely extensible bead–spring chain model*. JNNFM 7, 213–235. DOI: [10.1016/0377-0257(80)85007-5](https://doi.org/10.1016/0377-0257(80)85007-5). El resumen editorial identifica la aproximación FENE-P/Peterlin.
- **[R31] Saramito, P. (2007).** *A new constitutive equation for elastoviscoplastic fluid flows*. JNNFM 145, 1–14. DOI: [10.1016/j.jnnfm.2007.04.004](https://doi.org/10.1016/j.jnnfm.2007.04.004). [Texto primario](https://web.mit.edu/nnf/education/Summer2009/Saramito1.pdf). Modelo EVP; distinguirlo de la extensión Herschel–Bulkley de 2009.
- **[R32] Pimenta, F.; Alves, M. A. (2017).** *Stabilization of an open-source finite-volume solver for viscoelastic fluid flows*. JNNFM 239, 85–104. DOI: [10.1016/j.jnnfm.2016.12.002](https://doi.org/10.1016/j.jnnfm.2016.12.002). [Código oficial RheoTool](https://github.com/fppimenta/rheoTool). Candidato a integración externa; no ejecutado por citarlo.
- **[R33] Basilisk, documentación/código oficial.** *The log-conformation method for some viscoelastic constitutive models*. [Código](https://basilisk.fr/src/log-conform.h). Describe acoplamiento NS con Oldroyd-B/FENE-P y log-conformation. No tiene DOI; verificar versión y tests antes de integrar.
