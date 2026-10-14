# Comparación de estimadores y fidelidad de las referencias

El objetivo central es comparar la recuperación de campos de velocidad contaminados bajo condiciones equivalentes. La dinámica de cada simulación proporciona la referencia numérica; su calidad debe verificarse antes de atribuir el error al estimador. «Modelo constitutivo» designa el fluido (Newtoniano, Oldroyd-B, FENE-P, HB o Bingham). «Estimador» designa el procedimiento de recuperación (POD, PCP/RPCA, DMD y variantes). Una diferencia entre dos constitutivas no es un error de reconstrucción.

## Preguntas y evidencias que las responden

| Pregunta | Comparación necesaria | Alcance de la respuesta |
| --- | --- | --- |
| ¿Qué estimador recupera mejor el campo? | Misma referencia, observación, semilla, intensidad, rango declarado y procesamiento posterior | Errores de velocidad y vorticidad, convergencia y criterios operativos por bloque |
| ¿La ventaja procede del modelo o del filtrado? | Estimadores crudos, después H y después H + corte físico; identidad con exactamente cada mismo tratamiento | Ganancia incremental del modelo frente al control filtrado |
| ¿El resultado depende del ruido y del fluido? | Comparaciones pareadas en cada combinación; peso igual por caso y tipo de ruido | Interacción descriptiva en siete trayectorias deterministas; no generalización poblacional |
| ¿La referencia representa bien las magnitudes comparadas? | Parámetros, tiempos y fronteras coincidentes, refinamiento espacial/temporal, referencias analíticas y residuos | Error por observable e intervalo; una velocidad convergida no garantiza esfuerzo ni viscosidad convergidos |
| ¿Cuál es el costo de alcanzar la calidad exigida? | Recursos de un ajuste aislado y calidad de ese mismo ajuste; frontera entre alternativas admisibles | CPU, memoria, trabajo estimado y costo del tratamiento; sin puntuación arbitraria ni inferencia de eficiencia energética |

## Bancos separados y unidad de comparación

El banco previo contiene 216 bloques caso/ruido/semilla y 13 configuraciones por bloque: 2808 condiciones, con tres mediciones de recursos por condición. Las tres mediciones no convierten una realización en tres observaciones independientes. Sus referencias pequeñas se vuelven a evaluar contra simulaciones refinadas **con sus parámetros originales**. Las campañas visuales ampliadas tienen parámetros distintos y no sustituyen esta comprobación.

El nuevo banco utiliza siete campos calculados por separado en 128²: dos Navier–Stokes, Oldroyd-B, FENE-P y tres HB/Bingham. Preserva cada horizonte original de esas historias: T=4 con 41 estados, T=2.4 con 25, y T=2 con 21, respectivamente. Las diferencias de parámetros y horizonte impiden atribuir todos los cambios exclusivamente a la constitutiva.

Se fijan cinco contaminaciones (gaussiano por celda, impulsivo de soporte 1%, gaussiano correlacionado espacialmente, correlacionado temporalmente y columnas contaminadas), RMS relativo 20% y tres semillas nuevas 101, 307 y 911. Se comparan diez estimadores: identidad, POD, DMD clásico, TLS-DMD, PCP, RPCA estable penalizada, Outlier Pursuit, PCP→TLS, RPCA estable→TLS y DMD optimizada. El rango solicitado es 8 para los métodos de rango prescrito; PCP y Outlier Pursuit seleccionan rango mediante su formulación, no mediante una truncación impuesta.

El diseño produce 1050 ajustes crudos y 3150 evaluaciones: cada ajuste se evalúa sin posprocesamiento, con Helmholtz H, y con H más un corte isotrópico k≤10 en unidades de longitud inversa. Las tres evaluaciones comparten el ajuste y no son tres replicaciones independientes. H y el corte utilizan sólo la geometría declarada. La identidad recibe exactamente los mismos tratamientos. El costo del ajuste y el de cada tratamiento se identifican por separado; no se atribuye todo el ahorro al estimador modal.

El corte k≤10 es un prior exploratorio de suavidad heredado del seguimiento anterior. No se selecciona minimizando el error frente al limpio. La evaluación mantiene todas las escalas de la referencia: filtrar el candidato no implica filtrar su objetivo. Los resultados pueden cambiar con otra banda o con estructuras más finas.

## Ruido y acceso a la referencia

La generación de corrupción usa la referencia para fijar la intensidad del experimento; el estimador recibe únicamente la observación, tiempos y geometría. La referencia limpia se reserva a la evaluación. El uso de la intensidad conocida en la penalización estable es una condición declarada del experimento, no una estimación ciega de ruido en una medición real.

El ancho espacial σ=0.10 es la desviación del **núcleo de filtrado** en coordenadas físicas. No se identifica con la desviación de la autocorrelación de ese campo. La correlación temporal se expresa mediante una constante física τ=0.30 y la separación real entre estados. La corrupción de columnas selecciona sin reemplazo floor(0.10T) estados; su fracción realizada se declara porque no existen exactamente 10% de columnas para todos los T. Outlier Pursuit usa la cota nominal gamma=0.10.

Todas las comparaciones dentro de un bloque comparten exactamente el mismo array observado y su hash. Las semillas representan realizaciones de contaminación del mismo flujo. No son nuevas condiciones iniciales físicas ni simulaciones independientes de un régimen.

## Calidad operativa, fidelidad física y error de referencia

Se conserva la regla nominal: error relativo L2 de velocidad ≤15%, vorticidad ≤25%, exceso de divergencia normalizado ≤10% y error de energía cinética ≤15%, junto con convergencia. Cumplir esta regla significa superar cuatro criterios operativos de reconstrucción. No demuestra por sí solo que el candidato satisfaga la ecuación de momento, la constitutiva ni una validación experimental. Se muestra sensibilidad al umbral de vorticidad para evitar presentar 25% como límite universal.

La densidad uno es una hipótesis explícita de estos solvers. Masa constante y cancelación de caras periódicas no aportan por sí solas capacidad discriminante entre estimadores. Se distinguen esas identidades de divergencia, energía, enstrofía y residuos que sí cambian con el candidato.

Los errores contra una solución fina requieren mismo parámetro, condición inicial, dominio y tiempo. La referencia fina permanece numérica. Las cotas triangulares permiten examinar si el error de discretización es relevante frente al error de reconstrucción, manteniendo una normalización común. La prolongación de Fourier se usa para comparar campos calculados, y no constituye una integración nueva.

El residuo calculado entre snapshots incluye error de diferenciación temporal y cuadratura de salida; debe examinarse con varias separaciones o fórmulas antes de llamarlo error del integrador. Los esfuerzos viscoelásticos y la viscosidad viscoplástica no se reconstruyen a partir de velocidad mediante una fórmula ficticia: sólo se evalúan cuando existen estados constitutivos correspondientes.

## Sensibilidad de rango y amplitud

Un banco separado fija dos campos (Navier–Stokes multiescala y Oldroyd-B), dos ruidos (gaussiano e impulsivo), intensidades 5%, 20% y 40%, rangos 2/4/8/12/16, POD y TLS-DMD y las tres semillas nuevas: 360 ajustes y 1080 evaluaciones con los mismos tres tratamientos. Los extremos entre semillas son rangos descriptivos, no intervalos de confianza poblacionales.

El mejor error POD de rango r calculado con el limpio se presenta exclusivamente como cota diagnóstica de compresión. No selecciona el rango del modelo ni entra al ajuste. Las condiciones que coinciden con el banco principal no se suman como nueva evidencia independiente.

## Controles sin contaminación

Se añaden 84 ajustes y 252 evaluaciones deterministas: los diez estimadores sobre los siete campos limpios, con exactamente las penalizaciones del contrato de ruido del 20% (70), más los dos procedimientos estables con intensidad conocida cero sobre cada campo (14). El primer control identifica sesgo de truncación, separación o regularización aun sin corrupción; el segundo distingue el efecto de aplicar una penalización densa correspondiente a un experimento ruidoso. Los controles se incorporaron tras observar resultados iniciales, por lo que son una exploración del mecanismo, no un contraste preregistrado independiente. Las salidas siguen sin seleccionar rango o penalización por mínimo error limpio.

La comparación nueva y la sensibilidad comparten 24 contratos rango 8/intensidad 20%; esos ajustes repetidos no se suman como evidencia independiente. Los 1494 ajustes ejecutados en los tres bancos corresponden a 1470 contratos distintos, además de los pilotos separados y las integraciones de referencia.

## Registro y reproducibilidad

Las configuraciones se fijan antes de la campaña; un piloto separado caracteriza costo y viabilidad, sin seleccionar modelos por error limpio. Se preservan fallos, tiempos límite, no convergencias y diagnósticos de ambas etapas de las cadenas. No se elimina una configuración fallida para mejorar su posición.

Los registros incluyen fuentes, parámetros, versiones, hashes de entradas, matriz observada y finalización. Cada ajuste usa un proceso nuevo y un hilo BLAS. El equipo tiene cuatro CPU efectivas y se comparte con otras tareas: las mediciones únicas del banco nuevo son descriptivas y no sustentan diferencias pequeñas de velocidad. Las mediciones previas de tres repeticiones mantienen su propia procedencia.

La RSS de cada tratamiento posterior conserva el máximo de vida del hijo, y puede incluir picos de operaciones anteriores; no equivale a memoria incremental de la proyección. El diagnóstico FFT de Helmholtz registra dos transformadas por lotes; la banda añade otras dos. La tabla derivada identifica cuatro para H más banda, conservando intacto el contador parcial original.

Las fuentes científicas originales quedan congeladas. El documento se compila con los .tex y las figuras incluidos; la regeneración de simulaciones o análisis de campos completos requiere los NPZ originales generados localmente. Las selecciones de tres o cinco estados descargables no reemplazan las historias usadas en las métricas.
