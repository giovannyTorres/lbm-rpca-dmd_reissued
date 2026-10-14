# Segunda revisión científica independiente

La revisión inspeccionó los nuevos solvers periódicos 2D, la proyección física, el residuo de vorticidad y el pronóstico cronológico. Se efectuó sin modificar sus fuentes. Los cálculos independientes respaldan la implementación del esfuerzo variable y de la memoria polimérica. Los tres defectos reproducibles encontrados en la API de evaluación física fueron corregidos y sus contraejemplos se verificaron de nuevo antes de reiniciar íntegramente la campaña definitiva.

## Hallazgos de evaluación física corregidos

| Hallazgo | Evidencia independiente | Alcance y corrección verificada |
|---|---|---|
| Laplaciano de Nyquist incorrecto en `physics.py`. | Campo estático `u=cos(y)cos(4x), v=0`, malla `8²` en `[0,2π)²`, cinco tiempos, viscosidad uno. El residuo de vorticidad RMS calculado era `0.70710678`; el Laplaciano del interpolante trigonométrico exige `12.02081528`, una diferencia de factor 17. | Se separó el Laplaciano con números de onda completos del operador compatible de primera derivada, que anula Nyquist para campos reales. El contraejemplo ahora devuelve `12.020815280171307`, igual al valor analítico. Afectaba residuos de candidatos con contaminación en Nyquist; no modificaba la referencia limpia truncada ni la selección de pronóstico. |
| Máscara opcional tratada como obligatoria. | Un dataset periódico válido sin `mask`, admitido por forecasting y métricas, producía `selected_test_failed` y errores `KeyError: mask`. Añadir la máscara vacía recuperaba `ok`. | Postprocesamiento y residuo físico usan ahora una máscara vacía por defecto. Un pronóstico DMD con proyección y sin máscara explícita termina en `ok` sin fallos de evaluación. Los generadores de la campaña incluyen máscara explícita. |
| Un campo de fuerza estático podía confundirse con una serie temporal. | Con `T=Y=16`, `forcing_curl` de forma `(Y,X)` se cortaba en `_subset_dataset` como si su primera dimensión fuera tiempo. La evaluación futura fallaba por broadcasting; representarlo explícitamente como `(T,Y,X)` funcionaba. | Se distingue cada campo espacial estático del que tiene un eje temporal explícito. Con `T=Y=16`, fuerza, presión y densidad estáticas `(16,16)` conservan su forma al tomar cuatro tiempos de prueba. El pronóstico combinado sin máscara explícita termina en `ok`, sin fallos y con error futuro de velocidad `1.496e-15`. |

El primero es un defecto del diagnóstico, no una prueba de error en los solvers. Una proyección puede ser correcta bajo su operador discreto y, a la vez, un residuo difusivo utilizar un Laplaciano incorrecto. Por ello se revisaron ambos cálculos por separado.

## Memoria polimérica y realimentación de momento

[viscoelastic_dns.py](../../python/src/fluid_denoise/research/viscoelastic_dns.py) implementa `G C+C Gᵀ`, con `Gij=∂j ui`, relajación y esfuerzo `νp(fC-I)/λ`. El tensor por defecto tiene dimensión tres, aunque la velocidad sea planar; FENE-P incluye `Czz` y emplea `f=(L²-3)/(L²-tr C)`, que hace `C=I` un equilibrio. Las fórmulas de energía libre, trabajo del esfuerzo, relajación y difusión de conformación son coherentes con esa normalización. La admisibilidad se comprueba en etapas y estados aceptados; las etapas inválidas se rechazan, sin reparar autovalores o recortar la traza.

Se comprobó la fuente inicial mediante fórmulas independientes para una velocidad Taylor–Green de amplitud `0.05` y una conformación uniforme

```text
C = [[1.8, 0.1, 0],
     [0.1, 0.8, 0.02],
     [0, 0.02, 0.7]].
```

Con `λ=0.4`, `νs=0.02`, `νp=0.03`, `L²=8`, malla `24²` y dos pasos de `1e-5`, la derivada inicial estimada con `(-3C0+4C1-C2)/(2Δt)` coincide con `GC+CGᵀ-(fC-I)/λ`: errores relativos `2.134e-10` en Oldroyd-B y `4.187e-10` en FENE-P. La componente `Czz` tiene tasa inicial `0.75` y `0.6382978723`, respectivamente, y ambas se recuperan.

La realimentación de esfuerzos se verificó por separado con `Cxy=Cyx=0.1cos(x)`, restantes componentes iguales a identidad. En Oldroyd-B, su fuerza inicial solenoidal es `∂t v|polymer=-(νp/λ)0.1sin(x)`. Sumada a la difusión solvente exacta de Taylor–Green, reproduce la derivada inicial de velocidad con error relativo `3.015e-10`. La prueba distingue una integración realmente acoplada de una evolución newtoniana cuyo tensor se calculara después.

Los productos cuadráticos retenidos usan el corte de dos tercios; el factor racional de Peterlin no queda exactamente libre de aliasing con ese corte. La implementación conserva el defecto constitutivo espacial y el residuo de energía sin forzarlos a cero. La difusión artificial de conformación es explícita y tiene valor cero por defecto. Estos hechos y la positividad en puntos de colocación delimitan la evidencia: no certifican alto Weissenberg, positividad entre nodos o turbulencia elástica.

## Transporte de esfuerzo viscoplástico

[viscoplastic_dns.py](../../python/src/fluid_denoise/research/viscoplastic_dns.py) integra momento con `div(2ηapp D)/ρ`; no sustituye ese término por `ηapp Δu` ni reescala una trayectoria newtoniana. La norma de tasa de deformación `sqrt(2D:D)`, el signo de presión, la conversión de viscosidad dinámica mediante densidad y el uso de los tres componentes simétricos del esfuerzo son correctos. La viscosidad no tiene un cap oculto. CFL y difusión se verifican en las etapas; se rechazan estados no finitos o aumentos de energía en el problema no forzado.

Para una cizalla periódica `u=0.2sin(y),v=0`, la fuerza inicial puede calcularse sin reutilizar la rutina constitutiva. Si `γ=0.2cos(y)` y `h=hypot(γ,ε)`,

```text
∂t u = [ηs + K h^(n-3)(ε²+nγ²) + τy m exp(-m|γ|)] ∂yγ / ρ.
```

Con `n=1.4`, `K=0.02`, `τy=0.01`, `m=5`, `ε=0.2`, `ηs=0.003`, `ρ=2`, malla `32²` y dos pasos de `1e-5`, se comparó la derivada inicial contra la fuente analítica evaluada en `1024` puntos y restringida a los modos activos. Los errores relativos fueron `1.787e-4`, `1.782e-5` y `9.555e-6` al usar factores de cuadratura `1.5`, `3` y `6`. El balance energético discreto era del orden de `1e-17` en los tres ensayos. Esto confirma el transporte de esfuerzo variable y muestra por qué cerrar un balance discreto no elimina el error de cuadratura constitutiva.

La regularización de Papanastasiou y el suavizado de baja cizalla son aproximaciones declaradas. Las regiones clasificadas por un umbral de cizalla no son superficies de cedencia exactas. El refinamiento de `m` o `ε` cambia la aproximación constitutiva; no equivale a refinamiento de malla. Los casos son inelásticos, periódicos y sin cilindro.

## Proyección y ausencia de fuga temporal

La revisión separada de [physics.py](../../python/src/fluid_denoise/research/physics.py) comprobó idempotencia, contracción de energía, conservación de curl sin filtrado y divergencia inferior a `1e-12` en mallas pares, impares y rectangulares. El filtrado adicional de dos tercios se declara como prior de ancho de banda. La proyección utiliza geometría y la velocidad candidata, no velocidad limpia, y rechaza la aplicación a paredes o sólidos que necesitan otro problema de presión.

En [forecasting.py](../../python/src/fluid_denoise/research/forecasting.py), el entrenamiento, RPCA y reducción de orden acceden únicamente al tramo de entrenamiento. La selección utiliza el tramo de validación ruidoso. El modelo seleccionado no se reajusta con observaciones de prueba y las predicciones se generan sin leer esas observaciones. Persistencia y POD parten del último estado de entrenamiento; las amplitudes DMD se ajustan sólo en entrenamiento. La proyección precede el puntaje correspondiente y usa la misma geometría permitida. La referencia limpia futura se reserva para evaluar después de seleccionar.

La revisión auxiliar de la fuente inicial ejecutó 20 pruebas y 10 subcasos existentes de forecasting y physics, además de los contraejemplos descritos. Pasar esas pruebas no detectaba los tres defectos de API. Tras las correcciones se repitieron específicamente los tres contraejemplos, incluyendo presión y densidad estáticas junto a la fuerza; todos recuperan el comportamiento esperado. La campaña se reinicia con las fuentes corregidas y su nueva procedencia, sin mezclar los residuos anteriores.

## Interpretación de las campañas

Los nuevos solvers representan simulaciones acopladas o de esfuerzo variable con alcance explícito; no establecen una certificación universal DNS, calles de Kármán no newtonianas o caos. La reconstrucción y el pronóstico actuales estiman velocidad; almacenar esfuerzos y conformación de la referencia no demuestra reconstruirlos. Un residuo de vorticidad newtoniano no debe aplicarse a estas referencias omitiendo el curl del esfuerzo no newtoniano; la API los declara no aplicables.

Tres semillas de ruido y tres repeticiones de costo tienen funciones diferentes. Las repeticiones sobre una misma entrada cuantifican variación computacional; no aumentan el número de realizaciones físicas independientes. El número de condiciones o candidatos no reemplaza validación fuera de muestra, tolerancias por observable y sensibilidad constitutiva. Esta revisión verifica formulaciones y casos adversarios concretos; los resultados finales de comparación se redactan a partir de las campañas completas, sus fallos y su procedencia.
