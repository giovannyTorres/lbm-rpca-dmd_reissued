# Pronóstico cronológico fuera de muestra

Esta ampliación distingue reconstrucción dentro de la ventana de observación y
pronóstico de campos futuros. `algorithms.reconstruct` produce una aproximación
de los tiempos recibidos; `forecasting.fit_forecast` devuelve las dinámicas
ajustadas y un método `predict(target_times)` que las evalúa en tiempos nuevos.
Los errores futuros se calculan contra la referencia limpia en esos mismos
tiempos. No se extiende una reconstrucción usando coeficientes de toda la serie.

## Contrato y modelos

```python
from fluid_denoise.research.forecasting import (
    chronological_split, fit_forecast, evaluate_forecasting,
    make_forecasting_noise,
)

blocks = chronological_split(times, train_fraction=.6, validation_fraction=.2)
model = fit_forecast(observed_matrix[:, blocks["train"]],
                     times[blocks["train"]], "dmd", rank=4)
future_matrix = model.predict(times[blocks["test"]])
```

La matriz tiene forma `(features, time)`. El modelo conserva las coordenadas
espaciales completas: ningún campo de validación o prueba se proyecta antes de
pronosticarlo. La máscara geométrica, conocida antes del ajuste, excluye sólidos
tanto del ajuste como de la comparación. Cada modelo mantiene un SHA-256 de sus
observaciones y tiempos de entrenamiento.

| Método | Dinámica ajustada con entrenamiento | Tiempos de predicción |
| --- | --- | --- |
| `persistence` | Última observación de entrenamiento repetida | Cualquier vector creciente finito |
| `pod_ar` | Media de entrenamiento, POD, AR(1) reducido afín con ridge | Pasos enteros no negativos desde el último tiempo de entrenamiento |
| `dmd` | Operador reducido exact DMD, modos espaciales exactos y amplitudes globales | Exponenciales continuas a partir de autovalores discretos |
| `tls_dmd` | Proyección TLS de pares consecutivos y modos exactos | Igual que DMD |
| `optimized_dmd` | Proyección variable de exponenciales en la base POD de entrenamiento | Tiempos regulares o irregulares |

DMD discreto y POD-AR requieren intervalos uniformes en entrenamiento. Los
autovalores discretos se convierten en tasas mediante el logaritmo complejo
principal dividido por `dt`; autovalores negativos conservan su alternancia en
los tiempos discretos. Un autovalor exactamente cero no define una tasa continua
finita y produce un fallo explícito. Una serie cero se pronostica como cero.

`pod_ar` centra exclusivamente con la media de entrenamiento y aprende también
un término afín. `ridge` penaliza el operador, no el término afín; el valor por
defecto es `1e-6` multiplicado por `||X_reducido||²/r`. `ridge=0` permite mínimos
cuadrados sin penalización. No se ajusta esta penalización con campos limpios.

Las amplitudes DMD se ajustan globalmente sobre entrenamiento; `amplitude_fit="first"`
usa sólo su primer estado. El modelo expone `modes`, `rates` y `amplitudes`;
POD-AR expone `basis`, `mean`, `operator` y `last_state`.

Optimized DMD usa `scipy.optimize.least_squares`, diferencias finitas y arranques
predeclarados. Las cotas de tasas/frecuencias limitan el problema durante el
ajuste y se reportan junto con la convergencia, costo y cotas activas. Pueden
existir mínimos locales y tasas cercanas mal identificadas. No es BOP-DMD ni
produce intervalos bootstrap. Ningún método recorta autovalores o exponentes
durante la extrapolación. El crecimiento estimado se conserva, incluso cuando
empeora el error. Un desbordamiento o un resultado no finito produce
`ForecastFailure`; nunca se sustituye por ceros o un error ficticio pequeño.

## Preprocesamiento y selección

`preprocessing="rpca"` (también `"pcp"`) o `"stable_rpca"` aplica la descomposición
existente exclusivamente al bloque de entrenamiento. `preprocessing_parameters`
controla sus parámetros. Se conservan sus diagnósticos de convergencia. Su rango
convexo se determina por regularización; el rango solicitado se aplica al modelo
dinámico posterior. `evaluate_forecasting` reutiliza la misma descomposición de
entrenamiento para candidatos con el mismo preprocesamiento y parámetros.

El bloque inicial contiene `floor(0.6*T)` instantáneas; el siguiente,
`floor(0.2*T)`; el resto forma prueba. Para 51 tiempos son 30/10/11 y para 61 son
36/12/13. No hay barajado ni ventanas que solapen los bloques. Los candidatos y
sus rangos 2, 4 y 8 están declarados en
[`forecasting_campaign.json`](../../configs/research/forecasting_campaign.json).
La persistencia sin preprocesamiento se incluye siempre como referencia y
candidato. Empates exactos se resuelven por el orden declarado.

Los candidatos también pueden declarar `postprocess="helmholtz"` o
`"helmholtz_dealias"`. Se transforma cada **predicción** de validación antes del
MSE observado y cada predicción de prueba antes de evaluar. La proyección usa
sólo geometría periódica y rechaza sólidos o fronteras incompatibles; no consulta
la velocidad limpia. La primera impone divergencia discreta cero preservando
curl; la segunda añade la hipótesis explícita de banda estricta de dos tercios
de Fourier. El ajuste dinámico se comparte entre variantes del mismo candidato.
Se incluye persistencia con la misma restricción de banda para poder separar
beneficios de filtrado y dinámica. `model.metadata` y los diagnósticos registran
estas restricciones; imponer incompresibilidad no certifica Navier–Stokes.

El perfil separado `postprocessing_profile` reporta segundos de la transformación
de validación/prueba y su conteo determinista de FFT. Estos segundos no incluyen
ajuste dinámico ni evaluación física y no participan en la selección.

Se minimiza el MSE entre predicción y observación **ruidosa de validación**.
No se refita después de seleccionar: el modelo elegido sigue usando sólo el
primer bloque. Por defecto, candidatos no convergentes o con fallos no son
elegibles; sus diagnósticos permanecen en los resultados. Si el ganador falla
en prueba, se declara `selected_test_failed`; no se sustituye usando errores
limpios de prueba.

Para ruido independiente de validación, de media cero e independiente del
entrenamiento, el MSE observado coincide en esperanza con el riesgo limpio más
una varianza común. Esta justificación no convierte cada realización finita en
un oráculo. Con ruido temporalmente correlacionado, el ruido de entrenamiento y
validación puede estar correlacionado; con columnas anómalas o ruido impulsivo,
unas pocas observaciones pueden dominar el criterio. Por ello se reportan los
errores verdaderos de prueba y la referencia de persistencia, aunque la elección
por validación resulte peor. La intensidad y familia de ruido son información
predeclarada de inyección, no parámetros estimados desde el futuro limpio.

Stable RPCA puede usar esa intensidad conocida para su regla de penalización
basada en RMS **observado de entrenamiento**. La aproximación de independencia
para estimar sigma es limitada para las familias mixtas, correlacionadas y
anómalas. No se suministra el residual limpio-ruidoso ni una sigma obtenida de
campos futuros al estimador.

## Contaminación sin fuga temporal

La campaña de reconstrucción anterior normaliza ruido con el RMS de toda su
ventana. Aplicar esa función a un experimento futuro permitiría que la amplitud
limpia de prueba cambiara la escala de ruido de entrenamiento. La campaña de
pronóstico utiliza `make_forecasting_noise`:

1. Calcula la escala de señal y ruido realizado sólo en entrenamiento.
2. Genera innovaciones y soporte sin consultar valores limpios futuros.
3. Evoluciona el ruido AR temporal hacia adelante, sin filtrado bidireccional.
4. Selecciona columnas anómalas con Bernoulli independiente; si no aparece ninguna
   en entrenamiento, fuerza una columna de ese bloque solamente.
5. Mantiene la misma escala en validación y prueba, sin renormalizarlas.

Las familias quedan igualadas en RMS **de entrenamiento**, no en RMS realizado
de cada bloque futuro. Con columnas anómalas puede haber pocas o ninguna columna
de validación/prueba contaminada; es una realización legítima y queda registrada.
El ruido añadido al futuro se usa para crear observaciones de validación. Las
observaciones ruidosas de prueba no se leen al ajustar, seleccionar ni predecir.

## Evaluación y evidencia reproducible

`evaluate_forecasting(dataset, noisy, config)` devuelve un objeto JSON con:

- selección y MSE observado de cada candidato, convergencia y fallos;
- índices y tiempos explícitos de entrenamiento, validación y prueba;
- métricas físicas escalares de cada candidato en prueba, después de congelar
  la selección, además de ganador y persistencia;
- errores relativos L2 de velocidad y vorticidad, residuos de divergencia,
  energía cinética y enstrofía mediante `metrics.physical_metrics` existente;
- residual de la ecuación de vorticidad para casos Newtonianos periódicos con
  viscosidad conocida; en casos viscoelásticos se declara no aplicable porque
  velocidad sola no proporciona el curl de tensión polimérica;
- mejora frente a persistencia (`error_persistencia - error_ganador`), donde un
  valor negativo representa empeoramiento;
- SHA-256 de pronósticos, observaciones de entrenamiento/validación,
  configuración y entradas de selección.

`return_forecasts=True` agrega `test_predictions={"selected": array,
"persistence": array}` con forma `(T_test,2,Y,X)` para guardar NPZ o visualizar;
elimina esta clave antes de serializar el resultado a JSON. Si un pronóstico de
prueba falla, su valor es `None`. `test_times` sigue siendo una lista JSON.

La densidad limpia se conserva para métricas físicas de reconstrucciones sólo
de velocidad. Por eso masa y continuidad son condicionales a esa densidad;
este experimento no pronostica presión, tensión ni conformación polimérica.
Un error de velocidad bajo no certifica vorticidad, fuerzas o constitutivas.

La campaña contiene un flujo Newtoniano de vórtices interactuantes y dos flujos
periódicos acoplados Oldroyd-B/FENE-P, con 61/51/51 tiempos, cuatro familias de
ruido y semillas 7, 23 y 61: **36 condiciones**, cada una con 21 candidatos
dinámicos originales, seis variantes físicas, persistencia filtrada y
persistencia original: **29 opciones**. Es una comparación de pronóstico sobre estos flujos
de referencia; no establece resolución DNS por sí sola. Los casos viscoelásticos
usan deformación moderada y `L²=50`, de modo que Oldroyd-B y FENE-P pueden dar
resultados muy parecidos. Esa similitud no identifica los modelos materiales.

La incertidumbre se limita a media, desviación estándar y rango entre las tres
semillas dentro de cada combinación flujo/ruido. Es variación de contaminación
condicional a un flujo determinista, sin intervalo de confianza poblacional ni
afirmación de generalización a otras geometrías o regímenes.

Auditorías ejecutables:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m unittest discover -s tests -p test_research_forecasting.py -v
```

Las pruebas comparan contra dinámicas exponenciales independientes en tiempos
futuros, incluyendo tiempos irregulares y autovalores negativos; verifican
crecimiento sin recorte y rechazo de desbordamiento. También mutan todos los
campos limpios futuros y sustituyen observaciones ruidosas de prueba por NaN:
la selección, sus huellas y las huellas de pronóstico deben permanecer iguales,
mientras cambian las métricas de evaluación. Otras pruebas muestran que modificar
validación ruidosa sí cambia la selección, aunque exista un candidato mejor
según la referencia limpia, y comprueban que escala, soporte y ruido temporal
no cambian al modificar valores limpios futuros.

Un caso controlado adicional deja visible el sesgo de DMD con ruido: el
autovalor de decaimiento se subestima y el error futuro queda grande. TLS reduce
ese sesgo en el ejemplo, pero mantiene error apreciable. No se usa un buen
resultado de reconstrucción dentro de muestra como sustituto de evidencia de
pronóstico.
