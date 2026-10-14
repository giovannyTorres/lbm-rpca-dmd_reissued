# Revisión independiente de comparación de modelos y fidelidad

Esta revisión separa tres preguntas: si una simulación es una referencia
numérica adecuada para el observable considerado, si un estimador reconstruye
esa referencia bajo una contaminación concreta y si su costo es aceptable al
exigir esa calidad. Un error pequeño respecto a datos sintéticos no resuelve
por sí solo las tres preguntas.

El verificador independiente es
[verify_model_comparison.py](../../python/scripts/verify_model_comparison.py).
No importa las funciones del analista, del runner ni del estimador para
recalcular contadores y admisibilidad. La evidencia numérica está en
[model_comparison_review.json](../../results/research/integration_validation/model_comparison_review.json).
Las campañas y las fuentes científicas anteriores se conservan.

## Diseño de una comparación interpretable

| Pregunta | Control exigido | Consecuencia al interpretar |
| --- | --- | --- |
| ¿Qué añade el estimador al dato observado? | Misma realización de ruido, flujo, tiempos y malla; identidad como baseline. | Diferencias entre métodos dentro del mismo bloque. No confundir una diferencia entre fluidos con error de reconstrucción. |
| ¿Qué añade una proyección o un filtro? | Identidad y estimador reciben exactamente el mismo postprocesamiento. | Una ganancia de Helmholtz o del corte espectral no se atribuye automáticamente a DMD o RPCA. |
| ¿Qué añade la familia de modelos? | Rango y regularización explícitos, información accesible documentada. | POD/DMD tienen rango solicitado; PCP/SPCP/Outlier Pursuit seleccionan rango mediante penalización. No afirmar presupuestos de rango idénticos. |
| ¿Es admisible una reconstrucción? | Umbrales simultáneos de velocidad, vorticidad, divergencia y energía, más convergencia. | La menor velocidad L2 o el menor costo no bastan para declarar ganador. |
| ¿Qué tan confiable es la referencia? | Refinamiento con las mismas condiciones físicas, tiempos de salida y observables. | Una referencia fina de otro régimen no certifica el benchmark original. |
| ¿Qué representa la variabilidad? | Tres semillas de ruido distintas; repeticiones de recursos identificadas por separado. | Mínimo, mediana y máximo son descriptivos. No son intervalos de confianza poblacionales. |

En la comparación nueva, raw, Helmholtz y Helmholtz con banda física proceden
de **la misma estimación**, sin volver a ajustar el modelo. Los tres resultados
comparten realización de ruido y coeficientes estimados; son una ablación de
postprocesamiento, no tres realizaciones estadísticas independientes. El corte
isótropo de número de onda 10 es un prior de suavidad de señal explícito.
Que coincida con el inverso de una escala del ruido no demuestra que la señal
limpia pertenezca a esa banda.

La intensidad de ruido sintético conocida puede entrar en la regla de
penalización de SPCP. Esto es información auxiliar del experimento, no una
estimación automática demostrada para datos reales. Los estimadores no reciben
la velocidad de referencia, esfuerzos ni conformación. El evaluador sí los
necesita después de producir el candidato.

## Reconstrucción y fidelidad de simulación

La comparación de una simulación gruesa prolongada con una referencia fina debe
retener todo el campo fino. Filtrar ambos a los modos comunes elimina
precisamente las escalas ausentes que se pretende detectar. Puede informarse
un error adicional de modos comunes, identificado como tal.

Para velocidad y vorticidad se pueden utilizar cotas de norma. Si `P` prolonga
al espacio fino, `U_h` es la referencia usada por el estimador y `U_f` la
referencia numérica más fina, se definen, con **el mismo denominador**,

```text
a = ||P(U_estimado - U_h)|| / ||U_f||
b = ||P(U_h) - U_f|| / ||U_f||
|a-b| <= ||P(U_estimado)-U_f||/||U_f|| <= a+b.
```

No sumar sin ajuste un porcentaje normalizado por `||U_h||` con otro
normalizado por `||U_f||`. Para usar el porcentaje grueso hace falta la razón
de normas y, cuando corresponda, la norma del operador de transferencia.
Energía, viscosidad aparente y otras transformaciones no lineales requieren
evaluación propia; la cota anterior no se transfiere a sus errores relativos
sin hipótesis adicionales. `U_f` sigue siendo una referencia numérica y no la
solución física exacta.

La convergencia de velocidad no basta para garantizar la de vorticidad,
esfuerzos, conformación o viscosidad aparente. Una memoria polimérica no se
recupera de manera instantánea a partir de una reconstrucción únicamente de
velocidad. En los modelos generalizados newtonianos regularizados, en cambio,
se puede derivar el esfuerzo desde el gradiente candidato y la ley constitutiva
conocida. Este diagnóstico debe identificar el cierre aplicado y emplear la
norma tensorial correcta: si se guardan `(xx, xy, yy)`, la norma de Frobenius
usa `xx² + 2 xy² + yy²`.

La disipación derivada de una reconstrucción viscoplástica se compara con
`sum(eta_ref_guardada * gamma_ref_guardada²) * dx * dy`, usando el mismo
operador y la misma malla de observación. El balance energético del solver
integra una disipación calculada en su cuadratura padded y acumulada en
etapas RK4. Son diagnósticos distintos: el error del cierre constitutivo
guardado no debe etiquetarse como defecto de ese balance de energía.

Una masa total constante obtenida conservando la densidad limpia no prueba
conservación de una reconstrucción de velocidad. Divergencia y condiciones de
frontera aportan evidencia relevante. La proyección periódica de Helmholtz
impone su restricción discreta, pero no demuestra resolver la ecuación de
cantidad de movimiento. El residuo newtoniano de vorticidad no debe aplicarse
a un material no newtoniano omitiendo el curl del esfuerzo.

Se volvieron a calcular seis trayectorias reconstruidas del banco visual en
la malla 512², con sus 41 tiempos: doce filas de velocidad/vorticidad. La
transferencia se implementó con padding real de Fourier y división explícita
del modo Nyquist, sin usar el helper del analista. Coinciden el error frente
a la referencia fina, la razón de normas y ambas cotas triangulares. En estos
vórtices, el cambio de referencia explica una fracción mucho menor del error
que la reconstrucción; esto no se extrapola a todos los fluidos.

También se recomputó la ecuación de vorticidad de la trayectoria NS 512²
de vórtices, en los mismos siete centros temporales. Con imágenes separadas
0.1, el residuo RMS dividido por la dinámica RMS es 0.0389221 para diferencias
centrales de orden dos y 0.00697346 para orden cuatro. La reducción depende
del diagnóstico aplicado a las imágenes; no significa que se haya cambiado
el paso de integración RK4 ni que ese valor sea su error interno.

## Auditoría de la campaña anterior

El ledger de `extended_physics_2026-10-14` contiene **2.808 condiciones** y
**8.424 ejecuciones de recursos**: ocho casos, nueve contaminaciones, tres
semillas, trece candidatos y tres ejecuciones sobre cada entrada. La cobertura
es completa y no contiene condiciones duplicadas. Las ejecuciones de recursos
no incrementan el número de realizaciones independientes de ruido.

La admisibilidad se volvió a calcular directamente desde los cuatro umbrales
y la convergencia de todas las ejecuciones. El verificador compara ese valor
con el resumen guardado, comprueba la estabilidad de las métricas entre
repeticiones y conserva cada condición en su denominador.

- Optimized DMD devuelve campos finitos, pero **seis de sus 216 condiciones no
  convergen**; son 18 ejecuciones no convergentes. No se convierten en éxitos
  por tener error finito y no se eliminan del denominador.
- El candidato `fb_dmd_fallback_tls` activa su alternativa TLS en **23 de sus
  216 condiciones**. Su comportamiento describe esa estrategia con fallback;
  no constituye una evaluación de fbDMD puro en esas condiciones.
- Las 15 fuentes científicas registradas coinciden byte a byte con sus hashes
  de procedencia. Esta comprobación vincula los resultados con la implementación
  conservada; no sustituye la validación numérica de cada referencia.

El reanálisis de esa campaña se verificó desde el ledger, sin importar el
script de tablas. Coinciden **430 condiciones admisibles**, **66 bloques de
216 sin ningún candidato admisible** y **1.469 condiciones con error de
velocidad aceptado pero vorticidad rechazada**. También coinciden los 52
contadores de sensibilidad al umbral de vorticidad y las cinco ablaciones.
Helmholtz conserva el error de curl en los 216 bloques de cada una de las
cuatro parejas sin filtrado. Añadir el corte de dos tercios a TLS con
Helmholtz rescata 100 aceptaciones y no pierde ninguna bajo los umbrales
nominales; ese resultado depende del requisito de fidelidad.

De las 150 reconstrucciones newtonianas admisibles, 147 tienen residuo extra
normalizado de la ecuación de vorticidad mayor que uno. Es una alerta de
dinámica bajo el operador y la escala declarados, no un residuo completo de
momento ni una tolerancia DNS. Los cuatro criterios nominales no incluyen
esa puerta, por lo que «admisible» no debe traducirse como satisfacción de
toda Navier–Stokes.

Las ganancias antiguas frente a identidad raw describen la **cadena completa**,
incluidos proyección y filtrado. El banco anterior carece de identidad con
Helmholtz y dos tercios; por ello no permite aislar el incremento de TLS con
ese mismo prior. Esta limitación se publica y el banco nuevo incorpora las
tres variantes de identidad. Las ablaciones algoritmo→algoritmo con H sí
conservan observación y modelo.

## Costo condicionado a calidad

CPU, tiempo transcurrido, RSS y proxy de trabajo deben permanecer separados.
Una frontera de Pareto sólo se compara dentro de un mismo flujo, ruido,
semilla y postprocesamiento; no se pueden agrupar problemas físicos distintos
como si fueran intercambiables. Un perfil de desempeño debe definir el
denominador y el tratamiento de los métodos no admisibles.

La RSS es el máximo residente de vida del proceso, no una medida de asignación
incremental de una función. Cuando raw, Helmholtz y banda se evalúan en el mismo
hijo, los picos de postprocesamiento pueden contener memoria usada por una
variante o evaluación anterior. Sus CPU y tiempos pueden separarse, pero esos
picos de RSS no permiten atribuir el incremento exclusivamente a la proyección.
La máquina compartida y una única medición de costo por ajuste tampoco permiten
afirmar una jerarquía precisa de tiempos ni un intervalo de incertidumbre de
recursos.

En el driver nuevo, `fft_calls=2` procede del diagnóstico de Helmholtz. El
corte posterior añade otra FFT y otra transformada inversa, de modo que el
campo heredado no cuenta las cuatro transformadas de la cadena completa. No
se usa ese contador parcial como costo total. El perfil CPU/tiempo del
postprocesamiento sí envuelve todos sus pasos.

## El error inevitable del prior de banda

Se verificó la cobertura única de las 1.890 filas del presupuesto espectral
y las 63 filas de pérdida de señal limpia, con nueve cortes declarados.
Se volvieron a calcular
independientemente los pisos de velocidad y vorticidad de los siete flujos
en el corte utilizado por la comparación, `k <= 10`. Se emplearon FFT y
derivadas desde primitivas, sin llamar a las funciones del analista.

| Referencia 128² | Piso de error agregado de vorticidad por banda `k <= 10` |
| --- | ---: |
| Cuatro vórtices NS | 3.5411 % |
| NS multiescala | 10.8366 % |
| Oldroyd-B | 1.44765 % |
| FENE-P | 1.18960 % |
| HB adelgazante | 2.01234 % |
| Bingham regularizado | 1.07190 % |
| HB espesante | 0.75611 % |

Estos pisos usan la norma sobre todas las salidas guardadas, no el máximo
instantáneo. Proceden de la componente limpia fuera del subespacio permitido:
ningún candidato confinado a esa banda puede recuperarla. En particular, el
corte 10 no permite un error agregado de vorticidad de 5 % sobre el caso
multiescala, incluso sin ruido y con un modelo perfecto dentro de banda.
Los pisos se refieren a la referencia numérica y al operador discreto
declarados; no son cotas de error respecto a la solución física exacta.

La recomputación en espacio real de tres bloques —NS multiescala con ruido
IID, Oldroyd-B con ruido espacial y HB adelgazante con ruido impulsivo,
semilla 101— coincide en ambos observables con la descomposición espectral:
`error_control² = señal_descartada² + ruido_proyectado_retenido²`.
Los términos son ortogonales bajo la cuadratura periódica uniforme; los
errores relativos sin elevar al cuadrado no se suman. Las curvas de nueve
cortes caracterizan un prior ya conocido y no eligen el mejor corte usando
la referencia limpia.

## Sensibilidad a rango e intensidad

La campaña de sensibilidad contiene **360 ajustes** y **1.080 evaluaciones**
de esos mismos ajustes, en 36 bloques de observación. No presenta fallos ni
no convergencias. La admisibilidad recalculada desde los cuatro criterios es
0 para raw, 12 para Helmholtz y 324 para Helmholtz con corte 10. Son resultados
de dos trayectorias, dos contaminaciones, tres intensidades y cinco rangos
declarados; no una estimación para todos los fluidos o niveles de ruido.

Las tres intensidades comparten el patrón aleatorio de cada combinación
flujo/ruido/semilla: hay 12 campos base escalados a tres RMS. Las curvas están
pareadas y los 36 arrays observados no son 36 sorteos independientes.

Se volvieron a calcular los diez pisos Frobenius de velocidad mediante SVD
de las matrices limpias. En NS multiescala, el piso de rango 2 es 29.8933 % y
el de rango 8 es 1.41107 %; en Oldroyd-B son 9.28263 % y 0.0805214 %.
La trayectoria multiescala de esta ventana requiere más grados temporales
para una misma fidelidad de velocidad. El error de vorticidad del POD limpio
se informa como diagnóstico y no como cota óptima de vorticidad.

Veinticuatro condiciones —rango 8, RMS 20 % y los métodos POD/TLS en ambos
flujos— reproducen el contrato y las observaciones del benchmark principal.
La auditoría compara SHA de observación, rango y métricas de las tres
variantes. Su coincidencia prueba consistencia entre campañas; no constituye
24 realizaciones adicionales independientes. La selección de un rango
«óptimo» mirando la referencia limpia permanece fuera del alcance de este
barrido descriptivo.

## Controles limpios: sesgo de representación y regularización

Los **84 ajustes y 252 evaluaciones** sin contaminación utilizan las mismas
siete trayectorias. Sus hashes de velocidad observada coinciden con los de
la velocidad limpia; no se inyectó ruido ni se renormalizó RMS. Se conservan
70 contratos del experimento de intensidad 20 %, y se añaden 14 controles
SPCP/SPCP→TLS con intensidad conocida cero. El presupuesto heredado 20 %
significa regularización aplicada a un dato limpio; no describe ruido
observado en este control.

La admisibilidad independiente coincide con **77 de 84** para cada variante;
no hay fallos ni no convergencias. Se calcularon por SVD propia los siete
pisos de velocidad de POD de rango 8 y coinciden con su salida limpia.
La comparación con DMD distingue la compresión de rango de la restricción
adicional de evolución exponencial/modal.

Con `alpha=beta=0`, todos los pares `L+S=M` minimizan el objetivo: no existe
una descomposición identificada de manera única. La implementación comienza
con `S=0` y devuelve `L≈M`; por eso SPCP con presupuesto cero coincide con
identidad y su cadena con TLS coincide con TLS directo. Se verificaron 42
parejas en las cuatro métricas de admisibilidad, con diferencia máxima
absoluta `3.51e-14`. La tolerancia de equivalencia es `1e-10`, explícita ante
SVD, ajuste modal y diferenciación; no se afirma igualdad de todos los
coeficientes o campos internos.

El diagnóstico `sparse_fraction` cuenta entradas distintas de cero exacto.
En este control puede aproximarse a 99 % por residuos de redondeo, aunque
la norma de la componente dispersa sea del orden de `1e-12`. No significa
que se haya contaminado 99 % del dato. No se infiere proporción física de
corrupción a partir de ese contador sin una tolerancia de amplitud.

La identidad limpia reproduce el cierre viscoplástico nativo en esfuerzo,
viscosidad, corte y disipación con errores inferiores a `1e-10`; la memoria
viscoelástica sigue sin inferirse de velocidad. Estos controles verifican
sesgo y contratos de estimación sobre datos existentes, no aportan nuevas
simulaciones ni una validación independiente de la referencia física.

## Cierre de la campaña comparativa principal

La verificación definitiva exige cobertura completa de **1.050 ajustes**,
**3.150 evaluaciones**, 105 observaciones pareadas y diez métodos. No se
registran fallos de ejecución. Se conservan **ocho no convergencias de
Optimized DMD** en el denominador de su método; no son éxitos admisibles.
Las aceptaciones recalculadas son 6 sin posprocesamiento, 44 con Helmholtz
y 825 con Helmholtz más corte 10.

Cada columna de la siguiente tabla tiene **105 condiciones por método**:
siete flujos, cinco contaminaciones y tres semillas. Las tres columnas
evalúan la misma estimación de cada ajuste.

| Método | Raw | Helmholtz | Helmholtz + corte 10 |
| --- | ---: | ---: | ---: |
| Identidad | 0 | 0 | 87 |
| POD r8 | 0 | 3 | 90 |
| DMD r8 | 0 | 3 | 105 |
| TLS-DMD r8 | 0 | 3 | 93 |
| PCP | 3 | 13 | 80 |
| SPCP penalizada | 0 | 3 | 105 |
| Outlier Pursuit | 0 | 0 | 0 |
| PCP → TLS r8 | 3 | 13 | 77 |
| SPCP → TLS r8 | 0 | 3 | 102 |
| Optimized DMD r8 | 0 | 3 | 86 |

La identidad con el mismo prior acepta 87 de 105 condiciones. Este control
impide atribuir las 825 aceptaciones exclusivamente a la estimación temporal
o de bajo rango. DMD y SPCP añaden cobertura en ese banco y bajo esas cuatro
puertas; el dato no demuestra superioridad universal, precisión constitutiva
total ni pronóstico. El verificador comprueba los 1.050 grupos de tres
semillas: contadores, medianas y extremos coinciden con el ledger.

Se regeneraron desde primitivas estadísticas las observaciones almacenadas
de identidad, comprobando su hash de trayectoria y sus tres estados exactos.
Los diez estimadores comparten el hash observado dentro de cada bloque. Las
épocas de driver declaradas conservan hashes propios: se verifica mediante
AST que todas las funciones numéricas y constantes permanecen idénticas.
La corrección de lectura distingue digest canónico de arrays/metadatos de
SHA del archivo NPZ comprimido; no altera los parámetros ni las fuentes de
los solvers.

## Fidelidad con condiciones del banco original

Se completaron **18 integraciones de refinamiento** que conservan la física,
el horizonte y la cadencia de las referencias originales. Las verificaciones
independientes comprueban sus archivos y fuentes, y vuelven a calcular 22
filas de velocidad, vorticidad, esfuerzo y viscosidad sobre todos los tiempos
guardados, utilizando prolongación Fourier propia.

Para HB adelgazante del banco original, 32² frente a 128² durante `T=2`
alcanza **0.536765 % en velocidad, 4.423839 % en vorticidad, 6.723072 % en
esfuerzo de Frobenius y 16.535409 % en viscosidad aparente**. Un control
adicional independiente de 64² frente a 128² confirma 1.803389 % en esfuerzo
y 7.994312 % en viscosidad. Esto sitúa la precisión de gradientes y cierre
constitutivo junto a la comparación de denoisers; no se deduce de imágenes
suaves ni del número de nodos. La referencia 128² sigue siendo numérica.

Dos integraciones adicionales conservan 161 campos NS128² a separación
0.025. Sus comparaciones de cadencia subsamplean exactamente esa misma
trayectoria. Se comprobó la identidad cuadrática de las veinte
descomposiciones del residuo y se recalculó desde arrays el residuo de
orden cuatro de ambos casos a 0.025, separando la parte temporal proyectada
y la parte espacial descartada. Se distinguen estas dos integraciones de
las 18 referencias coincidentes y de los ajustes de reconstrucción.

## Verificación de referencias originales

El 14 de octubre de 2026 se consultaron los registros públicos de Crossref.
Coinciden título, autores y año en las siguientes fuentes originales:

| Método o alcance | Fuente original verificada | Alcance que debe conservarse |
| --- | --- | --- |
| PCP | Candès, Li, Ma y Wright (2011), [10.1145/1970392.1970395](https://doi.org/10.1145/1970392.1970395). | Recuperación bajo hipótesis de bajo rango y corrupción dispersa; no denoiser universal. |
| SPCP restringida | Zhou, Li, Wright, Candès y Ma (2010), [10.1109/ISIT.2010.5513535](https://doi.org/10.1109/ISIT.2010.5513535). | La implementación local usa una formulación penalizada; no hereda automáticamente el teorema del programa restringido. |
| Sesgo por ruido y correcciones DMD | Dawson, Hemati, Williams y Rowley (2016), [10.1007/s00348-016-2127-7](https://doi.org/10.1007/s00348-016-2127-7). | Las hipótesis de ruido y condicionamiento delimitan TLS y forward–backward. |
| Optimized DMD | Askham y Kutz (2018), [10.1137/M1124176](https://doi.org/10.1137/M1124176). | Ajuste exponencial separable con proyección variable; éxito del optimizador no es exactitud ni optimalidad global. |
| Métodos robustos de DMD | Askham, Zheng, Aravkin y Kutz (2022), [10.1137/21M1417405](https://doi.org/10.1137/21M1417405). | Una pérdida robusta cambia el problema de ajuste; limpiar con RPCA y luego DMD no equivale a implementar esa formulación. |

La bibliografía y el texto distinguen métodos ejecutados, antecedentes y
líneas futuras. Las referencias científicas respaldan formulaciones; las
afirmaciones cuantitativas de comparación deben proceder de los ledgers
calculados en este repositorio.

## Reproducción de la auditoría

```bash
python python/scripts/verify_model_comparison.py --include-new
```

El comando completo requiere regenerar o disponer localmente de los NPZ
de trayectoria; las muestras públicas de tres o cinco estados sirven para
visualización y no permiten recalcular todos los máximos temporales.
Sin `--include-new`, el verificador audita los ledgers históricos y su
reanálisis sin requerir los archivos completos de las nuevas simulaciones.

La auditoría termina con código distinto de cero ante una condición ausente,
duplicada, cambio de fuente, discrepancia de resumen o aceptación incompatible
con convergencia. Las comprobaciones nuevas utilizan sus ledgers completos;
no se extrapolan desde el piloto ni se convierten en certificación DNS.
