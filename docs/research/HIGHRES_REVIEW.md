# Revisión independiente de resolución y figuras

Fecha: 14 de octubre de 2026. Revisión final de operadores, comparaciones entre
mallas y evidencia de las nuevas campañas. Se verificaron datos de las tres
familias y sus ilustraciones, además de los controles de reconstrucción.
Los solvers científicos conservan sus fuentes y hashes; la auditoría usó
comprobaciones analíticas y cálculos independientes sobre sus archivos.

## Qué demuestra una malla más grande

El número de celdas por sí solo no establece que una simulación represente el
flujo. Para la condición inicial de cuatro núcleos gaussianos, el radio
característico es `sigma = 0.065 L`, por lo que las mallas 64, 128, 256 y 512
tienen respectivamente 4.16, 8.32, 16.64 y 33.28 celdas por `sigma`. En el
estado inicial aleatorio de Fourier, cuyo corte es el modo 8, esas mallas
tienen 8, 16, 32 y 64 celdas por la menor longitud de onda inicial. Estas
cantidades describen el estado inicial; los gradientes que surjan durante la
evolución requieren evaluación adicional.

La comparación justa mantiene dominio físico, condición inicial, parámetros
constitutivos y tiempos de salida. La malla fina debe ser una integración
nueva del sistema, con sus propios pasos internos y registros. Reescalar una
imagen o interpolar el resultado de una malla pequeña no constituye un
experimento nuevo. La prolongación de Fourier se usa aquí exclusivamente para
comparar campos calculados por separado.

Se necesitan dos errores entre mallas:

- **Modos comunes:** comparación en los modos de Fourier que puede retener la
  malla gruesa. Aísla el cambio de dinámica de esos modos al refinar.
- **Campo completo:** prolongación de la solución gruesa a la malla fina,
  comparada con todos los modos de la referencia fina. Incluye las estructuras
  que la malla gruesa no puede representar.

El error de modos comunes puede ser pequeño aunque falten gradientes o
estructuras. Por ello, se debe mostrar también la fracción de energía y de
enstrofía de la referencia que queda fuera del corte grueso. La vorticidad es
más sensible a resolución que la velocidad porque multiplica cada modo por
su número de onda.

La evidencia útil conserva las curvas y el máximo de error entre todos los
tiempos guardados, además del tiempo final. Para los polímeros, la comprobación
incluye `C`, `C-I`, esfuerzo, autovalor mínimo y máxima traza: comparar solamente
`C` puede ocultar el error de deformación detrás del tensor identidad. Para
Herschel–Bulkley/Bingham, incluye esfuerzos, tasa de deformación, viscosidad y
balance de energía. La clasificación de cedencia sigue siendo una estimación
dependiente de un umbral, no una solución exacta de la frontera de cedencia.

Los estudios de paso temporal, cuadratura constitutiva y regularización
comprueban cosas distintas. Refinar la malla reduce error espacial; reducir el
paso efectivo reduce error temporal; variar el padding comprueba cuadratura;
variar `m` o `epsilon` cambia el modelo regularizado. Un límite adaptativo puede
hacer que dos pasos nominales diferentes produzcan pasos internos iguales.
Por eso se deben registrar máximos, mínimos, límites y rechazos reales, y no
inferir un orden temporal sólo a partir del paso solicitado.

## Comprobaciones independientes realizadas

Se construyó un campo periódico analítico independiente de las condiciones
de prueba: `u=sin(3x) cos(5y)`, `v=-(3/5) cos(3x) sin(5y)`. Tiene divergencia
nula y vorticidad `(9/5+5) sin(3x) sin(5y)`. Se comprobaron los operadores de
transferencia y diagnóstico de la campaña newtoniana sin ejecutar su
generador de datos.

| Transferencia | Máximo error absoluto al prolongar | Máximo error absoluto al restringir |
| --- | ---: | ---: |
| 64 → 128 | 3.89e-15 | 2.55e-15 |
| 128 → 256 | 4.36e-15 | 2.55e-15 |
| 256 → 512 | 4.89e-15 | 2.90e-15 |

La suma del espectro radial coincide con la energía del campo a menos de
`2.8e-17`, y con la enstrofía a menos de `8.9e-16`. La derivada espectral de
vorticidad coincide con la expresión analítica a menos de `2.7e-13`. Estas
comprobaciones verifican amplitudes, normalización FFT y signos; los valores
están al nivel del redondeo en doble precisión.

También se ejecutó Navier–Stokes en 64², 128² y 256² para Taylor–Green con
`nu=0.01`, `A=0.8`, `dt=0.01`, `T=0.2`. La comparación contra la expresión
analítica dio error relativo L2 de velocidad de `4.35e-16`, `4.10e-16` y
`3.94e-16`, respectivamente; la divergencia RMS máxima fue `8.9e-15`.
Taylor–Green sólo contiene un modo de Fourier. Este resultado comprueba
difusión, signos y representación del modo; no demuestra resolución de
interacciones no lineales ni certifica DNS.

La transferencia con `scipy.signal.resample` conserva amplitudes de campos
periódicos, y trata el Nyquist de campos reales. Los estados del solver usan
un corte estricto de dos tercios, por lo que no incluyen Nyquist. Los campos
constitutivos derivados pueden tener frecuencias adicionales; deben
compararse como campos completos además de cualquier restricción a los modos
activos. Las derivadas físicas de velocidad se aplican a estados sin Nyquist.

Una segunda comprobación añadió sólo a la referencia 128² una onda de
velocidad incompresible `v=0.05 cos(35x)`, que queda fuera de la banda activa
de 64². Los errores de modos comunes fueron inferiores a `3.5e-15`, pero los
errores de campo completo fueron `6.05 %` en velocidad y `34.20 %` en
vorticidad. Al añadir también un modo de esfuerzo `tau_xy=0.01 cos(35x)`, el
error común fue inferior a `9e-16` y el completo `7.05 %`. Este caso fabricado
no es un resultado físico de la campaña; verifica que los diagnósticos de
campo completo detectan las escalas que una comparación de modos comunes
descarta.

Se comprobó asimismo una onda coseno exactamente en Nyquist al transferir
64 → 128 y 128 → 256. La amplitud del interpolante real se conserva, con
errores máximos de `2.1e-14` y `4.1e-14`, respectivamente, y la transferencia
de vuelta conserva los datos originales. Esto verifica el tratamiento del
Nyquist discreto; no recupera información física que nunca se haya muestreado.

En el solver de viscoplasticidad se comprobó de forma independiente el límite
newtoniano de corte periódico: `n=1`, `K=0.04`, `tau_y=0`, `A=0.8`,
`dt=0.001`, `T=0.1`, mallas 64² y 128². Las referencias son
`u=0.8 exp(-0.04 t) sin(y)`, `v=0` y
`tau_xy=0.032 exp(-0.04 t) cos(y)`. El error relativo de velocidad fue
inferior a `7.2e-15`, el máximo error absoluto de esfuerzo inferior a
`4.1e-15` y el residuo del balance integrado de energía, relativo a la energía
inicial, inferior a `1.7e-14`. Comprueba el operador de esfuerzo y el límite
analítico; no sustituye la convergencia de la ley no polinómica regularizada.

## Requisitos para las ilustraciones del documento

Las secuencias temporales deben utilizar el mismo rango de colores dentro de
una comparación, llevar tiempo, dominio, magnitud y malla, y conservar las
estructuras físicas de los datos. La visualización debe declarar si muestra
vorticidad, rapidez, esfuerzo o conformación; una figura de rapidez por sí
sola puede hacer invisibles estructuras de signos opuestos. Los paneles de
error necesitan su propia escala, visible y distinta de la del campo físico.
La imagen de la referencia comparada consigo misma debe identificarse: su
error cero no es evidencia de exactitud física.

Los resultados se pueden llamar convergentes o bien resueltos **para las
condiciones y el intervalo ensayados** si los errores y controles lo
respaldan. No deben extrapolarse a estelas con cilindro, condiciones de pared,
flujos tridimensionales, alto Weissenberg o caos. Un espectro decreciente y
un balance de masa con densidad constante son diagnósticos útiles, pero no
certificados universales de DNS.

## Estado de la auditoría de campañas

### Cuatro vórtices newtonianos: comparación contra 512²

Se cargaron los archivos calculados por separado en 64², 128², 256² y 512²,
con 41 instantes en `0 <= t <= 4`, y se volvieron a calcular ambos errores de
campo completo sin llamar al comparador de la campaña. El SHA-256 del archivo
512² coincide con su manifiesto. Esa referencia contiene 2.325 pasos internos
y paso máximo `0.0020321383`, frente al nominal `0.005`: la integración fina
es nueva y respeta sus propios límites.

| Malla | Máximo error completo de velocidad | Máximo error completo de vorticidad |
| --- | ---: | ---: |
| 64² | 2.3373984e-4 | 3.6260439e-3 |
| 128² | 2.4713900e-7 | 7.4100333e-6 |
| 256² | 4.0265188e-10 | 2.6711058e-9 |

El máximo de 64² ocurre en `t=1.7` y el de 128² en `t=1.6`; el error final
es mucho menor. Para este estado suave, incluso 64² da errores inferiores al
1 % en velocidad y vorticidad frente a 512². La malla fina confirma con números
qué estaba resuelto y permite contrastar
casos constitutivos cuyos gradientes requieren más resolución. Tampoco
demuestra que los demás casos o parámetros estén resueltos.

### Primer control de Bingham regularizado: 64² frente a 128²

Los dos archivos de `0 <= t <= 2` tienen digest de datos correcto, esfuerzo
constitutivo coherente y energía decreciente. Una recomputación independiente
dio los siguientes máximos de campo completo entre los 21 instantes:

| Cantidad | Máximo error relativo |
| --- | ---: |
| Velocidad | 1.5466841e-4 |
| Esfuerzo desviador (norma tensorial de Frobenius) | 4.9604058e-3 |
| Viscosidad aparente | 2.3994563e-2 |
| Tasa de corte | 3.6225518e-3 |

La viscosidad aparente tiene un pico de error de `2.40 %`, aunque el promedio
global es `1.26 %`; es necesario informar la cantidad y el agregado. El
esfuerzo se calculó aquí con `tau_xx² + 2 tau_xy² + tau_yy²`, porque el archivo
guarda sólo los tres componentes independientes. La norma vectorial de los
tres componentes guardados da un máximo distinto, `5.1669806e-3`; ambas
convenciones son reproducibles, pero deben identificarse al comparar.
El
balance de energía relativo máximo es `3.95e-11` en 64² y `4.65e-13` en
128². Los pasos máximos efectivos son `0.0070106` y `0.0017226`, frente al
nominal común `0.02`. Esta comparación espacial mezcla errores temporales
residuales hasta que el control temporal independiente los acote. La
referencia 256² de horizonte corto debe distinguirse de esta comparación de
horizonte largo.

### Estado newtoniano multiescala: comparación contra 512²

El archivo 512² del caso denominado `decaying_turbulence` contiene otra
integración de 2.038 pasos, 41 salidas hasta `t=4` y paso máximo
`0.0028131004`. Se verificó su SHA-256 y se recomputaron los errores completos
para cada malla, sin utilizar el comparador del script.

| Malla | Máximo error completo de velocidad | Máximo error completo de vorticidad |
| --- | ---: | ---: |
| 64² | 2.0070423e-3 | 1.3119957e-2 |
| 128² | 2.4554636e-5 | 3.0319349e-4 |
| 256² | 7.5251506e-10 | 1.5648892e-8 |

La velocidad de 64² parece suficiente bajo una tolerancia operativa de 1 %,
pero su vorticidad alcanza `1.312 %` de error. Ambas cantidades convergen
mucho más en 128² y 256². Los picos de 64² ocurren en `t=1.5`, y los de
128² en `t=0.9`. El nombre del caso no prueba caos ni una cascada turbulenta
universal: son dos dimensiones periódicas y un intervalo finito.

Se revisó el CSV definitivo: 410 filas corresponden a seis comparaciones
espaciales y cuatro temporales, cada una en 41 tiempos. La refinación
temporal a 128² utiliza máximos de paso efectivamente distintos
`0.005/0.0025/0.00125`, sin límites de estabilidad activos. Los mínimos del
paso incluyen pequeños pasos de alineación/redondeo, próximos a `2e-15`; no
deben interpretarse como evidencia de rigidez. Los errores temporales
reportados son mucho menores que los espaciales de 64²/128² y el orden de diferencias
sucesivas es aproximadamente 4.08 en ambos casos.

### Memoria y esfuerzo viscoelásticos: comparación contra 256²

Los archivos Oldroyd-B/FENE-P 256² tienen SHA-256 coincidente con el manifiesto,
25 salidas hasta `t=2.4`, autovalores positivos y cero rechazos. Se volvieron
a calcular los máximos completos de velocidad, `C-I` y esfuerzo, prolongando
cada campo grueso al campo fino sin filtrar la referencia.

| Modelo | Malla | Máximo error u | Máximo error C-I | Máximo error de esfuerzo |
| --- | ---: | ---: | ---: | ---: |
| Oldroyd-B | 64² | 1.9584828e-4 | 1.2238668e-1 | 1.2238668e-1 |
| Oldroyd-B | 128² | 8.8721594e-6 | 4.3829168e-2 | 4.3829168e-2 |
| FENE-P | 64² | 1.2135211e-4 | 7.8904542e-3 | 8.7866834e-3 |
| FENE-P | 128² | 4.7378123e-6 | 1.0097451e-3 | 1.1648040e-3 |

La velocidad de Oldroyd-B está muy próxima a la referencia desde 64²,
mientras que su esfuerzo difiere hasta `12.24 %`. En 128², el esfuerzo
todavía difiere `4.38 %`. Una figura suave de velocidad no justifica afirmar
resolución del tensor constitutivo. La referencia 256² conserva una cola
de deformación/esfuerzo del orden de `1.16e-4` bajo el indicador declarado
de banda externa; sigue siendo una referencia numérica condicional para
Oldroyd-B y requiere mayor refinamiento para objetivos más estrictos.

En FENE-P, el esfuerzo de 128² difiere hasta `0.1165 %` frente a 256².
La traza máxima de etapas 256² fue `6.37627`, inferior al límite 12, y el
autovalor mínimo `0.320450`. En Oldroyd-B fueron `14.10792` y `0.364799`;
Oldroyd-B no impone el límite de traza de FENE-P. Son verificaciones de
admisibilidad distintas de la resolución espacial.

El control temporal definitivo de cada polímero a 128² compara máximos
efectivos `0.004` y `0.002`, con cero pasos limitados y cero rechazos. El
máximo error completo de esfuerzo es `1.514e-7` en Oldroyd-B y `1.935e-8`
en FENE-P, mucho menor que el cambio espacial. Los 1.204 pasos de la segunda
ejecución incluyen cuatro pasos diminutos de alineación/redondeo, frente a
los 1.200 pasos nominales. Los hashes de todos los módulos científicos
coinciden con la procedencia final de la campaña.

La procedencia viscoelástica conserva el hash histórico del driver que
generó cada integración. El análisis posterior se identifica por
`analysis_source_hashes` y `summary.analysis_script_sha256`, que coinciden
con el driver de análisis actual. Los módulos científicos de las
integraciones coinciden con sus hashes originales; un cambio de presentación
no reescribe ese historial ni obliga a repetir la simulación.

### Leyes HB: comparación de horizonte largo 64² frente a 128²

Se verificaron los digests de ambos archivos por modelo, el descenso de
energía y la disipación positiva. Los máximos completos recalculados son:

| Modelo | Error u | Error de esfuerzo Frobenius | Error de viscosidad | Error de tasa de corte |
| --- | ---: | ---: | ---: | ---: |
| HB adelgazante | 2.2702302e-4 | 9.6304286e-3 | 7.9062944e-2 | 4.2439868e-3 |
| HB espesante | 1.6708725e-4 | 4.4977808e-3 | 2.4522869e-2 | 3.7919895e-3 |

La viscosidad aparente cambia hasta `7.91 %` en el caso adelgazante,
mientras la velocidad sólo cambia `0.0227 %`. La resolución se debe
calificar por cantidad; no se puede extrapolar la convergencia de velocidad
a todos los campos constitutivos. Los residuos máximos relativos de energía
son menores a `3.1e-11` en estas ejecuciones. Un balance discreto pequeño
tampoco garantiza una viscosidad local bien resuelta.

El control temporal HB 128² solicita `0.0004/0.0002/0.0001`. El primer nivel
acepta 354 pasos, limita 304 y rechaza 145 pruebas; los siguientes usan 500 y
1.000 pasos sin límites ni rechazos. Sus máximos son efectivamente distintos,
pero no se puede atribuir un orden RK4 a las tres diferencias mediante sólo
las razones nominales: el nivel grueso tiene integración adaptativa. Los
límites de difusión medidos respetan `0.8`, salvo redondeo al último bit.

La campaña viscoplástica final contiene 18 configuraciones completas. Se
verificaron de forma independiente los tres archivos 256² de horizonte
corto: hashes canónicos correctos, dos salidas en `t=0` y `t=0.1`, y nuevas
integraciones de 354 pasos en Bingham/espesante y 1.928 en adelgazante. Sus
comparaciones de 128² frente a 256² son:

| Modelo | Máximo error completo u | Máximo error de esfuerzo Frobenius | Máximo error de viscosidad |
| --- | ---: | ---: | ---: |
| Bingham | 1.5592443e-5 | 1.1571845e-3 | 8.2175939e-3 |
| HB adelgazante | 1.9632425e-5 | 2.7138228e-3 | 5.0225356e-2 |
| HB espesante | 2.1652934e-5 | 1.1395406e-3 | 8.8055808e-3 |

La viscosidad adelgazante todavía difiere `5.02 %` entre 128² y 256² en ese
intervalo corto. Esta evidencia no resuelve el horizonte largo en 256² ni
demuestra convergencia de todos los campos de 128². El derivado
`derived_stress_frobenius.json` coincide con los cálculos independientes
ponderados, conserva la norma vectorial original por separado y especifica
los pesos `(1,2,1)`. La auditoría de integridad del generador verifica
finitud, tiempos, horizonte y hashes de las 18 configuraciones; las fuentes
de campaña y ambos módulos físicos coinciden con sus hashes finales.

### Reconstrucción y escala física del ruido

Se auditaron las 60 condiciones del banco visual: todas completaron, pero
ninguna supera conjuntamente las tolerancias de velocidad, vorticidad,
divergencia y energía. En el caso gaussiano IID 256², semilla 7, se volvió
a generar la observación con el RNG y la normalización declarados: coincide
exactamente con el archivo `identity`. El ruido tiene RMS relativo `0.2`.
El cálculo independiente del archivo TLS rango 8 + Helmholtz/2/3 reprodujo
`e_u=0.0455724416` y `e_omega=1.9959115264`, coincidentes con el ledger.

La mejora de velocidad no basta para reconstruir vorticidad. En ruido IID,
refinar la observación conserva RMS, pero añade contenido de frecuencias
altas y empeora el error de derivadas. En ruido de ancho de filtro físico
fijo, los resultados 128² y 256² permanecen prácticamente iguales. La
anchura `sigma_f=0.10` es la desviación del kernel gaussiano, no la de la
autocorrelación del ruido; ésta es aproximadamente `sqrt(2) sigma_f` en el
límite continuo.

La proyección de Helmholtz elimina una componente de gradiente y conserva
el rotacional bajo los operadores periódicos compatibles. Por tanto, la
proyección por sí sola no elimina ruido de vorticidad: las mejoras de esta
magnitud provienen del modelo temporal o del filtro de frecuencias que la
acompaña. En ruido espacial suave, las métricas de vorticidad TLS antes y
después de Helmholtz/2/3 son casi iguales, aunque mejore la velocidad.

Después de observar esas 60 condiciones se comprobó una sola hipótesis
exploratoria: Helmholtz seguido de un corte circular físico fijo
`|k|<=10=1/sigma_f`, aplicado a TLS rango 8 previamente guardado. No se
barrió un conjunto de cortes ni se eligió por el error limpio.

| Ruido | Malla | Semilla | Error u | Error de vorticidad |
| --- | ---: | ---: | ---: | ---: |
| IID por celda | 128² | 7 | 1.93345 % | 8.18880 % |
| IID por celda | 256² | 7 | 1.70832 % | 7.10765 % |
| Kernel físico fijo | 128² | 7 | 5.29084 % | 22.28457 % |
| Kernel físico fijo | 256² | 7 | 5.29084 % | 22.28457 % |

Estos cuatro puntos superan las tolerancias anteriores. El corte impone una
hipótesis adicional de suavidad y puede eliminar señal física en otros
regímenes. Es evidencia exploratoria posterior, no una validación ciega ni
prueba de superioridad de DMD por sí solo. Un comparador que aplique el mismo
corte directamente a las observaciones permite separar la aportación de
esa hipótesis de la del modelo temporal.

El seguimiento definitivo ejecutó 24 condiciones: dos estimadores, dos
protocolos de ruido, dos mallas y tres semillas. Se conservaron los
resultados originales `0/60`, y el seguimiento dio `18/24` aceptaciones.
Las 12 condiciones IID superan los cuatro criterios; con kernel físico fijo,
las seis TLS los superan y las seis de observación filtrada no. Se volvió
a calcular la observación filtrada de los cuatro casos de semilla 7 desde
los archivos originales; velocidad/vorticidad coinciden con el ledger a
menos de `1e-15`.

| Ruido | Malla | Mediana de error de ω, observación filtrada | Mediana de error de ω, TLS + mismo filtro |
| --- | ---: | ---: | ---: |
| IID por celda | 128² | 10.02751 % | 8.17100 % |
| IID por celda | 256² | 5.86686 % | 7.16104 % |
| Kernel físico fijo | 128² | 49.22696 % | 22.16777 % |
| Kernel físico fijo | 256² | 49.22696 % | 22.16777 % |

TLS aporta una mejora con ruido espacial correlacionado en estos casos,
pero la observación filtrada tiene menor error de vorticidad bajo IID en
256². La comparación preserva ese resultado: el modelo temporal no gana
universalmente. Las 24 condiciones son un seguimiento exploratorio que
utiliza los mismos flujos y semillas, sin validación independiente ciega.

## Verificación visual

Se inspeccionaron los PNG de mallas, acoplamiento constitutivo y evolución
newtoniana 512². Las secuencias identifican tiempos y magnitudes, usan escalas
de color fijas y coinciden con los signos y centros del estado inicial. La
fusión y formación de filamentos son evoluciones calculadas del campo
periódico. Los esquemas de mallas están rotulados como esquemas.

Las figuras de reconstrucción 256² declaran la saturación fuera del rango de
vorticidad limpia y muestran el error de vorticidad, además de la comparación
visual. La gráfica agregada presenta mediana y rango de tres semillas; esas
barras no son intervalos de confianza. Sus líneas de tolerancia se identifican
como tolerancias de reconstrucción, no de DNS.

También se inspeccionaron las figuras compactas definitivas de traza,
esfuerzo viscoelástico y refinamiento completo. El rango de colores se
mantiene entre Oldroyd-B/FENE-P y entre tiempos; el tensor inicialmente
equilibrado tiene esfuerzo cero. Las curvas de refinamiento completo
reproducen los errores independientes, hacen visible el esfuerzo mal
resuelto de Oldroyd-B y conservan leyendas legibles.

Se inspeccionaron también las figuras definitivas de evolución y
constitución viscoplástica: mantienen la escala común de vorticidad entre
modelos y tiempos, y presentan la viscosidad en una escala logarítmica común
rotulada. Los centros iniciales, el cambio de forma y las diferencias de
viscosidad corresponden a los archivos calculados.

La revisión queda completa para los controles y condiciones descritos.
Las afirmaciones de resolución siguen siendo específicas de cantidad,
geometría, régimen y horizonte. Oldroyd-B 256² conserva una referencia
tensorial condicional; HB adelgazante conserva discrepancia de viscosidad
del 5 % en la prueba corta. El filtro físico se presenta como hipótesis
exploratoria con un comparador equivalente. Estos límites deben conservarse
en las conclusiones y no convertirse en certificaciones de DNS, caos,
estelas externas o superioridad universal de un estimador.
