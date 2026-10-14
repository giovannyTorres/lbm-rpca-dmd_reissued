# Extensión no newtoniana: referencias verificables y campaña de estelas


Esta guía conserva las referencias analíticas/de canal y el diseño de estelas externas. Los nuevos solvers periódicos 2D ya implementados y ejecutados se documentan en [VISCOELASTIC_2D.md](VISCOELASTIC_2D.md) y [VISCOPLASTIC_2D.md](VISCOPLASTIC_2D.md); sus resultados actuales están en [EXTENDED_RESULTS.md](EXTENDED_RESULTS.md).

Este capítulo amplía el banco de reconstrucción con perfiles Herschel–Bulkley/Bingham exactos y una solución numérica de arranque de Oldroyd-B que evoluciona velocidad y esfuerzos poliméricos. El código está en `python/src/fluid_denoise/research/rheology.py`. Son referencias de canal plano plenamente desarrollado; las estelas de cilindro, las inestabilidades elásticas y los casos caóticos requieren otra simulación. La campaña para esos casos está especificada en `configs/research/nonnewtonian_wake_campaign.json` y todavía no se ha ejecutado ni conectado a un solver externo.

## 1. Planteamiento físico

La comparación LBM–RPCA–DMD se extiende en dos direcciones separables: cambiar la dinámica limpia mediante una ley constitutiva y cambiar el modelo de contaminación de las observaciones. No basta con sustituir una viscosidad constante por una viscosidad dependiente de la tasa de deformación para representar elasticidad. La elasticidad exige memoria y una ecuación de evolución del esfuerzo o de la conformación.

Las preguntas experimentales son:

1. ¿La reconstrucción recupera caudal, vorticidad y perfil de velocidad, además del error matricial, cuando existe un núcleo sin ceder?
2. ¿Los métodos conservan la respuesta transitoria y la fase bajo forzamiento oscilatorio de un fluido con memoria?
3. ¿Persiste una ventaja de RPCA cuando el ruido es gaussiano o correlacionado, y cuando la señal deja de ser aproximadamente de rango bajo?
4. ¿Los modos identifican frecuencias físicas que se reproducen al refinar malla, paso y duración de observación?

Los perfiles estacionarios de este banco tienen rango temporal uno o cero. Sirven para comprobar campos y observables físicos, pero favorecen de forma artificial la hipótesis de rango bajo si se usan como evidencia general a favor de RPCA. Deben acompañarse del arranque/oscilación de Oldroyd-B y, posteriormente, de las estelas convergidas. Un resultado sobre estos canales no establece desempeño en turbulencia o caos.

## 2. Marco teórico

### 2.1 Herschel–Bulkley y Bingham

Para el canal de semialtura $H$, eje transversal $y\in[-H,H]$ y fuerza de presión $G=-\partial_xp$, el balance estacionario produce

\[
\tau_{xy}(y)=-Gy.
\]

La ley constitutiva escalar de cortante es

\[
\dot\gamma=0\quad\text{si }|\tau_{xy}|\leq\tau_y,
\qquad
\tau_{xy}=\operatorname{sgn}(\dot\gamma)\left(\tau_y+K|\dot\gamma|^n\right)
\quad\text{si }|\tau_{xy}|>\tau_y.
\]

$K$ tiene unidades de Pa·s$^n$; comparar valores numéricos de $K$ con distintos $n$ sin fijar una tasa de referencia no equivale a comparar viscosidades iguales. $n=0.6$ representa adelgazamiento, $n=1$ con cedencia representa Bingham y $n=1.4$ representa engrosamiento. El núcleo sin ceder tiene semiancho

\[
y_p=\min(H,\tau_y/|G|).
\]

Con $[a]_+=\max(a,0)$, $q=1+1/n$ y $G\neq0$, la solución exacta con no deslizamiento es

\[
u(y)=\operatorname{sgn}(G)\frac{n}{n+1}\frac{1}{|G|K^{1/n}}
\left([|G|H-\tau_y]_+^q-[|G||y|-\tau_y]_+^q\right).
\]

Si $|G|H\leq\tau_y$, todo el material permanece sin ceder y $u=0$. El esfuerzo puede ser distinto de cero en esa región: el residuo constitutivo debe comprobar la **desigualdad de cedencia**, no imponer $\tau=0$. El límite $n=1$, $\tau_y=0$ recupera

\[
u(y)=\frac{G}{2K}(H^2-y^2).
\]

El caudal exacto por unidad de ancho, para $a=[|G|H-\tau_y]_+$ y $p=1/n$, es

\[
Q=\frac{2\operatorname{sgn}(G)}{|G|^2K^p}
\left(\frac{a^{p+2}}{p+2}+\frac{\tau_ya^{p+1}}{p+1}\right).
\]

El código suministra $Q$, esfuerzo, tasa de cortante y máscara del núcleo para contrastarlos con la integración numérica de la reconstrucción. Las superficies cedidas de estas referencias son exactas; el solver 2D regularizado documenta sensibilidad a Papanastasiou y al suavizado de baja cizalla, además de malla, paso y cuadratura.

### 2.2 Oldroyd-B: memoria y esfuerzo normal

Para $\boldsymbol u=(u(y,t),0)$, sin variación en $x$, las ecuaciones de Navier–Stokes acopladas con la derivada convectada superior de Oldroyd-B se reducen a

\[
\rho\partial_tu=\eta_s\partial_{yy}u+\partial_y\tau_{xy}+G(t),
\]
\[
\lambda\partial_t\tau_{xy}+\tau_{xy}=\eta_p\partial_yu,
\]
\[
\lambda\partial_t\tau_{xx}+\tau_{xx}=2\lambda\partial_yu\,\tau_{xy},
\qquad\tau_{yy}=0
\]

para condiciones iniciales con $\tau_{yy}=0$. $\eta_s$ y $\eta_p$ están en Pa·s y $\lambda$ en s. La ecuación del esfuerzo normal distingue esta solución de un modelo de viscosidad aparente. En régimen estacionario:

\[
u=\frac{G}{2(\eta_s+\eta_p)}(H^2-y^2),\quad
\tau_{xy}=\eta_p\partial_yu,\quad
\tau_{xx}=2\lambda\eta_p(\partial_yu)^2.
\]

Los límites $\lambda=0$ y $\eta_p=0$ son newtonianos y se comprueban de forma independiente. En el arranque newtoniano desde reposo se utiliza como referencia la serie

\[
u(y,t)=\frac{G}{2\eta}(H^2-y^2)
-\frac{16GH^2}{\eta\pi^3}\sum_{k=0}^{\infty}
\frac{(-1)^k}{(2k+1)^3}
\cos\left(\frac{(2k+1)\pi y}{2H}\right)
\exp\left[-\frac{\eta}{\rho}\left(\frac{(2k+1)\pi}{2H}\right)^2t\right].
\]

### 2.3 Dimensiones y parámetros de las estelas futuras

Para un cilindro de diámetro $D$ y velocidad de entrada $U$:

\[
Re=\frac{\rho UD}{\eta_s+\eta_p},\quad Wi=\frac{\lambda U}{D},\quad
\beta=\frac{\eta_s}{\eta_s+\eta_p}
\]

en Oldroyd-B/FENE-P. Para HB se usa la tasa de referencia $\dot\gamma_r=U/D$,

\[
\eta_r=K\dot\gamma_r^{n-1},\quad
Re_{HB}=\frac{\rho U^{2-n}D^n}{K},\quad
Bn_{HB}=\frac{\tau_y}{K(U/D)^n}.
\]

Para Saramito-HB, la definición del tiempo elástico y del número $Wi$ debe quedar ligada a las ecuaciones y a su módulo elástico. La opción $\lambda_r=\eta_r/G_{el}$ recupera el tiempo viscosidad/módulo para $n=1$; otras normalizaciones de la bibliografía deben traducirse explícitamente. El manifiesto documenta estas convenciones. FENE-P requiere además el parámetro de extensibilidad finita y su normalización.

## 3. Metodología computacional

### 3.1 Contrato y banco implementado

Todos los casos devuelven `velocity(T,2,ny,nx)`, `times`, `dx`, `dy`, `density`, `mask`, `boundary='channel'` y `metadata`. `mask=True` significa obstáculo; los canales no contienen obstáculos. Las paredes son nodos de frontera con velocidad cero; no se cuentan como un cilindro ni como celdas sólidas. El eje $x$ es periódico con `dx=length/nx`; el eje $y$ incluye ambos extremos y `dy=2H/(ny-1)`. `metadata.grid_convention='periodic_x_nodal_y'` evita confundir estas coordenadas con una cuadratura de centros de celda.

| API | Referencia y datos adicionales |
|---|---|
| `herschel_bulkley_profile(...)` | Perfil exacto para valores de $y$ suministrados |
| `herschel_bulkley_channel(...)` | Perfil, esfuerzo, tasa de cortante, núcleo y caudal exactos |
| `newtonian_startup_profile(...)` | Serie independiente del arranque newtoniano |
| `oldroyd_b_channel(...)` | Velocidad y esfuerzos cortante/normal transitorios acoplados |
| `channel_convergence_study(...)` | Refinamiento separado espacial, temporal o conjunto |
| `reference_rheology_bank(...)` | Ocho casos reproducibles con un número común de snapshots |

Los ocho casos son HB $n=0.6$, Bingham $n=1$, HB $n=1.4$, límite newtoniano analítico, HB totalmente sin ceder, dos arranques Oldroyd-B con $\lambda=0.1$ y $1$ s, y Oldroyd-B con $G(t)=\sin(2\pi t)$ Pa/m. La densidad por defecto es constante. Los nombres `low_wi` y `high_wi` ordenan los dos tiempos de relajación bajo la misma escala; los valores adimensionales efectivos están en los metadatos y no son los de una estela de cilindro.

La referencia totalmente sin ceder tiene RMS de velocidad cero. La contaminación definida como fracción de ese RMS también es cero. Para estudiar detección de falsas velocidades en ese caso se necesita un presupuesto **absoluto** de ruido en m/s, y reportarlo aparte; no se debe presentarlo como un experimento de igual intensidad relativa.

### 3.2 Discretización de Oldroyd-B

La velocidad se almacena en nodos y los esfuerzos en las caras entre nodos. Con $a=\lambda/(\lambda+\Delta t)$ y $b=\eta_p\Delta t/(\lambda+\Delta t)$, se elimina el esfuerzo en el paso implícito:

\[
\tau_{xy}^{m+1}=a\tau_{xy}^m+bD_yu^{m+1},
\]
\[
\left(\frac{\rho}{\Delta t}I-(\eta_s+b)L_y\right)u^{m+1}
=\frac{\rho}{\Delta t}u^m+aD_y\tau_{xy}^m+G^{m+1}.
\]

El sistema tridiagonal se resuelve con factorización reutilizada, sin depender de un solver CFD externo. Luego se actualiza $\tau_{xx}$ con Euler implícito en relajación y con $D_yu^{m+1}\tau_{xy}^{m+1}$ en la fuente. El esquema es de orden uno temporal y dos espacial para campos suaves. El esfuerzo nodal exportado se obtiene con interpolación de segundo orden y extrapolación lineal en paredes; los esfuerzos de cara permanecen disponibles para comprobar la discretización original.

Los residuos de momento y de las dos ecuaciones constitutivas se calculan en **cada paso interno**, incluso si solo se guardan snapshots cada varios pasos. Los residuos de cortante/normal tienen unidades de Pa y los de momento Pa/m. Un residuo cercano al redondeo verifica el sistema discreto, pero no equivale a error pequeño frente a la solución continua.

Para Oldroyd-B se comprueba también $\boldsymbol C=\boldsymbol I+(\lambda/\eta_p)\boldsymbol\tau$. En esta reducción $C_{yy}=1$, por lo que $\det C>0$ garantiza definitud positiva. El mínimo exportado se mide en caras en todos los pasos. El esfuerzo cortante tiene signo y **no** debe someterse a un supuesto de positividad del esfuerzo completo.

### 3.3 Uso reproducible

Desde la raíz del repositorio, con `PYTHONPATH=python/src` al ejecutar los ejemplos Python:

```python
from fluid_denoise.research.rheology import (
    reference_rheology_bank, channel_convergence_study,
)

bank = reference_rheology_bank(ny=65, nx=16, frames=31)
clean = bank["oldroyd_b_startup_high_wi"]["velocity"]
spatial = channel_convergence_study(refinement="space", end_time=.5)
temporal = channel_convergence_study(refinement="time", end_time=.5)
```

```bash
python -m unittest discover -s tests -p 'test_rheology.py' -v
```

El API permite $G(t)$ como función para comparar fase y memoria bajo otras frecuencias. No permite llamar viscoelástica a una ley HB ni representar una estela añadiendo una textura de vórtices a un perfil analítico.

## 4. Verificación calculada

Los catorce tests de `tests/test_rheology.py` pasan. Cubren el límite newtoniano exacto de HB, el núcleo Bingham, el límite completamente sin ceder, caudal por cuadratura independiente, velocidades y esfuerzos estacionarios Oldroyd-B, serie de arranque newtoniano, ausencia de polímero, relajación transitoria, forzamiento oscilatorio y refinamientos. Un test adicional de regresión verifica que parámetros finitos capaces de desbordar los esfuerzos se rechacen explícitamente: la velocidad finita no puede ocultar esfuerzos o residuos no finitos.

Las siguientes cifras pertenecen al baseline `baseline_2026-10-14`; no son evidencia regenerada del paquete integrado de fase 8. Las comprobaciones usan los parámetros indicados:

| Comprobación | Resultado |
|---|---:|
| Arranque newtoniano: $N_y=65$, $\Delta t=0.0005$ s, $t=0.2$ s; error relativo $L_2$ frente a serie | $6.30\times10^{-4}$ |
| Oldroyd-B inicializado en Poiseuille estacionario, $N_y=33$, 50 pasos | cambio máximo de velocidad $6.11\times10^{-16}$ m/s |
| Oldroyd-B desde reposo: $N_y=33$, $\Delta t=0.005$ s, $t=8$ s; error máximo de velocidad frente a Poiseuille | $1.44\times10^{-6}$ m/s |
| Mismo arranque a 8 s; diferencia máxima de $\tau_{xy}$ frente a $\eta_pu_y$ | $7.43\times10^{-7}$ Pa |
| Mismo arranque a 8 s; diferencia máxima de $\tau_{xx}$ frente a $2\lambda\eta_pu_y^2$ | $3.32\times10^{-6}$ Pa |

La comparación a tiempo finito de los últimos tres renglones incluye el transitorio físico restante, además del error numérico. Para el caso estacionario inicializado se prueba que el esquema mantiene la solución y los esfuerzos discretos correctos.

Un refinamiento conjunto del límite newtoniano a $t=0.2$ s produjo:

| $N_y$ | $\Delta t$ (s) | Error relativo $L_2$ frente a serie |
|---:|---:|---:|
| 33 | 0.020 | 0.019145 |
| 65 | 0.010 | 0.009544 |
| 129 | 0.005 | 0.004765 |

La reducción cercana a dos es compatible con que domine el error temporal de Euler implícito; ese refinamiento conjunto no permite medir aisladamente el orden espacial. Los tests incluyen refinamiento de espacio con paso fijo y de tiempo con malla fija para evitar esa confusión. En Oldroyd-B a $t=0.5$ s, con tres niveles y comparación con el más fino, los cambios relativos de velocidad fueron $4.79\times10^{-4}$ y $9.59\times10^{-5}$ en espacio, y $1.06\times10^{-2}$ y $3.56\times10^{-3}$ en tiempo. El nivel fino es una referencia numérica, no una solución exacta.

### 4.1 Validación que coincide con el piloto de reconstrucción

El archivo [`rheology_validation.json`](../../results/research/baseline_2026-10-14/solver_validation/rheology_validation.json) registra una comprobación con $H=1$ m, longitud periódica 2 m, $\rho=1$ kg/m³, $\eta_s=\eta_p=0.5$ Pa·s, $\lambda=0.5$ s, $G=1$ Pa/m, duración 3 s y $N_x=8$. Se comparan las mismas 31 observaciones ($t=0,0.1,\ldots,3$ s). El límite newtoniano se recalcula con $\lambda=0$ y se compara con la serie independiente; los esfuerzos poliméricos Oldroyd-B se comparan en los mismos nodos interpolados.

| $N_y$ | $\Delta t$ (s) | Error newtoniano $L_2$, todos los tiempos | Cambio Oldroyd-B de velocidad frente a nivel fino | Cambio Oldroyd-B de esfuerzo frente a nivel fino | Residuo máximo constitutivo de cortante (Pa) |
|---:|---:|---:|---:|---:|---:|
| 33 | 0.0100 | 0.0026634 | 0.0026836 | 0.0031067 | $3.77\times10^{-15}$ |
| 65 | 0.0050 | 0.0012956 | 0.0008722 | 0.0009596 | $1.08\times10^{-14}$ |
| 129 | 0.0025 | 0.0006387 | referencia numérica | referencia numérica | $3.45\times10^{-14}$ |

Los cambios relativos integran todos los tiempos guardados; no ocultan el arranque detrás del campo casi estacionario final. Las paredes cumplen exactamente no deslizamiento y el mínimo de $\det C$ en caras se mantiene en 1, dentro del redondeo. Los residuos discretos de momento permanecen por debajo de $6.5\times10^{-13}$ Pa/m. Esta evidencia proporciona una escala empírica independiente del error de discretización para contextualizar el error de reconstrucción. No es una cota rigurosa: diferencias de reconstrucción comparables con esa escala requieren repetir la comparación con referencias más refinadas. Tampoco valida estelas ni parámetros más extremos.

El mismo archivo documenta el caudal HB exacto y el error de cuadratura trapezoidal para $G=K=H=1$, $\tau_y=0.2$ y $n=0.6/1/1.4$. Al pasar de 33 a 129 nodos, el error relativo de caudal cae respectivamente de 0.0013883 a 0.0000867, de 0.0011181 a 0.0000692 y de 0.0010088 a 0.0000613. Los campos son analíticos; esas diferencias miden cuadratura, no error de un solver HB.

La evidencia se regenera en una carpeta nueva con `rheology_validation_report()`; el resultado solo contiene valores JSON finitos. El solver rechaza desbordamientos o conformación inadmisible durante la integración, antes de exportar una falsa validación.

```python
import json
from pathlib import Path
from fluid_denoise.research.rheology import rheology_validation_report

target = Path("results/research/solver_validation_local/rheology_validation.json")
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(rheology_validation_report(), indent=2, allow_nan=False) + "\n")
```

## 5. Campaña de estelas que amplía el trabajo

El manifiesto propone verificación constitutiva previa, estelas Oldroyd-B/FENE-P, estelas HB/Bingham, estelas elastoviscoplásticas Saramito-HB y, finalmente, estudio de transición con validación tridimensional. Los barridos incluyen $Re=100$, $Wi$ hasta 2, $\beta=0.5/0.85$, $n=0.6/1/1.4$, $Bn$ hasta 1 y extensibilidad FENE-P. Son parámetros de diseño, no casos ya calculados.

Basilisk proporciona `log-conform.h` para Oldroyd-B/FENE-P acoplados con Navier–Stokes; RheoTool es una alternativa de volúmenes finitos. No se ha instalado, compilado, validado ni implementado un adaptador de ninguno en esta extensión. La disponibilidad de HB o Saramito-HB en una versión concreta debe verificarse antes de elegir el solver. Cada corrida deberá fijar commit, ecuaciones, convención de esfuerzo, condiciones de frontera, malla y paso.

Se proponen 32/64/128 celdas por diámetro, refinamiento temporal independiente, estudio de dominio/salida y ventanas estadísticas suficientemente largas. Las condiciones de admisión comprueban balances de masa, convergencia de arrastre/sustentación/Strouhal, residuos constitutivos, disipación y admisibilidad de conformación. Para FENE-P se requiere $\operatorname{tr}C<L^2$ bajo la normalización elegida. La viscosidad aparente y las zonas cedidas necesitan convergencia; los esfuerzos de Oldroyd-B/EVP necesitan resolución y estabilidad constitutiva. Una tolerancia relativa sobre un observable cercano a cero debe sustituirse por una absoluta en unidades físicas.

Hamid, Sasmal y Chhabra ya estudiaron DMD en estelas Oldroyd-B a $Re=100$. Raffi y colaboradores aplicaron DMD a estelas elastoviscoplásticas. Corrochano y colaboradores aplicaron POD/HODMD a estelas Saramito-HB; su preprint informa 512 núcleos durante 192 h **por simulación**, con resolución de referencia $D/64$. Esta evidencia sitúa la línea propuesta dentro de una literatura existente y demuestra que un canal 1D no sustituye esa campaña. El dato de recursos es un contexto publicado, no una estimación del costo de este repositorio.

Una señal irregular o varios autovalores DMD no prueban caos. Se necesita descartar transitorios, comprobar ventanas independientes, espectros/recurrencia y sensibilidad de estimaciones de Lyapunov; para afirmar un comportamiento físico a $Re$ mayores también se requiere validar la hipótesis 2D frente a dinámica 3D. La campaña debe separar inestabilidad numérica, cedencia, memoria elástica e inestabilidad hidrodinámica.

## 6. Referencias primarias

- Herschel, W. H. y Bulkley, R. (1926), *Konsistenzmessungen von Gummi-Benzollösungen*. [DOI: 10.1007/BF01432034](https://doi.org/10.1007/BF01432034).
- Oldroyd, J. G. (1950), *On the formulation of rheological equations of state*. [DOI: 10.1098/rspa.1950.0035](https://doi.org/10.1098/rspa.1950.0035).
- Bird, R. B., Dotson, P. J. y Johnson, N. L. (1980), *Polymer solution rheology based on a finitely extensible bead–spring chain model*. [DOI: 10.1016/0377-0257(80)85007-5](https://doi.org/10.1016/0377-0257(80)85007-5).
- Papanastasiou, T. C. (1987), *Flows of Materials with Yield*. [DOI: 10.1122/1.549926](https://doi.org/10.1122/1.549926).
- Fattal, R. y Kupferman, R. (2004), formulación log-conformation. [DOI: 10.1016/j.jnnfm.2004.08.008](https://doi.org/10.1016/j.jnnfm.2004.08.008).
- Saramito, P. (2007), *A new constitutive equation for elastoviscoplastic fluid flows*. [DOI: 10.1016/j.jnnfm.2007.04.004](https://doi.org/10.1016/j.jnnfm.2007.04.004).
- Saramito, P. (2009), *A new elastoviscoplastic model based on the Herschel–Bulkley viscoplastic model*. [DOI: 10.1016/j.jnnfm.2008.12.001](https://doi.org/10.1016/j.jnnfm.2008.12.001); [PDF primario](https://web.mit.edu/nnf/education/Summer2009/Saramito2.pdf).
- Pimenta, F. y Alves, M. A. (2017), *Stabilization of an open-source finite-volume solver for viscoelastic fluid flows*. [DOI: 10.1016/j.jnnfm.2016.12.002](https://doi.org/10.1016/j.jnnfm.2016.12.002); [RheoTool](https://github.com/fppimenta/rheoTool).
- Hamid, F., Sasmal, C. y Chhabra, R. P. (2022), DMD de estelas Oldroyd-B. [DOI: 10.1063/5.0122103](https://doi.org/10.1063/5.0122103).
- Raffi, Chauhan, Hamid y Sasmal (2024), DMD de estelas elastoviscoplásticas. [DOI: 10.1063/5.0224004](https://doi.org/10.1063/5.0224004).
- Corrochano y colaboradores (2026), POD/HODMD de estelas EVP. [DOI: 10.1007/s00162-026-00775-3](https://doi.org/10.1007/s00162-026-00775-3); [preprint primario](https://arxiv.org/html/2508.03362v1).
- [Basilisk: código y documentación de log-conformation](https://basilisk.fr/src/log-conform.h).
