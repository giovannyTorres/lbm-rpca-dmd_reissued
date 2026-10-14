# Flujo viscoplástico periódico 2D

`fluid_denoise.research.viscoplastic_dns` integra un fluido generalizado
newtoniano **inelástico**. La velocidad procede de la ecuación de momentum;
no se obtiene multiplicando una trayectoria newtoniana por una curva escalar.
No representa un cilindro, paredes, elasticidad ni una demostración de caos.

## Ecuaciones y parámetros

En el toro rectangular, con densidad constante positiva:

\[
\partial_t\mathbf u+(\mathbf u\cdot\nabla)\mathbf u
 =-\frac{\nabla p}{\rho}+\frac{\nabla\cdot\boldsymbol\tau}{\rho},
\quad\nabla\cdot\mathbf u=0,
\quad\boldsymbol\tau=2\eta_{\rm app}\mathbf D,
\quad\mathbf D=\tfrac12(\nabla\mathbf u+\nabla\mathbf u^T).
\]

Se usa \(\dot\gamma=\sqrt{2\mathbf D:\mathbf D}\) y

\[
\eta_{\rm app}=\eta_s
 +K(\dot\gamma^2+\epsilon^2)^{(n-1)/2}
 +\tau_y\frac{1-e^{-m\dot\gamma}}{\dot\gamma}.
\]

El término de cedencia en \(\dot\gamma=0\) se evalúa por su límite
\(\tau_y m\), usando `expm1` cerca de cero. No hay un corte artificial de
viscosidad. \(\epsilon>0\) regulariza la potencia para \(n<1\); modifica
el modelo y debe declararse. El valor predeterminado es **0.001**, fijado
antes de las corridas. \(m\) y \(\epsilon\) requieren estudios separados.
\(n=1\) da Bingham regularizado; \(\tau_y=0,n=1\) da exactamente un
fluido newtoniano de viscosidad dinámica \(K+\eta_s\) y viscosidad cinemática
\((K+\eta_s)/\rho\). Se permite \(K=0\), pero la regularización sigue sin
crear regiones rígidas exactamente no cedidas.

Todos los argumentos son **adimensionales**. Para recuperar unidades se usan
las escalas \(L,U,\rho_{ref}\): tiempo \(L/U\), esfuerzo
\(\rho_{ref}U^2\), corte \(U/L\), viscosidad dinámica \(\rho_{ref}UL\),
consistencia \(\rho_{ref}U^2/(U/L)^n\), \(m\) en \(L/U\), y
\(\epsilon\) en \(U/L\). Las escalas aparecen también en los metadatos;
sin elegir valores físicos no se pueden atribuir Pa, segundos ni caudales
experimentales a estos campos.

## Discretización y estabilidad

La proyección FFT elimina la componente longitudinal de momentum en cada
etapa RK4. El esfuerzo completo se deriva espacialmente, incluyendo la
variación de \(\eta_{app}\); reemplazarlo por
\(\eta_{app}\nabla^2\mathbf u\) sería otra ecuación. La presión recuperada
tiene media cero. El estado conserva solamente modos con
\(|k_x^{integer}|<N_x/3\), \(|k_y^{integer}|<N_y/3\).

La advección y el esfuerzo se calculan por defecto sobre una malla de
cuadratura 1.5 veces mayor y se restringen a los modos del estado. Esta
operación elimina aliasing de los productos cuadráticos retenidos, pero
**no desaliasa exactamente una ley constitutiva no polinomial**. El reporte
incluye refinamiento de cuadratura a 2 y 3 veces la malla. La resolución del
estado y la resolución de cuadratura son controles diferentes.

Los tiempos de salida son los solicitados, incluido el instante final. Los
subpasos satisfacen las cotas

\[
\Delta t\max_{x,y}(|u_x|/\Delta x+|u_y|/\Delta y)\le C_{adv},
\quad
\Delta t\max(1,n)\max\eta_{app}\,k^2_{max}/\rho\le C_{diff}.
\]

Por defecto \(C_{adv}=0.35\), \(C_{diff}=0.4\). El factor
\(\max(1,n)\) acota la viscosidad tangente de la potencia; el término
Papanastasiou tiene tangente no negativa menor o igual a su viscosidad
secante. Se verifica la cota sobre todas las etapas, no solamente al inicio
del paso. Una etapa no finita, una violación de estabilidad o un aumento de
energía provoca rechazo y reducción a la mitad. Existen presupuestos
explícitos para pasos internos y rechazos; el subdesbordamiento produce un
error. Este mecanismo adapta **estabilidad**, no un estimador de error local;
el refinamiento temporal sigue siendo necesario.

En el sistema sin forzamiento se verifica

\[
E=\frac{\rho}{2}\langle|\mathbf u|^2\rangle,
\quad \mathcal D=\langle\eta_{app}\dot\gamma^2\rangle\ge0,
\quad E(t)-E(0)+\int_0^t\mathcal D\,dt=0.
\]

La disipación se calcula en la cuadratura constitutiva y se integra con los
pesos de las etapas RK4. El residuo de energía se devuelve, no se fuerza a
cero. Los campos espaciales de esfuerzo/corte/viscosidad se evalúan en la
malla de salida; promediar esos campos produce una cuadratura diferente y
puede diferir del valor `dissipation_rate`.

## API y contrato

```python
from fluid_denoise.research.viscoplastic_dns import (
    simulate_viscoplastic_periodic, validation_report,
)

run = simulate_viscoplastic_periodic(
    case="nonlinear_modes", nx=32, consistency=0.02, flow_index=0.6,
    yield_stress=0.02, regularization_m=10, shear_epsilon=0.001,
    dt=0.005, n_steps=400, sample_every=10,
)
```

| Clave | Forma / significado |
|---|---|
| `velocity` | `(T,2,Y,X)`, componentes `(u,v)` |
| `times` | `(T,)`, tiempo inicial y final exactos |
| `density` | `(T,Y,X)`, densidad constante |
| `mask` | `(Y,X)`, booleano; `True` significaría obstáculo, aquí todo `False` |
| `dx`, `dy`, `boundary` | espaciamientos positivos y `"periodic"` |
| `pressure` | `(T,Y,X)`, presión de media cero |
| `shear_rate`, `apparent_viscosity` | `(T,Y,X)` |
| `deviatoric_stress` | `(T,3,Y,X)`, orden `(xx,xy,yy)` |
| `stress_invariant` | `(T,Y,X)`, \(\sqrt{\tau:\tau/2}=\eta_{app}\dot\gamma\) |
| `approximate_yielded` | `(T,Y,X)`, `shear_rate > yield_shear_threshold` |
| `over_yield_stress` | `(T,Y,X)`, `stress_invariant > yield_stress` |
| `kinetic_energy`, `dissipation_rate` | `(T,)` |
| `cumulative_dissipation`, `energy_budget_residual` | `(T,)` |
| `metadata` | parámetros, unidades, condiciones iniciales, estabilidad y límites |

`approximate_yielded` usa por defecto corte 0.001. Es una **estimación por
umbral declarado**, no una solución de complementariedad ni un clasificador
de regiones exactamente no cedidas. `over_yield_stress` es otro diagnóstico,
con otra definición; los dos no deben intercambiarse. La ley regularizada
permite deformación aun debajo del esfuerzo nominal de cedencia.

Se admiten Taylor–Green, shear periódico, vórtices interactuantes, turbulencia
decayente con semilla, y `nonlinear_modes`. Este último combina tres modos
de función de corriente con distintos autovalores del Laplaciano. Su
advección y transporte variable de esfuerzo producen cambios espaciales que
no se reducen a reescalar el campo inicial. También se permite una velocidad
inicial explícita, filtrada y proyectada.

## Validación y alcance

`validation_report()` devuelve un objeto serializable a JSON. El reporte
ejecutado se guarda en
`results/research/solver_validation_2d/viscoplastic.json`:

- Límites analíticos newtonianos Taylor–Green y shear periódico.
- Corridas de momentum con \(n=0.6,1,1.4\) y \(\tau_y>0\), balance de
  energía, incompresibilidad y residuo frente a un reescalamiento escalar.
- Refinamiento de malla con \(m\) fijo y referencia numérica más fina.
- Refinamiento temporal en la misma malla, indicando subpasos realmente usados.
- Refinamiento de \(m\) en la misma malla: cambia el modelo regularizado y
  no constituye evidencia de convergencia espacial.
- Sensibilidad a \(\epsilon\) con \(m\) fijo y refinamiento de cuadratura
  constitutiva independiente.

La corrida de verificación usa duración 0.08, malla 24 y paso nominal 0.004
para los límites y la sensibilidad constitutiva. Los errores relativos de
velocidad de los dos límites analíticos son inferiores a
\(4\times10^{-15}\). Para los tres índices de flujo, el residuo relativo del
balance energético es inferior a \(1.4\times10^{-14}\), la divergencia RMS
es inferior a \(1.2\times10^{-15}\), y la evolución final difiere de un
reescalamiento escalar del estado inicial en aproximadamente 1.5%.

El refinamiento espacial 16/32/48 frente a referencia 64 reduce el error
final \(1.36\times10^{-4}\to1.21\times10^{-4}\to8.01\times10^{-5}\).
La mejora es modesta: no demuestra una solución espacialmente convergida.
La malla 48 requiere subpasos menores por estabilidad; los pasos usados
aparecen en el JSON. El estudio temporal separado 0.004/0.002 frente a 0.001
reduce el error \(1.09\times10^{-12}\to6.49\times10^{-14}\).

En malla fija, \(m=4,8,16\) frente a 32 reduce el error
\(1.09\times10^{-3}\to4.24\times10^{-4}\to1.09\times10^{-4}\).
Con \(m=8\) fijo, \(\epsilon=0.01,0.001,0.0001\) frente a 0.00001 reduce
el error \(1.34\times10^{-6}\to1.47\times10^{-8}\to1.46\times10^{-10}\).
La viscosidad de corte cero aumenta al reducir \(\epsilon\), conforme a
la ley, sin recortarse. La cuadratura 1.5/2 frente a 3 veces la malla da
\(5.16\times10^{-5}\to2.54\times10^{-5}\), evidencia de que su error
todavía afecta los campos.

También se ejecutaron casos 32×32 de duración 2, paso nominal 0.005 y 41
salidas para \(n=0.6,K=0.02,\tau_y=0.02,m=10,\epsilon=0.001\).
`nonlinear_modes`, `interacting_vortices` y `periodic_shear` utilizaron
406, 401 y 520 pasos internos, respectivamente. Los cambios frente al mejor
reescalamiento del campo inicial fueron 52.9%, 81.0% y 5.05%. La energía
disminuyó y el residuo relativo máximo fue \(5.4\times10^{-11}\).

Una referencia del mismo solver no es verdad analítica. Un residuo pequeño
de energía o de divergencia tampoco certifica resolución de pequeñas escalas.
El indicador `dns_certified` permanece `False`; los resultados describen
casos periódicos sintéticos, sin afirmaciones sobre wake, turbulencia
completamente resuelta, cedencia ideal, elasticidad o caos no lineal.
