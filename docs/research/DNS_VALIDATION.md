# Validación de Navier–Stokes periódico y referencias analíticas

Los ensayos de `baseline_2026-10-14` caracterizan la formulación que implementa [dns.py](../../python/src/fluid_denoise/research/dns.py) y cuantifican la sensibilidad a malla y paso temporal. Las cifras completas están en [dns_validation.json](../../results/research/baseline_2026-10-14/solver_validation/dns_validation.json). Los parámetros de los dos ensayos no lineales coinciden con el piloto: viscosidad cinemática `0.02`, velocidad RMS inicial `1`, semilla `19`, dominio `[0, 2π)²` y tiempo final `1.5`, todos adimensionales.

**Conclusión operativa:** la malla `24²` genera referencias subresueltas para este piloto. Se recomienda `64²` para turbulencia decadente y `48²` para vórtices interactuantes: ambas satisfacen una tolerancia propuesta de **1% en los errores L2 de velocidad y vorticidad en las 31 instantáneas del piloto**, frente a una referencia numérica `128²`. El paso de las mallas seleccionadas es `0.01` y el máximo nominal de la referencia es `0.005`; se guardan tiempos `0, 0.05, …, 1.5`. Esa evidencia no certifica automáticamente DNS, convergencia de todos los tiempos continuos, otros observables, parámetros o geometrías. El archivo de resultados conserva `dns_certified: false`.

## Formulación y condiciones iniciales

Se resuelve directamente la ecuación de vorticidad de Navier–Stokes incompresible bidimensional:

\[
\partial_t\omega + u\partial_x\omega + v\partial_y\omega
= \nu\nabla^2\omega,
\qquad \omega=\partial_x v-\partial_y u.
\]

La función de corriente cumple `ω = -∇²ψ`, con `u = ∂yψ` y `v = -∂xψ`. La inversión de Laplace por FFT fija a cero la velocidad media; la densidad es constante e igual a uno. No se incorporan modelos de turbulencia. La discretización emplea FFT, productos no lineales en espacio físico, corte estricto de dos tercios por componente y RK4. El corte estricto elimina también los modos situados exactamente en `N/3`, cuando existen.

El paso interno nunca excede el nominal ni los límites:

\[
\Delta t\,\max\left(\frac{|u|}{\Delta x}+\frac{|v|}{\Delta y}\right)\le 0.35,
\qquad \nu\Delta t\,k_{\max}^2\le2.
\]

El último límite permanece dentro de la región estable de RK4 para difusión. Se comprueba finitud en cada etapa y existe un presupuesto máximo de subpasos. Las instantáneas conservan los tiempos físicos solicitados, aunque la integración utilice subpasos adicionales. El metadato `stability_limited_steps` distingue estos subpasos de los necesarios para alcanzar una salida.

Los casos disponibles son:

- `taylor_green`: un modo de función de corriente, con velocidad y presión analíticas.
- `interacting_vortices`: cuatro núcleos gaussianos periodizados, dos positivos y dos negativos; ancho `0.065 min(Lx,Ly)`. La suma de imágenes periódicas evita discontinuidades en las fronteras. Se normaliza la velocidad inicial a RMS uno.
- `decaying_turbulence`: una banda de coeficientes de vorticidad con máximo alrededor del número de onda entero cuatro y corte radial ocho. Las fases y amplitudes dependen únicamente de la semilla, sin depender de la malla; cada malla proyecta sobre sus modos retenidos. Mallas `≥28` representan todos los modos iniciales. La malla `24²` elimina algunos modos iniciales y después genera un error dinámico mucho mayor.

El solver periódico complementa el caso de cilindro LBM del banco. No reproduce la estela de un cilindro ni una calle de Kármán, y no incorpora paredes o fluidos no newtonianos. Comparar directamente sus resultados con el cilindro LBM cambiaría simultáneamente geometría, condiciones de contorno y método. La comparación de algoritmos de reconstrucción debe utilizar exactamente la misma referencia limpia dentro de cada caso.

## Referencia exacta de Taylor–Green

Para `kx=2π/Lx`, `ky=2π/Ly` y `q=kx²+ky²`, la referencia rectangular es:

\[
u=A\sin(k_x x)\cos(k_y y)e^{-\nu q t},
\qquad
v=-A\frac{k_x}{k_y}\cos(k_x x)\sin(k_y y)e^{-\nu q t},
\]

\[
p=\frac{A^2}{4}\left[\cos(2k_xx)+\frac{k_x^2}{k_y^2}\cos(2k_yy)\right]e^{-2\nu q t}.
\]

La presión tiene media espacial cero. La energía cinética media exacta es

\[
E(t)=\frac{A^2}{8}\left(1+\frac{k_x^2}{k_y^2}\right)e^{-2\nu q t}.
\]

La validación temporal usa `ν=0.2`, `A=0.1`, `16²` y tiempo final uno para medir errores por encima del redondeo y sin activar límites de estabilidad. Estos parámetros son deliberadamente distintos de los ensayos no lineales.

| Δt | Error relativo de velocidad | Error relativo de energía | Orden observado |
|---:|---:|---:|---:|
| 0.2 | 1.4596e-7 | 2.9191e-7 | — |
| 0.1 | 8.8227e-9 | 1.7645e-8 | 4.048 |
| 0.05 | 5.4230e-10 | 1.0846e-9 | 4.024 |

Las mallas `8²`, `12²` y `16²` reproducen el modo espacial; la diferencia permanece en torno al error temporal `5.423e-10`. La divergencia RMS máxima es `5.55e-17`. Esto es una prueba de **representación de un modo**, no una demostración de convergencia espacial de la dinámica no lineal. Las pruebas verifican además que la velocidad y presión rectangulares satisfacen la ecuación de cantidad de movimiento, evitando una comparación exclusiva entre dos implementaciones de la misma fórmula.

## Refinamiento no lineal con parámetros del piloto

La referencia más fina es `128²` con paso máximo `0.005`. Los errores de la tabla incluyen todo el campo fino: se prolonga espectralmente la velocidad de la malla gruesa y se compara sobre la malla fina. De esta forma, los modos que la malla gruesa no puede representar contribuyen al error.

| Malla | Error final completo: turbulencia decadente | Error final completo: vórtices interactuantes |
|---:|---:|---:|
| 24² | 10.5159% | 1.45837% |
| 32² | 3.13029% | 0.350794% |
| 48² | 0.447526% | 0.049296% |
| 64² | 0.044080% | 0.004683% |
| 96² | 0.000881% | 0.0000655% |

El validador del módulo devuelve también el error frente a la referencia restringida a los modos compartidos, útil para identificar el error de sus coeficientes. Ese error no incluye modos finos ausentes y, por tanto, no debe confundirse con la columna anterior. El JSON almacena ambas medidas en el estudio extendido.

Si `f` es la fracción de energía de la referencia en los modos de la malla gruesa y `e_low` el error relativo restringido, Parseval da:

\[
e_{full}^2 = f\,e_{low}^2 + (1-f).
\]

Se verificó esta identidad mediante prolongación y comparación directa para `48²` en ambos casos, con discrepancia inferior a `1e-12`.

La turbulencia decadente disipa energía de `0.5` a `0.24091208` en la referencia `128²`; los vórtices interactuantes, de `0.5` a `0.43392121`. La divergencia RMS permanece por debajo de `8.48e-15`. Estas pruebas respaldan la incompresibilidad y el comportamiento disipativo; una divergencia pequeña por sí sola no establece resolución espacial.

La referencia `128²` de vórtices interactuantes activó el límite CFL en 91 subpasos. El máximo sigue siendo `0.005`. La integración utiliza esos pasos menores y el estudio documenta el ajuste, en lugar de interpretar todos los pasos nominales como pasos efectivos idénticos.

### Separación entre error temporal y espacial

Los estudios `24²`, `32²`, `48²` frente a `64²` se ejecutaron con pasos `0.01` y `0.005`. El cambio temporal apenas modifica el error espacial final. En `24²`, por ejemplo, el error restringido de turbulencia decadente es `9.7229704%` y `9.7229702%`; reducir únicamente el paso temporal no resuelve la insuficiencia espacial.

Se añadió un refinamiento temporal independiente para cada malla `24²`, `32²`, `48²` y `64²`, usando pasos `0.01`, `0.005`, `0.0025` y referencia temporal `0.00125` sobre la misma malla. Para la malla `48²`, evaluada inicialmente como candidata para ambos casos:

| Caso | Error temporal Δt=0.01 | Error temporal Δt=0.005 | Error temporal Δt=0.0025 | Orden 0.01→0.005 |
|---|---:|---:|---:|---:|
| Turbulencia decadente | 8.5121e-9 | 5.2551e-10 | 3.0783e-11 | 4.018 |
| Vórtices interactuantes | 2.7763e-8 | 1.7235e-9 | 1.0123e-10 | 4.010 |

No se informa un orden observado cuando CFL o difusión redujeron alguno de los pasos nominales comparados. Este es el caso de `64²`, `Δt=0.01`, para vórtices interactuantes. Un orden calculado con esos pasos nominales no representaría una comparación uniforme de RK4.

El pequeño error temporal sobre una malla fija no demuestra exactitud espacial. Para aceptar una referencia de vorticidad, enstrofía, balance energético o otros observables derivados deben fijarse y verificarse tolerancias propias: derivar espacialmente amplifica errores en números de onda altos. Además deben examinarse múltiples tiempos, semillas, viscosidades y regímenes. La referencia `128²` sigue siendo numérica y producida por el mismo solver; una validación independiente del régimen no lineal aportaría evidencia adicional.

### Vorticidad y selección de mallas

Se calculó explícitamente la vorticidad espectral de los campos prolongados y de la referencia fina. El error de velocidad no sirve como sustituto de este observable:

| Caso y malla | Error final completo de velocidad | Error final completo de vorticidad | Criterio propuesto u y ω < 1% |
|---|---:|---:|---|
| Turbulencia, 48² | 0.447526% | 1.692872% | No satisface vorticidad |
| Turbulencia, 64² | 0.044080% | 0.248000% | Satisface |
| Vórtices, 48² | 0.049296% | 0.566030% | Satisface |
| Vórtices, 64² | 0.004683% | 0.075031% | Satisface |

Por tanto, los pilotos de los dos casos deben usar mallas diferentes si se pretende controlar conjuntamente velocidad y vorticidad con esta tolerancia. Los porcentajes expresan diferencias respecto a una referencia numérica, no errores demostrados frente a la solución continua exacta del régimen no lineal.

Se verificó después toda la trayectoria guardada del piloto, con las mallas seleccionadas a `Δt=0.01` y referencia `128²` a paso máximo nominal `0.005`. Los máximos no se encuentran en el tiempo final:

| Caso y malla seleccionada | Máximo error completo de velocidad | Máximo error completo de vorticidad | Tiempo del máximo |
|---|---:|---:|---:|
| Turbulencia, 64² | 0.084185% | 0.517661% | 0.75 |
| Vórtices, 48² | 0.053693% | 0.619818% | 1.3 |

Ambos satisfacen el criterio propuesto en las 31 instantáneas guardadas. La diferencia frente al estudio a `Δt=0.005` es pequeña, consistente con el refinamiento temporal anterior. Esta verificación muestra por qué controlar únicamente el error en el último tiempo puede pasar por alto un máximo intermedio.

## Auditorías independientes de dinámica no lineal

Se usó una condición inicial que no es el modo estacionario de Euler de Taylor–Green:

\[
\omega_0=\sin x\sin y+0.35\cos(2x)\sin(3y)+0.1\sin(5x+2y).
\]

Con `32²`, `ν=0`, paso `0.001` y tiempo final `0.4`, los cambios relativos de energía y enstrofía son `1.96e-15` y `2.33e-15`. La enstrofía es `Z=⟨ω²⟩/2`. En el caso viscoso periódico se verifican las identidades:

\[
\frac{dE}{dt}=-\nu\langle\omega^2\rangle,
\qquad
\frac{dZ}{dt}=-\nu\langle|\nabla\omega|^2\rangle.
\]

Con `ν=0.02` y diferencia temporal centrada de paso `0.0001`, los defectos relativos de estas identidades en `t=0.0001` son `9.69e-11` y `6.47e-10`. Conservación o disipación por sí solas no detectan necesariamente un signo incorrecto de advección, por lo que se incluye una comprobación independiente de advección.

Para la función de corriente inicial `ψ=0.6sin(x)sin(y)+0.03cos(2x)sin(3y)`, se compara la derivada inicial estimada `(-3ω0+4ω1-ω2)/(2Δt)` con la expresión analítica de `-uωx-vωy+ν∇²ω`. Al reducir `Δt` de `0.001` a `0.0005` y `0.00025`, los errores relativos son `7.756e-7`, `1.939e-7` y `4.849e-8`. Los órdenes observados `1.9997` y `1.9998` corresponden al stencil temporal de segundo orden utilizado para esta comprobación, no al integrador RK4. Esta comprobación sí contrasta el signo y magnitud de la advección no lineal contra una derivación independiente.

## Reproducción

Las cifras anteriores describen el baseline. Sus hashes identifican aquella ejecución; para comprobar la integración actual, ejecutar las pruebas y guardar nueva evidencia con sus propias fuentes. Las pruebas se ejecutan desde la raíz con:

```bash
python -m pytest -q tests/test_dns.py
```

Las pruebas cubren referencia rectangular y ecuación de cantidad de movimiento, orden temporal RK4, incompresibilidad, disipación, adaptación CFL, tiempos de salida, reproducibilidad, refinamiento no lineal y fallos explícitos ante entradas inválidas o desbordamiento. El banco completo de tablas usa estas funciones públicas:

```python
from fluid_denoise.research.dns import (
    validate_taylor_green,
    validate_periodic_resolution,
    validate_periodic_time_refinement,
)

analytical = validate_taylor_green()
for case in ("decaying_turbulence", "interacting_vortices"):
    for dt in (0.01, 0.005):
        spatial = validate_periodic_resolution(
            case=case, grids=(24, 32, 48), reference_nx=64,
            viscosity=0.02, seed=19, duration=1.5, dt=dt,
        )
    extended = validate_periodic_resolution(
        case=case, grids=(24, 32, 48, 64, 96), reference_nx=128,
        viscosity=0.02, seed=19, duration=1.5, dt=0.005,
    )
    for nx in (24, 32, 48, 64):
        temporal = validate_periodic_time_refinement(
            case=case, nx=nx, viscosity=0.02, seed=19, duration=1.5,
            time_steps=(0.01, 0.005, 0.0025), reference_dt=0.00125,
        )
```

Para reproducir el error de campo completo, se usan los campos de `simulate_periodic_ns`, se prolonga con `scipy.signal.resample` en ambos ejes espaciales y se evalúa `norm(u_coarse_prolonged-u_reference)/norm(u_reference)`. El JSON conserva el criterio de prolongación y la verificación por Parseval. No sustituir esta medida por el error restringido al tomar decisiones sobre la tolerancia física del piloto.
