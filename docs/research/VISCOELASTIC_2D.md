# Flujos viscoelásticos periódicos 2D: Oldroyd-B y FENE-P

`python/src/fluid_denoise/research/viscoelastic_dns.py` integra velocidad incomprensible y memoria polimérica en dos dimensiones espaciales. El tensor de conformación evoluciona y su esfuerzo realimenta la ecuación de movimiento. Los casos ejecutados son vórtices periódicos con parámetros moderados; los refinamientos siguientes respaldan esas corridas concretas. No validan estelas, cilindros, turbulencia elástica, caos ni regímenes de alto Weissenberg.

## Ecuaciones y normalización

Se emplean variables adimensionales, densidad constante uno y un dominio rectangular periódico. La presión impone la incomprensibilidad mediante la proyección de Leray en Fourier:

\[
\partial_t\boldsymbol u+\boldsymbol u\cdot\nabla\boldsymbol u
=-\nabla p+\nu_s\Delta\boldsymbol u+\nabla\cdot\boldsymbol\tau_p+\boldsymbol F,
\qquad\nabla\cdot\boldsymbol u=0,
\]

\[
\partial_t C+\boldsymbol u\cdot\nabla C
=GC+CG^T-\frac{f(C)C-I}{\lambda}+\kappa\Delta C,
\qquad G_{ij}=\partial_j u_i,
\qquad\tau_p=\frac{\nu_p}{\lambda}\bigl[f(C)C-I\bigr].
\]

En Oldroyd-B, \(f=1\). En FENE-P:

\[
f(C)=\frac{L^2-d}{L^2-\operatorname{tr}C},\quad
C\succ0,\quad\operatorname{tr}C<L^2.
\]

`extensibility_squared` representa \(L^2\), y debe exceder la dimensión constitutiva \(d\). Esta normalización hace que \(C=I\) sea equilibrio con esfuerzo cero. No se mezcla con la normalización \(L^2/(L^2-\operatorname{tr}C)\) sin transformar sus parámetros.

Por defecto \(d=3\): la velocidad y las coordenadas son planares, pero la conformación del polímero es un tensor \(3\times3\). Se incluyen \(C_{zz}\), la traza completa y el esfuerzo \(\tau_{zz}\). En Oldroyd-B con \(C(0)=I\), \(C_{zz}=1\); en FENE-P, el factor de Peterlin también modifica este componente. La divergencia del esfuerzo que aparece en momento usa los dos componentes espaciales. `tensor_dimension=2` permite estudiar explícitamente la variante constitutiva bidimensional, con \(d=2\) en su normalización.

\(\kappa=0\) por defecto. `stress_diffusivity>0` agrega difusión artificial **de conformación**; no se introduce de manera oculta. Se registra su valor, su contribución disipativa y un estudio de sensibilidad con \(0,\kappa/2,\kappa\) cuando se usa en `validation_report`. No equivale a cambiar la viscosidad solvente ni a estabilización log-conformation.

## Discretización y admisibilidad

La malla excluye el extremo del período. Se reutilizan las condiciones iniciales de `research/dns.py`: Taylor–Green, cuatro núcleos gaussianos periodizados y una banda de vorticidad con semilla fija. Los campos de velocidad son solenoidales desde el estado inicial; la condición polimérica inicial es \(C=I\), salvo suministro explícito.

La advección, el estiramiento y el esfuerzo se evalúan con colocación de Fourier. La proyección solenoidal se aplica a momento y al estado aceptado. Un corte estricto componente a componente de dos tercios elimina el aliasing de los productos cuadráticos conservados. El factor racional de FENE-P se evalúa en los puntos de colocación y se filtra: el corte no elimina exactamente todos los armónicos de una función racional. Su error se controla con el refinamiento espacial y el residuo de energía medido.

`rk_order=4` es la opción por defecto; también se ofrecen RK2 y SSP-RK3. El paso interno se limita por el paso nominal, CFL advectivo, difusión solvente o artificial, relajación \(0.2\lambda/\max(1,f)\), tasa de estiramiento y una escala de propagación elástica \(\sqrt{\nu_p f\lambda_{\max}(C)/\lambda}\). Estos límites son criterios conservadores de integración, no una prueba de estabilidad general del sistema no lineal.

Cada etapa de Runge–Kutta y cada estado aceptado se verifica: valores finitos, simetría, autovalor mínimo mayor que `eigenvalue_floor` y, en FENE-P, traza estrictamente menor que \(L^2\). Una etapa inválida se descarta y se intenta de nuevo con la mitad del paso. **No se recorta la traza, no se reemplazan autovalores y no se repara silenciosamente \(C\)**. Si el paso deja de representar avance temporal o se agota el límite de rechazos, se lanza una excepción. Si la resolución espacial es insuficiente, dividir el paso puede no resolver el problema.

La admisibilidad se verifica en los puntos de la malla; no se certifica positividad entre puntos de colocación. `rejected_steps`, límites realmente alcanzados y extremos de las etapas se conservan en los metadatos.

## API y datos conservados

```python
from fluid_denoise.research.viscoelastic_dns import simulate_viscoelastic_periodic

run = simulate_viscoelastic_periodic(
    model="fene_p", case="interacting_vortices", nx=48,
    viscosity_s=.02, viscosity_p=.04, relaxation_time=.5,
    extensibility_squared=12., amplitude=.6,
    dt=.0025, n_steps=240, sample_every=8,
    stress_diffusivity=0.,
)
```

El diccionario incluye:

| Campo | Forma o significado |
|---|---|
| `velocity` | \((T,2,Y,X)\), componentes \(u_x,u_y\) |
| `times` | Tiempos nominales de muestreo; incluye inicio y final exacto |
| `dx`, `dy` | Espaciamientos de la malla |
| `density` | \((T,Y,X)\), densidad uno |
| `mask` | \((Y,X)\), booleano; `True` sería obstáculo; aquí todo `False` |
| `boundary` | `"periodic"` |
| `conformation`, `stress` | \((T,d,d,Y,X)\), todos los componentes del tensor |
| `constitutive_rhs` | \((T,d,d,Y,X)\), lado derecho constitutivo proyectado en cada instantánea |
| `metadata` | Modelo, parámetros, fuerzas, resolución, pasos internos, admisibilidad y balances |

Se pueden añadir fuerzas `forcing="kolmogorov"` o `forcing="cellular"`, con amplitud y modo explícitos. No se activa una fuerza sin solicitarla. El campo celular es solenoidal también en dominios rectangulares.

Los metadatos definen \(Wi=\lambda U_{rms,0}/\ell\) y \(Re=U_{rms,0}\ell/(\nu_s+\nu_p)\). Para vórtices, \(\ell\) es el ancho gaussiano inicial; para Taylor–Green, el inverso del número de onda fundamental; para la banda aleatoria, el inverso del número de onda pico. No deben compararse estos valores con \(Wi\) y \(Re\) de un diámetro de cilindro sin transformar la escala.

## Residuos e identidad energética

El residuo constitutivo independiente usa una derivada temporal de segundo orden de las instantáneas, menos `constitutive_rhs`. Se mide en RMS sobre componentes y espacio; en los extremos usa una fórmula unilateral. Es un diagnóstico de muestreo temporal, no el residuo cero que resultaría de restar una rutina de sí misma. Se entrega como `constitutive_residual_rms`; vale `None` si hay menos de tres instantáneas. `constitutive_pde_residual_rms` compara esa misma derivada temporal con el RHS físico **sin filtrar** y conserva también el error de proyección espacial. `constitutive_spatial_projection_defect_rms` mide por separado la diferencia entre los RHS filtrado y sin filtrar. Ninguna de estas cantidades certifica por sí sola la PDE continua entre puntos de colocación.

Sea \(\langle\cdot\rangle\) el promedio espacial y \(K=\langle|u|^2\rangle/2\). La energía polimérica es:

\[
E_p=\frac{\nu_p}{2\lambda}\langle\Psi(C)\rangle,
\qquad
\Psi_{OB}=\operatorname{tr}C-\log\det C-d,
\]

\[
\Psi_{FENE}=-(L^2-d)\log\!\left(\frac{L^2-\operatorname{tr}C}{L^2-d}\right)
-\log\det C.
\]

Para el problema continuo periódico:

\[
K'=-\nu_s\langle|\nabla u|^2\rangle-\langle\tau_p:G\rangle+\langle u\cdot F\rangle,
\]

\[
(K+E_p)'=-D_s-D_r-D_\kappa+\langle u\cdot F\rangle,
\qquad
D_r=\frac{\nu_p}{2\lambda^2}
\left\langle f^2\operatorname{tr}C-2fd+\operatorname{tr}C^{-1}\right\rangle\ge0.
\]

Con difusión artificial:

\[
D_\kappa=\frac{\nu_p\kappa}{2\lambda}
\left\langle\sum_{a=x,y}
\operatorname{tr}(C^{-1}C_{,a}C^{-1}C_{,a})
+f'(\operatorname{tr}C)|\nabla\operatorname{tr}C|^2\right\rangle,
\]

donde \(f'=0\) en Oldroyd-B y \((L^2-d)/(L^2-\operatorname{tr}C)^2\) en FENE-P; el segundo término aparece una sola vez en la suma global. Los términos son no negativos para conformación admisible. El código evalúa las tasas instantáneas de la energía sobre el RHS discreto y reporta `kinetic_exchange_residual` y `total_energy_rate_residual`. El segundo conserva el defecto de proyección y cuadratura de la energía no lineal; no se fuerza a cero.

## Validación ejecutada

El archivo [viscoelastic.json](../../results/research/solver_validation_2d/viscoelastic.json) contiene los dos modelos. La prueba espacial usa vórtices interactuantes, \(\nu_s=.02\), \(\nu_p=.04\), \(\lambda=.5\), \(U_{rms,0}=.6\), \(L^2=12\) en FENE-P y \(\kappa=0\), hasta \(t=.6\). El ancho inicial es \(.065(2\pi)\), de modo que \(Wi\simeq.735\) bajo esa escala local. La referencia numérica usa \(64^2\) puntos y \(\Delta t=.0025\), y se restringe a los modos activos de cada malla evaluada.

| Malla | Error relativo velocidad Oldroyd-B | Error relativo velocidad FENE-P |
|---|---:|---:|
| \(24^2\) | \(6.998\times10^{-4}\) | \(6.994\times10^{-4}\) |
| \(32^2\) | \(3.264\times10^{-5}\) | \(2.764\times10^{-5}\) |
| \(48^2\) | \(2.430\times10^{-6}\) | \(3.092\times10^{-6}\) |

Un control adicional lleva la referencia de \(64^2\) a \(96^2\). El cambio relativo de velocidad final, restringido a la malla de \(64^2\), es \(5.691\times10^{-7}\) en Oldroyd-B y \(9.580\times10^{-7}\) en FENE-P. Al dividir el paso de la referencia \(64^2\) por dos, los cambios son \(9.695\times10^{-12}\) y \(9.442\times10^{-12}\), respectivamente. Estas cifras son cambios frente a referencias numéricas, no errores frente a una verdad analítica.

La prueba temporal separada mantiene \(48^2\) puntos y compara tres pasos reales sin limitación nominal, \(.01,.005,.0025\), con \(.00125\). El primer ensayo con paso nominal \(.02\) quedó limitado por CFL y se conserva en el JSON identificado como tal; no se usa para inferir orden nominal cuatro.

| Paso real | Error relativo velocidad Oldroyd-B | Error relativo velocidad FENE-P |
|---|---:|---:|
| \(.01\) | \(2.670\times10^{-9}\) | \(2.603\times10^{-9}\) |
| \(.005\) | \(1.653\times10^{-10}\) | \(1.610\times10^{-10}\) |
| \(.0025\) | \(9.694\times10^{-12}\) | \(9.439\times10^{-12}\) |

Los órdenes observados son aproximadamente \(4.014\) y \(4.092\) en ambos modelos; el último incluye la influencia de la referencia temporal finita.

Los controles constitutivos independientes incluyen cizalla homogénea Oldroyd-B, inicialmente \(C=I\):

\[
C_{xy}=\lambda\dot\gamma(1-e^{-t/\lambda}),\quad
C_{xx}=1+2(\lambda\dot\gamma)^2[1-(1+t/\lambda)e^{-t/\lambda}],
\quad C_{yy}=C_{zz}=1.
\]

`integrate_uniform_shear` integra únicamente esa ODE con el mismo término constitutivo; `uniform_shear_reference` da la solución exacta. La velocidad de cizalla uniforme no se presenta como una solución periódica del dominio. Los tests verifican orden cuarto de la ODE. Con \(\nu_p=0\), la velocidad coincide con el solver independiente de vorticidad newtoniana al nivel de redondeo. En ambos modelos, con \(\lambda=.02,.01,.005\), las diferencias frente a Newtoniano de viscosidad total \(.04\) disminuyen aproximadamente a la mitad en un Taylor–Green débil. Este último ensayo incluye la capa inicial de esfuerzo porque \(C(0)=I\); no compara falsamente el esfuerzo transitorio inicial con un esfuerzo newtoniano ya establecido.

También se ejecutaron 31 instantáneas por modelo en \(48^2\), de \(t=0\) a \(.6\), disponibles en `viscoelastic_oldroyd_b.npz` y `viscoelastic_fene_p.npz`, con sus archivos `.metadata.json`, dentro del mismo directorio de resultados. Conservan conformación, esfuerzos y RHS, además de la velocidad. Los autovalores mínimos son aproximadamente \(.448\) y \(.429\); los esfuerzos máximos absolutos son \(.115\) y \(.132\). Entre las instantáneas inicial y final de la referencia \(64^2\), la traza FENE-P máxima es \(4.36<12\), la divergencia RMS máxima es aproximadamente \(2.1\times10^{-15}\) y el defecto instantáneo máximo de energía total es \(8.1\times10^{-10}\). En Oldroyd-B el último defecto es \(2.3\times10^{-9}\). El defecto espacial disminuye en la referencia \(96^2\); la identidad de intercambio cinético se conserva cerca del redondeo.

En las 31 instantáneas de \(48^2\), el residuo constitutivo temporal frente al operador filtrado tiene un máximo RMS de \(9.36\times10^{-4}\) y \(9.55\times10^{-4}\). Los residuos frente a la ecuación física sin filtrar son \(2.43\times10^{-3}\) y \(1.69\times10^{-3}\); el defecto espacial de proyección es \(2.33\times10^{-3}\) y \(1.59\times10^{-3}\), respectivamente. Se reportan en unidades adimensionales absolutas, sin normalización. Estos valores no se sustituyen por cero ni por el residuo del esquema de integración y mantienen visible la resolución constitutiva finita.

`tests/test_viscoelastic_dns.py` reúne 14 pruebas: campos finitos/admisibles, incomprensibilidad, desacoplamiento newtoniano, retroacción polimérica, fuente exacta de cizalla, extensibilidad FENE-P y \(C_{zz}\), estados iniciales inválidos, rechazo de desbordamiento ante entradas extremas finitas, rechazo real de una etapa cercana a la barrera, límite de relajación, disipación/intercambio energético, refinamiento del residuo constitutivo, fuerzas/RK/dimensión y sensibilidad explícita a difusión.

```bash
python -m unittest discover -s tests -p test_viscoelastic_dns.py -v
```

`validation_report()` ofrece una campaña reproducible más corta por defecto; sus parámetros se pueden ampliar. Devuelve un diccionario serializable a JSON con al menos tres mallas y tres pasos, referencias claramente numéricas y `dns_certified=False`. La positividad, finitud y convergencia de estas corridas no justifican extrapolar a otros \(Wi\), geometrías ni tiempos. Para altos esfuerzos y alto \(Wi\), la representación directa puede requerir log-conformation y una validación específica.

## Referencias primarias

- Oldroyd, J. G. (1950), *On the formulation of rheological equations of state*. [DOI: 10.1098/rspa.1950.0035](https://doi.org/10.1098/rspa.1950.0035). Base de la constitución objetiva con memoria.
- Bird, R. B., Dotson, P. J. y Johnson, N. L. (1980), *Polymer solution rheology based on a finitely extensible bead–spring chain model*. [DOI: 10.1016/0377-0257(80)85007-5](https://doi.org/10.1016/0377-0257(80)85007-5). Modelo FENE-P y aproximación de Peterlin.
- Fattal, R. y Kupferman, R. (2004), *Constitutive laws for the matrix-logarithm of the conformation tensor*. [DOI: 10.1016/j.jnnfm.2004.08.008](https://doi.org/10.1016/j.jnnfm.2004.08.008). Referencia para una futura estabilización logarítmica; esta implementación integra \(C\) directamente.

Las entradas bibliográficas están en [references.bib](references.bib). Estos trabajos justifican los modelos, no certifican automáticamente las simulaciones de este repositorio.
