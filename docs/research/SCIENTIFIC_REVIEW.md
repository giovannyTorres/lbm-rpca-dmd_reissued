# Revisión científica independiente de la ampliación

Las comprobaciones independientes incluyen formulaciones, contratos numéricos, fronteras, convergencia y comparabilidad. El baseline de 14 de octubre de 2026 se conserva en [adversarial_review.json](../../results/research/baseline_2026-10-14/solver_validation/adversarial_review.json); sus hashes identifican las fuentes de aquella ejecución. La integración de fase 8 debe verificarse con [verify_adversarial_physics.py](../../python/scripts/verify_adversarial_physics.py), guardando nueva evidencia fuera del baseline. Un resultado aprobado del baseline no certifica automáticamente las fuentes integradas.

## Contraejemplos y criterios de verificación

| Riesgo numérico | Contraejemplo o consecuencia | Criterio de verificación |
|---|---|---|
| Usar la regla de λ de PCP, dependiente del número de características, en Outlier Pursuit. | Para `M=ones(300,20)`, la regla por características da `λ=1/√300<1/√20`. En el problema por columnas, ese valor obliga al óptimo `L=0` aun para una señal perfectamente limpia. | Regla por columnas `λ=3/(7√(γn))`, con `γ=0.1` predeclarado y `n` instantáneas. La reconstrucción limpia es exacta en el contraejemplo. La regla queda invariante al replicar características. Se declaran las hipótesis no verificadas de la regla de Xu et al. |
| Detener Outlier Pursuit comprobando sólo factibilidad primal. | Con esa regla de λ, la factibilidad primal puede aceptar `||L||/||M||=0.406559` y `converged=True`, aunque el óptimo demostrable era cero. | ADMM con μ constante y residuos primal y dual. Con esa regla de λ explícita, debe devolver el óptimo cero; ambos residuos son del orden de `1e-15`. PCP conserva su parada IALM por factibilidad y la identifica como tal, con `optimality_certified=False`. |
| Usar la raíz principal de fbDMD sin contrastar la rama con el operador hacia delante. | La señal limpia `(-0.8)^t` puede dar autovalor `+0.8`, error relativo `0.97561` y una reconstrucción real, por lo que el residuo imaginario no detectaba el problema. | Selección de la rama más cercana al operador hacia delante, con pruebas de condicionamiento y del residuo de la raíz. El contraejemplo tiene error `3.24e-16`; rotaciones amortiguadas de `0.6` y `2.1` radianes por muestra se reconstruyen con error menor que `3e-15`. La degeneración a `π/2` se rechaza explícitamente. |
| Aplicar criterios de calidad sin condición de pared. | En un canal HB, sumar `0.005` a toda la componente tangencial satisface los cuatro criterios de velocidad, vorticidad, divergencia y energía y produce deslizamiento en paredes inmóviles. Una componente normal constante puede penetrar ambas paredes y cancelar el flujo neto. | Métricas independientes de velocidad en pared, error respecto a la pared limpia y componente normal. El runner añade umbrales específicos de canal. Ambas perturbaciones pasan los cuatro criterios anteriores y fallan los criterios de pared de `0.02`. |
| Aceptar velocidad finita sin comprobar esfuerzos y residuos de Oldroyd-B. | Con parámetros finitos extremos, `τxx` puede desbordar y `max(0, NaN)` puede ocultar el fallo. La velocidad puede seguir siendo finita. | Excepciones ante desbordamiento y controles de finitud en estado inicial, sistema discreto, cada etapa, residuos, determinante de conformación y campos exportados. El caso extremo se rechaza; un control estacionario normal conserva el esfuerzo normal analítico. |
| Convertir indiscriminadamente errores numéricos en `null` al serializar. | Una operación desbordada puede confundirse con una métrica físicamente indefinida. | El runner comprueba recursivamente la finitud antes de aceptar una reconstrucción y después de generar el dataset. Los valores `None` explícitos se permiten; `NaN`/`Inf` en escalares, listas y arrays numéricos se rechazan con su ubicación. |

La razón matemática del primer contraejemplo es independiente del solver: `||L||_* ≥ ||L||F` y `||L||2,1 ≤ √n ||L||F`. Por desigualdad triangular, si `λ√n<1`, cualquier descomposición con `L≠0` cuesta más que `L=0,C=M`. La corrección distingue las dos normas de corrupción; cambiar únicamente una tolerancia no habría resuelto la formulación.

## Comprobaciones físicas independientes

El solver periódico usa la relación correcta `ω=-Δψ`, `u=ψy`, `v=-ψx`; la advección y difusión tienen los signos apropiados. Se inspeccionaron el corte estricto de dos tercios y los límites internos de CFL y difusión. La malla periódica no contiene extremos duplicados.

Una vorticidad inicial con tres modos, `sin(x)sin(y)+0.35cos(2x)sin(3y)+0.1sin(5x+2y)`, produce interacción no lineal y evita limitar la revisión a Taylor–Green. En `32²`, viscosidad cero, paso `0.001` y duración `0.4`, la energía y enstrofía varían relativamente menos de `2e-15`. Estos invariantes son una comprobación de la discretización no lineal, no una certificación de resolución.

Con viscosidad `0.02`, dos pasos de `0.0001` satisfacen las identidades de disipación periódica:

- `dE/dt = -ν mean(ω²)`: discrepancia absoluta `5.54e-13`.
- `dZ/dt = -ν mean(ωx²+ωy²)`: discrepancia absoluta `1.35e-11`.

Los invariantes no distinguen por sí solos un signo equivocado en advección. Por eso se comprobó además el término derecho inicial con la función de corriente `ψ=0.6sin(x)sin(y)+0.03cos(2x)sin(3y)`. Las derivadas y el término `-uωx-vωy+νΔω` se calcularon con fórmulas trigonométricas independientes. La derivada temporal del campo numérico, estimada con `(-3ω0+4ω1-ω2)/(2Δt)`, tiene errores relativos `7.756e-7`, `1.939e-7` y `4.849e-8` al reducir el paso de `0.001` a `0.0005` y `0.00025`. El orden observado es dos, correspondiente al estimador de derivada usado; no se confunde con el orden cuatro de RK4.

En reología, el control estacionario Oldroyd-B verifica `τxx=2ληp(uy)²` en caras y mantiene un determinante positivo de conformación. La finitud del campo de velocidad no sustituye la finitud o admisibilidad de sus esfuerzos. Los ensayos más amplios de refinamiento están en [rheology_validation.json](../../results/research/baseline_2026-10-14/solver_validation/rheology_validation.json).

## Alcance que debe acompañar los resultados

La validación del baseline documentada en [DNS_VALIDATION.md](DNS_VALIDATION.md) compara los parámetros del piloto frente a una referencia `128²` en las **31 instantáneas**. Para turbulencia decadente, `64²` alcanza errores relativos máximos de **0.08418537% en velocidad** y **0.51766149% en vorticidad**; para vórtices interactuantes, `48²` alcanza **0.05369304%** y **0.61981839%**, respectivamente. Ambas mallas satisfacen la tolerancia propuesta de 1% en esos dos campos y tiempos concretos. Esta evidencia no establece convergencia de todos los observables, otros parámetros o geometrías ni una certificación universal DNS. El piloto utiliza esas mallas verificadas y conserva `dns_certified=False`.

Las referencias HB/Bingham son estacionarias, con instantáneas limpias idénticas; verifican perfiles y reconstrucción espacial, pero no complejidad modal temporal. Oldroyd-B del baseline es una reducción acoplada de canal totalmente desarrollado en una dimensión; no contiene estela de cilindro, inestabilidad elástica o caos. Las campañas de estelas no newtonianas requieren ejecutar y validar los solvers externos descritos en [NON_NEWTONIAN.md](NON_NEWTONIAN.md). Un manifiesto preparado no equivale a una simulación ejecutada.

El baseline utiliza tres semillas de contaminación y tres repeticiones de costo sobre cada entrada. Las repeticiones no son realizaciones independientes de ruido. Los estimadores acceden a observaciones, tiempos y parámetros predeclarados; la referencia limpia se usa para inyectar contaminación y evaluar, sin ajustar rangos o seleccionar el mejor resultado. La estimación de ruido de SPCP utiliza intensidad de inyección conocida y escala observada bajo una aproximación de independencia; su validez debe reevaluarse con correlación y datos reales.

La comparación computacional conserva tiempo y añade CPU, RSS absoluta del proceso aislado, asignaciones visibles a `tracemalloc`, iteraciones y proxies declarados de operaciones. La RSS incluye runtime y dataset; los proxies no son conteos hardware ni mediciones de energía. El trazado introduce sobrecosto. La comparación de Pareto se realiza dentro de un mismo caso, ruido y semilla. La frontera adicional condicionada a los criterios físicos distingue el costo bajo los umbrales fijados; aprobar esos umbrales no demuestra satisfacer toda la ecuación de cantidad de movimiento.

Las métricas de materia son condicionales cuando se conserva la densidad limpia en una reconstrucción sólo de velocidad. En densidad constante, la masa total es trivial: divergencia, continuidad, flujo y fronteras aportan la evidencia útil. En dominios con sólido, los operadores excluyen un halo de diferencias finitas y el flujo en la superficie del sólido no se integra por separado. El balance neto global no garantiza impermeabilidad local, como muestra el contraejemplo de las paredes.

## Reproducción

```bash
python python/scripts/verify_adversarial_physics.py --output results/research/solver_validation_local/adversarial_review.json
```

Este comando produce `results/research/solver_validation_local/adversarial_review.json`, registra fórmulas, entradas, umbrales, versiones y SHA-256 de las fuentes, y termina con código distinto de cero si falla cualquier grupo. Si cambia una fuente registrada, se debe regenerar el JSON para vincular la evidencia al código que se entrega.

La revisión de los solvers 2D acoplados, la proyección física y el pronóstico cronológico nuevos está en [SECOND_REVIEW.md](SECOND_REVIEW.md).
