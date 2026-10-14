# Restricciones físicas y evaluación de ecuaciones

`physics.py` añade estimadores que sólo usan geometría y observaciones. La proyección de Helmholtz periódica descompone cada campo en componentes solenoidal e irrotacional: en Fourier aplica `I-kkᵀ/|k|²`, preservando el modo medio. Es una proyección ortogonal en la malla uniforme; no puede recuperar un sesgo uniforme incorrecto ni certificar la dinámica.

`helmholtz_dealias` añade un prior de ancho de banda: conserva modos estrictamente inferiores a un tercio de la malla por eje. Ese conocimiento del operador del banco es explícito. No se usa la velocidad limpia para ajustar una banda conveniente. El método se rechaza en canales y dominios con obstáculos, donde se necesita un Poisson con condiciones de frontera diferentes. Las paredes no se resuelven enmascarando un campo periódico.

En mallas pares, una primera derivada real de Nyquist no tiene representación no nula en los nodos. El proyector usa el operador discreto compatible y declara ese convenio. La segunda derivada sí conserva el factor de frecuencia física; el Laplaciano del diagnóstico de vorticidad usa la frecuencia completa. Una prueba adversarial detecta la subestimación de 17 veces que produciría reutilizar el operador del proyector en un campo de Nyquist.

Se mide además el residuo de vorticidad newtoniano periódico:

`Rω = ∂tω + u∂xω + v∂yω − νΔω − curl(f)`.

La derivada temporal se calcula con las instantáneas guardadas, incluyendo error de muestreo y estencils de extremos. Se guardan el residuo limpio, el reconstruido y la alteración adicional; la escala combina vorticidad, velocidad, tamaño de dominio y viscosidad. Un residuo pequeño no demuestra por sí solo una solución exacta o conservación de todos los esfuerzos.

Los campos no newtonianos se marcan como no aplicables a este diagnóstico de velocidad sola: omitir `curl(div τ/ρ)` y llamar al resultado residuo de Navier–Stokes sería incorrecto. Sus solvers sí guardan esfuerzos y residuos constitutivos de la simulación limpia, junto con defectos de proyección y cuadratura. La campaña reconstruye velocidad; no afirma haber reconstruido los esfuerzos poliméricos.

Las variantes físicas entran en el costo medido: ajuste, reconstrucción vectorial y proyección se ejecutan dentro del proceso instrumentado. Los conteos de FFT acompañan las SVD y los proxies de trabajo. Las métricas físicas y el guardado de archivos se calculan fuera de ese intervalo. Los umbrales son predeclarados y la referencia se usa sólo para evaluación.
