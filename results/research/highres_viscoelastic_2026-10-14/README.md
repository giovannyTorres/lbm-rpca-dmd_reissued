# Vórtices viscoelásticos: refinamiento y evolución visual

Se ejecutaron Oldroyd-B y FENE-P en 64², 128², 256², con 25 estados entre t=0 y 2.4, y una verificación dt/2 en 128² por modelo.

El control newtoniano usa la misma velocidad inicial, densidad, dominio, malla fina y tiempos. La diferencia constitutiva es un resultado material, no error de reconstrucción. Se conserva C tridimensional en flujo plano, κ=0 y rechazo de pasos que violen SPD/traza; no se repara C.

Las curvas comparan modos Fourier comunes y el JSON/CSV también compara cada solución gruesa prolongada al campo fino completo sin filtrar la referencia; registra aparte la energía que la malla gruesa no representa. La representatividad se limita a este régimen periódico moderado, con 25 instantes almacenados; no demuestra estelas de cilindros, caos ni DNS universal.

| Modelo | Malla | Máx. error u | Máx. error ω | Máx. error C−I | Máx. error τ |
|---|---:|---:|---:|---:|---:|
| Oldroyd-B | 64² | 1.050e-04 | 1.805e-03 | 5.794e-02 | 5.794e-02 |
| Oldroyd-B | 128² | 5.925e-06 | 2.073e-04 | 2.338e-02 | 2.338e-02 |
| FENE-P | 64² | 6.779e-05 | 1.099e-03 | 5.037e-03 | 4.582e-03 |
| FENE-P | 128² | 2.515e-06 | 8.360e-05 | 6.725e-04 | 6.235e-04 |

## Figuras y datos

- `vorticity_evolution.pdf/png/gif`: evolución real y líneas de corriente instantáneas; escala de color común.
- `polymer_evolution.pdf/png/gif`: tr(C)−3 y τxy; escalas comunes por cantidad.
- `energy_memory_and_residuals.pdf/png`: intercambio energía cinética/polimérica y defectos medidos.
- `grid_refinement.pdf/png`, `spectral_resolution.pdf/png`: refinamiento y espectros finales.
- `compact_vorticity_evolution.pdf/png`, `compact_polymer_trace_evolution.pdf/png`, `compact_polymer_stress_evolution.pdf/png`: tres tiempos con letra mayor para páginas del documento.
- `compact_full_field_refinement.pdf/png`, `compact_spectral_resolution.pdf/png`: controles de resolución legibles en papel.
- `summary.json`, `grid_summary.csv`, `datasets/*json`: series, admisibilidad, parámetros, hashes, tiempos y fallos.
- `datasets/*.npz`: campos calculados completos de velocidad, C y esfuerzo; regenerables mediante este script.

Los residuos obtenidos por diferencias temporales de snapshots dependen de Δt_salida, y se distinguen del defecto espacial de proyección y del balance instantáneo del operador semidiscreto.
