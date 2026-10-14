# Vórtices viscoplásticos con mallas mayores

Esta campaña integra el momentum 2D con transporte del esfuerzo completo
`div(2 η_app D)`, presión e incompresibilidad. Compara Bingham regularizado y
Herschel–Bulkley con `n=0.6` y `n=1.4`. Los tres materiales parten exactamente
de la misma velocidad inicial, densidad, dominio y tiempos de salida.

La viscosidad aparente coincide en **un corte de referencia declarado**,
`γ_ref=0.8/(0.065·2π)`, con valor `0.04`. Esta coincidencia en un punto no hace
equivalentes las curvas constitutivas completas. Se conserva `τ_y=0.01`,
`m=10` y `ε=0.001`, sin recortar la viscosidad.

## Resolución y horizontes

| Estudio | Mallas del estado | Horizonte | Interpretación |
|---|---:|---:|---|
| Evolución visual y comparación espacial larga | 64² y 128² | 0–2, 21 instantes | La referencia larga es 128²; no es una solución exacta |
| Referencia espacial más fina | 256² | 0–0.1 | Comprueba el horizonte corto; no valida toda la trayectoria larga |
| Refinamiento temporal fijo | 128² | 0–0.1 | Máximos nominales `Δt=0.0004/0.0002/0.0001`; se informa el paso aceptado real |
| Cuadratura constitutiva y sensibilidad de regularización | 64² | 0–0.1 | Cambiar la cuadratura cambia la integración; cambiar `m` o `ε` cambia el modelo |

Los estados conservan modos `|k_x|,|k_y|<N/3`; la cuadratura de los productos
constitutivos usa normalmente `1.5N` puntos por dirección. Un estado 128² no
resuelve cualquier fenómeno hasta su frecuencia de Nyquist. La comparación
espacial incluye tanto el campo de referencia restringido a la malla de
salida gruesa como el campo grueso interpolado frente al **campo fino completo**.
Se informa el error de velocidad, vorticidad, esfuerzo, viscosidad y corte en
todos los instantes comunes y su máximo. La primera comparación no debe
describirse como una comparación exclusiva de los modos activos 2/3.

La desviación estándar inicial de cada núcleo gaussiano es `0.065L`. Por
tanto, ese ancho contiene sólo **2.08 celdas en la antigua malla 32²**, frente
a **4.16, 8.32 y 16.64 celdas** en 64², 128² y 256². Esta relación entre
geometría y espaciamiento explica la mejora del banco visual. Aun así, el
esfuerzo constitutivo puede generar estructuras más finas que el núcleo
inicial: también deben observarse los errores de esfuerzo/viscosidad y los
espectros, no solamente el tamaño nominal de la malla.

El CFL difusivo declarado es `0.8`; todas las etapas deben satisfacer las
cotas advectiva/difusiva y el criterio de energía. Los pasos rechazados se
cuentan y reducen; aumentar ese parámetro no elimina esos controles. El
refinamiento temporal separado evita confundir estabilidad con exactitud.

La rigidez explícita también es un resultado: para `n≤1`, la cota de
viscosidad de corte cero es `η(0)=η_s+K ε^(n−1)+τ_y m` (`η_s=0` aquí).
En HB adelgazante se aproxima a 0.824, aunque los tres materiales coinciden
en viscosidad 0.04 en el corte de referencia. Ello restringe subpasos y
genera más rechazos. Un paso viscoso implícito o una cota global conservadora
constituyen líneas futuras; esta campaña no sustituye la viscosidad por un
valor recortado. Los máximos de viscosidad y de paso pueden ocurrir en
instantes distintos: el cumplimiento de CFL se comprueba por etapa, no
multiplicando esos dos máximos agregados.

## Lectura de las ilustraciones

- **vorticity_evolution.pdf/png:** cinco instantes por material, con la misma
  escala de color para los quince paneles. Revela desplazamiento, deformación
  y difusión, sin normalizar cada imagen por separado.
- **constitutive_fields.pdf/png:** corte, viscosidad, esfuerzo y el proxy de
  esfuerzo por debajo de `τ_y` al final. El proxy no demuestra regiones rígidas
  exactamente no cedidas; el modelo regularizado admite deformación allí.
- **evolution_diagnostics.pdf/png:** energía, enstrofía, disipación, residuo
  del balance energético, proxy de cedencia y espectro inicial/final.
- **resolution_comparison.pdf/png:** diferencias de velocidad/vorticidad a lo
  largo del horizonte y espectros finales 64²/128².
- **viscoplastic_evolution.gif:** los tres materiales evolucionan sincronizados
  en malla 128² y con escala de vorticidad común.

Los compañeros para el cuerpo del documento tienen menos paneles y tipografía
de 12–14 puntos: `vorticity_publication` (3 materiales por 3 tiempos),
`shear_viscosity_publication` (2×3), `stress_yield_publication` (2×3) y
`diagnostics_publication` (2×2), en PDF y PNG bajo `figures/`. La viscosidad usa una escala
logarítmica común declarada; la vorticidad conserva límites idénticos para
todos los materiales y tiempos. Se regeneran desde los campos guardados:

```bash
python results/research/highres_viscoplastic_2026-10-14/render_publication.py
```

El mismo compañero produce `full_field_refinement` (curvas largas del campo
completo con esfuerzo Frobenius), `spatial_refinement_publication` (máximos
entre los dos instantes guardados `t=0,0.1` frente a 256²) y
`sensitivity_publication` (cuadratura, `m` y `ε` en ejes separados).
Los máximos sólo cubren instantes guardados, no todos los subpasos internos.
El render debe ejecutarse después de `derive_stress_frobenius.py`.
La opción `--primary-only` permite dibujar las cuatro figuras de las
trayectorias 128² completas mientras siguen los controles secundarios;
`publication_provenance.json` distingue esa etapa de la campaña completa.

Los campos, parámetros y tiempos efectivos están en `datasets/*.npz` y
`datasets/*.json`; los NPZ se distribuyen como artefactos, no en el historial
Git. `summary.csv`, `comparisons.json`, `sensitivity.json`, `provenance.json` y
`completion.json` documentan métricas, límites y hashes de fuente. Una campaña
sólo está completa cuando `completion.json` declara `complete=true`.

## Reproducción

Desde la raíz del repositorio:

```bash
python python/scripts/run_highres_viscoplastic.py
```

La caché requiere configuración y hashes de código idénticos. El tamaño de
malla y la evolución visual aportan evidencia numérica; no certifican DNS,
caos, estelas de cilindro, superficies de cedencia exactas ni validación
experimental. El esfuerzo no polinomial conserva error de cuadratura aun con
padding 3/2, por lo que ese refinamiento se declara por separado.

Para obtener además la norma física de Frobenius del tensor simétrico,
después de la campaña:

```bash
python results/research/highres_viscoplastic_2026-10-14/derive_stress_frobenius.py
```

`comparisons.json` usa L₂ del vector de componentes almacenados `(xx,xy,yy)`.
El archivo complementario `derived_stress_frobenius.json` pondera el cortante
con factor 2 en la suma de cuadrados, `xx²+2xy²+yy²`, para representar el tensor
2×2. Esta comprobación utiliza los NPZ guardados y no reintegra el momentum.

La auditoría de los campos publicados se reproduce con:

```bash
python results/research/highres_viscoplastic_2026-10-14/verify_data_integrity.py --require-complete
```

Verifica finitud, formas, tiempo ordenado, horizonte, densidad/geometría
positivas y los hashes canónicos del dataset y de las fuentes. El calendario
de dos trabajadores independientes para los controles se declara en
`execution_notes.json`; CPU, pasos y límites reales acompañan al tiempo de
pared, que puede reflejar contención de otros experimentos.
