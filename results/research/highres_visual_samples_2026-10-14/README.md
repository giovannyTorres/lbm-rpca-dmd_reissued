# Muestras visuales de los campos calculados

Esta carpeta contiene cinco instantáneas exactas por cada una de ocho trayectorias de mayor resolución: dos flujos newtonianos 512², Oldroyd-B/FENE-P y su control newtoniano 256², y tres HB/Bingham 128². No se interpolan ni agrandan imágenes para crear datos. `manifest.json` identifica índices, tiempos, mallas, campos y hashes del NPZ completo de origen y de la selección.

Los NPZ se publican en la rama `data/high-resolution-visuals-2026-10-14`. La rama principal conserva este manifiesto y los metadatos, junto con el código y las figuras. Para inspeccionar datos, descargar el archivo ZIP de la rama de muestras o clonar esa rama.

Los metadatos conservan las configuraciones originales. La densidad uno es una hipótesis explícita de estos solvers. Las referencias son simulaciones numéricas; no establecen por sí solas DNS universal o validación experimental. Las selecciones sirven para inspeccionar y reproducir imágenes de campos; no permiten recomputar los máximos de error de todos los tiempos del banco completo.

Las historias completas se regeneran con `python/scripts/run_highres_newtonian.py`, `run_highres_viscoelastic.py` y `run_highres_viscoplastic.py`. El documento y las tablas completas están en las carpetas originales.
