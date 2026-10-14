# Entradas completas para comparar modelos

La rama `data/model-comparison-fidelity-2026-10-14` conserva siete NPZ calculados en 128² en sus rutas originales, con todos sus estados: Navier–Stokes 41, Oldroyd-B/FENE-P 25 y HB/Bingham 21. Los sidecars JSON conservan parámetros y trazabilidad. `manifest.json` distingue el SHA-256 del archivo comprimido del digest canónico de arrays y metadatos usado por el solver viscoplástico.

Estas historias permiten ejecutar el benchmark principal, la sensibilidad de rango/intensidad y los controles sin ruido. Los estimadores reciben velocidad observada y tiempos; los campos constitutivos están disponibles al evaluador. No se interpola ni modifica ninguna historia.

La rama principal conserva el código, las tablas, el documento, las figuras y este manifiesto. La rama de datos añade los siete NPZ originales y 40 selecciones de reconstrucciones: diez modelos, dos observaciones concretas y dos tratamientos, con tres estados exactos por selección. Estas muestras permiten regenerar las cuatro láminas de campos del benchmark. No contienen todas las historias reconstruidas ni las 1050 selecciones de la campaña local; los errores y máximos completos permanecen en el ledger. No incluye todo el banco de refinamiento 512²/256² ni los controles de referencia nuevos; sus scripts y registros permanecen reproducibles por separado.
