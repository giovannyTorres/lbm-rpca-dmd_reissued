# Campos sintéticos de las campañas ampliadas

Esta rama añade 81 NPZ generados por los solvers y los escenarios de pronóstico. La implementación y la evidencia científica corresponden al commit 450a8d3d84877afeddd7a98ad0d15cb1584bb116 de main. No contiene datos experimentales privados, originales históricos de la tesis ni copias de historia Git privada.

Los campos limpios y sus metadatos están en extended_physics_2026-10-14/datasets, forecasting_extended_2026-10-14/datasets, rheology_matched_2026-10-14/datasets y solver_validation_2d. Los archivos forecasting_extended_2026-10-14/predictions guardan tiempos y predicciones seleccionadas/persistencia en prueba. Los hashes de cada NPZ están en synthetic_data_manifest.json; los hashes científicos de arrays y de las fuentes están en los manifiestos originales de las campañas.

Las observaciones contaminadas se regeneran con las semillas y parámetros declarados. Los esfuerzos y conformación corresponden a referencias simuladas; las reconstrucciones y pronósticos actuales estiman velocidad. Un campo disponible no certifica DNS universal, caos ni alto Weissenberg.

Para descargar código, evidencia y campos juntos:

```bash
git clone --branch data/synthetic-campaign-2026-10-14 --single-branch https://github.com/giovannyTorres/lbm-rpca-dmd_reissued.git
```

Instalar requirements.txt y seguir docs/research/RUNNING.md. Los NPZ científicos se omiten de main y se regeneran allí al ejecutar una campaña.
