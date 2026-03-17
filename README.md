# lbm-rpca-dmd_reissued

Plataforma de investigación reproducible para simulación de fluidos y remoción de ruido.

## Estado actual
- ✅ FASE 0: diseño de arquitectura inicial.
- 🚧 FASE 1: solver LBM 2D D2Q9 BGK (arranque).
- ⏳ FASE 2+ pendientes.

## Ejecución rápida (FASE 1)

```bash
python3 python/scripts/run_phase1_example.py --config configs/lbm_cylinder_base.cfg
```

Salida esperada en `data/raw/<run_id>/` con snapshots CSV y `manifest.txt`.

## Prueba de humo

```bash
python3 -m unittest tests/test_phase1_smoke.py
```
