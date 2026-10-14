#!/usr/bin/env python3
"""Export exact computed frames for a compact public visualization supplement.

Full trajectories remain regenerable with the study scripts. No resampling,
interpolation or synthetic rendering changes the values of the selected frames.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
RESULTS=ROOT/'results/research'
OUTPUT=RESULTS/'highres_visual_samples_2026-10-14'
EXPERIMENTS=(
 ('newtonian_vortices_512','highres_newtonian_2026-10-14','interacting_vortices_512_dt0.005'),
 ('newtonian_multiscale_512','highres_newtonian_2026-10-14','decaying_turbulence_512_dt0.005'),
 ('oldroyd_b_256','highres_viscoelastic_2026-10-14','oldroyd_b_256'),
 ('fene_p_256','highres_viscoelastic_2026-10-14','fene_p_256'),
 ('visco_newtonian_256','highres_viscoelastic_2026-10-14','newtonian_256'),
 ('hb_thinning_128','highres_viscoplastic_2026-10-14','hb_thinning_128'),
 ('bingham_128','highres_viscoplastic_2026-10-14','bingham_128'),
 ('hb_thickening_128','highres_viscoplastic_2026-10-14','hb_thickening_128'),
)
TEMPORAL_FIELDS=('velocity','conformation','stress','pressure','shear_rate','apparent_viscosity',
                 'deviatoric_stress','stress_invariant','approximate_yielded','over_yield_stress')


def digest(path):
    value=hashlib.sha256()
    with path.open('rb')as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def main():
    (OUTPUT/'datasets').mkdir(parents=True,exist_ok=True)
    manifest={'scope':'Five exact computed frames per long visual trajectory, not complete time histories. Full campaigns, errors and provenance remain in the study directories; regenerate full NPZ with their scripts.',
              'interpolated':False,'private_data':False,'density':1.,
              'density_note':'These specific periodic solvers explicitly declare constant unit density; no missing experimental density is inferred.',
              'script_sha256':digest(Path(__file__)),'datasets':[]}
    for name,campaign,stem in EXPERIMENTS:
        path=RESULTS/campaign/'datasets'/f'{stem}.npz'
        source_meta=json.loads(path.with_suffix('.json').read_text())
        with np.load(path)as arrays:
            times=arrays['times'];indices=np.linspace(0,len(times)-1,5,dtype=int)
            source_shape=list(arrays['velocity'].shape)
            selected={'times':times[indices].copy()}
            for key in TEMPORAL_FIELDS:
                if key not in arrays.files:continue
                field=arrays[key]
                if field.shape[0]!=len(times):raise ValueError(f'Unexpected time axis: {path} {key}')
                if not np.isfinite(field).all():raise ValueError(f'Nonfinite source: {path} {key}')
                selected[key]=field[indices].copy()
            if 'mask'in arrays.files:selected['mask']=arrays['mask'].copy()
        target=OUTPUT/'datasets'/f'{name}.npz'
        np.savez_compressed(target,**selected)
        with np.load(target)as exported:
            for key,value in selected.items():np.testing.assert_array_equal(exported[key],value)
        n=selected['velocity'].shape[-1]
        record={'name':name,'path':str(target.relative_to(ROOT)),'sha256':digest(target),'bytes':target.stat().st_size,
                'source_npz':str(path.relative_to(ROOT)),'source_npz_sha256':digest(path),
                'source_shape':source_shape, 'frame_indices':indices.tolist(),'times':selected['times'].tolist(),
                'nx':n,'ny':n,'dx':float(2*np.pi/n),'dy':float(2*np.pi/n),'boundary':'periodic',
                'fields':{key:list(value.shape)for key,value in selected.items()}}
        manifest['datasets'].append(record)
        (OUTPUT/'datasets'/f'{name}.json').write_text(json.dumps({'selection':record,'source_metadata':source_meta},indent=2)+'\n')
        print(name,target.stat().st_size,flush=True)
    (OUTPUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (OUTPUT/'README.md').write_text('''# Muestras visuales de los campos calculados\n\nEsta carpeta contiene cinco instantáneas exactas por cada una de ocho trayectorias de mayor resolución: dos flujos newtonianos 512², Oldroyd-B/FENE-P y su control newtoniano 256², y tres HB/Bingham 128². No se interpolan ni agrandan imágenes para crear datos. `manifest.json` identifica índices, tiempos, mallas, campos y hashes del NPZ completo de origen y de la selección.\n\nLos NPZ se publican en la rama `data/high-resolution-visuals-2026-10-14`. La rama principal conserva este manifiesto y los metadatos, junto con el código y las figuras. Para inspeccionar datos, descargar el archivo ZIP de la rama de muestras o clonar esa rama.\n\nLos metadatos conservan las configuraciones originales. La densidad uno es una hipótesis explícita de estos solvers. Las referencias son simulaciones numéricas; no establecen por sí solas DNS universal o validación experimental. Las selecciones sirven para inspeccionar y reproducir imágenes de campos; no permiten recomputar los máximos de error de todos los tiempos del banco completo.\n\nLas historias completas se regeneran con `python/scripts/run_highres_newtonian.py`, `run_highres_viscoelastic.py` y `run_highres_viscoplastic.py`. El documento y las tablas completas están en las carpetas originales.\n''')
    print(OUTPUT)


if __name__=='__main__':main()
