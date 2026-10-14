#!/usr/bin/env python3
"""Independent tensor-norm check, using saved fields without rerunning the PDE.

Run after run_highres_viscoplastic.py has produced comparisons.json. This
companion keeps the packed-three-component L2 metric and additionally uses
the physical symmetric-tensor Frobenius norm, xx²+2xy²+yy².
"""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.signal import resample

OUTPUT = Path(__file__).resolve().parent


def fields(name, times):
    path=OUTPUT/'datasets'/f'{name}.npz'
    if not path.exists() and name.endswith('_short'):
        path=OUTPUT/'datasets'/f'{name[:-6]}.npz'
    with np.load(path) as saved:
        indices=[]
        for time in times:
            candidates=np.flatnonzero(np.isclose(saved['times'],time,rtol=0,atol=2e-14))
            if len(candidates)!=1:raise ValueError(f'unmatched time {time} in {path.name}')
            indices.append(int(candidates[0]))
        value=saved['deviatoric_stress'][indices]
    if value.shape[1]!=3 or not np.all(np.isfinite(value)):
        raise ValueError('expected finite symmetric stress components xx, xy, yy')
    return value


def interpolate(field, shape):
    return resample(resample(field,shape[-2],axis=-2),shape[-1],axis=-1)


def error(a,b):
    # Weight xx/xy/yy by 1/2/1 before summing squares.
    weights=np.sqrt(np.asarray([1.,2.,1.]))[None,:,None,None]
    return float(np.linalg.norm((a-b)*weights)/max(np.linalg.norm(b*weights),np.finfo(float).tiny))


def main():
    records=[]
    for filename in ('comparisons.json','sensitivity.json'):
        checks=json.loads((OUTPUT/filename).read_text())
        for check in checks:
            a=fields(check['run'],check['times']);b=fields(check['reference'],check['times'])
            restricted=interpolate(b,a.shape);prolonged=interpolate(a,b.shape)
            per_time=[error(x[None],y[None]) for x,y in zip(prolonged,b)]
            records.append({'run':check['run'],'reference':check['reference'],
                'scope':check['scope'],'times':check['times'],
                'coarse_output_grid_all_times_relative_l2':error(a,restricted),
                'full_field_all_times_relative_l2':error(prolonged,b),
                'full_field_final_relative_l2':per_time[-1],
                'full_field_maximum_per_time_relative_l2':max(per_time),
                'full_field_per_time_relative_l2':per_time})
    result={'stress_order':['xx','xy','yy'],
        'norm_definition':'symmetric 2x2 tensor Frobenius: sqrt(sum(xx^2+2xy^2+yy^2))',
        'difference_from_comparisons_json':'the primary comparison uses an unweighted packed-component vector L2 norm; both definitions are explicit',
        'spatial_transfer':'periodic Fourier resampling with scipy.signal.resample, including even-grid Nyquist handling',
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_file_hashes':{p:hashlib.sha256((OUTPUT/p).read_bytes()).hexdigest() for p in ('comparisons.json','sensitivity.json')},
        'comparisons':records}
    (OUTPUT/'derived_stress_frobenius.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f'{len(records)} tensor-norm comparisons written')


if __name__=='__main__':main()
