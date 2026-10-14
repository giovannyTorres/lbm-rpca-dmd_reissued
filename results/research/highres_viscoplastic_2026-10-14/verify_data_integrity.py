#!/usr/bin/env python3
"""Audit saved campaign fields, canonical dataset hashes and source hashes."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np

OUTPUT=Path(__file__).resolve().parent
ROOT=OUTPUT.parents[2]
sys.path.insert(0,str(ROOT/'python/src'))
from fluid_denoise.research.runner import assert_finite,dataset_digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-complete',action='store_true')
    args=parser.parse_args()
    completion=json.loads((OUTPUT/'completion.json').read_text())
    if args.require_complete and not completion.get('complete'):
        raise RuntimeError('The campaign is not complete')
    records=[]
    for path in sorted((OUTPUT/'datasets').glob('*.json')):
        meta=json.loads(path.read_text())
        with np.load(path.with_suffix('.npz')) as saved:data={k:saved[k] for k in saved.files}
        data.update(metadata=meta['metadata'],dx=meta['dx'],dy=meta['dy'],boundary='periodic')
        assert_finite(data)
        if list(data['velocity'].shape)!=meta['velocity_shape']:raise ValueError('Wrong velocity shape')
        if data['times'][0]!=0 or not np.all(np.diff(data['times'])>0):raise ValueError('Invalid snapshot times')
        final_time=meta['configuration']['dt']*meta['configuration']['n_steps']
        if not np.isclose(data['times'][-1],final_time,rtol=0,atol=1e-12):raise ValueError('Wrong horizon')
        if not np.all(data['density']>0) or data['dx']<=0 or data['dy']<=0:raise ValueError('Invalid geometry/density')
        digest=dataset_digest(data)
        if digest!=meta['dataset_sha256']:raise ValueError(f'Dataset digest mismatch: {path.name}')
        for name,expected in meta['source_hashes'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
                raise ValueError(f'Source hash mismatch: {name}')
        records.append({'run':path.stem,'dataset_sha256':digest,'all_fields_finite':True,
            'canonical_dataset_hash_matches':True,'all_source_hashes_match':True,
            'horizon_matches_configuration':True,'velocity_shape':list(data['velocity'].shape)})
    if args.require_complete and len(records)!=completion['runs']:
        raise ValueError('Dataset count disagrees with completion record')
    result={'valid':True,'campaign_complete':completion.get('complete',False),
        'datasets_verified':len(records),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'audit_scope':'finite fields, source and canonical dataset hashes, positive density/geometry, strict time ordering and configured horizon',
        'records':records}
    (OUTPUT/'data_integrity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f'{len(records)} saved datasets verified; campaign complete={completion.get("complete",False)}')


if __name__=='__main__':main()
