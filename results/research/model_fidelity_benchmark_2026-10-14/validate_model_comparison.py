#!/usr/bin/env python3
"""Check ledger denominators, source epochs, observed inputs and physical controls."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
from pathlib import Path
import json
import sys
import hashlib
import numpy as np

DEST=Path(__file__).resolve().parent
ROOT=DEST.parents[2]
sys.path.insert(0,str(ROOT/'python/scripts'))
import run_model_fidelity_benchmark as benchmark
from fluid_denoise.research.cost import quality_gates


def main():
    records=[json.loads(line) for line in (DEST/'ledger.jsonl').read_text().splitlines() if line.strip()]
    completion=json.loads((DEST/'completion.json').read_text())
    provenance=json.loads((DEST/'provenance.json').read_text())
    config=provenance['configuration']
    if len(records)!=1050 or not completion['complete']:raise ValueError('Incomplete primary ledger')
    expected={(flow,noise,seed,method) for flow in benchmark.FLOW_FILES
              for noise in config['noise_models'] for seed in config['seeds'] for method in benchmark.METHODS}
    actual={(r['flow'],r['noise'],r['seed'],r['method']) for r in records}
    if actual!=expected or len(actual)!=len(records):raise ValueError('Condition design/denominator mismatch')
    source_hashes={str(p.relative_to(ROOT)):benchmark.sha256(p)
                  for p in (ROOT/'python/src/fluid_denoise/research').glob('*.py')}
    if source_hashes!=provenance['estimator_source_hashes']:raise ValueError('Scientific modules changed')
    driver_hashes=sorted({r['driver_sha256'] for r in records})
    if any(digest not in provenance['driver_epochs'] for digest in driver_hashes):
        raise ValueError('A historical executing driver is missing from provenance epochs')
    for digest in driver_hashes:
        current=benchmark.sha256(benchmark.__file__)
        if digest!=current:
            archived=DEST/'driver_epochs'/f'driver_{digest}.py'
            if not archived.exists() or benchmark.sha256(archived)!=digest:
                raise ValueError('Historical driver source hash is not reproducible')
    identity_observations=0;archive_controls=0;equation_applicable=0;nonconverged=0
    h_curl_max_difference=0.;acceptances={v:0 for v in config['postprocessing']}
    archive_manifest={}
    for flow in benchmark.FLOW_FILES:
        reference,source=benchmark.load_flow(flow)
        sidecar=source['source_metadata']
        if flow in ('hb_thinning','bingham','hb_thickening'):
            with np.load(ROOT/source['path']) as source_fields:
                if not np.all(source_fields['density']==1.):
                    raise ValueError('VP explicit source density differs from rho=1 contract')
        elif sidecar.get('density') not in (1.,'constant rho=1; implicit rather than duplicated in NPZ'):
            raise ValueError('Source metadata must explicitly declare rho=1')
        if provenance['clean_input_sources'][flow]['sha256']!=source['sha256']:
            raise ValueError('Raw source archive differs from recorded input')
        for noise in config['noise_models']:
            for seed in config['seeds']:
                perturbation=benchmark.observation_noise(reference['velocity'].shape,reference['times'],
                    noise,seed,float(np.sqrt(np.mean(reference['velocity']**2))),config)
                observed=reference['velocity']+perturbation
                digest=hashlib.sha256(observed.tobytes()).hexdigest()
                selected=[r for r in records if (r['flow'],r['noise'],r['seed'])==(flow,noise,seed)]
                if any(r['noise_metadata']['observation_sha256']!=digest for r in selected):
                    raise ValueError('Paired models did not receive the same regenerated observation')
                identity=next(r for r in selected if r['method']=='identity')
                if identity['status']=='ok':
                    raw=next(e for e in identity['evaluations'] if e['variant']=='raw')
                    if not np.isclose(raw['metrics']['velocity_relative_l2'],.2,rtol=1e-12,atol=1e-14):
                        raise ValueError('Exact20% RMS identity control does not match')
                    identity_observations+=1
                    archive=raw['reconstruction_archive']
                    if archive is not None:
                        with np.load(ROOT/archive['path']) as saved:
                            indices=saved['frame_indices']
                            np.testing.assert_array_equal(saved['velocity'],observed[indices])
                            np.testing.assert_array_equal(saved['times'],reference['times'][indices])
                        archive_controls+=1
    for record in records:
        if record['status']!='ok':continue
        diagnostics=record['diagnostics']
        nonconverged+=not diagnostics['converged']
        if record['method'] in ('pcp_tls_r8','stable_tls_r8'):
            combined=diagnostics['preprocessing']['converged'] and diagnostics['modal']['converged']
            if diagnostics['converged']!=combined:raise ValueError('Hybrid convergence drops a stage')
        raw=next(e for e in record['evaluations'] if e['variant']=='raw')
        h=next(e for e in record['evaluations'] if e['variant']=='helmholtz')
        discrepancy=abs(raw['metrics']['vorticity_relative_l2']-h['metrics']['vorticity_relative_l2'])
        h_curl_max_difference=max(h_curl_max_difference,discrepancy)
        if not np.isclose(raw['metrics']['vorticity_relative_l2'],h['metrics']['vorticity_relative_l2'],rtol=1e-11,atol=1e-12):
            raise ValueError('Helmholtz projection unexpectedly changes discrete vorticity error')
        for evaluation in record['evaluations']:
            metrics=evaluation['metrics'];variant=evaluation['variant']
            expected_pass=quality_gates(metrics,config['quality_thresholds'])['passed'] and diagnostics['converged']
            if evaluation['quality_pass']!=expected_pass:raise ValueError('Operational acceptance differs from declared gate')
            acceptances[variant]+=bool(expected_pass)
            if not metrics['constant_density_mass_trivial'] or not metrics['candidate_density_assumed']:
                raise ValueError('Density assumption is not declared')
            applicable=record['flow'].startswith('ns_')
            if metrics['vorticity_equation_applicable']!=applicable:
                raise ValueError('Non-Newtonian stress omitted from a claimed Newtonian PDE diagnostic')
            equation_applicable+=bool(metrics['vorticity_equation_applicable'])
            if record['flow'] in ('oldroyd_b','fene_p') and metrics.get('stress_frobenius_relative_l2') is not None:
                raise ValueError('Unidentified polymer memory cannot be reported as reconstructed')
            archive=evaluation['reconstruction_archive']
            if archive is not None:
                path=ROOT/archive['path']
                if benchmark.sha256(path)!=archive['sha256']:raise ValueError('Changed estimate archive')
                if archive['full_trajectory_saved'] is not False:raise ValueError('Samples incorrectly described as full trajectory')
                archive_manifest[archive['path']]=archive['sha256']
    if acceptances!=completion['quality_passes_by_variant']:raise ValueError('Acceptance denominator disagrees')
    result=dict(valid=True,conditions_verified=1050,expected_evaluations=3150,
        failed_fits=sum(r['status']!='ok' for r in records),nonconverged_fits=nonconverged,
        regenerated_paired_observations=105,identity_rms_controls=identity_observations,
        exact_identity_sample_archives=archive_controls,archive_samples_verified=len(archive_manifest),
        historical_executing_driver_epochs=driver_hashes,
        maximum_H_vorticity_error_difference=h_curl_max_difference,
        Newtonian_PDE_evaluations=equation_applicable,quality_passes_by_variant=acceptances,
        scientific_sources_unchanged=True,validator_sha256=benchmark.sha256(__file__),
        scope='Arithmetic/provenance/controls audit, not independent scientific replication or DNS certification.')
    benchmark.write_json(DEST/'integrity_validation.json',result)
    benchmark.write_json(DEST/'reconstruction_sample_manifest.json',archive_manifest)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
