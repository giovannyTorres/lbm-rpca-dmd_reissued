#!/usr/bin/env python3
"""Paired velocity-reconstruction comparison on seven actual 128² trajectories.

Clean reference never enters ``estimate_observation`` or ``postprocess_field``.
An append-only ledger records all failures and nonconvergence; verified hashes
and condition fingerprints make interrupted campaigns resumable.
"""
from __future__ import annotations
import os
for _thread_variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                         'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_thread_variable] = '1'
import argparse
import csv
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python/src'))
import numpy as np
from scipy.ndimage import gaussian_filter
from fluid_denoise.research.algorithms import reconstruct, ReconstructionFailure
from fluid_denoise.research.cost import measure_call, quality_gates, hardware_metadata
from fluid_denoise.research.metrics import physical_metrics, spatial_derivative
from fluid_denoise.research.physics import helmholtz_project, vorticity_residual_metrics
from fluid_denoise.research.runner import json_safe
from fluid_denoise.research.viscoplastic_dns import apparent_viscosity

DEFAULT_CONFIG = ROOT / 'configs/research/model_fidelity_benchmark.json'
DEST = ROOT / 'results/research/model_fidelity_benchmark_2026-10-14'
FLOW_FILES = {
    'ns_vortices': 'highres_newtonian_2026-10-14/datasets/interacting_vortices_128_dt0.005.npz',
    'ns_multiscale': 'highres_newtonian_2026-10-14/datasets/decaying_turbulence_128_dt0.005.npz',
    'oldroyd_b': 'highres_viscoelastic_2026-10-14/datasets/oldroyd_b_128.npz',
    'fene_p': 'highres_viscoelastic_2026-10-14/datasets/fene_p_128.npz',
    'hb_thinning': 'highres_viscoplastic_2026-10-14/datasets/hb_thinning_128.npz',
    'bingham': 'highres_viscoplastic_2026-10-14/datasets/bingham_128.npz',
    'hb_thickening': 'highres_viscoplastic_2026-10-14/datasets/hb_thickening_128.npz',
}
METHODS = {
    'identity': 'identity', 'pod_r8': 'pod', 'dmd_r8': 'dmd', 'tls_r8': 'tls_dmd',
    'pcp': 'rpca', 'stable_rpca': 'stable_rpca', 'outlier_pursuit': 'outlier_pursuit',
    'pcp_tls_r8': 'rpca_dmd', 'stable_tls_r8': 'stable_rpca_dmd',
    'optimized_r8': 'optimized_dmd',
}
METRIC_KEYS = ('velocity_relative_l2', 'vorticity_relative_l2', 'vorticity_correlation',
               'kinetic_energy_relative_l2', 'enstrophy_relative_l2',
               'extra_divergence_normalized', 'extra_vorticity_equation_residual_normalized',
               'dissipation_relative_l2', 'stress_frobenius_relative_l2',
               'apparent_viscosity_relative_l2', 'shear_rate_relative_l2')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(json_safe(value), sort_keys=True,
                         separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(json_safe(value), indent=2, ensure_ascii=False,
                                    allow_nan=False) + '\n')
    temporary.replace(path)


def load_flow(name):
    path = ROOT / 'results/research' / FLOW_FILES[name]
    sidecar = json.loads(path.with_suffix('.json').read_text())
    digest = sha256(path)
    declared = sidecar.get('npz_sha256', sidecar.get('archive_sha256', sidecar.get('dataset_sha256')))
    if digest != declared:
        raise ValueError(f'Clean input hash does not match published metadata: {path}')
    with np.load(path) as source:
        velocity = source['velocity'].copy()
        times = source['times'].copy()
        extra = {key: source[key].copy() for key in ('shear_rate', 'apparent_viscosity',
                  'deviatoric_stress') if key in source.files}
    if not np.isfinite(velocity).all() or velocity.shape[2:] != (128, 128):
        raise ValueError('Expected finite computed 128x128 velocity')
    metadata = dict(sidecar['metadata'])
    if name.startswith('ns_'):
        metadata.update(model='newtonian', viscosity=sidecar['parameters']['viscosity'])
    else:
        # Explicit discriminator prevents omission of polymer/variable-stress forces.
        metadata['model'] = name
    data = dict(velocity=velocity, times=times, dx=2*np.pi/128, dy=2*np.pi/128,
                boundary='periodic', density=1., metadata=metadata,
                mask=np.zeros((128, 128), bool), **extra)
    return data, dict(path=str(path.relative_to(ROOT)), sha256=digest,
                     metadata_sha256=sha256(path.with_suffix('.json')),
                     times=times.tolist(), shape=list(velocity.shape), source_metadata=sidecar)


def observation_noise(shape, times, kind, seed, clean_rms, config):
    """Generate then normalize one realization to the declared global RMS.

    Temporal AR is stationary OU sampling at physical time spacings; spatial
    sigma is the smoothing-kernel standard deviation, not covariance length.
    """
    rng = np.random.default_rng(seed)
    value = rng.standard_normal(shape)
    if kind == 'sparse_impulsive':
        value *= rng.random(shape) < config['sparse_probability']
    elif kind == 'column_outliers':
        count=max(1,int(np.floor(config['column_fraction_upper_bound']*shape[0])))
        support=np.zeros(shape[0],dtype=bool)
        support[rng.choice(shape[0],size=count,replace=False)]=True
        value*=support[:,None,None,None]
    elif kind == 'spatial_gaussian':
        sigma = config['spatial_kernel_sigma'] / (2*np.pi/shape[-1])
        value = gaussian_filter(value, sigma=(0, 0, sigma, sigma), mode='wrap')
    elif kind == 'temporal_ar':
        for i in range(1, shape[0]):
            correlation = np.exp(-(times[i] - times[i-1]) / config['temporal_correlation_time'])
            value[i] = correlation*value[i-1] + np.sqrt(1-correlation**2)*value[i]
    elif kind != 'gaussian_iid':
        raise ValueError(kind)
    value *= config['intensity_relative_rms'] * clean_rms / np.sqrt(np.mean(value**2))
    if not np.isfinite(value).all():
        raise FloatingPointError('Nonfinite synthetic observation')
    return value


def method_parameters(label, seed, config):
    params = dict(max_iter=config['max_iter'], tol=config['tol'])
    if 'stable' in label:
        params['noise_intensity'] = config['intensity_relative_rms']
    if label == 'outlier_pursuit':
        params['gamma'] = config['outlier_column_fraction_upper_bound']
    if label in ('pcp_tls_r8', 'stable_tls_r8'):
        params['modal_method'] = 'tls_dmd'
    if label == 'optimized_r8':
        params.update(max_nfev=config['optimized_max_nfev'],
                      n_starts=config['optimized_starts'], seed=seed)
    return params


def estimate_observation(observed, times, method, seed, config):
    """Estimator contract: there is no reference or material field argument."""
    estimate, diagnostics = reconstruct(observed.reshape(len(observed), -1).T,
        method=METHODS[method], rank=config['rank'], times=times,
        **method_parameters(method, seed, config))
    return estimate.T.reshape(observed.shape), diagnostics


def postprocess_field(value, geometry, variant, cutoff):
    if variant == 'raw':
        return value, dict(postprocess='raw', reference_velocity_used=False, relative_change=0.)
    projected, diagnostics = helmholtz_project(value, geometry['dx'], geometry['dy'])
    if variant == 'helmholtz_k10':
        kx, ky = np.meshgrid(2*np.pi*np.fft.fftfreq(value.shape[-1], d=geometry['dx']),
                             2*np.pi*np.fft.fftfreq(value.shape[-2], d=geometry['dy']))
        output = np.fft.ifft2(np.fft.fft2(projected)* (np.hypot(kx, ky) <= cutoff)).real
        diagnostics.update(postprocess='helmholtz_k10', physical_cutoff=cutoff,
            isotropic_circle=True, explicit_signal_smoothness_assumption=True,
            relative_change=float(np.linalg.norm(output-value)/max(np.linalg.norm(value), np.finfo(float).tiny)),
            mean_velocity_change_max=float(np.max(abs(output.mean(axis=(-2,-1))-value.mean(axis=(-2,-1))))))
        return output, diagnostics
    if variant != 'helmholtz':
        raise ValueError(variant)
    return projected, diagnostics


def relative_l2(reference, estimate):
    scale = np.linalg.norm(reference.ravel())
    return float(np.linalg.norm((estimate-reference).ravel())/scale) if scale > 0 else None


def gradients(value, spacing):
    ux = spatial_derivative(value[:, 0], spacing, -1, True)
    uy = spatial_derivative(value[:, 0], spacing, -2, True)
    vx = spatial_derivative(value[:, 1], spacing, -1, True)
    vy = spatial_derivative(value[:, 1], spacing, -2, True)
    return ux, uy, vx, vy


def material_observables(reference, estimate):
    """Instantaneous VP closure is identifiable from velocity; VE memory is not.

    The complete constitutive stress is derived independently from the
    candidate gradient for generalized-Newtonian flows. It is never copied
    from the clean reference and never fed to an estimator.
    """
    name = reference['metadata']['model']
    ux, uy, vx, vy = gradients(estimate, reference['dx'])
    shear = np.sqrt(2*(ux**2 + vy**2 + 2*(.5*(uy+vx))**2))
    if name == 'newtonian':
        _, ref_uy, ref_vx, _ = gradients(reference['velocity'], reference['dx'])
        ref_ux, _, _, ref_vy = gradients(reference['velocity'], reference['dx'])
        ref_shear = np.sqrt(2*(ref_ux**2 + ref_vy**2 + 2*(.5*(ref_uy+ref_vx))**2))
        nu = reference['metadata']['viscosity']
        dissipation = nu*np.sum(shear**2, axis=(-2,-1))*reference['dx']*reference['dy']
        clean_dissipation = nu*np.sum(ref_shear**2, axis=(-2,-1))*reference['dx']*reference['dy']
        return dict(dissipation_relative_l2=relative_l2(clean_dissipation, dissipation),
                    dissipation=dissipation.tolist(), reference_dissipation=clean_dissipation.tolist(),
                    constitutive_observable_scope='Newtonian instantaneous viscous dissipation')
    if name in ('oldroyd_b', 'fene_p'):
        return dict(dissipation_relative_l2=None, stress_frobenius_relative_l2=None,
            constitutive_observable_scope='Polymer memory/conformation not estimated from velocity-only low-rank reconstruction; polymer stress and total dissipation unavailable')
    metadata = reference['metadata']
    params = {key:metadata[key] for key in ('consistency', 'flow_index', 'yield_stress',
                'regularization_m', 'shear_epsilon', 'solvent_viscosity')}
    eta = apparent_viscosity(shear, **params)
    stress = 2*eta[:,None]*np.stack((ux, .5*(uy+vx), vy), axis=1)
    # Packed symmetric tensor has weights (1,2,1) in its true Frobenius norm.
    weights = np.sqrt(np.array([1.,2.,1.]))[None,:,None,None]
    dissipation = np.sum(eta*shear**2, axis=(-2,-1))*reference['dx']*reference['dy']
    clean_dissipation = np.sum(reference['apparent_viscosity']*reference['shear_rate']**2,
                              axis=(-2,-1))*reference['dx']*reference['dy']
    return dict(stress_frobenius_relative_l2=relative_l2(reference['deviatoric_stress']*weights,stress*weights),
        apparent_viscosity_relative_l2=relative_l2(reference['apparent_viscosity'], eta),
        shear_rate_relative_l2=relative_l2(reference['shear_rate'], shear),
        dissipation_relative_l2=relative_l2(clean_dissipation, dissipation),
        dissipation=dissipation.tolist(), reference_dissipation=clean_dissipation.tolist(),
        constitutive_observable_scope='Instantaneous regularized generalized-Newtonian closure applied to candidate velocity gradients; not viscoelastic polymer memory',
        stress_norm_definition='Symmetric planar Frobenius: xx^2+2*xy^2+yy^2')


def frame_errors(reference, estimate):
    spacing=reference['dx']
    clean=reference['velocity']
    omega=lambda v: spatial_derivative(v[:,1], spacing,-1,True)-spatial_derivative(v[:,0],spacing,-2,True)
    clean_w=omega(clean); estimated_w=omega(estimate)
    result = {'times':reference['times'].tolist()}
    for key, actual, predicted in [('velocity_relative_l2', clean, estimate),
                                    ('vorticity_relative_l2', clean_w, estimated_w)]:
        values=[relative_l2(a,b) for a,b in zip(actual,predicted)]
        result[key]=values
        result[key+'_max_saved']=max(v for v in values if v is not None)
    return result


def worker(connection, observed, reference, condition, config, save_folder):
    try:
        (value, diagnostics), profile = measure_call(estimate_observation,
            observed, reference['times'], condition['method'], condition['seed'], config)
        evaluations=[]
        for variant in config['postprocessing']:
            (output, projection), projection_profile=measure_call(postprocess_field,
                value, {'dx':reference['dx'], 'dy':reference['dy']}, variant, config['physical_cutoff'])
            metrics=physical_metrics(reference, output)
            metrics.update(vorticity_residual_metrics(reference, output))
            metrics.update(material_observables(reference, output))
            error_series=frame_errors(reference, output)
            gates=quality_gates(metrics, config['quality_thresholds'])
            archive=None
            if save_folder is not None:
                path=Path(save_folder)/(condition['id']+'__'+variant+'.npz')
                indices=np.array([0,len(output)//2,len(output)-1],dtype=int)
                np.savez_compressed(path, velocity=output[indices], times=reference['times'][indices],
                                    frame_indices=indices)
                archive=dict(path=str(path.relative_to(ROOT)),sha256=sha256(path),bytes=path.stat().st_size,
                    frame_indices=indices.tolist(), full_trajectory_saved=False,
                    sampling_scope='Three exact saved states, no interpolation. All-time error series remain in ledger.')
            evaluations.append(dict(variant=variant, metrics=metrics, frame_errors=error_series,
                quality=gates, quality_pass=bool(gates['passed'] and diagnostics['converged']),
                projection=projection, postprocess_profile=projection_profile,
                reconstruction_archive=archive))
        connection.send(json_safe(dict(status='ok', diagnostics=diagnostics,
                         fit_profile=profile, evaluations=evaluations)))
    except Exception as error:
        connection.send(dict(status='failed', failure_type=type(error).__name__,
            failure=str(error), diagnostics=getattr(error,'diagnostics',{}),
            traceback=traceback.format_exc()))
    finally:
        connection.close()


def isolated(observed, reference, condition, config, save_folder):
    context=mp.get_context('fork' if 'fork' in mp.get_all_start_methods() else 'spawn')
    parent,child=context.Pipe(duplex=False)
    process=context.Process(target=worker,args=(child, observed, reference, condition,config,save_folder))
    started=time.monotonic();process.start();child.close()
    try:
        while time.monotonic()-started < config['timeout_seconds']:
            if parent.poll(.1):
                result=parent.recv();process.join(5)
                result['condition_wall_seconds_including_evaluation']=time.monotonic()-started
                return result
            if not process.is_alive():
                return dict(status='failed',failure_type='worker_exit',failure=f'Exit {process.exitcode}')
        return dict(status='failed', failure_type='timeout',
                    failure=f"Whole condition budget {config['timeout_seconds']} seconds",
                    condition_wall_seconds_including_evaluation=time.monotonic()-started)
    finally:
        if process.is_alive():process.terminate()
        process.join();parent.close()


def run_batch(observed, reference, pending, config, save_folder, workers):
    """At most two fresh children; no process pool reuses estimator state."""
    context=mp.get_context('fork' if 'fork' in mp.get_all_start_methods() else 'spawn')
    queue=list(pending);active={}
    try:
        while queue or active:
            while queue and len(active)<workers:
                condition,condition_hash=queue.pop(0)
                parent,child=context.Pipe(duplex=False)
                process=context.Process(target=worker,args=(child,observed,reference,
                    condition,config,save_folder))
                started=time.monotonic();process.start();child.close()
                active[condition['id']]=(process,parent,started,condition,condition_hash)
            ready=[]
            for identifier,(process,parent,started,condition,condition_hash) in active.items():
                if parent.poll():
                    try: result=parent.recv()
                    except EOFError:result=dict(status='failed',failure_type='worker_pipe_closed',failure=f'Exit {process.exitcode}')
                    ready.append((identifier,result))
                elif not process.is_alive():
                    ready.append((identifier,dict(status='failed',failure_type='worker_exit',failure=f'Exit {process.exitcode}')))
                elif time.monotonic()-started>=config['timeout_seconds']:
                    ready.append((identifier,dict(status='failed',failure_type='timeout',
                        failure=f"Whole condition budget {config['timeout_seconds']} seconds")))
            for identifier,result in ready:
                process,parent,started,condition,condition_hash=active.pop(identifier)
                if process.is_alive() and result.get('failure_type')=='timeout':process.terminate()
                process.join(5)
                if process.is_alive():process.terminate();process.join()
                parent.close()
                result['condition_wall_seconds_including_evaluation']=time.monotonic()-started
                yield condition,condition_hash,result
            if active and not ready:time.sleep(.02)
    finally:
        for process,parent,*_ in active.values():
            if process.is_alive():process.terminate()
            process.join();parent.close()


def summarize(records, destination, expected):
    rows=[]
    for record in records:
        common={key:record[key] for key in ('flow','noise','seed','method','id','status')}
        diagnostics=record.get('diagnostics',{})
        common.update(converged=bool(diagnostics.get('converged',False)),
                      effective_rank=diagnostics.get('effective_rank'),
                      iterations=diagnostics.get('iterations'),
                      svd_calls=diagnostics.get('svd_calls'),
                      flops_proxy=diagnostics.get('flops_proxy'),
                      failure_type=record.get('failure_type'),failure=record.get('failure'))
        common.update({key:record.get('fit_profile',{}).get(key)
                       for key in ('wall_seconds','cpu_seconds','rss_peak_bytes','python_traced_peak_bytes')})
        evaluations=record.get('evaluations',[]) or [dict(variant=v) for v in ('raw','helmholtz','helmholtz_k10')]
        for evaluation in evaluations:
            row=dict(common, variant=evaluation['variant'], quality_pass=evaluation.get('quality_pass',False))
            row.update({key:evaluation.get('metrics',{}).get(key) for key in METRIC_KEYS})
            rows.append(row)
    write_json(destination/'summary.json',rows)
    if rows:
        with (destination/'summary.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    groups=[]
    for flow in FLOW_FILES:
        for noise in ('gaussian_iid','sparse_impulsive','spatial_gaussian','temporal_ar','column_outliers'):
            for method in METHODS:
                for variant in ('raw','helmholtz','helmholtz_k10'):
                    sample=[r for r in rows if (r['flow'],r['noise'],r['method'],r['variant'])==(flow,noise,method,variant)]
                    if not sample:continue
                    group=dict(flow=flow,noise=noise,method=method,variant=variant,
                        conditions=len(sample), failures=sum(r['status']!='ok' for r in sample),
                        nonconverged=sum(r['status']=='ok' and not r['converged'] for r in sample),
                        quality_passes=sum(r['quality_pass'] for r in sample))
                    for metric in METRIC_KEYS+('wall_seconds','cpu_seconds','rss_peak_bytes','flops_proxy'):
                        values=[r[metric] for r in sample if r[metric] is not None and r['status']=='ok']
                        group[metric+'_median']=float(np.median(values)) if values else None
                        group[metric+'_min']=min(values) if values else None
                        group[metric+'_max']=max(values) if values else None
                    groups.append(group)
    write_json(destination/'group_summary.json',groups)
    completion=dict(complete=len(records)==expected, expected_fits=expected, recorded_fits=len(records),
        expected_evaluations=3*expected, recorded_evaluations=sum(len(r.get('evaluations',[])) for r in records),
        failed_fits=sum(r['status']!='ok' for r in records),
        nonconverged_fits=sum(r['status']=='ok' and not r['diagnostics']['converged'] for r in records),
        quality_passes_by_variant={v:sum(r['quality_pass'] for r in rows if r['variant']==v)
                                 for v in ('raw','helmholtz','helmholtz_k10')},
        elapsed_conditions_seconds=sum(r.get('condition_wall_seconds_including_evaluation',0) for r in records),
        scope='In-window paired velocity reconstruction; fixed reference density gives trivial total mass; sampled-time PDE residual only Newtonian; no blind or forecast certification')
    write_json(destination/'completion.json',completion)
    return completion


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--summarize-only',action='store_true')
    parser.add_argument('--workers',type=int,choices=(1,2),default=2)
    parser.add_argument('--flows',nargs='+',choices=tuple(FLOW_FILES))
    parser.add_argument('--methods',nargs='+',choices=tuple(METHODS))
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    destination=DEST/'pilot' if args.pilot else DEST
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'reconstructions').mkdir(exist_ok=True)
    ledger=destination/'ledger.jsonl'
    names=args.flows or (['ns_vortices'] if args.pilot else list(FLOW_FILES))
    methods=args.methods or (['tls_r8','pcp','stable_rpca','outlier_pursuit','optimized_r8'] if args.pilot else list(METHODS))
    noises=['gaussian_iid'] if args.pilot else config['noise_models']
    seeds=[7] if args.pilot else config['seeds']
    expected=len(names)*len(methods)*len(noises)*len(seeds)
    records=[]
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            if line.strip():records.append(json.loads(line))
    if args.summarize_only:
        print(json.dumps(summarize(records,destination,expected),indent=2));return
    sources={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'python/src/fluid_denoise/research').glob('*.py')}
    provenance=dict(configuration=config,configuration_sha256=sha256(args.config),
        estimator_source_hashes=sources,driver_sha256=sha256(__file__),hardware=hardware_metadata(),
        selected_flows=names,selected_methods=methods,selected_noises=noises,selected_seeds=seeds,
        design_note=config['design'],cost_note=config['cost_scope'], clean_input_sources={})
    campaign_hash=fingerprint(dict(configuration=config,estimator_source_hashes=sources,
                                  selected_flows=names,selected_methods=methods,selected_noises=noises,selected_seeds=seeds))
    existing={r['id']:r for r in records}
    if len(existing)!=len(records):raise ValueError('Duplicated ledger conditions')
    for name in names:
        dataset,source=load_flow(name)
        provenance['clean_input_sources'][name]=source
        write_json(destination/'provenance.json',provenance)
        clean_rms=float(np.sqrt(np.mean(dataset['velocity']**2)))
        for noise in noises:
            for seed in seeds:
                perturbation=observation_noise(dataset['velocity'].shape,dataset['times'],noise,seed,clean_rms,config)
                observed=dataset['velocity']+perturbation
                noise_meta=dict(realized_relative_rms=float(np.sqrt(np.mean(perturbation**2))/clean_rms),
                    nonzero_fraction=float(np.count_nonzero(perturbation)/perturbation.size),
                    contaminated_snapshot_indices=np.where(np.any(perturbation!=0,axis=(1,2,3)))[0].tolist() if noise=='column_outliers' else None,
                    observation_sha256=hashlib.sha256(observed.tobytes()).hexdigest(),
                    injection_reference_used_for_amplitude_only=True,
                    spatial_sigma_definition='Gaussian smoothing kernel standard deviation, not autocorrelation standard deviation',
                    temporal_model='Stationary AR1 sampled as exp(-dt/.30); global RMS normalization preserves correlation')
                pending=[]
                for method in methods:
                    identifier=f'{name}__{noise}__s{seed}__{method}'
                    condition=dict(id=identifier,flow=name,noise=noise,seed=seed,method=method)
                    condition_hash=fingerprint(dict(campaign=campaign_hash,condition=condition,
                        clean_input_sha256=source['sha256'],observation_sha256=noise_meta['observation_sha256']))
                    if identifier in existing:
                        if existing[identifier]['fingerprint']!=condition_hash:
                            raise ValueError(f'Condition changed: {identifier}; choose a new destination rather than mixing ledgers')
                        continue
                    pending.append((condition,condition_hash))
                save=destination/'reconstructions' if seed==config['seeds'][0] and not args.pilot else None
                for condition,condition_hash,record in run_batch(observed,dataset,pending,config,str(save) if save else None,
                                                               1 if args.pilot else args.workers):
                    record.update(condition,fingerprint=condition_hash,noise_metadata=noise_meta,
                                  clean_input_sha256=source['sha256'],driver_sha256=provenance['driver_sha256'])
                    with ledger.open('a') as handle:
                        handle.write(json.dumps(json_safe(record),allow_nan=False)+'\n');handle.flush();os.fsync(handle.fileno())
                    records.append(record)
                    print(f'{len(records)}/{expected} {condition["id"]} {record["status"]} converged={record.get("diagnostics",{}).get("converged")} fit={record.get("fit_profile",{}).get("wall_seconds",0):.3f}s total={record.get("condition_wall_seconds_including_evaluation",0):.3f}s',flush=True)
                    if len(records)%10==0:summarize(records,destination,expected)
    final_sources={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'python/src/fluid_denoise/research').glob('*.py')}
    if final_sources!=sources:raise ValueError('Scientific source modules changed during campaign')
    print(json.dumps(summarize(records,destination,expected),indent=2),flush=True)


if __name__=='__main__':main()
