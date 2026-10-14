#!/usr/bin/env python3
"""Fixed rank/noise sweep, physical fidelity, and separate clean POD diagnostic.

This campaign never chooses rank from reference error. It uses exactly the
paired-model benchmark's noise and estimator contracts. Every fitted condition
has a fresh process and three geometry-only postprocessing evaluations.
"""
from __future__ import annotations
import os
for _variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                  'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_variable] = '1'
import argparse
import csv
import hashlib
import inspect
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python/src'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import run_model_fidelity_benchmark as paired
from fluid_denoise.research.cost import measure_call, quality_gates, hardware_metadata
from fluid_denoise.research.metrics import physical_metrics
from fluid_denoise.research.runner import json_safe

DEFAULT_CONFIG = ROOT / 'configs/research/rank_noise_sensitivity.json'
DEST = ROOT / 'results/research/rank_noise_sensitivity_2026-10-14'
METRICS = ('velocity_relative_l2', 'vorticity_relative_l2', 'vorticity_correlation',
           'extra_divergence_normalized', 'kinetic_energy_relative_l2',
           'enstrophy_relative_l2')
METHOD_LABELS = {'pod_r8': 'POD', 'tls_r8': 'TLS-DMD'}
COLORS = {'pod_r8': '#1965b0', 'tls_r8': '#b85b25'}
FLOW_LABELS = {'ns_multiscale': 'NS multiescala', 'oldroyd_b': 'Oldroyd-B'}
NOISE_LABELS = {'gaussian_iid': 'Gaussiano IID', 'sparse_impulsive': 'Impulsivo: 1% de entradas'}


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def condition_identity(flow, noise, intensity, seed, rank, method, observed, config):
    estimator_config = dict(config, rank=rank, intensity_relative_rms=intensity)
    return dict(flow=flow, noise=noise, intensity_relative_rms=intensity, seed=seed,
        rank=rank, method=method, observed_sha256=array_sha(observed),
        algorithm=paired.METHODS[method],
        algorithm_parameters=paired.method_parameters(method, seed, estimator_config))


def fit_worker(connection, observed, times, reference, method, seed, config, selected_path):
    """Reference is evaluated only after the observed-only estimator returns."""
    try:
        (estimate, diagnostics), fit_profile = measure_call(
            paired.estimate_observation, observed, times, method, seed, config)
        evaluations = []
        geometry = {key: reference[key] for key in ('dx', 'dy', 'boundary')}
        for variant in config['postprocessing']:
            (value, post_diagnostics), post_profile = measure_call(
                paired.postprocess_field, estimate, geometry, variant,
                config['physical_cutoff'])
            metrics = physical_metrics(reference, value)
            gates = quality_gates(metrics, config['quality_thresholds'])
            evaluations.append(dict(postprocessing=variant, metrics=metrics,
                postprocessing_diagnostics=post_diagnostics,
                postprocessing_profile=post_profile, quality=gates,
                accepted=bool(gates['passed']) and bool(diagnostics.get('converged'))))
            if selected_path is not None and variant == 'helmholtz_k10':
                np.savez_compressed(selected_path, velocity=value, times=times)
        connection.send(json_safe(dict(status='ok', diagnostics=diagnostics,
            fit_profile=fit_profile, evaluations=evaluations)))
    except Exception as error:
        connection.send(json_safe(dict(status='failed', failure=str(error),
            failure_type=type(error).__name__, traceback=traceback.format_exc(),
            diagnostics=getattr(error, 'diagnostics', {}), evaluations=[])))
    finally:
        connection.close()


def isolated_fit(observed, dataset, method, seed, config, selected_path):
    context = mp.get_context('fork' if 'fork' in mp.get_all_start_methods() else 'spawn')
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=fit_worker, args=(child, observed,
        dataset['times'], dataset, method, seed, config, selected_path))
    process.start()
    child.close()
    try:
        deadline = time.monotonic() + config['timeout_seconds']
        while time.monotonic() < deadline:
            if parent.poll(.2):
                try:
                    result = parent.recv()
                except EOFError:
                    result = dict(status='failed', failure='Worker closed without result', evaluations=[])
                process.join(5)
                return result
            if not process.is_alive():
                return dict(status='failed', failure=f'Worker exit {process.exitcode}', evaluations=[])
        return dict(status='failed', failure='Condition timeout', evaluations=[])
    finally:
        if process.is_alive():
            process.terminate()
        process.join()
        parent.close()


def write_table(path, rows):
    fields = list(rows[0]) if rows else []
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def table_rows(records, config):
    rows = []
    for record in records:
        identity = record['condition']
        available = {value['postprocessing']: value for value in record['evaluations']}
        for variant in config['postprocessing']:
            value = available.get(variant, {})
            metrics = value.get('metrics', {})
            profile = record.get('fit_profile', {})
            post_profile = value.get('postprocessing_profile', {})
            rows.append({key: identity[key] for key in
                ('flow', 'noise', 'intensity_relative_rms', 'seed', 'rank', 'method', 'observed_sha256')}
                | dict(postprocessing=variant, status=record['status'],
                    converged=bool(record.get('diagnostics', {}).get('converged')),
                    accepted=bool(value.get('accepted')), failure=record.get('failure'),
                    fit_cpu_seconds=profile.get('cpu_seconds'),
                    fit_wall_seconds=profile.get('wall_seconds'),
                    fit_rss_peak_bytes=profile.get('rss_peak_bytes'),
                    post_cpu_seconds=post_profile.get('cpu_seconds'),
                    post_wall_seconds=post_profile.get('wall_seconds'),
                    post_rss_peak_bytes=post_profile.get('rss_peak_bytes'))
                | {key: metrics.get(key) for key in METRICS})
    return rows


def aggregate(rows):
    groups = {}
    group_keys = ('flow', 'noise', 'intensity_relative_rms', 'rank', 'method', 'postprocessing')
    for row in rows:
        key = tuple(row[name] for name in group_keys)
        groups.setdefault(key, []).append(row)
    output = []
    for key, samples in groups.items():
        row = dict(zip(group_keys, key), seeds=[sample['seed'] for sample in samples],
                   sample_count=len(samples), successful_count=sum(s['status']=='ok' for s in samples),
                   accepted_count=sum(s['accepted'] for s in samples))
        for metric in METRICS:
            values = [sample[metric] for sample in samples if sample[metric] is not None]
            row[metric] = dict(median=float(np.median(values)), minimum=float(min(values)),
                maximum=float(max(values)), count=len(values)) if values else None
        output.append(row)
    return output


def clean_oracle(datasets, config):
    """Clean rank-r POD is diagnostic only; derivatives have no optimality claim."""
    rows = []
    for name, dataset in datasets.items():
        clean = dataset['velocity']
        matrix = clean.reshape(len(clean), -1).T
        u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
        geometry = {key: dataset[key] for key in ('dx', 'dy', 'boundary')}
        clean_band, _ = paired.postprocess_field(clean, geometry, 'helmholtz_k10', config['physical_cutoff'])
        band_metrics = physical_metrics(dataset, clean_band)
        for rank in config['ranks']:
            value = ((u[:, :rank]*singular[:rank])@vh[:rank]).T.reshape(clean.shape)
            metrics = physical_metrics(dataset, value)
            direct_bound = np.sqrt(np.sum(singular[rank:]**2)/np.sum(singular**2))
            if not np.isclose(metrics['velocity_relative_l2'], direct_bound, atol=2e-14, rtol=1e-10):
                raise AssertionError('Clean SVD Frobenius lower bound mismatch')
            rows.append(dict(flow=name, rank=rank,
                velocity_rank_frobenius_lower_bound=float(direct_bound),
                vorticity_clean_pod_diagnostic=metrics['vorticity_relative_l2'],
                clean_band_velocity_error=band_metrics['velocity_relative_l2'],
                clean_band_vorticity_error=band_metrics['vorticity_relative_l2'],
                singular_values=singular.tolist(),
                clean_pod_energy_fraction=float(np.sum(singular[:rank]**2)/np.sum(singular**2)),
                oracle_reference_used=True, estimator_input=False,
                qualification='Velocity Frobenius rank lower bound; vorticity only diagnostic. No noisy rank selection.'))
    return rows


def subset(groups, **where):
    return sorted((row for row in groups if all(row[key] == value for key, value in where.items())),
                  key=lambda row: row['rank'])


def draw_curves(groups, config, metric, filename, post='helmholtz_k10'):
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.1), layout='constrained')
    styles = {0.05: ':', 0.2: '-', 0.4: '--'}
    for row, flow in enumerate(config['flows']):
        for col, noise in enumerate(config['noise_models']):
            axis = axes[row, col]
            for method in config['methods']:
                for intensity in config['intensities_relative_rms']:
                    series = subset(groups, flow=flow, noise=noise, method=method,
                        intensity_relative_rms=intensity, postprocessing=post)
                    valid = [sample for sample in series if sample[metric] is not None]
                    if not valid: continue
                    x = [s['rank'] for s in valid]
                    median = np.array([s[metric]['median'] for s in valid])*100
                    lo = np.array([s[metric]['minimum'] for s in valid])*100
                    hi = np.array([s[metric]['maximum'] for s in valid])*100
                    axis.plot(x, median, color=COLORS[method], ls=styles[intensity], marker='o',
                              markersize=4, label=f'{METHOD_LABELS[method]}, RMS {int(100*intensity)}%')
                    axis.fill_between(x, lo, hi, color=COLORS[method], alpha=.09)
            axis.set_title(f'{FLOW_LABELS[flow]} — {NOISE_LABELS[noise]}', fontsize=12)
            axis.set_xlabel('Rango fijado r');axis.set_ylabel('Error L2 relativo [%]')
            axis.set_xticks(config['ranks']);axis.grid(alpha=.2)
            axis.axhline(15 if metric=='velocity_relative_l2' else 25, color='#333333', ls='--', lw=.8)
    axes[0,0].legend(ncol=2, fontsize=8.5)
    fig.suptitle(('Fidelidad de velocidad' if metric=='velocity_relative_l2' else 'Fidelidad de vorticidad')
        +' frente a rango e intensidad\nH + corte físico k≤10; mediana y rango de tres semillas, sin intervalo de confianza', fontsize=13)
    for suffix in ('pdf', 'png'): fig.savefig(DEST/'figures'/f'{filename}.{suffix}')
    plt.close(fig)


def figures(groups, oracle, config):
    plt.rcParams.update({'font.size':11, 'axes.spines.top':False,
        'axes.spines.right':False, 'pdf.fonttype':42, 'savefig.dpi':170})
    draw_curves(groups, config, 'velocity_relative_l2', 'rank_velocity')
    draw_curves(groups, config, 'vorticity_relative_l2', 'rank_vorticity')
    fig, axes = plt.subplots(2,2, figsize=(11.4,8.1), layout='constrained')
    post_styles = {'raw':(':','Raw'), 'helmholtz':('--','H'), 'helmholtz_k10':('-','H+k≤10')}
    for row, flow in enumerate(config['flows']):
        for col, noise in enumerate(config['noise_models']):
            axis = axes[row,col]
            for method in config['methods']:
                for post, (style, label) in post_styles.items():
                    series = subset(groups, flow=flow, noise=noise, method=method,
                         intensity_relative_rms=.2, postprocessing=post)
                    values = [s['vorticity_relative_l2'] for s in series]
                    if any(v is None for v in values):continue
                    median = np.array([v['median'] for v in values])*100
                    lo=np.array([v['minimum'] for v in values])*100;hi=np.array([v['maximum'] for v in values])*100
                    axis.plot(config['ranks'], median, color=COLORS[method], ls=style,
                        marker='o', markersize=4,label=f'{METHOD_LABELS[method]}, {label}')
                    axis.fill_between(config['ranks'], lo, hi, color=COLORS[method], alpha=.07)
            axis.axhline(25,color='#333333',ls='--',lw=.8);axis.set_yscale('log')
            axis.set_xticks(config['ranks']);axis.set_xlabel('Rango fijado r')
            axis.set_ylabel('Error de vorticidad [%]');axis.grid(alpha=.2)
            axis.set_title(f'{FLOW_LABELS[flow]} — {NOISE_LABELS[noise]}',fontsize=12)
    axes[0,0].legend(ncol=2,fontsize=8.5)
    fig.suptitle('Separar algoritmo, incomprensibilidad y suavizado: RMS 20%\nRaw y H conservan la misma vorticidad; mediana/rango de tres semillas',fontsize=13)
    for suffix in ('pdf','png'):fig.savefig(DEST/'figures'/f'rank_postprocessing.{suffix}')
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11.4,7.4),layout='constrained')
    labels=[f'{METHOD_LABELS[m]}, {int(100*i)}%' for m in config['methods'] for i in config['intensities_relative_rms']]
    for row,flow in enumerate(config['flows']):
        for col,noise in enumerate(config['noise_models']):
            axis=axes[row,col];matrix=np.zeros((len(labels),len(config['ranks'])))
            for i,(method,intensity) in enumerate((m,j) for m in config['methods'] for j in config['intensities_relative_rms']):
                series=subset(groups,flow=flow,noise=noise,method=method,
                    intensity_relative_rms=intensity,postprocessing='helmholtz_k10')
                for j,sample in enumerate(series):matrix[i,j]=sample['accepted_count']
            picture=axis.imshow(matrix,vmin=0,vmax=3,cmap='YlGnBu',aspect='auto')
            for (i,j),value in np.ndenumerate(matrix):axis.text(j,i,f'{int(value)}/3',ha='center',va='center',color='white' if value>=2 else 'black',fontsize=11)
            axis.set_xticks(range(len(config['ranks'])),config['ranks']);axis.set_yticks(range(len(labels)),labels)
            axis.set_xlabel('Rango fijado r');axis.set_title(f'{FLOW_LABELS[flow]} — {NOISE_LABELS[noise]}',fontsize=12)
    fig.colorbar(picture,ax=axes,label='Semillas que cumplen los cuatro umbrales',ticks=[0,1,2,3],shrink=.8)
    fig.suptitle('Rangos e intensidades aceptados: H + corte físico k≤10\nNo se selecciona el mínimo de error de la referencia; tres realizaciones por celda',fontsize=13)
    for suffix in ('pdf','png'):fig.savefig(DEST/'figures'/f'acceptance_rank_noise.{suffix}')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11.4,4.6),layout='constrained')
    for axis,flow in zip(axes,config['flows']):
        series=[row for row in oracle if row['flow']==flow]
        for key,label,color,style in (
            ('velocity_rank_frobenius_lower_bound','u: cota Frobenius de rango','#1965b0','-'),
            ('vorticity_clean_pod_diagnostic','ω: diagnóstico POD limpio','#b85b25','-'),
            ('clean_band_velocity_error','u: sólo corte k≤10','#1965b0',':'),
            ('clean_band_vorticity_error','ω: sólo corte k≤10','#b85b25',':')):
            axis.plot(config['ranks'],[100*s[key] for s in series],color=color,ls=style,marker='o',label=label)
        axis.set_yscale('log');axis.set_xlabel('Rango fijado r');axis.set_ylabel('Error relativo [%]')
        axis.set_xticks(config['ranks']);axis.grid(alpha=.2);axis.set_title(FLOW_LABELS[flow])
    axes[0].legend(fontsize=9)
    fig.suptitle('Oráculo limpio separado: compresibilidad y sesgo del prior\nDiagnóstico: no entra en estimadores ni elige rango; ω no es una cota de optimalidad',fontsize=13)
    for suffix in ('pdf','png'):fig.savefig(DEST/'figures'/f'clean_oracle_floor.{suffix}')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--render-only',action='store_true')
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    for subdirectory in ('cache','figures','reconstructions'):(DEST/subdirectory).mkdir(parents=True,exist_ok=True)
    datasets={};source_inputs={}
    for flow in config['flows']:datasets[flow],source_inputs[flow]=paired.load_flow(flow)
    helper_names=('load_flow','observation_noise','estimate_observation','method_parameters','postprocess_field')
    helper_sources={name:inspect.getsource(getattr(paired,name)) for name in helper_names}
    scientific_sources={str(path.relative_to(ROOT)):paired.sha256(path)
        for path in (ROOT/'python/src/fluid_denoise/research').glob('*.py')}
    design_fingerprint=paired.fingerprint(dict(config=config,source_inputs=source_inputs,
        helper_sources=helper_sources,scientific_sources=scientific_sources))
    provenance=dict(configuration=config,design_fingerprint=design_fingerprint,
        script_sha256=paired.sha256(__file__), helper_contract_sources=helper_sources,
        helper_module_sha256_at_start=paired.sha256(paired.__file__),
        source_files_sha256=scientific_sources, source_datasets=source_inputs,
        hardware=hardware_metadata(), created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        cost_warning='Fresh fit child per condition, postprocess measured separately. Absolute RSS includes inherited input/runtime. One repetition on shared host; do not rank precise wall timings.',
        independent_evidence='New noise seeds on two existing trajectories, not new physical flows. Rank8/intensity20 conditions overlap model_fidelity_benchmark; observed SHA and parameters identify duplicates.')
    if args.render_only:
        records=[json.loads(line) for line in (DEST/'ledger.jsonl').read_text().splitlines()]
        previous=json.loads((DEST/'provenance.json').read_text())
        if previous['design_fingerprint']!=design_fingerprint:raise ValueError('Render-only design changed')
        oracle=json.loads((DEST/'oracle_floor.json').read_text())
    else:
        paired.write_json(DEST/'provenance.json',provenance)
        oracle=clean_oracle(datasets,config)
        paired.write_json(DEST/'oracle_floor.json',oracle)
        oracle_table=[{key:value for key,value in row.items() if key!='singular_values'} for row in oracle]
        write_table(DEST/'oracle_floor.csv',oracle_table)
        records=[]
        with (DEST/'ledger.jsonl').open('w') as ledger:
            for flow,dataset in datasets.items():
                clean_rms=float(np.sqrt(np.mean(dataset['velocity']**2)))
                for noise in config['noise_models']:
                    for intensity in config['intensities_relative_rms']:
                        estimator_config=dict(config,intensity_relative_rms=intensity)
                        for seed in config['seeds']:
                            perturbation=paired.observation_noise(dataset['velocity'].shape,dataset['times'],noise,seed,clean_rms,estimator_config)
                            observed=dataset['velocity']+perturbation
                            for rank in config['ranks']:
                                condition_config=dict(estimator_config,rank=rank)
                                for method in config['methods']:
                                    identity=condition_identity(flow,noise,intensity,seed,rank,method,observed,condition_config)
                                    fingerprint=paired.fingerprint(dict(design=design_fingerprint,condition=identity))
                                    cache=DEST/'cache'/f'{fingerprint}.json'
                                    record=json.loads(cache.read_text()) if args.resume and cache.exists() else None
                                    if record is not None and record.get('condition_fingerprint')!=fingerprint:raise ValueError('Invalid condition cache')
                                    if record is None:
                                        selected=DEST/'reconstructions'/f'{flow}_{noise}_i{int(100*intensity)}_s{seed}_r{rank}_{method}.npz' if seed==101 and rank==8 and intensity==.2 else None
                                        record=isolated_fit(observed,dataset,method,seed,condition_config,str(selected) if selected else None)
                                        record.update(condition=identity,condition_fingerprint=fingerprint,
                                            overlap_model_benchmark=bool(rank==8 and intensity==.2),
                                            exact_relative_noise_rms=float(np.sqrt(np.mean(perturbation**2))/clean_rms))
                                        paired.write_json(cache,record)
                                    records.append(record)
                                    ledger.write(json.dumps(json_safe(record),ensure_ascii=False,allow_nan=False)+'\n');ledger.flush()
                                    print(f'{len(records)}/360 {flow} {noise} RMS{int(100*intensity)} s{seed} r{rank} {method}: {record["status"]}',flush=True)
    rows=table_rows(records,config)
    groups=aggregate(rows)
    paired.write_json(DEST/'summary.json',rows);write_table(DEST/'summary.csv',rows)
    paired.write_json(DEST/'aggregates.json',groups)
    figures(groups,oracle,config)
    expected=len(config['flows'])*len(config['noise_models'])*len(config['intensities_relative_rms'])*len(config['seeds'])*len(config['ranks'])*len(config['methods'])
    paired.write_json(DEST/'completion.json',dict(complete=len(records)==expected,
        expected_fits=expected,fits=len(records),evaluations=len(rows),
        failed_fits=sum(r['status']!='ok' for r in records),
        nonconverged_fits=sum(r['status']=='ok' and not r['diagnostics'].get('converged') for r in records),
        accepted_evaluations=sum(r['accepted'] for r in rows),
        accepted_by_postprocessing={post:sum(r['accepted'] for r in rows if r['postprocessing']==post) for post in config['postprocessing']},
        overlap_model_benchmark_fits=sum(r.get('overlap_model_benchmark',False) for r in records),
        interpretation='360 distinct noisy/rank/method fits; three postprocess evaluations per fit. Overlap with main model benchmark is not independent evidence.'))
    print(DEST,flush=True)


if __name__=='__main__':main()
