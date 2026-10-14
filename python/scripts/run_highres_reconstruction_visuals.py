#!/usr/bin/env python3
"""Contrast grid-dependent observation noise and physics in actual large fields.

This is in-window reconstruction, with predeclared ranks and three noise seeds.
It does not select models using clean fields or estimate a forecasting skill.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
import csv
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'python/src'))
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import resample
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from fluid_denoise.research.algorithms import reconstruct
from fluid_denoise.research.cost import measure_call,quality_gates
from fluid_denoise.research.metrics import physical_metrics,spatial_derivative
from fluid_denoise.research.physics import apply_postprocessing
from fluid_denoise.research.runner import assert_finite,json_safe,write_json

DEST=ROOT/'results/research/highres_reconstruction_2026-10-14'
SOURCE=ROOT/'results/research/highres_newtonian_2026-10-14/datasets'
GRIDS=(128,256)
SEEDS=(7,23,61)
METHODS=(('identity','identity',1,'none'),('pod_r8','pod',8,'none'),
         ('tls_r8','tls_dmd',8,'none'),('tls_r8_projected','tls_dmd',8,'helmholtz_dealias'),
         ('tls_r16_projected','tls_dmd',16,'helmholtz_dealias'))
NOISES=('gaussian_iid','spatial_fixed_length')
THRESHOLDS={'velocity_relative_l2':.15,'vorticity_relative_l2':.25,
            'extra_divergence_normalized':.1,'kinetic_energy_relative_l2':.15}
COLORS={'identity':'#666666','pod_r8':'#1965b0','tls_r8':'#b38b24',
        'tls_r8_projected':'#b85b25','tls_r16_projected':'#6e7d30'}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42,'savefig.dpi':180})


def load_dataset(n):
    path=SOURCE/f'interacting_vortices_{n}_dt0.005.npz'
    side=json.loads(path.with_suffix('.json').read_text())
    with np.load(path) as arrays:
        velocity=arrays['velocity'].copy();times=arrays['times'].copy()
    if hashlib.sha256(path.read_bytes()).hexdigest()!=side['npz_sha256']:
        raise ValueError(f'Changed simulation: {path}')
    return dict(velocity=velocity,times=times,dx=2*np.pi/n,dy=2*np.pi/n,
                boundary='periodic',density=1.,mask=np.zeros((n,n),bool),metadata=side['metadata'])


def noise_field(reference,kind,seed):
    rng=np.random.default_rng(seed)
    value=rng.standard_normal(reference.shape)
    if kind=='spatial_fixed_length':
        # One actual physical realization on 256 is Fourier-transferred to 128.
        # sigma=.10 in domain units, unchanged across observation grids.
        value=gaussian_filter(value,sigma=(0,0,.10/(2*np.pi/256),.10/(2*np.pi/256)),mode='wrap')
    value*=.2*np.sqrt(np.mean(reference**2))/np.sqrt(np.mean(value**2))
    return value


def omega(value,spacing):
    return (spatial_derivative(value[:,1],spacing,-1,True)
            -spatial_derivative(value[:,0],spacing,-2,True))


def fit(observed,dataset,method,rank,postprocess):
    matrix=observed.reshape(len(observed),-1).T
    result,diagnostics=reconstruct(matrix,method=method,rank=rank,times=dataset['times'])
    result=result.T.reshape(observed.shape)
    result,projection=apply_postprocessing(dataset,result,postprocess)
    assert_finite(result)
    return result,{**diagnostics,'postprocessing':projection}


def worker(connection,observed,dataset,specification,path):
    try:
        label,method,rank,postprocess=specification
        (result,diagnostics),profile=measure_call(fit,observed,dataset,method,rank,postprocess)
        metrics=physical_metrics(dataset,result)
        gates=quality_gates(metrics,THRESHOLDS)
        if path is not None:
            np.savez_compressed(path,velocity=result,times=dataset['times'])
        connection.send(json_safe({'status':'ok','diagnostics':diagnostics,'profile':profile,
                                   'metrics':metrics,'quality':gates}))
    except Exception as error:
        connection.send({'status':'failed','failure':str(error)})
    finally:
        connection.close()


def isolated(observed,dataset,specification,path):
    context=mp.get_context('fork' if 'fork' in mp.get_all_start_methods() else 'spawn')
    parent,child=context.Pipe(duplex=False)
    process=context.Process(target=worker,args=(child,observed,dataset,specification,path))
    process.start();child.close()
    try:
        deadline=time.monotonic()+180
        while time.monotonic()<deadline:
            if parent.poll(.2):
                output=parent.recv();process.join(5);return output
            if not process.is_alive():return {'status':'failed','failure':f'worker exit {process.exitcode}'}
        return {'status':'failed','failure':'180 seconds condition budget'}
    finally:
        if process.is_alive():process.terminate()
        process.join();parent.close()


def image_comparison(dataset,observed,estimate,kind,n):
    indices=[0,len(dataset['times'])//2,len(dataset['times'])-1]
    clean_w=omega(dataset['velocity'],dataset['dx'])
    observed_w=omega(observed,dataset['dx']);estimated_w=omega(estimate,dataset['dx'])
    clean_scale=float(np.max(np.abs(clean_w)))
    reference_rms=float(np.sqrt(np.mean(clean_w**2)))
    velocity_error=np.linalg.norm((estimate-dataset['velocity']).ravel())/np.linalg.norm(dataset['velocity'].ravel())
    vorticity_error=np.linalg.norm((estimated_w-clean_w).ravel())/np.linalg.norm(clean_w.ravel())
    fig,axes=plt.subplots(3,4,figsize=(12.8,9.4),layout='constrained')
    columns=['Referencia limpia','Observación ruidosa','TLS rango 8 + proyección','Error absoluto de vorticidad']
    for row,index in enumerate(indices):
        fields=[clean_w[index],observed_w[index],estimated_w[index],np.abs(estimated_w[index]-clean_w[index])/reference_rms]
        for col,field in enumerate(fields):
            axis=axes[row,col]
            if col<3:
                field_image=axis.imshow(field,origin='lower',extent=(0,2*np.pi,0,2*np.pi),
                     cmap='RdBu_r',vmin=-clean_scale,vmax=clean_scale,interpolation='nearest')
                saturated=float(np.mean(abs(field)>clean_scale))
                if col:axis.text(.03,.04,f'Saturación: {100*saturated:.1f}%',transform=axis.transAxes,
                                fontsize=8,bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
            else:
                error_image=axis.imshow(np.maximum(field,1e-4),origin='lower',extent=(0,2*np.pi,0,2*np.pi),
                     cmap='magma',norm=LogNorm(vmin=1e-3,vmax=10),interpolation='nearest')
            axis.set_xticks([0,np.pi,2*np.pi],['0','π','2π'])
            axis.set_yticks([0,np.pi,2*np.pi],['0','π','2π'])
            if row==0:axis.set_title(columns[col],fontsize=10)
            if col==0:axis.set_ylabel(f't={dataset["times"][index]:.1f}\ny')
            if row==2:axis.set_xlabel('x')
    fig.colorbar(field_image,ax=axes[:,:3],label='ω [1/tiempo adimensional]',extend='both',shrink=.65)
    fig.colorbar(error_image,ax=axes[:,3],label='|ω̂−ω| / RMS(ω limpia)',extend='both',shrink=.65)
    title='Gaussiano por celda' if kind=='gaussian_iid' else 'Ruido espacial: ancho del filtro σ=0.10'
    fig.suptitle(f'{title} — malla {n}², ruido RMS 20%, semilla 7\nTLS proyectado: e(u)={100*velocity_error:.1f}%, e(ω)={100*vorticity_error:.1f}% en toda la ventana; saturación declarada',fontsize=12)
    stem=DEST/'figures'/f'{kind}_{n}_reconstruction'
    fig.savefig(str(stem)+'.png');fig.savefig(str(stem)+'.pdf');plt.close(fig)
    if n==128:
        # One row per exported time fits a landscape page without tiny labels.
        for index in indices:
            fig,axes=plt.subplots(1,4,figsize=(12.8,3.8),layout='constrained')
            fields=[clean_w[index],observed_w[index],estimated_w[index],
                    np.abs(estimated_w[index]-clean_w[index])/reference_rms]
            for col,(axis,field)in enumerate(zip(axes,fields)):
                if col<3:
                    field_image=axis.imshow(field,origin='lower',extent=(0,2*np.pi,0,2*np.pi),
                       cmap='RdBu_r',vmin=-clean_scale,vmax=clean_scale,interpolation='nearest')
                    if col:
                        axis.text(.03,.04,f'Sat.: {100*np.mean(abs(field)>clean_scale):.1f}%',
                                  transform=axis.transAxes,fontsize=10,
                                  bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
                else:
                    error_image=axis.imshow(np.maximum(field,1e-4),origin='lower',extent=(0,2*np.pi,0,2*np.pi),
                       cmap='magma',norm=LogNorm(vmin=1e-3,vmax=10),interpolation='nearest')
                axis.set_title(columns[col],fontsize=11)
                axis.set_xticks([0,np.pi,2*np.pi],['0','π','2π'])
                axis.set_yticks([0,np.pi,2*np.pi],['0','π','2π'])
                axis.tick_params(labelsize=11);axis.set_xlabel('x',fontsize=12)
            axes[0].set_ylabel('y',fontsize=12)
            fig.colorbar(field_image,ax=axes[:3],label='ω',extend='both',shrink=.7)
            fig.colorbar(error_image,ax=axes[3],label='|ω̂−ω| / RMS(ω)',extend='both',shrink=.7)
            fig.suptitle(f'{title}; 128²; t={dataset["times"][index]:.1f}; RMS ruido 20%\nTLS proyectado, ventana completa: e(u)={100*velocity_error:.1f}%, e(ω)={100*vorticity_error:.1f}%',fontsize=13)
            stem=DEST/'figures'/f'{kind}_{n}_frame_{index}'
            fig.savefig(str(stem)+'.pdf');fig.savefig(str(stem)+'.png');plt.close(fig)


def summary_figures(rows):
    fig,axes=plt.subplots(2,2,figsize=(12,8.3),layout='constrained')
    for col,kind in enumerate(NOISES):
        for row,key in enumerate(('velocity_relative_l2','vorticity_relative_l2')):
            axis=axes[row,col]
            for label,_,_,_ in METHODS:
                values=[[record[key] for record in rows if record['nx']==n and record['noise']==kind
                         and record['method']==label and record['status']=='ok'] for n in GRIDS]
                if any(not group for group in values):continue
                medians=np.array([np.median(group)for group in values])*100
                lo=np.array([min(group)for group in values])*100;hi=np.array([max(group)for group in values])*100
                axis.errorbar(GRIDS,medians,yerr=[medians-lo,hi-medians],color=COLORS[label],
                              label=label,marker='o',capsize=3,lw=1.5)
            axis.set_xticks(GRIDS);axis.set_xlabel('N: puntos por lado')
            axis.set_ylabel('Error L2 relativo [%]');axis.set_yscale('log');axis.grid(alpha=.2)
            axis.axhline(15 if row==0 else 25,color='#333333',ls='--',lw=1)
            axis.set_title(('Velocidad' if row==0 else 'Vorticidad')+' — '+('gaussiano por celda'if col==0 else 'σ del filtro fijo'))
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Resolución de observación y fidelidad de reconstrucción\nMediana y rango de tres semillas; líneas discontinuas: tolerancias de reconstrucción, no error DNS',fontsize=12)
    fig.savefig(DEST/'figures/noise_grid_comparison.pdf');fig.savefig(DEST/'figures/noise_grid_comparison.png');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--reuse-completed',action='store_true')
    args=parser.parse_args()
    (DEST/'figures').mkdir(parents=True,exist_ok=True);(DEST/'reconstructions').mkdir(exist_ok=True)
    config={'grids':GRIDS,'seeds':SEEDS,'methods':METHODS,'noise_models':NOISES,'relative_rms':.2,
            'spatial_correlation_length':.10,'spatial_master_grid':256,'thresholds':THRESHOLDS,
            'scope':'in-window reconstruction; fixed ranks, no clean field selection; one resource repetition; simultaneous solver jobs may affect wall times'}
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in (ROOT/'python/src/fluid_denoise/research').glob('*.py')}
    fingerprint=hashlib.sha256(json.dumps({'configuration':config,'sources':sources},sort_keys=True).encode()).hexdigest()
    datasets={n:load_dataset(n)for n in GRIDS}
    provenance={'configuration':config,'fingerprint':fingerprint,'source_files_sha256':sources,
                'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'source_datasets':{str(n):json.loads((SOURCE/f'interacting_vortices_{n}_dt0.005.json').read_text())['npz_sha256']for n in GRIDS},
                'filter_width_convention':{'physical_gaussian_kernel_sigma':.10,'continuum_gaussian_autocorrelation_standard_deviation':float(np.sqrt(2)*.10),'note':'spatial_correlation_length is the legacy configuration name for kernel sigma, not autocorrelation width'},
                'noise_resolution_note':'IID per-cell noise is not the same continuum realization on different meshes. Spatial noise is one Gaussian-smoothed master realization at 256 transferred to 128 without per-grid renormalization. Three seeds do not imply a confidence interval.',
                'cost_scope':'Fresh child per fit; fit+reshape+postprocess with tracemalloc; reference evaluation and NPZ writing excluded. One repetition, contended host: no precision timing ranking.'}
    rows=[];records=[]
    stored=json.loads((DEST/'provenance.json').read_text())if (DEST/'provenance.json').exists()else {}
    if (args.reuse_completed and stored.get('fingerprint')==fingerprint
            and stored.get('source_datasets')==provenance['source_datasets']):
        records=[json.loads(line)for line in (DEST/'ledger.jsonl').read_text().splitlines()]
        if len(records)!=60 or any(r['status']!='ok'for r in records):records=[]
    if not records:
        for kind in NOISES:
            for seed in SEEDS:
                master=noise_field(datasets[256]['velocity'],kind,seed)
                for n,dataset in datasets.items():
                    if kind=='spatial_fixed_length':
                        perturbation=master if n==256 else resample(resample(master,n,axis=-1),n,axis=-2)
                    else:perturbation=noise_field(dataset['velocity'],kind,seed)
                    observed=dataset['velocity']+perturbation
                    noise_rms=float(np.sqrt(np.mean(perturbation**2))/np.sqrt(np.mean(dataset['velocity']**2)))
                    for spec in METHODS:
                        path=str(DEST/'reconstructions'/f'{kind}_{n}_s{seed}_{spec[0]}.npz')if seed==7 else None
                        record=isolated(observed,dataset,spec,path)
                        record.update(nx=n,noise=kind,seed=seed,method=spec[0],relative_noise_rms=noise_rms)
                        records.append(record)
                        print(f'{len(records)}/60 {kind} N={n} seed={seed} {spec[0]} {record["status"]}',flush=True)
        with (DEST/'ledger.jsonl').open('w')as handle:
            for record in records:handle.write(json.dumps(json_safe(record),allow_nan=False)+'\n')
    for record in records:
        metrics=record.get('metrics',{})
        rows.append({key:record[key]for key in ('nx','noise','seed','method','status','relative_noise_rms')}|
                    {key:metrics.get(key)for key in ('velocity_relative_l2','vorticity_relative_l2','extra_divergence_normalized','kinetic_energy_relative_l2','enstrophy_relative_l2')}|
                    {'converged':record.get('diagnostics',{}).get('converged'),
                     'quality_pass':bool(record.get('quality',{}).get('passed')) and bool(record.get('diagnostics',{}).get('converged')),
                     'cpu_seconds':record.get('profile',{}).get('cpu_seconds')})
    with (DEST/'summary.csv').open('w',newline='')as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(DEST/'summary.json',rows);write_json(DEST/'provenance.json',provenance)
    for kind in NOISES:
        master=noise_field(datasets[256]['velocity'],kind,7)
        for n,dataset in datasets.items():
            perturbation=(master if n==256 else resample(resample(master,n,axis=-1),n,axis=-2))if kind=='spatial_fixed_length'else noise_field(dataset['velocity'],kind,7)
            path=DEST/'reconstructions'/f'{kind}_{n}_s7_tls_r8_projected.npz'
            if path.exists():
                with np.load(path)as arrays:image_comparison(dataset,dataset['velocity']+perturbation,arrays['velocity'],kind,n)
    summary_figures(rows)
    write_json(DEST/'completion.json',{'complete':len(records)==60,'conditions':len(records),
               'failed_conditions':sum(r['status']!='ok'for r in records),'fingerprint':fingerprint})
    print(DEST)


if __name__=='__main__':main()
