#!/usr/bin/env python3
"""Exploratory follow-up: fixed physical smoothing with an identity control.

The original 60-condition study remains unchanged. A single cutoff k_c=10 is
investigated after those failures, without a clean-reference parameter sweep.
The prior is an additional signal-smoothness assumption, not a DNS certificate.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
from pathlib import Path
import json
import csv
import hashlib
import multiprocessing as mp
import time
import numpy as np
from scipy.signal import resample
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import run_highres_reconstruction_visuals as base
from fluid_denoise.research.cost import measure_call,quality_gates
from fluid_denoise.research.metrics import physical_metrics
from fluid_denoise.research.physics import vorticity_residual_metrics
from fluid_denoise.research.runner import json_safe,write_json,assert_finite

DEST=base.ROOT/'results/research/highres_fixed_bandwidth_2026-10-14'
CUTOFF=10.
OPTIONS=(('identity_fixed_band','identity'),('tls_r8_fixed_band','tls_dmd'))


def fixed_fit(observed,dataset,method):
    value,diagnostics=base.fit(observed,dataset,method,8,'helmholtz')
    n=value.shape[-1]
    kx,ky=np.meshgrid(2*np.pi*np.fft.fftfreq(n,d=dataset['dx']),
                       2*np.pi*np.fft.fftfreq(n,d=dataset['dy']))
    support=np.hypot(kx,ky)<=CUTOFF
    output=np.fft.ifft2(np.fft.fft2(value)*support).real
    assert_finite(output)
    diagnostics['physical_bandwidth_prior']={'cutoff_inverse_length':CUTOFF,'isotropic_circle':True,
         'smoothness_assumption':True,'reference_velocity_used':False,
         'mean_change_max':float(np.max(abs(output.mean(axis=(-2,-1))-value.mean(axis=(-2,-1)))))}
    return output,diagnostics


def worker(connection,observed,dataset,method,path):
    try:
        (result,diagnostics),profile=measure_call(fixed_fit,observed,dataset,method)
        metrics=physical_metrics(dataset,result)
        metrics.update(vorticity_residual_metrics(dataset,result))
        gates=quality_gates(metrics,base.THRESHOLDS)
        if path is not None:np.savez_compressed(path,velocity=result,times=dataset['times'])
        connection.send(json_safe(dict(status='ok',diagnostics=diagnostics,profile=profile,
                                        metrics=metrics,quality=gates)))
    except Exception as error:connection.send(dict(status='failed',failure=str(error)))
    finally:connection.close()


def isolated(observed,dataset,method,path):
    context=mp.get_context('fork'if 'fork'in mp.get_all_start_methods()else'spawn')
    parent,child=context.Pipe(duplex=False)
    process=context.Process(target=worker,args=(child,observed,dataset,method,path))
    process.start();child.close()
    try:
        deadline=time.monotonic()+180
        while time.monotonic()<deadline:
            if parent.poll(.2):value=parent.recv();process.join(5);return value
            if not process.is_alive():return dict(status='failed',failure=f'worker exit {process.exitcode}')
        return dict(status='failed',failure='180 seconds budget')
    finally:
        if process.is_alive():process.terminate()
        process.join();parent.close()


def figures(datasets,rows):
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for col,kind in enumerate(base.NOISES):
        for row,key in enumerate(('velocity_relative_l2','vorticity_relative_l2')):
            axis=axes[row,col]
            for label,color in [('identity_fixed_band','#666666'),('tls_r8_fixed_band','#1965b0')]:
                samples=[[r[key]for r in rows if r['nx']==n and r['noise']==kind and r['method']==label and r['status']=='ok']for n in base.GRIDS]
                if any(not group for group in samples):continue
                med=np.array([np.median(group)for group in samples])*100
                lo=np.array([min(group)for group in samples])*100;hi=np.array([max(group)for group in samples])*100
                axis.errorbar(base.GRIDS,med,yerr=[med-lo,hi-med],color=color,label=label,marker='o',capsize=4)
            axis.axhline(15 if row==0 else 25,color='#333333',ls='--',lw=1)
            axis.set_xticks(base.GRIDS);axis.set_xlabel('N: puntos por lado');axis.set_ylabel('Error L2 relativo [%]')
            axis.grid(alpha=.2);axis.set_title(('Velocidad'if row==0 else'Vorticidad')+' — '+('IID por celda'if col==0 else'σ filtro fijo'))
    axes[0,0].legend(fontsize=10)
    fig.suptitle('Seguimiento exploratorio con corte físico fijo k≤10\nMediana/rango de tres semillas; observación filtrada controla el aporte del prior',fontsize=13)
    fig.savefig(DEST/'figures/noise_grid_comparison.pdf');fig.savefig(DEST/'figures/noise_grid_comparison.png');plt.close(fig)
    n=128;dataset=datasets[n];index=-1;clean=base.omega(dataset['velocity'],dataset['dx'])
    scale=float(np.max(abs(clean)));rms=float(np.sqrt(np.mean(clean**2)))
    for kind in base.NOISES:
        outputs=[]
        for label,_ in OPTIONS:
            with np.load(DEST/'reconstructions'/f'{kind}_{n}_s7_{label}.npz')as arrays:
                outputs.append(base.omega(arrays['velocity'],dataset['dx']))
        fig,axes=plt.subplots(1,4,figsize=(12.8,3.9),layout='constrained')
        fields=[clean[index],outputs[0][index],outputs[1][index],abs(outputs[1][index]-clean[index])/rms]
        labels=['Referencia limpia','Observación + prior k≤10','TLS rango 8 + prior k≤10','Error de ω / RMS limpia']
        for j,(axis,field)in enumerate(zip(axes,fields)):
            if j<3:picture=axis.imshow(field,origin='lower',extent=(0,2*np.pi,0,2*np.pi),vmin=-scale,vmax=scale,cmap='RdBu_r',interpolation='nearest')
            else:error=axis.imshow(field,origin='lower',extent=(0,2*np.pi,0,2*np.pi),vmin=0,vmax=.8,cmap='magma',interpolation='nearest')
            axis.set_title(labels[j],fontsize=11);axis.set_xlabel('x',fontsize=12)
            axis.set_xticks([0,np.pi,2*np.pi],['0','π','2π']);axis.set_yticks([0,np.pi,2*np.pi],['0','π','2π']);axis.tick_params(labelsize=11)
        axes[0].set_ylabel('y',fontsize=12)
        fig.colorbar(picture,ax=axes[:3],label='ω',extend='both',shrink=.7)
        fig.colorbar(error,ax=axes[3],label='|ω̂−ω| / RMS(ω)',extend='max',shrink=.7)
        selected=next(r for r in rows if r['nx']==n and r['noise']==kind and r['seed']==7 and r['method']=='tls_r8_fixed_band')
        fig.suptitle(f'{kind}; 128²; t=4; semilla 7; corte único exploratorio\nTLS + prior, toda la ventana: e(u)={100*selected["velocity_relative_l2"]:.2f}%, e(ω)={100*selected["vorticity_relative_l2"]:.2f}%',fontsize=13)
        stem=DEST/'figures'/f'{kind}_{n}_bandwidth_comparison'
        fig.savefig(str(stem)+'.pdf');fig.savefig(str(stem)+'.png');plt.close(fig)


def main():
    (DEST/'reconstructions').mkdir(parents=True,exist_ok=True);(DEST/'figures').mkdir(exist_ok=True)
    datasets={n:base.load_dataset(n)for n in base.GRIDS};records=[];rows=[]
    configuration={'grids':base.GRIDS,'seeds':base.SEEDS,'noise_models':base.NOISES,'options':OPTIONS,
                   'physical_cutoff':CUTOFF,'rank':8,'thresholds':base.THRESHOLDS}
    provenance={'configuration':configuration,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       'base_script_sha256':hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
       'source_files_sha256':{str(p.relative_to(base.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in(base.ROOT/'python/src/fluid_denoise/research').glob('*.py')},
       'source_datasets':{str(n):json.loads((base.SOURCE/f'interacting_vortices_{n}_dt0.005.json').read_text())['npz_sha256']for n in base.GRIDS},
       'design':'Follow-up after the original 60 failures. One k_c=10 chosen as reciprocal of kernel sigma=.10, used as an explicit signal-smoothness prior. No sweep or clean optimization. Same original seeds/flows: exploratory, without independent blind validation.',
       'identity_control':'Both identity and TLS use the identical Helmholtz and physical cutoff; this separates smoothness gains from additional temporal modeling.',
       'limitations':'A noise filter width does not establish clean-signal bandwidth. Filtering can remove physical filaments. All clean modes remain in evaluation; pass applies only to four reconstruction gates, not an exact PDE or DNS/forecast certificate.',
       'cost':'One fit per isolated child; absolute RSS includes inherited data/runtime. Contended host, no precision timing comparison.'}
    for kind in base.NOISES:
        for seed in base.SEEDS:
            master=base.noise_field(datasets[256]['velocity'],kind,seed)
            for n,dataset in datasets.items():
                noise=(master if n==256 else resample(resample(master,n,axis=-1),n,axis=-2))if kind=='spatial_fixed_length'else base.noise_field(dataset['velocity'],kind,seed)
                observed=dataset['velocity']+noise
                for label,method in OPTIONS:
                    path=str(DEST/'reconstructions'/f'{kind}_{n}_s{seed}_{label}.npz')if seed==7 else None
                    record=isolated(observed,dataset,method,path)
                    record.update(nx=n,noise=kind,seed=seed,method=label);records.append(record)
                    metrics=record.get('metrics',{})
                    rows.append({key:record[key]for key in('nx','noise','seed','method','status')}|
                         {key:metrics.get(key)for key in('velocity_relative_l2','vorticity_relative_l2','kinetic_energy_relative_l2','extra_divergence_normalized','enstrophy_relative_l2')}|
                         {'quality_pass':bool(record.get('quality',{}).get('passed'))and bool(record.get('diagnostics',{}).get('converged'))})
                    print(f'{len(records)}/24 {kind} {n} {label} {record["status"]}',flush=True)
    with(DEST/'ledger.jsonl').open('w')as handle:
        for record in records:handle.write(json.dumps(json_safe(record),allow_nan=False)+'\n')
    with(DEST/'summary.csv').open('w',newline='')as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(DEST/'summary.json',rows);write_json(DEST/'provenance.json',provenance)
    if all(r['status']=='ok'for r in records):figures(datasets,rows)
    write_json(DEST/'completion.json',{'complete':len(records)==24,'conditions':len(records),
             'failed_conditions':sum(r['status']!='ok'for r in records),'quality_pass_conditions':sum(r['quality_pass']for r in rows),
             'exploratory_followup':True})
    print(DEST)


if __name__=='__main__':main()
