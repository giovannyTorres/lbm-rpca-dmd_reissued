#!/usr/bin/env python3
"""Rebuild publication comparison figures/tables from the complete audited ledger.

The primary numerical figures do not need reconstruction NPZ files. Optional
snapshot plates load three exact computed states, never interpolate fields.
"""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
from pathlib import Path
import json
import csv
import sys
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

DEST=Path(__file__).resolve().parent
ROOT=DEST.parents[2]
sys.path.insert(0,str(ROOT/'python/scripts'))
from run_model_fidelity_benchmark import METHODS,FLOW_FILES,load_flow,observation_noise
from fluid_denoise.research.metrics import spatial_derivative
METHOD_NAMES=['Identidad','POD r8','DMD r8','TLS-DMD r8','PCP','RPCA estable',
              'Outlier Pursuit','PCP → TLS r8','Estable → TLS r8','Optimized DMD r8']
NOISES=['gaussian_iid','sparse_impulsive','spatial_gaussian','temporal_ar','column_outliers']
NOISE_NAMES=['Gaussiano IID','Impulsivo 1%','Gaussiano espacial','AR temporal','Columnas atípicas']
VARIANTS=['raw','helmholtz','helmholtz_k10']
VARIANT_NAMES=['Sin proyección','Helmholtz H','H + corte físico k≤10']
FLOW_NAMES=['NS vórtices','NS multiescala','Oldroyd-B','FENE-P','HB n=0.6','Bingham','HB n=1.4']
COLORS=['#787878','#1965b0','#b9770e']
FLOW_TITLES=dict(zip(FLOW_FILES,FLOW_NAMES))
NOISE_TITLES=dict(zip(NOISES,NOISE_NAMES))
plt.rcParams.update({'font.size':12,'axes.titlesize':13,'axes.labelsize':12,
    'xtick.labelsize':11,'ytick.labelsize':11,'pdf.fonttype':42,'savefig.dpi':180,
    'axes.spines.top':False,'axes.spines.right':False})


def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def save(fig,name):
    fig.savefig(DEST/'figures'/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(DEST/'figures'/f'{name}.png',bbox_inches='tight')
    plt.close(fig)


def valid_values(rows,key):
    return [r[key] for r in rows if r['status']=='ok' and r.get(key) is not None
            and np.isfinite(r[key])]


def pipeline_resources(rows,records):
    lookup={record['id']:record for record in records}
    output=[]
    for original in rows:
        row=dict(original)
        record=lookup[row['id']]
        evaluations={e['variant']:e for e in record.get('evaluations',[])}
        evaluation=evaluations.get(row['variant'],{})
        profile=evaluation.get('postprocess_profile',{})
        calls={'raw':0,'helmholtz':2,'helmholtz_k10':4}[row['variant']]
        points=int(record.get('diagnostics',{}).get('shape',[0,0])[0])*int(record.get('diagnostics',{}).get('shape',[0,0])[1])
        # One transform is over the128x128 spatial grid for each of2*T fields.
        row.update(postprocess_cpu_seconds=profile.get('cpu_seconds'),
                   postprocess_wall_seconds=profile.get('wall_seconds'),
                   fft_calls_total_postprocess=calls if evaluation else None,
                   fft_calls_declared_pipeline_design=calls,
                   fft_points_per_batched_call=points if evaluation else None,
                   fft_nlog2n_work_proxy=float(calls*points*np.log2(128*128)) if evaluation else None,
                   fft_proxy_definition='calls*(2*T*128*128)*log2(128*128); batched2D FFT asymptotic proxy, omitted constants/complex-real factors, not exact FLOPs',
                   flops_proxy_scope='Estimator dominant SVD/lstsq/eigen/matmul arithmetic; excludes FFT postprocessing and elementwise work',
                   original_projection_fft_counter_scope='Partial Helmholtz-only diagnostic; full H+k10 has two additional transforms',
                   postprocess_rss_scope='Accumulated process lifetime peak, may inherit fit/evaluation allocations; no incremental projection attribution')
        for metric in ['cpu_seconds','wall_seconds']:
            fit=row.get(metric);post=profile.get(metric)
            row['pipeline_'+metric]=fit+post if fit is not None and post is not None else None
        output.append(row)
    dump(DEST/'pipeline_resource_summary.json',output)
    return output


def pareto_controls(rows):
    output=[]
    blocks=sorted({(r['flow'],r['noise'],r['seed'],r['variant']) for r in rows})
    for flow,noise,seed,variant in blocks:
        attempted=[r for r in rows if (r['flow'],r['noise'],r['seed'],r['variant'])==(flow,noise,seed,variant)]
        eligible=[r for r in attempted if r['quality_pass']]
        row=dict(flow=flow,noise=noise,seed=seed,variant=variant,attempted=len(attempted),
                 operationally_accepted=len(eligible))
        for label,keys in [('dominant_matrix_work',('flops_proxy','vorticity_relative_l2')),
                          ('pipeline_CPU_and_RSS',('pipeline_cpu_seconds','rss_peak_bytes','vorticity_relative_l2'))]:
            valid=[r for r in eligible if all(r.get(key) is not None and np.isfinite(r[key]) for key in keys)]
            frontier=[]
            for candidate in valid:
                values=np.array([candidate[key] for key in keys])
                if not any(np.all(np.array([other[key] for key in keys])<=values)
                           and np.any(np.array([other[key] for key in keys])<values)
                           for other in valid if other['method']!=candidate['method']):
                    frontier.append(candidate['method'])
            row[label+'_pareto']=frontier
        output.append(row)
    dump(DEST/'operational_pareto.json',dict(blocks=output,
        scope='Frontiers only within identical observed flow/noise/seed/variant; operational gates and convergence first. Dominant-work FLOP proxy excludes postprocess FFT, which is identical for methods within each block. CPU/RSS frontier is descriptive with n=1 per shared-host condition, not a hardware ranking or statistical superiority test.'))


def aggregate(rows):
    output=[]
    metrics=['velocity_relative_l2','vorticity_relative_l2','vorticity_correlation',
             'extra_divergence_normalized','kinetic_energy_relative_l2','dissipation_relative_l2',
             'wall_seconds','cpu_seconds','rss_peak_bytes','flops_proxy','effective_rank',
             'pipeline_cpu_seconds','pipeline_wall_seconds','fft_nlog2n_work_proxy']
    for noise in NOISES+['all']:
        for method in METHODS:
            for variant in VARIANTS:
                selected=[r for r in rows if r['method']==method and r['variant']==variant
                           and (noise=='all' or r['noise']==noise)]
                count=len(selected)
                row=dict(noise=noise,method=method,variant=variant,attempted=count,
                    failed=sum(r['status']!='ok' for r in selected),
                    nonconverged=sum(r['status']=='ok' and not r['converged'] for r in selected),
                    operational_acceptances=sum(r['quality_pass'] for r in selected),
                    operational_acceptance_fraction=sum(r['quality_pass'] for r in selected)/count if count else None)
                profiled=[r for r in selected if r.get('cpu_seconds') is not None]
                cpu_total=sum(r['cpu_seconds'] for r in profiled)
                pipeline_cpu_total=sum(r['pipeline_cpu_seconds'] for r in profiled if r['pipeline_cpu_seconds'] is not None)
                work_total=sum(r['flops_proxy'] for r in profiled if r['flops_proxy'] is not None)
                accepted=row['operational_acceptances']
                row.update(fit_profiles_available=len(profiled),fit_cpu_seconds_total=cpu_total,
                    pipeline_cpu_seconds_total=pipeline_cpu_total,
                    fit_flops_proxy_total=work_total,
                    fit_cpu_seconds_per_operational_acceptance=cpu_total/accepted if accepted else None,
                    pipeline_cpu_seconds_per_operational_acceptance=pipeline_cpu_total/accepted if accepted else None,
                    fit_flops_proxy_per_operational_acceptance=work_total/accepted if accepted else None,
                    accepted_cost_definition='Sum of measured fit CPU/dominant-work proxies over all profiled attempts divided by operational acceptances; one fit reused across its three evaluated postprocesses. Excludes evaluation and postprocess time. Undefined when zero acceptances; lower bound if any failed attempt lacks a profile.',
                    accepted_cost_incomplete_profiles=len(profiled)!=count)
                for key in metrics:
                    values=valid_values(selected,key)
                    converged=valid_values([r for r in selected if r['converged']],key)
                    row[key+'_finite_median']=float(np.median(values)) if values else None
                    row[key+'_finite_min']=min(values) if values else None
                    row[key+'_finite_max']=max(values) if values else None
                    row[key+'_converged_median']=float(np.median(converged)) if converged else None
                output.append(row)
    dump(DEST/'aggregate_comparison.json',output)
    with (DEST/'aggregate_comparison.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(output[0]));writer.writeheader();writer.writerows(output)
    return output


def heatmap(aggregates,variant,key,name):
    # One cell contains21 cases:7 trajectories×3 seeds; this is descriptive,
    # and the per-flow table is supplied separately to expose heterogeneity.
    values=np.full((len(METHODS),len(NOISES)),np.nan)
    counts=np.zeros_like(values,dtype=int)
    for i,method in enumerate(METHODS):
        for j,noise in enumerate(NOISES):
            row=next(r for r in aggregates if (r['method'],r['noise'],r['variant'])==(method,noise,variant))
            value=row[key+'_finite_median'];values[i,j]=100*value if value is not None else np.nan
            counts[i,j]=row['failed']+row['nonconverged']
    fig,ax=plt.subplots(figsize=(10.6,7.7),layout='constrained')
    image=ax.imshow(np.ma.masked_invalid(values),cmap='viridis_r',norm=LogNorm(vmin=.5,vmax=500),aspect='auto')
    for i in range(len(METHODS)):
        for j in range(len(NOISES)):
            text='—' if not np.isfinite(values[i,j]) else f'{values[i,j]:.1f}'
            if counts[i,j]:text+=f'\n!{counts[i,j]}/21'
            ax.text(j,i,text,ha='center',va='center',fontsize=12,
                    color='white' if np.isfinite(values[i,j]) and values[i,j]>20 else '#111111')
    ax.set_xticks(range(len(NOISES)),NOISE_NAMES,rotation=20,ha='right')
    ax.set_yticks(range(len(METHODS)),METHOD_NAMES)
    ax.set_title(f'{"Vorticidad" if key.startswith("vorticity") else "Velocidad"}: mediana de error L2 [%]\n'+VARIANT_NAMES[VARIANTS.index(variant)],fontsize=15)
    fig.colorbar(image,ax=ax,label='Error L2 relativo [%], escala logarítmica',extend='both',shrink=.85)
    fig.suptitle('7 trayectorias × 3 semillas nuevas por celda; ruido global 20% RMS\n! = fallos o ajustes sin converger; las medianas muestran todas las salidas finitas',fontsize=12)
    save(fig,name)


def model_overview(rows,variant,name):
    fig,axes=plt.subplots(1,2,figsize=(14.0,7.1),layout='constrained',
                          gridspec_kw={'width_ratios':[5,7]})
    for ax,group_key,groups,labels,denominator in [
        (axes[0],'noise',NOISES,NOISE_NAMES,21),
        (axes[1],'flow',list(FLOW_FILES),FLOW_NAMES,15)]:
        values=np.full((10,len(groups)),np.nan)
        counts=np.zeros_like(values,dtype=int)
        for i,method in enumerate(METHODS):
            for j,group in enumerate(groups):
                selected=[r for r in rows if r['method']==method and r['variant']==variant and r[group_key]==group]
                actual=valid_values(selected,'vorticity_relative_l2')
                values[i,j]=100*np.median(actual) if actual else np.nan
                counts[i,j]=sum(r['status']!='ok' or not r['converged'] for r in selected)
        image=ax.imshow(np.ma.masked_invalid(values),cmap='viridis_r',norm=LogNorm(vmin=.5,vmax=500),aspect='auto')
        for i in range(10):
            for j in range(len(groups)):
                text=f'{values[i,j]:.1f}' if np.isfinite(values[i,j]) else '—'
                if counts[i,j]:text+=f'\n!{counts[i,j]}/{denominator}'
                ax.text(j,i,text,ha='center',va='center',fontsize=12,
                    color='white' if np.isfinite(values[i,j]) and values[i,j]>20 else '#111111')
        ax.set_xticks(range(len(groups)),labels,rotation=32,ha='right')
        ax.set_yticks(range(10),METHOD_NAMES)
        ax.set_title(('Por ruido: mediana de 21 realizaciones' if group_key=='noise' else 'Por flujo: mediana de 15 realizaciones'))
    fig.colorbar(image,ax=axes,label='Error relativo de vorticidad [%], escala logarítmica común',shrink=.85,extend='both')
    fig.suptitle('Comparación de diez modelos — '+VARIANT_NAMES[VARIANTS.index(variant)]+'\nSemillas 101/307/911; todas las salidas finitas; ! = sin converger o fallo incluido en denominador',fontsize=14)
    save(fig,name)


def prior_contributions(rows):
    lookup={(r['id'],r['variant']):r for r in rows}
    paired=[]
    for row in rows:
        if row['variant']!='raw' or row['status']!='ok':continue
        h=lookup[(row['id'],'helmholtz')];k=lookup[(row['id'],'helmholtz_k10')]
        pair=dict(id=row['id'],flow=row['flow'],noise=row['noise'],seed=row['seed'],method=row['method'],converged=row['converged'])
        identity=next(r for r in rows if (r['flow'],r['noise'],r['seed'],r['method'],r['variant'])==
                      (row['flow'],row['noise'],row['seed'],'identity','helmholtz_k10'))
        for key in ['velocity_relative_l2','vorticity_relative_l2','kinetic_energy_relative_l2']:
            pair[key+'_H_reduction_pp']=100*(row[key]-h[key])
            pair[key+'_band_reduction_pp']=100*(h[key]-k[key])
            pair[key+'_beyond_identity_band_pp']=100*(identity[key]-k[key])
        paired.append(pair)
    dump(DEST/'paired_prior_contributions.json',paired)
    fig,axes=plt.subplots(1,2,figsize=(13.7,6.9),layout='constrained')
    for ax,key,label in zip(axes,['velocity_relative_l2','vorticity_relative_l2'],['Velocidad','Vorticidad']):
        y=np.arange(len(METHODS))
        for offset,suffix,title,color in [(-.18,'_H_reduction_pp','Cambio por H','#1965b0'),(.18,'_band_reduction_pp','Cambio adicional por k≤10','#b9770e')]:
            values=[np.median([p[key+suffix] for p in paired if p['method']==method]) for method in METHODS]
            ax.barh(y+offset,values,height=.32,label=title,color=color)
        ax.set_yticks(y,METHOD_NAMES);ax.invert_yaxis();ax.axvline(0,color='#333333',lw=.8)
        ax.set_xlabel('Reducción mediana de error [puntos porcentuales]');ax.set_title(label);ax.grid(axis='x',alpha=.2)
    axes[0].legend(fontsize=10,loc='lower right')
    fig.suptitle('Mismo ajuste: aporte de cada posproceso, sin volver a entrenar\n105 pares por método; H preserva la vorticidad discreta y k≤10 añade una hipótesis de suavidad',fontsize=13)
    save(fig,'prior_contributions')
    fig,ax=plt.subplots(figsize=(11.5,7.4),layout='constrained')
    offsets=(.16,.08,0.,-.08,-.16)
    markers=('o','s','^','D','P')
    for j,noise in enumerate(NOISES):
        values=[np.median([p['vorticity_relative_l2_beyond_identity_band_pp'] for p in paired if p['method']==m and p['noise']==noise]) for m in METHODS]
        ax.scatter(values,np.arange(len(METHODS))+offsets[j],marker=markers[j],s=48,
                   color=plt.cm.tab10(j),label=NOISE_NAMES[j],zorder=3)
    ax.set_yticks(range(len(METHODS)),METHOD_NAMES);ax.invert_yaxis();ax.axvline(0,color='#333333',lw=.8)
    ax.set_xlabel('Error de ω: identidad+H+k10 menos modelo+H+k10 [puntos porcentuales]')
    ax.set_title('Aporte del modelo más allá del mismo filtro espacial\nMediana de 21 pares por ruido; valor positivo = menor error que la identidad filtrada')
    fig.legend(*ax.get_legend_handles_labels(),loc='outside lower center',ncol=3,fontsize=10)
    ax.grid(axis='x',alpha=.2)
    save(fig,'model_vs_identity')


def convergence_acceptance(aggregates):
    fig,axes=plt.subplots(2,3,figsize=(15.7,10.5),layout='constrained')
    for j,noise in enumerate(NOISES):
        ax=axes.flat[j]
        for offset,variant,label,color in zip([-.24,0,.24],VARIANTS,VARIANT_NAMES,COLORS):
            values=[next(r['operational_acceptances'] for r in aggregates if (r['noise'],r['method'],r['variant'])==(noise,m,variant)) for m in METHODS]
            ax.barh(np.arange(len(METHODS))+offset,values,height=.22,label=label,color=color)
        ax.set_yticks(range(len(METHODS)),METHOD_NAMES);ax.invert_yaxis();ax.set_xlim(0,21)
        ax.set_xticks([0,7,14,21]);ax.set_xlabel('Cumplen criterios operativos / 21 intentos')
        ax.set_title(NOISE_NAMES[j]);ax.grid(axis='x',alpha=.2)
    axes.flat[5].axis('off')
    axes.flat[5].text(.04,.85,'Aceptación exige:\n• Convergencia de todas las etapas\n• e(u) ≤15%\n• e(ω) ≤25%\n• Div extra/RMS(ω) ≤10%\n• e(energía) ≤15%\n\nFallos/sin converger permanecen\nen el denominador.\nNo certifica la ecuación constitutiva\nni una solución exacta de PDE.',va='top',fontsize=12)
    axes.flat[5].legend(*axes.flat[0].get_legend_handles_labels(),loc='lower left',fontsize=11)
    fig.suptitle('Aceptación y posprocesado por tipo de ruido\n7 trayectorias × 3 semillas = 21 intentos por método/ruido',fontsize=15)
    save(fig,'convergence_acceptance')


def cost_fidelity(rows):
    fig,axes=plt.subplots(1,2,figsize=(13.7,6.1),layout='constrained')
    fig.get_layout_engine().set(rect=(0,.20,1,.80))
    context=[r for r in rows if (r['flow'],r['noise'],r['seed'],r['variant'])==
             ('ns_multiscale','gaussian_iid',101,'helmholtz_k10')]
    for i,method in enumerate(METHODS):
        row=next(r for r in context if r['method']==method)
        marker='o' if row['quality_pass'] else ('X' if not row['converged'] else 's')
        for ax,key,xlabel in [(axes[0],'pipeline_cpu_seconds','CPU de ajuste + H+k10 [s]'),(axes[1],'flops_proxy','Trabajo matricial dominante [proxy de FLOPs]')]:
            x=row.get(key);y=row.get('vorticity_relative_l2')
            if x is None or y is None:continue
            # A genuine zero dominant-operation proxy has no log coordinate.
            if x<=0:continue
            ax.scatter(x,100*y,s=85,color=plt.cm.tab10(i),marker=marker,zorder=3)
            offset={4:(-12,-16),7:(8,5),9:(-13,-15)}.get(i,(5,3))
            ax.annotate(str(i+1),(x,100*y),xytext=offset,textcoords='offset points',fontsize=11)
            ax.set_xscale('log');ax.set_yscale('log');ax.grid(alpha=.2)
            ax.margins(y=.15)
            ax.set_xlabel(xlabel);ax.set_ylabel('Error relativo de vorticidad [%]')
    fig.suptitle('Fidelidad y recursos en la misma observación\nNS multiescala 128², gaussiano IID 20%, semilla 101; H+k10 idéntico, n = 1 por ajuste',fontsize=14)
    method_key='\n'.join('; '.join(f'{i+1} {METHOD_NAMES[i]}' for i in indices)
                         for indices in (range(5),range(5,10)))
    fig.text(.02,.025,method_key+'\nCírculo = cumple criterios; cuadrado = no cumple; X = no convergió. Host compartido, sin ranking hardware preciso.\nIdentidad: proxy matricial 0 (omitido del eje log); todas las opciones suman 4 FFT por H+k10, contabilizadas por separado.',fontsize=10)
    save(fig,'cost_fidelity')


def evolution(records,flow,noise):
    fig,axes=plt.subplots(1,2,figsize=(13.4,6.4),layout='constrained')
    for i,method in enumerate(METHODS):
        selected=[r for r in records if (r['flow'],r['noise'],r['method'])==(flow,noise,method) and r['status']=='ok']
        missing=3-len(selected)
        not_converged=sum(not r['diagnostics']['converged'] for r in selected)
        label=METHOD_NAMES[i]+(f' ({not_converged} NC)' if not_converged else '')+(f' ({missing} fallos)' if missing else '')
        for ax,key in zip(axes,['velocity_relative_l2','vorticity_relative_l2']):
            sample=[next(e['frame_errors'] for e in r['evaluations'] if e['variant']=='helmholtz_k10') for r in selected]
            if not sample:continue
            values=np.array([s[key] for s in sample])*100
            times=sample[0]['times']
            color=plt.cm.tab10(i)
            ax.plot(times,np.median(values,axis=0),label=label,color=color)
            ax.fill_between(times,values.min(axis=0),values.max(axis=0),alpha=.10,color=color)
            ax.set_yscale('log');ax.set_xlabel('Tiempo adimensional');ax.set_ylabel('Error L2 relativo [%]');ax.grid(alpha=.2)
    axes[0].set_title('Velocidad');axes[1].set_title('Vorticidad')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=5,fontsize=10)
    fig.suptitle(f'{FLOW_TITLES[flow]}, {NOISE_TITLES[noise]}: evolución del error con el mismo H+k10\nMediana/mínimo–máximo de salidas finitas; NC indica no convergencia; tiempos guardados',fontsize=14)
    save(fig,f'model_evolution_{flow}_{noise}')


def snapshots(records,flow,noise,variant):
    selected=[r for r in records if r['flow']==flow and r['noise']==noise and r['seed']==101 and r['status']=='ok']
    if len(selected)!=10:return
    reference,_=load_flow(flow)
    spacing=reference['dx']
    omega=lambda v:spatial_derivative(v[:,1],spacing,-1,True)-spatial_derivative(v[:,0],spacing,-2,True)
    clean=omega(reference['velocity'])[-1]
    scale=float(np.max(abs(clean)))
    fig,axes=plt.subplots(2,5,figsize=(15.6,7.4),layout='constrained')
    for i,(ax,method) in enumerate(zip(axes.flat,METHODS)):
        record=next(r for r in selected if r['method']==method)
        evaluation=next(e for e in record['evaluations'] if e['variant']==variant)
        archive=evaluation['reconstruction_archive']
        if archive is None:plt.close(fig);return
        path=ROOT/archive['path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=archive['sha256']:
            raise ValueError(f'Changed estimate archive: {path}')
        with np.load(path) as source:
            actual=omega(source['velocity'])[-1]
            t=float(source['times'][-1])
        saturated=100*float(np.mean(abs(actual)>scale))
        image=ax.imshow(actual,origin='lower',extent=(0,2*np.pi,0,2*np.pi),cmap='RdBu_r',vmin=-scale,vmax=scale,interpolation='nearest')
        error=evaluation['metrics']['vorticity_relative_l2']*100
        ax.set_title(f'{METHOD_NAMES[i]}\ne(ω) ventana={error:.1f}%',fontsize=14)
        if not record['diagnostics']['converged']:
            ax.text(.02,.92,'SIN CONVERGER',transform=ax.transAxes,fontsize=12,color='#9e1b1b',
                    bbox=dict(facecolor='white',alpha=.9,edgecolor='none'))
        ax.text(.02,.03,f'Sat. {saturated:.1f}%',transform=ax.transAxes,fontsize=12,
                bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
        ax.set_xticks([0,np.pi,2*np.pi],['0','π','2π']);ax.set_yticks([0,np.pi,2*np.pi],['0','π','2π'])
        ax.set_xlabel('x',fontsize=14);ax.set_ylabel('y',fontsize=14);ax.tick_params(labelsize=13)
    fig.colorbar(image,ax=axes,label='ω; escala común: ±máximo de referencia limpia al tiempo final',shrink=.8,extend='both')
    fig.suptitle(f'{FLOW_TITLES[flow]}, {NOISE_TITLES[noise]}; semilla 101; t = {t:g}; '+VARIANT_NAMES[VARIANTS.index(variant)]+'\nDiez modelos reciben la misma observación; colores saturados y error cuantitativo declarado',fontsize=14)
    save(fig,f'model_snapshots_{flow}_{noise}_{variant}')


def reference_observation_context(records,flow,noise,configuration):
    record=next(r for r in records if r['flow']==flow and r['noise']==noise and r['seed']==101)
    reference,_=load_flow(flow)
    velocity=reference['velocity'];spacing=reference['dx']
    perturbation=observation_noise(velocity.shape,reference['times'],noise,101,
                                  float(np.sqrt(np.mean(velocity**2))),configuration)
    observed=velocity+perturbation
    if hashlib.sha256(observed.tobytes()).hexdigest()!=record['noise_metadata']['observation_sha256']:
        raise ValueError('Context observation does not match the paired benchmark')
    omega=lambda v:spatial_derivative(v[:,1],spacing,-1,True)-spatial_derivative(v[:,0],spacing,-2,True)
    clean=omega(velocity)[-1];noisy=omega(observed)[-1]
    scale=float(np.max(abs(clean)));rms=float(np.sqrt(np.mean(clean**2)))
    fig,axes=plt.subplots(1,3,figsize=(12.9,4.9),layout='constrained')
    for i,field in enumerate([clean,noisy,abs(noisy-clean)/rms]):
        if i<2:
            image=axes[i].imshow(field,origin='lower',extent=(0,2*np.pi,0,2*np.pi),cmap='RdBu_r',vmin=-scale,vmax=scale,interpolation='nearest')
            axes[i].text(.02,.03,f'Sat. {100*np.mean(abs(field)>scale):.1f}%',transform=axes[i].transAxes,
                fontsize=11,bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
        else:
            error_image=axes[i].imshow(np.maximum(field,1e-4),origin='lower',extent=(0,2*np.pi,0,2*np.pi),cmap='magma',norm=LogNorm(vmin=1e-3,vmax=10),interpolation='nearest')
        axes[i].set_title(['Referencia calculada128²','Misma observación de diez modelos','Error de observación |Δω|/RMS(ω)'][i],fontsize=12)
        axes[i].set_xticks([0,np.pi,2*np.pi],['0','π','2π']);axes[i].set_yticks([0,np.pi,2*np.pi],['0','π','2π'])
        axes[i].set_xlabel('x');axes[i].set_ylabel('y')
    fig.colorbar(image,ax=axes[:2],label='ω: escala física común',extend='both',shrink=.8)
    fig.colorbar(error_image,ax=axes[2],label='Error normalizado (log)',extend='both',shrink=.8)
    fig.suptitle(f'{FLOW_TITLES[flow]}, {NOISE_TITLES[noise]}, semilla 101, t = {reference["times"][-1]:g}\nInyección 20% RMS de velocidad; la derivación amplifica los modos de alta frecuencia',fontsize=14)
    save(fig,f'reference_observation_{flow}_{noise}')


def constitutive_fidelity(rows):
    fig,axes=plt.subplots(2,2,figsize=(13.2,8.8),layout='constrained')
    keys=['kinetic_energy_relative_l2','dissipation_relative_l2',
          'stress_frobenius_relative_l2','apparent_viscosity_relative_l2']
    labels=['Energía cinética','Disipación viscosa instantánea','Esfuerzo planar: norma Frobenius','Viscosidad aparente']
    for ax,key,label in zip(axes.flat,keys,labels):
        for offset,variant,color,title in zip([-.24,0,.24],VARIANTS,COLORS,VARIANT_NAMES):
            selected=[[r for r in rows if (r['flow'],r['noise'],r['method'],r['variant'])==
                        ('hb_thinning','gaussian_iid',method,variant)] for method in METHODS]
            values=[valid_values(group,key) for group in selected]
            med=np.array([np.median(group)*100 if group else np.nan for group in values])
            low=np.array([min(group)*100 if group else np.nan for group in values])
            high=np.array([max(group)*100 if group else np.nan for group in values])
            # Log x accommodates derivative-driven errors with very different scales.
            ax.errorbar(med,np.arange(10)+offset,xerr=[med-low,high-med],fmt='o',
                        markersize=5,capsize=2,color=color,label=title)
        ax.set_xscale('log');ax.set_yticks(range(10),METHOD_NAMES);ax.invert_yaxis();ax.tick_params(labelsize=13)
        ax.set_xlabel('Error L2 relativo [%], escala logarítmica',fontsize=14);ax.set_title(label,fontsize=15)
        ax.grid(axis='x',alpha=.2)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=3,fontsize=11)
    fig.suptitle('Fidelidad física derivada de la velocidad reconstruida\nHB n = 0.6 en 128², gaussiano IID 20%; mediana y rango de tres semillas\nη y τ se obtienen del cierre regularizado conocido, sin estimar parámetros ni memoria polimérica',fontsize=14)
    save(fig,'constitutive_fidelity_hb_thinning_gaussian_iid')


def main():
    completion=json.loads((DEST/'completion.json').read_text())
    if not completion['complete'] or completion['recorded_fits']!=1050:
        raise ValueError('Publication analysis requires the full1050-condition ledger')
    records=[json.loads(line) for line in (DEST/'ledger.jsonl').read_text().splitlines() if line.strip()]
    rows=json.loads((DEST/'summary.json').read_text())
    configuration=json.loads((DEST/'provenance.json').read_text())['configuration']
    if len(records)!=1050 or len(rows)!=3150:raise ValueError('Unexpected completed denominator')
    rows=pipeline_resources(rows,records)
    pareto_controls(rows)
    (DEST/'figures').mkdir(exist_ok=True)
    aggregates=aggregate(rows)
    model_overview(rows,'raw','model_raw_overview')
    model_overview(rows,'helmholtz_k10','model_filtered_overview')
    for variant in VARIANTS:
        heatmap(aggregates,variant,'vorticity_relative_l2',f'heatmap_{variant}_vorticity')
    heatmap(aggregates,'raw','velocity_relative_l2','heatmap_raw_velocity')
    prior_contributions(rows);convergence_acceptance(aggregates);cost_fidelity(rows)
    for flow,noise in [('ns_multiscale','gaussian_iid'),('oldroyd_b','spatial_gaussian')]:
        evolution(records,flow,noise);reference_observation_context(records,flow,noise,configuration)
        snapshots(records,flow,noise,'raw');snapshots(records,flow,noise,'helmholtz_k10')
    constitutive_fidelity(rows)
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((DEST/'figures').iterdir())}
    dump(DEST/'figure_manifest.json',manifest)
    print(f'Complete: {len(aggregates)} aggregate groups, {len(manifest)//2} PDF+PNG figures')


if __name__=='__main__':main()
