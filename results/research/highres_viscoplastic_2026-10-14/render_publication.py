#!/usr/bin/env python3
"""Readable compact companions for the larger-grid viscoplastic experiment.

Uses saved 128² trajectories; no PDE is rerun. Long 15-panel illustrations
remain supplementary. A common color range makes cross-material comparison
possible, and the logarithmic viscosity colorbar is explicitly labeled.
"""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

OUTPUT=Path(__file__).resolve().parent
FIGURES=OUTPUT/'figures'
MODELS=(('hb_thinning','HB n=0.6'),('bingham','Bingham'),('hb_thickening','HB n=1.4'))
plt.rcParams.update({'font.size':12,'axes.titlesize':13,'axes.labelsize':12,
    'xtick.labelsize':11,'ytick.labelsize':11,'legend.fontsize':10,'figure.titlesize':14})


def load():
    runs={};metadata={}
    for name,title in MODELS:
        path=OUTPUT/'datasets'/f'{name}_128.npz'
        metadata[name]=json.loads(path.with_suffix('.json').read_text())
        with np.load(path) as arrays:runs[name]={k:arrays[k] for k in arrays.files}
        if not all(np.all(np.isfinite(v)) for v in runs[name].values()):
            raise ValueError(f'Nonfinite saved field: {name}')
    return runs,metadata


def omega(run,meta):
    ny,nx=run['velocity'].shape[-2:]
    kx,ky=np.meshgrid(2*np.pi*np.fft.fftfreq(nx,d=meta['dx']),
                      2*np.pi*np.fft.fftfreq(ny,d=meta['dy']))
    return np.fft.ifft2(1j*kx*np.fft.fft2(run['velocity'][:,1])
        -1j*ky*np.fft.fft2(run['velocity'][:,0])).real


def coordinate_axes(axis,x_label=True,y_label=True):
    axis.set_xticks([0,np.pi,2*np.pi],['0','π','2π'])
    axis.set_yticks([0,np.pi,2*np.pi],['0','π','2π'])
    if x_label:axis.set_xlabel('x')
    if y_label:axis.set_ylabel('y')


def save(fig,name):
    fig.savefig(FIGURES/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(FIGURES/f'{name}.png',dpi=230,bbox_inches='tight')
    plt.close(fig)


def provenance(metas,names,primary_only):
    result={'render_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'saved_data_metadata_sha256':{name:hashlib.sha256((OUTPUT/'datasets'/f'{name}_128.json').read_bytes()).hexdigest() for name,_ in MODELS},
        'state_grid':128,'common_times':[0,1,2],'font_points':{'base':12,'axes_title':13,'figure_title':14},
        'color_scaling':'all models and all saved times share vorticity limits; final constitutive fields share per-quantity limits; viscosity scale explicitly logarithmic',
        'quantity_units':'nondimensional','figure_directory':'figures','main_figures':names,
        'primary_trajectories_complete':True,'primary_figures_only':primary_only,
        'full_campaign_complete':json.loads((OUTPUT/'completion.json').read_text()).get('complete',False),
        'limitations':'periodic regularized inelastic flow, no cylinder/chaos/DNS certification; proxy is not exact unyielded region'}
    (OUTPUT/'publication_provenance.json').write_text(json.dumps(result,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--primary-only',action='store_true',help='Render four complete primary trajectories figures before secondary checks finish')
    args=parser.parse_args()
    FIGURES.mkdir(exist_ok=True)
    runs,metas=load();curves={name:omega(run,metas[name]) for name,run in runs.items()}
    limit=max(float(np.abs(v).max()) for v in curves.values())
    fig,axes=plt.subplots(3,3,figsize=(10.5,6.7),layout='constrained')
    for row,(name,title) in enumerate(MODELS):
        times=runs[name]['times']
        for col,requested in enumerate((0.,1.,2.)):
            candidates=np.flatnonzero(np.isclose(times,requested,rtol=0,atol=2e-14))
            if len(candidates)!=1:raise ValueError('Expected saved t=0,1,2 exactly once')
            im=axes[row,col].imshow(curves[name][candidates[0]],origin='lower',
                extent=(0,2*np.pi,0,2*np.pi),cmap='RdBu_r',vmin=-limit,vmax=limit)
            coordinate_axes(axes[row,col],x_label=row==2,y_label=False)
            if row==0:axes[row,col].set_title(f't={requested:g}')
            if col==0:axes[row,col].set_ylabel(title+'\ny',fontsize=12)
    fig.colorbar(im,ax=axes,shrink=.85,label='Vorticidad ω, escala común')
    fig.suptitle('Vórtices y momentum constitutivo 2D: malla 128×128',fontsize=14)
    save(fig,'vorticity_publication')

    fig,axes=plt.subplots(2,3,figsize=(10.5,5.8),layout='constrained')
    max_shear=max(float(r['shear_rate'][-1].max()) for r in runs.values())
    min_eta=min(float(r['apparent_viscosity'][-1].min()) for r in runs.values())
    max_eta=max(float(r['apparent_viscosity'][-1].max()) for r in runs.values())
    for col,(name,title) in enumerate(MODELS):
        gamma_image=axes[0,col].imshow(runs[name]['shear_rate'][-1],origin='lower',
            extent=(0,2*np.pi,0,2*np.pi),cmap='viridis',vmin=0,vmax=max_shear)
        eta_image=axes[1,col].imshow(runs[name]['apparent_viscosity'][-1],origin='lower',
            extent=(0,2*np.pi,0,2*np.pi),cmap='magma',norm=LogNorm(vmin=min_eta,vmax=max_eta))
        axes[0,col].set_title(title)
        for row in range(2):coordinate_axes(axes[row,col],x_label=row==1,y_label=False)
    axes[0,0].set_ylabel('Corte γ̇\ny');axes[1,0].set_ylabel('Viscosidad η\ny')
    fig.colorbar(gamma_image,ax=axes[0,:],shrink=.82,label='γ̇ (escala común)')
    fig.colorbar(eta_image,ax=axes[1,:],shrink=.82,label='η (escala logarítmica común)')
    fig.suptitle('Campos constitutivos a t=2; malla 128×128')
    save(fig,'shear_viscosity_publication')

    fig,axes=plt.subplots(2,3,figsize=(10.5,5.8),layout='constrained')
    max_stress=max(float(r['stress_invariant'][-1].max()) for r in runs.values())
    for col,(name,title) in enumerate(MODELS):
        stress_image=axes[0,col].imshow(runs[name]['stress_invariant'][-1],origin='lower',
            extent=(0,2*np.pi,0,2*np.pi),cmap='viridis',vmin=0,vmax=max_stress)
        yield_image=axes[1,col].imshow(~runs[name]['over_yield_stress'][-1],origin='lower',
            extent=(0,2*np.pi,0,2*np.pi),cmap='Greys',vmin=0,vmax=1)
        axes[0,col].set_title(title)
        for row in range(2):coordinate_axes(axes[row,col],x_label=row==1,y_label=False)
    axes[0,0].set_ylabel('Esfuerzo |τ|\ny');axes[1,0].set_ylabel('Proxy de cedencia\ny')
    fig.colorbar(stress_image,ax=axes[0,:],shrink=.82,label='sqrt(τ:τ/2), escala común')
    bar=fig.colorbar(yield_image,ax=axes[1,:],shrink=.82,ticks=[0,1])
    bar.set_ticklabels(['|τ|>τ_y','|τ|≤τ_y'])
    fig.suptitle('Esfuerzo y proxy regularizado a t=2; no son regiones rígidas exactas')
    save(fig,'stress_yield_publication')

    fig,axes=plt.subplots(2,2,figsize=(10,6),layout='constrained')
    for name,title in MODELS:
        run=runs[name];times=run['times']
        axes[0,0].plot(times,run['kinetic_energy']/run['kinetic_energy'][0],label=title)
        axes[0,1].plot(times,.5*np.mean(curves[name]**2,axis=(-2,-1)),label=title)
        axes[1,0].plot(times,run['energy_budget_residual']/run['kinetic_energy'][0],label=title)
        axes[1,1].plot(times,np.mean(~run['over_yield_stress'],axis=(-2,-1)),label=title)
    for axis,title in zip(axes.flat,('Energía cinética E/E₀','Enstrofía ⟨ω²⟩/2',
                            'Residuo del balance / E₀','Fracción |τ|≤τ_y (proxy)')):
        axis.set_title(title);axis.set_xlabel('Tiempo adimensional');axis.grid(alpha=.25)
    axes[0,0].legend()
    save(fig,'diagnostics_publication')
    primary_names=['vorticity_publication','shear_viscosity_publication','stress_yield_publication','diagnostics_publication']
    if args.primary_only:
        provenance(metas,primary_names,True)
        print('Four compact figures of complete primary trajectories written')
        return
    comparisons=json.loads((OUTPUT/'comparisons.json').read_text())
    frobenius=json.loads((OUTPUT/'derived_stress_frobenius.json').read_text())['comparisons']
    tensor={(c['run'],c['reference']):c for c in frobenius}
    fig,axes=plt.subplots(2,2,figsize=(10,6),layout='constrained')
    for name,title in MODELS:
        check=next(c for c in comparisons if c['run']==f'{name}_64'
                   and c['scope']=='long_horizon_spatial_64_vs_128')
        for axis,key in zip(axes.flat,('velocity','vorticity','deviatoric_stress','apparent_viscosity')):
            values=(tensor[(check['run'],check['reference'])]['full_field_per_time_relative_l2']
                    if key=='deviatoric_stress' else check['errors'][key]['full_field_per_time_relative_l2'])
            axis.semilogy(check['times'],np.maximum(values,1e-16),label=title)
    for axis,title in zip(axes.flat,('Velocidad u','Vorticidad ω','Esfuerzo τ (Frobenius)','Viscosidad η')):
        axis.set_title(title);axis.set_xlabel('Tiempo adimensional');axis.set_ylabel('Error relativo L₂');axis.grid(alpha=.25)
    axes[0,0].legend()
    fig.suptitle('Campo completo: 64² frente a referencia de 128², horizonte 0–2')
    save(fig,'full_field_refinement')

    fig,axes=plt.subplots(1,3,figsize=(10.5,3.5),layout='constrained')
    for name,title in MODELS:
        for axis,key in zip(axes,('velocity','vorticity','deviatoric_stress')):
            values=[]
            for grid in (64,128):
                check=next(c for c in comparisons if c['run']==f'{name}_{grid}_short'
                           and c['reference']==f'{name}_256_short')
                metric=(tensor[(check['run'],check['reference'])]['full_field_maximum_per_time_relative_l2']
                        if key=='deviatoric_stress' else check['errors'][key]['full_field_maximum_per_time_relative_l2'])
                values.append(metric)
            axis.semilogy([64,128],np.maximum(values,1e-16),'-o',label=title)
    for axis,title in zip(axes,('Velocidad u','Vorticidad ω','Esfuerzo τ (Frobenius)')):
        axis.set_title(title);axis.set_xlabel('Puntos por dirección N');axis.set_xticks([64,128])
        axis.set_ylabel('Máximo guardado del error L₂');axis.grid(alpha=.25)
    axes[0].legend(fontsize=10)
    fig.suptitle('Campo completo frente a 256²: sólo t=0 y t=0.1, horizonte corto')
    save(fig,'spatial_refinement_publication')

    sensitivities=json.loads((OUTPUT/'sensitivity.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(10.5,3.7),layout='constrained')
    quantities=(('velocity','u'),('vorticity','ω'),('deviatoric_stress','τ Frobenius'),('apparent_viscosity','η'))
    for axis,parameter,title in zip(axes,('quadrature_factor','regularization_m','shear_epsilon'),
          ('Cuadratura constitutiva','Cambio de m','Cambio de ε')):
        checks=[c for c in sensitivities if f'_{parameter}_' in c['run']]
        for key,label in quantities:
            values=[]
            for check in checks:
                values.append(tensor[(check['run'],check['reference'])]['full_field_maximum_per_time_relative_l2']
                              if key=='deviatoric_stress' else check['errors'][key]['full_field_maximum_per_time_relative_l2'])
            axis.semilogy(range(len(checks)),np.maximum(values,1e-16),'o-',label=label)
        labels=[c['run'].split(parameter+'_')[1].replace('p','.') for c in checks]
        axis.set_xticks(range(len(checks)),labels)
        axis.set_xlabel({'quadrature_factor':'Factor de cuadratura','regularization_m':'m','shear_epsilon':'ε'}[parameter])
        axis.set_title(title);axis.set_ylabel('Diferencia relativa frente al control');axis.grid(alpha=.25)
    axes[0].legend(fontsize=9)
    fig.suptitle('Sensibilidades separadas, malla 64²: m y ε cambian el modelo; T=0.1')
    save(fig,'sensitivity_publication')
    provenance(metas,primary_names+['full_field_refinement','spatial_refinement_publication','sensitivity_publication'],False)
    print('Seven compact publication companions written')


if __name__=='__main__':main()
