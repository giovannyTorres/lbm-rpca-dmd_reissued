#!/usr/bin/env python3
"""Generate a genuinely matched periodic vortex bank and visual comparison."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
from pathlib import Path
import sys
import csv
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'python/src'))
from fluid_denoise.research.dns import simulate_periodic_ns
from fluid_denoise.research.viscoelastic_dns import simulate_viscoelastic_periodic
from fluid_denoise.research.viscoplastic_dns import simulate_viscoplastic_periodic
from fluid_denoise.research.metrics import physical_metrics
from fluid_denoise.research.runner import assert_finite, dataset_digest, write_json


def vorticity(dataset):
    velocity=dataset['velocity'];ny,nx=velocity.shape[-2:]
    kx,ky=np.meshgrid(2*np.pi*np.fft.fftfreq(nx,d=dataset['dx']),2*np.pi*np.fft.fftfreq(ny,d=dataset['dy']))
    return np.fft.ifft2(1j*kx*np.fft.fft2(velocity[:,1])-1j*ky*np.fft.fft2(velocity[:,0])).real


def main():
    output=ROOT/'results/research/rheology_matched_2026-10-14'
    (output/'datasets').mkdir(parents=True,exist_ok=True)
    common=dict(nx=32,case='interacting_vortices',amplitude=.6,dt=.005,n_steps=240,sample_every=8)
    nominal_viscosity=.06;yield_stress=.02;regularization=10.;epsilon=.001
    characteristic_length=.065*2*np.pi
    reference_shear=common['amplitude']/characteristic_length
    yield_viscosity=yield_stress*(-np.expm1(-regularization*reference_shear))/reference_shear
    regularized_shear=np.hypot(reference_shear,epsilon)
    models={'Newtonian':simulate_periodic_ns(viscosity=nominal_viscosity,**common)}
    for name,model in [('Oldroyd-B','oldroyd_b'),('FENE-P','fene_p')]:
        models[name]=simulate_viscoelastic_periodic(model=model,viscosity_s=.02,viscosity_p=.04,
                       relaxation_time=.5,extensibility_squared=12.,stress_diffusivity=0.,**common)
    material_parameters={}
    for name,n in [('HB n=0.6',.6),('Bingham',1.),('HB n=1.4',1.4)]:
        consistency=(nominal_viscosity-yield_viscosity)/regularized_shear**(n-1)
        material_parameters[name]={'flow_index':n,'consistency':consistency,'yield_stress':yield_stress,
                                  'regularization_m':regularization,'shear_epsilon':epsilon}
        models[name]=simulate_viscoplastic_periodic(**material_parameters[name],**common)
    initial=models['Newtonian']['velocity'][0]
    times=models['Newtonian']['times']
    curves={};rows=[]
    for name,dataset in models.items():
        assert_finite(dataset)
        np.testing.assert_allclose(dataset['velocity'][0],initial,atol=3e-14)
        np.testing.assert_allclose(dataset['times'],times,atol=1e-14)
        safe_name=name.lower().replace(' ','_').replace('=','').replace('.','p')
        np.savez_compressed(output/'datasets'/(safe_name+'.npz'),**{k:v for k,v in dataset.items() if isinstance(v,np.ndarray)})
        write_json(output/'datasets'/(safe_name+'_metadata.json'),{'metadata':dataset['metadata'],'dataset_sha256':dataset_digest(dataset),
                   'dx':dataset['dx'],'dy':dataset['dy'],'shape':list(dataset['velocity'].shape)})
        omega=vorticity(dataset);curves[name]=omega
        comparisons=physical_metrics(models['Newtonian'],dataset['velocity'])
        rows.append({'model':name,'velocity_relative_l2_vs_newtonian':comparisons['velocity_relative_l2'],
             'vorticity_relative_l2_vs_newtonian':comparisons['vorticity_relative_l2'],
             'kinetic_energy_relative_l2_vs_newtonian':comparisons['kinetic_energy_relative_l2'],
             'divergence_rms':comparisons['divergence_rms'],'final_velocity_relative_l2_vs_newtonian':float(np.linalg.norm(dataset['velocity'][-1]-models['Newtonian']['velocity'][-1])/np.linalg.norm(models['Newtonian']['velocity'][-1]))})
    with (output/'summary.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    write_json(output/'provenance.json',{'common':common,'density':1.,'nominal_viscosity':nominal_viscosity,
         'reference_shear_rate':reference_shear,'material_parameters':material_parameters,
         'comparison_scope':'same initial velocity, grid, density, domain and times; HB viscosity matched at declared shear, FENE nominal zero-shear material constants; distinct constitutive physics',
         'limitations':'periodic vortices, no cylinder wake, no chaos claim; differences are constitutive changes, not reconstruction errors',
         'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    maximum=max(np.max(np.abs(values)) for values in curves.values())
    fig,axes=plt.subplots(2,3,figsize=(12,8),layout='constrained')
    pictures=[]
    for axis,(name,values) in zip(axes.flat,curves.items()):
        pictures.append(axis.imshow(values[0],origin='lower',extent=[0,2*np.pi,0,2*np.pi],cmap='RdBu_r',vmin=-maximum,vmax=maximum))
        axis.set_title(name);axis.set_xlabel('x');axis.set_ylabel('y')
    fig.colorbar(pictures[0],ax=axes,label='Vorticidad [1/tiempo]')
    title=fig.suptitle('Mismo estado inicial; modelos constitutivos distintos; t=0')
    def update(index):
        for picture,values in zip(pictures,curves.values()):picture.set_data(values[index])
        title.set_text(f'Vórtices periódicos con distintas reologías; t={times[index]:.2f}')
        return pictures+[title]
    animation=FuncAnimation(fig,update,frames=len(times),interval=120,blit=False)
    animation.save(output/'vortices_rheology.gif',writer=PillowWriter(fps=8),dpi=100)
    update(len(times)-1);fig.savefig(output/'final_vorticity.png',dpi=180);fig.savefig(output/'final_vorticity.pdf')
    plt.close(fig)
    write_json(output/'completion.json',{'complete':True,'models':len(models),'snapshots':len(times),'all_initial_velocities_equal':True})
    print(output)


if __name__=='__main__':main()
