#!/usr/bin/env python3
"""Larger-grid regularized HB/Bingham vortex evolution and numerical checks.

No solver modifications, viscosity clipping, or exact-yield/DNS claims are made.
Long evolutions use 64/128 states; 256 is a shorter numerical-reference study.
"""
import os
for _thread_name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_thread_name] = '1'

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.signal import resample

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python/src'))
from fluid_denoise.research.viscoplastic_dns import simulate_viscoplastic_periodic
from fluid_denoise.research.dns import periodic_diagnostics
from fluid_denoise.research.runner import assert_finite, dataset_digest, write_json

OUTPUT = ROOT / 'results/research/highres_viscoplastic_2026-10-14'
NAMES = {0.6: ('HB adelgazante, n=0.6', 'hb_thinning'),
         1.0: ('Bingham regularizado', 'bingham'),
         1.4: ('HB espesante, n=1.4', 'hb_thickening')}
SOLVER = ROOT / 'python/src/fluid_denoise/research/viscoplastic_dns.py'
DNS = ROOT / 'python/src/fluid_denoise/research/dns.py'
SHARED = dict(case='interacting_vortices', amplitude=.8, density=1.,
              length_x=2*np.pi, length_y=2*np.pi, seed=0,
              diffusion_cfl=.8, cfl=.35, quadrature_factor=1.5,
              yield_shear_threshold=.001, max_internal_steps=200000)
REFERENCE_SHEAR = SHARED['amplitude'] / (.065 * 2*np.pi)
NOMINAL_VISCOSITY = .04


def source_hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), SOLVER, DNS)}


def material(n):
    ty, m, eps = .01, 10., .001
    yield_viscosity = ty * (-np.expm1(-m * REFERENCE_SHEAR)) / REFERENCE_SHEAR
    consistency = ((NOMINAL_VISCOSITY - yield_viscosity)
                   / np.hypot(REFERENCE_SHEAR, eps) ** (n - 1))
    return dict(flow_index=n, consistency=float(consistency), yield_stress=ty,
                regularization_m=m, shear_epsilon=eps)


def run(name, **params):
    """Cached data are reused only for identical configuration and source hashes."""
    config = {**SHARED, **params}
    npz_path = OUTPUT / 'datasets' / f'{name}.npz'
    meta_path = npz_path.with_suffix('.json')
    if meta_path.exists() and npz_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get('configuration') == config and meta.get('source_hashes') == source_hashes():
            with np.load(npz_path) as arrays:
                data = {key: arrays[key] for key in arrays.files}
            data.update(metadata=meta['metadata'], dx=meta['dx'], dy=meta['dy'], boundary='periodic')
            assert_finite(data)
            print(f'reused {name}', flush=True)
            return data, meta
    print(f'running {name}', flush=True)
    begin = time.perf_counter()
    cpu = time.process_time()
    data = simulate_viscoplastic_periodic(**config)
    elapsed = time.perf_counter() - begin
    cpu_seconds = time.process_time() - cpu
    assert_finite(data)
    meta = {'configuration': config, 'metadata': data['metadata'],
            'wall_seconds_simulation': elapsed, 'cpu_seconds_simulation': cpu_seconds,
            'dx': data['dx'], 'dy': data['dy'], 'dataset_sha256': dataset_digest(data),
            'source_hashes': source_hashes(), 'velocity_shape': list(data['velocity'].shape),
            'complete': True}
    np.savez_compressed(npz_path, **{k: v for k, v in data.items() if isinstance(v, np.ndarray)})
    write_json(meta_path, meta)
    print(json.dumps({'run': name, 'seconds': elapsed,
                      'internal_steps': data['metadata']['internal_steps'],
                      'rejected_steps': data['metadata']['rejected_steps']}), flush=True)
    return data, meta


def omega(run):
    velocity = run['velocity']
    ny, nx = velocity.shape[-2:]
    kx, ky = np.meshgrid(2*np.pi*np.fft.fftfreq(nx, d=run['dx']),
                         2*np.pi*np.fft.fftfreq(ny, d=run['dy']))
    return np.fft.ifft2(1j*kx*np.fft.fft2(velocity[:, 1])
                        -1j*ky*np.fft.fft2(velocity[:, 0])).real


def relative(a, b):
    return float(np.linalg.norm(a-b) / max(np.linalg.norm(b), np.finfo(float).tiny))


def restrict(field, ny, nx):
    # scipy's periodic Fourier resampling handles the even-grid Nyquist mode.
    return resample(resample(field, ny, axis=-2), nx, axis=-1)


def comparison(coarse, finer, name, reference_name, scope):
    np.testing.assert_allclose(coarse['times'], finer['times'], rtol=0, atol=2e-14)
    ny, nx = coarse['velocity'].shape[-2:]
    quantities = {key: (coarse[key], restrict(finer[key], ny, nx), finer[key])
                  for key in ('velocity', 'shear_rate', 'deviatoric_stress', 'apparent_viscosity')}
    finer_omega = omega(finer)
    quantities['vorticity'] = (omega(coarse), restrict(finer_omega, ny, nx), finer_omega)
    record = {'run': name, 'reference': reference_name, 'scope': scope,
              'reference_kind': 'same-solver numerical reference; not analytical truth',
              'spatial_transfer': 'periodic Fourier low-pass resampling with Nyquist handling',
              'times': coarse['times'].tolist(), 'errors': {}}
    for key, (a, b, full_reference) in quantities.items():
        per_time = [relative(x, y) for x, y in zip(a, b)]
        prolongated = restrict(a, *full_reference.shape[-2:])
        full_per_time = [relative(x, y) for x, y in zip(prolongated, full_reference)]
        record['errors'][key] = {'all_times_relative_l2': relative(a, b),
                                 'final_relative_l2': relative(a[-1], b[-1]),
                                 'maximum_per_time_relative_l2': max(per_time),
                                 'per_time_relative_l2': per_time,
                                 'common_band_definition': 'Fourier resampled fine reference on complete coarse output grid; constitutive fields may include Nyquist modes',
                                 'full_field_all_times_relative_l2': relative(prolongated, full_reference),
                                 'full_field_final_relative_l2': relative(prolongated[-1], full_reference[-1]),
                                 'full_field_maximum_per_time_relative_l2': max(full_per_time),
                                 'full_field_per_time_relative_l2': full_per_time,
                                 'full_field_definition': 'coarse output field periodically Fourier interpolated onto entire finer output grid, compared with unfiltered fine reference'}
    record['energy_relative_l2'] = relative(coarse['kinetic_energy'], finer['kinetic_energy'])
    record['dissipation_relative_l2'] = relative(coarse['dissipation_rate'], finer['dissipation_rate'])
    return record


def summary(name, data, meta):
    velocity = data['velocity']; initial, final = velocity[[0, -1]]
    scalar = np.sum(initial*final)/np.sum(initial**2)
    return {'run': name, 'grid': velocity.shape[-1], 'snapshots': len(data['times']),
            'duration': float(data['times'][-1]), 'flow_index': data['metadata']['flow_index'],
            'wall_seconds_simulation': meta['wall_seconds_simulation'],
            'cpu_seconds_simulation': meta['cpu_seconds_simulation'],
            'internal_steps': data['metadata']['internal_steps'],
            'rejected_steps': data['metadata']['rejected_steps'],
            'internal_dt_max': data['metadata']['internal_dt_max'],
            'internal_dt_min': data['metadata']['internal_dt_min'],
            'maximum_stage_viscosity': data['metadata']['maximum_stage_viscosity'],
            'relative_energy_budget_residual': float(np.max(np.abs(data['energy_budget_residual']))/data['kinetic_energy'][0]),
            'energy_fraction_final': float(data['kinetic_energy'][-1]/data['kinetic_energy'][0]),
            'nonrescaling_relative_change': relative(final, scalar*initial),
            'final_relative_change_from_initial': relative(final, initial),
            'max_divergence_rms': float(max(periodic_diagnostics(velocity, data['dx'], data['dy'])['divergence_rms'])),
            'minimum_dissipation': float(data['dissipation_rate'].min()),
            'fraction_stress_below_yield_final': float(np.mean(~data['over_yield_stress'][-1])),
            'fraction_shear_below_threshold_final': float(np.mean(~data['approximate_yielded'][-1])),
            'maximum_energy_increase': float(max(0., np.max(np.diff(data['kinetic_energy']))))}


def save(fig, name, dpi=190):
    fig.savefig(OUTPUT / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(OUTPUT / f'{name}.png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def spectrum(field):
    ny, nx = field.shape[-2:]
    kx, ky = np.meshgrid(np.fft.fftfreq(nx)*nx, np.fft.fftfreq(ny)*ny)
    shells = np.rint(np.hypot(kx, ky)).astype(int)
    values = .5*np.abs(np.fft.fft2(field))**2/(nx*ny)**2
    return np.bincount(shells.ravel(), weights=values.ravel())


def figures(main_runs, checks, animate=True):
    fine = {n: main_runs[(n, 128)] for n in NAMES}
    curves = {n: omega(data) for n, data in fine.items()}
    times = next(iter(fine.values()))['times']
    max_omega = max(float(np.abs(field).max()) for field in curves.values())
    chosen = np.linspace(0, len(times)-1, 5).round().astype(int)
    fig, axes = plt.subplots(3, 5, figsize=(15, 9), layout='constrained')
    for row, (n, fields) in enumerate(curves.items()):
        for col, index in enumerate(chosen):
            im = axes[row, col].imshow(fields[index], origin='lower', cmap='RdBu_r',
                  vmin=-max_omega, vmax=max_omega, extent=(0, 2*np.pi, 0, 2*np.pi))
            axes[row, col].set_title(f't={times[index]:.2f}')
            axes[row, col].set_xticks([0, np.pi, 2*np.pi], ['0', 'π', '2π'])
            axes[row, col].set_yticks([0, np.pi, 2*np.pi], ['0', 'π', '2π'])
            if col == 0: axes[row, col].set_ylabel(NAMES[n][0]+'\ny')
            if row == 2: axes[row, col].set_xlabel('x')
    fig.colorbar(im, ax=axes, shrink=.85, label='Vorticidad (adimensional), escala común')
    fig.suptitle('Evolución de vórtices: momentum constitutivo completo, malla 128×128')
    save(fig, 'vorticity_evolution')

    fig, axes = plt.subplots(3, 4, figsize=(13, 9), layout='constrained')
    keys = ('shear_rate', 'apparent_viscosity', 'stress_invariant')
    titles = ('Tasa de corte γ̇', 'Viscosidad aparente η', 'Invariante del esfuerzo |τ|',
              'Esfuerzo ≤ τ_y (proxy)')
    limits = {k: max(float(run[k][-1].max()) for run in fine.values()) for k in keys}
    for row, (n, data) in enumerate(fine.items()):
        for col, key in enumerate(keys):
            im = axes[row, col].imshow(data[key][-1], origin='lower', cmap='viridis',
                       vmin=0, vmax=limits[key], extent=(0, 2*np.pi, 0, 2*np.pi))
            fig.colorbar(im, ax=axes[row, col], shrink=.78)
        axes[row, 3].imshow(~data['over_yield_stress'][-1], origin='lower', cmap='Greys',
                 vmin=0, vmax=1, extent=(0, 2*np.pi, 0, 2*np.pi))
        for col, ax in enumerate(axes[row]):
            ax.set_title(titles[col]); ax.set_xlabel('x'); ax.set_ylabel('y')
            ax.set_xticks([0, np.pi, 2*np.pi], ['0', 'π', '2π'])
            ax.set_yticks([0, np.pi, 2*np.pi], ['0', 'π', '2π'])
        axes[row, 0].set_ylabel(NAMES[n][0]+'\ny')
    fig.suptitle(f'Campos constitutivos a t={times[-1]:.2f}; el proxy no identifica regiones rígidas exactas')
    save(fig, 'constitutive_fields')

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout='constrained')
    for n, data in fine.items():
        label=NAMES[n][0]
        axes[0,0].plot(times, data['kinetic_energy']/data['kinetic_energy'][0], label=label)
        axes[0,1].plot(times, .5*np.mean(curves[n]**2,axis=(-2,-1)), label=label)
        axes[0,2].plot(times, data['dissipation_rate'], label=label)
        axes[1,0].plot(times, data['energy_budget_residual']/data['kinetic_energy'][0], label=label)
        axes[1,1].plot(times, np.mean(~data['over_yield_stress'],axis=(-2,-1)), label=label)
        for index, linestyle in ((0, '--'), (-1, '-')):
            spec=spectrum(curves[n][index]); k=np.arange(len(spec))
            axes[1,2].loglog(k[1:], np.maximum(spec[1:],1e-30), linestyle, label=label+(' inicial' if index==0 else ' final'))
    labels = ('Energía cinética E/E₀', 'Enstrofía ⟨ω²⟩/2', 'Disipación constitutiva',
              'Residuo del balance / E₀', 'Fracción esfuerzo≤τ_y (proxy)', 'Espectro de enstrofía')
    for ax,title in zip(axes.flat,labels):
        ax.set_title(title); ax.set_xlabel('Número de onda k' if ax is axes[1,2] else 'Tiempo adimensional');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);axes[1,2].legend(fontsize=6)
    save(fig,'evolution_diagnostics')

    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for n in NAMES:
        check=next(c for c in checks if c['run']==f'{NAMES[n][1]}_64' and c['scope']=='long_horizon_spatial_64_vs_128')
        axes[0].semilogy(check['times'],np.maximum(check['errors']['velocity']['per_time_relative_l2'],1e-16),label=NAMES[n][0])
        axes[1].semilogy(check['times'],np.maximum(check['errors']['vorticity']['per_time_relative_l2'],1e-16),label=NAMES[n][0])
        for grid,style in ((64,'--'),(128,'-')):
            field=omega(main_runs[(n,grid)])[-1];sp=spectrum(field)
            axes[2].loglog(np.arange(1,len(sp)),np.maximum(sp[1:],1e-30),style,label=f'n={n}, N={grid}')
    axes[0].set_title('Velocidad: 64 frente a 128');axes[1].set_title('Vorticidad: 64 frente a 128')
    axes[2].set_title('Espectro final por resolución')
    for ax in axes[:2]:ax.set_xlabel('Tiempo');ax.set_ylabel('Error relativo L₂');ax.grid(alpha=.2)
    axes[2].set_xlabel('Número de onda');axes[2].set_ylabel('Enstrofía por cascarón');axes[2].grid(alpha=.2)
    axes[0].legend(fontsize=7);axes[2].legend(fontsize=6)
    save(fig,'resolution_comparison')

    if animate:
        fig,axes=plt.subplots(1,3,figsize=(11,4),layout='constrained');pictures=[]
        for axis,(n,fields) in zip(axes,curves.items()):
            pictures.append(axis.imshow(fields[0],origin='lower',cmap='RdBu_r',vmin=-max_omega,vmax=max_omega,
                            extent=(0,2*np.pi,0,2*np.pi)))
            axis.set_title(NAMES[n][0]);axis.set_xlabel('x');axis.set_ylabel('y')
        fig.colorbar(pictures[0],ax=axes,label='Vorticidad, escala común',shrink=.8)
        title=fig.suptitle('Malla 128×128; tiempo=0')
        def update(index):
            for picture,values in zip(pictures,curves.values()):picture.set_data(values[index])
            title.set_text(f'Momentum 2D con distintas reologías; malla128×128; t={times[index]:.2f}')
            return pictures+[title]
        animation=FuncAnimation(fig,update,frames=len(times),interval=140)
        animation.save(OUTPUT/'viscoplastic_evolution.gif',writer=PillowWriter(fps=7),dpi=120)
        plt.close(fig)


def prefix(data, maximum_time):
    length=np.count_nonzero(data['times']<=maximum_time+1e-12)
    return {key: value[:length] if isinstance(value,np.ndarray) and value.ndim
            and value.shape[0]==len(data['times']) and key!='mask' else value
            for key,value in data.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-256',action='store_true',help='Omit only the short 256 reference checks')
    parser.add_argument('--no-animation',action='store_true')
    args=parser.parse_args()
    (OUTPUT/'datasets').mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();rows=[];main_runs={};checks=[]
    try:
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    except (OSError,subprocess.CalledProcessError):commit=None
    write_json(OUTPUT/'completion.json',{'complete':False,'status':'running','start_utc':datetime.now(timezone.utc).isoformat()})
    # The long horizon is identical in all three materials and both grids.
    # Bingham is less explicitly stiff and yields the first larger-grid data early.
    for n in (1., 1.4, .6):
        for grid in (64,128):
            name=f'{NAMES[n][1]}_{grid}'
            data,meta=run(name,nx=grid,dt=.02,n_steps=100,sample_every=5,**material(n))
            main_runs[(n,grid)]=data;rows.append(summary(name,data,meta))
        checks.append(comparison(main_runs[(n,64)],main_runs[(n,128)],
                    f'{NAMES[n][1]}_64',f'{NAMES[n][1]}_128','long_horizon_spatial_64_vs_128'))
        if not args.skip_256:
            name=f'{NAMES[n][1]}_256_short'
            data,meta=run(name,nx=256,dt=.02,n_steps=5,sample_every=5,**material(n))
            rows.append(summary(name,data,meta))
            for grid in (64,128):
                checks.append(comparison(prefix(main_runs[(n,grid)],.1),data,
                      f'{NAMES[n][1]}_{grid}_short',name,'short_horizon_spatial_vs_256'))
    # Temporal check changes the actual maximum allowed dt on a fixed state grid.
    temporal=[]
    for dt in (.0004,.0002,.0001):
        name=f'hb_thinning_128_dt{str(dt).replace(".","p")}'
        data,meta=run(name,nx=128,dt=dt,n_steps=int(round(.1/dt)),
                      sample_every=int(round(.05/dt)),**material(.6))
        temporal.append((name,data));rows.append(summary(name,data,meta))
    for name,data in temporal[:-1]:
        checks.append(comparison(data,temporal[-1][1],name,temporal[-1][0],'fixed_grid_temporal_refinement'))
    # Constitutive regularization and quadrature are different sensitivity axes.
    sensitivities=[]
    reference,reference_meta=run('hb_thinning_64_short_control',nx=64,dt=.005,n_steps=20,sample_every=10,**material(.6))
    rows.append(summary('hb_thinning_64_short_control',reference,reference_meta))
    for axis,value in (('quadrature_factor',2.),('regularization_m',5.),
                       ('regularization_m',20.),('shear_epsilon',.002),('shear_epsilon',.0005)):
        params=material(.6);params[axis]=value
        name=f'hb_thinning_64_{axis}_{str(value).replace(".","p")}'
        data,meta=run(name,nx=64,dt=.005,n_steps=20,sample_every=10,**params)
        rows.append(summary(name,data,meta))
        sensitivities.append(comparison(data,reference,name,'hb_thinning_64_short_control',
                'quadrature_sensitivity' if axis=='quadrature_factor' else 'constitutive_model_sensitivity'))
    write_json(OUTPUT/'comparisons.json',checks)
    write_json(OUTPUT/'sensitivity.json',sensitivities)
    with (OUTPUT/'summary.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    np.testing.assert_allclose(main_runs[(.6,128)]['velocity'][0],main_runs[(1.,128)]['velocity'][0],rtol=0,atol=3e-14)
    np.testing.assert_allclose(main_runs[(.6,128)]['velocity'][0],main_runs[(1.4,128)]['velocity'][0],rtol=0,atol=3e-14)
    figures(main_runs,checks,not args.no_animation)
    provenance={'created_utc':datetime.now(timezone.utc).isoformat(),'source_commit':commit,
        'source_hashes':source_hashes(),'common_configuration':SHARED,
        'matching_shear_rate':REFERENCE_SHEAR,'matching_apparent_viscosity':NOMINAL_VISCOSITY,
        'material_parameters':{NAMES[n][1]:material(n) for n in NAMES},
        'long_evolution':{'state_grids':[64,128],'duration':2.,'snapshots':21},
        'short_reference':{'state_grid':256,'duration':.1,'executed':not args.skip_256},
        'resolution_interpretation':'Long 64/128 comparison and short 64/128/256 comparison are separate horizons. A larger mesh alone is not a representativeness or DNS certificate.',
        'spatial_comparison':'Both fine same-solver field low-pass Fourier sampled onto coarse output grid and coarse field Fourier prolonged onto full unfiltered fine reference; all saved common times and maximum errors',
        'matching_scope':'same initial velocity/density/domain/output times and eta_app at one declared shear. Constitutive curves remain different; no universal Reynolds equivalence.',
        'temporal_check':'128 fixed state grid, maximum allowed dt 0.0004/0.0002/0.0001; actual dt diagnostics are reported',
        'sensitivity_interpretation':'m and epsilon change the constitutive model; quadrature_factor changes nonlinear integration error; none is mesh convergence',
        'limitations':['periodic unforced inelastic 2D flow; no cylinder, walls, elasticity or certified chaos',
          'regularized yield indicators are declared proxies, not exactly unyielded rigid regions',
          'nonpolynomial stress is not exactly dealiased by 3/2 padding',
          'stress/shear reference fields are numerical, not experimental measurements',
          'energy balance uses padded constitutive quadrature; output-grid quadrature may disagree'],
        'viscosity_cap':None,'dns_certification':False,
        'total_wall_seconds':time.perf_counter()-start,'runs':len(rows)}
    write_json(OUTPUT/'provenance.json',provenance)
    write_json(OUTPUT/'completion.json',{'complete':True,'runs':len(rows),'long_evolutions':6,
        'long_primary_state_grid':128,'short_reference_state_grid':None if args.skip_256 else 256,
        'all_initial_velocities_equal_at_fixed_grid':True,'figures_pdf_png':4,
        'gif_generated':not args.no_animation,'source_hashes':source_hashes(),
        'total_wall_seconds':time.perf_counter()-start})
    print(OUTPUT,flush=True)


if __name__=='__main__':
    main()
