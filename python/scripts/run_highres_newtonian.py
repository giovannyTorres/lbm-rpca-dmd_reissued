#!/usr/bin/env python3
"""Larger-grid periodic Navier--Stokes experiments with inspectable evolution.

Run from the repository root. Numerical references are finer runs of the same
solver, not analytical truth. Outputs explicitly distinguish resolved-mode
error from total field error, and report effective adaptive timesteps.
"""
from __future__ import annotations

import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python/src"))
import numpy as np
import scipy
from scipy.signal import resample
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from fluid_denoise.research.dns import simulate_periodic_ns, taylor_green_reference

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 180, "pdf.fonttype": 42})
CASES = ("interacting_vortices", "decaying_turbulence")
LABELS = {"interacting_vortices": "Interacción de cuatro vórtices", "decaying_turbulence": "Turbulencia bidimensional en decaimiento"}
OPTIONS = {"viscosity": .01, "amplitude": .8, "seed": 17, "length_x": 2*np.pi,
           "length_y": 2*np.pi, "cfl": .35}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wave_numbers(n):
    modes = np.rint(np.fft.fftfreq(n)*n)
    return np.meshgrid(modes, modes)


def omega(velocity):
    n = velocity.shape[-1]
    kx, ky = wave_numbers(n)
    return np.fft.ifft2(1j*kx*np.fft.fft2(velocity[1])-1j*ky*np.fft.fft2(velocity[0])).real


def restrict(field, n, active=True):
    reduced = resample(resample(field, n, axis=-1), n, axis=-2)
    if active:
        kx, ky = wave_numbers(n)
        mask = (np.abs(kx)<n/3)&(np.abs(ky)<n/3)
        reduced = np.fft.ifft2(np.fft.fft2(reduced, axes=(-2,-1))*mask, axes=(-2,-1)).real
    return reduced


def ratio_norm(a, b):
    denominator = np.linalg.norm(b.ravel())
    return float(np.linalg.norm((a-b).ravel())/denominator) if denominator else None


def diagnostics(velocity):
    n = velocity.shape[-1]
    kx, ky = wave_numbers(n)
    radius = np.hypot(kx, ky)
    active = (np.abs(kx)<n/3)&(np.abs(ky)<n/3)
    tail = active & (radius >= .8*np.max(radius[active]))
    # Also report tails relative to the component cutoff, to expose energetic
    # axes hidden by a radial corner-based tail definition.
    cutoff_tail = active & (np.maximum(np.abs(kx), np.abs(ky)) >= .8*n/3)
    rows = []
    for frame in velocity:
        transformed = np.fft.fft2(frame, axes=(-2,-1))/n**2
        mode_e = .5*np.sum(abs(transformed)**2, axis=0)
        vorticity_modes = 1j*kx*transformed[1]-1j*ky*transformed[0]
        mode_z = .5*abs(vorticity_modes)**2
        divergence = np.fft.ifft2(1j*kx*transformed[0]+1j*ky*transformed[1]).real*n**2
        w = omega(frame)
        rows.append({"energy": float(.5*np.mean(np.sum(frame**2,axis=0))),
                     "enstrophy": float(.5*np.mean(w**2)),
                     "divergence_rms": float(np.sqrt(np.mean(divergence**2))),
                     "radial_energy_tail": float(np.sum(mode_e[tail])/np.sum(mode_e)),
                     "component_energy_tail": float(np.sum(mode_e[cutoff_tail])/np.sum(mode_e)),
                     "component_enstrophy_tail": float(np.sum(mode_z[cutoff_tail])/np.sum(mode_z)),
                     "max_abs_vorticity": float(np.max(abs(w))),
                     "max_speed": float(np.max(np.sqrt(np.sum(frame**2,axis=0))))})
    return rows


def run_one(out, case, n, dt, duration, frames, reuse):
    tag = f"{case}_{n}_dt{dt:g}"
    data_path = out/"datasets"/(tag+".npz")
    meta_path = out/"datasets"/(tag+".json")
    options = {"case":case,"nx":n,"dt":dt,"n_steps":int(round(duration/dt)),
               "sample_every":int(round(duration/dt/(frames-1))), **OPTIONS}
    assert np.isclose(options["dt"]*options["n_steps"], duration)
    assert options["n_steps"] % options["sample_every"] == 0
    solver_hash = sha(ROOT/"python/src/fluid_denoise/research/dns.py")
    if reuse and data_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta["parameters"] != options or meta["solver_sha256"] != solver_hash:
            raise RuntimeError(f"Refusing stale reuse: {tag}")
        if meta.get("npz_sha256") != sha(data_path):
            raise RuntimeError(f"Refusing corrupted dataset reuse: {tag}")
        with np.load(data_path) as stored:
            velocity, times = stored["velocity"], stored["times"]
        diagnostics_hash=sha(Path(__file__).resolve())
        if meta.get("diagnostics_script_sha256") != diagnostics_hash:
            meta["diagnostics"]=diagnostics(velocity)
            meta["diagnostics_script_sha256"]=diagnostics_hash
            write_json(meta_path,meta)
        print(f"REUSED {tag}",flush=True)
        return velocity, times, meta
    print(f"RUN {tag} T={duration:g}",flush=True)
    wall = time.perf_counter(); cpu = time.process_time()
    try:
        run = simulate_periodic_ns(**options)
        seconds = time.perf_counter()-wall
        cpu_seconds = time.process_time()-cpu
        velocity = run["velocity"]
        times = run["times"]
        if not np.all(np.isfinite(velocity)) or len(times) != frames:
            raise RuntimeError("Nonfinite output or unexpected frame count")
        meta = {"status":"completed","parameters":options,"solver_sha256":solver_hash,
                "wall_seconds":seconds,"cpu_seconds":cpu_seconds,"metadata":run["metadata"],
                "shape":list(velocity.shape),"density":"constant rho=1; implicit rather than duplicated in NPZ",
                "diagnostics":diagnostics(velocity),"diagnostics_script_sha256":sha(Path(__file__).resolve())}
        np.savez_compressed(data_path,velocity=velocity,times=times)
        meta["npz_sha256"] = sha(data_path)
        write_json(meta_path,meta)
        print(f"DONE {tag}: {seconds:.2f}s, {meta['metadata']['internal_steps']} internal steps",flush=True)
        return velocity,times,meta
    except Exception as exc:
        write_json(out/(tag+"_failure.json"),{"status":"failed","parameters":options,
                   "exception":repr(exc),"wall_seconds":time.perf_counter()-wall})
        raise


def compare(run, ref, times, case, n, ref_n, kind="spatial"):
    rows=[]
    for i,t in enumerate(times):
        reference = ref[i]
        common = restrict(reference,n)
        w = omega(run[i]); ref_w = omega(reference)
        common_w = restrict(ref_w,n)
        expanded = restrict(run[i],ref_n,active=False)
        expanded_w = restrict(w,ref_n,active=False)
        energy = .5*np.mean(np.sum(run[i]**2,axis=0))
        ref_energy = .5*np.mean(np.sum(reference**2,axis=0))
        enstrophy = .5*np.mean(w**2); ref_z = .5*np.mean(ref_w**2)
        rows.append({"case":case,"comparison":kind,"nx":n,"reference_nx":ref_n,"time":float(t),
                     "velocity_common_modes_relative_l2":ratio_norm(run[i],common),
                     "vorticity_common_modes_relative_l2":ratio_norm(w,common_w),
                     "velocity_full_field_relative_l2":ratio_norm(expanded,reference),
                     "vorticity_full_field_relative_l2":ratio_norm(expanded_w,ref_w),
                     "energy_relative_error":float(abs(energy-ref_energy)/ref_energy),
                     "enstrophy_relative_error":float(abs(enstrophy-ref_z)/ref_z),
                     "reference_unresolved_energy_fraction":float(max(0,1-np.mean(np.sum(common**2,axis=0))/np.mean(np.sum(reference**2,axis=0)))),
                     "reference_unresolved_enstrophy_fraction":float(max(0,1-np.mean(common_w**2)/np.mean(ref_w**2)))})
    return rows


def spectra(frame):
    n = frame.shape[-1]
    kx,ky=wave_numbers(n); radius=np.hypot(kx,ky)
    shells=np.floor(radius+.5).astype(int)
    active=(np.abs(kx)<n/3)&(np.abs(ky)<n/3)
    transformed=np.fft.fft2(frame,axes=(-2,-1))/n**2
    density=.5*np.sum(abs(transformed)**2,axis=0)
    energy=np.bincount(shells[active].ravel(),weights=density[active].ravel())
    vorticity_modes=1j*kx*transformed[1]-1j*ky*transformed[0]
    enstrophy=np.bincount(shells[active].ravel(),weights=(.5*abs(vorticity_modes)**2)[active].ravel())
    return np.arange(len(energy)),energy,enstrophy


def mathematical_checks():
    """Independent known Fourier modes verify exported physical quantities."""
    n=32;coordinates=np.arange(n)*2*np.pi/n;x,y=np.meshgrid(coordinates,coordinates)
    field=np.stack((2*np.sin(3*x)*np.cos(2*y),-3*np.cos(3*x)*np.sin(2*y)))
    analytic_curl=13*np.sin(3*x)*np.sin(2*y)
    _,e,z=spectra(field)
    interpolation_error=float(np.max(abs(restrict(restrict(field,16),32,active=False)-field)))
    curl_error=float(np.max(abs(omega(field)-analytic_curl)))
    # E=(2²+3²)/8 and Z=(3²+2²)²/8 are analytical identities,
    # independent of implementation FFT normalization.
    energy_error=float(abs(np.sum(e)-13/8))
    enstrophy_error=float(abs(np.sum(z)-169/8))
    if max(interpolation_error,curl_error,energy_error,enstrophy_error)>1e-11:
        raise AssertionError("Fourier interpolation/curl/spectrum mathematical verification failed")
    n=64;coordinates=np.arange(n)*2*np.pi/n;x,y=np.meshgrid(coordinates,coordinates);a=.1
    rich=np.stack((np.sin(x)*np.cos(y)+a*np.sin(11*x)*np.cos(11*y),
                   -np.cos(x)*np.sin(y)-a*np.cos(11*x)*np.sin(11*y)))
    coarse=restrict(rich,16)
    expected_u=a/np.sqrt(1+a*a);expected_w=11*a/np.sqrt(1+(11*a)**2)
    actual_u=ratio_norm(restrict(coarse,64,active=False),rich)
    actual_w=ratio_norm(restrict(omega(coarse),64,active=False),omega(rich))
    if max(abs(expected_u-actual_u),abs(expected_w-actual_w))>1e-11:
        raise AssertionError("Full-field metric failed independent unresolved-mode test")
    return {"retained_mode_interpolation_max_error":interpolation_error,
            "analytic_curl_max_error":curl_error,"ring_summed_energy_analytic_error":energy_error,
            "ring_summed_enstrophy_analytic_error":enstrophy_error,
            "unresolved_mode_test":{"expected_velocity_full_field_error":float(expected_u),
              "observed_velocity_full_field_error":actual_u,"expected_vorticity_full_field_error":float(expected_w),
              "observed_vorticity_full_field_error":actual_w,
              "common_modes_error":ratio_norm(coarse,restrict(rich,16))},
            "status":"passed","scope":"Known divergence-free Fourier modes and deliberate discarded high mode verify interpolation, curl, Parseval units and full-field versus common-mode errors."}


def save_figure(fig,out,name):
    fig.savefig(out/"figures"/(name+".pdf"),bbox_inches="tight")
    fig.savefig(out/"figures"/(name+".png"),bbox_inches="tight",dpi=180)
    plt.close(fig)


def visual_evolution(out,case,reference,times,animate=True):
    n=reference.shape[-1]; indices=np.linspace(0,len(times)-1,5,dtype=int)
    ws=[omega(reference[i]) for i in indices]
    speeds=[np.sqrt(np.sum(reference[i]**2,axis=0)) for i in indices]
    wlim=max(np.max(abs(w)) for w in ws); ulim=max(np.max(s) for s in speeds)
    fig,axes=plt.subplots(2,5,figsize=(15,6.2),layout="constrained")
    for j,i in enumerate(indices):
        iw=axes[0,j].imshow(ws[j],origin="lower",extent=(0,2*np.pi,0,2*np.pi),cmap="RdBu_r",vmin=-wlim,vmax=wlim,interpolation="none")
        iu=axes[1,j].imshow(speeds[j],origin="lower",extent=(0,2*np.pi,0,2*np.pi),cmap="viridis",vmin=0,vmax=ulim,interpolation="none")
        skip=max(1,n//16); coords=np.arange(n)*2*np.pi/n
        axes[1,j].quiver(coords[::skip],coords[::skip],reference[i,0,::skip,::skip],reference[i,1,::skip,::skip],color="white",alpha=.7,scale=20,linewidth=.2)
        axes[0,j].set_title(f"t={times[i]:g}")
        for ax in axes[:,j]:
            ax.set_xlabel("x");ax.set_xticks([0,np.pi,2*np.pi],["0","π","2π"])
            ax.set_yticks([0,np.pi,2*np.pi],["0","π","2π"])
    axes[0,0].set_ylabel("Vorticidad ω\ny");axes[1,0].set_ylabel("Rapidez |u|\ny")
    fig.colorbar(iw,ax=axes[0,:],label="ω [adimensional]",shrink=.85)
    fig.colorbar(iu,ax=axes[1,:],label="|u| [adimensional]",shrink=.85)
    fig.suptitle(f"{LABELS[case]} · {n}×{n} · escalas de color constantes")
    save_figure(fig,out,case+"_evolution")

    fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout="constrained")
    for i,color in zip(indices,plt.cm.plasma(np.linspace(.05,.8,len(indices)))):
        k,e,z=spectra(reference[i])
        for ax,values in zip(axes,(e,z)):
            valid=(k>0)&(values>1e-20)
            ax.loglog(k[valid],values[valid],label=f"t={times[i]:g}",color=color)
    for ax in axes:
        ax.set_xlabel("Número de onda |k|");ax.grid(alpha=.2);ax.legend()
    axes[0].set_ylabel("E(k)");axes[0].set_title("Redistribución espectral de energía")
    axes[1].set_ylabel("Z(k)");axes[1].set_title("Enstrofía: evolución de escalas finas")
    fig.suptitle(LABELS[case]+f" · {n}² · espectros por anillos")
    save_figure(fig,out,case+"_spectra_evolution")

    if not animate:
        return

    fig,axes=plt.subplots(1,2,figsize=(9.5,4.6),layout="constrained")
    initial_w=omega(reference[0]); initial_speed=np.sqrt(np.sum(reference[0]**2,axis=0))
    imw=axes[0].imshow(initial_w,origin="lower",extent=(0,2*np.pi,0,2*np.pi),cmap="RdBu_r",vmin=-wlim,vmax=wlim,interpolation="none")
    imu=axes[1].imshow(initial_speed,origin="lower",extent=(0,2*np.pi,0,2*np.pi),cmap="viridis",vmin=0,vmax=ulim,interpolation="none")
    for ax in axes: ax.set_xlabel("x");ax.set_ylabel("y")
    fig.colorbar(imw,ax=axes[0],label="ω");fig.colorbar(imu,ax=axes[1],label="|u|")
    title=fig.suptitle(f"{LABELS[case]} · {n}×{n} · t=0")
    axes[0].set_title("Vorticidad");axes[1].set_title("Rapidez")
    def update(i):
        imw.set_data(omega(reference[i]));imu.set_data(np.sqrt(np.sum(reference[i]**2,axis=0)))
        title.set_text(f"{LABELS[case]} · {n}×{n} · t={times[i]:.2f}")
        return imw,imu,title
    FuncAnimation(fig,update,frames=len(times),interval=125,blit=False).save(out/"figures"/(case+"_evolution.gif"),writer=PillowWriter(fps=8),dpi=110)
    plt.close(fig)


def visual_comparison(out,case,runs,times,all_rows):
    grids=sorted(runs); ref=runs[grids[-1]][0]
    fig,axes=plt.subplots(2,len(grids),figsize=(13.5,6.4),layout="constrained")
    ws=[omega(runs[n][0][-1]) for n in grids];lim=max(np.max(abs(w)) for w in ws)
    reference_w=omega(ref[-1])
    errors=[restrict(omega(runs[n][0][-1]),grids[-1],active=False)-reference_w for n in grids]
    elim=max(np.max(abs(w)) for w in errors[:-1]);elim=max(elim,1e-15)
    for j,n in enumerate(grids):
        im=axes[0,j].imshow(ws[j],origin="lower",extent=(0,2*np.pi,0,2*np.pi),cmap="RdBu_r",vmin=-lim,vmax=lim,interpolation="none")
        err=axes[1,j].imshow(errors[j],origin="lower",extent=(0,2*np.pi,0,2*np.pi),cmap="PuOr",vmin=-elim,vmax=elim,interpolation="none")
        axes[0,j].set_title(f"{n}×{n}")
        for ax in axes[:,j]: ax.set_xlabel("x");ax.set_ylabel("y")
    axes[1,-1].text(.5,.5,"Referencia numérica\n(error consigo misma = 0)",ha="center",va="center",transform=axes[1,-1].transAxes,fontsize=9,bbox=dict(facecolor="white",alpha=.85,edgecolor="none"))
    fig.colorbar(im,ax=axes[0,:],label="ω final",shrink=.8)
    fig.colorbar(err,ax=axes[1,:],label=f"Error completo de ω\n(malla menor interpolada a {grids[-1]}²)",shrink=.8)
    fig.suptitle(f"{LABELS[case]} · comparación de mallas · t={times[-1]:g}")
    save_figure(fig,out,case+"_grid_comparison")

    fig,axes=plt.subplots(2,3,figsize=(14,8),layout="constrained")
    colors=plt.cm.viridis(np.linspace(.1,.9,len(grids)))
    for n,color in zip(grids,colors):
        _,_,meta=runs[n];diags=meta["diagnostics"]
        for ax,key in zip(axes[0,:2],("energy","enstrophy")):
            ax.plot(times,[r[key] for r in diags],label=f"{n}²",color=color)
        k,e,z=spectra(runs[n][0][-1]);valid=(k>0)&(e>1e-20)
        axes[0,2].loglog(k[valid],e[valid],color=color,label=f"{n}²")
        if n<grids[-1]:
            rows=[r for r in all_rows if r["case"]==case and r["nx"]==n and r["comparison"]=="spatial"]
            for ax,key in zip(axes[1,:2],("velocity_full_field_relative_l2","vorticity_full_field_relative_l2")):
                ax.semilogy(times[1:],[r[key] for r in rows][1:],color=color,label=f"{n}²")
            axes[1,2].semilogy(n,rows[-1]["velocity_full_field_relative_l2"],"o",color=color)
            axes[1,2].semilogy(n,rows[-1]["vorticity_full_field_relative_l2"],"s",color=color)
    axes[0,0].set_title("Energía cinética por unidad de masa");axes[0,1].set_title("Enstrofía: sensibilidad a gradientes")
    axes[0,2].set_title("Espectro de energía final por anillo")
    axes[0,2].set_xlabel("Número de onda |k|");axes[0,2].set_ylabel("E(k)")
    axes[1,0].set_title("Error L₂ de velocidad frente a referencia");axes[1,1].set_title("Error L₂ de vorticidad frente a referencia")
    for ax in (axes[0,0],axes[0,1],axes[1,0],axes[1,1]): ax.set_xlabel("t")
    axes[1,2].set_title("Error final al refinar la malla");axes[1,2].set_xlabel("N por eje")
    axes[1,2].set_xticks(grids[:-1]);axes[1,2].plot([],[],"ko",label="Velocidad");axes[1,2].plot([],[],"ks",label="Vorticidad")
    for ax in axes.flat: ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle(LABELS[case]+" · referencia numérica "+str(grids[-1])+"²")
    save_figure(fig,out,case+"_diagnostics")

    # Audit the periodic unforced kinetic-energy balance independently of
    # the time integrator. Quadrature of sampled fields has its own O(dt²)
    # error and is not an equation residual at internal solver timesteps.
    from scipy.integrate import cumulative_trapezoid
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout="constrained")
    balance_rows=[]
    for n,color in zip(grids,colors):
        diags=runs[n][2]["diagnostics"]
        energy=np.array([row["energy"] for row in diags])
        enstrophy=np.array([row["enstrophy"] for row in diags])
        integrated=energy[0]-2*OPTIONS["viscosity"]*cumulative_trapezoid(enstrophy,times,initial=0)
        defect=(energy-integrated)/energy[0]
        axes[0].plot(times,energy/energy[0],color=color,label=f"E/E₀ · {n}²")
        axes[0].plot(times,integrated/energy[0],color=color,ls="--",alpha=.7)
        axes[1].plot(times,defect,color=color,label=f"{n}²")
        balance_rows.append({"nx":n,"maximum_absolute_normalized_sampled_energy_balance_defect":float(np.max(abs(defect))),
                             "final_normalized_sampled_energy_balance_defect":float(defect[-1])})
    axes[0].set_title("Balance físico: E(t) = E₀ − 2ν ∫ Z dt")
    axes[1].set_title("Defecto normalizado de cuadratura del balance")
    axes[1].set_ylabel("[E − (E₀−2ν∫Zdt)] / E₀")
    for ax in axes: ax.set_xlabel("t");ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle(LABELS[case]+" · integración trapezoidal de salidas cada Δt=0.1")
    save_figure(fig,out,case+"_energy_balance")
    write_json(out/(case+"_energy_balance.json"),{"quadrature":"cumulative trapezoid at sampled output times; includes sampling quadrature error",
                "equation":"E(t)=E(0)-2*nu*integral(Z dt); E=.5 mean |u|², Z=.5 mean omega²",
                "rows":balance_rows})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default="results/research/highres_newtonian_2026-10-14")
    parser.add_argument("--grids",type=int,nargs="+",default=[64,128,256,512])
    parser.add_argument("--duration",type=float,default=4.)
    parser.add_argument("--frames",type=int,default=41)
    parser.add_argument("--reuse",action="store_true")
    parser.add_argument("--no-animations",action="store_true")
    args=parser.parse_args()
    if len(set(args.grids))<2 or args.frames<5 or not np.isfinite(args.duration) or args.duration<=0:
        parser.error("At least two different grids, five frames and positive finite duration are required")
    out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True)
    (out/"datasets").mkdir(exist_ok=True);(out/"figures").mkdir(exist_ok=True)
    write_json(out/"mathematical_checks.json",mathematical_checks())
    grids=sorted(set(args.grids));reference_n=grids[-1];base_dt=.005
    started=time.perf_counter()
    write_json(out/"completion.json",{"status":"running","cases":list(CASES),"grids":grids})
    all_rows=[]; summaries=[]; temporal=[]; manifest=[]
    for case in CASES:
        runs={}
        for n in grids:
            runs[n]=run_one(out,case,n,base_dt,args.duration,args.frames,args.reuse)
            manifest.append({"case":case,"nx":n,"dt":base_dt,"path":str((out/"datasets"/f"{case}_{n}_dt{base_dt:g}.npz").relative_to(ROOT)),"wall_seconds":runs[n][2]["wall_seconds"]})
        times=runs[reference_n][1];reference=runs[reference_n][0]
        for n in grids[:-1]:
            assert np.array_equal(times,runs[n][1])
            rows=compare(runs[n][0],reference,times,case,n,reference_n)
            all_rows.extend(rows)
            summaries.append({"case":case,"nx":n,"reference_nx":reference_n,
                              "initial":rows[0],"final":rows[-1],
                              "maximum_velocity_full_field_error":max(r["velocity_full_field_relative_l2"] for r in rows),
                              "maximum_vorticity_full_field_error":max(r["vorticity_full_field_relative_l2"] for r in rows),
                              "maximum_component_enstrophy_tail":max(r["component_enstrophy_tail"] for r in runs[n][2]["diagnostics"]),
                              "meets_scoped_exploratory_spatial_targets":bool(
                                  max(r["velocity_full_field_relative_l2"] for r in rows)<=.01
                                  and max(r["vorticity_full_field_relative_l2"] for r in rows)<=.01
                                  and max(r["component_enstrophy_tail"] for r in runs[n][2]["diagnostics"])<=1e-4)})
        write_json(out/(case+"_grid_refinement.json"),{"case":case,"reference_nx":reference_n,
                  "grids":grids,"duration":args.duration,"frames":args.frames,
                  "grid_refinement":[row for row in summaries if row["case"]==case]})
        # 128² permits nominal timesteps .005/.0025/.00125 without severe
        # spatial-grid stability caps. Report every actual cap rather than
        # treating nominal dt as the effective temporal refinement.
        temporal_n=128 if 128 in runs else grids[0]
        dt_runs={base_dt:runs[temporal_n]}
        for step_dt in (.0025,.00125):
            dt_runs[step_dt]=run_one(out,case,temporal_n,step_dt,args.duration,args.frames,args.reuse)
            manifest.append({"case":case,"nx":temporal_n,"dt":step_dt,"path":str((out/"datasets"/f"{case}_{temporal_n}_dt{step_dt:g}.npz").relative_to(ROOT)),"wall_seconds":dt_runs[step_dt][2]["wall_seconds"]})
        dt_reference=dt_runs[.00125][0]
        last_error=None
        for step_dt in (.005,.0025):
            rows=compare(dt_runs[step_dt][0],dt_reference,times,case,temporal_n,temporal_n,kind=f"temporal_dt_{step_dt:g}")
            all_rows.extend(rows)
            meta=dt_runs[step_dt][2]["metadata"]
            error=rows[-1]["velocity_full_field_relative_l2"]
            temporal.append({"case":case,"nx":temporal_n,"dt":step_dt,"reference_dt":.00125,
                             "internal_dt_max":meta["internal_dt_max"],"internal_dt_min":meta["internal_dt_min"],
                             "stability_limited_steps":meta["stability_limited_steps"],
                             "final_velocity_relative_l2":error,"final_vorticity_relative_l2":rows[-1]["vorticity_full_field_relative_l2"],
                             "maximum_velocity_relative_l2":max(row["velocity_full_field_relative_l2"] for row in rows),
                             "maximum_vorticity_relative_l2":max(row["vorticity_full_field_relative_l2"] for row in rows),
                             "reference_internal_dt_max":dt_runs[.00125][2]["metadata"]["internal_dt_max"],
                             "reference_stability_limited_steps":dt_runs[.00125][2]["metadata"]["stability_limited_steps"],
                             "meets_scoped_exploratory_temporal_targets":bool(
                                 max(row["velocity_full_field_relative_l2"] for row in rows)<=1e-4
                                 and max(row["vorticity_full_field_relative_l2"] for row in rows)<=1e-4),
                             "observed_order":float(np.log2(last_error/error)) if last_error and error>1e-13 else None})
            last_error=error
        visual_evolution(out,case,reference,times,animate=not args.no_animations)
        visual_comparison(out,case,runs,times,all_rows)
        del dt_reference, dt_runs, reference, runs
    tg_rows=[]
    for step_dt in (.1,.05,.025):
        t=time.perf_counter()
        run=simulate_periodic_ns(case="taylor_green",nx=128,viscosity=.2,amplitude=.02,dt=step_dt,n_steps=int(round(1/step_dt)),sample_every=int(round(1/step_dt)))
        ref=taylor_green_reference(nx=128,times=run["times"],viscosity=.2,amplitude=.02)
        tg_rows.append({"dt":step_dt,"nx":128,"velocity_error":ratio_norm(run["velocity"][-1],ref["velocity"][-1]),"wall_seconds":time.perf_counter()-t,
                        "stability_limited_steps":run["metadata"]["stability_limited_steps"],"internal_dt_max":run["metadata"]["internal_dt_max"],
                        "scope":"exact single-mode Taylor-Green; not nonlinear spatial certification"})
    # Large viscosity caps the TG step at 128²; additionally perform the
    # classical independent low-grid RK4 exact refinement where nominal
    # timesteps are distinct. This separation avoids a false order claim.
    tg_order=[]
    for step_dt in (.1,.05,.025):
        run=simulate_periodic_ns(case="taylor_green",nx=16,viscosity=.2,amplitude=.02,dt=step_dt,n_steps=int(round(1/step_dt)),sample_every=int(round(1/step_dt)))
        ref=taylor_green_reference(nx=16,times=run["times"],viscosity=.2,amplitude=.02)
        error=ratio_norm(run["velocity"][-1],ref["velocity"][-1])
        tg_order.append({"dt":step_dt,"nx":16,"velocity_error":error,"actual_dt_max":run["metadata"]["internal_dt_max"],"stability_limited_steps":run["metadata"]["stability_limited_steps"],
                         "observed_order":float(np.log2(tg_order[-1]["velocity_error"]/error)) if tg_order else None})
    with (out/"comparison_timeseries.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    write_json(out/"summary.json",{"status":"completed","cases":list(CASES),"grids":grids,"reference_nx":reference_n,"duration":args.duration,"frames":args.frames,
        "parameters":OPTIONS,"grid_refinement":summaries,"temporal_refinement":temporal,
        "exploratory_resolution_targets":{"maximum_all_times_velocity_relative_l2":.01,
          "maximum_all_times_vorticity_relative_l2":.01,"maximum_component_enstrophy_tail":1e-4,
          "maximum_all_times_temporal_velocity_and_vorticity_relative_l2":1e-4,
          "interpretation":"Engineering criteria aligned with preceding 1% velocity/vorticity protocol; tail is a heuristic and targets are exploratory, not blind preregistration or universal DNS certification."},
        "taylor_green_high_grid":tg_rows,"taylor_green_distinct_rk4_refinement":tg_order,
        "resolution_geometry":[{"nx":n,"dx":float(2*np.pi/n),"gaussian_core_width":float(.065*2*np.pi),"cells_per_core_sigma":.065*n,"cells_per_initial_turbulent_wavelength_cutoff8":n/8,
                                "strict_component_cutoff":int(np.ceil(n/3)-1),"grid_cells":n*n,
                                "retained_fourier_vorticity_modes":int((2*(np.ceil(n/3)-1)+1)**2)} for n in grids],
        "initial_characteristic_scales":{"initial_speed_rms":OPTIONS["amplitude"],
          "domain_reynolds_number":float(OPTIONS["amplitude"]*2*np.pi/OPTIONS["viscosity"]),
          "gaussian_core_sigma_reynolds_number":float(OPTIONS["amplitude"]*.065*2*np.pi/OPTIONS["viscosity"]),
          "duration_in_domain_advection_times":float(args.duration*OPTIONS["amplitude"]/(2*np.pi)),
          "duration_in_gaussian_sigma_advection_times":float(args.duration*OPTIONS["amplitude"]/(.065*2*np.pi)),
          "limitation":"Core-sigma scales refer to interacting-vortex initial condition; domain Reynolds is a chosen global scale, not a universal flow classification."},
        "interpretation":{"reference_kind":"same-solver finer numerical run; Fourier restriction/prolongation preserves periodic fields",
        "velocity_full_field_error":"coarse field Fourier-prolongated to full fine grid, includes missing high modes",
        "common_modes_error":"fine reference restricted to coarse active strict-two-thirds modes, measures dynamics only",
        "effective_timestep_interpretation":"internal_dt_min includes shortened steps for output-time alignment, occasionally machine-scale roundoff steps; it must not be interpreted as a stiffness or stability estimate. internal_dt_max, stability_limited_steps and actual distinct nominal refinements are reported separately.",
        "dns_certified":False,"limitations":["Two-dimensional periodic unforced incompressible constant-density flow; no external cylinder wake, wall or 3D cascade.","Only seed17 and nu=.01 were studied; convergence applies to these initial states and finite time interval.","Spatial refinement and fixed128 temporal refinement are separate; finest512 grid also uses adaptive stability caps.","A smooth low spectral tail is necessary evidence but is not a universal DNS certificate.","No Lyapunov exponent or chaos claim; decaying turbulence is the seeded benchmark case name.","Mass conservation with fixed rho=1 is not a compressible-density balance experiment."]},"datasets":manifest})
    files=[ROOT/"python/src/fluid_denoise/research/dns.py",Path(__file__)]
    write_json(out/"provenance.json",{"command":sys.argv,"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "source_sha256":{str(p.relative_to(ROOT)):sha(p) for p in files},"python":sys.version,"numpy":np.__version__,"scipy":scipy.__version__,"matplotlib":matplotlib.__version__,"platform":platform.platform(),
        "threads":{key:os.environ[key] for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS")},
        "duration_wall_seconds":time.perf_counter()-started,"sum_recorded_nonlinear_solver_wall_seconds":sum(row["wall_seconds"] for row in manifest),
        "reuse_requested":args.reuse,"cost_scope":"solver wall/CPU includes initial construction, integration and output field construction; per-run diagnostics, compression, I/O and visual export excluded",
        "timing_limitation":"Shared computational host may run concurrent campaigns; these per-run timings are resource records, not controlled solver-speed comparisons."})
    write_json(out/"completion.json",{"status":"completed","nonlinear_spatial_runs":len(grids)*len(CASES),"additional_temporal_runs":4,"analytical_checks":6,"frames_per_nonlinear_run":args.frames,
               "largest_grid":reference_n,"spatial_cells_at_largest_grid":reference_n**2,"comparison_rows":len(all_rows),"failures":0,"duration_wall_seconds":time.perf_counter()-started})
    print(f"COMPLETE {out}",flush=True)

if __name__=="__main__":
    main()
