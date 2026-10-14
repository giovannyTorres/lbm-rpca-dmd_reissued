#!/usr/bin/env python3
"""Render compact thesis figures from existing validated fields, without integration."""
from __future__ import annotations
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("highres_helpers",ROOT/"python/scripts/run_highres_newtonian.py")
helper=importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":14,"axes.titlesize":15,"axes.labelsize":14,
                     "xtick.labelsize":12,"ytick.labelsize":12,"legend.fontsize":12,
                     "figure.titlesize":16,"pdf.fonttype":42,"savefig.dpi":180})


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(fig,name):
    paths=[]
    for suffix in ("pdf","png"):
        path=OUT/"figures"/(name+"."+suffix)
        fig.savefig(path,bbox_inches="tight",dpi=180)
        paths.append(path)
    plt.close(fig)
    return paths


def evolution(case,velocity,times):
    selected=[int(np.argmin(abs(times-t))) for t in (0.,2.,4.)]
    assert np.allclose(times[selected],(0.,2.,4.),atol=1e-12)
    # Match the color limits of the original five-snapshot evolution figure.
    original_indices=np.linspace(0,len(times)-1,5,dtype=int)
    wlim=max(np.max(abs(helper.omega(velocity[i]))) for i in original_indices)
    ulim=max(np.max(np.sqrt(np.sum(velocity[i]**2,axis=0))) for i in original_indices)
    n=velocity.shape[-1]
    fig,axes=plt.subplots(2,3,figsize=(10.2,7.0),layout="constrained")
    coords=np.arange(n)*2*np.pi/n;skip=n//16
    for j,i in enumerate(selected):
        vorticity=helper.omega(velocity[i]);speed=np.sqrt(np.sum(velocity[i]**2,axis=0))
        imw=axes[0,j].imshow(vorticity,origin="lower",extent=(0,2*np.pi,0,2*np.pi),
                            vmin=-wlim,vmax=wlim,cmap="RdBu_r",interpolation="none")
        imu=axes[1,j].imshow(speed,origin="lower",extent=(0,2*np.pi,0,2*np.pi),
                            vmin=0,vmax=ulim,cmap="viridis",interpolation="none")
        axes[1,j].quiver(coords[::skip],coords[::skip],velocity[i,0,::skip,::skip],
                          velocity[i,1,::skip,::skip],color="white",alpha=.75,scale=20)
        axes[0,j].set_title(f"t={times[i]:g}",fontsize=16)
        for row,ax in enumerate(axes[:,j]):
            ax.set_xticks([0,np.pi,2*np.pi],["0","π","2π"])
            ax.set_yticks([0,np.pi,2*np.pi],["0","π","2π"])
            ax.set_xlabel("x" if row==1 else "")
            ax.set_ylabel("y" if j==0 else "")
    axes[0,0].set_ylabel("Vorticidad ω\ny")
    axes[1,0].set_ylabel("Rapidez |u|\ny")
    fig.colorbar(imw,ax=axes[0,:],label="ω",shrink=.82,pad=.025)
    fig.colorbar(imu,ax=axes[1,:],label="|u|",shrink=.82,pad=.025)
    fig.suptitle(helper.LABELS[case]+" · 512² · escalas constantes")
    return save(fig,case+"_evolution_publication")


def diagnostics(case,times,records):
    grids=(64,128,256,512)
    colors=plt.cm.viridis(np.linspace(.1,.9,4))
    fig,axes=plt.subplots(2,2,figsize=(10.2,7.0),layout="constrained")
    for n,color in zip(grids,colors):
        metadata=json.loads((OUT/"datasets"/f"{case}_{n}_dt0.005.json").read_text())
        series=metadata["diagnostics"]
        label=f"{n}²"+(" (referencia)" if n==512 else "")
        axes[0,0].plot(times,[row["energy"] for row in series],color=color,label=label,lw=2)
        axes[0,1].plot(times,[row["enstrophy"] for row in series],color=color,lw=2)
        if n<512:
            rows=[row for row in records if row["case"]==case and int(row["nx"])==n and row["comparison"]=="spatial"]
            assert len(rows)==len(times)
            assert np.allclose([float(row["time"]) for row in rows],times,atol=1e-12)
            # Skip only t=0 in the log display; all-times acceptance includes it.
            for ax,key in zip(axes[1,:],("velocity_full_field_relative_l2","vorticity_full_field_relative_l2")):
                ax.semilogy(times[1:],[float(row[key]) for row in rows[1:]],color=color,lw=2)
    axes[0,0].set_title("Energía cinética E(t)")
    axes[0,1].set_title("Enstrofía Z(t)")
    axes[1,0].set_title("Error del campo completo de velocidad")
    axes[1,1].set_title("Error del campo completo de vorticidad")
    axes[0,0].set_ylabel("E");axes[0,1].set_ylabel("Z")
    axes[1,0].set_ylabel("Error relativo L₂ de u")
    axes[1,1].set_ylabel("Error relativo L₂ de ω")
    for ax in axes.flat:
        ax.set_xlabel("t");ax.grid(alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=4,frameon=False,fontsize=13)
    fig.suptitle(helper.LABELS[case]+" · comparación con 512²")
    return save(fig,case+"_diagnostics_publication")


def main():
    start=time.perf_counter()
    with (OUT/"comparison_timeseries.csv").open(newline="") as handle:
        records=list(csv.DictReader(handle))
    artifacts=[];inputs={}
    for case in helper.CASES:
        path=OUT/"datasets"/f"{case}_512_dt0.005.npz"
        meta=json.loads(path.with_suffix(".json").read_text())
        checksum=digest(path)
        if checksum!=meta["npz_sha256"]:
            raise RuntimeError(f"Dataset hash mismatch: {path}")
        inputs[str(path.relative_to(ROOT))]=checksum
        with np.load(path) as data:
            velocity,times=data["velocity"],data["times"]
        artifacts.extend(evolution(case,velocity,times))
        artifacts.extend(diagnostics(case,times,records))
        del velocity
        print(f"Rendered publication companions: {case}",flush=True)
    helper.write_json(OUT/"publication_rendering.json",{
        "status":"completed","integration_performed":False,
        "sources_sha256":{str(Path(__file__).resolve().relative_to(ROOT)):digest(Path(__file__).resolve()),
                           "python/scripts/run_highres_newtonian.py":digest(ROOT/"python/scripts/run_highres_newtonian.py")},
        "datasets_sha256":inputs,"csv_sha256":digest(OUT/"comparison_timeseries.csv"),
        "artifacts_sha256":{str(path.relative_to(ROOT)):digest(path) for path in artifacts},
        "evolution":{"layout":"2 rows x 3 columns","times":[0.,2.,4.],"fields":["vorticity","speed with velocity arrows"],
                     "constant_color_scale_per_row":True,"matches_original_five_frame_color_limits":True},
        "diagnostics":{"layout":"2 rows x 2 columns","panels":["energy","enstrophy","velocity full-field relative L2","vorticity full-field relative L2"],
                       "log_error_display":"from t=.1; all-times acceptance still includes initial t=0"},
        "fontsize_points":{"base":14,"ticks":12,"titles":15,"figure_title":16,"global_legend":13},
        "numpy":np.__version__,"matplotlib":matplotlib.__version__,"wall_seconds":time.perf_counter()-start})

if __name__=="__main__":main()
