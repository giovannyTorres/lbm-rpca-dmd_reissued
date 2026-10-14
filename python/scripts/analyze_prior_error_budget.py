#!/usr/bin/env python3
"""Exact spectral bias/noise decomposition for the fixed physical prior.

This characterizes an inherited exploratory prior, never selects a cutoff by
clean-reference error. Numerical arrays are not changed or resimulated.
"""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key]="1"
from pathlib import Path
import ast
import csv
import hashlib
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"python/scripts"))
import run_model_fidelity_benchmark as benchmark
OUTPUT=ROOT/"results/research/prior_spectral_budget_2026-10-14"
CUTOFFS=(4.,6.,8.,10.,12.,16.,24.,32.,40.)
LABELS={"ns_vortices":"NS: cuatro vórtices","ns_multiscale":"NS: multiescala",
        "oldroyd_b":"Oldroyd-B","fene_p":"FENE-P","hb_thinning":"HB adelgazante",
        "bingham":"Bingham","hb_thickening":"HB espesante"}

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as f:
        for value in iter(lambda:f.read(1024*1024),b""):digest.update(value)
    return digest.hexdigest()

def squared_norm(x):return float(np.vdot(x.ravel(),x.ravel()).real)

def project_spectrum(value,kx,ky,cutoff):
    k2=kx*kx+ky*ky
    dot=kx*value[:,0]+ky*value[:,1]
    factor=np.divide(dot,k2,out=np.zeros_like(dot),where=k2>0)
    projected=value-np.stack((kx*factor,ky*factor),axis=1)
    return projected*(np.hypot(kx,ky)<=cutoff)

def curl_spectrum(value,kx,ky):return 1j*(kx*value[:,1]-ky*value[:,0])

def write_table(name,rows):
    with (OUTPUT/name).open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    OUTPUT.mkdir(exist_ok=True,parents=True);(OUTPUT/"figures").mkdir(exist_ok=True)
    config=json.loads(benchmark.DEFAULT_CONFIG.read_text())
    floors=[];budget=[];inputs=[];max_defect=0.
    for case,relative in benchmark.FLOW_FILES.items():
        path=ROOT/"results/research"/relative
        with np.load(path) as source:clean=source["velocity"].copy();times=source["times"].copy()
        n=clean.shape[-1];spacing=2*np.pi/n
        kx,ky=np.meshgrid(2*np.pi*np.fft.fftfreq(n,d=spacing),2*np.pi*np.fft.fftfreq(n,d=spacing))
        # All declared cutoffs are below Nyquist; clean solver states have no
        # Nyquist power. Setting derivative Nyquist to zero agrees with real
        # sampled spectral derivatives used in the physical metrics.
        deriv_kx=kx.copy();deriv_ky=ky.copy()
        deriv_kx[:,n//2]=0.;deriv_ky[n//2,:]=0.
        signal=np.fft.fft2(clean);omega=curl_spectrum(signal,deriv_kx,deriv_ky)
        unorm=squared_norm(signal);wnorm=squared_norm(omega)
        inputs.append(dict(case=case,path=str(path.relative_to(ROOT)),sha256=sha(path),
                           shape=list(clean.shape),times=times.tolist()))
        for cutoff in CUTOFFS:
            optimal=project_spectrum(signal,kx,ky,cutoff)
            floors.append(dict(case=case,cutoff=cutoff,
                velocity_bias_relative_l2=np.sqrt(squared_norm(optimal-signal)/unorm),
                vorticity_bias_relative_l2=np.sqrt(squared_norm(curl_spectrum(optimal,deriv_kx,deriv_ky)-omega)/wnorm)))
        rms=float(np.sqrt(np.mean(clean**2)))
        for kind in config["noise_models"]:
            for seed in config["seeds"]:
                noise=benchmark.observation_noise(clean.shape,times,kind,seed,rms,config)
                noise_spec=np.fft.fft2(noise)
                for cutoff in CUTOFFS:
                    optimal=project_spectrum(signal,kx,ky,cutoff)
                    retained=project_spectrum(noise_spec,kx,ky,cutoff)
                    for key,reference,bias,perturbation in (
                        ("velocity",signal,optimal-signal,retained),
                        ("vorticity",omega,curl_spectrum(optimal,deriv_kx,deriv_ky)-omega,
                         curl_spectrum(retained,deriv_kx,deriv_ky))):
                        scale=squared_norm(reference)
                        b=squared_norm(bias)/scale;s=squared_norm(perturbation)/scale
                        exact=squared_norm(bias+perturbation)/scale
                        defect=abs(exact-b-s)/max(exact,np.finfo(float).tiny)
                        max_defect=max(max_defect,defect)
                        budget.append(dict(case=case,noise=kind,seed=seed,cutoff=cutoff,observable=key,
                            bias_relative_l2=np.sqrt(b),retained_noise_relative_l2=np.sqrt(s),
                            identity_prior_error_relative_l2=np.sqrt(exact),
                            squared_decomposition_relative_defect=defect))
    assert max_defect<2e-10,max_defect
    write_table("clean_prior_floors.csv",floors);write_table("identity_bias_noise_budget.csv",budget)
    selected=[r for r in floors if r["cutoff"]==10.]
    report=dict(complete=True,selected_physical_cutoff=10.,cutoffs=list(CUTOFFS),
        selected_prior_floors=selected,identity_budget_rows=len(budget),source_datasets=inputs,
        max_squared_decomposition_relative_defect=max_defect,
        definition="For a band-limited candidate on the periodic grid, the discarded clean spectral component is orthogonal to all retained errors. For identity+H+band, squared error = squared clean bias + squared retained projected noise.",
        statistical_scope="Three noise seeds are descriptive realizations of fixed trajectories, not population confidence intervals. Cutoff curves characterize bias; no cutoff is selected or estimator tuned against clean data.",
        derivative_scope="Real sampled Fourier derivative with zero Nyquist multiplier. All cutoffs below Nyquist; original clean solver spectra exclude Nyquist.")
    (OUTPUT/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    (OUTPUT/"provenance.json").write_text(json.dumps(dict(script_sha256=sha(Path(__file__)),
        noise_driver_sha256=sha(Path(benchmark.__file__)), noise_function_ast_sha256=hashlib.sha256(ast.dump(next(node for node in ast.parse(Path(benchmark.__file__).read_text()).body if isinstance(node,ast.FunctionDef) and node.name=="observation_noise"),include_attributes=False).encode()).hexdigest(), config_sha256=sha(benchmark.DEFAULT_CONFIG),
        sources=inputs,source_role="Only clean spectral loss and synthetic-noise characterization; no estimator fit and no clean hyperparameter selection."),indent=2)+"\n")
    plt.rcParams.update({"font.size":12,"axes.titlesize":13,"axes.labelsize":12,"legend.fontsize":10})
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout="constrained")
    for axis,key,title in zip(axes,("velocity","vorticity"),("Velocidad","Vorticidad")):
        for case in LABELS:
            rows=[r for r in floors if r["case"]==case]
            values=[100*r[key+"_bias_relative_l2"] for r in rows]
            axis.plot(CUTOFFS,values,marker="o",ms=3,label=LABELS[case])
        axis.axvline(10.,ls="--",color="black",lw=1)
        axis.set_xscale("log");axis.set_xticks([4,8,10,16,32,40],[4,8,10,16,32,40])
        axis.set_xlabel("Corte físico k [longitud⁻¹]");axis.set_ylabel("Pérdida del campo limpio [%]")
        axis.set_title(title);axis.grid(alpha=.2)
    axes[1].legend(ncol=2)
    fig.suptitle("El prior elimina señal física: pérdida mínima por banda\nCurvas diagnósticas; corte heredado k≤10, sin selección contra el limpio")
    for ext in ("pdf","png"):fig.savefig(OUTPUT/"figures"/("clean_bandwidth_bias."+ext),dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8.4),layout="constrained")
    for i,case in enumerate(("ns_multiscale","oldroyd_b")):
        for j,kind in enumerate(("gaussian_iid","spatial_gaussian")):
            axis=axes[i,j];subset=[r for r in budget if r["case"]==case and r["noise"]==kind and r["observable"]=="vorticity"]
            curves={key:[] for key in ("bias","retained_noise","identity_prior_error")}
            lower=[];upper=[]
            for cutoff in CUTOFFS:
                group=[r for r in subset if r["cutoff"]==cutoff]
                for key in curves:curves[key].append(100*np.median([r[key+"_relative_l2"] for r in group]))
                lower.append(100*min(r["identity_prior_error_relative_l2"] for r in group))
                upper.append(100*max(r["identity_prior_error_relative_l2"] for r in group))
            for key,label,color in (("bias","Señal descartada","#a75518"),("retained_noise","Ruido retenido","#1877a8"),("identity_prior_error","Error total del control","#111111")):
                axis.plot(CUTOFFS,curves[key],marker="o",ms=3,label=label,color=color)
            axis.fill_between(CUTOFFS,lower,upper,color="#111111",alpha=.12,label="Mín.–máx. de 3 semillas")
            axis.axvline(10.,ls="--",color="black",lw=1);axis.set_xscale("log");axis.set_yscale("log")
            axis.set_xticks([4,8,10,16,32,40],[4,8,10,16,32,40]);axis.grid(alpha=.2)
            axis.set_xlabel("Corte físico k [longitud⁻¹]");axis.set_ylabel("Error de vorticidad [%]")
            axis.set_title(LABELS[case]+(" — IID" if j==0 else " — espacial, σ núcleo=0.10"))
    axes[0,0].legend(fontsize=9)
    fig.suptitle("Identidad con el mismo prior: sesgo físico y ruido residual\nLos cuadrados se suman exactamente; la suma directa de errores no es válida")
    for ext in ("pdf","png"):fig.savefig(OUTPUT/"figures"/("identity_bias_noise_tradeoff."+ext),dpi=180)
    plt.close(fig)
    (OUTPUT/"PRIOR_ERROR_BUDGET.md").write_text("# Error del prior físico\n\nEl corte no distingue por sí solo ruido y estructura física. La identidad con el mismo prior permite identificar cuánto gana un modelo temporal frente al control espacial. Para campos periódicos y una banda fija, Parseval da una descomposición ortogonal del error del control: error² = pérdida de señal² + ruido proyectado retenido². Las curvas de nueve cortes son diagnósticas y no seleccionan parámetros usando la referencia.\n\nEl error de vorticidad de cualquier candidato limitado a esa banda no puede ser menor que la norma relativa de la vorticidad limpia fuera de la banda. La comparación conserva como objetivo todas las escalas calculadas. Los extremos de tres semillas describen esas realizaciones.\n\nVer `summary.json`, `clean_prior_floors.csv` e `identity_bias_noise_budget.csv` para valores y fuentes exactos.\n")
    print(json.dumps({"complete":True,"budget_rows":len(budget),"max_defect":max_defect,"selected_floors":selected}))

if __name__=="__main__":main()
