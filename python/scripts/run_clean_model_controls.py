#!/usr/bin/env python3
"""Mechanism diagnostics: exact clean inputs, inherited versus zero noise budget.

No observation noise is generated. Every child calls the primary estimator
with observed velocity and timestamps only, and subsequently evaluates the
same estimate as raw, Helmholtz and Helmholtz plus physical k<=10.
"""
from __future__ import annotations
import os
for _thread in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_thread]="1"
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"python/src"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import run_model_fidelity_benchmark as primary
from fluid_denoise.research.cost import hardware_metadata
from fluid_denoise.research.runner import json_safe

DEFAULT_CONFIG=ROOT/"configs/research/clean_model_controls.json"
DEST=ROOT/"results/research/clean_model_controls_2026-10-14"
VARIANTS=("raw","helmholtz","helmholtz_k10")
CONTRACT="inherited_20pct_regularization"
ZERO="known_zero_noise"
FLOW_LABELS={"ns_vortices":"Vórtices NS","ns_multiscale":"Multiescala NS","oldroyd_b":"Oldroyd-B",
             "fene_p":"FENE-P","hb_thinning":"HB n=0.6","bingham":"Bingham","hb_thickening":"HB n=1.4"}
METHOD_LABELS={"identity":"Identidad","pod_r8":"POD r=8","dmd_r8":"DMD r=8","tls_r8":"TLS-DMD r=8",
 "pcp":"PCP","stable_rpca":"SPCP","outlier_pursuit":"Outlier Pursuit","pcp_tls_r8":"PCP → TLS r=8",
 "stable_tls_r8":"SPCP → TLS r=8","optimized_r8":"Optimized DMD r=8"}


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def write_table(path,rows):
    primary.write_json(path.with_suffix(".json"),rows)
    if rows:
        fields=list(dict.fromkeys(key for row in rows for key in row))
        with path.with_suffix(".csv").open("w",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def summarize(records,out,config,oracles):
    rows=[]
    for record in records:
        diagnostics=record.get("diagnostics",{})
        common={key:record[key] for key in ("id","flow","method","contract","seed","status","noise")}
        common.update(true_noise_intensity=0.,estimator_noise_intensity=record["estimator_noise_intensity"],
            converged=bool(diagnostics.get("converged",False)),effective_rank=diagnostics.get("effective_rank"),
            iterations=diagnostics.get("iterations"),svd_calls=diagnostics.get("svd_calls"),
            flops_proxy=diagnostics.get("flops_proxy"),alpha=diagnostics.get("alpha"),beta=diagnostics.get("beta"),
            sparse_fraction=diagnostics.get("sparse_fraction"),sparse_norm=diagnostics.get("sparse_norm"),
            failure_type=record.get("failure_type"),failure=record.get("failure"))
        common.update({key:record.get("fit_profile",{}).get(key) for key in
                       ("wall_seconds","cpu_seconds","rss_peak_bytes","python_traced_peak_bytes")})
        evaluations=record.get("evaluations",[]) or [dict(variant=v) for v in VARIANTS]
        for evaluation in evaluations:
            row=dict(common,variant=evaluation["variant"],quality_pass=evaluation.get("quality_pass",False))
            row.update({key:evaluation.get("metrics",{}).get(key) for key in primary.METRIC_KEYS})
            row.update({key:evaluation.get("frame_errors",{}).get(key) for key in
                         ("velocity_relative_l2_max_saved","vorticity_relative_l2_max_saved")})
            frames=evaluation.get("frame_errors",{})
            row.update(velocity_relative_l2_last_saved=(frames.get("velocity_relative_l2") or [None])[-1],
                       vorticity_relative_l2_last_saved=(frames.get("vorticity_relative_l2") or [None])[-1])
            row["constitutive_observable_scope"]=evaluation.get("metrics",{}).get("constitutive_observable_scope")
            rows.append(row)
    write_table(out/"summary",rows)
    comparisons=[]
    for flow in config["flows"]:
        for method in config["known_zero_noise_methods"]:
            for variant in VARIANTS:
                a=next((r for r in rows if (r["flow"],r["method"],r["contract"],r["variant"])==(flow,method,CONTRACT,variant)),None)
                b=next((r for r in rows if (r["flow"],r["method"],r["contract"],r["variant"])==(flow,method,ZERO,variant)),None)
                if a is None or b is None:continue
                item=dict(flow=flow,method=method,variant=variant,primary_converged=a["converged"],zero_converged=b["converged"])
                for metric in primary.METRIC_KEYS:
                    item[metric+"_inherited_20pct"]=a[metric]
                    item[metric+"_known_zero"]=b[metric]
                    item[metric+"_delta_zero_minus_inherited"]=(b[metric]-a[metric]) if a[metric] is not None and b[metric] is not None else None
                comparisons.append(item)
    write_table(out/"regularization_contrasts",comparisons)
    band_floors=[]
    for flow in config["flows"]:
        identity=next((r for r in rows if (r["flow"],r["method"],r["contract"],r["variant"])==(flow,"identity",CONTRACT,"helmholtz_k10")),None)
        pod=next((r for r in rows if (r["flow"],r["method"],r["contract"],r["variant"])==(flow,"pod_r8",CONTRACT,"raw")),None)
        band_pod=next((r for r in rows if (r["flow"],r["method"],r["contract"],r["variant"])==(flow,"pod_r8",CONTRACT,"helmholtz_k10")),None)
        if identity and pod and band_pod:
            band_floors.append(dict(flow=flow,physical_cutoff=10.,clean_velocity_band_floor=identity["velocity_relative_l2"],
                clean_vorticity_band_floor=identity["vorticity_relative_l2"],pod_r8_vorticity_raw=pod["vorticity_relative_l2"],
                pod_r8_vorticity_band=band_pod["vorticity_relative_l2"],
                qualification="Orthogonal periodic spectral-band floor relative to the computed full clean field, not an absolute continuum DNS error."))
    write_table(out/"clean_band_floors",band_floors)
    expected=84
    completion=dict(complete=len(records)==expected,expected_fits=expected,recorded_fits=len(records),expected_evaluations=252,
        recorded_evaluations=sum(len(r.get("evaluations",[])) for r in records),
        failed_fits=sum(r["status"]!="ok" for r in records),
        nonconverged_fits=sum(r["status"]=="ok" and not r["diagnostics"].get("converged",False) for r in records),
        primary_contract_fits=sum(r["contract"]==CONTRACT for r in records),
        known_zero_noise_fits=sum(r["contract"]==ZERO for r in records),
        quality_passes_by_variant={v:sum(r["quality_pass"] for r in rows if r["variant"]==v) for v in VARIANTS},
        seed=config["seed"],noise_realizations=0,resource_repeats_per_fit=1,workers=1,
        scope="Same seven computed trajectories without observation noise; diagnostics of representational/regularization bias, not new simulations or population inference.")
    primary.write_json(out/"completion.json",completion)
    if completion["complete"]:
        assert len(rows)==252 and completion["primary_contract_fits"]==70 and completion["known_zero_noise_fits"]==14
        audit=[]
        def find(flow,method,contract,variant):
            return next(r for r in rows if (r["flow"],r["method"],r["contract"],r["variant"])==(flow,method,contract,variant))
        for oracle in oracles:
            flow=oracle["flow"];pod=find(flow,"pod_r8",CONTRACT,"raw")
            if pod["status"]=="ok":
                delta=pod["velocity_relative_l2"]-oracle["velocity_rank8_frobenius_lower_bound"]
                assert abs(delta)<2e-12
                audit.append(dict(flow=flow,check="POD r8 equals SVD Frobenius lower bound",absolute_difference=abs(delta),passed=True))
            for variant in VARIANTS:
                identity=find(flow,"identity",CONTRACT,variant)
                for zero_method,control_method in [("stable_rpca","identity"),("stable_tls_r8","tls_r8")]:
                    zero=find(flow,zero_method,ZERO,variant);control=find(flow,control_method,CONTRACT,variant)
                    for metric in ("velocity_relative_l2","vorticity_relative_l2"):
                        if zero[metric] is None or control[metric] is None:continue
                        delta=abs(zero[metric]-control[metric])
                        audit.append(dict(flow=flow,check=f"{zero_method} known-zero vs {control_method}: {variant}/{metric}",
                            absolute_difference=delta,passed=delta<1e-10))
                if variant in ("raw","helmholtz"):
                    assert identity["velocity_relative_l2"]<1e-12 and identity["vorticity_relative_l2"]<1e-12
        primary.write_json(out/"numerical_controls_audit.json",dict(checks=audit,all_passed=all(r["passed"] for r in audit),
            definition="Error-metric equality diagnostic; not identity of every predicted field or proof of constitutive dynamics."))
    return rows,completion


def draw(rows,out,config):
    if len(rows)!=252:return
    folder=out/"figures";folder.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":14,"axes.labelsize":14,
                        "xtick.labelsize":13,"ytick.labelsize":13,"pdf.fonttype":42,"savefig.dpi":170})
    def save(fig,name,caption,bottom=.09):
        fig.set_layout_engine("constrained",rect=(0,bottom,1,1-bottom))
        import textwrap
        fig.text(.5,.012,textwrap.fill(caption,width=155),ha="center",va="bottom",fontsize=11)
        for suffix in ("pdf","png"):fig.savefig(folder/f"{name}.{suffix}",bbox_inches="tight")
        plt.close(fig)
    methods=config["methods"]+[m+"__zero" for m in config["known_zero_noise_methods"]]
    for metric,title,name in [("velocity_relative_l2","Velocidad","clean_bias_velocity"),
                               ("vorticity_relative_l2","Vorticidad","clean_bias_vorticity")]:
        values=[];labels=[]
        for label in methods:
            method=label.removesuffix("__zero");contract=ZERO if label.endswith("__zero") else CONTRACT
            values.append([100*next(r[metric] for r in rows if
                (r["flow"],r["method"],r["contract"],r["variant"])==(flow,method,contract,"raw")) for flow in config["flows"]])
            labels.append(METHOD_LABELS[method]+("; presupuesto 0" if contract==ZERO else "; presupuesto 20%" if "stable" in method else ""))
        values=np.asarray(values);fig,ax=plt.subplots(figsize=(14,9),layout="constrained")
        im=ax.imshow(values,aspect="auto",vmin=0,vmax=50,cmap="magma_r")
        ax.set_xticks(range(7),[FLOW_LABELS[f] for f in config["flows"]],rotation=25,ha="right")
        ax.set_yticks(range(len(labels)),labels)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                value=values[i,j];label="<10⁻⁶" if value<1e-6 else f"{value:.3g}"
                row=next(r for r in rows if (r["flow"],r["method"],r["contract"],r["variant"])==
                    (config["flows"][j],methods[i].removesuffix("__zero"),ZERO if methods[i].endswith("__zero") else CONTRACT,"raw"))
                if not row["converged"]:label+=" !"
                ax.text(j,i,label,ha="center",va="center",fontsize=12,color="white" if value>25 else "black")
        ax.set_title(f"{title}: sesgo de reconstrucción sin ruido, salida sin posproceso (%)",fontsize=16,pad=14)
        fig.colorbar(im,ax=ax,label="Error L2 relativo (%)",shrink=.8)
        save(fig,name,"Mismas siete referencias128²; un ajuste determinista por contrato. SPCP conserva presupuesto20% heredado o usa control conocido0. Colores saturados50%; cifras sin truncar; !=sin convergencia. No son nuevas simulaciones.")
    fig,axes=plt.subplots(1,2,figsize=(14,8),layout="constrained",sharey=True)
    for ax,method in zip(axes,config["known_zero_noise_methods"]):
        for contract,color,offset,label in [(CONTRACT,"#b85b25",-.12,"Presupuesto heredado20%"),(ZERO,"#1965b0",.12,"Ruido conocido0")]:
            values=[100*next(r["velocity_relative_l2"] for r in rows if
               (r["flow"],r["method"],r["contract"],r["variant"])==(flow,method,contract,"raw")) for flow in config["flows"]]
            ax.scatter(values,np.arange(7)+offset,s=65,color=color,label=label)
        ax.set_title(METHOD_LABELS[method],fontsize=16);ax.set_xlabel("Error de velocidad sin ruido (%)")
        ax.set_yticks(range(7),[FLOW_LABELS[f] for f in config["flows"]]);ax.grid(axis="x",alpha=.2)
    axes[0].invert_yaxis()
    fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",bbox_to_anchor=(.5,.080),ncol=2,fontsize=12,frameon=False)
    save(fig,"regularization_controls","Solo cambia el presupuesto noise_intensity:.20→0; observación, rango, tolerancia y límites de iteración idénticos. Control0 de SPCP equivale a identidad; control0 de SPCP→TLS equivale a TLS. No selecciona un parámetro usando error limpio.",bottom=.18)


def report(out,rows,oracles):
    if len(rows)!=252:return
    lines=["# Controles sin contaminación: representación y regularización", "",
      "Estos 84 ajustes y 252 evaluaciones usan **las mismas siete trayectorias128²** que la comparación principal. "
      "La observación es copia exacta de la velocidad calculada: sin ruido aditivo y sin normalización RMS. "
      "El diseño es exploratorio posterior a observar el sesgo de Outlier Pursuit en el primer flujo con ruido; no está prerregistrado y no añade simulaciones nuevas.","",
      "- **70 ajustes:** diez modelos, siete trayectorias, parámetros idénticos al experimento20%. Para SPCP/SPCP→TLS se mantiene noise_intensity=.20; se interpreta como contrato regularizado heredado aplicado sin ruido, no como ruido observado20%.",
      "- **14 ajustes adicionales:** SPCP y SPCP→TLS usan noise_intensity=0 conocido. Solo cambia ese presupuesto; el resto del contrato permanece igual.",
      "- **252 evaluaciones:** cada estimación se evalúa sin posproceso, con H y con H+k≤10. No se vuelve a ajustar el modelo.",
      "- Una ejecución por contrato: la semilla101 fija Optimized DMD. No se añaden copias por semillas de ruido inexistente, ni repeticiones de coste como muestras independientes.","",
      "## Qué separan estos controles", "",
      "POD r=8 sobre datos limpios alcanza el límite de error Frobenius de una matriz de rango8, calculado directamente con SVD. "
      "Es un diagnóstico de representación y no una elección del rango para datos ruidosos; no certifica error óptimo de vorticidad. "
      "DMD añade una restricción de evolución lineal/modal y ajuste de amplitudes sobre la misma ventana, de modo que su error limpio puede superar el de POD. "
      "PCP y Outlier Pursuit seleccionan rango mediante penalización nuclear; el rango solicitado8 no se les impone. "
      "La convergencia del objetivo de separación no garantiza que toda dinámica limpia sea identificable como su componente de bajo rango.","",
      "![Sesgo de velocidad sin ruido](figures/clean_bias_velocity.png)","",
      "![Sesgo de vorticidad sin ruido](figures/clean_bias_vorticity.png)","",
      "## Papel del presupuesto de regularización", "",
      "Con presupuesto0, el problema SPCP implementado tiene alpha=beta=0 y admite infinitos minimizadores L+S=M; "
      "no identifica L=M de forma única. La implementación inicia S=0 y devuelve L≈M a redondeo, eliminando penalización sin efectuar limpieza. "
      "Por esa elección de implementación, el híbrido SPCP→TLS con presupuesto0 reproduce TLS aplicado directamente. "
      "Los nonzeros de S a redondeo pueden dar sparse_fraction alta con norma ínfima; no indican contaminación real en estas observaciones exactas. "
      "Estos controles separan contracción por regularización de la restricción modal; "
      "no demuestran que presupuesto0 sea una opción adecuada con contaminación real.","",
      "![Control del presupuesto heredado](figures/regularization_controls.png)","",
      "## Límites físicos", "",
      "Para Navier–Stokes se evalúa disipación viscosa y residual muestreado de vorticidad. Para los tres viscoplásticos se deriva viscosidad y esfuerzo de "
      "los gradientes de la velocidad candidata usando el cierre instantáneo regularizado y la misma ley material. El Frobenius simétrico pondera(xx,xy,yy) con(1,2,1). "
      "Para Oldroyd-B y FENE-P, esfuerzo polimérico y disipación total quedan nulos/no disponibles: memoria y conformation no se infieren de un estimador de velocidad. "
      "El piso limpio introducido por H+k10 se conserva en identidad+k10; no se adjudica al algoritmo ni se elimina de la evaluación.","",
      "### Estructura limpia eliminada por el prior de banda", "",
      "El filtro k≤10 retiene la misma banda física en todas las referencias, pero su sesgo depende del espectro de cada flujo. "
      "En un dominio periódico, eliminar modos fuera de banda impone un piso ortogonal de error respecto al campo limpio completo. "
      "Es una pérdida de estructuras calculadas, no una prueba de error absoluto frente a Navier–Stokes continuo.", "",
      "| Referencia | Piso de banda eᵤ (%) | Piso de banda eω (%) | POD r8 sin filtro eω (%) | POD r8 con banda eω (%) |",
      "| --- | --- | --- | --- | --- |"]
    for flow in FLOW_LABELS:
        def find(method,variant,key):
            return 100*next(r[key] for r in rows if (r["flow"],r["method"],r["contract"],r["variant"])==(flow,method,CONTRACT,variant))
        values=[find("identity","helmholtz_k10","velocity_relative_l2"),find("identity","helmholtz_k10","vorticity_relative_l2"),
                find("pod_r8","raw","vorticity_relative_l2"),find("pod_r8","helmholtz_k10","vorticity_relative_l2")]
        lines.append("| "+FLOW_LABELS[flow]+" | "+" | ".join(f"{v:.5g}" for v in values)+" |")
    lines.extend(["", "En NS multiescala el piso de vorticidad de la banda es 10.8366%: un modelo restringido a k≤10 no puede "
      "alcanzar eω≤10% respecto a esta referencia completa. POD r8 limpio tiene eω=4.2103% sin filtro y 11.1249% con la banda. "
      "Un prior útil para eliminar ruido iid puede empeorar fidelidad sobre una observación limpia.", "",
      "Incluso POD r8 sin ruido y con eᵤ≈0.969% en NS vórtices presenta residual extra muestreado de vorticidad "
      "normalizado≈1.979; TLS r8≈2.391. Conservar energía o un error integrado pequeño no garantiza dinámica fiel. "
      "Es un diagnóstico de la ecuación de vorticidad con derivadas temporales muestreadas, no del momento completo.", "",
      "## Valores por referencia", "",
      "| Referencia | Piso POD r8, eᵤ (%) | PCP limpio eω (%) | Outlier Pursuit limpio eᵤ (%) | TLS limpio eᵤ (%) | SPCP presupuesto20%, eᵤ (%) | SPCP presupuesto0, eᵤ (%) |",
      "| --- | --- | --- | --- | --- | --- | --- |"])
    for oracle in oracles:
        flow=oracle["flow"]
        def metric(method,key,contract=CONTRACT):
            return 100*next(r[key] for r in rows if (r["flow"],r["method"],r["contract"],r["variant"])==(flow,method,contract,"raw"))
        values=[100*oracle["velocity_rank8_frobenius_lower_bound"],metric("pcp","vorticity_relative_l2"),
                metric("outlier_pursuit","velocity_relative_l2"),metric("tls_r8","velocity_relative_l2"),
                metric("stable_rpca","velocity_relative_l2"),metric("stable_rpca","velocity_relative_l2",ZERO)]
        lines.append("| "+FLOW_LABELS[flow]+" | "+" | ".join(f"{v:.5g}" for v in values)+" |")
    lines.extend(["","Los valores son deterministas sobre estas referencias; no son medianas de realizaciones nuevas. "
                  "`summary.csv/json` preserva convergencia, todas las variantes, máximos entre snapshots guardados, estados finales y observables materiales; "
                  "`ledger.jsonl` conserva diagnósticos y series. `clean_pod_rank8_diagnostic.json` publica singular values y el límite Frobenius.","",
                  "```bash\npython python/scripts/run_clean_model_controls.py\n```",""])
    (out/"CLEAN_MODEL_CONTROLS.md").write_text("\n".join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=DEFAULT_CONFIG)
    parser.add_argument("--out",type=Path,default=DEST)
    parser.add_argument("--summarize-only",action="store_true")
    args=parser.parse_args();config=json.loads(args.config.read_text());out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    assert config["workers"]==1 and len(config["flows"])==7 and len(config["methods"])==10
    assert primary.sha256(primary.__file__)==config["helper_driver_sha256"]
    assert primary.sha256(ROOT/config["base_config_path"])==config["base_config_sha256"]
    inherited=config["estimator_configuration"]
    assert inherited==json.loads((ROOT/config["base_config_path"]).read_text())
    ledger=out/"ledger.jsonl"
    records=[json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    oracle_path=out/"clean_pod_rank8_diagnostic.json"
    oracles=json.loads(oracle_path.read_text()) if oracle_path.exists() else []
    if args.summarize_only:
        rows,complete=summarize(records,out,config,oracles);draw(rows,out,config);report(out,rows,oracles)
        provenance=json.loads((out/"provenance.json").read_text())
        provenance["script_fit_epochs"]=sorted({r["script_sha256"] for r in records})
        provenance["script_sha256"]=primary.sha256(__file__)
        provenance["analysis_only_amendment"]="Fit ledger and numerical estimator unchanged. Editorial amendment moves legends, restores shared-axis ordering, adds explicit orthogonal clean-band floors and mechanism discussion. Original executing fit script is archived by SHA under driver_epochs/."
        provenance["artifact_sha256"]={str(p.relative_to(out)):primary.sha256(p) for p in sorted(out.rglob("*"))
                                       if p.is_file() and p.name!="provenance.json"}
        primary.write_json(out/"provenance.json",provenance)
        print(json.dumps(complete,indent=2));return
    sources={str(p.relative_to(ROOT)):primary.sha256(p) for p in sorted((ROOT/"python/src/fluid_denoise/research").glob("*.py"))}
    previous_provenance=json.loads((out/"provenance.json").read_text()) if records and (out/"provenance.json").exists() else {}
    provenance=dict(configuration=config,configuration_sha256=primary.sha256(args.config),
        script_sha256=primary.sha256(__file__),helper_driver_sha256=primary.sha256(primary.__file__),
        estimator_source_hashes=sources,hardware=previous_provenance.get("hardware",hardware_metadata()),clean_input_sources={},
        design_note=config["design"],cost_note=config["cost_scope"],frozen_before_first_fit=True,
        freeze_utc=previous_provenance.get("freeze_utc",datetime.now(timezone.utc).isoformat()),estimator_receives_reference=False,
        script_fit_epochs=sorted({r["script_sha256"] for r in records}),
        script_epoch_note="Each record retains its executing script SHA; original fit script snapshot is archived under driver_epochs when an analysis-only editorial amendment is made. Numerical helper contract remains frozen.")
    campaign_hash=primary.fingerprint(dict(config=config,sources=sources,helper_driver=provenance["helper_driver_sha256"]))
    existing={r["id"]:r for r in records};assert len(existing)==len(records)
    primary.write_json(out/"configuration_frozen.json",config)
    primary.write_json(out/"provenance.json",provenance)
    # Whole-bank preflight before any estimator; release loaded arrays before fits.
    for flow in config["flows"]:
        checked,source=primary.load_flow(flow);provenance["clean_input_sources"][flow]=source;del checked
    primary.write_json(out/"input_preflight.json",dict(valid=True,checked_flows=7,sources=provenance["clean_input_sources"],
        validation_completed_before_any_new_fit=True))
    primary.write_json(out/"provenance.json",provenance)
    print("Preflight: all seven clean128² trajectories, byte/canonical digests and scientific source hashes validated.",flush=True)
    for flow in config["flows"]:
        data,source=primary.load_flow(flow);observed=data["velocity"].copy()
        observed_sha=array_sha(observed);clean_sha=array_sha(data["velocity"]);assert observed_sha==clean_sha
        for contract,methods in [(CONTRACT,config["methods"]),(ZERO,config["known_zero_noise_methods"])]:
            estimator_config=dict(inherited)
            if contract==ZERO:estimator_config["intensity_relative_rms"]=0.
            for method in methods:
                condition=dict(id=f"{flow}__clean__{contract}__{method}",flow=flow,noise="none",seed=config["seed"],method=method,contract=contract)
                params=primary.method_parameters(method,config["seed"],estimator_config)
                condition_hash=primary.fingerprint(dict(campaign=campaign_hash,condition=condition,
                    observed_sha256=observed_sha,clean_input_sha256=source["sha256"],estimator_parameters=params))
                if condition["id"] in existing:
                    if existing[condition["id"]]["fingerprint"]!=condition_hash:raise ValueError("Changed clean condition; use a new destination")
                    continue
                record=primary.isolated(observed,data,condition,estimator_config,None)
                record.update(condition,fingerprint=condition_hash,true_noise_intensity=0.,
                    estimator_noise_intensity=params.get("noise_intensity"),estimator_parameters=params,
                    observed_sha256=observed_sha,clean_velocity_sha256=clean_sha,no_additive_noise=True,no_rms_normalization=True,
                    clean_input_sha256=source["sha256"],helper_driver_sha256=provenance["helper_driver_sha256"],
                    script_sha256=provenance["script_sha256"])
                with ledger.open("a") as handle:
                    handle.write(json.dumps(json_safe(record),allow_nan=False)+"\n");handle.flush();os.fsync(handle.fileno())
                records.append(record)
                print(f'{len(records)}/84 {flow} {method} {contract} {record["status"]} converged={record.get("diagnostics",{}).get("converged")} fit={record.get("fit_profile",{}).get("wall_seconds",0):.3f}s',flush=True)
                if len(records)%12==0:summarize(records,out,config,oracles)
        if not any(o["flow"]==flow for o in oracles):
            singular=np.linalg.svd(observed.reshape(len(observed),-1).T,compute_uv=False,full_matrices=False)
            oracles.append(dict(flow=flow,rank=8,velocity_rank8_frobenius_lower_bound=float(np.sqrt(np.sum(singular[8:]**2)/np.sum(singular**2))),
                singular_values=singular.tolist(),clean_velocity_sha256=clean_sha,oracle_reference_used=True,estimator_input=False,
                qualification="Eckart–Young Frobenius bound on clean velocity matrix rank8, not a vorticity optimum or rank selection."))
            primary.write_json(oracle_path,oracles)
        del data,observed
    assert sources=={str(p.relative_to(ROOT)):primary.sha256(p) for p in sorted((ROOT/"python/src/fluid_denoise/research").glob("*.py"))}
    assert primary.sha256(primary.__file__)==provenance["helper_driver_sha256"]
    rows,complete=summarize(records,out,config,oracles);draw(rows,out,config);report(out,rows,oracles)
    provenance["script_fit_epochs"]=sorted({r["script_sha256"] for r in records})
    provenance["completion"]=complete
    provenance["artifact_sha256"]={str(p.relative_to(out)):primary.sha256(p) for p in sorted(out.rglob("*"))
                                    if p.is_file() and p.name!="provenance.json"}
    primary.write_json(out/"provenance.json",provenance)
    print(json.dumps(complete,indent=2),flush=True)


if __name__=="__main__":main()
