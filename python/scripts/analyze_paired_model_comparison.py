#!/usr/bin/env python3
"""Descriptive, paired model comparison from the immutable campaign ledger.

Quality has one observation per case/noise/seed/method; three resource repeats
describe execution variability and are never counted as independent data.
No estimator is fitted and no clean array is used to choose a hyperparameter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
METHODS = ["identity", "identity_helmholtz", "pod", "pod_helmholtz", "rpca",
           "stable_rpca", "stable_rpca_helmholtz", "dmd", "tls_dmd",
           "tls_dmd_helmholtz", "tls_dmd_helmholtz_dealias", "optimized_dmd",
           "fb_dmd_fallback_tls"]
ML = {"identity": "Identidad", "identity_helmholtz": "Identidad + H",
      "pod": "POD", "pod_helmholtz": "POD + H", "rpca": "RPCA (PCP)",
      "stable_rpca": "SPCP", "stable_rpca_helmholtz": "SPCP + H",
      "dmd": "DMD", "tls_dmd": "TLS-DMD", "tls_dmd_helmholtz": "TLS-DMD + H",
      "tls_dmd_helmholtz_dealias": "TLS-DMD + H + 2/3",
      "optimized_dmd": "Optimized DMD", "fb_dmd_fallback_tls": "fbDMD → TLS si falla"}
NOISES = ["gaussian", "sparse_impulsive", "mixed", "spatial_correlated",
          "temporal_correlated", "column_outliers", "student_t",
          "heteroscedastic", "sensor_drift"]
NL = {"gaussian": "Gaussiano", "sparse_impulsive": "Impulsivo disperso",
      "mixed": "Mixto", "spatial_correlated": "Correlación espacial",
      "temporal_correlated": "Correlación temporal", "column_outliers": "Columnas corruptas",
      "student_t": "Student-t", "heteroscedastic": "Heterocedástico", "sensor_drift": "Deriva"}
CASES = ["taylor_green_ns", "interacting_vortices_ns", "decaying_turbulence_ns",
         "oldroyd_b_2d", "fene_p_2d", "hb_thinning_2d", "bingham_2d", "hb_thickening_2d"]
CL = {"taylor_green_ns": "Taylor–Green", "interacting_vortices_ns": "Vórtices NS",
      "decaying_turbulence_ns": "Multiescala NS", "oldroyd_b_2d": "Oldroyd-B",
      "fene_p_2d": "FENE-P", "hb_thinning_2d": "HB n=0.6", "bingham_2d": "Bingham",
      "hb_thickening_2d": "HB n=1.4"}
GATES = {"velocity_relative_l2": .15, "vorticity_relative_l2": .25,
         "extra_divergence_normalized": .10, "kinetic_energy_relative_l2": .15}
GL = {"velocity_relative_l2": "Velocidad ≤15%", "vorticity_relative_l2": "Vorticidad ≤25%",
      "extra_divergence_normalized": "Divergencia ≤10%", "kinetic_energy_relative_l2": "Energía ≤15%"}
BLOCK = ["case", "noise", "seed"]
QUALITY_METRICS = list(GATES) + ["vorticity_correlation", "enstrophy_relative_l2"]
RESOURCES = ["wall_seconds", "cpu_seconds", "rss_peak_bytes", "python_traced_peak_bytes"]
PAIRS = [("identity", "identity_helmholtz"), ("pod", "pod_helmholtz"),
         ("stable_rpca", "stable_rpca_helmholtz"), ("tls_dmd", "tls_dmd_helmholtz"),
         ("tls_dmd_helmholtz", "tls_dmd_helmholtz_dealias")]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value) -> bool:
    return value is not None and np.isfinite(value)


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def json_records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", double_precision=15))


def save_table(out: Path, name: str, frame: pd.DataFrame) -> None:
    frame.to_csv(out / f"{name}.csv", index=False)
    write_json(out / f"{name}.json", json_records(frame))


def stats(values) -> dict:
    a = np.asarray(list(values), dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {k: None for k in ("median", "min", "max", "q25", "q75", "iqr")}
    q25, q75 = np.quantile(a, [.25, .75])
    return dict(median=float(np.median(a)), min=float(a.min()), max=float(a.max()),
                q25=float(q25), q75=float(q75), iqr=float(q75-q25))


def load_conditions(source: Path) -> tuple[pd.DataFrame, list[dict], dict]:
    ledger = [json.loads(x) for x in (source / "ledger.jsonl").read_text().splitlines()]
    provenance = json.loads((source / "provenance.json").read_text())
    assert provenance["config"]["quality_thresholds"] == GATES
    assert len(ledger) == 2808
    assert len({r["id"] for r in ledger}) == 2808
    rows = []
    for r in ledger:
        fits = r["fits"]
        assert len(fits) == 3 and all("metrics" in f and "profile" in f for f in fits)
        metrics = fits[0]["metrics"]
        row = dict(id=r["id"], case=r["case"], noise=r["noise"]["kind"], seed=r["seed"],
                   method=r["method"],base_method=r["base_method"],postprocess=r["postprocess"],
                   rank_requested=r["rank_requested"], resource_repeats=3,
                   converged=all(f["diagnostics"]["converged"] for f in fits),
                   fallback_used=any(f["diagnostics"].get("fallback_used", False) for f in fits),
                   fallback_reason=" | ".join(sorted({f["diagnostics"].get("primary_failure", {}).get("reason", "")
                                                       for f in fits if f["diagnostics"].get("fallback_used")})),
                   noise_relative_rms=r["noise"]["realized_relative_rms"])
        for k in QUALITY_METRICS:
            vals = [f["metrics"].get(k) for f in fits]
            row[k] = metrics.get(k)
            row[k+"_repeat_range"] = max(vals)-min(vals) if all(finite(v) for v in vals) else None
        for key in ["vorticity_equation_applicable", "extra_vorticity_equation_residual_normalized",
                    "reference_vorticity_equation_residual_rms", "vorticity_equation_residual_rms"]:
            row[key] = metrics.get(key)
        for key, maximum in GATES.items():
            row["pass_"+key] = all(finite(f["metrics"].get(key)) and f["metrics"][key] <= maximum for f in fits)
        row["physical_pass"] = all(row["pass_"+k] for k in GATES)
        assert row["physical_pass"] == r["quality_gate"]["passed"]
        row["admissible"] = row["physical_pass"] and row["converged"]
        for resource in RESOURCES:
            for stat, value in stats(f["profile"][resource] for f in fits).items():
                row[resource+"_"+stat] = value
        for resource in ["flops_proxy", "svd_calls", "lstsq_calls", "fft_calls", "iterations"]:
            values = [f["diagnostics"].get(resource) for f in fits]
            row[resource] = float(np.median(values)) if all(finite(v) for v in values) else None
        row["fallback_method"] = ",".join(sorted({f["diagnostics"].get("fallback_method", "") for f in fits
                                                  if f["diagnostics"].get("fallback_used")}))
        rows.append(row)
    frame = pd.DataFrame(rows)
    assert set(frame.method) == set(METHODS) and set(frame.case) == set(CASES) and set(frame.noise) == set(NOISES)
    counts = frame.groupby(BLOCK).method.nunique()
    assert len(counts) == 216 and counts.eq(13).all()
    assert frame.seed.nunique() == 3 and set(frame.seed) == {7, 23, 61}
    assert frame.rank_requested.eq(4).all()
    assert (~frame.converged).sum() == 6
    return frame, ledger, provenance


def paired_against_identity(frame: pd.DataFrame) -> pd.DataFrame:
    base = frame[frame.method.eq("identity")].set_index(BLOCK)
    rows = []
    for _, r in frame.iterrows():
        b = base.loc[tuple(r[k] for k in BLOCK)]
        item = {k: r[k] for k in BLOCK + ["method", "id", "admissible", "converged"]}
        for metric in GATES:
            item[metric] = r[metric]
            item[metric+"_identity"] = b[metric]
            # A spatially uniform sensor drift has essentially zero curl/divergence;
            # dividing by roundoff would manufacture enormous, meaningless gains.
            item[metric+"_gain_relative"] = 1-r[metric]/b[metric] if b[metric] > 1e-12 else None
            item[metric+"_delta"] = r[metric]-b[metric]
        item["improves_velocity"] = r.velocity_relative_l2 < b.velocity_relative_l2 - 1e-12
        item["improves_vorticity"] = r.vorticity_relative_l2 < b.vorticity_relative_l2 - 1e-12
        item["improves_both"] = item["improves_velocity"] and item["improves_vorticity"]
        rows.append(item)
    return pd.DataFrame(rows)


def prior_controlled_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    """Use an identity control with the SAME physical prior when one exists."""
    indexed=frame.set_index(BLOCK+["method"])
    rows=[]
    for _,r in frame.iterrows():
        control="identity" if r.postprocess=="none" else "identity_helmholtz" if r.postprocess=="helmholtz" else None
        item={k:r[k] for k in BLOCK+["id","method","postprocess"]}
        item.update(control_method=control,control_available=control is not None,
                    interpretation="algorithm increment under same prior" if control else "missing identity with H+2/3 in historical campaign")
        b=indexed.loc[tuple(r[k] for k in BLOCK)+(control,)] if control else None
        for metric in GATES:
            item[metric+"_delta"]=r[metric]-b[metric] if b is not None else None
            item[metric+"_gain_relative"]=1-r[metric]/b[metric] if b is not None and b[metric]>1e-12 else None
        rows.append(item)
    return pd.DataFrame(rows)


def grouped_summary(frame: pd.DataFrame, paired: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    merged = frame.merge(paired[["id"]+[k+"_gain_relative" for k in GATES]], on="id", validate="one_to_one")
    rows = []
    for keys, g in merged.groupby(by, sort=False):
        if not isinstance(keys, tuple): keys = (keys,)
        item = dict(zip(by, keys))
        item.update(conditions=len(g), physical_pass=int(g.physical_pass.sum()),
                    admissible=int(g.admissible.sum()), nonconverged=int((~g.converged).sum()),
                    fallback_conditions=int(g.fallback_used.sum()), acceptance_fraction=float(g.admissible.mean()))
        for metric in QUALITY_METRICS + [k+"_gain_relative" for k in GATES]:
            for stat, value in stats(g[metric]).items():
                item[metric+"_"+stat] = value
        for key in GATES:
            item["pass_"+key+"_count"] = int(g["pass_"+key].sum())
        for resource in RESOURCES:
            # Distribution of per-condition repeat medians, separate from within-repeat variation.
            for stat, value in stats(g[resource+"_median"]).items():
                item[resource+"_condition_median_"+stat] = value
            item[resource+"_within_repeat_iqr_median"] = float(g[resource+"_iqr"].median())
        rows.append(item)
    return pd.DataFrame(rows)


def ablations(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for before, after in PAIRS:
        a = frame[frame.method.eq(before)].set_index(BLOCK)
        b = frame[frame.method.eq(after)].set_index(BLOCK)
        for index, r in a.iterrows():
            s = b.loc[index]
            item = dict(zip(BLOCK, index), before=before, after=after,
                        before_admissible=bool(r.admissible), after_admissible=bool(s.admissible),
                        rescued=not r.admissible and bool(s.admissible), lost=bool(r.admissible) and not s.admissible)
            for metric in GATES:
                item[metric+"_before"] = r[metric]
                item[metric+"_after"] = s[metric]
                item[metric+"_delta"] = s[metric]-r[metric]
                item[metric+"_gain_relative"] = 1-s[metric]/r[metric] if r[metric] > 1e-12 else None
            item["vorticity_equal_1e-10"] = abs(s.vorticity_relative_l2-r.vorticity_relative_l2) <= 1e-10
            rows.append(item)
    paired = pd.DataFrame(rows)
    summary = []
    for (before, after), g in paired.groupby(["before", "after"], sort=False):
        item = dict(before=before, after=after, conditions=len(g), before_admissible=int(g.before_admissible.sum()),
                    after_admissible=int(g.after_admissible.sum()), rescued=int(g.rescued.sum()), lost=int(g.lost.sum()),
                    vorticity_equal_count=int(g["vorticity_equal_1e-10"].sum()))
        for metric in GATES:
            for stat, value in stats(g[metric+"_delta"]).items():
                item[metric+"_delta_"+stat] = value
        summary.append(item)
    return paired, pd.DataFrame(summary)


def pareto_and_rank(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = frame.copy()
    def tolerant_ranks(series: pd.Series) -> pd.Series:
        # H preserves curl: machine-roundoff differences must not create a hierarchy.
        ordered=series.sort_values()
        ranks=pd.Series(index=series.index,dtype=float)
        groups=[]
        for index,value in ordered.items():
            if not groups or value-groups[-1][0] > 1e-10:
                groups.append([value,[(index,value)]])
            else:
                groups[-1][1].append((index,value))
        position=1
        for _,items in groups:
            rank=position+(len(items)-1)/2
            for index,_ in items:ranks.loc[index]=rank
            position+=len(items)
        return ranks
    for metric in ["velocity_relative_l2", "vorticity_relative_l2"]:
        frame[metric+"_descriptive_rank"] = frame.groupby(BLOCK)[metric].transform(tolerant_ranks)
    frame["feasible_pareto_3d"] = False
    frame["feasible_pareto_4d"] = False
    frame["admissible_alternatives_in_block"] = frame.groupby(BLOCK).admissible.transform("sum")
    rows = []
    for key, block in frame.groupby(BLOCK, sort=False):
        feasible = block[block.admissible]
        for flag, cols in [("feasible_pareto_3d", ["velocity_relative_l2", "vorticity_relative_l2", "cpu_seconds_median"]),
                           ("feasible_pareto_4d", ["velocity_relative_l2", "vorticity_relative_l2", "cpu_seconds_median", "rss_peak_bytes_median"])]:
            values = feasible[cols].to_numpy()
            for i, (idx, _) in enumerate(feasible.iterrows()):
                scale = np.maximum(np.abs(values[i]), 1)
                eps = 1e-12*scale
                dominated = any(np.all(other <= values[i]+eps) and np.any(other < values[i]-eps)
                                for j, other in enumerate(values) if j != i)
                frame.loc[idx, flag] = not dominated
        rows.append(dict(zip(BLOCK, key), admissible_methods=len(feasible),
                         no_admissible_method=len(feasible)==0,
                         feasible_methods=";".join(feasible.method),
                         feasible_pareto_3d_methods=";".join(frame.loc[block.index][lambda x: x.feasible_pareto_3d].method),
                         feasible_pareto_4d_methods=";".join(frame.loc[block.index][lambda x: x.feasible_pareto_4d].method)))
    macro = []
    for method, g in frame.groupby("method", sort=False):
        feasible = g[g.admissible]
        item = dict(method=method, conditions=len(g), admissible_conditions=len(feasible),
                    acceptance_fraction=float(g.admissible.mean()),
                    feasible_pareto_3d_count=int(g.feasible_pareto_3d.sum()), feasible_pareto_4d_count=int(g.feasible_pareto_4d.sum()))
        for metric in ["velocity_relative_l2", "vorticity_relative_l2"]:
            cell = g.groupby(["case", "noise"])[metric+"_descriptive_rank"].mean()
            assert len(cell)==72
            item[metric+"_macro_descriptive_rank"] = float(cell.mean())
            item[metric+"_admissible_median"] = float(feasible[metric].median()) if len(feasible) else None
        macro.append(item)
    return frame, pd.DataFrame(rows), pd.DataFrame(macro)


def heatmap(ax, values, labels_x, labels_y, title, *, vmin=0, vmax=100, cmap="viridis", fmt=".0f"):
    im = ax.imshow(values, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(range(len(labels_x)), labels_x, rotation=35, ha="right",fontsize=13)
    ax.set_yticks(range(len(labels_y)), labels_y,fontsize=13)
    ax.set_title(title, pad=16, fontsize=16)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if not np.isfinite(value):
                ax.text(j,i,"—",ha="center",va="center",fontsize=12,color="black")
                continue
            fraction = (value-vmin)/(vmax-vmin) if vmax != vmin else 0
            ax.text(j, i, format(value, fmt), ha="center", va="center", fontsize=12,
                    color="white" if fraction < .48 or cmap == "RdBu" and fraction > .7 else "black")
    return im


def render_figures(out: Path, frame: pd.DataFrame, paired: pd.DataFrame, ablation: pd.DataFrame,
                   macro: pd.DataFrame, groups: dict[str, pd.DataFrame]) -> list[str]:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11,
                         "pdf.fonttype": 42, "savefig.dpi": 170})
    figures = out / "figures"; figures.mkdir(exist_ok=True)
    names = []
    def save(fig, name, footnote=None, canonical=True):
        if footnote:
            fig.set_layout_engine("constrained",rect=(0,.08,1,.92))
            fig.text(.5, .01, textwrap.fill(footnote,width=int(fig.get_figwidth()*11)), ha="center", va="bottom",fontsize=10)
        fig.savefig(figures / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(figures / f"{name}.png", bbox_inches="tight")
        plt.close(fig)
        if canonical:names.append(name)
    for by, labels, cats, name in [("noise", NL, NOISES, "acceptance_by_noise"),
                                   ("case", CL, CASES, "acceptance_by_fluid")]:
        table = groups[by].pivot(index="method", columns=by, values="acceptance_fraction").reindex(index=METHODS, columns=cats)
        fig, ax = plt.subplots(figsize=(14, 9), layout="constrained")
        im = heatmap(ax, 100*table.to_numpy(), [labels[x] for x in cats], [ML[x] for x in METHODS],
                     "Aceptación física conjunta y convergencia (%)",fmt=".1f")
        fig.colorbar(im, ax=ax, label="Porcentaje de condiciones admisibles", shrink=.8)
        save(fig, name, "Cada celda: " + ("8 casos × 3 semillas = 24" if by=="noise" else "9 ruidos × 3 semillas = 27") +
             " condiciones. Las tres repeticiones de coste no incrementan esta muestra.")
    fig, axes = plt.subplots(1, 2, figsize=(22, 10), layout="constrained")
    for ax, metric, title in zip(axes, list(GATES)[:2], ["Ganancia en velocidad", "Ganancia en vorticidad"]):
        table = paired.groupby(["method", "noise"])[metric+"_gain_relative"].median().unstack().reindex(index=METHODS, columns=NOISES)
        # Saturation deliberately disclosed: raw values are preserved in CSV/JSON.
        im=heatmap(ax, 100*table.to_numpy(), [NL[x] for x in NOISES], [ML[x] for x in METHODS],
                   title+" frente a identidad (%)", vmin=-100, vmax=100, cmap="RdBu")
        fig.colorbar(im, ax=ax, shrink=.75, label="100·(1 − error del modelo/error de identidad)")
    save(fig, "paired_gains_velocity_vorticity", "Ganancia de la cadena completa, incluido posprocesamiento; no aísla aporte del algoritmo. Mediana de 24 cocientes pareados; negativo = empeora. Color saturado a ±100%; cifras sin truncar. — = baseline ≤10⁻¹² (deriva uniforme sin curl).")
    for metric,title,name in [("velocity_relative_l2","Velocidad","paired_gain_velocity"),
                              ("vorticity_relative_l2","Vorticidad","paired_gain_vorticity")]:
        fig,ax=plt.subplots(figsize=(14,9),layout="constrained")
        table=paired.groupby(["method","noise"])[metric+"_gain_relative"].median().unstack().reindex(index=METHODS,columns=NOISES)
        im=heatmap(ax,100*table.to_numpy(),[NL[n] for n in NOISES],[ML[m] for m in METHODS],
                   f"{title}: ganancia pareada frente a identidad (%)",vmin=-100,vmax=100,cmap="RdBu")
        fig.colorbar(im,ax=ax,label="100·(1 − error del modelo/error de identidad)",shrink=.75)
        save(fig,name,"Ganancia de la cadena completa, incluido posprocesamiento; no aísla aporte algorítmico. Mediana de 24 cocientes; negativo = empeora. — = baseline ≤10⁻¹². Color saturado a ±100%; cifras sin truncar.",canonical=False)
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={"width_ratios": [1.5, 1]}, layout="constrained")
    pass_rates=np.array([[100*frame[frame.method.eq(m)]["pass_"+k].mean() for k in GATES] for m in METHODS])
    im=heatmap(axes[0], pass_rates, list(GL.values()), [ML[x] for x in METHODS], "Cada puerta por separado (%)",fmt=".1f")
    fig.colorbar(im, ax=axes[0], shrink=.7)
    y=np.arange(len(METHODS)); totals=[100*frame[frame.method.eq(m)].admissible.mean() for m in METHODS]
    axes[1].barh(y, totals, color="#167d8d"); axes[1].set_yticks(y, [ML[m] for m in METHODS]); axes[1].invert_yaxis()
    axes[1].set_xlim(0, 100); axes[1].set_xlabel("Aceptación conjunta (%)"); axes[1].set_title("Todas las puertas + convergencia")
    for i, value in enumerate(totals): axes[1].text(value+1, i, f"{value:.1f}", va="center", fontsize=10)
    save(fig, "physical_gates_and_joint_acceptance", "Puertas marginales no se suman: deben cumplirse simultáneamente. 216 condiciones por método; masa fija no es una puerta informativa.")
    fig, axes = plt.subplots(1, 2, figsize=(18, 9), layout="constrained")
    colors=["#147d92", "#d16b25", "#7a4aa2"]
    for seed, color, offset in zip([7,23,61],colors,[-.2,0,.2]):
        g=frame[frame.seed.eq(seed)]
        values=[100*g[g.method.eq(m)].admissible.mean() for m in METHODS]
        axes[0].scatter(values, np.arange(len(METHODS))+offset, label=f"Semilla {seed}", color=color, s=45)
    axes[0].set_yticks(range(len(METHODS)), [ML[m] for m in METHODS]); axes[0].invert_yaxis()
    axes[0].set_xlim(-2,102); axes[0].set_xlabel("Aceptación (%) sobre 72 casos/ruidos por semilla")
    axes[0].set_title("Variación observada entre tres semillas"); axes[0].legend(fontsize=10)
    for j, metric in enumerate(["velocity_relative_l2", "vorticity_relative_l2"]):
        ranks=macro.set_index("method").reindex(METHODS)[metric+"_macro_descriptive_rank"]
        axes[1].scatter(ranks, np.arange(len(METHODS))+(j-.5)*.2, s=60,
                        color=["#147d92","#a14b33"][j], label=["Velocidad", "Vorticidad"][j])
    axes[1].set_yticks(range(len(METHODS)),[ML[m] for m in METHODS]); axes[1].invert_yaxis()
    axes[1].set_xlim(.5,13.5); axes[1].set_xlabel("Rango descriptivo medio (1 = menor error)")
    axes[1].set_title("Peso igual: 8 casos × 9 ruidos"); axes[1].legend(fontsize=10)
    save(fig,"seed_stability_and_macro_ranks", "Rangos incluyen todas las salidas, incluso seis no convergentes; describen error, no elegibilidad. Tres semillas no sustentan intervalos poblacionales.")
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), layout="constrained",sharey=True)
    for i, metric in enumerate(["velocity_relative_l2", "vorticity_relative_l2", "extra_divergence_normalized"]):
        data=[100*ablation[(ablation.before.eq(a))&(ablation.after.eq(b))][metric+"_delta"].to_numpy() for a,b in PAIRS]
        axes[i].boxplot(data, orientation="horizontal", showfliers=True,
                        flierprops={"markersize":2,"alpha":.25})
        axes[i].axvline(0,color="black",linewidth=1)
        if i==0:
            axes[i].set_yticks(range(1,6),["Identidad → +H","POD → +H","SPCP → +H","TLS-DMD → +H","TLS-DMD + H → +2/3"], fontsize=11)
        else:
            axes[i].tick_params(axis="y",labelleft=False)
        axes[i].set_xlabel("Δ error (puntos porcentuales); negativo = mejora")
        axes[i].set_title(["Velocidad", "Vorticidad", "Divergencia"][i])
        if i==1:
            axes[i].set_xscale("symlog",linthresh=.01)
            axes[i].set_xticks([-100,-10,-1,-.1,0,.1,1],["−100","−10","−1","−0.1","0","0.1","1"])
    # boxplot sets numbered labels on its own; restore labels after every shared-y panel exists.
    axes[0].set_yticks(range(1,6),["Identidad → +H","POD → +H","SPCP → +H","TLS-DMD → +H","TLS-DMD + H → +2/3"],fontsize=11)
    save(fig,"paired_physical_ablations", "216 diferencias por pareja. H conserva la vorticidad en el dominio periódico; 2/3 modifica el espectro. Eje de vorticidad simétrico logarítmico cerca de cero.")
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), layout="constrained")
    label_offsets={"dmd":(-18,9),"tls_dmd":(7,-13),"fb_dmd_fallback_tls":(-18,-13),"pod_helmholtz":(7,10)}
    for number,m in enumerate(METHODS,1):
        g=frame[frame.method.eq(m)]; cpu=stats(g.cpu_seconds_median)
        accept=100*g.admissible.mean(); axes[0].scatter(cpu["median"],accept,s=70)
        axes[0].errorbar(cpu["median"],accept,xerr=[[cpu["median"]-cpu["q25"]],[cpu["q75"]-cpu["median"]]],
                         color="gray",alpha=.5, capsize=3)
        axes[0].annotate(f"{number:02d}",(cpu["median"],accept),xytext=label_offsets.get(m,(4,6)),
                         textcoords="offset points",fontsize=11)
    axes[0].set_xscale("log");axes[0].set_ylim(-5,78)
    axes[0].set_xlabel("Mediana de CPU por condición (s; barras: Q25–Q75 entre condiciones)")
    axes[0].set_ylabel("Condiciones admisibles (%)");axes[0].set_title("Calidad exigida y recursos")
    counts=macro.set_index("method").reindex(METHODS).feasible_pareto_3d_count.to_numpy()
    axes[1].barh(range(len(METHODS)), counts, color="#167d8d");axes[1].invert_yaxis()
    axes[1].set_yticks(range(len(METHODS)),[f"{i:02d}  {ML[m]}" for i,m in enumerate(METHODS,1)]);axes[1].set_xlabel("Bloques donde es Pareto admisible (máximo 216)")
    axes[1].set_title("Pareto por observación: eᵤ, eω y CPU")
    save(fig,"resources_and_feasible_pareto", "Cada coste: mediana de tres procesos nuevos. Las barras mezclan problemas distintos y no son incertidumbre del cronómetro. Pareto nunca admite salidas que incumplen puertas.")
    fig,ax=plt.subplots(figsize=(12,8),layout="constrained")
    for i,m in enumerate(METHODS):
        g=frame[frame.method.eq(m)];cost=stats(g.cpu_seconds_median)
        ax.errorbar(cost["median"],i,xerr=[[cost["median"]-cost["q25"]],[cost["q75"]-cost["median"]]],
                    marker="o",color="#167d8d",capsize=4,markersize=7)
    ax.set_yticks(range(len(METHODS)),[ML[m] for m in METHODS],fontsize=13);ax.invert_yaxis();ax.set_xscale("log")
    ax.set_xlabel("Tiempo de CPU por condición (s)",fontsize=13);ax.set_title("Recursos de los modelos: mediana y Q25–Q75 entre condiciones",fontsize=15)
    save(fig,"resources_cpu","Cada valor agrega medianas de tres ejecuciones nuevas por condición. Barras entre problemas distintos; no son intervalos de confianza ni incertidumbre del cronómetro.",canonical=False)
    fig,ax=plt.subplots(figsize=(12,8),layout="constrained")
    ax.barh(range(len(METHODS)),counts,color="#167d8d");ax.invert_yaxis()
    ax.set_yticks(range(len(METHODS)),[ML[m] for m in METHODS],fontsize=13)
    for i,value in enumerate(counts):ax.text(value+1,i,str(int(value)),va="center",fontsize=12)
    ax.set_xlim(0,160);ax.set_xlabel("Bloques donde integra la frontera Pareto admisible (máximo 216)",fontsize=12)
    ax.set_title("Pareto por observación: velocidad, vorticidad y CPU",fontsize=15)
    save(fig,"feasible_pareto_coverage","Solo candidatos convergentes que pasan las cuatro puertas. Fronteras independientes por caso, ruido y semilla; cobertura observada, no ganador universal.",canonical=False)
    # Robust coverage: a method must be admissible for every one of the three seeds in a case/noise cell.
    stable=frame.groupby(["case","noise","method"]).admissible.all().groupby(["case","noise"]).sum().unstack().reindex(index=CASES,columns=NOISES)
    fig, ax=plt.subplots(figsize=(15,8),layout="constrained")
    im=heatmap(ax,stable.to_numpy(),[NL[n] for n in NOISES],[CL[c] for c in CASES],
               "Alternativas admisibles en las tres semillas de cada caso y ruido",vmin=0,vmax=13,fmt=".0f")
    fig.colorbar(im,ax=ax,label="Número de métodos (0 = ninguna alternativa verificada)")
    save(fig,"admissible_coverage_all_seeds", "Mismo caso, ruido e intensidad; tres realizaciones. Cero significa que este banco no ofrece un modelo aceptable en todas las semillas; no se elige un ganador forzado.")
    fig,ax=plt.subplots(figsize=(14,8),layout="constrained")
    line_colors=plt.get_cmap("tab20")(np.linspace(0,1,len(METHODS)))
    for method,color in zip(METHODS,line_colors):
        g=frame[frame.method.eq(method)]
        other=g.converged & g.pass_velocity_relative_l2 & g.pass_extra_divergence_normalized & g.pass_kinetic_energy_relative_l2
        rates=[100*(other & g.vorticity_relative_l2.le(t)).mean() for t in [.05,.10,.25,.50]]
        ax.plot([5,10,25,50],rates,marker="o",label=ML[method],linewidth=2,color=color,
                alpha=.9 if method not in ["identity","identity_helmholtz"] else .5)
    ax.axvline(25,color="black",linestyle="--",alpha=.6,label="Puerta nominal: 25%")
    ax.set_xticks([5,10,25,50]);ax.set_xlabel("Límite de error relativo de vorticidad (%)")
    ax.set_ylabel("Condiciones admisibles (%)");ax.set_title("Sensibilidad al criterio de fidelidad: las otras puertas se mantienen")
    ax.set_ylim(-2,102);ax.legend(bbox_to_anchor=(1.02,1),loc="upper left",fontsize=10)
    save(fig,"tolerance_sensitivity", "Reevaluación descriptiva sobre las mismas salidas: no cambia ajustes ni elige un umbral óptimo. No equivale a cuatro campañas independientes.")
    tolerance_matrix=[]
    for method in METHODS:
        g=frame[frame.method.eq(method)]
        other=g.converged & g.pass_velocity_relative_l2 & g.pass_extra_divergence_normalized & g.pass_kinetic_energy_relative_l2
        tolerance_matrix.append([int((other & g.vorticity_relative_l2.le(t)).sum()) for t in [.05,.10,.25,.50]])
    fig,ax=plt.subplots(figsize=(12,8),layout="constrained")
    im=heatmap(ax,np.asarray(tolerance_matrix),["eω ≤5%","eω ≤10%","eω ≤25%","eω ≤50%"],
               [ML[m] for m in METHODS],"Aceptaciones al cambiar la fidelidad de vorticidad",vmin=0,vmax=216)
    ax.tick_params(axis="x",labelrotation=0)
    for label in ax.get_xticklabels():label.set_ha("center");label.set_fontsize(15)
    for label in ax.get_yticklabels():label.set_fontsize(15)
    for label in ax.texts:label.set_fontsize(14)
    fig.colorbar(im,ax=ax,label="Número de condiciones admisibles de 216",shrink=.75)
    save(fig,"tolerance_sensitivity_matrix","Las otras tres puertas y la convergencia permanecen fijas. 25% es la puerta nominal del banco; no una exactitud física universal. Mismas salidas, sin reajuste.",canonical=False)
    assert len(names)==9
    return names


def markdown_table(frame: pd.DataFrame, columns: list[str], headers: list[str]) -> str:
    lines=["| "+" | ".join(headers)+" |", "| "+" | ".join(["---"]*len(columns))+" |"]
    for _,r in frame.iterrows():
        values=[]
        for col in columns:
            v=r[col]
            if col in {"method","before","after"}:v=ML[v]
            elif isinstance(v,(float,np.floating)):v=f"{v:.3g}"
            values.append(str(v))
        lines.append("| "+" | ".join(values)+" |")
    return "\n".join(lines)


def write_report(out: Path, frame: pd.DataFrame, macro: pd.DataFrame, groups: dict[str,pd.DataFrame],
                 ablation_summary: pd.DataFrame, blocks: pd.DataFrame, figures: list[str], stable: pd.DataFrame) -> None:
    methods=groups["method"].set_index("method").reindex(METHODS).reset_index()
    no_eligible=int(blocks.no_admissible_method.sum())
    eligible=int(frame.admissible.sum())
    fallback=int(frame.fallback_used.sum())
    disagree=frame.velocity_relative_l2.le(.15)&frame.vorticity_relative_l2.gt(.25)
    table=methods[["method","admissible","conditions","velocity_relative_l2_median","vorticity_relative_l2_median","cpu_seconds_condition_median_median"]].copy()
    table["velocity_relative_l2_median"]*=100;table["vorticity_relative_l2_median"]*=100
    narrative=["# Comparación pareada de modelos y fidelidad de reconstrucción", "",
      "## Resultado central", "",
      f"El banco completo conserva 2 808 condiciones, 216 observaciones pareadas y 8 424 mediciones de recursos. "
      f"Hay **{eligible} salidas admisibles** al exigir simultáneamente cuatro puertas físicas y convergencia; "
      f"**{no_eligible}/216 observaciones no tienen ningún método admisible**. La ausencia de alternativa se publica, en lugar de imponer un ganador.", "",
      "TLS-DMD + H + 2/3 alcanza 142/216 aceptaciones; SPCP + H alcanza 72/216. Esta jerarquía corresponde a los casos, tamaños, rango 4, intensidad RMS 20% y ventanas de esta campaña. "
      "No demuestra superioridad universal, resolución DNS de las referencias ni pronóstico fuera de muestra.", "",
      "Este banco usa mallas 24² (Taylor–Green), 48² (vórtices NS), 64² (multiescala NS) y 32² (cinco casos no newtonianos). "
      "Es una comparación exploratoria sobre referencias propias de esas mallas; el refinamiento de las campañas de mayor resolución "
      "no certifica retroactivamente estas ocho trayectorias. En ruido iid, la contaminación de gradientes cambia con N incluso cuando el RMS de velocidad es igual.", "",
      "## Qué se compara y cómo leer los números", "",
      "- Unidad de calidad: un caso × tipo de ruido × semilla × método. Los 31–41 snapshots de una trayectoria no se cuentan como observaciones independientes.",
      "- Pareamiento: todos los métodos reciben la misma realización de ruido dentro de cada caso/ruido/semilla. Semillas 7, 23 y 61; nueve ruidos normalizados al mismo RMS relativo realizado de 20%.",
      "- Recursos: tres procesos nuevos por condición; mediana, IQR y rango del tiempo de CPU, reloj, RSS y memoria Python. Tres repeticiones de coste no son tres realizaciones nuevas de ruido.",
      "- Puertas predeclaradas: error de velocidad ≤15%, vorticidad ≤25%, divergencia extra normalizada ≤10% y energía ≤15%. Densidad fija y masa exacta son supuestos del banco; no prueban conservación dinámica.",
      "- H designa proyección de Helmholtz periódica; 2/3 es una restricción espectral asociada a la malla. El filtro 2/3 del reconstructor no debe confundirse con eliminar aliasing de un producto no lineal del solver.",
      "- Errores de velocidad y vorticidad se presentan separados. Los rangos macro dan peso igual a cada uno de los 72 pares caso/ruido y luego a sus tres semillas. Los costes heterogéneos no se agregan como una carga universal.",
      "- Los rangos empatan errores que difieren ≤10⁻¹⁰ dentro del bloque; las diferencias de redondeo entre un modelo y su proyección H no crean una jerarquía artificial de vorticidad.",
      "- No se calculan intervalos de confianza poblacionales, significancias ni bootstrap con tres semillas. Mediana, rango e IQR describen exclusivamente estas realizaciones.", "",
      "## Comparación completa", "",
      markdown_table(table,list(table.columns),["Método","Admisibles","Condiciones","Mediana eᵤ (%)","Mediana eω (%)","CPU mediana (s)"]),"",
      "Las medianas de esta tabla incluyen todas las salidas, incluso las seis no convergentes identificadas abajo. "
      "Una mediana baja no implica que el método pase todas las puertas ni todas las realizaciones. Los CSV preservan la distribución y cada bloque.", "",
      "![Aceptación por ruido](figures/acceptance_by_noise.png)","",
      "![Aceptación por referencia física](figures/acceptance_by_fluid.png)","",
      "## H1: la jerarquía depende del ruido", "",
      "El ruido disperso favorece la separación PCP; las contaminaciones densas, dependientes o con deriva requieren comprobar el objetivo por separado. "
      "`method_noise_summary.csv` muestra 24 condiciones por celda y `paired_vs_identity.csv` conserva cada ganancia pareada, definida como 1−e_modelo/e_identidad. "
      "Una ganancia negativa se conserva: normalizar a igual RMS no iguala estructura espectral ni identificabilidad.", "",
      "Las ganancias frente a identidad sin procesar atribuyen el efecto de la cadena completa (modelo + posprocesamiento). "
      "No aíslan el aporte algorítmico de DMD o RPCA. `paired_prior_control.csv` usa identidad con el mismo prior para "
      "las variantes sin H y con H. El banco histórico carece de identidad+H+2/3: sus 216 comparaciones controladas "
      "se marcan no disponibles, sin inventar ese experimento. La campaña nueva debe aportar ese control antes de atribuir a TLS "
      "una ventaja que podría explicar el filtrado.", "",
      "La deriva espacialmente uniforme modifica velocidad y mantiene curl/divergencia a redondeo; dividir por su error de vorticidad ≤10⁻¹² carece de significado. "
      "Esos cocientes se marcan indefinidos (—), y se conserva la diferencia absoluta pareada. No se reportan ganancias gigantes originadas por dividir por ruido numérico.", "",
      "![Ganancias pareadas en dos objetivos](figures/paired_gains_velocity_vorticity.png)","",
      "## H2: mejorar velocidad no garantiza preservar gradientes o continuidad", "",
      f"En **{int(disagree.sum())}/2 808 salidas**, la puerta de velocidad pasa y la puerta de vorticidad falla. "
      "Estos son eventos descriptivos sobre una malla común, no una probabilidad para otros flujos. `gate_events.csv` informa fallos marginales, únicos y conjuntos sin sumarlos como si fueran disjuntos.","",
      "![Puertas individuales y conjuntas](figures/physical_gates_and_joint_acceptance.png)","",
      "### Sensibilidad a la fidelidad exigida", "",
      "La puerta eω≤25% es un requisito exploratorio de esta campaña, no una ley física. `tolerance_sensitivity.csv` reevalúa "
      "eω≤5%, 10%, 25% y 50%, manteniendo las puertas nominales de velocidad, divergencia, energía y convergencia. "
      "Se muestran todos los valores del pequeño grid sin elegir el que favorece a un algoritmo; son las mismas salidas, no cuatro ensayos independientes.", "",
      "La conclusión cambia al exigir mayor fidelidad: con eω≤5%, RPCA acepta 9/216 y TLS-DMD+H+2/3 solo 1/216; "
      "con eω≤10% son 23/216 y 26/216; con la puerta nominal 25%, 42/216 y 142/216. "
      "Por eso la cobertura al 25% no basta para afirmar superioridad bajo una exigencia del 5%.", "",
      "![Sensibilidad al requisito de vorticidad](figures/tolerance_sensitivity.png)", "",
      "La aceptación no impone una puerta sobre el residual PDE. `admissible_pde_audit.csv` conserva los "
      "residuales extra de la ecuación de vorticidad para los tres casos NS donde se aplican. Un residual normalizado "
      "mayor que uno significa perturbación comparable o mayor que la escala de evolución definida por la referencia, "
      "pero no es una tolerancia predeclarada de exactitud. Este operador usa snapshots y derivadas temporales de segundo orden. "
      "No es una evaluación del residual de momento completo ni de la ecuación constitutiva de los otros cinco casos. "
      "Cumplir las cuatro puertas no certifica Navier–Stokes ni garantiza dinámica correcta.", "",
      "De las 150 salidas NS que pasan las cuatro puertas y convergen, 147 presentan residual extra de la ecuación "
      "de vorticidad normalizado mayor que uno. El denominador representa una escala de evolución física; "
      "esta alerta exige revisar la dinámica y la sensibilidad temporal antes de presentar esas reconstrucciones como físicamente fieles.", "",
      "## Ablaciones: cuánto aportan H y 2/3", "",
      "Se compara cada posprocesamiento con su algoritmo idéntico sin ese paso, manteniendo observación, rango y semilla. "
      "H elimina la componente de gradiente pero conserva curl sobre el dominio periódico; por eso una mejora en divergencia no debe atribuirse a recuperación de vorticidad. "
      "El CSV de ablaciones publica 216 diferencias por pareja, aceptaciones rescatadas y perdidas y pruebas de igualdad de eω a 10⁻¹⁰.","",
      markdown_table(ablation_summary,["before","after","before_admissible","after_admissible","rescued","lost","vorticity_equal_count"],
                     ["Antes","Después","Aceptadas antes","Después","Rescatadas","Perdidas","eω igual (216)"]),"",
      "![Ablaciones físicas pareadas](figures/paired_physical_ablations.png)","",
      "## H3: coste al nivel de fidelidad requerido", "",
      "Pareto se calcula dentro de cada observación y únicamente con candidatos que cumplen todas las puertas y convergen. "
      "Se publican dos fronteras: eᵤ/eω/CPU y eᵤ/eω/CPU/RSS. RSS es pico absoluto del proceso, incluye intérprete y datos; FLOPs es una aproximación de operaciones dominantes y excluye varios costes. "
      "Identidad es barata y falla todas las observaciones, por lo que no pertenece a ninguna frontera admisible. "
      "La fracción de bloques donde un modelo está en Pareto informa cobertura, no un premio universal para su coste medio.","",
      "![Calidad y recursos con Pareto admisible](figures/resources_and_feasible_pareto.png)","",
      "`conditions.csv` distingue IQR dentro de las tres repeticiones de cada condición de la dispersión entre problemas. "
      "El gráfico de CPU usa barras Q25–Q75 entre las medianas de 216 condiciones; no son error del reloj ni intervalos de confianza. "
      "`macro_descriptive_ranks.csv` conserva por separado orden por eᵤ, orden por eω y cobertura física; no combina objetivos con pesos arbitrarios.","",
      "## H4: dependencia del caso físico", "",
      "Las ocho referencias no constituyen un ensayo controlado de reología: difieren malla, horizonte, parámetros y estado inicial. "
      "Sus diferencias de aceptación son asociaciones con el caso, no efectos causales atribuibles solo a elasticidad, cedencia o índice n. "
      "La prueba de transferibilidad debe repetir un protocolo de observación común sobre referencias de fidelidad independiente, y evaluar tensores y esfuerzos además de velocidad. "
      "Este análisis compara reconstrucción de velocidad; no reconstruye C ni τ y no convierte eᵤ bajo en una prueba de precisión constitutiva.","",
      "![Cobertura en las tres semillas](figures/admissible_coverage_all_seeds.png)","",
      "## Variabilidad y fragilidad numérica", "",
      "![Variación por semilla y órdenes descriptivos](figures/seed_stability_and_macro_ranks.png)","",
      f"Las **seis no convergencias** de Optimized DMD se conservan en `nonconvergence.csv`. No se sustituyen sus errores por éxitos ni se eliminan del denominador. "
      f"La ruta fbDMD→TLS emplea respaldo en **{fallback}/216 condiciones**; `fallbacks.csv` conserva motivo y método efectivo. "
      "Solo las condiciones sin respaldo prueban una realización de fbDMD puro. Comparar la ruta completa con TLS incluye coste del intento fallido; no equivale a otro algoritmo independiente.","",
      "## Alcance y siguientes experimentos decisivos", "",
      "1. Separar calidad de referencia y de reconstrucción: antes de juzgar un denoiser, contrastar malla/tiempo, PDE, balances y variables constitutivas de su referencia.",
      "2. Repetir una selección fijada antes de mirar la referencia limpia, sobre mallas grandes, banda física y ruido comparable entre resoluciones; 2/3 cambia con N y no es una banda física constante.",
      "3. Usar nuevas trayectorias y semillas para evaluar selección sin reutilizar los 216 bloques como validación independiente; mantener un grupo sin método admisible.",
      "4. Separar reconstrucción de la ventana observada de predicción cronológica, y evaluar desplazamiento de fase, amplitud y espectro fuera de muestra.", "",
      "## Artefactos reproducibles", "",
      "Todos los CSV tienen compañero JSON. `provenance.json` guarda SHA-256 del ledger, resumen, manifiestos y configuración de origen, del script y de las tablas/figuras. "
      "`completion.json` verifica cardinalidades, las seis no convergencias y las tres repeticiones. Reproducción:","",
      "```bash\npython python/scripts/analyze_paired_model_comparison.py\n```", "",
      "La reproducibilidad de estas tablas no requiere volver a ejecutar 8 424 ajustes; usa el ledger completo publicado.", ""]
    # Human-readable exact noise/fluid contrasts, computed rather than hard-coded.
    sections=[]
    for axis, labels, cats in [("noise",NL,NOISES),("case",CL,CASES)]:
        sections.append("### " + ("Resultados por ruido" if axis=="noise" else "Resultados por caso") + "\n")
        for value in cats:
            group=groups[axis][groups[axis][axis].eq(value)]
            best=group.sort_values(["admissible","vorticity_relative_l2_median"],ascending=[False,True]).head(3)
            text="; ".join(f"{ML[r.method]}: {int(r.admissible)}/{int(r.conditions)}" for _,r in best.iterrows())
            sections.append(f"- **{labels[value]}:** mayor cobertura observada: {text}.")
        sections.append("")
    insert=narrative.index("## H2: mejorar velocidad no garantiza preservar gradientes o continuidad")
    narrative[insert:insert]=sections
    (out / "MODEL_COMPARISON.md").write_text("\n".join(narrative))


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=ROOT/"results/research/extended_physics_2026-10-14")
    parser.add_argument("--out",type=Path,default=ROOT/"results/research/paired_model_comparison_2026-10-14")
    args=parser.parse_args();source=args.source.resolve();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    frame,ledger,source_provenance=load_conditions(source)
    paired=paired_against_identity(frame)
    prior_control=prior_controlled_comparison(frame)
    frame,blocks,macro=pareto_and_rank(frame)
    ablation,ablation_summary=ablations(frame)
    groups={"method":grouped_summary(frame,paired,["method"]),
            "noise":grouped_summary(frame,paired,["method","noise"]),
            "case":grouped_summary(frame,paired,["method","case"]),
            "cell":grouped_summary(frame,paired,["method","case","noise"]),
            "seed":grouped_summary(frame,paired,["method","seed"])}
    stable=frame.groupby(["case","noise","method"],as_index=False).agg(all_three_seeds_admissible=("admissible","all"),
                admissible_seeds=("admissible","sum"),velocity_min=("velocity_relative_l2","min"),velocity_max=("velocity_relative_l2","max"),
                vorticity_min=("vorticity_relative_l2","min"),vorticity_max=("vorticity_relative_l2","max"))
    gates=[]
    for method,g in frame.groupby("method",sort=False):
        failed=pd.DataFrame({k:~g["pass_"+k] for k in GATES});n_fail=failed.sum(axis=1)
        for metric in GATES:
            gates.append(dict(method=method,gate=metric,conditions=len(g),passed=int((~failed[metric]).sum()),
                              failed=int(failed[metric].sum()),only_failed_gate=int((failed[metric]&n_fail.eq(1)).sum()),
                              velocity_pass_vorticity_fail=int((g.pass_velocity_relative_l2&~g.pass_vorticity_relative_l2).sum())))
    tolerance=[]
    for method,g in frame.groupby("method",sort=False):
        other=g.converged & g.pass_velocity_relative_l2 & g.pass_extra_divergence_normalized & g.pass_kinetic_energy_relative_l2
        for maximum in [.05,.10,.25,.50]:
            accepted=other & g.vorticity_relative_l2.le(maximum)
            tolerance.append(dict(method=method,vorticity_maximum=maximum,conditions=len(g),admissible=int(accepted.sum()),
                                  acceptance_fraction=float(accepted.mean()),other_gates_nominal=True))
    pde=frame[frame.admissible & frame.vorticity_equation_applicable.fillna(False).astype(bool)][
        BLOCK+["method","id","velocity_relative_l2","vorticity_relative_l2",
               "extra_vorticity_equation_residual_normalized","reference_vorticity_equation_residual_rms","vorticity_equation_residual_rms"]].copy()
    pde["extra_residual_greater_than_one"]=pde.extra_vorticity_equation_residual_normalized.gt(1)
    tables={"conditions":frame,"paired_vs_identity":paired,"paired_prior_control":prior_control,
            "paired_ablations":ablation,"ablation_summary":ablation_summary,
            "paired_block_feasibility":blocks,"macro_descriptive_ranks":macro,"seed_ranges_by_cell":stable,
            "gate_events":pd.DataFrame(gates),"nonconvergence":frame[~frame.converged],"fallbacks":frame[frame.fallback_used],
            "tolerance_sensitivity":pd.DataFrame(tolerance),"admissible_pde_audit":pde}
    tables.update({f"method_{axis}_summary":g for axis,g in groups.items()})
    tables["method_summary"]=groups["method"]
    for name,table in tables.items():save_table(out,name,table)
    figures=render_figures(out,frame,paired,ablation,macro,groups)
    write_report(out,frame,macro,groups,ablation_summary,blocks,figures,stable)
    completion=dict(complete=True,conditions=len(frame),paired_blocks=len(blocks),methods=13,cases=8,noises=9,seeds=[7,23,61],
                    resource_measurements=8424,resource_repeats_per_condition=3,quality_observations_per_condition=1,
                    physical_pass_conditions=int(frame.physical_pass.sum()),admissible_conditions=int(frame.admissible.sum()),
                    nonconverged_conditions=int((~frame.converged).sum()),fallback_conditions=int(frame.fallback_used.sum()),
                    blocks_without_admissible_method=int(blocks.no_admissible_method.sum()),figures=figures,
                    publication_companions=["paired_gain_velocity","paired_gain_vorticity","resources_cpu","feasible_pareto_coverage","tolerance_sensitivity_matrix"],
                    inferential_intervals=False,rank_requested=4)
    write_json(out/"completion.json",completion)
    artifacts={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name!="provenance.json"}
    inputs={str(p.relative_to(ROOT)):sha(p) for p in [source/"ledger.jsonl",source/"summary.json",source/"completion.json",
                                                    source/"provenance.json",source/"dataset_manifests.json"]}
    write_json(out/"provenance.json",dict(script=str(Path(__file__).relative_to(ROOT)),script_sha256=sha(Path(__file__)),
                 source_sha256=inputs,source_config_sha256=source_provenance["config_sha256"],artifact_sha256=artifacts,
                 methodology={"quality_unit":"case/noise/seed/method; snapshots and cost repeats are not additional quality samples",
                  "paired_blocks":"same observed trajectory across all 13 methods", "admissible":"all four physical gates plus convergence in all three executions",
                  "descriptive_ranks":"equal weighting of 72 case/noise cells, average rank over three seeds; all outputs retained; absolute tie tolerance 1e-10",
                  "pareto":"within one paired block; admissible only; 3D velocity/vorticity/CPU and 4D adding absolute RSS",
                  "inference":"no population CI, significance tests, seed bootstrap or causal rheological attribution"}))
    print(json.dumps(completion,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
