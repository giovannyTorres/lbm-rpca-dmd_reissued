#!/usr/bin/env python3
"""Export inspectable tables, figures, and thesis discussion from a completed run."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def tex_escape(value):
    return str(value).replace("_", r"\_\allowbreak ").replace("%", r"\%")


def percentage(value, latex=False):
    if value is None or not np.isfinite(float(value)):
        return "N/D"
    percent = 100 * float(value)
    if 0 < percent < .001:
        return r"$<0.001$" if latex else "<0.001"
    return f"{percent:.3f}"


def analyze(campaign, update_docs=False):
    directory = ROOT / "results" / "research" / campaign
    completion = json.loads((directory / "completion.json").read_text())
    if not completion.get("complete"):
        raise RuntimeError("Campaign incomplete; do not write final conclusions")
    frame = pd.read_csv(directory / "summary.csv")
    for key in ("all_repetitions_converged", "quality_pass", "pareto_member", "feasible_pareto_member"):
        if key not in frame:
            frame[key] = False
        frame[key] = frame[key].astype(str).str.lower().eq("true")
    provenance = json.loads((directory / "provenance.json").read_text())
    config = provenance["config"]
    expected_seed_count = len(config["seeds"])
    repeat_count = config["repetitions"]
    variable_ranks = frame.groupby("method")["rank_requested"].nunique().gt(1)
    frame["method"] = [f"{method}[r={rank}]" if variable_ranks.get(method, False) else method
                       for method, rank in zip(frame["method"], frame["rank_requested"])]
    output = directory / "figures"
    output.mkdir(exist_ok=True)
    metrics = ["velocity_relative_l2", "vorticity_relative_l2", "extra_divergence_normalized",
               "kinetic_energy_relative_l2", "cpu_seconds_median", "wall_seconds_median", "rss_peak_bytes_median"]
    for metric in metrics:
        if metric not in frame:
            frame[metric] = np.nan
    groups = frame.groupby(["case", "noise", "method"], sort=False)
    aggregate = groups[metrics].median()
    for metric in metrics:
        aggregate[metric + "_seed_iqr"] = groups[metric].quantile(.75) - groups[metric].quantile(.25)
    aggregate["converged_seeds"] = groups["all_repetitions_converged"].sum()
    aggregate["quality_pass_seeds"] = groups["quality_pass"].sum()
    aggregate["seed_conditions"] = groups.size()
    aggregate.to_csv(directory / "quality_by_seed.csv")
    aggregate = aggregate.reset_index()
    methods = list(dict.fromkeys(frame["method"]))
    noise_order = [n["kind"] for n in config["noise_models"]]
    for case in frame["case"].unique():
        subset = aggregate[aggregate["case"].eq(case)]
        matrix = subset.pivot(index="noise", columns="method", values="velocity_relative_l2").reindex(index=noise_order, columns=methods)
        fig, axis = plt.subplots(figsize=(13, 5))
        values = np.maximum(matrix.to_numpy(float), 1e-8)
        picture = axis.imshow(np.log10(values), aspect="auto", cmap="viridis", vmin=-4, vmax=0)
        axis.set_xticks(range(len(methods)), methods, rotation=35, ha="right")
        axis.set_yticks(range(len(noise_order)), noise_order)
        for y in range(values.shape[0]):
            for x in range(values.shape[1]):
                original = matrix.iloc[y, x]
                text = "fallo" if not np.isfinite(original) else ("<0.01%" if 0 < original < .0001 else f"{100 * original:.2f}%")
                row = subset[(subset["noise"] == noise_order[y]) & (subset["method"] == methods[x])]
                if not row.empty and row["converged_seeds"].iloc[0] < expected_seed_count:
                    text += "*"
                axis.text(x, y, text, ha="center", va="center", fontsize=7,
                          color="white" if np.isfinite(original) and original < .025 else "black")
        axis.set_title(f"{case}: mediana de error de velocidad entre {expected_seed_count} semillas\n* alguna semilla no convergió; estas estimaciones no certifican aceptación física", fontsize=10)
        fig.colorbar(picture, ax=axis, label="log10(error relativo)")
        fig.tight_layout()
        fig.savefig(output / (case + "_velocity.png"), dpi=180)
        fig.savefig(output / (case + "_velocity.pdf"))
        plt.close(fig)
    # Matched-condition cost frontier, without pooling physically different cases.
    sample_case = "taylor_green_ns" if "taylor_green_ns" in set(frame["case"]) else frame["case"].iloc[0]
    sample_noise = "gaussian" if "gaussian" in set(frame["noise"]) else frame["noise"].iloc[0]
    sample_seed = config["seeds"][0]
    sample = frame[(frame["case"] == sample_case) & (frame["noise"] == sample_noise) & (frame["seed"] == sample_seed)]
    fig, axis = plt.subplots(figsize=(12, 6))
    markers = ["o", "s", "D", "^", "v", ">", "<", "p", "h", "*", "X"]
    for index, (_, row) in enumerate(sample.iterrows()):
        if not np.isfinite(row.get("cpu_seconds_median", np.nan)):
            continue
        color = "#247747" if row["quality_pass"] else "#ab4b38"
        marker = markers[index % len(markers)] if row["all_repetitions_converged"] else "x"
        axis.scatter(row["cpu_seconds_median"], max(row["velocity_relative_l2"], 1e-8),
                     c=color, marker=marker, s=75, label=row["method"])
    axis.set(xscale="log", yscale="log", xlabel=f"CPU, mediana de {repeat_count} procesos [s]", ylabel="Error relativo de velocidad",
             title=f"Costo y fidelidad: {sample_case}, {sample_noise}, semilla {sample_seed}\nVerde: supera todos los umbrales físicos; rojo: no los supera")
    axis.grid(True, which="both", alpha=.2)
    if axis.get_legend_handles_labels()[0]:
        axis.legend(loc="center left", bbox_to_anchor=(1.02, .5), fontsize=8, title="Estimador")
    else:
        axis.text(.5, .5, "Sin ajustes exitosos medidos", transform=axis.transAxes,
                  ha="center", va="center")
    fig.tight_layout()
    fig.savefig(output / "cost_quality.png", dpi=180)
    fig.savefig(output / "cost_quality.pdf")
    plt.close(fig)
    best_rows = []
    for (case, noise), subset in aggregate.groupby(["case", "noise"], sort=False):
        eligible = subset[subset["converged_seeds"].eq(expected_seed_count)].dropna(subset=["velocity_relative_l2"])
        if eligible.empty:
            continue
        best = eligible.loc[eligible["velocity_relative_l2"].idxmin()]
        best_rows.append({key: best[key] for key in ["case", "noise", "method", "velocity_relative_l2", "vorticity_relative_l2", "quality_pass_seeds"]})
    pd.DataFrame(best_rows, columns=["case", "noise", "method", "velocity_relative_l2",
                                    "vorticity_relative_l2", "quality_pass_seeds"]).to_csv(
        directory / "lowest_velocity_error.csv", index=False)
    # Cost at the predeclared fidelity tolerance; absence of a feasible estimator
    # is recorded as absence, never substituted by the least inaccurate one.
    feasible_cost = []
    for (case, noise, seed), subset in frame.groupby(["case", "noise", "seed"], sort=False):
        eligible = subset[subset["quality_pass"]]
        if eligible.empty:
            feasible_cost.append({"case": case, "noise": noise, "seed": seed, "status": "no_feasible_estimator"})
        else:
            chosen = eligible.loc[eligible["cpu_seconds_median"].idxmin()]
            feasible_cost.append({"case": case, "noise": noise, "seed": seed, "status": "feasible",
                                  "method": chosen["method"], "cpu_seconds_median": chosen["cpu_seconds_median"],
                                  "rss_peak_bytes_median": chosen["rss_peak_bytes_median"], "velocity_relative_l2": chosen["velocity_relative_l2"]})
    pd.DataFrame(feasible_cost).to_csv(directory / "cost_at_physical_tolerance.csv", index=False)
    selected = best_rows
    doc = ["# Resultados de la ampliación ejecutada", "",
           f"Campaña `{campaign}`: **{completion['conditions']} condiciones, {completion['fits']} ajustes**. "
           f"{frame['case'].nunique()} campos limpios, {frame['noise'].nunique()} ruidos, {frame['method'].nunique()} configuraciones de métodos, "
           f"{expected_seed_count} semillas y {repeat_count} repeticiones de recursos por condición.", "",
           f"Se registraron **{completion['failed_fits']} ajustes fallidos**, **{completion['not_converged_conditions']} condiciones sin convergencia completa** "
           f"y **{completion['quality_pass_conditions']} condiciones que cumplen los umbrales físicos y la convergencia declarada**.", "",
           "## Qué muestran los cálculos", "",
           f"Esta tabla selecciona, para cada caso y ruido, el menor error de velocidad mediano entre métodos que convergieron en las {expected_seed_count} semillas. "
           "Es una comparación descriptiva dentro del piloto, no una selección de hiperparámetros ni una prueba de superioridad universal. "
           "La última columna indica cuántas semillas cumplen además todos los umbrales físicos.", "",
           "| Caso | Ruido | Menor error de velocidad | Error u | Error ω | Aceptación física |", "|---|---|---|---:|---:|---:|"]
    for row in selected:
        doc.append(f"| {row['case']} | {row['noise']} | {row['method']} | {percentage(row['velocity_relative_l2'])}% | {percentage(row['vorticity_relative_l2'])}% | {int(row['quality_pass_seeds'])}/{expected_seed_count} |")
    noise_acceptance = frame.groupby("noise")["quality_pass"].agg(["sum", "size"])
    feasible_count = sum(item["status"] == "feasible" for item in feasible_cost)
    group_count = len(feasible_cost)
    doc.extend(["", f"Sólo **{feasible_count}/{group_count} grupos caso/ruido/semilla** tienen algún estimador admisible con estos umbrales. "
                f"Los otros **{group_count-feasible_count}** quedan registrados como ausencia de solución a la tolerancia fijada, no como métodos rápidos exitosos.", "",
                "| Ruido | Condiciones que cumplen todos los criterios |", "|---|---:|"])
    for noise, row in noise_acceptance.iterrows():
        doc.append(f"| {noise} | {int(row['sum'])}/{int(row['size'])} |")
    doc.extend(["", "En Taylor–Green gaussiano, la familia DMD alcanza menor error de velocidad que PCP en este rango y presupuesto. "
                "Una mejora de velocidad por sí sola no garantiza cumplir simultáneamente los criterios físicos. La diferencia entre las columnas de velocidad, vorticidad y aceptación "
                "justifica investigar restricciones físicas, sensibilidad de rango y parámetros en validación independiente. Los conteos son descriptivos del banco; "
                "no representan probabilidades de éxito en flujos reales."])
    validation_prefix = os.path.relpath(ROOT / "docs" / "research", directory)
    doc.extend(["", "Los ruidos y presupuestos de intensidad son los declarados en provenance.json; el RMS efectivamente realizado aparece en summary.csv. "
                "La intensidad conocida sólo se usa como presupuesto experimental en la regularización estable; "
                "no se ajustan parámetros con la referencia limpia.", "",
                "## Costo y fidelidad física", "",
                f"Se conservan CPU, tiempo de pared, RSS pico, memoria trazada, mediana e IQR de {repeat_count} ejecuciones, número de SVD, rango efectivo, iteraciones "
                "y proxy de operaciones. `cost_at_physical_tolerance.csv` encuentra el ajuste de menor CPU entre los que cumplen tolerancias; también registra los grupos sin solución admisible. "
                "`pareto_member` identifica la frontera exploratoria y `feasible_pareto_member` la frontera restringida a tolerancias físicas, dentro de cada caso/ruido/semilla.", "",
                "![Costo y calidad](figures/cost_quality.png)", "",
                "Los perfiles estacionarios HB/Bingham son referencias temporales de rango uno y no representan estelas ni caos. "
                "Consulte los metadatos para distinguir referencias paralelas de canales y solvers bidimensionales. La evaluación de reconstrucción es dentro de muestra; "
                "ninguna cifra demuestra pronóstico, desempeño PIV real o generalización entre fluidos.", "",
                "## Evidencia y límites", "",
                "- [Resumen de cada condición](summary.csv).",
                "- [Repeticiones y diagnósticos](ledger.jsonl).",
                "- [Proveniencia, parámetros y hashes](provenance.json).",
                "- [Variación entre semillas](quality_by_seed.csv).",
                f"- [Convergencia del solver NS]({validation_prefix}/DNS_VALIDATION.md) y [reología]({validation_prefix}/NON_NEWTONIAN.md).", "",
                "Las métricas de masa y energía requieren densidad declarada; una reconstrucción de velocidad no estima densidad. La continuidad, divergencia y flujo se comparan tanto "
                "con cero como con el residual de la referencia. Los canales agregan comprobación de no deslizamiento y penetración. La convergencia de PCP "
                "se refiere a factibilidad primal, sin certificado de optimalidad; los fallos y no convergencias no se eliminan de los registros. "
                "El tiempo incluye instrumentación `tracemalloc` y el RSS incluye runtime y entradas; estas mediciones no se equiparan a los tiempos históricos de la tesis.", "",
                "Las tolerancias son exploratorias predeclaradas. Una configuración no demuestra que una campaña se haya ejecutado; "
                "las conclusiones se limitan a las condiciones registradas y al alcance declarado por cada generador."])
    (directory / "discussion.md").write_text("\n".join(doc) + "\n")
    if update_docs:
        document = "\n".join(doc) + "\n"
        for link in ("figures/cost_quality.png", "summary.csv", "ledger.jsonl", "provenance.json", "quality_by_seed.csv"):
            document = document.replace(f"]({link})", f"](../../results/research/{campaign}/{link})")
        document = document.replace(f"]({validation_prefix}/", "](")
        (ROOT / "docs" / "research" / "RESULTS.md").write_text(document)
    latex = [r"\section{Resultados calculados}",
             f"La campaña ejecutó {completion['conditions']} condiciones y {completion['fits']} ajustes. "
             f"Hubo {completion['failed_fits']} ajustes fallidos y {completion['not_converged_conditions']} condiciones sin convergencia completa. "
             f"Cumplieron convergencia declarada y umbrales físicos {completion['quality_pass_conditions']} condiciones.",
             r"{\scriptsize\setlength{\tabcolsep}{3pt}",
             r"\begin{longtable}{p{.17\textwidth}p{.16\textwidth}p{.30\textwidth}rrr}",
             r"\caption{Menor error mediano de velocidad entre estimadores convergidos en todas las semillas declaradas. AP: semillas aceptadas físicamente.}\\",
             r"\toprule Caso & Ruido & Método & $e_u$ [\%] & $e_\omega$ [\%] & AP \\\midrule\endfirsthead",
             r"\toprule Caso & Ruido & Método & $e_u$ [\%] & $e_\omega$ [\%] & AP \\\midrule\endhead"]
    for row in selected:
        case = row['case'].replace('_ns','').replace('_legacy','').replace('hb_shear_','HB ')
        noise = {"gaussian": "gaussiano", "sparse_impulsive": "impulsivo"}.get(row["noise"], row["noise"])
        latex.append(f"{tex_escape(case)} & {tex_escape(noise)} & {tex_escape(row['method'])} & {percentage(row['velocity_relative_l2'], True)} & {percentage(row['vorticity_relative_l2'], True)} & {int(row['quality_pass_seeds'])}/{expected_seed_count} " + r"\\")
    latex.extend([r"\bottomrule\end{longtable}}",
                  f"Sólo {feasible_count} de {group_count} grupos caso/ruido/semilla tienen algún estimador admisible a estos umbrales. "
                  f"Los {group_count-feasible_count} restantes no ofrecen solución al costo y fidelidad fijados. "
                  "Los valores pequeños se indican mediante desigualdad para no representar una aproximación como error exactamente cero. "
                  "Estos conteos son descriptivos del banco y no probabilidades de éxito en mediciones reales.",
                  r"\begin{figure}[htbp]\centering",
                  r"\includegraphics[width=.95\textwidth]{results/research/"+campaign+r"/figures/cost_quality.pdf}",
                  r"\caption{Costo medido y error de velocidad en una condición coincidente. La aceptación incluye vorticidad, divergencia y energía; en canales, paredes.}\end{figure}",
                  r"\section{Límites de la inferencia}",
                  "Las semillas son réplicas de contaminación del mismo campo; las repeticiones de recursos no aumentan el tamaño de muestra de calidad. "
                  "La evaluación es reconstrucción dentro de la ventana, sin pronóstico ni mediciones PIV reales. El parámetro de ruido conocido "
                  "es información del experimento sintético. Los perfiles HB estacionarios favorecen bajo rango temporal y no representan estelas; consulte los metadatos para distinguir perfiles de canales y solvers bidimensionales. "
                  "La reconstrucción de velocidad no estima densidad: las métricas de masa y energía requieren densidad declarada. "
                  "Los tiempos y métricas históricas de magnitud de velocidad no son directamente comparables a estas componentes vectoriales ni al entorno instrumentado.",
                  r"Los CSV, ledger JSONL, configuraciones y hashes acompañan esta discusión en \texttt{results/research/"+tex_escape(campaign)+"}. "
                  "Las figuras de todos los casos se exportan a PNG y PDF; el resumen distingue convergencia y factibilidad física."])
    (directory / "results_generated.tex").write_text("\n".join(latex) + "\n")
    if update_docs:
        (ROOT / "docs" / "research" / "results_generated.tex").write_text("\n".join(latex) + "\n")
    print(directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", default="baseline_2026-10-14/pilot_2026-10-14")
    parser.add_argument("--update-docs", action="store_true",
                        help="Also replace docs/research/RESULTS.md and results_generated.tex")
    arguments = parser.parse_args()
    analyze(arguments.campaign, arguments.update_docs)
