#!/usr/bin/env python3
"""Build the thesis comparison chapter from complete, verified evidence.

This renderer performs no fit and selects no hyperparameter. JSON/CSV evidence
and published figure files suffice; absent raw NPZ archives do not prevent a
rebuild. Available inputs with declared hashes are verified before any output
is replaced. Missing/incomplete campaigns are fatal by default.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs/research"
RESULTS = ROOT / "results/research"
PAIRED = RESULTS / "paired_model_comparison_2026-10-14"
FIDELITY = RESULTS / "reference_fidelity_2026-10-14"
BENCHMARK = RESULTS / "model_fidelity_benchmark_2026-10-14"
SENSITIVITY = RESULTS / "rank_noise_sensitivity_2026-10-14"
PRIOR = RESULTS / "prior_spectral_budget_2026-10-14"
CLEAN = RESULTS / "clean_model_controls_2026-10-14"
METHODS = ["identity", "pod_r8", "dmd_r8", "tls_r8", "pcp", "stable_rpca",
           "outlier_pursuit", "pcp_tls_r8", "stable_tls_r8", "optimized_r8"]
VARIANTS = ["raw", "helmholtz", "helmholtz_k10"]
METHOD_LABELS = {"identity": "Identity", "pod_r8": "POD r=8", "dmd_r8": "DMD r=8",
                "tls_r8": "TLS-DMD r=8", "pcp": "RPCA/PCP", "stable_rpca": "RPCA estable",
                "outlier_pursuit": "Outlier Pursuit", "pcp_tls_r8": "PCP + TLS-DMD",
                "stable_tls_r8": "RPCA estable + TLS-DMD", "optimized_r8": "Optimized DMD r=8"}
VARIANT_LABELS = {"raw": "Raw", "helmholtz": "H", "helmholtz_k10": "H+B"}
NOISE_LABELS = {"gaussian_iid": "Gaussiano IID", "sparse_impulsive": "Impulsivo disperso",
                "spatial_gaussian": "Espacial correlacionado", "temporal_ar": "Temporal AR",
                "column_outliers": "Columnas corruptas"}
FLOW_LABELS = {"ns_vortices": "Vórtices NS", "ns_multiscale": "Multiescala NS",
               "interacting_vortices": "Vórtices NS", "decaying_turbulence": "Multiescala NS",
               "oldroyd_b": "Oldroyd-B", "fene_p": "FENE-P", "hb_thinning": "HB n=0.6",
               "bingham": "Bingham", "hb_thickening": "HB n=1.4"}
OLD_FLOW_LABELS={"taylor_green_ns":"Taylor–Green", "interacting_vortices_ns":"Vórtices NS",
 "decaying_turbulence_ns":"Multiescala NS", "oldroyd_b_2d":"Oldroyd-B", "fene_p_2d":"FENE-P",
 "hb_thinning_2d":"HB n=0.6", "bingham_2d":"Bingham", "hb_thickening_2d":"HB n=1.4"}
OLD_METHOD_LABELS = {"identity": "Identity", "identity_helmholtz": "Identity + H",
 "pod": "POD", "pod_helmholtz": "POD + H", "rpca": "RPCA/PCP",
 "stable_rpca": "RPCA estable", "stable_rpca_helmholtz": "RPCA estable + H",
 "dmd": "DMD", "tls_dmd": "TLS-DMD", "tls_dmd_helmholtz": "TLS-DMD + H",
 "tls_dmd_helmholtz_dealias": "TLS-DMD + H + 2/3", "optimized_dmd": "Optimized DMD",
 "fb_dmd_fallback_tls": "fbDMD con alternativa TLS"}
GATES = {"velocity_relative_l2": .15, "vorticity_relative_l2": .25,
         "extra_divergence_normalized": .10, "kinetic_energy_relative_l2": .15}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    require(path.is_file(), f"Missing evidence: {path.relative_to(ROOT)}")
    return json.loads(path.read_text())


def read_csv(path):
    require(path.is_file(), f"Missing evidence: {path.relative_to(ROOT)}")
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def fmt(value):
    if value is None:
        return "ND"
    require(number(value), f"Nonnumeric reported value: {value!r}")
    if value == 0:
        return "0"
    return f"{value:.3e}" if abs(value) < .001 or abs(value) >= 10000 else f"{value:.4g}"


def pct(value):
    return fmt(100*value) + " %" if value is not None else "ND"


def points(value):
    return fmt(100*value) + " pp" if value is not None else "ND"


def median(rows, key):
    values = [row[key] for row in rows if number(row.get(key))]
    return statistics.median(values) if values else None


def tex(value):
    text = str(value)
    text = text.translate(str.maketrans({"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
      "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
      "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}))
    for old, new in {"ω": r"$\omega$", "τ": r"$\tau$", "η": r"$\eta$",
     "γ": r"$\gamma$", "σ": r"$\sigma$", "λ": r"$\lambda$", "ν": r"$\nu$",
     "ρ": r"$\rho$", "π": r"$\pi$", "²": r"$^2$", "≤": r"$\leq$",
      "≥": r"$\geq$", "→": r"$\to$", "−": "-", "Δ": r"$\Delta$",
      "10⁻¹⁰": r"$10^{-10}$", "E₀": r"$E_0$"}.items():
        text = text.replace(old, new)
    text = text.replace(r"e\_u", r"$e_u$").replace(r"e\_$\omega$", r"$e_\omega$")
    text = text.replace(r" \%", r"~\%").replace("\n", r" \newline ")
    return text


class Evidence:
    def __init__(self):
        self.sources = {}
        self.verified = []
        self.unavailable_archives = []

    def source(self, path):
        require(path.is_file(), f"Missing evidence: {path.relative_to(ROOT)}")
        self.sources[str(path.relative_to(ROOT))] = sha(path)

    def declared(self, declared, base):
        """Verify explicit path-to-hash maps, permitting separately archived NPZ."""
        for name, digest in declared.items():
            require(isinstance(digest, str) and len(digest) == 64, f"Invalid SHA256 for {name}")
            relative = Path(name)
            root_path, base_path = ROOT / relative, base / relative
            path = root_path if root_path.is_file() else base_path
            if not path.is_file() and relative.suffix == ".npz":
                self.unavailable_archives.append(str(relative))
                continue
            require(path.is_file(), f"Declared source missing: {name}")
            require(sha(path) == digest, f"SHA256 mismatch: {name}")
            self.source(path)
            self.verified.append(str(path.relative_to(ROOT)))

    def campaign(self, folder):
        completion_path=folder / "completion.json"
        if not completion_path.is_file():
            completion_path=folder / "summary.json"
        completion = read(completion_path)
        require(isinstance(completion,dict), f"Missing completion marker: {folder.name}")
        require(completion.get("complete") is True, f"Incomplete campaign: {folder.name}")
        provenance = read(folder / "provenance.json")
        self.source(completion_path)
        self.source(folder / "provenance.json")
        for key in ("source_sha256", "artifact_sha256", "estimator_source_hashes", "scientific_source_hashes",
                    "scientific_source_sha256", "input_sha256", "source_files_sha256"):
            if key in provenance:
                require(isinstance(provenance[key], dict), f"Invalid hash map: {key}")
                self.declared(provenance[key], folder)
        datasets=provenance.get("clean_input_sources",provenance.get("source_datasets",{}))
        if isinstance(datasets,list):
            datasets={str(i):value for i,value in enumerate(datasets)}
        if not datasets and isinstance(provenance.get("sources"),list):
            datasets={str(i):value for i,value in enumerate(provenance["sources"])}
        for input_record in datasets.values():
            self.declared({input_record["path"]: input_record["sha256"]}, folder)
            metadata = (ROOT / input_record["path"]).with_suffix(".json")
            if "metadata_sha256" in input_record:
                self.declared({str(metadata.relative_to(ROOT)): input_record["metadata_sha256"]}, folder)
        driver_names={PAIRED:"python/scripts/analyze_paired_model_comparison.py",
                      FIDELITY:"python/scripts/analyze_reference_fidelity.py",
                      BENCHMARK:"python/scripts/run_model_fidelity_benchmark.py",
                      SENSITIVITY:"python/scripts/run_rank_noise_sensitivity.py",
                      PRIOR:"python/scripts/analyze_prior_error_budget.py",
                      CLEAN:"python/scripts/run_clean_model_controls.py"}
        driver_hash=provenance.get("script_sha256",provenance.get("driver_sha256"))
        if driver_hash is not None:
            self.declared({provenance.get("script",driver_names[folder]):driver_hash},folder)
        if provenance.get("helper_driver_sha256") is not None:
            self.declared({"python/scripts/run_model_fidelity_benchmark.py":provenance["helper_driver_sha256"]},folder)
        configuration_files={BENCHMARK:"configs/research/model_fidelity_benchmark.json",
                             CLEAN:"configs/research/clean_model_controls.json",
                             SENSITIVITY:"configs/research/rank_noise_sensitivity.json"}
        if folder in configuration_files:
            configuration_path=ROOT/configuration_files[folder]
            self.source(configuration_path)
            require(read(configuration_path)==provenance["configuration"], f"Configuration differs from executed design: {folder.name}")
            if "configuration_sha256" in provenance:
                self.declared({configuration_files[folder]:provenance["configuration_sha256"]},folder)
        return completion, provenance


class Document:
    def __init__(self, evidence):
        self.evidence = evidence
        self.markdown = ["# Comparación de modelos y fidelidad de simulaciones", "",
                         "Análisis generado exclusivamente desde campañas completas y verificadas.", ""]
        self.latex = ["% Generated by python/scripts/build_comparison_chapter.py; do not edit."]
        self.figure_count = 0

    def section(self, title):
        self.markdown += ["## " + title, ""]
        self.latex += [r"\clearpage", r"\section{" + tex(title) + "}", ""]

    def subsection(self, title):
        self.markdown += ["### " + title, ""]
        self.latex += [r"\subsection{" + tex(title) + "}", ""]

    def paragraph(self, text, latex=None):
        self.markdown += [text, ""]
        self.latex += [latex if latex is not None else tex(text), ""]

    def table(self, headers, rows, caption, widths=None):
        markdown_labels={"Trap. |ΔE|/E₀":"Trapecios: defecto absoluto/E₀",
                         "Simpson |ΔE|/E₀":"Simpson: defecto absoluto/E₀"}
        markdown_headers=[markdown_labels.get(header,header) for header in headers]
        self.markdown += [caption, "", "| " + " | ".join(markdown_headers) + " |",
                          "| " + " | ".join("---" for _ in headers) + " |"]
        self.markdown += ["| " + " | ".join(str(item).replace("\n","<br>") for item in row) + " |" for row in rows] + [""]
        if widths is None:
            widths = [3.7] + [(15 - 3.7)/(len(headers)-1)]*(len(headers)-1)
        available=15.9-(len(headers)-1)*6/28.4527559
        if sum(widths)>available:
            widths=[round(width*available/sum(widths),4) for width in widths]
        spec = "@{}" + "".join(r">{\raggedright\arraybackslash}p{" + str(width) + "cm}" for width in widths) + "@{}"
        self.latex += [r"\begingroup\small\setlength{\tabcolsep}{3pt}", r"\begin{longtable}{" + spec + "}",
          r"\caption{" + tex(caption) + r"}\\", r"\toprule", " & ".join(r"\textbf{"+tex(h)+"}" for h in headers) + r" \\",
          r"\midrule\endfirsthead", r"\toprule", " & ".join(r"\textbf{"+tex(h)+"}" for h in headers) + r" \\", r"\midrule\endhead"]
        self.latex += [" & ".join(tex(item) for item in row) + r" \\" for row in rows]
        self.latex += [r"\bottomrule\end{longtable}\endgroup", ""]

    def figure(self, folder, name, caption, label, landscape=False):
        path = folder / "figures" / (name + ".pdf")
        png = path.with_suffix(".png")
        self.evidence.source(path)
        self.evidence.source(png)
        relative = path.relative_to(ROOT).as_posix()
        self.markdown += ["![" + caption + "](" + "../../" + png.relative_to(ROOT).as_posix() + ")", "", caption, ""]
        self.latex += [r"\clearpage"]
        if landscape:
            self.latex += [r"\begin{landscape}"]
        self.latex += [r"\begin{figure}[p]\centering",
            r"\includegraphics[width=" + (r"\linewidth,height=.80\textwidth" if landscape else r"\textwidth,height=.71\textheight")
            + ",keepaspectratio]{" + relative + "}", r"\caption{" + tex(caption) + "}",
            r"\label{" + label + r"}\end{figure}"]
        if landscape:
            self.latex += [r"\end{landscape}"]
        self.latex += [r"\clearpage", ""]
        self.figure_count += 1

    def optional_figure(self, folder, names, caption, label, landscape=False):
        """Render exactly one existing named artifact, never invent a figure."""
        for name in names:
            if (folder / "figures" / (name + ".pdf")).is_file():
                self.figure(folder, name, caption, label, landscape)
                return name
        raise ValueError(f"Missing required comparative illustration: {folder.name}: {names}")


def checked_inputs():
    evidence = Evidence()
    paired_completion, paired_provenance = evidence.campaign(PAIRED)
    fidelity_completion, fidelity_provenance = evidence.campaign(FIDELITY)
    benchmark_completion, benchmark_provenance = evidence.campaign(BENCHMARK)
    sensitivity_completion, sensitivity_provenance = evidence.campaign(SENSITIVITY)
    prior_completion, prior_provenance = evidence.campaign(PRIOR)
    clean_completion, clean_provenance = evidence.campaign(CLEAN)
    rows = read(BENCHMARK / "summary.json")
    groups = read(BENCHMARK / "group_summary.json")
    figure_hashes=read(BENCHMARK/"figure_manifest.json")
    evidence.source(BENCHMARK/"figure_manifest.json")
    evidence.declared(figure_hashes,BENCHMARK/"figures")
    integrity=read(BENCHMARK/"integrity_validation.json")
    evidence.source(BENCHMARK/"integrity_validation.json")
    evidence.source(BENCHMARK/"ledger.jsonl")
    require(integrity.get("valid") is True and integrity["conditions_verified"]==1050
            and integrity["regenerated_paired_observations"]==105, "Benchmark arithmetic/provenance audit incomplete")
    fidelity = read(FIDELITY / "summary.json")
    dense=read(FIDELITY/"dense_sampling.json")
    require(dense.get("complete") is True, "Dense sampling isolation is incomplete")
    evidence.source(FIDELITY/"dense_sampling.json")
    old_groups = read(PAIRED / "method_method_summary.json")
    old_ablations = read(PAIRED / "ablation_summary.json")
    old_blocks = read(PAIRED / "paired_block_feasibility.json")
    old_conditions = read(PAIRED / "conditions.json")
    for folder, names in ((BENCHMARK, ["summary.json", "summary.csv", "group_summary.json"]),
                         (FIDELITY, ["summary.json"]),
                         (PAIRED, ["method_method_summary.json", "ablation_summary.json", "paired_block_feasibility.json", "conditions.json"]),
                         (SENSITIVITY, ["aggregates.json", "summary.csv", "oracle_floor.csv"])):
        for name in names:
            evidence.source(folder / name)
    require(len(rows) == benchmark_completion["expected_evaluations"], "Evaluation cardinality mismatch")
    unique_keys = {(r["flow"], r["noise"], r["seed"], r["method"], r["variant"]) for r in rows}
    require(len(unique_keys) == len(rows), "Duplicated paired benchmark evaluations")
    require(set(r["method"] for r in rows) == set(METHODS), "Incomplete method panel")
    require(set(r["variant"] for r in rows) == set(VARIANTS), "Incomplete physical ablation panel")
    require(len({(r["flow"], r["noise"], r["seed"], r["method"]) for r in rows}) == benchmark_completion["expected_fits"],
            "Fit cardinality mismatch")
    for variant in VARIANTS:
        sample = [r for r in rows if r["variant"] == variant]
        require(sum(bool(r["quality_pass"]) for r in sample) == benchmark_completion["quality_passes_by_variant"][variant],
                f"Acceptance count mismatch: {variant}")
        for row in sample:
            if row["quality_pass"]:
                require(row["status"] == "ok" and row["converged"], "Nonconverged output marked admissible")
                require(all(number(row.get(key)) and row[key] <= maximum for key, maximum in GATES.items()),
                        "Physical gate inconsistent with reported acceptance")
    require(paired_completion["conditions"] == 2808 and paired_completion["resource_measurements"] == 8424,
            "Historical paired analysis changed its evidence base")
    require(paired_completion["nonconverged_conditions"] == 6, "Historical nonconvergence omitted")
    require(paired_completion["fallback_conditions"] == 23, "Historical fbDMD alternatives omitted")
    sensitivity_rows=read_csv(SENSITIVITY / "summary.csv")
    sensitivity_aggregates=read(SENSITIVITY / "aggregates.json")
    require(sensitivity_completion["fits"] == sensitivity_completion["expected_fits"] == 360,
            "Sensitivity fit cardinality changed")
    require(len(sensitivity_rows) == sensitivity_completion["evaluations"] == 1080,
            "Sensitivity evaluation cardinality changed")
    sensitivity_keys={(r["flow"],r["noise"],r["intensity_relative_rms"],r["seed"],r["rank"],r["method"],r["postprocessing"])
                      for r in sensitivity_rows}
    require(len(sensitivity_keys)==len(sensitivity_rows), "Duplicated sensitivity evaluation")
    require(len(sensitivity_aggregates)==360, "Sensitivity group cardinality changed")
    prior=read(PRIOR/"summary.json")
    evidence.source(PRIOR/"summary.json")
    evidence.source(PRIOR/"clean_prior_floors.csv")
    evidence.source(PRIOR/"identity_bias_noise_budget.csv")
    require(len(prior["selected_prior_floors"])==7 and prior["selected_physical_cutoff"]==10,
            "Fixed-band prior panel is incomplete or changed its cutoff")
    require(prior["max_squared_decomposition_relative_defect"]<1e-10,
            "Spectral bias/noise squared-error decomposition did not verify")
    pipeline=read(BENCHMARK/"pipeline_resource_summary.json")
    require(len(pipeline)==len(rows), "Pipeline resource table is incomplete")
    evidence.source(BENCHMARK/"pipeline_resource_summary.json")
    evidence.source(BENCHMARK/"operational_pareto.json")
    pipeline_index={(r["flow"],r["noise"],r["seed"],r["method"],r["variant"]):r for r in pipeline}
    require(set(pipeline_index)==unique_keys, "Pipeline resource keys differ from quality evaluations")
    for r in pipeline:
        if number(r.get("cpu_seconds")) and number(r.get("postprocess_cpu_seconds")):
            require(math.isclose(r["pipeline_cpu_seconds"],r["cpu_seconds"]+r["postprocess_cpu_seconds"],rel_tol=1e-12,abs_tol=1e-14),
                    "Pipeline CPU is not fit plus this postprocessing variant")
    clean_rows=read(CLEAN/"summary.json")
    for name in ("summary.json","summary.csv","regularization_contrasts.json","clean_pod_rank8_diagnostic.json"):
        evidence.source(CLEAN/name)
    require(clean_completion["recorded_fits"]==clean_completion["expected_fits"]==84,
            "No-noise control fit panel is incomplete")
    require(len(clean_rows)==clean_completion["recorded_evaluations"]==252,
            "No-noise control output panel is incomplete")
    require(clean_completion["noise_realizations"]==0 and clean_completion["primary_contract_fits"]==70
            and clean_completion["known_zero_noise_fits"]==14, "No-noise control contracts changed")
    clean_keys={(r["flow"],r["method"],r["contract"],r["variant"]) for r in clean_rows}
    require(len(clean_keys)==252,"Duplicated no-noise control evaluation")
    return dict(evidence=evidence, rows=rows, groups=groups, fidelity=fidelity,dense_sampling=dense,
                paired_completion=paired_completion, paired_provenance=paired_provenance,
                fidelity_completion=fidelity_completion, fidelity_provenance=fidelity_provenance,
                benchmark_completion=benchmark_completion, benchmark_provenance=benchmark_provenance,
                sensitivity_completion=sensitivity_completion, sensitivity_provenance=sensitivity_provenance,
                old_groups=old_groups, old_ablations=old_ablations, old_blocks=old_blocks, old_conditions=old_conditions,
                prior=prior, prior_provenance=prior_provenance,
                pipeline=pipeline,
                clean_rows=clean_rows,clean_completion=clean_completion,clean_provenance=clean_provenance,
                sensitivity_aggregates=sensitivity_aggregates, sensitivity_rows=sensitivity_rows,
                oracle_floor=read_csv(SENSITIVITY / "oracle_floor.csv"))


def aggregate(rows, method=None, variant=None, noise=None, flow=None):
    sample = [r for r in rows if (method is None or r["method"] == method)
              and (variant is None or r["variant"] == variant)
              and (noise is None or r["noise"] == noise) and (flow is None or r["flow"] == flow)]
    return dict(conditions=len(sample), accepted=sum(bool(r["quality_pass"]) for r in sample),
                failures=sum(r["status"] != "ok" for r in sample),
                nonconverged=sum(r["status"] == "ok" and not r["converged"] for r in sample),
                velocity=median(sample, "velocity_relative_l2"),
                vorticity=median(sample, "vorticity_relative_l2"),
                energy=median(sample, "kinetic_energy_relative_l2"),
                cpu=median(sample, "cpu_seconds"), rss=median(sample, "rss_peak_bytes"),
                proxy=median(sample, "flops_proxy"))


def fixed_contrast(rows, method_a, method_b, noise, variant="raw"):
    """Descriptive signed paired differences; finite diagnostics remain visible."""
    index={(r["flow"],r["noise"],r["seed"],r["method"],r["variant"]):r for r in rows}
    differences={"velocity_relative_l2":[],"vorticity_relative_l2":[]}
    wins=ties=losses=nonconverged=0
    for key,a in index.items():
        if key[1]!=noise or key[3]!=method_a or key[4]!=variant:
            continue
        b=index[key[:3]+(method_b,variant)]
        if not all(number(r.get(q)) for r in (a,b) for q in differences):
            continue
        nonconverged+=not(a["converged"] and b["converged"])
        for q in differences:
            differences[q].append(a[q]-b[q])
        delta=a["vorticity_relative_l2"]-b["vorticity_relative_l2"]
        wins+=delta < -1e-10
        losses+=delta > 1e-10
        ties+=abs(delta)<=1e-10
    return {"velocity":statistics.median(differences["velocity_relative_l2"]) if differences["velocity_relative_l2"] else None,
            "vorticity":statistics.median(differences["vorticity_relative_l2"]) if differences["vorticity_relative_l2"] else None,
            "paired":wins+ties+losses,"wins":wins,"ties":ties,"losses":losses,"nonconverged":nonconverged}


def make_document(data):
    evidence, rows = data["evidence"], data["rows"]
    document = Document(evidence)
    completion = data["benchmark_completion"]
    nblocks = len({(r["flow"], r["noise"], r["seed"]) for r in rows})
    document.section("Fidelidad de la referencia antes de comparar estimadores")
    document.paragraph("La pregunta sobre el mejor modelo de reconstrucción depende primero de la calidad de los campos limpios. "
        "Una recuperación fiel de una solución subresuelta reproduce sus limitaciones. Se separan error de malla, error temporal, "
        "residuo medido entre instantáneas y error frente a la observación limpia que generó el ruido. "
        "Las referencias finas son controles numéricos del mismo solver; Taylor–Green y los límites analíticos añaden verificaciones independientes.")
    credibility = data["fidelity"]["credibility_matrix"]
    chosen = []
    reference_groups = defaultdict(dict)
    # One row per case/horizon, preserving separate observable errors.
    for row in credibility:
        if row.get("campaign", "visual_highres") != "visual_highres":
            continue
        grid = row.get("grid")
        n = int(grid[0]) if isinstance(grid, list) else int(grid)
        if n != 128:
            continue
        observable = row.get("observable", "")
        if observable not in ("velocity", "vorticity", "stress_frobenius", "apparent_viscosity"):
            continue
        key=(row["case"], row["reference_grid"], row["start_time"], row["end_time"])
        reference_groups[key][observable]=row["maximum_relative_l2"]
    for (case,ref,t0,tf), values in reference_groups.items():
        constitutive=[]
        if "stress_frobenius" in values:
            constitutive.append("τ: "+pct(values["stress_frobenius"]))
        if "apparent_viscosity" in values:
            constitutive.append("η: "+pct(values["apparent_viscosity"]))
        chosen.append([FLOW_LABELS.get(case,case), "128² → "+str(ref)+"²", fmt(t0)+"–"+fmt(tf),
                       pct(values.get("velocity")),pct(values.get("vorticity")),"\n".join(constitutive) or "No aplica"])
    require(chosen, "No reference-credibility rows for evaluation grid")
    require(len(chosen)<=12, "Reference table requires a deliberate horizon-aware layout")
    document.table(["Referencia", "Mallas", "Intervalo", "Máx. e_u", "Máx. e_ω", "Máx. constitutivo"], chosen,
        "Discrepancia de campo completo en 128² frente a una solución fina; máximos en tiempos guardados. "
        "Los intervalos cortos y largos se mantienen separados. τ usa norma de Frobenius del esfuerzo.", [3.0,2.5,1.8,2.0,2.0,3.3])
    document.paragraph("La prioridad depende del observable. En Oldroyd-B la velocidad puede converger mucho antes que el tensor, "
        "y una cola de esfuerzo todavía visible en la referencia 256² limita una afirmación de resolución constitutiva completa. "
        "En HB adelgazante, la viscosidad aparente responde especialmente a zonas de corte pequeño y conserva sensibilidad a malla. "
        "El control 256² hasta t=0.1 no certifica todo el intervalo viscoplástico hasta t=2.")
    document.optional_figure(FIDELITY, ["fidelity_observable_grid"],
        "Fidelidad por observable y malla. Cada serie compara integraciones independientes con su referencia fina, "
        "sobre el horizonte indicado. La velocidad y sus derivadas o cierres constitutivos requieren precisiones diferentes; "
        "una diferencia pequeña entre mallas del mismo solver no elimina posibles sesgos comunes.", "fig:reference-credibility")
    document.subsection("Cómo modifica la referencia el error atribuido al estimador")
    bounds = data["fidelity"].get("triangle_bounds", [])
    if bounds:
        selected = [b for b in bounds if b.get("observable") == "vorticity"][:8]
        document.table(["Ruido / método", "Error vs. gruesa", "Error ref.", "Intervalo vs. fina", "Medido vs. fina"],
            [[str(b["noise"]) + " / " + str(b["method"]), pct(b["reconstruction_vs_coarse_reference"]),
              pct(b["coarse_reference_relative_error"]), pct(b["lower_vs_fine"]) + "–" + pct(b["upper_vs_fine"]),
              pct(b["actual_vs_fine"])] for b in selected],
            "Presupuesto de error de vorticidad: la cota utiliza denominador y espacio fino comunes, "
            "mientras que el error frente a la solución gruesa conserva su normalización original.", [4.5,2.3,2.0,3.2,2.3])
    document.optional_figure(FIDELITY, ["reconstruction_error_budget"],
        "Reconstrucción y discretización son fuentes de error distintas. Las cotas triangulares prolongan los campos "
        "al mismo espacio y normalizan por la referencia fina; no suman porcentajes con denominadores distintos. "
        "La referencia fina sigue siendo numérica y el intervalo describe discrepancia entre soluciones, no incertidumbre experimental.",
        "fig:reconstruction-budget")
    document.optional_figure(FIDELITY, ["residual_sampling"],
        "Dependencia de los residuos con la separación entre estados guardados y el operador temporal. "
        "La misma trayectoria puede producir residuos distintos al muestrearse con menor frecuencia. "
        "Estos diagnósticos no equivalen al residuo de las etapas internas del integrador ni certifican el solver.", "fig:sampled-residual")
    sampling_rows=[]
    for case in ("interacting_vortices","decaying_turbulence"):
        residuals=[r for r in data["fidelity"]["residual_sampling"] if r["case"]==case and abs(r["spacing"]-.1)<1e-12]
        energy=[r for r in data["fidelity"]["energy_sampling"] if r["case"]==case and abs(r["spacing"]-.1)<1e-12]
        r2=next(r for r in residuals if r["order"]==2)
        r4=next(r for r in residuals if r["order"]==4)
        require(r2["times"]==r4["times"], "Temporal residual stencils evaluated on different support")
        trap=next(r for r in energy if r["quadrature"]=="trapezoid")
        simpson=next(r for r in energy if r["quadrature"]=="simpson")
        sampling_rows.append([FLOW_LABELS[case],pct(r2["normalized_residual"]),pct(r4["normalized_residual"]),
                              fmt(abs(trap["final_normalized_defect"])),fmt(abs(simpson["final_normalized_defect"]))])
    document.table(["Referencia 512²", "Residuo orden 2", "Residuo orden 4", "Trap. |ΔE|/E₀", "Simpson |ΔE|/E₀"],
        sampling_rows,"Efecto del tratamiento de las mismas salidas cada 0.1: el residuo se normaliza por RMS de "
        "convección menos difusión sobre siete centros comunes entre t=0.8 y t=3.2; la energía usa el estado final t=4.",
        [3.2,2.9,2.9,2.7,2.7])
    document.paragraph("Esta normalización del residuo es la dinámica RMS de la propia trayectoria, distinta de la escala "
        "heurística de vorticidad utilizada en las métricas históricas. No se comparan directamente sus porcentajes. "
        "Cambiar stencil o cuadratura de salida puede reducir mucho el diagnóstico manteniendo exactamente la misma "
        "integración RK4: por tanto, el error de muestreo de salida debe separarse del error del solver.")
    document.subsection("Aislar el muestreo dentro de una misma integración")
    dense_rows=[]
    for case in ("interacting_vortices","decaying_turbulence"):
        for spacing in (.1,.025):
            r=next(r for r in data["dense_sampling"]["residual_sampling"]
                   if r["case"]==case and r["order"]==4 and abs(r["spacing"]-spacing)<1e-12)
            dense_rows.append([FLOW_LABELS[case],fmt(spacing),pct(r["normalized_residual"]),
                pct(r["projected_temporal_residual_normalized"]),pct(r["spatial_projection_defect_normalized"])])
    document.table(["Referencia 128²", "Separación de salida", "Residuo completo", "Componente temporal", "Defecto de proyección"],
        dense_rows,"Una única integración nueva por caso guarda 161 estados cada 0.025; las demás cadencias "
        "son submuestras exactas. Orden temporal cuatro y normalización dinámica RMS comunes. La malla permanece en 128².",
        [3.3,2.1,2.6,3.0,3.4])
    document.paragraph("Al guardar más estados, la componente temporal del residuo disminuye mientras el defecto "
        "de proyección espacial conserva su valor. El residuo completo alcanza así un piso asociado al operador "
        "discreto de esa malla. Las 161 salidas y sus submuestras proceden de una misma integración, por lo que no "
        "son nuevas realizaciones físicas. El contraste contra la integración original a tiempos comunes verifica "
        "por separado el pequeño cambio de paso interno. Esta prueba 128² se mantiene distinta de los residuos 512² anteriores.")
    document.optional_figure(FIDELITY,["dense_sampling_isolation"],
        "Aislamiento del muestreo temporal sobre dos integraciones 128² con 161 estados. Al reducir la separación "
        "entre salidas disminuye el error de diferenciación, mientras la proyección espacial fija un piso de residuo. "
        "La normalización y los centros temporales son comunes dentro de cada caso; el gráfico no atribuye a RK4 "
        "un defecto producido por guardar pocas instantáneas.","fig:dense-output-isolation",landscape=True)

    document.section("Comparación pareada de modelos sobre la misma observación")
    document.paragraph(f"El banco principal registra {completion['recorded_fits']} ajustes y {len(rows)} evaluaciones: "
        f"{nblocks} observaciones compartidas por diez estimadores y tres salidas físicas. "
        "Se utilizan siete referencias 128², cinco ruidos al 20 % RMS y semillas nuevas 101, 307 y 911. "
        "Los estimadores reciben sólo observación, tiempo, geometría y parámetros declarados; el limpio evalúa después. "
        "Raw, H y H+B comparten cada ajuste, por lo que sus evaluaciones no añaden entrenamientos ni nuevas realizaciones de ruido.")
    table_rows = []
    for method in METHODS:
        raw = aggregate(rows, method, "raw")
        filtered = aggregate(rows, method, "helmholtz_k10")
        table_rows.append([METHOD_LABELS[method], pct(raw["velocity"]), pct(raw["vorticity"]),
            f"{raw['accepted']}/{raw['conditions']}", pct(filtered["vorticity"]),
            f"{filtered['accepted']}/{filtered['conditions']}", str(raw["failures"] + raw["nonconverged"])])
    document.table(["Estimador", "Raw e_u", "Raw e_ω", "Raw adm.", "H+B e_ω", "H+B adm.", "Fallo / no conv."],
       table_rows, "Panel de configuraciones fijas: medianas de salidas finitas, incluidas las no convergentes identificadas, y conteos de aceptación conjunta. "
       "La distribución mezcla casos y ruidos; no constituye una clasificación universal ni sustituye las tablas estratificadas.",
       [4.0,1.6,1.8,1.65,1.9,1.75,2.0])
    document.paragraph(f"Se conservan {completion['failed_fits']} ajustes fallidos y {completion['nonconverged_fits']} ajustes no convergentes. "
        "Los fallos no se reemplazan por el mejor inicio ni se borran de los denominadores. El rango ocho es un presupuesto "
        "de los métodos modales; PCP, RPCA estable y Outlier Pursuit producen rangos efectivos por sus regularizadores. "
        "Una norma observada menor no demuestra que el modelo haya identificado mejor la señal limpia.")
    document.optional_figure(BENCHMARK, ["model_raw_overview", "heatmap_raw_vorticity", "raw_vorticity_by_noise"],
        "Comparación sin posprocesamiento físico. Los errores de vorticidad se calculan con el mismo operador para "
        "cada referencia y estimación. La estratificación revela ventajas condicionadas a contaminación y dinámica; "
        "los marcadores de fallo y no convergencia permanecen diferenciados de un error grande.", "fig:models-raw", landscape=True)
    document.subsection("Qué controla cada familia de estimadores")
    document.paragraph("POD controla compresión espacial-temporal, sin imponer una evolución autónoma. DMD y TLS-DMD "
        "ajustan una dinámica lineal reducida; TLS trata ruido en ambos bloques consecutivos. PCP separa bajo rango y "
        "entradas dispersas; Outlier Pursuit separa columnas corruptas completas. RPCA estable admite un residuo denso, "
        "con regularización heurística informada por la intensidad sintética conocida. Las cadenas PCP/TLS y RPCA estable/TLS "
        "permiten preguntar si la limpieza previa ayuda o si introduce sesgo que el ajuste dinámico amplifica. "
        "Estas funciones diferentes justifican comparar errores y recursos, sin transformar sus hipótesis en promesas de superioridad.")
    document.paragraph("La corrupción de columnas necesita un criterio específico: recuperar el subespacio no identifica por sí solo "
        "cada valor limpio oculto en un estado totalmente corrupto. Los errores en esas instantáneas siguen incluidos. "
        "Una trayectoria de velocidad más regular puede favorecer bajo rango sin representar mejor memoria de esfuerzo, "
        "viscosidad o dinámica física; por ello se examinan observables derivados por separado.")
    document.subsection("Control sin ruido: sesgo del modelo y de su regularización")
    cc=data["clean_completion"]
    document.paragraph(f"El control sin contaminación realiza {cc['recorded_fits']} ajustes deterministas y "
        f"{cc['recorded_evaluations']} evaluaciones sobre las mismas siete trayectorias, con "
        f"{cc['failed_fits']} fallos y {cc['nonconverged_fits']} ajustes no convergentes. "
        "Setenta ajustes heredan las configuraciones del banco principal. En RPCA estable, aplicar el presupuesto "
        "de ruido del 20 % a una observación sin ruido es una prueba deliberada de sesgo por regularización; "
        "no representa calibración correcta de ruido cero. Los catorce ajustes adicionales informan intensidad "
        "conocida cero para RPCA estable y su cadena TLS, sin cambiar rango, tolerancia ni límites.")
    clean_index={(r["flow"],r["method"],r["contract"],r["variant"]):r for r in data["clean_rows"]}
    inherited="inherited_20pct_regularization"
    clean_bias=[]
    for flow in ("ns_vortices","ns_multiscale","oldroyd_b","fene_p","hb_thinning","bingham","hb_thickening"):
        methods=("pcp","outlier_pursuit","tls_r8","stable_rpca")
        values=[clean_index[(flow,method,inherited,"raw")] for method in methods]
        zero=clean_index[(flow,"stable_rpca","known_zero_noise","raw")]
        clean_bias.append([FLOW_LABELS[flow],*[pct(r["vorticity_relative_l2"]) for r in values],pct(zero["vorticity_relative_l2"])])
    document.table(["Referencia", "PCP e_ω", "OP e_ω", "TLS e_ω", "RPCA estable, 20 %", "RPCA estable, 0"],clean_bias,
        "Sesgo Raw sin ruido de observación. Las dos columnas de RPCA estable separan el presupuesto heredado "
        "del control de ruido conocido cero. Se conservan salidas finitas; la figura identifica las no convergentes.",
        [3.0,2.2,2.2,2.2,2.7,2.3])
    document.paragraph("Un error presente antes de añadir ruido no puede atribuirse a la contaminación. POD y DMD "
        "introducen restricciones de representación; PCP y Outlier Pursuit pueden asignar parte de la señal física "
        "a su componente de corrupción cuando sus supuestos no describen esa trayectoria. RPCA estable introduce "
        "además sesgo por umbralización. El control de presupuesto cero distingue esa regla de regularización de "
        "la dificultad dinámica: con penalizaciones cero el objetivo tiene múltiples minimizadores L+S=M. "
        "La implementación, iniciada con componente dispersa cero, devuelve aproximadamente identity; su cadena TLS "
        "mantiene la restricción modal. La fracción de entradas dispersas puede contar redondeo aunque su norma sea "
        "despreciable; no mide ruido inyectado en este control. Estos controles explican mecanismos sin escoger "
        "parámetros por menor error limpio.")
    document.optional_figure(CLEAN,["clean_bias_vorticity"],
        "Error de vorticidad en observaciones sin ruido sobre las mismas siete referencias. Las configuraciones "
        "modales y las nucleares tienen presupuestos diferentes; el rango solicitado ocho no limita el rango de PCP "
        "u Outlier Pursuit. Los presupuestos 20 % y cero de RPCA estable permanecen etiquetados, junto con convergencia.",
        "fig:clean-model-bias",landscape=True)
    document.optional_figure(CLEAN,["regularization_controls"],
        "Control de la intensidad informada a la regularización: de 20 % heredado a ruido conocido cero, "
        "con la misma observación limpia. El contraste identifica sesgo de la regla de umbralización y de la cadena "
        "modal; no modifica el protocolo del banco contaminado ni convierte el control en una calibración PIV real.",
        "fig:regularization-contract",landscape=True)
    document.optional_figure(BENCHMARK, ["model_evolution_ns_multiscale_gaussian_iid"],
        "Errores de reconstrucción durante la misma evolución multiescala newtoniana. Comparar estados intermedios "
        "permite detectar un deterioro que la norma global o el último estado pueden ocultar. "
        "Las curvas pertenecen al intervalo ajustado y no constituyen un pronóstico independiente.",
        "fig:model-temporal-fidelity", landscape=True)

    document.section("Ablación física: qué añade el modelo sobre proyección y filtrado")
    indexes = {(r["flow"],r["noise"],r["seed"],r["method"],r["variant"]):r for r in rows}
    ablation_rows = []
    for method in METHODS:
        raw, h, hb = [aggregate(rows,method,v) for v in VARIANTS]
        delta_h, delta_b, wins = [], [], 0
        for key, r in indexes.items():
            if key[3] != method or key[4] != "raw":
                continue
            hp = indexes[key[:4]+("helmholtz",)]
            bp = indexes[key[:4]+("helmholtz_k10",)]
            control = indexes[key[:3]+("identity","helmholtz_k10")]
            if number(r.get("vorticity_relative_l2")) and number(hp.get("vorticity_relative_l2")):
                delta_h.append(hp["vorticity_relative_l2"]-r["vorticity_relative_l2"])
            if number(hp.get("vorticity_relative_l2")) and number(bp.get("vorticity_relative_l2")):
                delta_b.append(bp["vorticity_relative_l2"]-hp["vorticity_relative_l2"])
            if number(bp.get("vorticity_relative_l2")) and number(control.get("vorticity_relative_l2")):
                wins += bp["vorticity_relative_l2"] < control["vorticity_relative_l2"] - 1e-10
        ablation_rows.append([METHOD_LABELS[method], f"{raw['accepted']} → {h['accepted']} → {hb['accepted']}",
            points(statistics.median(delta_h) if delta_h else None), points(statistics.median(delta_b) if delta_b else None),
            str(wins)+"/"+str(raw["conditions"])])
    document.table(["Estimador", "Admisibles Raw→H→H+B", "Δ e_ω por H", "Δ e_ω por B", "Supera identity H+B"],
        ablation_rows, "Ablación pareada de la misma estimación: diferencias en puntos porcentuales (pp), negativas cuando favorecen la operación añadida; "
        "el último conteo usa vorticidad y tolerancia de empate de 10⁻¹⁰, sin equivaler a aceptación conjunta.",
        [4.0,3.2,2.4,2.4,2.4])
    document.paragraph("Helmholtz impone solenoidalidad y puede aumentar la admisibilidad aun cuando la vorticidad no cambia. "
        "Esa mejora informa el cumplimiento de una restricción, no una recuperación adicional de estructuras vorticales. "
        "El filtro de banda elimina gradientes de ruido y puede aumentar mucho más la fidelidad de vorticidad. "
        "Su control identity H+B establece cuánto se logra sin estimar ninguna dinámica. "
        "Las diferencias pareadas anteriores no se sustituyen por restar medianas independientes.")
    document.optional_figure(BENCHMARK, ["prior_contributions", "paired_prior_contributions"],
        "Aportes pareados de imponer divergencia y limitar banda física. Cada transición compara el mismo ajuste "
        "y la misma contaminación. El control directo identity ayuda a atribuir la ganancia al estimador, al prior espacial "
        "o a su combinación, mientras los tres estados de aceptación conservan sus denominadores.", "fig:paired-priors", landscape=True)
    document.optional_figure(BENCHMARK, ["model_filtered_overview", "heatmap_helmholtz_k10_vorticity", "filtered_vorticity_by_noise"],
        "Comparación con idéntica proyección y banda física fija. El prior espacial se aplica también a identity; "
        "por ello una mejora frente a Raw no se atribuye enteramente a DMD o RPCA. "
        "Los errores que persisten identifican límites de la combinación de observación, modelo y conocimiento físico.", "fig:models-common-prior", landscape=True)
    document.optional_figure(BENCHMARK, ["model_vs_identity"],
        "Aporte del estimador frente al control directo sobre la misma observación y con el mismo posprocesamiento. "
        "Una ventaja de vorticidad requiere superar ese control físico; aceptar una salida y mejorar una métrica son "
        "criterios diferentes. Se conserva el emparejamiento por caso, ruido y semilla.", "fig:model-control", landscape=True)
    document.subsection("Fidelidad que el prior elimina incluso con observación limpia")
    document.table(["Referencia 128²", "Corte físico", "Señal perdida e_u", "Señal perdida e_ω"],
        [[FLOW_LABELS[r["case"]],fmt(r["cutoff"]),pct(r["velocity_bias_relative_l2"]),pct(r["vorticity_bias_relative_l2"])]
         for r in data["prior"]["selected_prior_floors"]],
        "Pérdida de señal limpia por el prior k_c=10 ya fijado. La referencia original completa sigue siendo "
        "el objetivo de fidelidad: no se cambia por una verdad filtrada más fácil de recuperar.",[4.0,2.2,4.2,4.2])
    document.paragraph("El piso de vorticidad de una estimación limitada a esa banda es la norma de vorticidad "
        "limpia que queda fuera de la banda. El error restante proviene de reconstruir la parte retenida. "
        "Para identity H+B, Parseval separa exactamente pérdida de señal y ruido retenido en el error cuadrático, "
        "sin suponer independencia estadística entre ambos: sus soportes espectrales son ortogonales. "
        "Para otro estimador, la parte retenida incluye también su error de modelado y no se etiqueta únicamente como ruido.")
    document.optional_figure(PRIOR,["clean_bandwidth_bias"],
        "Pérdida de estructura física con distintos cortes espaciales. Las curvas describen la referencia limpia "
        "y el corte k_c=10 utilizado en la campaña permanece fijado. No se selecciona un nuevo corte minimizando "
        "error contra limpio; el diagnóstico muestra el costo de aceptar ese prior.","fig:clean-prior-bias",landscape=True)
    document.optional_figure(PRIOR,["identity_bias_noise_tradeoff"],
        "Descomposición del error de identity con proyección y banda fija: pérdida de señal fuera de banda "
        "y ruido proyectado que permanece dentro. La identidad sirve como control espacial del aporte del estimador. "
        "Las realizaciones son descriptivas y el error se mide contra todas las escalas de la referencia.",
        "fig:prior-noise-budget",landscape=True)
    document.optional_figure(BENCHMARK, ["model_snapshots_ns_multiscale_gaussian_iid_helmholtz_k10"],
        "Inspección espacial de las reconstrucciones multiescala con conocimiento físico común. "
        "Cada panel procede de un campo calculado; las escalas físicas y de diferencia se mantienen explícitas. "
        "El parecido visual se contrasta con los errores de campo completo y la aceptación, sin reemplazarlos.",
        "fig:model-spatial-fidelity", landscape=True)

    document.section("Ruido, referencia física y costo a fidelidad exigida")
    noise_rows=[]
    for noise,label in NOISE_LABELS.items():
        counts=[]
        for variant in VARIANTS:
            blocks=defaultdict(list)
            for r in rows:
                if r["noise"]==noise and r["variant"]==variant:
                    blocks[(r["flow"],r["seed"])].append(r)
            counts.append(sum(any(r["quality_pass"] for r in block) for block in blocks.values()))
        all_blocks=len(blocks)
        control=aggregate(rows,"identity","helmholtz_k10",noise=noise)
        noise_rows.append([label,*[str(c)+"/"+str(all_blocks) for c in counts],pct(control["vorticity"])])
    document.table(["Contaminación", "Con solución Raw", "Con solución H", "Con solución H+B", "Identity H+B e_ω"],
       noise_rows, "Cobertura por contaminación: una observación tiene solución cuando al menos una opción fija satisface "
       "convergencia y las cuatro puertas físicas. La última columna es un control fijo, no un ganador seleccionado con limpio.",
       [4.0,2.5,2.5,2.7,2.7])
    document.paragraph("RMS igual controla energía inyectada, no dificultad de las derivadas. El ruido IID concentra "
        "parte de esa energía en frecuencias altas; el ruido espacial correlacionado solapa estructuras suaves de la señal. "
        "La memoria AR puede parecer dinámica modal y la corrupción de columnas elimina información temporal completa. "
        "Las conclusiones se apoyan en estas cinco contaminaciones y en sus parámetros, no en todo sensor real posible.")
    comparisons=[]
    for noise,label in NOISE_LABELS.items():
        pcp=aggregate(rows,"pcp","raw",noise=noise)
        dmd=aggregate(rows,"dmd_r8","raw",noise=noise)
        tls=aggregate(rows,"tls_r8","raw",noise=noise)
        tls_b=aggregate(rows,"tls_r8","helmholtz_k10",noise=noise)
        control=aggregate(rows,"identity","helmholtz_k10",noise=noise)
        comparisons.append([label,pct(pcp["vorticity"]),pct(dmd["vorticity"]),pct(tls["vorticity"]),
                            pct(tls_b["vorticity"]),pct(control["vorticity"])])
    document.table(["Contaminación", "PCP Raw", "DMD Raw", "TLS Raw", "TLS H+B", "Identity H+B"],comparisons,
        "Comparación centrada en RPCA y DMD: mediana de e_ω de salidas finitas entre siete referencias y tres semillas para cada ruido. "
        "No hay elección retrospectiva de hiperparámetro o ganador; las opciones son fijas y las condiciones siguen emparejadas.",
        [3.7,2.2,2.2,2.2,2.2,2.1])
    contrast_rows=[]
    for noise,label in NOISE_LABELS.items():
        contrast=fixed_contrast(rows,"pcp","tls_r8",noise)
        contrast_rows.append([label,points(contrast["velocity"]),points(contrast["vorticity"]),
            f"{contrast['wins']}/{contrast['ties']}/{contrast['losses']}",
            f"{contrast['paired']}/21",str(contrast["nonconverged"])])
    document.table(["Contaminación", "Δ e_u PCP−TLS", "Δ e_ω PCP−TLS", "PCP gana / empata / pierde ω", "Pares finitos", "No conv. en par"],
        contrast_rows,"Contraste pareado fijo de RPCA/PCP y TLS-DMD sin prior físico, en puntos porcentuales. Valores negativos favorecen PCP. "
        "Se usan diagnósticos finitos aun si no convergen, identificados en la última columna; la aceptación se exige por separado.",
        [3.3,2.4,2.4,3.0,1.8,1.7])
    document.subsection("Del campo de velocidad a la fidelidad física")
    vp_rows=[]
    for flow in ("hb_thinning","bingham","hb_thickening"):
        for method in ("identity","tls_r8","pcp","stable_rpca"):
            sample=[r for r in rows if r["flow"]==flow and r["method"]==method and r["variant"]=="helmholtz_k10"]
            vp_rows.append([FLOW_LABELS[flow],METHOD_LABELS[method],pct(median(sample,"vorticity_relative_l2")),
                           pct(median(sample,"stress_frobenius_relative_l2")),pct(median(sample,"apparent_viscosity_relative_l2"))])
    document.table(["Fluido", "Estimador H+B", "e_ω", "e_esfuerzo", "e_viscosidad"],vp_rows,
        "Diagnóstico viscoplástico condicionado a la constitutiva conocida: medianas sobre cinco ruidos y tres semillas. "
        "La norma de esfuerzo simétrico incluye dos veces la componente cortante; no es la norma de un vector de tres componentes.",
        [3.0,4.3,2.3,2.8,2.8])
    document.paragraph("El esfuerzo y la viscosidad viscoplásticos se derivan del gradiente reconstruido y de parámetros "
        "materiales conocidos, sin copiar el esfuerzo limpio. Su error puede diferir del de vorticidad porque la ley de viscosidad "
        "es no lineal y sensible a corte pequeño. En Oldroyd-B/FENE-P el esfuerzo depende de la memoria de conformación y no "
        "queda identificado por velocidad sola; se informa como no disponible. Esas ausencias no se convierten en ceros.")
    document.paragraph("Una serie de energía próxima a la referencia no demuestra por sí sola su balance dinámico. "
        "La disipación integra gradientes y la ley constitutiva, mientras el balance relaciona esa disipación con "
        "la derivada temporal de energía. El banco informa el error de disipación cuando es identificable y mantiene "
        "separados esos diagnósticos de la aceptación operacional. La masa total con densidad fija es un control "
        "trivial; la continuidad local y la evolución del campo contienen información adicional.")
    if (BENCHMARK/"figures/constitutive_fidelity_hb_thinning_gaussian_iid.pdf").is_file():
        document.figure(BENCHMARK,"constitutive_fidelity_hb_thinning_gaussian_iid",
            "Fidelidad física adicional en HB adelgazante con ruido gaussiano IID: energía, disipación viscosa, "
            "esfuerzo de Frobenius y viscosidad aparente bajo Raw/H/H+B. Las medianas y extremos representan tres "
            "realizaciones del mismo caso. El cierre material conocido se aplica al gradiente reconstruido y no "
            "identifica parámetros ni certifica la ecuación de movimiento.","fig:constitutive-reconstruction-fidelity",landscape=True)
    document.optional_figure(BENCHMARK,["cost_fidelity"],
        "Recursos y fidelidad en una misma observación Multiescala NS, ruido gaussiano IID y semilla 101, con H+B. "
        "El costo suma ajuste y esta variante de posprocesamiento. La admisibilidad exige las cuatro puertas y convergencia, "
        "además de un error de velocidad pequeño. Las mediciones de este banco tienen una réplica por ajuste en el host "
        "declarado: describen ese entorno y no justifican diferencias finas de rendimiento ni memoria incremental exclusiva.","fig:cost-at-fidelity", landscape=True)
    resource_rows=[]
    for method in METHODS:
        r=next(r for r in data["pipeline"] if r["flow"]=="ns_multiscale" and r["noise"]=="gaussian_iid"
               and r["seed"]==101 and r["method"]==method and r["variant"]=="helmholtz_k10")
        status="Fallo" if r["status"]!="ok" else ("No converge" if not r["converged"] else ("Admisible" if r["quality_pass"] else "Fuera de puertas"))
        resource_rows.append([METHOD_LABELS[method],fmt(r.get("pipeline_cpu_seconds")),fmt(r.get("flops_proxy")),
            fmt(r["rss_peak_bytes"]/2**20) if number(r.get("rss_peak_bytes")) else "ND",status])
    document.table(["Estimador H+B", "CPU cadena, s", "Trabajo matricial proxy", "RSS ajuste, MiB", "Estado físico"],
        resource_rows,"Comparación de recursos en el mismo bloque de la figura. La CPU suma ajuste y su posprocesamiento; "
        "la RSS es pico absoluto del ajuste y el trabajo FFT se informa aparte del proxy matricial. ND indica dato no disponible.",
        [4.3,2.4,3.2,2.5,3.0])
    document.paragraph("La CPU, RSS absoluta y proxy de operaciones complementan el tiempo de pared. La memoria incluye "
        "intérprete y entrada; el proxy registra trabajo dominante aproximado, no FLOPs instrumentados ni energía eléctrica. "
        "El pico RSS del posprocesamiento hereda máximos previos del proceso, incluso de una evaluación anterior; "
        "se conserva separado de la memoria del ajuste y no se interpreta como memoria incremental de la proyección. "
        "La comparación de costo empieza por filtrar soluciones admisibles dentro del mismo bloque. Si no existe ninguna, "
        "el costo a la tolerancia queda sin solución, en vez de premiar al método más rápido que falla. "
        "El reanálisis histórico conserva tres mediciones por condición y permite revisar variabilidad de costo sin "
        "aumentar artificialmente el número de observaciones de calidad.")

    document.section("Sensibilidad y límites de la comparación")
    sc=data["sensitivity_completion"]
    document.paragraph(f"El estudio de sensibilidad registra {sc['fits']} ajustes y {sc['evaluations']} salidas físicas, "
        f"con {sc['failed_fits']} fallos y {sc['nonconverged_fits']} ajustes no convergentes. "
        "Cambia rango e intensidad sobre Multiescala NS y Oldroyd-B en 128². "
        "Cruza r=2,4,8,12,16; intensidades RMS 5 %, 20 % y 40 %; ruido gaussiano e impulsivo; POD y TLS-DMD; "
        "y las mismas tres semillas nuevas. El diseño fija los valores antes de ejecutar. Es un análisis exploratorio "
        "de configuraciones, sin escoger rango contra limpio para transferirlo a otra medición. "
        "Los puntos r=8 y 20 % coincidentes con el banco principal no se cuentan como evidencia independiente.")
    sensitivity_pass=sc["accepted_by_postprocessing"]
    document.paragraph(f"La aceptación operacional es {sensitivity_pass['raw']}/{sc['fits']} con Raw, "
        f"{sensitivity_pass['helmholtz']}/{sc['fits']} con H y {sensitivity_pass['helmholtz_k10']}/{sc['fits']} con H+B. "
        "Estos conteos cruzan sólo POD y TLS-DMD, dos flujos y los valores de rango/intensidad declarados. "
        "La interpretación del aporte del filtro requiere el control identity del banco principal, que comparte el mismo prior.")
    oracle_by_flow=defaultdict(list)
    for row in data["oracle_floor"]:
        oracle_by_flow[row["flow"]].append(row)
    floor_rows=[]
    for flow, sample in oracle_by_flow.items():
        sample.sort(key=lambda r:int(r["rank"]))
        for rank in (2,8,16):
            r=next(row for row in sample if int(row["rank"])==rank)
            floor_rows.append([FLOW_LABELS.get(flow,flow),str(rank),
                pct(float(r["velocity_rank_frobenius_lower_bound"])),
                pct(float(r["vorticity_clean_pod_diagnostic"])),pct(float(r["clean_band_vorticity_error"]))])
    document.table(["Referencia", "Rango", "POD limpio e_u", "POD limpio e_ω", "Banda limpia e_ω"],floor_rows,
        "Diagnóstico del sesgo de compresión y filtrado. La SVD limpia minimiza Frobenius de velocidad a rango fijado; "
        "su error de vorticidad es diagnóstico y no un óptimo universal. La banda física elimina también señal real.",
        [3.3,1.3,3.0,3.0,3.8])
    document.paragraph("El filtro no es gratuito: la pérdida de vorticidad limpia depende del caso, aunque se mantenga k_c=10. "
        "La menor pérdida de una referencia frente a otra puede facilitar su aceptación con la misma puerta del 25 %. "
        "POD de mayor rango aproxima mejor la velocidad limpia, pero en datos ruidosos puede retener más gradientes de contaminación. "
        "La curva de error observada, y no el rango óptimo del limpio, describe ese intercambio entre compresión y ruido.")
    document.optional_figure(SENSITIVITY,["rank_vorticity", "rank_noise_fidelity", "rank_intensity_fidelity", "rank_noise_sensitivity"],
        "Sensibilidad del error al rango y a la intensidad. Las bandas muestran mínimo y máximo de tres realizaciones "
        "de observación, no intervalos de confianza poblacionales. Aumentar rango puede recuperar señal omitida y también "
        "incorporar ruido; el comportamiento debe leerse junto con la vorticidad y el prior aplicado.","fig:rank-intensity", landscape=True)
    sensitivity_examples=[]
    for flow,noise,intensity,rank,method in (("ns_multiscale","gaussian_iid",.2,8,"pod_r8"),
                                           ("ns_multiscale","gaussian_iid",.2,16,"pod_r8"),
                                           ("oldroyd_b","sparse_impulsive",.4,16,"pod_r8"),
                                           ("oldroyd_b","sparse_impulsive",.4,16,"tls_r8")):
        selection=[r for r in data["sensitivity_aggregates"] if r["flow"]==flow and r["noise"]==noise
             and r["intensity_relative_rms"]==intensity and r["rank"]==rank and r["method"]==method]
        raw=next(r for r in selection if r["postprocessing"]=="raw")
        hb=next(r for r in selection if r["postprocessing"]=="helmholtz_k10")
        require(raw["sample_count"]==hb["sample_count"]==3, "Sensitivity example lacks three seeds")
        sensitivity_examples.append([FLOW_LABELS[flow]+", "+NOISE_LABELS[noise]+", RMS "+pct(intensity),
            "POD" if method=="pod_r8" else "TLS-DMD",str(rank),
            pct(raw["vorticity_relative_l2"]["median"]),pct(hb["velocity_relative_l2"]["median"]),
            pct(hb["vorticity_relative_l2"]["median"])])
    document.table(["Referencia y observación", "Modelo", "Rango", "Raw e_ω", "H+B e_u", "H+B e_ω"],
        sensitivity_examples,"Ejemplos descriptivos de sensibilidad, medianas de tres semillas: aumentar rango puede "
        "aumentar vorticidad de ruido, y dos estimadores pueden intercambiar ventaja en velocidad y gradientes.",
        [4.5,2.4,1.2,2.1,2.1,2.1])
    document.optional_figure(SENSITIVITY,["clean_oracle_floor", "oracle_floor", "oracle_bias_floor", "oracle_rank_floor"],
        "Piso diagnóstico de compresión y filtrado del campo limpio. Este cálculo usa referencia sólo después del "
        "experimento para explicar sesgo de rango o banda; no interviene en entrenamiento ni selección. "
        "No es un método utilizable cuando la señal verdadera es desconocida.","fig:diagnostic-oracle", landscape=True)
    document.subsection("Qué conserva el reanálisis de las mallas pequeñas")
    matched=[r for r in data["fidelity"]["credibility_matrix"]
             if r.get("campaign") != "visual_highres" and r.get("observable") in ("velocity","vorticity","stress_frobenius")]
    if matched:
        matched_groups=defaultdict(dict)
        minimum_grids={case:min(r["grid"] for r in matched if r["case"]==case)
                       for case in {r["case"] for r in matched}}
        for r in matched:
            if r["grid"]!=minimum_grids[r["case"]]:
                continue
            key=(r["case"],r["grid"],r["reference_grid"],r["end_time"])
            matched_groups[key][r["observable"]]=r["maximum_relative_l2"]
        document.table(["Referencia histórica", "Mallas", "T final", "Máx. e_u", "Máx. e_ω", "Máx. e_τ"],
            [[OLD_FLOW_LABELS.get(case,case),str(n)+"² → "+str(nf)+"²",fmt(tf),pct(v.get("velocity")),pct(v.get("vorticity")),pct(v.get("stress_frobenius"))]
             for (case,n,nf,tf),v in matched_groups.items()],
            "Controles adicionales con parámetros físicos exactamente iguales a los del banco histórico. "
            "Taylor–Green usa solución analítica en la misma malla. El refinamiento NS incluye malla y paso nominal; "
            "las campañas visuales usan otras intensidades físicas y no reemplazan estas verificaciones.",
            [4.2,2.5,1.3,2.1,2.1,2.4])
        document.optional_figure(FIDELITY,["matching_original_references"],
            "Calidad de las referencias históricas bajo su propia configuración física. "
            "Cambiar viscosidad, condición inicial o intervalo invalidaría atribuir el error al refinamiento de esa campaña. "
            "Los esfuerzos auxiliares juzgan la simulación aunque el estimador histórico sólo reconstruya velocidad.",
            "fig:matched-original-references",landscape=True)
    pc=data["paired_completion"]
    document.paragraph(f"El reanálisis descriptivo conserva {pc['conditions']} condiciones, {pc['resource_measurements']} mediciones "
        f"de costo, {pc['nonconverged_conditions']} condiciones no convergentes y {pc['fallback_conditions']} uso de alternativa fbDMD→TLS. "
        f"En {pc['blocks_without_admissible_method']} de {pc['paired_blocks']} observaciones no existe estimador admisible. "
        "Su rango cuatro, nueve ruidos y referencias de menor malla difieren del banco principal: no se combinan en "
        "una sola tasa ni se atribuye a la resolución el cambio entre configuraciones distintas.")
    velocity_without_vorticity=sum(r["pass_velocity_relative_l2"] and not r["pass_vorticity_relative_l2"]
                                  for r in data["old_conditions"])
    document.paragraph(f"En {velocity_without_vorticity} salidas históricas se cumple e_u≤15 % pero se incumple e_ω≤25 %. "
        "Ese desacuerdo cuantifica por qué el error de velocidad aislado no basta para declarar fidelidad del flujo. "
        "Las salidas con divergencia impuesta tampoco certifican la ecuación de movimiento: un residuo de vorticidad "
        "medido entre instantáneas conserva error de muestreo y exige interpretación adicional.")
    document.optional_figure(PAIRED,["acceptance_by_noise"],
        "Cobertura del banco anterior en nueve contaminaciones. Cada celda cuenta 24 condiciones de calidad "
        "(ocho referencias y tres semillas); las tres mediciones de costo no aumentan ese número. "
        "Los parámetros y mallas difieren del banco principal, de modo que sus porcentajes se mantienen separados.",
        "fig:historical-noise-coverage", landscape=True)
    old_table=[]
    for ab in data["old_ablations"]:
        old_table.append([OLD_METHOD_LABELS[ab["before"]]+" → "+OLD_METHOD_LABELS[ab["after"]],
             str(ab["before_admissible"])+" → "+str(ab["after_admissible"]),
             str(ab["rescued"]),str(ab["lost"]),str(ab["vorticity_equal_count"])+"/"+str(ab["conditions"])])
    document.table(["Transición histórica", "Admisibles", "Rescata", "Pierde", "ω sin cambio"],old_table,
        "Ablaciones del banco amplio anterior con parámetros propios. La proyección H y el corte 2/3 "
        "no se equiparan al corte físico k_c=10 del banco nuevo.",[5.5,2.5,2.0,2.0,2.4])
    document.optional_figure(PAIRED,["tolerance_sensitivity_matrix", "tolerance_sensitivity"],
        "Sensibilidad histórica a la tolerancia de vorticidad, manteniendo las demás puertas. "
        "Un conteo de aceptación depende del umbral declarado y debe acompañarse de errores continuos. "
        "Este análisis posterior no redefine las puertas con que se evaluó el banco principal.","fig:tolerance-sensitivity", landscape=True)
    document.paragraph("Las tres semillas exploran observaciones del mismo flujo; no estiman variabilidad entre condiciones "
        "iniciales, regímenes ni laboratorios. Las referencias son periódicas bidimensionales y la campaña reconstruye "
        "dentro de la ventana. No demuestra identificación de material, esfuerzo viscoelástico recuperado, pronóstico "
        "futuro, calle de vórtices externa ni generalización PIV. Los residuos de salida, los controles de resolución "
        "y los negativos anteriores permanecen separados y auditables.")
    document.subsection("Acceso a la evidencia y lectura complementaria")
    document.paragraph("Las tablas de condición, resúmenes estratificados, ledger, hashes y figuras acompañan el capítulo. "
        "Los apéndices conservan el catálogo de evolución, el pronóstico cronológico y la tabla amplia histórica. "
        "La reproducción documental utiliza JSON/CSV y figuras publicadas; las trayectorias completas se regeneran "
        "con los scripts y se archivan por separado cuando exceden el límite por archivo de GitHub.")
    document.markdown += ["Fuentes verificadas:", ""] + ["- ["+path+"]("+"../../"+path+")" for path in sorted(evidence.sources)] + [""]
    return document


def executive(data):
    rows=data["rows"]; completion=data["benchmark_completion"]
    blocks=len({(r["flow"],r["noise"],r["seed"]) for r in rows})
    cover={}
    for variant in VARIANTS:
        grouped=defaultdict(list)
        for r in rows:
            if r["variant"]==variant:grouped[(r["flow"],r["noise"],r["seed"])].append(r)
        cover[variant]=sum(any(r["quality_pass"] for r in group) for group in grouped.values())
    multiscale_floor=next(r["vorticity_bias_relative_l2"] for r in data["prior"]["selected_prior_floors"] if r["case"]=="ns_multiscale")
    text=["% Generated; evidence-dependent academic executive summary.",
      r"La ampliación compara modelos de reconstrucción mediante fidelidad física, después de examinar la calidad numérica de las referencias. "
      r"La pregunta exige una misma observación, errores de velocidad y "
      r"vorticidad, convergencia y costo a la tolerancia. El documento conserva cinco capítulos y desplaza el catálogo de evolución a los apéndices.", "",
      tex(f"El banco principal incluye {completion['recorded_fits']} ajustes sobre siete referencias 128², cinco ruidos y tres semillas nuevas; "
          f"sus {len(rows)} evaluaciones Raw/H/H+B comparten esos ajustes. El reanálisis previo conserva 2808 condiciones y 8424 mediciones "
          "de costo. Los controles físicos alcanzan 512² en Navier–Stokes, 256² en viscoelasticidad y 256² en un intervalo viscoplástico corto."), "",
      r"\begin{enumerate}",
      r"\item \textbf{Fidelidad de simulación.} La resolución se juzga por observable y horizonte. Oldroyd-B puede tener velocidad "
      r"casi invariable entre mallas y esfuerzo aún sensible; el refinamiento corto HB no valida automáticamente su evolución larga.",
      r"\item \textbf{Comparación del modelo.} Raw compara estimadores sin conocimiento físico añadido. Las diferencias se estratifican "
      r"por contaminación y referencia, y los fallos permanecen en los denominadores.",
      r"\item \textbf{Aporte de restricciones.} H impone solenoidalidad; H+B limita banda física. Identity recibe las mismas "
      r"transformaciones para distinguir qué mejora procede del estimador y qué mejora procede del prior.",
      r"\item \textbf{Costo útil.} CPU, RSS y trabajo dominante complementan el tiempo. La frontera de recursos sólo tiene "
      r"sentido una vez exigida fidelidad; una observación sin método admisible conserva ese resultado negativo.",
      r"\end{enumerate}", "",
      tex(f"La cobertura del banco, cuando basta alguna de las diez opciones fijas, es {cover['raw']}/{blocks} con Raw, "
          f"{cover['helmholtz']}/{blocks} con H y {cover['helmholtz_k10']}/{blocks} con H+B. "
          f"Se registran {completion['failed_fits']} fallos y {completion['nonconverged_fits']} ajustes no convergentes. "
          "Estos conteos describen el banco y las cuatro puertas predeclaradas, no probabilidades poblacionales."), "",
      tex(f"El prior tiene un costo físico medible: en Multiescala NS elimina {pct(multiscale_floor)} de vorticidad limpia "
          "en norma relativa. Cualquier reconstrucción limitada a esa banda conserva al menos ese error frente a la "
          "referencia completa, incluso sin ruido. Aceptar una salida con puerta del 25 % no equivale a fidelidad del 10 %."), "",
      r"\textbf{Guía de lectura.} El capítulo 2 define supuestos y observables; el 3 fija unidades de comparación y controles; "
      r"el 4 responde las cuatro preguntas anteriores y examina sensibilidad. Los apéndices permiten inspeccionar evoluciones, "
      r"resolución y resultados históricos completos sin diluir la comparación central.", "",
      r"El error de reconstrucción se evalúa contra la solución que generó la observación. Las diferencias contra una malla fina "
      r"acotan qué interpretación física puede darse a ese error, pero no convierten una solución numérica en verdad experimental. "
      r"El campo limpio no selecciona hiperparámetros; las sensibilidades y los pisos de sesgo se interpretan después del ajuste."]
    return "\n".join(text)+"\n",cover,blocks


def conclusions(data,cover,blocks):
    c=data["benchmark_completion"]
    floor=next(r["vorticity_bias_relative_l2"] for r in data["prior"]["selected_prior_floors"] if r["case"]=="ns_multiscale")
    impulse=fixed_contrast(data["rows"],"pcp","tls_r8","sparse_impulsive")
    gaussian=fixed_contrast(data["rows"],"pcp","tls_r8","gaussian_iid")
    spatial=fixed_contrast(data["rows"],"pcp","tls_r8","spatial_gaussian")
    return "\n".join(["% Generated from completed comparison evidence.",r"\section{Conclusiones de comparación y fidelidad}",
       tex(f"El banco de {c['recorded_fits']} ajustes conserva diferencias entre estimadores antes y después de añadir restricciones. "
           f"En {blocks-cover['raw']} de {blocks} observaciones no existe solución admisible Raw; "
           f"con H+B quedan {blocks-cover['helmholtz_k10']} observaciones sin solución. "
           "El resultado se interpreta como cobertura de configuraciones fijas bajo los umbrales, no como selección "
           "automática de un modelo en una medición cuyo limpio no está disponible."),"",
       tex(f"El contraste Raw fijo PCP−TLS produce una mediana de diferencia de e_ω de {points(impulse['vorticity'])} "
           f"bajo contaminación impulsiva y {points(gaussian['vorticity'])} bajo gaussiano IID. El signo negativo favorece PCP; "
           "los diagnósticos finitos incluyen los estados de convergencia documentados y la aceptación se comprueba aparte. "
           "Esta comparación conserva la misma observación por par y no transforma una ventaja condicionada en superioridad universal."),"",
       tex(f"El criterio modifica la preferencia: con gaussiano IID, la diferencia de e_u PCP−TLS es "
           f"{points(gaussian['velocity'])}, aun cuando PCP tiene menor e_ω. Con ruido espacial correlacionado "
           f"la diferencia de e_ω cambia a {points(spatial['vorticity'])}. Un error menor que el de otro estimador "
           "puede seguir fuera de tolerancia; los conteos de cobertura mantienen esa distinción."),"",
       r"Los contrastes pareados y el control identity con el mismo prior impiden atribuir toda limpieza espacial "
       r"al modelo dinámico. Un error bajo en velocidad puede coexistir con vorticidad o cierres constitutivos degradados. "
       r"En fluidos de memoria, el esfuerzo no se identifica mediante reconstrucción de velocidad sola; en viscoplasticidad "
       r"se evalúa bajo la hipótesis explícita de ley material conocida.","",
       tex(f"El filtro k_c=10 impone un piso de {pct(floor)} de error relativo de vorticidad en Multiescala NS "
           "contra su referencia completa. Por ortogonalidad espectral, una salida limitada a esa banda no puede "
           "alcanzar la tolerancia del 10 %, incluso con observación perfecta. Aumentar aceptación con una puerta "
           "del 25 % requiere reconocer esa pérdida de señal, además del ruido que se elimina."),"",
       r"La suficiencia de simulación depende del observable, la malla y el intervalo. Las referencias finas del mismo solver "
       r"aportan convergencia numérica, mientras las soluciones analíticas y controles temporales examinan causas diferentes. "
       r"Los errores entre mallas no acreditan validación experimental, y el control viscoplástico corto conserva su horizonte.","",
       r"La sensibilidad a rango, intensidad y tolerancia explica límites de las configuraciones; no utiliza el limpio para "
       r"escoger el mejor hiperparámetro desplegable. La contribución comprobable es una comparación trazable de observación, "
       r"estimador, prior físico, referencia y recursos, con resultados negativos preservados.",""])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only",action="store_true")
    args=parser.parse_args()
    data=checked_inputs()
    document=make_document(data)
    executive_tex,cover,blocks=executive(data)
    conclusion_tex=conclusions(data,cover,blocks)
    manifest=dict(complete=True,script="python/scripts/build_comparison_chapter.py",script_sha256=sha(Path(__file__)),
         sources_sha256=data["evidence"].sources,declared_sources_verified=sorted(set(data["evidence"].verified)),
         unavailable_separately_archived_npz=sorted(set(data["evidence"].unavailable_archives)),
         main_comparative_figures=document.figure_count,fit_count=data["benchmark_completion"]["recorded_fits"],
         evaluated_outputs=len(data["rows"]),paired_observation_blocks=blocks,coverage_by_variant=cover,
         scope="Five numbered chapters; full historical evidence and evolution retained in appendices; no fits or parameter selection in renderer")
    if args.check_only:
        print(json.dumps({"verified":True,"sources":len(data["evidence"].sources),"figures":document.figure_count,
                          "fit_count":manifest["fit_count"],"coverage":cover},ensure_ascii=False,indent=2))
        return
    outputs={"COMPARISON_RESULTS.md":"\n".join(document.markdown),
             "comparison_generated.tex":"\n".join(document.latex),
             "executive_generated.tex":executive_tex,
             "comparison_conclusions_generated.tex":conclusion_tex,
             "comparison_document_manifest.json":json.dumps(manifest,ensure_ascii=False,indent=2)+"\n"}
    for name,content in outputs.items():
        temporary=DOCS/(name+".tmp")
        temporary.write_text(content)
        temporary.replace(DOCS/name)
    print(json.dumps({"generated":list(outputs),"sources":len(data["evidence"].sources),
                      "figures":document.figure_count,"coverage":cover},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
