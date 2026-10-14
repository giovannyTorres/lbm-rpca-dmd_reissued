#!/usr/bin/env python3
"""Reference credibility by observable, matched physics, horizon and estimator.

This independent analysis never changes scientific solver modules. It audits
existing visual campaigns and, separately, integrates finer references with
the exact physical configuration of the earlier extended_physics benchmark.
Run --matched-only first to inspect integration cost; cached integrations are
reused only when configuration, scientific sources and archives agree.
"""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python/src"))
import numpy as np
from scipy.integrate import cumulative_trapezoid, cumulative_simpson
from scipy.signal import resample
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fluid_denoise.research.dns import simulate_periodic_ns, taylor_green_reference
from fluid_denoise.research.viscoelastic_dns import simulate_viscoelastic_periodic
from fluid_denoise.research.viscoplastic_dns import simulate_viscoplastic_periodic

OUT = ROOT / "results/research/reference_fidelity_2026-10-14"
RESULTS = ROOT / "results/research"
OLD = RESULTS / "extended_physics_2026-10-14"
NS = RESULTS / "highres_newtonian_2026-10-14"
VE = RESULTS / "highres_viscoelastic_2026-10-14"
VP = RESULTS / "highres_viscoplastic_2026-10-14"
plt.rcParams.update({"font.size": 12, "axes.labelsize": 12, "axes.titlesize": 13,
                     "legend.fontsize": 12, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 180})


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def scientific_hashes():
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / "python/src/fluid_denoise/research").glob("*.py"))}


def transfer(field, n):
    return resample(resample(field, n, axis=-2), n, axis=-1)


def waves(n, length=2 * np.pi):
    modes = 2 * np.pi * np.fft.fftfreq(n, d=length / n)
    # Real first derivatives use a zero Nyquist. Solver states exclude it.
    if n % 2 == 0:
        modes[n // 2] = 0
    return np.meshgrid(modes, modes)


def vorticity(velocity):
    kx, ky = waves(velocity.shape[-1])
    return np.fft.ifft2(1j * kx * np.fft.fft2(velocity[..., 1, :, :])
                        - 1j * ky * np.fft.fft2(velocity[..., 0, :, :])).real


def l2(value):
    return float(np.sqrt(np.sum(np.asarray(value, dtype=float) ** 2)))


def ratios(a, b):
    denominator = l2(b)
    return None if denominator == 0 else l2(a - b) / denominator


def array_data(path):
    with np.load(path) as saved:
        data = {key: saved[key] for key in saved.files}
    if not all(np.isfinite(value).all() for value in data.values()):
        raise ValueError(f"Nonfinite field: {path}")
    if not np.all(np.diff(data["times"]) > 0):
        raise ValueError(f"Invalid times: {path}")
    return data


def matched_integrations(only_solvers=None, only_cases=None, only_grids=None, vp128_budget=1200.):
    """Generate new runs using earlier manifests; no visual-physics substitution."""
    (OUT / "datasets").mkdir(parents=True, exist_ok=True)
    hashes = scientific_hashes()
    manifests = read(OLD / "dataset_manifests.json")
    records = []
    omissions=[]
    # A first 64² viscoplastic pilot records actual cost before the finer runs.
    ordered = sorted(manifests, key=lambda m: (m["case"]["solver"] != "viscoplastic_periodic", m["case"]["id"]))
    for manifest in ordered:
        case = manifest["case"]
        solver = case["solver"]
        if only_solvers and solver not in only_solvers:
            continue
        if only_cases and case["id"] not in only_cases:
            continue
        if case["id"] == "taylor_green_ns":
            continue  # Exact reference is evaluated without an unnecessary new PDE run.
        grids = (64, 128) if solver == "viscoplastic_periodic" else ((64, 128, 256) if solver == "viscoelastic_periodic" else (128, 256))
        for grid in grids:
            if only_grids and grid not in only_grids:
                continue
            variants = (.0025, .01) if solver == "spectral_ns" and grid == 128 else (None,)
            for variant in variants:
                config = dict(case["parameters"])
                config["nx"] = grid
                if solver == "spectral_ns":
                    duration = config["dt"] * config["n_steps"]
                    spacing = config["dt"] * config["sample_every"]
                    config["dt"] = .0025 if variant is None else variant
                    config["n_steps"] = int(round(duration / config["dt"]))
                    config["sample_every"] = int(round(spacing / config["dt"]))
                tag = f"matched_{case['id']}_{grid}_dt{config['dt']:g}"
                archive = OUT / "datasets" / (tag + ".npz")
                meta_path = archive.with_suffix(".json")
                if solver=="viscoplastic_periodic" and grid==128 and not archive.exists():
                    pilot_path=OUT/"datasets"/f"matched_{case['id']}_64_dt{config['dt']:g}.json"
                    if pilot_path.exists():
                        predicted=16*read(pilot_path)["wall_seconds"]
                        if predicted>vp128_budget:
                            omission={"case_id":case["id"],"grid":128,"parameters":config,
                                      "status":"not_executed_cost_budget", "predicted_wall_seconds":predicted,
                                      "budget_seconds":vp128_budget,
                                      "prediction_method":"64-to128: about4 times more spatial nodes and4 times more explicit diffusion-limited steps; shared-host timing is uncertain",
                                      "qualification":"64 reference remains numerical and is not independently validated at128 over the full horizon"}
                            omissions.append(omission)
                            print("OMIT MATCHED",tag,"predicted_seconds",round(predicted),"budget",vp128_budget,flush=True)
                            continue
                key = {"case_id": case["id"], "solver": solver, "parameters": config,
                       "old_manifest_sha256": sha(OLD / "dataset_manifests.json"),
                       "old_archive_sha256": sha(OLD / "datasets" / f"{case['id']}_clean.npz"),
                       "scientific_source_sha256": hashes}
                if archive.exists() and meta_path.exists():
                    meta = read(meta_path)
                    if meta["cache_key"] != key or sha(archive) != meta["archive_sha256"]:
                        raise ValueError(f"Refusing stale matched reference: {tag}")
                    print("REUSED", tag, flush=True)
                else:
                    print("RUN MATCHED", tag, flush=True)
                    wall, cpu = time.perf_counter(), time.process_time()
                    routine = {"spectral_ns": simulate_periodic_ns,
                               "viscoelastic_periodic": simulate_viscoelastic_periodic,
                               "viscoplastic_periodic": simulate_viscoplastic_periodic}[solver]
                    data = routine(**config)
                    elapsed, processor = time.perf_counter() - wall, time.process_time() - cpu
                    if not all(np.isfinite(v).all() for v in data.values() if isinstance(v, np.ndarray)):
                        raise FloatingPointError(f"Nonfinite matched reference: {tag}")
                    np.savez_compressed(archive, **{k: v for k, v in data.items() if isinstance(v, np.ndarray)})
                    meta = {"complete": True, "cache_key": key, "metadata": data["metadata"],
                            "dx": data["dx"], "dy": data["dy"], "shape": list(data["velocity"].shape),
                            "wall_seconds": elapsed, "cpu_seconds": processor,
                            "archive_sha256": sha(archive),
                            "scope": "same physical parameters, initial condition and saved times as extended_physics; new independent fine-grid integration"}
                    write(meta_path, meta)
                    print("DONE MATCHED", tag, "seconds", round(elapsed, 2), "steps", meta["metadata"]["internal_steps"], flush=True)
                records.append({"archive": str(archive.relative_to(ROOT)), **meta})
    if only_solvers or only_cases or only_grids:
        records=[]
        for path in sorted((OUT/"datasets").glob("matched_*.json")):
            meta=read(path)
            records.append({"archive":str(path.with_suffix(".npz").relative_to(ROOT)),**meta})
    previous_omissions=read(OUT/"omitted_matched_controls.json") if (OUT/"omitted_matched_controls.json").exists() else []
    omissions=list({(x["case_id"],x["grid"]):x for x in previous_omissions+omissions}.values())
    available={(x["cache_key"]["case_id"],x["cache_key"]["parameters"]["nx"]) for x in records}
    omissions=[x for x in omissions if (x["case_id"],x["grid"]) not in available]
    write(OUT/"omitted_matched_controls.json",omissions)
    write(OUT / "matched_integrations.json", {"complete": len(records)+len(omissions)==18, "runs": records,
          "count": len(records), "planned_count":18,"omitted_controls":omissions,"scientific_source_sha256": hashes,
          "recorded_cpu_seconds_sum": sum(x["cpu_seconds"] for x in records),
          "cost_qualification": "shared host; integration CPU/wall only, no fair solver-speed ranking"})
    return records


def field_views(data):
    values = {"velocity": data["velocity"], "vorticity": vorticity(data["velocity"])}
    if "conformation" in data:
        values["conformation"] = data["conformation"]
        values["excess_conformation"] = data["conformation"] - np.eye(3)[None, :, :, None, None]
        values["stress_frobenius"] = data["stress"]  # Full 3×3 tensor contains both off-diagonals.
    if "deviatoric_stress" in data:
        values["stress_frobenius"] = data["deviatoric_stress"] * np.sqrt([1., 2., 1.])[None, :, None, None]
        values["apparent_viscosity"] = data["apparent_viscosity"]
        values["shear_rate"] = data["shear_rate"]
    return values


def compare_arrays(coarse, fine, family, case, campaign, reference_kind="numerical_same_solver"):
    np.testing.assert_allclose(coarse["times"], fine["times"], rtol=0, atol=2e-14)
    coarse_fields, fine_fields = field_views(coarse), field_views(fine)
    rows = []
    for observable in coarse_fields:
        a, b = coarse_fields[observable], fine_fields[observable]
        expanded = transfer(a, b.shape[-1])
        errors = [ratios(x, y) for x, y in zip(expanded, b)]
        defined = [(i, v) for i, v in enumerate(errors) if v is not None]
        i, maximum = max(defined, key=lambda pair: pair[1])
        rows.append({"family": family, "case": case, "campaign": campaign,
                     "observable": observable, "grid": a.shape[-1], "reference_grid": b.shape[-1],
                     "reference_kind": reference_kind, "start_time": float(fine["times"][0]),
                     "end_time": float(fine["times"][-1]), "saved_times": len(fine["times"]),
                     "maximum_relative_l2": maximum, "maximum_time": float(fine["times"][i]),
                     "final_relative_l2": errors[-1], "all_times_relative_l2": ratios(expanded, b),
                     "times": fine["times"].tolist(), "per_time_relative_l2": errors,
                     "relative_error_units": "dimensionless; multiply by 100 for percent",
                     "scope": "full coarse field Fourier-prolonged to all fine-grid modes, equal saved times; maximum among saved frames, not continuous-time bound",
                     "caveat": "same-solver difference does not bound exact continuum error; density=1 is imposed"})
    return rows


def credibility_matrix(include_matched=True):
    """Keep numerical error, material identity and integration horizon together."""
    rows = []
    with (NS / "comparison_timeseries.csv").open() as stream:
        ns_curves = list(csv.DictReader(stream))
    ns_summary = read(NS / "summary.json")
    for item in ns_summary["grid_refinement"]:
        curve = [x for x in ns_curves if x["case"] == item["case"] and x["comparison"] == "spatial" and int(x["nx"]) == item["nx"]]
        times = [float(x["time"]) for x in curve]
        for observable in ("velocity", "vorticity"):
            errors = [float(x[f"{observable}_full_field_relative_l2"]) for x in curve]
            if not np.isclose(max(errors), item[f"maximum_{observable}_full_field_error"], rtol=1e-12):
                raise ValueError("NS summary and saved curve disagree")
            rows.append({"family": "newtonian", "case": item["case"], "campaign": "visual_highres",
                         "observable": observable, "grid": item["nx"], "reference_grid": 512,
                         "reference_kind": "numerical_same_solver", "start_time": times[0], "end_time": times[-1],
                         "saved_times": len(times), "maximum_relative_l2": max(errors), "maximum_time": times[int(np.argmax(errors))],
                         "final_relative_l2": errors[-1], "all_times_relative_l2": None,
                         "times": times, "per_time_relative_l2": errors,
                         "caveat": "nu=.01, initial U_rms=.8, seed17; not a validation of the older nu=.02/U_rms=1/seed19 benchmark"})
    ve_summary = read(VE / "summary.json")
    for model, records in ve_summary["spatial_refinement"].items():
        for item in records:
            for observable, key in (("velocity", "velocity"), ("vorticity", "vorticity"),
                                    ("conformation", "conformation"), ("excess_conformation", "excess_conformation"),
                                    ("stress_frobenius", "stress")):
                errors = item[f"{key}_full_field_relative_l2"]
                defined = [(i, x) for i, x in enumerate(errors) if x is not None]
                i, maximum = max(defined, key=lambda pair: pair[1])
                rows.append({"family": "viscoelastic", "case": model, "campaign": "visual_highres",
                             "observable": observable, "grid": item["nx"], "reference_grid": 256,
                             "reference_kind": "numerical_same_solver", "start_time": item["times"][0],
                             "end_time": item["times"][-1], "saved_times": len(item["times"]),
                             "maximum_relative_l2": maximum, "maximum_time": item["times"][i],
                             "final_relative_l2": errors[-1], "all_times_relative_l2": None,
                             "times": item["times"], "per_time_relative_l2": errors,
                             "caveat": "nu_s=.02,nu_p=.04,lambda=.5,U_rms=.8; older benchmark uses U_rms=.6,T=.6; zero initial stress makes relative stress undefined at t=0"})
    stress_checks = {(x["run"], x["reference"]): x for x in read(VP / "derived_stress_frobenius.json")["comparisons"]}
    for item in read(VP / "comparisons.json"):
        if "spatial" not in item["scope"]:
            continue
        model = item["run"].split("_64")[0].split("_128")[0]
        for observable, error in item["errors"].items():
            if observable == "deviatoric_stress":
                observable = "stress_frobenius"
                error = stress_checks[item["run"], item["reference"]]
            errors = error["full_field_per_time_relative_l2"]
            rows.append({"family": "viscoplastic", "case": model, "campaign": "visual_highres",
                         "observable": observable, "grid": 64 if "_64" in item["run"] else 128,
                         "reference_grid": 128 if "long_horizon" in item["scope"] else 256,
                         "reference_kind": "numerical_same_solver", "start_time": item["times"][0],
                         "end_time": item["times"][-1], "saved_times": len(item["times"]),
                         "maximum_relative_l2": max(errors), "maximum_time": item["times"][int(np.argmax(errors))],
                         "final_relative_l2": errors[-1], "all_times_relative_l2": error["full_field_all_times_relative_l2"],
                         "times": item["times"], "per_time_relative_l2": errors,
                         "caveat": "matched eta_app=.04 at gamma_ref=1.95883, not identical viscosity everywhere; finest256 only t=0,.1, does not validate T=2 reference128"})
    if not include_matched:
        return rows
    # Integrations matching the earlier benchmark use its exact manifest configuration.
    matched = read(OUT / "matched_integrations.json")
    manifests = read(OLD / "dataset_manifests.json")
    for manifest in manifests:
        case = manifest["case"]
        coarse = array_data(OLD / "datasets" / f"{case['id']}_clean.npz")
        if case["id"] == "taylor_green_ns":
            fine = taylor_green_reference(nx=24, times=coarse["times"], viscosity=.05, amplitude=1.)
            rows.extend(compare_arrays(coarse, fine, "newtonian", case["id"], "matched_original_benchmark", "analytical_exact_single_mode"))
            continue
        records = [x for x in matched["runs"] if x["cache_key"]["case_id"] == case["id"]]
        largest = max(records, key=lambda x: x["cache_key"]["parameters"]["nx"])
        fine = array_data(ROOT / largest["archive"])
        family = {"spectral_ns": "newtonian", "viscoelastic_periodic": "viscoelastic", "viscoplastic_periodic": "viscoplastic"}[case["solver"]]
        original_comparisons=compare_arrays(coarse, fine, family, case["id"], "matched_original_benchmark")
        for comparison in original_comparisons:
            comparison.update(coarse_nominal_dt=case["parameters"]["dt"],
                              reference_nominal_dt=largest["cache_key"]["parameters"]["dt"],
                              coarse_run_id=case["id"]+"_clean",reference_run_id=Path(largest["archive"]).stem,
                              coarse_parameters=case["parameters"],reference_parameters=largest["cache_key"]["parameters"])
        rows.extend(original_comparisons)
        for record in records:
            if record is largest:
                continue
            middle = array_data(ROOT / record["archive"])
            refined_comparisons=compare_arrays(middle, fine, family, case["id"], "matched_original_benchmark")
            for comparison in refined_comparisons:
                comparison.update(coarse_nominal_dt=record["cache_key"]["parameters"]["dt"],
                                  reference_nominal_dt=largest["cache_key"]["parameters"]["dt"],
                                  coarse_run_id=Path(record["archive"]).stem,reference_run_id=Path(largest["archive"]).stem,
                                  coarse_parameters=record["cache_key"]["parameters"],reference_parameters=largest["cache_key"]["parameters"])
            rows.extend(refined_comparisons)
    return rows


def residual_sampling():
    """Same physical frames, centered derivatives and independent energy quadrature."""
    residual_rows, energy_rows = [], []
    for case in ("interacting_vortices", "decaying_turbulence"):
        data = array_data(NS / "datasets" / f"{case}_512_dt0.005.npz")
        velocity, times = data["velocity"], data["times"]
        omega = vorticity(velocity)
        n = velocity.shape[-1]
        kx, ky = waves(n)
        omega_hat = np.fft.fft2(omega)
        wx = np.fft.ifft2(1j * kx * omega_hat).real
        wy = np.fft.ifft2(1j * ky * omega_hat).real
        laplace = np.fft.ifft2(-(kx * kx + ky * ky) * omega_hat).real
        convection = velocity[:, 0] * wx + velocity[:, 1] * wy
        dynamics = convection - .01 * laplace
        # Common centers and common support for every spacing/order: no endpoint stencil bias.
        centers = np.arange(8, 33, 4)
        scale = float(np.sqrt(np.mean(dynamics[centers] ** 2)))
        for stride in (1, 2, 4):
            spacing = float(times[stride] - times[0])
            for order in (2, 4):
                if order == 2:
                    temporal = (omega[centers + stride] - omega[centers - stride]) / (2 * spacing)
                else:
                    temporal = (-omega[centers + 2 * stride] + 8 * omega[centers + stride]
                                - 8 * omega[centers - stride] + omega[centers - 2 * stride]) / (12 * spacing)
                residual = temporal + dynamics[centers]
                per_time = np.sqrt(np.mean(residual ** 2, axis=(-2, -1)))
                rms = float(np.sqrt(np.mean(residual ** 2)))
                residual_rows.append({"case": case, "grid": n, "spacing": spacing, "order": order,
                                      "residual_rms": rms, "normalized_residual": rms / scale,
                                      "normalization": "RMS(convection - nu*laplace omega) on exactly the same seven centers",
                                      "scale_rms": scale, "support_start_time": .8, "support_end_time": 3.2,
                                      "times": times[centers].tolist(), "per_time_rms": per_time.tolist(),
                                      "units": "nondimensional omega/time", "scope": "unfiltered physical vorticity equation; includes output-spacing error; not the RK4 internal residual"})
            selected = np.arange(0, len(times), stride)
            ts = times[selected]
            energy = .5 * np.mean(np.sum(velocity[selected] ** 2, axis=1), axis=(-2, -1))
            enstrophy = .5 * np.mean(omega[selected] ** 2, axis=(-2, -1))
            for method in ("trapezoid", "simpson"):
                integral = cumulative_trapezoid(enstrophy, ts, initial=0) if method == "trapezoid" else cumulative_simpson(enstrophy, x=ts, initial=0)
                defect = (energy - energy[0] + .02 * integral) / energy[0]
                energy_rows.append({"case": case, "grid": n, "spacing": spacing, "quadrature": method,
                                    "maximum_absolute_normalized_defect": float(np.max(np.abs(defect))),
                                    "final_normalized_defect": float(defect[-1]), "times": ts.tolist(),
                                    "per_time_normalized_defect": defect.tolist(),
                                    "scope": "E-E0+2nu*integral(Zdt); sampled quadrature, not a PDE or internal RK4 error bound"})
        del data, velocity, omega, omega_hat, wx, wy, laplace, convection, dynamics
    return residual_rows, energy_rows


def dense_sampling_controls():
    """Isolate output cadence by subsampling the very same new integration."""
    (OUT/"datasets").mkdir(exist_ok=True)
    records=[]
    sources=scientific_hashes()
    for case in ("interacting_vortices","decaying_turbulence"):
        config={"case":case,"nx":128,"viscosity":.01,"amplitude":.8,"seed":17,
                "dt":.00125,"n_steps":3200,"sample_every":20,"cfl":.35,
                "length_x":2*np.pi,"length_y":2*np.pi}
        path=OUT/"datasets"/f"dense_{case}_128_dt0.00125.npz"
        meta_path=path.with_suffix(".json")
        key={"parameters":config,"scientific_source_sha256":sources}
        if path.exists() and meta_path.exists():
            meta=read(meta_path)
            if meta["cache_key"]!=key or meta["archive_sha256"]!=sha(path):raise ValueError("Stale dense sampling control")
            print("REUSED DENSE",case,flush=True)
        else:
            print("RUN DENSE",case,flush=True)
            wall,cpu=time.perf_counter(),time.process_time()
            data=simulate_periodic_ns(**config)
            seconds,processor=time.perf_counter()-wall,time.process_time()-cpu
            np.savez_compressed(path,**{k:v for k,v in data.items() if isinstance(v,np.ndarray)})
            meta={"complete":True,"cache_key":key,"metadata":data["metadata"],"shape":list(data["velocity"].shape),
                  "archive_sha256":sha(path),"wall_seconds":seconds,"cpu_seconds":processor,
                  "density":"constant rho=1; stored explicitly","scope":"new exact same-physics NS128 integration,161 outputs at spacing0.025; no change to reconstruction training data"}
            write(meta_path,meta)
            print("DONE DENSE",case,"seconds",round(seconds,2),flush=True)
        records.append({"archive":str(path.relative_to(ROOT)),**meta})
    write(OUT/"dense_integrations.json",{"complete":True,"count":2,"runs":records,
          "scope":"sampling experiments, separate from18 matched original-benchmark references"})
    return records


def dense_sampling_analysis():
    records=dense_sampling_controls()
    residuals=[]
    comparisons=[]
    for record in records:
        data=array_data(ROOT/record["archive"])
        case=record["cache_key"]["parameters"]["case"]
        u=data["velocity"];times=data["times"];omega=vorticity(u)
        centers=np.arange(32,129,16)
        kx,ky=waves(128)
        selected_hat=np.fft.fft2(omega[centers])
        wx=np.fft.ifft2(1j*kx*selected_hat).real
        wy=np.fft.ifft2(1j*ky*selected_hat).real
        laplace=np.fft.ifft2(-(kx*kx+ky*ky)*selected_hat).real
        dynamics=u[centers,0]*wx+u[centers,1]*wy-.01*laplace
        integer_modes=np.rint(np.fft.fftfreq(128)*128)
        mx,my=np.meshgrid(integer_modes,integer_modes)
        active=(abs(mx)<128/3)&(abs(my)<128/3)
        projected=np.fft.ifft2(np.fft.fft2(dynamics)*active).real
        projection=dynamics-projected
        scale=float(np.sqrt(np.mean(dynamics**2)))
        projection_rms=float(np.sqrt(np.mean(projection**2)))
        for stride in (1,2,4,8,16):
            spacing=.025*stride
            for order in (2,4):
                temporal=(omega[centers+stride]-omega[centers-stride])/(2*spacing) if order==2 else\
                        (-omega[centers+2*stride]+8*omega[centers+stride]-8*omega[centers-stride]+omega[centers-2*stride])/(12*spacing)
                physical_residual=temporal+dynamics
                temporal_residual=temporal+projected
                physical_rms=float(np.sqrt(np.mean(physical_residual**2)))
                temporal_rms=float(np.sqrt(np.mean(temporal_residual**2)))
                # Fixed Fourier projection is orthogonal: the Galerkin residual
                # and discarded physical RHS occupy disjoint mode subspaces.
                orthogonality=float(np.mean(temporal_residual*projection))
                if abs(orthogonality)>1e-10*max(scale**2,np.finfo(float).tiny):
                    raise AssertionError("Projected temporal and discarded RHS unexpectedly overlap")
                residuals.append({"case":case,"grid":128,"spacing":spacing,"order":order,
                                  "residual_rms":physical_rms,"normalized_residual":physical_rms/scale,
                                  "projected_temporal_residual_rms":temporal_rms,
                                  "projected_temporal_residual_normalized":temporal_rms/scale,
                                  "spatial_projection_defect_rms":projection_rms,
                                  "spatial_projection_defect_normalized":projection_rms/scale,
                                  "decomposition_cross_mean":orthogonality,
                                  "scale_rms":scale,"times":times[centers].tolist(),
                                  "normalization":"RMS(convection - nu*laplace omega), same seven centers; own128 integration",
                                  "scope":"all cadence variants subsample the exact same161-frame integration; Galerkin split is discrete consistency, not a continuum truth certificate"})
        sampled={k:v[::4] for k,v in data.items() if isinstance(v,np.ndarray) and v.shape[0]==161}
        primary=array_data(NS/"datasets"/f"{case}_128_dt0.005.npz")
        fine512=array_data(NS/"datasets"/f"{case}_512_dt0.005.npz")
        for reference,label in ((sampled,"primary_dt0.005_vs_dense_dt0.00125_at128"),(fine512,"dense128_vs_fine512_same_saved_times")):
            coarse=primary if reference is sampled else sampled
            rows=compare_arrays(coarse,reference,"newtonian",case,label)
            comparisons.extend(rows)
    result={"complete":True,"integrations":records,"residual_sampling":residuals,"same_time_comparisons":comparisons,
            "qualification":"128 dense controls cannot substitute512 spatial certification; cadence effect is isolated within one128 integration"}
    write(OUT/"dense_sampling.json",result)
    export_csv(OUT/"dense_sampling_residuals.csv",residuals)
    export_csv(OUT/"dense_same_time_comparisons.csv",comparisons)
    return result


def plot_dense_sampling(result):
    fig,axes=plt.subplots(1,2,figsize=(13,5.5),constrained_layout=True)
    for ax,case in zip(axes,("interacting_vortices","decaying_turbulence")):
        for order,color in ((2,"#0072B2"),(4,"#D55E00")):
            rows=sorted([r for r in result["residual_sampling"] if r["case"]==case and r["order"]==order],key=lambda r:r["spacing"])
            ax.loglog([r["spacing"] for r in rows],[r["normalized_residual"] for r in rows],"o-",color=color,label=f"Residuo físico, orden{order}")
        rows=sorted([r for r in result["residual_sampling"] if r["case"]==case and r["order"]==4],key=lambda r:r["spacing"])
        ax.loglog([r["spacing"] for r in rows],[r["projected_temporal_residual_normalized"] for r in rows],"s--",color="#009E73",label="Componente temporal, orden4")
        ax.axhline(rows[0]["spatial_projection_defect_normalized"],linestyle=":",color="#CC79A7",label="Componente espacial descartada")
        ax.set_xticks([.025,.05,.1,.2,.4],["0.025","0.05","0.1","0.2","0.4"])
        ax.set_xlabel("Espaciado de salida dentro de la misma integración")
        ax.set_ylabel("Residuo RMS / dinámica RMS")
        ax.set_title(LABELS[case]+": NS128, ν=0.01, T=4")
        ax.legend(fontsize=12);ax.grid(alpha=.2)
    fig.suptitle("Guardar 161 campos separa error temporal de salida y proyección espacial",fontsize=15)
    save_figure(fig,"dense_sampling_isolation")


def triangle_bounds():
    """Actual estimator-vs-finer-reference error and rigorous discrete norm bounds.

    r is the coarse clean simulation, f is fine512, c is the reconstructed
    velocity. All three are represented at the same fine grid and times.
    If a=||c-r||/||r||, q=||r||/||f|| and delta=||r-f||/||f||, then
    |a*q-delta| <= ||c-f||/||f|| <= a*q+delta.
    These bounds apply to f, not to the unknown exact continuum solution.
    """
    fine = array_data(NS / "datasets/interacting_vortices_512_dt0.005.npz")
    fine_velocity = fine["velocity"]
    fine_omega = vorticity(fine_velocity)
    records = []
    paths = [(RESULTS / "highres_reconstruction_2026-10-14", "tls_r8_projected"),
             (RESULTS / "highres_fixed_bandwidth_2026-10-14", "tls_r8_fixed_band"),
             (RESULTS / "highres_fixed_bandwidth_2026-10-14", "identity_fixed_band")]
    for n in (128, 256):
        clean = array_data(NS / "datasets" / f"interacting_vortices_{n}_dt0.005.npz")
        np.testing.assert_allclose(clean["times"], fine["times"], rtol=0, atol=2e-14)
        for noise in ("gaussian_iid", "spatial_fixed_length"):
            for folder, method in paths:
                path = folder / "reconstructions" / f"{noise}_{n}_s7_{method}.npz"
                candidate = array_data(path)
                np.testing.assert_allclose(candidate["times"], fine["times"], rtol=0, atol=2e-14)
                sums = {name: np.zeros(5) for name in ("velocity", "vorticity")}
                for i, (reference, estimated) in enumerate(zip(clean["velocity"], candidate["velocity"])):
                    r, c = transfer(reference, 512), transfer(estimated, 512)
                    for observable, (a, b, f) in {
                            "velocity": (c, r, fine_velocity[i]),
                            "vorticity": (vorticity(c), vorticity(r), fine_omega[i])}.items():
                        sums[observable] += [np.sum((a-b)**2), np.sum((b-f)**2), np.sum((a-f)**2), np.sum(b**2), np.sum(f**2)]
                for observable, values in sums.items():
                    cr, rf, cf, rr, ff = np.sqrt(values)
                    a, delta, actual, q = cr / rr, rf / ff, cf / ff, rr / ff
                    lower, upper = abs(a * q - delta), a * q + delta
                    if actual < lower - 5e-14 or actual > upper + 5e-14:
                        raise AssertionError("Discrete norm triangle inequality failed")
                    records.append({"nx": n, "noise": noise, "seed": 7, "method": method,
                                    "observable": observable, "reference_grid": 512,
                                    "start_time": 0., "end_time": 4., "saved_times": 41,
                                    "coarse_reference_relative_error": delta,
                                    "reconstruction_vs_coarse_reference": a, "reference_norm_ratio": q,
                                    "lower_vs_fine": lower, "actual_vs_fine": actual, "upper_vs_fine": upper,
                                    "bound_width": upper-lower,
                                    "scope": "aggregate L2 over equally weighted saved frames and the same full512 Fourier grid; finest reference remains numerical",
                                    "qualification": "postprocessed projected states exclude Nyquist; derivative and Fourier transfer commute; not applicable to unmatched physical campaigns"})
    return records


def pairwise_reference_rank_stability(bounds):
    """Sufficient ordering condition against a specified finer numerical field."""
    from itertools import combinations
    output=[]
    for n in (128,256):
        for noise in ("gaussian_iid","spatial_fixed_length"):
            for observable in ("velocity","vorticity"):
                rows=[r for r in bounds if r["nx"]==n and r["noise"]==noise and r["observable"]==observable]
                for a,b in combinations(rows,2):
                    low,high=sorted((a,b),key=lambda r:r["reconstruction_vs_coarse_reference"])
                    np.testing.assert_allclose([a["coarse_reference_relative_error"],a["reference_norm_ratio"]],
                                               [b["coarse_reference_relative_error"],b["reference_norm_ratio"]],rtol=1e-12,atol=1e-14)
                    delta=a["coarse_reference_relative_error"];q=a["reference_norm_ratio"]
                    gap=(high["reconstruction_vs_coarse_reference"]-low["reconstruction_vs_coarse_reference"])*q
                    certified=gap>2*delta
                    if certified and low["actual_vs_fine"]>=high["actual_vs_fine"]:
                        raise AssertionError("Certified rank ordering contradicts computed fine-reference errors")
                    output.append({"nx":n,"noise":noise,"observable":observable,"seed":7,
                                   "lower_error_method_to_coarse":low["method"],"higher_error_method_to_coarse":high["method"],
                                   "scaled_coarse_error_gap":gap,"two_reference_differences":2*delta,
                                   "order_stable_against_fine512_reference":certified,
                                   "actual_lower_method_error_to_fine":low["actual_vs_fine"],
                                   "actual_higher_method_error_to_fine":high["actual_vs_fine"],
                                   "criterion":"(a_high-a_low)*q>2delta is sufficient for upper_low<lower_high against this fine reference",
                                   "qualification":"single seed7, same flow/grid/cadence/noise; stability to reference512 only, not universal algorithm superiority or exact-continuum error certification"})
    return output


def analytic_controls():
    records = []
    coefficients = ((3, 5, .17), (7, 2, -.08), (4, 6, .11))
    def analytical(n):
        x, y = np.meshgrid(np.arange(n) * 2*np.pi/n, np.arange(n) * 2*np.pi/n)
        u, v, omega = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
        for mx, my, amplitude in coefficients:
            u += amplitude * my * np.sin(mx*x) * np.cos(my*y)
            v -= amplitude * mx * np.cos(mx*x) * np.sin(my*y)
            omega += amplitude * (mx*mx+my*my) * np.sin(mx*x) * np.sin(my*y)
        return np.stack((u, v)), omega
    for grid in (32, 64, 128):
        velocity, exact_omega = analytical(grid)
        fine_velocity, fine_omega = analytical(512)
        kx, ky = waves(grid)
        divergence = np.fft.ifft2(1j*kx*np.fft.fft2(velocity[0]) + 1j*ky*np.fft.fft2(velocity[1])).real
        records.append({"kind": "independent_three_mode_streamfunction", "grid": grid, "reference_grid": 512,
                        "coefficients": [list(x) for x in coefficients],
                        "prolongation_max_absolute_error": float(np.max(np.abs(transfer(velocity,512)-fine_velocity))),
                        "vorticity_max_absolute_error": float(np.max(np.abs(vorticity(velocity)-exact_omega))),
                        "vorticity_prolongation_max_absolute_error": float(np.max(np.abs(transfer(exact_omega,512)-fine_omega))),
                        "divergence_rms": float(np.sqrt(np.mean(divergence**2))),
                        "qualification": "operator/FFT normalization verification; manufactured field, not a physical evolution or nonlinear DNS certificate"})
    return records


def ranges_record(data, family, case, campaign):
    output = []
    values = field_views(data)
    for observable, field in values.items():
        components = tuple(range(1, field.ndim-2))
        magnitude = np.sqrt(np.sum(field**2,axis=components)) if components else field
        output.append({"family": family, "case": case, "campaign": campaign, "observable": observable,
                       "grid": field.shape[-1], "start_time": float(data["times"][0]), "end_time": float(data["times"][-1]),
                       "representation": "pointwise vector/tensor magnitude" if components else "signed scalar" if observable == "vorticity" else "scalar",
                       "minimum": float(np.min(magnitude)), "maximum": float(np.max(magnitude)),
                       "percentile_05": float(np.quantile(magnitude,.05)), "median": float(np.median(magnitude)),
                       "percentile_95": float(np.quantile(magnitude,.95)),
                       "rms": float(np.sqrt(np.mean(magnitude**2))),
                       "units": {"velocity":"U", "vorticity":"U/L", "stress_frobenius":"rho_ref*U^2",
                                 "conformation":"dimensionless", "excess_conformation":"dimensionless",
                                 "apparent_viscosity":"rho_ref*U*L", "shear_rate":"U/L"}[observable],
                       "scaling": "nondimensional; U,L,rho_ref are declared reference scales, not measured SI values"})
    return output


def model_separation():
    """Material effect and mesh difference with a common Newtonian denominator."""
    reference = array_data(VE / "datasets/newtonian_256.npz")
    newton = field_views(reference)
    records = []
    for model in ("oldroyd_b", "fene_p"):
        fine = array_data(VE / "datasets" / f"{model}_256.npz")
        coarse = array_data(VE / "datasets" / f"{model}_128.npz")
        np.testing.assert_allclose(fine["times"],reference["times"],rtol=0,atol=2e-14)
        np.testing.assert_allclose(coarse["times"],reference["times"],rtol=0,atol=2e-14)
        a,b=field_views(fine),field_views(coarse)
        for observable in ("velocity","vorticity"):
            effect=[ratios(x,y) for x,y in zip(a[observable],newton[observable])]
            discrepancy=[l2(transfer(x,256)-y)/l2(z) for x,y,z in zip(b[observable],a[observable],newton[observable])]
            records.append({"model":model,"observable":observable,"times":reference["times"].tolist(),
                            "constitutive_effect_vs_newtonian":effect,"mesh_difference_128_256_normalized_newtonian":discrepancy,
                            "newtonian_viscosity":.06,"nu_s":.02,"nu_p":.04,"relaxation_time":.5,
                            "scope":"same initial U_rms=.8/domain/output times; fixed Newtonian nu=.06 equals zero-shear total polymer+solvent viscosity, not identical rheology",
                            "qualification":"constitutive differences are physical model contrasts, not reconstruction errors"})
    return records


def source_audit():
    actual = scientific_hashes()
    assertions=[]
    for folder in (OLD,NS,VE,VP,RESULTS/"highres_reconstruction_2026-10-14",RESULTS/"highres_fixed_bandwidth_2026-10-14"):
        provenance=read(folder/"provenance.json")
        expected=provenance.get("source_sha256",provenance.get("source_hashes",provenance.get("source_files_sha256",{})))
        for path,digest in expected.items():
            if path.startswith("python/src/"):
                if actual[path]!=digest:raise ValueError(f"Scientific source changed: {path}")
                assertions.append({"campaign":folder.name,"source":path,"sha256":digest,"matches":True})
    archive_checks=[]
    for folder,hash_key in ((NS,"npz_sha256"),(VE,"archive_sha256")):
        for meta_path in sorted((folder/"datasets").glob("*.json")):
            metadata=read(meta_path)
            path=meta_path.with_suffix(".npz")
            if sha(path)!=metadata[hash_key]:raise ValueError(f"Corrupt archive: {path}")
            archive_checks.append({"archive":str(path.relative_to(ROOT)),"sha256":metadata[hash_key],"matches":True})
    # Canonical digest includes all arrays, geometry and physical solver metadata.
    from fluid_denoise.research.runner import dataset_digest
    for manifest in read(OLD/"dataset_manifests.json"):
        case_id=manifest["case"]["id"]
        path=OLD/"datasets"/f"{case_id}_clean.npz"
        data=array_data(path)
        data.update(metadata=manifest["metadata"],dx=manifest["dx"],dy=manifest["dy"],boundary=manifest["boundary"])
        digest=dataset_digest(data)
        if digest!=manifest["dataset_sha256"]:raise ValueError(f"Original benchmark digest changed: {path}")
        archive_checks.append({"archive":str(path.relative_to(ROOT)),"sha256":sha(path),"canonical_sha256":digest,"matches":True})
    for path in sorted((VP/"datasets").glob("*.npz")):
        data=array_data(path);meta=read(path.with_suffix(".json"))
        data.update(metadata=meta["metadata"],dx=meta["dx"],dy=meta["dy"],boundary="periodic")
        digest=dataset_digest(data)
        if digest!=meta["dataset_sha256"]:raise ValueError(f"Canonical digest changed: {path}")
        archive_checks.append({"archive":str(path.relative_to(ROOT)),"sha256":sha(path),"canonical_sha256":digest,"matches":True})
    for record in read(OUT/"matched_integrations.json")["runs"]:
        path=ROOT/record["archive"]
        digest=sha(path)
        if digest!=record["archive_sha256"]:raise ValueError(f"Matched reference archive changed: {path}")
        if record["cache_key"]["scientific_source_sha256"]!=actual:raise ValueError("Matched reference solver changed")
        case_id=record["cache_key"]["case_id"]
        if sha(OLD/"datasets"/f"{case_id}_clean.npz")!=record["cache_key"]["old_archive_sha256"]:
            raise ValueError("Matched reference copied a changed original benchmark")
        archive_checks.append({"archive":record["archive"],"sha256":digest,"matches":True})
    if (OUT/"dense_integrations.json").exists():
        for record in read(OUT/"dense_integrations.json")["runs"]:
            digest=sha(ROOT/record["archive"])
            if digest!=record["archive_sha256"]:raise ValueError("Dense sampling archive changed")
            if record["cache_key"]["scientific_source_sha256"]!=actual:raise ValueError("Dense sampling solver changed")
            archive_checks.append({"archive":record["archive"],"sha256":digest,"matches":True})
    return {"scientific_sources_unchanged":True,"scientific_source_sha256":actual,"assertions":assertions,
            "archives":archive_checks,"qualification":"historical driver/rendering hashes can differ after figure changes; these checks compare frozen scientific solver modules and saved numerical arrays"}


def save_figure(fig, name):
    fig.savefig(OUT/"figures"/(name+".pdf"),bbox_inches="tight")
    fig.savefig(OUT/"figures"/(name+".png"),bbox_inches="tight")
    plt.close(fig)


LABELS={"interacting_vortices":"Cuatro vórtices","decaying_turbulence":"Estado multiescala",
        "oldroyd_b":"Oldroyd-B","fene_p":"FENE-P","hb_thinning":"HB n=0.6",
        "bingham":"Bingham","hb_thickening":"HB n=1.4"}
OBS={"velocity":r"$u$","vorticity":r"$\omega$","stress_frobenius":r"$\tau$ (Frobenius)",
     "excess_conformation":r"$C-I$","conformation":r"$C$","apparent_viscosity":r"$\eta_{app}$","shear_rate":r"$\dot\gamma$"}
COLORS={"velocity":"#0072B2","vorticity":"#D55E00","stress_frobenius":"#009E73",
        "apparent_viscosity":"#CC79A7","excess_conformation":"#009E73"}


def style_error_axis(ax):
    ax.set_yscale("log")
    ax.set_ylabel("Máximo error completo (%)")
    ax.grid(axis="y",which="major",alpha=.2)


def plot_fidelity(rows):
    visual=[r for r in rows if r["campaign"]=="visual_highres"]
    fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    for case,marker in (("interacting_vortices","o"),("decaying_turbulence","s")):
        for observable,style in (("velocity","--"),("vorticity","-")):
            selected=sorted([r for r in visual if r["case"]==case and r["observable"]==observable],key=lambda r:r["grid"])
            axes[0,0].plot([r["grid"] for r in selected],[100*r["maximum_relative_l2"] for r in selected],marker=marker,
                           linestyle=style,label=LABELS[case]+" "+OBS[observable])
    axes[0,0].set_title("Navier–Stokes: referencia 512², T=4")
    axes[0,0].set_xticks([64,128,256]);axes[0,0].set_xlabel("Malla de la simulación comparada")
    axes[0,0].legend(fontsize=12)
    for model,marker in (("oldroyd_b","o"),("fene_p","s")):
        for observable,style in (("velocity","--"),("stress_frobenius","-")):
            selected=sorted([r for r in visual if r["case"]==model and r["observable"]==observable],key=lambda r:r["grid"])
            axes[0,1].plot([r["grid"] for r in selected],[100*r["maximum_relative_l2"] for r in selected],marker=marker,
                           linestyle=style,label=LABELS[model]+" "+OBS[observable])
    axes[0,1].set_title("Polímeros: referencia 256², T=2.4")
    axes[0,1].set_xticks([64,128]);axes[0,1].set_xlabel("Malla de la simulación comparada")
    axes[0,1].legend(fontsize=12)
    for ax,horizon,grid,reference in ((axes[1,0],2.,64,128),(axes[1,1],.1,128,256)):
        x=np.arange(3)
        for j,observable in enumerate(("velocity","vorticity","stress_frobenius","apparent_viscosity")):
            selected=[next(r for r in visual if r["case"]==case and r["observable"]==observable and r["grid"]==grid and r["end_time"]==horizon)
                      for case in ("hb_thinning","bingham","hb_thickening")]
            ax.bar(x+(j-1.5)*.19,[100*r["maximum_relative_l2"] for r in selected],.18,color=COLORS[observable],label=OBS[observable])
        ax.set_xticks(x,["HB n=0.6","Bingham","HB n=1.4"])
        ax.set_title(f"Viscoplásticos: {grid}² → {reference}², T={horizon:g}")
        ax.legend(ncol=2,fontsize=12)
    for ax in axes.ravel():style_error_axis(ax)
    fig.suptitle("La resolución suficiente depende del observable y del intervalo",fontsize=16)
    save_figure(fig,"fidelity_observable_grid")

    fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    selectors=[("interacting_vortices",128,4.,("velocity","vorticity")),
               ("oldroyd_b",128,2.4,("velocity","vorticity","excess_conformation","stress_frobenius")),
               ("fene_p",128,2.4,("velocity","vorticity","stress_frobenius")),
               ("hb_thinning",64,2.,("velocity","vorticity","stress_frobenius","apparent_viscosity"))]
    for ax,(case,grid,horizon,observables) in zip(axes.ravel(),selectors):
        for observable in observables:
            r=next(x for x in visual if x["case"]==case and x["grid"]==grid and x["end_time"]==horizon and x["observable"]==observable)
            error=np.asarray([np.nan if x is None else x for x in r["per_time_relative_l2"]])*100
            ax.plot(r["times"],np.maximum(error,1e-12),label=OBS[observable],color=COLORS[observable],
                    linestyle="--" if observable=="excess_conformation" else "-")
            if observable in ("stress_frobenius","apparent_viscosity"):
                ax.plot(r["maximum_time"],100*r["maximum_relative_l2"],"o",color=COLORS[observable])
        ax.set_yscale("log");ax.set_ylim(bottom=1e-8)
        ax.set_title(f"{LABELS[case]}: {grid}² vs {r['reference_grid']}²")
        ax.set_xlabel("Tiempo adimensional");ax.set_ylabel("Error completo por instante (%)")
        ax.grid(axis="y",alpha=.2);ax.legend(fontsize=12)
    fig.suptitle("Una tabla del instante final puede ocultar el máximo del error",fontsize=16)
    save_figure(fig,"fidelity_time_evolution")


def plot_residuals(residuals,energy):
    fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    for j,case in enumerate(("interacting_vortices","decaying_turbulence")):
        for order,marker in ((2,"o"),(4,"s")):
            records=sorted([r for r in residuals if r["case"]==case and r["order"]==order],key=lambda r:r["spacing"])
            axes[0,j].loglog([r["spacing"] for r in records],[r["normalized_residual"] for r in records],marker=marker,label=f"Derivada temporal orden {order}")
        axes[0,j].set_title(LABELS[case]+": ecuación de vorticidad")
        axes[0,j].set_ylabel("Residuo RMS / dinámica RMS")
        axes[0,j].set_xlabel("Espaciado entre salidas Δt (no paso RK4)")
        axes[0,j].set_xticks([.1,.2,.4],["0.1","0.2","0.4"])
        axes[0,j].legend(fontsize=12);axes[0,j].grid(alpha=.2)
        for method,marker in (("trapezoid","o"),("simpson","s")):
            records=sorted([r for r in energy if r["case"]==case and r["quadrature"]==method],key=lambda r:r["spacing"])
            axes[1,j].loglog([r["spacing"] for r in records],[abs(r["final_normalized_defect"]) for r in records],marker=marker,
                            label="Trapecios" if method=="trapezoid" else "Simpson")
        axes[1,j].set_title("Balance de energía: misma trayectoria 512²")
        axes[1,j].set_xlabel("Espaciado de la cuadratura de salida")
        axes[1,j].set_ylabel(r"$|E(T)-E(0)+2\nu\int Zdt|/E(0)$")
        axes[1,j].set_xticks([.1,.2,.4],["0.1","0.2","0.4"])
        axes[1,j].legend();axes[1,j].grid(alpha=.2)
    fig.suptitle("El residuo calculado desde imágenes incluye el error de muestreo temporal",fontsize=16)
    save_figure(fig,"residual_sampling")


def plot_budget(bounds):
    fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    options=[(n,method) for n in (128,256) for method in ("tls_r8_projected","tls_r8_fixed_band","identity_fixed_band")]
    labels=[name+f"\n{n}²" for n in (128,256) for name in ("TLS\nH⅔","TLS\nkc=10","Identidad\nkc=10")]
    for i,noise in enumerate(("gaussian_iid","spatial_fixed_length")):
        for j,observable in enumerate(("velocity","vorticity")):
            ax=axes[i,j]
            selected=[next(r for r in bounds if (r["nx"],r["method"])==option and r["noise"]==noise and r["observable"]==observable) for option in options]
            x=np.arange(len(selected))
            ax.bar(x-.16,[100*r["actual_vs_fine"] for r in selected],.32,color="#D55E00",label="Reconstrucción vs referencia 512²")
            ax.bar(x+.16,[100*r["coarse_reference_relative_error"] for r in selected],.32,color="#0072B2",label="Cambio de referencia gruesa → 512²")
            ax.set_yscale("log");ax.set_xticks(x,labels,fontsize=12)
            ax.set_ylabel("Error L2 de la ventana completa (%)")
            ax.set_title(("Gaussiano iid" if noise=="gaussian_iid" else "Ruido espacial σ_kernel=0.10")+" — "+OBS[observable])
            ax.grid(axis="y",alpha=.2)
    axes[0,0].legend(fontsize=12)
    fig.suptitle("En estos vórtices, el error del estimador domina el cambio de referencia",fontsize=16)
    save_figure(fig,"reconstruction_error_budget")


def plot_model_separation(records):
    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    for ax,observable in zip(axes,("velocity","vorticity")):
        for model,color in (("oldroyd_b","#0072B2"),("fene_p","#D55E00")):
            r=next(x for x in records if x["model"]==model and x["observable"]==observable)
            ax.semilogy(r["times"][1:],100*np.array(r["constitutive_effect_vs_newtonian"][1:]),color=color,label=LABELS[model]+" vs Newtoniano")
            ax.semilogy(r["times"][1:],100*np.array(r["mesh_difference_128_256_normalized_newtonian"][1:]),color=color,linestyle="--",label=LABELS[model]+" cambio 128² → 256²")
        ax.set_title(OBS[observable]+": efectos constitutivos y espaciales")
        ax.set_xlabel("Tiempo adimensional");ax.set_ylabel("Diferencia / norma Newtoniana (%)")
        ax.legend(fontsize=12);ax.grid(axis="y",alpha=.2)
    fig.suptitle("Control Newtoniano ν=0.06; polímeros νs=0.02, νp=0.04, λ=0.5",fontsize=15)
    save_figure(fig,"constitutive_effect_vs_resolution")


def plot_matched(rows):
    selected=[r for r in rows if r["campaign"]=="matched_original_benchmark"]
    grids={m["case"]["id"]:m["shape"][-1] for m in read(OLD/"dataset_manifests.json")}
    fig,axes=plt.subplots(1,3,figsize=(14,6),constrained_layout=True)
    groups=[("newtonian",("interacting_vortices_ns","decaying_turbulence_ns"),("velocity","vorticity")),
            ("viscoelastic",("oldroyd_b_2d","fene_p_2d"),("velocity","vorticity","stress_frobenius")),
            ("viscoplastic",("hb_thinning_2d","bingham_2d","hb_thickening_2d"),("velocity","vorticity","stress_frobenius","apparent_viscosity"))]
    for ax,(family,cases,observables) in zip(axes,groups):
        x=np.arange(len(cases))
        width=.75/len(observables)
        for j,observable in enumerate(observables):
            records=[next(r for r in selected if r["case"]==case and r["grid"]==grids[case] and r["observable"]==observable) for case in cases]
            ax.bar(x+(j-(len(observables)-1)/2)*width,[100*r["maximum_relative_l2"] for r in records],width*.95,
                   color=COLORS[observable],label=OBS[observable])
        labels=[]
        for case in cases:
            r=next(r for r in selected if r["case"]==case and r["grid"]==grids[case])
            label=LABELS.get(case.removesuffix("_ns").removesuffix("_2d"),case)
            labels.append(label+f"\n{grids[case]}² → {r['reference_grid']}²\nT={r['end_time']:g}")
        ax.set_xticks(x,labels,fontsize=12);style_error_axis(ax)
        ax.legend(fontsize=12,ncol=1)
    fig.suptitle("Fidelidad del banco original: nuevos controles con física y tiempos coincidentes",fontsize=16)
    save_figure(fig,"matching_original_references")


def solver_diagnostics():
    records=[]
    for folder in (NS,VE,VP):
        for path in sorted((folder/"datasets").glob("*.json")):
            meta=read(path);m=meta["metadata"]
            record={"campaign":folder.name,"run":path.stem,"units":"nondimensional",
                    "actual_internal_steps":m["internal_steps"],"dt_max":m["internal_dt_max"],
                    "dt_min":m["internal_dt_min"],"rejected_steps":m.get("rejected_steps",0),
                    "nominal_dt":m["nominal_dt"],"density_constant_one":True,
                    "mass_test_interpretation":"fixed rho=1 makes total mass constant by construction; velocity reconstruction must instead assess divergence and admissible boundary flux"}
            if folder==NS:
                diag=meta["diagnostics"]
                record.update(max_divergence_rms=max(x["divergence_rms"] for x in diag),
                              maximum_component_enstrophy_tail=max(x["component_enstrophy_tail"] for x in diag),
                              tail_definition="power in retained modes with max(|mx|,|my|)>=0.8*N/3; divided by all retained enstrophy")
            if folder==VE and m.get("model") in ("oldroyd_b","fene_p"):
                record.update(minimum_stage_eigenvalue=m["stage_min_conformation_eigenvalue"],
                              maximum_stage_trace=m["stage_max_conformation_trace"],
                              extensibility_squared=m.get("extensibility_squared"),
                              maximum_constitutive_projection_defect_rms=max(m["constitutive_spatial_projection_defect_rms"]),
                              maximum_snapshot_constitutive_residual_rms=max(m["constitutive_pde_residual_rms"]),
                              maximum_total_energy_rate_residual=max(abs(x) for x in m["diagnostics"]["total_energy_rate_residual"]),
                              residual_interpretation="constitutive residual mixes second-order saved-time differentiation and spatial projection; instantaneous total-energy rate uses solver RHS and is a separate discrete consistency check")
            if folder==VP:
                data=array_data(path.with_suffix(".npz"))
                record.update(maximum_stage_viscosity=m["maximum_stage_viscosity"],viscosity_cap=m["viscosity_cap"],
                              maximum_energy_budget_relative=max(abs(data["energy_budget_residual"]))/data["kinetic_energy"][0],
                              minimum_padded_dissipation=float(np.min(data["dissipation_rate"])),
                              energy_interpretation="RK4-stage accumulated padded constitutive dissipation; a small same-discretization budget does not prove spatially accurate stress or apparent viscosity")
            records.append(record)
    return records


def tolerance_table(rows):
    """Sensitivity to reporting tolerance, without declaring universal acceptability."""
    output=[]
    for r in rows:
        output.append({"campaign":r["campaign"],"case":r["case"],"grid":r["grid"],
                       "reference_grid":r["reference_grid"],"observable":r["observable"],
                       "end_time":r["end_time"],"maximum_error_percent":100*r["maximum_relative_l2"],
                       "difference_below_1_percent":r["maximum_relative_l2"]<=.01,
                       "difference_below_5_percent":r["maximum_relative_l2"]<=.05,
                       "difference_below_10_percent":r["maximum_relative_l2"]<=.10,
                       "qualification":"conditional comparison thresholds, not proven error bounds to exact truth or a DNS certificate"})
    return output


def export_csv(path,records):
    scalar=[{k:v for k,v in row.items() if not isinstance(v,(list,dict))} for row in records]
    keys=list(dict.fromkeys(k for row in scalar for k in row))
    with Path(path).open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=keys,lineterminator="\n")
        writer.writeheader();writer.writerows(scalar)


def report(summary):
    rows=summary["credibility_matrix"]
    old=[r for r in rows if r["campaign"]=="matched_original_benchmark"]
    original_grids={m["case"]["id"]:m["shape"][-1] for m in read(OLD/"dataset_manifests.json")}
    lines=["# Fidelidad de referencias y comparación de reconstrucciones","",
           "Este análisis separa tres preguntas: qué sistema físico se integra, cuánto cambia la referencia al refinar y cuánto error añade el estimador. El número de nodos no responde por sí solo a ninguna de ellas.","",
           "## Resultado que afecta directamente la comparación de modelos","",
           "El banco visual de mallas grandes utiliza condiciones distintas del banco original. Para evitar transferir indebidamente su convergencia, se ejecutaron nuevos controles que copian exactamente los parámetros físicos, condición inicial y tiempos guardados de `extended_physics_2026-10-14`. Taylor–Green se contrasta con su solución analítica; las demás referencias son integraciones nuevas del mismo solver en mallas mayores.","",
           "| Caso original | Malla → referencia | T | Error máximo u (%) | Error máximo ω (%) | Error máximo τ (%) | Error máximo η (%) |",
           "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for case,grid in original_grids.items():
        selection=[r for r in old if r["case"]==case and r["grid"]==grid]
        anchor=selection[0]
        def value(observable):
            match=[r for r in selection if r["observable"]==observable]
            return "—" if not match else f"{100*match[0]['maximum_relative_l2']:.5g}"
        display_case="Taylor–Green" if case=="taylor_green_ns" else LABELS.get(case.removesuffix("_ns").removesuffix("_2d"),case)
        display_reference="solución analítica" if anchor["reference_kind"].startswith("analytical") else f"{anchor['reference_grid']}²"
        lines.append(f"| {display_case} | {grid}² → {display_reference} | {anchor['end_time']:g} | {value('velocity')} | {value('vorticity')} | {value('stress_frobenius')} | {value('apparent_viscosity')} |")
    matched=read(OUT/"matched_integrations.json")
    lines.extend(["",f"Se completaron **{matched['count']} integraciones nuevas** de las 18 configuraciones propuestas. Su costo se declara en `matched_integrations.json`."+(" No hubo controles omitidos." if not matched.get("omitted_controls") else " Los controles omitidos se identifican por separado.")])
    for omission in matched.get("omitted_controls",[]):
        lines.extend(["",f"El control {omission['case_id']} de 128² y horizonte completo no se ejecutó: el piloto 64² proyectó ≈{omission['predicted_wall_seconds']/60:.1f} minutos frente al presupuesto de {omission['budget_seconds']/60:g} minutos por control. La comparación 32² frente a 64² sigue siendo evidencia numérica condicionada; no se presenta 64² como solución exacta o ya convergente frente a 128²."])
    lines += ["","Los valores son máximos entre los instantes guardados, sobre el campo completo prolongado por Fourier. Las normas de τ usan Frobenius: para los tres componentes independientes viscoplásticos se pondera `xx² + 2xy² + yy²`. El tensor conformacional se evalúa como `C` y como `C−I`; la identidad puede ocultar un error importante en deformación. A t=0, el esfuerzo y `C−I` son cero: su error relativo es indefinido y se guarda como `null`, sin declararlo exacto.","",
              "![Controles que coinciden con el banco original](figures/matching_original_references.png)","",
              "## Qué aporta y qué no aporta la referencia grande","",
              "Las campañas visuales conservan las comparaciones 64/128/256 frente a 512 (NS, ν=0.01, U_rms=0.8, T=4), 64/128 frente a 256 (Oldroyd-B/FENE-P, U_rms=0.8, T=2.4), y viscoplásticos 64 frente a 128 durante T=2, más controles 128 frente a 256 sólo entre t=0 y 0.1. Cada conjunto tiene parámetros y denominadores propios. Ningún control de horizonte corto valida automáticamente el intervalo largo.","",
              "En el banco visual, Oldroyd-B 128 tiene máximo error de velocidad ≈0.00089%, pero de esfuerzo ≈4.38% frente a 256. El control temporal está varios órdenes por debajo; la limitación es principalmente espacial para esos observables. La cola de esfuerzo de 256 alcanza ≈1.157e-4 bajo la definición declarada; no se afirma convergencia absoluta del tensor. En HB adelgazante 128 frente a 256, el error de η alcanza ≈5.02% aunque u≈0.00196%; esta evidencia sólo cubre t≤0.1.","",
              "![Fidelidad según observable y malla](figures/fidelity_observable_grid.png)","",
              "![Error completo a lo largo del tiempo](figures/fidelity_time_evolution.png)","",
              "## Presupuesto de error del estimador","",
              "Se usa una comparación verificable de tres campos en la misma malla 512 y los mismos 41 tiempos: `r` (referencia gruesa prolongada), `f` (referencia 512) y `c` (reconstrucción prolongada). Para `a=||c−r||/||r||`, `q=||r||/||f||` y `δ=||r−f||/||f||`, la desigualdad triangular da **`|a q−δ| ≤ ||c−f||/||f|| ≤ a q+δ`**. Los CSV guardan ambas cotas y el valor calculado directamente. La afirmación se refiere a la referencia numérica f; no es una cota contra la solución exacta desconocida.","",
              "En los vórtices suaves con semilla de ruido 7, estos intervalos son extremadamente estrechos comparados con el error de reconstrucción: el cambio de referencia explica una fracción mínima de los errores del estimador, especialmente de la vorticidad. La identidad filtrada se conserva como control, de modo que el beneficio de limitar banda no se atribuye automáticamente a TLS-DMD. `kc=10` fue una hipótesis exploratoria posterior a los 60 fallos iniciales y no constituye validación ciega con datos independientes.","",
              "![Error del estimador y cambio de referencia](figures/reconstruction_error_budget.png)","",
              "Además de los errores individuales se comprueba si la ordenación entre dos estimadores podría cambiar sólo por sustituir la referencia gruesa por 512: la condición suficiente es `(a_alto−a_bajo) q > 2δ`. `pairwise_reference_rank_stability.csv` registra la diferencia, el margen y el orden calculado directamente para cada pareja. Esto certifica estabilidad de esa comparación contra 512 en esos campos, no superioridad universal ni una cota a la verdad continua. Se conserva el resultado cuando la identidad con el mismo filtro supera a TLS-DMD: la resolución de la referencia no se usa como excusa para invertir esa conclusión.","",
              "## Residuos físicos: integración y muestreo son cosas distintas","",
              "Se recomputó la ecuación de vorticidad `ω_t+u·∇ω−νΔω=0` en NS 512 con derivadas centrales de orden 2 y 4, usando separaciones de salida 0.1/0.2/0.4 sobre exactamente los mismos siete centros t=0.8,1.2,…,3.2. Se guarda el residuo RMS en unidades adimensionales de ω/tiempo y dividido por la dinámica RMS. La escala se define en el JSON; no se confunde con el paso adaptativo interno RK4, que es mucho menor.","",
              "El balance `E−E0+2ν∫Zdt` se integra con trapecios y Simpson sobre los mismos campos guardados. Que cambie al variar el operador de salida demuestra que el valor incorpora error de cuadratura/muestreo; no permite deducir por sí solo el error del solver. El balance viscoplástico usa integral acumulada en etapas RK4 y cuadratura constitutiva con padding; es una comprobación de coherencia discreta diferente, que puede ser pequeña aunque η esté mal resuelta.","",
              "![Error de los diagnósticos calculados desde salidas](figures/residual_sampling.png)","",
              "Dos controles adicionales en 128² guardan 161 campos cada 0.025, con el mismo flujo, ν, amplitud, semilla y horizonte que los experimentos visuales. Al usar todos los campos o sólo uno de cada cuatro se cambia únicamente la cadencia del diagnóstico dentro de una única integración. El JSON separa residuo temporal respecto a la dinámica proyectada de la parte del RHS físico descartada por el corte de Fourier. Esa descomposición es coherencia discreta; no certifica por sí sola la solución continua.","",
              "La comparación a los 41 tiempos comunes registra por separado el cambio de dt nominal (primario 128 a 0.005 frente al control 128 a 0.00125) y la diferencia de campo 128 frente a 512. Los controles 128 no reemplazan la validación espacial 512. Sus dos integraciones se cuentan aparte de las 18 referencias coincidentes del banco original.","",
              "![Aislamiento del espaciado de salida dentro de una única integración](figures/dense_sampling_isolation.png)","",
              "## Interpretación de modelos físicos","",
              "El solver newtoniano integra Navier–Stokes incompresible 2D en un toro periódico mediante vorticidad, Fourier, corte estricto 2/3 y RK4 adaptado a CFL y difusión. No incluye paredes, cilindro ni cierre turbulento. Taylor–Green valida signos, difusión y orden temporal de un modo; no somete a prueba todo el aliasing de interacciones no lineales.","",
              "Oldroyd-B/FENE-P integran conjuntamente velocidad y conformación 3×3 en flujo plano. El esfuerzo acopla al momento mediante su divergencia; se exige conformación definida positiva y, en FENE-P, trC<L². Las etapas inválidas se rechazan: no se recorta C para esconder el fallo. La parte racional FENE-P no obtiene desaliasado exacto sólo por el corte cuadrático. Los residuos constitutivos de imágenes incluyen diferenciación temporal y proyección espacial.","",
              "Los estimadores de la comparación reconstruyen velocidad. C, τ y η en la matriz de fidelidad son observables auxiliares de la simulación, no campos que todos los algoritmos hayan reconstruido. En viscoelasticidad, el esfuerzo depende de la conformación y de su historia; no se deduce de una imagen de velocidad. La ley viscoplástica permitiría derivar esfuerzo o viscosidad desde una velocidad reconstruida si se asumen parámetros materiales conocidos, pero esa evaluación debe implementarse y declararse por separado.","",
              "Bingham/Herschel–Bulkley integran momento con divergencia del esfuerzo variable `τ=2η_app D`. La regularización Papanastasiou tiene viscosidad finita a corte cero; no produce regiones rígidas exactamente sin cedencia. Variar m o ε cambia el modelo regularizado; variar padding prueba cuadratura no polinómica; variar malla prueba resolución. El indicador `γdot>umbral` se declara aproximación dependiente de un umbral, no frontera exacta de cedencia.","",
              "El control Newtoniano ν=0.06 para polímeros coincide con ν_s+ν_p en el límite de corte cero y conserva velocidad inicial y tiempos. Las curvas de contraste constitutivo no son errores de reconstrucción. En los viscoplásticos visuales se igualó η_app=0.04 en una sola tasa de corte declarada; eso no iguala toda la ley ni proporciona un Reynolds universal equivalente.","",
              "![Efecto constitutivo frente al cambio de malla](figures/constitutive_effect_vs_resolution.png)","",
              "El LBM D2Q9-BGK legado con cilindro usa transporte transversal periódico, entrada de equilibrio, salida por copia y rebote en el obstáculo. Es débilmente compresible en unidades de retícula, con ν=(τ_relajación−0.5)/3. Su referencia limpia conserva ρ, u, v y máscara; con ρ variable el balance de masa tiene contenido, pero debe incluir flujo por las fronteras abiertas. No hay convergencia de esa estela demostrada por los experimentos periódicos. La rapidez escalar no determina el vector velocidad, vorticidad, continuidad ni esfuerzo y no permite recuperar esas magnitudes de manera única.","",
              "## Unidades, tolerancias y procedencia","",
              "`evidence_ranges.csv` registra mínimos, máximos, RMS y percentiles 5/50/95 de los campos físicos. Las simulaciones periódicas son adimensionales: velocidad escala U, vorticidad/tasa de corte U/L, viscosidad ρ_ref U L y esfuerzo ρ_ref U²; sin fijar U, L y ρ_ref físicos no se inventan unidades SI. Con ρ=1 constante y dominio fijo, masa total=área por construcción; no demuestra que una reconstrucción de velocidad conserve materia localmente. Esa pregunta requiere divergencia y flujos admisibles, además de ρ reconstruida cuando corresponda.","",
              "`tolerance_sensitivity.csv` muestra si la diferencia numérica cae debajo de 1%, 5% y 10% para cada observable e intervalo. Son niveles de comparación condicionados; no semáforos universales de certeza ni cotas al continuo. Los errores de referencia deben ser menores que la diferencia entre estimadores que se pretenda interpretar; si la diferencia de modelos es comparable al cambio por refinamiento, una ordenación fina no queda respaldada sólo por esos datos.","",
              "Los hashes de todas las fuentes científicas permanecen iguales a los de las campañas de origen. Se verificaron los SHA-256 de archivos NS/VE y los digest canónicos de viscoplasticidad. Los controles nuevos identifican el manifiesto y el NPZ antiguo exacto que copian, configuración completa, pasos internos, CPU y hash del nuevo archivo. Un cambio de hash de un programa de figuras histórico no se confunde con un cambio del solver.","",
              "Reproducción: `python python/scripts/analyze_reference_fidelity.py`. La primera ejecución genera los controles coincidentes; ejecuciones posteriores reutilizan datos sólo con configuración/fuentes/hashes iguales. `--skip-matched-integrations` exige controles ya disponibles y repite el análisis. Los NPZ completos se regeneran localmente; los JSON/CSV/figuras permiten inspeccionar el resultado publicado. No se interpolan nuevas trayectorias para simular refinamiento.","",
              "## Archivos calculados","",
              "- `summary.json`: matriz de credibilidad, cotas de error, residuos y controles analíticos.",
              "- `credibility_matrix.csv`: errores máximos/finales/ventana, observable, mallas e intervalo.",
              "- `triangle_bounds.csv`: cotas discretas y cálculo directo de error frente a 512.",
              "- `pairwise_reference_rank_stability.csv`: ordenación de estimadores que queda separada de la incertidumbre por cambiar a la referencia 512.",
              "- `residual_sampling.csv`, `energy_sampling.csv`: sensibilidad de diagnóstico al muestreo.",
              "- `evidence_ranges.csv`, `tolerance_sensitivity.csv`: magnitudes y niveles condicionados.",
              "- `matched_integrations.json`: nuevas integraciones con física original y costo registrado.",
              "- `source_verification.json`: fuentes/archivos verificados; `provenance.json`: entradas de este análisis.",""]
    (OUT/"FIDELITY.md").write_text("\n".join(lines))


def visual_analysis():
    """Auditable reusable analysis of the finished visual campaigns."""
    cache=OUT/"visual_summary.json"
    source_paths=[NS/"summary.json",NS/"comparison_timeseries.csv",VE/"summary.json",
                  VP/"comparisons.json",VP/"derived_stress_frobenius.json"]
    source_paths += [p for folder in (NS,VE,VP) for p in sorted((folder/"datasets").glob("*.npz"))]
    for folder,method in ((RESULTS/"highres_reconstruction_2026-10-14","tls_r8_projected"),
                          (RESULTS/"highres_fixed_bandwidth_2026-10-14","tls_r8_fixed_band"),
                          (RESULTS/"highres_fixed_bandwidth_2026-10-14","identity_fixed_band")):
        source_paths += [folder/"reconstructions"/f"{noise}_{grid}_s7_{method}.npz"
                         for noise in ("gaussian_iid","spatial_fixed_length") for grid in (128,256)]
    import inspect
    science_functions=(array_data,field_views,waves,vorticity,transfer,l2,ratios,credibility_matrix,
                       residual_sampling,triangle_bounds,analytic_controls,ranges_record,model_separation,solver_diagnostics)
    science_hash=hashlib.sha256("\n".join(inspect.getsource(f) for f in science_functions).encode()).hexdigest()
    fingerprint={"analysis_functions_sha256":science_hash,"scientific_source_sha256":scientific_hashes(),
                 "input_sha256":{str(p.relative_to(ROOT)):sha(p) for p in source_paths}}
    if cache.exists():
        cached=read(cache)
        if cached.get("fingerprint")==fingerprint:
            print("REUSED CHECKED VISUAL ANALYSIS",flush=True)
            # Rendering changes do not require repeating numerical comparisons.
            if cached.get("rendering_script_sha256")!=sha(__file__):
                a=cached["analysis"]
                plot_fidelity(a["credibility_matrix"]);plot_residuals(a["residual_sampling"],a["energy_sampling"])
                plot_budget(a["triangle_bounds"]);plot_model_separation(a["constitutive_separation"])
                cached["rendering_script_sha256"]=sha(__file__)
                write(cache,cached)
            return cached["analysis"]
    print("VISUAL CREDIBILITY MATRIX",flush=True)
    matrix=credibility_matrix(include_matched=False)
    print("SAMPLING RESIDUALS",flush=True)
    residuals,energy=residual_sampling()
    print("RECONSTRUCTION TRIANGLE BOUNDS",flush=True)
    bounds=triangle_bounds()
    print("PHYSICAL CONTRASTS",flush=True)
    models=model_separation()
    physical_ranges=[]
    for family,case,path in [("newtonian",x,NS/"datasets"/f"{x}_512_dt0.005.npz") for x in ("interacting_vortices","decaying_turbulence")]+\
                            [("viscoelastic",x,VE/"datasets"/f"{x}_256.npz") for x in ("oldroyd_b","fene_p")]+\
                            [("viscoplastic",x,VP/"datasets"/f"{x}_128.npz") for x in ("hb_thinning","bingham","hb_thickening")]:
        physical_ranges.extend(ranges_record(array_data(path),family,case,"visual_highres"))
    analysis={"credibility_matrix":matrix,"residual_sampling":residuals,"energy_sampling":energy,
              "triangle_bounds":bounds,"constitutive_separation":models,"evidence_ranges":physical_ranges,
              "analytic_controls":analytic_controls(),"solver_diagnostics":solver_diagnostics()}
    plot_fidelity(matrix);plot_residuals(residuals,energy);plot_budget(bounds);plot_model_separation(models)
    write(cache,{"complete":True,"scope":"finished visual campaigns, separate from matched original-benchmark controls",
                 "fingerprint":fingerprint,"rendering_script_sha256":sha(__file__),"analysis":analysis})
    return analysis


def analyze():
    begin=time.perf_counter()
    matched=read(OUT/"matched_integrations.json")
    if not matched["complete"]:
        raise RuntimeError("The matched original-benchmark controls are incomplete")
    (OUT/"figures").mkdir(exist_ok=True)
    print("DENSE SAMPLING ANALYSIS",flush=True)
    dense=dense_sampling_analysis()
    print("AUDIT SOURCES AND ARCHIVES",flush=True)
    sources=source_audit()
    write(OUT/"source_verification.json",sources)
    visual=visual_analysis()
    print("FULL CREDIBILITY MATRIX",flush=True)
    matrix=credibility_matrix()
    residuals,energy,bounds,models,physical_ranges=(visual[k] for k in ("residual_sampling","energy_sampling","triangle_bounds","constitutive_separation","evidence_ranges"))
    physical_ranges=list(physical_ranges)
    for manifest in read(OLD/"dataset_manifests.json"):
        case=manifest["case"]
        family={"spectral_ns":"newtonian","viscoelastic_periodic":"viscoelastic","viscoplastic_periodic":"viscoplastic"}[case["solver"]]
        physical_ranges.extend(ranges_record(array_data(OLD/"datasets"/f"{case['id']}_clean.npz"),family,case["id"],"original_benchmark"))
        controls=[r for r in matched["runs"] if r["cache_key"]["case_id"]==case["id"]]
        if controls:
            finest=max(controls,key=lambda r:r["cache_key"]["parameters"]["nx"])
            physical_ranges.extend(ranges_record(array_data(ROOT/finest["archive"]),family,case["id"],"matched_original_reference"))
    summary={"complete":True,"credibility_matrix":matrix,"residual_sampling":residuals,"energy_sampling":energy,
             "triangle_bounds":bounds,"analytic_controls":visual["analytic_controls"],"evidence_ranges":physical_ranges,
             "constitutive_separation":models,"solver_diagnostics":visual["solver_diagnostics"],
             "tolerance_sensitivity":tolerance_table(matrix),
             "source_verification":{"scientific_sources_unchanged":True,"verified_archives":len(sources["archives"])},
             "matched_controls":{"planned_count":18,"completed_count":matched["count"],"omitted_controls":matched.get("omitted_controls",[])},
             "dense_sampling":dense,
             "pairwise_reference_rank_stability":pairwise_reference_rank_stability(bounds),
             "qualified_conclusion":"reference fidelity belongs to a specified flow, observable, physical configuration and horizon; highest-grid self agreement is not exact truth or universal DNS certification"}
    for key in ("credibility_matrix","triangle_bounds","pairwise_reference_rank_stability","residual_sampling","energy_sampling","evidence_ranges","tolerance_sensitivity"):
        export_csv(OUT/(key+".csv"),summary[key])
    write(OUT/"summary.json",summary)
    plot_matched(matrix);plot_dense_sampling(dense)
    report(summary)
    input_files=[NS/"summary.json",NS/"comparison_timeseries.csv",VE/"summary.json",VP/"comparisons.json",VP/"derived_stress_frobenius.json",OLD/"dataset_manifests.json",OUT/"matched_integrations.json"]
    write(OUT/"provenance.json",{"script_sha256":sha(__file__),"scientific_source_sha256":scientific_hashes(),
          "input_sha256":{str(p.relative_to(ROOT)):sha(p) for p in input_files},
          "source_archive_checks":"source_verification.json; matched references archive+old-source hashes in matched_integrations.json",
          "counts":{"credibility_records":len(matrix),"triangle_records":len(bounds),"residual_sampling_records":len(residuals),
                    "energy_quadrature_records":len(energy),"figures":7,"matched_integrations":matched["count"],"dense_integrations":2},
          "wall_seconds_analysis":time.perf_counter()-begin,
          "analysis_cost_scope":"verification, loading, differentiation, field comparisons and visualization; additional matched integrations have separate cost records"})
    write(OUT/"completion.json",{"complete":True,"credibility_records":len(matrix),"figures":7,
          "matched_integrations":matched["count"],"dense_integrations":2,"omitted_controls":len(matched.get("omitted_controls",[])),"scientific_sources_unchanged":True,
          "summary_sha256":sha(OUT/"summary.json"),"script_sha256":sha(__file__)})
    print("COMPLETE",len(matrix),"credibility records; 7 figures",flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched-only", action="store_true")
    parser.add_argument("--visual-only",action="store_true",
                        help="Analyze completed visual campaigns separately while matched controls are running")
    parser.add_argument("--dense-only",action="store_true",help="Generate and analyze two dense saved-time NS128 controls")
    parser.add_argument("--only-solvers", nargs="+", choices=("spectral_ns","viscoelastic_periodic","viscoplastic_periodic"))
    parser.add_argument("--only-cases",nargs="+")
    parser.add_argument("--only-grids",nargs="+",type=int)
    parser.add_argument("--vp128-budget-seconds",type=float,default=2700.,
                        help="Skip a new128 viscoplastic control if its64 pilot predicts more than this wall-time budget; skipped runs stay explicit")
    parser.add_argument("--skip-matched-integrations", action="store_true",
                        help="Require existing complete matched references; never substitute changed physical configurations")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"figures").mkdir(exist_ok=True)
    if args.visual_only:
        visual_analysis()
        return
    if args.dense_only:
        dense=dense_sampling_analysis()
        plot_dense_sampling(dense)
        return
    if not args.skip_matched_integrations:
        matched_integrations(args.only_solvers,args.only_cases,args.only_grids,args.vp128_budget_seconds)
    if args.matched_only:
        return
    analyze()


if __name__ == "__main__":
    main()
