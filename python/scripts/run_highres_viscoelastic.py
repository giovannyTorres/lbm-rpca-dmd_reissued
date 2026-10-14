#!/usr/bin/env python3
"""Run, check and illustrate a matched high-resolution viscoelastic vortex bank.

The public numerical solvers are called unchanged.  This script keeps rejected
steps, finite-extensibility and SPD checks, and reports refinement evidence
rather than declaring a DNS certification from the number of pixels.
"""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

from pathlib import Path
import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.signal import resample

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python/src"))
from fluid_denoise.research.dns import simulate_periodic_ns
from fluid_denoise.research.viscoelastic_dns import simulate_viscoelastic_periodic

DEFAULT_OUTPUT = ROOT / "results/research/highres_viscoelastic_2026-10-14"
MODEL_LABEL = {"oldroyd_b": "Oldroyd-B", "fene_p": "FENE-P", "newtonian": "Newtoniano"}


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def file_sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def source_hashes():
    paths = [ROOT / "python/scripts/run_highres_viscoelastic.py"]
    paths += sorted((ROOT / "python/src/fluid_denoise/research").glob("*.py"))
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in paths}


def vorticity(velocity, length=2 * np.pi):
    ny, nx = velocity.shape[-2:]
    kx, ky = np.meshgrid(2 * np.pi * np.fft.fftfreq(nx, d=length / nx),
                         2 * np.pi * np.fft.fftfreq(ny, d=length / ny))
    return np.fft.ifft2(1j * kx * np.fft.fft2(velocity[..., 1, :, :])
                       - 1j * ky * np.fft.fft2(velocity[..., 0, :, :])).real


def relative_series(value, reference):
    axes = tuple(range(1, value.ndim))
    numerator = np.sqrt(np.sum((value - reference) ** 2, axis=axes))
    denominator = np.sqrt(np.sum(reference ** 2, axis=axes))
    # At t=0 the polymer stress and excess conformation are exactly zero.
    # Their relative error is undefined, not a fictitious successful zero.
    return [float(a / b) if b > 1e-13 else None for a, b in zip(numerator, denominator)]


def rms_series(value):
    return np.sqrt(np.mean(value ** 2, axis=tuple(range(1, value.ndim)))).tolist()


def restrict(field, size, retain_solver_band=True):
    """Fourier restriction, optionally to the exact common strict 2/3 band.

    SciPy's periodic resampling normalizes Fourier coefficients for the target
    grid.  The retained-band comparison is accompanied by the reference tail
    so missing fine scales are not erased from the scientific report.
    """
    value = resample(resample(field, size, axis=-1), size, axis=-2)
    if retain_solver_band:
        kx, ky = np.meshgrid(np.fft.fftfreq(size) * size, np.fft.fftfreq(size) * size)
        retained = (np.abs(kx) < size / 3) & (np.abs(ky) < size / 3)
        value = np.fft.ifft2(np.fft.fft2(value) * retained).real
    return value


def tail_fraction(field, cutoff=None):
    size = field.shape[-1]
    kx, ky = np.meshgrid(np.fft.fftfreq(size) * size, np.fft.fftfreq(size) * size)
    # Outer 20% of the actually retained component-wise Fourier band.
    threshold = .8 * size / 3 if cutoff is None else float(cutoff)
    outer = (np.abs(kx) >= threshold) | (np.abs(ky) >= threshold)
    transformed = np.abs(np.fft.fft2(field)) ** 2
    axes = tuple(range(1, transformed.ndim))
    total = transformed.sum(axis=axes)
    tail = (transformed * outer).sum(axis=axes)
    return [float(a / b) if b > 1e-24 else None for a, b in zip(tail, total)]


def spectra(field):
    size = field.shape[-1]
    kx, ky = np.meshgrid(np.fft.fftfreq(size) * size, np.fft.fftfreq(size) * size)
    shell = np.floor(np.sqrt(kx ** 2 + ky ** 2)).astype(int)
    power = np.abs(np.fft.fft2(field)) ** 2 / size ** 4
    if field.ndim > 2:
        power = power.sum(axis=tuple(range(field.ndim - 2)))
    return np.arange(shell.max() + 1), np.bincount(shell.ravel(), weights=power.ravel())


def run_case(job):
    output, model, size, dt, duration, amplitude, snapshots, label = job
    output = Path(output)
    destination = output / "datasets" / label
    destination.parent.mkdir(parents=True, exist_ok=True)
    steps = int(round(duration / dt))
    every = steps // (snapshots - 1)
    if not np.isclose(steps * dt, duration) or steps % (snapshots - 1):
        raise ValueError("duration/dt must be divisible by snapshots-1")
    common = {"nx": size, "case": "interacting_vortices", "amplitude": amplitude,
              "dt": dt, "n_steps": steps, "sample_every": every}
    physical = {"viscosity_s": .02, "viscosity_p": .04, "relaxation_time": .5,
                "extensibility_squared": 12., "stress_diffusivity": 0.}
    config = {"model": model, "common": common, "physical": physical if model != "newtonian" else {"viscosity": .06}}
    hashes = source_hashes()
    manifest = destination.with_suffix(".json")
    archive = destination.with_suffix(".npz")
    if archive.exists() and manifest.exists():
        previous = json.loads(manifest.read_text())
        previous_simulation_hashes = {key: value for key, value in previous.get("source_hashes", {}).items()
                                      if key.startswith("python/src/")}
        simulation_hashes = {key: value for key, value in hashes.items() if key.startswith("python/src/")}
        if (previous.get("complete") and previous.get("config") == config
                and previous_simulation_hashes == simulation_hashes
                and previous.get("archive_sha256") == file_sha256(archive)):
            print(f"reuse {label}", flush=True)
            return previous
    started = time.perf_counter()
    cpu_started = time.process_time()
    print(f"start {label}: {size}² t={duration} dt={dt}", flush=True)
    try:
        if model == "newtonian":
            dataset = simulate_periodic_ns(viscosity=.06, **common)
        else:
            dataset = simulate_viscoelastic_periodic(model=model, **physical, **common)
        arrays = {key: value for key, value in dataset.items()
                  if isinstance(value, np.ndarray) and key in ("velocity", "times", "conformation", "stress")}
        for key, array in arrays.items():
            if not np.all(np.isfinite(array)):
                raise FloatingPointError(f"nonfinite saved {key}")
        np.savez_compressed(archive, **arrays)
        summary = {"complete": True, "label": label, "config": config, "config_sha256": config_sha256(config), "source_hashes": hashes,
                   "shape": list(dataset["velocity"].shape), "metadata": dataset["metadata"],
                   "wall_seconds": time.perf_counter() - started, "cpu_seconds": time.process_time() - cpu_started,
                   "archive_sha256": file_sha256(archive), "archive_bytes": archive.stat().st_size,
                   "dx": dataset["dx"], "dy": dataset["dy"], "density": 1.,
                   "saved_fields": list(arrays), "constitutive_rhs_saved": False}
        summary["timing_scope"] = "simulation, finite-array checks, compressed NPZ write; metadata write and archive checksum computation excluded"
        write_json(manifest, summary)
        print(f"complete {label}: {summary['wall_seconds']:.1f}s", flush=True)
        return summary
    except BaseException as error:
        write_json(manifest, {"complete": False, "label": label, "config": config, "source_hashes": hashes,
                             "wall_seconds": time.perf_counter() - started, "error_type": type(error).__name__,
                             "error": str(error)})
        print(f"failed {label}: {error}", flush=True)
        raise


def load_arrays(output, label):
    with np.load(output / "datasets" / (label + ".npz")) as archive:
        return {key: archive[key] for key in archive.files}


def compare(value, reference):
    size = value["velocity"].shape[-1]
    np.testing.assert_allclose(value["times"], reference["times"], rtol=0, atol=2e-13)
    result = {"nx": size, "times": value["times"].tolist(), "error_reference": "finest-grid solution Fourier-restricted to the common strict component-wise 2/3 band"}
    identity = np.eye(3)[None, :, :, None, None]
    for key in ("velocity", "conformation", "stress"):
        if key not in value or key not in reference:
            continue
        fine = restrict(reference[key], size)
        # FENE-P physical stress is rational in C and may contain modes outside
        # the solver's C band. Both fields must be filtered for common modes.
        common_value = restrict(value[key], size)
        result[key + "_relative_l2"] = relative_series(common_value, fine)
        result[key + "_rms_error"] = rms_series(common_value - fine)
        prolonged = restrict(value[key], reference[key].shape[-1], retain_solver_band=False)
        result[key + "_full_field_relative_l2"] = relative_series(prolonged, reference[key])
        result[key + "_full_field_rms_error"] = rms_series(prolonged - reference[key])
        reference_peak_rms = max(rms_series(reference[key]))
        result[key + "_full_field_error_normalized_peak_reference_rms"] = [float(error / reference_peak_rms)
            if reference_peak_rms > 1e-13 else None for error in result[key + "_full_field_rms_error"]]
        if key == "conformation":
            result["excess_conformation_relative_l2"] = relative_series(common_value - identity, fine - identity)
            result["excess_conformation_full_field_relative_l2"] = relative_series(prolonged - identity, reference[key] - identity)
            peak_deformation_rms = max(rms_series(reference[key] - identity))
            result["excess_conformation_full_field_error_normalized_peak_reference_rms"] = [
                float(error / peak_deformation_rms) if peak_deformation_rms > 1e-13 else None
                for error in result[key + "_full_field_rms_error"]]
    omega = vorticity(value["velocity"])
    omega_ref = vorticity(restrict(reference["velocity"], size))
    result["vorticity_relative_l2"] = relative_series(omega, omega_ref)
    result["vorticity_rms_error"] = rms_series(omega - omega_ref)
    omega_full_ref = vorticity(reference["velocity"])
    omega_prolonged = vorticity(restrict(value["velocity"], reference["velocity"].shape[-1], retain_solver_band=False))
    result["vorticity_full_field_relative_l2"] = relative_series(omega_prolonged, omega_full_ref)
    result["vorticity_full_field_rms_error"] = rms_series(omega_prolonged - omega_full_ref)
    result["full_field_error_reference"] = "coarse computed solution Fourier-prolongated to finest grid; includes omitted fine scales, no filtered reference"
    # A separate full-finest-grid energy fraction describes the scales a
    # coarse simulation cannot retain; it is never called a grid error.
    result["reference_velocity_energy_outside_coarse_band"] = tail_fraction(reference["velocity"], size / 3)
    if "conformation" in reference:
        result["reference_deformation_energy_outside_coarse_band"] = tail_fraction(reference["conformation"] - identity, size / 3)
        result["reference_stress_energy_outside_coarse_band"] = tail_fraction(reference["stress"], size / 3)
    return result


def save_figure(fig, output, name):
    fig.savefig(output / (name + ".pdf"), bbox_inches="tight")
    fig.savefig(output / (name + ".png"), dpi=190, bbox_inches="tight")
    plt.close(fig)


def compact_publication_figures(output, finest):
    """Three-state companions with type sized for a thesis PDF page."""
    times = finest["oldroyd_b"]["times"]
    indices = [0, int(np.argmin(np.abs(times - times[-1] / 2))), len(times) - 1]
    extent = [0, 2 * np.pi, 0, 2 * np.pi]
    models = ["newtonian", "oldroyd_b", "fene_p"]
    omega = {model: vorticity(finest[model]["velocity"]) for model in models}
    limit = max(float(np.max(np.abs(field))) for field in omega.values())
    with plt.rc_context({"font.size": 15, "axes.labelsize": 14, "axes.titlesize": 15,
                         "xtick.labelsize": 13, "ytick.labelsize": 13}):
        fig, axes = plt.subplots(3, 3, figsize=(10.5, 8.5), layout="constrained")
        for row, model in enumerate(models):
            for col, index in enumerate(indices):
                axis = axes[row, col]
                picture = axis.imshow(omega[model][index], origin="lower", extent=extent,
                                      cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
                if row == 0:
                    axis.set_title(f"t={times[index]:.1f}")
                if col == 0:
                    axis.set_ylabel(MODEL_LABEL[model] + "\ny")
                if row == 2:
                    axis.set_xlabel("x")
                axis.set_xticks([0, np.pi, 2 * np.pi], ["0", "π", "2π"] if row == 2 else ["", "", ""])
                axis.set_yticks([0, np.pi, 2 * np.pi], ["0", "π", "2π"] if col == 0 else ["", "", ""])
        fig.colorbar(picture, ax=axes, shrink=.8, label="Vorticidad ω; escala común")
        size = finest["oldroyd_b"]["velocity"].shape[-1]
        fig.suptitle(f"Evolución calculada en {size}²: mismo campo inicial", fontsize=16)
        save_figure(fig, output, "compact_vorticity_evolution")

        polymer_models = ["oldroyd_b", "fene_p"]
        fields = {"trace": {m: np.trace(finest[m]["conformation"], axis1=1, axis2=2) - 3
                            for m in polymer_models},
                  "stress": {m: finest[m]["stress"][:, 0, 1] for m in polymer_models}}
        for kind, values in fields.items():
            field_limit = max(float(np.max(np.abs(field))) for field in values.values())
            fig, axes = plt.subplots(2, 3, figsize=(10.5, 6), layout="constrained")
            for row, model in enumerate(polymer_models):
                for col, index in enumerate(indices):
                    axis = axes[row, col]
                    picture = axis.imshow(values[model][index], origin="lower", extent=extent,
                        cmap="magma" if kind == "trace" else "RdBu_r",
                        vmin=0 if kind == "trace" else -field_limit, vmax=field_limit, interpolation="nearest")
                    if row == 0:
                        axis.set_title(f"t={times[index]:.1f}")
                    if col == 0:
                        axis.set_ylabel(MODEL_LABEL[model] + "\ny")
                    if row == 1:
                        axis.set_xlabel("x")
                    axis.set_xticks([0, np.pi, 2 * np.pi], ["0", "π", "2π"] if row == 1 else ["", "", ""])
                    axis.set_yticks([0, np.pi, 2 * np.pi], ["0", "π", "2π"] if col == 0 else ["", "", ""])
            fig.colorbar(picture, ax=axes, shrink=.8,
                         label="tr(C)−3; escala común" if kind == "trace" else "τxy; escala común")
            fig.suptitle("Deformación del polímero" if kind == "trace" else "Memoria y esfuerzo polimérico", fontsize=16)
            save_figure(fig, output, "compact_polymer_" + kind + "_evolution")


def compact_publication_checks(output, finest, spatial):
    """Readable full-field and spectral companions for the printed document."""
    times = finest["oldroyd_b"]["times"]
    size = finest["oldroyd_b"]["velocity"].shape[-1]
    with plt.rc_context({"font.size": 14, "axes.labelsize": 14, "axes.titlesize": 15,
                         "xtick.labelsize": 12, "ytick.labelsize": 12}):
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4), layout="constrained")
        metric_keys = ["velocity", "vorticity", "excess_conformation", "stress"]
        for model in ["oldroyd_b", "fene_p"]:
            for row in spatial[model]:
                linestyle = "--" if row["nx"] == min(r["nx"] for r in spatial[model]) else "-"
                for axis, key in zip(axes.flat, metric_keys):
                    values = [np.nan if value is None else max(value, 1e-17)
                              for value in row[key + "_full_field_relative_l2"]]
                    # t=0 is retained in JSON; exclude it from log plots because
                    # equal initial conditions give roundoff/undefined errors.
                    axis.semilogy(times[1:], values[1:], linestyle,
                                  label=f"{MODEL_LABEL[model]} {row['nx']}²")
        for axis, title in zip(axes.flat, ["Velocidad u", "Vorticidad ω", "Conformación C−I", "Esfuerzo τ"]):
            axis.set_title(title); axis.set_ylabel("Error relativo del campo completo")
            axis.set_xlabel("Tiempo adimensional"); axis.grid(alpha=.25); axis.legend(fontsize=10.5)
        fig.suptitle(f"Cada solución gruesa prolongada frente a {size}² sin filtrar\n"
                     "Se muestran estados posteriores a la condición inicial común", fontsize=15)
        save_figure(fig, output, "compact_full_field_refinement")

        fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4), layout="constrained")
        for model in ["oldroyd_b", "fene_p"]:
            for grid_size in sorted([r["nx"] for r in spatial[model]] + [size]):
                dataset = finest[model] if grid_size == size else load_arrays(output, f"{model}_{grid_size}")
                fields = [dataset["velocity"][-1], vorticity(dataset["velocity"][-1:])[0],
                          dataset["conformation"][-1] - np.eye(3)[:, :, None, None], dataset["stress"][-1]]
                for axis, field in zip(axes.flat, fields):
                    shells, power = spectra(field)
                    retained = (shells >= 1) & (power > 1e-30)
                    axis.loglog(shells[retained], power[retained],
                                label=f"{MODEL_LABEL[model]} {grid_size}²")
        for axis, title in zip(axes.flat, ["Velocidad u", "Vorticidad ω", "Conformación C−I", "Esfuerzo τ"]):
            axis.set_title(title); axis.set_ylabel("Potencia por anillo espectral")
            axis.set_xlabel("Número de onda radial k"); axis.grid(alpha=.25)
        axes.flat[-1].legend(fontsize=10)
        fig.suptitle(f"Espectros finales calculados en t={times[-1]:.1f}\n"
                     "La banda retenida y los defectos constitutivos se documentan por separado", fontsize=15)
        save_figure(fig, output, "compact_spectral_resolution")


def make_figures(output, finest, metadata, spatial, temporal):
    labels = ["newtonian", "oldroyd_b", "fene_p"]
    times = finest["oldroyd_b"]["times"]
    indices = np.linspace(0, len(times) - 1, 5).round().astype(int)
    omega = {model: vorticity(finest[model]["velocity"]) for model in labels}
    omega_limit = max(np.max(np.abs(field)) for field in omega.values())
    extent = [0, 2 * np.pi, 0, 2 * np.pi]
    fig, axes = plt.subplots(3, 5, figsize=(15, 8.7), layout="constrained")
    for row, model in enumerate(labels):
        for col, index in enumerate(indices):
            axis = axes[row, col]
            picture = axis.imshow(omega[model][index], origin="lower", extent=extent,
                                  cmap="RdBu_r", vmin=-omega_limit, vmax=omega_limit)
            u = finest[model]["velocity"][index]
            coordinates = np.arange(u.shape[-1]) * 2 * np.pi / u.shape[-1]
            axis.streamplot(coordinates, coordinates, u[0], u[1], color="k", density=.55,
                            linewidth=.35, arrowsize=.45)
            if row == 0:
                axis.set_title(f"t={times[index]:.2f}")
            if col == 0:
                axis.set_ylabel(MODEL_LABEL[model] + "\ny")
            if row == 2:
                axis.set_xlabel("x")
            axis.set_xticks([0, np.pi, 2 * np.pi], ["0", "π", "2π"])
            axis.set_yticks([0, np.pi, 2 * np.pi], ["0", "π", "2π"])
    fig.colorbar(picture, ax=axes, shrink=.7, label="Vorticidad ω [1/t], escala común")
    size = finest["oldroyd_b"]["velocity"].shape[-1]
    fig.suptitle(f"Evolución calculada de vórtices periódicos: {size}×{size}, misma condición inicial\n"
                 "Líneas de corriente instantáneas; no representan trayectorias de partículas", fontsize=13)
    save_figure(fig, output, "vorticity_evolution")

    trace = {model: np.trace(finest[model]["conformation"], axis1=1, axis2=2) - 3
             for model in ["oldroyd_b", "fene_p"]}
    tau_xy = {model: finest[model]["stress"][:, 0, 1] for model in trace}
    trace_limit = max(float(field.max()) for field in trace.values())
    stress_limit = max(float(np.abs(field).max()) for field in tau_xy.values())
    fig, axes = plt.subplots(4, 5, figsize=(15, 11), layout="constrained")
    trace_axes, stress_axes = [], []
    for row, (model, kind) in enumerate([(m, k) for m in trace for k in ("trace", "stress")]):
        for col, index in enumerate(indices):
            axis = axes[row, col]
            if kind == "trace":
                picture_trace = axis.imshow(trace[model][index], origin="lower", extent=extent,
                                            cmap="magma", vmin=0, vmax=trace_limit)
                trace_axes.append(axis)
            else:
                picture_stress = axis.imshow(tau_xy[model][index], origin="lower", extent=extent,
                                             cmap="RdBu_r", vmin=-stress_limit, vmax=stress_limit)
                stress_axes.append(axis)
            if row == 0:
                axis.set_title(f"t={times[index]:.2f}")
            if col == 0:
                axis.set_ylabel(MODEL_LABEL[model] + ("\ntr(C)−3" if kind == "trace" else "\nτxy"))
            if row == 3:
                axis.set_xlabel("x")
            axis.set_xticks([0, np.pi, 2 * np.pi], ["0", "π", "2π"])
            axis.set_yticks([0, np.pi, 2 * np.pi], ["0", "π", "2π"])
    fig.colorbar(picture_trace, ax=trace_axes, shrink=.7, label="Exceso de traza de conformación")
    fig.colorbar(picture_stress, ax=stress_axes, shrink=.7, label="Esfuerzo polimérico de corte τxy")
    fig.suptitle("Deformación y memoria del polímero; escalas comunes entre modelos y tiempos", fontsize=13)
    save_figure(fig, output, "polymer_evolution")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for model in labels:
        dataset = finest[model]
        kinetic = .5 * np.mean(np.sum(dataset["velocity"] ** 2, axis=1), axis=(1, 2))
        axes[0, 0].plot(times, kinetic, label=MODEL_LABEL[model])
        if model != "newtonian":
            diagnostics = metadata[model]["metadata"]["diagnostics"]
            polymer = np.asarray(diagnostics["polymer_free_energy"])
            axes[0, 0].plot(times, kinetic + polymer, "--", label=MODEL_LABEL[model] + " total")
            axes[0, 1].plot(times, diagnostics["polymer_stress_work"], label=MODEL_LABEL[model] + " trabajo τ:∇u")
            axes[0, 1].plot(times, diagnostics["polymer_relaxation_dissipation"], "--", label=MODEL_LABEL[model] + " relajación")
            axes[1, 0].plot(times, diagnostics["max_conformation_trace"], label=MODEL_LABEL[model] + " tr(C) máxima")
            axes[1, 0].plot(times, diagnostics["min_conformation_eigenvalue"], "--", label=MODEL_LABEL[model] + " λmin(C)")
            axes[1, 1].semilogy(times, np.maximum(np.abs(diagnostics["total_energy_rate_residual"]), 1e-18), label=MODEL_LABEL[model] + " balance total")
            axes[1, 1].semilogy(times, np.maximum(metadata[model]["metadata"]["constitutive_spatial_projection_defect_rms"], 1e-18), "--", label=MODEL_LABEL[model] + " defecto proyección C")
    axes[0, 0].set_ylabel("Energía media por unidad de masa")
    axes[0, 1].set_ylabel("Tasa media de intercambio/disipación")
    axes[1, 0].set_ylabel("Admisibilidad de C: cantidades distintas")
    axes[1, 1].set_ylabel("Residuos absolutos [unidades distintas]")
    for axis in axes.flat:
        axis.set_xlabel("Tiempo adimensional"); axis.grid(alpha=.25); axis.legend(fontsize=8)
    fig.suptitle("Balance energético, memoria y defectos numéricos medidos\n"
                 "Sin difusión artificial de C; tr(C)<12 en FENE-P y autovalores positivos", fontsize=12)
    save_figure(fig, output, "energy_memory_and_residuals")

    fig, axes = plt.subplots(2, 4, figsize=(15.5, 8), layout="constrained")
    for model in ["oldroyd_b", "fene_p"]:
        for row in spatial[model]:
            style = "--" if row["nx"] == min(r["nx"] for r in spatial[model]) else "-"
            for col, key in enumerate(["velocity", "vorticity", "excess_conformation", "stress"]):
                for row_index, suffix in enumerate(["_relative_l2", "_full_field_relative_l2"]):
                    values = [np.nan if value is None else max(value, 1e-17) for value in row[key + suffix]]
                    axes[row_index, col].semilogy(times, values, style, label=f"{MODEL_LABEL[model]} {row['nx']}²")
    for col, ylabel in enumerate(["Velocidad u", "Vorticidad ω", "Conformación C−I", "Esfuerzo τ"]):
        axes[0, col].set_title(ylabel)
        for row_index in [0, 1]:
            axis = axes[row_index, col]
            axis.set_xlabel("Tiempo adimensional"); axis.grid(alpha=.25)
            axis.legend(fontsize=7.5)
        axes[0, col].set_ylabel("Error relativo: modos comunes")
        axes[1, col].set_ylabel("Error relativo: campo fino completo")
    fig.suptitle(f"Refinamiento espacial frente a {size}² durante toda la evolución\n"
                 "Arriba: bandas comunes filtradas; abajo: prolongación gruesa frente al campo fino sin filtrar", fontsize=12)
    save_figure(fig, output, "grid_refinement")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), layout="constrained")
    for model in ["oldroyd_b", "fene_p"]:
        for grid_size in sorted([r["nx"] for r in spatial[model]] + [size]):
            label = f"{MODEL_LABEL[model]} {grid_size}²"
            dataset = finest[model] if grid_size == size else load_arrays(output, f"{model}_{grid_size}")
            for axis, field in zip(axes, [dataset["velocity"][-1], vorticity(dataset["velocity"][-1:])[0],
                                        dataset["conformation"][-1] - np.eye(3)[:, :, None, None], dataset["stress"][-1]]):
                shells, power = spectra(field)
                mask = (shells >= 1) & (power > 1e-30)
                axis.loglog(shells[mask], power[mask], label=label)
    for axis, ylabel in zip(axes, ["Potencia de velocidad por anillo", "Potencia de vorticidad por anillo", "Potencia de C−I por anillo", "Potencia de esfuerzo τ por anillo"]):
        axis.set_xlabel("Número de onda radial k"); axis.set_ylabel(ylabel); axis.grid(alpha=.25)
    axes[-1].legend(fontsize=8)
    fig.suptitle("Espectro final y solapamiento de escalas resueltas; no se asume turbulencia", fontsize=12)
    save_figure(fig, output, "spectral_resolution")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), layout="constrained")
    images = [axis.imshow(omega[model][0], origin="lower", extent=extent, cmap="RdBu_r",
                         vmin=-omega_limit, vmax=omega_limit) for axis, model in zip(axes, labels)]
    for axis, model in zip(axes, labels):
        axis.set_title(MODEL_LABEL[model]); axis.set_xlabel("x"); axis.set_ylabel("y")
    fig.colorbar(images[0], ax=axes, label="Vorticidad ω, escala común")
    title = fig.suptitle("")
    def update(index):
        for picture, model in zip(images, labels):
            picture.set_data(omega[model][index])
        title.set_text(f"Vórtices calculados en {size}²; t={times[index]:.2f}; mismo estado inicial")
        return images + [title]
    animation = FuncAnimation(fig, update, frames=len(times), interval=160, blit=False)
    animation.save(output / "vorticity_evolution.gif", writer=PillowWriter(fps=6), dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(9, 8), layout="constrained")
    polymer_images = []
    for row, model in enumerate(trace):
        polymer_images.append(axes[row, 0].imshow(trace[model][0], origin="lower", extent=extent, cmap="magma", vmin=0, vmax=trace_limit))
        polymer_images.append(axes[row, 1].imshow(tau_xy[model][0], origin="lower", extent=extent, cmap="RdBu_r", vmin=-stress_limit, vmax=stress_limit))
        axes[row, 0].set_title(MODEL_LABEL[model] + ": tr(C)−3")
        axes[row, 1].set_title(MODEL_LABEL[model] + ": τxy")
    fig.colorbar(polymer_images[0], ax=axes[:, 0], shrink=.8, label="tr(C)−3")
    fig.colorbar(polymer_images[1], ax=axes[:, 1], shrink=.8, label="τxy")
    title = fig.suptitle("")
    def update_polymer(index):
        for row, model in enumerate(trace):
            polymer_images[row * 2].set_data(trace[model][index])
            polymer_images[row * 2 + 1].set_data(tau_xy[model][index])
        title.set_text(f"Memoria y esfuerzo calculados en {size}²; t={times[index]:.2f}")
        return polymer_images + [title]
    animation = FuncAnimation(fig, update_polymer, frames=len(times), interval=160, blit=False)
    animation.save(output / "polymer_evolution.gif", writer=PillowWriter(fps=6), dpi=110)
    plt.close(fig)
    compact_publication_figures(output, finest)
    compact_publication_checks(output, finest, spatial)


def analyze(output, sizes, amplitude, duration, dt, snapshots):
    finest_size = max(sizes)
    spatial, temporal, metadata, finest = {}, {}, {}, {}
    for model in ["oldroyd_b", "fene_p"]:
        reference = load_arrays(output, f"{model}_{finest_size}")
        finest[model] = reference
        metadata[model] = json.loads((output / "datasets" / f"{model}_{finest_size}.json").read_text())
        spatial[model] = []
        for size in sizes[:-1]:
            candidate = load_arrays(output, f"{model}_{size}")
            spatial[model].append(compare(candidate, reference))
            initial = restrict(reference["velocity"][:1], size)
            np.testing.assert_allclose(candidate["velocity"][0], initial[0], atol=2e-8, rtol=2e-8)
        temporal_size = sizes[-2]
        temporal[model] = compare(load_arrays(output, f"{model}_{temporal_size}"),
                                  load_arrays(output, f"{model}_{temporal_size}_dt_half"))
        temporal[model]["error_reference"] = "same grid at half nominal timestep, same output times; adaptive internal caps reported in individual metadata"
    finest["newtonian"] = load_arrays(output, f"newtonian_{finest_size}")
    metadata["newtonian"] = json.loads((output / "datasets" / f"newtonian_{finest_size}.json").read_text())
    for model in ["oldroyd_b", "fene_p"]:
        np.testing.assert_allclose(finest[model]["velocity"][0], finest["newtonian"]["velocity"][0], atol=3e-14, rtol=0)
        np.testing.assert_allclose(finest[model]["times"], finest["newtonian"]["times"], atol=1e-13, rtol=0)
    archive_verification = {}
    for manifest_path in sorted((output / "datasets").glob("*.json")):
        manifest = json.loads(manifest_path.read_text())
        if not manifest.get("complete"):
            raise RuntimeError(f"simulation failed: {manifest_path.name}")
        archive_path = manifest_path.with_suffix(".npz")
        if not archive_path.exists() or file_sha256(archive_path) != manifest.get("archive_sha256"):
            raise RuntimeError(f"archive checksum mismatch: {archive_path}")
        archive_verification[manifest_path.stem] = manifest["archive_sha256"]
        if "timing_scope" not in manifest or "config_sha256" not in manifest:
            manifest["timing_scope"] = "simulation, finite-array checks, compressed NPZ write; metadata write and archive checksum computation excluded"
            manifest["config_sha256"] = config_sha256(manifest["config"])
            write_json(manifest_path, manifest)
    summary = {"complete": True, "grids": sizes, "duration": duration, "snapshots": snapshots,
               "amplitude": amplitude, "nominal_dt": dt, "nominal_half_dt": dt / 2,
               "same_initial_velocity_at_finest": True, "same_times": True, "verified_archives": archive_verification,
               "analysis_script_sha256": file_sha256(Path(__file__)),
               "models": {}, "spatial_refinement": spatial, "temporal_refinement": temporal,
               "interpretation": "Mesh representativeness is evidenced by field/spectral/energy checks for this finite moderate-parameter periodic case. A larger mesh alone is insufficient. No cylinder wake, chaos, high-Wi or universal DNS certification.",
               "constitutive_difference_interpretation": "Differences from matched Newtonian nu=.06 are material dynamics, not reconstruction error; initial C=I has zero polymer stress, while Newtonian stress acts immediately."}
    identity = np.eye(3)[None, :, :, None, None]
    rows = []
    for model in ["oldroyd_b", "fene_p"]:
        dataset, manifest = finest[model], metadata[model]
        diagnostics = manifest["metadata"]["diagnostics"]
        velocity_difference = relative_series(dataset["velocity"], finest["newtonian"]["velocity"])
        omega_difference = relative_series(vorticity(dataset["velocity"]), vorticity(finest["newtonian"]["velocity"]))
        summary["models"][model] = {"weissenberg_reference": manifest["metadata"]["weissenberg_reference"],
            "reynolds_reference": manifest["metadata"]["reynolds_reference"],
            "stage_min_conformation_eigenvalue": manifest["metadata"]["stage_min_conformation_eigenvalue"],
            "stage_max_conformation_trace": manifest["metadata"]["stage_max_conformation_trace"],
            "finest_velocity_spectral_tail_fraction": tail_fraction(dataset["velocity"]),
            "finest_deformation_spectral_tail_fraction": tail_fraction(dataset["conformation"] - identity),
            "finest_stress_spectral_tail_fraction": tail_fraction(dataset["stress"]),
            "velocity_difference_vs_newtonian_relative_l2": velocity_difference,
            "vorticity_difference_vs_newtonian_relative_l2": omega_difference,
            "total_energy_rate_residual_absolute_max": float(np.max(np.abs(diagnostics["total_energy_rate_residual"]))),
            "constitutive_spatial_projection_defect_rms_max": float(np.max(manifest["metadata"]["constitutive_spatial_projection_defect_rms"])),
            "snapshot_constitutive_pde_residual_rms_max": float(np.max(manifest["metadata"]["constitutive_pde_residual_rms"])),
            "snapshot_time_residual_caveat": "Second-order time derivative from coarse output snapshots; includes time-differencing error and is not the internal RK local error."}
        total_energy = np.asarray(diagnostics["kinetic_energy"]) + np.asarray(diagnostics["polymer_free_energy"])
        dissipation = np.asarray(diagnostics["solvent_dissipation"]) + np.asarray(diagnostics["polymer_relaxation_dissipation"])
        summary["models"][model]["maximum_positive_total_energy_snapshot_increment"] = max(0., float(np.diff(total_energy).max()))
        summary["models"][model]["total_energy_rate_residual_relative_dissipation_peak"] = float(
            np.max(np.abs(diagnostics["total_energy_rate_residual"])) / max(float(np.max(dissipation)), 1e-30))
        summary["models"][model]["temporal_actual_dt_primary"] = {
            key: json.loads((output / "datasets" / f"{model}_{sizes[-2]}.json").read_text())["metadata"][key]
            for key in ("internal_dt_min", "internal_dt_max", "stability_limited_steps", "rejected_steps", "internal_steps")}
        summary["models"][model]["temporal_actual_dt_half_reference"] = {
            key: json.loads((output / "datasets" / f"{model}_{sizes[-2]}_dt_half.json").read_text())["metadata"][key]
            for key in ("internal_dt_min", "internal_dt_max", "stability_limited_steps", "rejected_steps", "internal_steps")}
        for row in spatial[model]:
            rows.append({"model": model, "nx": row["nx"], "reference_nx": finest_size,
                         "final_velocity_error": row["velocity_relative_l2"][-1],
                         "max_velocity_error": max(v for v in row["velocity_relative_l2"] if v is not None),
                         "final_vorticity_error": row["vorticity_relative_l2"][-1],
                         "max_vorticity_error": max(v for v in row["vorticity_relative_l2"] if v is not None),
                         "final_excess_conformation_error": row["excess_conformation_relative_l2"][-1],
                         "max_excess_conformation_error": max(v for v in row["excess_conformation_relative_l2"] if v is not None),
                         "final_stress_error": row["stress_relative_l2"][-1],
                         "max_stress_error": max(v for v in row["stress_relative_l2"] if v is not None)})
            for key in ("velocity", "vorticity", "excess_conformation", "stress"):
                full_errors = row[key + "_full_field_relative_l2"]
                rows[-1]["max_full_field_" + key + "_error"] = max(v for v in full_errors if v is not None)
    with (output / "grid_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    write_json(output / "summary.json", summary)
    make_figures(output, finest, metadata, spatial, temporal)
    explanation = ["# Vórtices viscoelásticos: refinamiento y evolución visual", "",
        f"Se ejecutaron Oldroyd-B y FENE-P en {', '.join(str(n)+'²' for n in sizes)}, "
        f"con {snapshots} estados entre t=0 y {duration}, y una verificación dt/2 en {sizes[-2]}² por modelo.", "",
        "El control newtoniano usa la misma velocidad inicial, densidad, dominio, malla fina y tiempos. "
        "La diferencia constitutiva es un resultado material, no error de reconstrucción. "
        "Se conserva C tridimensional en flujo plano, κ=0 y rechazo de pasos que violen SPD/traza; no se repara C.", "",
        "Las curvas comparan modos Fourier comunes y el JSON/CSV también compara cada solución gruesa "
        "prolongada al campo fino completo sin filtrar la referencia; registra aparte la energía que la malla gruesa no representa. "
        "La representatividad se limita a este régimen periódico moderado, con 25 instantes almacenados; "
        "no demuestra estelas de cilindros, caos ni DNS universal.", "",
        "| Modelo | Malla | Máx. error u | Máx. error ω | Máx. error C−I | Máx. error τ |", "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        explanation.append(f"| {MODEL_LABEL[row['model']]} | {row['nx']}² | {row['max_velocity_error']:.3e} | {row['max_vorticity_error']:.3e} | {row['max_excess_conformation_error']:.3e} | {row['max_stress_error']:.3e} |")
    explanation += ["", "## Figuras y datos", "",
        "- `vorticity_evolution.pdf/png/gif`: evolución real y líneas de corriente instantáneas; escala de color común.",
        "- `polymer_evolution.pdf/png/gif`: tr(C)−3 y τxy; escalas comunes por cantidad.",
        "- `energy_memory_and_residuals.pdf/png`: intercambio energía cinética/polimérica y defectos medidos.",
        "- `grid_refinement.pdf/png`, `spectral_resolution.pdf/png`: refinamiento y espectros finales.",
        "- `compact_vorticity_evolution.pdf/png`, `compact_polymer_trace_evolution.pdf/png`, `compact_polymer_stress_evolution.pdf/png`: tres tiempos con letra mayor para páginas del documento.",
        "- `compact_full_field_refinement.pdf/png`, `compact_spectral_resolution.pdf/png`: controles de resolución legibles en papel.",
        "- `summary.json`, `grid_summary.csv`, `datasets/*json`: series, admisibilidad, parámetros, hashes, tiempos y fallos.",
        "- `datasets/*.npz`: campos calculados completos de velocidad, C y esfuerzo; regenerables mediante este script.", "",
        "Los residuos obtenidos por diferencias temporales de snapshots dependen de Δt_salida, "
        "y se distinguen del defecto espacial de proyección y del balance instantáneo del operador semidiscreto.", ""]
    (output / "README.md").write_text("\n".join(explanation))
    write_json(output / "completion.json", {"complete": True, "simulations": len(sizes) * 2 + 3,
               "models": 3, "grids": sizes, "snapshots": snapshots, "duration": duration,
               "source_hashes": source_hashes(), "figures_pdf": 10, "animations_gif": 2})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--grids", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--duration", type=float, default=2.4)
    parser.add_argument("--amplitude", type=float, default=.8)
    parser.add_argument("--dt", type=float, default=.004)
    parser.add_argument("--snapshots", type=int, default=25)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if len(args.grids) < 3 or sorted(set(args.grids)) != args.grids:
        raise ValueError("three or more strictly increasing grids required")
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy(); config["output"] = str(output)
    provenance = {"config": config, "config_sha256": config_sha256(config), "source_hashes": source_hashes(),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip(),
        "reference_scope": "same periodic vortex initial condition, units nondimensional/rho=1, Oldroyd-B/FENE-P full 3x3 C, no stress diffusion, unmodified numerical sources"}
    if args.analyze_only and (output / "provenance.json").exists():
        original = json.loads((output / "provenance.json").read_text())
        original["analysis_config"] = config
        original["analysis_config_sha256"] = config_sha256(config)
        original["analysis_source_hashes"] = provenance["source_hashes"]
        provenance = original
    write_json(output / "provenance.json", provenance)
    if not args.analyze_only:
        jobs = [(str(output), model, size, args.dt, args.duration, args.amplitude, args.snapshots, f"{model}_{size}")
                for size in reversed(args.grids) for model in ["oldroyd_b", "fene_p"]]
        jobs += [(str(output), model, args.grids[-2], args.dt / 2, args.duration, args.amplitude, args.snapshots,
                  f"{model}_{args.grids[-2]}_dt_half") for model in ["oldroyd_b", "fene_p"]]
        jobs.append((str(output), "newtonian", max(args.grids), args.dt, args.duration, args.amplitude, args.snapshots, f"newtonian_{max(args.grids)}"))
        # Spawn avoids inheriting a partially initialized numerical library.
        import multiprocessing
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as executor:
            futures = {executor.submit(run_case, job): job[-1] for job in jobs}
            for future in as_completed(futures):
                future.result()
    analyze(output, args.grids, args.amplitude, args.duration, args.dt, args.snapshots)
    print(f"done {output}", flush=True)


if __name__ == "__main__":
    main()
