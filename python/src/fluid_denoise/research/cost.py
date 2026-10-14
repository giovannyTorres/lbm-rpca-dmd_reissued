"""Repeatable computational cost reporting without a hardware-dependent score.

Run each measured repetition in a fresh child process to make process RSS
peaks comparable. ``measure_call`` cannot reset the OS historical RSS peak.
"""
from __future__ import annotations

import contextlib
import io
import os
import platform
import resource
import sys
import time
import tracemalloc
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np


def _rss_peak_bytes() -> int:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw if sys.platform == "darwin" else raw * 1024)


def hardware_metadata() -> dict[str, Any]:
    """Record architecture, numerical runtime, and effective BLAS thread pools."""
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        np.show_config()
    try:
        from threadpoolctl import threadpool_info
        pools = threadpool_info()
    except ImportError:
        pools = []
    return {"platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor(), "cpu_count": os.cpu_count(),
            "python": platform.python_version(), "numpy": np.__version__,
            "numpy_configuration": stream.getvalue(), "threadpools": pools,
            "thread_environment": {name: os.environ.get(name) for name in
                                   ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}}


# Stable runner-facing name; the helper describes the execution environment.
environment_metadata = hardware_metadata


def measure_call(function: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
    """Measure one call; RSS includes interpreter, dependencies, and input data.

    Python-traced peak reports allocations visible to tracemalloc, which is not
    an all-native allocator census. Tracing overhead affects both time measures.
    If tracing is already active it is left enabled and its peak is not reset.
    """
    rss_before = _rss_peak_bytes()
    already_tracing = tracemalloc.is_tracing()
    if not already_tracing:
        tracemalloc.start()
    traced_before, traced_peak_before = tracemalloc.get_traced_memory()
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    try:
        value = function(*args, **kwargs)
    finally:
        wall_seconds = time.perf_counter() - wall_start
        cpu_seconds = time.process_time() - cpu_start
        traced_current, traced_peak = tracemalloc.get_traced_memory()
        if not already_tracing:
            tracemalloc.stop()
    profile = {"wall_seconds": float(wall_seconds), "cpu_seconds": float(cpu_seconds),
               "rss_peak_bytes": _rss_peak_bytes(), "rss_peak_before_bytes": rss_before,
               "python_traced_peak_bytes": int(traced_peak),
               "python_traced_current_bytes": int(traced_current),
               "python_traced_before_bytes": int(traced_before),
               "python_traced_peak_before_bytes": int(traced_peak_before),
               "rss_scope": "absolute process lifetime peak; use a fresh subprocess per repetition",
               "rss_native_unit": "bytes" if sys.platform == "darwin" else "KiB converted to bytes",
               "tracemalloc_scope": "Python-visible allocations; includes measurement overhead, not all native memory",
               "tracemalloc_preexisting": already_tracing}
    return value, profile


def summarize_profiles(profiles: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Median and interquartile range over genuine independent repetitions."""
    if not profiles:
        raise ValueError("at least one profile is required")
    output: dict[str, Any] = {"repetitions": len(profiles), "iqr_definition": "75th minus 25th percentile, linear interpolation"}
    for key in ("wall_seconds", "cpu_seconds", "rss_peak_bytes", "python_traced_peak_bytes"):
        values = np.asarray([profile[key] for profile in profiles], dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError(f"{key} must be finite and nonnegative")
        q25, q75 = np.percentile(values, (25, 75))
        output.update({key + "_median": float(np.median(values)), key + "_iqr": float(q75 - q25),
                       key + "_q25": float(q25), key + "_q75": float(q75)})
    if len(profiles) < 3:
        output["warning"] = "Fewer than three repetitions; variability estimate is weak."
    return output


def operation_proxies(matrix_shape: Sequence[int], *, svd_calls: int = 0,
                      fft_shapes: Sequence[Sequence[int]] = ()) -> dict[str, Any]:
    """Diagnostic asymptotic work proxies, explicitly not measured FLOP counts.

    Full thin SVD proxy is m*n*min(m,n). FFT proxy is N*log2(N) per
    supplied transform shape. Algorithms with iterative/partial SVD must expose
    their actual ranks/calls separately instead of claiming this as a count.
    """
    if len(matrix_shape) != 2 or any(int(x) != x or x <= 0 for x in matrix_shape):
        raise ValueError("matrix_shape must contain two positive integers")
    if int(svd_calls) != svd_calls or svd_calls < 0:
        raise ValueError("svd_calls must be a nonnegative integer")
    m, n = map(int, matrix_shape)
    fft_work = 0.0
    for shape in fft_shapes:
        if not shape or any(int(x) != x or x <= 0 for x in shape):
            raise ValueError("FFT shapes must contain positive integers")
        size = int(np.prod(shape))
        fft_work += size * np.log2(size)
    return {"svd_full_thin_work_proxy": int(svd_calls * m * n * min(m, n)),
            "svd_calls": int(svd_calls), "fft_nlog2n_work_proxy": float(fft_work),
            "matrix_elements": int(m * n),
            "qualification": "Asymptotic diagnostic proxies with omitted constants; not FLOPs, energy, or elapsed time."}


def quality_gates(metrics: Mapping[str, Any], thresholds: Mapping[str, float]) -> dict[str, Any]:
    """Apply a priori upper thresholds. Missing/undefined metrics fail a gate."""
    gates = {}
    for key, threshold in thresholds.items():
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("thresholds must be finite and nonnegative")
        value = metrics.get(key)
        valid = isinstance(value, (int, float, np.number)) and np.isfinite(value)
        gates[key] = {"value": float(value) if valid else None, "maximum": float(threshold),
                      "passed": bool(valid and value <= threshold)}
    return {"passed": bool(gates) and all(gate["passed"] for gate in gates.values()),
            "gates": gates, "definition": "Predeclared upper limits; missing or undefined values fail."}


def pareto_front(records: Sequence[Mapping[str, Any]],
                 cost_keys: Sequence[str] = ("wall_seconds_median", "rss_peak_bytes_median"),
                 quality_key: str = "velocity_relative_l2") -> list[int]:
    """Indices of nondominated rows, minimizing each cost and reconstruction error.

    Invalid rows are excluded. Equal rows are retained; no arbitrary combined
    weighting or ranking is imposed. Feed only rows passing physical gates when
    a feasibility-constrained cost frontier is required.
    """
    keys = tuple(cost_keys) + (quality_key,)
    if not cost_keys:
        raise ValueError("at least one cost key is required")
    values: dict[int, np.ndarray] = {}
    for index, record in enumerate(records):
        try:
            row = np.asarray([record[key] for key in keys], dtype=float)
        except (KeyError, TypeError, ValueError):
            continue
        if np.all(np.isfinite(row)) and np.all(row >= 0):
            values[index] = row
    return [index for index, row in values.items()
            if not any(np.all(other <= row) and np.any(other < row)
                       for other_index, other in values.items() if other_index != index)]
