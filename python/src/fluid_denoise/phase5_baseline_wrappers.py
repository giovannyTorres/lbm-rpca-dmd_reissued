from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from fluid_denoise.phase4_model_data import (
    ModelRun,
    NormalizationSpec,
    SnapshotBatch,
    SpaceTimeMatrix,
    load_model_run,
    pack_snapshot_batch,
    pack_space_time_matrix,
    unpack_snapshot_batch,
    unpack_space_time_matrix,
)
from fluid_denoise.phase5_baseline_impl import (
    dmd_reconstruct,
    median_filter_2d,
    rpca_ialm,
    truncated_svd_reconstruct,
    wiener_filter_2d,
)


SUPPORTED_INPUT_KINDS = ("snapshot_batch", "space_time_matrix")
SUPPORTED_MODEL_NAMES = (
    "rpca_ialm",
    "dmd",
    "truncated_svd",
    "median_filter",
    "wiener_filter",
)


@dataclass(frozen=True)
class PreparedBaselineInput:
    input_kind: str
    run: ModelRun
    variables: tuple[str, ...]
    steps: np.ndarray
    data: np.ndarray
    payload: SnapshotBatch | SpaceTimeMatrix
    observed_grid: np.ndarray


def _normalize_normalization_config(config: dict[str, Any] | None) -> NormalizationSpec | None:
    if config is None:
        return None
    return NormalizationSpec(
        mode=str(config.get("mode", "none")),
        scope=str(config.get("scope", "global")),
        epsilon=float(config.get("epsilon", 1e-8)),
    )


def prepare_baseline_input(
    run_dir: Path | str,
    *,
    input_kind: str,
    variables: Sequence[str],
    steps: Sequence[int] | None = None,
    snapshot_indices: Sequence[int] | None = None,
    mask_policy: str | None = None,
    obstacle_fill_value: float = 0.0,
    include_mask_channel: bool = False,
    normalization: dict[str, Any] | NormalizationSpec | None = None,
) -> PreparedBaselineInput:
    normalized_kind = input_kind.strip().lower()
    if normalized_kind not in SUPPORTED_INPUT_KINDS:
        raise ValueError(f"Unsupported baseline input kind: {input_kind}")

    if isinstance(normalization, NormalizationSpec):
        normalization_spec = normalization
    else:
        normalization_spec = _normalize_normalization_config(normalization)

    run = load_model_run(run_dir)

    if normalized_kind == "snapshot_batch":
        payload = pack_snapshot_batch(
            run,
            variables=variables,
            snapshot_indices=snapshot_indices,
            steps=steps,
            mask_policy="keep" if mask_policy is None else mask_policy,
            obstacle_fill_value=obstacle_fill_value,
            include_mask_channel=include_mask_channel,
            normalization=normalization_spec,
        )
        observed_grid = unpack_snapshot_batch(payload)
        return PreparedBaselineInput(
            input_kind=normalized_kind,
            run=run,
            variables=payload.variables,
            steps=np.asarray(payload.steps),
            data=np.asarray(payload.data),
            payload=payload,
            observed_grid=np.asarray(observed_grid),
        )

    payload = pack_space_time_matrix(
        run,
        variables=variables,
        snapshot_indices=snapshot_indices,
        steps=steps,
        mask_policy="flatten_fluid" if mask_policy is None else mask_policy,
        obstacle_fill_value=obstacle_fill_value,
        normalization=normalization_spec,
    )
    observed_grid = unpack_space_time_matrix(payload)
    return PreparedBaselineInput(
        input_kind=normalized_kind,
        run=run,
        variables=payload.variables,
        steps=np.asarray(payload.steps),
        data=np.asarray(payload.data),
        payload=payload,
        observed_grid=np.asarray(observed_grid),
    )


def materialize_reconstruction_grid(
    prepared_input: PreparedBaselineInput,
    reconstructed_data: np.ndarray,
) -> np.ndarray:
    if prepared_input.input_kind == "snapshot_batch":
        payload = prepared_input.payload
        assert isinstance(payload, SnapshotBatch)
        reconstructed_payload = SnapshotBatch(
            run_dir=payload.run_dir,
            dataset_kind=payload.dataset_kind,
            experiment_id=payload.experiment_id,
            variables=payload.variables,
            steps=payload.steps,
            data=np.asarray(reconstructed_data, dtype=np.float64),
            mask=payload.mask,
            context=payload.context,
            normalization=payload.normalization,
        )
        return np.asarray(unpack_snapshot_batch(reconstructed_payload), dtype=np.float64)

    payload = prepared_input.payload
    assert isinstance(payload, SpaceTimeMatrix)
    reconstructed_payload = SpaceTimeMatrix(
        run_dir=payload.run_dir,
        dataset_kind=payload.dataset_kind,
        experiment_id=payload.experiment_id,
        variables=payload.variables,
        steps=payload.steps,
        data=np.asarray(reconstructed_data, dtype=np.float64),
        mask=payload.mask,
        context=payload.context,
        normalization=payload.normalization,
        feature_layout=payload.feature_layout,
    )
    return np.asarray(unpack_space_time_matrix(reconstructed_payload), dtype=np.float64)


class BaselineModel(ABC):
    def __init__(self, **params: Any) -> None:
        self._params = dict(params)
        self._fitted_input: PreparedBaselineInput | None = None
        self._reconstructed_data: np.ndarray | None = None
        self._fit_metadata: dict[str, Any] = {}
        self._warnings: list[str] = []

    def fit(self, prepared_input: PreparedBaselineInput) -> "BaselineModel":
        self._fitted_input = prepared_input
        self._warnings = list(self._estimate_warnings(prepared_input))
        self._reconstructed_data, self._fit_metadata = self._fit_impl(prepared_input)
        return self

    def reconstruct(self, prepared_input: PreparedBaselineInput | None = None) -> np.ndarray:
        if self._reconstructed_data is None or self._fitted_input is None:
            raise RuntimeError("The model must be fitted before calling reconstruct()")
        if prepared_input is not None and prepared_input.input_kind != self._fitted_input.input_kind:
            raise ValueError("Reconstruction input kind does not match the fitted input kind")
        return np.asarray(self._reconstructed_data, dtype=np.float64)

    def get_params(self) -> dict[str, Any]:
        return dict(self._params)

    def get_name(self) -> str:
        return self.NAME

    def get_warnings(self) -> tuple[str, ...]:
        return tuple(self._warnings)

    def get_fit_metadata(self) -> dict[str, Any]:
        return dict(self._fit_metadata)

    def _estimate_warnings(self, prepared_input: PreparedBaselineInput) -> list[str]:
        return []

    @abstractmethod
    def _fit_impl(
        self,
        prepared_input: PreparedBaselineInput,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        raise NotImplementedError


class RPCAIALMBaselineModel(BaselineModel):
    NAME = "rpca_ialm"

    def _estimate_warnings(self, prepared_input: PreparedBaselineInput) -> list[str]:
        warnings: list[str] = []
        if prepared_input.input_kind != "space_time_matrix":
            raise ValueError("RPCA-IALM requires a space_time_matrix input")
        if prepared_input.data.shape[1] < 3:
            warnings.append("RPCA-IALM has very few temporal samples; low-rank/sparse separation may be unstable.")
        return warnings

    def _fit_impl(self, prepared_input: PreparedBaselineInput) -> tuple[np.ndarray, dict[str, Any]]:
        result = rpca_ialm(
            prepared_input.data,
            lam=self._params.get("lam"),
            mu=self._params.get("mu"),
            rho=float(self._params.get("rho", 1.5)),
            max_iter=int(self._params.get("max_iter", 500)),
            tol=float(self._params.get("tol", 1e-7)),
        )
        metadata = {key: value for key, value in result.items() if key not in {"low_rank", "sparse"}}
        metadata["sparse_norm_fro"] = float(np.linalg.norm(result["sparse"], ord="fro"))
        return np.asarray(result["low_rank"], dtype=np.float64), metadata


class DMDBaselineModel(BaselineModel):
    NAME = "dmd"

    def _estimate_warnings(self, prepared_input: PreparedBaselineInput) -> list[str]:
        warnings: list[str] = []
        if prepared_input.input_kind != "space_time_matrix":
            raise ValueError("DMD requires a space_time_matrix input")
        if prepared_input.data.shape[1] < 4:
            warnings.append("DMD with fewer than four snapshots gives a very weak dynamical estimate.")
        if prepared_input.payload.context.mask_policy != "flatten_fluid":
            warnings.append("DMD on full-grid features may entangle obstacle cells with fluid dynamics.")
        return warnings

    def _fit_impl(self, prepared_input: PreparedBaselineInput) -> tuple[np.ndarray, dict[str, Any]]:
        result = dmd_reconstruct(
            prepared_input.data,
            rank=self._params.get("rank"),
            exact=bool(self._params.get("exact", True)),
        )
        metadata = {
            "effective_rank": int(result["effective_rank"]),
            "num_eigenvalues": int(result["eigenvalues"].shape[0]),
            "eigenvalues_real": [float(value.real) for value in result["eigenvalues"]],
            "eigenvalues_imag": [float(value.imag) for value in result["eigenvalues"]],
        }
        return np.asarray(result["reconstruction"], dtype=np.float64), metadata


class TruncatedSVDBaselineModel(BaselineModel):
    NAME = "truncated_svd"

    def _estimate_warnings(self, prepared_input: PreparedBaselineInput) -> list[str]:
        warnings: list[str] = []
        if prepared_input.input_kind != "space_time_matrix":
            raise ValueError("Truncated SVD requires a space_time_matrix input")
        if prepared_input.data.shape[1] < 2:
            warnings.append("Truncated SVD with a single time sample reduces to a static projection.")
        return warnings

    def _fit_impl(self, prepared_input: PreparedBaselineInput) -> tuple[np.ndarray, dict[str, Any]]:
        result = truncated_svd_reconstruct(
            prepared_input.data,
            rank=int(self._params.get("rank", 2)),
        )
        metadata = {
            "effective_rank": int(result["effective_rank"]),
            "retained_energy_ratio": float(result["retained_energy_ratio"]),
            "singular_values": [float(value) for value in result["singular_values"]],
        }
        return np.asarray(result["reconstruction"], dtype=np.float64), metadata


class MedianFilterBaselineModel(BaselineModel):
    NAME = "median_filter"

    def _estimate_warnings(self, prepared_input: PreparedBaselineInput) -> list[str]:
        warnings: list[str] = []
        if prepared_input.input_kind != "snapshot_batch":
            raise ValueError("Median filter requires a snapshot_batch input")
        payload = prepared_input.payload
        assert isinstance(payload, SnapshotBatch)
        if payload.context.layout != "batch_channel_y_x":
            raise ValueError("Median filter requires grid snapshots, not flattened fluid vectors")
        if payload.context.mask_policy == "keep":
            warnings.append("Median filter with mask_policy='keep' may mix obstacle values into local neighborhoods.")
        return warnings

    def _fit_impl(self, prepared_input: PreparedBaselineInput) -> tuple[np.ndarray, dict[str, Any]]:
        kernel_size = self._params.get("kernel_size", 3)
        padding_mode = str(self._params.get("padding_mode", "reflect"))
        payload = prepared_input.payload
        assert isinstance(payload, SnapshotBatch)
        source = prepared_input.data[:, : payload.context.variable_channel_count, :, :]
        filtered = np.empty_like(source, dtype=np.float64)
        for snapshot_index in range(source.shape[0]):
            for channel_index in range(source.shape[1]):
                filtered[snapshot_index, channel_index] = median_filter_2d(
                    source[snapshot_index, channel_index],
                    kernel_size=kernel_size,
                    padding_mode=padding_mode,
                )
        if payload.context.include_mask_channel:
            filtered = np.concatenate([filtered, prepared_input.data[:, payload.context.variable_channel_count :, :, :]], axis=1)
        metadata = {
            "kernel_size": kernel_size,
            "padding_mode": padding_mode,
        }
        return filtered, metadata


class WienerFilterBaselineModel(BaselineModel):
    NAME = "wiener_filter"

    def _estimate_warnings(self, prepared_input: PreparedBaselineInput) -> list[str]:
        warnings: list[str] = []
        if prepared_input.input_kind != "snapshot_batch":
            raise ValueError("Wiener filter requires a snapshot_batch input")
        payload = prepared_input.payload
        assert isinstance(payload, SnapshotBatch)
        if payload.context.layout != "batch_channel_y_x":
            raise ValueError("Wiener filter requires grid snapshots, not flattened fluid vectors")
        if payload.context.mask_policy == "keep":
            warnings.append("Wiener filter with mask_policy='keep' may blur across obstacle boundaries.")
        warnings.append("Wiener filtering assumes locally stationary additive noise; structured corruption may violate this assumption.")
        return warnings

    def _fit_impl(self, prepared_input: PreparedBaselineInput) -> tuple[np.ndarray, dict[str, Any]]:
        kernel_size = self._params.get("kernel_size", 3)
        noise_power = self._params.get("noise_power")
        padding_mode = str(self._params.get("padding_mode", "reflect"))
        payload = prepared_input.payload
        assert isinstance(payload, SnapshotBatch)
        source = prepared_input.data[:, : payload.context.variable_channel_count, :, :]
        filtered = np.empty_like(source, dtype=np.float64)
        for snapshot_index in range(source.shape[0]):
            for channel_index in range(source.shape[1]):
                filtered[snapshot_index, channel_index] = wiener_filter_2d(
                    source[snapshot_index, channel_index],
                    kernel_size=kernel_size,
                    noise_power=None if noise_power is None else float(noise_power),
                    padding_mode=padding_mode,
                )
        if payload.context.include_mask_channel:
            filtered = np.concatenate([filtered, prepared_input.data[:, payload.context.variable_channel_count :, :, :]], axis=1)
        metadata = {
            "kernel_size": kernel_size,
            "noise_power": None if noise_power is None else float(noise_power),
            "padding_mode": padding_mode,
        }
        return filtered, metadata


def create_baseline_model(name: str, params: dict[str, Any] | None = None) -> BaselineModel:
    normalized_name = name.strip().lower()
    normalized_params = {} if params is None else dict(params)
    if normalized_name == "rpca_ialm":
        return RPCAIALMBaselineModel(**normalized_params)
    if normalized_name == "dmd":
        return DMDBaselineModel(**normalized_params)
    if normalized_name == "truncated_svd":
        return TruncatedSVDBaselineModel(**normalized_params)
    if normalized_name == "median_filter":
        return MedianFilterBaselineModel(**normalized_params)
    if normalized_name == "wiener_filter":
        return WienerFilterBaselineModel(**normalized_params)
    raise ValueError(f"Unsupported baseline model: {name}")
