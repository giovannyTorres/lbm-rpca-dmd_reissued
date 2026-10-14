from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python" / "src"))

from fluid_denoise.research.forecasting import (
    ForecastFailure, chronological_split, evaluate_forecasting, fit_forecast,
    make_forecasting_noise,
)


def decaying_dataset():
    times = np.arange(31) * .1
    xy = np.arange(8) * 2 * np.pi / 8
    x, y = np.meshgrid(xy, xy)
    first = np.stack((np.sin(x) * np.cos(y), -np.cos(x) * np.sin(y)))
    second = np.stack((np.sin(2 * x) * np.cos(2 * y), -np.cos(2 * x) * np.sin(2 * y)))
    velocity = np.exp(-.3 * times[:, None, None, None]) * first
    velocity += .2 * np.exp(-.8 * times[:, None, None, None]) * second
    return {"velocity": velocity, "times": times, "dx": 2 * np.pi / 8,
            "dy": 2 * np.pi / 8, "boundary": "periodic", "mask": np.zeros((8, 8), bool),
            "density": np.ones((31, 8, 8)), "metadata": {"rho0": 1.}}


class ChronologicalForecastTest(unittest.TestCase):
    def test_split_has_ordered_disjoint_blocks_and_declared_rounding(self):
        split = chronological_split(np.arange(51))
        self.assertEqual([len(split[k]) for k in ("train", "validation", "test")], [30, 10, 11])
        np.testing.assert_array_equal(np.concatenate(list(split.values())), np.arange(51))
        self.assertLess(split["train"][-1], split["validation"][0])
        self.assertLess(split["validation"][-1], split["test"][0])
        with self.assertRaises(ValueError):
            chronological_split(np.arange(4))
        with self.assertRaises(ValueError):
            chronological_split(np.arange(20), .8, .3)

    def test_full_feature_extrapolation_matches_independent_exact_dynamics(self):
        train = np.arange(20) * .1 + 7.
        future = np.arange(20, 31) * .1 + 7.
        modes = np.array([[1., 2.], [3., -.5], [-2., 4.], [.3, 1.]])
        matrix = modes @ np.exp(np.array([-.4, -.7])[:, None] * (train - train[0]))
        reference = modes @ np.exp(np.array([-.4, -.7])[:, None] * (future - train[0]))
        for method in ("dmd", "tls_dmd", "pod_ar", "optimized_dmd"):
            with self.subTest(method=method):
                model = fit_forecast(matrix, train, method, 2, ridge=0., max_nfev=200)
                np.testing.assert_allclose(model.predict(future), reference, rtol=2e-7, atol=2e-9)
                self.assertEqual(model.predict(future).shape, (4, 11))
                self.assertEqual(model.diagnostics["fit_scope"], "train_only")
                self.assertTrue(model.diagnostics["converged"])
                if method in ("dmd", "tls_dmd", "optimized_dmd"):
                    self.assertIsNotNone(model.rates)
                    self.assertIsNotNone(model.modes)

    def test_optimized_dmd_handles_irregular_physical_times(self):
        train = np.array([0., .04, .11, .18, .32, .49, .64, .82, 1.05])
        target = np.array([1.2, 1.55, 2.])
        modes = np.array([[1., 2.], [-1., .4], [.5, -2.]])
        rates = np.array([-.3, -.7])
        model = fit_forecast(modes @ np.exp(rates[:, None] * train), train, "optimized_dmd", 2,
                             max_nfev=300)
        np.testing.assert_allclose(model.predict(target), modes @ np.exp(rates[:, None] * target),
                                   rtol=1e-5, atol=1e-7)
        with self.assertRaises(ValueError):
            fit_forecast(modes @ np.exp(rates[:, None] * train), train, "dmd", 2)

    def test_pod_ar_requires_forecast_times_on_its_learned_step_grid(self):
        model = fit_forecast(np.exp(-.1 * np.arange(8))[None], np.arange(8), "pod_ar", 1)
        with self.assertRaises(ValueError):
            model.predict([8.5])
        with self.assertRaises(ValueError):
            model.predict([6.])

    def test_persistence_and_zero_series_are_constant_without_future_inputs(self):
        observations = np.array([[2., 3., 5.], [4., 2., 9.]])
        model = fit_forecast(observations, [0., 1., 2.], "persistence", 8)
        observations[:, -1] = 1000
        np.testing.assert_array_equal(model.predict([3., 12.]), [[5., 5.], [9., 9.]])
        zero = fit_forecast(np.zeros((3, 4)), np.arange(4), "dmd", 2)
        np.testing.assert_array_equal(zero.predict([5., 10.]), np.zeros((3, 2)))

    def test_growing_spectrum_is_not_clipped_and_overflow_is_rejected(self):
        model = fit_forecast((2. ** np.arange(5))[None], np.arange(5), "dmd", 1)
        self.assertGreater(model.diagnostics["spectral_radius"], 1.)
        np.testing.assert_allclose(model.predict([8.]), [[256.]], rtol=1e-10)
        with self.assertRaises(ForecastFailure):
            model.predict([2000.])

    def test_negative_eigenvalue_preserves_discrete_alternating_dynamics(self):
        model = fit_forecast((-1.2) ** np.arange(8)[None], np.arange(8), "dmd", 1)
        np.testing.assert_allclose(model.predict([8., 9., 10.]), (-1.2) ** np.array([[8., 9., 10.]]),
                                   rtol=1e-10, atol=1e-10)

    def test_raw_dmd_noise_bias_remains_visible_in_true_future_errors(self):
        rng = np.random.default_rng(42)
        spatial = rng.normal(size=100)
        training_times = np.arange(40)
        future_times = np.arange(40, 60)
        observations = spatial[:, None] * (.98 ** training_times)[None]
        observations += .8 * rng.normal(size=observations.shape)
        reference = spatial[:, None] * (.98 ** future_times)[None]
        raw = fit_forecast(observations, training_times, "dmd", 1)
        tls = fit_forecast(observations, training_times, "tls_dmd", 1)
        raw_error = np.linalg.norm(raw.predict(future_times) - reference) / np.linalg.norm(reference)
        tls_error = np.linalg.norm(tls.predict(future_times) - reference) / np.linalg.norm(reference)
        self.assertLess(raw.diagnostics["spectral_radius"], .97)
        self.assertGreater(raw_error, .5)
        self.assertLess(abs(tls.diagnostics["spectral_radius"] - .98),
                        abs(raw.diagnostics["spectral_radius"] - .98))
        self.assertGreater(tls_error, .2)  # Bias reduction does not certify an accurate forecast.

    def test_train_only_preprocessors_produce_real_forecast_models(self):
        t = np.arange(15) * .1
        modes = np.array([[1., 2.], [.3, -.4], [-1., 1.]])
        observed = modes @ np.exp(np.array([-.2, -.6])[:, None] * t)
        for preprocessing in ("rpca", "stable_rpca"):
            with self.subTest(preprocessing=preprocessing):
                parameters = {"max_iter": 300, "tol": 1e-6}
                if preprocessing == "stable_rpca":
                    parameters.update(alpha=0., beta=0.)
                model = fit_forecast(observed, t, "dmd", 2, preprocessing=preprocessing,
                                     preprocessing_parameters=parameters)
                self.assertEqual(model.diagnostics["preprocessing"]["fit_scope"], "train_only")
                self.assertTrue(np.isfinite(model.predict([1.6, 1.7])).all())

    def test_future_clean_and_test_observations_cannot_change_selection_or_forecasts(self):
        dataset = decaying_dataset()
        noisy, noise = make_forecasting_noise(dataset, kind="mixed", intensity=.15, seed=7)
        config = {"candidates": [{"id": "dmd_r2", "method": "dmd", "rank": 2},
                                  {"id": "tls_r2", "method": "tls_dmd", "rank": 2},
                                  {"id": "projected_r2", "method": "dmd", "rank": 2,
                                   "postprocess": "helmholtz_dealias"},
                                  {"id": "stable_dmd_r2", "method": "dmd", "rank": 2,
                                   "preprocessing": "stable_rpca"}],
                  "known_noise": {"kind": "mixed", "intensity": .15},
                  "preprocessing_parameters": {"max_iter": 300, "tol": 1e-5}}
        original = evaluate_forecasting(dataset, noisy, config)
        split = chronological_split(dataset["times"])
        changed = copy.deepcopy(dataset)
        future = np.r_[split["validation"], split["test"]]
        changed["velocity"][future] *= 3.7
        corrupted_test = noisy.copy()
        corrupted_test[split["test"]] = np.nan
        repeated = evaluate_forecasting(changed, corrupted_test, config)
        self.assertEqual(original["selection"], repeated["selection"])
        self.assertEqual(original["forecast_sha256"], repeated["forecast_sha256"])
        self.assertEqual(original["provenance"]["selection_inputs_sha256"],
                         repeated["provenance"]["selection_inputs_sha256"])
        self.assertNotEqual(original["selected_test_metrics"], repeated["selected_test_metrics"])
        self.assertEqual(noise["normalization"],
                         "realized training fluid-entry RMS only; future blocks are not renormalized")
        json.dumps(original, allow_nan=False)

    def test_noisy_validation_controls_choice_without_clean_oracle(self):
        dataset = decaying_dataset()
        split = chronological_split(dataset["times"])
        config = {"candidates": [{"id": "exact_r2", "method": "dmd", "rank": 2}]}
        exact_validation = dataset["velocity"].copy()
        chosen_dynamic = evaluate_forecasting(dataset, exact_validation, config)
        self.assertEqual(chosen_dynamic["selection"]["selected_id"], "exact_r2")
        persistence_validation = exact_validation.copy()
        persistence_validation[split["validation"]] = dataset["velocity"][split["train"][-1]]
        chosen_constant = evaluate_forecasting(dataset, persistence_validation, config)
        self.assertEqual(chosen_constant["selection"]["selected_id"], "persistence")
        self.assertGreater(chosen_constant["selected_test_metrics"]["velocity_relative_l2"],
                           chosen_constant["test_metrics"]["exact_r2"]["velocity_relative_l2"])
        self.assertFalse(chosen_constant["selection"]["clean_fields_used_for_selection"])

    def test_noise_scale_temporal_state_and_column_support_ignore_clean_future(self):
        dataset = decaying_dataset()
        split = chronological_split(dataset["times"])
        changed = copy.deepcopy(dataset)
        changed["velocity"][len(split["train"]):] *= 50
        for kind in ("gaussian", "mixed", "temporal_correlated", "column_outliers"):
            with self.subTest(kind=kind):
                a, meta_a = make_forecasting_noise(dataset, kind=kind, intensity=.2, seed=23)
                b, meta_b = make_forecasting_noise(changed, kind=kind, intensity=.2, seed=23)
                np.testing.assert_array_equal(a[split["train"]], b[split["train"]])
                self.assertEqual(meta_a, meta_b)
                np.testing.assert_allclose(a - dataset["velocity"], b - changed["velocity"], atol=1e-14)

    def test_failed_or_nonconverged_candidates_are_visible_and_not_selected(self):
        dataset = decaying_dataset()
        config = {"candidates": [{"id": "bad", "method": "dmd", "rank": 2,
                                  "parameters": {"amplitude_fit": "unsupported"}},
                                 {"id": "short_optimized", "method": "optimized_dmd", "rank": 4,
                                  "parameters": {"max_nfev": 1}}]}
        noisy = dataset["velocity"] + .01 * np.random.default_rng(13).standard_normal(dataset["velocity"].shape)
        result = evaluate_forecasting(dataset, noisy, config)
        self.assertEqual(result["selection"]["selected_id"], "persistence")
        bad, short = result["selection"]["candidates"][1:]
        self.assertEqual(bad["status"], "failed")
        self.assertFalse(bad["eligible"])
        self.assertEqual(short["status"], "not_converged")
        self.assertFalse(short["eligible"])

    def test_test_metrics_compare_matching_future_times_with_existing_physics(self):
        from fluid_denoise.research.metrics import physical_metrics
        dataset = decaying_dataset()
        result = evaluate_forecasting(dataset, dataset["velocity"],
                                      {"candidates": [{"id": "dmd", "method": "dmd", "rank": 2}]})
        indices = chronological_split(dataset["times"])["test"]
        reference = dict(dataset, velocity=dataset["velocity"][indices], times=dataset["times"][indices],
                         density=dataset["density"][indices])
        persistence = np.repeat(dataset["velocity"][[17]], len(indices), axis=0)
        expected = physical_metrics(reference, persistence)
        self.assertEqual(result["test_times"], dataset["times"][indices].tolist())
        self.assertEqual(result["persistence_test_metrics"]["velocity_relative_l2"], expected["velocity_relative_l2"])
        self.assertEqual(result["persistence_test_metrics"]["vorticity_relative_l2"], expected["vorticity_relative_l2"])
        self.assertLess(result["selected_test_metrics"]["velocity_relative_l2"], 1e-10)
        self.assertLess(result["selected_test_metrics"]["vorticity_relative_l2"], 1e-10)

    def test_geometry_projection_precedes_noisy_selection_and_optional_arrays(self):
        dataset = decaying_dataset()
        noisy, _ = make_forecasting_noise(dataset, kind="gaussian", intensity=.2, seed=61)
        config = {"candidates": [{"id": "projected", "method": "dmd", "rank": 2,
                                  "postprocess": "helmholtz_dealias"}]}
        result = evaluate_forecasting(dataset, noisy, config, return_forecasts=True)
        record = result["selection"]["candidates"][1]
        constraints = record["diagnostics"]["constraints"]
        self.assertFalse(constraints["reference_velocity_used"])
        self.assertEqual(constraints["bandwidth_prior"], "strict two-thirds Fourier truncation")
        self.assertEqual(record["diagnostics"]["validation_postprocessing"]["postprocess"], "helmholtz_dealias")
        self.assertEqual(record["diagnostics"]["validation_postprocessing"]["fft_calls"], 2)
        for key in ("selected", "persistence"):
            self.assertEqual(result["test_predictions"][key].shape, (7, 2, 8, 8))
        projected_metrics = result["test_metrics"]["projected"]
        self.assertLess(projected_metrics["extra_divergence_normalized"], 1e-12)
        self.assertIn("test_elapsed_seconds", result["postprocessing_profile"]["projected"])
        result.pop("test_predictions")
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
