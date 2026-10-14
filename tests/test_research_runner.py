"""Integration gates for isolation, provenance, and scientific failure reporting."""
import json
import unittest
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python" / "src"))
import numpy as np
from fluid_denoise.research.dns import taylor_green_reference
from fluid_denoise.research.runner import assert_finite, dataset_digest, isolated_fit, json_safe


class RunnerTests(unittest.TestCase):
    def test_independent_fit_has_physics_and_measured_profile(self):
        dataset = taylor_green_reference(nx=8, times=np.linspace(0, 1, 8))
        result = isolated_fit(dataset, dataset['velocity'], 'dmd', 1, {}, 10)
        self.assertNotIn('failure', result)
        self.assertLess(result['metrics']['velocity_relative_l2'], 1e-10)
        self.assertGreater(result['profile']['rss_peak_bytes'], 0)
        json.dumps(json_safe(result), allow_nan=False)

    def test_failure_remains_a_failure(self):
        dataset = taylor_green_reference(nx=8, times=np.linspace(0, 1, 8))
        result = isolated_fit(dataset, dataset['velocity'], 'invalid_method', 1, {}, 10)
        self.assertIn('failure', result)
        self.assertNotIn('metrics', result)

    def test_source_digest_detects_changed_field(self):
        dataset = taylor_green_reference(nx=8, times=np.linspace(0, 1, 8))
        before = dataset_digest(dataset)
        dataset['velocity'][0, 0, 0, 0] += .001
        self.assertNotEqual(before, dataset_digest(dataset))

    def test_nonfinite_output_is_explicit_json_null(self):
        self.assertEqual(json_safe({'error': np.nan, 'zero': np.float64(0)}), {'error': None, 'zero': 0.})

    def test_unexpected_nonfinite_result_cannot_be_success(self):
        assert_finite({'undefined_relative_error': None})
        with self.assertRaises(FloatingPointError):
            assert_finite({'metrics': {'energy': np.inf}})
        with self.assertRaises(FloatingPointError):
            assert_finite({'stress': np.array([0., np.nan])})
