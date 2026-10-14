"""End-to-end selection tests using real clean/noisy phase 2/3 fixtures."""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "python" / "src"))
sys.path.insert(0, str(REPOSITORY / "tests"))

import test_phase5_baselines as fixtures
from fluid_denoise import phase5_baseline_io as baseline_io
from fluid_denoise.phase2_clean_dataset import validate_clean_dataset
from fluid_denoise.phase4_model_data import align_model_runs


@contextmanager
def working_directory(path):
    original = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)


class Phase5SourceSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="phase5_source_selection_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.clean, self.noisy = fixtures.Phase5BaselineTest()._build_clean_and_noisy(self.root / "data")
        self.fake_module = self.root / "python" / "src" / "fluid_denoise" / "phase5_baseline_io.py"
        self.module_patch = patch.object(baseline_io, "__file__", str(self.fake_module))
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        self.config_dir = self.root / "configuration"
        self.config_dir.mkdir()
        self.external = self.root / "unrelated_cwd"
        self.external.mkdir()
        self._age(self.noisy / "metadata.json", 10)

    @staticmethod
    def _age(path, ordinal):
        nanoseconds = 1_000_000_000 + ordinal * 1_000_000_000
        os.utime(path, ns=(nanoseconds, nanoseconds))

    def _config(self, automatic=True):
        return {"source": {"run_dir": "latest_valid_noisy" if automatic else str(self.noisy),
                           "reference_run_dir": "source_clean" if automatic else str(self.clean)},
                "input": {"kind": "space_time_matrix", "variables": ["ux", "uy"],
                          "mask_policy": "flatten_fluid"},
                "model": {"name": "truncated_svd", "params": {"rank": 2}},
                "output": {"root": str(self.root / "output"), "reconstruction_id": "selection-test",
                           "overwrite": True}}

    def _run(self, config=None, config_path=None):
        output = baseline_io.run_baseline_pipeline(config or self._config(), config_path=config_path)
        metadata = json.loads((output / "metadata.json").read_text())
        with np.load(output / "reconstruction.npz") as data:
            reference = data["reference"].copy()
        return output, metadata, reference

    def _assert_paired_reference(self, metadata, reference, expected_noisy=None):
        self.assertEqual(Path(metadata["source"]["run_dir"]), expected_noisy or self.noisy)
        with np.load(self.clean / "fields.npz") as clean:
            expected = np.stack((clean["ux"], clean["uy"]), axis=1)
        np.testing.assert_array_equal(reference, expected)

    def _clone_noisy(self, name, age):
        target = self.root / "data" / "noisy" / name
        shutil.copytree(self.noisy, target)
        self._age(target / "metadata.json", age)
        return target

    def test_skips_newest_invalid_json_and_unaligned_noisy_run(self):
        invalid = self.root / "data" / "noisy" / "newest-malformed"
        invalid.mkdir()
        (invalid / "metadata.json").write_text("{incomplete")
        self._age(invalid / "metadata.json", 30)
        unaligned = self._clone_noisy("newer-unaligned", 20)
        with np.load(unaligned / "fields.npz") as data:
            arrays = {key: data[key].copy() for key in data.files}
        arrays["steps"][-1] += 1
        np.savez_compressed(unaligned / "fields.npz", **arrays)
        _, metadata, reference = self._run()
        self._assert_paired_reference(metadata, reference)

    def test_skips_structurally_malformed_newest_metadata(self):
        invalid = self.root / "data" / "noisy" / "newest-bad-structure"
        invalid.mkdir()
        for payload in ([], {"source": None},
                        {"source": {"clean_run_dir": None, "source_experiment_id": self.clean.name}}):
            with self.subTest(payload=payload):
                (invalid / "metadata.json").write_text(json.dumps(payload))
                self._age(invalid / "metadata.json", 20)
                _, metadata, reference = self._run()
                self._assert_paired_reference(metadata, reference)

    def test_uses_declared_pair_instead_of_newest_unrelated_clean(self):
        unrelated = self.root / "data" / "clean" / "newest-unrelated"
        shutil.copytree(self.clean, unrelated)
        meta = json.loads((unrelated / "metadata.json").read_text())
        meta["experiment_id"] = unrelated.name
        (unrelated / "metadata.json").write_text(json.dumps(meta))
        self._age(unrelated / "metadata.json", 100)
        with np.load(unrelated / "fields.npz") as data:
            arrays = {key: data[key].copy() for key in data.files}
        arrays["ux"][:, arrays["mask"] == 0] += .2
        arrays["speed"] = np.sqrt(arrays["ux"] ** 2 + arrays["uy"] ** 2)
        np.savez_compressed(unrelated / "fields.npz", **arrays)
        validate_clean_dataset(unrelated)
        _, metadata, reference = self._run()
        self._assert_paired_reference(metadata, reference)

    def test_conflicting_existing_clean_path_does_not_override_source_id(self):
        unrelated = self.root / "data" / "clean" / "wrong-existing-reference"
        shutil.copytree(self.clean, unrelated)
        meta = json.loads((unrelated / "metadata.json").read_text())
        meta["experiment_id"] = unrelated.name
        (unrelated / "metadata.json").write_text(json.dumps(meta))
        with np.load(unrelated / "fields.npz") as data:
            arrays = {key: data[key].copy() for key in data.files}
        arrays["ux"][:, arrays["mask"] == 0] += .2
        arrays["speed"] = np.sqrt(arrays["ux"] ** 2 + arrays["uy"] ** 2)
        np.savez_compressed(unrelated / "fields.npz", **arrays)
        validate_clean_dataset(unrelated)
        self.assertEqual(align_model_runs(self.noisy, unrelated).reference.experiment_id, unrelated.name)
        newest = self._clone_noisy("newest-conflicting-reference", 20)
        meta = json.loads((newest / "metadata.json").read_text())
        meta["source"]["clean_run_dir"] = str(unrelated)
        # Source identity still names the actual pair, not this aligned geometry.
        self.assertEqual(meta["source"]["source_experiment_id"], self.clean.name)
        (newest / "metadata.json").write_text(json.dumps(meta))
        self._age(newest / "metadata.json", 20)
        _, _, reference = self._run()
        with np.load(self.clean / "fields.npz") as data:
            expected = np.stack((data["ux"], data["uy"]), axis=1)
        np.testing.assert_array_equal(reference, expected)

    def test_stale_absolute_source_path_falls_back_to_portable_source_id(self):
        meta_path = self.noisy / "metadata.json"
        meta = json.loads(meta_path.read_text())
        meta["source"]["clean_run_dir"] = str(self.root / "missing_old_checkout" / self.clean.name)
        meta_path.write_text(json.dumps(meta))
        with working_directory(self.external):
            _, metadata, reference = self._run()
        self._assert_paired_reference(metadata, reference)

    def test_relative_recorded_clean_path_is_portable_outside_repository_cwd(self):
        meta_path = self.noisy / "metadata.json"
        meta = json.loads(meta_path.read_text())
        meta["source"]["clean_run_dir"] = str(self.clean.relative_to(self.root))
        meta_path.write_text(json.dumps(meta))
        with working_directory(self.external):
            _, metadata, reference = self._run()
        self._assert_paired_reference(metadata, reference)

    def test_config_relative_paths_default_to_config_directory(self):
        config = self._config(automatic=False)
        config["source"] = {"run_dir": str(Path("..") / self.noisy.relative_to(self.root)),
                            "reference_run_dir": str(Path("..") / self.clean.relative_to(self.root))}
        config["output"]["root"] = "relative-output"
        with working_directory(self.external):
            output, metadata, reference = self._run(config, self.config_dir / "baseline.json")
        self.assertEqual(output, self.config_dir / "relative-output" / "selection-test")
        self._assert_paired_reference(metadata, reference)

    def test_repository_path_base_is_independent_of_config_directory_and_cwd(self):
        config = self._config(automatic=False)
        config["path_base"] = "repository"
        config["source"] = {"run_dir": str(self.noisy.relative_to(self.root)),
                            "reference_run_dir": str(self.clean.relative_to(self.root))}
        config["output"]["root"] = "relative-output"
        with working_directory(self.external):
            output, metadata, reference = self._run(config, self.config_dir / "baseline.json")
        self.assertEqual(output, self.root / "relative-output" / "selection-test")
        self._assert_paired_reference(metadata, reference)

    def test_default_output_root_respects_selected_path_base(self):
        for path_base, expected_base in ((None, self.config_dir), ("repository", self.root)):
            with self.subTest(path_base=path_base):
                config = self._config(automatic=False)
                config["output"].pop("root")
                if path_base is not None:
                    config["path_base"] = path_base
                with working_directory(self.external):
                    output, _, _ = self._run(config, self.config_dir / "baseline.json")
                self.assertEqual(output, expected_base / "data" / "processed" / "baselines" / "selection-test")

    def test_no_valid_pair_raises_actionable_error(self):
        shutil.rmtree(self.noisy)
        with self.assertRaisesRegex(FileNotFoundError, "No valid noisy/clean pair"):
            self._run()


if __name__ == "__main__":
    unittest.main()
