"""Tiny temporary bundles verify replay/accounting, not Monte Carlo coverage."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from online import calibration_runner as runner


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "protocol.md").write_text("Synthetic fixture, no statistical claim.\n", encoding="utf-8")
        self.plan = {"schema": runner.SCHEMA, "stage": "preflight", "method": "percentile",
                     "population_names": ["zero_constant", "positive_constant", "mixed_null_positive"],
                     "counts": [2], "outer_trials": 2, "bootstrap_repeats": 5, "alpha": [1, 20],
                     "seed_namespace": "test-only", "protocol": "protocol.md"}
        self.plan_path = self.root / "plan.json"
        runner.write_json(self.plan_path, self.plan)
        self.bundle = self.root / "bundle"

    def test_exact_replay_uses_saved_outer_inputs_and_constant_trials_count(self):
        manifest = runner.run(self.plan_path, self.bundle)
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["outer_trials"], 6)
        with patch.object(runner.cal, "draw_families", side_effect=AssertionError("Do not regenerate outer inputs")):
            self.assertEqual(runner.replay(self.bundle)["outer_trials"], 6)
        cells = {row["population"]: row for row in runner.read_json(self.bundle / "summary.json")["cells"]}
        for name in ("zero_constant", "positive_constant"):
            cell = cells[name]
            self.assertEqual(cell["outer_trials"], 2)
            self.assertEqual(cell["conjunction_false_positive"], 0)
            for arm in cell["arms"]:
                self.assertEqual(arm["constant_samples"], 2)
                self.assertEqual(arm["covered"], 2)
                self.assertEqual(arm["false_positive"], 0)
                self.assertEqual(arm["rejected"], 0 if name == "zero_constant" else 2)

    def test_changed_file_rejected_by_integrity_check(self):
        runner.run(self.plan_path, self.bundle)
        with (self.bundle / "inputs.jsonl").open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "integrity mismatch"):
            runner.replay(self.bundle)

    def test_rehashed_wrong_verdict_still_fails_numerical_replay(self):
        runner.run(self.plan_path, self.bundle)
        records = list(runner.read_rows(self.bundle / "trials.jsonl"))
        records[0]["verdict"]["covered"][0] = False
        (self.bundle / "trials.jsonl").write_text("\n".join(map(runner.canonical, records)) + "\n", encoding="utf-8")
        manifest = runner.read_json(self.bundle / "manifest.json")
        manifest["files"]["trials.jsonl"] = runner.digest(self.bundle / "trials.jsonl")
        runner.write_json(self.bundle / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "Numerical replay mismatch"):
            runner.replay(self.bundle)

    def test_rehashed_snapshot_must_match_current_source(self):
        runner.run(self.plan_path, self.bundle)
        name = "sources/confirmation_stats.py.txt"
        with (self.bundle / name).open("a", encoding="utf-8") as stream:
            stream.write("\n# changed snapshot\n")
        manifest = runner.read_json(self.bundle / "manifest.json")
        manifest["files"][name] = runner.digest(self.bundle / name)
        runner.write_json(self.bundle / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "Current source mismatch"):
            runner.replay(self.bundle)

    def test_failure_is_recorded_and_never_complete(self):
        with patch.object(runner.cal, "assess_trial", side_effect=RuntimeError("injected failure")):
            with self.assertRaisesRegex(RuntimeError, "injected failure"):
                runner.run(self.plan_path, self.bundle)
        manifest = runner.read_json(self.bundle / "manifest.json")
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["error"]["type"], "RuntimeError")
        self.assertTrue((self.bundle / "inputs.jsonl").exists())
        with self.assertRaisesRegex(ValueError, "complete preflight bundle"):
            runner.replay(self.bundle)

    def test_existing_output_is_not_overwritten(self):
        self.bundle.mkdir()
        sentinel = self.bundle / "keep.txt"
        sentinel.write_text("keep", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            runner.run(self.plan_path, self.bundle)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_unsupported_plan_settings_rejected(self):
        for key, value in (("stage", "confirmation"), ("method", "BCa"), ("counts", [True]),
                           ("counts", [2, 2]), ("population_names", ["missing"]),
                           ("population_names", ["zero_constant", "zero_constant"]),
                           ("outer_trials", 0), ("bootstrap_repeats", False), ("alpha", [1, 0])):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                runner.validate_plan({**self.plan, key: value})


if __name__ == "__main__":
    unittest.main()
