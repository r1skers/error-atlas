"""Transport/replay checks on known deterministic fixtures, not a research sweep."""

import copy
import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

from online import pilot, pilot_runner as runner, schedules


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        (cls.root / "prediction.md").write_text("Known deterministic transport fixture.", encoding="utf-8")
        cls.plan = {"schema": runner.SCHEMA, "stage": "smoke", "prediction_record": "prediction.md",
                    "exp": runner.EXP, "schedules": ["chain", "balanced", "shfl_down", "split_k_4"],
                    "fused": [False, True],
                    "cells": [{"kind": "max_at_position", "count": 32, "mass": 32,
                               "gap": "20", "position": 31}]}
        cls.plan_path = cls.root / "plan.json"
        cls.plan_path.write_text(json.dumps(cls.plan), encoding="utf-8")
        cls.bundle = cls.root / "bundle"
        runner.run(cls.plan_path, cls.bundle)

    def test_exact_replay_uses_saved_words_and_graphs_without_rng(self):
        with patch.object(pilot, "uniform_spread", side_effect=AssertionError("regenerated input")), \
                patch.object(pilot, "max_at_position", side_effect=AssertionError("regenerated input")), \
                patch.object(schedules, "balanced_pairwise", side_effect=AssertionError("regenerated graph")):
            result = runner.replay(self.bundle)
        self.assertEqual(result["status"], "exact_match")
        self.assertEqual(result["measurements"], 8)
        self.assertEqual(result["paired_differences"], 6)
        measurements = json.loads((self.bundle / "measurements.json").read_text(encoding="utf-8"))
        chain = next(row for row in measurements if row["graph_id"] == "chain:32" and not row["fused"])
        self.assertEqual(runner.from_word(chain["computed_fp32"]), 32 + Fraction(1, 2**18))

    def test_refuses_existing_output_without_mutating_it(self):
        before = (self.bundle / "manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            runner.run(self.plan_path, self.bundle)
        self.assertEqual(before, (self.bundle / "manifest.json").read_bytes())

    def test_integrity_source_and_result_corruption_are_detected(self):
        # Mock disk reads of copies; leave the original bundle untouched for other tests.
        original_read = runner._read
        manifest = original_read(self.bundle / "manifest.json")
        bad_manifest = copy.deepcopy(manifest)
        bad_manifest["files"]["inputs.json"] = "0" * 64
        with patch.object(runner, "_read", side_effect=lambda p: bad_manifest if p.name == "manifest.json" else original_read(p)):
            with self.assertRaisesRegex(ValueError, "integrity mismatch: inputs"):
                runner.replay(self.bundle)
        with patch.object(runner, "MODULE_DIR", self.root):
            for name in runner.SOURCE_NAMES:
                (self.root / name).write_text("changed source", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Source mismatch"):
                runner.replay(self.bundle)
        bad_results = original_read(self.bundle / "measurements.json")
        bad_results[0]["computed_fp32"] = "00000000"
        with patch.object(runner, "_read", side_effect=lambda p: bad_results if p.name == "measurements.json" else original_read(p)):
            with self.assertRaisesRegex(ValueError, "Measurement replay mismatch"):
                runner.replay(self.bundle)

    def test_unsupported_shuffle_width_is_explicitly_not_applicable(self):
        plan = copy.deepcopy(self.plan)
        plan["cells"][0]["count"] = 128
        graph = next(g for g in runner.build_graphs(plan) if g["requested"] == "shfl_down")
        self.assertEqual(graph["status"], "not_applicable")
        self.assertNotIn("nodes", graph)

    def test_signed_and_subnormal_words_round_trip_exactly(self):
        for word in ("00000000", "00000001", "80000001", "3eaaaaab", "beaaaaab", "7f7fffff"):
            self.assertEqual(runner.fp32_word(runner.from_word(word)), word)
        for word in ("80000000", "7f800000", "7fc00000", "1234"):
            with self.assertRaises(ValueError):
                runner.from_word(word)
        with self.assertRaises(ValueError):
            runner.fp32_word(Fraction(1, 3))

    def test_plan_rejects_unsupported_or_ambiguous_arithmetic(self):
        for field, value in (("exp", "math.exp"), ("fused", ["false"]),
                             ("schedules", ["chain"]), ("stage", "confirmation")):
            plan = dict(self.plan, **{field: value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                runner.validate_plan(plan)

    def test_uniform_input_records_keep_cell_seed_mass_and_exact_words(self):
        # Input-serialization fixture only: no numerical measurement or random sweep.
        plan = dict(self.plan, cells=[{"kind": "uniform_spread", "count": 4, "mass": 7,
                                      "spread": 8, "seeds": [123, 124]}])
        rows = runner.build_inputs(plan)
        self.assertEqual(rows, runner.build_inputs(plan))
        self.assertEqual([row["seed"] for row in rows], [123, 124])
        self.assertNotEqual(rows[0]["maxima_fp32"], rows[1]["maxima_fp32"])
        for row in rows:
            self.assertEqual(row["cell_index"], 0)
            self.assertEqual(row["masses"], [7] * 4)
            self.assertEqual([runner.fp32_word(runner.from_word(word))
                              for word in row["maxima_fp32"]], row["maxima_fp32"])

    def test_checked_in_plans_validate_and_prediction_paths_exist(self):
        for name in ("smoke_v1.json", "expansion_v1.json"):
            path = runner.MODULE_DIR / "plans" / name
            plan = json.loads(path.read_text(encoding="utf-8"))
            runner.validate_plan(plan)
            self.assertTrue((path.parent / plan["prediction_record"]).is_file())

    def test_failure_never_marks_a_partial_bundle_complete(self):
        destination = self.root / "failed"
        with patch.object(runner, "evaluate", side_effect=ValueError("reference could not certify")):
            with self.assertRaisesRegex(ValueError, "could not certify"):
                runner.run(self.plan_path, destination)
        with self.assertRaisesRegex(ValueError, "complete supported bundle"):
            runner.replay(destination)


if __name__ == "__main__":
    unittest.main()
