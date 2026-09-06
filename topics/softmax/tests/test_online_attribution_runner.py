"""End-to-end attribution transport on tiny test-only inputs, not the research sweep."""

import copy
import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

from online import attribution_runner as runner, pilot_runner


class AttributionRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.prediction = cls.root / "prediction.md"
        cls.prediction.write_text("Unit test only; no research inference.", encoding="utf-8")
        plan = {"schema": pilot_runner.SCHEMA, "stage": "smoke", "prediction_record": "prediction.md",
                "exp": pilot_runner.EXP, "schedules": ["chain", "balanced"], "fused": [False, True],
                "cells": [{"kind": "uniform_spread", "count": 4, "mass": 32, "spread": 8, "seeds": [123, 124]}]}
        plan_path = cls.root / "plan.json"
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        cls.parent, cls.output = cls.root / "parent", cls.root / "attribution"
        pilot_runner.run(plan_path, cls.parent)
        runner.run(cls.parent, cls.output, cls.prediction)

    def test_replay_matches_raw_terms_and_summary(self):
        self.assertEqual(runner.replay(self.parent, self.output),
                         {"status": "exact_match", "measurements": 8, "pairs": 4})

    def test_exact_serialization_handles_large_and_negative_fractions(self):
        original = {"huge": Fraction(-(10**5000 + 1), 10**5001 + 3),
                    "interval": (Fraction(-1, 3), Fraction(2, 7))}
        restored = runner.unpack(json.loads(json.dumps(runner.pack(original))))
        self.assertEqual(restored, original)

    def test_saved_components_close_the_signed_and_absolute_identities(self):
        rows = runner.unpack(runner.read(self.output / "attributions.json"))
        for row in rows:
            terms = row["terms"]
            r, w = terms["rounding_error"], terms["weight_error"]
            self.assertEqual(terms["total_error"], (r + w[0], r + w[1]))
            self.assertEqual(row["A"], runner.absolute_interval(terms["relative_total"]))
            self.assertEqual(sum(row["residual_sign_counts"].values()), 3)
            self.assertGreaterEqual(terms["rounding_budget"], abs(r))
            self.assertGreaterEqual(row["internal_savings"][0], 0)
            self.assertGreaterEqual(row["between_savings"][0], 0)

    def test_mismatched_original_result_is_rejected(self):
        original_read = runner.read
        broken = copy.deepcopy(original_read(self.parent / "measurements.json"))
        broken[0]["computed_fp32"] = "00000000"
        with patch.object(runner, "read", side_effect=lambda path: broken if path.name == "measurements.json" else original_read(path)):
            with self.assertRaisesRegex(ValueError, "Original root/error replay mismatch"):
                runner.analyze(self.parent)

    def test_integrity_mismatch_and_existing_destination_are_rejected(self):
        original_digest = runner.digest
        with patch.object(runner, "digest", side_effect=lambda path: "bad" if path.name == "inputs.json" else original_digest(path)):
            with self.assertRaisesRegex(ValueError, "integrity mismatch"):
                runner.analyze(self.parent)
        with self.assertRaises(FileExistsError):
            runner.run(self.parent, self.output, self.prediction)

    def test_outward_summary_enclosures_and_negative_division(self):
        interval = runner.divide_constant(Fraction(-1, 3), (Fraction(2), Fraction(4)))
        self.assertEqual(interval, (Fraction(-1, 6), Fraction(-1, 12)))
        encoded = runner.decimal_interval(interval)
        self.assertLessEqual(Fraction(encoded["low"]), interval[0])
        self.assertGreaterEqual(Fraction(encoded["high"]), interval[1])
        summary = runner.read(self.output / "summary.json")
        self.assertEqual(len(summary["cells"]), 4)
        self.assertEqual(len(summary["comparisons"]), 2)
        self.assertTrue(all(cell["families"] == 2 for cell in summary["cells"]))


if __name__ == "__main__":
    unittest.main()
