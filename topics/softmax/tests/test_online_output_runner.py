"""Tiny saved-input output study, replay and integrity checks; not the 64-family run."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from online import output_runner as runner, pilot_runner


class OutputRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.protocol = cls.root / "protocol.md"
        cls.protocol.write_text("Test-only fixtures, not research evidence.", encoding="utf-8")
        plan = {"schema": pilot_runner.SCHEMA, "stage": "smoke", "prediction_record": "protocol.md",
                "exp": pilot_runner.EXP, "schedules": ["chain", "balanced"], "fused": [False, True],
                "cells": [{"kind": "uniform_spread", "count": 4, "mass": 32, "spread": 8, "seeds": [123, 124]}]}
        path = cls.root / "plan.json"
        path.write_text(json.dumps(plan), encoding="utf-8")
        cls.parent, cls.output = cls.root / "parent", cls.root / "output"
        pilot_runner.run(path, cls.parent)
        runner.run(cls.parent, cls.output, cls.protocol)

    def test_full_replay_on_saved_bits_and_graphs(self):
        self.assertEqual(runner.replay(self.parent, self.output),
                         {"status": "exact_match", "counts": {"families": 2, "references": 10, "measurements": 60, "pairs": 30}})

    def test_shared_preparation_matches_user_reference_assembly(self):
        _, inputs, _, _, _ = runner.io.load_verified_bundle(self.parent)
        for row in inputs:
            family = runner.pilot.BlockFamily(tuple(pilot_runner.from_word(w) for w in row["maxima_fp32"]), tuple(row["masses"]))
            prepared = runner.fixed.prepare(family)
            for values, ref in runner.prepare_references(family, prepared).values():
                self.assertEqual(ref, runner.reference.reference_output(family, values))

    def test_zero_one_controls_and_division_residual_identity(self):
        result = runner.io.unpack(runner.io.read(self.output / "results.json"))
        for row in result["measurements"]:
            self.assertEqual(row["formats"]["fp32"]["value"], row["before_division_rounding"] + row["division_residual"])
            if row["probe"] in ("zero", "one"):
                for entry in row["formats"].values():
                    self.assertEqual(entry["value"], int(row["probe"] == "one"))
                    self.assertEqual(entry["abs_error"], (0, 0))
        self.assertEqual(sum(r["arm"] == "fixed" for r in result["measurements"]), 20)

    def test_existing_output_and_wrong_parent_are_rejected(self):
        with self.assertRaises(FileExistsError):
            runner.run(self.parent, self.output, self.protocol)
        original = runner.io.digest
        with patch.object(runner.io, "digest", side_effect=lambda p: "bad" if p == self.parent / "manifest.json" else original(p)):
            with self.assertRaisesRegex(ValueError, "Wrong parent"):
                runner.replay(self.parent, self.output)

    def test_tampering_or_bad_original_result_rejected(self):
        original = runner.io.digest
        with patch.object(runner.io, "digest", side_effect=lambda p: "bad" if p.name == "results.json" else original(p)):
            with self.assertRaisesRegex(ValueError, "integrity mismatch"):
                runner.replay(self.parent, self.output)
        loaded = copy.deepcopy(runner.io.load_verified_bundle(self.parent))
        loaded[3][0]["computed_fp32"] = "00000000"
        with patch.object(runner.io, "load_verified_bundle", return_value=loaded):
            with self.assertRaisesRegex(ValueError, "Original denominator root/error"):
                runner.analyze(self.parent)

    def test_summary_keeps_modes_formats_and_shared_arms_separate(self):
        summary = runner.io.read(self.output / "summary.json")
        self.assertEqual(len(summary["cells"]), 90)
        self.assertEqual(len(summary["comparisons"]), 45)
        self.assertTrue(all(r["families"] == 2 for r in summary["comparisons"]))
        self.assertEqual(sum(r["arm"] == "fixed" for r in summary["comparisons"]), 15)


if __name__ == "__main__":
    unittest.main()
