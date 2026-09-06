"""End-to-end fixed ablation on test-only saved inputs, with replay/tamper checks."""

import copy
import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

from online import fixed_ablation_runner as runner, pilot_runner
from online.merge import merge_reduce, weighted_residuals


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.protocol = cls.root / "protocol.md"
        cls.protocol.write_text("Unit tests; no research inference.", encoding="utf-8")
        plan = {"schema": pilot_runner.SCHEMA, "stage": "smoke", "prediction_record": "protocol.md",
                "exp": pilot_runner.EXP, "schedules": ["chain", "balanced"], "fused": [False, True],
                "cells": [{"kind": "uniform_spread", "count": 4, "mass": 32, "spread": 8, "seeds": [123, 124]}]}
        path = cls.root / "plan.json"
        path.write_text(json.dumps(plan), encoding="utf-8")
        cls.parent, cls.output = cls.root / "parent", cls.root / "ablation"
        pilot_runner.run(path, cls.parent)
        runner.run(cls.parent, cls.output, cls.protocol)

    def test_exact_replay_from_saved_inputs(self):
        self.assertEqual(runner.replay(self.parent, self.output),
                         {"status": "exact_match", "families": 2, "fixed": 4, "online": 8, "pairs": 4})

    def test_shared_reference_and_signed_identities(self):
        data = runner.transport.unpack(runner.transport.read(self.output / "results.json"))
        families = {r["family_id"]: r for r in data["families"]}
        fixed = {(r["family_id"], r["schedule"]): r for r in data["fixed"]}
        for row in data["fixed"]:
            family = families[row["family_id"]]
            self.assertEqual(row["S"], row["root"] - family["C"])
            self.assertEqual(row["E"], tuple(row["S"] + j for j in family["J"]))
        for row in data["online"]:
            self.assertEqual(row["E"], tuple(row["R"] + w for w in row["W"]))
            self.assertEqual(row["root_delta"], row["root"] - fixed[row["family_id"], row["schedule"]]["root"])
        for row in data["pairs"]:
            self.assertEqual(row["D_fixed"], runner.gap(fixed[row["family_id"], "chain"], fixed[row["family_id"], "balanced"]))
            if row["both_roots_equal"]:
                self.assertEqual(row["H"], (0, 0))
            else:
                self.assertEqual(row["H"], runner.subtract(row["D_online"], row["D_fixed"]))

    def test_existing_output_is_not_overwritten(self):
        with self.assertRaises(FileExistsError):
            runner.run(self.parent, self.output, self.protocol)

    def test_artifact_tampering_is_rejected(self):
        original = runner.transport.digest
        with patch.object(runner.transport, "digest", side_effect=lambda p: "bad" if p.name == "results.json" else original(p)):
            with self.assertRaisesRegex(ValueError, "integrity mismatch"):
                runner.replay(self.parent, self.output)

    def test_wrong_old_root_rejected_even_if_loaded(self):
        original = runner.transport.load_verified_bundle
        values = copy.deepcopy(original(self.parent))
        values[3][0]["computed_fp32"] = "00000000"
        with patch.object(runner.transport, "load_verified_bundle", return_value=values):
            with self.assertRaisesRegex(ValueError, "Original root/error replay mismatch"):
                runner.analyze(self.parent)

    def test_duplicate_or_missing_old_record_rejected(self):
        original = runner.transport.load_verified_bundle
        for mode in ("duplicate", "missing"):
            values = copy.deepcopy(original(self.parent))
            if mode == "duplicate":
                values[3].append(values[3][0])
            else:
                values[3].pop()
            with patch.object(runner.transport, "load_verified_bundle", return_value=values):
                with self.assertRaises(ValueError):
                    runner.analyze(self.parent)

    def test_summary_has_single_fixed_arm_and_both_online_arms(self):
        summary = runner.transport.read(self.output / "summary.json")
        self.assertEqual(len(summary["cells"]), 6)
        self.assertEqual(len(summary["comparisons"]), 2)
        self.assertTrue(all(r["families"] == 2 for r in summary["cells"]))
        self.assertEqual(sum(r["fused"] is None for r in summary["cells"]), 2)
        data = runner.transport.unpack(runner.transport.read(self.output / "results.json"))
        equal = next(r for r in data["controls"] if r["label"] == "equal_max")
        self.assertTrue(all(r["root_delta"] == 0 for r in equal["online"]))

    def test_online_R_matches_sum_of_propagated_node_residuals(self):
        family = runner.pilot.BlockFamily(tuple(Fraction(i) for i in (-3, -2, -1, 0)), (17, 3, 32, 1))
        trees = {"chain": runner.schedules.sequential_chain(4), "balanced": runner.schedules.balanced_pairwise(4)}
        _, _, rows = runner.compare(family, trees, (False, True))
        for (name, fused), row in rows.items():
            dump = merge_reduce(family.leaf_max, family.leaf_ell, trees[name], fused=fused)
            self.assertEqual(row["R"], sum(weighted_residuals(dump)))


if __name__ == "__main__":
    unittest.main()
