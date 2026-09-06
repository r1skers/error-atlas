"""Synthetic, hand-computable checks; no new research inputs are measured here."""

import copy
import unittest
from fractions import Fraction

from online import pilot_summary as summary


def fraction(value):
    value = Fraction(value)
    return {"numerator": str(value.numerator), "denominator": str(value.denominator)}


class SummaryTests(unittest.TestCase):
    def fixture(self):
        plan = {"schedules": ["chain", "balanced"], "fused": [False, True],
                "cells": [{"kind": "uniform_spread", "count": n, "mass": 32,
                           "spread": 8, "seeds": [i * 10, i * 10 + 1]}
                          for i, n in enumerate((32, 128))]}
        inputs, measures, pairs = [], [], []
        for i, cell in enumerate(plan["cells"]):
            for j, seed in enumerate(cell["seeds"]):
                fid = f"{i}:{j}"
                inputs.append({"family_id": fid, "cell_index": i, "seed": seed})
                for fused in plan["fused"]:
                    d = Fraction(((-1, 3), (5, 7))[i][j]) * (-1 if fused else 1)
                    for name, error in (("chain", 10 + d), ("balanced", 10)):
                        measures.append({"family_id": fid, "graph_id": f"{name}:{cell['count']}",
                                         "fused": fused, "status": "ok", "error_low": fraction(error),
                                         "error_high": fraction(error)})
                    pairs.append({"family_id": fid, "fused": fused, "schedule": "chain",
                                  "baseline": "balanced", "difference_low": fraction(d),
                                  "difference_high": fraction(d)})
        return plan, inputs, measures, pairs

    def test_cell_means_and_adjacent_changes_keep_fma_arms_separate(self):
        result = summary.summarize(*self.fixture())
        cells = result["cells"]
        self.assertEqual([Fraction(row["mean_difference"]["low"]) for row in cells], [1, -1, 6, -6])
        self.assertEqual(cells[0]["individual_orders"], {"negative": 1, "equal": 0, "positive": 1, "unresolved": 0})
        changes = result["adjacent_count_changes"]
        self.assertEqual([Fraction(row["mean_difference_change"]["low"]) for row in changes], [5, -5])
        self.assertTrue(all(row["families"] == 2 for row in cells))

    def test_mean_is_an_interval_not_a_point_and_encoding_is_outward(self):
        interval = summary._mean([(Fraction(-1, 3), Fraction(1, 3)), (Fraction(1), Fraction(2))])
        self.assertEqual(interval, (Fraction(1, 3), Fraction(7, 6)))
        encoded = summary._encoded(interval)
        self.assertLessEqual(Fraction(encoded["low"]), interval[0])
        self.assertGreaterEqual(Fraction(encoded["high"]), interval[1])
        self.assertEqual(summary._order((Fraction(-1), Fraction(1))), "unresolved")
        self.assertEqual(summary._order((Fraction(0), Fraction(0))), "equal")

    def test_missing_and_duplicate_observations_fail_instead_of_biasing_means(self):
        data = self.fixture()
        with self.assertRaises((ValueError, KeyError)):
            summary.summarize(data[0], data[1][:-1], data[2], data[3])
        data = self.fixture()
        with self.assertRaisesRegex(ValueError, "Duplicate observations"):
            summary.summarize(*data[:2], data[2] + [copy.deepcopy(data[2][0])], data[3])


if __name__ == "__main__":
    unittest.main()
