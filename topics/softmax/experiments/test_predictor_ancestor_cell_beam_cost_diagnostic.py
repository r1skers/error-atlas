"""Tests for the ancestor-cell beam cost diagnostic."""

import math
import unittest

from predictor_ancestor_cell_beam_cost_diagnostic import _benchmark_width


class AncestorCellBeamCostDiagnosticTests(unittest.TestCase):
    def test_small_cost_profile_is_positive(self) -> None:
        row = _benchmark_width(16, graphs_per_family=1, repeats=1, budget=4)

        self.assertEqual(row.width, 16)
        self.assertEqual(row.tree_count, 2)
        for value in (
            row.q_metadata_ms,
            row.oracle_ms,
            row.shadow_trace_ms,
            row.beam1_extra_ms,
            row.beam3_extra_ms,
        ):
            self.assertTrue(math.isfinite(value) and value > 0.0)


if __name__ == "__main__":
    unittest.main()
