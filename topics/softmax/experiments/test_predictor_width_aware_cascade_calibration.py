"""Tests for the width-aware energy-mass cascade calibration."""

import unittest

from predictor_ancestor_cell_beam_score_calibration import _beam_tree
from predictor_calibration_inputs import wide_range_random
from predictor_two_stage_cheap_score_calibration import _graphs
from predictor_width_aware_cascade_calibration import (
    _adaptive_shortlist_size,
    _energy_budget,
    _evaluate_policy,
    _robust_gap,
)


class WidthAwareCascadeCalibrationTests(unittest.TestCase):
    def test_robust_gap_uses_within_group_scale(self) -> None:
        values = [0.0, 0.1, 0.2, 0.3, 2.0, 2.1, 2.2, 2.3]
        self.assertAlmostEqual(_robust_gap(values, 4), 1.7 / 1.0)
        self.assertAlmostEqual(
            _robust_gap([10.0 * value for value in values], 4),
            _robust_gap(values, 4),
        )

    def test_adaptive_shortlist_stops_only_on_supported_gaps(self) -> None:
        separated_at_four = [0.0] * 4 + [1.0] * 12
        self.assertEqual(
            _adaptive_shortlist_size(separated_at_four, median_energy_budget=8.0),
            4,
        )
        separated_at_eight = [0.0] * 4 + [0.01] * 4 + [1.0] * 8
        self.assertEqual(
            _adaptive_shortlist_size(separated_at_eight, median_energy_budget=9.0),
            8,
        )
        no_gaps = [1.0] * 16
        self.assertEqual(
            _adaptive_shortlist_size(no_gaps, median_energy_budget=8.0),
            16,
        )

    def test_invalid_gap_boundary_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _robust_gap([], 1)
        with self.assertRaises(ValueError):
            _robust_gap([1.0, 2.0], 2)

    def test_policy_rejects_misaligned_score_budgets(self) -> None:
        with self.assertRaises(ValueError):
            _evaluate_policy([], object(), 1, 1, score_budgets=[object()])

    def test_energy_budget_keeps_tree_metadata_aligned(self) -> None:
        values = wide_range_random(16, seed=22260821).values
        family, graph = next(_graphs(16, 0, 2))
        tree = _beam_tree(values, graph, family, budget=8)

        budget = _energy_budget(tree, 0.5, minimum_budget=2, maximum_budget=8)

        self.assertGreaterEqual(budget.node_count, 2)
        self.assertEqual(
            {sample.node_index for sample in budget.tree.transitions},
            set(budget.tree.selected_order),
        )
        self.assertGreaterEqual(budget.captured_fraction, 0.5)


if __name__ == "__main__":
    unittest.main()
