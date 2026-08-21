"""Tests for Q shortlist plus beam reranking."""

import unittest

from predictor_q_beam_shortlist_cascade_calibration import (
    _random_best_coverage,
    _mean_or_none,
    _selection_metrics,
    _shortlist_indices,
)


class QBeamShortlistCascadeCalibrationTests(unittest.TestCase):
    def test_shortlist_uses_lowest_q_with_stable_ties(self) -> None:
        self.assertEqual(_shortlist_indices([3.0, 1.0, 1.0, 2.0], 2), (1, 2))
        self.assertEqual(_shortlist_indices([3.0, 1.0], 9), (1, 0))

    def test_random_coverage_is_exact_without_replacement(self) -> None:
        self.assertAlmostEqual(_random_best_coverage(4, 1, 2), 0.5)
        self.assertEqual(_random_best_coverage(4, 2, 3), 1.0)

    def test_selection_metrics_report_perfect_ranking(self) -> None:
        metric = _selection_metrics([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])

        self.assertEqual(metric.rho, 1.0)
        self.assertEqual(metric.pairwise_accuracy, 1.0)
        self.assertEqual(metric.best_tier_hit, 1.0)
        self.assertEqual(metric.normalized_regret, 0.0)

    def test_invalid_shortlist_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _shortlist_indices([], 1)
        with self.assertRaises(ValueError):
            _shortlist_indices([1.0], 0)

    def test_empty_metric_summary_is_undefined(self) -> None:
        self.assertIsNone(_mean_or_none([]))
        self.assertEqual(_mean_or_none([1.0, 3.0]), 2.0)


if __name__ == "__main__":
    unittest.main()
