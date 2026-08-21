"""Tests for the two-stage macro-energy plus sparse-coherence score."""

import math
import unittest
from fractions import Fraction

from predictor_calibration_inputs import wide_range_random
from predictor_tree_generator import random_pair_merge_graph
from predictor_two_stage_cheap_score_calibration import (
    _analyze,
    _predictor_trace,
)
from predictor_ulp_energy_cost_pareto_diagnostic import _tree_cost_profile
from summation_graph_predictor import balanced_reduction_graph


class TwoStageCheapScoreCalibrationTests(unittest.TestCase):
    def test_full_budget_matches_full_q_and_cost_diagnostic(self) -> None:
        values = tuple(Fraction(1) for _ in range(8))
        graph = balanced_reduction_graph(8)

        trace = _predictor_trace(values, graph, (1, 4, 7, 20))
        cost = _tree_cost_profile(values, graph, "balanced", (1, 4, 7, 20))

        self.assertAlmostEqual(trace.full_q, 1.75)
        self.assertEqual(trace.budget[7].q_budget, trace.full_q)
        self.assertEqual(trace.budget[20].q_budget, trace.full_q)
        self.assertEqual(trace.budget[4].q_budget, cost.budget_q[4])

    def test_exact_additions_keep_macro_energy_without_coherence(self) -> None:
        values = tuple(Fraction(1) for _ in range(8))
        trace = _predictor_trace(values, balanced_reduction_graph(8), (7,))

        self.assertEqual(trace.history_free, 0.0)
        self.assertEqual(trace.full_first, 0.0)
        self.assertEqual(trace.trajectory, 0.0)
        expected = trace.full_q / 12.0
        self.assertEqual(trace.budget[7].coherence_shadow, expected)
        self.assertEqual(trace.budget[7].coherence_first, expected)
        self.assertEqual(trace.budget[7].coherence_trajectory, expected)
        self.assertEqual(trace.budget[7].coherence_phase, expected)

    def test_scores_are_finite_and_nonnegative(self) -> None:
        values = wide_range_random(16, seed=22260821).values
        graph = random_pair_merge_graph(16, seed=46_000_000)

        trace = _predictor_trace(values, graph, (1, 4, 8))

        scalar_scores = [
            trace.full_q,
            trace.history_free,
            trace.full_first,
            trace.trajectory,
        ]
        for scores in trace.budget.values():
            scalar_scores.extend(
                (
                    scores.q_budget,
                    scores.coherence_shadow,
                    scores.coherence_first,
                    scores.coherence_trajectory,
                    scores.coherence_phase,
                    scores.sparse_first,
                )
            )
        self.assertTrue(all(math.isfinite(score) and score >= 0.0 for score in scalar_scores))

    def test_oracle_micro_counts_are_bounded_by_selected_nodes(self) -> None:
        values = wide_range_random(16, seed=22260821).values
        graph = random_pair_merge_graph(16, seed=46_000_001)

        row = _analyze(values, graph, "pair_merge", (4, 8))

        self.assertGreaterEqual(row.target, 0.0)
        for counts in row.micro.values():
            self.assertLessEqual(counts.shadow_sign_correct, counts.selected)
            self.assertLessEqual(counts.first_sign_correct, counts.selected)
            self.assertLessEqual(counts.trajectory_sign_correct, counts.selected)
            self.assertLessEqual(counts.predicted_cross, counts.selected)
            self.assertLessEqual(counts.actual_cross, counts.selected)
            self.assertLessEqual(counts.cross_true_positive, counts.predicted_cross)
            self.assertLessEqual(counts.cross_true_positive, counts.actual_cross)
            self.assertLessEqual(counts.predicted_phase, counts.selected)
            self.assertLessEqual(counts.actual_phase, counts.selected)
            self.assertLessEqual(counts.phase_true_positive, counts.predicted_phase)
            self.assertLessEqual(counts.phase_true_positive, counts.actual_phase)

    def test_invalid_budget_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _predictor_trace(
                (Fraction(1), Fraction(1)),
                balanced_reduction_graph(2),
                (0,),
            )


if __name__ == "__main__":
    unittest.main()
