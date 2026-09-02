"""Tests for the discrete ancestor-chain phase score."""

import math
import unittest

from predictor_calibration_inputs import wide_range_random
from predictor_discrete_ancestor_phase_score_calibration import (
    SIGMA_STATES,
    SIGMA_WEIGHTS,
    _ancestor_gap,
    _predictor_scores,
)
from predictor_tree_generator import random_pair_merge_graph
from predictor_ulp_energy_convergence_diagnostic import _parent_map
from summation_graph_predictor import balanced_reduction_graph


class DiscreteAncestorPhaseScoreCalibrationTests(unittest.TestCase):
    def test_sigma_states_have_zero_mean_and_unit_variance(self) -> None:
        state_mean = sum(
            weight * state
            for weight, state in zip(SIGMA_WEIGHTS, SIGMA_STATES, strict=True)
        )
        state_variance = sum(
            weight * state * state
            for weight, state in zip(SIGMA_WEIGHTS, SIGMA_STATES, strict=True)
        )

        self.assertAlmostEqual(state_mean, 0.0)
        self.assertAlmostEqual(state_variance, 1.0)

    def test_ancestor_gap_is_symmetric_for_related_nodes(self) -> None:
        graph = balanced_reduction_graph(8)
        parent = _parent_map(graph)
        child = graph.nodes[-1].left

        self.assertEqual(_ancestor_gap(child, graph.root, parent), 1)
        self.assertEqual(_ancestor_gap(graph.root, child, parent), 1)
        self.assertIsNone(_ancestor_gap(graph.nodes[-1].left, graph.nodes[-1].right, parent))

    def test_scores_are_finite_nonnegative_and_clip_budget(self) -> None:
        values = wide_range_random(16, seed=22260821).values
        graph = random_pair_merge_graph(16, seed=46_000_000)

        scores = _predictor_scores(values, graph, (1, 4, 15, 30))

        scalar_scores = [scores.full_q]
        for mapping in (
            scores.q_budget,
            scores.prior_phase,
            scores.corr4_zero,
            scores.corr4_shadow,
            scores.geometric_shadow,
        ):
            scalar_scores.extend(mapping.values())
        self.assertTrue(
            all(math.isfinite(score) and score >= 0.0 for score in scalar_scores)
        )
        self.assertEqual(scores.q_budget[15], scores.q_budget[30])


if __name__ == "__main__":
    unittest.main()
