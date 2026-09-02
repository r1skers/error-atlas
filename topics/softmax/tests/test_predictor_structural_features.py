"""Tests for cheap sibling-scale mismatch structural features."""

import math
import unittest
from fractions import Fraction

from predictor_structural_features import sibling_scale_mismatch_features
from summation_graph_predictor import (
    balanced_reduction_graph,
    sequential_reduction_graph,
)


class SiblingScaleMismatchFeatureTests(unittest.TestCase):
    def test_equal_scale_balanced_tree_has_zero_mismatch(self) -> None:
        values = (Fraction(1),) * 4
        features = sibling_scale_mismatch_features(
            values,
            balanced_reduction_graph(4),
            top_k=2,
        )

        self.assertEqual(features.node_count, 3)
        self.assertEqual(features.mean_log2_gap, 0.0)
        self.assertEqual(features.max_log2_gap, 0.0)
        self.assertEqual(features.top_k, 2)
        self.assertEqual(features.top_k_mean_log2_gap, 0.0)

    def test_sequential_head_tail_is_more_mismatched_than_balanced(self) -> None:
        values = (
            Fraction(1),
            Fraction(1, 8),
            Fraction(1, 8),
            Fraction(1, 8),
        )
        sequential = sibling_scale_mismatch_features(
            values,
            sequential_reduction_graph(4),
            top_k=2,
        )
        balanced = sibling_scale_mismatch_features(
            values,
            balanced_reduction_graph(4),
            top_k=2,
        )

        self.assertGreater(sequential.mean_log2_gap, balanced.mean_log2_gap)
        self.assertGreater(sequential.max_log2_gap, balanced.max_log2_gap)
        self.assertGreater(
            sequential.top_k_mean_log2_gap,
            balanced.top_k_mean_log2_gap,
        )

    def test_node_gap_uses_subtree_mass_not_fp32_rounding(self) -> None:
        values = (
            Fraction(1),
            Fraction(1, 4),
            Fraction(1, 4),
            Fraction(1, 4),
        )
        features = sibling_scale_mismatch_features(
            values,
            balanced_reduction_graph(4),
            top_k=2,
        )
        gaps = [node.log2_gap for node in features.nodes]

        self.assertAlmostEqual(gaps[0], 2.0)
        self.assertAlmostEqual(gaps[1], 0.0)
        self.assertAlmostEqual(gaps[2], math.log2(2.5))

    def test_zero_mass_boundary_is_explicit(self) -> None:
        one_sided_zero = sibling_scale_mismatch_features(
            (Fraction(1), Fraction(0)),
            balanced_reduction_graph(2),
        )
        both_zero = sibling_scale_mismatch_features(
            (Fraction(0), Fraction(0)),
            balanced_reduction_graph(2),
        )

        self.assertTrue(math.isinf(one_sided_zero.max_log2_gap))
        self.assertEqual(both_zero.max_log2_gap, 0.0)

    def test_single_leaf_has_no_internal_mismatch(self) -> None:
        features = sibling_scale_mismatch_features(
            (Fraction(1),),
            balanced_reduction_graph(1),
        )

        self.assertEqual(features.node_count, 0)
        self.assertEqual(features.top_k, 0)
        self.assertEqual(features.mean_log2_gap, 0.0)
        self.assertEqual(features.max_log2_gap, 0.0)
        self.assertEqual(features.top_k_mean_log2_gap, 0.0)

    def test_rejects_negative_or_mismatched_inputs(self) -> None:
        graph = balanced_reduction_graph(2)
        with self.assertRaises(ValueError):
            sibling_scale_mismatch_features((Fraction(1),), graph)
        with self.assertRaises(ValueError):
            sibling_scale_mismatch_features((Fraction(1), Fraction(-1)), graph)
        with self.assertRaises(TypeError):
            sibling_scale_mismatch_features((Fraction(1), 1.0), graph)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            sibling_scale_mismatch_features((Fraction(1), Fraction(1)), graph, top_k=0)


if __name__ == "__main__":
    unittest.main()
