"""Tests for cheap dominant-leaf exposure structural features."""

import math
import unittest
from fractions import Fraction

from predictor_dominant_exposure import dominant_leaf_exposure_features
from summation_graph_predictor import (
    balanced_reduction_graph,
    sequential_reduction_graph,
)


class DominantLeafExposureFeatureTests(unittest.TestCase):
    def test_head_first_sequential_has_repeated_exposure(self) -> None:
        values = (
            Fraction(1),
            Fraction(1, 8),
            Fraction(1, 8),
            Fraction(1, 8),
        )
        features = dominant_leaf_exposure_features(
            values,
            sequential_reduction_graph(4),
        )

        self.assertEqual(features.dominant_leaf_index, 0)
        self.assertEqual(features.path_length, 3)
        self.assertEqual(
            [step.severity_log2 for step in features.steps],
            [3.0, 3.0, 3.0],
        )
        self.assertEqual(features.total_severity_log2, 9.0)
        self.assertEqual(features.mean_severity_log2, 3.0)
        self.assertEqual(features.max_severity_log2, 3.0)

    def test_balanced_tree_exposes_head_later_and_less(self) -> None:
        values = (
            Fraction(1),
            Fraction(1, 8),
            Fraction(1, 8),
            Fraction(1, 8),
        )
        sequential = dominant_leaf_exposure_features(
            values,
            sequential_reduction_graph(4),
        )
        balanced = dominant_leaf_exposure_features(
            values,
            balanced_reduction_graph(4),
        )

        self.assertEqual(balanced.path_length, 2)
        self.assertAlmostEqual(balanced.steps[0].severity_log2, 3.0)
        self.assertAlmostEqual(
            balanced.steps[1].severity_log2,
            math.log2(4.0),
        )
        self.assertLess(
            balanced.total_severity_log2,
            sequential.total_severity_log2,
        )

    def test_sibling_larger_than_dominant_leaf_has_zero_severity(self) -> None:
        values = (
            Fraction(1),
            Fraction(3, 4),
            Fraction(3, 4),
            Fraction(3, 4),
        )
        features = dominant_leaf_exposure_features(
            values,
            balanced_reduction_graph(4),
        )

        self.assertEqual(features.dominant_leaf_index, 0)
        self.assertGreater(features.steps[0].severity_log2, 0.0)
        self.assertEqual(features.steps[1].severity_log2, 0.0)

    def test_dominant_tie_uses_earliest_leaf(self) -> None:
        features = dominant_leaf_exposure_features(
            (Fraction(1), Fraction(1), Fraction(1, 2)),
            balanced_reduction_graph(3),
        )
        self.assertEqual(features.dominant_leaf_index, 0)

    def test_zero_sibling_and_all_zero_boundaries_are_explicit(self) -> None:
        one_sided_zero = dominant_leaf_exposure_features(
            (Fraction(1), Fraction(0)),
            balanced_reduction_graph(2),
        )
        all_zero = dominant_leaf_exposure_features(
            (Fraction(0), Fraction(0)),
            balanced_reduction_graph(2),
        )

        self.assertTrue(math.isinf(one_sided_zero.max_severity_log2))
        self.assertEqual(all_zero.dominant_leaf_index, 0)
        self.assertEqual(all_zero.total_severity_log2, 0.0)
        self.assertEqual(all_zero.max_severity_log2, 0.0)

    def test_single_leaf_has_zero_length_path(self) -> None:
        features = dominant_leaf_exposure_features(
            (Fraction(1),),
            balanced_reduction_graph(1),
        )

        self.assertEqual(features.path_length, 0)
        self.assertEqual(features.total_severity_log2, 0.0)
        self.assertEqual(features.mean_severity_log2, 0.0)
        self.assertEqual(features.max_severity_log2, 0.0)

    def test_rejects_negative_or_mismatched_inputs(self) -> None:
        graph = balanced_reduction_graph(2)
        with self.assertRaises(ValueError):
            dominant_leaf_exposure_features((Fraction(1),), graph)
        with self.assertRaises(ValueError):
            dominant_leaf_exposure_features((Fraction(1), Fraction(-1)), graph)
        with self.assertRaises(TypeError):
            dominant_leaf_exposure_features((Fraction(1), 1.0), graph)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
