"""Tests for the top-energy ancestor-transition predictability diagnostic."""

import math
import unittest

import numpy as np

from predictor_ancestor_transition_predictability_diagnostic import (
    SHADOW_DIMENSION,
    TransitionSample,
    _auc,
    _fit_probe,
    _metrics,
    _nearest_selected_ancestor,
    _predict_probe,
    _tree_transitions,
)
from predictor_calibration_inputs import wide_range_random
from predictor_tree_generator import random_pair_merge_graph
from predictor_ulp_energy_convergence_diagnostic import _parent_map
from summation_graph_predictor import balanced_reduction_graph


class AncestorTransitionPredictabilityDiagnosticTests(unittest.TestCase):
    def test_nearest_selected_ancestor_reports_gap_and_branch(self) -> None:
        graph = balanced_reduction_graph(8)
        parent = _parent_map(graph)
        ancestor = graph.root
        branch = graph.nodes[-1].left
        descendant = graph.nodes[branch - graph.leaf_count].left

        relation = _nearest_selected_ancestor(
            descendant,
            {descendant, ancestor},
            parent,
        )

        self.assertEqual(relation, (ancestor, 2, branch))

    def test_transition_features_and_labels_are_well_formed(self) -> None:
        values = wide_range_random(16, seed=22260821).values
        graph = random_pair_merge_graph(16, seed=46_000_000)

        rows = _tree_transitions(values, graph, "pair_merge", 8)

        self.assertGreater(len(rows), 0)
        self.assertEqual(len({row.node_index for row in rows}), len(rows))
        for row in rows:
            self.assertNotEqual(row.node_index, row.ancestor_index)
            self.assertGreater(row.gap, 0)
            self.assertEqual(len(row.features), SHADOW_DIMENSION)
            self.assertTrue(all(math.isfinite(value) for value in row.features))
            self.assertIn(row.crossing, (0, 1))
            self.assertIn(row.sign_flip, (0, 1))
            self.assertIn(row.wrong_cell, (0, 1))
            self.assertEqual(row.wrong_cell, int(row.cell_shift != 0))
            self.assertIsInstance(row.innovation_shift, int)

    def test_auc_handles_perfect_order_and_ties(self) -> None:
        target = np.asarray([0.0, 0.0, 1.0, 1.0])

        self.assertEqual(_auc(target, np.asarray([0.0, 0.2, 0.8, 1.0])), 1.0)
        self.assertEqual(_auc(target, np.asarray([0.5, 0.5, 0.5, 0.5])), 0.5)

    def test_probe_learns_a_separable_feature(self) -> None:
        samples = [
            TransitionSample(
                family="test",
                node_index=index,
                ancestor_index=100 + index,
                gap=1,
                features=(value,) + (0.0,) * (SHADOW_DIMENSION - 1),
                crossing=int(value > 0.0),
                sign_flip=0,
                wrong_cell=0,
                cell_shift=0,
                innovation_shift=0,
                predicted_crossing=0,
                predicted_sign_flip=0,
            )
            for index, value in enumerate((-3.0, -2.0, -1.0, 1.0, 2.0, 3.0))
        ]

        model = _fit_probe(samples, "crossing", 1)
        probability = _predict_probe(model, samples, 1)

        self.assertGreater(probability[-1], probability[0])
        self.assertEqual(_auc(np.asarray([0, 0, 0, 1, 1, 1]), probability), 1.0)

    def test_constant_baseline_has_zero_gain(self) -> None:
        target = np.asarray([0.0, 1.0, 0.0, 1.0])
        probability = np.full(4, 0.5)

        metric = _metrics(target, probability, 0.5)

        self.assertAlmostEqual(metric.log_gain_bits, 0.0)
        self.assertAlmostEqual(metric.brier_skill, 0.0)


if __name__ == "__main__":
    unittest.main()
