"""Tests for the ULP-energy cost/accuracy Pareto diagnostic."""

import unittest
from fractions import Fraction

from predictor_ulp_energy_cost_pareto_diagnostic import (
    _pareto_indices,
    _root_band_internal_order,
    _tree_cost_profile,
)
from summation_graph_predictor import (
    balanced_reduction_graph,
    sequential_reduction_graph,
)


class UlpEnergyCostParetoDiagnosticTests(unittest.TestCase):
    def test_balanced_and_sequential_span(self) -> None:
        values = tuple(Fraction(1) for _ in range(8))
        budgets = (1, 2, 4, 7, 12)

        balanced = _tree_cost_profile(
            values,
            balanced_reduction_graph(8),
            "balanced",
            budgets,
        )
        sequential = _tree_cost_profile(
            values,
            sequential_reduction_graph(8),
            "sequential",
            budgets,
        )

        self.assertEqual(balanced.work, 7)
        self.assertEqual(sequential.work, 7)
        self.assertEqual(balanced.span, 3)
        self.assertEqual(sequential.span, 7)
        self.assertLess(balanced.full_q, sequential.full_q)

    def test_budgeted_q_is_monotone_and_clips_to_full_tree(self) -> None:
        values = tuple(Fraction(1) for _ in range(8))
        profile = _tree_cost_profile(
            values,
            balanced_reduction_graph(8),
            "balanced",
            (1, 2, 4, 7, 12),
        )

        budgeted = list(profile.budget_q.values())
        self.assertEqual(budgeted, sorted(budgeted))
        self.assertEqual(profile.budget_q[7], profile.full_q)
        self.assertEqual(profile.budget_q[12], profile.full_q)

    def test_profile_rejects_invalid_budget(self) -> None:
        with self.assertRaises(ValueError):
            _tree_cost_profile(
                (Fraction(1), Fraction(1)),
                balanced_reduction_graph(2),
                "balanced",
                (0,),
            )

    def test_root_band_is_root_first_and_complete_when_unbounded(self) -> None:
        graph = balanced_reduction_graph(8)
        subtree_leaves = [1] * graph.leaf_count
        for node in graph.nodes:
            subtree_leaves.append(
                subtree_leaves[node.left] + subtree_leaves[node.right]
            )

        order = _root_band_internal_order(graph, subtree_leaves, 20)

        self.assertEqual(order[0], graph.root)
        self.assertEqual(len(order), len(graph.nodes))
        self.assertEqual(len(set(order)), len(order))

    def test_pareto_indices_minimize_q_and_span(self) -> None:
        values = tuple(Fraction(1) for _ in range(8))
        balanced = _tree_cost_profile(
            values,
            balanced_reduction_graph(8),
            "balanced",
            (1,),
        )
        sequential = _tree_cost_profile(
            values,
            sequential_reduction_graph(8),
            "sequential",
            (1,),
        )

        self.assertEqual(_pareto_indices([balanced, sequential]), {0})


if __name__ == "__main__":
    unittest.main()
