"""Differential and replication tests for the closed-book A/C decomposition rewrite.

Skips while the rewrite raises NotImplementedError. The differential part requires
exact Fraction agreement with reduction_analysis; the replication part re-derives the
"coherence dominates tree-to-tree variation" observation on wide-range inputs using
only the rewritten oracle and decomposition.
"""

import random
import unittest
from fractions import Fraction

import summation_graph_predictor as legacy
from predictor_calibration_inputs import (
    head_tail_random,
    same_scale_random,
    wide_range_random,
)
from predictor_tree_generator import (
    random_contiguous_split_graph,
    random_pair_merge_graph,
)
from reduction_analysis import CoherenceAnalysis, replay
from rewrite import coherence as rewrite
from rewrite import fp32_oracle

SEED = 20260904
DIFFERENTIAL_CASES = 30
REPLICATION_WIDTH = 256
REPLICATION_INPUT_SEEDS = (1, 2, 3, 22260821)
REPLICATION_TREES = 32
# Legacy implementation gives 2.5-3.6 on these seeds; 1.5 is a loose guard, not a research claim.
REPLICATION_MIN_RATIO = 1.5


def _to_tree(graph: legacy.BinaryReductionGraph) -> fp32_oracle.Tree:
    return fp32_oracle.Tree(
        leaf_count=graph.leaf_count,
        nodes=tuple((node.left, node.right) for node in graph.nodes),
    )


def _skip_unless_implemented(test: unittest.TestCase, call) -> None:
    try:
        call()
    except NotImplementedError:
        test.skipTest("rewrite not implemented yet")


def _toy_trace(deltas: tuple[Fraction, ...]) -> fp32_oracle.Trace:
    error = sum(deltas, Fraction(0))
    return fp32_oracle.Trace(
        values=(), node_values=(), deltas=deltas, exact_sum=Fraction(0), error=error
    )


class ACDecompositionTests(unittest.TestCase):
    def setUp(self) -> None:
        _skip_unless_implemented(
            self, lambda: rewrite.ac_decomposition(_toy_trace((Fraction(1),)))
        )

    def test_toy_example_from_docstring(self) -> None:
        first = rewrite.ac_decomposition(_toy_trace((Fraction(-1), Fraction(1, 4))))
        second = rewrite.ac_decomposition(_toy_trace((Fraction(1, 4), Fraction(1))))
        self.assertEqual((first.a_local, first.c_coherence, first.e2),
                         (Fraction(17, 16), Fraction(-1, 2), Fraction(9, 16)))
        self.assertEqual((second.a_local, second.c_coherence, second.e2),
                         (Fraction(17, 16), Fraction(1, 2), Fraction(25, 16)))

    def test_single_node_has_no_coherence(self) -> None:
        split = rewrite.ac_decomposition(_toy_trace((Fraction(3, 8),)))
        self.assertEqual(split.c_coherence, 0)
        self.assertEqual(split.a_local, split.e2)

    def test_matches_reduction_analysis_exactly(self) -> None:
        rng = random.Random(SEED)
        families = (wide_range_random, head_tail_random, same_scale_random)
        for case in range(DIFFERENTIAL_CASES):
            width = rng.choice((8, 33, 128))
            values = families[case % 3](width, seed=rng.randrange(1, 10**6)).values
            builder = random_contiguous_split_graph if case % 2 else random_pair_merge_graph
            graph = builder(width, seed=case)
            expected = CoherenceAnalysis(replay(values, graph)).ac
            actual = rewrite.ac_decomposition(
                fp32_oracle.reduce_tree(values, _to_tree(graph))
            )
            with self.subTest(case=case):
                self.assertEqual(actual.e2, expected.e2)
                self.assertEqual(actual.a_local, expected.a_local)
                self.assertEqual(actual.c_coherence, expected.c_coherence)
                self.assertEqual(actual.e2, actual.a_local + actual.c_coherence)


class CoherenceDominanceReplicationTests(unittest.TestCase):
    def setUp(self) -> None:
        _skip_unless_implemented(
            self, lambda: rewrite.ac_decomposition(_toy_trace((Fraction(1),)))
        )
        _skip_unless_implemented(self, lambda: rewrite.variation_ratio(()))

    def test_variation_ratio_edge_cases(self) -> None:
        split = rewrite.ac_decomposition(_toy_trace((Fraction(1),)))
        self.assertTrue(rewrite.variation_ratio(()) != rewrite.variation_ratio(()))  # nan
        self.assertTrue(rewrite.variation_ratio((split, split)) != rewrite.variation_ratio((split, split)))

    def test_coherence_dominates_tree_variation_on_wide_range_inputs(self) -> None:
        for input_seed in REPLICATION_INPUT_SEEDS:
            values = wide_range_random(REPLICATION_WIDTH, seed=input_seed).values
            splits = []
            for k in range(REPLICATION_TREES):
                builder = random_contiguous_split_graph if k % 2 else random_pair_merge_graph
                graph = builder(REPLICATION_WIDTH, seed=1000 + k)
                splits.append(
                    rewrite.ac_decomposition(
                        fp32_oracle.reduce_tree(values, _to_tree(graph))
                    )
                )
            ratio = rewrite.variation_ratio(splits)
            with self.subTest(input_seed=input_seed, ratio=ratio):
                self.assertGreater(ratio, REPLICATION_MIN_RATIO)


if __name__ == "__main__":
    unittest.main()
