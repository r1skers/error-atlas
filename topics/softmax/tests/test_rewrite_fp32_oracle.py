"""Differential tests for the closed-book oracle rewrite against the legacy oracle.

Tests skip while the rewrite still raises NotImplementedError, so the suite stays
green before the user has written the core. Once implemented, agreement must be exact.
"""

import random
import struct
import unittest
from fractions import Fraction

import numpy as np

import summation_graph_predictor as legacy
from predictor_tree_generator import (
    random_contiguous_split_graph,
    random_pair_merge_graph,
)
from rewrite import fp32_oracle as rewrite

SEED = 20260903
PAIR_CASES = 20_000
TREE_CASES = 40
TREE_WIDTHS = (8, 33, 256, 1024)
Q = rewrite.SUBNORMAL_QUANTUM


def _to_tree(graph: legacy.BinaryReductionGraph) -> rewrite.Tree:
    return rewrite.Tree(
        leaf_count=graph.leaf_count,
        nodes=tuple((node.left, node.right) for node in graph.nodes),
    )


def _random_fp32(rng: random.Random, *, max_exponent_field: int = 253) -> Fraction:
    mode = rng.random()
    if mode < 0.15:
        bits = rng.randrange(1, 1 << 23)  # subnormal
    else:
        low = max(1, max_exponent_field - 13) if mode < 0.30 else 1
        bits = (rng.randrange(low, max_exponent_field + 1) << 23) | rng.randrange(1 << 23)
    return Fraction(struct.unpack("<f", struct.pack("<I", bits))[0])


def _tie_addend(rng: random.Random, a: Fraction) -> Fraction:
    """An odd multiple of half an ULP of ``a`` so that a + b is an exact rounding tie."""
    exponent = int(np.frexp(np.float32(float(a)))[1])
    half_ulp = Fraction(2) ** (exponent - 24 - 1)
    return half_ulp * (2 * rng.randrange(1, 8) + 1)


def _skip_unless_implemented(test: unittest.TestCase, call) -> None:
    try:
        call()
    except NotImplementedError:
        test.skipTest("rewrite not implemented yet")


class RoundToFp32Tests(unittest.TestCase):
    def setUp(self) -> None:
        _skip_unless_implemented(self, lambda: rewrite.round_to_fp32(Fraction(1)))

    def test_hand_picked_edge_cases(self) -> None:
        cases = {
            Fraction(0): Fraction(0),
            Q / 2: Fraction(0),  # tie below the smallest subnormal -> even (0)
            Q * Fraction(3, 2): 2 * Q,  # tie -> even significand
            Q * Fraction(5, 2): 2 * Q,  # tie -> even significand
            Q * Fraction(7, 4): 2 * Q,  # nearest
            Fraction(1) + Fraction(1, 2**24): Fraction(1),  # tie at 1 -> even
            Fraction(1) + Fraction(3, 2**24): Fraction(1) + Fraction(1, 2**22),  # 1.5 ULP tie -> even
            Fraction(1) + Fraction(5, 2**25): Fraction(1) + Fraction(1, 2**23),  # 1.25 ULP -> nearest
            Fraction(2**24 - 1) + Fraction(1, 2): Fraction(2**24),  # tie -> even, carry
            rewrite.MAX_FINITE: rewrite.MAX_FINITE,
        }
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(rewrite.round_to_fp32(value), expected)

    def test_overflow_and_sign_contract(self) -> None:
        with self.assertRaises(OverflowError):
            rewrite.round_to_fp32(rewrite.MAX_FINITE + Fraction(2**104))
        with self.assertRaises(ValueError):
            rewrite.round_to_fp32(Fraction(-1))

    def test_matches_legacy_oracle_on_random_sums(self) -> None:
        rng = random.Random(SEED)
        for _ in range(PAIR_CASES):
            a = _random_fp32(rng)
            b = _tie_addend(rng, a) if rng.random() < 0.2 else _random_fp32(rng)
            exact = a + b
            try:
                expected = legacy.round_nonnegative_fraction_to_fp32(exact).value
            except OverflowError:
                with self.assertRaises(OverflowError):
                    rewrite.round_to_fp32(exact)
                continue
            with self.subTest(a=float(a).hex(), b=float(b).hex()):
                self.assertEqual(rewrite.round_to_fp32(exact), expected)


class ReduceTreeTests(unittest.TestCase):
    def setUp(self) -> None:
        tree = rewrite.Tree(leaf_count=2, nodes=((0, 1),))
        _skip_unless_implemented(
            self, lambda: rewrite.reduce_tree((Fraction(1), Fraction(1)), tree)
        )

    def test_rejects_non_stored_inputs(self) -> None:
        tree = rewrite.Tree(leaf_count=2, nodes=((0, 1),))
        for bad in ((Fraction(1, 3), Fraction(1)), (Fraction(-1), Fraction(1))):
            with self.subTest(values=bad):
                with self.assertRaises(ValueError):
                    rewrite.reduce_tree(bad, tree)

    def test_matches_legacy_oracle_on_random_trees(self) -> None:
        rng = random.Random(SEED + 1)
        for case in range(TREE_CASES):
            width = TREE_WIDTHS[case % len(TREE_WIDTHS)]
            values = tuple(
                _random_fp32(rng, max_exponent_field=200) for _ in range(width)
            )
            builder = random_contiguous_split_graph if case % 2 else random_pair_merge_graph
            graph = builder(width, seed=case)
            expected = legacy.predict_fp32_tree_error(values, graph)
            trace = rewrite.reduce_tree(values, _to_tree(graph))
            with self.subTest(case=case):
                self.assertEqual(
                    trace.node_values,
                    tuple(n.rounded_sum for n in expected.node_predictions),
                )
                self.assertEqual(
                    trace.deltas,
                    tuple(n.local_rounding_error for n in expected.node_predictions),
                )
                self.assertEqual(trace.exact_sum, sum(values, Fraction(0)))
                self.assertEqual(trace.error, expected.signed_error)
                self.assertEqual(trace.error, sum(trace.deltas, Fraction(0)))


if __name__ == "__main__":
    unittest.main()
