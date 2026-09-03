"""Differential check: the Fraction-based RN-even oracle must agree with hardware binary32.

This is an engineering guard for the exact oracle, not research evidence. It compares
`round_nonnegative_fraction_to_fp32` and whole-tree `predict_fp32_tree_error` against
NumPy float32 addition (IEEE 754 round-to-nearest, ties-to-even on the host CPU),
covering subnormals, wide exponent gaps, the near-overflow band and forced exact ties.
Any future closed-book rewrite of the oracle should pass this same test unchanged.
"""

import random
import struct
import unittest
from fractions import Fraction

import numpy as np

from predictor_tree_generator import (
    random_contiguous_split_graph,
    random_pair_merge_graph,
)
from summation_graph_predictor import (
    predict_fp32_tree_error,
    round_nonnegative_fraction_to_fp32,
)

SEED = 20260902
PAIR_CASES = 20_000
TREE_CASES = 40
TREE_WIDTHS = (8, 33, 256, 1024)


def _bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def _random_positive_float32(rng: random.Random, *, max_exponent_field: int = 253) -> float:
    mode = rng.random()
    if mode < 0.15:
        bits = rng.randrange(1, 1 << 23)  # subnormal
    elif mode < 0.30:
        high = max(1, max_exponent_field - 13)  # top band of the allowed range
        bits = (rng.randrange(high, max_exponent_field + 1) << 23) | rng.randrange(1 << 23)
    else:
        bits = (rng.randrange(1, max_exponent_field + 1) << 23) | rng.randrange(1 << 23)
    return _from_bits(bits)


def _forced_tie_addend(rng: random.Random, a: float) -> float:
    """An odd multiple of half an ULP of `a`, so a + b lies exactly on a rounding tie."""
    exponent = int(np.frexp(np.float32(a))[1])
    half_ulp = 2.0 ** (exponent - 24 - 1)
    return float(np.float32(half_ulp * (2 * rng.randrange(1, 8) + 1)))


class OracleHardwareDifferentialTests(unittest.TestCase):
    def test_pairwise_rounding_matches_float32_hardware(self) -> None:
        rng = random.Random(SEED)
        checked = 0
        for _ in range(PAIR_CASES):
            a = _random_positive_float32(rng)
            b = _random_positive_float32(rng)
            if rng.random() < 0.2:
                b = _forced_tie_addend(rng, a)
                if b == 0.0:
                    continue
            hardware = np.float32(a) + np.float32(b)
            exact = Fraction(a) + Fraction(b)
            if not np.isfinite(hardware):
                with self.assertRaises(OverflowError):
                    round_nonnegative_fraction_to_fp32(exact)
                continue
            oracle = round_nonnegative_fraction_to_fp32(exact)
            with self.subTest(a=a.hex(), b=b.hex()):
                self.assertEqual(oracle.value, Fraction(float(hardware)))
                self.assertEqual(int(oracle.bits_hex, 16), _bits(float(hardware)))
            checked += 1
        self.assertGreater(checked, PAIR_CASES // 2)

    def test_whole_tree_states_match_float32_hardware(self) -> None:
        rng = random.Random(SEED + 1)
        for case in range(TREE_CASES):
            width = TREE_WIDTHS[case % len(TREE_WIDTHS)]
            # Keep leaves well below overflow so every internal sum stays finite.
            values = tuple(
                Fraction(_random_positive_float32(rng, max_exponent_field=200))
                for _ in range(width)
            )
            builder = random_contiguous_split_graph if case % 2 else random_pair_merge_graph
            graph = builder(width, seed=case)
            prediction = predict_fp32_tree_error(values, graph)

            state: list = [np.float32(float(v)) for v in values] + [None] * len(graph.nodes)
            for offset, node in enumerate(graph.nodes):
                index = graph.leaf_count + offset
                state[index] = np.float32(state[node.left] + state[node.right])
                with self.subTest(case=case, node=index):
                    self.assertEqual(
                        prediction.node_predictions[offset].rounded_sum,
                        Fraction(float(state[index])),
                    )
            with self.subTest(case=case, root=True):
                self.assertEqual(prediction.predicted_sum, Fraction(float(state[graph.root])))


if __name__ == "__main__":
    unittest.main()
