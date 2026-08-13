"""Tests for the independent graph-semantic summation predictor."""

import unittest
from fractions import Fraction

from summation_graph_predictor import (
    AdditionNode,
    BinaryReductionGraph,
    balanced_reduction_graph,
    predict_fp32_tree_error,
    round_nonnegative_fraction_to_fp32,
    sequential_reduction_graph,
)
from summation_graph_predictor_validation import (
    SCALE_EXPONENTS,
    validation_rows,
)


class ExactFP32RoundingTests(unittest.TestCase):
    def test_rounds_midpoint_with_ties_to_even(self) -> None:
        midpoint = Fraction(1) + Fraction(1, 2**24)
        minimally_above = midpoint + Fraction(1, 2**35)
        odd_lower_midpoint = Fraction(1) + Fraction(3, 2**24)

        self.assertEqual(
            round_nonnegative_fraction_to_fp32(midpoint).bits_hex,
            "0x3f800000",
        )
        self.assertEqual(
            round_nonnegative_fraction_to_fp32(minimally_above).bits_hex,
            "0x3f800001",
        )
        self.assertEqual(
            round_nonnegative_fraction_to_fp32(odd_lower_midpoint).bits_hex,
            "0x3f800002",
        )

    def test_handles_zero_and_subnormal_tie(self) -> None:
        self.assertEqual(
            round_nonnegative_fraction_to_fp32(Fraction(0)).bits_hex,
            "0x00000000",
        )
        self.assertEqual(
            round_nonnegative_fraction_to_fp32(Fraction(1, 2**150)).bits_hex,
            "0x00000000",
        )
        self.assertEqual(
            round_nonnegative_fraction_to_fp32(Fraction(1, 2**149)).bits_hex,
            "0x00000001",
        )
        self.assertEqual(
            round_nonnegative_fraction_to_fp32(Fraction(1, 2**126)).bits_hex,
            "0x00800000",
        )

    def test_handles_binade_carry_and_maximum_finite_value(self) -> None:
        midpoint_below_two = Fraction(2) - Fraction(1, 2**24)
        maximum_finite = Fraction((2**24) - 1) * Fraction(2**104)

        self.assertEqual(
            round_nonnegative_fraction_to_fp32(midpoint_below_two).bits_hex,
            "0x40000000",
        )
        self.assertEqual(
            round_nonnegative_fraction_to_fp32(maximum_finite).bits_hex,
            "0x7f7fffff",
        )

    def test_rejects_values_outside_the_declared_domain(self) -> None:
        with self.assertRaises(ValueError):
            round_nonnegative_fraction_to_fp32(Fraction(-1))
        with self.assertRaises(TypeError):
            round_nonnegative_fraction_to_fp32(1.0)  # type: ignore[arg-type]


class ExplicitReductionGraphTests(unittest.TestCase):
    def test_graphs_are_full_trees_with_distinct_shapes(self) -> None:
        sequential = sequential_reduction_graph(5)
        balanced = balanced_reduction_graph(5)

        self.assertEqual(len(sequential.nodes), 4)
        self.assertEqual(len(balanced.nodes), 4)
        self.assertNotEqual(sequential.nodes, balanced.nodes)

    def test_rejects_a_disconnected_or_reused_value(self) -> None:
        with self.assertRaises(ValueError):
            BinaryReductionGraph(
                name="invalid",
                leaf_count=3,
                nodes=(
                    AdditionNode(0, 1),
                    AdditionNode(2, 2),
                ),
                root=4,
            )

    def test_requires_already_stored_fp32_inputs(self) -> None:
        graph = sequential_reduction_graph(1)
        with self.assertRaises(ValueError):
            predict_fp32_tree_error((Fraction(1, 10),), graph)


class RegisteredScaledMidpointBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = validation_rows()

    def test_every_preregistered_and_observed_prediction_matches(self) -> None:
        self.assertEqual(len(self.rows), len(SCALE_EXPONENTS) * 3 * 2 * 2)
        self.assertTrue(
            all(row["predictor_matched_preregistered"] for row in self.rows)
        )
        self.assertTrue(all(row["predictor_matched_observation"] for row in self.rows))

    def test_local_residual_sum_is_the_predicted_forward_error(self) -> None:
        for row in self.rows:
            self.assertEqual(
                Fraction(row["local_error_sum_fraction"]),
                Fraction(row["predicted_signed_error_fraction"]),
            )

    def test_each_ordered_input_has_one_hash_shared_by_both_graphs(self) -> None:
        hashes_by_input: dict[tuple[int, int, str], set[str]] = {}
        for row in self.rows:
            key = (
                row["scale_exponent_k"],
                row["tail_count"],
                row["layout"],
            )
            hashes_by_input.setdefault(key, set()).add(row["input_hash"])
        self.assertTrue(all(len(hashes) == 1 for hashes in hashes_by_input.values()))

    def test_above_midpoint_tail_first_cases_are_graph_sensitive(self) -> None:
        for scale_exponent_k in SCALE_EXPONENTS:
            above_count = 2**scale_exponent_k + 1
            rows = {
                row["graph"]: row
                for row in self.rows
                if row["scale_exponent_k"] == scale_exponent_k
                and row["tail_count"] == above_count
                and row["layout"] == "tail_then_head"
            }
            self.assertEqual(
                rows["sequential"]["predicted_sum_bits"],
                "0x3f800001",
            )
            self.assertEqual(
                rows["pairwise"]["predicted_sum_bits"],
                "0x3f800000",
            )


if __name__ == "__main__":
    unittest.main()
