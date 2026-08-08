import math
import unittest

import numpy as np

from fp32_summation_stress import (
    FP32_UNIT_ROUNDOFF,
    alternating_terms,
    adversarial_terms,
    compensated_sum_fp32,
    controlled_permutation_rows,
    fp64_reference_sum,
    pairwise_sum_fp32,
    sequential_sum_fp32,
    sequential_summation_probe,
)


class FP32SequentialSummationTests(unittest.TestCase):
    def test_adversarial_tail_terms_are_nonzero(self) -> None:
        values = adversarial_terms(2)

        self.assertEqual(values.dtype, np.dtype(np.float32))
        self.assertTrue(np.all(values[1:] == FP32_UNIT_ROUNDOFF))
        self.assertEqual(np.count_nonzero(values[1:]), 2)

    def test_large_first_order_loses_two_half_ulp_terms(self) -> None:
        computed = sequential_sum_fp32(adversarial_terms(2))

        self.assertIsInstance(computed, np.float32)
        self.assertEqual(float(computed), 1.0)

    def test_registered_probe_separates_nonzero_input_from_sum_loss(self) -> None:
        probe = sequential_summation_probe(2**20)

        self.assertEqual(probe.nonzero_tail_count, 2**20)
        self.assertEqual(probe.computed_sum, 1.0)
        self.assertEqual(probe.reference_sum, 1.0 + 2.0**-4)
        self.assertEqual(probe.absolute_error, -(2.0**-4))
        self.assertEqual(probe.relative_error, -1.0 / 17.0)

    def test_matrix_dimension_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            sequential_sum_fp32(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))

    def test_invalid_dtype_raises(self) -> None:
        with self.assertRaises(TypeError):
            sequential_sum_fp32(np.array([1.0, 2.0], dtype=np.float64))


class FP32PairwiseSummationTests(unittest.TestCase):
    def test_three_term_tree_combines_the_two_tail_terms_first(self) -> None:
        computed = pairwise_sum_fp32(adversarial_terms(2))

        self.assertIsInstance(computed, np.float32)
        self.assertEqual(float(computed), 1.0 + 2.0**-23)

    def test_tree_error_respects_depth_bound_and_beats_large_first(self) -> None:
        values = adversarial_terms(2**10)
        computed = float(pairwise_sum_fp32(values))
        reference = fp64_reference_sum(values)
        relative_error = abs(computed - reference) / reference
        depth = math.ceil(math.log2(values.size))
        unit_roundoff = float(FP32_UNIT_ROUNDOFF)
        gamma_depth = (depth * unit_roundoff) / (1.0 - depth * unit_roundoff)

        self.assertLessEqual(relative_error, gamma_depth)
        self.assertLess(
            abs(computed - reference),
            abs(float(sequential_sum_fp32(values)) - reference),
        )

    def test_tree_rejects_invalid_shape_and_dtype(self) -> None:
        with self.assertRaises(ValueError):
            pairwise_sum_fp32(np.ones((2, 2), dtype=np.float32))
        with self.assertRaises(TypeError):
            pairwise_sum_fp32(np.ones(4, dtype=np.float64))

    def test_alternating_layout_exposes_lowest_level_tree_rounding(self) -> None:
        pair_count = 2**10
        values = alternating_terms(pair_count)
        computed = float(pairwise_sum_fp32(values))
        reference = fp64_reference_sum(values)
        expected_absolute_error = -pair_count * float(FP32_UNIT_ROUNDOFF)

        self.assertEqual(np.count_nonzero(values[1::2]), pair_count)
        self.assertEqual(computed, float(pair_count))
        self.assertEqual(computed - reference, expected_absolute_error)
        self.assertEqual(
            (computed - reference) / reference,
            -float(FP32_UNIT_ROUNDOFF) / (1.0 + float(FP32_UNIT_ROUNDOFF)),
        )

    def test_same_multiset_different_order_changes_fixed_tree_result(self) -> None:
        favorable = np.array(
            [1.0, FP32_UNIT_ROUNDOFF, FP32_UNIT_ROUNDOFF],
            dtype=np.float32,
        )
        unfavorable = np.array(
            [FP32_UNIT_ROUNDOFF, 1.0, FP32_UNIT_ROUNDOFF],
            dtype=np.float32,
        )

        np.testing.assert_array_equal(np.sort(favorable), np.sort(unfavorable))
        self.assertEqual(
            fp64_reference_sum(favorable),
            fp64_reference_sum(unfavorable),
        )
        self.assertEqual(
            float(pairwise_sum_fp32(favorable)),
            1.0 + 2.0**-23,
        )
        self.assertEqual(float(pairwise_sum_fp32(unfavorable)), 1.0)


class FP32CompensatedSummationTests(unittest.TestCase):
    def test_compensation_recovers_lost_term_in_unfavorable_order(self) -> None:
        values = np.array(
            [FP32_UNIT_ROUNDOFF, 1.0, FP32_UNIT_ROUNDOFF],
            dtype=np.float32,
        )

        computed = compensated_sum_fp32(values)

        self.assertIsInstance(computed, np.float32)
        self.assertEqual(float(computed), fp64_reference_sum(values))
        self.assertEqual(float(computed), 1.0 + 2.0**-23)

    def test_compensation_rejects_invalid_shape_and_dtype(self) -> None:
        with self.assertRaises(ValueError):
            compensated_sum_fp32(np.ones((2, 2), dtype=np.float32))
        with self.assertRaises(TypeError):
            compensated_sum_fp32(np.ones(4, dtype=np.float64))


class ControlledPermutationExperimentTests(unittest.TestCase):
    def test_all_six_registered_predictions_match(self) -> None:
        rows = controlled_permutation_rows()

        self.assertEqual(len(rows), 6)
        self.assertTrue(all(row["prediction_matched"] for row in rows))
        self.assertEqual(
            {row["reference_sum"] for row in rows},
            {1.0 + 2.0**-23},
        )



if __name__ == "__main__":
    unittest.main()
