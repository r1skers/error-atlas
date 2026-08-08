import unittest

import numpy as np

from fp32_softmax_summation import (
    IDEAL_DENOMINATOR,
    controlled_end_to_end_rows,
    softmax_with_summation_fp32,
)
from fp32_summation_stress import sequential_sum_fp32


class FP32SoftmaxSummationTests(unittest.TestCase):
    def test_equal_logits_have_exact_inspectable_stages(self) -> None:
        probe = softmax_with_summation_fp32(
            np.array([0.0, 0.0], dtype=np.float32),
            sequential_sum_fp32,
        )

        np.testing.assert_array_equal(
            probe.stored_logits,
            np.array([0.0, 0.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            probe.shifted_logits,
            np.array([0.0, 0.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            probe.exponentials,
            np.array([1.0, 1.0], dtype=np.float32),
        )
        self.assertIsInstance(probe.denominator, np.float32)
        self.assertEqual(float(probe.denominator), 2.0)
        self.assertEqual(probe.probabilities.dtype, np.dtype(np.float32))
        np.testing.assert_array_equal(
            probe.probabilities,
            np.array([0.5, 0.5], dtype=np.float32),
        )

    def test_invalid_logit_dtype_shape_and_size_raise(self) -> None:
        with self.assertRaises(TypeError):
            softmax_with_summation_fp32(
                np.ones(2, dtype=np.float64),
                sequential_sum_fp32,
            )
        with self.assertRaises(ValueError):
            softmax_with_summation_fp32(
                np.ones((1, 2), dtype=np.float32),
                sequential_sum_fp32,
            )
        with self.assertRaises(ValueError):
            softmax_with_summation_fp32(
                np.array([], dtype=np.float32),
                sequential_sum_fp32,
            )

    def test_all_end_to_end_predictions_match_and_errors_decompose(self) -> None:
        rows = controlled_end_to_end_rows()

        self.assertEqual(len(rows), 6)
        self.assertTrue(all(row["prediction_matched"] for row in rows))
        self.assertTrue(all(row["nonzero_tail_count"] == 2 for row in rows))
        for row in rows:
            with self.subTest(dataset=row["dataset"], method=row["method"]):
                self.assertEqual(row["ideal_reference_sum"], IDEAL_DENOMINATOR)
                self.assertAlmostEqual(
                    row["total_error"],
                    row["pre_reduction_error"] + row["summation_error"],
                    places=20,
                )
                self.assertEqual(
                    row["probability_l1_error"],
                    2.0 * row["total_variation"],
                )

        positive_mass_residual = {
            (row["dataset"], row["method"])
            for row in rows
            if row["mass_residual"] > 0.0
        }
        self.assertTrue(
            {
                ("favorable_0_tail_tail", "sequential"),
                ("unfavorable_tail_0_tail", "sequential"),
                ("unfavorable_tail_0_tail", "pairwise"),
            }.issubset(positive_mass_residual)
        )


if __name__ == "__main__":
    unittest.main()
