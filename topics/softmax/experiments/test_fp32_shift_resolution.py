import math
import unittest

from fp32_shift_resolution import fp32_softmax_probe
from rewrite_fp32_shift_resolution import rewritten_fp32_softmax_probe


class FP32ShiftResolutionTests(unittest.TestCase):
    def test_unit_gap_is_preserved_at_two_to_the_twenty_three(self) -> None:
        stored_difference, first_probability = fp32_softmax_probe(2**23)

        self.assertEqual(stored_difference, 1.0)
        self.assertAlmostEqual(
            first_probability,
            1.0 / (1.0 + math.exp(-1.0)),
            places=6,
        )

    def test_unit_gap_collapses_at_two_to_the_twenty_four(self) -> None:
        stored_difference, first_probability = fp32_softmax_probe(2**24)

        self.assertEqual(stored_difference, 0.0)
        self.assertEqual(first_probability, 0.5)


class ClosedBookRewriteTests(unittest.TestCase):
    def test_rewrite_preserves_the_boundary_mechanism(self) -> None:
        expected = {
            2**23: (1.0, 1.0 / (1.0 + math.exp(-1.0))),
            2**24: (0.0, 0.5),
            2**25: (0.0, 0.5),
        }

        for common_offset, (expected_difference, expected_probability) in expected.items():
            with self.subTest(common_offset=common_offset):
                stored_difference, first_probability = (
                    rewritten_fp32_softmax_probe(common_offset)
                )
                self.assertEqual(stored_difference, expected_difference)
                self.assertAlmostEqual(
                    first_probability,
                    expected_probability,
                    places=6,
                )


if __name__ == "__main__":
    unittest.main()
