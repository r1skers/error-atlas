"""Tests for the sparse shadow-repair calibration diagnostic."""

import unittest
from fractions import Fraction

from predictor_shadow_sparse_repair_ablation import _crosses_boundary


class SparseShadowRepairTests(unittest.TestCase):
    def test_boundary_cross_requires_a_different_rn_cell(self) -> None:
        self.assertFalse(_crosses_boundary(Fraction(1, 4), Fraction(49, 100)))
        self.assertTrue(_crosses_boundary(Fraction(1, 4), Fraction(51, 100)))

    def test_boundary_cross_is_symmetric(self) -> None:
        left = Fraction(33_554_433, 4)
        right = Fraction(16_777_665, 2)
        self.assertTrue(_crosses_boundary(left, right))
        self.assertTrue(_crosses_boundary(right, left))

    def test_boundary_cross_respects_ties_to_even(self) -> None:
        self.assertFalse(_crosses_boundary(Fraction(1, 2), Fraction(49, 100)))
        self.assertTrue(_crosses_boundary(Fraction(1, 2), Fraction(51, 100)))
        self.assertTrue(_crosses_boundary(Fraction(3, 2), Fraction(149, 100)))


if __name__ == "__main__":
    unittest.main()
