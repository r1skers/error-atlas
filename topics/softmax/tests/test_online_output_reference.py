"""Signed interval boundaries and the declared final-division contract."""

import unittest
from fractions import Fraction as F
from unittest.mock import patch

from online import output_reference as output
from online.fp32_signed import round_to_fp32
from online.merge import merge_reduce
from online.output_probe import leaf_numerators
from online.pilot import BlockFamily, denominator_interval
from online.schedules import sequential_chain, balanced_pairwise
from test_online_pilot import _decimal_real_exp_interval


class DivisionScaffoldingTests(unittest.TestCase):
    def test_single_leaf_signed_third_is_rounded_once(self):
        dump = merge_reduce((F(0),), (F(3),), sequential_chain(1))
        result = output.finish_output(dump, (F(-1),))
        self.assertEqual(result.before_division_rounding, F(-1, 3))
        self.assertEqual(result.value, F(-11184811, 33554432))
        self.assertEqual(result.division_residual, result.value + F(1, 3))

    def test_constant_values_are_exact_final_outputs(self):
        family = BlockFamily(tuple(map(F, (-2, 0, -1, -3))), (32,) * 4)
        for tree in (sequential_chain(4), balanced_pairwise(4)):
            for fused in (False, True):
                dump = merge_reduce(family.leaf_max, family.leaf_ell, tree, fused=fused)
                for v in (F(0), F(1), F(-1), F(1, 2)):
                    result = output.finish_output(dump, leaf_numerators(family, (v,) * 4))
                    self.assertEqual(result.value, v)
                    self.assertEqual(result.division_residual, 0)

    def test_nonpositive_computed_denominator_rejected(self):
        dump = merge_reduce((F(0),), (F(0),), sequential_chain(1))
        with self.assertRaises(ValueError):
            output.finish_output(dump, (F(0),))


class ReferenceGuardsTests(unittest.TestCase):
    def test_bad_intervals_rejected_before_pending_core(self):
        for num, den in (((F(1), F(-1)), (F(1), F(2))),
                         ((F(-1), F(1)), (F(2), F(1))),
                         ((F(0), F(1)), (F(0), F(1))),
                         ((F(0), F(1)), (F(-2), F(-1)))):
            with self.assertRaises(ValueError):
                output.quotient_interval(num, den)

    def test_bad_V_length_rejected_before_pending_core(self):
        with self.assertRaises(ValueError):
            output.numerator_interval(BlockFamily((F(0),), (32,)), ())


class SignedNumeratorTests(unittest.TestCase):
    def setUp(self):
        try:
            output.numerator_interval(BlockFamily((F(0),), (32,)), (F(0),))
        except NotImplementedError:
            self.skipTest("USER-WRITTEN numerator interval pending")

    def test_negative_coefficient_reverses_exp_endpoints(self):
        family = BlockFamily((F(0), F(-1)), (32, 32))
        with patch.object(output, "real_exp_interval", side_effect=lambda gap: (F(1), F(1)) if gap == 0 else (F(1, 4), F(1, 2))):
            self.assertEqual(output.numerator_interval(family, (F(0), F(-1))), (F(-16), F(-8)))

    def test_signed_terms_are_added_including_mass_and_half_values(self):
        family = BlockFamily((F(0), F(-1)), (3, 7))
        with patch.object(output, "real_exp_interval", side_effect=lambda gap: (F(1), F(1)) if gap == 0 else (F(1, 4), F(1, 2))):
            self.assertEqual(output.numerator_interval(family, (F(1, 2), F(-1, 2))), (F(-1, 4), F(5, 8)))

    def test_exact_non_fp32_gap_encloses_independent_decimal_reference(self):
        family = BlockFamily((round_to_fp32(F(1, 3)), F(-10)), (1, 3))
        gap = family.maxima[1] - family.global_max
        self.assertNotEqual(gap, round_to_fp32(gap))
        lo, hi = output.numerator_interval(family, (F(0), F(-1)))
        # A much tighter independent Decimal bracket should fit within Taylor bounds.
        dlo, dhi = _decimal_real_exp_interval(gap)
        self.assertLessEqual(lo, -3 * dhi)
        self.assertGreaterEqual(hi, -3 * dlo)

    def test_zero_and_all_negative_numerator(self):
        family = BlockFamily((F(0), F(-1)), (32, 32))
        self.assertEqual(output.numerator_interval(family, (F(0), F(0))), (0, 0))
        lo, hi = output.numerator_interval(family, (F(-1), F(-1)))
        dlow, dhigh = denominator_interval(family)
        self.assertLessEqual(lo, hi)
        self.assertLess(hi, 0)
        self.assertLessEqual(lo, -dlow)
        self.assertGreaterEqual(hi, -dhigh)


class QuotientTests(unittest.TestCase):
    def setUp(self):
        try:
            output.quotient_interval((F(0), F(1)), (F(1), F(2)))
        except NotImplementedError:
            self.skipTest("USER-WRITTEN quotient interval pending")

    def test_positive_negative_crossing_and_zero_cases(self):
        for numerator, expected in (((2, 3), (F(1, 4), F(3, 4))),
                                    ((-3, -2), (F(-3, 4), F(-1, 4))),
                                    ((-3, 2), (F(-3, 4), F(1, 2))),
                                    ((0, 0), (F(0), F(0)))):
            self.assertEqual(output.quotient_interval(tuple(map(F, numerator)), (F(4), F(8))), expected)

    def test_no_binary_float_conversion_for_tiny_signed_values(self):
        q = F(1, 10**400)
        self.assertEqual(output.quotient_interval((-q, q), (F(2), F(3))), (-q / 2, q / 2))


class CombinedReferenceTests(unittest.TestCase):
    def setUp(self):
        try:
            output.reference_output(BlockFamily((F(0), F(-1)), (32, 32)), (F(1), F(0)))
        except NotImplementedError:
            self.skipTest("USER-WRITTEN output interval cores pending")

    def test_constant_V_reference_is_exact_despite_component_interval_width(self):
        family = BlockFamily((F(0), F(-1)), (32, 32))
        for v in (F(0), F(1), F(-1), F(1, 2)):
            result = output.reference_output(family, (v, v))
            self.assertEqual(result.output, (v, v))
            self.assertEqual(result.denominator, denominator_interval(family))

    def test_equal_logits_signed_values_have_known_exact_output(self):
        family = BlockFamily((F(0), F(0)), (3, 1))
        result = output.reference_output(family, (F(1), F(-1)))
        self.assertLessEqual(result.output[0], F(1, 2))
        self.assertGreaterEqual(result.output[1], F(1, 2))


if __name__ == "__main__":
    unittest.main()
