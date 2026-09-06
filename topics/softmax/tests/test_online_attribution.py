"""Attribution specification: guards run now, core checks skip until user implementation.

Only a known flat probe may skip in setUp. A later NotImplementedError in an edge case
is a failure, so a partial implementation cannot silently skip the harder cases.
"""

import unittest
from dataclasses import replace
from decimal import Context, Decimal
from fractions import Fraction
from unittest.mock import patch

from online import attribution, pilot, schedules
from online.fp32_exp import correctly_rounded_exp
from online.merge import merge_reduce, weighted_residuals


def flat(count=4):
    return pilot.BlockFamily((Fraction(0),) * count, (32,) * count)


def dump_for(family, fused=False):
    return merge_reduce(family.leaf_max, family.leaf_ell,
                        schedules.sequential_chain(len(family.maxima)), fused=fused)


class GuardTests(unittest.TestCase):
    def test_valid_inputs_clear_guards_without_implementing_the_core(self):
        family = flat()
        attribution._validate_inputs(family, dump_for(family))

    def test_rejects_a_different_reference_family(self):
        family = flat()
        other = replace(family, masses=(16,) * 4)
        with self.assertRaisesRegex(ValueError, "same constant-block family"):
            attribution.decompose(other, dump_for(family))

    def test_rejects_malformed_dump(self):
        family = flat()
        with self.assertRaisesRegex(ValueError, "well-formed dump"):
            attribution.decompose(family, replace(dump_for(family), node_ell=()))

    def test_rejects_wrong_normalization_or_impossible_weights(self):
        family, dump = flat(), dump_for(flat())
        for invalid in (replace(dump, node_max=(Fraction(0), Fraction(0), Fraction(1))),
                        replace(dump, weight_left=(Fraction(2),) * 3),
                        replace(dump, node_ell=(Fraction(-1),) * 3)):
            with self.subTest(dump=invalid), self.assertRaises(ValueError):
                attribution.decompose(family, invalid)

    def test_rejects_reversed_cancellation_interval(self):
        with self.assertRaises(ValueError):
            attribution.cancellation_savings(Fraction(1), (Fraction(2), Fraction(1)))


class DecompositionTests(unittest.TestCase):
    def setUp(self):
        try:
            self.flat_result = attribution.decompose(flat(), dump_for(flat()))
        except NotImplementedError:
            self.skipTest("USER-WRITTEN decompose is not implemented")

    def test_flat_case_has_no_error_or_budget(self):
        result = self.flat_result
        self.assertEqual(result.reference, (128, 128))
        self.assertEqual((result.computed, result.frozen), (128, 128))
        self.assertEqual(result.rounding_error, 0)
        self.assertEqual(result.weight_error, (0, 0))
        self.assertEqual(result.total_error, (0, 0))
        self.assertEqual(result.residuals, (0, 0, 0))
        self.assertEqual(result.rounding_budget, 0)
        self.assertEqual(result.relative_rounding, (0, 0))
        self.assertEqual(result.relative_weight, (0, 0))
        self.assertEqual(result.relative_total, (0, 0))

    def test_single_leaf_has_an_empty_residual_sum(self):
        family = pilot.BlockFamily((Fraction(-3),), (7,))
        result = attribution.decompose(family, dump_for(family))
        self.assertEqual((result.computed, result.frozen, result.reference), (7, 7, (7, 7)))
        self.assertEqual(result.residuals, ())
        self.assertEqual(result.rounding_budget, 0)
        self.assertEqual(result.total_error, (0, 0))

    def test_known_max_position_fixtures_have_opposite_signed_rounding_errors(self):
        frozen = 32 + 992 * correctly_rounded_exp(Fraction(-20))
        for position, computed in ((0, Fraction(32)), (31, 32 + Fraction(1, 2**18))):
            family = pilot.max_at_position(32, 32, Fraction(20), position)
            result = attribution.decompose(family, dump_for(family))
            with self.subTest(position=position):
                self.assertEqual(result.frozen, frozen)
                self.assertEqual(result.computed, computed)
                self.assertEqual(result.rounding_error, computed - frozen)
                self.assertEqual(sum(result.residuals, Fraction(0)), result.rounding_error)
                self.assertEqual(result.total_error,
                                 tuple(result.rounding_error + endpoint for endpoint in result.weight_error))
                self.assertEqual(result.total_error,
                                 (computed - result.reference[1], computed - result.reference[0]))
                self.assertGreaterEqual(result.rounding_budget, abs(result.rounding_error))
                self.assertEqual(result.rounding_error < 0, position == 0)

    def test_weight_error_contains_independent_real_exp_reference_not_fp32_exp(self):
        family = pilot.max_at_position(32, 32, Fraction(20), 0)
        result = attribution.decompose(family, dump_for(family))
        context = Context(prec=200)
        exponential = context.exp(Decimal(-20))
        real_low = 32 + 992 * Fraction(context.next_minus(exponential))
        real_high = 32 + 992 * Fraction(context.next_plus(exponential))
        self.assertLessEqual(result.reference[0], real_low)
        self.assertGreaterEqual(result.reference[1], real_high)
        self.assertLessEqual(result.weight_error[0], result.frozen - real_high)
        self.assertGreaterEqual(result.weight_error[1], result.frozen - real_low)
        self.assertNotEqual(result.weight_error, (Fraction(0), Fraction(0)))
        self.assertLess(result.weight_error[1] - result.weight_error[0], Fraction(1, 10**100))

    def test_propagated_residuals_and_absolute_budget_in_both_arms(self):
        family = pilot.BlockFamily((Fraction(0), Fraction(20), Fraction(25), Fraction(1)), (32,) * 4)
        for fused in (False, True):
            dump = dump_for(family, fused)
            result = attribution.decompose(family, dump)
            terms = weighted_residuals(dump)
            self.assertEqual(result.residuals, terms)
            self.assertEqual(result.rounding_budget, sum(map(abs, terms), Fraction(0)))
            self.assertEqual(result.rounding_error, sum(terms, Fraction(0)))
            self.assertGreater(result.rounding_budget, abs(result.rounding_error))
            self.assertEqual(result.reference, pilot.denominator_interval(family))

    def test_broken_identity_is_rejected_not_reported_as_success(self):
        with patch.object(attribution, "weighted_residuals", return_value=(Fraction(1),) * 3):
            with self.assertRaises(ArithmeticError):
                attribution.decompose(flat(), dump_for(flat()))

    def test_normalization_keeps_sign_and_shared_reference_dependence(self):
        # Artificially widen only the reference in this unit test, to expose dependency
        # bugs that would be hidden by a ~1e-140-wide real bracket. A valid enclosure.
        for position in (0, 31):
            family = pilot.max_at_position(32, 32, Fraction(20), position)
            with patch.object(attribution, "denominator_interval", return_value=(Fraction(31), Fraction(34))):
                result = attribution.decompose(family, dump_for(family))
            r = result.rounding_error
            self.assertEqual(r < 0, position == 0)
            expected = (r / 31, r / 34) if position == 0 else (r / 34, r / 31)
            self.assertEqual(result.relative_rounding, expected)
            self.assertEqual(result.relative_weight, (result.frozen / 34 - 1, result.frozen / 31 - 1))
            self.assertEqual(result.relative_total, (result.computed / 34 - 1, result.computed / 31 - 1))
        # Separate interval addition can be wider; do not require endpoint-wise equality
        # of relative_rounding + relative_weight with the tight relative_total interval.


class CancellationTests(unittest.TestCase):
    def setUp(self):
        try:
            attribution.cancellation_savings(Fraction(0), (Fraction(0), Fraction(0)))
        except NotImplementedError:
            self.skipTest("USER-WRITTEN cancellation_savings is not implemented")

    def test_zero_and_same_sign_components(self):
        for r, interval in ((0, (-5, 7)), (2, (0, 5)), (-2, (-5, 0))):
            self.assertEqual(attribution.cancellation_savings(Fraction(r), tuple(map(Fraction, interval))), (0, 0))

    def test_complete_cancellation_and_saturation(self):
        for r, interval, expected in ((2, (-2, -2), (4, 4)), (2, (-5, -3), (4, 4)),
                                      (-2, (3, 5), (4, 4))):
            self.assertEqual(attribution.cancellation_savings(Fraction(r), tuple(map(Fraction, interval))), expected)

    def test_intervals_crossing_a_kink_and_sign_symmetry(self):
        for r, interval, expected in ((2, (-3, -1), (2, 4)), (2, (-1, 1), (0, 2)),
                                      (-2, (1, 3), (2, 4)), (-2, (-1, 1), (0, 2)),
                                      (2, (-3, 1), (0, 4))):
            self.assertEqual(attribution.cancellation_savings(Fraction(r), tuple(map(Fraction, interval))), expected)

    def test_exact_rationals_survive_without_float_or_fp32_rounding(self):
        result = attribution.cancellation_savings(Fraction(1, 10), (Fraction(-3, 20), Fraction(-1, 20)))
        self.assertEqual(result, (Fraction(1, 10), Fraction(1, 5)))
        self.assertTrue(all(isinstance(value, Fraction) for value in result))


if __name__ == "__main__":
    unittest.main()
