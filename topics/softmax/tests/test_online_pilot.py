"""Tests for the exploratory pilot's inputs, reference and metric.

Core tests skip while the core still raises NotImplementedError; scaffolding tests run
either way. Nothing here produces or reads an artifact.
"""

import random
import unittest
from fractions import Fraction

from online import pilot, schedules
from online.fp32_exp import correctly_rounded_exp
from online.fp32_signed import fp32_sub, round_to_fp32

SEED = 20260905


def _skip_unless_implemented(test: unittest.TestCase, call) -> None:
    try:
        call()
    except NotImplementedError:
        test.skipTest("pilot core not implemented yet")


def _flat_family(count: int = 4, mass: int = 32) -> pilot.BlockFamily:
    return pilot.BlockFamily(tuple(Fraction(0) for _ in range(count)), tuple([mass] * count))


class BlockFamilyTests(unittest.TestCase):
    """Scaffolding, so these do not skip."""

    def test_leaf_states_are_the_logit_and_the_mass(self) -> None:
        family = pilot.BlockFamily((Fraction(0), Fraction(-3)), (32, 8))
        self.assertEqual(family.leaf_max, (Fraction(0), Fraction(-3)))
        self.assertEqual(family.leaf_ell, (Fraction(32), Fraction(8)))
        self.assertEqual(family.global_max, Fraction(0))

    def test_malformed_families_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            pilot.BlockFamily((Fraction(0),), (1, 2))
        with self.assertRaises(ValueError):
            pilot.BlockFamily((Fraction(1, 3),), (1,))
        with self.assertRaises(ValueError):
            pilot.BlockFamily((Fraction(0),), (0,))
        with self.assertRaises(ValueError):
            pilot.BlockFamily((Fraction(0),), (2**24 + 1,))

    def test_the_block_mass_really_does_accumulate_exactly(self) -> None:
        """The design of contract section 12 rests on this, so it is checked, not assumed."""
        for mass in (1, 2, 32, 1000, 4096):
            self.assertTrue(pilot.accumulates_exactly(mass))

    def test_generators_place_the_maximum_where_asked(self) -> None:
        for position in (0, 3, 7):
            family = pilot.max_at_position(8, 32, Fraction(20), position)
            self.assertEqual(family.maxima[position], family.global_max)
            self.assertEqual(family.global_max, Fraction(20))
            self.assertEqual(sum(1 for m in family.maxima if m == 0), 7)
        with self.assertRaises(ValueError):
            pilot.max_at_position(8, 32, Fraction(20), 8)
        with self.assertRaises(ValueError):
            pilot.max_at_position(8, 32, Fraction(-1), 0)

    def test_uniform_spread_stays_in_range_and_is_reproducible(self) -> None:
        first = pilot.uniform_spread(16, 32, 12.0, random.Random(SEED))
        second = pilot.uniform_spread(16, 32, 12.0, random.Random(SEED))
        self.assertEqual(first, second)
        self.assertTrue(all(-13 <= m <= 0 for m in first.maxima))


class RealExpIntervalTests(unittest.TestCase):
    """Scaffolding, so these do not skip."""

    def test_brackets_are_ordered_and_positive(self) -> None:
        rng = random.Random(SEED)
        for _ in range(200):
            argument = Fraction(rng.uniform(-90.0, 0.0)).limit_denominator(10**6)
            low, high = pilot.real_exp_interval(argument)
            self.assertLessEqual(low, high)
            self.assertGreater(high, 0)

    def test_agrees_with_the_decimal_route_on_fp32_arguments(self) -> None:
        """Two independent routes to the same reference: Taylor here, decimal there."""
        rng = random.Random(SEED + 1)
        for _ in range(150):
            argument = round_to_fp32(Fraction(rng.uniform(-100.0, 0.0)))
            low, high = pilot.real_exp_interval(argument)
            expected = correctly_rounded_exp(argument)
            with self.subTest(argument=float(argument)):
                self.assertEqual(round_to_fp32(low), expected)
                self.assertEqual(round_to_fp32(high), expected)

    def test_accepts_rationals_that_are_not_fp32_values(self) -> None:
        """The exponent is an exact difference of two logits, which need not be FP32."""
        difference = round_to_fp32(Fraction(-1, 3) * 100) - round_to_fp32(Fraction(7, 9))
        low, high = pilot.real_exp_interval(difference)
        self.assertLessEqual(low, high)
        self.assertGreater(low, 0)

    def test_the_bracket_is_far_narrower_than_one_fp32_step(self) -> None:
        for argument in (Fraction(0), Fraction(-1), Fraction(-40), Fraction(-95)):
            low, high = pilot.real_exp_interval(argument)
            self.assertLess((high - low) / high, Fraction(1, 10**30))

    def test_positive_arguments_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            pilot.real_exp_interval(Fraction(1, 100))


class DenominatorTests(unittest.TestCase):
    def setUp(self) -> None:
        _skip_unless_implemented(self, lambda: pilot.denominator_interval(_flat_family()))

    def test_equal_maxima_give_the_total_mass(self) -> None:
        """Every gap is zero, so the denominator is exactly the number of elements."""
        family = pilot.BlockFamily(tuple(Fraction(5) for _ in range(4)), (32, 8, 100, 1))
        low, high = pilot.denominator_interval(family)
        self.assertLessEqual(low, 141)
        self.assertGreaterEqual(high, 141)
        self.assertLess((high - low) / high, Fraction(1, 10**30))

    def test_matches_the_term_by_term_formula(self) -> None:
        """Pins the shape of the sum: one bracketed exp per block, scaled by its mass.

        This is a specification check rather than an independent route; the independent
        route is real_exp_interval itself, which the scaffolding tests cross-check against
        the decimal implementation of correctly_rounded_exp.
        """
        rng = random.Random(SEED + 2)
        for _ in range(20):
            family = pilot.uniform_spread(8, 32, 25.0, rng)
            low, high = pilot.denominator_interval(family)
            maximum = family.global_max
            expected_low = sum(
                mass * pilot.real_exp_interval(logit - maximum)[0]
                for logit, mass in zip(family.maxima, family.masses)
            )
            expected_high = sum(
                mass * pilot.real_exp_interval(logit - maximum)[1]
                for logit, mass in zip(family.maxima, family.masses)
            )
            self.assertLessEqual(low, expected_low)
            self.assertGreaterEqual(high, expected_high)

    def test_the_exponent_is_the_exact_difference_not_the_fp32_one(self) -> None:
        """A gap that FP32 subtraction would round must not be rounded here.

        The choice of numbers matters. With a maximum of one third and a block ten below
        it, the exact gap needs bits FP32 cannot hold, and fp32_sub moves it by 3.3e-7
        relative. That shifts the block's term by the same relative amount, which is
        1e127 times wider than the reference bracket, so the two routes are separable.
        A far more negative gap would not work: the term would be negligible beside the
        maximum block and either route would pass.
        """
        maximum, lower = round_to_fp32(Fraction(1, 3)), round_to_fp32(Fraction(-10))
        gap = lower - maximum
        self.assertNotEqual(gap, fp32_sub(lower, maximum)[0])

        family = pilot.BlockFamily((maximum, lower), (1, 1))
        low, high = pilot.denominator_interval(family)
        exact_low, exact_high = pilot.real_exp_interval(gap)
        self.assertLessEqual(low, 1 + exact_low)
        self.assertGreaterEqual(high, 1 + exact_high)

        rounded_low, _ = pilot.real_exp_interval(fp32_sub(lower, maximum)[0])
        self.assertGreater(1 + rounded_low, high)  # the rounded route lands outside


class RelativeErrorTests(unittest.TestCase):
    def setUp(self) -> None:
        _skip_unless_implemented(
            self, lambda: pilot.relative_error(Fraction(1), (Fraction(1), Fraction(1)))
        )

    def test_a_point_reference_gives_the_plain_relative_error(self) -> None:
        for computed, reference in ((Fraction(3), Fraction(4)), (Fraction(5), Fraction(4))):
            low, high = pilot.relative_error(computed, (reference, reference))
            expected = abs(computed - reference) / reference
            self.assertEqual(low, expected)
            self.assertEqual(high, expected)

    def test_the_bracket_covers_every_reference_in_the_interval(self) -> None:
        """The conservative property: no admissible reference may fall outside."""
        rng = random.Random(SEED + 3)
        for _ in range(300):
            reference = Fraction(rng.uniform(0.5, 100.0)).limit_denominator(10**6)
            width = reference * Fraction(rng.randrange(1, 1000), 10**4)
            interval = (reference - width, reference + width)
            computed = Fraction(rng.uniform(0.4, 110.0)).limit_denominator(10**6)
            low, high = pilot.relative_error(computed, interval)
            for step in range(11):
                candidate = interval[0] + (interval[1] - interval[0]) * Fraction(step, 10)
                actual = abs(computed - candidate) / candidate
                with self.subTest(step=step):
                    self.assertLessEqual(low, actual)
                    self.assertGreaterEqual(high, actual)

    def test_zero_error_is_reachable_when_the_reference_straddles_the_value(self) -> None:
        low, _ = pilot.relative_error(Fraction(10), (Fraction(9), Fraction(11)))
        self.assertEqual(low, 0)


class MeasureTests(unittest.TestCase):
    def setUp(self) -> None:
        _skip_unless_implemented(self, lambda: pilot.denominator_interval(_flat_family()))

    def test_a_measurement_reports_a_usable_bracket(self) -> None:
        family = pilot.max_at_position(8, 32, Fraction(20), 7)
        for schedule in (schedules.sequential_chain(8), schedules.balanced_pairwise(8)):
            for fused in (False, True):
                result = pilot.measure(family, schedule, fused=fused)
                with self.subTest(kind=schedule.kind, fused=fused):
                    self.assertTrue(result.separated)
                    self.assertGreaterEqual(result.error_low, 0)
                    self.assertLessEqual(result.error_high, 1)
                    self.assertEqual(result.kind, schedule.kind)
                    self.assertGreaterEqual(result.absorbed_merges, 0)
                    self.assertLessEqual(result.absorbed_merges, len(schedule.nodes))

    def test_equal_maxima_leave_nothing_to_rescale(self) -> None:
        """All gaps zero, all weights one: the merge is a plain exact integer sum."""
        family = _flat_family(8, 32)
        result = pilot.measure(family, schedules.balanced_pairwise(8))
        self.assertEqual(result.computed, 256)
        self.assertEqual(result.error_high, 0)
        self.assertEqual(result.absorbed_merges, 0)


if __name__ == "__main__":
    unittest.main()
