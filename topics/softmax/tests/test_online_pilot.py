"""Tests for the exploratory pilot's inputs, reference and metric.

Core tests skip while the core still raises NotImplementedError; scaffolding tests run
either way. Nothing here produces or reads an artifact.
"""

import random
import unittest
from decimal import (
    Context, Decimal, DivisionByZero, InvalidOperation, Overflow,
    ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN,
)
from fractions import Fraction

from online import pilot, schedules
from online.fp32_exp import correctly_rounded_exp
from online.fp32_signed import fp32_sub, round_to_fp32
from online.merge import merge_reduce

SEED = 20260905


def _skip_unless_implemented(test: unittest.TestCase, call) -> None:
    try:
        call()
    except NotImplementedError:
        test.skipTest("pilot core not implemented yet")


def _flat_family(count: int = 4, mass: int = 32) -> pilot.BlockFamily:
    return pilot.BlockFamily(tuple(Fraction(0) for _ in range(count)), tuple([mass] * count))


def _decimal_real_exp_interval(argument: Fraction) -> tuple[Fraction, Fraction]:
    """Independent real-exp bracket for these test cases, without binary float conversion.

    Directed division brackets even nonterminating rational arguments. Monotonic exp
    maps that bracket; Decimal.exp is correctly rounded to nearest, so the adjacent
    decimal values enclose its rounding error. The precision resolves our sampled
    Taylor brackets; this helper is not a general-purpose arbitrary-argument oracle.
    """
    if argument == 0:
        return Fraction(1), Fraction(1)
    context = Context(
        prec=320, rounding=ROUND_HALF_EVEN, Emin=-999_999, Emax=999_999,
        capitals=1, clamp=0, flags=[], traps=[DivisionByZero, InvalidOperation, Overflow],
    )
    down, up = context.copy(), context.copy()
    down.rounding, up.rounding = ROUND_FLOOR, ROUND_CEILING
    numerator, denominator = Decimal(argument.numerator), Decimal(argument.denominator)
    argument_low = down.divide(numerator, denominator)
    argument_high = up.divide(numerator, denominator)
    low = context.next_minus(context.exp(argument_low))
    high = context.next_plus(context.exp(argument_high))
    return Fraction(low), Fraction(high)


class BlockFamilyTests(unittest.TestCase):
    """Scaffolding, so these do not skip."""

    def test_leaf_states_are_the_logit_and_the_mass(self) -> None:
        family = pilot.BlockFamily((Fraction(0), Fraction(-3)), (32, 8))
        self.assertEqual(family.leaf_max, (Fraction(0), Fraction(-3)))
        self.assertEqual(family.leaf_ell, (Fraction(32), Fraction(8)))
        self.assertEqual(family.global_max, Fraction(0))

    def test_malformed_families_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            pilot.BlockFamily((), ())
        with self.assertRaises(ValueError):
            pilot.BlockFamily((Fraction(0),), (1, 2))
        with self.assertRaises(ValueError):
            pilot.BlockFamily((Fraction(1, 3),), (1,))
        with self.assertRaises(ValueError):
            pilot.BlockFamily((Fraction(0),), (0,))
        with self.assertRaises(ValueError):
            pilot.BlockFamily((Fraction(0),), (2**24 + 1,))

    def test_masses_must_be_integer_element_counts(self) -> None:
        for mass in (True, False, 1.5, 2.0, Fraction(3, 2), Fraction(2)):
            with self.subTest(mass=mass):
                with self.assertRaises(TypeError):
                    pilot.BlockFamily((Fraction(0),), (mass,))
        self.assertEqual(pilot.BlockFamily((Fraction(0),), (2**24,)).leaf_ell,
                         (Fraction(2**24),))

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
        """FP32 agreement only; real containment is checked separately below."""
        rng = random.Random(SEED + 1)
        for _ in range(150):
            argument = round_to_fp32(Fraction(rng.uniform(-100.0, 0.0)))
            low, high = pilot.real_exp_interval(argument)
            expected = correctly_rounded_exp(argument)
            with self.subTest(argument=float(argument)):
                self.assertEqual(round_to_fp32(low), expected)
                self.assertEqual(round_to_fp32(high), expected)

    def test_contains_an_independent_high_precision_real_exp_bracket(self) -> None:
        rng = random.Random(SEED + 4)
        exact_gap = round_to_fp32(Fraction(-10)) - round_to_fp32(Fraction(1, 3))
        arguments = [Fraction(0), Fraction(-1), Fraction(-1, 3), exact_gap,
                     Fraction(-95), Fraction(-200), Fraction(-1000)]
        arguments.extend(Fraction(rng.uniform(-90.0, -0.125)).limit_denominator(10**6)
                         for _ in range(20))
        for argument in arguments:
            with self.subTest(argument=argument):
                low, high = pilot.real_exp_interval(argument)
                reference_low, reference_high = _decimal_real_exp_interval(argument)
                self.assertLessEqual(low, reference_low)
                self.assertGreaterEqual(high, reference_high)

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
        it, the exact gap needs bits FP32 cannot hold, and fp32_sub moves it by about
        3.3e-7 in absolute terms. That shifts exp(gap) by about 3.3e-7 relative, which is
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


class MeasurementIntervalTests(unittest.TestCase):
    """Scaffolding checks independent of the two unfinished core functions."""

    def measurement(self, computed, low, high, family=None):
        return pilot.Measurement(
            kind="test", fused=False, computed=Fraction(computed),
            error_low=Fraction(low), error_high=Fraction(high), absorbed_merges=0,
            exp_underflow_edges=0, product_underflow_edges=0,
            family=family if family is not None else _flat_family(2, 10),
        )

    def test_wide_or_touching_intervals_remain_unresolved(self) -> None:
        first = self.measurement(19, 0, 1)
        second = self.measurement(18, 0, 1)
        self.assertTrue(first.has_valid_error_interval)
        self.assertIsNone(first.error_order(second))
        first = self.measurement(19, 0, Fraction(1, 10))
        second = self.measurement(18, Fraction(1, 10), Fraction(1, 5))
        self.assertIsNone(first.error_order(second))

    def test_disjoint_intervals_determine_the_paired_sign(self) -> None:
        first = self.measurement(19, Fraction(4, 100), Fraction(6, 100))
        second = self.measurement(18, Fraction(9, 100), Fraction(11, 100))
        self.assertEqual(first.error_difference_interval(second),
                         (Fraction(-7, 100), Fraction(-3, 100)))
        self.assertEqual(first.error_order(second), -1)
        self.assertEqual(second.error_order(first), 1)

    def test_equal_outputs_share_exactly_the_same_error(self) -> None:
        first = self.measurement(19, 0, 1)
        second = self.measurement(19, Fraction(4, 100), Fraction(6, 100))
        self.assertEqual(first.error_difference_interval(second), (0, 0))
        self.assertEqual(first.error_order(second), 0)

    def test_invalid_intervals_and_unpaired_families_are_rejected(self) -> None:
        valid = self.measurement(19, 0, 1)
        for low, high in ((-1, 1), (2, 1)):
            invalid = self.measurement(18, low, high)
            self.assertFalse(invalid.has_valid_error_interval)
            with self.assertRaises(ValueError):
                valid.error_order(invalid)
        unpaired = self.measurement(19, 0, 1, _flat_family(2, 11))
        with self.assertRaisesRegex(ValueError, "same block family"):
            valid.error_order(unpaired)


class RoundingEventTests(unittest.TestCase):
    """Classify real merge dumps without requiring the unfinished metric core."""

    def test_exp_underflow_is_not_addition_absorption(self) -> None:
        for fused in (False, True):
            dump = merge_reduce((Fraction(0), Fraction(-200)), (Fraction(1), Fraction(1)),
                                schedules.sequential_chain(2), fused=fused)
            self.assertEqual(pilot._rounding_event_counts(dump), (0, 1, 0))

    def test_product_underflow_exists_only_in_the_separate_arm(self) -> None:
        # Synthetic merge state, outside constant-logit pilot blocks (whose ell >= 1).
        # A separate multiply loses the nonzero term; FMA loses it only at final rounding.
        maxima, ells = (Fraction(0), Fraction(-1)), (Fraction(1), Fraction(1, 2**149))
        for fused, expected in ((False, (0, 0, 1)), (True, (1, 0, 0))):
            dump = merge_reduce(maxima, ells, schedules.sequential_chain(2), fused=fused)
            self.assertEqual(pilot._rounding_event_counts(dump), expected)

    def test_fma_changes_absorption_for_integer_mass_blocks(self) -> None:
        for reverse in (False, True):
            maxima = (Fraction(0), Fraction(-16323477, 8388608))
            masses = (2**24, 7)
            if reverse:
                maxima, masses = maxima[::-1], masses[::-1]
            family = pilot.BlockFamily(maxima, masses)
            for fused, expected_count, expected_output in (
                (False, 1, 2**24), (True, 0, 2**24 + 2),
            ):
                with self.subTest(reverse=reverse, fused=fused):
                    dump = merge_reduce(family.leaf_max, family.leaf_ell,
                                        schedules.sequential_chain(2), fused=fused)
                    self.assertEqual(dump.node_ell[0], expected_output)
                    self.assertEqual(pilot._rounding_event_counts(dump),
                                     (expected_count, 0, 0))


class MeasureTests(unittest.TestCase):
    def setUp(self) -> None:
        _skip_unless_implemented(self, lambda: pilot.denominator_interval(_flat_family()))

    def test_a_measurement_reports_a_usable_bracket(self) -> None:
        family = pilot.max_at_position(8, 32, Fraction(20), 7)
        for schedule in (schedules.sequential_chain(8), schedules.balanced_pairwise(8)):
            for fused in (False, True):
                result = pilot.measure(family, schedule, fused=fused)
                with self.subTest(kind=schedule.kind, fused=fused):
                    self.assertTrue(result.has_valid_error_interval)
                    self.assertEqual(result.family, family)
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
