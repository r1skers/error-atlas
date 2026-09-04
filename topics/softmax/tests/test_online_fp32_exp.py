"""Tests for the correctly-rounded FP32 exp and the ULP profiling helpers.

Tests for the core skip while it still raises NotImplementedError; the scaffolding
tests run either way.
"""

import doctest
import math
import random
import struct
import unittest
from decimal import ROUND_UP, DefaultContext, FloatOperation, Inexact, localcontext
from fractions import Fraction
from unittest.mock import patch

from online import fp32_exp
from online.fp32_signed import MAX_FINITE, round_to_fp32

SEED = 20260905
SAMPLES = 3_000
DOMAIN = (-104.0, 0.0)


def _random_argument(rng: random.Random) -> Fraction:
    return round_to_fp32(Fraction(rng.uniform(*DOMAIN)))


def _next_fp32(value: Fraction) -> Fraction:
    """The next representable FP32 above ``value``; used to build ULP fixtures."""
    bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    return Fraction(struct.unpack("<f", struct.pack("<I", bits + 1))[0])


def _double_rounded_exp(argument: Fraction) -> Fraction:
    """The provisional path: compute in binary64, then round to FP32."""
    return round_to_fp32(Fraction(math.exp(float(argument))))


# --- an exp route that shares no code with the implementation under test ---------------
#
# Comparing two precisions of the same implementation cannot detect a shared bias. These
# helpers compute exp a second way, with no `decimal` involved at all: exact Fraction
# Taylor after argument reduction, then repeated squaring, carrying a rigorous bracket
# throughout. Agreement is then a differential result rather than a restatement.

_HALVINGS = 7          # keeps the reduced arguments small for all fixtures, including -200
_TERMS = 80
_GRID = 1 << 600       # widen the bracket onto this grid so the Fractions stay bounded


def _widen(low: Fraction, high: Fraction) -> tuple[Fraction, Fraction]:
    lower = low.numerator * _GRID // low.denominator
    upper = -((-high.numerator * _GRID) // high.denominator)
    return Fraction(lower, _GRID), Fraction(upper, _GRID)


def _independent_exp_interval(argument: Fraction) -> tuple[Fraction, Fraction]:
    """A rigorous bracket for exp(argument), for argument <= 0."""
    reduced = argument / (1 << _HALVINGS)
    total, term = Fraction(0), Fraction(1)
    for n in range(_TERMS + 1):
        total += term
        term = term * reduced / (n + 1)
    # For y <= 0, Taylor's remainder has e^c <= 1 (y <= c <= 0).
    # Thus |remainder| <= |next term|; the factor of three is conservative even at -200.
    bound = abs(term) * 3
    low, high = total - bound, total + bound
    for _ in range(_HALVINGS):
        low, high = _widen(low * low, high * high)
    return low, high


def _golden(argument: Fraction) -> Fraction:
    low, high = _independent_exp_interval(argument)
    lower, upper = round_to_fp32(low), round_to_fp32(high)
    if lower != upper:
        raise AssertionError(f"the independent route could not certify exp({float(argument)})")
    return lower


# Six arguments whose true exp lands unusually close to an FP32 rounding midpoint, found
# by searching the domain with the independent route; these are where a careless
# implementation goes wrong first.
_NEAR_MIDPOINT = (
    "-0x1.264b2c0000000p+4",
    "-0x1.52ab7c0000000p+6",
    "-0x1.a338860000000p+5",
    "-0x1.1805800000000p+6",
    "-0x1.5369720000000p+6",
    "-0x1.5a12840000000p+6",
)

# The two adjacent FP32 arguments straddling the underflow transition: exp crosses half
# the smallest subnormal at -150*ln2 = -103.972, so this is where the result stops being
# representable as anything but zero.
_LAST_NONZERO = "-0x1.9fe3680000000p+6"
_FIRST_ZERO = "-0x1.9fe36a0000000p+6"

GOLDEN_ARGUMENTS = tuple(
    Fraction(float.fromhex(h)) for h in _NEAR_MIDPOINT + (_LAST_NONZERO, _FIRST_ZERO)
) + tuple(
    Fraction(v) for v in (0, -1, -2, -10, -30, -87, -100, -103, -104, -105, -200)
) + (Fraction(-1, 2), Fraction(-1, 4), Fraction(-1, 1024))


def _skip_unless_implemented(test: unittest.TestCase, call) -> None:
    try:
        call()
    except NotImplementedError:
        test.skipTest("correctly_rounded_exp not implemented yet")


class DecimalScaffoldingTests(unittest.TestCase):
    """The Decimal plumbing is implemented and checked even while the core is blank."""

    def test_ambient_context_does_not_change_result_or_flags(self) -> None:
        with localcontext() as context:
            context.prec = 1
            context.Emin = 0
            context.Emax = 1
            context.rounding = ROUND_UP
            context.traps[FloatOperation] = True
            context.traps[Inexact] = True
            context.clear_flags()
            before = repr(context)
            self.assertEqual(fp32_exp._decimal_exp(Fraction(-10), 3), Fraction(227, 5_000_000))
            self.assertEqual(repr(context), before)

    def test_default_context_traps_are_not_inherited(self) -> None:
        saved_traps = DefaultContext.traps.copy()
        try:
            DefaultContext.traps[Inexact] = True
            self.assertEqual(fp32_exp._decimal_exp(Fraction(-10), 3), Fraction(227, 5_000_000))
        finally:
            DefaultContext.traps = saved_traps


class UlpDistanceTests(unittest.TestCase):
    """Scaffolding, so these do not skip."""

    def test_identical_values_are_zero_apart(self) -> None:
        for value in (Fraction(1), Fraction(1, 2), fp32_exp.MIN_USEFUL_ARGUMENT):
            self.assertEqual(fp32_exp.ulp_distance(value, value), 0)

    def test_adjacent_values_are_one_apart_and_signed(self) -> None:
        for value in (Fraction(1), Fraction(3, 4), round_to_fp32(Fraction(1, 10**30))):
            above = _next_fp32(value)
            self.assertEqual(fp32_exp.ulp_distance(above, value), 1)
            self.assertEqual(fp32_exp.ulp_distance(value, above), -1)

    def test_direction_is_by_value_on_the_negative_side_too(self) -> None:
        """Raw bit patterns run backwards for negatives, so this is where a naive
        implementation reports the wrong sign: -0.5 is above -1, not below it."""
        half, one = Fraction(-1, 2), Fraction(-1)
        self.assertEqual(fp32_exp.ulp_distance(half, one), 2**23)
        self.assertEqual(fp32_exp.ulp_distance(one, half), -(2**23))
        self.assertEqual(
            fp32_exp.ulp_distance(Fraction(-1), Fraction(-2)),
            fp32_exp.ulp_distance(Fraction(2), Fraction(1)),
        )

    def test_arguments_may_straddle_zero(self) -> None:
        self.assertGreater(fp32_exp.ulp_distance(Fraction(1), Fraction(-1)), 0)
        self.assertEqual(
            fp32_exp.ulp_distance(Fraction(1), Fraction(-1)),
            -fp32_exp.ulp_distance(Fraction(-1), Fraction(1)),
        )

    def test_unrepresentable_values_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            fp32_exp.ulp_distance(Fraction(1, 3), Fraction(1))
        with self.assertRaises(ValueError):
            fp32_exp.ulp_distance(Fraction(1), Fraction(-1, 3))


class CorrectlyRoundedExpTests(unittest.TestCase):
    def setUp(self) -> None:
        _skip_unless_implemented(self, lambda: fp32_exp.correctly_rounded_exp(Fraction(0)))
        self.rng = random.Random(SEED)

    def test_exp_of_zero_is_exactly_one(self) -> None:
        """The merge relies on this: the winning branch must cost nothing at all."""
        self.assertEqual(fp32_exp.correctly_rounded_exp(Fraction(0)), 1)

    def test_results_are_stored_fp32_and_nonnegative(self) -> None:
        for _ in range(SAMPLES):
            value = fp32_exp.correctly_rounded_exp(_random_argument(self.rng))
            self.assertGreaterEqual(value, 0)
            self.assertEqual(round_to_fp32(value), value)

    def test_monotonic_in_the_argument(self) -> None:
        arguments = sorted(_random_argument(self.rng) for _ in range(SAMPLES))
        values = [fp32_exp.correctly_rounded_exp(a) for a in arguments]
        for lower, upper in zip(values, values[1:]):
            self.assertLessEqual(lower, upper)

    def test_underflow_boundary(self) -> None:
        self.assertGreater(fp32_exp.correctly_rounded_exp(Fraction(-103)), 0)
        self.assertEqual(fp32_exp.correctly_rounded_exp(Fraction(-105)), 0)
        self.assertEqual(fp32_exp.correctly_rounded_exp(Fraction(-200)), 0)

    def test_matches_an_independently_computed_reference(self) -> None:
        """At the default budget every fixture must agree with the independent route."""
        for argument in GOLDEN_ARGUMENTS:
            with self.subTest(argument=float(argument)):
                self.assertEqual(fp32_exp.correctly_rounded_exp(argument), _golden(argument))

    def test_the_zero_shortcut_stays_below_the_real_transition(self) -> None:
        """MIN_USEFUL_ARGUMENT lets the implementation skip the decimal path entirely,
        which is not optional: without it _decimal_exp builds a seven-million-bit
        denominator at x = -5e6 and takes half a second, and finite FP32 gaps can have
        much larger magnitudes. The shortcut is only sound below the argument where
        exp actually stops rounding to zero.
        """
        last_nonzero = Fraction(float.fromhex(_LAST_NONZERO))
        self.assertGreater(fp32_exp.correctly_rounded_exp(last_nonzero), 0)
        self.assertLess(fp32_exp.MIN_USEFUL_ARGUMENT, last_nonzero)
        self.assertEqual(fp32_exp.correctly_rounded_exp(fp32_exp.MIN_USEFUL_ARGUMENT), 0)

    def test_extreme_negative_inputs_skip_high_precision_computation(self) -> None:
        """Guard the shortcut itself without a machine-dependent timing threshold."""
        with patch.object(
            fp32_exp, "_decimal_exp", side_effect=AssertionError("unexpected high-precision call")
        ) as decimal_exp:
            for argument in (
                fp32_exp.MIN_USEFUL_ARGUMENT, Fraction(-105), Fraction(-(2**30)), -MAX_FINITE
            ):
                with self.subTest(argument=argument):
                    self.assertEqual(fp32_exp.correctly_rounded_exp(argument), 0)
            decimal_exp.assert_not_called()

    def test_invalid_precision_is_rejected_before_any_shortcut(self) -> None:
        for argument in (Fraction(-105), Fraction(-1), Fraction(0)):
            for precision in (0, -1, True, False, 3.5, "60", None, Fraction(3)):
                with self.subTest(argument=argument, precision=precision):
                    with self.assertRaises(ValueError):
                        fp32_exp.correctly_rounded_exp(argument, precision=precision)

    def test_the_precision_budget_is_actually_honoured(self) -> None:
        """The prescribed three-digit intervals cannot decide these nonzero cases.

        Returning exp(0) = 1 analytically is allowed, so zero is deliberately excluded.
        Refusing these cases also catches implementations that silently use 60 digits.
        """
        for argument in (Fraction(-1), Fraction(-30), Fraction(-87)):
            with self.subTest(argument=float(argument)):
                with self.assertRaises(ValueError):
                    fp32_exp.correctly_rounded_exp(argument, precision=3)

    def test_non_stored_arguments_are_rejected(self) -> None:
        for argument in (Fraction(1, 3), Fraction(-313, 3)):
            with self.subTest(argument=argument):
                with self.assertRaises(ValueError):
                    fp32_exp.correctly_rounded_exp(argument)

    def test_limited_precision_never_returns_a_wrong_value(self) -> None:
        """Every returned value must match the independent route, at each fixed budget.

        Testing only 3 and 60 digits misses a plausible shortcut: reject fewer than 8
        digits, otherwise round Decimal.exp without a proof. The midpoint fixtures
        expose wrong answers at 8, 9, 10 and 11 digits. Refusal must be ValueError.
        """
        arguments = GOLDEN_ARGUMENTS + tuple(_random_argument(self.rng) for _ in range(200))
        for argument in arguments:
            reference = _golden(argument)
            for precision in (3, 8, 9, 10, 11):
                with self.subTest(argument=float(argument), precision=precision):
                    try:
                        value = fp32_exp.correctly_rounded_exp(argument, precision=precision)
                    except ValueError:  # refusing to answer is an allowed outcome
                        continue
                    self.assertEqual(value, reference)


class ExpProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        _skip_unless_implemented(self, lambda: fp32_exp.correctly_rounded_exp(Fraction(0)))
        self.rng = random.Random(SEED + 1)
        self.arguments = [_random_argument(self.rng) for _ in range(SAMPLES)]

    def test_profile_accounts_for_every_argument(self) -> None:
        histogram = fp32_exp.exp_ulp_profile(_double_rounded_exp, self.arguments)
        self.assertEqual(sum(histogram.values()), len(self.arguments))

    def test_the_provisional_double_rounded_path_stays_within_one_ulp(self) -> None:
        """merge.provisional_fp32_exp takes this path, so everything computed with it
        depends on this bound.

        Measured at 0 ULP for every sampled argument on the development platform. The
        assertion allows 1 ULP because the double-precision library exp is not itself
        correctly rounded everywhere, and that is a portability tolerance rather than a
        research claim.
        """
        histogram = fp32_exp.exp_ulp_profile(_double_rounded_exp, self.arguments)
        self.assertLessEqual(max(abs(d) for d in histogram), 1)


def load_tests(loader, tests, pattern):
    tests.addTests(doctest.DocTestSuite(fp32_exp))
    return tests


if __name__ == "__main__":
    unittest.main()
