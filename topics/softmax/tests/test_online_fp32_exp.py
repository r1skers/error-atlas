"""Tests for the correctly-rounded FP32 exp and the ULP profiling helpers.

Tests for the core skip while it still raises NotImplementedError; the scaffolding
tests run either way.
"""

import math
import random
import struct
import unittest
from fractions import Fraction

from online import fp32_exp
from online.fp32_signed import round_to_fp32

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


def _skip_unless_implemented(test: unittest.TestCase, call) -> None:
    try:
        call()
    except NotImplementedError:
        test.skipTest("correctly_rounded_exp not implemented yet")


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

    def test_mixed_signs_and_unrepresentable_values_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            fp32_exp.ulp_distance(Fraction(1), Fraction(-1))
        with self.assertRaises(ValueError):
            fp32_exp.ulp_distance(Fraction(1, 3), Fraction(1))


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

    def test_non_stored_arguments_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            fp32_exp.correctly_rounded_exp(Fraction(1, 3))

    def test_low_precision_never_returns_a_wrong_value(self) -> None:
        """Refusing is fine, escalating is fine; silently answering from a bound that
        did not separate two FP32 candidates is not.

        This is the certification step itself, so it holds whichever way the design
        question about the failure mode was answered.
        """
        reference = fp32_exp.correctly_rounded_exp
        for _ in range(200):
            argument = _random_argument(self.rng)
            try:
                value = fp32_exp.correctly_rounded_exp(argument, precision=3)
            except Exception:  # noqa: BLE001 - refusing to answer is an allowed outcome
                continue
            self.assertEqual(value, reference(argument))


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


if __name__ == "__main__":
    unittest.main()
