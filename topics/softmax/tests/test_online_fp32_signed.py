"""Differential tests for the signed FP32 contract of the online normalizer stage.

Tests skip while the core still raises NotImplementedError, so the suite stays green
before the user has written it. Once implemented, agreement must be exact.

References: rewrite.fp32_oracle on the nonnegative subset, and hardware float32.
"""

import random
import struct
import unittest
from fractions import Fraction

import numpy as np

from online import fp32_signed as signed
from rewrite import fp32_oracle as nonneg

SEED = 20260904
PAIR_CASES = 4_000
Q = signed.SUBNORMAL_QUANTUM


def _random_fp32(rng: random.Random, *, max_exponent_field: int = 240) -> Fraction:
    """A random stored FP32 value, signed, biased toward the interesting regions."""
    mode = rng.random()
    if mode < 0.15:
        bits = rng.randrange(1, 1 << 23)  # subnormal
    else:
        low = max(1, max_exponent_field - 13) if mode < 0.30 else 1
        bits = (rng.randrange(low, max_exponent_field + 1) << 23) | rng.randrange(1 << 23)
    if rng.random() < 0.5:
        bits |= 1 << 31
    return Fraction(struct.unpack("<f", struct.pack("<I", bits))[0])


def _skip_unless_implemented(test: unittest.TestCase, call) -> None:
    try:
        call()
    except NotImplementedError:
        test.skipTest("online signed core not implemented yet")


class RoundingTests(unittest.TestCase):
    def setUp(self) -> None:
        _skip_unless_implemented(self, lambda: signed.round_to_fp32(Fraction(1)))
        self.rng = random.Random(SEED)

    def test_matches_nonnegative_oracle_on_its_domain(self) -> None:
        """The signed contract must not move any value the old contract already fixed."""
        for _ in range(PAIR_CASES):
            perturbed = abs(_random_fp32(self.rng)) + Q * self.rng.randrange(-3, 4)
            if perturbed < 0:
                continue
            self.assertEqual(signed.round_to_fp32(perturbed), nonneg.round_to_fp32(perturbed))

    def test_sign_symmetry(self) -> None:
        """RN-even is sign-symmetric; a violation here means the tie rule saw the sign."""
        for _ in range(PAIR_CASES):
            v = _random_fp32(self.rng) + Q * self.rng.randrange(-3, 4)
            self.assertEqual(signed.round_to_fp32(-v), -signed.round_to_fp32(v))

    def test_ties_pick_the_even_significand_in_both_directions(self) -> None:
        """Exact midpoints must round to even, not away from zero, for either sign.

        Both offsets are needed: at an odd significand the even neighbour is upward,
        where ties-to-even and ties-away agree, so only the even-significand case
        separates them.
        """
        for exponent in (-140, -126, -10, 0, 5, 60):
            quantum = Fraction(2) ** max(exponent - 23, -149)
            base = Fraction(2) ** exponent
            for offset in (1, 2, 3, 4, 5, 6, 7, 8):
                midpoint = base + quantum * offset + quantum / 2
                pos = signed.round_to_fp32(midpoint)
                neg = signed.round_to_fp32(-midpoint)
                self.assertEqual(neg, -pos)
                significand = pos / quantum
                self.assertEqual(significand.denominator, 1)
                self.assertEqual(int(significand) % 2, 0, f"tie at {exponent}/{offset}")
                # even offset -> the even neighbour is downward; ties-away would go up
                expected = base + quantum * (offset + 1 if offset % 2 else offset)
                self.assertEqual(pos, expected)

    def test_signed_subnormals_and_zero(self) -> None:
        self.assertEqual(signed.round_to_fp32(Fraction(0)), 0)
        self.assertEqual(signed.round_to_fp32(-Q / 4), 0)
        self.assertEqual(signed.round_to_fp32(-Q * 3 / 2), -Q * 2)  # tie -> even
        self.assertEqual(signed.round_to_fp32(-Q * 5 / 2), -Q * 2)  # tie -> even

    def test_overflow_both_directions(self) -> None:
        big = signed.MAX_FINITE * 2
        with self.assertRaises(OverflowError):
            signed.round_to_fp32(big)
        with self.assertRaises(OverflowError):
            signed.round_to_fp32(-big)

    def test_overflow_threshold_is_the_midpoint_not_max_finite(self) -> None:
        """IEEE overflows above the midpoint, not above MAX_FINITE.

        The first grid point past MAX_FINITE is 2**128, so values in
        (MAX_FINITE, midpoint) round back down to MAX_FINITE and are finite in
        hardware. Deciding overflow on the input rather than on the rounded result
        turns that whole band into a spurious OverflowError.
        """
        maximum = signed.MAX_FINITE
        midpoint = (maximum + Fraction(2) ** 128) / 2
        gap = Fraction(2) ** 100
        for value in (maximum + gap, midpoint - gap):
            self.assertEqual(signed.round_to_fp32(value), maximum)
            self.assertEqual(signed.round_to_fp32(-value), -maximum)
            with np.errstate(over="ignore"):
                self.assertEqual(np.float32(float(value)), np.float32(float(maximum)))
        for value in (midpoint, midpoint + gap):  # ties-to-even selects 2**128
            with self.assertRaises(OverflowError):
                signed.round_to_fp32(value)
            with self.assertRaises(OverflowError):
                signed.round_to_fp32(-value)

    def test_is_stored_accepts_negatives(self) -> None:
        for _ in range(500):
            v = _random_fp32(self.rng)
            self.assertTrue(signed.is_stored_fp32(v))
            self.assertTrue(signed.is_stored_fp32(-v))
        self.assertFalse(signed.is_stored_fp32(Fraction(1, 3)))
        self.assertFalse(signed.is_stored_fp32(Fraction(-1, 3)))


class OperatorTests(unittest.TestCase):
    def setUp(self) -> None:
        _skip_unless_implemented(self, lambda: signed.fp32_add(Fraction(1), Fraction(1)))
        self.rng = random.Random(SEED + 1)

    def _pair(self):
        return _random_fp32(self.rng), _random_fp32(self.rng)

    def test_residual_convention(self) -> None:
        """residual == result - exact, for every operator."""
        for _ in range(PAIR_CASES):
            a, b = self._pair()
            for op, exact in (
                (signed.fp32_add, a + b),
                (signed.fp32_sub, a - b),
                (signed.fp32_mul, a * b),
            ):
                try:
                    result, residual = op(a, b)
                except OverflowError:
                    continue
                self.assertEqual(residual, result - exact)

    def test_add_sub_mul_match_hardware_float32(self) -> None:
        with np.errstate(over="ignore", under="ignore"):
            for _ in range(PAIR_CASES):
                a, b = self._pair()
                fa, fb = np.float32(float(a)), np.float32(float(b))
                for op, hw in (
                    (signed.fp32_add, fa + fb),
                    (signed.fp32_sub, fa - fb),
                    (signed.fp32_mul, fa * fb),
                ):
                    if not np.isfinite(hw):
                        continue
                    try:
                        result, _ = op(a, b)
                    except OverflowError:
                        self.fail(f"oracle overflowed where hardware did not: {a} {b}")
                    self.assertEqual(result, Fraction(float(hw)))

    def test_sub_equals_add_of_negation(self) -> None:
        for _ in range(PAIR_CASES):
            a, b = self._pair()
            try:
                self.assertEqual(signed.fp32_sub(a, b), signed.fp32_add(a, -b))
            except OverflowError:
                continue

    def test_sterbenz_subtraction_is_exact(self) -> None:
        """Same sign and within a factor of two: the subtraction must have zero residual."""
        for _ in range(PAIR_CASES):
            a = abs(_random_fp32(self.rng, max_exponent_field=200))
            if a == 0:
                continue
            b = signed.round_to_fp32(a * Fraction(self.rng.randrange(50, 200), 100))
            if b == 0 or not (b / 2 <= a <= 2 * b):
                continue
            self.assertEqual(signed.fp32_sub(a, b)[1], 0)
            self.assertEqual(signed.fp32_sub(-a, -b)[1], 0)

    def test_equal_operands_subtract_to_exact_zero(self) -> None:
        """The winning branch of every merge relies on m_a (-) m_v being exactly 0."""
        for _ in range(500):
            a = _random_fp32(self.rng)
            self.assertEqual(signed.fp32_sub(a, a), (Fraction(0), Fraction(0)))


class FusedMultiplyAddTests(unittest.TestCase):
    def setUp(self) -> None:
        _skip_unless_implemented(
            self, lambda: signed.fp32_fma(Fraction(1), Fraction(1), Fraction(1))
        )
        self.rng = random.Random(SEED + 2)

    def test_fma_rounds_once(self) -> None:
        """fma(a,b,c) must round a*b+c once, not round the product first."""
        for _ in range(PAIR_CASES):
            a, b, c = (_random_fp32(self.rng) for _ in range(3))
            try:
                fused, residual = signed.fp32_fma(a, b, c)
            except OverflowError:
                continue
            self.assertEqual(residual, fused - (a * b + c))
            self.assertEqual(fused, signed.round_to_fp32(a * b + c))

    def test_fma_differs_from_separate_rounding_somewhere(self) -> None:
        """If this never fires, the FMA arm of the contract would be vacuous."""
        found = 0
        for _ in range(PAIR_CASES):
            a, b, c = (_random_fp32(self.rng) for _ in range(3))
            try:
                fused, _ = signed.fp32_fma(a, b, c)
                product, _ = signed.fp32_mul(a, b)
                separate, _ = signed.fp32_add(product, c)
            except OverflowError:
                continue
            if fused != separate:
                found += 1
        self.assertGreater(found, 0, "no (a,b,c) separated fused from separate rounding")


if __name__ == "__main__":
    unittest.main()
