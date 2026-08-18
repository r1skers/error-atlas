"""Irregular stored-FP32 input generators for predictor calibration.

These generators are calibration-only scaffolding.  They sample positive *stored binary32*
values directly as exact ``Fraction`` objects so the resulting leaves are already valid
FP32 values.  This avoids source-value/materialization ambiguity while we diagnose whether
candidate reduction trees produce different oracle targets.

This module is not the frozen held-out input generator.  Its parameters may be replaced by
the protocol's eventual source-generation/materialization rules before held-out data exist.
It must not be tuned using predictor correlation or held-out behavior.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from fractions import Fraction


CALIBRATION_INPUT_GENERATOR_VERSION = "irregular_stored_fp32_v1"


@dataclass(frozen=True)
class CalibrationInput:
    family: str
    seed: int
    values: tuple[Fraction, ...]


def _validate(width: int, seed: int) -> None:
    if isinstance(width, bool) or not isinstance(width, int):
        raise TypeError("width must be an integer")
    if width < 2:
        raise ValueError("width must be at least 2")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")


def _normal_fp32_fraction(rng: random.Random, exponent: int) -> Fraction:
    """Sample one positive normal binary32 value with a random 23-bit fraction field."""
    if not -126 <= exponent <= 127:
        raise ValueError("normal FP32 exponent is out of range")
    significand = (1 << 23) | rng.randrange(1 << 23)
    shift = exponent - 23
    if shift >= 0:
        return Fraction(significand << shift)
    return Fraction(significand, 1 << (-shift))


def head_tail_random(width: int, *, seed: int) -> CalibrationInput:
    """One dominant normal leaf plus irregular smaller tails in random positions."""
    _validate(width, seed)
    rng = random.Random(seed)

    # Keep the head around order one while distributing tails across many FP32 scales.
    values = [_normal_fp32_fraction(rng, 0)]
    for _ in range(width - 1):
        tail_exponent = rng.randint(-28, -12)
        values.append(_normal_fp32_fraction(rng, tail_exponent))
    rng.shuffle(values)
    return CalibrationInput("head_tail_random", seed, tuple(values))


def same_scale_random(width: int, *, seed: int) -> CalibrationInput:
    """Irregular mantissas with exponents confined to a narrow three-binade band."""
    _validate(width, seed)
    rng = random.Random(seed)
    base_exponent = rng.randint(-4, 2)
    values = [
        _normal_fp32_fraction(rng, base_exponent + rng.randint(-1, 1))
        for _ in range(width)
    ]
    rng.shuffle(values)
    return CalibrationInput("same_scale_random", seed, tuple(values))


def wide_range_random(width: int, *, seed: int) -> CalibrationInput:
    """Irregular mantissas with exponents sampled over a broad dynamic range."""
    _validate(width, seed)
    rng = random.Random(seed)
    values = [
        _normal_fp32_fraction(rng, rng.randint(-32, 4))
        for _ in range(width)
    ]
    rng.shuffle(values)
    return CalibrationInput("wide_range_random", seed, tuple(values))


def calibration_input_families(width: int, *, seed: int) -> tuple[CalibrationInput, ...]:
    """Generate one reproducible input from each calibration family."""
    return (
        head_tail_random(width, seed=seed),
        same_scale_random(width, seed=seed + 1_000_003),
        wide_range_random(width, seed=seed + 2_000_003),
    )
