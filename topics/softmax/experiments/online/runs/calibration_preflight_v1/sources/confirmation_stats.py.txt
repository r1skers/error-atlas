"""USER-WRITTEN statistical core under construction; confirmation design is not frozen.

This module produces paired bootstrap MEAN DISTRIBUTIONS, not a selected confidence
interval or a pass/fail decision. Those choices follow method validation and design freeze.

Explain-back（用户填写）
目标：
为什么抽样单位是整个输入族，而不是一次 schedule 执行：
两种 FMA arm 为什么必须使用同一组抽样索引：
数值外包区间与统计置信区间有什么区别：
增加 bootstrap 次数和增加独立输入族数，分别改变什么：
伪代码：

Prediction record（核心测试/区间校准前填写）
Direction：
Scale：
Boundary：
Failure signature：
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

Interval = tuple[Fraction, Fraction]
ArmMeans = tuple[Interval, Interval]  # separate, fused, in this fixed order
GRID_BITS = 128
GRID = 1 << GRID_BITS


@dataclass(frozen=True)
class FamilySample:
    """One input family within ONE fixed (n, spread, mass, exp) stratum.

    Both arms hold intervals for D=A_chain-A_balanced from that same family.
    Caller groups strata before calling this module; schedule differences are already paired.
    """

    family_id: str
    separate: Interval
    fused: Interval


def _grid_rows(samples: Sequence[FamilySample]):
    """Scaffolding: validate source rows and enclose each endpoint on a common dyadic grid."""
    if not samples or len({row.family_id for row in samples}) != len(samples):
        raise ValueError("Need nonempty source samples with unique family IDs")
    rows = []
    for sample in samples:
        if not isinstance(sample.family_id, str) or not sample.family_id:
            raise ValueError("Each source family needs a nonempty string ID")
        arms = []
        for interval in (sample.separate, sample.fused):
            if len(interval) != 2 or any(not isinstance(value, Fraction) for value in interval):
                raise TypeError("Endpoints must be exact Fractions")
            low, high = interval
            if low > high:
                raise ValueError("Endpoints must be ordered")
            down = low.numerator * GRID // low.denominator
            up = -((-high.numerator * GRID) // high.denominator)
            arms.append((down, up))
        rows.append(tuple(arms))
    return tuple(rows)


def _validate_indices(indices: Sequence[int], count: int):
    if not indices or any(type(index) is not int or not 0 <= index < count for index in indices):
        raise ValueError("Need nonempty valid integer family indices; repeats are allowed")


def resample_indices(count: int, repeats: int, seed: int) -> tuple[tuple[int, ...], ...]:
    """Scaffolding: one deterministic draw matrix, shared by both arms of each family.

    One replicate draws exactly count indices with replacement, in list order, using
    random.Random(seed).randrange(count). This helper does not draw research inputs.
    """
    if type(count) is not int or count < 1 or type(repeats) is not int or repeats < 1:
        raise ValueError("count and repeats must be positive integers")
    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    rng = random.Random(seed)
    return tuple(tuple(rng.randrange(count) for _ in range(count)) for _ in range(repeats))


def paired_mean(samples: Sequence[FamilySample], indices: Sequence[int]) -> ArmMeans:
    """USER-WRITTEN CORE: two mean D intervals for the SAME selected family indices.

    grid_rows[i][arm] contains the lower/upper integer bounds, in units of 1/GRID.
    Sum selected lower endpoints and selected upper endpoints separately, then divide
    by the selected sample size and grid scale. Use exact Fraction outputs.
    D can be negative. Repeated indices count repeatedly; never deduplicate a bootstrap draw.
    The two arms share indices, and cannot be pooled into twice as many observations.
    """
    grid_rows = _grid_rows(samples)
    _validate_indices(indices, len(samples))
    return _mean_from_grid(grid_rows, indices)


def _mean_from_grid(grid_rows, indices: Sequence[int]) -> ArmMeans:
    """User's mean calculation on already validated, quantized rows and indices."""
    k = len(indices)
    total = 0

    for index in indices:
        total += grid_rows[index][0][0]
    mean_low = Fraction(total, k * GRID)
    # Calculate mean_high similarly
    total = 0
    for index in indices:
        total += grid_rows[index][0][1]
    mean_high = Fraction(total, k * GRID)

    total = 0
    for index in indices:
        total += grid_rows[index][1][0]
    mean_low_fused = Fraction(total, k * GRID)

    total = 0
    for index in indices:
        total += grid_rows[index][1][1]
    mean_high_fused = Fraction(total, k * GRID)

    return (
        (mean_low, mean_high),
        (mean_low_fused, mean_high_fused),
    )


def bootstrap_means(samples: Sequence[FamilySample], draws: Sequence[Sequence[int]]) -> tuple[ArmMeans, ...]:
    """USER-WRITTEN CORE: evaluate every fixed draw without changing the pairing.

    Every draw has len(samples) family indices, sampled with replacement. Return one
    ArmMeans per draw in exactly that order. Use the same draw for separate and fused.
    Quantize the source once, not separately for every replicate. The supplied draws
    make tests independent of the PRNG and let later audits compare identical resamples.

    This returns a distribution of interval-enclosed bootstrap means. It does not return
    a statistical CI and must not decide significance or select an interval method.
    """
    grid_rows = _grid_rows(samples)
    if not draws:
        raise ValueError("Need bootstrap draws")
    results = []
    for indices in draws:
        _validate_indices(indices, len(samples))
        if len(indices) != len(samples):
            raise ValueError("Each bootstrap draw must have the original family count")
        results.append(_mean_from_grid(grid_rows, indices))
    return tuple(results)


def linear_quantile(values: Sequence[Fraction], p: Fraction) -> Fraction:
    """USER-WRITTEN CORE: exact scalar quantile with position h=(B-1)*p.

    Sort a copy, retaining duplicates. For h between two integer indices, interpolate
    linearly between their values. At an integer h return that indexed value; this
    includes p=1 and the single-value case, without reading beyond the last index.
    Keep all arithmetic in Fraction; do not round to float or quantize again.

    This numerical helper neither selects a CI method nor decides significance.

    Explain-back / prediction before implementation (user fills):
    Why sorting is needed:
    What to do at p=0, p=1, and B=1:
    Failure expected if duplicate values are removed:
    """
    if not values:
        raise ValueError("Need at least one value")
    if not isinstance(p, Fraction) or any(not isinstance(value, Fraction) for value in values):
        raise TypeError("Values and p must be exact Fractions")
    if not 0 <= p <= 1:
        raise ValueError("p must lie in [0, 1]")
    ordered = sorted(values)
    h = (len(ordered) - 1) * p

    i = h.numerator // h.denominator
    t = h - i

    if t == 0:
        return ordered[i]
    else:
        return ordered[i] * (1 - t) + ordered[i + 1] * t


def quantile_interval(intervals: Sequence[Interval], p: Fraction) -> Interval:
    """USER-WRITTEN CORE: enclose the scalar quantile of interval-valued observations.

    Collect all lower endpoints and all upper endpoints into separate lists. Call
    linear_quantile on each list with the SAME p, returning (lower_q, upper_q).
    It handles sorting and interpolation; keep duplicates and leave inputs unchanged.
    Do not sort interval tuples and take one row, or replace intervals by midpoints.

    This bounds a numerical quantile, not the population mean's statistical CI.
    User's hand calculation: median of [0,100], [1,2], [3,4] is enclosed by [1,4].
    """
    if not intervals:
        raise ValueError("Need at least one interval")
    if not isinstance(p, Fraction):
        raise TypeError("p must be an exact Fraction")
    if not 0 <= p <= 1:
        raise ValueError("p must lie in [0, 1]")
    for interval in intervals:
        if len(interval) != 2 or any(not isinstance(value, Fraction) for value in interval):
            raise TypeError("Each interval needs two exact Fraction endpoints")
        if interval[0] > interval[1]:
            raise ValueError("Endpoints must be ordered")
    lows = []
    highs = []
    for interval in intervals:
        lows.append(interval[0])
        highs.append(interval[1])
    lower_q = linear_quantile(lows, p)
    upper_q = linear_quantile(highs, p)
    return (lower_q, upper_q)
