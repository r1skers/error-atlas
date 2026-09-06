"""Final RN32 division and two user-written real-output interval cores.

Explain-back (user fills):
When V is negative, what happens to the exp interval endpoints:乘以负的 n_b V_b 后，区间端点顺序反转。
Why the real reference uses exact logit gaps rather than dump's rounded gaps:dump 的 gap 已包含被测算法的 FP32 减法误差，不能放进真值参考。
Why signed numerator intervals need all four quotient corners:分子可能为负，分母严格正，所以四个角点都要算，才能保证包含真实比值。
Why V=1 has an exact output reference even if independent interval division is wider:常数 V=v 时 O_*=v ell_*，两者完全相关，因此 y_*=O_*/ell_*=v。
分别构造区间再相除会丢失这种相关性，产生不必要的宽区间。
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .fp32_signed import round_to_fp32
from .merge import MergeDump
from .output_probe import leaf_numerators, propagate_numerator
from .pilot import BlockFamily, denominator_interval, real_exp_interval

Interval = tuple[Fraction, Fraction]


@dataclass(frozen=True)
class ComputedOutput:
    numerator: Fraction
    denominator: Fraction
    before_division_rounding: Fraction
    value: Fraction
    division_residual: Fraction


@dataclass(frozen=True)
class OutputReference:
    numerator: Interval
    denominator: Interval
    output: Interval


def finish_output(dump: MergeDump, leaf_O: tuple[Fraction, ...]) -> ComputedOutput:
    """Scaffolding: existing O propagation, exact ratio, then ONE existing RN32 call.

    This defines the CPU division contract. It does not model approximate reciprocal
    instructions or the subsequent FP16/BF16 storage cast.
    """
    nodes = propagate_numerator(dump, leaf_O)
    numerator = nodes[-1] if nodes else leaf_O[0]
    denominator = dump.ell_at(dump.schedule.root)
    if denominator <= 0:
        raise ValueError("Output division requires a positive computed denominator")
    ratio = numerator / denominator
    value = round_to_fp32(ratio)
    return ComputedOutput(numerator, denominator, ratio, value, value - ratio)


def numerator_interval(family: BlockFamily, values: tuple[Fraction, ...]) -> Interval:
    """USER-WRITTEN CORE: enclose O_star = sum n_b V_b exp(m_b-M).

    Use real_exp_interval on the EXACT Fraction difference m_b-family.global_max.
    Multiply its endpoints by the exact signed coefficient n_b*V_b. A negative
    coefficient reverses their order. Add lower endpoints and upper endpoints
    separately. Keep Fraction throughout; no FP32 rounding or rounded-exp calls.

    Leaf validation below also checks the V alphabet; it does not compute a reference.
    Zero coefficients contribute exactly zero. exp(0)=1 may be handled exactly.
    """
    leaf_numerators(family, values)

    M = family.global_max
    low, high = Fraction(0), Fraction(0)

    for m, n, value in zip(family.maxima, family.masses, values):
        coefficient = Fraction(n) * value

        if coefficient == 0:
            continue

        exp_low, exp_high = real_exp_interval(m - M)

        endpoint_a = coefficient * exp_low
        endpoint_b = coefficient * exp_high

        low += min(endpoint_a, endpoint_b)
        high += max(endpoint_a, endpoint_b)

    return low, high


def quotient_interval(numerator: Interval, denominator: Interval) -> Interval:
    """USER-WRITTEN CORE: enclose O/ell for signed O and strictly positive ell.

    Given independent rectangular bounds, evaluate the four endpoint quotients and
    return their minimum and maximum. Do not assume O is positive. No float or RN32.
    This is conservative when the actual numerator and denominator are correlated;
    it is not generally the tight range of the correlated softmax expression.
    """
    if len(numerator) != 2 or len(denominator) != 2:
        raise ValueError("Need two endpoints per interval")
    if numerator[0] > numerator[1] or denominator[0] > denominator[1]:
        raise ValueError("Intervals must be ordered")
    if denominator[0] <= 0:
        raise ValueError("Denominator interval must be strictly positive")

    o_low, o_high = numerator
    l_low, l_high = denominator

    corners = (
        o_low / l_low,
        o_low / l_high,
        o_high / l_low,
        o_high / l_high,
    )

    return min(corners), max(corners)

def reference_output(family: BlockFamily, values: tuple[Fraction, ...]) -> OutputReference:
    """Assemble references; preserve the known constant-V correlation exactly.

    Identical V_b=v implies O_star=v*ell_star and y_star=v. Do not infer this
    correlation merely because two independently constructed intervals overlap.
    """
    numerator = numerator_interval(family, values)
    denominator = denominator_interval(family)
    output = ((values[0], values[0]) if all(v == values[0] for v in values)
              else quotient_interval(numerator, denominator))
    return OutputReference(numerator, denominator, output)
