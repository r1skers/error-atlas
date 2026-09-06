"""Certified fixed-global-scale control, implemented by agent at user's request.

The fixed contribution baseline knows the final maximum in advance. It is a mechanism
control, not a deployable streaming implementation or a new experiment result.
See notes/online_fixed_contribution_ablation_v1.md before running comparisons.

Explain-back (user fills):
Why the logit gap is exact in this reference preparation:
Why round(n * real_exp(gap)) differs from n * rounded_exp(gap):
Where fixed-leaf preparation error is recorded separately from addition error:
"""

from __future__ import annotations

from fractions import Fraction
from dataclasses import dataclass

from .fp32_signed import round_to_fp32
from .pilot import BlockFamily, real_exp_interval
from .schedules import Schedule, check_schedule

# Tests load ``online`` directly; the CLI loads the repository package.
if __package__ == "online":
    from rewrite.fp32_oracle import Tree, reduce_tree
else:
    from ..rewrite.fp32_oracle import Tree, reduce_tree


@dataclass(frozen=True)
class Preparation:
    values: tuple[Fraction, ...]
    reference: tuple[Fraction, Fraction]

    @property
    def exact_sum(self):
        return sum(self.values, Fraction(0))

    @property
    def initialization_error(self):
        return self.exact_sum - self.reference[1], self.exact_sum - self.reference[0]


def global_contributions(family: BlockFamily) -> tuple[Fraction, ...]:
    """c_hat[b] = RN32(n_b * exp(m_b - M)), rounded ONCE.

    For each block, use the exact Fraction gap m_b-family.global_max. Obtain real exp
    bounds with real_exp_interval; multiply BOTH by n_b before rounding to FP32.
    Equal rounded endpoints certify the unique answer; append it in original leaf order.
    If they differ, raise ValueError: the fixed precision did not certify rounding.
    Never return a rounded midpoint, silently increase precision, or skip the block.

    Do not use correctly_rounded_exp here: it requires a stored FP32 argument, rounds
    the weight separately, and can underflow a weight whose scaled contribution survives.
    Zero contributions may result from certified final underflow and must be retained.
    exp(0)=1 may be treated exactly. Keep all calculations in Fraction.
    """
    return prepare(family).values


def prepare(family: BlockFamily) -> Preparation:
    """Compute leaves and the original pilot reference in one exp pass.

    Keep the pilot's enclosure even at zero, so old A endpoints replay exactly.
    All computation is rational; no rounded exponential or logit subtraction.
    """
    if not isinstance(family, BlockFamily):
        raise TypeError("Need a validated constant-block BlockFamily")
    values = []
    total_low = total_high = Fraction(0)
    maximum = family.global_max
    for logit, mass in zip(family.maxima, family.masses):
        low, high = real_exp_interval(logit - maximum)
        low, high = mass * low, mass * high
        left, right = round_to_fp32(low), round_to_fp32(high)
        if left != right:
            raise ValueError("Global contribution rounding is not certified by the interval")
        values.append(left)
        total_low += low
        total_high += high
    return Preparation(tuple(values), (total_low, total_high))


def ordinary_reduce(values: tuple[Fraction, ...], schedule: Schedule):
    """Use the existing FP32 add oracle on exactly the saved online graph."""
    check_schedule(schedule)
    trace = reduce_tree(values, Tree(schedule.leaf_count, schedule.nodes))
    if trace.error != sum(trace.deltas, Fraction(0)):
        raise AssertionError("Ordinary addition residuals do not close")
    root = trace.values[0] if schedule.leaf_count == 1 else trace.node_values[-1]
    return root, trace.error
