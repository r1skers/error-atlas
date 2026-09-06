"""Synthetic calibration scaffolding; no softmax confirmation inputs are generated.

The percentile candidate composes the user's existing statistics. Trial scoring is
USER-WRITTEN CORE. No Monte Carlo study runs at import or from a CLI in this module.

Explain-back (user fills before implementing assess_trial):
What differs between a bootstrap replicate and a fresh outer trial:
When a lower confidence bound covers the known population mean:
Why rejection uses > 0, not >= 0:
When rejecting the conjunction of two positive means is a false positive:
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

from .confirmation_stats import ArmMeans, FamilySample, bootstrap_means, quantile_interval

ArmTruth = tuple[Fraction, Fraction]
ArmFlags = tuple[bool, bool]
SCALE = Fraction(1, 2**24)


@dataclass(frozen=True)
class Population:
    """Finite joint population; integer weights describe exact sampling probabilities."""

    name: str
    outcomes: tuple[ArmTruth, ...]
    weights: tuple[int, ...]

    def __post_init__(self):
        if not self.name or not self.outcomes or len(self.outcomes) != len(self.weights):
            raise ValueError("Need a named, nonempty population with matching weights")
        if any(type(weight) is not int or weight < 1 for weight in self.weights):
            raise ValueError("Weights must be positive integers")
        if any(len(row) != 2 or any(not isinstance(value, Fraction) for value in row)
               for row in self.outcomes):
            raise TypeError("Each outcome must contain two exact Fractions")

    @property
    def truth(self) -> ArmTruth:
        total = sum(self.weights)
        return tuple(sum((row[arm] * weight for row, weight in zip(self.outcomes, self.weights)),
                         Fraction(0)) / total for arm in (0, 1))


def populations() -> tuple[Population, ...]:
    """Analytically specified stress cases, not a fit to the observed pilot."""
    def case(name, rows, weights):
        return Population(name, tuple(tuple(Fraction(x) * SCALE for x in row) for row in rows),
                          tuple(weights))

    return (
        case("symmetric_identical", [(-1, -1), (1, 1)], [1, 1]),
        case("symmetric_corr08", [(-1, -1), (1, 1), (-1, 1), (1, -1)], [9, 9, 1, 1]),
        case("rare_negative", [(1, 1), (-31, -31)], [31, 1]),
        case("rare_positive", [(-1, -1), (31, 31)], [31, 1]),
        case("zero_constant", [(0, 0)], [1]),
        case("positive_constant", [(1, 1)], [1]),
        case("mixed_null_positive", [(-1, 1), (1, 3)], [1, 1]),
    )


def draw_families(population: Population, count: int, seed: int) -> tuple[FamilySample, ...]:
    """Sample entire joint outcomes with replacement using a local deterministic PRNG."""
    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive integer")
    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    rng = random.Random(seed)
    rows = []
    for index in range(count):
        ticket = rng.randrange(sum(population.weights))
        for outcome, weight in zip(population.outcomes, population.weights):
            if ticket < weight:
                separate, fused = outcome
                rows.append(FamilySample(f"{population.name}:{seed}:{index}",
                                         (separate, separate), (fused, fused)))
                break
            ticket -= weight
    return tuple(rows)


def percentile_candidate(samples: Sequence[FamilySample], draws: Sequence[Sequence[int]],
                         alpha: Fraction) -> ArmMeans:
    """Assemble the existing user functions for a candidate one-sided percentile bound.

    Returns numerical enclosures of the two lower confidence endpoints. The deployed
    conservative endpoint for each arm is its enclosure's lower endpoint, not midpoint.
    A constant observed sample is retained for scoring; it is never silently dropped.
    """
    if not isinstance(alpha, Fraction):
        raise TypeError("alpha must be an exact Fraction")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    distribution = bootstrap_means(samples, draws)
    return tuple(quantile_interval([row[arm] for row in distribution], alpha) for arm in (0, 1))


@dataclass(frozen=True)
class TrialVerdict:
    covered: ArmFlags
    rejected: ArmFlags
    false_positive: ArmFlags
    conjunction_rejected: bool
    conjunction_false_positive: bool


def assess_trial(lower_bounds: ArmMeans, truth: ArmTruth) -> TrialVerdict:
    """USER-WRITTEN CORE: score ONE outer trial using the known population means.

    For each arm choose L=lower_bounds[arm][0]. Its statistical interval is [L,+inf).
    covered: L <= truth[arm]. rejected: L > 0 (the claim is strictly positive mean).
    false_positive: rejected while truth[arm] <= 0. A positive truth can be missed
    by the interval without the positive-mean rejection being a false positive.

    The two-arm conjunction is rejected only when BOTH arms reject; its rejection
    is false if AT LEAST ONE true mean is <= 0. This is a one-stratum diagnostic,
    not yet the proposed four-condition confirmation decision.

    Return a TrialVerdict with actual bool values. Keep all numeric comparisons exact.
    """
    if len(lower_bounds) != 2 or len(truth) != 2:
        raise ValueError("Need exactly two arms")
    for bound, mean in zip(lower_bounds, truth):
        if len(bound) != 2 or any(not isinstance(value, Fraction) for value in (*bound, mean)):
            raise TypeError("Bounds and truths must be exact Fractions")
        if bound[0] > bound[1]:
            raise ValueError("Endpoints must be ordered")

    covered = []
    rejected = []
    false_positive = []
    for arm in (0, 1):
        L = lower_bounds[arm][0]
        theta = truth[arm]
        covered.append(L <= theta)
        arm_rejected = L > 0
        rejected.append(arm_rejected)
        false_positive.append(arm_rejected and theta <= 0)

    conjunction_rejected = lower_bounds[0][0] > 0 and lower_bounds[1][0] > 0
    conjunction_false_positive = conjunction_rejected and (truth[0] <= 0 or truth[1] <= 0)

    return TrialVerdict(
        covered=tuple(covered),
        rejected=tuple(rejected),
        false_positive=tuple(false_positive),
        conjunction_rejected=conjunction_rejected,
        conjunction_false_positive=conjunction_false_positive,
    )
