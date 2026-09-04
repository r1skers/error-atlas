"""Tests for the (m, l) merge recurrence and the frozen-weight weighted identity.

Tests skip while the core still raises NotImplementedError. The residual reference here
is recomputed independently from the dump rather than taken from the implementation, so
agreement is a differential result, not a restatement.
"""

import random
import unittest
from fractions import Fraction

from online import schedules
from online.fp32_signed import fp32_add, fp32_fma, fp32_mul, fp32_sub, round_to_fp32
from online.merge import (
    frozen_weight_reference,
    merge_reduce,
    provisional_fp32_exp,
    weighted_residuals,
)

SEED = 20260904
GROUPS = 12
LEAVES = 32
SPREADS = (2.0, 12.0, 25.0, 60.0)


def _schedule_family(size: int) -> tuple[schedules.Schedule, ...]:
    return (
        schedules.sequential_chain(size),
        schedules.balanced_pairwise(size),
        schedules.warp_shfl_down(size),
        schedules.split_k(size, 4),
    )


def _leaves(rng: random.Random, size: int, spread: float):
    maxima = tuple(round_to_fp32(Fraction(rng.uniform(-spread, spread))) for _ in range(size))
    ells = tuple(round_to_fp32(Fraction(rng.uniform(1.0, 4.0))) for _ in range(size))
    return maxima, ells


def _path_weight(dump, node: int) -> Fraction:
    """Product of the computed rescale factors from ``node`` up to the root."""
    parent = {}
    count = dump.schedule.leaf_count
    for k, (left, right) in enumerate(dump.schedule.nodes):
        parent[left] = (count + k, 0)
        parent[right] = (count + k, 1)
    weight, current = Fraction(1), node
    while current != dump.schedule.root:
        up, side = parent[current]
        weight *= dump.weights_at(up)[side]
        current = up
    return weight


def _expected_residuals(dump) -> tuple[Fraction, ...]:
    """Recompute each node's local residual from the dump alone, independently."""
    out = []
    for k, (left, right) in enumerate(dump.schedule.nodes):
        w_left, w_right = dump.weight_left[k], dump.weight_right[k]
        ell_left, ell_right = dump.ell_at(left), dump.ell_at(right)
        if dump.fused and w_left == 1:
            out.append(dump.node_ell[k] - (ell_right * w_right + ell_left))
        elif dump.fused and w_right == 1:
            out.append(dump.node_ell[k] - (ell_left * w_left + ell_right))
        else:
            product_left, mu_left = fp32_mul(ell_left, w_left)
            product_right, mu_right = fp32_mul(ell_right, w_right)
            _, alpha = fp32_add(product_left, product_right)
            out.append(mu_left + mu_right + alpha)
    return tuple(out)


def _skip_unless_implemented(test: unittest.TestCase, call) -> None:
    try:
        call()
    except NotImplementedError:
        test.skipTest("online merge core not implemented yet")


class MergeDumpTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = random.Random(SEED)
        schedule = schedules.balanced_pairwise(4)
        maxima, ells = _leaves(self.rng, 4, 12.0)
        _skip_unless_implemented(self, lambda: merge_reduce(maxima, ells, schedule))

    def test_every_dumped_field_is_a_stored_fp32_value(self) -> None:
        for spread in SPREADS:
            for schedule in _schedule_family(LEAVES):
                maxima, ells = _leaves(self.rng, LEAVES, spread)
                for fused in (False, True):
                    dump = merge_reduce(maxima, ells, schedule, fused=fused)
                    with self.subTest(kind=schedule.kind, spread=spread, fused=fused):
                        self.assertTrue(dump.is_well_formed())

    def test_winning_branch_costs_nothing(self) -> None:
        """m_a (-) m_v is exactly 0 on the larger side, exp(0) is 1, and l (*) 1 is exact."""
        for spread in SPREADS:
            for schedule in _schedule_family(LEAVES):
                maxima, ells = _leaves(self.rng, LEAVES, spread)
                dump = merge_reduce(maxima, ells, schedule)
                for k, (left, right) in enumerate(schedule.nodes):
                    winner = 0 if dump.max_at(left) >= dump.max_at(right) else 1
                    with self.subTest(kind=schedule.kind, node=k):
                        self.assertEqual(dump.weights_at(schedule.leaf_count + k)[winner], 1)

    def test_node_max_is_the_exact_maximum(self) -> None:
        for schedule in _schedule_family(LEAVES):
            maxima, ells = _leaves(self.rng, LEAVES, 25.0)
            dump = merge_reduce(maxima, ells, schedule)
            for k, (left, right) in enumerate(schedule.nodes):
                self.assertEqual(dump.node_max[k], max(dump.max_at(left), dump.max_at(right)))

    def test_gaps_are_the_rounded_difference(self) -> None:
        """Pins contract step 2 on its own: Delta goes through FP32 subtraction.

        Kept separate from the weight check so that a device dump can say which of the
        two channels drifted. The weighted identity cannot check either one: its residual
        is *defined* as ``l_v - exact``, so it holds for whatever the dump carries.
        """
        for spread in SPREADS:
            for schedule in _schedule_family(LEAVES):
                maxima, ells = _leaves(self.rng, LEAVES, spread)
                dump = merge_reduce(maxima, ells, schedule)
                for k, (left, right) in enumerate(schedule.nodes):
                    node_max = dump.node_max[k]
                    with self.subTest(kind=schedule.kind, spread=spread, node=k):
                        for child, gap in (
                            (left, dump.gap_left[k]),
                            (right, dump.gap_right[k]),
                        ):
                            self.assertEqual(gap, fp32_sub(dump.max_at(child), node_max)[0])
                        winner = 0 if dump.max_at(left) >= dump.max_at(right) else 1
                        self.assertEqual(dump.gaps_at(schedule.leaf_count + k)[winner], 0)

    def test_weights_are_the_exponential_of_the_dumped_gap(self) -> None:
        """Pins contract step 3 against the gap the dump actually carries."""
        for spread in SPREADS:
            for schedule in _schedule_family(LEAVES):
                maxima, ells = _leaves(self.rng, LEAVES, spread)
                dump = merge_reduce(maxima, ells, schedule)
                for k in range(len(schedule.nodes)):
                    with self.subTest(kind=schedule.kind, spread=spread, node=k):
                        self.assertEqual(
                            dump.weight_left[k], provisional_fp32_exp(dump.gap_left[k])
                        )
                        self.assertEqual(
                            dump.weight_right[k], provisional_fp32_exp(dump.gap_right[k])
                        )

    def test_node_ell_is_the_rounded_merge_of_its_children(self) -> None:
        """Pins contract steps 4 and 5, including that fused really rounds once."""
        for spread in SPREADS:
            for schedule in _schedule_family(LEAVES):
                maxima, ells = _leaves(self.rng, LEAVES, spread)
                for fused in (False, True):
                    dump = merge_reduce(maxima, ells, schedule, fused=fused)
                    for k, (left, right) in enumerate(schedule.nodes):
                        w_left, w_right = dump.weight_left[k], dump.weight_right[k]
                        ell_left, ell_right = dump.ell_at(left), dump.ell_at(right)
                        if fused and w_left == 1:
                            expected, _ = fp32_fma(ell_right, w_right, ell_left)
                        elif fused and w_right == 1:
                            expected, _ = fp32_fma(ell_left, w_left, ell_right)
                        else:
                            product_left, _ = fp32_mul(ell_left, w_left)
                            product_right, _ = fp32_mul(ell_right, w_right)
                            expected, _ = fp32_add(product_left, product_right)
                        with self.subTest(kind=schedule.kind, fused=fused, node=k):
                            self.assertEqual(dump.node_ell[k], expected)

    def test_non_stored_leaves_are_rejected(self) -> None:
        schedule = schedules.balanced_pairwise(4)
        maxima, ells = _leaves(self.rng, 4, 12.0)
        with self.assertRaises(ValueError):
            merge_reduce(maxima, (Fraction(1, 3),) + ells[1:], schedule)
        with self.assertRaises(ValueError):
            merge_reduce((Fraction(1, 3),) + maxima[1:], ells, schedule)


class IdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = random.Random(SEED + 1)
        schedule = schedules.balanced_pairwise(4)
        maxima, ells = _leaves(self.rng, 4, 12.0)
        _skip_unless_implemented(
            self, lambda: weighted_residuals(merge_reduce(maxima, ells, schedule))
        )

    def test_identity_is_bit_exact_across_the_schedule_family(self) -> None:
        for spread in SPREADS:
            for schedule in _schedule_family(LEAVES):
                for fused in (False, True):
                    with self.subTest(kind=schedule.kind, spread=spread, fused=fused):
                        for _ in range(GROUPS):
                            maxima, ells = _leaves(self.rng, LEAVES, spread)
                            dump = merge_reduce(maxima, ells, schedule, fused=fused)
                            error = dump.ell_at(schedule.root) - frozen_weight_reference(dump)
                            self.assertEqual(error, sum(weighted_residuals(dump)))

    def test_weighted_residuals_match_an_independent_recomputation(self) -> None:
        for spread in SPREADS:
            for schedule in _schedule_family(LEAVES):
                for fused in (False, True):
                    maxima, ells = _leaves(self.rng, LEAVES, spread)
                    dump = merge_reduce(maxima, ells, schedule, fused=fused)
                    expected = tuple(
                        _path_weight(dump, schedule.leaf_count + k) * residual
                        for k, residual in enumerate(_expected_residuals(dump))
                    )
                    with self.subTest(kind=schedule.kind, spread=spread, fused=fused):
                        self.assertEqual(weighted_residuals(dump), expected)

    def test_analytic_weights_break_the_identity(self) -> None:
        """Negative control: exp(m_v - m_root) is not the product of computed factors."""
        deviations = []
        for spread in (12.0, 25.0):
            for schedule in _schedule_family(LEAVES):
                for _ in range(GROUPS):
                    maxima, ells = _leaves(self.rng, LEAVES, spread)
                    dump = merge_reduce(maxima, ells, schedule)
                    error = dump.ell_at(schedule.root) - frozen_weight_reference(dump)
                    root_max = dump.max_at(schedule.root)
                    analytic = sum(
                        provisional_fp32_exp(fp32_sub(dump.node_max[k], root_max)[0]) * residual
                        for k, residual in enumerate(_expected_residuals(dump))
                    )
                    if error != 0:
                        deviations.append(abs(float((analytic - error) / error)))
        self.assertGreater(len(deviations), 0)
        self.assertGreater(max(deviations), 0, "analytic weights reproduced the identity")

    def test_reference_uses_no_rounding(self) -> None:
        """A case where the FP32 sum and the exact sum genuinely differ.

        ``1 + 2**-23`` is the FP32 value just above one, so the exact sum is
        ``2 + 2**-23``. That sits exactly halfway between 2 and the next FP32 value, and
        ties-to-even sends it back to 2. A reference that rounded anywhere would report
        2 and the test would not notice; 1 + 2 would not separate them at all.
        """
        schedule = schedules.balanced_pairwise(2)
        maxima = (Fraction(3), Fraction(3))
        ells = (Fraction(1), Fraction(1) + Fraction(1, 2**23))
        dump = merge_reduce(maxima, ells, schedule)
        self.assertEqual(dump.node_ell[0], Fraction(2))
        self.assertEqual(frozen_weight_reference(dump), Fraction(2) + Fraction(1, 2**23))
        self.assertEqual(sum(weighted_residuals(dump)), -Fraction(1, 2**23))


class AbsorptionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.schedule = schedules.balanced_pairwise(2)
        _skip_unless_implemented(
            self,
            lambda: merge_reduce((Fraction(0), Fraction(0)), (Fraction(1), Fraction(1)),
                                 self.schedule),
        )

    def test_total_absorption_loses_the_whole_downweighted_block(self) -> None:
        """Contract section 5: the residual is the full magnitude of the losing side."""
        dump = merge_reduce((Fraction(0), Fraction(-40)), (Fraction(1), Fraction(1)),
                            self.schedule)
        weight = dump.weight_right[0]
        self.assertGreater(weight, 0)
        self.assertEqual(dump.node_ell[0], Fraction(1))  # the addend vanished entirely
        self.assertEqual(sum(weighted_residuals(dump)), -weight)

    def test_weight_can_underflow_to_zero_without_breaking_the_identity(self) -> None:
        dump = merge_reduce((Fraction(0), Fraction(-200)), (Fraction(1), Fraction(1)),
                            self.schedule)
        self.assertEqual(dump.weight_right[0], 0)
        error = dump.ell_at(self.schedule.root) - frozen_weight_reference(dump)
        self.assertEqual(error, sum(weighted_residuals(dump)))
        self.assertEqual(error, 0)


if __name__ == "__main__":
    unittest.main()
