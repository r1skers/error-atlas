"""Preparation semantics only; no schedule comparison or new research sample."""

import unittest
from fractions import Fraction as F
from unittest.mock import patch

from online import fixed_contribution_ablation as ablation
from online.fp32_exp import correctly_rounded_exp
from online.fp32_signed import fp32_mul, round_to_fp32
from online.pilot import BlockFamily
from online.pilot import denominator_interval, relative_error
from online.schedules import sequential_chain as chain, balanced_pairwise as balanced, Schedule
from online.merge import merge_reduce
from test_online_pilot import _decimal_real_exp_interval


def reference(family):
    result = []
    for maximum, mass in zip(family.maxima, family.masses):
        low, high = _decimal_real_exp_interval(maximum - family.global_max)
        left, right = round_to_fp32(mass * low), round_to_fp32(mass * high)
        if left != right:
            raise AssertionError("Independent Decimal fixture did not certify rounding")
        result.append(left)
    return tuple(result)


class GuardTests(unittest.TestCase):
    def test_family_type_rejected_before_core(self):
        with self.assertRaises(TypeError):
            ablation.global_contributions(None)


class PreparationTests(unittest.TestCase):
    def test_equal_maxima_preserve_exact_masses_and_order(self):
        family = BlockFamily((F(3), F(3), F(3)), (32, 1, 7))
        self.assertEqual(ablation.global_contributions(family), (F(32), F(1), F(7)))

    def test_non_fp32_exact_gap_matches_independent_real_exp(self):
        family = BlockFamily((F(-10), round_to_fp32(F(1, 3))), (32, 1))
        self.assertNotEqual(family.maxima[0] - family.global_max,
                            round_to_fp32(family.maxima[0] - family.global_max))
        self.assertEqual(ablation.global_contributions(family), reference(family))

    def test_scale_before_rounding_not_after_rounded_exp(self):
        family = BlockFamily((F(-1), F(0)), (3, 1))
        expected = reference(family)
        two_roundings = fp32_mul(F(3), correctly_rounded_exp(F(-1)))[0]
        self.assertNotEqual(expected[0], two_roundings)
        self.assertEqual(ablation.global_contributions(family), expected)

    def test_small_weight_can_survive_after_mass_scaling(self):
        family = BlockFamily((F(0), F(-104)), (1, 2**24))
        expected = reference(family)
        self.assertEqual(correctly_rounded_exp(F(-104)), F(0))
        self.assertGreater(expected[1], F(0))
        self.assertEqual(ablation.global_contributions(family), expected)

    def test_unresolved_rounding_rejected_instead_of_using_midpoint(self):
        family = BlockFamily((F(0), F(-1)), (1, 1))
        def wide_bracket(gap):
            return (F(1), F(1)) if gap == 0 else (F(1, 4), F(1, 2))
        with patch.object(ablation, "real_exp_interval", side_effect=wide_bracket):
            with self.assertRaises(ValueError):
                ablation.global_contributions(family)

    def test_certified_zero_stays_in_original_position(self):
        family = BlockFamily((F(-1000), F(0), F(-1)), (32, 1, 3))
        self.assertEqual(ablation.global_contributions(family), reference(family))
        self.assertEqual(ablation.global_contributions(family)[0], 0)

    def test_preparation_reference_is_exactly_the_existing_pilot_interval(self):
        family = BlockFamily((F(0), F(-1), round_to_fp32(F(-1, 3))), (1, 3, 17))
        prepared = ablation.prepare(family)
        self.assertEqual(prepared.reference, denominator_interval(family))
        self.assertEqual(prepared.exact_sum, sum(prepared.values))
        self.assertEqual(prepared.initialization_error,
                         (sum(prepared.values) - prepared.reference[1],
                          sum(prepared.values) - prepared.reference[0]))


class AdditionControlTests(unittest.TestCase):
    def test_equal_maxima_match_online_with_both_fma_settings(self):
        family = BlockFamily((F(3),) * 5, (2**24, 1, 1, 1, 1))
        prepared = ablation.prepare(family)
        for tree in (chain(5), balanced(5)):
            root, error = ablation.ordinary_reduce(prepared.values, tree)
            for fused in (False, True):
                dump = merge_reduce(family.leaf_max, family.leaf_ell, tree, fused=fused)
                self.assertEqual(root, dump.ell_at(tree.root))
            j = prepared.initialization_error
            self.assertEqual((error + j[0], error + j[1]),
                             (root - prepared.reference[1], root - prepared.reference[0]))
        self.assertEqual(ablation.ordinary_reduce(prepared.values, chain(5))[0], 2**24)
        # Last +1 is a tie above 2**24+2; ties-to-even rounds it up to +4.
        self.assertEqual(ablation.ordinary_reduce(prepared.values, balanced(5))[0], 2**24 + 4)

    def test_single_leaf_has_no_addition_error(self):
        self.assertEqual(ablation.ordinary_reduce((F(7),), chain(1)), (F(7), F(0)))

    def test_malformed_graph_and_leaf_count_are_rejected(self):
        with self.assertRaises(ValueError):
            ablation.ordinary_reduce((F(1), F(2)), Schedule(2, ((0, 0),), "invalid"))
        with self.assertRaises(ValueError):
            ablation.ordinary_reduce((F(1),), chain(2))

    def test_max_first_and_ascending_share_multiset_and_reference(self):
        maxima = tuple(F(-i) for i in range(8))
        expected = None
        for logits in (maxima, maxima[1:] + maxima[:1], tuple(reversed(maxima))):
            family = BlockFamily(logits, (1,) * 8)
            prepared = ablation.prepare(family)
            if expected is None:
                expected = prepared
            self.assertEqual(sorted(prepared.values), sorted(expected.values))
            self.assertEqual(prepared.reference, expected.reference)
            for tree in (chain(8), balanced(8)):
                fixed_root, _ = ablation.ordinary_reduce(prepared.values, tree)
                for fused in (False, True):
                    dump = merge_reduce(logits, family.leaf_ell, tree, fused=fused)
                    if logits == maxima and tree.kind == chain(8).kind:
                        self.assertEqual(fixed_root, dump.ell_at(tree.root))
                    self.assertGreaterEqual(relative_error(dump.ell_at(tree.root), prepared.reference)[0], 0)


if __name__ == "__main__":
    unittest.main()
