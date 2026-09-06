"""Negative control for an n-independent online weight-error claim.

Correctly rounded exp can repeatedly round a weight below one to one. This is an
audit counterexample, not a new sample distribution or an engineering result.
"""

import unittest
from fractions import Fraction as F

from online.attribution import decompose
from online.fp32_exp import correctly_rounded_exp
from online.merge import merge_reduce
from online.pilot import BlockFamily
from online.schedules import sequential_chain


class WeightPathCounterexampleTests(unittest.TestCase):
    def test_correct_exp_can_accumulate_weight_error_without_merge_rounding(self):
        delta, u = F(1, 2**26), F(1, 2**24)
        self.assertEqual(correctly_rounded_exp(-delta), 1)
        for fused in (False, True):
            prior_upper = None
            for n, lower_units, upper_units in ((32, 3, 4), (128, 15, 16)):
                family = BlockFamily(tuple(b * delta for b in range(n)), (32,) * n)
                tree = sequential_chain(n)
                dump = merge_reduce(family.leaf_max, family.leaf_ell, tree, fused=fused)
                self.assertTrue(all(w == 1 for w in dump.weight_left + dump.weight_right))
                result = decompose(family, dump)
                self.assertEqual(result.computed, 32 * n)
                self.assertEqual(result.frozen, 32 * n)
                self.assertEqual(result.rounding_error, 0)
                self.assertTrue(all(r == 0 for r in result.residuals))
                low, high = result.relative_weight
                self.assertGreater(low, lower_units * u)
                self.assertLess(high, upper_units * u)
                if prior_upper is not None:
                    self.assertGreater(low, 3 * prior_upper)
                prior_upper = high


if __name__ == "__main__":
    unittest.main()
