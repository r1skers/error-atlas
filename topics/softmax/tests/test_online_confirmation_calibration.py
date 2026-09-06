"""Deterministic plumbing and hand-scored trials; no Monte Carlo calibration claims."""

import unittest
from fractions import Fraction as F

from online import confirmation_calibration as cal
from online.confirmation_stats import FamilySample


class ScaffoldTests(unittest.TestCase):
    def test_population_means_and_joint_correlation_are_exact(self):
        cases = {case.name: case for case in cal.populations()}
        for name in ("symmetric_identical", "symmetric_corr08", "rare_negative", "rare_positive", "zero_constant"):
            self.assertEqual(cases[name].truth, (F(0), F(0)))
        self.assertEqual(cases["positive_constant"].truth, (cal.SCALE, cal.SCALE))
        self.assertEqual(cases["mixed_null_positive"].truth, (F(0), 2 * cal.SCALE))
        case = cases["symmetric_corr08"]
        covariance = sum(row[0] * row[1] * weight for row, weight in zip(case.outcomes, case.weights)) / sum(case.weights)
        self.assertEqual(covariance / cal.SCALE**2, F(4, 5))

    def test_draws_reproduce_and_keep_joint_outcomes_and_unique_ids(self):
        case = cal.populations()[0]
        rows = cal.draw_families(case, 12, 123)
        self.assertEqual(rows, cal.draw_families(case, 12, 123))
        self.assertEqual(len({row.family_id for row in rows}), 12)
        for row in rows:
            self.assertEqual(row.separate, row.fused)
            self.assertIn(row.separate[0], (-cal.SCALE, cal.SCALE))

    def test_candidate_composes_existing_means_and_quantiles(self):
        rows = [FamilySample("a", (F(-2), F(-2)), (F(2), F(2))),
                FamilySample("b", (F(2), F(2)), (F(-2), F(-2)))]
        self.assertEqual(cal.percentile_candidate(rows, [(0, 0), (0, 1), (1, 1)], F(1, 4)),
                         ((F(-1), F(-1)), (F(-1), F(-1))))

    def test_constant_observed_sample_is_retained(self):
        rows = [FamilySample("a", (F(1), F(1)), (F(1), F(1)))]
        self.assertEqual(cal.percentile_candidate(rows, [(0,), (0,)], F(1, 20)),
                         ((F(1), F(1)), (F(1), F(1))))


class VerdictGuardTests(unittest.TestCase):
    def test_bad_shape_order_and_inexact_truth_rejected(self):
        with self.assertRaises(ValueError):
            cal.assess_trial(((F(0), F(0)),), (F(0), F(0)))
        with self.assertRaises(ValueError):
            cal.assess_trial(((F(1), F(0)), (F(0), F(0))), (F(0), F(0)))
        with self.assertRaises(TypeError):
            cal.assess_trial(((F(0), F(0)), (F(0), F(0))), (0.0, F(0)))


class VerdictTests(unittest.TestCase):
    def setUp(self):
        try:
            cal.assess_trial(((F(0), F(0)), (F(0), F(0))), (F(0), F(0)))
        except NotImplementedError:
            self.skipTest("USER-WRITTEN assess_trial not implemented")

    def test_zero_is_covered_and_not_strictly_positive(self):
        self.assertEqual(cal.assess_trial(((F(0), F(0)), (F(-1), F(2))), (F(0), F(0))),
                         cal.TrialVerdict((True, True), (False, False), (False, False), False, False))

    def test_coverage_failure_and_false_positive_are_different(self):
        self.assertEqual(cal.assess_trial(((F(3), F(4)), (F(1), F(2))), (F(2), F(0))),
                         cal.TrialVerdict((False, False), (True, True), (False, True), True, True))

    def test_one_rejection_does_not_reject_conjunction(self):
        self.assertEqual(cal.assess_trial(((F(1), F(1)), (F(-2), F(0))), (F(0), F(0))),
                         cal.TrialVerdict((False, True), (True, False), (True, False), False, False))

    def test_positive_truth_and_endpoint_choice(self):
        self.assertEqual(cal.assess_trial(((F(1), F(9)), (F(2), F(10))), (F(1), F(3))),
                         cal.TrialVerdict((True, True), (True, True), (False, False), True, False))

    def test_negative_truth_noncoverage_without_positive_rejection(self):
        self.assertEqual(cal.assess_trial(((F(-1), F(3)), (F(1), F(1))), (F(-2), F(-1))),
                         cal.TrialVerdict((False, False), (False, True), (False, True), False, False))


if __name__ == "__main__":
    unittest.main()
