"""Hand-calculated scalar and interval quantiles, not statistical coverage tests."""

import unittest
from fractions import Fraction as F

from online import confirmation_stats as stats


class GuardTests(unittest.TestCase):
    def test_empty_input_and_outside_probability_rejected(self):
        for values, p in (([], F(0)), ([F(1)], F(-1, 4)), ([F(1)], F(5, 4))):
            with self.subTest(values=values, p=p), self.assertRaises(ValueError):
                stats.linear_quantile(values, p)

    def test_float_values_and_probability_rejected(self):
        for values, p in (([0.5], F(1, 2)), ([F(1)], 0.5)):
            with self.subTest(values=values, p=p), self.assertRaises(TypeError):
                stats.linear_quantile(values, p)


class QuantileTests(unittest.TestCase):
    def setUp(self):
        try:
            stats.linear_quantile([F(0)], F(0))
        except NotImplementedError:
            self.skipTest("USER-WRITTEN linear_quantile not implemented")

    def test_unsorted_hand_examples_and_input_unchanged(self):
        values = list(map(F, [6, -2, 12, 0, 4]))
        original = values.copy()
        for p, expected in ((F(1, 4), F(0)), (F(1, 8), F(-1)), (F(3, 8), F(2))):
            with self.subTest(p=p):
                self.assertEqual(stats.linear_quantile(values, p), expected)
        self.assertEqual(values, original)

    def test_endpoints_and_singleton(self):
        self.assertEqual(stats.linear_quantile([F(7), F(-3)], F(0)), F(-3))
        self.assertEqual(stats.linear_quantile([F(7), F(-3)], F(1)), F(7))
        for p in (F(0), F(1, 3), F(1)):
            with self.subTest(p=p):
                self.assertEqual(stats.linear_quantile([F(-7, 3)], p), F(-7, 3))

    def test_duplicate_values_retain_their_frequency(self):
        self.assertEqual(stats.linear_quantile(list(map(F, [0, 0, 0, 8])), F(1, 2)), F(0))

    def test_interpolation_keeps_sub_float_and_sub_grid_information(self):
        epsilon = F(1, 2**180)
        result = stats.linear_quantile([F(1), F(1) + epsilon], F(1, 3))
        self.assertIsInstance(result, F)
        self.assertEqual(result, F(1) + epsilon / 3)


class IntervalGuardTests(unittest.TestCase):
    def test_empty_reversed_and_invalid_probability_rejected(self):
        for intervals, p in (([], F(0)), ([(F(2), F(1))], F(1, 2)),
                             ([(F(0), F(1))], F(-1)), ([(F(0), F(1))], F(2))):
            with self.subTest(intervals=intervals, p=p), self.assertRaises(ValueError):
                stats.quantile_interval(intervals, p)

    def test_inexact_or_malformed_endpoints_rejected(self):
        for intervals, p in (([(F(0), 1.0)], F(0)), ([(F(0),)], F(0)),
                             ([(F(0), F(1))], 0.5)):
            with self.subTest(intervals=intervals, p=p), self.assertRaises(TypeError):
                stats.quantile_interval(intervals, p)


class IntervalQuantileTests(unittest.TestCase):
    def setUp(self):
        try:
            stats.quantile_interval([(F(0), F(0))], F(0))
        except NotImplementedError:
            self.skipTest("USER-WRITTEN quantile_interval not implemented")

    def test_crossing_endpoint_orders_and_no_input_mutation(self):
        intervals = [(F(0), F(100)), (F(1), F(2)), (F(3), F(4))]
        original = intervals.copy()
        self.assertEqual(stats.quantile_interval(intervals, F(1, 2)), (F(1), F(4)))
        self.assertEqual(intervals, original)

    def test_interpolated_quantile_uses_same_probability_for_both_columns(self):
        self.assertEqual(stats.quantile_interval([(F(0), F(10)), (F(4), F(6))], F(1, 4)),
                         (F(1), F(7)))

    def test_single_interval_preserves_width_at_every_probability(self):
        interval = (F(-1, 3), F(5, 7))
        for p in (F(0), F(1, 3), F(1)):
            with self.subTest(p=p):
                result = stats.quantile_interval([interval], p)
                self.assertEqual(result, interval)
                self.assertTrue(all(isinstance(value, F) for value in result))


if __name__ == "__main__":
    unittest.main()
