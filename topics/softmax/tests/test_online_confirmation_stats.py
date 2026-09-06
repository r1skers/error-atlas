"""Synthetic tests for the statistics core; no new research inputs or confidence claims."""

import unittest
from fractions import Fraction as F

from online import confirmation_stats as stats


def sample(name, separate, fused):
    return stats.FamilySample(name, (F(separate), F(separate)), (F(fused), F(fused)))


class GuardTests(unittest.TestCase):
    def test_quantization_encloses_negative_and_positive_non_dyadics(self):
        row = stats.FamilySample("a", (F(-1, 3), F(1, 7)), (F(2, 3), F(4, 3)))
        grid = stats._grid_rows([row])[0]
        for original, encoded in zip((row.separate, row.fused), grid):
            low, high = (F(value, stats.GRID) for value in encoded)
            self.assertLessEqual(low, original[0])
            self.assertGreaterEqual(high, original[1])
            self.assertLess(original[0] - low, F(1, stats.GRID))
            self.assertLess(high - original[1], F(1, stats.GRID))

    def test_source_duplicates_invalid_intervals_and_bad_indices_rejected(self):
        row = sample("a", 1, 2)
        for rows, indices in (([], [0]), ([row, row], [0]), ([row], []), ([row], [1]), ([row], [True])):
            with self.subTest(rows=rows, indices=indices), self.assertRaises(ValueError):
                stats.paired_mean(rows, indices)
        with self.assertRaises(ValueError):
            stats.paired_mean([stats.FamilySample("a", (F(2), F(1)), (F(0), F(0)))], [0])

    def test_rng_reproducible_and_draws_have_original_size(self):
        draws = stats.resample_indices(3, 8, 123)
        self.assertEqual(draws, stats.resample_indices(3, 8, 123))
        self.assertEqual(len(draws), 8)
        self.assertTrue(all(len(row) == 3 and all(0 <= index < 3 for index in row) for row in draws))
        self.assertTrue(any(len(set(row)) < 3 for row in draws))

    def test_wrong_bootstrap_sample_size_is_rejected(self):
        with self.assertRaises(ValueError):
            stats.bootstrap_means([sample("a", 0, 1), sample("b", 1, 0)], [(0,)])


class MeanTests(unittest.TestCase):
    def setUp(self):
        try:
            stats.paired_mean([sample("a", 0, 0)], [0])
        except NotImplementedError:
            self.skipTest("USER-WRITTEN paired_mean not implemented")

    def test_negative_differences_and_repeated_indices(self):
        rows = [sample("a", -2, 2), sample("b", 4, -4)]
        self.assertEqual(stats.paired_mean(rows, [0, 0, 1]), ((F(0), F(0)), (F(0), F(0))))
        self.assertEqual(stats.paired_mean(rows, [1, 1]), ((F(4), F(4)), (F(-4), F(-4))))

    def test_interval_width_is_preserved_and_outputs_remain_fractions(self):
        rows = [stats.FamilySample("a", (F(-2), F(0)), (F(3), F(5))),
                stats.FamilySample("b", (F(2), F(4)), (F(-1), F(1)))]
        result = stats.paired_mean(rows, [0, 1])
        self.assertEqual(result, ((F(0), F(2)), (F(1), F(3))))
        self.assertTrue(all(isinstance(endpoint, F) for arm in result for endpoint in arm))


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        try:
            stats.bootstrap_means([sample("a", 0, 0)], [(0,)])
        except NotImplementedError:
            self.skipTest("USER-WRITTEN bootstrap_means not implemented")

    def test_draw_order_multiplicity_and_pairing_are_preserved(self):
        rows = [sample("a", -2, 2), sample("b", 2, -2)]
        draws = ((1, 1), (0, 1), (0, 0))
        result = stats.bootstrap_means(rows, draws)
        self.assertEqual(result, (((F(2), F(2)), (F(-2), F(-2))),
                                  ((F(0), F(0)), (F(0), F(0))),
                                  ((F(-2), F(-2)), (F(2), F(2)))))
        self.assertTrue(all(arms[0][0] + arms[1][0] == 0 for arms in result))

    def test_bootstrap_matches_selected_means_on_wide_intervals(self):
        rows = [stats.FamilySample("a", (F(-3), F(2)), (F(1), F(5))),
                stats.FamilySample("b", (F(4), F(7)), (F(-5), F(-1)))]
        self.assertEqual(stats.bootstrap_means(rows, ((0, 1), (1, 1))),
                         (((F(1, 2), F(9, 2)), (F(-2), F(2))),
                          ((F(4), F(7)), (F(-5), F(-1)))))


if __name__ == "__main__":
    unittest.main()
