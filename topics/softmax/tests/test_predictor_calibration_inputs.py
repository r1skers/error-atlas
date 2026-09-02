"""Tests for calibration-only irregular stored-FP32 input generators."""

import unittest

from predictor_calibration_inputs import (
    calibration_input_families,
    head_tail_random,
    same_scale_random,
    wide_range_random,
)


class CalibrationInputGeneratorTests(unittest.TestCase):
    def test_same_seed_is_reproducible(self) -> None:
        first = calibration_input_families(64, seed=12345)
        second = calibration_input_families(64, seed=12345)
        self.assertEqual(first, second)

    def test_different_seeds_change_values(self) -> None:
        first = head_tail_random(64, seed=1)
        second = head_tail_random(64, seed=2)
        self.assertNotEqual(first.values, second.values)

    def test_all_values_are_positive_and_irregular(self) -> None:
        for generated in calibration_input_families(128, seed=9):
            self.assertEqual(len(generated.values), 128)
            self.assertTrue(all(value > 0 for value in generated.values))
            self.assertGreater(len(set(generated.values)), 100)

    def test_head_tail_contains_one_clear_dominant_value(self) -> None:
        generated = head_tail_random(128, seed=7)
        ordered = sorted(generated.values, reverse=True)
        self.assertGreater(ordered[0], ordered[1] * 100)

    def test_same_scale_stays_within_narrow_dynamic_range(self) -> None:
        generated = same_scale_random(128, seed=11)
        ratio = max(generated.values) / min(generated.values)
        self.assertLess(ratio, 8)

    def test_wide_range_spans_many_orders_of_two(self) -> None:
        generated = wide_range_random(256, seed=13)
        ratio = max(generated.values) / min(generated.values)
        self.assertGreater(ratio, 2**20)

    def test_rejects_invalid_width_or_seed(self) -> None:
        with self.assertRaises(ValueError):
            head_tail_random(1, seed=0)
        with self.assertRaises(TypeError):
            same_scale_random(64.0, seed=0)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            wide_range_random(64, seed=True)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
