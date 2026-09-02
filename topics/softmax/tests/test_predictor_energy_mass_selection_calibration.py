"""Tests for train-only energy-mass selection."""

import unittest

from predictor_energy_mass_selection_calibration import (
    MassSummary,
    _defined_mean,
    _select_mass,
)


def _summary(
    mass: float,
    *,
    hit: float = 1.0,
    regret: float = 0.0,
    pair: float = 0.8,
    nodes: float = 8.0,
) -> MassSummary:
    return MassSummary(mass, nodes, 1.0, hit, regret, pair, 0.5)


class EnergyMassSelectionCalibrationTests(unittest.TestCase):
    def test_selection_prioritizes_utility_before_cost(self) -> None:
        expensive_hit = _summary(0.9, hit=1.0, nodes=16.0)
        cheap_miss = _summary(0.7, hit=0.75, nodes=4.0)
        self.assertIs(_select_mass([cheap_miss, expensive_hit]), expensive_hit)

    def test_selection_uses_lower_cost_as_final_tie_break(self) -> None:
        expensive = _summary(0.9, nodes=16.0)
        cheap = _summary(0.7, nodes=6.0)
        self.assertIs(_select_mass([expensive, cheap]), cheap)

    def test_defined_mean_ignores_undefined_folds(self) -> None:
        self.assertEqual(_defined_mean([None, 1.0, 3.0]), 2.0)
        self.assertIsNone(_defined_mean([None]))

    def test_empty_selection_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _select_mass([])


if __name__ == "__main__":
    unittest.main()
