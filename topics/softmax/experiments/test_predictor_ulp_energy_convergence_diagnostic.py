"""Tests for the ULP-energy convergence diagnostic."""

import unittest
from fractions import Fraction

from predictor_ulp_energy_convergence_diagnostic import (
    _metrics,
    _quantile,
    _safe_ratio,
)
from summation_graph_predictor import (
    balanced_reduction_graph,
    sequential_reduction_graph,
)


class UlpEnergyConvergenceDiagnosticTests(unittest.TestCase):
    def test_balanced_power_of_two_has_geometric_ulp_energy(self) -> None:
        values = tuple(Fraction(1) for _ in range(8))

        metrics = _metrics(values, balanced_reduction_graph(8), "balanced")

        self.assertAlmostEqual(metrics.q_ulp, 1.75)
        self.assertEqual(metrics.normalized_error, 0.0)

    def test_sequential_energy_exceeds_balanced_energy(self) -> None:
        values = tuple(Fraction(1) for _ in range(8))

        balanced = _metrics(values, balanced_reduction_graph(8), "balanced")
        sequential = _metrics(values, sequential_reduction_graph(8), "sequential")

        self.assertGreater(sequential.q_ulp, balanced.q_ulp)

    def test_quantile_interpolates(self) -> None:
        self.assertEqual(_quantile([0.0, 10.0], 0.5), 5.0)
        self.assertEqual(_quantile([3.0], 0.9), 3.0)

    def test_quantile_rejects_empty_input(self) -> None:
        with self.assertRaises(ValueError):
            _quantile([], 0.5)

    def test_safe_ratio_handles_zero_denominator(self) -> None:
        self.assertEqual(_safe_ratio(3.0, 2.0), 1.5)
        self.assertIsNone(_safe_ratio(3.0, 0.0))


if __name__ == "__main__":
    unittest.main()
