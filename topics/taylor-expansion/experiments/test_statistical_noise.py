import math
import random
import unittest

from statistical_noise import (
    correlated_noise_pair,
    empirical_metrics,
    mean_noisy_central_difference,
    repeated_estimates,
    theoretical_metrics,
)


class CorrelatedNoiseTests(unittest.TestCase):
    def test_rho_one_produces_common_mode_noise(self) -> None:
        rng = random.Random(11)
        for _ in range(20):
            epsilon_plus, epsilon_minus = correlated_noise_pair(0.3, 1.0, rng)
            self.assertEqual(epsilon_plus, epsilon_minus)

    def test_rho_minus_one_produces_opposite_noise(self) -> None:
        rng = random.Random(12)
        for _ in range(20):
            epsilon_plus, epsilon_minus = correlated_noise_pair(0.3, -1.0, rng)
            self.assertEqual(epsilon_plus, -epsilon_minus)


class EstimatorTests(unittest.TestCase):
    def test_same_seed_reproduces_all_trials(self) -> None:
        arguments = {
            "h": 0.02,
            "sigma": 1e-3,
            "rho": 0.25,
            "n_samples": 20,
            "n_trials": 30,
            "seed": 1234,
        }
        self.assertEqual(repeated_estimates(**arguments), repeated_estimates(**arguments))

    def test_zero_noise_matches_deterministic_central_difference(self) -> None:
        h = 0.02
        estimate = mean_noisy_central_difference(
            h=h,
            sigma=0.0,
            rho=0.0,
            n_samples=10,
            rng=random.Random(1),
        )
        self.assertAlmostEqual(estimate, math.sinh(h) / h, places=14)

    def test_empirical_mse_decomposition(self) -> None:
        metrics = empirical_metrics([0.8, 1.1, 1.2])
        self.assertAlmostEqual(
            metrics["rmse"] ** 2,
            metrics["bias"] ** 2 + metrics["variance"],
            places=15,
        )

    def test_empirical_variance_matches_theory_for_fixed_seed(self) -> None:
        h = 0.02
        sigma = 1e-3
        rho = 0.25
        n_samples = 40
        estimates = repeated_estimates(
            h=h,
            sigma=sigma,
            rho=rho,
            n_samples=n_samples,
            n_trials=2_000,
            seed=20260728,
        )
        empirical = empirical_metrics(estimates)
        theoretical = theoretical_metrics(h, sigma, rho, n_samples)
        relative_gap = abs(
            empirical["variance"] / theoretical["variance"] - 1.0
        )
        self.assertLess(relative_gap, 0.08)

    def test_invalid_collection_sizes_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            repeated_estimates(0.1, 1e-3, 0.0, 10, 0, 1)
        with self.assertRaises(ValueError):
            empirical_metrics([])

    def test_invalid_theoretical_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            theoretical_metrics(0.0, 1e-3, 0.0, 10)
        with self.assertRaises(ValueError):
            theoretical_metrics(0.1, -1e-3, 0.0, 10)
        with self.assertRaises(ValueError):
            theoretical_metrics(0.1, 1e-3, 1.1, 10)
        with self.assertRaises(ValueError):
            theoretical_metrics(0.1, 1e-3, 0.0, 0)


if __name__ == "__main__":
    unittest.main()
