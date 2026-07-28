"""Closed-book rewrite exercise for correlated Gaussian noise."""

import random


def correlated_noise_pair(
    sigma: float,
    rho: float,
    rng: random.Random,
) -> tuple[float, float]:
    """Return a zero-mean Gaussian noise pair.

    Required invariants:
    - both marginal variances equal sigma**2;
    - the correlation coefficient equals rho;
    - exactly two independent standard-normal draws are consumed per call;
    - rho=1 gives common-mode noise;
    - rho=-1 gives opposite noise;
    - sigma=0 gives (0, 0).

    Valid inputs satisfy sigma >= 0 and -1 <= rho <= 1.
    Do not import or call the original implementation.
    """
    if not (-1 <= rho <= 1):
        raise ValueError(f"rho={rho} is out of range [-1, 1]")
    if sigma < 0:
        raise ValueError(f"sigma={sigma} is negative")
    Z1 = rng.gauss(0, 1)
    Z2 = rng.gauss(0, 1)
    epsilon_plus = sigma * Z1
    epsilon_minus = sigma * (rho * Z1 + (1 - rho**2)**0.5 * Z2)
    return epsilon_plus, epsilon_minus
