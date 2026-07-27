"""Monte Carlo study of noisy central differences for exp'(0).

Research model
--------------
Each function evaluation receives additive, paired noise.  The two noises have
standard deviation ``sigma`` and correlation ``rho``.  A single derivative
estimate averages ``n_samples`` noisy central differences.

The research-core functions are learner-owned.  The surrounding orchestration
writes reproducible tabular data, metadata, and a comparison plot.
"""

import csv
import json
import math
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXACT_DERIVATIVE = 1.0
SIGMA = 1e-3
RHO = 0.0
N_SAMPLES = 100
N_TRIALS = 2_000
SEED = 20_260_728
H_VALUES = tuple(10.0 ** (-3.0 + 3.0 * index / 40.0) for index in range(41))

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CSV_PATH = RESULTS_DIR / "statistical_noise_comparison.csv"
METADATA_PATH = RESULTS_DIR / "statistical_noise_metadata.json"
PLOT_PATH = RESULTS_DIR / "statistical_noise_error.png"


def correlated_noise_pair(
    sigma: float,
    rho: float,
    rng: random.Random,
) -> tuple[float, float]:
    """Return two zero-mean noises with std ``sigma`` and correlation ``rho``.

    Invariants to preserve:
    - E[epsilon_plus] = E[epsilon_minus] = 0
    - Var(epsilon_plus) = Var(epsilon_minus) = sigma**2
    - Corr(epsilon_plus, epsilon_minus) = rho

    Use two independent standard-normal draws from ``rng.gauss(0.0, 1.0)``.
    """
    Z1 = rng.gauss(0.0, 1.0)
    Z2 = rng.gauss(0.0, 1.0)
    epsilon_plus = sigma * Z1
    epsilon_minus = sigma * (rho * Z1 + math.sqrt(1 - rho**2) * Z2)
    return epsilon_plus, epsilon_minus


def mean_noisy_central_difference(
    h: float,
    sigma: float,
    rho: float,
    n_samples: int,
    rng: random.Random,
) -> float:
    """Return one estimate formed by averaging ``n_samples`` noisy differences.

    For every sample, add one correlated noise pair to f(h) and f(-h), form
    the central difference, and finally average the N resulting estimates.
    """
    if sigma < 0.0:
        raise ValueError("sigma must be non-negative")
    if not -1.0 <= rho <= 1.0:
        raise ValueError("rho must be in [-1, 1]")
    if h == 0.0:
        raise ValueError("h must be non-zero")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    total = 0.0
    for _ in range(n_samples):
        epsilon_plus, epsilon_minus = correlated_noise_pair(sigma, rho, rng)
        D_i = (
            math.exp(h)
            + epsilon_plus
            - (math.exp(-h) + epsilon_minus)
        ) / (2 * h)
        total += D_i
    return total / n_samples


def exact_central_bias(h: float) -> float:
    """Return the deterministic central-difference bias for exp'(0)."""
    return math.sinh(h) / h - EXACT_DERIVATIVE


def repeated_estimates(
    h: float,
    sigma: float,
    rho: float,
    n_samples: int,
    n_trials: int,
    seed: int,
) -> list[float]:
    """Repeat the complete N-sample estimator ``n_trials`` times.

    Create one seeded random-number generator before the M-trial loop.  Do not
    re-seed inside the loop, or every trial will repeat the same random stream.
    """
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")

    rng = random.Random(seed)
    estimates = []
    for _ in range(n_trials):
        D_m = mean_noisy_central_difference(h, sigma, rho, n_samples, rng)
        estimates.append(D_m)
    return estimates


def empirical_metrics(
    estimates: list[float],
    target: float = EXACT_DERIVATIVE,
) -> dict[str, float]:
    """Return empirical mean, bias, variance, and RMSE over repeated trials.

    Use denominator M for the empirical variance so that the finite dataset
    obeys RMSE**2 == bias**2 + variance up to floating-point roundoff.
    """
    if not estimates:
        raise ValueError("estimates must not be empty")

    D_bar = sum(estimates) / len(estimates)
    bias = D_bar - target
    variance = sum((D - D_bar) ** 2 for D in estimates) / len(estimates)
    rmse = math.sqrt(sum((D - target) ** 2 for D in estimates) / len(estimates))
    return {
        "mean": D_bar,
        "bias": bias,
        "variance": variance,
        "rmse": rmse,
    }


def theoretical_metrics(
    h: float,
    sigma: float,
    rho: float,
    n_samples: int,
) -> dict[str, float]:
    """Return the theoretical bias, variance, and RMSE for this model."""
    if sigma < 0.0:
        raise ValueError("sigma must be non-negative")
    if not -1.0 <= rho <= 1.0:
        raise ValueError("rho must be in [-1, 1]")
    if h == 0.0:
        raise ValueError("h must be non-zero")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    b_h = exact_central_bias(h)
    V = sigma**2 * (1 - rho) / (2 * n_samples * h**2)
    RMSE_theory = math.sqrt(b_h**2 + V)
    return {
        "bias": b_h,
        "variance": V,
        "rmse": RMSE_theory,
    }


CSV_FIELDS = [
    "h",
    "sigma",
    "rho",
    "n_samples",
    "n_trials",
    "seed",
    "empirical_mean",
    "empirical_bias",
    "empirical_variance",
    "empirical_random_std",
    "empirical_rmse",
    "theoretical_bias",
    "theoretical_variance",
    "theoretical_random_std",
    "theoretical_rmse",
]


def run_experiment() -> list[dict[str, float | int]]:
    """Sweep h and compare Monte Carlo metrics with the theoretical model."""
    rows: list[dict[str, float | int]] = []

    for index, h in enumerate(H_VALUES):
        configuration_seed = SEED + index
        estimates = repeated_estimates(
            h=h,
            sigma=SIGMA,
            rho=RHO,
            n_samples=N_SAMPLES,
            n_trials=N_TRIALS,
            seed=configuration_seed,
        )
        empirical = empirical_metrics(estimates)
        theoretical = theoretical_metrics(h, SIGMA, RHO, N_SAMPLES)

        rows.append(
            {
                "h": h,
                "sigma": SIGMA,
                "rho": RHO,
                "n_samples": N_SAMPLES,
                "n_trials": N_TRIALS,
                "seed": configuration_seed,
                "empirical_mean": empirical["mean"],
                "empirical_bias": empirical["bias"],
                "empirical_variance": empirical["variance"],
                "empirical_random_std": math.sqrt(empirical["variance"]),
                "empirical_rmse": empirical["rmse"],
                "theoretical_bias": theoretical["bias"],
                "theoretical_variance": theoretical["variance"],
                "theoretical_random_std": math.sqrt(theoretical["variance"]),
                "theoretical_rmse": theoretical["rmse"],
            }
        )

    return rows


def predicted_optimal_h() -> float:
    """Return the leading-order optimum for central differences of exp at 0."""
    leading_bias_coefficient = 1.0 / 6.0
    numerator = SIGMA**2 * (1.0 - RHO)
    denominator = 4.0 * N_SAMPLES * leading_bias_coefficient**2
    return (numerator / denominator) ** (1.0 / 6.0)


def write_csv(rows: list[dict[str, float | int]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_metadata() -> None:
    metadata = {
        "experiment": "Monte Carlo error model for noisy central differences",
        "target": "derivative of exp(x) at x = 0",
        "exact_derivative": EXACT_DERIVATIVE,
        "noise_model": "additive paired Gaussian function-value noise",
        "sigma": SIGMA,
        "rho": RHO,
        "n_samples_per_estimate": N_SAMPLES,
        "n_trials_per_h": N_TRIALS,
        "base_seed": SEED,
        "configuration_seed": "base_seed + h_index",
        "h_min": min(H_VALUES),
        "h_max": max(H_VALUES),
        "h_count": len(H_VALUES),
        "h_spacing": "logarithmic",
        "theoretical_bias": "sinh(h) / h - 1",
        "theoretical_variance": "sigma^2 * (1-rho) / (2*N*h^2)",
        "float_format": "Python float (IEEE 754 binary64)",
        "machine_epsilon": sys.float_info.epsilon,
        "python_version": sys.version.split()[0],
        "matplotlib_version": matplotlib.__version__,
    }
    METADATA_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def plot_metrics(rows: list[dict[str, float | int]]) -> None:
    h_values = [float(row["h"]) for row in rows]
    empirical_rmse = [float(row["empirical_rmse"]) for row in rows]
    theoretical_rmse = [float(row["theoretical_rmse"]) for row in rows]
    absolute_bias = [abs(float(row["theoretical_bias"])) for row in rows]
    random_std = [float(row["theoretical_random_std"]) for row in rows]

    figure, axis = plt.subplots(figsize=(10.5, 6.4), layout="constrained")
    axis.plot(
        h_values,
        empirical_rmse,
        color="#111827",
        marker="o",
        markersize=4.0,
        markevery=2,
        linewidth=0.0,
        label="Empirical RMSE",
        zorder=4,
    )
    axis.plot(
        h_values,
        theoretical_rmse,
        color="#111827",
        linewidth=2.0,
        label="Theoretical RMSE",
        zorder=3,
    )
    axis.plot(
        h_values,
        absolute_bias,
        color="#d97706",
        linestyle="--",
        linewidth=1.8,
        label="Absolute truncation bias",
    )
    axis.plot(
        h_values,
        random_std,
        color="#2563eb",
        linestyle="--",
        linewidth=1.8,
        label="Random standard deviation",
    )

    h_star = predicted_optimal_h()
    axis.axvline(
        h_star,
        color="#6b7280",
        linestyle=":",
        linewidth=1.3,
        label=rf"Leading-order optimum $h_*={h_star:.3g}$",
    )

    empirical_best = min(rows, key=lambda row: float(row["empirical_rmse"]))
    axis.scatter(
        [float(empirical_best["h"])],
        [float(empirical_best["empirical_rmse"])],
        color="#dc2626",
        marker="*",
        s=110,
        label="Empirical grid minimum",
        zorder=5,
    )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Step size h")
    axis.set_ylabel("Error scale")
    axis.set_title(
        "Noisy central difference for exp'(0)\n"
        f"sigma={SIGMA:g}, N={N_SAMPLES}, rho={RHO:g}, M={N_TRIALS}"
    )
    axis.grid(which="major", color="#d1d5db", linewidth=0.7)
    axis.grid(which="minor", color="#e5e7eb", linewidth=0.45, alpha=0.65)
    axis.legend(loc="best", frameon=False)
    figure.savefig(PLOT_PATH, dpi=200)
    plt.close(figure)


def print_summary(rows: list[dict[str, float | int]]) -> None:
    empirical_best = min(rows, key=lambda row: float(row["empirical_rmse"]))
    theoretical_best = min(rows, key=lambda row: float(row["theoretical_rmse"]))
    print(f"Wrote {len(rows)} rows to {CSV_PATH}")
    print(f"Wrote metadata to {METADATA_PATH}")
    print(f"Wrote plot to {PLOT_PATH}")
    print(f"Leading-order optimum h={predicted_optimal_h():.8e}")
    print(
        "Empirical grid optimum "
        f"h={float(empirical_best['h']):.8e}, "
        f"RMSE={float(empirical_best['empirical_rmse']):.8e}"
    )
    print(
        "Theoretical grid optimum "
        f"h={float(theoretical_best['h']):.8e}, "
        f"RMSE={float(theoretical_best['theoretical_rmse']):.8e}"
    )


def main() -> None:
    rows = run_experiment()
    write_csv(rows)
    write_metadata()
    plot_metrics(rows)
    print_summary(rows)


if __name__ == "__main__":
    main()
