import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXACT_DERIVATIVE = 1.0
K_VALUES = range(1, 56)
RESULTS_DIR = Path(__file__).resolve().parent / "results"
CSV_PATH = RESULTS_DIR / "finite_difference_comparison.csv"
METADATA_PATH = RESULTS_DIR / "finite_difference_metadata.json"
PLOT_PATH = RESULTS_DIR / "finite_difference_error.png"


def forward_difference(h: float) -> float:
    """用前向差分逼近 exp'(0)。"""
    A_h = (math.exp(h) - math.exp(0)) / h
    return A_h


def observed_order(
    a_h: float,
    a_h2: float,
    a_h4: float,
) -> float:
    """用三个尺度的近似值估计收敛阶。"""
    coarse_difference = abs(a_h - a_h2)
    fine_difference = abs(a_h2 - a_h4)

    if coarse_difference == 0.0 or fine_difference == 0.0:
        return math.nan

    return math.log2(coarse_difference / fine_difference)


def richardson(
    a_coarse: float,
    a_fine: float,
    p: float,
) -> float:
    """用粗、细两个结果和收敛阶进行外推。(收敛阶：p)"""
    scale = 2.0**p
    denominator = scale - 1.0

    if not math.isfinite(scale) or math.isclose(denominator, 0.0, abs_tol=1e-15):
        return math.nan

    return (scale * a_fine - a_coarse) / denominator


def estimated_error(
    a_coarse: float,
    a_fine: float,
    p: float,
) -> float:
    """估计细尺度结果与真值之间的误差。(收敛阶：p)"""
    scale = 2.0**p
    denominator = scale - 1.0

    if not math.isfinite(scale) or denominator <= 0.0:
        return math.nan

    return abs(a_fine - a_coarse) / denominator

def forward_difference_stable(h: float) -> float:
    """用前向稳定差分逼近 exp'(0)。"""
    return (math.expm1(h)) / h  # 使用 expm1 提高数值稳定性


def central_difference_naive(h: float) -> float:
    """用中心朴素差分逼近 exp'(0)。"""
    return (math.exp(h) - math.exp(-h)) / (2 * h)


def central_difference_stable(h: float) -> float:
    """用中心稳定差分逼近 exp'(0)。"""
    return math.sinh(h) / h


METHODS = {
    "forward_naive": forward_difference,
    "forward_stable": forward_difference_stable,
    "central_naive": central_difference_naive,
    "central_stable": central_difference_stable,
}

METHOD_ORDERS = {
    "forward_naive": 1,
    "forward_stable": 1,
    "central_naive": 2,
    "central_stable": 2,
}

METHOD_LABELS = {
    "forward_naive": "Forward, naive",
    "forward_stable": "Forward, stable",
    "central_naive": "Central, naive",
    "central_stable": "Central, stable",
}

CSV_FIELDS = [
    "method",
    "theoretical_order",
    "k",
    "h_coarse",
    "h_fine",
    "approximation",
    "true_error",
    "p_true",
    "p_observed",
    "estimated_error",
    "richardson",
    "richardson_true_error",
    "status",
]


def measured_order(coarse_error: float, fine_error: float) -> float:
    """用两个已知真误差测量收敛阶；零误差时无法在对数尺度定义。"""
    if coarse_error == 0.0 or fine_error == 0.0:
        return math.nan
    return math.log2(coarse_error / fine_error)


def run_experiment() -> list[dict[str, float | int | str]]:
    """运行四种差分实现，并保留误差分析所需的完整逐尺度记录。"""
    rows: list[dict[str, float | int | str]] = []

    for k in K_VALUES:
        h = 2.0 ** (-k)

        for method_name, method in METHODS.items():
            a_h = method(h)
            a_h2 = method(h / 2)
            a_h4 = method(h / 4)

            h_fine = h / 4
            half_true_error = abs(a_h2 - EXACT_DERIVATIVE)
            fine_true_error = abs(a_h4 - EXACT_DERIVATIVE)
            p_true = measured_order(half_true_error, fine_true_error)
            p_observed = observed_order(a_h, a_h2, a_h4)
            valid_order = math.isfinite(p_observed) and p_observed > 0.0

            if valid_order:
                a_r = richardson(a_h2, a_h4, p_observed)
                eta = estimated_error(a_h2, a_h4, p_observed)
                richardson_true_error = abs(a_r - EXACT_DERIVATIVE)
                status = "ok"
            else:
                a_r = math.nan
                eta = math.nan
                richardson_true_error = math.nan
                status = "roundoff-dominated"

            rows.append(
                {
                    "method": method_name,
                    "theoretical_order": METHOD_ORDERS[method_name],
                    "k": k,
                    "h_coarse": h,
                    "h_fine": h_fine,
                    "approximation": a_h4,
                    "true_error": fine_true_error,
                    "p_true": p_true,
                    "p_observed": p_observed,
                    "estimated_error": eta,
                    "richardson": a_r,
                    "richardson_true_error": richardson_true_error,
                    "status": status,
                }
            )

    return rows


def write_csv(rows: list[dict[str, float | int | str]]) -> None:
    """写出便于复查和后续绘图的原始实验表。"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: ""
                    if isinstance(value, float) and not math.isfinite(value)
                    else value
                    for key, value in row.items()
                }
            )


def write_metadata() -> None:
    """记录实验定义和运行环境，避免图表脱离计算上下文。"""
    metadata = {
        "experiment": "finite-difference error comparison",
        "target": "derivative of exp(x) at x = 0",
        "exact_derivative": EXACT_DERIVATIVE,
        "float_format": "Python float (IEEE 754 binary64)",
        "machine_epsilon": sys.float_info.epsilon,
        "k_values": [K_VALUES.start, K_VALUES.stop - 1],
        "h_coarse": "2**(-k)",
        "h_fine": "h_coarse / 4",
        "methods": {
            name: {
                "label": METHOD_LABELS[name],
                "theoretical_order": METHOD_ORDERS[name],
            }
            for name in METHODS
        },
        "zero_error_policy": "kept in CSV; omitted from logarithmic plot",
        "python_version": sys.version.split()[0],
        "matplotlib_version": matplotlib.__version__,
    }

    METADATA_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def plot_errors(rows: list[dict[str, float | int | str]]) -> None:
    """绘制四种实现的双对数总误差曲线。"""
    styles = {
        "forward_naive": {"color": "#1f77b4", "linestyle": "--", "marker": "x"},
        "forward_stable": {"color": "#1f77b4", "linestyle": "-", "marker": "o"},
        "central_naive": {"color": "#d97706", "linestyle": "--", "marker": "x"},
        "central_stable": {"color": "#d97706", "linestyle": "-", "marker": "o"},
    }

    figure, axis = plt.subplots(figsize=(10.5, 6.4), layout="constrained")

    for method_name in METHODS:
        method_rows = sorted(
            (row for row in rows if row["method"] == method_name),
            key=lambda row: float(row["h_fine"]),
        )
        h_values = [float(row["h_fine"]) for row in method_rows]
        error_values = [
            float(row["true_error"]) if float(row["true_error"]) > 0.0 else math.nan
            for row in method_rows
        ]
        axis.plot(
            h_values,
            error_values,
            label=METHOD_LABELS[method_name],
            linewidth=1.8,
            markersize=4.0,
            markevery=4,
            **styles[method_name],
        )

    reference_h = [2.0 ** (-k) for k in range(4, 19)]
    axis.plot(
        reference_h,
        [h / 2 for h in reference_h],
        color="#6b7280",
        linewidth=1.1,
        linestyle=":",
        label=r"Forward leading term $h/2$",
    )
    axis.plot(
        reference_h,
        [h**2 / 6 for h in reference_h],
        color="#374151",
        linewidth=1.1,
        linestyle=":",
        label=r"Central leading term $h^2/6$",
    )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Fine step size h")
    axis.set_ylabel("Absolute total error")
    axis.set_title(
        "Finite-difference error for exp'(0)\n"
        "Zero-valued binary64 errors are omitted from the logarithmic scale"
    )
    axis.grid(which="major", color="#d1d5db", linewidth=0.7)
    axis.grid(which="minor", color="#e5e7eb", linewidth=0.45, alpha=0.65)
    axis.legend(loc="best", frameon=False, ncols=2)
    figure.savefig(PLOT_PATH, dpi=200)
    plt.close(figure)


def print_summary(rows: list[dict[str, float | int | str]]) -> None:
    print(f"Wrote {len(rows)} rows to {CSV_PATH}")
    print(f"Wrote metadata to {METADATA_PATH}")
    print(f"Wrote plot to {PLOT_PATH}")

    for method_name in METHODS:
        method_rows = [row for row in rows if row["method"] == method_name]
        positive_rows = [row for row in method_rows if float(row["true_error"]) > 0.0]
        best_row = min(positive_rows, key=lambda row: float(row["true_error"]))
        zero_count = len(method_rows) - len(positive_rows)
        print(
            f"{method_name}: minimum positive error={float(best_row['true_error']):.5e} "
            f"at h={float(best_row['h_fine']):.5e}; zero-error rows={zero_count}"
        )


def main() -> None:
    rows = run_experiment()
    write_csv(rows)
    write_metadata()
    plot_errors(rows)
    print_summary(rows)



if __name__ == "__main__":
    main()
