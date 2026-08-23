"""Calibration-only online risk-certificate diagnostic for balanced FP32 reduction.

The previous selector ranked many candidate trees and therefore remained expensive even after its
exact oracle was removed.  This experiment changes the object being predicted.  It executes one
fixed balanced tree and carries a small annotation beside every rounded subtree result:

* ``B`` is a worst-case sum of local half-ULP envelopes;
* ``Q`` is a local-rounding variance proxy;
* an optional four-gap term carries the already observed ancestor-history correlation kernel.

The cheap trace sees only nonnegative finite binary32 bit patterns.  Integer-aligned operand
addition is used to detect whether a hardware-style FP32 addition is exact, but the sign and
magnitude of every local residual are deliberately discarded.  ``Fraction`` is used only after the
trace to produce calibration labels and verify invariants.

The Gaussian model is empirical: across similar generated inputs, not physical randomness within
one deterministic reduction.  Its CDF is used to estimate interval coverage and the probability
that the exact stored-leaf sum lies in the computed root's rounding cell.  It is never called a
rigorous certificate; only ``B`` has that role under this experiment's stored-input/reduction-only
contract.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from statistics import mean

from predictor_calibration_inputs import wide_range_random
from predictor_fixed_k8_beam_inference import (
    FP32_MAX_FINITE_BITS,
    FP32_MIN_SUBNORMAL_EXPONENT,
    _bits_to_float,
    _bits_to_units,
    _round_units_to_fp32_bits,
    _ulp_exponent_from_units,
)
from predictor_wide_range_fixed_k8_beam_v2_heldout import (
    _bits_hash,
    _git_state,
    _json_dump,
    _reserve_output_directory,
    _sha256,
    _stored_leaf_bits,
)
from summation_graph_predictor import (
    BinaryReductionGraph,
    balanced_reduction_graph,
    predict_fp32_tree_error,
    round_nonnegative_fraction_to_fp32,
)


EXPERIMENT_ID = "wide_range_online_risk_certificate_calibration_v1"
HERE = Path(__file__).resolve().parent
RESULT_ROOT = HERE / "results" / EXPERIMENT_ID
PREREGISTRATION = RESULT_ROOT / f"{EXPERIMENT_ID}_preregistration.json"
OUTPUT_DIRECTORY = RESULT_ROOT / "calibration"

EXPECTED_WIDTHS = (256, 512, 1024)
EXPECTED_GROUPS_PER_WIDTH = 64
EXPECTED_FOLDS = 5
TOP_ENERGY_COUNT = 8
CORRELATION_KERNEL = (0.476, 0.267, 0.168, 0.055)
PROXIES = ("q_all", "q_inexact", "q_corr4_all", "q_corr4_inexact")
MODEL_VARIANTS = ("zero_mean", "bias_aware")
PRIMARY_PROXY = "q_inexact"
PRIMARY_VARIANT = "bias_aware"
NORMAL_COVERAGE_Z = {
    "90": 1.6448536269514722,
    "99": 2.5758293035489004,
    "99.9": 3.2905267314919255,
}
RELIABILITY_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


@dataclass(frozen=True)
class OnlineCertificateState:
    """One subtree result plus root-normalized online certificate statistics.

    ``layers_*[d]`` is the sum of local ULP scales at exact graph distance ``d``
    from this subtree root, normalized by this state's output ULP.  Four entries
    are sufficient for the frozen four-gap ancestor kernel.
    """

    bits: int
    ulp_exponent: int
    q_all: float
    q_inexact: float
    q_corr4_all: float
    q_corr4_inexact: float
    b_all: float
    b_inexact: float
    layers_all: tuple[float, float, float, float]
    layers_inexact: tuple[float, float, float, float]
    top_energy_all: tuple[float, ...]
    top_energy_inexact: tuple[float, ...]
    internal_count: int
    inexact_count: int


@dataclass(frozen=True)
class CrossFitPrediction:
    mu: float
    sigma: float
    z: float
    p_safe: float
    prevalence_baseline: float


def _derived_seed(width: int, index: int) -> int:
    if width not in EXPECTED_WIDTHS:
        raise ValueError("unexpected width")
    if not 0 <= index < EXPECTED_GROUPS_PER_WIDTH:
        raise ValueError("unexpected input index")
    payload = f"{EXPERIMENT_ID}|calibration|{width}|{index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


def _power_of_two(exponent: int) -> Fraction:
    if exponent >= 0:
        return Fraction(1 << exponent)
    return Fraction(1, 1 << (-exponent))


def _outward_positive_sum(*terms: float) -> float:
    """Add nonnegative binary64 terms with an outward step after each addition."""
    total = 0.0
    for term in terms:
        if not math.isfinite(term) or term < 0.0:
            raise ValueError("bound terms must be finite and nonnegative")
        if term:
            total = math.nextafter(total + term, math.inf)
    return total


def _top_energy(terms: list[float]) -> tuple[float, ...]:
    return tuple(sorted(terms, reverse=True)[:TOP_ENERGY_COUNT])


def _leaf_state(bits: int) -> OnlineCertificateState:
    units = _bits_to_units(bits)
    if units <= 0:
        raise ValueError("the frozen diagnostic requires positive leaves")
    exponent = _ulp_exponent_from_units(units)
    zeros = (0.0, 0.0, 0.0, 0.0)
    return OnlineCertificateState(
        bits=bits,
        ulp_exponent=exponent,
        q_all=0.0,
        q_inexact=0.0,
        q_corr4_all=0.0,
        q_corr4_inexact=0.0,
        b_all=0.0,
        b_inexact=0.0,
        layers_all=zeros,
        layers_inexact=zeros,
        top_energy_all=(),
        top_energy_inexact=(),
        internal_count=0,
        inexact_count=0,
    )


def _merge_states(
    left: OnlineCertificateState,
    right: OnlineCertificateState,
) -> OnlineCertificateState:
    """Execute one FP32 add and merge the frozen certificate state."""
    exact_addend_units = _bits_to_units(left.bits) + _bits_to_units(right.bits)
    rounded_bits = _round_units_to_fp32_bits(exact_addend_units)
    rounded_units = _bits_to_units(rounded_bits)
    if rounded_units <= 0:
        raise AssertionError("positive addition produced a nonpositive result")

    output_exponent = _ulp_exponent_from_units(rounded_units)
    local_exponent = _ulp_exponent_from_units(exact_addend_units)
    left_scale = math.ldexp(1.0, left.ulp_exponent - output_exponent)
    right_scale = math.ldexp(1.0, right.ulp_exponent - output_exponent)
    local_scale = math.ldexp(1.0, local_exponent - output_exponent)
    left_energy_scale = left_scale * left_scale
    right_energy_scale = right_scale * right_scale
    local_energy = local_scale * local_scale
    local_variance = local_energy / 12.0
    inexact = rounded_units != exact_addend_units

    q_all = (
        left.q_all * left_energy_scale
        + right.q_all * right_energy_scale
        + local_variance
    )
    q_inexact = (
        left.q_inexact * left_energy_scale
        + right.q_inexact * right_energy_scale
        + (local_variance if inexact else 0.0)
    )

    descendants_all = tuple(
        left.layers_all[gap] * left_scale
        + right.layers_all[gap] * right_scale
        for gap in range(4)
    )
    descendants_inexact = tuple(
        left.layers_inexact[gap] * left_scale
        + right.layers_inexact[gap] * right_scale
        for gap in range(4)
    )
    corr_all = sum(
        2.0 * rho * local_scale * descendants_all[gap] / 12.0
        for gap, rho in enumerate(CORRELATION_KERNEL)
    )
    corr_inexact = (
        sum(
            2.0 * rho * local_scale * descendants_inexact[gap] / 12.0
            for gap, rho in enumerate(CORRELATION_KERNEL)
        )
        if inexact
        else 0.0
    )
    q_corr4_all = (
        left.q_corr4_all * left_energy_scale
        + right.q_corr4_all * right_energy_scale
        + local_variance
        + corr_all
    )
    q_corr4_inexact = (
        left.q_corr4_inexact * left_energy_scale
        + right.q_corr4_inexact * right_energy_scale
        + (local_variance if inexact else 0.0)
        + corr_inexact
    )

    b_all = _outward_positive_sum(
        math.ldexp(left.b_all, left.ulp_exponent - output_exponent),
        math.ldexp(right.b_all, right.ulp_exponent - output_exponent),
        0.5 * local_scale,
    )
    b_inexact = _outward_positive_sum(
        math.ldexp(left.b_inexact, left.ulp_exponent - output_exponent),
        math.ldexp(right.b_inexact, right.ulp_exponent - output_exponent),
        0.5 * local_scale if inexact else 0.0,
    )

    layers_all = (
        local_scale,
        descendants_all[0],
        descendants_all[1],
        descendants_all[2],
    )
    layers_inexact = (
        local_scale if inexact else 0.0,
        descendants_inexact[0],
        descendants_inexact[1],
        descendants_inexact[2],
    )
    top_all = _top_energy(
        [value * left_energy_scale for value in left.top_energy_all]
        + [value * right_energy_scale for value in right.top_energy_all]
        + [local_energy]
    )
    active_terms = (
        [value * left_energy_scale for value in left.top_energy_inexact]
        + [value * right_energy_scale for value in right.top_energy_inexact]
    )
    if inexact:
        active_terms.append(local_energy)
    top_inexact = _top_energy(active_terms)

    return OnlineCertificateState(
        bits=rounded_bits,
        ulp_exponent=output_exponent,
        q_all=q_all,
        q_inexact=q_inexact,
        q_corr4_all=q_corr4_all,
        q_corr4_inexact=q_corr4_inexact,
        b_all=b_all,
        b_inexact=b_inexact,
        layers_all=layers_all,
        layers_inexact=layers_inexact,
        top_energy_all=top_all,
        top_energy_inexact=top_inexact,
        internal_count=left.internal_count + right.internal_count + 1,
        inexact_count=left.inexact_count + right.inexact_count + int(inexact),
    )


def trace_online_certificate(
    leaf_bits: tuple[int, ...],
    graph: BinaryReductionGraph,
) -> OnlineCertificateState:
    """Run one explicit graph while maintaining only the frozen online state."""
    if len(leaf_bits) != graph.leaf_count:
        raise ValueError("leaf count does not match graph")
    states = [_leaf_state(bits) for bits in leaf_bits]
    for node in graph.nodes:
        states.append(_merge_states(states[node.left], states[node.right]))
    return states[graph.root]


def _root_error_cell(bits: int, root_ulp_exponent: int) -> tuple[float, float]:
    """Return the open continuous-model interval of E=root-exact in root ULPs."""
    if not 0 < bits < FP32_MAX_FINITE_BITS:
        raise ValueError("cell diagnostic requires an interior positive finite result")
    root_units = _bits_to_units(bits)
    previous_units = _bits_to_units(bits - 1)
    next_units = _bits_to_units(bits + 1)
    ulp_units = 1 << (root_ulp_exponent - FP32_MIN_SUBNORMAL_EXPONENT)
    lower = -0.5 * (next_units - root_units) / ulp_units
    upper = 0.5 * (root_units - previous_units) / ulp_units
    if not lower < 0.0 < upper:
        raise AssertionError("invalid rounding cell")
    return lower, upper


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _normal_cell_probability(
    low: float,
    high: float,
    mu: float,
    sigma: float,
) -> float:
    if sigma <= 0.0 or not math.isfinite(sigma):
        return float(low < mu < high)
    probability = _normal_cdf((high - mu) / sigma) - _normal_cdf(
        (low - mu) / sigma
    )
    return min(1.0, max(0.0, probability))


def _percentile(values: list[float], probability: float) -> float:
    if not values or not 0.0 <= probability <= 1.0:
        raise ValueError("invalid percentile request")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _population_std(values: list[float]) -> float:
    center = mean(values)
    return math.sqrt(mean((value - center) ** 2 for value in values))


def _shape(values: list[float]) -> dict[str, float]:
    center = mean(values)
    std = _population_std(values)
    if std:
        skew = mean(((value - center) / std) ** 3 for value in values)
        excess = mean(((value - center) / std) ** 4 for value in values) - 3.0
    else:
        skew = 0.0
        excess = -3.0
    ordered = sorted(values)
    ks = 0.0
    count = len(ordered)
    for index, value in enumerate(ordered):
        cdf = _normal_cdf(value)
        ks = max(ks, cdf - index / count, (index + 1) / count - cdf)
    absolute = [abs(value) for value in values]
    return {
        "mean": center,
        "std": std,
        "skew": skew,
        "excess_kurtosis": excess,
        "ks_distance_standard_normal": ks,
        "abs_p90": _percentile(absolute, 0.90),
        "abs_p99": _percentile(absolute, 0.99),
        "abs_max": max(absolute),
    }


def _model_key(proxy: str, variant: str) -> str:
    return f"{proxy}__{variant}"


def _cross_fit(
    rows: list[dict],
    proxy: str,
    variant: str,
) -> tuple[list[CrossFitPrediction], list[dict]]:
    if proxy not in PROXIES or variant not in MODEL_VARIANTS:
        raise ValueError("unknown frozen model")
    predictions: list[CrossFitPrediction | None] = [None] * len(rows)
    fits = []
    for fold in range(EXPECTED_FOLDS):
        train_indices = [i for i, row in enumerate(rows) if row["fold"] != fold]
        test_indices = [i for i, row in enumerate(rows) if row["fold"] == fold]
        standardized = []
        for index in train_indices:
            q = float(rows[index][proxy])
            if q <= 0.0:
                raise ValueError(f"{proxy} must be positive for cross-fitting")
            standardized.append(
                float(rows[index]["signed_error_root_ulp"]) / math.sqrt(q)
            )
        beta = mean(standardized) if variant == "bias_aware" else 0.0
        scale = math.sqrt(mean((value - beta) ** 2 for value in standardized))
        if not scale > 0.0:
            raise ValueError("cross-fitted scale must be positive")
        prevalence = mean(bool(rows[index]["correctly_rounded"]) for index in train_indices)
        fits.append(
            {
                "fold": fold,
                "train_count": len(train_indices),
                "test_count": len(test_indices),
                "beta_standardized": beta,
                "scale_standardized": scale,
                "train_correct_prevalence": prevalence,
            }
        )
        for index in test_indices:
            feature = math.sqrt(float(rows[index][proxy]))
            mu = beta * feature
            sigma = scale * feature
            error = float(rows[index]["signed_error_root_ulp"])
            low = float(rows[index]["cell_error_low_root_ulp"])
            high = float(rows[index]["cell_error_high_root_ulp"])
            predictions[index] = CrossFitPrediction(
                mu=mu,
                sigma=sigma,
                z=(error - mu) / sigma,
                p_safe=_normal_cell_probability(low, high, mu, sigma),
                prevalence_baseline=prevalence,
            )
    if any(prediction is None for prediction in predictions):
        raise AssertionError("cross-fitting left an input without a prediction")
    return [prediction for prediction in predictions if prediction is not None], fits


def _reliability(rows: list[dict], predictions: list[CrossFitPrediction]) -> list[dict]:
    result = []
    for index in range(len(RELIABILITY_EDGES) - 1):
        low = RELIABILITY_EDGES[index]
        high = RELIABILITY_EDGES[index + 1]
        selected = [
            i
            for i, prediction in enumerate(predictions)
            if low <= prediction.p_safe < high
            or (index == len(RELIABILITY_EDGES) - 2 and prediction.p_safe == 1.0)
        ]
        result.append(
            {
                "low": low,
                "high": high,
                "count": len(selected),
                "mean_predicted": (
                    mean(predictions[i].p_safe for i in selected) if selected else None
                ),
                "observed_correct": (
                    mean(bool(rows[i]["correctly_rounded"]) for i in selected)
                    if selected
                    else None
                ),
            }
        )
    return result


def _model_summary(
    rows: list[dict],
    predictions: list[CrossFitPrediction],
    fits: list[dict],
) -> dict:
    z_values = [prediction.z for prediction in predictions]
    coverage = {
        label: mean(abs(prediction.z) <= cutoff for prediction in predictions)
        for label, cutoff in NORMAL_COVERAGE_Z.items()
    }
    per_width = {}
    for width in EXPECTED_WIDTHS:
        indices = [i for i, row in enumerate(rows) if row["width"] == width]
        per_width[str(width)] = {
            "count": len(indices),
            "z_shape": _shape([predictions[i].z for i in indices]),
            "coverage": {
                label: mean(abs(predictions[i].z) <= cutoff for i in indices)
                for label, cutoff in NORMAL_COVERAGE_Z.items()
            },
            "correct_prevalence": mean(
                bool(rows[i]["correctly_rounded"]) for i in indices
            ),
            "mean_p_safe": mean(predictions[i].p_safe for i in indices),
        }
    outcome = [float(bool(row["correctly_rounded"])) for row in rows]
    brier = mean(
        (prediction.p_safe - target) ** 2
        for prediction, target in zip(predictions, outcome, strict=True)
    )
    baseline_brier = mean(
        (prediction.prevalence_baseline - target) ** 2
        for prediction, target in zip(predictions, outcome, strict=True)
    )
    certification = {}
    for threshold in (0.90, 0.99):
        selected = [
            i for i, prediction in enumerate(predictions) if prediction.p_safe >= threshold
        ]
        certification[f"{threshold:.2f}"] = {
            "count": len(selected),
            "coverage": len(selected) / len(rows),
            "observed_precision": (
                mean(bool(rows[i]["correctly_rounded"]) for i in selected)
                if selected
                else None
            ),
        }
    return {
        "fold_fits": fits,
        "z_shape": _shape(z_values),
        "coverage": coverage,
        "per_width": per_width,
        "correct_prevalence": mean(outcome),
        "mean_p_safe": mean(prediction.p_safe for prediction in predictions),
        "brier_score": brier,
        "prevalence_baseline_brier_score": baseline_brier,
        "brier_improvement": baseline_brier - brier,
        "reliability": _reliability(rows, predictions),
        "certification": certification,
    }


def _finite_ratio(bound: float, error: float) -> float | None:
    return bound / error if error > 0.0 else None


def _bound_summary(rows: list[dict], field: str) -> dict:
    ratios = [
        ratio
        for row in rows
        if (
            ratio := _finite_ratio(
                float(row[field]), float(row["absolute_error_root_ulp"])
            )
        )
        is not None
    ]
    cell_cert_field = f"{field}_strict_cell_certificate"
    return {
        "coverage": mean(
            float(row["absolute_error_root_ulp"]) <= float(row[field])
            for row in rows
        ),
        "zero_error_count": sum(
            float(row["absolute_error_root_ulp"]) == 0.0 for row in rows
        ),
        "bound_to_abs_error_ratio": {
            "p50": _percentile(ratios, 0.50),
            "p90": _percentile(ratios, 0.90),
            "p99": _percentile(ratios, 0.99),
            "max": max(ratios),
        },
        "strict_rounding_cell_certificate_rate": mean(
            bool(row[cell_cert_field]) for row in rows
        ),
        "strict_rounding_cell_certificate_count": sum(
            bool(row[cell_cert_field]) for row in rows
        ),
    }


def _macro_summary(rows: list[dict]) -> dict:
    fields = (
        "inexact_fraction",
        "inexact_energy_fraction",
        "top8_energy_concentration",
        "top8_inexact_energy_concentration",
        "max_leaf_root_ratio",
        "effective_leaf_count",
    )
    result = {}
    for width in EXPECTED_WIDTHS:
        group = [row for row in rows if row["width"] == width]
        result[str(width)] = {
            field: {
                "mean": mean(float(row[field]) for row in group),
                "p10": _percentile([float(row[field]) for row in group], 0.10),
                "p50": _percentile([float(row[field]) for row in group], 0.50),
                "p90": _percentile([float(row[field]) for row in group], 0.90),
            }
            for field in fields
        }
    return result


def _evaluate_group(width: int, index: int) -> dict:
    seed = _derived_seed(width, index)
    generated = wide_range_random(width, seed=seed)
    values = generated.values
    bits = tuple(_stored_leaf_bits(values))
    graph = balanced_reduction_graph(width)
    state = trace_online_certificate(bits, graph)

    oracle = predict_fp32_tree_error(values, graph)
    oracle_bits = int(oracle.predicted_sum_bits, 16)
    if state.bits != oracle_bits:
        raise AssertionError("online root bits disagree with the exact graph oracle")
    root_ulp = _power_of_two(state.ulp_exponent)
    signed_error = float(oracle.signed_error / root_ulp)
    absolute_error = abs(signed_error)
    if absolute_error > state.b_all or absolute_error > state.b_inexact:
        raise AssertionError("online rigorous error envelope was violated")

    exact_sum = sum(values, start=Fraction(0))
    correct_bits = round_nonnegative_fraction_to_fp32(exact_sum).bits
    cell_low, cell_high = _root_error_cell(state.bits, state.ulp_exponent)
    strict_margin = min(-cell_low, cell_high)
    all_energy = state.q_all * 12.0
    inexact_energy = state.q_inexact * 12.0
    leaf_floats = [_bits_to_float(value) for value in bits]
    approximate_total = math.fsum(leaf_floats)
    square_total = math.fsum(value * value for value in leaf_floats)
    return {
        "schema_version": "1",
        "input_group_id": f"w{width}_g{index:03d}",
        "width": width,
        "input_index": index,
        "fold": index % EXPECTED_FOLDS,
        "seed": seed,
        "input_bits_sha256": _bits_hash(list(bits)),
        "computed_root_bits": f"0x{state.bits:08x}",
        "correct_root_bits": f"0x{correct_bits:08x}",
        "correctly_rounded": state.bits == correct_bits,
        "root_ulp_exponent": state.ulp_exponent,
        "signed_error_root_ulp": signed_error,
        "absolute_error_root_ulp": absolute_error,
        "cell_error_low_root_ulp": cell_low,
        "cell_error_high_root_ulp": cell_high,
        "q_all": state.q_all,
        "q_inexact": state.q_inexact,
        "q_corr4_all": state.q_corr4_all,
        "q_corr4_inexact": state.q_corr4_inexact,
        "b_all": state.b_all,
        "b_inexact": state.b_inexact,
        "b_all_strict_cell_certificate": state.b_all < strict_margin,
        "b_inexact_strict_cell_certificate": state.b_inexact < strict_margin,
        "addition_count": state.internal_count,
        "inexact_addition_count": state.inexact_count,
        "inexact_fraction": state.inexact_count / state.internal_count,
        "inexact_energy_fraction": inexact_energy / all_energy,
        "top8_energy_concentration": sum(state.top_energy_all) / all_energy,
        "top8_inexact_energy_concentration": (
            sum(state.top_energy_inexact) / inexact_energy
        ),
        "max_leaf_root_ratio": max(leaf_floats) / approximate_total,
        "effective_leaf_count": approximate_total * approximate_total / square_total,
    }


def _load_and_validate_preregistration() -> dict:
    with PREREGISTRATION.open(encoding="utf-8") as handle:
        config = json.load(handle)
    expected = {
        "experiment_id": EXPERIMENT_ID,
        "status": "frozen_calibration_not_opened",
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"frozen preregistration mismatch: {key}")
    boundary = config["data_boundary"]
    if boundary["widths"] != list(EXPECTED_WIDTHS):
        raise ValueError("frozen widths changed")
    if boundary["groups_per_width"] != EXPECTED_GROUPS_PER_WIDTH:
        raise ValueError("frozen group count changed")
    proxy = config["frozen_variance_proxies"]
    if proxy["primary"] != PRIMARY_PROXY:
        raise ValueError("frozen primary proxy changed")
    if tuple(proxy["comparators"]) != (
        "q_all",
        "q_corr4_all",
        "q_corr4_inexact",
    ):
        raise ValueError("frozen comparator set changed")
    recorded_kernel = config["online_state"]["correlation_kernel"]
    if tuple(recorded_kernel[str(i)] for i in range(1, 5)) != CORRELATION_KERNEL:
        raise ValueError("frozen correlation kernel changed")
    seeds = [
        _derived_seed(width, index)
        for width in EXPECTED_WIDTHS
        for index in range(EXPECTED_GROUPS_PER_WIDTH)
    ]
    if len(seeds) != len(set(seeds)):
        raise ValueError("derived calibration seeds collided")
    return config


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write an empty artifact")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _decision(summary: dict) -> dict:
    primary = summary["models"][_model_key(PRIMARY_PROXY, PRIMARY_VARIANT)]
    pooled = primary["coverage"]
    per_width = primary["per_width"]
    gaussian_viable = (
        pooled["90"] >= 0.88
        and pooled["99"] >= 0.98
        and primary["z_shape"]["abs_p99"] <= 3.5
        and all(record["coverage"]["90"] >= 0.85 for record in per_width.values())
        and all(record["coverage"]["99"] >= 0.95 for record in per_width.values())
    )
    p90 = primary["certification"]["0.90"]
    probability_useful = (
        primary["brier_improvement"] > 0.0
        and p90["coverage"] >= 0.10
        and p90["observed_precision"] is not None
        and p90["observed_precision"] >= 0.90
    )
    hard_valid = (
        summary["rigorous_bounds"]["b_all"]["coverage"] == 1.0
        and summary["rigorous_bounds"]["b_inexact"]["coverage"] == 1.0
    )
    return {
        "hard_invariants_pass": hard_valid,
        "primary_gaussian_viable": gaussian_viable,
        "primary_probability_threshold_gate": probability_useful,
        "rigorous_cell_certificate_nonzero": summary["rigorous_bounds"][
            "b_inexact"
        ]["strict_rounding_cell_certificate_count"]
        > 0,
        "stage_authorizes_confirmation": hard_valid
        and gaussian_viable
        and probability_useful,
        "stage_note": "Calibration-only decision; a positive result would still require a separately frozen confirmation run.",
    }


def run() -> tuple[list[dict], dict, dict]:
    _load_and_validate_preregistration()
    repo_root = HERE.parents[2]
    commit, tree = _git_state(repo_root)
    _reserve_output_directory(OUTPUT_DIRECTORY)

    rows = [
        _evaluate_group(width, index)
        for width in EXPECTED_WIDTHS
        for index in range(EXPECTED_GROUPS_PER_WIDTH)
    ]
    models = {}
    for proxy in PROXIES:
        for variant in MODEL_VARIANTS:
            predictions, fits = _cross_fit(rows, proxy, variant)
            key = _model_key(proxy, variant)
            models[key] = _model_summary(rows, predictions, fits)
            for row, prediction in zip(rows, predictions, strict=True):
                row[f"{key}__mu_root_ulp"] = prediction.mu
                row[f"{key}__sigma_root_ulp"] = prediction.sigma
                row[f"{key}__z"] = prediction.z
                row[f"{key}__p_safe"] = prediction.p_safe
                row[f"{key}__prevalence_baseline"] = prediction.prevalence_baseline

    summary = {
        "schema_version": "1",
        "experiment_id": EXPERIMENT_ID,
        "status": "completed_calibration_only",
        "group_count": len(rows),
        "widths": list(EXPECTED_WIDTHS),
        "models": models,
        "rigorous_bounds": {
            "b_all": _bound_summary(rows, "b_all"),
            "b_inexact": _bound_summary(rows, "b_inexact"),
        },
        "macro_by_width": _macro_summary(rows),
    }
    summary["decision"] = _decision(summary)

    observations_path = OUTPUT_DIRECTORY / "observations.csv"
    summary_path = OUTPUT_DIRECTORY / "model_summary.json"
    _write_csv(observations_path, rows)
    _json_dump(summary_path, summary)
    artifacts = {}
    for path in (observations_path, summary_path):
        artifacts[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    metadata = {
        "schema_version": "1",
        "experiment_id": EXPERIMENT_ID,
        "status": "completed_calibration_only",
        "git_commit_before_opening": commit,
        "git_tree_before_opening": tree,
        "python": sys.version,
        "platform": platform.platform(),
        "input_generator": "irregular_stored_fp32_v1/wide_range_random",
        "widths": list(EXPECTED_WIDTHS),
        "groups_per_width": EXPECTED_GROUPS_PER_WIDTH,
        "fold_count": EXPECTED_FOLDS,
        "source_sha256": _sha256(Path(__file__)),
        "preregistration_sha256": _sha256(PREREGISTRATION),
        "artifacts": artifacts,
    }
    _json_dump(OUTPUT_DIRECTORY / "metadata.json", metadata)
    return rows, summary, metadata


def main() -> int:
    rows, summary, _ = run()
    primary = summary["models"][_model_key(PRIMARY_PROXY, PRIMARY_VARIANT)]
    decision = summary["decision"]
    hard = summary["rigorous_bounds"]["b_inexact"]
    print("Online FP32 reduction risk-certificate calibration")
    print("CALIBRATION ONLY — no confirmation/held-out inputs")
    print(f"groups={len(rows)} widths={','.join(map(str, EXPECTED_WIDTHS))}")
    print(
        "primary q_inexact/bias-aware: "
        f"coverage90={primary['coverage']['90']:.3f} "
        f"coverage99={primary['coverage']['99']:.3f} "
        f"absZ_p99={primary['z_shape']['abs_p99']:.3f} "
        f"Brier={primary['brier_score']:.4f} "
        f"baseline={primary['prevalence_baseline_brier_score']:.4f}"
    )
    print(
        "rigorous B_inexact: "
        f"coverage={hard['coverage']:.3f} "
        f"cell_cert_rate={hard['strict_rounding_cell_certificate_rate']:.3f} "
        f"ratio_p50={hard['bound_to_abs_error_ratio']['p50']:.1f}"
    )
    print(
        "decision: "
        f"gaussian_viable={decision['primary_gaussian_viable']} "
        f"probability_gate={decision['primary_probability_threshold_gate']} "
        f"authorize_confirmation={decision['stage_authorizes_confirmation']}"
    )
    print(f"artifacts={OUTPUT_DIRECTORY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
