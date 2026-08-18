"""Calibration-only smoke test for within-input graph ranking signal.

This runner is exploratory calibration infrastructure, not held-out validation evidence.
It intentionally uses a tiny deterministic set of stored-FP32 inputs and a small fixed
candidate-tree sample.  Its only purpose is to answer whether the currently defined cheap
structural features show any obvious within-input ranking signal worth investigating.

Nothing printed by this script freezes the final target, predictor score, tree budget,
input distribution, or metric protocol.  Do not promote these rows into held-out evidence.
"""

from __future__ import annotations

import math
from fractions import Fraction
from statistics import mean

from predictor_dominant_exposure import dominant_leaf_exposure_features
from predictor_structural_features import sibling_scale_mismatch_features
from predictor_tree_generator import (
    random_contiguous_split_graph,
    random_pair_merge_graph,
)
from summation_graph_predictor import predict_fp32_tree_error


SMOKE_WIDTH = 64
RANDOM_GRAPH_COUNT = 16
BASE_SEED = 20260818


def _rankdata(values: list[float]) -> list[float]:
    """Return average ranks with ties, using rank 1 for the smallest value."""
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        value = values[order[start]]
        while stop < len(order) and values[order[stop]] == value:
            stop += 1
        average_rank = (start + 1 + stop) / 2.0
        for position in range(start, stop):
            ranks[order[position]] = average_rank
        start = stop
    return ranks


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right):
        raise ValueError("vectors must have equal length")
    if len(left) < 2:
        return None
    left_mean = mean(left)
    right_mean = mean(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    left_ss = sum(value * value for value in left_centered)
    right_ss = sum(value * value for value in right_centered)
    if left_ss == 0.0 or right_ss == 0.0:
        return None
    covariance = sum(a * b for a, b in zip(left_centered, right_centered, strict=True))
    return covariance / math.sqrt(left_ss * right_ss)


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right):
        raise ValueError("vectors must have equal length")
    if any(not math.isfinite(value) for value in (*left, *right)):
        return None
    return _pearson(_rankdata(left), _rankdata(right))


def _head_tail() -> tuple[Fraction, ...]:
    values = [Fraction(1)]
    values.extend(Fraction(1, 2**18) for _ in range(SMOKE_WIDTH - 1))
    return tuple(values)


def _same_scale() -> tuple[Fraction, ...]:
    # Exactly representable dyadic values spanning only a narrow scale band.
    cycle = (Fraction(1), Fraction(5, 8), Fraction(3, 4), Fraction(7, 8))
    return tuple(cycle[index % len(cycle)] for index in range(SMOKE_WIDTH))


def _wide_range() -> tuple[Fraction, ...]:
    exponents = (0, -4, -8, -12, -16, -20, -24, -28)
    return tuple(
        Fraction(1, 2 ** (-exponents[index % len(exponents)]))
        if exponents[index % len(exponents)] < 0
        else Fraction(1)
        for index in range(SMOKE_WIDTH)
    )


def _graphs() -> list[object]:
    graphs = []
    for index in range(RANDOM_GRAPH_COUNT):
        seed = BASE_SEED + index
        if index % 2 == 0:
            graph = random_contiguous_split_graph(SMOKE_WIDTH, seed=seed)
        else:
            graph = random_pair_merge_graph(SMOKE_WIDTH, seed=seed)
        graphs.append(graph)
    return graphs


def _format_rho(value: float | None) -> str:
    return "undefined" if value is None else f"{value:+.3f}"


def _run_input(name: str, values: tuple[Fraction, ...]) -> None:
    target: list[float] = []
    mismatch_mean: list[float] = []
    mismatch_max: list[float] = []
    mismatch_top4: list[float] = []
    exposure_total: list[float] = []
    exposure_mean: list[float] = []
    exposure_max: list[float] = []

    for graph in _graphs():
        oracle = predict_fp32_tree_error(values, graph)
        target.append(float(abs(oracle.signed_error)))

        mismatch = sibling_scale_mismatch_features(values, graph, top_k=4)
        mismatch_mean.append(mismatch.mean_log2_gap)
        mismatch_max.append(mismatch.max_log2_gap)
        mismatch_top4.append(mismatch.top_k_mean_log2_gap)

        exposure = dominant_leaf_exposure_features(values, graph)
        exposure_total.append(exposure.total_severity_log2)
        exposure_mean.append(exposure.mean_severity_log2)
        exposure_max.append(exposure.max_severity_log2)

    print(f"input={name} width={len(values)} random_graphs={len(target)}")
    print("  provisional_target = abs(exact signed forward error)")
    print(f"  mismatch.mean   rho={_format_rho(_spearman(mismatch_mean, target))}")
    print(f"  mismatch.max    rho={_format_rho(_spearman(mismatch_max, target))}")
    print(f"  mismatch.top4   rho={_format_rho(_spearman(mismatch_top4, target))}")
    print(f"  exposure.total  rho={_format_rho(_spearman(exposure_total, target))}")
    print(f"  exposure.mean   rho={_format_rho(_spearman(exposure_mean, target))}")
    print(f"  exposure.max    rho={_format_rho(_spearman(exposure_max, target))}")
    print()


def main() -> int:
    print("Predictor within-input ranking smoke test")
    print("CALIBRATION ONLY — not held-out evidence; no protocol choice is frozen here")
    print(f"tree_sample=random-only K={RANDOM_GRAPH_COUNT}; anchors excluded")
    print()

    for name, values in (
        ("head_tail", _head_tail()),
        ("same_scale", _same_scale()),
        ("wide_range", _wide_range()),
    ):
        _run_input(name, values)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
