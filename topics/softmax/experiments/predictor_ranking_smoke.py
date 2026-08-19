"""Calibration-only smoke test for within-input graph ranking signal.

This runner is exploratory calibration infrastructure, not held-out validation evidence.
It reuses the predeclared irregular stored-FP32 calibration inputs that were checked for
graph-sensitive target variation, then compares the current hand-designed features and a
second-moment theory baseline against the exact within-input graph ranking target.

Nothing printed by this script freezes the final target, predictor score, tree budget,
input distribution, or metric protocol.  No input seed is selected or rejected using the
rank correlations printed here.  Do not promote these rows into held-out evidence.
"""

from __future__ import annotations

import math
from statistics import mean

from predictor_calibration_inputs import calibration_input_families
from predictor_dominant_exposure import dominant_leaf_exposure_features
from predictor_second_moment_baseline import second_moment_tree_cost
from predictor_structural_features import sibling_scale_mismatch_features
from predictor_tree_generator import (
    random_contiguous_split_graph,
    random_pair_merge_graph,
)
from summation_graph_predictor import predict_fp32_tree_error


WIDTH = 256
INPUT_SEEDS = (20260818, 20260819, 20260820, 20260821)
RANDOM_GRAPH_COUNT = 64
TREE_BASE_SEED = 31000000
PROGRESS_EVERY = 16


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


def _graphs(width: int, *, input_index: int):
    """Use the same deterministic tree schedule, extended to the larger K."""
    for graph_index in range(RANDOM_GRAPH_COUNT):
        seed = TREE_BASE_SEED + input_index * 10_000 + graph_index
        if graph_index % 2 == 0:
            yield random_contiguous_split_graph(width, seed=seed)
        else:
            yield random_pair_merge_graph(width, seed=seed)


def _format_rho(value: float | None) -> str:
    return "undefined" if value is None else f"{value:+.3f}"


def _run_input(*, family: str, seed: int, values, input_index: int) -> None:
    target: list[float] = []
    second_moment: list[float] = []
    mismatch_mean: list[float] = []
    mismatch_max: list[float] = []
    mismatch_top4: list[float] = []
    exposure_total: list[float] = []
    exposure_mean: list[float] = []
    exposure_max: list[float] = []

    print(
        f"running family={family} seed={seed} width={len(values)} "
        f"random_graphs={RANDOM_GRAPH_COUNT}",
        flush=True,
    )

    for graph_number, graph in enumerate(
        _graphs(len(values), input_index=input_index),
        start=1,
    ):
        oracle = predict_fp32_tree_error(values, graph)
        target.append(float(abs(oracle.signed_error)))

        theory = second_moment_tree_cost(values, graph)
        second_moment.append(theory.partial_sum_square_cost)

        mismatch = sibling_scale_mismatch_features(values, graph, top_k=4)
        mismatch_mean.append(mismatch.mean_log2_gap)
        mismatch_max.append(mismatch.max_log2_gap)
        mismatch_top4.append(mismatch.top_k_mean_log2_gap)

        exposure = dominant_leaf_exposure_features(values, graph)
        exposure_total.append(exposure.total_severity_log2)
        exposure_mean.append(exposure.mean_severity_log2)
        exposure_max.append(exposure.max_severity_log2)

        if graph_number % PROGRESS_EVERY == 0 or graph_number == RANDOM_GRAPH_COUNT:
            print(f"  progress {graph_number}/{RANDOM_GRAPH_COUNT}", flush=True)

    print(
        f"family={family:<24} seed={seed:<10d} width={len(values)} "
        f"random_graphs={len(target)} target_unique={len(set(target))}"
    )
    print(f"  second_moment   rho={_format_rho(_spearman(second_moment, target))}")
    print(f"  mismatch.mean   rho={_format_rho(_spearman(mismatch_mean, target))}")
    print(f"  mismatch.max    rho={_format_rho(_spearman(mismatch_max, target))}")
    print(f"  mismatch.top4   rho={_format_rho(_spearman(mismatch_top4, target))}")
    print(f"  exposure.total  rho={_format_rho(_spearman(exposure_total, target))}")
    print(f"  exposure.mean   rho={_format_rho(_spearman(exposure_mean, target))}")
    print(f"  exposure.max    rho={_format_rho(_spearman(exposure_max, target))}")
    print(flush=True)


def main() -> int:
    print("Predictor within-input ranking smoke test")
    print("CALIBRATION ONLY — not held-out evidence; no protocol choice is frozen here")
    print(
        f"width={WIDTH} input_seeds={len(INPUT_SEEDS)} "
        f"tree_sample=random-only K={RANDOM_GRAPH_COUNT}; anchors excluded"
    )
    print("Inputs are predeclared; no row is selected or rejected using predictor metrics.")
    print("second_moment = sum of squared cheap binary64 internal partial sums")
    print()

    input_index = 0
    for base_seed in INPUT_SEEDS:
        for generated in calibration_input_families(WIDTH, seed=base_seed):
            _run_input(
                family=generated.family,
                seed=generated.seed,
                values=generated.values,
                input_index=input_index,
            )
            input_index += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
