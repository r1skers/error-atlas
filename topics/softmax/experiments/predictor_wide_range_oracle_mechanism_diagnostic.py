"""Calibration-only oracle mechanism diagnostic for wide-range reductions.

This script deliberately uses the exact FP32 oracle as a microscope, not as a cheap
predictor.  It inspects node-level deterministic RN-even behavior on the same predeclared
width-256 wide-range calibration inputs and 64-tree schedule used by the earlier ranking
and stagnation diagnostics.

The purpose is to distinguish three mechanisms that cheap structural proxies cannot
separate reliably:

1. actual stagnation: a positive smaller addend is completely absorbed by a larger one;
2. local rounding activity: the total magnitude of exact node-level rounding residuals;
3. cancellation: positive and negative local residuals partially cancel before the root.

Nothing printed here is held-out evidence and no predictor formula is frozen by this run.
"""

from __future__ import annotations

import math
from fractions import Fraction
from statistics import mean

from predictor_calibration_inputs import calibration_input_families
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


def _format_rho(value: float | None) -> str:
    return "undefined" if value is None else f"{value:+.3f}"


def _graph(width: int, *, input_index: int, graph_index: int):
    seed = TREE_BASE_SEED + input_index * 10_000 + graph_index
    if graph_index % 2 == 0:
        return "contiguous", random_contiguous_split_graph(width, seed=seed)
    return "pair_merge", random_pair_merge_graph(width, seed=seed)


def _oracle_mechanism_metrics(values, graph):
    prediction = predict_fp32_tree_error(values, graph)
    states = list(values)

    actual_stagnation_count = 0
    actual_stagnated_mass = Fraction(0)
    sum_abs_local_error = Fraction(0)

    for node, node_prediction in zip(graph.nodes, prediction.node_predictions, strict=True):
        left_value = states[node.left]
        right_value = states[node.right]
        rounded_value = node_prediction.rounded_sum
        local_error = node_prediction.local_rounding_error

        small = min(left_value, right_value)
        large = max(left_value, right_value)
        if small > 0 and rounded_value == large:
            actual_stagnation_count += 1
            actual_stagnated_mass += small

        sum_abs_local_error += abs(local_error)
        states.append(rounded_value)

    if states[graph.root] != prediction.predicted_sum:
        raise AssertionError("oracle replay did not reproduce predicted root")

    final_abs_error = abs(prediction.signed_error)
    cancellation_ratio = (
        0.0
        if sum_abs_local_error == 0
        else 1.0 - float(final_abs_error / sum_abs_local_error)
    )
    root_mass = prediction.exact_input_sum

    return {
        "target": float(final_abs_error),
        "actual_stagnation_frac": actual_stagnation_count / len(graph.nodes),
        "actual_stagnated_mass_frac": (
            0.0 if root_mass == 0 else float(actual_stagnated_mass / root_mass)
        ),
        "sum_abs_local_error": float(sum_abs_local_error),
        "cancellation_ratio": cancellation_ratio,
        "inexact_addition_frac": prediction.inexact_addition_count / len(graph.nodes),
    }


def _print_group(name: str, rows: list[dict[str, float]]) -> None:
    target = [row["target"] for row in rows]
    print(f"  {name}:")
    for metric in (
        "actual_stagnation_frac",
        "actual_stagnated_mass_frac",
        "sum_abs_local_error",
        "cancellation_ratio",
        "inexact_addition_frac",
    ):
        values = [row[metric] for row in rows]
        print(
            f"    {metric:<29} "
            f"rho={_format_rho(_spearman(values, target))} mean={mean(values):.6g}"
        )


def main() -> int:
    print("Wide-range exact-oracle mechanism diagnostic")
    print("CALIBRATION ONLY — oracle used as microscope; no cheap predictor defined")
    print(
        f"width={WIDTH} input_seeds={len(INPUT_SEEDS)} K={RANDOM_GRAPH_COUNT}; "
        "same predeclared tree schedule"
    )
    print()

    input_index = 0
    for base_seed in INPUT_SEEDS:
        for generated in calibration_input_families(WIDTH, seed=base_seed):
            if generated.family != "wide_range_random":
                input_index += 1
                continue

            print(
                f"running family=wide_range_random seed={generated.seed} "
                f"width={len(generated.values)} random_graphs={RANDOM_GRAPH_COUNT}",
                flush=True,
            )

            all_rows: list[dict[str, float]] = []
            contiguous_rows: list[dict[str, float]] = []
            pair_rows: list[dict[str, float]] = []

            for graph_index in range(RANDOM_GRAPH_COUNT):
                graph_family, graph = _graph(
                    len(generated.values),
                    input_index=input_index,
                    graph_index=graph_index,
                )
                row = _oracle_mechanism_metrics(generated.values, graph)
                all_rows.append(row)
                if graph_family == "contiguous":
                    contiguous_rows.append(row)
                else:
                    pair_rows.append(row)

                completed = graph_index + 1
                if completed % PROGRESS_EVERY == 0 or completed == RANDOM_GRAPH_COUNT:
                    print(f"  progress {completed}/{RANDOM_GRAPH_COUNT}", flush=True)

            print(
                f"seed={generated.seed:<10d} target_unique(all/contig/pair)="
                f"{len(set(row['target'] for row in all_rows))}/"
                f"{len(set(row['target'] for row in contiguous_rows))}/"
                f"{len(set(row['target'] for row in pair_rows))}"
            )
            _print_group("all", all_rows)
            _print_group("contiguous", contiguous_rows)
            _print_group("pair_merge", pair_rows)
            print(flush=True)
            input_index += 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
