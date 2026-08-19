"""Calibration-only diagnostic for second-moment behavior on wide-range inputs.

This script does not change any predictor, target, input generator, or tree schedule.  It
reuses the predeclared width-256 calibration inputs and the same 64-tree schedule as
``predictor_ranking_smoke.py``, then separates the second-moment rank correlation by random
graph family.  Its purpose is to distinguish an input-regime failure from a graph-family
interaction before changing the theory baseline.

Nothing printed here is held-out evidence and no input is selected or rejected.
"""

from __future__ import annotations

import math
from statistics import mean

from predictor_calibration_inputs import calibration_input_families
from predictor_second_moment_baseline import second_moment_tree_cost
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


def _run_wide_input(*, seed: int, values, input_index: int) -> tuple[float | None, float | None, float | None]:
    target_all: list[float] = []
    theory_all: list[float] = []
    target_contiguous: list[float] = []
    theory_contiguous: list[float] = []
    target_pair: list[float] = []
    theory_pair: list[float] = []

    print(
        f"running family=wide_range_random seed={seed} width={len(values)} "
        f"random_graphs={RANDOM_GRAPH_COUNT}",
        flush=True,
    )

    for graph_index in range(RANDOM_GRAPH_COUNT):
        graph_family, graph = _graph(
            len(values), input_index=input_index, graph_index=graph_index
        )
        oracle = predict_fp32_tree_error(values, graph)
        target = float(abs(oracle.signed_error))
        theory = second_moment_tree_cost(values, graph).partial_sum_square_cost

        target_all.append(target)
        theory_all.append(theory)
        if graph_family == "contiguous":
            target_contiguous.append(target)
            theory_contiguous.append(theory)
        else:
            target_pair.append(target)
            theory_pair.append(theory)

        completed = graph_index + 1
        if completed % PROGRESS_EVERY == 0 or completed == RANDOM_GRAPH_COUNT:
            print(f"  progress {completed}/{RANDOM_GRAPH_COUNT}", flush=True)

    rho_all = _spearman(theory_all, target_all)
    rho_contiguous = _spearman(theory_contiguous, target_contiguous)
    rho_pair = _spearman(theory_pair, target_pair)

    print(
        f"seed={seed:<10d} "
        f"target_unique(all/contig/pair)="
        f"{len(set(target_all))}/{len(set(target_contiguous))}/{len(set(target_pair))}"
    )
    print(f"  second_moment.all         rho={_format_rho(rho_all)} n={len(target_all)}")
    print(
        f"  second_moment.contiguous  rho={_format_rho(rho_contiguous)} "
        f"n={len(target_contiguous)}"
    )
    print(
        f"  second_moment.pair_merge  rho={_format_rho(rho_pair)} "
        f"n={len(target_pair)}"
    )
    print(flush=True)
    return rho_all, rho_contiguous, rho_pair


def main() -> int:
    print("Wide-range second-moment graph-family diagnostic")
    print("CALIBRATION ONLY — no predictor changes; no held-out evidence")
    print(
        f"width={WIDTH} input_seeds={len(INPUT_SEEDS)} K={RANDOM_GRAPH_COUNT}; "
        "same predeclared tree schedule as ranking smoke"
    )
    print()

    all_rhos: list[float] = []
    contiguous_rhos: list[float] = []
    pair_rhos: list[float] = []

    input_index = 0
    for base_seed in INPUT_SEEDS:
        for generated in calibration_input_families(WIDTH, seed=base_seed):
            if generated.family == "wide_range_random":
                rho_all, rho_contiguous, rho_pair = _run_wide_input(
                    seed=generated.seed,
                    values=generated.values,
                    input_index=input_index,
                )
                if rho_all is not None:
                    all_rhos.append(rho_all)
                if rho_contiguous is not None:
                    contiguous_rhos.append(rho_contiguous)
                if rho_pair is not None:
                    pair_rhos.append(rho_pair)
            input_index += 1

    print("wide-range mean rho across the four predeclared inputs:")
    print(f"  all         {_format_rho(mean(all_rhos) if all_rhos else None)}")
    print(
        f"  contiguous  "
        f"{_format_rho(mean(contiguous_rhos) if contiguous_rhos else None)}"
    )
    print(f"  pair_merge  {_format_rho(mean(pair_rhos) if pair_rhos else None)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
