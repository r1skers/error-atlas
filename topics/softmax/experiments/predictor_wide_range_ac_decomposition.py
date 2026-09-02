"""Calibration-only decomposition of tree-level FP32 error into local magnitude and coherence.

For one explicit reduction tree with exact local FP32 rounding residuals delta_v,

    E = sum_v delta_v
    E^2 = A + C

where

    A = sum_v delta_v^2
    C = 2 * sum_{u<v} delta_u delta_v = E^2 - A.

A measures local squared rounding activity. C is the signed cross-node coherence term:
positive C means same-sign residuals reinforce overall; negative C means cancellation dominates.

This script uses the exact FP32 tree oracle intentionally and is CALIBRATION ONLY.  It does not
construct a cheap predictor.  The purpose is to determine whether tree-to-tree variation in
wide-range root error is primarily associated with A or with C before designing another score.
"""

from __future__ import annotations

import argparse
import math
from fractions import Fraction
from statistics import mean

from predictor_calibration_inputs import wide_range_random
from predictor_tree_generator import (
    random_contiguous_split_graph,
    random_pair_merge_graph,
)
from summation_graph_predictor import BinaryReductionGraph
from reduction_analysis import ACTree, CoherenceAnalysis, replay


DEFAULT_WIDTH = 256
DEFAULT_INPUT_SEEDS = (22260821, 22260822, 22260823, 22260824)
DEFAULT_GRAPH_COUNT = 64
TREE_BASE_SEED = 32_000_000


def _graph(width: int, *, graph_index: int, input_index: int):
    seed = TREE_BASE_SEED + input_index * 10_000 + graph_index
    if graph_index % 2 == 0:
        return "contiguous", seed, random_contiguous_split_graph(width, seed=seed)
    return "pair_merge", seed, random_pair_merge_graph(width, seed=seed)


def _rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    return ranks


def _pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    mx = mean(x)
    my = mean(y)
    dx = [v - mx for v in x]
    dy = [v - my for v in y]
    sx = math.sqrt(sum(v * v for v in dx))
    sy = math.sqrt(sum(v * v for v in dy))
    if sx == 0.0 or sy == 0.0:
        return None
    return sum(a * b for a, b in zip(dx, dy, strict=True)) / (sx * sy)


def _spearman(x: list[float], y: list[float]) -> float | None:
    return _pearson(_rankdata(x), _rankdata(y))


def _std(values: list[float]) -> float:
    if not values:
        return float("nan")
    m = mean(values)
    return math.sqrt(mean((v - m) ** 2 for v in values))


def _fmt_rho(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.3f}"


def diagnose_tree(
    values: tuple[Fraction, ...],
    graph: BinaryReductionGraph,
    *,
    graph_family: str,
    graph_seed: int,
) -> ACTree:
    """Compatibility wrapper; compose multiple views with CoherenceAnalysis instead."""
    return CoherenceAnalysis(
        replay(values, graph), graph_family=graph_family, graph_seed=graph_seed
    ).ac


def _summary(label: str, trees: list[ACTree]) -> None:
    if not trees:
        return
    e2 = [float(t.e2) for t in trees]
    a = [float(t.a_local) for t in trees]
    c = [float(t.c_coherence) for t in trees]
    rho_a = _spearman(a, e2)
    rho_c = _spearman(c, e2)
    std_a = _std(a)
    std_c = _std(c)
    variation_ratio = float("inf") if std_a == 0 else std_c / std_a
    positive_c = mean(v > 0 for v in c)
    negative_c = mean(v < 0 for v in c)
    mean_abs_c_over_a = mean(
        abs(t.c_over_a) for t in trees if math.isfinite(t.c_over_a)
    )

    print(
        f"  {label:<10} n={len(trees):2d} "
        f"rho_A_vs_E2={_fmt_rho(rho_a)} rho_C_vs_E2={_fmt_rho(rho_c)} "
        f"stdC/stdA={variation_ratio:.3f} "
        f"mean|C/A|={mean_abs_c_over_a:.3f} "
        f"frac_C_pos={positive_c:.3f} frac_C_neg={negative_c:.3f}"
    )


def _print_extremes(trees: list[ACTree], *, count: int = 3) -> None:
    ranked = sorted(enumerate(trees), key=lambda item: item[1].e2)
    picks = ranked[:count] + ranked[-count:]
    print("  extremes columns: tree family E2 A C C/A")
    for tree_index, tree in picks:
        print(
            f"    tree={tree_index:02d} family={tree.graph_family:<10} "
            f"E2={float(tree.e2):.9e} A={float(tree.a_local):.9e} "
            f"C={float(tree.c_coherence):+.9e} C/A={tree.c_over_a:+.3f}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--graphs", type=int, default=DEFAULT_GRAPH_COUNT)
    parser.add_argument(
        "--input-seeds", type=int, nargs="+", default=list(DEFAULT_INPUT_SEEDS)
    )
    parser.add_argument("--show-extremes", action="store_true")
    args = parser.parse_args()
    if args.width <= 1:
        parser.error("--width must exceed 1")
    if args.graphs <= 1:
        parser.error("--graphs must exceed 1")
    return args


def main() -> int:
    args = _parse_args()
    print("Wide-range A/C error decomposition")
    print("CALIBRATION ONLY — exact local FP32 residuals are intentionally inspected")
    print("A=sum(delta^2), C=E^2-A=2*sum_{u<v}(delta_u delta_v)")
    print(
        f"width={args.width} graphs_per_input={args.graphs} "
        f"input_seeds={','.join(str(seed) for seed in args.input_seeds)}"
    )
    print()

    per_seed: list[tuple[float | None, float | None, float]] = []
    for input_index, input_seed in enumerate(args.input_seeds):
        generated = wide_range_random(args.width, seed=input_seed)
        trees: list[ACTree] = []
        for graph_index in range(args.graphs):
            family, graph_seed, graph = _graph(
                len(generated.values), graph_index=graph_index, input_index=input_index
            )
            trees.append(
                diagnose_tree(
                    generated.values,
                    graph,
                    graph_family=family,
                    graph_seed=graph_seed,
                )
            )

        print(
            f"INPUT seed={input_seed} family={generated.family} width={len(generated.values)}"
        )
        _summary("all", trees)
        _summary("contiguous", [t for t in trees if t.graph_family == "contiguous"])
        _summary("pair_merge", [t for t in trees if t.graph_family == "pair_merge"])
        if args.show_extremes:
            _print_extremes(trees)

        e2 = [float(t.e2) for t in trees]
        a = [float(t.a_local) for t in trees]
        c = [float(t.c_coherence) for t in trees]
        rho_a = _spearman(a, e2)
        rho_c = _spearman(c, e2)
        std_a = _std(a)
        ratio = float("inf") if std_a == 0 else _std(c) / std_a
        per_seed.append((rho_a, rho_c, ratio))
        print()

    valid_a = [x[0] for x in per_seed if x[0] is not None]
    valid_c = [x[1] for x in per_seed if x[1] is not None]
    ratios = [x[2] for x in per_seed]
    print("SEED SUMMARY all-tree decomposition mean/min/max")
    if valid_a:
        print(
            f"  rho_A_vs_E2 mean={mean(valid_a):+.3f} "
            f"min={min(valid_a):+.3f} max={max(valid_a):+.3f}"
        )
    if valid_c:
        print(
            f"  rho_C_vs_E2 mean={mean(valid_c):+.3f} "
            f"min={min(valid_c):+.3f} max={max(valid_c):+.3f}"
        )
    print(
        f"  stdC/stdA mean={mean(ratios):.3f} "
        f"min={min(ratios):.3f} max={max(ratios):.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
