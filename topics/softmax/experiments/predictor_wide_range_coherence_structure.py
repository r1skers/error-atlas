"""Calibration-only structural decomposition of the cross-node coherence term C.

For exact local FP32 rounding residuals delta_v,

    C = 2 * sum_{u<v} delta_u delta_v.

This diagnostic asks where that signed pairwise term lives in the reduction tree.  Internal-node
pairs are partitioned into three mutually exclusive classes:

    parent_child : one node is the direct parent of the other
    far_ancestor : one node is an ancestor of the other, with graph-distance >= 2
    disjoint     : neither node is an ancestor of the other

The ancestor contribution is also split by ancestor gap 1, 2, 3, and 4+.  Finally, the script
reports how concentrated the absolute pairwise interaction mass is among the largest pair terms.

This intentionally uses exact FP32 oracle residuals and is CALIBRATION ONLY.  It does not create
a cheap predictor and does not touch held-out data.
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
from reduction_analysis import CoherenceTree, CoherenceAnalysis, replay


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
) -> CoherenceTree:
    """Compatibility wrapper; compose multiple views with CoherenceAnalysis instead."""
    return CoherenceAnalysis(
        replay(values, graph), graph_family=graph_family, graph_seed=graph_seed
    ).structure


def _component_summary(label: str, trees: list[CoherenceTree]) -> None:
    if not trees:
        return
    e2 = [t.e2 for t in trees]
    c_total = [t.c_total for t in trees]
    parent = [t.c_parent for t in trees]
    far = [t.c_far_ancestor for t in trees]
    ancestor = [t.c_ancestor for t in trees]
    disjoint = [t.c_disjoint for t in trees]
    std_total = _std(c_total)

    def std_ratio(values: list[float]) -> float:
        return float("nan") if std_total == 0 else _std(values) / std_total

    abs_total_mass = [t.abs_pair_mass for t in trees]
    ancestor_abs_share = mean(
        0.0 if total == 0 else (t.abs_parent_mass + t.abs_far_ancestor_mass) / total
        for t, total in zip(trees, abs_total_mass, strict=True)
    )
    disjoint_abs_share = mean(
        0.0 if total == 0 else t.abs_disjoint_mass / total
        for t, total in zip(trees, abs_total_mass, strict=True)
    )

    print(
        f"  {label:<10} n={len(trees):2d} "
        f"rho_parent={_fmt_rho(_spearman(parent, e2))} "
        f"rho_farAnc={_fmt_rho(_spearman(far, e2))} "
        f"rho_ancestor={_fmt_rho(_spearman(ancestor, e2))} "
        f"rho_disjoint={_fmt_rho(_spearman(disjoint, e2))}"
    )
    print(
        f"    std/component_vs_C parent={std_ratio(parent):.3f} "
        f"farAnc={std_ratio(far):.3f} ancestor={std_ratio(ancestor):.3f} "
        f"disjoint={std_ratio(disjoint):.3f} "
        f"mean_abs_pair_mass_share ancestor={ancestor_abs_share:.3f} "
        f"disjoint={disjoint_abs_share:.3f}"
    )
    print(
        f"    top_pair_abs_mass_share "
        f"top1%={mean(t.top1pct_abs_mass_share for t in trees):.3f} "
        f"top5%={mean(t.top5pct_abs_mass_share for t in trees):.3f} "
        f"top10%={mean(t.top10pct_abs_mass_share for t in trees):.3f}"
    )


def _gap_summary(trees: list[CoherenceTree]) -> None:
    if not trees:
        return
    e2 = [t.e2 for t in trees]
    print(
        "    ancestor_gap_rho_vs_E2 "
        f"gap1={_fmt_rho(_spearman([t.c_gap1 for t in trees], e2))} "
        f"gap2={_fmt_rho(_spearman([t.c_gap2 for t in trees], e2))} "
        f"gap3={_fmt_rho(_spearman([t.c_gap3 for t in trees], e2))} "
        f"gap4+={_fmt_rho(_spearman([t.c_gap4plus for t in trees], e2))}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--graphs", type=int, default=DEFAULT_GRAPH_COUNT)
    parser.add_argument(
        "--input-seeds", type=int, nargs="+", default=list(DEFAULT_INPUT_SEEDS)
    )
    args = parser.parse_args()
    if args.width <= 1:
        parser.error("--width must exceed 1")
    if args.graphs <= 1:
        parser.error("--graphs must exceed 1")
    return args


def main() -> int:
    args = _parse_args()
    print("Wide-range coherence structure decomposition")
    print("CALIBRATION ONLY — exact local FP32 residuals are intentionally inspected")
    print(
        "C = parent-child + far-ancestor + disjoint; ancestor also split by graph gap"
    )
    print(
        f"width={args.width} graphs_per_input={args.graphs} "
        f"input_seeds={','.join(str(seed) for seed in args.input_seeds)}"
    )
    print()

    for input_index, input_seed in enumerate(args.input_seeds):
        generated = wide_range_random(args.width, seed=input_seed)
        trees: list[CoherenceTree] = []
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
        _component_summary("all", trees)
        _gap_summary(trees)
        contiguous = [t for t in trees if t.graph_family == "contiguous"]
        pair_merge = [t for t in trees if t.graph_family == "pair_merge"]
        _component_summary("contiguous", contiguous)
        _gap_summary(contiguous)
        _component_summary("pair_merge", pair_merge)
        _gap_summary(pair_merge)
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
