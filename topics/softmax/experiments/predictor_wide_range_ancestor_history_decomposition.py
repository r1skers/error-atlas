"""Calibration-only diagnostic connecting ancestor coherence to recursive rounding history.

For exact local FP32 rounding residuals delta_v on a binary reduction tree, define

    H_v = sum_{u internal proper descendant of v} delta_u.

Then the signed coherence contributed by ancestor-related internal-node pairs is exactly

    C_ancestor = 2 * sum_v delta_v * H_v.

This script verifies that identity and asks whether the node-local terms

    K_v = 2 * delta_v * H_v

are concentrated enough to motivate an O(n) recursive approximation.  It is CALIBRATION ONLY:
exact FP32 oracle residuals are intentionally inspected and no predictor evidence is claimed.
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
from reduction_analysis import TreeDiagnostic, CoherenceAnalysis, replay
from reduction_analysis.coherence import TOP_FRACS as TOP_FRACS

DEFAULT_WIDTH = 256
DEFAULT_INPUT_SEEDS = (22260821, 22260822, 22260823, 22260824)
DEFAULT_GRAPH_COUNT = 64
TREE_BASE_SEED = 33_000_000


def _graph(width: int, graph_index: int, input_index: int):
    seed = TREE_BASE_SEED + input_index * 10_000 + graph_index
    if graph_index % 2 == 0:
        return "contiguous", random_contiguous_split_graph(width, seed=seed)
    return "pair_merge", random_pair_merge_graph(width, seed=seed)


def diagnose(
    values: tuple[Fraction, ...], graph: BinaryReductionGraph, family: str
) -> TreeDiagnostic:
    """Compatibility wrapper; compose multiple views with CoherenceAnalysis instead."""
    return CoherenceAnalysis(replay(values, graph), graph_family=family).history


def _fmt(v: float) -> str:
    return "nan" if not math.isfinite(v) else f"{v:+.3f}"


def _vec(vs: tuple[float, ...]) -> str:
    return "/".join("nan" if not math.isfinite(v) else f"{v:.3f}" for v in vs)


def _summary(label: str, trees: list[TreeDiagnostic]) -> None:
    if not trees:
        return

    def avg_tuple(attr: str) -> tuple[float, ...]:
        rows = [getattr(t, attr) for t in trees]
        return tuple(
            mean(row[i] for row in rows if math.isfinite(row[i]))
            for i in range(len(rows[0]))
        )

    def avg_scalar(attr: str) -> float:
        vals = [getattr(t, attr) for t in trees if math.isfinite(getattr(t, attr))]
        return mean(vals) if vals else float("nan")

    ancestor_frac = []
    for t in trees:
        if t.c_total != 0:
            ancestor_frac.append(float(t.c_ancestor / t.c_total))
    print(
        f"  {label:<10} n={len(trees):2d} identity={sum(t.k_total == t.c_ancestor for t in trees)}/{len(trees)} "
        f"mean_Cancestor/C={_fmt(mean(ancestor_frac) if ancestor_frac else float('nan'))}"
    )
    print(
        f"    top|K|_absMass[1/5/10/20%]={_vec(avg_tuple('top_abs_k_mass'))} "
        f"top|K|_signedRecovery[1/5/10/20%]={_vec(avg_tuple('top_signed_k_recovery'))}"
    )
    print(
        f"    mean_node_rho |K|~|delta|={_fmt(avg_scalar('rho_abs_k_vs_abs_delta'))} "
        f"|K|~|H|={_fmt(avg_scalar('rho_abs_k_vs_abs_history'))}"
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    p.add_argument("--graphs", type=int, default=DEFAULT_GRAPH_COUNT)
    p.add_argument(
        "--input-seeds", type=int, nargs="+", default=list(DEFAULT_INPUT_SEEDS)
    )
    args = p.parse_args()
    if args.width <= 1:
        p.error("--width must exceed 1")
    if args.graphs <= 1:
        p.error("--graphs must exceed 1")

    print("Wide-range ancestor/history coherence decomposition")
    print("CALIBRATION ONLY — exact FP32 residuals/history are intentionally inspected")
    print("Testing C_ancestor == 2 sum_v delta_v H_v and concentration of K_v")
    print(
        f"width={args.width} graphs_per_input={args.graphs} input_seeds={','.join(map(str, args.input_seeds))}"
    )
    print()

    for input_index, seed in enumerate(args.input_seeds):
        generated = wide_range_random(args.width, seed=seed)
        trees: list[TreeDiagnostic] = []
        for graph_index in range(args.graphs):
            family, graph = _graph(len(generated.values), graph_index, input_index)
            trees.append(diagnose(generated.values, graph, family))
        print(
            f"INPUT seed={seed} family={generated.family} width={len(generated.values)}"
        )
        _summary("all", trees)
        _summary("contiguous", [t for t in trees if t.graph_family == "contiguous"])
        _summary("pair_merge", [t for t in trees if t.graph_family == "pair_merge"])
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
