"""Calibration-only diagnostic for predicting rounding-history scale.

This bridges the microscopic rounding mechanism back toward a cheap score without
using real FP32 intermediate states on the predictor side.

For each node v, let T*_v be the exact mathematical subtree sum from the stored-FP32
leaves.  Approximate one local round-to-nearest residual as uniform on one FP32 cell,

    Var(delta_v) ~= ulp32(T*_v)^2 / 12.

Propagate those local variances through the tree:

    Var_hat(H_v) = Var_hat(E_left) + Var_hat(E_right)
    Var_hat(E_v) = Var_hat(H_v) + Var_hat(delta_v),

where H_v is the descendant rounding history entering node v.  The resulting
sigma_hat(H_v) depends only on stored leaves and graph topology.  The true H_v is
read from the exact FP32 oracle only as a calibration target.

The diagnostic reports whether sigma_hat(H_v) tracks |H_v|, plus a boundary-pressure
quantity sigma_hat(H_v) / distance_to_nearest_residual_sign_boundary.  No AUROC is
computed and no predictor is frozen here.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from fractions import Fraction
from statistics import mean, median

from predictor_calibration_inputs import wide_range_random
from predictor_tree_generator import random_contiguous_split_graph, random_pair_merge_graph
from summation_graph_predictor import BinaryReductionGraph, predict_fp32_tree_error


DEFAULT_WIDTH = 16
DEFAULT_INPUT_SEEDS = (22260824,)
DEFAULT_GRAPH_COUNT = 16
DEFAULT_TRACE_TREES = (0, 1, 5, 7)
TREE_BASE_SEED = 32_000_000
FP32_FRACTION_BITS = 23
FP32_MIN_NORMAL_EXPONENT = -126
FP32_MIN_SUBNORMAL_EXPONENT = -149


@dataclass(frozen=True)
class HistoryScaleNode:
    node_index: int
    left_index: int
    right_index: int
    shadow_sum: Fraction
    ulp_shadow: Fraction
    shadow_phase: Fraction
    boundary_distance_ulp: Fraction
    predicted_history_variance: Fraction
    predicted_subtree_variance: Fraction
    actual_history_shift: Fraction

    @property
    def predicted_history_sigma_ulp(self) -> float:
        if self.predicted_history_variance == 0:
            return 0.0
        return math.sqrt(float(self.predicted_history_variance)) / float(self.ulp_shadow)

    @property
    def actual_history_abs_ulp(self) -> float:
        return float(abs(self.actual_history_shift) / self.ulp_shadow)

    @property
    def standardized_abs_history(self) -> float | None:
        sigma = self.predicted_history_sigma_ulp
        if sigma == 0.0:
            return None
        return self.actual_history_abs_ulp / sigma

    @property
    def boundary_pressure(self) -> float:
        distance = float(self.boundary_distance_ulp)
        sigma = self.predicted_history_sigma_ulp
        if distance == 0.0:
            return float("inf") if sigma > 0 else 0.0
        return sigma / distance


@dataclass(frozen=True)
class HistoryScaleTree:
    graph_family: str
    graph_seed: int
    nodes: tuple[HistoryScaleNode, ...]


def _graph(width: int, *, graph_index: int, input_index: int):
    seed = TREE_BASE_SEED + input_index * 10_000 + graph_index
    if graph_index % 2 == 0:
        return "contiguous", seed, random_contiguous_split_graph(width, seed=seed)
    return "pair_merge", seed, random_pair_merge_graph(width, seed=seed)


def _power_of_two(exponent: int) -> Fraction:
    if exponent >= 0:
        return Fraction(2**exponent)
    return Fraction(1, 2 ** (-exponent))


def _floor_log2(value: Fraction) -> int:
    if value <= 0:
        raise ValueError("log2 requires a positive value")
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if value < _power_of_two(exponent):
        exponent -= 1
    return exponent


def _ulp32_at_positive(value: Fraction) -> Fraction:
    if value <= 0:
        raise ValueError("ULP is defined here only for positive values")
    exponent = _floor_log2(value)
    if exponent < FP32_MIN_NORMAL_EXPONENT:
        return _power_of_two(FP32_MIN_SUBNORMAL_EXPONENT)
    return _power_of_two(exponent - FP32_FRACTION_BITS)


def _phase_on_grid(value: Fraction, ulp: Fraction) -> Fraction:
    scaled = value / ulp
    lower = scaled.numerator // scaled.denominator
    return scaled - lower


def _boundary_distance(phase: Fraction) -> Fraction:
    """Distance in ULP units to the nearest residual-sign boundary.

    Residual sign changes at integer grid points and half-integer midpoints, so on
    one unit cell the boundaries are 0, 1/2, and 1.
    """
    return min(phase, abs(phase - Fraction(1, 2)), 1 - phase)


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
    dx = [value - mx for value in x]
    dy = [value - my for value in y]
    sx = math.sqrt(sum(value * value for value in dx))
    sy = math.sqrt(sum(value * value for value in dy))
    if sx == 0.0 or sy == 0.0:
        return None
    return sum(a * b for a, b in zip(dx, dy, strict=True)) / (sx * sy)


def _spearman(x: list[float], y: list[float]) -> float | None:
    return _pearson(_rankdata(x), _rankdata(y))


def diagnose_history_scale(
    values: tuple[Fraction, ...],
    graph: BinaryReductionGraph,
    *,
    graph_family: str,
    graph_seed: int,
) -> HistoryScaleTree:
    actual = predict_fp32_tree_error(values, graph)

    shadow_sums = list(values)
    subtree_variances = [Fraction(0) for _ in values]
    records: list[HistoryScaleNode] = []

    for node, actual_node in zip(graph.nodes, actual.node_predictions, strict=True):
        shadow_sum = shadow_sums[node.left] + shadow_sums[node.right]
        ulp_shadow = _ulp32_at_positive(shadow_sum)
        phase = _phase_on_grid(shadow_sum, ulp_shadow)

        history_variance = (
            subtree_variances[node.left] + subtree_variances[node.right]
        )
        local_variance = ulp_shadow * ulp_shadow / 12
        subtree_variance = history_variance + local_variance
        history_shift = actual_node.exact_addend_sum - shadow_sum

        records.append(
            HistoryScaleNode(
                node_index=actual_node.node_index,
                left_index=node.left,
                right_index=node.right,
                shadow_sum=shadow_sum,
                ulp_shadow=ulp_shadow,
                shadow_phase=phase,
                boundary_distance_ulp=_boundary_distance(phase),
                predicted_history_variance=history_variance,
                predicted_subtree_variance=subtree_variance,
                actual_history_shift=history_shift,
            )
        )
        shadow_sums.append(shadow_sum)
        subtree_variances.append(subtree_variance)

    return HistoryScaleTree(
        graph_family=graph_family,
        graph_seed=graph_seed,
        nodes=tuple(records),
    )


def _eligible_nodes(trees: list[HistoryScaleTree]) -> list[HistoryScaleNode]:
    return [
        node
        for tree in trees
        for node in tree.nodes
        if node.predicted_history_variance > 0
    ]


def _summary_stats(trees: list[HistoryScaleTree]) -> tuple[float | None, float, float, float, float]:
    nodes = _eligible_nodes(trees)
    predicted = [node.predicted_history_sigma_ulp for node in nodes]
    actual = [node.actual_history_abs_ulp for node in nodes]
    rho = _spearman(predicted, actual)
    z = [
        node.standardized_abs_history
        for node in nodes
        if node.standardized_abs_history is not None
    ]
    coverage1 = mean(value <= 1.0 for value in z) if z else float("nan")
    coverage2 = mean(value <= 2.0 for value in z) if z else float("nan")
    median_z = median(z) if z else float("nan")
    mean_pressure = mean(node.boundary_pressure for node in nodes) if nodes else float("nan")
    return rho, coverage1, coverage2, median_z, mean_pressure


def _fmt_rho(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.3f}"


def _print_tree_summary(input_seed: int, tree_index: int, tree: HistoryScaleTree) -> None:
    rho, coverage1, coverage2, median_z, mean_pressure = _summary_stats([tree])
    nodes = _eligible_nodes([tree])
    print(
        "TREE "
        f"input_seed={input_seed} tree={tree_index:02d} family={tree.graph_family} "
        f"graph_seed={tree.graph_seed} nontrivial={len(nodes)} "
        f"rho_sigma_vs_absH={_fmt_rho(rho)} "
        f"cover<=1sigma={coverage1:.3f} cover<=2sigma={coverage2:.3f} "
        f"median|H|/sigma={median_z:.3f} mean_boundary_pressure={mean_pressure:.3f}"
    )


def _print_trace(tree: HistoryScaleTree) -> None:
    print(
        "  history_trace columns: node left right phase boundary_dist "
        "sigmaH_ulp actual|H|_ulp |H|/sigma boundary_pressure"
    )
    for node in tree.nodes:
        z = node.standardized_abs_history
        z_text = "x" if z is None else f"{z:.3f}"
        print(
            "  NODE "
            f"{node.node_index:>3d} {node.left_index:>3d} {node.right_index:>3d} "
            f"{float(node.shadow_phase):.6f} {float(node.boundary_distance_ulp):.6f} "
            f"{node.predicted_history_sigma_ulp:.6f} {node.actual_history_abs_ulp:.6f} "
            f"{z_text:>7s} {node.boundary_pressure:.6f}"
        )


def _print_family_summary(family: str, trees: list[HistoryScaleTree]) -> None:
    if not trees:
        return
    rho, coverage1, coverage2, median_z, mean_pressure = _summary_stats(trees)
    nodes = _eligible_nodes(trees)
    print(
        f"  {family:<10} trees={len(trees):2d} nodes={len(nodes):3d} "
        f"rho_sigma_vs_absH={_fmt_rho(rho)} "
        f"cover<=1sigma={coverage1:.6f} cover<=2sigma={coverage2:.6f} "
        f"median|H|/sigma={median_z:.6f} mean_boundary_pressure={mean_pressure:.6f}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, choices=(8, 16), default=DEFAULT_WIDTH)
    parser.add_argument("--graphs", type=int, default=DEFAULT_GRAPH_COUNT)
    parser.add_argument(
        "--input-seeds", type=int, nargs="+", default=list(DEFAULT_INPUT_SEEDS)
    )
    parser.add_argument(
        "--trace-trees",
        type=int,
        nargs="+",
        default=list(DEFAULT_TRACE_TREES),
        metavar="INDEX",
    )
    args = parser.parse_args()
    if args.graphs <= 0:
        parser.error("--graphs must be positive")
    bad = [index for index in args.trace_trees if not 0 <= index < args.graphs]
    if bad:
        parser.error(f"--trace-trees indices must be in [0,{args.graphs - 1}]: {bad}")
    return args


def main() -> int:
    args = _parse_args()
    trace_trees = set(args.trace_trees)

    print("Wide-range rounding-history scale diagnostic")
    print("CALIBRATION ONLY — oracle history is used only as the target")
    print("PREDICTOR SIDE — stored leaves + graph + shadow ULP variance only")
    print(
        f"width={args.width} graphs_per_input={args.graphs} "
        f"input_seeds={','.join(str(seed) for seed in args.input_seeds)} "
        f"trace_trees={','.join(f'{index:02d}' for index in args.trace_trees)}"
    )
    print()

    for input_index, input_seed in enumerate(args.input_seeds):
        generated = wide_range_random(args.width, seed=input_seed)
        trees: list[HistoryScaleTree] = []
        print(
            f"INPUT seed={generated.seed} family={generated.family} width={len(generated.values)}"
        )

        for graph_index in range(args.graphs):
            graph_family, graph_seed, graph = _graph(
                len(generated.values), graph_index=graph_index, input_index=input_index
            )
            tree = diagnose_history_scale(
                generated.values,
                graph,
                graph_family=graph_family,
                graph_seed=graph_seed,
            )
            trees.append(tree)
            _print_tree_summary(input_seed, graph_index, tree)
            if graph_index in trace_trees:
                _print_trace(tree)

        print("INPUT SUMMARY")
        _print_family_summary("all", trees)
        _print_family_summary(
            "contiguous", [tree for tree in trees if tree.graph_family == "contiguous"]
        )
        _print_family_summary(
            "pair_merge", [tree for tree in trees if tree.graph_family == "pair_merge"]
        )
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
