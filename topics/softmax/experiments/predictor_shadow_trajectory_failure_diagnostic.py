"""Calibration-only diagnostic for the deterministic shadow error trajectory.

The recursive Gaussian prototype accidentally exposed a useful control: when propagated variance
stays zero, the score becomes a deterministic phase-preserving shadow trajectory and showed weak
positive within-input ranking signal.  This script asks where that shadow trajectory succeeds and
where it loses the true FP32 path.

For every internal node it compares:
  * actual versus shadow local rounding-residual sign,
  * actual versus shadow accumulated history before the node,
  * history drift in units of the node shadow ULP,
  * whether that drift crosses a rounding boundary,
  * how these quantities change with tree depth.

It also reports tree-level Spearman correlation of |shadow root error| with the true |root error|.
Exact FP32 states are inspected only for diagnostics/targets.  CALIBRATION ONLY.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from fractions import Fraction
from statistics import mean

from predictor_calibration_inputs import wide_range_random
from predictor_gaussian_ancestor_coherence_calibration import _fp32_ulp
from predictor_tree_generator import random_contiguous_split_graph, random_pair_merge_graph
from summation_graph_predictor import BinaryReductionGraph, predict_fp32_tree_error

DEFAULT_WIDTH = 256
DEFAULT_GRAPH_COUNT = 64
DEFAULT_INPUT_SEEDS = (22260821, 22260822, 22260823, 22260824)
TREE_BASE_SEED = 37_000_000


@dataclass(frozen=True)
class NodeSample:
    family: str
    depth: int
    depth_frac: float
    sign_match: int
    history_drift_ulp: float
    boundary_cross: int
    actual_abs_delta: float


@dataclass(frozen=True)
class TreeSample:
    family: str
    actual_root_error: float
    shadow_root_error: float
    root_sign_match: int
    node_sign_match: float
    weighted_sign_match: float
    mean_abs_drift_ulp: float
    max_abs_drift_ulp: float
    boundary_cross_rate: float


def _graph(width: int, graph_index: int, input_index: int):
    seed = TREE_BASE_SEED + input_index * 10_000 + graph_index
    if graph_index % 2 == 0:
        return "contiguous", random_contiguous_split_graph(width, seed=seed)
    return "pair_merge", random_pair_merge_graph(width, seed=seed)


def _sgn(x: float) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


def _rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
    return ranks


def _corr(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or len(x) != len(y):
        return None
    mx, my = mean(x), mean(y)
    dx = [v - mx for v in x]
    dy = [v - my for v in y]
    sx = math.sqrt(sum(v*v for v in dx))
    sy = math.sqrt(sum(v*v for v in dy))
    if sx == 0.0 or sy == 0.0:
        return None
    return sum(a*b for a,b in zip(dx,dy,strict=True))/(sx*sy)


def _spearman(x: list[float], y: list[float]) -> float | None:
    return _corr(_rankdata(x), _rankdata(y))


def _boundary_cross_count(phase: float, shift: float) -> int:
    """Number of half-integer rounding boundaries crossed by moving phase by shift."""
    a = phase
    b = phase + shift
    lo, hi = (a, b) if a <= b else (b, a)
    # Boundaries are k+0.5. Ignore a boundary exactly at the starting point.
    first = math.floor(lo - 0.5) + 1
    last = math.floor(hi - 0.5)
    return max(0, last - first + 1)


def _diagnose(values: tuple[Fraction, ...], graph: BinaryReductionGraph, family: str) -> tuple[TreeSample, list[NodeSample]]:
    pred = predict_fp32_tree_error(values, graph)
    n = graph.leaf_count
    actual_delta = {n+i: float(p.local_rounding_error) for i,p in enumerate(pred.node_predictions)}

    shadow_sum = [float(v) for v in values]
    shadow_error = [0.0 for _ in values]
    actual_error = [0.0 for _ in values]
    depth = [0 for _ in values]
    rows_raw: list[tuple[int,int,float,int,float]] = []
    weighted_ok = 0.0
    weighted_total = 0.0

    for offset, node in enumerate(graph.nodes):
        idx = n + offset
        exact_shadow_sum = shadow_sum[node.left] + shadow_sum[node.right]
        ulp = _fp32_ulp(exact_shadow_sum)
        phase = exact_shadow_sum / ulp
        phase -= math.floor(phase)

        h_shadow = shadow_error[node.left] + shadow_error[node.right]
        h_actual = actual_error[node.left] + actual_error[node.right]
        scaled = phase + h_shadow / ulp
        out_shadow_error = ulp * (round(scaled) - phase)
        shadow_delta = out_shadow_error - h_shadow
        a_delta = actual_delta[idx]

        drift_ulp = (h_actual - h_shadow) / ulp
        crossings = _boundary_cross_count(phase + h_shadow/ulp, drift_ulp)
        match = int(_sgn(a_delta) == _sgn(shadow_delta))
        w = abs(a_delta)
        weighted_ok += w * match
        weighted_total += w

        node_depth = max(depth[node.left], depth[node.right]) + 1
        depth.append(node_depth)
        rows_raw.append((node_depth, match, abs(drift_ulp), int(crossings > 0), w))

        shadow_sum.append(exact_shadow_sum)
        shadow_error.append(out_shadow_error)
        actual_error.append(h_actual + a_delta)

    max_depth = max(depth)
    node_rows = [
        NodeSample(
            family=family,
            depth=d,
            depth_frac=d/max_depth if max_depth else 0.0,
            sign_match=m,
            history_drift_ulp=drift,
            boundary_cross=cross,
            actual_abs_delta=w,
        )
        for d,m,drift,cross,w in rows_raw
    ]
    actual_root = float(pred.signed_error)
    shadow_root = shadow_error[-1]
    return TreeSample(
        family=family,
        actual_root_error=actual_root,
        shadow_root_error=shadow_root,
        root_sign_match=int(_sgn(actual_root) == _sgn(shadow_root)),
        node_sign_match=mean(r.sign_match for r in node_rows),
        weighted_sign_match=(weighted_ok/weighted_total if weighted_total else 1.0),
        mean_abs_drift_ulp=mean(r.history_drift_ulp for r in node_rows),
        max_abs_drift_ulp=max(r.history_drift_ulp for r in node_rows),
        boundary_cross_rate=mean(r.boundary_cross for r in node_rows),
    ), node_rows


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:+.3f}"


def _tree_summary(label: str, trees: list[TreeSample]) -> None:
    if not trees:
        return
    rho = _spearman([abs(t.shadow_root_error) for t in trees], [abs(t.actual_root_error) for t in trees])
    print(
        f"  {label:<10} n={len(trees):2d} rho_shadow_abs={_fmt(rho)} "
        f"root_sign_match={mean(t.root_sign_match for t in trees):.3f} "
        f"node_sign_match={mean(t.node_sign_match for t in trees):.3f} "
        f"weighted_match={mean(t.weighted_sign_match for t in trees):.3f}"
    )
    print(
        f"    mean|history_drift|/ulp={mean(t.mean_abs_drift_ulp for t in trees):.3f} "
        f"mean_tree_max_drift/ulp={mean(t.max_abs_drift_ulp for t in trees):.3f} "
        f"boundary_cross_rate={mean(t.boundary_cross_rate for t in trees):.3f}"
    )


def _depth_summary(nodes: list[NodeSample]) -> None:
    print("  depth-conditioned node trajectory:")
    for q in range(4):
        lo, hi = q/4, (q+1)/4
        subset = [r for r in nodes if (lo < r.depth_frac <= hi) or (q == 0 and r.depth_frac == 0)]
        if not subset:
            continue
        total_w = sum(r.actual_abs_delta for r in subset)
        weighted = sum(r.actual_abs_delta*r.sign_match for r in subset)/total_w if total_w else 1.0
        print(
            f"    depthQ{q+1} n={len(subset):6d} sign_match={mean(r.sign_match for r in subset):.3f} "
            f"weighted_match={weighted:.3f} mean|drift|/ulp={mean(r.history_drift_ulp for r in subset):.3f} "
            f"cross_rate={mean(r.boundary_cross for r in subset):.3f}"
        )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    p.add_argument("--graphs", type=int, default=DEFAULT_GRAPH_COUNT)
    p.add_argument("--input-seeds", type=int, nargs="+", default=list(DEFAULT_INPUT_SEEDS))
    args = p.parse_args()
    if args.width <= 1: p.error("--width must exceed 1")
    if args.graphs <= 1: p.error("--graphs must exceed 1")

    print("Wide-range deterministic shadow-trajectory failure diagnostic")
    print("CALIBRATION ONLY — exact FP32 states used only to diagnose shadow failures")
    print(f"width={args.width} graphs_per_input={args.graphs} input_seeds={','.join(map(str,args.input_seeds))}")
    print()

    for input_index, seed in enumerate(args.input_seeds):
        generated = wide_range_random(args.width, seed=seed)
        trees: list[TreeSample] = []
        nodes: list[NodeSample] = []
        for graph_index in range(args.graphs):
            family, graph = _graph(len(generated.values), graph_index, input_index)
            t, ns = _diagnose(generated.values, graph, family)
            trees.append(t); nodes.extend(ns)
        print(f"INPUT seed={seed} family={generated.family} width={len(generated.values)}")
        _tree_summary("all", trees)
        _tree_summary("contiguous", [t for t in trees if t.family == "contiguous"])
        _tree_summary("pair_merge", [t for t in trees if t.family == "pair_merge"])
        _depth_summary(nodes)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
