"""Calibration-only diagnostic for the signed rounding-history distribution.

The previous history-scale diagnostic estimates sigma_H cheaply but not the sign of the true
rounding history

    H_v = (actual FP32 addend sum at v) - (exact mathematical subtree sum at v).

This script asks whether the standardized history

    Z_v = H_v / sigma_hat(H_v)

looks roughly symmetric/zero-centered, or whether cheap conditions such as graph family, node
depth, shadow phase, or predicted sigma expose a systematic sign bias P(H_v > 0) != 1/2.

Exact H_v is used only as a CALIBRATION target. No predictor is frozen and held-out data is not
used.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from statistics import mean, median

from predictor_calibration_inputs import wide_range_random
from predictor_tree_generator import random_contiguous_split_graph, random_pair_merge_graph
from predictor_wide_range_history_scale_diagnostic import diagnose_history_scale
from summation_graph_predictor import BinaryReductionGraph

DEFAULT_WIDTH = 256
DEFAULT_INPUT_SEEDS = (22260821, 22260822, 22260823, 22260824)
DEFAULT_GRAPH_COUNT = 64
TREE_BASE_SEED = 34_000_000
PHASE_EDGES = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class Sample:
    family: str
    depth: int
    phase: float
    sigma_ulp: float
    h_ulp: float
    z: float


def _graph(width: int, graph_index: int, input_index: int):
    seed = TREE_BASE_SEED + input_index * 10_000 + graph_index
    if graph_index % 2 == 0:
        return "contiguous", seed, random_contiguous_split_graph(width, seed=seed)
    return "pair_merge", seed, random_pair_merge_graph(width, seed=seed)


def _depths(graph: BinaryReductionGraph) -> list[int]:
    total = graph.leaf_count + len(graph.nodes)
    children: dict[int, tuple[int, int]] = {}
    for offset, node in enumerate(graph.nodes):
        children[graph.leaf_count + offset] = (node.left, node.right)
    depth = [0] * total
    stack = [(graph.root, 0)]
    while stack:
        idx, d = stack.pop()
        depth[idx] = d
        pair = children.get(idx)
        if pair is not None:
            stack.append((pair[0], d + 1))
            stack.append((pair[1], d + 1))
    return depth


def _quantile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    pos = q * (len(ys) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ys[lo]
    w = pos - lo
    return ys[lo] * (1.0 - w) + ys[hi] * w


def _std(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    m = mean(xs)
    return math.sqrt(mean((x - m) ** 2 for x in xs))


def _skew(xs: list[float]) -> float:
    if len(xs) < 3:
        return float("nan")
    m = mean(xs)
    s = _std(xs)
    if s == 0.0:
        return 0.0
    return mean(((x - m) / s) ** 3 for x in xs)


def _sign_stats(samples: list[Sample]) -> tuple[int, float, float, float]:
    nonzero = [s for s in samples if s.h_ulp != 0.0]
    if not nonzero:
        return 0, float("nan"), float("nan"), float("nan")
    ppos = mean(s.h_ulp > 0 for s in nonzero)
    bias = abs(ppos - 0.5)
    zmean = mean(s.z for s in nonzero)
    return len(nonzero), ppos, bias, zmean


def _print_distribution(label: str, samples: list[Sample]) -> None:
    nonzero = [s for s in samples if s.h_ulp != 0.0]
    zs = [s.z for s in nonzero]
    n, ppos, bias, zmean = _sign_stats(samples)
    if not zs:
        print(f"  {label:<12} n=0")
        return
    print(
        f"  {label:<12} n={n:5d} P(H>0)={ppos:.3f} bias={bias:.3f} "
        f"meanZ={zmean:+.3f} medianZ={median(zs):+.3f} stdZ={_std(zs):.3f} skewZ={_skew(zs):+.3f} "
        f"q05/q25/q75/q95={_quantile(zs,.05):+.3f}/{_quantile(zs,.25):+.3f}/"
        f"{_quantile(zs,.75):+.3f}/{_quantile(zs,.95):+.3f}"
    )


def _phase_bin(phase: float) -> int:
    for i in range(len(PHASE_EDGES) - 1):
        if PHASE_EDGES[i] <= phase < PHASE_EDGES[i + 1]:
            return i
    return len(PHASE_EDGES) - 2


def _quartile_edges(values: list[float]) -> tuple[float, float, float]:
    return (_quantile(values, 0.25), _quantile(values, 0.50), _quantile(values, 0.75))


def _quartile(value: float, edges: tuple[float, float, float]) -> int:
    if value <= edges[0]:
        return 0
    if value <= edges[1]:
        return 1
    if value <= edges[2]:
        return 2
    return 3


def _conditional_bias_report(samples: list[Sample]) -> None:
    candidates: list[tuple[str, int, float, float]] = []

    for family in ("contiguous", "pair_merge"):
        group = [s for s in samples if s.family == family]
        n, p, b, _ = _sign_stats(group)
        if n:
            candidates.append((f"family={family}", n, p, b))

    for i in range(4):
        lo, hi = PHASE_EDGES[i], PHASE_EDGES[i + 1]
        group = [s for s in samples if _phase_bin(s.phase) == i]
        n, p, b, _ = _sign_stats(group)
        if n:
            candidates.append((f"phase=[{lo:.2f},{hi:.2f})", n, p, b))

    depths = [float(s.depth) for s in samples]
    sigmas = [s.sigma_ulp for s in samples]
    d_edges = _quartile_edges(depths)
    s_edges = _quartile_edges(sigmas)
    for q in range(4):
        group = [s for s in samples if _quartile(float(s.depth), d_edges) == q]
        n, p, b, _ = _sign_stats(group)
        if n:
            candidates.append((f"depthQ{q+1}", n, p, b))
        group = [s for s in samples if _quartile(s.sigma_ulp, s_edges) == q]
        n, p, b, _ = _sign_stats(group)
        if n:
            candidates.append((f"sigmaQ{q+1}", n, p, b))

    candidates.sort(key=lambda row: row[3], reverse=True)
    print("  strongest cheap-condition sign biases:")
    for label, n, p, b in candidates[:8]:
        print(f"    {label:<24} n={n:5d} P(H>0)={p:.3f} bias={b:.3f}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    p.add_argument("--graphs", type=int, default=DEFAULT_GRAPH_COUNT)
    p.add_argument("--input-seeds", type=int, nargs="+", default=list(DEFAULT_INPUT_SEEDS))
    args = p.parse_args()
    if args.width <= 1:
        p.error("--width must exceed 1")
    if args.graphs <= 1:
        p.error("--graphs must exceed 1")

    print("Wide-range signed-history distribution diagnostic")
    print("CALIBRATION ONLY — true signed H is inspected only as a target")
    print("Cheap side uses graph, exact subtree sums, phase, depth, and sigma_H")
    print(f"width={args.width} graphs_per_input={args.graphs} input_seeds={','.join(map(str,args.input_seeds))}")
    print()

    pooled: list[Sample] = []
    for input_index, seed in enumerate(args.input_seeds):
        generated = wide_range_random(args.width, seed=seed)
        samples: list[Sample] = []
        for graph_index in range(args.graphs):
            family, graph_seed, graph = _graph(len(generated.values), graph_index, input_index)
            tree = diagnose_history_scale(
                generated.values,
                graph,
                graph_family=family,
                graph_seed=graph_seed,
            )
            depths = _depths(graph)
            for node in tree.nodes:
                sigma = node.predicted_history_sigma_ulp
                if sigma <= 0.0:
                    continue
                h = float(node.actual_history_shift / node.ulp_shadow)
                idx = node.node_index
                samples.append(
                    Sample(
                        family=family,
                        depth=depths[idx],
                        phase=float(node.shadow_phase),
                        sigma_ulp=sigma,
                        h_ulp=h,
                        z=h / sigma,
                    )
                )
        pooled.extend(samples)
        print(f"INPUT seed={seed} family={generated.family} width={len(generated.values)}")
        _print_distribution("all", samples)
        _print_distribution("contiguous", [s for s in samples if s.family == "contiguous"])
        _print_distribution("pair_merge", [s for s in samples if s.family == "pair_merge"])
        _conditional_bias_report(samples)
        print()

    print("POOLED SUMMARY")
    _print_distribution("all", pooled)
    _print_distribution("contiguous", [s for s in pooled if s.family == "contiguous"])
    _print_distribution("pair_merge", [s for s in pooled if s.family == "pair_merge"])
    _conditional_bias_report(pooled)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
