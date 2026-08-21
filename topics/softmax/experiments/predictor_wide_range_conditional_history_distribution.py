"""Calibration-only conditional distribution diagnostic for standardized signed history Z=H/sigma_H.

The pooled marginal distribution looked approximately N(0,1), but that can hide structure.
This script asks whether cheap conditions (shadow phase, depth, predicted sigma_H, graph family)
change the conditional mean, scale, skew, or tails of Z enough to invalidate a single global
Gaussian model.

Exact H is inspected only as a calibration target.  No predictor is frozen and held-out data is
not used.
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
TREE_BASE_SEED = 35_000_000
PHASE_EDGES = (0.0, 0.25, 0.5, 0.75, 1.0)

@dataclass(frozen=True)
class Sample:
    family: str
    depth: int
    phase: float
    sigma_ulp: float
    z: float


def _graph(width: int, graph_index: int, input_index: int):
    seed = TREE_BASE_SEED + input_index * 10_000 + graph_index
    if graph_index % 2 == 0:
        return "contiguous", seed, random_contiguous_split_graph(width, seed=seed)
    return "pair_merge", seed, random_pair_merge_graph(width, seed=seed)


def _depths(graph: BinaryReductionGraph) -> list[int]:
    total = graph.leaf_count + len(graph.nodes)
    children = {graph.leaf_count + i: (node.left, node.right) for i, node in enumerate(graph.nodes)}
    depth = [0] * total
    stack = [(graph.root, 0)]
    while stack:
        idx, d = stack.pop()
        depth[idx] = d
        if idx in children:
            left, right = children[idx]
            stack.append((left, d + 1)); stack.append((right, d + 1))
    return depth


def _quantile(xs: list[float], q: float) -> float:
    ys = sorted(xs)
    if not ys:
        return float("nan")
    if len(ys) == 1:
        return ys[0]
    pos = q * (len(ys) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ys[lo]
    w = pos - lo
    return ys[lo] * (1 - w) + ys[hi] * w


def _std(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    m = mean(xs)
    return math.sqrt(mean((x - m) ** 2 for x in xs))


def _skew(xs: list[float]) -> float:
    if len(xs) < 3:
        return float("nan")
    m, s = mean(xs), _std(xs)
    if s == 0:
        return 0.0
    return mean(((x - m) / s) ** 3 for x in xs)


def _stats(label: str, samples: list[Sample]) -> None:
    zs = [s.z for s in samples]
    if not zs:
        return
    print(
        f"    {label:<24} n={len(zs):5d} mean={mean(zs):+.3f} med={median(zs):+.3f} "
        f"std={_std(zs):.3f} skew={_skew(zs):+.3f} "
        f"q05/q25/q75/q95={_quantile(zs,.05):+.3f}/{_quantile(zs,.25):+.3f}/"
        f"{_quantile(zs,.75):+.3f}/{_quantile(zs,.95):+.3f}"
    )


def _quartile_edges(xs: list[float]) -> tuple[float, float, float]:
    return (_quantile(xs, .25), _quantile(xs, .50), _quantile(xs, .75))


def _quartile(x: float, e: tuple[float, float, float]) -> int:
    return 0 if x <= e[0] else 1 if x <= e[1] else 2 if x <= e[2] else 3


def _conditional_report(samples: list[Sample]) -> None:
    print("  phase-conditioned Z:")
    for i in range(4):
        lo, hi = PHASE_EDGES[i], PHASE_EDGES[i+1]
        _stats(f"phase=[{lo:.2f},{hi:.2f})", [s for s in samples if lo <= s.phase < hi])

    print("  family-conditioned Z:")
    for family in ("contiguous", "pair_merge"):
        _stats(f"family={family}", [s for s in samples if s.family == family])

    d_edges = _quartile_edges([float(s.depth) for s in samples])
    s_edges = _quartile_edges([s.sigma_ulp for s in samples])
    print("  depth-conditioned Z:")
    for q in range(4):
        _stats(f"depthQ{q+1}", [s for s in samples if _quartile(float(s.depth), d_edges) == q])
    print("  sigma-conditioned Z:")
    for q in range(4):
        _stats(f"sigmaQ{q+1}", [s for s in samples if _quartile(s.sigma_ulp, s_edges) == q])

    # Compact heterogeneity indicators relative to N(0,1).
    groups = []
    for i in range(4):
        lo, hi = PHASE_EDGES[i], PHASE_EDGES[i+1]
        groups.append((f"phase{i+1}", [s for s in samples if lo <= s.phase < hi]))
    for family in ("contiguous", "pair_merge"):
        groups.append((family, [s for s in samples if s.family == family]))
    for q in range(4):
        groups.append((f"depthQ{q+1}", [s for s in samples if _quartile(float(s.depth), d_edges) == q]))
        groups.append((f"sigmaQ{q+1}", [s for s in samples if _quartile(s.sigma_ulp, s_edges) == q]))
    rows = []
    for label, group in groups:
        zs = [s.z for s in group]
        if len(zs) < 50:
            continue
        rows.append((max(abs(mean(zs)), abs(_std(zs)-1.0), abs(_skew(zs))), label, len(zs), mean(zs), _std(zs), _skew(zs)))
    rows.sort(reverse=True)
    print("  strongest conditional departures from N(0,1):")
    for score, label, n, m, sd, sk in rows[:8]:
        print(f"    {label:<12} n={n:5d} departure={score:.3f} mean={m:+.3f} std={sd:.3f} skew={sk:+.3f}")


def _collect(seed: int, input_index: int, width: int, graphs: int) -> list[Sample]:
    generated = wide_range_random(width, seed=seed)
    out: list[Sample] = []
    for graph_index in range(graphs):
        family, graph_seed, graph = _graph(width, graph_index, input_index)
        tree = diagnose_history_scale(generated.values, graph, graph_family=family, graph_seed=graph_seed)
        depths = _depths(graph)
        for node in tree.nodes:
            sigma = node.predicted_history_sigma_ulp
            if sigma <= 0:
                continue
            h = float(node.actual_history_shift / node.ulp_shadow)
            if h == 0.0:
                continue
            out.append(Sample(family, depths[node.node_index], float(node.shadow_phase), sigma, h/sigma))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    p.add_argument("--graphs", type=int, default=DEFAULT_GRAPH_COUNT)
    p.add_argument("--input-seeds", type=int, nargs="+", default=list(DEFAULT_INPUT_SEEDS))
    args = p.parse_args()
    print("Wide-range conditional signed-history distribution diagnostic")
    print("CALIBRATION ONLY — checking whether pooled N(0,1)-like behavior survives conditioning")
    print(f"width={args.width} graphs_per_input={args.graphs} input_seeds={','.join(map(str,args.input_seeds))}")
    print()
    pooled: list[Sample] = []
    for input_index, seed in enumerate(args.input_seeds):
        samples = _collect(seed, input_index, args.width, args.graphs)
        pooled.extend(samples)
        print(f"INPUT seed={seed} n={len(samples)}")
        _stats("all", samples)
        _conditional_report(samples)
        print()
    print(f"POOLED n={len(pooled)}")
    _stats("all", pooled)
    _conditional_report(pooled)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
