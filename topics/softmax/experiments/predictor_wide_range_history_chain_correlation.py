"""Calibration-only diagnostic for joint dependence along ancestor history chains.

The standardized marginal history Z_v = H_v / sigma_hat(H_v) looks close to N(0,1), even after
conditioning on simple cheap covariates. This script asks whether the missing information is in
the joint structure along ancestor chains rather than in the one-node marginal distribution.

For internal-node ancestor/descendant pairs, report Pearson correlations of standardized histories
Z at ancestor gaps 1, 2, 3, and 4+, together with same-sign probabilities. Results are pooled and
split by graph family. Exact H is inspected only as a CALIBRATION target.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from statistics import mean

from predictor_calibration_inputs import wide_range_random
from predictor_tree_generator import random_contiguous_split_graph, random_pair_merge_graph
from predictor_wide_range_history_scale_diagnostic import diagnose_history_scale
from summation_graph_predictor import BinaryReductionGraph

DEFAULT_WIDTH = 256
DEFAULT_INPUT_SEEDS = (22260821, 22260822, 22260823, 22260824)
DEFAULT_GRAPH_COUNT = 64
TREE_BASE_SEED = 36_000_000


@dataclass(frozen=True)
class PairSample:
    family: str
    gap_bucket: str
    z_desc: float
    z_anc: float


def _graph(width: int, graph_index: int, input_index: int):
    seed = TREE_BASE_SEED + input_index * 10_000 + graph_index
    if graph_index % 2 == 0:
        return "contiguous", seed, random_contiguous_split_graph(width, seed=seed)
    return "pair_merge", seed, random_pair_merge_graph(width, seed=seed)


def _parents(graph: BinaryReductionGraph) -> list[int | None]:
    total = graph.leaf_count + len(graph.nodes)
    parent: list[int | None] = [None] * total
    for offset, node in enumerate(graph.nodes):
        idx = graph.leaf_count + offset
        parent[node.left] = idx
        parent[node.right] = idx
    return parent


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    mx, my = mean(xs), mean(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    sx = math.sqrt(sum(x * x for x in dx))
    sy = math.sqrt(sum(y * y for y in dy))
    if sx == 0.0 or sy == 0.0:
        return float("nan")
    return sum(x * y for x, y in zip(dx, dy)) / (sx * sy)


def _bucket(gap: int) -> str:
    if gap <= 3:
        return f"gap{gap}"
    return "gap4+"


def _report(label: str, samples: list[PairSample]) -> None:
    print(f"  {label}")
    for bucket in ("gap1", "gap2", "gap3", "gap4+"):
        group = [s for s in samples if s.gap_bucket == bucket]
        if not group:
            print(f"    {bucket:<5} n=0")
            continue
        xd = [s.z_desc for s in group]
        xa = [s.z_anc for s in group]
        corr = _pearson(xd, xa)
        same = mean((x > 0) == (y > 0) for x, y in zip(xd, xa) if x != 0 and y != 0)
        mean_product = mean(x * y for x, y in zip(xd, xa))
        print(
            f"    {bucket:<5} n={len(group):7d} corrZ={corr:+.3f} "
            f"P(same_sign)={same:.3f} mean(Zd*Za)={mean_product:+.3f}"
        )


def collect(values, graph: BinaryReductionGraph, family: str, graph_seed: int) -> list[PairSample]:
    tree = diagnose_history_scale(values, graph, graph_family=family, graph_seed=graph_seed)
    z_by_idx: dict[int, float] = {}
    for node in tree.nodes:
        sigma = node.predicted_history_sigma_ulp
        if sigma <= 0.0:
            continue
        h_ulp = float(node.actual_history_shift / node.ulp_shadow)
        z_by_idx[node.node_index] = h_ulp / sigma

    parent = _parents(graph)
    internal_start = graph.leaf_count
    out: list[PairSample] = []
    for desc, z_desc in z_by_idx.items():
        cur = parent[desc]
        gap = 1
        while cur is not None:
            if cur >= internal_start and cur in z_by_idx:
                out.append(PairSample(family=family, gap_bucket=_bucket(gap), z_desc=z_desc, z_anc=z_by_idx[cur]))
            cur = parent[cur]
            gap += 1
    return out


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

    print("Wide-range ancestor history joint-correlation diagnostic")
    print("CALIBRATION ONLY — true signed H is inspected only as a target")
    print("Testing whether standardized histories Z remain correlated along ancestor chains")
    print(f"width={args.width} graphs_per_input={args.graphs} input_seeds={','.join(map(str,args.input_seeds))}")
    print()

    pooled: list[PairSample] = []
    for input_index, seed in enumerate(args.input_seeds):
        generated = wide_range_random(args.width, seed=seed)
        samples: list[PairSample] = []
        for graph_index in range(args.graphs):
            family, graph_seed, graph = _graph(len(generated.values), graph_index, input_index)
            samples.extend(collect(generated.values, graph, family, graph_seed))
        pooled.extend(samples)
        print(f"INPUT seed={seed} family={generated.family} width={len(generated.values)}")
        _report("all", samples)
        _report("contiguous", [s for s in samples if s.family == "contiguous"])
        _report("pair_merge", [s for s in samples if s.family == "pair_merge"])
        print()

    print("POOLED SUMMARY")
    _report("all", pooled)
    _report("contiguous", [s for s in pooled if s.family == "contiguous"])
    _report("pair_merge", [s for s in pooled if s.family == "pair_merge"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
