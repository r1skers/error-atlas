"""Calibration-only sparsity diagnostic for the coherence term C.

For exact local FP32 rounding residuals delta_v on one reduction tree,

    C = 2 * sum_{u<v} delta_u delta_v.

This diagnostic asks whether the signed coherence term is effectively controlled by a small
subset of large-|delta| internal nodes and/or a small set of dominant node pairs. It is
CALIBRATION ONLY: exact FP32 oracle residuals are intentionally inspected and no cheap predictor
is constructed here.

Reported quantities include:
- top-k node recovery of signed C and absolute pair-interaction mass,
- the ancestor share among the largest-|2 delta_u delta_v| pairs,
- concentration of pair-interaction mass on a small set of hub nodes.

The purpose is to determine whether a future cheap coherence score can focus on a sparse set of
candidate nodes rather than model all O(n^2) internal-node pairs.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from fractions import Fraction
from statistics import mean

from predictor_calibration_inputs import wide_range_random
from predictor_tree_generator import random_contiguous_split_graph, random_pair_merge_graph
from summation_graph_predictor import BinaryReductionGraph, predict_fp32_tree_error


DEFAULT_WIDTH = 256
DEFAULT_INPUT_SEEDS = (22260821, 22260822, 22260823, 22260824)
DEFAULT_GRAPH_COUNT = 64
TREE_BASE_SEED = 32_000_000
TOP_NODE_FRACS = (0.01, 0.05, 0.10, 0.20)
TOP_PAIR_FRACS = (0.01, 0.05, 0.10)
HUB_NODE_FRACS = (0.01, 0.05, 0.10)


@dataclass(frozen=True)
class PairTerm:
    u: int
    v: int
    value: Fraction
    abs_value: Fraction
    ancestor_related: bool


@dataclass(frozen=True)
class SparsityTree:
    graph_family: str
    graph_seed: int
    c_total: Fraction
    abs_pair_mass: Fraction
    node_signed_c_recovery: tuple[float, ...]
    node_abs_mass_recovery: tuple[float, ...]
    node_ancestor_signed_c_recovery: tuple[float, ...]
    top_pair_ancestor_share: tuple[float, ...]
    top_pair_ancestor_abs_mass_share: tuple[float, ...]
    hub_abs_mass_share: tuple[float, ...]


def _graph(width: int, *, graph_index: int, input_index: int):
    seed = TREE_BASE_SEED + input_index * 10_000 + graph_index
    if graph_index % 2 == 0:
        return "contiguous", seed, random_contiguous_split_graph(width, seed=seed)
    return "pair_merge", seed, random_pair_merge_graph(width, seed=seed)


def _parents_and_depths(graph: BinaryReductionGraph) -> tuple[list[int | None], list[int]]:
    value_count = graph.leaf_count + len(graph.nodes)
    parent: list[int | None] = [None] * value_count
    children: dict[int, tuple[int, int]] = {}
    for offset, node in enumerate(graph.nodes):
        idx = graph.leaf_count + offset
        children[idx] = (node.left, node.right)
        parent[node.left] = idx
        parent[node.right] = idx

    depth = [0] * value_count
    stack = [(graph.root, 0)]
    while stack:
        idx, d = stack.pop()
        depth[idx] = d
        if idx in children:
            left, right = children[idx]
            stack.append((left, d + 1))
            stack.append((right, d + 1))
    return parent, depth


def _is_ancestor(a: int, b: int, parent: list[int | None], depth: list[int]) -> bool:
    if a == b:
        return False
    if depth[a] >= depth[b]:
        return False
    cur = b
    while depth[cur] > depth[a]:
        p = parent[cur]
        if p is None:
            return False
        cur = p
    return cur == a


def _top_count(total: int, frac: float) -> int:
    return max(1, min(total, math.ceil(total * frac)))


def _safe_signed_recovery(part: Fraction, total: Fraction) -> float:
    if total == 0:
        return float("nan")
    return float(part / total)


def diagnose_tree(
    values: tuple[Fraction, ...],
    graph: BinaryReductionGraph,
    *,
    graph_family: str,
    graph_seed: int,
) -> SparsityTree:
    pred = predict_fp32_tree_error(values, graph)
    internal_indices = [graph.leaf_count + i for i in range(len(graph.nodes))]
    deltas = {idx: pred.node_predictions[i].local_rounding_error for i, idx in enumerate(internal_indices)}
    parent, depth = _parents_and_depths(graph)

    pairs: list[PairTerm] = []
    c_total = Fraction(0)
    abs_pair_mass = Fraction(0)
    for i, u in enumerate(internal_indices):
        for v in internal_indices[i + 1 :]:
            term = 2 * deltas[u] * deltas[v]
            abs_term = abs(term)
            ancestor_related = _is_ancestor(u, v, parent, depth) or _is_ancestor(v, u, parent, depth)
            pairs.append(PairTerm(u=u, v=v, value=term, abs_value=abs_term, ancestor_related=ancestor_related))
            c_total += term
            abs_pair_mass += abs_term

    expected_c = pred.signed_error * pred.signed_error - sum(
        (node.local_rounding_error * node.local_rounding_error for node in pred.node_predictions),
        start=Fraction(0),
    )
    if c_total != expected_c:
        raise AssertionError("pairwise C decomposition mismatch")

    ranked_nodes = sorted(internal_indices, key=lambda idx: abs(deltas[idx]), reverse=True)
    node_signed: list[float] = []
    node_abs: list[float] = []
    node_ancestor_signed: list[float] = []
    for frac in TOP_NODE_FRACS:
        keep = set(ranked_nodes[: _top_count(len(ranked_nodes), frac)])
        signed = sum((p.value for p in pairs if p.u in keep and p.v in keep), start=Fraction(0))
        abs_mass = sum((p.abs_value for p in pairs if p.u in keep and p.v in keep), start=Fraction(0))
        ancestor_signed = sum(
            (p.value for p in pairs if p.ancestor_related and p.u in keep and p.v in keep),
            start=Fraction(0),
        )
        node_signed.append(_safe_signed_recovery(signed, c_total))
        node_abs.append(1.0 if abs_pair_mass == 0 else float(abs_mass / abs_pair_mass))
        node_ancestor_signed.append(_safe_signed_recovery(ancestor_signed, c_total))

    ranked_pairs = sorted(pairs, key=lambda p: p.abs_value, reverse=True)
    pair_ancestor_share: list[float] = []
    pair_ancestor_abs_share: list[float] = []
    for frac in TOP_PAIR_FRACS:
        chosen = ranked_pairs[: _top_count(len(ranked_pairs), frac)]
        pair_ancestor_share.append(mean(p.ancestor_related for p in chosen))
        chosen_abs = sum((p.abs_value for p in chosen), start=Fraction(0))
        chosen_ancestor_abs = sum((p.abs_value for p in chosen if p.ancestor_related), start=Fraction(0))
        pair_ancestor_abs_share.append(
            0.0 if chosen_abs == 0 else float(chosen_ancestor_abs / chosen_abs)
        )

    incident_abs = {idx: Fraction(0) for idx in internal_indices}
    for p in pairs:
        incident_abs[p.u] += p.abs_value
        incident_abs[p.v] += p.abs_value
    ranked_hubs = sorted(internal_indices, key=lambda idx: incident_abs[idx], reverse=True)
    hub_shares: list[float] = []
    # Sum each pair once if at least one endpoint is in the selected hub set.
    for frac in HUB_NODE_FRACS:
        hubs = set(ranked_hubs[: _top_count(len(ranked_hubs), frac)])
        captured = sum((p.abs_value for p in pairs if p.u in hubs or p.v in hubs), start=Fraction(0))
        hub_shares.append(1.0 if abs_pair_mass == 0 else float(captured / abs_pair_mass))

    return SparsityTree(
        graph_family=graph_family,
        graph_seed=graph_seed,
        c_total=c_total,
        abs_pair_mass=abs_pair_mass,
        node_signed_c_recovery=tuple(node_signed),
        node_abs_mass_recovery=tuple(node_abs),
        node_ancestor_signed_c_recovery=tuple(node_ancestor_signed),
        top_pair_ancestor_share=tuple(pair_ancestor_share),
        top_pair_ancestor_abs_mass_share=tuple(pair_ancestor_abs_share),
        hub_abs_mass_share=tuple(hub_shares),
    )


def _fmt_vec(values: tuple[float, ...]) -> str:
    return "/".join("nan" if not math.isfinite(v) else f"{v:.3f}" for v in values)


def _summary(label: str, trees: list[SparsityTree]) -> None:
    if not trees:
        return
    def avg_tuple(attr: str) -> tuple[float, ...]:
        rows = [getattr(tree, attr) for tree in trees]
        out = []
        for i in range(len(rows[0])):
            vals = [row[i] for row in rows if math.isfinite(row[i])]
            out.append(mean(vals) if vals else float("nan"))
        return tuple(out)

    print(
        f"  {label:<10} n={len(trees):2d} "
        f"topNode_C_recovery[1/5/10/20%]={_fmt_vec(avg_tuple('node_signed_c_recovery'))} "
        f"topNode_absPair[1/5/10/20%]={_fmt_vec(avg_tuple('node_abs_mass_recovery'))}"
    )
    print(
        f"    topNode_ancestorC/C[1/5/10/20%]={_fmt_vec(avg_tuple('node_ancestor_signed_c_recovery'))} "
        f"topPair_ancestorFrac[1/5/10%]={_fmt_vec(avg_tuple('top_pair_ancestor_share'))} "
        f"topPair_ancestorAbs[1/5/10%]={_fmt_vec(avg_tuple('top_pair_ancestor_abs_mass_share'))}"
    )
    print(
        f"    hub_absPair_capture[top1/5/10% nodes]={_fmt_vec(avg_tuple('hub_abs_mass_share'))}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--graphs", type=int, default=DEFAULT_GRAPH_COUNT)
    parser.add_argument("--input-seeds", type=int, nargs="+", default=list(DEFAULT_INPUT_SEEDS))
    args = parser.parse_args()
    if args.width <= 1:
        parser.error("--width must exceed 1")
    if args.graphs <= 1:
        parser.error("--graphs must exceed 1")
    return args


def main() -> int:
    args = _parse_args()
    print("Wide-range coherence sparsity diagnostic")
    print("CALIBRATION ONLY — exact FP32 residuals are intentionally inspected")
    print("Node selection is by oracle |delta| only to diagnose sparsity, not as a predictor")
    print(
        f"width={args.width} graphs_per_input={args.graphs} "
        f"input_seeds={','.join(str(seed) for seed in args.input_seeds)}"
    )
    print()

    for input_index, input_seed in enumerate(args.input_seeds):
        generated = wide_range_random(args.width, seed=input_seed)
        trees: list[SparsityTree] = []
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

        print(f"INPUT seed={input_seed} family={generated.family} width={len(generated.values)}")
        _summary("all", trees)
        _summary("contiguous", [t for t in trees if t.graph_family == "contiguous"])
        _summary("pair_merge", [t for t in trees if t.graph_family == "pair_merge"])
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
