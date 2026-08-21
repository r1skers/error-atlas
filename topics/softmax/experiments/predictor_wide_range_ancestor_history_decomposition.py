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
from dataclasses import dataclass
from fractions import Fraction
from statistics import mean

from predictor_calibration_inputs import wide_range_random
from predictor_tree_generator import random_contiguous_split_graph, random_pair_merge_graph
from summation_graph_predictor import BinaryReductionGraph, predict_fp32_tree_error

DEFAULT_WIDTH = 256
DEFAULT_INPUT_SEEDS = (22260821, 22260822, 22260823, 22260824)
DEFAULT_GRAPH_COUNT = 64
TREE_BASE_SEED = 33_000_000
TOP_FRACS = (0.01, 0.05, 0.10, 0.20)


@dataclass(frozen=True)
class TreeDiagnostic:
    graph_family: str
    c_ancestor: Fraction
    k_total: Fraction
    c_total: Fraction
    abs_k_mass: Fraction
    top_abs_k_mass: tuple[float, ...]
    top_signed_k_recovery: tuple[float, ...]
    rho_abs_k_vs_abs_delta: float
    rho_abs_k_vs_abs_history: float


def _graph(width: int, graph_index: int, input_index: int):
    seed = TREE_BASE_SEED + input_index * 10_000 + graph_index
    if graph_index % 2 == 0:
        return "contiguous", random_contiguous_split_graph(width, seed=seed)
    return "pair_merge", random_pair_merge_graph(width, seed=seed)


def _rankdata(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=xs.__getitem__)
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and xs[order[j]] == xs[order[i]]:
            j += 1
        rank = (i + j - 1) / 2.0
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    rx, ry = _rankdata(xs), _rankdata(ys)
    mx, my = mean(rx), mean(ry)
    vx = sum((x - mx) ** 2 for x in rx)
    vy = sum((y - my) ** 2 for y in ry)
    if vx == 0 or vy == 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(rx, ry)) / math.sqrt(vx * vy)


def _top_count(n: int, frac: float) -> int:
    return max(1, min(n, math.ceil(n * frac)))


def diagnose(values: tuple[Fraction, ...], graph: BinaryReductionGraph, family: str) -> TreeDiagnostic:
    pred = predict_fp32_tree_error(values, graph)
    leaf_count = graph.leaf_count
    internal = [leaf_count + i for i in range(len(graph.nodes))]
    delta = {idx: pred.node_predictions[i].local_rounding_error for i, idx in enumerate(internal)}

    # Recursive subtree history. history[v] contains all proper internal descendants' residuals.
    subtree_delta: dict[int, Fraction] = {}
    history: dict[int, Fraction] = {}
    k: dict[int, Fraction] = {}
    for offset, node in enumerate(graph.nodes):
        idx = leaf_count + offset
        left_sub = subtree_delta.get(node.left, Fraction(0))
        right_sub = subtree_delta.get(node.right, Fraction(0))
        history[idx] = left_sub + right_sub
        k[idx] = 2 * delta[idx] * history[idx]
        subtree_delta[idx] = history[idx] + delta[idx]

    k_total = sum(k.values(), start=Fraction(0))

    # Independent pairwise ancestor calculation for an exact identity check.
    parent: list[int | None] = [None] * (leaf_count + len(graph.nodes))
    for offset, node in enumerate(graph.nodes):
        idx = leaf_count + offset
        parent[node.left] = idx
        parent[node.right] = idx

    def is_ancestor(a: int, b: int) -> bool:
        cur = b
        while parent[cur] is not None:
            cur = parent[cur]  # type: ignore[assignment]
            if cur == a:
                return True
        return False

    c_ancestor = Fraction(0)
    c_total = Fraction(0)
    for i, u in enumerate(internal):
        for v in internal[i + 1:]:
            term = 2 * delta[u] * delta[v]
            c_total += term
            if is_ancestor(u, v) or is_ancestor(v, u):
                c_ancestor += term
    if k_total != c_ancestor:
        raise AssertionError(f"ancestor/history identity mismatch: {k_total} != {c_ancestor}")

    abs_k_mass = sum((abs(x) for x in k.values()), start=Fraction(0))
    ranked = sorted(internal, key=lambda idx: abs(k[idx]), reverse=True)
    abs_recovery: list[float] = []
    signed_recovery: list[float] = []
    for frac in TOP_FRACS:
        chosen = ranked[:_top_count(len(ranked), frac)]
        part_abs = sum((abs(k[idx]) for idx in chosen), start=Fraction(0))
        part_signed = sum((k[idx] for idx in chosen), start=Fraction(0))
        abs_recovery.append(1.0 if abs_k_mass == 0 else float(part_abs / abs_k_mass))
        signed_recovery.append(float("nan") if k_total == 0 else float(part_signed / k_total))

    abs_k = [float(abs(k[idx])) for idx in internal]
    abs_delta = [float(abs(delta[idx])) for idx in internal]
    abs_history = [float(abs(history[idx])) for idx in internal]
    return TreeDiagnostic(
        graph_family=family,
        c_ancestor=c_ancestor,
        k_total=k_total,
        c_total=c_total,
        abs_k_mass=abs_k_mass,
        top_abs_k_mass=tuple(abs_recovery),
        top_signed_k_recovery=tuple(signed_recovery),
        rho_abs_k_vs_abs_delta=_spearman(abs_k, abs_delta),
        rho_abs_k_vs_abs_history=_spearman(abs_k, abs_history),
    )


def _fmt(v: float) -> str:
    return "nan" if not math.isfinite(v) else f"{v:+.3f}"


def _vec(vs: tuple[float, ...]) -> str:
    return "/".join("nan" if not math.isfinite(v) else f"{v:.3f}" for v in vs)


def _summary(label: str, trees: list[TreeDiagnostic]) -> None:
    if not trees:
        return
    def avg_tuple(attr: str) -> tuple[float, ...]:
        rows = [getattr(t, attr) for t in trees]
        return tuple(mean(row[i] for row in rows if math.isfinite(row[i])) for i in range(len(rows[0])))
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
    p.add_argument("--input-seeds", type=int, nargs="+", default=list(DEFAULT_INPUT_SEEDS))
    args = p.parse_args()
    if args.width <= 1:
        p.error("--width must exceed 1")
    if args.graphs <= 1:
        p.error("--graphs must exceed 1")

    print("Wide-range ancestor/history coherence decomposition")
    print("CALIBRATION ONLY — exact FP32 residuals/history are intentionally inspected")
    print("Testing C_ancestor == 2 sum_v delta_v H_v and concentration of K_v")
    print(f"width={args.width} graphs_per_input={args.graphs} input_seeds={','.join(map(str,args.input_seeds))}")
    print()

    for input_index, seed in enumerate(args.input_seeds):
        generated = wide_range_random(args.width, seed=seed)
        trees: list[TreeDiagnostic] = []
        for graph_index in range(args.graphs):
            family, graph = _graph(len(generated.values), graph_index, input_index)
            trees.append(diagnose(generated.values, graph, family))
        print(f"INPUT seed={seed} family={generated.family} width={len(generated.values)}")
        _summary("all", trees)
        _summary("contiguous", [t for t in trees if t.graph_family == "contiguous"])
        _summary("pair_merge", [t for t in trees if t.graph_family == "pair_merge"])
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
