"""Calibration-only sparse local-residual repair ablation for the shadow model.

The shadow model intentionally ignores descendant rounding history at every internal node:

    delta_shadow(v) = RN32(S_v) - S_v,

where S_v is the exact stored-FP32 subtree sum.  The true oracle residual is

    delta_actual(v) = RN32(S_v + H_v) - (S_v + H_v).

Because the exact root error is the sum of true local residuals, this ablation asks whether replacing
only a sparse subset of shadow residuals by their oracle counterparts is enough to recover tree-level
ranking:

    E_repaired = sum_{v in R} delta_actual(v) + sum_{v not in R} delta_shadow(v).

This is NOT a predictor.  Oracle histories/residuals are intentionally used only to diagnose whether
shadow failure is sparse.  Importantly, the script never recursively replays RN32(child outputs),
because doing so would exactly execute the candidate tree and trivially reproduce the oracle.

Repair policies:
  * depth: topmost nodes by distance from leaves
  * oracle_drift: largest |H_v| / ulp(S_v)
  * oracle_cross: nodes where H_v crosses an RN boundary, then largest drift

Fractions 1%, 5%, 10%, 20% are tested.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from fractions import Fraction
from statistics import mean

from predictor_calibration_inputs import wide_range_random
from predictor_tree_generator import random_contiguous_split_graph, random_pair_merge_graph
from summation_graph_predictor import BinaryReductionGraph, predict_fp32_tree_error, round_nonnegative_fraction_to_fp32

DEFAULT_WIDTH = 256
DEFAULT_GRAPH_COUNT = 64
DEFAULT_INPUT_SEEDS = (22260821, 22260822, 22260823, 22260824)
TREE_BASE_SEED = 37_000_000
FRACTIONS = (0.01, 0.05, 0.10, 0.20)


@dataclass(frozen=True)
class TreeRow:
    family: str
    target: float
    shadow: float
    repaired: dict[tuple[str, float], float]


def _graph(width: int, graph_index: int, input_index: int):
    seed = TREE_BASE_SEED + input_index * 10_000 + graph_index
    if graph_index % 2 == 0:
        return "contiguous", random_contiguous_split_graph(width, seed=seed)
    return "pair_merge", random_pair_merge_graph(width, seed=seed)


def _rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        r = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = r
        i = j
    return ranks


def _pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    mx, my = mean(x), mean(y)
    dx = [v - mx for v in x]
    dy = [v - my for v in y]
    sx = math.sqrt(sum(v*v for v in dx))
    sy = math.sqrt(sum(v*v for v in dy))
    if sx == 0 or sy == 0:
        return None
    return sum(a*b for a,b in zip(dx,dy,strict=True))/(sx*sy)


def _spearman(x: list[float], y: list[float]) -> float | None:
    return _pearson(_rankdata(x), _rankdata(y))


def _fp32_ulp_fraction(value: Fraction) -> Fraction:
    if value <= 0:
        raise ValueError("positive sums only")
    e = value.numerator.bit_length() - value.denominator.bit_length()
    p2 = Fraction(2**e) if e >= 0 else Fraction(1, 2**(-e))
    if value < p2:
        e -= 1
    if e < -126:
        return Fraction(1, 2**149)
    qexp = e - 23
    return Fraction(2**qexp) if qexp >= 0 else Fraction(1, 2**(-qexp))


def _topological_depths(graph: BinaryReductionGraph) -> dict[int, int]:
    d = {i: 0 for i in range(graph.leaf_count)}
    for off,node in enumerate(graph.nodes):
        idx = graph.leaf_count + off
        d[idx] = max(d[node.left], d[node.right]) + 1
    return d


def _crosses_boundary(x0: Fraction, x1: Fraction) -> bool:
    """Whether the exact shift changes the RN-even quantization cell.

    Comparing the rounded integer coordinates handles half-integer ties exactly and avoids
    converting the (roughly 24-bit) FP32-grid coordinate to a binary64 approximation.
    """
    return round(x0) != round(x1)


def _analyze(values: tuple[Fraction,...], graph: BinaryReductionGraph, family: str) -> TreeRow:
    oracle = predict_fp32_tree_error(values, graph)
    nleaf = graph.leaf_count
    internal = [nleaf+i for i in range(len(graph.nodes))]
    depths = _topological_depths(graph)

    exact_subtree = [*values]
    actual_output = [*values]
    actual_delta: dict[int, Fraction] = {}
    shadow_delta: dict[int, Fraction] = {}
    drift_ulp: dict[int, float] = {}
    crossed: dict[int, bool] = {}

    for off, (node, pred) in enumerate(zip(graph.nodes, oracle.node_predictions, strict=True)):
        idx = nleaf + off
        exact_sum = exact_subtree[node.left] + exact_subtree[node.right]
        exact_subtree.append(exact_sum)

        actual_output.append(pred.rounded_sum)
        actual_delta[idx] = pred.local_rounding_error

        shadow_rounded = round_nonnegative_fraction_to_fp32(exact_sum).value
        shadow_delta[idx] = shadow_rounded - exact_sum

        history = (
            actual_output[node.left] - exact_subtree[node.left]
            + actual_output[node.right] - exact_subtree[node.right]
        )
        ulp = _fp32_ulp_fraction(exact_sum)
        drift_ulp[idx] = float(abs(history / ulp))
        crossed[idx] = _crosses_boundary(exact_sum / ulp, (exact_sum + history) / ulp)

    # Hard identities/guards: target must equal sum(actual delta), while shadow is a distinct
    # history-free local model rather than a replay of candidate execution.
    actual_sum = sum((actual_delta[i] for i in internal), start=Fraction(0))
    if actual_sum != oracle.signed_error:
        raise AssertionError("oracle local-residual identity failed")
    shadow_sum = sum((shadow_delta[i] for i in internal), start=Fraction(0))

    target = abs(float(actual_sum))
    shadow_error = abs(float(shadow_sum))

    policy_orders = {
        "depth": sorted(internal, key=lambda i: depths[i], reverse=True),
        "oracle_drift": sorted(internal, key=lambda i: drift_ulp[i], reverse=True),
        "oracle_cross": sorted(internal, key=lambda i: (crossed[i], drift_ulp[i]), reverse=True),
    }

    repaired_scores: dict[tuple[str,float], float] = {}
    for policy, order in policy_orders.items():
        for frac in FRACTIONS:
            k = max(1, math.ceil(len(internal)*frac))
            repair = set(order[:k])
            repaired_sum = sum(
                (actual_delta[i] if i in repair else shadow_delta[i] for i in internal),
                start=Fraction(0),
            )
            repaired_scores[(policy,frac)] = abs(float(repaired_sum))

    return TreeRow(family=family,target=target,shadow=shadow_error,repaired=repaired_scores)


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:+.3f}"


def _report(label: str, rows: list[TreeRow]) -> dict[str,float|None]:
    target=[r.target for r in rows]
    out: dict[str,float|None] = {"shadow": _spearman([r.shadow for r in rows], target)}
    print(f"  {label:<10} n={len(rows):2d} rho_shadow={_fmt(out['shadow'])}")
    for policy in ("depth","oracle_drift","oracle_cross"):
        vals=[]
        for frac in FRACTIONS:
            rho=_spearman([r.repaired[(policy,frac)] for r in rows],target)
            out[f"{policy}_{frac}"]=rho
            vals.append(_fmt(rho))
        print(f"    {policy:<12} rho_repair[1/5/10/20%]={'/'.join(vals)}")
    return out


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--width',type=int,default=DEFAULT_WIDTH)
    p.add_argument('--graphs',type=int,default=DEFAULT_GRAPH_COUNT)
    p.add_argument('--input-seeds',type=int,nargs='+',default=list(DEFAULT_INPUT_SEEDS))
    args=p.parse_args()
    if args.width<=1: p.error('--width must exceed 1')
    if args.graphs<=1: p.error('--graphs must exceed 1')

    print('Sparse shadow local-residual repair ablation')
    print('CALIBRATION ONLY — repaired residuals use oracle FP32 history; not a predictor')
    print('Shadow = sum of history-free per-node residuals; candidate execution is never replayed')
    print(f"width={args.width} graphs_per_input={args.graphs} input_seeds={','.join(map(str,args.input_seeds))}")
    print()

    pooled: dict[str,list[float]]={}
    for input_index,seed in enumerate(args.input_seeds):
        generated=wide_range_random(args.width,seed=seed)
        rows=[]
        for gi in range(args.graphs):
            family,graph=_graph(len(generated.values),gi,input_index)
            rows.append(_analyze(generated.values,graph,family))
        print(f"INPUT seed={seed} family={generated.family} width={len(generated.values)}")
        stats=_report('all',rows)
        _report('contiguous',[r for r in rows if r.family=='contiguous'])
        _report('pair_merge',[r for r in rows if r.family=='pair_merge'])
        for k,v in stats.items():
            if v is not None: pooled.setdefault(k,[]).append(v)
        print()

    print('SEED SUMMARY all-tree rho mean/min/max')
    for key,vals in pooled.items():
        print(f"  {key:<24} mean={mean(vals):+.3f} min={min(vals):+.3f} max={max(vals):+.3f}")
    return 0

if __name__=='__main__':
    raise SystemExit(main())
