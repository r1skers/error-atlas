"""Calibration-only sparse repair ablation for the deterministic shadow trajectory.

The preceding shadow diagnostic showed that most shallow nodes preserve rounding direction, while
history drift, boundary crossings, and sign mismatches concentrate near the top of the tree.  This
script asks whether tree-level ranking can be recovered by repairing only a sparse subset of
internal nodes with oracle history.

This is NOT a predictor: repaired nodes intentionally use exact FP32 accumulated history.  The goal
is purely mechanistic: quantify how many nodes must be corrected before the shadow ranking recovers.

Repair policies:
  * depth: deepest/topmost nodes by distance from leaves
  * oracle_drift: nodes with largest |actual_history - shadow_history| / ulp
  * oracle_cross: nodes whose shadow->actual history shift crosses a rounding boundary, then drift

Fractions 1%, 5%, 10%, 20% are tested.  At repaired nodes we inject the exact oracle subtree output;
unrepaired nodes continue the deterministic shadow recursion from their current children.
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
    dx = [v-mx for v in x]
    dy = [v-my for v in y]
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
    while value < (Fraction(2) ** e if e >= 0 else Fraction(1, 2**(-e))):
        e -= 1
    if e < -126:
        return Fraction(1, 2**149)
    qexp = e - 23
    return Fraction(2**qexp) if qexp >= 0 else Fraction(1, 2**(-qexp))


def _distance_to_boundary(shadow_sum: Fraction, ulp: Fraction) -> float:
    scaled = shadow_sum / ulp
    floor = scaled.numerator // scaled.denominator
    phase = float(scaled - floor)
    return min(phase, abs(0.5-phase), 1.0-phase)


def _topological_depths(graph: BinaryReductionGraph) -> dict[int, int]:
    d = {i: 0 for i in range(graph.leaf_count)}
    for off,node in enumerate(graph.nodes):
        idx = graph.leaf_count + off
        d[idx] = max(d[node.left], d[node.right]) + 1
    return d


def _analyze(values: tuple[Fraction,...], graph: BinaryReductionGraph, family: str) -> TreeRow:
    oracle = predict_fp32_tree_error(values, graph)
    nleaf = graph.leaf_count
    exact_total = sum(values, start=Fraction(0))

    actual_out = [*values]
    for pred in oracle.node_predictions:
        actual_out.append(pred.rounded_sum)

    shadow_out = [*values]
    shadow_history: dict[int, Fraction] = {}
    actual_history: dict[int, Fraction] = {}
    drift_ulp: dict[int, float] = {}
    crossed: dict[int, bool] = {}
    depths = _topological_depths(graph)

    # Exact subtree input sums, independent of execution history.
    exact_subtree = [*values]
    for off,node in enumerate(graph.nodes):
        idx = nleaf + off
        exact_sum = exact_subtree[node.left] + exact_subtree[node.right]
        exact_subtree.append(exact_sum)

        ssum = shadow_out[node.left] + shadow_out[node.right]
        sout = round_nonnegative_fraction_to_fp32(ssum).value
        shadow_out.append(sout)

        ah = (actual_out[node.left] - exact_subtree[node.left]) + (actual_out[node.right] - exact_subtree[node.right])
        sh = (shadow_out[node.left] - exact_subtree[node.left]) + (shadow_out[node.right] - exact_subtree[node.right])
        actual_history[idx] = ah
        shadow_history[idx] = sh
        ulp = _fp32_ulp_fraction(exact_sum)
        drift = ah - sh
        drift_ulp[idx] = float(abs(drift / ulp))

        # Crossing test against the nearest RN boundary using exact scaled positions.
        x0 = (exact_sum + sh) / ulp
        x1 = (exact_sum + ah) / ulp
        lo, hi = sorted((float(x0), float(x1)))
        first = math.floor(lo - 0.5) + 0.5
        crossed[idx] = first <= hi and first > lo + 1e-15

    shadow_error = abs(float(shadow_out[-1] - exact_total))
    internal = [nleaf+i for i in range(len(graph.nodes))]

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
            out = [*values]
            for off,node in enumerate(graph.nodes):
                idx = nleaf + off
                if idx in repair:
                    out.append(actual_out[idx])
                else:
                    s = out[node.left] + out[node.right]
                    out.append(round_nonnegative_fraction_to_fp32(s).value)
            repaired_scores[(policy,frac)] = abs(float(out[-1] - exact_total))

    target = abs(float(oracle.signed_error))
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

    print('Sparse shadow repair ablation')
    print('CALIBRATION ONLY — repaired nodes use oracle FP32 history/output; not a predictor')
    print('Question: can sparse high-risk repairs recover tree ranking?')
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
