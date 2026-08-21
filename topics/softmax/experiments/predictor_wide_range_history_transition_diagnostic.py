"""Calibration-only diagnostic for the exact history transition across tree levels.

For an internal parent p with children l,r, define each internal child's accumulated history H
and local residual delta. Then the exact parent history is

    H_p = I_p + J_p,
    I_p = H_l + H_r,
    J_p = delta_l + delta_r,

where leaf children contribute zero to both terms.

This script verifies the identity and measures how much of parent-history variation is inherited
state versus new local rounding innovation, including their covariance. It is CALIBRATION ONLY:
true FP32 histories/residuals are intentionally inspected and no predictor evidence is claimed.
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
TREE_BASE_SEED = 35_000_000


@dataclass(frozen=True)
class Sample:
    family: str
    parent_h: float
    inherited: float
    innovation: float


def _graph(width: int, graph_index: int, input_index: int):
    seed = TREE_BASE_SEED + input_index * 10_000 + graph_index
    if graph_index % 2 == 0:
        return "contiguous", random_contiguous_split_graph(width, seed=seed)
    return "pair_merge", random_pair_merge_graph(width, seed=seed)


def _var(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    m = mean(xs)
    return mean((x-m)**2 for x in xs)


def _cov(xs: list[float], ys: list[float]) -> float:
    if not xs or len(xs) != len(ys):
        return float("nan")
    mx, my = mean(xs), mean(ys)
    return mean((x-mx)*(y-my) for x,y in zip(xs,ys,strict=True))


def _corr(xs: list[float], ys: list[float]) -> float:
    vx, vy = _var(xs), _var(ys)
    if not math.isfinite(vx) or not math.isfinite(vy) or vx <= 0 or vy <= 0:
        return float("nan")
    return _cov(xs,ys)/math.sqrt(vx*vy)


def _collect(values: tuple[Fraction,...], graph: BinaryReductionGraph, family: str) -> tuple[list[Sample], int, int]:
    pred = predict_fp32_tree_error(values, graph)
    leaf_count = graph.leaf_count
    delta: dict[int, Fraction] = {}
    subtree_error: dict[int, Fraction] = {}
    history: dict[int, Fraction] = {}
    for i,node_pred in enumerate(pred.node_predictions):
        idx = leaf_count + i
        delta[idx] = node_pred.local_rounding_error
    ok = 0
    total = 0
    out: list[Sample] = []
    for offset,node in enumerate(graph.nodes):
        idx = leaf_count + offset
        left_err = subtree_error.get(node.left, Fraction(0))
        right_err = subtree_error.get(node.right, Fraction(0))
        hp = left_err + right_err
        history[idx] = hp

        inherited = history.get(node.left, Fraction(0)) + history.get(node.right, Fraction(0))
        innovation = delta.get(node.left, Fraction(0)) + delta.get(node.right, Fraction(0))
        total += 1
        if hp == inherited + innovation:
            ok += 1
        else:
            raise AssertionError("history transition identity mismatch")

        out.append(Sample(family=family,parent_h=float(hp),inherited=float(inherited),innovation=float(innovation)))
        subtree_error[idx] = hp + delta[idx]
    return out, ok, total


def _summary(label: str, samples: list[Sample]) -> None:
    if not samples:
        return
    h=[s.parent_h for s in samples]
    i=[s.inherited for s in samples]
    j=[s.innovation for s in samples]
    vh,vi,vj=_var(h),_var(i),_var(j)
    cij=_cov(i,j)
    closure=(vi+vj+2*cij)/vh if vh>0 else float('nan')
    print(
        f"  {label:<10} n={len(samples):5d} "
        f"varI/varH={vi/vh if vh>0 else float('nan'):.3f} "
        f"varJ/varH={vj/vh if vh>0 else float('nan'):.3f} "
        f"2covIJ/varH={2*cij/vh if vh>0 else float('nan'):+.3f} closure={closure:.3f}"
    )
    print(
        f"    corr(H,I)={_corr(h,i):+.3f} corr(H,J)={_corr(h,j):+.3f} "
        f"corr(I,J)={_corr(i,j):+.3f} mean|I|/|H|={mean(abs(a)/(abs(b) if b!=0 else 1.0) for a,b in zip(i,h,strict=True)):.3f} "
        f"mean|J|/|H|={mean(abs(a)/(abs(b) if b!=0 else 1.0) for a,b in zip(j,h,strict=True)):.3f}"
    )


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--width',type=int,default=DEFAULT_WIDTH)
    p.add_argument('--graphs',type=int,default=DEFAULT_GRAPH_COUNT)
    p.add_argument('--input-seeds',type=int,nargs='+',default=list(DEFAULT_INPUT_SEEDS))
    args=p.parse_args()
    if args.width<=1: p.error('--width must exceed 1')
    if args.graphs<=1: p.error('--graphs must exceed 1')

    print('Wide-range history transition decomposition diagnostic')
    print('CALIBRATION ONLY — exact FP32 histories/residuals are intentionally inspected')
    print('Testing H_parent = inherited_child_history + child_local_innovation')
    print(f"width={args.width} graphs_per_input={args.graphs} input_seeds={','.join(map(str,args.input_seeds))}")
    print()

    pooled:list[Sample]=[]
    for input_index,seed in enumerate(args.input_seeds):
        generated=wide_range_random(args.width,seed=seed)
        samples:list[Sample]=[]
        ok=total=0
        for graph_index in range(args.graphs):
            family,graph=_graph(len(generated.values),graph_index,input_index)
            rows,a,b=_collect(generated.values,graph,family)
            samples.extend(rows); ok+=a; total+=b
        pooled.extend(samples)
        print(f"INPUT seed={seed} family={generated.family} width={len(generated.values)} identity={ok}/{total}")
        _summary('all',samples)
        _summary('contiguous',[s for s in samples if s.family=='contiguous'])
        _summary('pair_merge',[s for s in samples if s.family=='pair_merge'])
        print()

    print('POOLED SUMMARY')
    _summary('all',pooled)
    _summary('contiguous',[s for s in pooled if s.family=='contiguous'])
    _summary('pair_merge',[s for s in pooled if s.family=='pair_merge'])
    return 0


if __name__=='__main__':
    raise SystemExit(main())
