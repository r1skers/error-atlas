"""Calibration-only recursive Gaussian moment-closure tree scores.

Two predictor-side recursions are compared:

1. deterministic-shadow recursion: propagate only the phase-aware rounded output moments. Starting
   from exact leaves this collapses to zero variance and is retained as a useful control.
2. stochastic-innovation recursion: after the same phase-aware transform, inject an unresolved
   local rounding innovation variance u_v^2/12 before propagating upward. This is the first cheap
   stochastic recursion motivated by the transition diagnostic H_p = I_p + J_p, where inherited
   state dominates but local rounding innovation supplies substantial new variance.

Only stored leaves and the candidate graph are used by the scores. Exact FP32 execution is used
only for the calibration ranking target. Held-out inputs remain untouched.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from fractions import Fraction
from statistics import mean

from predictor_calibration_inputs import wide_range_random
from predictor_gaussian_ancestor_coherence_calibration import _fp32_ulp, _normal_cdf
from predictor_second_moment_baseline import second_moment_tree_cost
from predictor_tree_generator import random_contiguous_split_graph, random_pair_merge_graph
from summation_graph_predictor import BinaryReductionGraph, predict_fp32_tree_error

DEFAULT_WIDTH = 256
DEFAULT_GRAPH_COUNT = 64
DEFAULT_INPUT_SEEDS = (22260821, 22260822, 22260823, 22260824)
TREE_BASE_SEED = 36_000_000
NORMAL_TAIL_SIGMAS = 8.0


@dataclass(frozen=True)
class Moment:
    mean: float
    variance: float


@dataclass(frozen=True)
class Scores:
    second_moment: float
    shadow_mean_abs: float
    stochastic_mean_abs: float
    stochastic_rms: float
    stochastic_variance: float


def _graph(width: int, graph_index: int, input_index: int):
    seed = TREE_BASE_SEED + input_index * 10_000 + graph_index
    if graph_index % 2 == 0:
        return "contiguous", random_contiguous_split_graph(width, seed=seed)
    return "pair_merge", random_pair_merge_graph(width, seed=seed)


def _rounding_output_moments(phase: float, mu: float, sigma: float) -> tuple[float, float]:
    """Return E[y], E[y^2] for y=RN(phi+h)-phi, h~N(mu,sigma^2)."""
    if sigma <= 1e-15:
        y = round(phase + mu) - phase
        return y, y * y

    lo = math.floor(phase + mu - NORMAL_TAIL_SIGMAS * sigma - 0.5)
    hi = math.ceil(phase + mu + NORMAL_TAIL_SIGMAS * sigma + 0.5)
    ey = 0.0
    ey2 = 0.0
    for k in range(lo, hi + 1):
        lower = k - 0.5 - phase
        upper = k + 0.5 - phase
        alpha = (lower - mu) / sigma
        beta = (upper - mu) / sigma
        p = _normal_cdf(beta) - _normal_cdf(alpha)
        if p <= 0.0:
            continue
        y = k - phase
        ey += y * p
        ey2 += y * y * p
    return ey, ey2


def _normal_mean_abs(mu: float, variance: float) -> float:
    if variance <= 0.0:
        return abs(mu)
    sigma = math.sqrt(variance)
    z = mu / sigma
    return sigma * math.sqrt(2.0 / math.pi) * math.exp(-0.5 * z * z) + mu * (2.0 * _normal_cdf(z) - 1.0)


def _propagate(values: tuple[Fraction, ...], graph: BinaryReductionGraph, *, inject_innovation: bool) -> Moment:
    shadow = [float(v) for v in values]
    error = [Moment(0.0, 0.0) for _ in values]

    for node in graph.nodes:
        shadow_sum = shadow[node.left] + shadow[node.right]
        left = error[node.left]
        right = error[node.right]

        h_mu = left.mean + right.mean
        h_var = max(left.variance + right.variance, 0.0)

        ulp = _fp32_ulp(shadow_sum)
        scaled = shadow_sum / ulp
        phase = scaled - math.floor(scaled)
        mu_ulp = h_mu / ulp
        sigma_ulp = math.sqrt(h_var) / ulp if h_var > 0.0 else 0.0

        ey, ey2 = _rounding_output_moments(phase, mu_ulp, sigma_ulp)
        out_mu = ulp * ey
        out_second = ulp * ulp * ey2
        out_var = max(out_second - out_mu * out_mu, 0.0)

        if inject_innovation:
            out_var += ulp * ulp / 12.0

        shadow.append(shadow_sum)
        error.append(Moment(out_mu, out_var))

    return error[-1]


def cheap_scores(values: tuple[Fraction, ...], graph: BinaryReductionGraph) -> Scores:
    shadow_root = _propagate(values, graph, inject_innovation=False)
    stochastic_root = _propagate(values, graph, inject_innovation=True)
    second = second_moment_tree_cost(values, graph).partial_sum_square_cost
    return Scores(
        second_moment=second,
        shadow_mean_abs=_normal_mean_abs(shadow_root.mean, shadow_root.variance),
        stochastic_mean_abs=_normal_mean_abs(stochastic_root.mean, stochastic_root.variance),
        stochastic_rms=math.sqrt(max(stochastic_root.variance + stochastic_root.mean * stochastic_root.mean, 0.0)),
        stochastic_variance=stochastic_root.variance,
    )


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


def _pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    mx, my = mean(x), mean(y)
    dx = [v - mx for v in x]
    dy = [v - my for v in y]
    sx = math.sqrt(sum(v*v for v in dx))
    sy = math.sqrt(sum(v*v for v in dy))
    if sx == 0.0 or sy == 0.0:
        return None
    return sum(a*b for a,b in zip(dx,dy,strict=True)) / (sx*sy)


def _spearman(x: list[float], y: list[float]) -> float | None:
    return _pearson(_rankdata(x), _rankdata(y))


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:+.3f}"


def _report(label: str, rows: list[tuple[str, Scores, float]]) -> dict[str, float | None]:
    target = [r[2] for r in rows]
    rhos = {
        "second_moment": _spearman([r[1].second_moment for r in rows], target),
        "shadow_abs": _spearman([r[1].shadow_mean_abs for r in rows], target),
        "stochastic_abs": _spearman([r[1].stochastic_mean_abs for r in rows], target),
        "stochastic_rms": _spearman([r[1].stochastic_rms for r in rows], target),
        "stochastic_variance": _spearman([r[1].stochastic_variance for r in rows], target),
    }
    print(
        f"  {label:<10} n={len(rows):2d} "
        f"rho_second={_fmt(rhos['second_moment'])} "
        f"rho_shadow={_fmt(rhos['shadow_abs'])} "
        f"rho_stoch_abs={_fmt(rhos['stochastic_abs'])} "
        f"rho_stoch_rms={_fmt(rhos['stochastic_rms'])} "
        f"rho_stoch_var={_fmt(rhos['stochastic_variance'])}"
    )
    return rhos


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

    print("Recursive Gaussian moment-closure cheap-score calibration")
    print("CALIBRATION ONLY — formula not frozen; held-out remains untouched")
    print("PREDICTOR SIDE — stored FP32 leaves + graph only; oracle used only for ranking target")
    print("Compare deterministic shadow recursion against stochastic recursion with u^2/12 innovation")
    print(f"width={args.width} graphs_per_input={args.graphs} input_seeds={','.join(map(str,args.input_seeds))}")
    print()

    keys = ("second_moment", "shadow_abs", "stochastic_abs", "stochastic_rms", "stochastic_variance")
    pooled = {k: [] for k in keys}
    for input_index, seed in enumerate(args.input_seeds):
        generated = wide_range_random(args.width, seed=seed)
        rows: list[tuple[str, Scores, float]] = []
        for graph_index in range(args.graphs):
            family, graph = _graph(len(generated.values), graph_index, input_index)
            scores = cheap_scores(generated.values, graph)
            target = float(abs(predict_fp32_tree_error(generated.values, graph).signed_error))
            rows.append((family, scores, target))

        print(f"INPUT seed={seed} family={generated.family} width={len(generated.values)}")
        all_rhos = _report("all", rows)
        _report("contiguous", [r for r in rows if r[0] == "contiguous"])
        _report("pair_merge", [r for r in rows if r[0] == "pair_merge"])
        for key, value in all_rhos.items():
            if value is not None:
                pooled[key].append(value)
        print()

    print("SEED SUMMARY all-tree rho mean/min/max")
    for key, values in pooled.items():
        if not values:
            print(f"  {key:<20} n/a")
            continue
        print(f"  {key:<20} mean={mean(values):+.3f} min={min(values):+.3f} max={max(values):+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
