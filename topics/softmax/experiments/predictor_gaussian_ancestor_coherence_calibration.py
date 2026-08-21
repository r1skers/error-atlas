"""Calibration-only Gaussian ancestor-coherence cheap-score experiment.

The signed-history diagnostics suggest that, after normalization by the cheap history-scale
estimate sigma_H, the oracle history H is close to a centered unit-scale symmetric distribution.
This experiment therefore uses the working model

    H_v / ulp_v ~ Normal(0, s_v^2),

where s_v = sigma_H,v / ulp_v is computed from stored FP32 leaves and the candidate graph only.
For shadow phase phi_v and normalized rounding residual

    q(phi+h) = RN(phi+h) - (phi+h),

we analytically integrate two node-local quantities over the Gaussian history distribution:

    E[delta_v^2]          = ulp_v^2 E[q(phi+h)^2]
    E[2 H_v delta_v]      = 2 ulp_v^2 E[h q(phi+h)].

The second term is the expected node-local ancestor-coherence contribution K_v=2 H_v delta_v.
Summing both terms yields an O(n) predictor-side score for A + C_ancestor under the Gaussian
history approximation.  It does not use real FP32 intermediate states.  The exact FP32 oracle is
used only for the calibration ranking target; held-out data remain untouched.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from fractions import Fraction
from statistics import mean

from predictor_calibration_inputs import wide_range_random
from predictor_second_moment_baseline import second_moment_tree_cost
from predictor_tree_generator import random_contiguous_split_graph, random_pair_merge_graph
from summation_graph_predictor import BinaryReductionGraph, predict_fp32_tree_error

DEFAULT_WIDTH = 256
DEFAULT_GRAPH_COUNT = 64
DEFAULT_INPUT_SEEDS = (22260821, 22260822, 22260823, 22260824)
TREE_BASE_SEED = 32_000_000
FP32_FRACTION_BITS = 23
FP32_MIN_NORMAL_EXPONENT = -126
FP32_MIN_SUBNORMAL_EXPONENT = -149
NORMAL_TAIL_SIGMAS = 8.0


@dataclass(frozen=True)
class Scores:
    second_moment: float
    gaussian_local_mse: float
    gaussian_ancestor_correction: float
    gaussian_local_plus_ancestor: float


def _graph(width: int, *, graph_index: int, input_index: int):
    seed = TREE_BASE_SEED + input_index * 10_000 + graph_index
    if graph_index % 2 == 0:
        return "contiguous", random_contiguous_split_graph(width, seed=seed)
    return "pair_merge", random_pair_merge_graph(width, seed=seed)


def _fp32_ulp(value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("score requires positive finite shadow sums")
    exponent = math.floor(math.log2(value))
    if exponent < FP32_MIN_NORMAL_EXPONENT:
        return math.ldexp(1.0, FP32_MIN_SUBNORMAL_EXPONENT)
    return math.ldexp(1.0, exponent - FP32_FRACTION_BITS)


def _normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _normal_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _gaussian_rounding_moments(phase: float, sigma: float) -> tuple[float, float]:
    """Return (E[q^2], E[h*q]) for h~N(0,sigma^2), q=RN(phase+h)-(phase+h).

    Each rounding cell has q = (k-phase) - h, so both moments follow from exact truncated-normal
    zeroth, first, and second moments. Ties have probability zero for sigma>0.
    """
    if sigma <= 1e-15:
        q = round(phase) - phase
        return q * q, 0.0

    lo = math.floor(phase - NORMAL_TAIL_SIGMAS * sigma - 0.5)
    hi = math.ceil(phase + NORMAL_TAIL_SIGMAS * sigma + 0.5)
    q2_total = 0.0
    hq_total = 0.0

    for k in range(lo, hi + 1):
        lower = k - 0.5 - phase
        upper = k + 0.5 - phase
        alpha = lower / sigma
        beta = upper / sigma
        p = _normal_cdf(beta) - _normal_cdf(alpha)
        if p <= 0.0:
            continue

        # Integrals over h on this cell under N(0,sigma^2).
        first = sigma * (_normal_pdf(alpha) - _normal_pdf(beta))
        second = sigma * sigma * (
            p + alpha * _normal_pdf(alpha) - beta * _normal_pdf(beta)
        )
        center = k - phase
        q2_total += center * center * p - 2.0 * center * first + second
        hq_total += center * first - second

    return max(q2_total, 0.0), hq_total


def cheap_scores(values: tuple[Fraction, ...], graph: BinaryReductionGraph) -> Scores:
    shadow_sums = [float(value) for value in values]
    subtree_variances = [0.0 for _ in values]
    local_mse = 0.0
    ancestor_correction = 0.0

    for node in graph.nodes:
        shadow_sum = shadow_sums[node.left] + shadow_sums[node.right]
        ulp = _fp32_ulp(shadow_sum)
        scaled = shadow_sum / ulp
        phase = scaled - math.floor(scaled)

        history_variance = subtree_variances[node.left] + subtree_variances[node.right]
        sigma_h_ulp = math.sqrt(history_variance) / ulp if history_variance > 0.0 else 0.0
        eq2, ehq = _gaussian_rounding_moments(phase, sigma_h_ulp)

        local_mse += ulp * ulp * eq2
        ancestor_correction += 2.0 * ulp * ulp * ehq

        # Keep the same history-scale model used in the preceding diagnostics.
        local_uniform_variance = ulp * ulp / 12.0
        subtree_variances.append(history_variance + local_uniform_variance)
        shadow_sums.append(shadow_sum)

    second = second_moment_tree_cost(values, graph).partial_sum_square_cost
    return Scores(
        second_moment=second,
        gaussian_local_mse=local_mse,
        gaussian_ancestor_correction=ancestor_correction,
        gaussian_local_plus_ancestor=local_mse + ancestor_correction,
    )


def _rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    return ranks


def _pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    mx, my = mean(x), mean(y)
    dx = [v - mx for v in x]
    dy = [v - my for v in y]
    sx = math.sqrt(sum(v * v for v in dx))
    sy = math.sqrt(sum(v * v for v in dy))
    if sx == 0.0 or sy == 0.0:
        return None
    return sum(a * b for a, b in zip(dx, dy, strict=True)) / (sx * sy)


def _spearman(x: list[float], y: list[float]) -> float | None:
    return _pearson(_rankdata(x), _rankdata(y))


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:+.3f}"


def _report(label: str, rows: list[tuple[str, Scores, float]]) -> dict[str, float | None]:
    target = [r[2] for r in rows]
    rhos = {
        "second_moment": _spearman([r[1].second_moment for r in rows], target),
        "gaussian_local": _spearman([r[1].gaussian_local_mse for r in rows], target),
        "ancestor_correction": _spearman([r[1].gaussian_ancestor_correction for r in rows], target),
        "gaussian_AplusCanc": _spearman([r[1].gaussian_local_plus_ancestor for r in rows], target),
    }
    print(
        f"  {label:<10} n={len(rows):2d} "
        f"rho_second={_fmt(rhos['second_moment'])} "
        f"rho_gaussian_local={_fmt(rhos['gaussian_local'])} "
        f"rho_Cancestor={_fmt(rhos['ancestor_correction'])} "
        f"rho_AplusCanc={_fmt(rhos['gaussian_AplusCanc'])}"
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

    print("Wide-range Gaussian ancestor-coherence cheap-score calibration")
    print("CALIBRATION ONLY — formula not frozen; held-out remains untouched")
    print("PREDICTOR SIDE — stored FP32 leaves + graph only; oracle used only for ranking target")
    print("H/ulp ~ Normal(0,sigma_H^2/ulp^2); score = sum E[delta^2 + 2 H delta]")
    print(f"width={args.width} graphs_per_input={args.graphs} input_seeds={','.join(map(str,args.input_seeds))}")
    print()

    pooled: dict[str, list[float]] = {
        "second_moment": [],
        "gaussian_local": [],
        "ancestor_correction": [],
        "gaussian_AplusCanc": [],
    }
    for input_index, seed in enumerate(args.input_seeds):
        generated = wide_range_random(args.width, seed=seed)
        rows: list[tuple[str, Scores, float]] = []
        for graph_index in range(args.graphs):
            family, graph = _graph(len(generated.values), graph_index=graph_index, input_index=input_index)
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
        print(f"  {key:<20} mean={mean(values):+.3f} min={min(values):+.3f} max={max(values):+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
