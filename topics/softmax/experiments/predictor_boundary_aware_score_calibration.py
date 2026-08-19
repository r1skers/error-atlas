"""Calibration-only boundary-aware cheap-score experiment.

This is the first tree-level score derived from the rounding-history mechanism work.
It remains calibration-only: the score formula is not frozen and held-out data must not
be opened because of this experiment.

Predictor-side quantities use only stored FP32 leaves and the known reduction graph.
No real FP32 intermediate state or oracle residual enters the score.

For node v, let T*_v be the shadow subtree sum and let

    sigma_H,v^2 = estimated descendant rounding-history variance.

The already-calibrated scale model propagates a simple local variance ulp(T*_v)^2 / 12.
Instead of the singular heuristic sigma / boundary_distance, this experiment uses a
bounded phase-conditioned quantity.  With

    X_v = phase(T*_v) + Z,    Z ~ Normal(0, sigma_H,v^2 / ulp(T*_v)^2),

let q(x) = round_to_nearest_integer(x) - x.  The node contribution is

    ulp(T*_v)^2 * E[q(X_v)^2].

Summing this over internal nodes gives ``boundary_phase_mse``.  Two ablations are reported:
``uniform_ulp_mse`` = sum ulp^2/12 and the existing second-moment baseline.

The exact FP32 oracle is used only for the calibration target |root signed error|.
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
class CheapScores:
    second_moment: float
    uniform_ulp_mse: float
    boundary_phase_mse: float


def _graph(width: int, *, graph_index: int, input_index: int):
    seed = TREE_BASE_SEED + input_index * 10_000 + graph_index
    if graph_index % 2 == 0:
        return "contiguous", seed, random_contiguous_split_graph(width, seed=seed)
    return "pair_merge", seed, random_pair_merge_graph(width, seed=seed)


def _fp32_ulp(value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("boundary-aware score requires positive finite shadow sums")
    exponent = math.floor(math.log2(value))
    if exponent < FP32_MIN_NORMAL_EXPONENT:
        return math.ldexp(1.0, FP32_MIN_SUBNORMAL_EXPONENT)
    return math.ldexp(1.0, exponent - FP32_FRACTION_BITS)


def _normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _normal_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _truncated_normal_q2(phase: float, sigma: float) -> float:
    """Return E[(RN(phase+Z)-(phase+Z))^2] for Z~N(0,sigma^2).

    The quantization residual is periodic with period one, so only the fractional shadow
    phase is needed.  Ties have probability zero for sigma>0.  For sigma=0 Python's
    round() supplies ties-to-even, matching IEEE round-to-nearest ties-to-even here.
    """
    if sigma <= 1e-15:
        residual = round(phase) - phase
        return residual * residual

    lo = math.floor(phase - NORMAL_TAIL_SIGMAS * sigma - 0.5)
    hi = math.ceil(phase + NORMAL_TAIL_SIGMAS * sigma + 0.5)
    total = 0.0

    for k in range(lo, hi + 1):
        lower = k - 0.5 - phase
        upper = k + 0.5 - phase
        alpha = lower / sigma
        beta = upper / sigma
        p = _normal_cdf(beta) - _normal_cdf(alpha)
        if p <= 0.0:
            continue

        first = sigma * (_normal_pdf(alpha) - _normal_pdf(beta))
        second = sigma * sigma * (
            p + alpha * _normal_pdf(alpha) - beta * _normal_pdf(beta)
        )
        center = k - phase
        total += center * center * p - 2.0 * center * first + second

    return max(total, 0.0)


def cheap_scores(values: tuple[Fraction, ...], graph: BinaryReductionGraph) -> CheapScores:
    """Compute predictor-side scores without replaying FP32 intermediate states."""
    shadow_sums = [float(value) for value in values]
    subtree_history_variances = [0.0 for _ in values]
    uniform_score = 0.0
    boundary_score = 0.0

    for node in graph.nodes:
        shadow_sum = shadow_sums[node.left] + shadow_sums[node.right]
        ulp = _fp32_ulp(shadow_sum)
        scaled = shadow_sum / ulp
        phase = scaled - math.floor(scaled)

        history_variance = (
            subtree_history_variances[node.left]
            + subtree_history_variances[node.right]
        )
        sigma_h_ulp = math.sqrt(history_variance) / ulp if history_variance > 0.0 else 0.0

        local_uniform_variance = (ulp * ulp) / 12.0
        uniform_score += local_uniform_variance
        boundary_score += (ulp * ulp) * _truncated_normal_q2(phase, sigma_h_ulp)

        # Keep the already-calibrated history-scale model fixed for this experiment so
        # boundary conditioning is the only new ingredient being tested.
        subtree_history_variances.append(history_variance + local_uniform_variance)
        shadow_sums.append(shadow_sum)

    baseline = second_moment_tree_cost(values, graph)
    return CheapScores(
        second_moment=baseline.partial_sum_square_cost,
        uniform_ulp_mse=uniform_score,
        boundary_phase_mse=boundary_score,
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
    mx = mean(x)
    my = mean(y)
    dx = [value - mx for value in x]
    dy = [value - my for value in y]
    sx = math.sqrt(sum(value * value for value in dx))
    sy = math.sqrt(sum(value * value for value in dy))
    if sx == 0.0 or sy == 0.0:
        return None
    return sum(a * b for a, b in zip(dx, dy, strict=True)) / (sx * sy)


def _spearman(x: list[float], y: list[float]) -> float | None:
    return _pearson(_rankdata(x), _rankdata(y))


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.3f}"


def _report(label: str, rows: list[tuple[str, CheapScores, float]]) -> None:
    if not rows:
        return
    target = [row[2] for row in rows]
    print(
        f"  {label:<10} n={len(rows):2d} "
        f"rho_second_moment={_fmt(_spearman([r[1].second_moment for r in rows], target))} "
        f"rho_uniform_ulp={_fmt(_spearman([r[1].uniform_ulp_mse for r in rows], target))} "
        f"rho_boundary_phase={_fmt(_spearman([r[1].boundary_phase_mse for r in rows], target))}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--graphs", type=int, default=DEFAULT_GRAPH_COUNT)
    parser.add_argument("--input-seeds", type=int, nargs="+", default=list(DEFAULT_INPUT_SEEDS))
    args = parser.parse_args()
    if args.width <= 1:
        parser.error("--width must be greater than one")
    if args.graphs <= 1:
        parser.error("--graphs must be greater than one")
    return args


def main() -> int:
    args = _parse_args()
    print("Wide-range boundary-aware cheap-score calibration")
    print("CALIBRATION ONLY — score formula is not frozen; do not open held-out")
    print("PREDICTOR SIDE — stored leaves + graph only; oracle used only for target")
    print(
        f"width={args.width} graphs_per_input={args.graphs} "
        f"input_seeds={','.join(str(seed) for seed in args.input_seeds)}"
    )
    print()

    all_seed_results: dict[str, list[float]] = {
        "second_moment": [],
        "uniform_ulp": [],
        "boundary_phase": [],
    }

    for input_index, input_seed in enumerate(args.input_seeds):
        generated = wide_range_random(args.width, seed=input_seed)
        rows: list[tuple[str, CheapScores, float]] = []

        for graph_index in range(args.graphs):
            family, _, graph = _graph(
                len(generated.values), graph_index=graph_index, input_index=input_index
            )
            scores = cheap_scores(generated.values, graph)
            target = float(abs(predict_fp32_tree_error(generated.values, graph).signed_error))
            rows.append((family, scores, target))

        print(f"INPUT seed={input_seed} family=wide_range_random width={args.width}")
        _report("all", rows)
        _report("contiguous", [row for row in rows if row[0] == "contiguous"])
        _report("pair_merge", [row for row in rows if row[0] == "pair_merge"])

        target = [row[2] for row in rows]
        all_seed_results["second_moment"].append(
            _spearman([row[1].second_moment for row in rows], target) or 0.0
        )
        all_seed_results["uniform_ulp"].append(
            _spearman([row[1].uniform_ulp_mse for row in rows], target) or 0.0
        )
        all_seed_results["boundary_phase"].append(
            _spearman([row[1].boundary_phase_mse for row in rows], target) or 0.0
        )
        print()

    print("SEED SUMMARY pooled-tree-ranking rho mean/min/max")
    for name, values in all_seed_results.items():
        print(
            f"  {name:<15} mean={mean(values):+.3f} "
            f"min={min(values):+.3f} max={max(values):+.3f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
