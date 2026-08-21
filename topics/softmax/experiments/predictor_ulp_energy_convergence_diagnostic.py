"""Calibration-only diagnostic for normalized-error saturation versus ULP energy.

For exact stored-leaf subtree sums S_v, define the predictor-side local quantum

    U_v = ulp32(S_v),

and normalize by the root quantum U_r.  The independent local-rounding energy is

    Q = sum_v (U_v / U_r)^2,

which gives the heuristic RMS envelope sqrt(Q/12) root ULPs when local residuals are centered and
uniform inside one rounding cell.  A balanced equal-scale tree has Q < 2 as width grows, whereas a
sequential tree need not have a finite-width-independent limit.

This script compares balanced, sequential, random contiguous-split, and random pair-merge trees
across widths.  It reports Q, normalized oracle error |E|/U_r, local-energy calibration, coherence
amplification, and the fraction of Q concentrated in the largest-ULP nodes.

An explicitly exploratory ``Qcorr4`` also adds a four-gap truncated ancestor kernel using the
previously observed standardized-history correlations (0.476, 0.267, 0.168, 0.055).  Those are
history correlations, not a proved residual covariance law, so Qcorr4 is reported as a diagnostic
proxy rather than named as a bound or predictor.

The width-specific input samples are reproducible draws from the same generator, not nested prefixes,
so width trends are distributional rather than trajectories of one growing vector.  All inputs and
graph samples are calibration-only.  Held-out inputs remain untouched.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from fractions import Fraction
from statistics import mean

from predictor_calibration_inputs import wide_range_random
from predictor_shadow_sparse_repair_ablation import _fp32_ulp_fraction
from predictor_tree_generator import (
    random_contiguous_split_graph,
    random_pair_merge_graph,
)
from summation_graph_predictor import (
    BinaryReductionGraph,
    balanced_reduction_graph,
    predict_fp32_tree_error,
    sequential_reduction_graph,
)

DEFAULT_WIDTHS = (32, 64, 128, 256, 512, 1024)
DEFAULT_INPUT_SEEDS = (22260821, 22260822, 22260823, 22260824)
DEFAULT_RANDOM_GRAPHS_PER_FAMILY = 8
CONTIGUOUS_TREE_BASE_SEED = 41_000_000
PAIR_TREE_BASE_SEED = 42_000_000
TOP_COUNTS = (8, 16, 32)
HISTORY_CORRELATION_KERNEL = {
    1: 0.476,
    2: 0.267,
    3: 0.168,
    4: 0.055,
}
FAMILY_ORDER = ("balanced", "sequential", "contiguous", "pair_merge")


@dataclass(frozen=True)
class Metrics:
    width: int
    family: str
    q_ulp: float
    q_corr4: float
    independent_rms_proxy: float
    corr4_rms_proxy: float
    normalized_error: float
    normalized_local_energy: float
    local_energy_ratio: float
    coherence_amplification: float | None
    worst_case_bound: float
    top_energy_share: dict[int, float]


def _parent_map(graph: BinaryReductionGraph) -> list[int | None]:
    parent: list[int | None] = [None] * (graph.leaf_count + len(graph.nodes))
    for offset, node in enumerate(graph.nodes):
        index = graph.leaf_count + offset
        parent[node.left] = index
        parent[node.right] = index
    return parent


def _metrics(
    values: tuple[Fraction, ...],
    graph: BinaryReductionGraph,
    family: str,
) -> Metrics:
    exact_subtree = [*values]
    node_ulps: dict[int, Fraction] = {}
    for offset, node in enumerate(graph.nodes):
        index = graph.leaf_count + offset
        exact_sum = exact_subtree[node.left] + exact_subtree[node.right]
        exact_subtree.append(exact_sum)
        node_ulps[index] = _fp32_ulp_fraction(exact_sum)

    root_ulp = _fp32_ulp_fraction(exact_subtree[graph.root])
    normalized_ulps = {
        index: float(ulp / root_ulp) for index, ulp in node_ulps.items()
    }
    q_ulp = sum(value * value for value in normalized_ulps.values())
    worst_case_bound = 0.5 * sum(normalized_ulps.values())

    parent = _parent_map(graph)
    q_corr4 = q_ulp
    for index, normalized_ulp in normalized_ulps.items():
        ancestor = parent[index]
        for gap in range(1, 5):
            if ancestor is None:
                break
            if ancestor >= graph.leaf_count:
                q_corr4 += (
                    2.0
                    * HISTORY_CORRELATION_KERNEL[gap]
                    * normalized_ulp
                    * normalized_ulps[ancestor]
                )
            ancestor = parent[ancestor]

    oracle = predict_fp32_tree_error(values, graph)
    normalized_error = abs(float(oracle.signed_error / root_ulp))
    normalized_local_energy = math.sqrt(
        sum(
            float(prediction.local_rounding_error / root_ulp) ** 2
            for prediction in oracle.node_predictions
        )
    )
    expected_local_energy_sq = q_ulp / 12.0
    actual_local_energy_sq = normalized_local_energy * normalized_local_energy
    local_energy_ratio = (
        actual_local_energy_sq / expected_local_energy_sq
        if expected_local_energy_sq
        else 0.0
    )
    coherence_amplification = (
        normalized_error / normalized_local_energy
        if normalized_local_energy
        else None
    )

    descending_energy = sorted(
        (value * value for value in normalized_ulps.values()),
        reverse=True,
    )
    top_energy_share = {
        count: sum(descending_energy[:count]) / q_ulp if q_ulp else 0.0
        for count in TOP_COUNTS
    }
    return Metrics(
        width=len(values),
        family=family,
        q_ulp=q_ulp,
        q_corr4=q_corr4,
        independent_rms_proxy=math.sqrt(q_ulp / 12.0),
        corr4_rms_proxy=math.sqrt(q_corr4 / 12.0),
        normalized_error=normalized_error,
        normalized_local_energy=normalized_local_energy,
        local_energy_ratio=local_energy_ratio,
        coherence_amplification=coherence_amplification,
        worst_case_bound=worst_case_bound,
        top_energy_share=top_energy_share,
    )


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile requires at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _fmt_ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _graphs(
    width: int,
    input_index: int,
    random_graphs_per_family: int,
):
    yield "balanced", balanced_reduction_graph(width)
    yield "sequential", sequential_reduction_graph(width)
    for graph_index in range(random_graphs_per_family):
        contiguous_seed = (
            CONTIGUOUS_TREE_BASE_SEED + input_index * 10_000 + graph_index
        )
        pair_seed = PAIR_TREE_BASE_SEED + input_index * 10_000 + graph_index
        yield "contiguous", random_contiguous_split_graph(
            width,
            seed=contiguous_seed,
        )
        yield "pair_merge", random_pair_merge_graph(width, seed=pair_seed)


def _summary(rows: list[Metrics]) -> dict[str, float]:
    q = [row.q_ulp for row in rows]
    q_corr4 = [row.q_corr4 for row in rows]
    error = [row.normalized_error for row in rows]
    local_ratio = [row.local_energy_ratio for row in rows]
    coherence = [
        row.coherence_amplification
        for row in rows
        if row.coherence_amplification is not None
    ]
    return {
        "n": float(len(rows)),
        "q_median": _quantile(q, 0.5),
        "q_p90": _quantile(q, 0.9),
        "q_corr4_median": _quantile(q_corr4, 0.5),
        "proxy_median": _quantile(
            [row.independent_rms_proxy for row in rows],
            0.5,
        ),
        "corr4_proxy_median": _quantile(
            [row.corr4_rms_proxy for row in rows],
            0.5,
        ),
        "error_median": _quantile(error, 0.5),
        "error_p90": _quantile(error, 0.9),
        "error_max": max(error),
        "local_ratio_median": _quantile(local_ratio, 0.5),
        "coherence_median": _quantile(coherence, 0.5) if coherence else 0.0,
        "coherence_p90": _quantile(coherence, 0.9) if coherence else 0.0,
        "worst_bound_median": _quantile(
            [row.worst_case_bound for row in rows],
            0.5,
        ),
        **{
            f"top{count}_mean": mean(
                row.top_energy_share[count] for row in rows
            )
            for count in TOP_COUNTS
        },
    }


def _print_summary(width: int, family: str, summary: dict[str, float]) -> None:
    print(
        f"  width={width:<4d} family={family:<10} n={int(summary['n']):3d} "
        f"Qmed/p90={summary['q_median']:.3f}/{summary['q_p90']:.3f} "
        f"Qcorr4med={summary['q_corr4_median']:.3f} "
        f"proxy/corr4={summary['proxy_median']:.3f}/"
        f"{summary['corr4_proxy_median']:.3f} "
        f"|E|/Ur med/p90/max={summary['error_median']:.3f}/"
        f"{summary['error_p90']:.3f}/{summary['error_max']:.3f} "
        f"A/(Q/12)med={summary['local_ratio_median']:.3f} "
        f"|E|/sqrt(A) med/p90={summary['coherence_median']:.3f}/"
        f"{summary['coherence_p90']:.3f} "
        f"topQ[8/16/32]={summary['top8_mean']:.3f}/"
        f"{summary['top16_mean']:.3f}/{summary['top32_mean']:.3f} "
        f"BworstMed={summary['worst_bound_median']:.1f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--widths",
        type=int,
        nargs="+",
        default=list(DEFAULT_WIDTHS),
    )
    parser.add_argument(
        "--input-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_INPUT_SEEDS),
    )
    parser.add_argument(
        "--random-graphs-per-family",
        type=int,
        default=DEFAULT_RANDOM_GRAPHS_PER_FAMILY,
    )
    args = parser.parse_args()
    if any(width < 2 for width in args.widths):
        parser.error("--widths must contain only integers >= 2")
    if args.random_graphs_per_family <= 0:
        parser.error("--random-graphs-per-family must be positive")
    widths = tuple(dict.fromkeys(args.widths))

    print("Wide-range ULP-energy convergence diagnostic")
    print("CALIBRATION ONLY — no held-out inputs; no predictor is frozen")
    print("Q is predictor-side; oracle E/A are diagnostic targets only")
    print("balanced/sequential quantiles use one anchor per input seed and are descriptive")
    print(
        f"widths={','.join(map(str, widths))} "
        f"input_seeds={','.join(map(str, args.input_seeds))} "
        f"random_graphs_per_family={args.random_graphs_per_family}"
    )
    print()

    grouped: dict[tuple[int, str], list[Metrics]] = {}
    for width in widths:
        for input_index, input_seed in enumerate(args.input_seeds):
            generated = wide_range_random(width, seed=input_seed)
            for family, graph in _graphs(
                width,
                input_index,
                args.random_graphs_per_family,
            ):
                grouped.setdefault((width, family), []).append(
                    _metrics(generated.values, graph, family)
                )

    summaries: dict[tuple[int, str], dict[str, float]] = {}
    for width in widths:
        for family in FAMILY_ORDER:
            summary = _summary(grouped[(width, family)])
            summaries[(width, family)] = summary
            _print_summary(width, family, summary)
        print()

    if len(widths) >= 2:
        previous_width, final_width = widths[-2:]
        print("ENDPOINT SATURATION CHECK final/previous ratios")
        for family in FAMILY_ORDER:
            previous = summaries[(previous_width, family)]
            final = summaries[(final_width, family)]
            q_ratio = _safe_ratio(final["q_median"], previous["q_median"])
            error_ratio = _safe_ratio(final["error_p90"], previous["error_p90"])
            bound_ratio = _safe_ratio(
                final["worst_bound_median"],
                previous["worst_bound_median"],
            )
            print(
                f"  {family:<10} widths={previous_width}->{final_width} "
                f"Qmed_ratio={_fmt_ratio(q_ratio)} "
                f"errorP90_ratio={_fmt_ratio(error_ratio)} "
                f"worstBound_ratio={_fmt_ratio(bound_ratio)}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
