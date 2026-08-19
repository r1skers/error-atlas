"""Calibration-only ULP-scale stagnation diagnostic for wide-range reductions.

This script does not define a new predictor.  It tests a concrete mechanism hypothesis:
second-moment ranking may fail on positive wide-dynamic-range inputs because some merges
enter a quantized regime where the smaller child is tiny relative to one FP32 ulp at the
larger child's scale.

The diagnostic intentionally uses cheap binary64 subtree masses, not simulated FP32
partial sums.  Therefore ``small / ulp32(large)`` is an *exposure proxy*, not a claim that
a real FP32 merge necessarily stagnates.  Exact graph error is used only as the calibration
target for rank-correlation diagnostics.

Nothing printed here is held-out evidence and no predictor formula is frozen by this run.
"""

from __future__ import annotations

import math
from statistics import mean

from predictor_calibration_inputs import calibration_input_families
from predictor_tree_generator import (
    random_contiguous_split_graph,
    random_pair_merge_graph,
)
from summation_graph_predictor import predict_fp32_tree_error


WIDTH = 256
INPUT_SEEDS = (20260818, 20260819, 20260820, 20260821)
RANDOM_GRAPH_COUNT = 64
TREE_BASE_SEED = 31000000
PROGRESS_EVERY = 16


def _rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        value = values[order[start]]
        while stop < len(order) and values[order[stop]] == value:
            stop += 1
        average_rank = (start + 1 + stop) / 2.0
        for position in range(start, stop):
            ranks[order[position]] = average_rank
        start = stop
    return ranks


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right):
        raise ValueError("vectors must have equal length")
    if len(left) < 2:
        return None
    left_mean = mean(left)
    right_mean = mean(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    left_ss = sum(value * value for value in left_centered)
    right_ss = sum(value * value for value in right_centered)
    if left_ss == 0.0 or right_ss == 0.0:
        return None
    covariance = sum(a * b for a, b in zip(left_centered, right_centered, strict=True))
    return covariance / math.sqrt(left_ss * right_ss)


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right):
        raise ValueError("vectors must have equal length")
    if any(not math.isfinite(value) for value in (*left, *right)):
        return None
    return _pearson(_rankdata(left), _rankdata(right))


def _format_rho(value: float | None) -> str:
    return "undefined" if value is None else f"{value:+.3f}"


def _ulp32_at_scale(value: float) -> float:
    """Return the binary32 ulp spacing at positive finite ``value``'s scale.

    This is a scale-only quantity.  It does not round ``value`` to FP32 and therefore does
    not expose mantissa/tie phase; that limitation is deliberate for this mechanism check.
    """
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("value must be positive and finite")
    exponent = math.floor(math.log2(value))
    if exponent < -126:
        return math.ldexp(1.0, -149)
    return math.ldexp(1.0, exponent - 23)


def _graph(width: int, *, input_index: int, graph_index: int):
    seed = TREE_BASE_SEED + input_index * 10_000 + graph_index
    if graph_index % 2 == 0:
        return "contiguous", random_contiguous_split_graph(width, seed=seed)
    return "pair_merge", random_pair_merge_graph(width, seed=seed)


def _stagnation_exposure(values, graph) -> tuple[float, float, float, float]:
    """Return four graph-level ULP-scale exposure summaries.

    For each internal merge, subtree masses are propagated in binary64.  With
    ``small <= large`` define ``r = small / ulp32(large)``.

    Returned summaries are:
    - fraction of nodes with r < 0.5;
    - fraction of nodes with r < 1.0;
    - fraction of nodes with 0.5 <= r < 2.0 (near-threshold exposure);
    - total small-child mass from nodes with r < 0.5, normalized by root mass.
    """
    masses = [float(value) for value in values]
    below_half = 0
    below_one = 0
    near_threshold = 0
    below_half_mass = 0.0

    for node in graph.nodes:
        left = masses[node.left]
        right = masses[node.right]
        small = min(left, right)
        large = max(left, right)
        if large == 0.0:
            ratio = math.inf
        else:
            ratio = small / _ulp32_at_scale(large)

        if ratio < 0.5:
            below_half += 1
            below_half_mass += small
        if ratio < 1.0:
            below_one += 1
        if 0.5 <= ratio < 2.0:
            near_threshold += 1
        masses.append(left + right)

    node_count = len(graph.nodes)
    root_mass = masses[-1]
    return (
        below_half / node_count,
        below_one / node_count,
        near_threshold / node_count,
        0.0 if root_mass == 0.0 else below_half_mass / root_mass,
    )


def _run_wide_input(*, seed: int, values, input_index: int) -> None:
    targets: dict[str, list[float]] = {"all": [], "contiguous": [], "pair_merge": []}
    metrics: dict[str, dict[str, list[float]]] = {
        group: {
            "below_half_frac": [],
            "below_one_frac": [],
            "near_threshold_frac": [],
            "below_half_mass_frac": [],
        }
        for group in targets
    }

    print(
        f"running family=wide_range_random seed={seed} width={len(values)} "
        f"random_graphs={RANDOM_GRAPH_COUNT}",
        flush=True,
    )

    for graph_index in range(RANDOM_GRAPH_COUNT):
        graph_family, graph = _graph(
            len(values), input_index=input_index, graph_index=graph_index
        )
        oracle = predict_fp32_tree_error(values, graph)
        target = float(abs(oracle.signed_error))
        exposure = _stagnation_exposure(values, graph)
        named = dict(
            zip(
                (
                    "below_half_frac",
                    "below_one_frac",
                    "near_threshold_frac",
                    "below_half_mass_frac",
                ),
                exposure,
                strict=True,
            )
        )

        for group in ("all", graph_family):
            targets[group].append(target)
            for name, value in named.items():
                metrics[group][name].append(value)

        completed = graph_index + 1
        if completed % PROGRESS_EVERY == 0 or completed == RANDOM_GRAPH_COUNT:
            print(f"  progress {completed}/{RANDOM_GRAPH_COUNT}", flush=True)

    print(
        f"seed={seed:<10d} target_unique(all/contig/pair)="
        f"{len(set(targets['all']))}/{len(set(targets['contiguous']))}/"
        f"{len(set(targets['pair_merge']))}"
    )
    for group in ("all", "contiguous", "pair_merge"):
        print(f"  {group}:")
        for name in (
            "below_half_frac",
            "below_one_frac",
            "near_threshold_frac",
            "below_half_mass_frac",
        ):
            values_for_metric = metrics[group][name]
            rho = _spearman(values_for_metric, targets[group])
            print(
                f"    {name:<22} rho={_format_rho(rho)} "
                f"mean={mean(values_for_metric):.6g}"
            )
    print(flush=True)


def main() -> int:
    print("Wide-range ULP-scale stagnation exposure diagnostic")
    print("CALIBRATION ONLY — mechanism diagnostic; no new predictor is defined")
    print(
        "r = cheap small-child mass / binary32 ulp at cheap large-child scale; "
        "r is an exposure proxy, not simulated FP32 stagnation"
    )
    print()

    input_index = 0
    for base_seed in INPUT_SEEDS:
        for generated in calibration_input_families(WIDTH, seed=base_seed):
            if generated.family == "wide_range_random":
                _run_wide_input(
                    seed=generated.seed,
                    values=generated.values,
                    input_index=input_index,
                )
            input_index += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
