"""Calibrate a sparse deterministic first-order phase-correction score.

For each internal node v, first compute the history-free shadow residual

    delta0_v = RN32(S_v) - S_v,

where S_v is the exact stored-leaf subtree sum.  Accumulating only these residuals inside each
subtree gives a deterministic first-order history estimate

    H0_v = sum_{u proper-descendant of v} delta0_u.

Use that estimate once, without feeding corrected residuals back into ancestors, to obtain

    delta1_v = RN32(S_v + H0_v) - (S_v + H0_v).

Candidate scores replace delta0 by delta1 either for the top k nodes by subtree height or for the k
nodes with largest predicted |delta1-delta0|.  ``full_first`` replaces every node independently.
The deliberately non-recursive correction prevents this predictor from collapsing into an exact
replay of the candidate FP32 reduction.

Only stored FP32 leaves and the graph are used by these scores.  The exact oracle is used solely as
the calibration ranking target; held-out inputs remain untouched.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
from statistics import mean

from predictor_calibration_inputs import wide_range_random
from predictor_shadow_high_layer_mistake_ablation import DEFAULT_REPAIR_COUNTS
from predictor_shadow_sparse_repair_ablation import (
    DEFAULT_GRAPH_COUNT,
    DEFAULT_INPUT_SEEDS,
    DEFAULT_WIDTH,
    _graph,
    _spearman,
    _topological_depths,
)
from summation_graph_predictor import (
    BinaryReductionGraph,
    predict_fp32_tree_error,
    round_nonnegative_fraction_to_fp32,
)


@dataclass(frozen=True)
class Scores:
    shadow: float
    full_first: float
    top: dict[int, float]
    predicted_gap: dict[int, float]


def _sparse_sum(
    internal: list[int],
    delta0: dict[int, Fraction],
    delta1: dict[int, Fraction],
    repair: set[int],
) -> float:
    value = sum(
        (delta1[index] if index in repair else delta0[index] for index in internal),
        start=Fraction(0),
    )
    return abs(float(value))


def cheap_scores(
    values: tuple[Fraction, ...],
    graph: BinaryReductionGraph,
    repair_counts: tuple[int, ...],
) -> Scores:
    nleaf = graph.leaf_count
    internal = [nleaf + offset for offset in range(len(graph.nodes))]
    depths = _topological_depths(graph)

    exact_subtree = [*values]
    first_order_error = [Fraction(0) for _ in values]
    delta0: dict[int, Fraction] = {}
    delta1: dict[int, Fraction] = {}

    for offset, node in enumerate(graph.nodes):
        index = nleaf + offset
        exact_sum = exact_subtree[node.left] + exact_subtree[node.right]
        exact_subtree.append(exact_sum)

        rounded0 = round_nonnegative_fraction_to_fp32(exact_sum).value
        delta0[index] = rounded0 - exact_sum

        history0 = first_order_error[node.left] + first_order_error[node.right]
        shifted_sum = exact_sum + history0
        rounded1 = round_nonnegative_fraction_to_fp32(shifted_sum).value
        delta1[index] = rounded1 - shifted_sum

        # Keep the history estimate strictly first order: corrected delta1 never propagates.
        first_order_error.append(history0 + delta0[index])

    depth_order = sorted(internal, key=lambda index: depths[index], reverse=True)
    gap_order = sorted(
        internal,
        key=lambda index: abs(delta1[index] - delta0[index]),
        reverse=True,
    )

    top: dict[int, float] = {}
    predicted_gap: dict[int, float] = {}
    for requested_count in repair_counts:
        budget = min(requested_count, len(internal))
        top[requested_count] = _sparse_sum(
            internal,
            delta0,
            delta1,
            set(depth_order[:budget]),
        )
        predicted_gap[requested_count] = _sparse_sum(
            internal,
            delta0,
            delta1,
            set(gap_order[:budget]),
        )

    return Scores(
        shadow=_sparse_sum(internal, delta0, delta1, set()),
        full_first=_sparse_sum(internal, delta0, delta1, set(internal)),
        top=top,
        predicted_gap=predicted_gap,
    )


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.3f}"


def _report(
    label: str,
    rows: list[tuple[str, Scores, float]],
    repair_counts: tuple[int, ...],
) -> dict[str, float | None]:
    target = [row[2] for row in rows]
    out: dict[str, float | None] = {
        "shadow": _spearman([row[1].shadow for row in rows], target),
        "full_first": _spearman([row[1].full_first for row in rows], target),
    }
    print(
        f"  {label:<10} n={len(rows):2d} target_unique={len(set(target)):2d} "
        f"rho_shadow={_fmt(out['shadow'])} rho_full={_fmt(out['full_first'])}"
    )
    for policy in ("top", "predicted_gap"):
        values: list[str] = []
        for count in repair_counts:
            rho = _spearman(
                [getattr(row[1], policy)[count] for row in rows],
                target,
            )
            out[f"{policy}_{count}"] = rho
            values.append(_fmt(rho))
        print(f"    {policy:<13} rho={'/'.join(values)}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--graphs", type=int, default=DEFAULT_GRAPH_COUNT)
    parser.add_argument(
        "--input-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_INPUT_SEEDS),
    )
    parser.add_argument(
        "--repair-counts",
        type=int,
        nargs="+",
        default=list(DEFAULT_REPAIR_COUNTS),
    )
    args = parser.parse_args()
    if args.width <= 1:
        parser.error("--width must exceed 1")
    if args.graphs <= 1:
        parser.error("--graphs must exceed 1")
    if any(count <= 0 for count in args.repair_counts):
        parser.error("--repair-counts must contain only positive integers")
    repair_counts = tuple(dict.fromkeys(args.repair_counts))

    print("Sparse first-order phase-correction cheap-score calibration")
    print("CALIBRATION ONLY — formula not frozen; held-out remains untouched")
    print("PREDICTOR SIDE — stored FP32 leaves + graph only; oracle is target only")
    print("delta1 corrections are independent and never recursively replayed")
    print(
        f"width={args.width} graphs_per_input={args.graphs} "
        f"input_seeds={','.join(map(str, args.input_seeds))} "
        f"repair_counts={','.join(map(str, repair_counts))}"
    )
    print()

    pooled: dict[str, list[float]] = {}
    for input_index, seed in enumerate(args.input_seeds):
        generated = wide_range_random(args.width, seed=seed)
        rows: list[tuple[str, Scores, float]] = []
        for graph_index in range(args.graphs):
            family, graph = _graph(len(generated.values), graph_index, input_index)
            scores = cheap_scores(generated.values, graph, repair_counts)
            target = abs(float(predict_fp32_tree_error(generated.values, graph).signed_error))
            rows.append((family, scores, target))

        print(f"INPUT seed={seed} family={generated.family} width={len(generated.values)}")
        stats = _report("all", rows, repair_counts)
        _report("contiguous", [row for row in rows if row[0] == "contiguous"], repair_counts)
        _report("pair_merge", [row for row in rows if row[0] == "pair_merge"], repair_counts)
        for key, value in stats.items():
            if value is not None:
                pooled.setdefault(key, []).append(value)
        print()

    print("SEED SUMMARY all-tree rho mean/min/max")
    for key, values in pooled.items():
        print(
            f"  {key:<24} mean={mean(values):+.3f} "
            f"min={min(values):+.3f} max={max(values):+.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
