"""Calibration-only ablation for sparse phase mistakes in the shadow model's high layers.

The corrected sparse-repair experiment shows that replacing the shadow residuals of a small
top-of-tree band can recover much of the within-input ranking signal.  That result alone does not
show whether the useful repairs are discrete phase mistakes or a more diffuse high-layer drift.

Compare five oracle repair policies at budgets k=1,2,4,...:

* ``top_all``: repair every node in the high-layer budget;
* ``ulp_all``: repair the k nodes with the largest local FP32 ULP;
* ``top_cell``: repair only budget nodes whose actual history changes the RN-even cell;
* ``top_sign``: repair only budget nodes whose actual and shadow residual signs differ;
* ``oracle_gap``: repair the globally largest |delta_actual - delta_shadow| nodes (upper bound).

The cell/sign policies report their effective repair counts because they may repair fewer than k
nodes.  All repairs are local-residual substitutions in a sum; none recursively executes the
candidate tree.  Exact FP32 histories are used intentionally, so this remains a mechanistic
calibration diagnostic rather than a deployable predictor.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from fractions import Fraction
from statistics import mean

from predictor_calibration_inputs import wide_range_random
from predictor_shadow_sparse_repair_ablation import (
    DEFAULT_GRAPH_COUNT,
    DEFAULT_INPUT_SEEDS,
    DEFAULT_WIDTH,
    _crosses_boundary,
    _fp32_ulp_fraction,
    _graph,
    _spearman,
    _topological_depths,
)
from summation_graph_predictor import (
    BinaryReductionGraph,
    predict_fp32_tree_error,
    round_nonnegative_fraction_to_fp32,
)

DEFAULT_REPAIR_COUNTS = (1, 2, 4, 8, 16, 32)
POLICIES = ("top_all", "ulp_all", "top_cell", "top_sign", "oracle_gap")


@dataclass(frozen=True)
class TreeRow:
    family: str
    target: float
    shadow: float
    repaired: dict[tuple[str, int], float]
    effective_counts: dict[tuple[str, int], int]


def _sign(value: Fraction) -> int:
    return (value > 0) - (value < 0)


def _analyze(
    values: tuple[Fraction, ...],
    graph: BinaryReductionGraph,
    family: str,
    repair_counts: tuple[int, ...],
) -> TreeRow:
    oracle = predict_fp32_tree_error(values, graph)
    nleaf = graph.leaf_count
    internal = [nleaf + offset for offset in range(len(graph.nodes))]
    depths = _topological_depths(graph)

    exact_subtree = [*values]
    actual_output = [*values]
    actual_delta: dict[int, Fraction] = {}
    shadow_delta: dict[int, Fraction] = {}
    node_ulp: dict[int, Fraction] = {}
    changed_cell: dict[int, bool] = {}
    changed_sign: dict[int, bool] = {}

    for offset, (node, pred) in enumerate(
        zip(graph.nodes, oracle.node_predictions, strict=True)
    ):
        index = nleaf + offset
        exact_sum = exact_subtree[node.left] + exact_subtree[node.right]
        exact_subtree.append(exact_sum)

        actual_output.append(pred.rounded_sum)
        actual_delta[index] = pred.local_rounding_error

        shadow_rounded = round_nonnegative_fraction_to_fp32(exact_sum).value
        shadow_delta[index] = shadow_rounded - exact_sum

        history = (
            actual_output[node.left]
            - exact_subtree[node.left]
            + actual_output[node.right]
            - exact_subtree[node.right]
        )
        ulp = _fp32_ulp_fraction(exact_sum)
        node_ulp[index] = ulp
        changed_cell[index] = _crosses_boundary(
            exact_sum / ulp,
            (exact_sum + history) / ulp,
        )
        changed_sign[index] = _sign(actual_delta[index]) != _sign(shadow_delta[index])

    actual_sum = sum((actual_delta[index] for index in internal), start=Fraction(0))
    if actual_sum != oracle.signed_error:
        raise AssertionError("oracle local-residual identity failed")
    shadow_sum = sum((shadow_delta[index] for index in internal), start=Fraction(0))

    depth_order = sorted(internal, key=lambda index: depths[index], reverse=True)
    ulp_order = sorted(internal, key=node_ulp.__getitem__, reverse=True)
    gap_order = sorted(
        internal,
        key=lambda index: abs(actual_delta[index] - shadow_delta[index]),
        reverse=True,
    )

    repaired: dict[tuple[str, int], float] = {}
    effective_counts: dict[tuple[str, int], int] = {}
    for requested_count in repair_counts:
        budget = min(requested_count, len(internal))
        top = set(depth_order[:budget])
        repair_sets = {
            "top_all": top,
            "ulp_all": set(ulp_order[:budget]),
            "top_cell": {index for index in top if changed_cell[index]},
            "top_sign": {index for index in top if changed_sign[index]},
            "oracle_gap": set(gap_order[:budget]),
        }
        for policy, repair_set in repair_sets.items():
            repaired_sum = sum(
                (
                    actual_delta[index]
                    if index in repair_set
                    else shadow_delta[index]
                )
                for index in internal
            )
            repaired[(policy, requested_count)] = abs(float(repaired_sum))
            effective_counts[(policy, requested_count)] = len(repair_set)

    return TreeRow(
        family=family,
        target=abs(float(actual_sum)),
        shadow=abs(float(shadow_sum)),
        repaired=repaired,
        effective_counts=effective_counts,
    )


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.3f}"


def _fmt_count(value: float) -> str:
    return f"{value:.1f}" if value % 1 else f"{value:.0f}"


def _report(
    label: str,
    rows: list[TreeRow],
    repair_counts: tuple[int, ...],
) -> dict[str, float | None]:
    target = [row.target for row in rows]
    shadow_rho = _spearman([row.shadow for row in rows], target)
    out: dict[str, float | None] = {"shadow": shadow_rho}
    print(
        f"  {label:<10} n={len(rows):2d} target_unique={len(set(target)):2d} "
        f"rho_shadow={_fmt(shadow_rho)}"
    )
    for policy in POLICIES:
        rhos: list[str] = []
        effective: list[str] = []
        for count in repair_counts:
            rho = _spearman(
                [row.repaired[(policy, count)] for row in rows],
                target,
            )
            out[f"{policy}_{count}"] = rho
            rhos.append(_fmt(rho))
            effective.append(
                _fmt_count(mean(row.effective_counts[(policy, count)] for row in rows))
            )
        print(f"    {policy:<10} rho={'/'.join(rhos)} mean_n={'/'.join(effective)}")
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

    print("Shadow high-layer phase-mistake ablation")
    print("CALIBRATION ONLY — all repair policies use oracle FP32 residuals")
    print("candidate execution is never replayed")
    print(
        f"width={args.width} graphs_per_input={args.graphs} "
        f"input_seeds={','.join(map(str, args.input_seeds))} "
        f"repair_counts={','.join(map(str, repair_counts))}"
    )
    print()

    pooled: dict[str, list[float]] = {}
    for input_index, seed in enumerate(args.input_seeds):
        generated = wide_range_random(args.width, seed=seed)
        rows: list[TreeRow] = []
        for graph_index in range(args.graphs):
            family, graph = _graph(len(generated.values), graph_index, input_index)
            rows.append(
                _analyze(
                    generated.values,
                    graph,
                    family,
                    repair_counts,
                )
            )
        print(f"INPUT seed={seed} family={generated.family} width={len(generated.values)}")
        stats = _report("all", rows, repair_counts)
        _report("contiguous", [row for row in rows if row.family == "contiguous"], repair_counts)
        _report("pair_merge", [row for row in rows if row.family == "pair_merge"], repair_counts)
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
