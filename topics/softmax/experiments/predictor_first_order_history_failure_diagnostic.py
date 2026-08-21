"""Diagnose why the deterministic first-order history estimate fails at high layers.

The sparse oracle ablation found that a handful of local-residual corrections can recover tree
ranking, but the predictor-side estimate

    H0_v = sum_{u proper-descendant of v} delta0_u,
    delta0_u = RN32(S_u) - S_u,

did not turn that sparsity into a useful score.  This calibration-only diagnostic separates three
possible failures in the top-of-tree band:

* history alignment: signed/scale agreement between H0_v and the actual H_v;
* phase alignment: whether H0_v predicts the same RN-even cell change as H_v;
* selector alignment: whether cheap node orderings recover the largest oracle correction gaps
  |delta_actual-delta0|.

All predictor-side quantities use only stored FP32 leaves and the candidate graph.  Actual history,
residuals, and oracle top-k sets are used only for diagnosis.  Held-out inputs remain untouched.
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
    _crosses_boundary,
    _fp32_ulp_fraction,
    _graph,
    _pearson,
    _spearman,
    _topological_depths,
)
from summation_graph_predictor import (
    BinaryReductionGraph,
    predict_fp32_tree_error,
    round_nonnegative_fraction_to_fp32,
)

SELECTORS = ("depth", "pred_gap", "pred_history", "ulp")


@dataclass(frozen=True)
class NodeRow:
    index: int
    depth: int
    actual_history_ulp: float
    predicted_history_ulp: float
    actual_gap_ulp: float
    predicted_gap_ulp: float
    actual_cell_change: bool
    predicted_cell_change: bool
    ulp: float


@dataclass(frozen=True)
class TreeRow:
    family: str
    nodes: tuple[NodeRow, ...]
    orders: dict[str, tuple[int, ...]]


@dataclass(frozen=True)
class Alignment:
    history_corr: float | None
    history_sign_match: float
    history_scale_ratio: float
    history_mae_ulp: float
    gap_corr: float | None
    gap_abs_rho: float | None
    gap_sign_match: float
    cell_precision: float | None
    cell_recall: float | None
    actual_cell_rate: float
    predicted_cell_rate: float


@dataclass(frozen=True)
class SelectionQuality:
    recall: float
    mass_recovery: float


def _sign(value: float) -> int:
    return (value > 0.0) - (value < 0.0)


def _analyze(
    values: tuple[Fraction, ...],
    graph: BinaryReductionGraph,
    family: str,
) -> TreeRow:
    oracle = predict_fp32_tree_error(values, graph)
    nleaf = graph.leaf_count
    depths = _topological_depths(graph)

    exact_subtree = [*values]
    actual_output = [*values]
    first_order_error = [Fraction(0) for _ in values]
    rows: list[NodeRow] = []

    for offset, (node, prediction) in enumerate(
        zip(graph.nodes, oracle.node_predictions, strict=True)
    ):
        index = nleaf + offset
        exact_sum = exact_subtree[node.left] + exact_subtree[node.right]
        exact_subtree.append(exact_sum)

        actual_output.append(prediction.rounded_sum)
        actual_history = (
            actual_output[node.left]
            - exact_subtree[node.left]
            + actual_output[node.right]
            - exact_subtree[node.right]
        )

        rounded0 = round_nonnegative_fraction_to_fp32(exact_sum).value
        delta0 = rounded0 - exact_sum
        predicted_history = (
            first_order_error[node.left] + first_order_error[node.right]
        )
        shifted_sum = exact_sum + predicted_history
        rounded1 = round_nonnegative_fraction_to_fp32(shifted_sum).value
        delta1 = rounded1 - shifted_sum
        first_order_error.append(predicted_history + delta0)

        ulp = _fp32_ulp_fraction(exact_sum)
        actual_gap = prediction.local_rounding_error - delta0
        predicted_gap = delta1 - delta0
        rows.append(
            NodeRow(
                index=index,
                depth=depths[index],
                actual_history_ulp=float(actual_history / ulp),
                predicted_history_ulp=float(predicted_history / ulp),
                actual_gap_ulp=float(actual_gap / ulp),
                predicted_gap_ulp=float(predicted_gap / ulp),
                actual_cell_change=_crosses_boundary(
                    exact_sum / ulp,
                    (exact_sum + actual_history) / ulp,
                ),
                predicted_cell_change=_crosses_boundary(
                    exact_sum / ulp,
                    (exact_sum + predicted_history) / ulp,
                ),
                ulp=float(ulp),
            )
        )

    by_index = {row.index: row for row in rows}
    indices = [row.index for row in rows]
    orders = {
        "depth": tuple(
            sorted(indices, key=lambda index: by_index[index].depth, reverse=True)
        ),
        "pred_gap": tuple(
            sorted(
                indices,
                key=lambda index: (
                    abs(by_index[index].predicted_gap_ulp) * by_index[index].ulp
                ),
                reverse=True,
            )
        ),
        "pred_history": tuple(
            sorted(
                indices,
                key=lambda index: abs(by_index[index].predicted_history_ulp),
                reverse=True,
            )
        ),
        "ulp": tuple(
            sorted(indices, key=lambda index: by_index[index].ulp, reverse=True)
        ),
        "oracle_gap": tuple(
            sorted(
                indices,
                key=lambda index: (
                    abs(by_index[index].actual_gap_ulp) * by_index[index].ulp
                ),
                reverse=True,
            )
        ),
    }
    return TreeRow(family=family, nodes=tuple(rows), orders=orders)


def _top_nodes(tree: TreeRow, count: int) -> list[NodeRow]:
    by_index = {row.index: row for row in tree.nodes}
    return [by_index[index] for index in tree.orders["depth"][:count]]


def _alignment(trees: list[TreeRow], count: int) -> Alignment:
    nodes = [node for tree in trees for node in _top_nodes(tree, count)]
    actual_history = [node.actual_history_ulp for node in nodes]
    predicted_history = [node.predicted_history_ulp for node in nodes]
    actual_gap = [node.actual_gap_ulp for node in nodes]
    predicted_gap = [node.predicted_gap_ulp for node in nodes]

    actual_abs_mean = mean(abs(value) for value in actual_history)
    predicted_abs_mean = mean(abs(value) for value in predicted_history)
    true_positive = sum(
        node.actual_cell_change and node.predicted_cell_change for node in nodes
    )
    predicted_positive = sum(node.predicted_cell_change for node in nodes)
    actual_positive = sum(node.actual_cell_change for node in nodes)

    return Alignment(
        history_corr=_pearson(predicted_history, actual_history),
        history_sign_match=mean(
            _sign(predicted) == _sign(actual)
            for predicted, actual in zip(
                predicted_history,
                actual_history,
                strict=True,
            )
        ),
        history_scale_ratio=(
            predicted_abs_mean / actual_abs_mean if actual_abs_mean else 0.0
        ),
        history_mae_ulp=mean(
            abs(predicted - actual)
            for predicted, actual in zip(
                predicted_history,
                actual_history,
                strict=True,
            )
        ),
        gap_corr=_pearson(predicted_gap, actual_gap),
        gap_abs_rho=_spearman(
            [abs(value) for value in predicted_gap],
            [abs(value) for value in actual_gap],
        ),
        gap_sign_match=mean(
            _sign(predicted) == _sign(actual)
            for predicted, actual in zip(predicted_gap, actual_gap, strict=True)
        ),
        cell_precision=(
            true_positive / predicted_positive if predicted_positive else None
        ),
        cell_recall=true_positive / actual_positive if actual_positive else None,
        actual_cell_rate=actual_positive / len(nodes),
        predicted_cell_rate=predicted_positive / len(nodes),
    )


def _selection_quality(
    tree: TreeRow,
    selector: str,
    count: int,
) -> SelectionQuality:
    budget = min(count, len(tree.nodes))
    if budget == 0:
        return SelectionQuality(recall=0.0, mass_recovery=0.0)
    by_index = {row.index: row for row in tree.nodes}
    oracle = set(tree.orders["oracle_gap"][:budget])
    selected = set(tree.orders[selector][:budget])
    oracle_mass = sum(
        abs(by_index[index].actual_gap_ulp) * by_index[index].ulp
        for index in oracle
    )
    selected_mass = sum(
        abs(by_index[index].actual_gap_ulp) * by_index[index].ulp
        for index in selected
    )
    return SelectionQuality(
        recall=len(oracle & selected) / budget,
        mass_recovery=selected_mass / oracle_mass if oracle_mass else 1.0,
    )


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.3f}"


def _fmt_unsigned(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _report_alignment(
    label: str,
    trees: list[TreeRow],
    counts: tuple[int, ...],
) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    print(f"  {label:<10} trees={len(trees):2d}")
    for count in counts:
        result = _alignment(trees, count)
        out[f"h_corr_{count}"] = result.history_corr
        out[f"gap_abs_rho_{count}"] = result.gap_abs_rho
        print(
            f"    top{count:<2d} Hcorr={_fmt(result.history_corr)} "
            f"Hsign={result.history_sign_match:.3f} "
            f"Hscale={result.history_scale_ratio:.3f} "
            f"Hmae={result.history_mae_ulp:.3f} "
            f"gapCorr={_fmt(result.gap_corr)} "
            f"gapAbsRho={_fmt(result.gap_abs_rho)} "
            f"gapSign={result.gap_sign_match:.3f} "
            f"cellP/R={_fmt_unsigned(result.cell_precision)}/"
            f"{_fmt_unsigned(result.cell_recall)} "
            f"cellRate(a/p)={result.actual_cell_rate:.3f}/"
            f"{result.predicted_cell_rate:.3f}"
        )
    return out


def _report_selectors(
    trees: list[TreeRow],
    counts: tuple[int, ...],
) -> dict[str, float]:
    out: dict[str, float] = {}
    print("  oracle-gap top-k selector quality: mean recall / abs-gap-mass recovery")
    for selector in SELECTORS:
        values: list[str] = []
        for count in counts:
            quality = [
                _selection_quality(tree, selector, count) for tree in trees
            ]
            recall = mean(item.recall for item in quality)
            mass = mean(item.mass_recovery for item in quality)
            out[f"{selector}_recall_{count}"] = recall
            out[f"{selector}_mass_{count}"] = mass
            values.append(f"{recall:.3f}/{mass:.3f}")
        print(f"    {selector:<12} {'  '.join(values)}")
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
        "--top-counts",
        type=int,
        nargs="+",
        default=list(DEFAULT_REPAIR_COUNTS),
    )
    args = parser.parse_args()
    if args.width <= 1:
        parser.error("--width must exceed 1")
    if args.graphs <= 1:
        parser.error("--graphs must exceed 1")
    if any(count <= 0 for count in args.top_counts):
        parser.error("--top-counts must contain only positive integers")
    counts = tuple(dict.fromkeys(args.top_counts))

    print("First-order high-layer history failure diagnostic")
    print("CALIBRATION ONLY — actual histories and oracle gaps are diagnostic only")
    print("PREDICTOR SIDE — H0 uses stored FP32 leaves + graph only")
    print(
        f"width={args.width} graphs_per_input={args.graphs} "
        f"input_seeds={','.join(map(str, args.input_seeds))} "
        f"top_counts={','.join(map(str, counts))}"
    )
    print()

    seed_stats: list[dict[str, float | None]] = []
    for input_index, seed in enumerate(args.input_seeds):
        generated = wide_range_random(args.width, seed=seed)
        trees: list[TreeRow] = []
        for graph_index in range(args.graphs):
            family, graph = _graph(len(generated.values), graph_index, input_index)
            trees.append(_analyze(generated.values, graph, family))

        print(f"INPUT seed={seed} family={generated.family} width={len(generated.values)}")
        stats = _report_alignment("all", trees, counts)
        _report_alignment(
            "contiguous",
            [tree for tree in trees if tree.family == "contiguous"],
            counts,
        )
        _report_alignment(
            "pair_merge",
            [tree for tree in trees if tree.family == "pair_merge"],
            counts,
        )
        stats.update(_report_selectors(trees, counts))
        seed_stats.append(stats)
        print()

    summary_keys = [
        key
        for key in seed_stats[0]
        if key.startswith("h_corr_")
        or key.startswith("gap_abs_rho_")
        or key.startswith("depth_recall_")
        or key.startswith("pred_gap_recall_")
    ]
    print("SEED SUMMARY mean/min/max")
    for key in summary_keys:
        values = [
            float(stats[key])
            for stats in seed_stats
            if stats.get(key) is not None
        ]
        print(
            f"  {key:<24} mean={mean(values):+.3f} "
            f"min={min(values):+.3f} max={max(values):+.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
