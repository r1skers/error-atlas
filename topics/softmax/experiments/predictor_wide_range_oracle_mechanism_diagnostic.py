"""Calibration-only microscopic rounding diagnostic for wide-range reductions.

This is an oracle/mechanism diagnostic, not a cheap predictor. It intentionally inspects
the actual FP32 intermediate state at every internal node.

For each tree node v:

    fl_v = RN32(a_v + b_v)
    delta_v = fl_v - (a_v + b_v)

and the script verifies the exact tree identity

    root_fp32 - sum_i x_i = sum_v delta_v.

Besides sign counts and cancellation, selected trees can be printed as compact microscopic
traces with cumulative signed error, per-node contribution share, and child magnitude ratio.
Nothing produced here is held-out evidence and no predictor formula is frozen by this run.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from fractions import Fraction
from statistics import mean

from predictor_calibration_inputs import wide_range_random
from predictor_tree_generator import (
    random_contiguous_split_graph,
    random_pair_merge_graph,
)
from summation_graph_predictor import BinaryReductionGraph, predict_fp32_tree_error


DEFAULT_WIDTH = 16
DEFAULT_INPUT_SEEDS = (22260821, 22260822, 22260823, 22260824)
DEFAULT_GRAPH_COUNT = 16
TREE_BASE_SEED = 32_000_000


@dataclass(frozen=True)
class NodeRoundingRecord:
    node_index: int
    left_index: int
    right_index: int
    a: Fraction
    b: Fraction
    exact_sum: Fraction
    rounded_sum: Fraction
    rounded_sum_bits: str
    delta: Fraction


@dataclass(frozen=True)
class TreeRoundingDiagnostic:
    graph_family: str
    graph_seed: int
    graph_name: str
    exact_input_sum: Fraction
    root_fp32: Fraction
    root_signed_error: Fraction
    local_delta_sum: Fraction
    n_positive: int
    n_negative: int
    n_zero: int
    sum_delta_positive: Fraction
    sum_delta_negative: Fraction
    sum_abs_delta: Fraction
    abs_sum_delta: Fraction
    nodes: tuple[NodeRoundingRecord, ...]

    @property
    def cancellation_ratio(self) -> float:
        if self.sum_abs_delta == 0:
            return 0.0
        return 1.0 - float(self.abs_sum_delta / self.sum_abs_delta)

    @property
    def dominant_node_share(self) -> float:
        if self.sum_abs_delta == 0:
            return 0.0
        return max(float(abs(node.delta) / self.sum_abs_delta) for node in self.nodes)


def _graph(width: int, *, graph_index: int, input_index: int):
    seed = TREE_BASE_SEED + input_index * 10_000 + graph_index
    if graph_index % 2 == 0:
        return "contiguous", seed, random_contiguous_split_graph(width, seed=seed)
    return "pair_merge", seed, random_pair_merge_graph(width, seed=seed)


def diagnose_tree(
    values: tuple[Fraction, ...],
    graph: BinaryReductionGraph,
    *,
    graph_family: str,
    graph_seed: int,
) -> TreeRoundingDiagnostic:
    """Replay one tree and expose its exact node-level FP32 rounding residuals."""
    prediction = predict_fp32_tree_error(values, graph)
    states = list(values)
    records: list[NodeRoundingRecord] = []

    for node, node_prediction in zip(graph.nodes, prediction.node_predictions, strict=True):
        a = states[node.left]
        b = states[node.right]
        exact_sum = a + b
        if exact_sum != node_prediction.exact_addend_sum:
            raise AssertionError("oracle replay disagrees on the exact node addend sum")

        rounded_sum = node_prediction.rounded_sum
        delta = rounded_sum - exact_sum
        if delta != node_prediction.local_rounding_error:
            raise AssertionError("oracle replay disagrees on the local rounding error")

        records.append(
            NodeRoundingRecord(
                node_index=node_prediction.node_index,
                left_index=node.left,
                right_index=node.right,
                a=a,
                b=b,
                exact_sum=exact_sum,
                rounded_sum=rounded_sum,
                rounded_sum_bits=node_prediction.rounded_sum_bits,
                delta=delta,
            )
        )
        states.append(rounded_sum)

    root_fp32 = states[graph.root]
    local_delta_sum = sum((record.delta for record in records), start=Fraction(0))
    root_signed_error = root_fp32 - sum(values, start=Fraction(0))

    if root_fp32 != prediction.predicted_sum:
        raise AssertionError("oracle replay did not reproduce the predicted root")
    if root_signed_error != prediction.signed_error:
        raise AssertionError("oracle replay disagrees on the signed root error")
    if local_delta_sum != prediction.local_error_sum:
        raise AssertionError("local delta sum disagrees with the core oracle")
    if local_delta_sum != root_signed_error:
        raise AssertionError("tree identity root_error == sum(delta_v) was violated")

    positive = [record.delta for record in records if record.delta > 0]
    negative = [record.delta for record in records if record.delta < 0]
    sum_delta_positive = sum(positive, start=Fraction(0))
    sum_delta_negative = sum(negative, start=Fraction(0))
    sum_abs_delta = sum((abs(record.delta) for record in records), start=Fraction(0))

    return TreeRoundingDiagnostic(
        graph_family=graph_family,
        graph_seed=graph_seed,
        graph_name=graph.name,
        exact_input_sum=prediction.exact_input_sum,
        root_fp32=root_fp32,
        root_signed_error=root_signed_error,
        local_delta_sum=local_delta_sum,
        n_positive=len(positive),
        n_negative=len(negative),
        n_zero=len(records) - len(positive) - len(negative),
        sum_delta_positive=sum_delta_positive,
        sum_delta_negative=sum_delta_negative,
        sum_abs_delta=sum_abs_delta,
        abs_sum_delta=abs(local_delta_sum),
        nodes=tuple(records),
    )


def _sci(value: Fraction) -> str:
    return f"{float(value):+.9e}"


def _fraction(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _magnitude_ratio(a: Fraction, b: Fraction) -> float:
    """Return min(|a|,|b|)/max(|a|,|b|), in [0,1]."""
    aa = abs(a)
    bb = abs(b)
    high = max(aa, bb)
    if high == 0:
        return 1.0
    return float(min(aa, bb) / high)


def _log2_magnitude_ratio(a: Fraction, b: Fraction) -> float:
    ratio = _magnitude_ratio(a, b)
    if ratio == 0.0:
        return float("-inf")
    return math.log2(ratio)


def _print_tree_summary(
    input_seed: int,
    tree_index: int,
    diagnostic: TreeRoundingDiagnostic,
) -> None:
    print(
        "TREE "
        f"input_seed={input_seed} tree={tree_index:02d} "
        f"family={diagnostic.graph_family} graph_seed={diagnostic.graph_seed} "
        f"root_error={_sci(diagnostic.root_signed_error)} "
        f"sum_delta={_sci(diagnostic.local_delta_sum)} "
        f"identity_ok={diagnostic.root_signed_error == diagnostic.local_delta_sum}"
    )
    print(
        "  signs "
        f"N+={diagnostic.n_positive} N-={diagnostic.n_negative} N0={diagnostic.n_zero} "
        f"sum_delta+={_sci(diagnostic.sum_delta_positive)} "
        f"sum_delta-={_sci(diagnostic.sum_delta_negative)}"
    )
    print(
        "  cancellation "
        f"sum_abs_delta={_sci(diagnostic.sum_abs_delta)} "
        f"abs_sum_delta={_sci(diagnostic.abs_sum_delta)} "
        f"ratio={diagnostic.cancellation_ratio:.6f} "
        f"dominant_node_share={diagnostic.dominant_node_share:.6f}"
    )


def _print_node_trace(
    input_seed: int,
    tree_index: int,
    diagnostic: TreeRoundingDiagnostic,
    *,
    compact: bool,
) -> None:
    cumulative = Fraction(0)
    if compact:
        print(
            "  compact_trace columns: "
            "node left right delta cumulative delta_share child_ratio log2_child_ratio"
        )
    else:
        print(
            "  node_trace columns: "
            "input_seed tree family graph_seed node left right "
            "a b fl32 delta cumulative delta_share child_ratio log2_child_ratio "
            "fl32_bits delta_exact"
        )

    for record in diagnostic.nodes:
        cumulative += record.delta
        share = (
            0.0
            if diagnostic.sum_abs_delta == 0
            else float(abs(record.delta) / diagnostic.sum_abs_delta)
        )
        ratio = _magnitude_ratio(record.a, record.b)
        log2_ratio = _log2_magnitude_ratio(record.a, record.b)

        if compact:
            print(
                "  NODE "
                f"{record.node_index:>3d} {record.left_index:>3d} {record.right_index:>3d} "
                f"{_sci(record.delta)} {_sci(cumulative)} "
                f"{share:.4f} {ratio:.4e} {log2_ratio:+.2f}"
            )
        else:
            print(
                "  NODE "
                f"{input_seed} {tree_index:02d} {diagnostic.graph_family} "
                f"{diagnostic.graph_seed} {record.node_index} "
                f"{record.left_index} {record.right_index} "
                f"{float(record.a):.9e} {float(record.b):.9e} "
                f"{float(record.rounded_sum):.9e} {_sci(record.delta)} "
                f"{_sci(cumulative)} {share:.6f} {ratio:.9e} {log2_ratio:+.6f} "
                f"{record.rounded_sum_bits} {_fraction(record.delta)}"
            )

    if cumulative != diagnostic.root_signed_error:
        raise AssertionError("printed cumulative delta does not reach root error")


def _print_family_summary(
    family: str,
    diagnostics: list[TreeRoundingDiagnostic],
) -> None:
    if not diagnostics:
        return
    print(
        f"  {family:<10} trees={len(diagnostics):2d} "
        f"mean|root_error|={mean(float(abs(item.root_signed_error)) for item in diagnostics):.9e} "
        f"mean_sum|delta|={mean(float(item.sum_abs_delta) for item in diagnostics):.9e} "
        f"mean_cancel={mean(item.cancellation_ratio for item in diagnostics):.6f} "
        f"mean_dominant_share={mean(item.dominant_node_share for item in diagnostics):.6f} "
        f"mean_N+={mean(item.n_positive for item in diagnostics):.2f} "
        f"mean_N-={mean(item.n_negative for item in diagnostics):.2f}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--width",
        type=int,
        choices=(8, 16),
        default=DEFAULT_WIDTH,
        help="microscopic leaf count; intentionally restricted to 8 or 16",
    )
    parser.add_argument(
        "--graphs",
        type=int,
        default=DEFAULT_GRAPH_COUNT,
        help="number of trees per input; alternates contiguous and pair-merge",
    )
    parser.add_argument(
        "--input-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_INPUT_SEEDS),
        help="wide-range stored-FP32 calibration seeds",
    )
    parser.add_argument(
        "--show-nodes",
        action="store_true",
        help="print full internal-node trace for selected trees",
    )
    parser.add_argument(
        "--compact-trace",
        action="store_true",
        help="print compact node trace for selected trees",
    )
    parser.add_argument(
        "--trace-trees",
        type=int,
        nargs="+",
        default=None,
        metavar="INDEX",
        help="only expand these zero-based tree indices; summaries are still printed for all trees",
    )
    args = parser.parse_args()
    if args.graphs <= 0:
        parser.error("--graphs must be positive")
    if args.trace_trees is not None:
        bad = [index for index in args.trace_trees if not 0 <= index < args.graphs]
        if bad:
            parser.error(f"--trace-trees indices must be in [0,{args.graphs - 1}]: {bad}")
    if args.show_nodes and args.compact_trace:
        parser.error("choose at most one of --show-nodes and --compact-trace")
    return args


def main() -> int:
    args = _parse_args()
    trace_tree_set = None if args.trace_trees is None else set(args.trace_trees)

    print("Wide-range microscopic FP32 rounding diagnostic")
    print("CALIBRATION ONLY — exact intermediate states are intentionally inspected")
    print("ORACLE/MECHANISM DIAGNOSTIC — not a predictor and not held-out evidence")
    print(
        f"width={args.width} graphs_per_input={args.graphs} "
        f"input_seeds={','.join(str(seed) for seed in args.input_seeds)}"
    )
    if args.trace_trees is not None:
        print("trace_trees=" + ",".join(f"{index:02d}" for index in args.trace_trees))
    print()

    for input_index, input_seed in enumerate(args.input_seeds):
        generated = wide_range_random(args.width, seed=input_seed)
        diagnostics: list[TreeRoundingDiagnostic] = []

        print(
            f"INPUT seed={generated.seed} family={generated.family} "
            f"width={len(generated.values)}"
        )
        for graph_index in range(args.graphs):
            graph_family, graph_seed, graph = _graph(
                len(generated.values),
                graph_index=graph_index,
                input_index=input_index,
            )
            diagnostic = diagnose_tree(
                generated.values,
                graph,
                graph_family=graph_family,
                graph_seed=graph_seed,
            )
            diagnostics.append(diagnostic)
            _print_tree_summary(input_seed, graph_index, diagnostic)

            should_trace = trace_tree_set is None or graph_index in trace_tree_set
            if should_trace and args.show_nodes:
                _print_node_trace(input_seed, graph_index, diagnostic, compact=False)
            if should_trace and args.compact_trace:
                _print_node_trace(input_seed, graph_index, diagnostic, compact=True)

        print("INPUT SUMMARY")
        _print_family_summary("all", diagnostics)
        _print_family_summary(
            "contiguous",
            [item for item in diagnostics if item.graph_family == "contiguous"],
        )
        _print_family_summary(
            "pair_merge",
            [item for item in diagnostics if item.graph_family == "pair_merge"],
        )
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
