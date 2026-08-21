"""Measure prototype cost components of the selected ancestor cell beam.

This benchmark separates Q metadata, the exact FP32 oracle, the deterministic full-tree shadow
trace, and the incremental B=1/3 beam work after a trace exists.  Model fitting and calibration-label
construction are excluded from inference timing.  The implementation uses Python ``Fraction`` and
is intended only to expose algorithmic/prototype bottlenecks; it is not a hardware performance claim.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

from predictor_ancestor_cell_beam_score_calibration import (
    _beam_scores,
    _beam_tree,
)
from predictor_ancestor_transition_predictability_diagnostic import SHADOW_DIMENSION
from predictor_calibration_inputs import wide_range_random
from predictor_signed_cell_shift_predictability_diagnostic import _fit_probe
from predictor_two_stage_cheap_score_calibration import _graphs, _predictor_trace
from predictor_ulp_energy_cost_pareto_diagnostic import _tree_cost_profile
from summation_graph_predictor import predict_fp32_tree_error


DEFAULT_WIDTHS = (256, 1024)
DEFAULT_GRAPHS_PER_FAMILY = 4
DEFAULT_REPEATS = 1
DEFAULT_INPUT_SEED = 22260821
DEFAULT_BUDGET = 8
DEFAULT_CANDIDATE_COUNT = 64
DEFAULT_SHORTLISTS = (4, 8, 16, 32)


@dataclass(frozen=True)
class CostRow:
    width: int
    tree_count: int
    q_metadata_ms: float
    oracle_ms: float
    shadow_trace_ms: float
    beam1_extra_ms: float
    beam3_extra_ms: float


def _average_ms(call, count: int, repeats: int) -> float:
    start = time.perf_counter()
    for _ in range(repeats):
        call()
    return 1000.0 * (time.perf_counter() - start) / (count * repeats)


def _cascade_average_ms(
    q_metadata_ms: float,
    beam_total_ms: float,
    candidate_count: int,
    shortlist: int,
) -> float:
    """Return per-candidate cost when Q filters all trees and beam reranks M trees."""
    if candidate_count <= 0 or shortlist <= 0 or shortlist > candidate_count:
        raise ValueError("shortlist must be between 1 and candidate_count")
    return q_metadata_ms + shortlist * beam_total_ms / candidate_count


def _benchmark_width(
    width: int,
    graphs_per_family: int,
    repeats: int,
    budget: int,
) -> CostRow:
    values = wide_range_random(width, seed=DEFAULT_INPUT_SEED).values
    graphs = list(_graphs(width, 0, graphs_per_family))
    trees = [
        _beam_tree(values, graph, family, budget)
        for family, graph in graphs
    ]
    training_samples = [sample for tree in trees for sample in tree.transitions]
    # Repetition supplies enough rows for rare innovation classes without changing inference state.
    model = _fit_probe(
        training_samples * 4,
        SHADOW_DIMENSION,
        label="innovation_shift",
    )

    def q_metadata() -> None:
        for family, graph in graphs:
            _tree_cost_profile(values, graph, family, (budget,))

    def oracle() -> None:
        for _, graph in graphs:
            predict_fp32_tree_error(values, graph)

    def shadow_trace() -> None:
        for _, graph in graphs:
            _predictor_trace(values, graph, (budget,))

    def beam1() -> None:
        for tree in trees:
            _beam_scores(tree, model, 1)

    def beam3() -> None:
        for tree in trees:
            _beam_scores(tree, model, 3)

    count = len(graphs)
    return CostRow(
        width=width,
        tree_count=count,
        q_metadata_ms=_average_ms(q_metadata, count, repeats),
        oracle_ms=_average_ms(oracle, count, repeats),
        shadow_trace_ms=_average_ms(shadow_trace, count, repeats),
        beam1_extra_ms=_average_ms(beam1, count, repeats),
        beam3_extra_ms=_average_ms(beam3, count, repeats),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--widths", type=int, nargs="+", default=list(DEFAULT_WIDTHS))
    parser.add_argument(
        "--graphs-per-family",
        type=int,
        default=DEFAULT_GRAPHS_PER_FAMILY,
    )
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--candidate-count", type=int, default=DEFAULT_CANDIDATE_COUNT)
    parser.add_argument(
        "--shortlists",
        type=int,
        nargs="+",
        default=list(DEFAULT_SHORTLISTS),
    )
    args = parser.parse_args()
    if not args.widths or any(width < 2 for width in args.widths):
        parser.error("--widths must contain integers of at least 2")
    if args.graphs_per_family <= 0 or args.repeats <= 0 or args.budget <= 1:
        parser.error("graph count/repeats must be positive and budget must exceed 1")
    if args.candidate_count <= 0:
        parser.error("--candidate-count must be positive")
    if not args.shortlists or any(
        shortlist <= 0 or shortlist > args.candidate_count
        for shortlist in args.shortlists
    ):
        parser.error("--shortlists must be between 1 and --candidate-count")

    print("Selected ancestor-cell beam prototype cost diagnostic")
    print("CALIBRATION ONLY — Python Fraction timings are not hardware claims")
    print("training and label construction excluded; milliseconds per tree")
    print()
    for width in args.widths:
        row = _benchmark_width(
            width,
            args.graphs_per_family,
            args.repeats,
            args.budget,
        )
        beam1_total = row.shadow_trace_ms + row.beam1_extra_ms
        beam3_total = row.shadow_trace_ms + row.beam3_extra_ms
        print(f"width={width} trees={row.tree_count} budget={args.budget}")
        print(
            f"  Q_metadata={row.q_metadata_ms:.3f} oracle={row.oracle_ms:.3f} "
            f"shadow_trace={row.shadow_trace_ms:.3f}"
        )
        print(
            f"  beam_extra B1/B3={row.beam1_extra_ms:.3f}/{row.beam3_extra_ms:.3f} "
            f"total={beam1_total:.3f}/{beam3_total:.3f}"
        )
        print(
            f"  total/oracle B1/B3={beam1_total / row.oracle_ms:.2f}/"
            f"{beam3_total / row.oracle_ms:.2f} "
            f"total/Q B1/B3={beam1_total / row.q_metadata_ms:.2f}/"
            f"{beam3_total / row.q_metadata_ms:.2f}"
        )
        print(f"  cascade amortized over N={args.candidate_count} candidates")
        for shortlist in args.shortlists:
            cascade1 = _cascade_average_ms(
                row.q_metadata_ms,
                beam1_total,
                args.candidate_count,
                shortlist,
            )
            cascade3 = _cascade_average_ms(
                row.q_metadata_ms,
                beam3_total,
                args.candidate_count,
                shortlist,
            )
            print(
                f"    M={shortlist:<2} avg_ms B1/B3={cascade1:.3f}/{cascade3:.3f} "
                f"avg/Q={cascade1 / row.q_metadata_ms:.2f}/"
                f"{cascade3 / row.q_metadata_ms:.2f} "
                f"avg/oracle={cascade1 / row.oracle_ms:.2f}/"
                f"{cascade3 / row.oracle_ms:.2f}"
            )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
