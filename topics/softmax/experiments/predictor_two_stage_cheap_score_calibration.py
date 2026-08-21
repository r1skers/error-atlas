"""Calibrate a two-stage macro-energy plus sparse-coherence cheap score.

The macro layer uses the normalized ULP energy

    Q = sum_v (U_v / U_root)**2,

while a structural root-band traversal selects at most K important internal nodes B_K.  The micro
layer computes history-free residuals delta0 and one non-recursive first-order correction delta1:

    H0_v     = sum of delta0 over proper descendants of v,
    delta1_v = RN32(S_v + H0_v) - (S_v + H0_v).

No delta1 value is propagated into an ancestor, so this is not a replay of the candidate FP32 tree.
The primary two-stage ablation preserves the macro local-energy term and adds predicted coherence
only among selected nodes:

    coherence_first(K)
      = Q / 12
        + (sum_{v in B_K} delta1_v / U_root)**2
        - sum_{v in B_K} (delta1_v / U_root)**2.

The final two terms are exactly the selected pairwise cross terms.  ``coherence_shadow`` substitutes
delta0; ``coherence_trajectory`` uses the previously diagnosed recursive binary64 shadow; and
``coherence_phase`` uses the trajectory residual only when that shadow expects an RN-cell crossing
or a residual-sign change.  All unselected pairs remain zero rather than being independently
propagated.  A prior sparse-first-order score that keeps the deterministic full shadow tail is
retained as a baseline.

All score values use only stored FP32 leaves and the graph.  Exact candidate execution is used after
scoring solely for the within-input ranking target and for diagnostic sign/crossing/phase metrics.
Inputs and graphs are calibration-only; held-out inputs remain untouched and no K is frozen.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
from statistics import mean

from predictor_calibration_inputs import wide_range_random
from predictor_gaussian_ancestor_coherence_calibration import _fp32_ulp
from predictor_ranking_smoke import _spearman
from predictor_shadow_sparse_repair_ablation import (
    _crosses_boundary,
    _fp32_ulp_fraction,
)
from predictor_shadow_trajectory_failure_diagnostic import _boundary_cross_count
from predictor_tree_generator import (
    random_contiguous_split_graph,
    random_pair_merge_graph,
)
from predictor_ulp_energy_cost_pareto_diagnostic import _root_band_internal_order
from summation_graph_predictor import (
    BinaryReductionGraph,
    predict_fp32_tree_error,
    round_nonnegative_fraction_to_fp32,
)


DEFAULT_WIDTH = 256
DEFAULT_INPUT_SEEDS = (22260821, 22260822, 22260823, 22260824)
DEFAULT_RANDOM_GRAPHS_PER_FAMILY = 32
DEFAULT_BUDGETS = (4, 8, 16, 32)
CONTIGUOUS_TREE_BASE_SEED = 45_000_000
PAIR_TREE_BASE_SEED = 46_000_000
FAMILIES = ("contiguous", "pair_merge")


@dataclass(frozen=True)
class BudgetScores:
    q_budget: float
    coherence_shadow: float
    coherence_first: float
    coherence_trajectory: float
    coherence_phase: float
    sparse_first: float


@dataclass(frozen=True)
class PredictorTrace:
    root_ulp: Fraction
    exact_subtree: tuple[Fraction, ...]
    node_ulp: dict[int, Fraction]
    delta0: dict[int, Fraction]
    delta1: dict[int, Fraction]
    trajectory_delta: dict[int, float]
    predicted_cross: dict[int, bool]
    predicted_phase: dict[int, bool]
    selected_order: tuple[int, ...]
    full_q: float
    history_free: float
    full_first: float
    trajectory: float
    budget: dict[int, BudgetScores]


@dataclass(frozen=True)
class MicroCounts:
    selected: int
    shadow_sign_correct: int
    first_sign_correct: int
    trajectory_sign_correct: int
    predicted_cross: int
    actual_cross: int
    cross_true_positive: int
    predicted_phase: int
    actual_phase: int
    phase_true_positive: int


@dataclass(frozen=True)
class TreeRow:
    family: str
    target: float
    trace: PredictorTrace
    micro: dict[int, MicroCounts]


def _sign(value: Fraction | float) -> int:
    return (value > 0) - (value < 0)


def _predictor_trace(
    values: tuple[Fraction, ...],
    graph: BinaryReductionGraph,
    budgets: tuple[int, ...],
) -> PredictorTrace:
    """Compute every score-side quantity without executing the candidate tree."""
    if len(values) != graph.leaf_count:
        raise ValueError("value count must match graph leaf count")
    if not budgets or any(budget <= 0 for budget in budgets):
        raise ValueError("budgets must contain positive integers")

    exact_subtree = [*values]
    subtree_leaves = [1] * graph.leaf_count
    first_order_error = [Fraction(0) for _ in values]
    node_ulp: dict[int, Fraction] = {}
    delta0: dict[int, Fraction] = {}
    delta1: dict[int, Fraction] = {}
    predicted_cross: dict[int, bool] = {}
    predicted_phase: dict[int, bool] = {}

    for offset, node in enumerate(graph.nodes):
        index = graph.leaf_count + offset
        exact_sum = exact_subtree[node.left] + exact_subtree[node.right]
        exact_subtree.append(exact_sum)
        subtree_leaves.append(subtree_leaves[node.left] + subtree_leaves[node.right])

        ulp = _fp32_ulp_fraction(exact_sum)
        node_ulp[index] = ulp
        rounded0 = round_nonnegative_fraction_to_fp32(exact_sum).value
        delta0[index] = rounded0 - exact_sum

        history0 = first_order_error[node.left] + first_order_error[node.right]
        shifted_sum = exact_sum + history0
        rounded1 = round_nonnegative_fraction_to_fp32(shifted_sum).value
        delta1[index] = rounded1 - shifted_sum
        # Strictly first order: delta1 is never fed into an ancestor history.
        first_order_error.append(history0 + delta0[index])

    # This is the previously diagnosed deterministic binary64 shadow trajectory.  It recursively
    # propagates its own approximate error, never the exact candidate FP32 output/history.
    shadow_subtree = [float(value) for value in values]
    shadow_error = [0.0 for _ in values]
    trajectory_delta: dict[int, float] = {}
    for offset, node in enumerate(graph.nodes):
        index = graph.leaf_count + offset
        exact_shadow_sum = shadow_subtree[node.left] + shadow_subtree[node.right]
        ulp_float = _fp32_ulp(exact_shadow_sum)
        phase = exact_shadow_sum / ulp_float
        phase -= int(phase // 1)
        history_shadow = shadow_error[node.left] + shadow_error[node.right]
        shifted_phase = phase + history_shadow / ulp_float
        output_error = ulp_float * (round(shifted_phase) - phase)
        trajectory_delta[index] = output_error - history_shadow
        predicted_cross[index] = (
            _boundary_cross_count(phase, history_shadow / ulp_float) > 0
        )
        predicted_phase[index] = _sign(trajectory_delta[index]) != _sign(delta0[index])
        shadow_subtree.append(exact_shadow_sum)
        shadow_error.append(output_error)

    ordered_budgets = tuple(sorted(set(budgets)))
    selected_order = _root_band_internal_order(
        graph,
        subtree_leaves,
        ordered_budgets[-1],
    )
    internal = tuple(node_ulp)
    root_ulp = _fp32_ulp_fraction(exact_subtree[graph.root])
    energy = {
        index: float(node_ulp[index] / root_ulp) ** 2 for index in internal
    }
    full_q = sum(energy.values())
    shadow_sum = sum((delta0[index] for index in internal), start=Fraction(0))
    full_first_sum = sum((delta1[index] for index in internal), start=Fraction(0))

    budget_scores: dict[int, BudgetScores] = {}
    for requested_budget in ordered_budgets:
        effective_budget = min(requested_budget, len(selected_order))
        selected = selected_order[:effective_budget]
        q_budget = sum(energy[index] for index in selected)
        selected_shadow = [delta0[index] for index in selected]
        selected_first = [delta1[index] for index in selected]
        selected_trajectory = [trajectory_delta[index] for index in selected]
        selected_phase = [
            (
                trajectory_delta[index]
                if predicted_cross[index] or predicted_phase[index]
                else float(delta0[index])
            )
            for index in selected
        ]
        sparse_first_sum = shadow_sum + sum(
            (delta1[index] - delta0[index] for index in selected),
            start=Fraction(0),
        )

        def normalized_square(value: Fraction) -> float:
            return float(value / root_ulp) ** 2

        def coherence_score(predicted: list[Fraction | float]) -> float:
            normalized = [float(value) / float(root_ulp) for value in predicted]
            pairwise = sum(normalized) ** 2 - sum(value * value for value in normalized)
            return max(0.0, full_q / 12.0 + pairwise)

        budget_scores[requested_budget] = BudgetScores(
            q_budget=q_budget,
            coherence_shadow=coherence_score(selected_shadow),
            coherence_first=coherence_score(selected_first),
            coherence_trajectory=coherence_score(selected_trajectory),
            coherence_phase=coherence_score(selected_phase),
            sparse_first=normalized_square(sparse_first_sum),
        )

    return PredictorTrace(
        root_ulp=root_ulp,
        exact_subtree=tuple(exact_subtree),
        node_ulp=node_ulp,
        delta0=delta0,
        delta1=delta1,
        trajectory_delta=trajectory_delta,
        predicted_cross=predicted_cross,
        predicted_phase=predicted_phase,
        selected_order=selected_order,
        full_q=full_q,
        history_free=float(shadow_sum / root_ulp) ** 2,
        full_first=float(full_first_sum / root_ulp) ** 2,
        trajectory=(shadow_error[-1] / float(root_ulp)) ** 2,
        budget=budget_scores,
    )


def _analyze(
    values: tuple[Fraction, ...],
    graph: BinaryReductionGraph,
    family: str,
    budgets: tuple[int, ...],
) -> TreeRow:
    trace = _predictor_trace(values, graph, budgets)
    oracle = predict_fp32_tree_error(values, graph)
    target = float(oracle.signed_error / trace.root_ulp) ** 2

    actual_output = [*values]
    actual_cross: dict[int, bool] = {}
    actual_phase: dict[int, bool] = {}
    actual_delta: dict[int, Fraction] = {}
    for offset, (node, prediction) in enumerate(
        zip(graph.nodes, oracle.node_predictions, strict=True)
    ):
        index = graph.leaf_count + offset
        actual_output.append(prediction.rounded_sum)
        actual_delta[index] = prediction.local_rounding_error
        history = (
            actual_output[node.left]
            - trace.exact_subtree[node.left]
            + actual_output[node.right]
            - trace.exact_subtree[node.right]
        )
        exact_sum = trace.exact_subtree[index]
        ulp = trace.node_ulp[index]
        actual_cross[index] = _crosses_boundary(
            exact_sum / ulp,
            (exact_sum + history) / ulp,
        )
        actual_phase[index] = _sign(actual_delta[index]) != _sign(trace.delta0[index])

    micro: dict[int, MicroCounts] = {}
    for requested_budget in sorted(set(budgets)):
        selected = trace.selected_order[: min(requested_budget, len(trace.selected_order))]
        micro[requested_budget] = MicroCounts(
            selected=len(selected),
            shadow_sign_correct=sum(
                _sign(trace.delta0[index]) == _sign(actual_delta[index])
                for index in selected
            ),
            first_sign_correct=sum(
                _sign(trace.delta1[index]) == _sign(actual_delta[index])
                for index in selected
            ),
            trajectory_sign_correct=sum(
                _sign(trace.trajectory_delta[index]) == _sign(actual_delta[index])
                for index in selected
            ),
            predicted_cross=sum(trace.predicted_cross[index] for index in selected),
            actual_cross=sum(actual_cross[index] for index in selected),
            cross_true_positive=sum(
                trace.predicted_cross[index] and actual_cross[index]
                for index in selected
            ),
            predicted_phase=sum(trace.predicted_phase[index] for index in selected),
            actual_phase=sum(actual_phase[index] for index in selected),
            phase_true_positive=sum(
                trace.predicted_phase[index] and actual_phase[index]
                for index in selected
            ),
        )
    return TreeRow(family=family, target=target, trace=trace, micro=micro)


def _graphs(width: int, input_index: int, count: int):
    for graph_index in range(count):
        contiguous_seed = (
            CONTIGUOUS_TREE_BASE_SEED + input_index * 10_000 + graph_index
        )
        pair_seed = PAIR_TREE_BASE_SEED + input_index * 10_000 + graph_index
        yield "contiguous", random_contiguous_split_graph(
            width,
            seed=contiguous_seed,
        )
        yield "pair_merge", random_pair_merge_graph(width, seed=pair_seed)


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.3f}"


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _report(
    label: str,
    rows: list[TreeRow],
    budgets: tuple[int, ...],
) -> dict[str, float | None]:
    target = [row.target for row in rows]
    out: dict[str, float | None] = {
        "full_q": _spearman([row.trace.full_q for row in rows], target),
        "history_free": _spearman(
            [row.trace.history_free for row in rows],
            target,
        ),
        "trajectory": _spearman([row.trace.trajectory for row in rows], target),
        "full_first": _spearman([row.trace.full_first for row in rows], target),
    }
    print(
        f"  {label:<10} n={len(rows):3d} target_unique={len(set(target)):3d} "
        f"rho fullQ/d0/traj/first="
        f"{_fmt(out['full_q'])}/{_fmt(out['history_free'])}/"
        f"{_fmt(out['trajectory'])}/{_fmt(out['full_first'])}"
    )

    for budget in budgets:
        policies = {
            "qK": [row.trace.budget[budget].q_budget for row in rows],
            "coherence0": [row.trace.budget[budget].coherence_shadow for row in rows],
            "coherence1": [row.trace.budget[budget].coherence_first for row in rows],
            "trajectory": [
                row.trace.budget[budget].coherence_trajectory for row in rows
            ],
            "phase": [row.trace.budget[budget].coherence_phase for row in rows],
            "sparse": [row.trace.budget[budget].sparse_first for row in rows],
        }
        rhos = {name: _spearman(values, target) for name, values in policies.items()}
        for name, rho in rhos.items():
            out[f"{name}_{budget}"] = rho

        counts = [row.micro[budget] for row in rows]
        selected = sum(count.selected for count in counts)
        shadow_sign_accuracy = _ratio(
            sum(count.shadow_sign_correct for count in counts),
            selected,
        )
        first_sign_accuracy = _ratio(
            sum(count.first_sign_correct for count in counts),
            selected,
        )
        trajectory_sign_accuracy = _ratio(
            sum(count.trajectory_sign_correct for count in counts),
            selected,
        )
        cross_precision = _ratio(
            sum(count.cross_true_positive for count in counts),
            sum(count.predicted_cross for count in counts),
        )
        cross_recall = _ratio(
            sum(count.cross_true_positive for count in counts),
            sum(count.actual_cross for count in counts),
        )
        phase_precision = _ratio(
            sum(count.phase_true_positive for count in counts),
            sum(count.predicted_phase for count in counts),
        )
        phase_recall = _ratio(
            sum(count.phase_true_positive for count in counts),
            sum(count.actual_phase for count in counts),
        )
        print(
            f"    K={budget:<2d} rho qK/c0/c1/ctraj/phase/sparse="
            f"{_fmt(rhos['qK'])}/{_fmt(rhos['coherence0'])}/"
            f"{_fmt(rhos['coherence1'])}/{_fmt(rhos['trajectory'])}/"
            f"{_fmt(rhos['phase'])}/{_fmt(rhos['sparse'])} "
            f"micro signAcc0/1/traj={_fmt(shadow_sign_accuracy)}/"
            f"{_fmt(first_sign_accuracy)}/{_fmt(trajectory_sign_accuracy)} "
            f"crossP/R={_fmt(cross_precision)}/{_fmt(cross_recall)} "
            f"phaseP/R={_fmt(phase_precision)}/{_fmt(phase_recall)}"
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
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
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=list(DEFAULT_BUDGETS),
    )
    args = parser.parse_args()
    if args.width < 2:
        parser.error("--width must be at least 2")
    if args.random_graphs_per_family < 2:
        parser.error("--random-graphs-per-family must be at least 2")
    if not args.budgets or any(budget <= 0 for budget in args.budgets):
        parser.error("--budgets must contain positive integers")
    budgets = tuple(sorted(set(args.budgets)))

    print("Two-stage macro-energy + sparse-coherence cheap-score calibration")
    print("CALIBRATION ONLY — no held-out inputs; formula and K are not frozen")
    print("PREDICTOR SIDE — stored FP32 leaves + graph only; oracle is target/diagnostic only")
    print("delta1 is strictly first-order and never recursively replayed")
    print(
        f"width={args.width} "
        f"input_seeds={','.join(map(str, args.input_seeds))} "
        f"random_graphs_per_family={args.random_graphs_per_family} "
        f"budgets={','.join(map(str, budgets))}"
    )
    print()

    pooled: dict[str, list[float]] = {}
    for input_index, input_seed in enumerate(args.input_seeds):
        values = wide_range_random(args.width, seed=input_seed).values
        rows: list[TreeRow] = []
        for family, graph in _graphs(
            args.width,
            input_index,
            args.random_graphs_per_family,
        ):
            rows.append(_analyze(values, graph, family, budgets))

        print(f"INPUT seed={input_seed} family=wide_range_random width={args.width}")
        stats = _report("all", rows, budgets)
        for family in FAMILIES:
            _report(family, [row for row in rows if row.family == family], budgets)
        for key, value in stats.items():
            if value is not None:
                pooled.setdefault(key, []).append(value)
        print()

    print("SEED SUMMARY all-tree rho mean/min/max")
    for key, values in pooled.items():
        print(
            f"  {key:<18} mean={mean(values):+.3f} "
            f"min={min(values):+.3f} max={max(values):+.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
