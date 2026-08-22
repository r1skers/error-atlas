"""Calibrate a small joint cell-state beam over the selected root band.

The signed cell-shift diagnostic shows transferable directional information at K=8, but directly
multiplying node marginals would repeat the failed independent-coherence approximation.  This
ablation instead keeps explicit FP32 output-cell states on the connected root-band subtree.

For a selected node, already-selected child states are combined and rounded deterministically.
Unselected child branches use the deterministic shadow output.  A five-state learned innovation
``{-2,-1,0,+1,+2}`` is then applied to the deterministic parent cell.  The innovation label used for
training is exactly the actual parent cell minus the cell obtained from actual selected children and
shadow unselected children.  Thus selected-child corrections are propagated once, while only the
unobserved frontier contribution is learned.

Each node retains the B most probable output cells.  The root score is the beam expectation of
squared signed error in root-ULP units.  B=1,3,5 are compared with Q_K/12 and the deterministic shadow
root score.  Same-width calibration uses leave-one-input-seed-out fitting; cross-width mode transfers
one fixed model and may use disjoint evaluation seeds.

This is calibration-only.  Oracle execution supplies training labels and evaluation targets but is
never read by an evaluated tree's score.  Project held-out inputs remain untouched.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
from statistics import mean

from predictor_ancestor_transition_predictability_diagnostic import (
    SHADOW_DIMENSION,
    TransitionSample,
    _tree_transitions,
)
from predictor_calibration_inputs import wide_range_random
from predictor_ranking_smoke import _spearman
from predictor_reliability_weighted_coherence_calibration import _selection_utility
from predictor_signed_cell_shift_predictability_diagnostic import (
    SHIFT_STATES,
    SoftmaxProbeModel,
    _fit_probe,
    _predict_probe,
)
from predictor_two_stage_cheap_score_calibration import (
    DEFAULT_INPUT_SEEDS,
    DEFAULT_RANDOM_GRAPHS_PER_FAMILY,
    _graphs,
    _predictor_trace,
)
from summation_graph_predictor import (
    BinaryReductionGraph,
    FP32_MAX_FINITE,
    predict_fp32_tree_error,
    round_nonnegative_fraction_to_fp32,
)


DEFAULT_TRAIN_WIDTH = 256
DEFAULT_BUDGET = 8
DEFAULT_BEAM_WIDTHS = (1, 3, 5)
FP32_MAX_FINITE_BITS = 0x7F7FFFFF


@dataclass(frozen=True)
class BeamState:
    bits: int
    probability: float


@dataclass(frozen=True)
class BeamTree:
    family: str
    values: tuple[Fraction, ...]
    graph: BinaryReductionGraph
    signed_error: Fraction
    target: float
    q_score: float
    shadow_score: float
    root_ulp: Fraction
    exact_sum: Fraction
    selected_order: tuple[int, ...]
    exact_subtree: tuple[Fraction, ...]
    trajectory_delta: dict[int, float]
    transitions: tuple[TransitionSample, ...]


@dataclass(frozen=True)
class PolicyMetrics:
    rho: float | None
    pairwise_accuracy: float | None
    normalized_regret: float | None
    best_tier_hit: float


def _power_of_two(exponent: int) -> Fraction:
    return Fraction(2**exponent) if exponent >= 0 else Fraction(1, 2 ** (-exponent))


def _fp32_bits_to_fraction(bits: int) -> Fraction:
    if not 0 <= bits <= FP32_MAX_FINITE_BITS:
        raise ValueError("bits must encode a finite nonnegative FP32 value")
    exponent_bits = (bits >> 23) & 0xFF
    fraction_bits = bits & ((1 << 23) - 1)
    if exponent_bits == 0:
        return Fraction(fraction_bits) * _power_of_two(-149)
    exponent = exponent_bits - 127
    significand = (1 << 23) + fraction_bits
    return Fraction(significand) * _power_of_two(exponent - 23)


def _shift_bits(bits: int, shift: int) -> int:
    return max(0, min(FP32_MAX_FINITE_BITS, bits + shift))


def _beam_tree(
    values: tuple[Fraction, ...],
    graph: BinaryReductionGraph,
    family: str,
    budget: int,
) -> BeamTree:
    trace = _predictor_trace(values, graph, (budget,))
    oracle = predict_fp32_tree_error(values, graph)
    transitions = _tree_transitions(
        values,
        graph,
        family,
        budget,
        include_root=True,
    )
    return BeamTree(
        family=family,
        values=values,
        graph=graph,
        signed_error=oracle.signed_error,
        target=float(oracle.signed_error / trace.root_ulp) ** 2,
        q_score=trace.budget[budget].q_budget / 12.0,
        shadow_score=trace.trajectory,
        root_ulp=trace.root_ulp,
        exact_sum=trace.exact_subtree[graph.root],
        selected_order=trace.selected_order[: min(budget, len(trace.selected_order))],
        exact_subtree=trace.exact_subtree,
        trajectory_delta=trace.trajectory_delta,
        transitions=tuple(transitions),
    )


def _generate_width(
    width: int,
    seeds: tuple[int, ...],
    graphs_per_family: int,
    budget: int,
) -> list[list[BeamTree]]:
    groups: list[list[BeamTree]] = []
    for input_index, seed in enumerate(seeds):
        values = wide_range_random(width, seed=seed).values
        group = [
            _beam_tree(values, graph, family, budget)
            for family, graph in _graphs(width, input_index, graphs_per_family)
        ]
        groups.append(group)
    return groups


def _shadow_rounded_values(tree: BeamTree) -> dict[int, Fraction]:
    output_error = [0.0 for _ in tree.values]
    rounded: dict[int, Fraction] = {}
    for offset, node in enumerate(tree.graph.nodes):
        index = tree.graph.leaf_count + offset
        history = output_error[node.left] + output_error[node.right]
        output_error.append(history + tree.trajectory_delta[index])
        rounded[index] = round_nonnegative_fraction_to_fp32(
            tree.exact_subtree[index] + Fraction.from_float(output_error[index])
        ).value
    return rounded


def _prune(states: list[BeamState], width: int) -> list[BeamState]:
    combined: dict[int, float] = {}
    for state in states:
        combined[state.bits] = combined.get(state.bits, 0.0) + state.probability
    retained = sorted(combined.items(), key=lambda item: (-item[1], item[0]))[:width]
    total = sum(probability for _, probability in retained)
    if total <= 0.0:
        raise AssertionError("beam probability vanished")
    return [
        BeamState(bits=bits, probability=probability / total)
        for bits, probability in retained
    ]


def _beam_root_states(
    tree: BeamTree,
    model: SoftmaxProbeModel,
    beam_width: int,
) -> list[BeamState]:
    if beam_width <= 0:
        raise ValueError("beam_width must be positive")
    selected = set(tree.selected_order)
    sample_by_node = {sample.node_index: sample for sample in tree.transitions}
    if set(sample_by_node) != selected:
        raise AssertionError("beam requires exactly one transition sample per selected node")
    innovation_probability = {
        index: _predict_probe(
            model,
            [sample_by_node[index]],
            SHADOW_DIMENSION,
        )[0]
        for index in selected
    }
    shadow_rounded = _shadow_rounded_values(tree)
    beams: dict[int, list[BeamState]] = {}

    def child_states(index: int) -> list[BeamState]:
        if index in selected:
            return beams[index]
        if index < tree.graph.leaf_count:
            bits = round_nonnegative_fraction_to_fp32(tree.values[index]).bits
        else:
            bits = round_nonnegative_fraction_to_fp32(shadow_rounded[index]).bits
        return [BeamState(bits=bits, probability=1.0)]

    for index in reversed(tree.selected_order):
        node = tree.graph.nodes[index - tree.graph.leaf_count]
        candidates: list[BeamState] = []
        for left in child_states(node.left):
            for right in child_states(node.right):
                deterministic = round_nonnegative_fraction_to_fp32(
                    _fp32_bits_to_fraction(left.bits)
                    + _fp32_bits_to_fraction(right.bits)
                )
                inherited_probability = left.probability * right.probability
                for state_index, shift in enumerate(SHIFT_STATES):
                    candidates.append(
                        BeamState(
                            bits=_shift_bits(deterministic.bits, shift),
                            probability=(
                                inherited_probability
                                * float(innovation_probability[index][state_index])
                            ),
                        )
                    )
        beams[index] = _prune(candidates, beam_width)
    return beams[tree.graph.root]


def _beam_scores(
    tree: BeamTree,
    model: SoftmaxProbeModel,
    beam_width: int,
) -> tuple[float, float]:
    states = _beam_root_states(tree, model, beam_width)
    errors = [
        float((_fp32_bits_to_fraction(state.bits) - tree.exact_sum) / tree.root_ulp)
        for state in states
    ]
    expected_square = sum(
        state.probability * error * error
        for state, error in zip(states, errors, strict=True)
    )
    mean_error = sum(
        state.probability * error
        for state, error in zip(states, errors, strict=True)
    )
    return expected_square, mean_error * mean_error


def _policy_metrics(scores: list[float], target: list[float]) -> PolicyMetrics:
    utility = _selection_utility(scores, target)
    return PolicyMetrics(
        rho=_spearman(scores, target),
        pairwise_accuracy=utility.pairwise_accuracy,
        normalized_regret=utility.normalized_regret,
        best_tier_hit=utility.best_tier_hit,
    )


def _evaluate_group(
    trees: list[BeamTree],
    model: SoftmaxProbeModel,
    beam_widths: tuple[int, ...],
) -> dict[str, PolicyMetrics]:
    target = [tree.target for tree in trees]
    policies: dict[str, list[float]] = {
        "qK": [tree.q_score for tree in trees],
        "shadow": [tree.shadow_score for tree in trees],
    }
    for width in beam_widths:
        pairs = [_beam_scores(tree, model, width) for tree in trees]
        policies[f"beamE{width}"] = [pair[0] for pair in pairs]
        policies[f"beamMean{width}"] = [pair[1] for pair in pairs]
    return {
        policy: _policy_metrics(scores, target)
        for policy, scores in policies.items()
    }


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.3f}"


def _print_fold(
    seed: int,
    training: list[BeamTree],
    evaluation: list[BeamTree],
    beam_widths: tuple[int, ...],
) -> dict[str, PolicyMetrics]:
    training_samples = [sample for tree in training for sample in tree.transitions]
    model = _fit_probe(
        training_samples,
        SHADOW_DIMENSION,
        label="innovation_shift",
    )
    results = _evaluate_group(evaluation, model, beam_widths)
    print(
        f"EVAL seed={seed} train_trees={len(training)} eval_trees={len(evaluation)} "
        f"target_unique={len(set(tree.target for tree in evaluation))}"
    )
    for policy, metric in results.items():
        print(
            f"  {policy:<9} rho={_fmt(metric.rho)} "
            f"pairAcc={_fmt(metric.pairwise_accuracy)} "
            f"regret={_fmt(metric.normalized_regret)} "
            f"bestHit={metric.best_tier_hit:.0f}"
        )
    for family in sorted({tree.family for tree in evaluation}):
        family_results = _evaluate_group(
            [tree for tree in evaluation if tree.family == family],
            model,
            beam_widths,
        )
        compact = " ".join(
            f"{policy}={_fmt(family_results[policy].rho)}"
            for policy in (
                "qK",
                "shadow",
                *(f"beamE{width}" for width in beam_widths),
            )
        )
        print(f"    family={family:<10} rho {compact}")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-width", type=int, default=DEFAULT_TRAIN_WIDTH)
    parser.add_argument("--eval-width", type=int)
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument(
        "--beam-widths",
        type=int,
        nargs="+",
        default=list(DEFAULT_BEAM_WIDTHS),
    )
    parser.add_argument(
        "--input-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_INPUT_SEEDS),
        help="training seeds; also evaluation seeds unless overridden",
    )
    parser.add_argument(
        "--eval-input-seeds",
        type=int,
        nargs="+",
        help="optional disjoint evaluation seeds for fixed-model transfer",
    )
    parser.add_argument(
        "--random-graphs-per-family",
        type=int,
        default=DEFAULT_RANDOM_GRAPHS_PER_FAMILY,
    )
    args = parser.parse_args()
    eval_width = args.eval_width or args.train_width
    if args.train_width < 2 or eval_width < 2:
        parser.error("widths must be at least 2")
    if args.budget <= 1:
        parser.error("--budget must exceed 1")
    if not args.beam_widths or any(width <= 0 for width in args.beam_widths):
        parser.error("--beam-widths must contain positive integers")
    if len(args.input_seeds) < 2:
        parser.error("at least two training seeds are required")
    if args.random_graphs_per_family < 2:
        parser.error("--random-graphs-per-family must be at least 2")
    train_seeds = tuple(args.input_seeds)
    eval_seeds = tuple(args.eval_input_seeds or args.input_seeds)
    beam_widths = tuple(sorted(set(args.beam_widths)))

    print("Selected-root-band joint cell-beam score calibration")
    print("CALIBRATION ONLY — evaluated scores never read their oracle labels")
    print("selected-child states propagate; only frontier innovation is learned")
    print(
        f"train_width={args.train_width} eval_width={eval_width} "
        f"budget={args.budget} beam_widths={','.join(map(str, beam_widths))} "
        f"train_seeds={','.join(map(str, train_seeds))} "
        f"eval_seeds={','.join(map(str, eval_seeds))} "
        f"random_graphs_per_family={args.random_graphs_per_family}"
    )
    print()

    train_groups = _generate_width(
        args.train_width,
        train_seeds,
        args.random_graphs_per_family,
        args.budget,
    )
    same_groups = eval_width == args.train_width and eval_seeds == train_seeds
    eval_groups = (
        train_groups
        if same_groups
        else _generate_width(
            eval_width,
            eval_seeds,
            args.random_graphs_per_family,
            args.budget,
        )
    )

    pooled: dict[str, list[PolicyMetrics]] = {}
    if same_groups:
        for held_out_index, (seed, evaluation) in enumerate(
            zip(eval_seeds, eval_groups, strict=True)
        ):
            training = [
                tree
                for group_index, group in enumerate(train_groups)
                if group_index != held_out_index
                for tree in group
            ]
            fold = _print_fold(seed, training, evaluation, beam_widths)
            for policy, metric in fold.items():
                pooled.setdefault(policy, []).append(metric)
            print()
    else:
        training = [tree for group in train_groups for tree in group]
        for seed, evaluation in zip(eval_seeds, eval_groups, strict=True):
            fold = _print_fold(seed, training, evaluation, beam_widths)
            for policy, metric in fold.items():
                pooled.setdefault(policy, []).append(metric)
            print()

    print("FOLD SUMMARY mean/min rho and pairAcc")
    for policy, metrics in pooled.items():
        rhos = [metric.rho for metric in metrics if metric.rho is not None]
        pairs = [
            metric.pairwise_accuracy
            for metric in metrics
            if metric.pairwise_accuracy is not None
        ]
        regrets = [
            metric.normalized_regret
            for metric in metrics
            if metric.normalized_regret is not None
        ]
        print(
            f"  {policy:<9} rho_mean/min={mean(rhos):+.3f}/{min(rhos):+.3f} "
            f"pair_mean/min={mean(pairs):.3f}/{min(pairs):.3f} "
            f"regret_mean={mean(regrets):.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
