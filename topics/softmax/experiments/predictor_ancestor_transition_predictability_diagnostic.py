"""Probe cheap predictability of phase transitions on top-energy ancestor edges.

The current cheap-score skeleton is intentionally left unchanged::

    score = Q_K / 12 + sparse ancestor-coherence correction.

This calibration-only diagnostic asks whether predictor-side state contains transferable signal for
the missing correction.  Every selected root-band node is recorded once and connected to its nearest
selected ancestor as propagation context.  The transition labels belong to the selected node:

* ``crossing``: actual FP32 history moves the ancestor to a different RN-even output cell than the
  history-free exact-subtree rounding;
* ``sign_flip``: the actual local residual sign differs from the history-free residual sign;
* ``wrong_cell``: the deterministic binary64 shadow chooses a different RN-even output cell from
  the actual FP32 tree.

Three nested cheap feature sets are compared: structural edge context, parent exact phase, and the
deterministic shadow history/innovation state.  A small regularized logistic probe is trained with
leave-one-input-seed-out grouping when train and evaluation widths match.  Cross-width mode fits on
all declared train-width seeds and transfers the probe unchanged.  Positive held-out AUC and
log-loss gain show exploitable feature information; they do not prove an information-theoretic
upper bound or promote a new score.

Exact FP32 execution supplies labels only.  It never enters a held-out sample's feature vector.
Project held-out inputs remain untouched.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from fractions import Fraction
from statistics import mean

import numpy as np

from predictor_calibration_inputs import wide_range_random
from predictor_two_stage_cheap_score_calibration import (
    DEFAULT_INPUT_SEEDS,
    DEFAULT_RANDOM_GRAPHS_PER_FAMILY,
    _graphs,
    _predictor_trace,
    _sign,
)
from predictor_ulp_energy_convergence_diagnostic import _parent_map
from summation_graph_predictor import (
    BinaryReductionGraph,
    predict_fp32_tree_error,
    round_nonnegative_fraction_to_fp32,
)


DEFAULT_TRAIN_WIDTH = 256
DEFAULT_BUDGET = 32
L2_PENALTY = 1.0e-2
MAX_ITERATIONS = 400

STRUCTURE_DIMENSION = 6
PHASE_DIMENSION = 10
SHADOW_DIMENSION = 19
FEATURE_SETS = {
    "structure": STRUCTURE_DIMENSION,
    "phase": PHASE_DIMENSION,
    "shadow": SHADOW_DIMENSION,
}
LABELS = ("crossing", "sign_flip", "wrong_cell")


@dataclass(frozen=True)
class TransitionSample:
    family: str
    node_index: int
    ancestor_index: int
    gap: int
    features: tuple[float, ...]
    crossing: int
    sign_flip: int
    wrong_cell: int
    cell_shift: int
    innovation_shift: int
    predicted_crossing: int
    predicted_sign_flip: int

    def label(self, name: str) -> int:
        if name not in LABELS:
            raise ValueError(f"unknown label: {name}")
        return int(getattr(self, name))

    def heuristic(self, name: str) -> int:
        if name == "crossing":
            return self.predicted_crossing
        if name == "sign_flip":
            return self.predicted_sign_flip
        if name == "wrong_cell":
            return int(self.predicted_crossing or self.predicted_sign_flip)
        raise ValueError(f"unknown label: {name}")


@dataclass(frozen=True)
class ProbeModel:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    weights: np.ndarray
    train_prevalence: float


@dataclass(frozen=True)
class ProbeMetrics:
    count: int
    prevalence: float
    auc: float | None
    balanced_accuracy: float | None
    log_gain_bits: float
    brier_skill: float


def _fractional_ulp_coordinate(value: Fraction, ulp: Fraction) -> float:
    coordinate = value / ulp
    floor = coordinate.numerator // coordinate.denominator
    return float(coordinate - floor)


def _clamp(value: float, limit: float = 8.0) -> float:
    return max(-limit, min(limit, value))


def _nearest_selected_ancestor(
    node: int,
    selected: set[int],
    parent: list[int | None],
) -> tuple[int, int, int] | None:
    """Return ancestor, edge gap, and the ancestor's immediate child containing node."""
    gap = 0
    branch = node
    current = parent[node]
    while current is not None:
        gap += 1
        if current in selected:
            return current, gap, branch
        branch = current
        current = parent[current]
    return None


def _tree_transitions(
    values: tuple[Fraction, ...],
    graph: BinaryReductionGraph,
    family: str,
    budget: int,
    *,
    include_root: bool = False,
) -> list[TransitionSample]:
    trace = _predictor_trace(values, graph, (budget,))
    oracle = predict_fp32_tree_error(values, graph)
    selected_order = trace.selected_order[: min(budget, len(trace.selected_order))]
    selected = set(selected_order)
    selected_rank = {node: rank for rank, node in enumerate(selected_order)}
    parent = _parent_map(graph)

    size = [1 for _ in values]
    depth = [0 for _ in values]
    shadow_history: dict[int, float] = {}
    shadow_output_error = [0.0 for _ in values]
    actual_delta: dict[int, Fraction] = {}
    actual_rounded: dict[int, Fraction] = {}
    for prediction in oracle.node_predictions:
        actual_delta[prediction.node_index] = prediction.local_rounding_error
        actual_rounded[prediction.node_index] = prediction.rounded_sum

    for offset, node in enumerate(graph.nodes):
        index = graph.leaf_count + offset
        size.append(size[node.left] + size[node.right])
        depth.append(max(depth[node.left], depth[node.right]) + 1)
        history = shadow_output_error[node.left] + shadow_output_error[node.right]
        shadow_history[index] = history
        shadow_output_error.append(history + trace.trajectory_delta[index])

    maximum_depth = max(depth) or 1
    rows: list[TransitionSample] = []
    for descendant in selected_order:
        relation = _nearest_selected_ancestor(descendant, selected, parent)
        if relation is None:
            if not include_root:
                continue
            ancestor, gap = descendant, 0
        else:
            ancestor, gap, _branch = relation
        target_node = graph.nodes[descendant - graph.leaf_count]
        ancestor_ulp = float(trace.node_ulp[ancestor])
        descendant_ulp = float(trace.node_ulp[descendant])
        phase = _fractional_ulp_coordinate(
            trace.exact_subtree[descendant],
            trace.node_ulp[descendant],
        )
        parent_history_ulp = shadow_history[descendant] / descendant_ulp
        left_history_ulp = shadow_output_error[target_node.left] / descendant_ulp
        right_history_ulp = shadow_output_error[target_node.right] / descendant_ulp
        propagated_output_ulp = shadow_output_error[descendant] / ancestor_ulp
        descendant_history_ulp = shadow_history[descendant] / descendant_ulp
        descendant_innovation_ulp = (
            trace.trajectory_delta[descendant] / descendant_ulp
        )
        inherited_fraction = abs(descendant_history_ulp) / (
            abs(descendant_history_ulp) + abs(descendant_innovation_ulp) + 1.0e-12
        )

        rank_denominator = max(1, len(selected_order) - 1)
        ancestor_size = size[ancestor]
        structural = (
            selected_rank[descendant] / rank_denominator,
            selected_rank[ancestor] / rank_denominator,
            gap / maximum_depth,
            math.log2(descendant_ulp / ancestor_ulp),
            size[descendant] / ancestor_size,
            abs(size[target_node.left] - size[target_node.right]) / size[descendant],
        )
        phase_features = (
            math.sin(2.0 * math.pi * phase),
            math.cos(2.0 * math.pi * phase),
            abs(phase - 0.5),
            float(_sign(trace.delta0[descendant])),
        )
        shadow_features = (
            _clamp(parent_history_ulp),
            min(8.0, abs(parent_history_ulp)),
            _clamp(left_history_ulp),
            _clamp(right_history_ulp),
            _clamp(left_history_ulp - right_history_ulp),
            _clamp(propagated_output_ulp),
            _clamp(descendant_innovation_ulp),
            inherited_fraction,
            float(
                trace.predicted_cross[descendant]
                or trace.predicted_phase[descendant]
            ),
        )

        history_free_output = (
            trace.exact_subtree[descendant] + trace.delta0[descendant]
        )
        rounded_actual = actual_rounded[descendant]
        actual_rounded_bits = round_nonnegative_fraction_to_fp32(rounded_actual).bits
        shadow_rounded = round_nonnegative_fraction_to_fp32(
            trace.exact_subtree[descendant]
            + Fraction.from_float(shadow_output_error[descendant])
        )
        deterministic_children = []
        for child in (target_node.left, target_node.right):
            if child in selected:
                deterministic_children.append(actual_rounded[child])
            elif child < graph.leaf_count:
                deterministic_children.append(values[child])
            else:
                deterministic_children.append(
                    round_nonnegative_fraction_to_fp32(
                        trace.exact_subtree[child]
                        + Fraction.from_float(shadow_output_error[child])
                    ).value
                )
        deterministic_parent = round_nonnegative_fraction_to_fp32(
            deterministic_children[0] + deterministic_children[1]
        )
        rows.append(
            TransitionSample(
                family=family,
                node_index=descendant,
                ancestor_index=ancestor,
                gap=gap,
                features=structural + phase_features + shadow_features,
                crossing=int(rounded_actual != history_free_output),
                sign_flip=int(
                    _sign(actual_delta[descendant])
                    != _sign(trace.delta0[descendant])
                ),
                # Quantize the already-computed shadow output only for cell identity.  This exact
                # rational operation is not propagated into an ancestor, so it cannot replay the
                # candidate FP32 trajectory.  It also avoids cancellation from float(S + error).
                wrong_cell=int(rounded_actual != shadow_rounded.value),
                cell_shift=actual_rounded_bits - shadow_rounded.bits,
                innovation_shift=actual_rounded_bits - deterministic_parent.bits,
                predicted_crossing=int(trace.predicted_cross[descendant]),
                predicted_sign_flip=int(trace.predicted_phase[descendant]),
            )
        )
    return rows


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _fit_probe(
    samples: list[TransitionSample],
    label: str,
    dimension: int,
) -> ProbeModel:
    if not samples:
        raise ValueError("probe training requires samples")
    matrix = np.asarray([sample.features[:dimension] for sample in samples], dtype=float)
    target = np.asarray([sample.label(label) for sample in samples], dtype=float)
    feature_mean = matrix.mean(axis=0)
    feature_scale = matrix.std(axis=0)
    feature_scale[feature_scale < 1.0e-12] = 1.0
    standardized = (matrix - feature_mean) / feature_scale
    design = np.column_stack((np.ones(len(samples)), standardized))

    prevalence = float(target.mean())
    clipped_prevalence = min(1.0 - 1.0e-6, max(1.0e-6, prevalence))
    weights = np.zeros(design.shape[1], dtype=float)
    weights[0] = math.log(clipped_prevalence / (1.0 - clipped_prevalence))
    for _ in range(MAX_ITERATIONS):
        probability = _sigmoid(design @ weights)
        gradient = design.T @ (probability - target) / len(samples)
        gradient[1:] += L2_PENALTY * weights[1:]
        curvature = probability * (1.0 - probability)
        hessian = (design.T * curvature) @ design / len(samples)
        hessian[1:, 1:] += L2_PENALTY * np.eye(design.shape[1] - 1)
        hessian += 1.0e-8 * np.eye(design.shape[1])
        step = np.linalg.solve(hessian, gradient)
        weights -= step
        if float(np.max(np.abs(step))) < 1.0e-8:
            break
    return ProbeModel(feature_mean, feature_scale, weights, prevalence)


def _predict_probe(
    model: ProbeModel,
    samples: list[TransitionSample],
    dimension: int,
) -> np.ndarray:
    matrix = np.asarray([sample.features[:dimension] for sample in samples], dtype=float)
    standardized = (matrix - model.feature_mean) / model.feature_scale
    design = np.column_stack((np.ones(len(samples)), standardized))
    return _sigmoid(design @ model.weights)


def _auc(target: np.ndarray, score: np.ndarray) -> float | None:
    positives = int(target.sum())
    negatives = len(target) - positives
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(score), dtype=float)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and score[order[stop]] == score[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    positive_rank_sum = float(ranks[target == 1.0].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


def _balanced_accuracy(target: np.ndarray, prediction: np.ndarray) -> float | None:
    positive = target == 1.0
    negative = ~positive
    if not positive.any() or not negative.any():
        return None
    sensitivity = float(prediction[positive].mean())
    specificity = float((1.0 - prediction[negative]).mean())
    return (sensitivity + specificity) / 2.0


def _metrics(
    target: np.ndarray,
    probability: np.ndarray,
    baseline_probability: float,
) -> ProbeMetrics:
    epsilon = 1.0e-12
    probability = np.clip(probability, epsilon, 1.0 - epsilon)
    baseline_probability = min(1.0 - epsilon, max(epsilon, baseline_probability))
    baseline = np.full(len(target), baseline_probability)
    model_loss = -np.mean(
        target * np.log(probability) + (1.0 - target) * np.log(1.0 - probability)
    )
    baseline_loss = -np.mean(
        target * np.log(baseline) + (1.0 - target) * np.log(1.0 - baseline)
    )
    model_brier = float(np.mean((probability - target) ** 2))
    baseline_brier = float(np.mean((baseline - target) ** 2))
    prediction = (probability >= baseline_probability).astype(float)
    return ProbeMetrics(
        count=len(target),
        prevalence=float(target.mean()),
        auc=_auc(target, probability),
        balanced_accuracy=_balanced_accuracy(target, prediction),
        log_gain_bits=float((baseline_loss - model_loss) / math.log(2.0)),
        brier_skill=(1.0 - model_brier / baseline_brier if baseline_brier else 0.0),
    )


def _heuristic_balanced_accuracy(
    samples: list[TransitionSample],
    label: str,
) -> float | None:
    target = np.asarray([sample.label(label) for sample in samples], dtype=float)
    prediction = np.asarray([sample.heuristic(label) for sample in samples], dtype=float)
    return _balanced_accuracy(target, prediction)


def _generate_width(
    width: int,
    seeds: tuple[int, ...],
    graphs_per_family: int,
    budget: int,
    *,
    include_root: bool = False,
) -> list[list[TransitionSample]]:
    groups: list[list[TransitionSample]] = []
    for input_index, seed in enumerate(seeds):
        values = wide_range_random(width, seed=seed).values
        group: list[TransitionSample] = []
        for family, graph in _graphs(width, input_index, graphs_per_family):
            group.extend(
                _tree_transitions(
                    values,
                    graph,
                    family,
                    budget,
                    include_root=include_root,
                )
            )
        groups.append(group)
    return groups


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.3f}"


def _evaluate_fold(
    training: list[TransitionSample],
    evaluation: list[TransitionSample],
) -> dict[str, ProbeMetrics]:
    results: dict[str, ProbeMetrics] = {}
    for label in LABELS:
        for feature_set, dimension in FEATURE_SETS.items():
            model = _fit_probe(training, label, dimension)
            probability = _predict_probe(model, evaluation, dimension)
            target = np.asarray(
                [sample.label(label) for sample in evaluation],
                dtype=float,
            )
            results[f"{label}_{feature_set}"] = _metrics(
                target,
                probability,
                model.train_prevalence,
            )
    return results


def _print_fold(
    seed: int,
    training: list[TransitionSample],
    evaluation: list[TransitionSample],
) -> dict[str, ProbeMetrics]:
    results = _evaluate_fold(training, evaluation)
    print(f"EVAL seed={seed} train_edges={len(training)} eval_edges={len(evaluation)}")
    for label in LABELS:
        prevalence = results[f"{label}_structure"].prevalence
        heuristic = _heuristic_balanced_accuracy(evaluation, label)
        print(
            f"  {label:<10} prevalence={prevalence:.3f} "
            f"heuristic_balAcc={_fmt(heuristic)}"
        )
        for feature_set in FEATURE_SETS:
            metric = results[f"{label}_{feature_set}"]
            print(
                f"    {feature_set:<9} auc={_fmt(metric.auc)} "
                f"balAcc={_fmt(metric.balanced_accuracy)} "
                f"logGain={metric.log_gain_bits:+.4f} bits/edge "
                f"brierSkill={metric.brier_skill:+.3f}"
            )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-width", type=int, default=DEFAULT_TRAIN_WIDTH)
    parser.add_argument("--eval-width", type=int)
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
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
    if len(args.input_seeds) < 2:
        parser.error("at least two input seeds are required")
    if args.eval_input_seeds is not None and not args.eval_input_seeds:
        parser.error("--eval-input-seeds must not be empty")
    if args.random_graphs_per_family < 2:
        parser.error("--random-graphs-per-family must be at least 2")
    train_seeds = tuple(args.input_seeds)
    eval_seeds = tuple(args.eval_input_seeds or args.input_seeds)

    print("Top-energy ancestor-transition predictability diagnostic")
    print("CALIBRATION ONLY — exact FP32 execution supplies labels only")
    print("positive held-out logGain/AUC means cheap transition information exists")
    print(
        f"train_width={args.train_width} eval_width={eval_width} "
        f"budget={args.budget} train_seeds={','.join(map(str, train_seeds))} "
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

    pooled: dict[str, list[ProbeMetrics]] = {}
    if same_groups:
        for held_out_index, (seed, evaluation) in enumerate(
            zip(eval_seeds, eval_groups, strict=True)
        ):
            training = [
                sample
                for group_index, group in enumerate(train_groups)
                if group_index != held_out_index
                for sample in group
            ]
            fold = _print_fold(seed, training, evaluation)
            for key, metric in fold.items():
                pooled.setdefault(key, []).append(metric)
            print()
    else:
        training = [sample for group in train_groups for sample in group]
        for seed, evaluation in zip(eval_seeds, eval_groups, strict=True):
            fold = _print_fold(seed, training, evaluation)
            for key, metric in fold.items():
                pooled.setdefault(key, []).append(metric)
            print()

    print("FOLD SUMMARY mean/min AUC and logGain")
    for label in LABELS:
        print(f"  {label}")
        for feature_set in FEATURE_SETS:
            metrics = pooled[f"{label}_{feature_set}"]
            aucs = [metric.auc for metric in metrics if metric.auc is not None]
            gains = [metric.log_gain_bits for metric in metrics]
            print(
                f"    {feature_set:<9} "
                f"auc_mean/min={mean(aucs):+.3f}/{min(aucs):+.3f} "
                f"logGain_mean/min={mean(gains):+.4f}/{min(gains):+.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
