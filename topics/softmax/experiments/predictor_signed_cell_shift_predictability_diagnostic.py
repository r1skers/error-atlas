"""Calibrate directional FP32 cell-shift predictability on selected ancestor edges.

The binary ancestor-transition diagnostic establishes that cheap shadow state predicts whether a
selected node crosses a boundary or lands in the wrong FP32 cell.  Coherence needs direction, not
only event probability, so this follow-up predicts the signed difference between the actual and
shadow output bit patterns.  Nonnegative finite FP32 bit patterns are monotonically ordered; their
integer difference therefore counts representable output cells exactly, including binade edges.

Observed calibration shifts are concentrated near zero.  The probe uses five states
``{-2, -1, 0, +1, +2}``, clipping only the rare tails.  A regularized multinomial logistic model is
fit with input-seed grouping and the same nested structure/phase/shadow feature ablation as the
binary diagnostic.  Same-width evaluation is leave-one-seed-out.  Cross-width mode can use disjoint
evaluation seeds and transfers a fixed model unchanged.

This is a feature-information probe, not an ancestor beam or a promoted score.  Exact FP32 execution
supplies calibration labels only; project held-out inputs remain untouched.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from statistics import mean

import numpy as np

from predictor_ancestor_transition_predictability_diagnostic import (
    DEFAULT_BUDGET,
    DEFAULT_TRAIN_WIDTH,
    FEATURE_SETS,
    TransitionSample,
    _generate_width,
)
from predictor_ranking_smoke import _spearman
from predictor_two_stage_cheap_score_calibration import (
    DEFAULT_INPUT_SEEDS,
    DEFAULT_RANDOM_GRAPHS_PER_FAMILY,
)


SHIFT_STATES = (-2, -1, 0, 1, 2)
STATE_TO_INDEX = {state: index for index, state in enumerate(SHIFT_STATES)}
L2_PENALTY = 1.0e-2
MAX_ITERATIONS = 600
LEARNING_RATE = 5.0e-2


@dataclass(frozen=True)
class SoftmaxProbeModel:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    weights: np.ndarray
    train_prevalence: np.ndarray


@dataclass(frozen=True)
class ShiftMetrics:
    count: int
    log_gain_bits: float
    accuracy: float
    majority_accuracy: float
    macro_recall: float
    direction_accuracy: float | None
    expected_shift_rho: float | None
    expected_shift_mae: float


def _clip_shift(shift: int) -> int:
    return max(SHIFT_STATES[0], min(SHIFT_STATES[-1], shift))


def _target_indices(
    samples: list[TransitionSample],
    label: str = "cell_shift",
) -> np.ndarray:
    if label not in ("cell_shift", "innovation_shift"):
        raise ValueError("label must be cell_shift or innovation_shift")
    return np.asarray(
        [STATE_TO_INDEX[_clip_shift(getattr(sample, label))] for sample in samples],
        dtype=int,
    )


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponential = np.exp(np.clip(shifted, -80.0, 0.0))
    return exponential / exponential.sum(axis=1, keepdims=True)


def _fit_probe(
    samples: list[TransitionSample],
    dimension: int,
    label: str = "cell_shift",
) -> SoftmaxProbeModel:
    if not samples:
        raise ValueError("probe training requires samples")
    matrix = np.asarray([sample.features[:dimension] for sample in samples], dtype=float)
    target = _target_indices(samples, label)
    feature_mean = matrix.mean(axis=0)
    feature_scale = matrix.std(axis=0)
    feature_scale[feature_scale < 1.0e-12] = 1.0
    standardized = (matrix - feature_mean) / feature_scale
    design = np.column_stack((np.ones(len(samples)), standardized))
    one_hot = np.eye(len(SHIFT_STATES))[target]

    counts = np.bincount(target, minlength=len(SHIFT_STATES)).astype(float)
    prevalence = (counts + 1.0) / (len(samples) + len(SHIFT_STATES))
    weights = np.zeros((design.shape[1], len(SHIFT_STATES)), dtype=float)
    weights[0] = np.log(prevalence)
    first_moment = np.zeros_like(weights)
    second_moment = np.zeros_like(weights)
    for iteration in range(1, MAX_ITERATIONS + 1):
        probability = _softmax(design @ weights)
        gradient = design.T @ (probability - one_hot) / len(samples)
        gradient[1:] += L2_PENALTY * weights[1:]
        first_moment = 0.9 * first_moment + 0.1 * gradient
        second_moment = 0.999 * second_moment + 0.001 * gradient * gradient
        first_hat = first_moment / (1.0 - 0.9**iteration)
        second_hat = second_moment / (1.0 - 0.999**iteration)
        step = LEARNING_RATE * first_hat / (np.sqrt(second_hat) + 1.0e-8)
        weights -= step
        # Softmax is invariant to a common logit offset.  Centering prevents numerical drift.
        weights -= weights.mean(axis=1, keepdims=True)
        if float(np.max(np.abs(gradient))) < 1.0e-7:
            break
    return SoftmaxProbeModel(feature_mean, feature_scale, weights, prevalence)


def _predict_probe(
    model: SoftmaxProbeModel,
    samples: list[TransitionSample],
    dimension: int,
) -> np.ndarray:
    matrix = np.asarray([sample.features[:dimension] for sample in samples], dtype=float)
    standardized = (matrix - model.feature_mean) / model.feature_scale
    design = np.column_stack((np.ones(len(samples)), standardized))
    return _softmax(design @ model.weights)


def _macro_recall(target: np.ndarray, prediction: np.ndarray) -> float:
    recalls = []
    for state_index in range(len(SHIFT_STATES)):
        selected = target == state_index
        if selected.any():
            recalls.append(float((prediction[selected] == state_index).mean()))
    return mean(recalls)


def _metrics(
    samples: list[TransitionSample],
    probability: np.ndarray,
    baseline_probability: np.ndarray,
    label: str = "cell_shift",
) -> ShiftMetrics:
    target = _target_indices(samples, label)
    epsilon = 1.0e-12
    probability = np.clip(probability, epsilon, 1.0)
    probability /= probability.sum(axis=1, keepdims=True)
    baseline_probability = np.clip(baseline_probability, epsilon, 1.0)
    baseline_probability /= baseline_probability.sum()
    model_loss = -float(np.log(probability[np.arange(len(target)), target]).mean())
    baseline_loss = -float(np.log(baseline_probability[target]).mean())

    prediction = probability.argmax(axis=1)
    majority = int(baseline_probability.argmax())
    state_values = np.asarray(SHIFT_STATES, dtype=float)
    expected = probability @ state_values
    actual = state_values[target]
    nonzero = actual != 0.0
    direction_accuracy = (
        float((np.sign(expected[nonzero]) == np.sign(actual[nonzero])).mean())
        if nonzero.any()
        else None
    )
    return ShiftMetrics(
        count=len(samples),
        log_gain_bits=(baseline_loss - model_loss) / math.log(2.0),
        accuracy=float((prediction == target).mean()),
        majority_accuracy=float((target == majority).mean()),
        macro_recall=_macro_recall(target, prediction),
        direction_accuracy=direction_accuracy,
        expected_shift_rho=_spearman(expected.tolist(), actual.tolist()),
        expected_shift_mae=float(np.abs(expected - actual).mean()),
    )


def _evaluate_fold(
    training: list[TransitionSample],
    evaluation: list[TransitionSample],
) -> dict[str, ShiftMetrics]:
    results: dict[str, ShiftMetrics] = {}
    for feature_set, dimension in FEATURE_SETS.items():
        model = _fit_probe(training, dimension)
        probability = _predict_probe(model, evaluation, dimension)
        results[feature_set] = _metrics(
            evaluation,
            probability,
            model.train_prevalence.copy(),
        )
    return results


def _distribution(samples: list[TransitionSample]) -> str:
    counts = {state: 0 for state in SHIFT_STATES}
    for sample in samples:
        counts[_clip_shift(sample.cell_shift)] += 1
    return " ".join(
        f"{state:+d}:{counts[state] / len(samples):.3f}" for state in SHIFT_STATES
    )


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.3f}"


def _print_fold(
    seed: int,
    training: list[TransitionSample],
    evaluation: list[TransitionSample],
) -> dict[str, ShiftMetrics]:
    results = _evaluate_fold(training, evaluation)
    print(
        f"EVAL seed={seed} train_edges={len(training)} eval_edges={len(evaluation)} "
        f"shift_distribution={_distribution(evaluation)}"
    )
    for feature_set, metric in results.items():
        print(
            f"  {feature_set:<9} logGain={metric.log_gain_bits:+.4f} bits/edge "
            f"acc/base={metric.accuracy:.3f}/{metric.majority_accuracy:.3f} "
            f"macroRecall={metric.macro_recall:.3f} "
            f"directionAcc={_fmt(metric.direction_accuracy)} "
            f"Eshift_rho={_fmt(metric.expected_shift_rho)} "
            f"Eshift_MAE={metric.expected_shift_mae:.3f}"
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
        parser.error("at least two training seeds are required")
    if args.eval_input_seeds is not None and not args.eval_input_seeds:
        parser.error("--eval-input-seeds must not be empty")
    if args.random_graphs_per_family < 2:
        parser.error("--random-graphs-per-family must be at least 2")
    train_seeds = tuple(args.input_seeds)
    eval_seeds = tuple(args.eval_input_seeds or args.input_seeds)

    print("Signed selected-node FP32 cell-shift predictability diagnostic")
    print("CALIBRATION ONLY — exact FP32 execution supplies directional labels only")
    print("five states: -2,-1,0,+1,+2; rare tails are clipped")
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

    pooled: dict[str, list[ShiftMetrics]] = {}
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

    print("FOLD SUMMARY mean/min")
    for feature_set in FEATURE_SETS:
        metrics = pooled[feature_set]
        gains = [metric.log_gain_bits for metric in metrics]
        directions = [
            metric.direction_accuracy
            for metric in metrics
            if metric.direction_accuracy is not None
        ]
        rhos = [
            metric.expected_shift_rho
            for metric in metrics
            if metric.expected_shift_rho is not None
        ]
        print(
            f"  {feature_set:<9} "
            f"logGain_mean/min={mean(gains):+.4f}/{min(gains):+.4f} "
            f"directionAcc_mean/min={mean(directions):.3f}/{min(directions):.3f} "
            f"Eshift_rho_mean/min={mean(rhos):+.3f}/{min(rhos):+.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
