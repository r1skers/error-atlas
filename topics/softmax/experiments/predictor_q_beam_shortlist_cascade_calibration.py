"""Calibrate Q_K shortlist followed by joint cell-beam reranking.

The joint ancestor cell beam improves ranking but its current full-tree shadow implementation is too
expensive for every candidate.  This experiment applies Q_K/12 to all trees, keeps only the M lowest
Q candidates, and evaluates B=1/3 beam scores inside that shortlist.

For each input group it reports whether the shortlist contains any oracle-best-tier tree, the exact
without-replacement random coverage baseline, within-shortlist pairwise/rank quality, selected-tree
best-tier hit, and normalized regret.  Same-width fitting is leave-one-input-seed-out; cross-width
mode transfers one innovation model and may use disjoint evaluation seeds.

This is calibration-only.  Evaluated beam scores consume predictor features only.  Project held-out
inputs remain untouched.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from statistics import mean

from predictor_ancestor_cell_beam_score_calibration import (
    BeamTree,
    _beam_scores,
    _generate_width,
)
from predictor_ancestor_transition_predictability_diagnostic import SHADOW_DIMENSION
from predictor_ranking_smoke import _spearman
from predictor_reliability_weighted_coherence_calibration import _selection_utility
from predictor_signed_cell_shift_predictability_diagnostic import _fit_probe
from predictor_two_stage_cheap_score_calibration import (
    DEFAULT_INPUT_SEEDS,
    DEFAULT_RANDOM_GRAPHS_PER_FAMILY,
)


DEFAULT_TRAIN_WIDTH = 256
DEFAULT_BUDGET = 8
DEFAULT_SHORTLISTS = (4, 8, 16, 32)
DEFAULT_BEAM_WIDTHS = (1, 3)


@dataclass(frozen=True)
class SelectionMetrics:
    rho: float | None
    pairwise_accuracy: float | None
    best_tier_hit: float
    normalized_regret: float | None


@dataclass(frozen=True)
class CascadeMetrics:
    shortlist_size: int
    best_tier_coverage: float
    random_coverage: float
    selection: dict[int, SelectionMetrics]


def _shortlist_indices(q_scores: list[float], size: int) -> tuple[int, ...]:
    if not q_scores:
        raise ValueError("q_scores must be nonempty")
    if size <= 0:
        raise ValueError("shortlist size must be positive")
    return tuple(
        sorted(range(len(q_scores)), key=lambda index: (q_scores[index], index))[
            : min(size, len(q_scores))
        ]
    )


def _random_best_coverage(tree_count: int, best_count: int, size: int) -> float:
    if not 0 < best_count <= tree_count:
        raise ValueError("best_count must be in [1, tree_count]")
    selected = min(size, tree_count)
    misses = tree_count - best_count
    if selected > misses:
        return 1.0
    return 1.0 - math.comb(misses, selected) / math.comb(tree_count, selected)


def _selection_metrics(
    scores: list[float],
    target: list[float],
) -> SelectionMetrics:
    utility = _selection_utility(scores, target)
    return SelectionMetrics(
        rho=_spearman(scores, target),
        pairwise_accuracy=utility.pairwise_accuracy,
        best_tier_hit=utility.best_tier_hit,
        normalized_regret=utility.normalized_regret,
    )


def _evaluate_group(
    trees: list[BeamTree],
    model,
    shortlists: tuple[int, ...],
    beam_widths: tuple[int, ...],
) -> tuple[SelectionMetrics, dict[int, CascadeMetrics]]:
    target = [tree.target for tree in trees]
    q_scores = [tree.q_score for tree in trees]
    q_selection = _selection_metrics(q_scores, target)
    maximum_shortlist = min(max(shortlists), len(trees))
    maximum_indices = _shortlist_indices(q_scores, maximum_shortlist)
    beam_cache: dict[int, dict[int, float]] = {width: {} for width in beam_widths}
    for index in maximum_indices:
        for width in beam_widths:
            beam_cache[width][index] = _beam_scores(trees[index], model, width)[0]

    best = min(target)
    best_count = sum(value == best for value in target)
    results: dict[int, CascadeMetrics] = {}
    for requested_size in shortlists:
        indices = _shortlist_indices(q_scores, requested_size)
        subset_target = [target[index] for index in indices]
        selection = {
            width: _selection_metrics(
                [beam_cache[width][index] for index in indices],
                subset_target,
            )
            for width in beam_widths
        }
        results[requested_size] = CascadeMetrics(
            shortlist_size=len(indices),
            best_tier_coverage=float(any(target[index] == best for index in indices)),
            random_coverage=_random_best_coverage(
                len(trees),
                best_count,
                len(indices),
            ),
            selection=selection,
        )
    return q_selection, results


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.3f}"


def _mean_or_none(values: list[float]) -> float | None:
    return mean(values) if values else None


def _print_fold(
    seed: int,
    training: list[BeamTree],
    evaluation: list[BeamTree],
    shortlists: tuple[int, ...],
    beam_widths: tuple[int, ...],
) -> tuple[SelectionMetrics, dict[int, CascadeMetrics]]:
    training_samples = [sample for tree in training for sample in tree.transitions]
    model = _fit_probe(
        training_samples,
        SHADOW_DIMENSION,
        label="innovation_shift",
    )
    q_selection, results = _evaluate_group(
        evaluation,
        model,
        shortlists,
        beam_widths,
    )
    print(
        f"EVAL seed={seed} trees={len(evaluation)} "
        f"target_unique={len(set(tree.target for tree in evaluation))} "
        f"Q bestHit/regret={q_selection.best_tier_hit:.0f}/"
        f"{_fmt(q_selection.normalized_regret)}"
    )
    for requested_size in shortlists:
        metric = results[requested_size]
        print(
            f"  M={metric.shortlist_size:<2d} coverage/random="
            f"{metric.best_tier_coverage:.0f}/{metric.random_coverage:.3f}"
        )
        for width in beam_widths:
            selected = metric.selection[width]
            print(
                f"    B={width} rho={_fmt(selected.rho)} "
                f"pairAcc={_fmt(selected.pairwise_accuracy)} "
                f"bestHit={selected.best_tier_hit:.0f} "
                f"regret={_fmt(selected.normalized_regret)}"
            )
    return q_selection, results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-width", type=int, default=DEFAULT_TRAIN_WIDTH)
    parser.add_argument("--eval-width", type=int)
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument(
        "--shortlists",
        type=int,
        nargs="+",
        default=list(DEFAULT_SHORTLISTS),
    )
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
    if not args.shortlists or any(size <= 0 for size in args.shortlists):
        parser.error("--shortlists must contain positive integers")
    if not args.beam_widths or any(width <= 0 for width in args.beam_widths):
        parser.error("--beam-widths must contain positive integers")
    if len(args.input_seeds) < 2:
        parser.error("at least two training seeds are required")
    if args.random_graphs_per_family < 2:
        parser.error("--random-graphs-per-family must be at least 2")
    train_seeds = tuple(args.input_seeds)
    eval_seeds = tuple(args.eval_input_seeds or args.input_seeds)
    shortlists = tuple(sorted(set(args.shortlists)))
    beam_widths = tuple(sorted(set(args.beam_widths)))

    print("Q_K shortlist plus joint cell-beam reranking calibration")
    print("CALIBRATION ONLY — evaluated scores never read their oracle labels")
    print(
        f"train_width={args.train_width} eval_width={eval_width} "
        f"budget={args.budget} shortlists={','.join(map(str, shortlists))} "
        f"beam_widths={','.join(map(str, beam_widths))} "
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

    pooled_q: list[SelectionMetrics] = []
    pooled: dict[tuple[int, int], list[tuple[CascadeMetrics, SelectionMetrics]]] = {}
    if same_groups:
        iterator = []
        for held_out_index, (seed, evaluation) in enumerate(
            zip(eval_seeds, eval_groups, strict=True)
        ):
            training = [
                tree
                for group_index, group in enumerate(train_groups)
                if group_index != held_out_index
                for tree in group
            ]
            iterator.append((seed, training, evaluation))
    else:
        training = [tree for group in train_groups for tree in group]
        iterator = [
            (seed, training, evaluation)
            for seed, evaluation in zip(eval_seeds, eval_groups, strict=True)
        ]

    for seed, training, evaluation in iterator:
        q_metric, fold = _print_fold(
            seed,
            training,
            evaluation,
            shortlists,
            beam_widths,
        )
        pooled_q.append(q_metric)
        for size, cascade in fold.items():
            for width, selected in cascade.selection.items():
                pooled.setdefault((size, width), []).append((cascade, selected))
        print()

    print("FOLD SUMMARY")
    print(
        f"  Q full bestHit={mean(metric.best_tier_hit for metric in pooled_q):.3f} "
        f"regret={mean(metric.normalized_regret or 0.0 for metric in pooled_q):.3f}"
    )
    for size in shortlists:
        reference = pooled[(size, beam_widths[0])]
        print(
            f"  M={size:<2d} coverage/random="
            f"{mean(item[0].best_tier_coverage for item in reference):.3f}/"
            f"{mean(item[0].random_coverage for item in reference):.3f}"
        )
        for width in beam_widths:
            items = pooled[(size, width)]
            selections = [item[1] for item in items]
            rhos = [metric.rho for metric in selections if metric.rho is not None]
            pairs = [
                metric.pairwise_accuracy
                for metric in selections
                if metric.pairwise_accuracy is not None
            ]
            regrets = [
                metric.normalized_regret
                for metric in selections
                if metric.normalized_regret is not None
            ]
            print(
                f"    B={width} rho={_fmt(_mean_or_none(rhos))} "
                f"pairAcc={_fmt(_mean_or_none(pairs))} "
                f"bestHit={mean(metric.best_tier_hit for metric in selections):.3f} "
                f"regret={_fmt(_mean_or_none(regrets))}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
