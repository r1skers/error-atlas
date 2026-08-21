"""Calibrate an energy-mass and Q-gap adaptive shortlist cascade.

The fixed K=8 root band captures a similar fraction of ULP energy at widths 256 and 1024, even
though raw Q changes scale.  This experiment therefore replaces a width-proportional node count
with the smallest connected root-band prefix that captures a requested fraction of full-tree ULP
energy.  Candidate Q values are robustly standardized within each input group only to measure
shortlist-boundary gaps; this affine transform does not change the Q ordering.

The label-free shortlist rule considers M=4, 8, and 16.  It permits M=4 only when the median 80%
energy budget is at most eight nodes and the robust Q gap at four is at least 0.10.  Otherwise it
uses M=8 when the gap at eight is at least 0.05, falling back to M=16.  These rounded thresholds
were chosen from predictor-feature distributions, not oracle labels.  Fixed K/M policies are
reported beside the adaptive rule.

Same-width fitting is leave-one-input-seed-out.  Cross-width mode transfers one fixed innovation
model and may use disjoint evaluation seeds.  Evaluated scores never consume their oracle targets;
project held-out inputs remain untouched.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from statistics import mean, median

from predictor_ancestor_cell_beam_score_calibration import (
    BeamTree,
    _beam_scores,
    _generate_width,
)
from predictor_ancestor_transition_predictability_diagnostic import SHADOW_DIMENSION
from predictor_q_beam_shortlist_cascade_calibration import (
    SelectionMetrics,
    _fmt,
    _random_best_coverage,
    _selection_metrics,
    _shortlist_indices,
)
from predictor_signed_cell_shift_predictability_diagnostic import _fit_probe
from predictor_two_stage_cheap_score_calibration import (
    DEFAULT_INPUT_SEEDS,
    DEFAULT_RANDOM_GRAPHS_PER_FAMILY,
    _fp32_ulp_fraction,
)


DEFAULT_TRAIN_WIDTH = 256
DEFAULT_MAX_BUDGET = 32
DEFAULT_FIXED_BUDGET = 8
DEFAULT_ENERGY_MASS = 0.80
DEFAULT_BEAM_WIDTHS = (1, 3)
DEFAULT_SHORTLISTS = (4, 8, 16)


@dataclass(frozen=True)
class EnergyBudget:
    tree: BeamTree
    node_count: int
    captured_fraction: float
    q_score: float


@dataclass(frozen=True)
class PolicyResult:
    shortlist_size: int
    coverage: float
    random_coverage: float
    selection: SelectionMetrics


def _robust_gap(values: list[float], boundary: int) -> float:
    """Return the adjacent sorted gap in median-absolute-deviation units."""
    if not values:
        raise ValueError("values must be nonempty")
    if boundary <= 0 or boundary >= len(values):
        raise ValueError("boundary must split the values")
    ordered = sorted(values)
    center = median(ordered)
    scale = median(abs(value - center) for value in ordered)
    raw_gap = ordered[boundary] - ordered[boundary - 1]
    if scale == 0.0:
        return float("inf") if raw_gap > 0.0 else 0.0
    return raw_gap / scale


def _energy_budget(
    tree: BeamTree,
    energy_mass: float,
    minimum_budget: int,
    maximum_budget: int,
) -> EnergyBudget:
    """Choose the smallest connected root-band prefix reaching the energy target."""
    if not 0.0 < energy_mass <= 1.0:
        raise ValueError("energy_mass must be in (0, 1]")
    if minimum_budget <= 0 or maximum_budget < minimum_budget:
        raise ValueError("invalid budget range")
    available = min(maximum_budget, len(tree.selected_order))
    minimum = min(minimum_budget, available)
    energy = {
        index: float(_fp32_ulp_fraction(tree.exact_subtree[index]) / tree.root_ulp) ** 2
        for index in range(tree.graph.leaf_count, tree.graph.root + 1)
    }
    full_q = sum(energy.values())
    target = energy_mass * full_q
    accumulated = 0.0
    selected_count = available
    for count, index in enumerate(tree.selected_order[:available], 1):
        accumulated += energy[index]
        if count >= minimum and accumulated >= target:
            selected_count = count
            break
    selected = tree.selected_order[:selected_count]
    selected_q = sum(energy[index] for index in selected)
    selected_set = set(selected)
    return EnergyBudget(
        tree=replace(
            tree,
            selected_order=selected,
            transitions=tuple(
                sample
                for sample in tree.transitions
                if sample.node_index in selected_set
            ),
        ),
        node_count=selected_count,
        captured_fraction=selected_q / full_q,
        q_score=selected_q / 12.0,
    )


def _adaptive_shortlist_size(
    q_scores: list[float],
    median_energy_budget: float,
    shortlists: tuple[int, int, int] = DEFAULT_SHORTLISTS,
) -> int:
    """Apply the label-free concentration/gap stopping rule."""
    if len(shortlists) != 3 or tuple(sorted(shortlists)) != shortlists:
        raise ValueError("shortlists must contain three increasing sizes")
    small, medium, large = shortlists
    if large > len(q_scores):
        raise ValueError("largest shortlist exceeds candidate count")
    if median_energy_budget <= DEFAULT_FIXED_BUDGET:
        if _robust_gap(q_scores, small) >= 0.10:
            return small
    if _robust_gap(q_scores, medium) >= 0.05:
        return medium
    return large


def _evaluate_policy(
    shortlist_budgets: list[EnergyBudget],
    model,
    shortlist_size: int,
    beam_width: int,
    score_budgets: list[EnergyBudget] | None = None,
) -> PolicyResult:
    if score_budgets is None:
        score_budgets = shortlist_budgets
    if len(score_budgets) != len(shortlist_budgets):
        raise ValueError("shortlist and score budgets must align")
    q_scores = [item.q_score for item in shortlist_budgets]
    indices = _shortlist_indices(q_scores, shortlist_size)
    target = [item.tree.target for item in shortlist_budgets]
    subset_target = [target[index] for index in indices]
    scores = [
        _beam_scores(score_budgets[index].tree, model, beam_width)[0]
        for index in indices
    ]
    best = min(target)
    best_count = sum(value == best for value in target)
    return PolicyResult(
        shortlist_size=len(indices),
        coverage=float(any(target[index] == best for index in indices)),
        random_coverage=_random_best_coverage(
            len(shortlist_budgets), best_count, len(indices)
        ),
        selection=_selection_metrics(scores, subset_target),
    )


def _fixed_budget(tree: BeamTree, budget: int) -> EnergyBudget:
    return _energy_budget(tree, 1.0, budget, budget)


def _print_result(label: str, result: PolicyResult) -> None:
    metric = result.selection
    print(
        f"    {label:<16} M={result.shortlist_size:<2d} "
        f"coverage/random={result.coverage:.0f}/{result.random_coverage:.3f} "
        f"rho={_fmt(metric.rho)} pairAcc={_fmt(metric.pairwise_accuracy)} "
        f"bestHit={metric.best_tier_hit:.0f} regret={_fmt(metric.normalized_regret)}"
    )


def _evaluate_fold(
    seed: int,
    training: list[BeamTree],
    evaluation: list[BeamTree],
    energy_mass: float,
    fixed_budget: int,
    max_budget: int,
    shortlists: tuple[int, int, int],
    beam_widths: tuple[int, ...],
) -> dict[tuple[str, int], PolicyResult]:
    model = _fit_probe(
        [sample for tree in training for sample in tree.transitions],
        SHADOW_DIMENSION,
        label="innovation_shift",
    )
    fixed = [_fixed_budget(tree, fixed_budget) for tree in evaluation]
    adaptive = [
        _energy_budget(tree, energy_mass, 4, max_budget) for tree in evaluation
    ]
    median_k = median(item.node_count for item in adaptive)
    adaptive_m = _adaptive_shortlist_size(
        [item.q_score for item in adaptive], median_k, shortlists
    )
    best_prevalence = mean(
        tree.target == min(item.target for item in evaluation)
        for tree in evaluation
    )
    print(
        f"EVAL seed={seed} trees={len(evaluation)} best_prevalence={best_prevalence:.3f} "
        f"energyK mean/median/range="
        f"{mean(item.node_count for item in adaptive):.1f}/{median_k:.1f}/"
        f"{min(item.node_count for item in adaptive)}-"
        f"{max(item.node_count for item in adaptive)} adaptiveM={adaptive_m}"
    )
    results: dict[tuple[str, int], PolicyResult] = {}
    for beam_width in beam_widths:
        for shortlist in shortlists:
            label = f"fixedK{fixed_budget}/M{shortlist}"
            result = _evaluate_policy(fixed, model, shortlist, beam_width)
            results[(label, beam_width)] = result
        for shortlist in shortlists:
            label = f"massK/M{shortlist}"
            result = _evaluate_policy(adaptive, model, shortlist, beam_width)
            results[(label, beam_width)] = result
        for q_label, shortlist_budget, score_budget in (
            ("fixedQ/massBeam", fixed, adaptive),
            ("massQ/fixedBeam", adaptive, fixed),
        ):
            results[(q_label, beam_width)] = _evaluate_policy(
                shortlist_budget,
                model,
                shortlists[0],
                beam_width,
                score_budgets=score_budget,
            )
        label = "massK/adaptiveM"
        results[(label, beam_width)] = _evaluate_policy(
            adaptive, model, adaptive_m, beam_width
        )
        print(f"  B={beam_width}")
        for (policy, width), result in results.items():
            if width == beam_width:
                _print_result(policy, result)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-width", type=int, default=DEFAULT_TRAIN_WIDTH)
    parser.add_argument("--eval-width", type=int)
    parser.add_argument("--energy-mass", type=float, default=DEFAULT_ENERGY_MASS)
    parser.add_argument("--fixed-budget", type=int, default=DEFAULT_FIXED_BUDGET)
    parser.add_argument("--max-budget", type=int, default=DEFAULT_MAX_BUDGET)
    parser.add_argument(
        "--beam-widths", type=int, nargs="+", default=list(DEFAULT_BEAM_WIDTHS)
    )
    parser.add_argument(
        "--input-seeds", type=int, nargs="+", default=list(DEFAULT_INPUT_SEEDS)
    )
    parser.add_argument("--eval-input-seeds", type=int, nargs="+")
    parser.add_argument(
        "--random-graphs-per-family",
        type=int,
        default=DEFAULT_RANDOM_GRAPHS_PER_FAMILY,
    )
    args = parser.parse_args()
    eval_width = args.eval_width or args.train_width
    if args.train_width < 2 or eval_width < 2:
        parser.error("widths must be at least 2")
    if not 0.0 < args.energy_mass <= 1.0:
        parser.error("--energy-mass must be in (0, 1]")
    if args.fixed_budget < 4 or args.max_budget < args.fixed_budget:
        parser.error("budget range must satisfy 4 <= fixed <= max")
    if not args.beam_widths or any(width <= 0 for width in args.beam_widths):
        parser.error("--beam-widths must contain positive integers")
    if len(args.input_seeds) < 2:
        parser.error("at least two training seeds are required")
    if args.random_graphs_per_family < 8:
        parser.error("--random-graphs-per-family must be at least 8")

    train_seeds = tuple(args.input_seeds)
    eval_seeds = tuple(args.eval_input_seeds or args.input_seeds)
    beam_widths = tuple(sorted(set(args.beam_widths)))
    shortlists = DEFAULT_SHORTLISTS
    print("Width-aware energy-mass plus Q-gap cascade calibration")
    print("CALIBRATION ONLY — adaptive decisions consume predictor features only")
    print(
        f"train_width={args.train_width} eval_width={eval_width} "
        f"energy_mass={args.energy_mass:.2f} fixed_budget={args.fixed_budget} "
        f"max_budget={args.max_budget} train_seeds={','.join(map(str, train_seeds))} "
        f"eval_seeds={','.join(map(str, eval_seeds))}"
    )
    print()

    train_groups = _generate_width(
        args.train_width, train_seeds, args.random_graphs_per_family, args.max_budget
    )
    same_groups = eval_width == args.train_width and eval_seeds == train_seeds
    eval_groups = train_groups if same_groups else _generate_width(
        eval_width, eval_seeds, args.random_graphs_per_family, args.max_budget
    )
    folds: list[dict[tuple[str, int], PolicyResult]] = []
    if same_groups:
        iterator = []
        for held_out, (seed, evaluation) in enumerate(
            zip(eval_seeds, eval_groups, strict=True)
        ):
            training = [
                tree
                for index, group in enumerate(train_groups)
                if index != held_out
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
        folds.append(
            _evaluate_fold(
                seed,
                training,
                evaluation,
                args.energy_mass,
                args.fixed_budget,
                args.max_budget,
                shortlists,
                beam_widths,
            )
        )
        print()

    print("FOLD SUMMARY")
    for key in folds[0]:
        policy, beam_width = key
        items = [fold[key] for fold in folds]
        metrics = [item.selection for item in items]
        rhos = [item.rho for item in metrics if item.rho is not None]
        pairs = [
            item.pairwise_accuracy
            for item in metrics
            if item.pairwise_accuracy is not None
        ]
        regrets = [
            item.normalized_regret
            for item in metrics
            if item.normalized_regret is not None
        ]
        print(
            f"  B={beam_width} {policy:<16} "
            f"meanM={mean(item.shortlist_size for item in items):.1f} "
            f"coverage={mean(item.coverage for item in items):.3f} "
            f"rho={_fmt(mean(rhos) if rhos else None)} "
            f"pairAcc={_fmt(mean(pairs) if pairs else None)} "
            f"bestHit={mean(item.best_tier_hit for item in metrics):.3f} "
            f"regret={_fmt(mean(regrets) if regrets else None)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
