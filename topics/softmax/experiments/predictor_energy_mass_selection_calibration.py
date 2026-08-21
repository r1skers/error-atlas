"""Select an energy-mass budget on calibration folds, then lock it for width transfer.

This bounded follow-up compares energy masses 0.70, 0.80, and 0.90 using only leave-one-seed-out
width-256 calibration folds.  The winner is chosen lexicographically by global-best shortlist
coverage, selected-tree best-tier hit, normalized regret, pairwise accuracy, and finally lower mean
node count.  It is then frozen before evaluation on disjoint seeds and requested widths.

The deployed policy is deliberately fixed at M=4 and B=3.  A fixed K=8 policy fitted with the same
innovation model is reported as the baseline.  Project held-out inputs remain untouched.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from statistics import mean

from predictor_ancestor_cell_beam_score_calibration import BeamTree, _generate_width
from predictor_ancestor_transition_predictability_diagnostic import SHADOW_DIMENSION
from predictor_q_beam_shortlist_cascade_calibration import _fmt
from predictor_signed_cell_shift_predictability_diagnostic import _fit_probe
from predictor_two_stage_cheap_score_calibration import (
    DEFAULT_INPUT_SEEDS,
    DEFAULT_RANDOM_GRAPHS_PER_FAMILY,
)
from predictor_width_aware_cascade_calibration import (
    _energy_budget,
    _evaluate_policy,
    _fixed_budget,
)


DEFAULT_TRAIN_WIDTH = 256
DEFAULT_EVAL_WIDTHS = (128, 512, 1024)
DEFAULT_ENERGY_MASSES = (0.70, 0.80, 0.90)
DEFAULT_EVAL_SEEDS = tuple(range(22260825, 22260833))
DEFAULT_MAX_BUDGET = 32
DEFAULT_FIXED_BUDGET = 8
DEFAULT_SHORTLIST = 4
DEFAULT_BEAM_WIDTH = 3


@dataclass(frozen=True)
class MassSummary:
    energy_mass: float
    mean_node_count: float
    coverage: float
    best_tier_hit: float
    normalized_regret: float
    pairwise_accuracy: float | None
    rho: float | None


def _defined_mean(values: list[float | None]) -> float | None:
    defined = [value for value in values if value is not None]
    return mean(defined) if defined else None


def _selection_key(summary: MassSummary) -> tuple[float, ...]:
    """Prefer selection utility, then ranking quality, then lower predictor work."""
    return (
        summary.coverage,
        summary.best_tier_hit,
        -summary.normalized_regret,
        summary.pairwise_accuracy if summary.pairwise_accuracy is not None else 0.5,
        -summary.mean_node_count,
    )


def _select_mass(summaries: list[MassSummary]) -> MassSummary:
    if not summaries:
        raise ValueError("summaries must be nonempty")
    return max(summaries, key=_selection_key)


def _fit_model(trees: list[BeamTree]):
    return _fit_probe(
        [sample for tree in trees for sample in tree.transitions],
        SHADOW_DIMENSION,
        label="innovation_shift",
    )


def _summarize_mass(
    energy_mass: float,
    groups: list[list[BeamTree]],
    maximum_budget: int,
    shortlist: int,
    beam_width: int,
) -> MassSummary:
    fold_results = []
    node_counts: list[int] = []
    for held_out, evaluation in enumerate(groups):
        training = [
            tree
            for index, group in enumerate(groups)
            if index != held_out
            for tree in group
        ]
        model = _fit_model(training)
        budgets = [
            _energy_budget(tree, energy_mass, 4, maximum_budget)
            for tree in evaluation
        ]
        node_counts.extend(item.node_count for item in budgets)
        fold_results.append(
            _evaluate_policy(budgets, model, shortlist, beam_width)
        )
    metrics = [result.selection for result in fold_results]
    return MassSummary(
        energy_mass=energy_mass,
        mean_node_count=mean(node_counts),
        coverage=mean(result.coverage for result in fold_results),
        best_tier_hit=mean(metric.best_tier_hit for metric in metrics),
        normalized_regret=mean(
            metric.normalized_regret or 0.0 for metric in metrics
        ),
        pairwise_accuracy=_defined_mean(
            [metric.pairwise_accuracy for metric in metrics]
        ),
        rho=_defined_mean([metric.rho for metric in metrics]),
    )


def _print_summary(prefix: str, summary: MassSummary) -> None:
    print(
        f"{prefix} mass={summary.energy_mass:.2f} meanK={summary.mean_node_count:.2f} "
        f"coverage={summary.coverage:.3f} bestHit={summary.best_tier_hit:.3f} "
        f"regret={summary.normalized_regret:.3f} "
        f"pairAcc={_fmt(summary.pairwise_accuracy)} rho={_fmt(summary.rho)}"
    )


def _evaluate_width(
    width: int,
    groups: list[list[BeamTree]],
    model,
    selected_mass: float,
    maximum_budget: int,
    fixed_budget: int,
    shortlist: int,
    beam_width: int,
) -> tuple[MassSummary, MassSummary]:
    mass_results = []
    fixed_results = []
    mass_nodes: list[int] = []
    for seed_index, group in enumerate(groups):
        mass_budgets = [
            _energy_budget(tree, selected_mass, 4, maximum_budget)
            for tree in group
        ]
        fixed_budgets = [_fixed_budget(tree, fixed_budget) for tree in group]
        mass_nodes.extend(item.node_count for item in mass_budgets)
        mass_result = _evaluate_policy(
            mass_budgets, model, shortlist, beam_width
        )
        fixed_result = _evaluate_policy(
            fixed_budgets, model, shortlist, beam_width
        )
        mass_results.append(mass_result)
        fixed_results.append(fixed_result)
        print(
            f"  group={seed_index} massK={mean(item.node_count for item in mass_budgets):.1f} "
            f"mass hit/regret={mass_result.selection.best_tier_hit:.0f}/"
            f"{_fmt(mass_result.selection.normalized_regret)} "
            f"fixed hit/regret={fixed_result.selection.best_tier_hit:.0f}/"
            f"{_fmt(fixed_result.selection.normalized_regret)}"
        )

    def summarize(
        energy_mass: float,
        results,
        mean_nodes: float,
    ) -> MassSummary:
        metrics = [result.selection for result in results]
        return MassSummary(
            energy_mass=energy_mass,
            mean_node_count=mean_nodes,
            coverage=mean(result.coverage for result in results),
            best_tier_hit=mean(metric.best_tier_hit for metric in metrics),
            normalized_regret=mean(
                metric.normalized_regret or 0.0 for metric in metrics
            ),
            pairwise_accuracy=_defined_mean(
                [metric.pairwise_accuracy for metric in metrics]
            ),
            rho=_defined_mean([metric.rho for metric in metrics]),
        )

    mass_summary = summarize(
        selected_mass, mass_results, mean(mass_nodes)
    )
    fixed_summary = summarize(
        1.0, fixed_results, float(fixed_budget)
    )
    print(f"WIDTH SUMMARY width={width}")
    _print_summary("  selected", mass_summary)
    _print_summary("  fixedK8 ", fixed_summary)
    return mass_summary, fixed_summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-width", type=int, default=DEFAULT_TRAIN_WIDTH)
    parser.add_argument(
        "--eval-widths", type=int, nargs="+", default=list(DEFAULT_EVAL_WIDTHS)
    )
    parser.add_argument(
        "--energy-masses", type=float, nargs="+", default=list(DEFAULT_ENERGY_MASSES)
    )
    parser.add_argument(
        "--input-seeds", type=int, nargs="+", default=list(DEFAULT_INPUT_SEEDS)
    )
    parser.add_argument(
        "--eval-input-seeds", type=int, nargs="+", default=list(DEFAULT_EVAL_SEEDS)
    )
    parser.add_argument("--max-budget", type=int, default=DEFAULT_MAX_BUDGET)
    parser.add_argument("--fixed-budget", type=int, default=DEFAULT_FIXED_BUDGET)
    parser.add_argument("--shortlist", type=int, default=DEFAULT_SHORTLIST)
    parser.add_argument("--beam-width", type=int, default=DEFAULT_BEAM_WIDTH)
    parser.add_argument(
        "--random-graphs-per-family",
        type=int,
        default=DEFAULT_RANDOM_GRAPHS_PER_FAMILY,
    )
    args = parser.parse_args()
    if args.train_width < 2 or not args.eval_widths or any(
        width < 2 for width in args.eval_widths
    ):
        parser.error("widths must be at least 2")
    if not args.energy_masses or any(
        not 0.0 < mass <= 1.0 for mass in args.energy_masses
    ):
        parser.error("--energy-masses must lie in (0, 1]")
    if len(args.input_seeds) < 3 or not args.eval_input_seeds:
        parser.error("at least three training seeds and one evaluation seed are required")
    if set(args.input_seeds) & set(args.eval_input_seeds):
        parser.error("training and evaluation seeds must be disjoint")
    if args.max_budget < args.fixed_budget or args.fixed_budget < 4:
        parser.error("budget range must satisfy 4 <= fixed <= max")
    if args.shortlist <= 0 or args.beam_width <= 0:
        parser.error("shortlist and beam width must be positive")
    if args.random_graphs_per_family < args.shortlist:
        parser.error("graph count per family must cover the shortlist")

    masses = tuple(sorted(set(args.energy_masses)))
    eval_widths = tuple(dict.fromkeys(args.eval_widths))
    train_seeds = tuple(args.input_seeds)
    eval_seeds = tuple(args.eval_input_seeds)
    print("Train-only energy-mass selection and locked width transfer")
    print("CALIBRATION ONLY — project held-out inputs untouched")
    print(
        f"train_width={args.train_width} masses={','.join(f'{m:.2f}' for m in masses)} "
        f"M={args.shortlist} B={args.beam_width} eval_widths="
        f"{','.join(map(str, eval_widths))} eval_seeds={','.join(map(str, eval_seeds))}"
    )
    print()

    train_groups = _generate_width(
        args.train_width,
        train_seeds,
        args.random_graphs_per_family,
        args.max_budget,
    )
    summaries = [
        _summarize_mass(
            mass,
            train_groups,
            args.max_budget,
            args.shortlist,
            args.beam_width,
        )
        for mass in masses
    ]
    print("TRAIN-ONLY SELECTION")
    for summary in summaries:
        _print_summary("  candidate", summary)
    selected = _select_mass(summaries)
    print(f"  LOCKED mass={selected.energy_mass:.2f}")
    print()

    model = _fit_model([tree for group in train_groups for tree in group])
    for width in eval_widths:
        print(f"EVAL width={width}")
        groups = _generate_width(
            width,
            eval_seeds,
            args.random_graphs_per_family,
            args.max_budget,
        )
        _evaluate_width(
            width,
            groups,
            model,
            selected.energy_mass,
            args.max_budget,
            args.fixed_budget,
            args.shortlist,
            args.beam_width,
        )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
