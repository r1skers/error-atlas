"""Calibration-only cost/accuracy diagnostic for ULP-energy tree scores.

The full ULP-energy score

    Q = sum_v (ulp32(S_v) / ulp32(S_root))**2

requires one contribution from every internal node.  A full binary reduction tree always performs
``n - 1`` additions, so work alone cannot distinguish candidate trees.  Their critical-path span can,
and score evaluation itself must also be budgeted before Q can be called cheap.

This diagnostic therefore reports two separate costs:

* execution proxy: tree span (the longest leaf-to-root addition chain), while work is fixed;
* score-side node budget: a root-first best-first traversal that observes at most K internal-node ULP
  contributions, prioritizing the structurally largest available subtree.

The budgeted score assumes exact-subtree-sum/ULP metadata is already maintained while constructing a
candidate tree.  Building that metadata remains O(n) and is not hidden inside K.  Given the metadata,
the root-band read is O(K log K); full Q is O(n).  No wall-clock hardware claim is made here.

For each input and random-tree family, the script measures retained Q energy, Spearman agreement with
full-Q ranking, selected-best full-Q regret, Q/span Pareto structure, and balanced/sequential anchors.
All samples are calibration-only; held-out inputs remain untouched and no budget is frozen.
"""
from __future__ import annotations

import argparse
import heapq
from dataclasses import dataclass
from fractions import Fraction

from predictor_calibration_inputs import wide_range_random
from predictor_ranking_smoke import _spearman
from predictor_shadow_sparse_repair_ablation import _fp32_ulp_fraction
from predictor_tree_generator import (
    random_contiguous_split_graph,
    random_pair_merge_graph,
)
from predictor_ulp_energy_convergence_diagnostic import _quantile
from summation_graph_predictor import (
    BinaryReductionGraph,
    balanced_reduction_graph,
    sequential_reduction_graph,
)


DEFAULT_WIDTHS = (256, 1024)
DEFAULT_INPUT_SEEDS = (22260821, 22260822, 22260823, 22260824)
DEFAULT_RANDOM_GRAPHS_PER_FAMILY = 32
DEFAULT_BUDGETS = (4, 8, 16, 32, 64)
CONTIGUOUS_TREE_BASE_SEED = 43_000_000
PAIR_TREE_BASE_SEED = 44_000_000
RANDOM_FAMILIES = ("contiguous", "pair_merge")


@dataclass(frozen=True)
class TreeCostProfile:
    width: int
    family: str
    work: int
    span: int
    full_q: float
    budget_q: dict[int, float]


def _tree_cost_profile(
    values: tuple[Fraction, ...],
    graph: BinaryReductionGraph,
    family: str,
    budgets: tuple[int, ...],
) -> TreeCostProfile:
    if len(values) != graph.leaf_count:
        raise ValueError("value count must match graph leaf count")
    if not budgets or any(budget <= 0 for budget in budgets):
        raise ValueError("budgets must contain positive integers")

    exact_subtree = [*values]
    subtree_leaves = [1] * graph.leaf_count
    value_span = [0] * graph.leaf_count
    node_ulps: dict[int, Fraction] = {}
    for offset, node in enumerate(graph.nodes):
        index = graph.leaf_count + offset
        exact_sum = exact_subtree[node.left] + exact_subtree[node.right]
        exact_subtree.append(exact_sum)
        subtree_leaves.append(subtree_leaves[node.left] + subtree_leaves[node.right])
        value_span.append(1 + max(value_span[node.left], value_span[node.right]))
        node_ulps[index] = _fp32_ulp_fraction(exact_sum)

    root_ulp = _fp32_ulp_fraction(exact_subtree[graph.root])
    node_energy = {
        index: float(ulp / root_ulp) ** 2 for index, ulp in node_ulps.items()
    }
    full_q = sum(node_energy.values())

    ordered_budgets = tuple(sorted(set(budgets)))
    max_visits = min(ordered_budgets[-1], len(graph.nodes))
    frontier = [(-subtree_leaves[graph.root], graph.root)]
    cumulative_q = 0.0
    q_after_visit: list[float] = []
    while frontier and len(q_after_visit) < max_visits:
        _, index = heapq.heappop(frontier)
        cumulative_q += node_energy[index]
        q_after_visit.append(cumulative_q)
        node = graph.nodes[index - graph.leaf_count]
        for child in (node.left, node.right):
            if child >= graph.leaf_count:
                heapq.heappush(frontier, (-subtree_leaves[child], child))

    budget_q = {
        budget: q_after_visit[min(budget, len(q_after_visit)) - 1]
        for budget in ordered_budgets
    }
    return TreeCostProfile(
        width=len(values),
        family=family,
        work=len(graph.nodes),
        span=value_span[graph.root],
        full_q=full_q,
        budget_q=budget_q,
    )


def _pareto_indices(rows: list[TreeCostProfile]) -> set[int]:
    """Return indices not dominated when minimizing both full Q and span."""
    efficient: set[int] = set()
    for index, row in enumerate(rows):
        dominated = any(
            other.full_q <= row.full_q
            and other.span <= row.span
            and (other.full_q < row.full_q or other.span < row.span)
            for other_index, other in enumerate(rows)
            if other_index != index
        )
        if not dominated:
            efficient.add(index)
    return efficient


def _median(values: list[float]) -> float:
    return _quantile(values, 0.5)


def _format_rho(value: float | None) -> str:
    return "undefined" if value is None else f"{value:+.3f}"


def _random_graphs(width: int, input_index: int, count: int):
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


def _print_anchor_summary(
    width: int,
    anchors: dict[str, list[TreeCostProfile]],
) -> None:
    print(f"width={width} anchors (n={len(anchors['balanced'])}, descriptive)")
    for family in ("balanced", "sequential"):
        rows = anchors[family]
        print(
            f"  {family:<10} work={rows[0].work:<4d} "
            f"span_med={_median([float(row.span) for row in rows]):.1f} "
            f"Q_med={_median([row.full_q for row in rows]):.3f}"
        )


def _print_random_summary(
    width: int,
    family: str,
    groups: list[list[TreeCostProfile]],
    budgets: tuple[int, ...],
) -> None:
    rows = [row for group in groups for row in group]
    q_span_rhos = [
        _spearman(
            [row.full_q for row in group],
            [float(row.span) for row in group],
        )
        for group in groups
    ]
    defined_q_span = [rho for rho in q_span_rhos if rho is not None]
    pareto_counts = [float(len(_pareto_indices(group))) for group in groups]

    min_q_span_ratios: list[float] = []
    min_span_q_regrets: list[float] = []
    for group in groups:
        min_q = min(row.full_q for row in group)
        min_span = min(row.span for row in group)
        best_q_rows = [row for row in group if row.full_q == min_q]
        best_span_rows = [row for row in group if row.span == min_span]
        min_q_span_ratios.append(min(row.span for row in best_q_rows) / min_span)
        min_span_q_regrets.append(
            min(row.full_q for row in best_span_rows) / min_q
        )

    print(
        f"  family={family:<10} trees={len(rows):3d} groups={len(groups)} "
        f"Q_med={_median([row.full_q for row in rows]):.3f} "
        f"span_med/p90={_median([float(row.span) for row in rows]):.1f}/"
        f"{_quantile([float(row.span) for row in rows], 0.9):.1f} "
        f"rho(Q,span)_med={_format_rho(_median(defined_q_span) if defined_q_span else None)} "
        f"Pareto_count_med={_median(pareto_counts):.1f} "
        f"minQ_span/minSpan_med={_median(min_q_span_ratios):.3f} "
        f"minSpan_Qregret_med={_median(min_span_q_regrets):.3f}"
    )

    for budget in budgets:
        captures = [row.budget_q[budget] / row.full_q for row in rows]
        ranking_rhos: list[float] = []
        selection_regrets: list[float] = []
        for group in groups:
            rho = _spearman(
                [row.budget_q[budget] for row in group],
                [row.full_q for row in group],
            )
            if rho is not None:
                ranking_rhos.append(rho)
            selected = min(group, key=lambda row: row.budget_q[budget])
            selection_regrets.append(
                selected.full_q / min(row.full_q for row in group)
            )
        effective_budget = min(budget, rows[0].work)
        print(
            f"    K={budget:<3d} score_reads={100.0 * effective_budget / rows[0].work:5.1f}% "
            f"Qcapture_med/p10={_median(captures):.3f}/"
            f"{_quantile(captures, 0.1):.3f} "
            f"rho_med/min={_format_rho(_median(ranking_rhos) if ranking_rhos else None)}/"
            f"{_format_rho(min(ranking_rhos) if ranking_rhos else None)} "
            f"bestQ_regret_med/max={_median(selection_regrets):.3f}/"
            f"{max(selection_regrets):.3f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--widths", type=int, nargs="+", default=list(DEFAULT_WIDTHS))
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
    if any(width < 2 for width in args.widths):
        parser.error("--widths must contain only integers >= 2")
    if args.random_graphs_per_family < 2:
        parser.error("--random-graphs-per-family must be at least 2")
    if not args.budgets or any(budget <= 0 for budget in args.budgets):
        parser.error("--budgets must contain positive integers")

    widths = tuple(dict.fromkeys(args.widths))
    budgets = tuple(sorted(set(args.budgets)))
    print("Wide-range ULP-energy cost/accuracy Pareto diagnostic")
    print("CALIBRATION ONLY — no held-out inputs; no score budget is frozen")
    print("tree work is fixed at n-1; span is the execution critical-path proxy")
    print("score_reads assumes O(n) subtree metadata already exists; it prices only root-band reads")
    print(
        f"widths={','.join(map(str, widths))} "
        f"input_seeds={','.join(map(str, args.input_seeds))} "
        f"random_graphs_per_family={args.random_graphs_per_family} "
        f"budgets={','.join(map(str, budgets))}"
    )
    print()

    input_index = 0
    for width in widths:
        anchors: dict[str, list[TreeCostProfile]] = {
            "balanced": [],
            "sequential": [],
        }
        grouped: dict[str, list[list[TreeCostProfile]]] = {
            family: [] for family in RANDOM_FAMILIES
        }
        for input_seed in args.input_seeds:
            values = wide_range_random(width, seed=input_seed).values
            anchors["balanced"].append(
                _tree_cost_profile(
                    values,
                    balanced_reduction_graph(width),
                    "balanced",
                    budgets,
                )
            )
            anchors["sequential"].append(
                _tree_cost_profile(
                    values,
                    sequential_reduction_graph(width),
                    "sequential",
                    budgets,
                )
            )
            per_family = {family: [] for family in RANDOM_FAMILIES}
            for family, graph in _random_graphs(
                width,
                input_index,
                args.random_graphs_per_family,
            ):
                per_family[family].append(
                    _tree_cost_profile(values, graph, family, budgets)
                )
            for family in RANDOM_FAMILIES:
                grouped[family].append(per_family[family])
            input_index += 1

        _print_anchor_summary(width, anchors)
        for family in RANDOM_FAMILIES:
            _print_random_summary(width, family, grouped[family], budgets)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
