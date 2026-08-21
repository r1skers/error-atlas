"""Calibrate reliability-shrunk sparse coherence without seed leakage.

The unweighted two-stage score can improve at width 256 but fails to transfer to width 1024.  This
diagnostic asks whether the failure comes from treating every selected trajectory residual as equally
trustworthy.  For each root-band node it records only predictor-side conditioning features:

* absolute root-band rank bucket: 1--4, 5--8, 9--16, or 17--32;
* whether the shadow predicts an RN-cell crossing or residual-sign phase change;
* whether the shadow history magnitude is at least half a local ULP.

Node-level oracle residuals fit two zero-intercept reliability gains per bin:

* ``sign``: ULP-energy-weighted mean predicted/actual sign agreement, clipped to [0, 1];
* ``ols``: least-squares slope from predicted to actual residual, clipped to [0, 1].

The gains shrink selected residuals before adding their pairwise cross terms to Q/12.  When training
and evaluation widths match, every input seed is scored by a table fitted on the other seeds only.
When widths differ, all declared train-width seeds fit one table that is transferred unchanged to the
evaluation width.  Oracle tree errors and node residuals are calibration labels only; they never enter
the held-out fold's score.  Project held-out inputs remain untouched and no table is frozen.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from statistics import mean

from predictor_calibration_inputs import wide_range_random
from predictor_ranking_smoke import _spearman
from predictor_two_stage_cheap_score_calibration import (
    DEFAULT_BUDGETS,
    DEFAULT_INPUT_SEEDS,
    DEFAULT_RANDOM_GRAPHS_PER_FAMILY,
    _graphs,
    _predictor_trace,
    _sign,
)
from summation_graph_predictor import predict_fp32_tree_error


DEFAULT_TRAIN_WIDTH = 256
MIN_BIN_SAMPLES = 32


@dataclass(frozen=True)
class NodeRecord:
    rank: int
    event: bool
    large_drift: bool
    predicted: float
    actual: float
    energy: float


@dataclass(frozen=True)
class TreeSample:
    family: str
    target: float
    full_q: float
    q_budget: dict[int, float]
    unweighted_phase: dict[int, float]
    nodes: tuple[NodeRecord, ...]


@dataclass(frozen=True)
class GainPair:
    sign: float
    ols: float
    count: int


@dataclass(frozen=True)
class ReliabilityTable:
    global_gain: GainPair
    by_bin: dict[tuple[int, bool, bool], GainPair]

    def gain(self, record: NodeRecord, mode: str, *, binned: bool) -> float:
        if mode not in ("sign", "ols"):
            raise ValueError("mode must be 'sign' or 'ols'")
        selected = self.global_gain
        if binned:
            candidate = self.by_bin.get(_feature_bin(record))
            if candidate is not None and candidate.count >= MIN_BIN_SAMPLES:
                selected = candidate
        return getattr(selected, mode)


@dataclass(frozen=True)
class SelectionUtility:
    pairwise_accuracy: float | None
    best_tier_hit: float
    normalized_regret: float | None


@dataclass
class _GainAccumulator:
    count: int = 0
    sign_numerator: float = 0.0
    sign_denominator: float = 0.0
    ols_numerator: float = 0.0
    ols_denominator: float = 0.0

    def add(self, record: NodeRecord) -> None:
        self.count += 1
        self.sign_numerator += (
            record.energy * _sign(record.predicted) * _sign(record.actual)
        )
        self.sign_denominator += record.energy
        self.ols_numerator += record.predicted * record.actual
        self.ols_denominator += record.predicted * record.predicted

    def finish(self) -> GainPair:
        sign = (
            self.sign_numerator / self.sign_denominator
            if self.sign_denominator
            else 0.0
        )
        ols = self.ols_numerator / self.ols_denominator if self.ols_denominator else 0.0
        return GainPair(
            sign=min(1.0, max(0.0, sign)),
            ols=min(1.0, max(0.0, ols)),
            count=self.count,
        )


def _rank_bucket(rank: int) -> int:
    if rank < 0:
        raise ValueError("rank must be nonnegative")
    if rank < 4:
        return 0
    if rank < 8:
        return 1
    if rank < 16:
        return 2
    return 3


def _feature_bin(record: NodeRecord) -> tuple[int, bool, bool]:
    return (_rank_bucket(record.rank), record.event, record.large_drift)


def _fit_reliability(samples: list[TreeSample]) -> ReliabilityTable:
    if not samples:
        raise ValueError("at least one training tree is required")
    global_accumulator = _GainAccumulator()
    bins: dict[tuple[int, bool, bool], _GainAccumulator] = {}
    for sample in samples:
        for record in sample.nodes:
            global_accumulator.add(record)
            bins.setdefault(_feature_bin(record), _GainAccumulator()).add(record)
    return ReliabilityTable(
        global_gain=global_accumulator.finish(),
        by_bin={key: accumulator.finish() for key, accumulator in bins.items()},
    )


def _tree_sample(
    values,
    graph,
    family: str,
    budgets: tuple[int, ...],
) -> TreeSample:
    trace = _predictor_trace(values, graph, budgets)
    oracle = predict_fp32_tree_error(values, graph)
    actual_delta = {
        graph.leaf_count + offset: prediction.local_rounding_error
        for offset, prediction in enumerate(oracle.node_predictions)
    }

    trajectory_error = [0.0 for _ in values]
    history_ulp: dict[int, float] = {}
    for offset, node in enumerate(graph.nodes):
        index = graph.leaf_count + offset
        history = trajectory_error[node.left] + trajectory_error[node.right]
        history_ulp[index] = abs(history / float(trace.node_ulp[index]))
        trajectory_error.append(history + trace.trajectory_delta[index])

    records: list[NodeRecord] = []
    for rank, index in enumerate(trace.selected_order):
        normalized_ulp = float(trace.node_ulp[index] / trace.root_ulp)
        records.append(
            NodeRecord(
                rank=rank,
                event=trace.predicted_cross[index] or trace.predicted_phase[index],
                large_drift=history_ulp[index] >= 0.5,
                predicted=trace.trajectory_delta[index] / float(trace.root_ulp),
                actual=float(actual_delta[index] / trace.root_ulp),
                energy=normalized_ulp * normalized_ulp,
            )
        )

    return TreeSample(
        family=family,
        target=float(oracle.signed_error / trace.root_ulp) ** 2,
        full_q=trace.full_q,
        q_budget={budget: trace.budget[budget].q_budget for budget in budgets},
        unweighted_phase={
            budget: trace.budget[budget].coherence_phase for budget in budgets
        },
        nodes=tuple(records),
    )


def _reliability_score(
    sample: TreeSample,
    budget: int,
    table: ReliabilityTable,
    mode: str,
    *,
    binned: bool,
) -> float:
    selected = sample.nodes[: min(budget, len(sample.nodes))]
    means = [
        table.gain(record, mode, binned=binned) * record.predicted
        for record in selected
    ]
    pairwise = sum(means) ** 2 - sum(value * value for value in means)
    return max(0.0, sample.full_q / 12.0 + pairwise)


def _generate_width(
    width: int,
    seeds: tuple[int, ...],
    graphs_per_family: int,
    budgets: tuple[int, ...],
) -> list[list[TreeSample]]:
    groups: list[list[TreeSample]] = []
    for input_index, seed in enumerate(seeds):
        values = wide_range_random(width, seed=seed).values
        rows: list[TreeSample] = []
        for family, graph in _graphs(width, input_index, graphs_per_family):
            rows.append(_tree_sample(values, graph, family, budgets))
        groups.append(rows)
    return groups


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.3f}"


def _selection_utility(
    scores: list[float],
    target: list[float],
) -> SelectionUtility:
    if len(scores) != len(target) or not scores:
        raise ValueError("score and target vectors must be nonempty and equally sized")
    comparable = 0
    concordance = 0.0
    for left in range(len(target)):
        for right in range(left + 1, len(target)):
            target_difference = target[left] - target[right]
            if target_difference == 0.0:
                continue
            comparable += 1
            score_difference = scores[left] - scores[right]
            if score_difference == 0.0:
                concordance += 0.5
            elif (score_difference > 0.0) == (target_difference > 0.0):
                concordance += 1.0

    selected = min(range(len(scores)), key=scores.__getitem__)
    best = min(target)
    worst = max(target)
    regret = (target[selected] - best) / (worst - best) if worst > best else None
    return SelectionUtility(
        pairwise_accuracy=concordance / comparable if comparable else None,
        best_tier_hit=float(target[selected] == best),
        normalized_regret=regret,
    )


def _evaluate_group(
    rows: list[TreeSample],
    table: ReliabilityTable,
    budgets: tuple[int, ...],
) -> dict[str, float | None]:
    target = [row.target for row in rows]
    out: dict[str, float | None] = {
        "full_q": _spearman([row.full_q for row in rows], target),
        "random_hit": mean(value == min(target) for value in target),
    }
    for budget in budgets:
        policies = {
            "qK": [row.q_budget[budget] for row in rows],
            "phase": [row.unweighted_phase[budget] for row in rows],
            "sign_global": [
                _reliability_score(row, budget, table, "sign", binned=False)
                for row in rows
            ],
            "sign_bin": [
                _reliability_score(row, budget, table, "sign", binned=True)
                for row in rows
            ],
            "ols_global": [
                _reliability_score(row, budget, table, "ols", binned=False)
                for row in rows
            ],
            "ols_bin": [
                _reliability_score(row, budget, table, "ols", binned=True)
                for row in rows
            ],
        }
        for name, values in policies.items():
            out[f"{name}_{budget}"] = _spearman(values, target)
        for name in ("qK", "ols_bin"):
            utility = _selection_utility(policies[name], target)
            out[f"{name}_pair_{budget}"] = utility.pairwise_accuracy
            out[f"{name}_hit_{budget}"] = utility.best_tier_hit
            out[f"{name}_regret_{budget}"] = utility.normalized_regret
    return out


def _print_fold(
    seed: int,
    rows: list[TreeSample],
    table: ReliabilityTable,
    budgets: tuple[int, ...],
) -> dict[str, float | None]:
    stats = _evaluate_group(rows, table, budgets)
    print(
        f"EVAL seed={seed} trees={len(rows)} target_unique={len(set(row.target for row in rows))} "
        f"global_gain sign/ols={table.global_gain.sign:.3f}/{table.global_gain.ols:.3f} "
        f"rho_fullQ={_fmt(stats['full_q'])} "
        f"random_bestTier={stats['random_hit']:.3f}"
    )
    for budget in budgets:
        print(
            f"  K={budget:<2d} rho qK/phase/signG/signB/olsG/olsB="
            f"{_fmt(stats[f'qK_{budget}'])}/{_fmt(stats[f'phase_{budget}'])}/"
            f"{_fmt(stats[f'sign_global_{budget}'])}/"
            f"{_fmt(stats[f'sign_bin_{budget}'])}/"
            f"{_fmt(stats[f'ols_global_{budget}'])}/"
            f"{_fmt(stats[f'ols_bin_{budget}'])}"
        )
        print(
            f"       utility pairAcc/regret/bestHit "
            f"qK={_fmt(stats[f'qK_pair_{budget}'])}/"
            f"{_fmt(stats[f'qK_regret_{budget}'])}/"
            f"{stats[f'qK_hit_{budget}']:.0f} "
            f"olsB={_fmt(stats[f'ols_bin_pair_{budget}'])}/"
            f"{_fmt(stats[f'ols_bin_regret_{budget}'])}/"
            f"{stats[f'ols_bin_hit_{budget}']:.0f}"
        )
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-width", type=int, default=DEFAULT_TRAIN_WIDTH)
    parser.add_argument("--eval-width", type=int)
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
    eval_width = args.eval_width or args.train_width
    if args.train_width < 2 or eval_width < 2:
        parser.error("widths must be at least 2")
    if args.random_graphs_per_family < 2:
        parser.error("--random-graphs-per-family must be at least 2")
    if len(args.input_seeds) < 2:
        parser.error("at least two input seeds are required")
    if not args.budgets or any(budget <= 0 for budget in args.budgets):
        parser.error("--budgets must contain positive integers")
    seeds = tuple(args.input_seeds)
    budgets = tuple(sorted(set(args.budgets)))

    print("Reliability-weighted sparse-coherence calibration")
    print("CALIBRATION ONLY — project held-out inputs remain untouched")
    print("same-width mode uses leave-one-seed-out node-label fitting")
    print("different-width mode transfers one train-width table unchanged")
    print(
        f"train_width={args.train_width} eval_width={eval_width} "
        f"input_seeds={','.join(map(str, seeds))} "
        f"random_graphs_per_family={args.random_graphs_per_family} "
        f"budgets={','.join(map(str, budgets))}"
    )
    print()

    train_groups = _generate_width(
        args.train_width,
        seeds,
        args.random_graphs_per_family,
        budgets,
    )
    eval_groups = (
        train_groups
        if eval_width == args.train_width
        else _generate_width(
            eval_width,
            seeds,
            args.random_graphs_per_family,
            budgets,
        )
    )

    pooled: dict[str, list[float]] = {}
    if eval_width == args.train_width:
        for held_out_index, (seed, rows) in enumerate(zip(seeds, eval_groups, strict=True)):
            training = [
                row
                for group_index, group in enumerate(train_groups)
                if group_index != held_out_index
                for row in group
            ]
            table = _fit_reliability(training)
            stats = _print_fold(seed, rows, table, budgets)
            for key, value in stats.items():
                if value is not None:
                    pooled.setdefault(key, []).append(value)
    else:
        table = _fit_reliability([row for group in train_groups for row in group])
        for seed, rows in zip(seeds, eval_groups, strict=True):
            stats = _print_fold(seed, rows, table, budgets)
            for key, value in stats.items():
                if value is not None:
                    pooled.setdefault(key, []).append(value)

    print()
    print("SEED SUMMARY rho mean/min/max")
    for key, values in pooled.items():
        print(
            f"  {key:<18} mean={mean(values):+.3f} "
            f"min={min(values):+.3f} max={max(values):+.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
