"""Calibrate a discrete ancestor-chain phase score without Gaussian phase averaging.

The macro baseline remains Q/12.  For each selected root-band node v, descendant local-rounding
energy supplies a history scale sigma_H(v).  Three deterministic sigma states

    z in {-sqrt(3), 0, +sqrt(3)},  p = {1/6, 2/3, 1/6}

have mean zero and unit variance.  Each state is passed through the actual RN-even phase map at v,
so boundary crossings and residual signs remain discrete rather than being averaged before rounding.

Only selected ancestor pairs receive a coherence correction.  Two exploratory kernels are compared:

* ``corr4``: the measured gap-1..4 history correlations 0.476, 0.267, 0.168, 0.055;
* ``geometric``: 0.476**gap, corresponding to a copy-or-resample three-state Markov chain.

For centered state residuals d_v(z), the pair contribution is

    2 * kernel(gap) * sum_z p(z) d_u(z) d_v(z) / U_root**2.

``zero`` centers history states at zero; ``shadow`` centers them at the previously diagnosed
deterministic binary64 shadow history.  Descendant energy is used only as a scale.  The model never
uses an oracle history and never recursively executes RN32 candidate outputs.  Exact execution enters
after scoring solely as a calibration ranking target.  Held-out inputs remain untouched.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from fractions import Fraction
from statistics import mean

from predictor_calibration_inputs import wide_range_random
from predictor_ranking_smoke import _spearman
from predictor_two_stage_cheap_score_calibration import (
    DEFAULT_BUDGETS,
    DEFAULT_INPUT_SEEDS,
    DEFAULT_RANDOM_GRAPHS_PER_FAMILY,
    FAMILIES,
    _graphs,
    _predictor_trace,
)
from predictor_ulp_energy_convergence_diagnostic import (
    HISTORY_CORRELATION_KERNEL,
    _parent_map,
)
from summation_graph_predictor import BinaryReductionGraph, predict_fp32_tree_error


DEFAULT_WIDTH = 256
SIGMA_STATES = (-math.sqrt(3.0), 0.0, math.sqrt(3.0))
SIGMA_WEIGHTS = (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0)
GAP1_CORRELATION = HISTORY_CORRELATION_KERNEL[1]


@dataclass(frozen=True)
class DiscretePhaseScores:
    root_ulp: Fraction
    full_q: float
    q_budget: dict[int, float]
    prior_phase: dict[int, float]
    corr4_zero: dict[int, float]
    corr4_shadow: dict[int, float]
    geometric_shadow: dict[int, float]


@dataclass(frozen=True)
class TreeRow:
    family: str
    target: float
    scores: DiscretePhaseScores


def _fractional_ulp_coordinate(value: Fraction, ulp: Fraction) -> float:
    coordinate = value / ulp
    floor = coordinate.numerator // coordinate.denominator
    return float(coordinate - floor)


def _centered_state_residuals(
    exact_sum: Fraction,
    ulp: Fraction,
    history_center: float,
    history_sigma: float,
) -> tuple[float, ...]:
    phase = _fractional_ulp_coordinate(exact_sum, ulp)
    ulp_float = float(ulp)
    residuals: list[float] = []
    for state in SIGMA_STATES:
        shift_ulp = (history_center + state * history_sigma) / ulp_float
        shifted_phase = phase + shift_ulp
        residuals.append(ulp_float * (round(shifted_phase) - shifted_phase))
    state_mean = sum(
        weight * residual
        for weight, residual in zip(SIGMA_WEIGHTS, residuals, strict=True)
    )
    return tuple(residual - state_mean for residual in residuals)


def _ancestor_gap(
    first: int,
    second: int,
    parent: list[int | None],
) -> int | None:
    def upward(descendant: int, ancestor: int) -> int | None:
        gap = 0
        current: int | None = descendant
        while current is not None:
            if current == ancestor:
                return gap
            current = parent[current]
            gap += 1
        return None

    forward = upward(first, second)
    if forward is not None:
        return forward
    return upward(second, first)


def _kernel_corr4(gap: int) -> float:
    return HISTORY_CORRELATION_KERNEL.get(gap, 0.0)


def _kernel_geometric(gap: int) -> float:
    return GAP1_CORRELATION**gap


def _coherence_score(
    selected: tuple[int, ...],
    state_residuals: dict[int, tuple[float, ...]],
    parent: list[int | None],
    full_q: float,
    root_ulp: Fraction,
    kernel,
) -> float:
    correction = 0.0
    for left_position, left in enumerate(selected):
        for right in selected[left_position + 1 :]:
            gap = _ancestor_gap(left, right, parent)
            if gap is None or gap == 0:
                continue
            correlation = kernel(gap)
            if correlation == 0.0:
                continue
            covariance = sum(
                weight * left_value * right_value
                for weight, left_value, right_value in zip(
                    SIGMA_WEIGHTS,
                    state_residuals[left],
                    state_residuals[right],
                    strict=True,
                )
            )
            correction += 2.0 * correlation * covariance
    return max(0.0, full_q / 12.0 + correction / float(root_ulp * root_ulp))


def _predictor_scores(
    values: tuple[Fraction, ...],
    graph: BinaryReductionGraph,
    budgets: tuple[int, ...],
) -> DiscretePhaseScores:
    trace = _predictor_trace(values, graph, budgets)
    parent = _parent_map(graph)

    accumulated_variance = [0.0 for _ in values]
    trajectory_error = [0.0 for _ in values]
    history_sigma: dict[int, float] = {}
    shadow_history: dict[int, float] = {}
    for offset, node in enumerate(graph.nodes):
        index = graph.leaf_count + offset
        variance_before = (
            accumulated_variance[node.left] + accumulated_variance[node.right]
        )
        history_sigma[index] = math.sqrt(variance_before)
        node_variance = float(trace.node_ulp[index]) ** 2 / 12.0
        accumulated_variance.append(variance_before + node_variance)

        history = trajectory_error[node.left] + trajectory_error[node.right]
        shadow_history[index] = history
        trajectory_error.append(history + trace.trajectory_delta[index])

    zero_states = {
        index: _centered_state_residuals(
            trace.exact_subtree[index],
            trace.node_ulp[index],
            0.0,
            history_sigma[index],
        )
        for index in trace.selected_order
    }
    shadow_states = {
        index: _centered_state_residuals(
            trace.exact_subtree[index],
            trace.node_ulp[index],
            shadow_history[index],
            history_sigma[index],
        )
        for index in trace.selected_order
    }

    corr4_zero: dict[int, float] = {}
    corr4_shadow: dict[int, float] = {}
    geometric_shadow: dict[int, float] = {}
    for budget in sorted(set(budgets)):
        selected = trace.selected_order[: min(budget, len(trace.selected_order))]
        corr4_zero[budget] = _coherence_score(
            selected,
            zero_states,
            parent,
            trace.full_q,
            trace.root_ulp,
            _kernel_corr4,
        )
        corr4_shadow[budget] = _coherence_score(
            selected,
            shadow_states,
            parent,
            trace.full_q,
            trace.root_ulp,
            _kernel_corr4,
        )
        geometric_shadow[budget] = _coherence_score(
            selected,
            shadow_states,
            parent,
            trace.full_q,
            trace.root_ulp,
            _kernel_geometric,
        )

    return DiscretePhaseScores(
        root_ulp=trace.root_ulp,
        full_q=trace.full_q,
        q_budget={budget: trace.budget[budget].q_budget for budget in budgets},
        prior_phase={
            budget: trace.budget[budget].coherence_phase for budget in budgets
        },
        corr4_zero=corr4_zero,
        corr4_shadow=corr4_shadow,
        geometric_shadow=geometric_shadow,
    )


def _report(
    label: str,
    rows: list[TreeRow],
    budgets: tuple[int, ...],
) -> dict[str, float | None]:
    target = [row.target for row in rows]
    out: dict[str, float | None] = {
        "full_q": _spearman([row.scores.full_q for row in rows], target),
    }
    print(
        f"  {label:<10} n={len(rows):3d} target_unique={len(set(target)):3d} "
        f"rho_fullQ={_fmt(out['full_q'])}"
    )
    for budget in budgets:
        policies = {
            "qK": [row.scores.q_budget[budget] for row in rows],
            "prior": [row.scores.prior_phase[budget] for row in rows],
            "corr4_zero": [row.scores.corr4_zero[budget] for row in rows],
            "corr4_shadow": [row.scores.corr4_shadow[budget] for row in rows],
            "geometric_shadow": [
                row.scores.geometric_shadow[budget] for row in rows
            ],
        }
        rhos = {name: _spearman(values, target) for name, values in policies.items()}
        for name, rho in rhos.items():
            out[f"{name}_{budget}"] = rho
        print(
            f"    K={budget:<2d} rho qK/prior/c4zero/c4shadow/geomshadow="
            f"{_fmt(rhos['qK'])}/{_fmt(rhos['prior'])}/"
            f"{_fmt(rhos['corr4_zero'])}/{_fmt(rhos['corr4_shadow'])}/"
            f"{_fmt(rhos['geometric_shadow'])}"
        )
    return out


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.3f}"


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

    print("Discrete ancestor-chain phase-score calibration")
    print("CALIBRATION ONLY — held-out inputs remain untouched")
    print("three phase states preserve RN-cell transitions; oracle is target only")
    print(
        f"width={args.width} "
        f"input_seeds={','.join(map(str, args.input_seeds))} "
        f"random_graphs_per_family={args.random_graphs_per_family} "
        f"budgets={','.join(map(str, budgets))}"
    )
    print()

    pooled: dict[str, list[float]] = {}
    for input_index, seed in enumerate(args.input_seeds):
        values = wide_range_random(args.width, seed=seed).values
        rows: list[TreeRow] = []
        for family, graph in _graphs(
            args.width,
            input_index,
            args.random_graphs_per_family,
        ):
            scores = _predictor_scores(values, graph, budgets)
            oracle = predict_fp32_tree_error(values, graph)
            target = float(oracle.signed_error / scores.root_ulp) ** 2
            rows.append(TreeRow(family=family, target=target, scores=scores))

        print(f"INPUT seed={seed} family=wide_range_random width={args.width}")
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
            f"  {key:<22} mean={mean(values):+.3f} "
            f"min={min(values):+.3f} max={max(values):+.3f}"
        )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
