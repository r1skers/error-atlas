"""Calibration-only ideal-subtree shadow diagnostic for wide-range reductions.

This diagnostic asks whether the local FP32 rounding direction at an internal node can be
anticipated without replaying the real FP32 intermediate states.

For each graph node v, build the exact mathematical subtree sum from the stored-FP32 leaves,
without rounding internal nodes:

    T*_v = sum_{i in leaves(v)} x_i.

Then compute a shadow one-shot rounding residual

    delta*_v = RN32(T*_v) - T*_v.

The real candidate-tree residual is

    delta_v = RN32(a_v + b_v) - (a_v + b_v),

where a_v and b_v include all earlier FP32 rounding history. Their difference is

    H_v = (a_v + b_v) - T*_v,

which is exactly the accumulated descendant rounding error entering node v.

This version also checks the lattice-boundary mechanism. Within one binary32 binade, rounding
decision boundaries lie at half-integer ULP coordinates. We therefore measure the distance from
the shadow state to the next half-ULP boundary in the signed direction of H_v, count how many
such boundaries H_v crosses, and compare that with shadow/actual rounding-direction changes.
Binade changes and exact ties are reported separately instead of being forced into the simple
same-spacing rule.

CALIBRATION ONLY. This is a mechanism diagnostic, not held-out evidence and not a frozen
predictor.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from fractions import Fraction
from statistics import mean

from predictor_calibration_inputs import wide_range_random
from predictor_tree_generator import random_contiguous_split_graph, random_pair_merge_graph
from summation_graph_predictor import (
    BinaryReductionGraph,
    predict_fp32_tree_error,
    round_nonnegative_fraction_to_fp32,
)


DEFAULT_WIDTH = 16
DEFAULT_INPUT_SEEDS = (22260824,)
DEFAULT_GRAPH_COUNT = 16
DEFAULT_TRACE_TREES = (0, 1, 5, 7)
TREE_BASE_SEED = 32_000_000
FP32_FRACTION_BITS = 23
FP32_MIN_NORMAL_EXPONENT = -126
FP32_MIN_SUBNORMAL_EXPONENT = -149
HALF = Fraction(1, 2)


@dataclass(frozen=True)
class ShadowNodeDiagnostic:
    node_index: int
    left_index: int
    right_index: int
    shadow_exact_sum: Fraction
    shadow_delta: Fraction
    actual_exact_sum: Fraction
    actual_delta: Fraction
    history_shift: Fraction
    ulp_shadow: Fraction
    shadow_phase: Fraction
    actual_phase_on_shadow_grid: Fraction | None
    shadow_direction: int
    actual_direction: int
    direction_match: bool
    same_binade_after_shift: bool
    directional_boundary_distance_ulp: Fraction | None
    boundary_crossing_count: int | None

    @property
    def history_shift_ulp(self) -> Fraction:
        return self.history_shift / self.ulp_shadow

    @property
    def sign_flipped(self) -> bool:
        return self.shadow_direction != self.actual_direction

    @property
    def simple_crossing_explains_flip(self) -> bool | None:
        """Whether same-binade half-ULP crossings explain a direction change.

        Exact tie endpoints are excluded because ties-to-even needs lattice parity, not only
        the phase coordinate. A zero-vs-nonzero direction change is still considered a change.
        """
        if not self.same_binade_after_shift or self.boundary_crossing_count is None:
            return None
        shadow_tie = self.shadow_phase == HALF
        actual_tie = self.actual_phase_on_shadow_grid == HALF
        if shadow_tie or actual_tie:
            return None
        predicted_flip = self.boundary_crossing_count > 0
        return predicted_flip == self.sign_flipped


@dataclass(frozen=True)
class ShadowTreeDiagnostic:
    graph_family: str
    graph_seed: int
    nodes: tuple[ShadowNodeDiagnostic, ...]

    @property
    def sign_matches(self) -> int:
        return sum(node.direction_match for node in self.nodes)

    @property
    def sign_total(self) -> int:
        return len(self.nodes)

    @property
    def weighted_sign_agreement(self) -> float:
        total = sum((abs(node.actual_delta) for node in self.nodes), start=Fraction(0))
        if total == 0:
            return 1.0
        matched = sum(
            (abs(node.actual_delta) for node in self.nodes if node.direction_match),
            start=Fraction(0),
        )
        return float(matched / total)

    @property
    def mean_abs_history_shift_ulp(self) -> float:
        return mean(float(abs(node.history_shift_ulp)) for node in self.nodes)

    @property
    def max_abs_history_shift_ulp(self) -> float:
        return max(float(abs(node.history_shift_ulp)) for node in self.nodes)

    @property
    def crossing_explanation_counts(self) -> tuple[int, int]:
        eligible = [
            node.simple_crossing_explains_flip
            for node in self.nodes
            if node.simple_crossing_explains_flip is not None
        ]
        return sum(bool(value) for value in eligible), len(eligible)

    @property
    def binade_shift_count(self) -> int:
        return sum(not node.same_binade_after_shift for node in self.nodes)


def _graph(width: int, *, graph_index: int, input_index: int):
    seed = TREE_BASE_SEED + input_index * 10_000 + graph_index
    if graph_index % 2 == 0:
        return "contiguous", seed, random_contiguous_split_graph(width, seed=seed)
    return "pair_merge", seed, random_pair_merge_graph(width, seed=seed)


def _power_of_two(exponent: int) -> Fraction:
    if exponent >= 0:
        return Fraction(2**exponent)
    return Fraction(1, 2 ** (-exponent))


def _floor_log2(value: Fraction) -> int:
    if value <= 0:
        raise ValueError("log2 requires a positive value")
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if value < _power_of_two(exponent):
        exponent -= 1
    return exponent


def _ulp32_at_positive(value: Fraction) -> Fraction:
    if value <= 0:
        raise ValueError("ULP is defined here only for positive values")
    exponent = _floor_log2(value)
    if exponent < FP32_MIN_NORMAL_EXPONENT:
        return _power_of_two(FP32_MIN_SUBNORMAL_EXPONENT)
    return _power_of_two(exponent - FP32_FRACTION_BITS)


def _sign(value: Fraction) -> int:
    return (value > 0) - (value < 0)


def _phase_on_local_grid(value: Fraction, ulp: Fraction) -> Fraction:
    scaled = value / ulp
    lower = scaled.numerator // scaled.denominator
    return scaled - lower


def _directional_boundary_distance(phase: Fraction, shift_ulp: Fraction) -> Fraction | None:
    """Distance in ULPs to the next half-integer boundary along the shift direction."""
    if shift_ulp == 0:
        return None
    if shift_ulp > 0:
        return HALF - phase if phase < HALF else Fraction(3, 2) - phase
    return phase - HALF if phase > HALF else phase + HALF


def _count_half_integer_boundaries(start: Fraction, stop: Fraction) -> int:
    """Count half-integer lattice boundaries strictly between start and stop.

    Endpoints that are exact half-integers are excluded and handled as tie cases. Counting is
    done on exact rationals by mapping k+1/2 boundaries to odd integers under multiplication by 2.
    """
    if start == stop:
        return 0
    lo = min(start, stop) * 2
    hi = max(start, stop) * 2

    first_integer = lo.numerator // lo.denominator + 1
    if Fraction(first_integer) <= lo:
        first_integer += 1
    last_integer = math.ceil(hi) - 1
    if first_integer > last_integer:
        return 0

    if first_integer % 2 == 0:
        first_integer += 1
    if last_integer % 2 == 0:
        last_integer -= 1
    if first_integer > last_integer:
        return 0
    return (last_integer - first_integer) // 2 + 1


def diagnose_shadow_tree(
    values: tuple[Fraction, ...],
    graph: BinaryReductionGraph,
    *,
    graph_family: str,
    graph_seed: int,
) -> ShadowTreeDiagnostic:
    """Compare ideal-subtree one-shot rounding with actual recursive-tree rounding."""
    actual = predict_fp32_tree_error(values, graph)
    shadow_states = list(values)
    records: list[ShadowNodeDiagnostic] = []

    for node, actual_node in zip(graph.nodes, actual.node_predictions, strict=True):
        shadow_exact = shadow_states[node.left] + shadow_states[node.right]
        shadow_states.append(shadow_exact)

        shadow_rounded = round_nonnegative_fraction_to_fp32(shadow_exact).value
        shadow_delta = shadow_rounded - shadow_exact
        history_shift = actual_node.exact_addend_sum - shadow_exact
        ulp_shadow = _ulp32_at_positive(shadow_exact)
        shadow_phase = _phase_on_local_grid(shadow_exact, ulp_shadow)
        same_binade = _floor_log2(shadow_exact) == _floor_log2(actual_node.exact_addend_sum)

        actual_phase: Fraction | None = None
        boundary_distance: Fraction | None = None
        crossing_count: int | None = None
        shift_ulp = history_shift / ulp_shadow
        if same_binade:
            actual_phase = _phase_on_local_grid(actual_node.exact_addend_sum, ulp_shadow)
            boundary_distance = _directional_boundary_distance(shadow_phase, shift_ulp)
            crossing_count = _count_half_integer_boundaries(
                shadow_exact / ulp_shadow,
                actual_node.exact_addend_sum / ulp_shadow,
            )

        records.append(
            ShadowNodeDiagnostic(
                node_index=actual_node.node_index,
                left_index=node.left,
                right_index=node.right,
                shadow_exact_sum=shadow_exact,
                shadow_delta=shadow_delta,
                actual_exact_sum=actual_node.exact_addend_sum,
                actual_delta=actual_node.local_rounding_error,
                history_shift=history_shift,
                ulp_shadow=ulp_shadow,
                shadow_phase=shadow_phase,
                actual_phase_on_shadow_grid=actual_phase,
                shadow_direction=_sign(shadow_delta),
                actual_direction=_sign(actual_node.local_rounding_error),
                direction_match=_sign(shadow_delta) == _sign(actual_node.local_rounding_error),
                same_binade_after_shift=same_binade,
                directional_boundary_distance_ulp=boundary_distance,
                boundary_crossing_count=crossing_count,
            )
        )

    if shadow_states[graph.root] != sum(values, start=Fraction(0)):
        raise AssertionError("shadow root must equal the exact input sum")

    return ShadowTreeDiagnostic(
        graph_family=graph_family,
        graph_seed=graph_seed,
        nodes=tuple(records),
    )


def _symbol(direction: int) -> str:
    return {1: "+", 0: "0", -1: "-"}[direction]


def _sci(value: Fraction) -> str:
    return f"{float(value):+.9e}"


def _print_tree_summary(
    input_seed: int,
    tree_index: int,
    diagnostic: ShadowTreeDiagnostic,
) -> None:
    crossing_matches, crossing_total = diagnostic.crossing_explanation_counts
    crossing_rate = "n/a" if crossing_total == 0 else f"{crossing_matches / crossing_total:.3f}"
    print(
        "TREE "
        f"input_seed={input_seed} tree={tree_index:02d} "
        f"family={diagnostic.graph_family} graph_seed={diagnostic.graph_seed} "
        f"shadow_sign_match={diagnostic.sign_matches}/{diagnostic.sign_total} "
        f"rate={diagnostic.sign_matches / diagnostic.sign_total:.3f} "
        f"weighted_match={diagnostic.weighted_sign_agreement:.3f} "
        f"mean|history_shift|/ulp={diagnostic.mean_abs_history_shift_ulp:.3f} "
        f"max={diagnostic.max_abs_history_shift_ulp:.3f}"
    )
    print(
        "  boundary "
        f"crossing_explains_flip={crossing_matches}/{crossing_total} rate={crossing_rate} "
        f"binade_shifts={diagnostic.binade_shift_count}"
    )


def _print_trace(diagnostic: ShadowTreeDiagnostic) -> None:
    total_actual_abs = sum(
        (abs(node.actual_delta) for node in diagnostic.nodes), start=Fraction(0)
    )
    print(
        "  shadow_trace columns: node left right actual_delta shadow_delta actual_share "
        "shadow_phase history_shift_ulp boundary_dist crossing_count dirs match explained"
    )
    for node in diagnostic.nodes:
        share = 0.0 if total_actual_abs == 0 else float(abs(node.actual_delta) / total_actual_abs)
        dirs = f"{_symbol(node.shadow_direction)}/{_symbol(node.actual_direction)}"
        boundary_dist = (
            "x"
            if node.directional_boundary_distance_ulp is None
            else f"{float(node.directional_boundary_distance_ulp):.6f}"
        )
        crossings = "x" if node.boundary_crossing_count is None else str(node.boundary_crossing_count)
        explained = node.simple_crossing_explains_flip
        explained_text = "x" if explained is None else str(int(explained))
        print(
            "  NODE "
            f"{node.node_index:>3d} {node.left_index:>3d} {node.right_index:>3d} "
            f"{_sci(node.actual_delta)} {_sci(node.shadow_delta)} {share:.4f} "
            f"{float(node.shadow_phase):.6f} {float(node.history_shift_ulp):+.6f} "
            f"{boundary_dist:>8s} {crossings:>2s} {dirs:>3s} "
            f"{int(node.direction_match)} {explained_text}"
        )


def _print_family_summary(family: str, trees: list[ShadowTreeDiagnostic]) -> None:
    if not trees:
        return
    matches = sum(tree.sign_matches for tree in trees)
    total = sum(tree.sign_total for tree in trees)
    actual_mass = sum(
        (abs(node.actual_delta) for tree in trees for node in tree.nodes),
        start=Fraction(0),
    )
    matched_mass = sum(
        (
            abs(node.actual_delta)
            for tree in trees
            for node in tree.nodes
            if node.direction_match
        ),
        start=Fraction(0),
    )
    weighted = 1.0 if actual_mass == 0 else float(matched_mass / actual_mass)
    crossing_pairs = [tree.crossing_explanation_counts for tree in trees]
    crossing_matches = sum(pair[0] for pair in crossing_pairs)
    crossing_total = sum(pair[1] for pair in crossing_pairs)
    crossing_rate = "n/a" if crossing_total == 0 else f"{crossing_matches / crossing_total:.6f}"
    binade_shifts = sum(tree.binade_shift_count for tree in trees)
    print(
        f"  {family:<10} trees={len(trees):2d} "
        f"shadow_sign_match={matches}/{total}({matches / total:.6f}) "
        f"weighted_match={weighted:.6f} "
        f"crossing_explains_flip={crossing_matches}/{crossing_total}({crossing_rate}) "
        f"binade_shifts={binade_shifts} "
        f"mean_tree_history_shift_ulp={mean(tree.mean_abs_history_shift_ulp for tree in trees):.6f} "
        f"mean_tree_max_shift_ulp={mean(tree.max_abs_history_shift_ulp for tree in trees):.6f}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, choices=(8, 16), default=DEFAULT_WIDTH)
    parser.add_argument("--graphs", type=int, default=DEFAULT_GRAPH_COUNT)
    parser.add_argument(
        "--input-seeds", type=int, nargs="+", default=list(DEFAULT_INPUT_SEEDS)
    )
    parser.add_argument(
        "--trace-trees",
        type=int,
        nargs="+",
        default=list(DEFAULT_TRACE_TREES),
        metavar="INDEX",
    )
    args = parser.parse_args()
    if args.graphs <= 0:
        parser.error("--graphs must be positive")
    bad = [index for index in args.trace_trees if not 0 <= index < args.graphs]
    if bad:
        parser.error(f"--trace-trees indices must be in [0,{args.graphs - 1}]: {bad}")
    return args


def main() -> int:
    args = _parse_args()
    trace_trees = set(args.trace_trees)

    print("Wide-range ideal-subtree shadow phase diagnostic")
    print("CALIBRATION ONLY — stored leaves + known graph define the shadow state")
    print("MECHANISM BRIDGE — shadow residuals do not replay real FP32 intermediates")
    print(
        f"width={args.width} graphs_per_input={args.graphs} "
        f"input_seeds={','.join(str(seed) for seed in args.input_seeds)} "
        f"trace_trees={','.join(f'{index:02d}' for index in args.trace_trees)}"
    )
    print()

    for input_index, input_seed in enumerate(args.input_seeds):
        generated = wide_range_random(args.width, seed=input_seed)
        trees: list[ShadowTreeDiagnostic] = []
        print(
            f"INPUT seed={generated.seed} family={generated.family} width={len(generated.values)}"
        )

        for graph_index in range(args.graphs):
            graph_family, graph_seed, graph = _graph(
                len(generated.values), graph_index=graph_index, input_index=input_index
            )
            diagnostic = diagnose_shadow_tree(
                generated.values,
                graph,
                graph_family=graph_family,
                graph_seed=graph_seed,
            )
            trees.append(diagnostic)
            _print_tree_summary(input_seed, graph_index, diagnostic)
            if graph_index in trace_trees:
                _print_trace(diagnostic)

        print("INPUT SUMMARY")
        _print_family_summary("all", trees)
        _print_family_summary(
            "contiguous", [tree for tree in trees if tree.graph_family == "contiguous"]
        )
        _print_family_summary(
            "pair_merge", [tree for tree in trees if tree.graph_family == "pair_merge"]
        )
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
