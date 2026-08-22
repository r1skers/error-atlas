"""Score-only implementation of the frozen fixed-K8/B3 tree selector.

The v2 confirmation runner intentionally retained exact-oracle instrumentation so that every
selection could be evaluated.  That runner is evidence-generation code, not an inference path.
This module implements only the frozen selector:

* score every candidate with connected-root-band ``Q_8 / 12``;
* shortlist the four lowest-Q candidates, with graph index as the stable tie-breaker;
* rerank the shortlist with the frozen 19-feature innovation model and a width-three cell beam.

Inputs are nonnegative finite binary32 bit patterns.  Exact subtree sums are accumulated as Python
integers on the binary32 ``2**-149`` lattice.  This avoids ``Fraction`` while preserving the frozen
score's exact phase and ULP semantics.  No candidate FP32 trajectory, forward-error target, or
oracle function is imported or executed.

The implementation is a faithful research prototype, not a kernel-speed claim.  Macro scoring is
still O(number_of_candidates * input_width), and the four beam traces are O(input_width) each.
"""
from __future__ import annotations

import heapq
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np


class AdditionNodeLike(Protocol):
    left: int
    right: int


class BinaryReductionGraphLike(Protocol):
    leaf_count: int
    nodes: Sequence[AdditionNodeLike]
    root: int


FP32_MIN_SUBNORMAL_EXPONENT = -149
FP32_MIN_NORMAL_EXPONENT = -126
FP32_FRACTION_BITS = 23
FP32_MAX_NORMAL_EXPONENT = 127
FP32_MAX_FINITE_BITS = 0x7F7FFFFF
SHIFT_STATES = (-2, -1, 0, 1, 2)
FEATURE_DIMENSION = 19

FROZEN_ROOT_BAND_BUDGET = 8
FROZEN_FEATURE_BUDGET = 32
FROZEN_SHORTLIST_SIZE = 4
FROZEN_BEAM_WIDTH = 3

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = (
    HERE
    / "results"
    / "wide_range_fixed_k8_beam_v2"
    / "heldout"
    / "calibration_model.json"
)

_PACK_U32 = struct.Struct(">I")
_UNPACK_F32 = struct.Struct(">f")


@dataclass(frozen=True)
class InnovationModel:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    weights: np.ndarray

    @classmethod
    def from_record(cls, record: dict) -> "InnovationModel":
        if record.get("label") != "innovation_shift":
            raise ValueError("model label must be innovation_shift")
        if record.get("feature_dimension") != FEATURE_DIMENSION:
            raise ValueError("model feature dimension does not match the frozen selector")
        feature_mean = np.asarray(record["feature_mean"], dtype=float)
        feature_scale = np.asarray(record["feature_scale"], dtype=float)
        weights = np.asarray(record["weights"], dtype=float)
        if feature_mean.shape != (FEATURE_DIMENSION,):
            raise ValueError("model feature_mean has the wrong shape")
        if feature_scale.shape != (FEATURE_DIMENSION,):
            raise ValueError("model feature_scale has the wrong shape")
        if weights.shape != (FEATURE_DIMENSION + 1, len(SHIFT_STATES)):
            raise ValueError("model weights have the wrong shape")
        if not np.all(np.isfinite(feature_mean)):
            raise ValueError("model feature_mean must be finite")
        if not np.all(np.isfinite(feature_scale)) or np.any(feature_scale <= 0.0):
            raise ValueError("model feature_scale must be positive and finite")
        if not np.all(np.isfinite(weights)):
            raise ValueError("model weights must be finite")
        return cls(feature_mean, feature_scale, weights)

    @classmethod
    def from_json(cls, path: Path | str = DEFAULT_MODEL_PATH) -> "InnovationModel":
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_record(json.load(handle))

    def probabilities(self, features: Sequence[float]) -> tuple[float, ...]:
        matrix = np.asarray([features], dtype=float)
        if matrix.shape != (1, FEATURE_DIMENSION):
            raise ValueError(f"expected {FEATURE_DIMENSION} features")
        standardized = (matrix - self.feature_mean) / self.feature_scale
        design = np.column_stack((np.ones(1), standardized))
        logits = design @ self.weights
        shifted = logits - logits.max(axis=1, keepdims=True)
        exponential = np.exp(np.clip(shifted, -80.0, 0.0))
        probability = exponential / exponential.sum(axis=1, keepdims=True)
        return tuple(float(value) for value in probability[0])


@dataclass(frozen=True)
class MacroTrace:
    graph: BinaryReductionGraphLike
    leaf_units: tuple[int, ...]
    exact_internal_units: tuple[int, ...]
    subtree_internal_leaves: tuple[int, ...]
    node_ulp_exponents: tuple[int, ...]
    selected_order: tuple[int, ...]
    q_score: float

    @property
    def root_units(self) -> int:
        return self.exact_internal_units[-1]

    @property
    def root_ulp_exponent(self) -> int:
        return self.node_ulp_exponents[-1]

    def exact_units(self, index: int) -> int:
        if index < self.graph.leaf_count:
            return self.leaf_units[index]
        return self.exact_internal_units[index - self.graph.leaf_count]

    def subtree_leaves(self, index: int) -> int:
        if index < self.graph.leaf_count:
            return 1
        return self.subtree_internal_leaves[index - self.graph.leaf_count]

    def ulp_exponent(self, index: int) -> int:
        if index < self.graph.leaf_count:
            raise ValueError("leaf nodes do not have score-side ULP metadata")
        return self.node_ulp_exponents[index - self.graph.leaf_count]


@dataclass(frozen=True)
class BeamState:
    bits: int
    probability: float


@dataclass(frozen=True)
class SelectionResult:
    selected_index: int
    q_selected_index: int
    shortlist_indices: tuple[int, ...]
    q_scores: tuple[float, ...]
    beam_scores: tuple[float | None, ...]


def _validate_fp32_bits(bits: int) -> None:
    if isinstance(bits, bool) or not isinstance(bits, int):
        raise TypeError("every leaf bit pattern must be an integer")
    if not 0 <= bits <= FP32_MAX_FINITE_BITS:
        raise ValueError("leaves must be nonnegative finite binary32 values")


def _bits_to_units(bits: int) -> int:
    """Return a nonnegative binary32 value in integer multiples of 2**-149."""
    _validate_fp32_bits(bits)
    exponent_bits = (bits >> FP32_FRACTION_BITS) & 0xFF
    fraction_bits = bits & ((1 << FP32_FRACTION_BITS) - 1)
    if exponent_bits == 0:
        return fraction_bits
    significand = (1 << FP32_FRACTION_BITS) + fraction_bits
    return significand << (exponent_bits - 1)


def _bits_to_float(bits: int) -> float:
    _validate_fp32_bits(bits)
    return _UNPACK_F32.unpack(_PACK_U32.pack(bits))[0]


def _round_integer_ties_even(numerator: int, denominator_shift: int) -> int:
    """Round numerator / 2**denominator_shift to an integer, ties to even."""
    if numerator < 0 or denominator_shift < 0:
        raise ValueError("rounding arguments must be nonnegative")
    if denominator_shift == 0:
        return numerator
    quotient, remainder = divmod(numerator, 1 << denominator_shift)
    midpoint = 1 << (denominator_shift - 1)
    if remainder > midpoint or (remainder == midpoint and quotient & 1):
        quotient += 1
    return quotient


def _round_dyadic_to_fp32_bits(numerator: int, exponent: int) -> int:
    """Round nonnegative ``numerator * 2**exponent`` to finite binary32."""
    if numerator < 0:
        raise ValueError("the frozen selector supports only nonnegative values")
    if numerator == 0:
        return 0

    value_exponent = numerator.bit_length() - 1 + exponent
    quantum_exponent = (
        FP32_MIN_SUBNORMAL_EXPONENT
        if value_exponent < FP32_MIN_NORMAL_EXPONENT
        else value_exponent - FP32_FRACTION_BITS
    )
    shift = exponent - quantum_exponent
    if shift >= 0:
        significand = numerator << shift
    else:
        significand = _round_integer_ties_even(numerator, -shift)

    if value_exponent < FP32_MIN_NORMAL_EXPONENT:
        if significand == 0:
            return 0
        if significand < 1 << FP32_FRACTION_BITS:
            return significand
        value_exponent = FP32_MIN_NORMAL_EXPONENT
        significand = 1 << FP32_FRACTION_BITS
    elif significand == 1 << (FP32_FRACTION_BITS + 1):
        value_exponent += 1
        significand = 1 << FP32_FRACTION_BITS

    if value_exponent > FP32_MAX_NORMAL_EXPONENT:
        raise OverflowError("rounded selector state exceeds finite binary32")
    exponent_bits = value_exponent + 127
    fraction_bits = significand - (1 << FP32_FRACTION_BITS)
    bits = (exponent_bits << FP32_FRACTION_BITS) | fraction_bits
    if not 0 <= bits <= FP32_MAX_FINITE_BITS:
        raise OverflowError("rounded selector state exceeds finite binary32")
    return bits


def _round_units_to_fp32_bits(units: int) -> int:
    return _round_dyadic_to_fp32_bits(units, FP32_MIN_SUBNORMAL_EXPONENT)


def _float_dyadic(value: float) -> tuple[int, int]:
    if not math.isfinite(value):
        raise ValueError("shadow state must be finite")
    numerator, denominator = value.as_integer_ratio()
    return numerator, -(denominator.bit_length() - 1)


def _add_dyadics(
    left_numerator: int,
    left_exponent: int,
    right_numerator: int,
    right_exponent: int,
) -> tuple[int, int]:
    exponent = min(left_exponent, right_exponent)
    numerator = (
        (left_numerator << (left_exponent - exponent))
        + (right_numerator << (right_exponent - exponent))
    )
    return numerator, exponent


def _round_exact_plus_float_to_fp32_bits(exact_units: int, error: float) -> int:
    error_numerator, error_exponent = _float_dyadic(error)
    numerator, exponent = _add_dyadics(
        exact_units,
        FP32_MIN_SUBNORMAL_EXPONENT,
        error_numerator,
        error_exponent,
    )
    return _round_dyadic_to_fp32_bits(numerator, exponent)


def _ulp_exponent_from_units(units: int) -> int:
    if units <= 0:
        raise ValueError("the frozen score requires positive subtree sums")
    value_exponent = units.bit_length() - 1 + FP32_MIN_SUBNORMAL_EXPONENT
    if value_exponent < FP32_MIN_NORMAL_EXPONENT:
        return FP32_MIN_SUBNORMAL_EXPONENT
    return value_exponent - FP32_FRACTION_BITS


def _float_from_units(units: int) -> float:
    return math.ldexp(float(units), FP32_MIN_SUBNORMAL_EXPONENT)


def _sign(value: int | float) -> int:
    return (value > 0) - (value < 0)


def _root_band_internal_order(
    graph: BinaryReductionGraphLike,
    subtree_internal_leaves: Sequence[int],
    max_visits: int,
) -> tuple[int, ...]:
    def size(index: int) -> int:
        if index < graph.leaf_count:
            return 1
        return subtree_internal_leaves[index - graph.leaf_count]

    frontier = [(-size(graph.root), graph.root)]
    order: list[int] = []
    while frontier and len(order) < min(max_visits, len(graph.nodes)):
        _, index = heapq.heappop(frontier)
        order.append(index)
        node = graph.nodes[index - graph.leaf_count]
        for child in (node.left, node.right):
            if child >= graph.leaf_count:
                heapq.heappush(frontier, (-size(child), child))
    return tuple(order)


def _macro_trace(
    leaf_units: tuple[int, ...],
    graph: BinaryReductionGraphLike,
) -> MacroTrace:
    if len(leaf_units) != graph.leaf_count:
        raise ValueError("leaf count does not match candidate graph")
    exact_values = [*leaf_units]
    subtree_leaves = [1] * graph.leaf_count
    node_ulp_exponents: list[int] = []
    for node in graph.nodes:
        exact_sum = exact_values[node.left] + exact_values[node.right]
        exact_values.append(exact_sum)
        subtree_leaves.append(subtree_leaves[node.left] + subtree_leaves[node.right])
        node_ulp_exponents.append(_ulp_exponent_from_units(exact_sum))

    selected_order = _root_band_internal_order(
        graph,
        subtree_leaves[graph.leaf_count :],
        FROZEN_FEATURE_BUDGET,
    )
    fixed_order = selected_order[: min(FROZEN_ROOT_BAND_BUDGET, len(selected_order))]
    root_ulp_exponent = node_ulp_exponents[-1]
    q_budget = sum(
        math.ldexp(
            1.0,
            2
            * (
                node_ulp_exponents[index - graph.leaf_count]
                - root_ulp_exponent
            ),
        )
        for index in fixed_order
    )
    return MacroTrace(
        graph=graph,
        leaf_units=leaf_units,
        exact_internal_units=tuple(exact_values[graph.leaf_count :]),
        subtree_internal_leaves=tuple(subtree_leaves[graph.leaf_count :]),
        node_ulp_exponents=tuple(node_ulp_exponents),
        selected_order=selected_order,
        q_score=q_budget / 12.0,
    )


def _fp32_ulp(value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("score requires positive finite shadow sums")
    exponent = math.floor(math.log2(value))
    if exponent < FP32_MIN_NORMAL_EXPONENT:
        return math.ldexp(1.0, FP32_MIN_SUBNORMAL_EXPONENT)
    return math.ldexp(1.0, exponent - FP32_FRACTION_BITS)


def _boundary_cross_count(phase: float, shift: float) -> int:
    start = phase
    stop = phase + shift
    low, high = (start, stop) if start <= stop else (stop, start)
    first = math.floor(low - 0.5) + 1
    last = math.floor(high - 0.5)
    return max(0, last - first + 1)


def _clamp(value: float, limit: float = 8.0) -> float:
    return max(-limit, min(limit, value))


def _fractional_ulp_coordinate(units: int, ulp_exponent: int) -> float:
    quantum_units = 1 << (ulp_exponent - FP32_MIN_SUBNORMAL_EXPONENT)
    return (units % quantum_units) / quantum_units


def _parent_map(graph: BinaryReductionGraphLike) -> list[int | None]:
    parent: list[int | None] = [None] * (graph.leaf_count + len(graph.nodes))
    for offset, node in enumerate(graph.nodes):
        index = graph.leaf_count + offset
        parent[node.left] = index
        parent[node.right] = index
    return parent


def _nearest_selected_ancestor(
    node: int,
    selected: set[int],
    parent: Sequence[int | None],
) -> tuple[int, int] | None:
    gap = 0
    current = parent[node]
    while current is not None:
        gap += 1
        if current in selected:
            return current, gap
        current = parent[current]
    return None


def _beam_features_and_shadow_bits(
    trace: MacroTrace,
) -> tuple[dict[int, tuple[float, ...]], dict[int, int]]:
    graph = trace.graph
    leaf_count = graph.leaf_count
    selected_order = trace.selected_order
    selected = set(selected_order)
    selected_rank = {node: rank for rank, node in enumerate(selected_order)}
    fixed_order = selected_order[: min(FROZEN_ROOT_BAND_BUDGET, len(selected_order))]
    parent = _parent_map(graph)

    shadow_subtree = [_float_from_units(value) for value in trace.leaf_units]
    shadow_output_error = [0.0] * leaf_count
    shadow_history: dict[int, float] = {}
    trajectory_delta: dict[int, float] = {}
    predicted_cross: dict[int, bool] = {}
    predicted_phase: dict[int, bool] = {}
    delta0_sign: dict[int, int] = {}
    depth = [0] * leaf_count
    shadow_bits: dict[int, int] = {}

    for offset, node in enumerate(graph.nodes):
        index = leaf_count + offset
        exact_shadow_sum = shadow_subtree[node.left] + shadow_subtree[node.right]
        ulp_float = _fp32_ulp(exact_shadow_sum)
        phase = exact_shadow_sum / ulp_float
        phase -= int(phase // 1)
        history = shadow_output_error[node.left] + shadow_output_error[node.right]
        shifted_phase = phase + history / ulp_float
        output_error = ulp_float * (round(shifted_phase) - phase)
        innovation = output_error - history

        exact_units = trace.exact_internal_units[offset]
        rounded0_units = _bits_to_units(_round_units_to_fp32_bits(exact_units))
        sign0 = _sign(rounded0_units - exact_units)

        shadow_subtree.append(exact_shadow_sum)
        shadow_output_error.append(output_error)
        shadow_history[index] = history
        trajectory_delta[index] = innovation
        predicted_cross[index] = _boundary_cross_count(
            phase,
            history / ulp_float,
        ) > 0
        predicted_phase[index] = _sign(innovation) != sign0
        delta0_sign[index] = sign0
        depth.append(max(depth[node.left], depth[node.right]) + 1)
        shadow_bits[index] = _round_exact_plus_float_to_fp32_bits(
            exact_units,
            output_error,
        )

    maximum_depth = max(depth) or 1
    rank_denominator = max(1, len(selected_order) - 1)
    features: dict[int, tuple[float, ...]] = {}
    for descendant in fixed_order:
        relation = _nearest_selected_ancestor(descendant, selected, parent)
        if relation is None:
            ancestor, gap = descendant, 0
        else:
            ancestor, gap = relation
        target_node = graph.nodes[descendant - leaf_count]
        ancestor_ulp = math.ldexp(1.0, trace.ulp_exponent(ancestor))
        descendant_ulp = math.ldexp(1.0, trace.ulp_exponent(descendant))
        exact_units = trace.exact_units(descendant)
        phase = _fractional_ulp_coordinate(
            exact_units,
            trace.ulp_exponent(descendant),
        )
        parent_history_ulp = shadow_history[descendant] / descendant_ulp
        left_history_ulp = shadow_output_error[target_node.left] / descendant_ulp
        right_history_ulp = shadow_output_error[target_node.right] / descendant_ulp
        propagated_output_ulp = shadow_output_error[descendant] / ancestor_ulp
        descendant_innovation_ulp = trajectory_delta[descendant] / descendant_ulp
        inherited_fraction = abs(parent_history_ulp) / (
            abs(parent_history_ulp) + abs(descendant_innovation_ulp) + 1.0e-12
        )
        descendant_size = trace.subtree_leaves(descendant)
        ancestor_size = trace.subtree_leaves(ancestor)
        structural = (
            selected_rank[descendant] / rank_denominator,
            selected_rank[ancestor] / rank_denominator,
            gap / maximum_depth,
            float(trace.ulp_exponent(descendant) - trace.ulp_exponent(ancestor)),
            descendant_size / ancestor_size,
            abs(
                trace.subtree_leaves(target_node.left)
                - trace.subtree_leaves(target_node.right)
            )
            / descendant_size,
        )
        phase_features = (
            math.sin(2.0 * math.pi * phase),
            math.cos(2.0 * math.pi * phase),
            abs(phase - 0.5),
            float(delta0_sign[descendant]),
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
            float(predicted_cross[descendant] or predicted_phase[descendant]),
        )
        features[descendant] = structural + phase_features + shadow_features
    return features, shadow_bits


def _prune(states: Sequence[BeamState], width: int) -> list[BeamState]:
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


def _beam_score(trace: MacroTrace, model: InnovationModel) -> float:
    graph = trace.graph
    selected_order = trace.selected_order[
        : min(FROZEN_ROOT_BAND_BUDGET, len(trace.selected_order))
    ]
    selected = set(selected_order)
    features, shadow_bits = _beam_features_and_shadow_bits(trace)
    probabilities = {
        index: model.probabilities(features[index]) for index in selected_order
    }
    beams: dict[int, list[BeamState]] = {}

    def child_states(index: int) -> list[BeamState]:
        if index in selected:
            return beams[index]
        if index < graph.leaf_count:
            bits = _round_units_to_fp32_bits(trace.leaf_units[index])
        else:
            bits = shadow_bits[index]
        return [BeamState(bits=bits, probability=1.0)]

    for index in reversed(selected_order):
        node = graph.nodes[index - graph.leaf_count]
        candidates: list[BeamState] = []
        for left in child_states(node.left):
            for right in child_states(node.right):
                deterministic_bits = _round_units_to_fp32_bits(
                    _bits_to_units(left.bits) + _bits_to_units(right.bits)
                )
                inherited_probability = left.probability * right.probability
                for state_index, shift in enumerate(SHIFT_STATES):
                    candidates.append(
                        BeamState(
                            bits=max(
                                0,
                                min(FP32_MAX_FINITE_BITS, deterministic_bits + shift),
                            ),
                            probability=(
                                inherited_probability
                                * probabilities[index][state_index]
                            ),
                        )
                    )
        beams[index] = _prune(candidates, FROZEN_BEAM_WIDTH)

    root_states = beams[graph.root]
    root_quantum_units = 1 << (
        trace.root_ulp_exponent - FP32_MIN_SUBNORMAL_EXPONENT
    )
    errors = [
        (_bits_to_units(state.bits) - trace.root_units) / root_quantum_units
        for state in root_states
    ]
    return sum(
        state.probability * error * error
        for state, error in zip(root_states, errors, strict=True)
    )


def _shortlist_indices(q_scores: Sequence[float]) -> tuple[int, ...]:
    if not q_scores:
        raise ValueError("at least one candidate graph is required")
    return tuple(
        sorted(range(len(q_scores)), key=lambda index: (q_scores[index], index))[
            : min(FROZEN_SHORTLIST_SIZE, len(q_scores))
        ]
    )


def _stable_min(indices: Sequence[int], scores: Sequence[float | None]) -> int:
    return min(indices, key=lambda index: (float(scores[index]), index))


def select_tree(
    leaf_bits: Sequence[int],
    graphs: Sequence[BinaryReductionGraphLike],
    model: InnovationModel | None = None,
) -> SelectionResult:
    """Select one candidate without executing any candidate FP32 reduction tree."""
    if not leaf_bits:
        raise ValueError("leaf_bits must be nonempty")
    if not graphs:
        raise ValueError("graphs must be nonempty")
    leaf_units = tuple(_bits_to_units(bits) for bits in leaf_bits)
    if sum(leaf_units) <= 0:
        raise ValueError("the frozen score requires a positive input sum")
    for graph in graphs:
        if graph.leaf_count != len(leaf_units):
            raise ValueError("every candidate graph must match the input width")
    frozen_model = model or InnovationModel.from_json()
    traces = tuple(_macro_trace(leaf_units, graph) for graph in graphs)
    q_scores = tuple(trace.q_score for trace in traces)
    shortlist = _shortlist_indices(q_scores)
    beam_scores: list[float | None] = [None] * len(graphs)
    for index in shortlist:
        beam_scores[index] = _beam_score(traces[index], frozen_model)
    return SelectionResult(
        selected_index=_stable_min(shortlist, beam_scores),
        q_selected_index=_stable_min(range(len(graphs)), q_scores),
        shortlist_indices=shortlist,
        q_scores=q_scores,
        beam_scores=tuple(beam_scores),
    )


__all__ = [
    "DEFAULT_MODEL_PATH",
    "InnovationModel",
    "SelectionResult",
    "select_tree",
]
