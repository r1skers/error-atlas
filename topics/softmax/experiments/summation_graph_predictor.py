"""Exact pre-execution error predictor for explicit FP32 addition trees.

The predictor is deliberately independent of NumPy arithmetic.  It accepts
already-stored nonnegative binary32 inputs as exact ``Fraction`` values and an
explicit binary reduction tree.  Every internal node is evaluated with an
integer/rational implementation of round-to-nearest, ties-to-even.

For leaf values ``x_i`` and internal nodes ``v``, define

    a_v = y_left(v) + y_right(v)
    y_v = RN32(a_v)
    rho_v = y_v - a_v.

For a proper reduction tree, the predicted signed forward error is the exact
identity

    E_G(x) = y_root - sum_i(x_i) = sum_v(rho_v).

``E_G`` is computable before candidate execution and is falsified if either
the candidate output bits or its signed error disagree.  Agreement validates
the claimed graph/dtype/rounding contract for the controlled case; it does not
identify an unknown black-box reduction graph.

This first research step excludes negative inputs, overflow, fused operations,
compensated algorithms, and non-tree graphs.  Those require separate contracts
rather than silent generalization of this oracle.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction


FP32_FRACTION_BITS = 23
FP32_MIN_NORMAL_EXPONENT = -126
FP32_MIN_SUBNORMAL_EXPONENT = -149
FP32_MAX_NORMAL_EXPONENT = 127
FP32_MAX_FINITE = Fraction((2**24) - 1) * Fraction(2**104)


@dataclass(frozen=True)
class FP32Rounding:
    """One exact nonnegative binary32 rounding result."""

    value: Fraction
    bits: int

    @property
    def bits_hex(self) -> str:
        return f"0x{self.bits:08x}"


@dataclass(frozen=True)
class AdditionNode:
    """One topologically ordered binary-addition node."""

    left: int
    right: int


@dataclass(frozen=True)
class BinaryReductionGraph:
    """An explicit full binary tree over ordered input leaves.

    Leaves have indices ``0 .. leaf_count - 1``.  Internal node ``j`` has
    index ``leaf_count + j`` and may only reference earlier indices.
    """

    name: str
    leaf_count: int
    nodes: tuple[AdditionNode, ...]
    root: int

    def __post_init__(self) -> None:
        if isinstance(self.leaf_count, bool) or not isinstance(self.leaf_count, int):
            raise TypeError("leaf_count must be an integer")
        if self.leaf_count <= 0:
            raise ValueError("leaf_count must be positive")
        if len(self.nodes) != self.leaf_count - 1:
            raise ValueError("a full reduction tree needs leaf_count - 1 nodes")

        value_count = self.leaf_count + len(self.nodes)
        expected_root = value_count - 1
        if self.root != expected_root:
            raise ValueError("root must be the final topological value")

        parent_counts = [0] * value_count
        for node_offset, node in enumerate(self.nodes):
            node_index = self.leaf_count + node_offset
            for child in (node.left, node.right):
                if isinstance(child, bool) or not isinstance(child, int):
                    raise TypeError("node references must be integers")
                if not 0 <= child < node_index:
                    raise ValueError(
                        "each node must reference two earlier graph values"
                    )
                parent_counts[child] += 1

        for value_index, parent_count in enumerate(parent_counts):
            expected = 0 if value_index == self.root else 1
            if parent_count != expected:
                raise ValueError(
                    "graph must be one connected tree with every non-root "
                    "value consumed exactly once"
                )


@dataclass(frozen=True)
class NodePrediction:
    """Exact semantic prediction for one graph addition."""

    node_index: int
    left: int
    right: int
    exact_addend_sum: Fraction
    rounded_sum: Fraction
    rounded_sum_bits: str
    local_rounding_error: Fraction


@dataclass(frozen=True)
class GraphErrorPrediction:
    """Pre-execution result and forward-error prediction for one tree."""

    graph_name: str
    exact_input_sum: Fraction
    predicted_sum: Fraction
    predicted_sum_bits: str
    signed_error: Fraction
    absolute_relative_error: Fraction | None
    local_error_sum: Fraction
    inexact_addition_count: int
    node_predictions: tuple[NodePrediction, ...]


def _power_of_two(exponent: int) -> Fraction:
    if exponent >= 0:
        return Fraction(2**exponent)
    return Fraction(1, 2 ** (-exponent))


def _floor_log2(value: Fraction) -> int:
    """Return floor(log2(value)) for a positive rational without floats."""
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if value < _power_of_two(exponent):
        exponent -= 1
    return exponent


def _round_nonnegative_fraction_to_integer_ties_even(
    value: Fraction,
) -> int:
    """Round a nonnegative rational to an integer with ties-to-even."""
    quotient, remainder = divmod(value.numerator, value.denominator)
    midpoint_comparison = 2 * remainder - value.denominator
    if midpoint_comparison > 0:
        return quotient + 1
    if midpoint_comparison < 0:
        return quotient
    return quotient + (quotient & 1)


def round_nonnegative_fraction_to_fp32(value: Fraction) -> FP32Rounding:
    """Round an exact nonnegative value to finite IEEE binary32.

    The supported domain is intentionally explicit: ``value`` must be a
    ``Fraction`` in ``[0, max_finite_binary32]``.  Rejecting larger values
    keeps overflow semantics outside this first predictor-validation step.
    """
    if not isinstance(value, Fraction):
        raise TypeError("value must be a Fraction")
    if value < 0:
        raise ValueError("value must be nonnegative")
    if value > FP32_MAX_FINITE:
        raise OverflowError("overflow is outside the predictor domain")
    if value == 0:
        return FP32Rounding(value=Fraction(0), bits=0)

    exponent = _floor_log2(value)
    if exponent < FP32_MIN_NORMAL_EXPONENT:
        quantum = _power_of_two(FP32_MIN_SUBNORMAL_EXPONENT)
        significand = _round_nonnegative_fraction_to_integer_ties_even(value / quantum)
        if significand == 0:
            return FP32Rounding(value=Fraction(0), bits=0)
        if significand < 2**FP32_FRACTION_BITS:
            return FP32Rounding(
                value=Fraction(significand) * quantum,
                bits=significand,
            )
        exponent = FP32_MIN_NORMAL_EXPONENT
        significand = 2**FP32_FRACTION_BITS
    else:
        quantum = _power_of_two(exponent - FP32_FRACTION_BITS)
        significand = _round_nonnegative_fraction_to_integer_ties_even(value / quantum)
        if significand == 2 ** (FP32_FRACTION_BITS + 1):
            exponent += 1
            significand = 2**FP32_FRACTION_BITS

    if exponent > FP32_MAX_NORMAL_EXPONENT:
        raise OverflowError("rounded result overflows finite binary32")

    quantum = _power_of_two(exponent - FP32_FRACTION_BITS)
    rounded_value = Fraction(significand) * quantum
    exponent_bits = exponent - FP32_MIN_NORMAL_EXPONENT + 1
    fraction_bits = significand - 2**FP32_FRACTION_BITS
    bits = (exponent_bits << FP32_FRACTION_BITS) | fraction_bits
    return FP32Rounding(value=rounded_value, bits=bits)


def sequential_reduction_graph(leaf_count: int) -> BinaryReductionGraph:
    """Return the explicit left-to-right reduction tree for ``leaf_count``."""
    if isinstance(leaf_count, bool) or not isinstance(leaf_count, int):
        raise TypeError("leaf_count must be an integer")
    if leaf_count <= 0:
        raise ValueError("leaf_count must be positive")

    nodes: list[AdditionNode] = []
    root = 0
    for next_leaf in range(1, leaf_count):
        nodes.append(AdditionNode(left=root, right=next_leaf))
        root = leaf_count + len(nodes) - 1
    return BinaryReductionGraph(
        name="sequential_left_to_right",
        leaf_count=leaf_count,
        nodes=tuple(nodes),
        root=root,
    )


def balanced_reduction_graph(leaf_count: int) -> BinaryReductionGraph:
    """Return the fixed contiguous tree split at ``length // 2``."""
    if isinstance(leaf_count, bool) or not isinstance(leaf_count, int):
        raise TypeError("leaf_count must be an integer")
    if leaf_count <= 0:
        raise ValueError("leaf_count must be positive")

    nodes: list[AdditionNode] = []

    def build(start: int, stop: int) -> int:
        length = stop - start
        if length == 1:
            return start
        midpoint = start + length // 2
        left = build(start, midpoint)
        right = build(midpoint, stop)
        nodes.append(AdditionNode(left=left, right=right))
        return leaf_count + len(nodes) - 1

    root = build(0, leaf_count)
    return BinaryReductionGraph(
        name="balanced_contiguous_floor_half",
        leaf_count=leaf_count,
        nodes=tuple(nodes),
        root=root,
    )


def predict_fp32_tree_error(
    values: Sequence[Fraction],
    graph: BinaryReductionGraph,
) -> GraphErrorPrediction:
    """Predict one explicit tree's FP32 result and exact forward error."""
    if len(values) != graph.leaf_count:
        raise ValueError("value count does not match graph leaf_count")

    states: list[Fraction] = []
    for value in values:
        if not isinstance(value, Fraction):
            raise TypeError("every input value must be a Fraction")
        stored = round_nonnegative_fraction_to_fp32(value)
        if stored.value != value:
            raise ValueError("every input must already be exactly stored FP32")
        states.append(value)

    node_predictions: list[NodePrediction] = []
    for node_offset, node in enumerate(graph.nodes):
        node_index = graph.leaf_count + node_offset
        exact_addend_sum = states[node.left] + states[node.right]
        rounded = round_nonnegative_fraction_to_fp32(exact_addend_sum)
        local_error = rounded.value - exact_addend_sum
        states.append(rounded.value)
        node_predictions.append(
            NodePrediction(
                node_index=node_index,
                left=node.left,
                right=node.right,
                exact_addend_sum=exact_addend_sum,
                rounded_sum=rounded.value,
                rounded_sum_bits=rounded.bits_hex,
                local_rounding_error=local_error,
            )
        )

    exact_input_sum = sum(values, start=Fraction(0))
    predicted_sum = states[graph.root]
    predicted_rounding = round_nonnegative_fraction_to_fp32(predicted_sum)
    signed_error = predicted_sum - exact_input_sum
    local_error_sum = sum(
        (node.local_rounding_error for node in node_predictions),
        start=Fraction(0),
    )
    if local_error_sum != signed_error:
        raise AssertionError("tree residual identity was violated")

    absolute_relative_error = (
        None if exact_input_sum == 0 else abs(signed_error) / abs(exact_input_sum)
    )
    return GraphErrorPrediction(
        graph_name=graph.name,
        exact_input_sum=exact_input_sum,
        predicted_sum=predicted_sum,
        predicted_sum_bits=predicted_rounding.bits_hex,
        signed_error=signed_error,
        absolute_relative_error=absolute_relative_error,
        local_error_sum=local_error_sum,
        inexact_addition_count=sum(
            node.local_rounding_error != 0 for node in node_predictions
        ),
        node_predictions=tuple(node_predictions),
    )
