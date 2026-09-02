"""One exact oracle replay shared by multiple diagnostic views.

This is an oracle-labelled diagnostic interface, NOT a cheap predictor input.
The RN32 implementation remains in summation_graph_predictor.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property

from summation_graph_predictor import (
    BinaryReductionGraph,
    GraphErrorPrediction,
    NodePrediction,
    predict_fp32_tree_error,
)

from .topology import TreeTopology


@dataclass(frozen=True)
class ReductionTrace:
    """Inputs, topology and the existing oracle result, constructed by replay().

    Node IDs keep the oracle convention: leaves first, then topological additions.
    No second node-record schema or second rounding implementation is introduced.
    """

    values: tuple[Fraction, ...]
    graph: BinaryReductionGraph
    prediction: GraphErrorPrediction

    @property
    def nodes(self) -> tuple[NodePrediction, ...]:
        return self.prediction.node_predictions

    @cached_property
    def node_ids(self) -> tuple[int, ...]:
        return tuple(node.node_index for node in self.nodes)

    @cached_property
    def deltas(self) -> tuple[Fraction, ...]:
        return tuple(node.local_rounding_error for node in self.nodes)

    @cached_property
    def topology(self) -> TreeTopology:
        return TreeTopology.from_graph(self.graph)

    def value_at(self, index: int) -> Fraction:
        """Stored leaf or rounded internal output, not an exact shadow subtree sum."""
        if not 0 <= index < self.graph.leaf_count + len(self.nodes):
            raise IndexError("node index is outside this trace")
        if index < self.graph.leaf_count:
            return self.values[index]
        return self.nodes[index - self.graph.leaf_count].rounded_sum


def replay(values: Sequence[Fraction], graph: BinaryReductionGraph) -> ReductionTrace:
    """Evaluate a case exactly once; delegate all numerical validation to the oracle."""
    stored_values = tuple(values)
    prediction = predict_fp32_tree_error(stored_values, graph)
    return ReductionTrace(stored_values, graph, prediction)
