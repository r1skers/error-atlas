"""Calibration-only second-moment baseline for reduction-tree ranking.

This module implements the leading tree-dependent cost suggested by the 2026 paper
"A Second-Moment Theory for Floating-Point Reduction Trees".  For one fixed stored
input and binary reduction tree, let q_v be the (cheap binary64) partial sum at internal
node v.  The leading cost p^T K_T p is equivalently the sum of q_v^2 over internal nodes.

This is a theory baseline, not a frozen predictor and not held-out evidence.  It does not
simulate FP32 round-to-nearest and does not call the exact oracle.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

from summation_graph_predictor import BinaryReductionGraph


SECOND_MOMENT_BASELINE_VERSION = "partial_sum_square_v1"


@dataclass(frozen=True)
class SecondMomentBaseline:
    baseline_version: str
    graph_name: str
    internal_node_count: int
    partial_sum_square_cost: float
    root_mass: float


def second_moment_tree_cost(
    values: Sequence[Fraction],
    graph: BinaryReductionGraph,
) -> SecondMomentBaseline:
    """Return the cheap leading second-moment tree cost sum_v q_v^2.

    Stored leaves are converted once to ordinary Python floats.  Internal partial sums are
    then propagated in binary64 according to the candidate graph.  No FP32 rounding is
    simulated, so this remains a pre-execution structural/numerical baseline.
    """
    if len(values) != graph.leaf_count:
        raise ValueError("value count does not match graph leaf_count")
    for value in values:
        if not isinstance(value, Fraction):
            raise TypeError("every input value must be a Fraction")

    masses = [float(value) for value in values]
    cost = 0.0

    for node in graph.nodes:
        node_mass = masses[node.left] + masses[node.right]
        masses.append(node_mass)
        cost += node_mass * node_mass

    root_mass = masses[-1] if graph.nodes else masses[0]
    return SecondMomentBaseline(
        baseline_version=SECOND_MOMENT_BASELINE_VERSION,
        graph_name=graph.name,
        internal_node_count=len(graph.nodes),
        partial_sum_square_cost=cost,
        root_mass=root_mass,
    )
