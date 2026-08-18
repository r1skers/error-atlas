"""Cheap structural features for reduction-graph predictor validation.

The features in this module deliberately do not simulate FP32 round-to-nearest or call
the exact semantic oracle.  They may inspect the frozen stored leaves and the explicit
tree topology, then use ordinary binary64 arithmetic to build inexpensive structural
summaries.

Version 1 contains only sibling scale mismatch.  Dominant-leaf exposure and any combined
predictor score are intentionally left for later research decisions.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

from summation_graph_predictor import BinaryReductionGraph


STRUCTURAL_FEATURE_VERSION = "sibling_mismatch_v1"


@dataclass(frozen=True)
class NodeSiblingMismatch:
    """Scale mismatch for one internal merge."""

    node_index: int
    left: int
    right: int
    left_mass: float
    right_mass: float
    log2_gap: float


@dataclass(frozen=True)
class SiblingMismatchFeatures:
    """Simple graph-level summaries of node-wise sibling scale mismatch."""

    feature_version: str
    graph_name: str
    node_count: int
    mean_log2_gap: float
    max_log2_gap: float
    top_k: int
    top_k_mean_log2_gap: float
    nodes: tuple[NodeSiblingMismatch, ...]


def _validate_values(values: Sequence[Fraction], graph: BinaryReductionGraph) -> None:
    if len(values) != graph.leaf_count:
        raise ValueError("value count does not match graph leaf_count")
    for value in values:
        if not isinstance(value, Fraction):
            raise TypeError("every input value must be a Fraction")
        if value < 0:
            raise ValueError("sibling mismatch v1 requires nonnegative leaves")


def _log2_gap(left_mass: float, right_mass: float) -> float:
    if left_mass == 0.0 and right_mass == 0.0:
        return 0.0
    if left_mass == 0.0 or right_mass == 0.0:
        return math.inf
    return abs(math.log2(left_mass) - math.log2(right_mass))


def sibling_scale_mismatch_features(
    values: Sequence[Fraction],
    graph: BinaryReductionGraph,
    *,
    top_k: int = 4,
) -> SiblingMismatchFeatures:
    """Compute cheap sibling-scale summaries for one stored input and tree.

    For each internal node ``v`` with child subtrees ``L`` and ``R``, define the cheap
    magnitude summary ``M`` as the binary64 sum of the stored nonnegative leaves in that
    subtree and compute

        m_v = |log2 M(L) - log2 M(R)|.

    Subtree masses are propagated with ordinary Python ``float`` additions.  There is no
    FP32 rounding simulation and no use of local FP32 residuals, so this remains a cheap
    structural feature rather than a semantic-oracle calculation.

    ``top_k`` controls only the reported top-k mean.  All node mismatches are retained so
    later protocol work can choose a summary without recomputing the tree traversal.
    """
    _validate_values(values, graph)
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k must be an integer")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    masses = [float(value) for value in values]
    node_features: list[NodeSiblingMismatch] = []

    for node_offset, node in enumerate(graph.nodes):
        node_index = graph.leaf_count + node_offset
        left_mass = masses[node.left]
        right_mass = masses[node.right]
        gap = _log2_gap(left_mass, right_mass)
        node_features.append(
            NodeSiblingMismatch(
                node_index=node_index,
                left=node.left,
                right=node.right,
                left_mass=left_mass,
                right_mass=right_mass,
                log2_gap=gap,
            )
        )
        masses.append(left_mass + right_mass)

    gaps = [node.log2_gap for node in node_features]
    if not gaps:
        mean_gap = 0.0
        max_gap = 0.0
        effective_top_k = 0
        top_k_mean = 0.0
    else:
        mean_gap = sum(gaps) / len(gaps)
        max_gap = max(gaps)
        effective_top_k = min(top_k, len(gaps))
        largest = sorted(gaps, reverse=True)[:effective_top_k]
        top_k_mean = sum(largest) / effective_top_k

    return SiblingMismatchFeatures(
        feature_version=STRUCTURAL_FEATURE_VERSION,
        graph_name=graph.name,
        node_count=len(node_features),
        mean_log2_gap=mean_gap,
        max_log2_gap=max_gap,
        top_k=effective_top_k,
        top_k_mean_log2_gap=top_k_mean,
        nodes=tuple(node_features),
    )
