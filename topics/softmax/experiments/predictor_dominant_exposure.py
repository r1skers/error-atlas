"""Cheap dominant-leaf exposure feature for reduction-graph predictor validation.

This module inspects one stored nonnegative FP32 input and one explicit reduction tree.
It does not simulate FP32 round-to-nearest and does not call the exact semantic oracle.

Version 1 tracks only the single largest stored leaf.  Ties are resolved by the earliest
leaf index so the feature is deterministic.  Along that leaf's path to the root, each
merge receives an exposure severity based on how much smaller the sibling subtree mass is
than the dominant leaf itself.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

from summation_graph_predictor import BinaryReductionGraph


DOMINANT_EXPOSURE_VERSION = "dominant_leaf_exposure_v1"


@dataclass(frozen=True)
class DominantExposureStep:
    """One merge encountered by the dominant leaf on its path to the root."""

    child_index: int
    parent_index: int
    sibling_index: int
    sibling_mass: float
    severity_log2: float


@dataclass(frozen=True)
class DominantExposureFeatures:
    """Graph-level summaries of dominant-leaf exposure."""

    feature_version: str
    graph_name: str
    dominant_leaf_index: int
    dominant_leaf_mass: float
    path_length: int
    total_severity_log2: float
    mean_severity_log2: float
    max_severity_log2: float
    steps: tuple[DominantExposureStep, ...]


def _validate_values(values: Sequence[Fraction], graph: BinaryReductionGraph) -> None:
    if len(values) != graph.leaf_count:
        raise ValueError("value count does not match graph leaf_count")
    for value in values:
        if not isinstance(value, Fraction):
            raise TypeError("every input value must be a Fraction")
        if value < 0:
            raise ValueError("dominant exposure v1 requires nonnegative leaves")


def _severity(dominant_mass: float, sibling_mass: float) -> float:
    """Return positive log2 scale separation when the sibling is smaller."""
    if dominant_mass == 0.0:
        return 0.0
    if sibling_mass == 0.0:
        return math.inf
    return max(0.0, math.log2(dominant_mass) - math.log2(sibling_mass))


def dominant_leaf_exposure_features(
    values: Sequence[Fraction],
    graph: BinaryReductionGraph,
) -> DominantExposureFeatures:
    """Track the largest stored leaf through the reduction tree.

    Let ``d`` be the largest stored leaf, with earliest-index tie breaking.  For every
    merge on the unique path from ``d`` to the root, let ``M_s`` be the binary64 sum of
    the sibling subtree's stored leaves.  The step severity is

        e_v = max(0, log2(d) - log2(M_s)).

    A zero sibling mass gives infinite severity when ``d`` is positive.  An all-zero input
    is defined to have zero exposure.  The feature records sum, mean, and maximum path
    severity without selecting one as the final predictor score.
    """
    _validate_values(values, graph)

    float_values = [float(value) for value in values]
    dominant_index = max(range(len(values)), key=lambda index: (values[index], -index))
    dominant_mass = float_values[dominant_index]

    masses = list(float_values)
    parent_of: list[int | None] = [None] * (graph.leaf_count + len(graph.nodes))
    sibling_of: list[int | None] = [None] * (graph.leaf_count + len(graph.nodes))

    for node_offset, node in enumerate(graph.nodes):
        node_index = graph.leaf_count + node_offset
        masses.append(masses[node.left] + masses[node.right])
        parent_of[node.left] = node_index
        parent_of[node.right] = node_index
        sibling_of[node.left] = node.right
        sibling_of[node.right] = node.left

    steps: list[DominantExposureStep] = []
    current = dominant_index
    while current != graph.root:
        parent = parent_of[current]
        sibling = sibling_of[current]
        if parent is None or sibling is None:
            raise AssertionError("proper tree path metadata is incomplete")
        sibling_mass = masses[sibling]
        severity = _severity(dominant_mass, sibling_mass)
        steps.append(
            DominantExposureStep(
                child_index=current,
                parent_index=parent,
                sibling_index=sibling,
                sibling_mass=sibling_mass,
                severity_log2=severity,
            )
        )
        current = parent

    severities = [step.severity_log2 for step in steps]
    if not severities:
        total = 0.0
        mean = 0.0
        maximum = 0.0
    else:
        total = sum(severities)
        mean = total / len(severities)
        maximum = max(severities)

    return DominantExposureFeatures(
        feature_version=DOMINANT_EXPOSURE_VERSION,
        graph_name=graph.name,
        dominant_leaf_index=dominant_index,
        dominant_leaf_mass=dominant_mass,
        path_length=len(steps),
        total_severity_log2=total,
        mean_severity_log2=mean,
        max_severity_log2=maximum,
        steps=tuple(steps),
    )
