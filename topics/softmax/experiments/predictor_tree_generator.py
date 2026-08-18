"""Reproducible candidate-tree generators for predictor validation.

This module defines graph sampling only. It does not inspect input values, predictor
scores, oracle errors, or failure labels. A generated graph therefore cannot be selected
because of how well the predictor or oracle behaves on a particular input.
"""

from __future__ import annotations

import random

from summation_graph_predictor import AdditionNode, BinaryReductionGraph


TREE_GENERATOR_VERSION = "v1"


def _validate_leaf_count(leaf_count: int) -> None:
    if isinstance(leaf_count, bool) or not isinstance(leaf_count, int):
        raise TypeError("leaf_count must be an integer")
    if leaf_count <= 0:
        raise ValueError("leaf_count must be positive")


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")


def random_contiguous_split_graph(
    leaf_count: int,
    *,
    seed: int,
) -> BinaryReductionGraph:
    """Sample a full tree by recursively splitting contiguous leaf intervals.

    Each non-singleton interval chooses one split point uniformly from its valid
    interior positions. The ordered leaf sequence is never permuted.
    """
    _validate_leaf_count(leaf_count)
    _validate_seed(seed)
    rng = random.Random(seed)
    nodes: list[AdditionNode] = []

    def build(start: int, stop: int) -> int:
        if stop - start == 1:
            return start
        split = rng.randrange(start + 1, stop)
        left = build(start, split)
        right = build(split, stop)
        nodes.append(AdditionNode(left=left, right=right))
        return leaf_count + len(nodes) - 1

    root = build(0, leaf_count)
    return BinaryReductionGraph(
        name=f"random_contiguous_split_{TREE_GENERATOR_VERSION}_seed_{seed}",
        leaf_count=leaf_count,
        nodes=tuple(nodes),
        root=root,
    )


def _swap_pop(values: list[int], position: int) -> int:
    value = values[position]
    values[position] = values[-1]
    values.pop()
    return value


def random_pair_merge_graph(
    leaf_count: int,
    *,
    seed: int,
) -> BinaryReductionGraph:
    """Sample a full tree by repeatedly merging two random active nodes.

    The active pool initially contains every leaf index. At each step two distinct
    active nodes are sampled uniformly without replacement, merged, and replaced by
    their parent. Pool removal is O(1), so graph generation remains practical for the
    large width strata used by the validation protocol.

    This procedure defines a reproducible graph distribution; it does not claim to be
    uniform over unlabeled or labeled full binary-tree topologies.
    """
    _validate_leaf_count(leaf_count)
    _validate_seed(seed)
    rng = random.Random(seed)
    active = list(range(leaf_count))
    nodes: list[AdditionNode] = []

    while len(active) > 1:
        first_position = rng.randrange(len(active))
        second_position = rng.randrange(len(active) - 1)
        if second_position >= first_position:
            second_position += 1

        high = max(first_position, second_position)
        low = min(first_position, second_position)
        first = _swap_pop(active, high)
        second = _swap_pop(active, low)
        left, right = sorted((first, second))

        nodes.append(AdditionNode(left=left, right=right))
        active.append(leaf_count + len(nodes) - 1)

    root = active[0]
    return BinaryReductionGraph(
        name=f"random_pair_merge_{TREE_GENERATOR_VERSION}_seed_{seed}",
        leaf_count=leaf_count,
        nodes=tuple(nodes),
        root=root,
    )
