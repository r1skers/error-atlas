"""Merge schedules an actual CUDA kernel can express.

Agent scaffolding: pure combinatorics over block indices, no research judgement and no
arithmetic. The point of this module is that the online normalizer stage does not rank
arbitrary trees the way the frozen reduction line did. A real kernel can only express a
small, structured family, and that family is a fact about the programming model rather
than about the hardware, so it can be enumerated and simulated exactly on the CPU.

Reachable shapes (contract note section 7):

- intra-warp: ``__shfl_xor_sync`` / ``__shfl_down_sync``, balanced, depth log2(lanes);
- intra-block across warps: a small balanced tree or a short chain over warp partials;
- across K/V blocks: the FlashAttention main loop, a strict sequential chain;
- split-K / FlashDecoding: the top of that chain replaced by a combine over splits.

Index convention matches the frozen line: leaves are ``0 .. leaf_count-1``, internal node
``leaf_count + k`` is ``nodes[k]``, nodes are evaluated in list order, so both children of
a node are already defined when it is reached and the last node is the root.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Schedule:
    """One merge order over ``leaf_count`` blocks, plus a label for provenance."""

    leaf_count: int
    nodes: tuple[tuple[int, int], ...]
    kind: str

    @property
    def root(self) -> int:
        return self.leaf_count + len(self.nodes) - 1

    @property
    def depth(self) -> int:
        """Longest root-to-leaf path, the quantity an O(u log n) bound is stated in."""
        height = [0] * (self.leaf_count + len(self.nodes))
        for k, (left, right) in enumerate(self.nodes):
            height[self.leaf_count + k] = 1 + max(height[left], height[right])
        return height[self.root]


def _finish(leaf_count: int, nodes: list[tuple[int, int]], kind: str) -> Schedule:
    schedule = Schedule(leaf_count=leaf_count, nodes=tuple(nodes), kind=kind)
    check_schedule(schedule)
    return schedule


def check_schedule(schedule: Schedule) -> None:
    """Raise ``ValueError`` unless the schedule is a well-formed evaluable binary tree.

    There is no separate "the root is not a child" check: the root carries the largest
    index and every child must have a smaller index than its parent, so that case is
    already unreachable.
    """
    leaf_count = schedule.leaf_count
    if leaf_count < 1:
        raise ValueError("A schedule needs at least one leaf.")
    if len(schedule.nodes) != leaf_count - 1:
        raise ValueError("A full binary tree over L leaves has exactly L-1 internal nodes.")
    used: set[int] = set()
    for k, (left, right) in enumerate(schedule.nodes):
        node = leaf_count + k
        for child in (left, right):
            if not 0 <= child < node:
                raise ValueError(f"Child {child} of node {node} is not evaluated earlier.")
            if child in used:
                raise ValueError(f"Index {child} is used as a child more than once.")
            used.add(child)


def sequential_chain(leaf_count: int) -> Schedule:
    """The FlashAttention K/V loop: one accumulator, one block folded in at a time.

    Depth is ``leaf_count - 1``. This is the shape most exposed to repeated rescaling of
    an already-accumulated state, so it is the natural worst case for the stage.
    """
    nodes: list[tuple[int, int]] = []
    current = 0
    for leaf in range(1, leaf_count):
        nodes.append((current, leaf))
        current = leaf_count + len(nodes) - 1
    return _finish(leaf_count, nodes, "sequential_chain")


def balanced_pairwise(leaf_count: int) -> Schedule:
    """Adjacent pairs, level by level; an odd survivor is carried to the next level."""
    nodes: list[tuple[int, int]] = []
    level = list(range(leaf_count))
    while len(level) > 1:
        nxt: list[int] = []
        for i in range(0, len(level) - 1, 2):
            nodes.append((level[i], level[i + 1]))
            nxt.append(leaf_count + len(nodes) - 1)
        if len(level) % 2:
            nxt.append(level[-1])
        level = nxt
    return _finish(leaf_count, nodes, "balanced_pairwise")


def warp_shfl_xor(lane_count: int) -> Schedule:
    """Butterfly reduction via ``__shfl_xor_sync``.

    Worth stating explicitly: as a *reduction tree* this is exactly
    :func:`balanced_pairwise`. The butterfly is about data movement (every lane ends
    holding the full result), not about which leaves get summed together. Only the
    ``shfl_down`` variant below actually changes the pairing.
    """
    _require_power_of_two(lane_count)
    base = balanced_pairwise(lane_count)
    return Schedule(leaf_count=base.leaf_count, nodes=base.nodes, kind="warp_shfl_xor")


def warp_shfl_down(lane_count: int) -> Schedule:
    """Reduction via ``__shfl_down_sync`` with halving offsets.

    Same depth as the xor butterfly but a different leaf pairing: the first step joins
    lane ``i`` with lane ``i + lanes/2``, so the leaves that end up summed together are
    strided rather than adjacent. On skewed data the two are not interchangeable, which
    is a distinction the frozen line's random trees could not express.
    """
    _require_power_of_two(lane_count)
    nodes: list[tuple[int, int]] = []
    current = list(range(lane_count))
    offset = lane_count // 2
    while offset >= 1:
        nxt: list[int] = []
        for i in range(offset):
            nodes.append((current[i], current[i + offset]))
            nxt.append(lane_count + len(nodes) - 1)
        current = nxt
        offset //= 2
    return _finish(lane_count, nodes, "warp_shfl_down")


def split_k(leaf_count: int, splits: int) -> Schedule:
    """FlashDecoding-style split: chain inside each split, then combine the partials.

    Blocks are cut into ``splits`` contiguous groups as evenly as possible. Each group is
    a sequential chain, matching what one CTA does; the partials are then combined by a
    second chain, matching the usual flat combine kernel. ``splits == 1`` degenerates to
    :func:`sequential_chain`.
    """
    if not 1 <= splits <= leaf_count:
        raise ValueError("splits must be between 1 and leaf_count.")
    nodes: list[tuple[int, int]] = []
    bounds = [leaf_count * i // splits for i in range(splits + 1)]
    partials: list[int] = []
    for start, stop in zip(bounds, bounds[1:]):
        if start == stop:
            raise ValueError("An empty split would not correspond to a real launch.")
        current = start
        for leaf in range(start + 1, stop):
            nodes.append((current, leaf))
            current = leaf_count + len(nodes) - 1
        partials.append(current)
    combined = partials[0]
    for partial in partials[1:]:
        nodes.append((combined, partial))
        combined = leaf_count + len(nodes) - 1
    return _finish(leaf_count, nodes, f"split_k_{splits}")


def _require_power_of_two(value: int) -> None:
    if value < 1 or value & (value - 1):
        raise ValueError("Warp-shuffle schedules need a power-of-two lane count.")
