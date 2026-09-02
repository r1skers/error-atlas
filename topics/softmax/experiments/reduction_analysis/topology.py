"""Shared structural information; no numeric execution or experiment seed policy."""

from __future__ import annotations

from dataclasses import dataclass

from summation_graph_predictor import BinaryReductionGraph


@dataclass(frozen=True)
class TreeTopology:
    """Parent and root-depth arrays indexed by both leaf and internal node IDs."""

    parent: tuple[int | None, ...]
    depth: tuple[int, ...]

    @classmethod
    def from_graph(cls, graph: BinaryReductionGraph) -> TreeTopology:
        count = graph.leaf_count + len(graph.nodes)
        parent: list[int | None] = [None] * count
        depth = [0] * count
        # Reverse topological order visits every parent before its children.
        for offset in reversed(range(len(graph.nodes))):
            node = graph.nodes[offset]
            index = graph.leaf_count + offset
            for child in (node.left, node.right):
                parent[child] = index
                depth[child] = depth[index] + 1
        return cls(tuple(parent), tuple(depth))

    def ancestor_gap(self, u: int, v: int) -> int | None:
        """Distance for a proper ancestor pair, or None for disjoint/equal nodes."""
        if not (0 <= u < len(self.parent) and 0 <= v < len(self.parent)):
            raise IndexError("node index is outside this topology")
        for descendant, ancestor in ((u, v), (v, u)):
            current = self.parent[descendant]
            gap = 1
            while current is not None:
                if current == ancestor:
                    return gap
                current = self.parent[current]
                gap += 1
        return None
