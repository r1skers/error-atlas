"""Tests for reproducible predictor-validation tree generators."""

import unittest

from predictor_tree_generator import (
    random_contiguous_split_graph,
    random_pair_merge_graph,
)


class PredictorTreeGeneratorTests(unittest.TestCase):
    def test_contiguous_split_is_reproducible(self) -> None:
        first = random_contiguous_split_graph(32, seed=17)
        second = random_contiguous_split_graph(32, seed=17)

        self.assertEqual(first.name, second.name)
        self.assertEqual(first.nodes, second.nodes)
        self.assertEqual(first.root, second.root)

    def test_pair_merge_is_reproducible(self) -> None:
        first = random_pair_merge_graph(32, seed=23)
        second = random_pair_merge_graph(32, seed=23)

        self.assertEqual(first.name, second.name)
        self.assertEqual(first.nodes, second.nodes)
        self.assertEqual(first.root, second.root)

    def test_both_generators_return_full_trees(self) -> None:
        for graph in (
            random_contiguous_split_graph(19, seed=3),
            random_pair_merge_graph(19, seed=3),
        ):
            self.assertEqual(graph.leaf_count, 19)
            self.assertEqual(len(graph.nodes), 18)
            self.assertEqual(graph.root, 36)

    def test_contiguous_split_preserves_contiguous_subtrees(self) -> None:
        graph = random_contiguous_split_graph(25, seed=41)
        leaf_sets: list[frozenset[int]] = [
            frozenset((leaf,)) for leaf in range(graph.leaf_count)
        ]

        for node in graph.nodes:
            leaves = leaf_sets[node.left] | leaf_sets[node.right]
            ordered = sorted(leaves)
            self.assertEqual(ordered, list(range(ordered[0], ordered[-1] + 1)))
            leaf_sets.append(leaves)

        self.assertEqual(leaf_sets[graph.root], frozenset(range(graph.leaf_count)))

    def test_single_leaf_is_supported(self) -> None:
        for graph in (
            random_contiguous_split_graph(1, seed=0),
            random_pair_merge_graph(1, seed=0),
        ):
            self.assertEqual(graph.nodes, ())
            self.assertEqual(graph.root, 0)

    def test_rejects_invalid_arguments(self) -> None:
        for generator in (random_contiguous_split_graph, random_pair_merge_graph):
            with self.assertRaises(ValueError):
                generator(0, seed=1)
            with self.assertRaises(TypeError):
                generator(True, seed=1)  # type: ignore[arg-type]
            with self.assertRaises(TypeError):
                generator(4, seed=True)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
