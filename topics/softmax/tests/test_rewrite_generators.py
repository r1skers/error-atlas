"""Differential tests for the closed-book generator rewrite.

Three layers, in increasing strength:
  1. structural invariants of the tree validator;
  2. exact agreement with the legacy generators across widths and seeds;
  3. exact reproduction of the frozen 2026-08 wide_range_fixed_k8_beam_v2 evidence,
     which records only seeds plus stored_leaf_bits and graph_sha256.

Layer 3 is the reason the RNG call sequence is a frozen boundary rather than a free
implementation choice. These tests read frozen artifacts; they never write them.
"""

import csv
import hashlib
import json
import struct
import unittest
from fractions import Fraction
from pathlib import Path

from predictor_calibration_inputs import wide_range_random as legacy_wide_range
from predictor_tree_generator import (
    random_contiguous_split_graph,
    random_pair_merge_graph,
)
from rewrite import generators as rewrite
from rewrite.fp32_oracle import Tree, is_stored_fp32

FROZEN = (
    Path(__file__).resolve().parents[1]
    / "experiments/results/wide_range_fixed_k8_beam_v2/heldout"
)
GRAPH_ROW_STRIDE = 40  # 12288 rows total; a deterministic stride keeps the test fast
LEGACY_WIDTHS = (2, 3, 8, 33, 256)
LEGACY_SEEDS = (0, 1, 7, 1000, 55000000, 2055805352)


def _fp32_bits(value: Fraction) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _graph_sha256(tree: Tree) -> str:
    payload = json.dumps(
        {
            "leaf_count": tree.leaf_count,
            "root": tree.root,
            "nodes": [[left, right] for left, right in tree.nodes],
        },
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _skip_unless_implemented(test: unittest.TestCase, call) -> None:
    try:
        call()
    except NotImplementedError:
        test.skipTest("rewrite not implemented yet")


class TreeStructureTests(unittest.TestCase):
    def setUp(self) -> None:
        _skip_unless_implemented(
            self, lambda: rewrite.check_tree_structure(Tree(2, ((0, 1),)))
        )

    def test_accepts_well_formed_trees(self) -> None:
        good = (
            Tree(1, ()),  # degenerate: one leaf, no additions
            Tree(2, ((0, 1),)),
            Tree(3, ((0, 1), (3, 2))),
            Tree(4, ((0, 1), (2, 3), (4, 5))),
        )
        for tree in good:
            with self.subTest(tree=tree):
                rewrite.check_tree_structure(tree)

    def test_rejects_malformed_trees(self) -> None:
        bad = (
            Tree(4, ((0, 1), (2, 3))),  # too few nodes for 4 leaves
            Tree(3, ((0, 1), (3, 2), (0, 4))),  # too many nodes
            Tree(3, ((0, 1), (4, 2))),  # child index not yet evaluated
            Tree(3, ((0, 1), (3, 3))),  # same child used twice
            Tree(3, ((0, 0), (3, 1))),  # leaf 2 never consumed
            Tree(2, ((0, 5),)),  # child index out of range
            Tree(0, ()),  # no leaves
        )
        for tree in bad:
            with self.subTest(tree=tree):
                with self.assertRaises(ValueError):
                    rewrite.check_tree_structure(tree)

    def test_generated_trees_are_well_formed(self) -> None:
        for width in LEGACY_WIDTHS:
            for builder in (
                rewrite.random_contiguous_split_tree,
                rewrite.random_pair_merge_tree,
            ):
                with self.subTest(width=width, builder=builder.__name__):
                    rewrite.check_tree_structure(builder(width, seed=width * 31))


class LegacyAgreementTests(unittest.TestCase):
    def setUp(self) -> None:
        _skip_unless_implemented(
            self, lambda: rewrite.random_contiguous_split_tree(2, seed=0)
        )
        _skip_unless_implemented(self, lambda: rewrite.wide_range_random(2, seed=0))

    def test_tree_generators_match_legacy_node_for_node(self) -> None:
        for width in LEGACY_WIDTHS:
            for seed in LEGACY_SEEDS:
                for mine, legacy in (
                    (rewrite.random_contiguous_split_tree, random_contiguous_split_graph),
                    (rewrite.random_pair_merge_tree, random_pair_merge_graph),
                ):
                    expected = legacy(width, seed=seed)
                    actual = mine(width, seed=seed)
                    with self.subTest(width=width, seed=seed, builder=mine.__name__):
                        self.assertEqual(actual.leaf_count, expected.leaf_count)
                        self.assertEqual(actual.root, expected.root)
                        self.assertEqual(
                            actual.nodes,
                            tuple((n.left, n.right) for n in expected.nodes),
                        )

    def test_wide_range_matches_legacy_value_for_value(self) -> None:
        for width in (2, 8, 33, 256):
            for seed in LEGACY_SEEDS:
                with self.subTest(width=width, seed=seed):
                    self.assertEqual(
                        rewrite.wide_range_random(width, seed=seed),
                        legacy_wide_range(width, seed=seed).values,
                    )

    def test_wide_range_emits_stored_fp32_leaves(self) -> None:
        for value in rewrite.wide_range_random(64, seed=99):
            with self.subTest(value=value):
                self.assertTrue(is_stored_fp32(value))

    def test_argument_validation(self) -> None:
        for call, error in (
            (lambda: rewrite.wide_range_random(1, seed=0), ValueError),
            (lambda: rewrite.wide_range_random(8, seed=True), TypeError),
            (lambda: rewrite.random_contiguous_split_tree(0, seed=0), ValueError),
            (lambda: rewrite.random_pair_merge_tree(True, seed=0), TypeError),
        ):
            with self.subTest(error=error.__name__):
                with self.assertRaises(error):
                    call()


class FrozenEvidenceReproductionTests(unittest.TestCase):
    """Regenerate the frozen v2 held-out inputs and trees from their recorded seeds."""

    def setUp(self) -> None:
        _skip_unless_implemented(self, lambda: rewrite.wide_range_random(2, seed=0))
        _skip_unless_implemented(
            self, lambda: rewrite.random_contiguous_split_tree(2, seed=0)
        )

    def test_reproduces_every_frozen_input_group(self) -> None:
        checked = 0
        with (FROZEN / "input_groups.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                values = rewrite.wide_range_random(
                    record["width"], seed=record["seed"]
                )
                with self.subTest(group=record["input_group_id"]):
                    self.assertEqual(
                        [_fp32_bits(value) for value in values],
                        record["stored_leaf_bits"],
                    )
                checked += 1
        self.assertEqual(checked, 192)

    def test_reproduces_frozen_graph_hashes(self) -> None:
        builders = {
            "contiguous": rewrite.random_contiguous_split_tree,
            "pair_merge": rewrite.random_pair_merge_tree,
        }
        checked = 0
        with (FROZEN / "graph_observations.csv").open(encoding="utf-8") as handle:
            for index, row in enumerate(csv.DictReader(handle)):
                if index % GRAPH_ROW_STRIDE:
                    continue
                tree = builders[row["graph_family"]](
                    int(row["width"]), seed=int(row["graph_seed"])
                )
                with self.subTest(group=row["input_group_id"], tree=row["tree_index"]):
                    self.assertEqual(_graph_sha256(tree), row["graph_sha256"])
                checked += 1
        self.assertGreater(checked, 250)


if __name__ == "__main__":
    unittest.main()
