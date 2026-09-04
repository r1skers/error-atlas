"""Tests for the CUDA-expressible merge schedule family.

These cover agent scaffolding, so they do not skip: the generators are pure
combinatorics and must be correct before any core is written against them.
"""

import math
import unittest

from online import schedules


def _leaf_groups(schedule: schedules.Schedule) -> list[frozenset[int]]:
    """The leaf set under each internal node, in node order."""
    count = schedule.leaf_count
    groups: list[frozenset[int]] = []
    for left, right in schedule.nodes:
        under = []
        for child in (left, right):
            under.append(frozenset([child]) if child < count else groups[child - count])
        groups.append(under[0] | under[1])
    return groups


class WellFormednessTests(unittest.TestCase):
    def test_every_generator_produces_an_evaluable_tree(self) -> None:
        for size in (1, 2, 3, 4, 5, 8, 13, 32, 64):
            for schedule in (
                schedules.sequential_chain(size),
                schedules.balanced_pairwise(size),
            ):
                with self.subTest(kind=schedule.kind, size=size):
                    schedules.check_schedule(schedule)
                    if size > 1:  # a single block has no merge at all
                        self.assertEqual(_leaf_groups(schedule)[-1], frozenset(range(size)))
        for size in (1, 2, 4, 8, 32):
            for schedule in (
                schedules.warp_shfl_xor(size),
                schedules.warp_shfl_down(size),
            ):
                with self.subTest(kind=schedule.kind, size=size):
                    schedules.check_schedule(schedule)
                    if size > 1:  # a single block has no merge at all
                        self.assertEqual(_leaf_groups(schedule)[-1], frozenset(range(size)))
        for size, splits in ((8, 2), (8, 4), (13, 3), (64, 8)):
            schedule = schedules.split_k(size, splits)
            with self.subTest(kind=schedule.kind):
                schedules.check_schedule(schedule)
                self.assertEqual(_leaf_groups(schedule)[-1], frozenset(range(size)))

    def test_check_rejects_malformed_schedules(self) -> None:
        cases = {
            "wrong node count": schedules.Schedule(4, ((0, 1), (2, 3)), "bad"),
            "child used twice": schedules.Schedule(3, ((0, 1), (0, 3)), "bad"),
            "forward reference": schedules.Schedule(3, ((0, 4), (1, 3)), "bad"),
        }
        for name, schedule in cases.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                schedules.check_schedule(schedule)

    def test_warp_shuffles_are_confined_to_one_warp(self) -> None:
        """A shuffle reduction cannot span warps, so 64 lanes is not expressible."""
        for size in (0, 3, 6, 33, 64, 128):
            with self.subTest(size=size):
                with self.assertRaises(ValueError):
                    schedules.warp_shfl_xor(size)
                with self.assertRaises(ValueError):
                    schedules.warp_shfl_down(size)


class ShapeTests(unittest.TestCase):
    def test_chain_is_as_deep_as_it_can_be(self) -> None:
        for size in (2, 5, 16, 64):
            self.assertEqual(schedules.sequential_chain(size).depth, size - 1)

    def test_balanced_and_warp_shuffles_are_logarithmic(self) -> None:
        for size in (2, 4, 8, 32):
            expected = int(math.log2(size))
            self.assertEqual(schedules.balanced_pairwise(size).depth, expected)
            self.assertEqual(schedules.warp_shfl_xor(size).depth, expected)
            self.assertEqual(schedules.warp_shfl_down(size).depth, expected)

    def test_ascending_xor_butterfly_is_the_balanced_tree(self) -> None:
        """With ascending masks the butterfly moves data without changing the pairing."""
        for size in (2, 4, 8, 32):
            self.assertEqual(
                schedules.warp_shfl_xor(size).nodes, schedules.balanced_pairwise(size).nodes
            )

    def test_shfl_down_pairs_strided_lanes_not_adjacent_ones(self) -> None:
        """Same depth as the xor butterfly, different pairing, so not interchangeable."""
        for size in (4, 8, 32):
            with self.subTest(size=size):
                down = schedules.warp_shfl_down(size)
                self.assertNotEqual(down.nodes, schedules.warp_shfl_xor(size).nodes)
                self.assertEqual(down.nodes[0], (0, size // 2))
        self.assertEqual(
            _leaf_groups(schedules.warp_shfl_down(4))[:2],
            [frozenset({0, 2}), frozenset({1, 3})],
        )

    def test_single_split_degenerates_to_the_chain(self) -> None:
        for size in (2, 7, 32):
            self.assertEqual(
                schedules.split_k(size, 1).nodes, schedules.sequential_chain(size).nodes
            )

    def test_split_k_is_shallower_than_the_plain_chain(self) -> None:
        for size, splits in ((16, 2), (16, 4), (64, 8)):
            with self.subTest(size=size, splits=splits):
                self.assertLess(
                    schedules.split_k(size, splits).depth,
                    schedules.sequential_chain(size).depth,
                )

    def test_split_k_groups_are_contiguous_blocks(self) -> None:
        schedule = schedules.split_k(8, 4)
        groups = _leaf_groups(schedule)
        self.assertIn(frozenset({0, 1}), groups)
        self.assertIn(frozenset({2, 3}), groups)
        self.assertIn(frozenset({4, 5}), groups)
        self.assertIn(frozenset({6, 7}), groups)

    def test_split_k_rejects_impossible_launches(self) -> None:
        for size, splits in ((8, 0), (8, 9), (4, 5)):
            with self.subTest(size=size, splits=splits), self.assertRaises(ValueError):
                schedules.split_k(size, splits)


if __name__ == "__main__":
    unittest.main()
