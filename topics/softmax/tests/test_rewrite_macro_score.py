"""Replication tests for the closed-book Q_8/12 macro score against frozen v2 evidence.

Skips while the rewrite raises NotImplementedError. Once implemented, the rewritten
generators + oracle + macro score must reproduce, for every frozen v2 held-out group,
the CSV columns fixed_q_score, fixed_energy_capture, shortlisted and q_selected.
The trained-probe beam is out of scope here; beam_selected is not recomputed.
"""

import csv
import unittest
from collections import defaultdict
from pathlib import Path

from rewrite import macro_score
from rewrite.generators import (
    random_contiguous_split_tree,
    random_pair_merge_tree,
    wide_range_random,
)

HELDOUT = (
    Path(__file__).resolve().parents[1]
    / "experiments/results/wide_range_fixed_k8_beam_v2/heldout"
)
OBSERVATIONS = HELDOUT / "graph_observations.csv"


def _skip_unless_implemented(test: unittest.TestCase, call) -> None:
    try:
        call()
    except NotImplementedError:
        test.skipTest("rewrite not implemented yet")


def _tree_for_row(width: int, family: str, graph_seed: int):
    if family == "contiguous":
        return random_contiguous_split_tree(width, seed=graph_seed)
    return random_pair_merge_tree(width, seed=graph_seed)


class MacroScoreReplicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows = list(csv.DictReader(OBSERVATIONS.open()))
        cls.by_group = defaultdict(list)
        for row in rows:
            cls.by_group[row["input_group_id"]].append(row)
        # Deterministic order within each group.
        for group in cls.by_group.values():
            group.sort(key=lambda r: int(r["tree_index"]))

    def setUp(self):
        tree = random_contiguous_split_tree(4, seed=0)
        _skip_unless_implemented(
            self,
            lambda: macro_score.q_macro_score(
                wide_range_random(4, seed=0), tree
            ),
        )

    def test_reproduces_frozen_q_score_capture_shortlist_and_selection(self):
        checked_groups = 0
        for group_id, group in self.by_group.items():
            width = int(group[0]["width"])
            seed = int(group[0]["seed"])
            values = wide_range_random(width, seed=seed)
            q_scores = []
            for row in group:
                tree = _tree_for_row(width, row["graph_family"], int(row["graph_seed"]))
                q_score, capture = macro_score.q_macro_score(values, tree)
                with self.subTest(group=group_id, tree=row["tree_index"], col="q"):
                    self.assertEqual(q_score, float(row["fixed_q_score"]))
                with self.subTest(group=group_id, tree=row["tree_index"], col="capture"):
                    self.assertEqual(capture, float(row["fixed_energy_capture"]))
                q_scores.append(q_score)

            shortlist = set(macro_score.shortlist_indices(q_scores))
            for row in group:
                with self.subTest(group=group_id, tree=row["tree_index"], col="shortlisted"):
                    self.assertEqual(
                        int(row["tree_index"]) in shortlist, row["shortlisted"] == "1"
                    )

            q_selected = macro_score.shortlist_indices(q_scores, size=1)[0]
            frozen_q_selected = {
                int(r["tree_index"]) for r in group if r["q_selected"] == "1"
            }
            with self.subTest(group=group_id, col="q_selected"):
                self.assertEqual({q_selected}, frozen_q_selected)
            checked_groups += 1
        self.assertEqual(checked_groups, 192)


if __name__ == "__main__":
    unittest.main()
