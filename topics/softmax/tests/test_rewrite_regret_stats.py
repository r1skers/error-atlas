"""Independent replication of the frozen v2 headline from the rewrite oracle.

Skips while the rewrite raises NotImplementedError. Recomputes, per frozen v2 group,
each tree's target from the rewritten oracle, the Q-selected tree from the rewritten
macro score, and the beam-selected tree from the frozen CSV (beam is out of scope).
Then it recomputes per-group normalized regret, the paired improvement, and the
width-stratified group bootstrap CI, and checks them against the frozen
metric_summary.json overall block.
"""

import csv
import json
import unittest
from collections import defaultdict
from pathlib import Path

from rewrite.fp32_oracle import reduce_tree
from rewrite.generators import (
    random_contiguous_split_tree,
    random_pair_merge_tree,
    wide_range_random,
)
from rewrite.macro_score import q_macro_score, shortlist_indices, ulp_fraction
from rewrite import regret_stats

HELDOUT = (
    Path(__file__).resolve().parents[1]
    / "experiments/results/wide_range_fixed_k8_beam_v2/heldout"
)
OBSERVATIONS = HELDOUT / "graph_observations.csv"
SUMMARY = HELDOUT / "metric_summary.json"
WIDTH_ORDER = (256, 512, 1024)


def _tree(width, family, graph_seed):
    if family == "contiguous":
        return random_contiguous_split_tree(width, seed=graph_seed)
    return random_pair_merge_tree(width, seed=graph_seed)


def _skip_unless_implemented(test, call):
    try:
        call()
    except NotImplementedError:
        test.skipTest("rewrite not implemented yet")


class HeadlineReplicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.overall = json.loads(SUMMARY.read_text())["overall"]
        rows = list(csv.DictReader(OBSERVATIONS.open()))
        groups = defaultdict(list)
        for row in rows:
            groups[row["input_group_id"]].append(row)
        for group in groups.values():
            group.sort(key=lambda r: int(r["tree_index"]))
        cls.groups = groups

    def setUp(self):
        _skip_unless_implemented(
            self, lambda: regret_stats.normalized_regret([1.0, 2.0], 0)
        )
        _skip_unless_implemented(
            self, lambda: regret_stats.paired_improvement([1.0], [1.0])
        )

    def _recompute_rows(self):
        # Group order must match the frozen runner: widths 256, 512, 1024, groups g000..
        ordered_ids = sorted(
            self.groups,
            key=lambda gid: (WIDTH_ORDER.index(int(self.groups[gid][0]["width"])), gid),
        )
        rows = []
        for gid in ordered_ids:
            group = self.groups[gid]
            width = int(group[0]["width"])
            seed = int(group[0]["seed"])
            values = wide_range_random(width, seed=seed)
            targets = []
            q_scores = []
            beam_selected = None
            for row in group:
                tree = _tree(width, row["graph_family"], int(row["graph_seed"]))
                trace = reduce_tree(values, tree)
                root_ulp = ulp_fraction(trace.exact_sum)
                # The frozen CSV squared the exact rational ratio and rounded once
                # (float((E/ulp)**2)); float(E/ulp)**2 rounds twice and differs by
                # 1 ULP on rare rows. Reproduce the frozen convention exactly.
                targets.append(float((trace.error / root_ulp) ** 2))
                q_scores.append(q_macro_score(values, tree)[0])
                if row["beam_selected"] == "1":
                    beam_selected = int(row["tree_index"])
            q_selected = shortlist_indices(q_scores, size=1)[0]
            rows.append(
                {
                    "width": width,
                    "q_regret": regret_stats.normalized_regret(targets, q_selected),
                    "beam_regret": regret_stats.normalized_regret(targets, beam_selected),
                }
            )
        return rows

    def test_recomputed_primary_and_ci_match_frozen_overall(self):
        rows = self._recompute_rows()
        self.assertEqual(len(rows), 192)
        primary = regret_stats.paired_improvement(
            [r["q_regret"] for r in rows], [r["beam_regret"] for r in rows]
        )
        self.assertEqual(primary, self.overall["primary_fixed_q_minus_beam_regret"])

        low, high = regret_stats.stratified_group_bootstrap_ci(
            rows, lambda r: r["q_regret"] - r["beam_regret"]
        )
        self.assertEqual([low, high], self.overall["primary_95_ci"])
        self.assertTrue(low > 0.0)


if __name__ == "__main__":
    unittest.main()
