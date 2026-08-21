"""Tests for the first-order history failure diagnostic helpers."""

import unittest

from predictor_first_order_history_failure_diagnostic import (
    NodeRow,
    TreeRow,
    _selection_quality,
)


def _node(index: int, gap: float, *, ulp: float = 1.0) -> NodeRow:
    return NodeRow(
        index=index,
        depth=index,
        actual_history_ulp=0.0,
        predicted_history_ulp=0.0,
        actual_gap_ulp=gap,
        predicted_gap_ulp=gap,
        actual_cell_change=False,
        predicted_cell_change=False,
        ulp=ulp,
    )


class FirstOrderHistoryFailureDiagnosticTests(unittest.TestCase):
    def test_selection_quality_reports_overlap_and_mass(self) -> None:
        tree = TreeRow(
            family="test",
            nodes=(_node(1, 4.0), _node(2, 3.0), _node(3, 2.0), _node(4, 1.0)),
            orders={
                "oracle_gap": (1, 2, 3, 4),
                "depth": (2, 3, 4, 1),
            },
        )

        quality = _selection_quality(tree, "depth", 2)

        self.assertEqual(quality.recall, 0.5)
        self.assertAlmostEqual(quality.mass_recovery, 5.0 / 7.0)

    def test_selection_quality_caps_budget_at_node_count(self) -> None:
        tree = TreeRow(
            family="test",
            nodes=(_node(1, 2.0), _node(2, 1.0)),
            orders={"oracle_gap": (1, 2), "depth": (2, 1)},
        )

        quality = _selection_quality(tree, "depth", 8)

        self.assertEqual(quality.recall, 1.0)
        self.assertEqual(quality.mass_recovery, 1.0)

    def test_selection_mass_uses_raw_error_units(self) -> None:
        tree = TreeRow(
            family="test",
            nodes=(
                _node(1, 4.0),
                _node(2, 3.0, ulp=2.0),
                _node(3, 1.0),
            ),
            orders={"oracle_gap": (2, 1, 3), "depth": (1, 3, 2)},
        )

        quality = _selection_quality(tree, "depth", 1)

        self.assertEqual(quality.recall, 0.0)
        self.assertAlmostEqual(quality.mass_recovery, 4.0 / 6.0)


if __name__ == "__main__":
    unittest.main()
