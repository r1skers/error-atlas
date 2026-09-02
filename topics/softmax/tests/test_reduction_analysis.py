"""Characterization and invariants for shared-trace extraction, not new research evidence."""

import contextlib
import dataclasses
import io
import json
import math
import sys
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import predictor_wide_range_ac_decomposition as legacy_ac
import predictor_wide_range_ancestor_history_decomposition as legacy_history
import predictor_wide_range_coherence_structure as legacy_structure
from reduction_analysis import CoherenceAnalysis, replay
from reduction_analysis.topology import TreeTopology
from summation_graph_predictor import (
    AdditionNode,
    BinaryReductionGraph,
    balanced_reduction_graph,
    predict_fp32_tree_error,
)

FIXTURE = Path(__file__).parent / "fixtures/coherence_pre_refactor.json"


def _encode(value):
    """Preserve exact fractions, float bit patterns and NaN explicitly in JSON."""
    if isinstance(value, Fraction):
        return {"fraction": str(value)}
    if isinstance(value, float):
        return {"float_hex": value.hex()}
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(item) for item in value]
    return value


def _case(record):
    graph = record["graph"]
    return (
        tuple(Fraction(value) for value in record["values"]),
        BinaryReductionGraph(
            name=graph["name"],
            leaf_count=graph["leaf_count"],
            nodes=tuple(AdditionNode(**node) for node in graph["nodes"]),
            root=graph["root"],
        ),
    )


class CharacterizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_shared_views_match_all_pre_refactor_fields(self):
        for record in self.fixture["cases"]:
            values, graph = _case(record)
            analysis = CoherenceAnalysis(
                replay(values, graph), record["graph_family"], record["graph_seed"]
            )
            for view in ("ac", "structure", "history"):
                with self.subTest(case=record["name"], view=view):
                    actual = _encode(dataclasses.asdict(getattr(analysis, view)))
                    self.assertEqual(actual, record["expected"][view])

    def test_legacy_function_signatures_and_results_remain_compatible(self):
        for record in self.fixture["cases"]:
            values, graph = _case(record)
            labels = {
                "graph_family": record["graph_family"],
                "graph_seed": record["graph_seed"],
            }
            actual = {
                "ac": legacy_ac.diagnose_tree(values, graph, **labels),
                "structure": legacy_structure.diagnose_tree(values, graph, **labels),
                "history": legacy_history.diagnose(
                    values, graph, record["graph_family"]
                ),
            }
            for view, result in actual.items():
                with self.subTest(case=record["name"], view=view):
                    self.assertEqual(
                        _encode(dataclasses.asdict(result)), record["expected"][view]
                    )

    def test_legacy_cli_output_and_seed_schedules_match(self):
        for name, module in (
            ("ac", legacy_ac),
            ("structure", legacy_structure),
            ("history", legacy_history),
        ):
            record = self.fixture["cli"][name]
            output = io.StringIO()
            with self.subTest(cli=name):
                with patch.object(sys, "argv", [name, *record["argv"]]):
                    with contextlib.redirect_stdout(output):
                        code = module.main()
                self.assertEqual(code, record["exit_code"])
                self.assertEqual(output.getvalue(), record["stdout"])

    def test_trace_preserves_node_and_root_residual_identities(self):
        for record in self.fixture["cases"]:
            values, graph = _case(record)
            trace = replay(values, graph)
            with self.subTest(case=record["name"]):
                self.assertEqual(
                    trace.prediction.signed_error, sum(trace.deltas, Fraction(0))
                )
                self.assertEqual(
                    trace.value_at(graph.root) - sum(values, Fraction(0)),
                    trace.prediction.signed_error,
                )
                for node in trace.nodes:
                    self.assertEqual(
                        trace.value_at(node.left) + trace.value_at(node.right),
                        node.exact_addend_sum,
                    )
                    self.assertEqual(
                        node.rounded_sum - node.exact_addend_sum,
                        node.local_rounding_error,
                    )


class TraceCompositionTests(unittest.TestCase):
    def setUp(self):
        self.values = (
            Fraction(1),
            Fraction(1, 2**24),
            Fraction(1, 2**25),
            Fraction(1, 2**25),
        )
        self.graph = balanced_reduction_graph(4)

    def test_three_views_share_one_oracle_call_and_one_topology(self):
        with patch(
            "reduction_analysis.trace.predict_fp32_tree_error",
            wraps=predict_fp32_tree_error,
        ) as oracle:
            with patch.object(
                TreeTopology, "from_graph", wraps=TreeTopology.from_graph
            ) as topology:
                trace = replay(self.values, self.graph)
                analysis = CoherenceAnalysis(trace)
                energy = analysis.ac
                topology.assert_not_called()
                self.assertIs(analysis.ac, energy)
                self.assertIs(analysis.structure, analysis.structure)
                self.assertIs(analysis.history, analysis.history)
                self.assertIs(trace.topology, trace.topology)
                oracle.assert_called_once()
                topology.assert_called_once_with(self.graph)

    def test_analysis_views_do_not_execute_oracle(self):
        analysis = CoherenceAnalysis(replay(self.values, self.graph))
        with patch(
            "reduction_analysis.trace.predict_fp32_tree_error",
            side_effect=AssertionError("analysis must not replay"),
        ):
            self.assertEqual(
                analysis.ac.e2, analysis.ac.a_local + analysis.ac.c_coherence
            )
            self.assertEqual(analysis.history.c_total, analysis.ac.c_coherence)
            self.assertEqual(analysis.history.c_ancestor, analysis.history.k_total)
            self.assertEqual(analysis.structure.c_total, float(analysis.ac.c_coherence))

    def test_replay_copies_mutable_input_sequence(self):
        values = list(self.values)
        trace = replay(values, self.graph)
        values[0] = Fraction(2)
        self.assertEqual(trace.values, self.values)

    def test_oracle_input_contract_is_not_widened(self):
        bad_inputs = (
            (self.values[:-1], ValueError),
            ((Fraction(-1), *self.values[1:]), ValueError),
            ((Fraction(1, 3), *self.values[1:]), ValueError),
            ((1.0, *self.values[1:]), TypeError),
        )
        for values, error in bad_inputs:
            with self.subTest(values=values):
                with self.assertRaises(error):
                    replay(values, self.graph)

    def test_value_lookup_rejects_out_of_range_ids(self):
        trace = replay(self.values, self.graph)
        for index in (-1, 7):
            with self.assertRaises(IndexError):
                trace.value_at(index)

    def test_single_leaf_has_zero_energy_and_empty_pair_partitions(self):
        analysis = CoherenceAnalysis(
            replay((Fraction(1),), balanced_reduction_graph(1))
        )
        self.assertEqual(analysis.trace.nodes, ())
        self.assertEqual(analysis.ac.e2, 0)
        self.assertTrue(math.isnan(analysis.ac.c_over_a))
        self.assertEqual(analysis.structure.abs_pair_mass, 0.0)
        self.assertEqual(analysis.history.c_ancestor, 0)

    def test_topology_depth_and_proper_ancestor_gap(self):
        topology = TreeTopology.from_graph(self.graph)
        self.assertEqual(topology.depth, (2, 2, 2, 2, 1, 1, 0))
        self.assertEqual(topology.ancestor_gap(4, 6), 1)
        self.assertEqual(topology.ancestor_gap(6, 0), 2)
        self.assertIsNone(topology.ancestor_gap(4, 5))
        self.assertIsNone(topology.ancestor_gap(6, 6))
        for pair in ((-1, 0), (0, 7)):
            with self.assertRaises(IndexError):
                topology.ancestor_gap(*pair)


if __name__ == "__main__":
    unittest.main()
