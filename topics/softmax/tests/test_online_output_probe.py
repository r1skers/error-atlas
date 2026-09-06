"""V semantics and independent invariants for the next user-written O core."""

import unittest
from fractions import Fraction as F

from online import output_probe as probe
from online.merge import merge_reduce
from online.pilot import BlockFamily
from online.schedules import sequential_chain, balanced_pairwise


class ProbeScaffoldingTests(unittest.TestCase):
    def setUp(self):
        self.family = BlockFamily((F(-2), F(0), F(0), F(-1)), (32,) * 4)

    def test_patterns_are_values_not_normalized_probabilities(self):
        self.assertEqual(probe.probe_values(self.family, "one"), (1, 1, 1, 1))
        self.assertEqual(probe.probe_values(self.family, "zero"), (0, 0, 0, 0))
        self.assertEqual(probe.probe_values(self.family, "alternating_sign"), (1, -1, 1, -1))
        self.assertEqual(probe.leaf_numerators(self.family, (F(1), F(-1), F(0), F(1, 2))),
                         (32, -32, 0, 16))

    def test_indicator_choice_uses_original_order_with_first_max_tie_break(self):
        self.assertEqual(probe.probe_values(self.family, "max_block"), (0, 1, 0, 0))
        self.assertEqual(probe.probe_values(self.family, "first_block"), (1, 0, 0, 0))

    def test_all_patterns_keep_input_unchanged_and_local_leaf_initialization(self):
        before = self.family
        for name in probe.PROBE_NAMES:
            values = probe.probe_values(self.family, name)
            self.assertEqual(probe.leaf_numerators(self.family, values), tuple(32 * v for v in values))
        self.assertEqual(self.family, before)

    def test_unknown_mode_bad_value_and_wrong_leaf_count_rejected(self):
        with self.assertRaises(ValueError):
            probe.probe_values(self.family, "uniform_probabilities")
        with self.assertRaises(ValueError):
            probe.leaf_numerators(self.family, (F(1, 3),) * 4)
        with self.assertRaises(ValueError):
            probe.leaf_numerators(self.family, (F(1),))

    def test_core_guards_run_before_pending_implementation(self):
        with self.assertRaises(ValueError):
            probe.propagate_numerator(None, ())
        dump = merge_reduce(self.family.leaf_max, self.family.leaf_ell, sequential_chain(4))
        for leaves in ((F(1),), (F(1, 3),) * 4):
            with self.assertRaises(ValueError):
                probe.propagate_numerator(dump, leaves)


class NumeratorCoreTests(unittest.TestCase):
    def test_one_leaf_has_no_internal_nodes(self):
        dump = merge_reduce((F(0),), (F(32),), sequential_chain(1))
        self.assertEqual(probe.propagate_numerator(dump, (F(-32),)), ())

    def test_constant_values_match_denominator_node_by_node(self):
        family = BlockFamily(tuple(map(F, (-3, 0, -1, -2, -5))), (32,) * 5)
        for tree in (sequential_chain(5), balanced_pairwise(5)):
            for fused in (False, True):
                dump = merge_reduce(family.leaf_max, family.leaf_ell, tree, fused=fused)
                for value in (F(0), F(1), F(-1), F(1, 2)):
                    with self.subTest(kind=tree.kind, fused=fused, value=value):
                        leaves = probe.leaf_numerators(family, (value,) * 5)
                        self.assertEqual(probe.propagate_numerator(dump, leaves),
                                         tuple(value * ell for ell in dump.node_ell))

    def test_signed_children_and_saved_nonstandard_weights(self):
        # Test-only exp subject: exp(-1) returns 1/4. Recomputing real exp is wrong.
        def subject(gap):
            return F(1) if gap == 0 else F(1, 4)
        for fused in (False, True):
            for maxima, leaves in (((F(0), F(-1)), (F(32), F(-32))),
                                    ((F(-1), F(0)), (F(-32), F(32)))):
                dump = merge_reduce(maxima, (F(32), F(32)), sequential_chain(2), subject, fused=fused)
                self.assertEqual(probe.propagate_numerator(dump, leaves), (F(24),))

    def test_indicator_output_depends_on_logits_even_with_equal_masses(self):
        dump = merge_reduce((F(0), F(-1)), (F(32), F(32)), sequential_chain(2))
        result = probe.propagate_numerator(dump, (F(32), F(0)))
        self.assertEqual(result, (F(32),))
        self.assertGreater(result[0] / dump.node_ell[0], F(1, 2))

    def test_signed_cancellation_and_tree_child_indices(self):
        for tree in (sequential_chain(4), balanced_pairwise(4)):
            dump = merge_reduce((F(0),) * 4, (F(32),) * 4, tree)
            result = probe.propagate_numerator(dump, (F(32), F(-32), F(16), F(-16)))
            self.assertEqual(result[-1], 0)
            self.assertEqual(len(result), 3)

    def test_fma_boundary_is_not_silently_evaluated_as_two_roundings(self):
        maxima = (F(0), F(-2))
        leaves = (F(2**24), F(7747987, 1048576))
        for reverse in (False, True):
            for fused in (False, True):
                m = tuple(reversed(maxima)) if reverse else maxima
                o = tuple(reversed(leaves)) if reverse else leaves
                dump = merge_reduce(m, (F(32), F(32)), sequential_chain(2), fused=fused)
                self.assertEqual(probe.propagate_numerator(dump, o), (F(2**24 + (2 if fused else 0)),))


if __name__ == "__main__":
    unittest.main()
