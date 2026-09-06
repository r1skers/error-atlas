"""Storage boundaries and output error bookkeeping, independent of research inputs."""

from fractions import Fraction as F
import random
import unittest

import numpy as np

from online import output_measure as measure
from online.output_reference import ComputedOutput, OutputReference
from online.pilot_runner import from_word


class StorageCastTests(unittest.TestCase):
    def test_fp16_matches_numpy_on_fp32_bit_samples(self):
        rng = random.Random(2048)
        for _ in range(500):
            word = rng.randrange(0x40000000) | (rng.randrange(2) << 31)
            if word == 0x80000000:
                continue
            value = from_word(f"{word:08x}")
            result, bits = measure.storage_cast(value, "fp16")
            expected = F(float(np.float16(float(value))))
            self.assertEqual(result, expected)
            if expected == 0:
                self.assertEqual(bits, "0000")

    def test_bf16_even_odd_ties_and_midpoint_neighbors(self):
        quantum, small = F(1, 128), F(1, 2**23)
        for lower, tie in ((F(1), F(1)), (1 + quantum, 1 + 2 * quantum)):
            midpoint = lower + quantum / 2
            for sign in (1, -1):
                self.assertEqual(measure.storage_cast(sign * midpoint, "bf16")[0], sign * tie)
                self.assertEqual(measure.storage_cast(sign * (midpoint - small), "bf16")[0], sign * lower)
                self.assertEqual(measure.storage_cast(sign * (midpoint + small), "bf16")[0], sign * (lower + quantum))

    def test_gradual_underflow_and_canonical_zero(self):
        for dtype, exponent in (("fp16", -24), ("bf16", -133)):
            quantum = F(1, 2**(-exponent))
            delta = max(F(1, 2**149), quantum / 2 / 2**23)
            for sign in (1, -1):
                self.assertEqual(measure.storage_cast(sign * quantum / 2, dtype), (0, "0000"))
                self.assertEqual(measure.storage_cast(sign * (quantum / 2 + delta), dtype)[0], sign * quantum)
                self.assertEqual(measure.storage_cast(sign * quantum, dtype)[0], sign * quantum)

    def test_tiny_fp32_difference_can_change_low_precision_output(self):
        for dtype, exponent in (("fp16", -11), ("bf16", -8)):
            middle = 1 + F(1, 2**(-exponent))
            left, right = middle - F(1, 2**23), middle + F(1, 2**23)
            self.assertNotEqual(measure.storage_cast(left, dtype)[1], measure.storage_cast(right, dtype)[1])

    def test_cast_contract_rejects_non_fp32_out_of_range_and_unknown_format(self):
        for value, dtype in ((F(1, 3), "fp16"), (F(3), "bf16"), (F(1), "unknown")):
            with self.assertRaises(ValueError):
                measure.storage_cast(value, dtype)


class BookkeepingTests(unittest.TestCase):
    def test_signed_errors_and_counterfactuals(self):
        computed = ComputedOutput(F(2), F(4), F(1, 2), F(1, 2), F(0))
        ref = OutputReference((F(1), F(1)), (F(4), F(4)), (F(1, 4), F(1, 4)))
        row = measure.record(computed, ref, F(1))
        self.assertEqual(row["delta_O"], (1, 1))
        self.assertEqual(row["delta_ell"], (0, 0))
        self.assertEqual(row["only_denominator"], (F(1, 4), F(1, 4)))
        self.assertEqual(row["only_numerator"], (F(1, 2), F(1, 2)))
        for entry in row["formats"].values():
            self.assertEqual(entry["abs_error"], (F(1, 4), F(1, 4)))
            self.assertEqual(entry["normalized_abs_error"], entry["abs_error"])
        self.assertTrue(all(v["D"] == (0, 0) for v in measure.paired(row, row).values()))

    def test_zero_value_scale_does_not_divide_by_zero(self):
        computed = ComputedOutput(F(0), F(4), F(0), F(0), F(0))
        ref = OutputReference((F(0), F(0)), (F(4), F(4)), (F(0), F(0)))
        row = measure.record(computed, ref, F(0))
        self.assertTrue(all(v["normalized_abs_error"] is None and v["abs_error"] == (0, 0) for v in row["formats"].values()))


if __name__ == "__main__":
    unittest.main()
