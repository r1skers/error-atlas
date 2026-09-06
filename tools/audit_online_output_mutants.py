"""Audit delivered output cores/casts with in-memory mutations; source stays intact."""

import inspect
import io
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "topics/softmax/experiments"), str(ROOT / "topics/softmax/tests")]
from online import output_probe, output_reference, output_measure
import test_online_output_probe
import test_online_output_reference
import test_online_output_measure


def main():
    modules = (test_online_output_probe, test_online_output_reference, test_online_output_measure)
    mutations = (
        ("negative_endpoint_order", output_reference, "numerator_interval", "low += min(endpoint_a, endpoint_b)", "low += max(endpoint_a, endpoint_b)"),
        ("omit_block_mass", output_reference, "numerator_interval", "Fraction(n) * value", "value"),
        ("rounded_reference_gap", output_reference, "numerator_interval", "real_exp_interval(m - M)", "real_exp_interval(round_to_fp32(m - M))"),
        ("collapse_upper_bound", output_reference, "numerator_interval", "high += max(endpoint_a, endpoint_b)", "high += min(endpoint_a, endpoint_b)"),
        ("float_quotient", output_reference, "quotient_interval", "return min(corners), max(corners)", "return Fraction(float(min(corners))), Fraction(float(max(corners)))"),
        ("omit_division_rounding", output_reference, "finish_output", "value = round_to_fp32(ratio)", "value = ratio"),
        ("wrong_fma_operand", output_probe, "propagate_numerator", "fp32_fma(o_b, w_b, o_a)", "fp32_fma(o_a, w_a, o_b)"),
        ("bf16_tie_parity", output_measure, "storage_cast", "0x7fff + ((bits >> 16) & 1)", "0x7fff"),
    )
    survivors = []
    for label, module, name, before, after in mutations:
        original = getattr(module, name)
        source = inspect.getsource(original)
        if source.count(before) != 1:
            raise ValueError(f"Mutation anchor changed: {label}")
        try:
            exec(compile(source.replace(before, after), f"<mutant:{label}>", "exec"), vars(module))
            suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(m) for m in modules)
            result = unittest.TextTestRunner(stream=io.StringIO()).run(suite)
        finally:
            setattr(module, name, original)
        killed = [case.id().split(".")[-1] for case, _ in result.failures + result.errors]
        print(f"{label}: {'KILLED' if killed else 'SURVIVED'}; {', '.join(killed)}")
        if not killed:
            survivors.append(label)
    if survivors:
        raise SystemExit(f"Surviving mutants: {survivors}")
    print(f"{len(mutations)}/{len(mutations)} killed; source files unchanged")


if __name__ == "__main__":
    main()
