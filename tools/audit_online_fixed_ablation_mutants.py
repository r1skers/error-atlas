"""Mutate the delivered preparation function in memory; never edit source files."""

import ast
import io
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "topics/softmax/experiments"))
sys.path.insert(0, str(ROOT / "topics/softmax/tests"))
from online import fixed_contribution_ablation as core
import test_online_fixed_contribution_ablation as tests


def main():
    source = Path(core.__file__).read_text(encoding="utf-8")
    function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "prepare")
    original = ast.unparse(function)
    mutations = {
        "rounded_logit_gap": ("real_exp_interval(logit - maximum)", "real_exp_interval(round_to_fp32(logit - maximum))"),
        "round_exp_before_mass": ("low, high = (mass * low, mass * high)",
                                  "low, high = (mass * round_to_fp32(low), mass * round_to_fp32(high))"),
        "omit_mass": ("low, high = (mass * low, mass * high)", "low, high = (low, high)"),
        "accept_ambiguous": ("if left != right:", "if False:"),
        "reverse_leaves": ("Preparation(tuple(values),", "Preparation(tuple(reversed(values)),"),
        "drop_underflow_zeros": ("values.append(left)", "values.extend([left] if left else [])"),
    }
    survived = []
    for name, (before, after) in mutations.items():
        if original.count(before) != 1:
            raise ValueError(f"Mutation anchor changed: {name}")
        saved = core.prepare
        try:
            # Preserve actual module globals so test patches still reach the mutant.
            exec(compile(original.replace(before, after), f"<mutant:{name}>", "exec"), vars(core))
            result = unittest.TextTestRunner(stream=io.StringIO()).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
        finally:
            core.prepare = saved
        killers = [case.id().split(".")[-1] for case, _ in result.failures + result.errors]
        print(f"{name}: {'KILLED' if killers else 'SURVIVED'}; {', '.join(killers)}")
        if not killers:
            survived.append(name)
    if survived:
        raise SystemExit(f"Surviving mutants: {survived}")
    print(f"{len(mutations)}/{len(mutations)} killed; delivered files unchanged")


if __name__ == "__main__":
    main()
