"""Audit the delivered attribution implementation with in-memory mutations.

Run from the repository root. No research measurements or source-file mutations occur.
Each replacement must match exactly once; a changed implementation requires an explicit
audit update, not silently testing an unrelated mutant. Optional JSON output is exclusive.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "topics/softmax/experiments"
TEST_DIR = ROOT / "topics/softmax/tests"

MUTATIONS = (
    ("M1_abs_R", "R = computed - frozen", "R = abs(computed - frozen)"),
    ("M2_net_instead_of_budget", "sum((abs(r) for r in residuals), Fraction(0))",
     "abs(sum(residuals, Fraction(0)))"),
    ("M3_reverse_weight_interval", "W = (frozen - U, frozen - L)",
     "W = (frozen - L, frozen - U)"),
    ("M4_negative_R_wrong_endpoints", "else:\n        relative_rounding = (R / L, R / U)",
     "else:\n        relative_rounding = (R / U, R / L)"),
    ("M5_discard_shared_reference", "relative_weight = (frozen/U - 1, frozen/L - 1)",
     "relative_weight = (min(w / d for w in W for d in reference), max(w / d for w in W for d in reference))"),
    ("M6_ignore_broken_identity", "if sum(residuals, Fraction(0)) != R:", "if False:"),
    ("M7_add_instead_of_cancel", "abs(rounding_error) + abs(weight) - abs(rounding_error + weight)",
     "abs(rounding_error) + abs(weight) + abs(rounding_error + weight)"),
    ("M8_forget_cancellation_saturation", "abs(rounding_error) + abs(weight) - abs(rounding_error + weight)",
     "2 * abs(weight) if rounding_error * weight < 0 else Fraction(0)"),
    ("M9_frozen_as_real_reference", "reference = denominator_interval(family)",
     "reference = (frozen, frozen)"),
)


def run_tests():
    suite = unittest.TestLoader().discover(str(TEST_DIR), pattern="test_online_attribution.py",
                                         top_level_dir=str(TEST_DIR))
    return unittest.TextTestRunner(stream=io.StringIO()).run(suite)


def audit() -> dict:
    sys.path.insert(0, str(SOURCE_DIR))
    from online import attribution

    path = SOURCE_DIR / "online/attribution.py"
    source = path.read_text(encoding="utf-8")
    baseline = run_tests()
    if not baseline.wasSuccessful() or baseline.skipped or baseline.testsRun == 0:
        raise RuntimeError("Baseline must pass real core tests with no skips before mutation")
    original = {name: getattr(attribution, name) for name in ("decompose", "cancellation_savings")}
    records = []
    try:
        for name, old, new in MUTATIONS:
            if source.count(old) != 1:
                raise ValueError(f"{name}: expected exactly one source anchor")
            tree = ast.parse(source.replace(old, new, 1), filename=str(path))
            functions = [node for node in tree.body
                         if isinstance(node, ast.FunctionDef) and node.name in original]
            # Keep the original globals so test mocks of weighted_residuals/reference
            # still affect the delivered functions. All changes live in this process.
            exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), attribution.__dict__)
            result = run_tests()
            records.append({"name": name, "killed": bool(result.failures or result.errors),
                            "tests_run": result.testsRun, "skips": len(result.skipped),
                            "failed_tests": [test.id() for test, _ in result.failures],
                            "errored_tests": [test.id() for test, _ in result.errors]})
            for function, value in original.items():
                setattr(attribution, function, value)
    finally:
        for function, value in original.items():
            setattr(attribution, function, value)
    return {"stage": "implementation_audit_not_research_measurement",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "tests_sha256": hashlib.sha256((TEST_DIR / "test_online_attribution.py").read_bytes()).hexdigest(),
            "audit_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "baseline_passed": baseline.testsRun, "mutations": records,
            "all_killed": all(r["killed"] and not r["skips"] for r in records)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit()
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    print(f"Baseline: {report['baseline_passed']} passed, no skips")
    for row in report["mutations"]:
        print(f"{row['name']}: {'KILLED' if row['killed'] else 'SURVIVED'} "
              f"({len(row['failed_tests'])} failures, {len(row['errored_tests'])} errors)")
    return 0 if report["all_killed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
