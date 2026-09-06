"""Audit delivered paired statistics with in-memory mutations, without research data.

Run from the repository root. Source anchors must match exactly once; optional output
is created exclusively. Test failures kill mutants; skips or test errors fail the audit.
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
FUNCTIONS = ("paired_mean", "_mean_from_grid", "bootstrap_means", "linear_quantile", "quantile_interval")
TEST_FILES = ("test_online_confirmation_stats.py", "test_online_confirmation_quantiles.py")
MUTATIONS = (
    ("M1_deduplicate_draw", "k = len(indices)",
     "indices = tuple(dict.fromkeys(indices))\n    k = len(indices)"),
    ("M2_forget_grid_scale", "mean_low = Fraction(total, k * GRID)",
     "mean_low = Fraction(total, k)"),
    ("M3_discard_negative_sign", "mean_low = Fraction(total, k * GRID)",
     "mean_low = abs(Fraction(total, k * GRID))"),
    ("M4_collapse_interval", "mean_high = Fraction(total, k * GRID)",
     "mean_high = mean_low"),
    ("M5_mispair_fused_family", "total += grid_rows[index][1][0]",
     "total += grid_rows[(index + 1) % len(grid_rows)][1][0]"),
    ("M6_keep_previous_endpoint_total",
     "total = 0\n    for index in indices:\n        total += grid_rows[index][0][1]",
     "for index in indices:\n        total += grid_rows[index][0][1]"),
    ("M7_reverse_replicates", "return tuple(results)", "return tuple(reversed(results))"),
    ("M8_ignore_draw", "results.append(_mean_from_grid(grid_rows, indices))",
     "results.append(_mean_from_grid(grid_rows, tuple(range(len(samples)))))"),
    ("M9_float_mean", "mean_low = Fraction(total, k * GRID)",
     "mean_low = total / (k * GRID)"),
    ("Q1_skip_sort", "ordered = sorted(values)", "ordered = list(values)"),
    ("Q2_deduplicate_values", "ordered = sorted(values)", "ordered = sorted(set(values))"),
    ("Q3_wrong_position", "h = (len(ordered) - 1) * p",
     "h = min(Fraction(len(ordered) - 1), len(ordered) * p)"),
    ("Q4_floor_instead_of_interpolate", "t = h - i", "t = Fraction(0)"),
    ("Q5_round_interpolation_to_float",
     "return ordered[i] * (1 - t) + ordered[i + 1] * t",
     "return Fraction(float(ordered[i] * (1 - t) + ordered[i + 1] * t))"),
    ("Q6_take_whole_interval_row", "return (lower_q, upper_q)",
     "return sorted(intervals)[int((len(intervals) - 1) * p)]"),
    ("Q7_complement_upper_probability", "upper_q = linear_quantile(highs, p)",
     "upper_q = linear_quantile(highs, 1 - p)"),
    ("Q8_midpoints_discard_enclosure", "return (lower_q, upper_q)",
     "return (linear_quantile([(low + high) / 2 for low, high in intervals], p),) * 2"),
    ("Q9_reverse_interpolation_weights",
     "return ordered[i] * (1 - t) + ordered[i + 1] * t",
     "return ordered[i] * t + ordered[i + 1] * (1 - t)"),
)

CALIBRATION_MUTATIONS = (
    ("C1_exclude_equality_from_coverage", "covered.append(L <= theta)", "covered.append(L < theta)"),
    ("C2_reject_zero_bound", "arm_rejected = L > 0", "arm_rejected = L >= 0"),
    ("C3_truth_alone_is_false_positive", "false_positive.append(arm_rejected and theta <= 0)",
     "false_positive.append(theta <= 0)"),
    ("C4_noncoverage_is_false_positive", "false_positive.append(arm_rejected and theta <= 0)",
     "false_positive.append(L > theta)"),
    ("C5_any_arm_passes_conjunction", "lower_bounds[0][0] > 0 and lower_bounds[1][0] > 0",
     "lower_bounds[0][0] > 0 or lower_bounds[1][0] > 0"),
    ("C6_require_both_null_means", "truth[0] <= 0 or truth[1] <= 0",
     "truth[0] <= 0 and truth[1] <= 0"),
    ("C7_use_upper_endpoint", "L = lower_bounds[arm][0]", "L = lower_bounds[arm][1]"),
    ("C8_reuse_first_truth", "theta = truth[arm]", "theta = truth[0]"),
)


def run_tests(test_files=TEST_FILES):
    suite = unittest.TestSuite()
    for name in test_files:
        suite.addTests(unittest.TestLoader().discover(str(TEST_DIR), pattern=name,
                                                    top_level_dir=str(TEST_DIR)))
    return unittest.TextTestRunner(stream=io.StringIO()).run(suite)


def audit(calibration=False):
    sys.path.insert(0, str(SOURCE_DIR))
    if calibration:
        from online import confirmation_calibration as stats
        source_name = "confirmation_calibration.py"
        function_names = ("assess_trial",)
        test_files = ("test_online_confirmation_calibration.py",)
        mutations = CALIBRATION_MUTATIONS
    else:
        from online import confirmation_stats as stats
        source_name, function_names, test_files, mutations = "confirmation_stats.py", FUNCTIONS, TEST_FILES, MUTATIONS

    path = SOURCE_DIR / "online" / source_name
    source = path.read_text(encoding="utf-8")
    baseline = run_tests(test_files)
    if not baseline.wasSuccessful() or baseline.skipped or not baseline.testsRun:
        raise RuntimeError("Baseline must pass with no skips")
    original = {name: getattr(stats, name) for name in function_names}
    records = []
    try:
        for name, old, new in mutations:
            if source.count(old) != 1:
                raise ValueError(f"{name}: expected exactly one source anchor")
            tree = ast.parse(source.replace(old, new, 1), filename=str(path))
            functions = [node for node in tree.body
                         if isinstance(node, ast.FunctionDef) and node.name in original]
            exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), stats.__dict__)
            result = run_tests(test_files)
            records.append({"name": name, "killed": bool(result.failures),
                            "tests_run": result.testsRun, "skips": len(result.skipped),
                            "failed_tests": [test.id() for test, _ in result.failures],
                            "errored_tests": [test.id() for test, _ in result.errors]})
            for function, value in original.items():
                setattr(stats, function, value)
    finally:
        for function, value in original.items():
            setattr(stats, function, value)
    return {"stage": "implementation_audit_not_research_measurement",
            "source": source_name,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "tests_sha256": {name: hashlib.sha256((TEST_DIR / name).read_bytes()).hexdigest()
                             for name in test_files},
            "audit_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "baseline_passed": baseline.testsRun, "mutations": records,
            "all_killed": all(r["killed"] and not r["skips"] and not r["errored_tests"] for r in records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--calibration", action="store_true", help="Audit trial scoring instead of means/quantiles")
    args = parser.parse_args()
    report = audit(args.calibration)
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
