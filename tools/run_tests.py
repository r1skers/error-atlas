"""Run regression tests without launching experiment CLIs or regenerating evidence."""

from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUITES = {
    "softmax": (ROOT / "topics/softmax/tests", ROOT / "topics/softmax/experiments"),
    "taylor": (
        ROOT / "topics/taylor-expansion/tests",
        ROOT / "topics/taylor-expansion/experiments",
    ),
    "maintenance": (ROOT / "tools/tests", ROOT / "tools"),
}


def collect_tests(names: list[str], pattern: str) -> unittest.TestSuite:
    """Keep a separate loader per topic so unittest does not retain another root."""
    suite = unittest.TestSuite()
    for name in names:
        test_dir, source_dir = SUITES[name]
        if not test_dir.is_dir():
            raise ValueError(f"Missing test directory: {test_dir}")
        # Historical scripts intentionally keep their original top-level imports.
        source = str(source_dir)
        if source not in sys.path:
            sys.path.insert(0, source)
        discovered = unittest.TestLoader().discover(
            str(test_dir), pattern=pattern, top_level_dir=str(test_dir)
        )
        suite.addTests(discovered)
    if suite.countTestCases() == 0:
        raise ValueError(f"No tests matched {pattern!r} in {', '.join(names)}")
    return suite


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", choices=("all", *SUITES), default="all", help="default: all"
    )
    parser.add_argument(
        "-p", "--pattern", default="test_*.py", help="test filename glob"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    names = list(SUITES) if args.suite == "all" else [args.suite]
    try:
        suite = collect_tests(names, args.pattern)
    except ValueError as error:
        parser.error(str(error))
    result = unittest.TextTestRunner(verbosity=2 if args.verbose else 1).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
