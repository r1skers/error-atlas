"""Compatibility entry for the historical unittest discovery command."""

from pathlib import Path


def load_tests(loader, tests, pattern):
    test_dir = Path(__file__).resolve().parents[1] / "tests"
    return loader.discover(
        str(test_dir), pattern=pattern or "test_*.py", top_level_dir=str(test_dir)
    )
