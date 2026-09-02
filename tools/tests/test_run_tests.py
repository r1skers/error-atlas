"""Tests for the maintenance harness, not for research algorithms."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import run_tests


class TestRunnerTests(unittest.TestCase):
    def test_layout_has_source_and_test_directories(self):
        for name, (tests, source) in run_tests.SUITES.items():
            with self.subTest(suite=name):
                self.assertTrue(tests.is_dir())
                self.assertTrue(source.is_dir())
                self.assertNotEqual(tests, source)

    def test_empty_pattern_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with patch.dict(run_tests.SUITES, {"empty": (path, path)}, clear=True):
                with self.assertRaisesRegex(ValueError, "No tests matched"):
                    run_tests.collect_tests(["empty"], "test_missing_*.py")

    def test_missing_directory_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing"
            with patch.dict(
                run_tests.SUITES, {"missing": (missing, missing)}, clear=True
            ):
                with self.assertRaisesRegex(ValueError, "Missing test directory"):
                    run_tests.collect_tests(["missing"], "test_*.py")

    def test_cli_returns_test_outcome_and_passes_selection(self):
        for success, expected in ((True, 0), (False, 1)):
            with self.subTest(success=success):
                result = Mock()
                result.wasSuccessful.return_value = success
                with patch.object(run_tests, "collect_tests") as collect:
                    with patch.object(run_tests.unittest, "TextTestRunner") as runner:
                        runner.return_value.run.return_value = result
                        actual = run_tests.main(
                            ["--suite", "softmax", "-p", "test_one.py"]
                        )
                self.assertEqual(actual, expected)
                collect.assert_called_once_with(["softmax"], "test_one.py")

    def test_cli_does_not_report_empty_collection_as_success(self):
        with patch.object(
            run_tests, "collect_tests", side_effect=ValueError("No tests matched")
        ):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    run_tests.main([])
        self.assertEqual(error.exception.code, 2)
