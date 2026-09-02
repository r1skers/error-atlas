"""Keep the new navigation and source/test boundary from silently drifting."""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class RepositoryLayoutTests(unittest.TestCase):
    def test_softmax_index_covers_every_research_module_once(self):
        directory = ROOT / "topics/softmax/experiments"
        sources = {
            path.relative_to(directory).as_posix()
            for path in directory.rglob("*.py")
            if not path.name.startswith("test_")
        }
        links = re.findall(
            r"\]\(([^()]+\.py)\)", (directory / "README.md").read_text(encoding="utf-8")
        )
        self.assertEqual(set(links), sources)
        self.assertEqual(len(links), len(set(links)))

    def test_experiment_directories_contain_only_compatibility_test_entries(self):
        for topic, entry in (
            ("softmax", "test_softmax_suite.py"),
            ("taylor-expansion", "test_taylor_suite.py"),
        ):
            with self.subTest(topic=topic):
                directory = ROOT / "topics" / topic / "experiments"
                self.assertEqual(
                    {path.name for path in directory.glob("test_*.py")}, {entry}
                )

    def test_current_navigation_links_resolve(self):
        # These entry pages use simple local Markdown links (not generated research text).
        entries = (
            "README.md",
            "TOPICS.md",
            "NEXT_SESSION.md",
            "docs/maintenance.md",
            "topics/softmax/README.md",
            "topics/softmax/experiments/README.md",
            "topics/softmax/experiments/reduction_analysis/README.md",
        )
        for name in entries:
            path = ROOT / name
            targets = re.findall(
                r"\[[^\]\n]+\]\(([^)\s]+)\)", path.read_text(encoding="utf-8")
            )
            for target in targets:
                if "://" in target or target.startswith("#"):
                    continue
                with self.subTest(page=name, target=target):
                    self.assertTrue((path.parent / target.split("#", 1)[0]).exists())
