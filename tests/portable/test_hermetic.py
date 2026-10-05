"""Tier A must not depend on what tier B installs (docs/design/test-strategy.md, rule 1)."""

import importlib.util
import os
import unittest

from tests import guard


class HermeticTests(unittest.TestCase):
    def test_the_real_libraries_cannot_be_imported_by_a_portable_test(self):
        if not guard.active():
            self.skipTest("the guard is off: a real-tool test shares this process")
        for name in guard.BLOCKED:
            with self.subTest(name), self.assertRaises(ImportError):
                __import__(name)
            self.assertIsNone(importlib.util.find_spec(name), name)      # "not installed", not an error
        self.assertEqual(os.environ.get("HF_HUB_OFFLINE"), "1")


if __name__ == "__main__":
    unittest.main()
