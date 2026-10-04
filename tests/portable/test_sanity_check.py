"""scripts/sanity_check.py: the generated documents are what the checks assume, and a failing machine is reported, not raised."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.portable.test_conv_routed import HAVE_PDF

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "sanity_check.py"


def load():
    spec = importlib.util.spec_from_file_location("sanity_check", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@unittest.skipUnless(HAVE_PDF, "needs pypdfium2 and Pillow")
class SanityCheckTests(unittest.TestCase):
    def test_generated_documents(self):
        from rag_search.core.conversion import profiler

        sc = load()
        with tempfile.TemporaryDirectory() as d:
            dig = profiler.profile_file(sc.digital_pdf(Path(d) / "a.pdf"))["pages"][0]
            scan = profiler.profile_file(sc.scan_image(Path(d) / "b.pdf"))["pages"][0]
        self.assertTrue(dig["text_ok"])
        self.assertFalse(scan["text_ok"])
        self.assertGreater(scan["image_cover"], 0.9)

    def test_number_match(self):
        sc = load()
        self.assertTrue(sc.has_number("Total amount 1,234.50 Cr"))
        self.assertFalse(sc.has_number("Total amount 1,234.05"))

    def test_checks_never_raise_and_fail_is_exit_1(self):
        sc = load()
        with mock.patch.object(sc, "check_gpu", side_effect=RuntimeError("no metal")):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = sc.main(["--json"])
        rows = json.loads(buf.getvalue())
        gpu = next(r for r in rows if r["check"].startswith("GPU"))
        self.assertEqual(gpu["status"], "FAIL")
        self.assertIn("no metal", gpu["detail"])
        self.assertEqual(rc, 1)
        self.assertTrue(all(r["status"] == "SKIP" for r in rows if "Playground" in r["check"] or "verify" in r["check"]))


if __name__ == "__main__":
    unittest.main()
