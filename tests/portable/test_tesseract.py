"""Tesseract as the last-resort reader: the module (with a stand-in binary) and the router."""

from __future__ import annotations

import os
import stat
import unittest
from unittest import mock

from tests.helpers import TempHome
from tests.portable.test_conv_vlm import HAVE_PDF, RoutedVlmBase

from rag_search.core.conversion import tesseract, vlm

FAKE_BIN = """#!/bin/sh
if [ "$1" = "--list-langs" ]; then
  echo 'List of available languages in "/fake/tessdata/" (3):'
  printf 'eng\\nhin\\nosd\\n'
  exit 0
fi
if [ -n "$FAKE_TESS_FAIL" ]; then echo "Error: could not open the image" >&2; exit 1; fi
echo "language argument: $4 | psm $6"
echo "Page text read by the stand-in binary with plenty of ordinary words in it."
"""


class ModuleTests(TempHome):
    def setUp(self):
        super().setUp()
        self.bindir = self.tmp / "bin"
        self.bindir.mkdir()
        exe = self.bindir / "tesseract"
        exe.write_text(FAKE_BIN)
        exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
        self.old_path = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{self.bindir}{os.pathsep}{self.old_path}"
        for k in ("RAG_SEARCH_TESSERACT", "RAG_SEARCH_TESSERACT_LANG", "FAKE_TESS_FAIL"):
            os.environ.pop(k, None)
        tesseract._langs_cache.clear()
        self.addCleanup(self.restore)

    def restore(self):
        os.environ["PATH"] = self.old_path
        for k in ("RAG_SEARCH_TESSERACT", "RAG_SEARCH_TESSERACT_LANG", "FAKE_TESS_FAIL"):
            os.environ.pop(k, None)
        tesseract._langs_cache.clear()

    def test_the_requested_languages_are_cut_down_to_the_installed_ones(self):
        self.assertEqual(tesseract.installed_languages(), {"eng", "hin", "osd"})
        self.assertEqual(tesseract.languages(), "hin+eng")                # mar is not installed
        os.environ["RAG_SEARCH_TESSERACT_LANG"] = "eng,fra"
        self.assertEqual(tesseract.languages(), "eng")
        self.assertEqual(tesseract.why_not(), "")

    def test_it_says_why_it_cannot_be_used(self):
        os.environ["RAG_SEARCH_TESSERACT"] = "off"
        self.assertIn("switched off", tesseract.why_not())
        del os.environ["RAG_SEARCH_TESSERACT"]
        os.environ["RAG_SEARCH_TESSERACT_LANG"] = "fra"
        self.assertIn("none of the requested languages", tesseract.why_not())
        os.environ["PATH"] = str(self.tmp / "nowhere")
        tesseract._langs_cache.clear()
        self.assertIn("not installed", tesseract.why_not())

    def test_an_image_is_read_with_the_languages_and_page_segmentation_mode(self):
        text = tesseract.read_image(self.tmp / "x.png")
        self.assertIn("language argument: hin+eng | psm 4", text)
        self.assertIn("ordinary words", text)

    def test_a_failing_binary_is_an_error(self):
        os.environ["FAKE_TESS_FAIL"] = "1"
        with self.assertRaises(RuntimeError):
            tesseract.read_image(self.tmp / "x.png")


@unittest.skipUnless(HAVE_PDF, "needs pypdfium2 and Pillow")
class LastResortTests(RoutedVlmBase):
    TEXT = ("Clause one binds the seller to deliver the property. Clause two binds the buyer to pay the sum "
            "agreed on the date named in the schedule below. " * 3)

    def reader(self):
        return vlm.VlmReader("fake/model", style="instruct", need_gb=1.0)

    def page3(self, res):
        return [x for x in res["records"] if x["page"] == 3][0]

    def test_a_page_the_reader_cannot_stop_repeating_is_read_by_tesseract(self):
        self.plan(always_loop=True)
        with mock.patch.object(tesseract, "why_not", return_value=""), \
                mock.patch.object(tesseract, "read_page", return_value=self.TEXT) as read:
            res = self.convert(self.reader())
            rec = self.page3(res)
            self.assertEqual((rec["reader"]["tool"], rec["branch"]), ("tesseract", "fallback"))
            self.assertEqual([c["name"] for c in rec["gate"]["checks"]], ["low_resolution"])   # the fixture is 72 dpi; the runaway is gone
            self.assertIn("read by Tesseract", rec["note"])
            self.assertIn("Clause one binds", self.out.read_text())
            self.assertEqual(read.call_count, 1)
            calls = self.calls()
            self.convert(self.reader())                                   # the next run: from the cache
            self.assertEqual(read.call_count, 1)
            self.assertEqual(self.calls(), calls)

    def test_without_tesseract_the_page_stays_flagged_and_says_so(self):
        self.plan(always_loop=True)
        with mock.patch.object(tesseract, "why_not", return_value="tesseract is not installed"):
            rec = self.page3(self.convert(self.reader()))
        self.assertEqual(rec["outcome"], "low")
        self.assertIn("Tesseract not used (tesseract is not installed)", rec["note"])

    def test_a_runaway_from_tesseract_is_not_taken(self):
        self.plan(always_loop=True)
        loop = "The said property shall be conveyed to the purchaser free of all charges.\n" * 80
        with mock.patch.object(tesseract, "why_not", return_value=""), \
                mock.patch.object(tesseract, "read_page", return_value=loop):
            rec = self.page3(self.convert(self.reader()))
        self.assertEqual(rec["outcome"], "low")
        self.assertIn("found no usable text either", rec["note"])

    def test_a_healthy_page_never_reaches_tesseract(self):
        with mock.patch.object(tesseract, "read_page", side_effect=AssertionError("must not be called")):
            rec = self.page3(self.convert(self.reader()))
        self.assertEqual(rec["reader"]["tool"], "vlm")


if __name__ == "__main__":
    unittest.main()
