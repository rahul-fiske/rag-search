"""Runway 3.2b first (``RAG_SEARCH_OCR_FIRST=auto``): a clean scan is read by docling's OCR, a doubted page goes on to the
document reader; the scan-side gate checks that decide it; the page cache and the trace."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from tests.helpers import TempHome
from tests.portable.test_conv_routed import FakeReader, write_routing_pdf
from tests.portable.test_conversion import HAVE_PDF

from rag_search.core.conversion import gate, pagecache, profiler, routed, vlm

GOOD = ("The storage array replicates every volume to the second site and the policy sets the quota for each pool. "
        "Customers receive a statement of the account balance each month and the payment schedule follows the agreement. ") * 4
GIBBERISH = " ".join(["xkqzrtp", "b3rn7", "qwrtpsd", "lkjhgfd", "m1x9z", "vbnmxcz"] * 20)


class StubVlm:
    """Stands in for the document reader: the page text it returns, and the calls it was given."""

    id = "stub"
    model = "stub/model"
    dead = ""

    def __init__(self, fail: bool = False) -> None:
        self.calls: list[tuple[int, int]] = []
        self.fail = fail

    def usable(self):
        return True

    def check(self):
        pass

    def close(self):
        pass

    def read(self, src, first, last, mode):
        self.calls.append((first, last))
        if self.fail:
            raise vlm.ReaderError("failed", "the model fell over")
        return {"pages": {n: f"{GOOD} read by the document reader" for n in range(first, last + 1)},
                "stats": {n: {"read_s": 2.0, "tokens": 50} for n in range(first, last + 1)}, "failed": {}}


@unittest.skipUnless(HAVE_PDF, "needs pypdfium2 and Pillow")
class OcrFirstTests(TempHome):
    def setUp(self):
        super().setUp()
        self.pdf = self.tmp / "r.pdf"
        write_routing_pdf(self.pdf)                             # pages: text, text, scan, blank
        self.profile = profiler.profile_file(self.pdf)
        for pg in self.profile["pages"]:
            if pg.get("ink"):
                pg["ink"] = 0.1                           # a page with about as much text as GOOD
        self.cache = pagecache.PageCache(self.paths.workspace)
        os.environ["RAG_SEARCH_OCR_FIRST"] = "auto"
        self.addCleanup(os.environ.pop, "RAG_SEARCH_OCR_FIRST", None)
        tess = mock.patch("rag_search.core.conversion.tesseract.why_not", return_value="the real binary is not used by tier A")
        tess.start()
        self.addCleanup(tess.stop)
        self.runway = mock.patch("rag_search.core.conversion.router.decide_scan", return_value=("b", ["clean print"]))
        self.runway.start()
        self.addCleanup(self.runway.stop)
        facts = mock.patch("rag_search.core.conversion.scanfacts.pages_facts", side_effect=lambda s, pages: {n: {} for n in pages})
        facts.start()
        self.addCleanup(facts.stop)

    def convert(self, reader, scan):
        return routed.convert_pdf(self.pdf, self.tmp / "out.md", self.profile, cache=self.cache, reader=reader, scan_reader=scan)

    def rec(self, res, n):
        return {x["page"]: x for x in res["records"]}[n]

    def test_a_good_ocr_page_stays_on_runway_b(self):
        r, v = FakeReader(bad={3: GOOD}), StubVlm()
        res = self.convert(r, v)
        self.assertEqual(v.calls, [])
        self.assertIn((3, 3, "scan"), r.calls)
        rec = self.rec(res, 3)
        self.assertEqual(rec["branch"], "fallback")
        self.assertEqual(rec["route"], {"runway": "b", "reasons": ["clean print"], "final": "b", "engine": "docling"})
        self.assertEqual(rec["reader"]["mode"], "ocr-full-page")

    def test_a_doubted_page_goes_to_the_document_reader_and_is_remembered(self):
        r, v = FakeReader(bad={3: GIBBERISH}), StubVlm()
        res = self.convert(r, v)
        self.assertEqual(v.calls, [(3, 3)])
        rec = self.rec(res, 3)
        self.assertEqual(rec["branch"], "raster")
        self.assertEqual(rec["reader"]["tool"], "vlm")
        self.assertEqual(rec["route"]["final"], "d")
        self.assertIn("plausibility", rec["route"]["escalated_from"]["checks"])
        self.assertIn("first_try", rec["time_s"])
        r2, v2 = FakeReader(bad={3: GIBBERISH}), StubVlm()
        res2 = self.convert(r2, v2)                           # the document reader's text is in the cache: nothing is read
        self.assertEqual((r2.calls, v2.calls), ([], []))
        self.assertEqual(self.rec(res2, 3)["branch"], "cached")

    def test_a_cached_ocr_page_that_the_gate_doubts_goes_to_the_document_reader(self):
        self.convert(FakeReader(bad={3: GIBBERISH}), None)    # no document reader: OCR's text is cached
        v = StubVlm()
        res = self.convert(FakeReader(), v)
        self.assertEqual(v.calls, [(3, 3)])
        self.assertEqual(self.rec(res, 3)["route"]["final"], "d")

    def test_when_the_document_reader_fails_ocr_text_is_kept(self):
        r, v = FakeReader(bad={3: GIBBERISH}), StubVlm(fail=True)
        res = self.convert(r, v)
        rec = self.rec(res, 3)
        self.assertEqual(rec["reader"]["tool"], "fake")
        self.assertEqual(rec["route"]["final"], "b")
        self.assertIn("OCR's text kept", rec["note"])
        self.assertEqual(rec["outcome"], "low")

    def test_off_by_default_the_document_reader_takes_every_scan(self):
        os.environ["RAG_SEARCH_OCR_FIRST"] = "off"
        r, v = FakeReader(bad={3: GOOD}), StubVlm()
        res = self.convert(r, v)
        self.assertEqual(v.calls, [(3, 3)])
        self.assertEqual(self.rec(res, 3)["route"]["runway"], "d")      # every page records its lane, whatever the setting
        self.assertEqual(self.rec(res, 3)["route"]["final"], "d")


class ScanGateTests(unittest.TestCase):
    def check(self, md, ink=0.1, table=False):
        return gate.check_page(md, branch_kind="scan", profile={"ink": ink, "dpi": 300}, ocr=True)

    def names(self, g):
        return gate.failed(g)

    def test_good_text_passes_and_does_not_escalate(self):
        g = self.check(GOOD, ink=0.05)
        self.assertEqual(g["verdict"], "ok")
        self.assertFalse(g["escalate"])

    def test_text_that_is_not_words_escalates(self):
        g = self.check(GIBBERISH, ink=0.1)
        self.assertIn("plausibility", self.names(g))
        self.assertTrue(g["escalate"])

    def test_too_little_text_for_the_ink_escalates(self):
        g = self.check("A few words only here and nothing more " * 2, ink=0.2)
        self.assertIn("expected_size", self.names(g))
        self.assertTrue(g["escalate"])

    def test_a_number_column_with_a_letter_o_for_zero(self):
        md = "| Item | Amount |\n|---|---|\n| a | 105 |\n| b | 1O5 |\n| c | 220 |\n| d | 310 |\n"
        g = self.check(md, ink=0.01)
        self.assertIn("column_types", self.names(g))

    def test_the_ocr_checks_apply_only_when_asked(self):
        g = gate.check_page(GIBBERISH, branch_kind="scan", profile={"ink": 0.1, "dpi": 300})
        self.assertNotIn("plausibility", self.names(g))
        self.assertNotIn("escalate", g)

    def test_low_resolution_alone_does_not_escalate(self):
        g = gate.check_page(GOOD, branch_kind="scan", profile={"ink": 0.05, "dpi": 100}, ocr=True)
        self.assertEqual(self.names(g), ["low_resolution"])
        self.assertFalse(g["escalate"])


if __name__ == "__main__":
    unittest.main()
