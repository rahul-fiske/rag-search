"""The four lanes of the conversion (3.2a docling on the text layer, 3.2b OCR, 3.2c text layer plus the document reader on
pictures and residue regions, 3.2d the document reader): the router's decisions, the residue finder, the gate's checks per
lane, the ladders between them, and the record every page keeps."""

from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np

from tests.helpers import TempHome
from tests.portable.test_conv_ocrfirst import GIBBERISH, GOOD, StubVlm
from tests.portable.test_conv_routed import FakeReader, write_routing_pdf
from tests.portable.test_conversion import HAVE_PDF

from rag_search.core.conversion import gate, pagecache, profiler, residue, routed, router, scanfacts

CLEAN = {"ink": 0.05, "contrast": 0.9, "sharp": 2.0, "mid_tones": 0.1, "skew": 0.0, "text_lines": 30, "h_rules": 0, "v_rules": 0,
         "bg_std": 3.0, "speckle": 0.0005, "render_dpi": 100, "px": [827, 1170]}


class RouterLaneTests(unittest.TestCase):
    def test_a_text_page_is_lane_a_and_one_with_pictures_or_regions_is_lane_c(self):
        self.assertEqual(router.decide_digital({}, [], {"vlm": True})[0], "a")
        self.assertEqual(router.decide_digital({"big_pics": [[0, 0, 1, 1]]}, [], {"vlm": True})[0], "c")
        lane, why = router.decide_digital({}, [[0.1, 0.1, 0.4, 0.3]], {"vlm": True})
        self.assertEqual(lane, "c")
        self.assertIn("region", why[0])
        self.assertEqual(router.decide_digital({"big_pics": [[0, 0, 1, 1]]}, [], {"vlm": False})[0], "a")   # nobody to read them

    def test_a_skewed_page_stays_in_lane_b_when_it_can_be_straightened(self):
        skewed = dict(CLEAN, skew=4.0)
        prof = {"dpi": 200}
        self.assertEqual(router.decide_scan(skewed, prof, {"ocr": True})[0], "d")
        self.assertEqual(router.decide_scan(skewed, prof, {"ocr": True, "deskew": True})[0], "b")
        self.assertEqual(router.decide_scan(dict(CLEAN, skew=12.0), prof, {"ocr": True, "deskew": True})[0], "d")
        self.assertEqual(router.b_engine(skewed, {"deskew": True}), "tesseract")
        self.assertEqual(router.b_engine(CLEAN, {"deskew": True}), "docling")
        self.assertEqual(router.b_engine(CLEAN, {"deskew": True, "docling_pages": False}), "tesseract")   # an image file


class ResidueTests(unittest.TestCase):
    def mask(self):
        return np.zeros((660, 500), bool)

    def blob(self, ink, y0=100, y1=200, x0=100, x1=250, seed=1):
        rng = np.random.default_rng(seed)
        ink[y0:y1, x0:x1] = rng.random((y1 - y0, x1 - x0)) < 0.5

    def test_a_region_of_ink_outside_the_text_is_found(self):
        ink = self.mask()
        self.blob(ink)
        boxes = residue.regions_of(ink, np.zeros_like(ink))
        self.assertEqual(len(boxes), 1)
        x0, y0, x1, y1 = boxes[0]
        self.assertTrue(x0 < 0.2 < 0.5 < x1 and y0 < 0.16 < 0.3 < y1)

    def test_ink_inside_the_text_rectangles_rules_and_thin_frames_are_not_residue(self):
        ink, text = self.mask(), self.mask()
        self.blob(ink)
        text[90:210, 90:260] = True                                      # the text layer explains it
        ink[300, 50:450] = True                                          # a rule
        ink[320:420, 50] = True                                          # a table border
        ink[320, 50:300] = ink[420, 50:300] = ink[320:420, 300] = True   # a frame
        self.assertEqual(residue.regions_of(ink, text), [])

    def test_a_picture_the_profile_already_has_is_not_found_again(self):
        ink = self.mask()
        self.blob(ink)
        boxes = residue.regions_of(ink, np.zeros_like(ink))
        self.assertEqual(residue.regions_of(ink, np.zeros_like(ink), boxes), [])

    def test_a_small_mark_is_not_a_region(self):
        ink = self.mask()
        self.blob(ink, 100, 120, 100, 130)
        self.assertEqual(residue.regions_of(ink, np.zeros_like(ink)), [])


@unittest.skipUnless(HAVE_PDF, "needs pypdfium2 and Pillow")
class StraightenTests(unittest.TestCase):
    def test_a_skewed_page_is_turned_upright(self):
        from PIL import Image, ImageDraw

        im = Image.new("L", (827, 1170), 255)
        d = ImageDraw.Draw(im)
        for y in range(100, 1050, 26):
            d.rectangle((90, y, 730, y + 9), fill=0)
        tilted = im.rotate(4.0, resample=Image.BICUBIC, fillcolor=255)
        before = abs(scanfacts.facts_of_image(tilted)["skew"])
        after = abs(scanfacts.facts_of_image(scanfacts.straighten(tilted, scanfacts.facts_of_image(tilted)["skew"]))["skew"])
        self.assertGreater(before, 2.0)
        self.assertLess(after, 1.0)

    def test_a_straight_page_is_left_alone(self):
        from PIL import Image

        im = Image.new("RGB", (100, 100), "white")
        self.assertIs(scanfacts.straighten(im, 0.1), im)


class GateLaneTests(unittest.TestCase):
    def test_a_picture_that_was_not_read_leaves_the_page_low(self):
        g = gate.check_page(GOOD, branch_kind="digital", profile={"chars": 800}, residue={"asked": 2, "read": 1})
        self.assertIn("residue_read", gate.failed(g))
        g = gate.check_page(GOOD, branch_kind="digital", profile={"chars": 800}, residue={"asked": 2, "read": 2})
        self.assertNotIn("residue_read", gate.failed(g))

    def test_devanagari_words_are_not_judged_by_latin_vowels(self):
        marathi = " ".join(["कंपनीच्या", "व्यवहारासाठी", "प्रमाणपत्र", "सोसायटीचे", "सदस्यत्व", "नोंदणी", "खरेदीखत", "दस्तऐवज",
                            "मालमत्ता", "हस्तांतरण", "अर्जदार", "स्वाक्षरी"] * 3)
        self.assertTrue(gate._plausibility(marathi)["ok"])
        self.assertFalse(gate._plausibility(GIBBERISH)["ok"])


class PicVlm(StubVlm):
    """A document reader that reads pictures (text per picture) and pages."""

    def __init__(self, tmp, fail_pictures: bool = False, **kw):
        super().__init__(**kw)
        self.tmp, self.fail_pictures, self.pictures = tmp, fail_pictures, 0

    def _tmp_dir(self):
        return self.tmp

    def read_image(self, img, kind="page"):
        from rag_search.core.conversion import vlm

        if self.fail_pictures:
            raise vlm.ReaderError("failed", "no memory")
        self.pictures += 1
        return {"md": "Rubber stamp: received 4,512.00 rupees on the fifth of March", "tokens": 12, "seconds": 0.5}


@unittest.skipUnless(HAVE_PDF, "needs pypdfium2 and Pillow")
class LaneFlowTests(TempHome):
    def setUp(self):
        super().setUp()
        self.pdf = self.tmp / "r.pdf"
        write_routing_pdf(self.pdf)                             # pages: text, text, scan, blank
        self.profile = profiler.profile_file(self.pdf)
        for pg in self.profile["pages"]:
            if pg.get("ink"):
                pg["ink"] = 0.1
        self.cache = pagecache.PageCache(self.paths.workspace)
        for var in ("RAG_SEARCH_OCR_FIRST", "RAG_SEARCH_RESIDUE", "RAG_SEARCH_ESCALATE_DIGITAL"):
            self.addCleanup(os.environ.pop, var, None)
        tess = mock.patch("rag_search.core.conversion.tesseract.why_not", return_value="the real binary is not used by tier A")
        self.tess = tess.start()
        self.addCleanup(tess.stop)

    def convert(self, reader, scan):
        return routed.convert_pdf(self.pdf, self.tmp / "out.md", self.profile, cache=self.cache, reader=reader, scan_reader=scan)

    def rec(self, res, n):
        return {x["page"]: x for x in res["records"]}[n]

    def tesseract_on(self, text=GOOD):
        self.tess.return_value = ""
        usable = mock.patch("rag_search.core.conversion.tesseract.usable", return_value=True)
        read = mock.patch("rag_search.core.conversion.tesseract.read_page", return_value=text)
        usable.start()
        self.addCleanup(usable.stop)
        self.addCleanup(read.stop)
        return read.start()

    def lane_b(self, skew=0.0):
        os.environ["RAG_SEARCH_OCR_FIRST"] = "auto"
        for target, value in (("router.decide_scan", ("b", ["clean print"])),):
            p = mock.patch(f"rag_search.core.conversion.{target}", return_value=value)
            p.start()
            self.addCleanup(p.stop)
        f = mock.patch("rag_search.core.conversion.scanfacts.pages_facts",
                       side_effect=lambda s, pages: {n: dict(CLEAN, skew=skew) for n in pages})
        f.start()
        self.addCleanup(f.stop)

    def test_every_page_records_its_lane(self):
        res = self.convert(FakeReader(), StubVlm())
        self.assertEqual([self.rec(res, n)["route"]["runway"] for n in (1, 2, 3)], ["a", "a", "d"])
        self.assertEqual([self.rec(res, n)["route"]["final"] for n in (1, 2, 3)], ["a", "a", "d"])
        self.assertNotIn("route", self.rec(res, 4))                                  # a blank page is not read

    def test_without_a_document_reader_a_scan_is_lane_b_by_necessity(self):
        res = self.convert(FakeReader(), None)
        rt = self.rec(res, 3)["route"]
        self.assertEqual((rt["runway"], rt["final"]), ("b", "b"))
        self.assertIn("no document reader", rt["reasons"][0])

    def test_a_region_the_text_layer_does_not_explain_makes_the_page_lane_c(self):
        os.environ["RAG_SEARCH_RESIDUE"] = "auto"
        v = PicVlm(self.tmp)
        with mock.patch("rag_search.core.conversion.residue.pages_regions", return_value={1: [[0.1, 0.5, 0.5, 0.8]]}):
            res = self.convert(FakeReader(), v)
        rec = self.rec(res, 1)
        self.assertEqual((rec["route"]["runway"], rec["route"]["final"], rec["branch"]), ("c", "c", "embedded"))
        self.assertEqual(v.pictures, 1)
        self.assertIn("4,512.00", (self.tmp / "out.md").read_text())
        self.assertEqual(self.rec(res, 2)["route"]["runway"], "a")
        with mock.patch("rag_search.core.conversion.residue.pages_regions", return_value={1: [[0.1, 0.5, 0.5, 0.8]]}):
            res2 = self.convert(FakeReader(), PicVlm(self.tmp))              # cached together
        self.assertEqual(self.rec(res2, 1)["route"]["final"], "c")

    def test_residue_is_not_looked_for_unless_it_is_switched_on(self):
        with mock.patch("rag_search.core.conversion.residue.pages_regions") as found:
            self.convert(FakeReader(), PicVlm(self.tmp))
        found.assert_not_called()

    def test_a_region_that_could_not_be_read_leaves_the_page_low(self):
        os.environ["RAG_SEARCH_RESIDUE"] = "auto"
        with mock.patch("rag_search.core.conversion.residue.pages_regions", return_value={1: [[0.1, 0.5, 0.5, 0.8]]}):
            res = self.convert(FakeReader(), PicVlm(self.tmp, fail_pictures=True))
        rec = self.rec(res, 1)
        self.assertEqual(rec["outcome"], "low")
        self.assertIn("residue_read", [c["name"] for c in rec["gate"]["checks"]])

    def test_a_text_page_that_lost_its_text_goes_to_the_document_reader_when_that_is_switched_on(self):
        os.environ["RAG_SEARCH_ESCALATE_DIGITAL"] = "auto"
        v = PicVlm(self.tmp)
        res = self.convert(FakeReader(bad={2: "�" * 60 + " x" * 30}), v)
        rec = self.rec(res, 2)
        self.assertEqual((rec["route"]["runway"], rec["route"]["final"]), ("a", "d"))
        self.assertEqual(rec["route"]["escalated_from"]["runway"], "a")
        self.assertIn("script", rec["route"]["escalated_from"]["checks"])
        self.assertEqual(rec["reader"]["tool"], "vlm")
        self.assertTrue(any(a <= 2 <= b for a, b in v.calls))
        self.assertEqual(self.rec(res, 1)["route"]["final"], "a")

    def test_the_document_reader_result_for_a_text_page_is_reused_on_the_next_run(self):
        os.environ["RAG_SEARCH_ESCALATE_DIGITAL"] = "auto"
        bad = {2: "�" * 60 + " x" * 30}
        self.convert(FakeReader(bad=bad), PicVlm(self.tmp))
        v = PicVlm(self.tmp)
        res = self.convert(FakeReader(bad=bad), v)
        rec = self.rec(res, 2)
        self.assertEqual(v.calls, [])                                                # nothing is read again
        self.assertEqual((rec["branch"], rec["route"]["final"]), ("cached", "d"))
        self.assertEqual(rec["route"]["escalated_from"]["runway"], "a")

    def test_the_text_layer_result_is_kept_when_the_document_reader_cannot_take_over(self):
        os.environ["RAG_SEARCH_ESCALATE_DIGITAL"] = "auto"
        res = self.convert(FakeReader(bad={2: "�" * 60 + " x" * 30}), PicVlm(self.tmp, fail=True))
        rec = self.rec(res, 2)
        self.assertEqual(rec["branch"], "digital")
        self.assertEqual(rec["route"]["final"], "a")                                  # it was handed over and handed back
        self.assertIn("text-layer result is kept", rec["note"])

    def test_off_by_default_a_text_page_that_lost_its_text_is_kept_and_flagged(self):
        v = PicVlm(self.tmp)
        res = self.convert(FakeReader(bad={2: "�" * 60 + " x" * 30}), v)
        self.assertEqual(self.rec(res, 2)["outcome"], "low")
        self.assertEqual(self.rec(res, 2)["route"]["final"], "a")
        self.assertFalse(any(a <= 2 <= b for a, b in v.calls))

    def test_a_doubted_ocr_page_gets_a_tesseract_try_before_the_document_reader(self):
        self.lane_b()
        read = self.tesseract_on()
        v = StubVlm()
        res = self.convert(FakeReader(bad={3: GIBBERISH}), v)
        rec = self.rec(res, 3)
        self.assertEqual(v.calls, [])
        self.assertEqual((rec["route"]["runway"], rec["route"]["final"], rec["route"]["engine"]), ("b", "b", "tesseract"))
        self.assertEqual(rec["reader"]["tool"], "tesseract")
        self.assertIn("docling's OCR was doubted", rec["note"])
        read.assert_called_once()

    def test_a_page_with_a_table_goes_straight_to_the_document_reader(self):
        self.lane_b()
        read = self.tesseract_on()
        table = "| a | b | c |\n|---|---|---|\n| 1 | 2 |\n| 3 | 4 | 5 | 6 |\n" + GIBBERISH
        v = StubVlm()
        res = self.convert(FakeReader(bad={3: table}), v)
        self.assertEqual(v.calls, [(3, 3)])
        read.assert_not_called()
        self.assertEqual(self.rec(res, 3)["route"]["final"], "d")

    def test_when_tesseract_is_doubted_too_the_document_reader_takes_over(self):
        self.lane_b()
        self.tesseract_on(GIBBERISH)
        v = StubVlm()
        res = self.convert(FakeReader(bad={3: GIBBERISH}), v)
        self.assertEqual(v.calls, [(3, 3)])
        self.assertEqual(self.rec(res, 3)["route"]["final"], "d")

    def test_a_skewed_page_is_straightened_and_read_by_tesseract_first(self):
        self.lane_b(skew=4.0)
        read = self.tesseract_on()
        r = FakeReader()
        res = self.convert(r, StubVlm())
        rec = self.rec(res, 3)
        self.assertEqual(rec["route"]["engine"], "tesseract")
        self.assertNotIn((3, 3, "scan"), r.calls)                                    # docling's OCR was not asked
        self.assertEqual(read.call_args.kwargs["skew"], 4.0)
        self.assertIn("straightening", rec["note"])
        res2 = self.convert(FakeReader(), StubVlm())                                 # and cached under Tesseract's tag
        self.assertEqual(self.rec(res2, 3)["branch"], "cached")

    def test_the_lane_of_a_cached_page_is_still_known(self):
        self.convert(FakeReader(), StubVlm())
        res = self.convert(FakeReader(), StubVlm())
        self.assertEqual([self.rec(res, n)["route"]["final"] for n in (1, 3)], ["a", "d"])


@unittest.skipUnless(HAVE_PDF, "needs pypdfium2 and Pillow")
class ImageLaneTests(TempHome):
    @staticmethod
    def write_letter(path):
        from PIL import Image, ImageDraw

        im = Image.new("RGB", (1240, 1754), "white")
        d = ImageDraw.Draw(im)
        for y in range(150, 1600, 40):
            d.rectangle((120, y, 1100, y + 14), fill="black")
        im.save(path)

    def test_a_clean_photograph_of_a_page_is_read_by_tesseract_through_the_router(self):
        png = self.tmp / "letter.png"
        self.write_letter(png)
        prof = profiler.profile_file(png)
        prof["pages"][0]["ink"] = 0.1                           # about as much text as GOOD
        os.environ["RAG_SEARCH_OCR_FIRST"] = "auto"
        self.addCleanup(os.environ.pop, "RAG_SEARCH_OCR_FIRST", None)
        v = StubVlm()
        with (mock.patch("rag_search.core.conversion.scanfacts.image_facts", return_value=CLEAN),
              mock.patch("rag_search.core.conversion.tesseract.why_not", return_value=""),
              mock.patch("rag_search.core.conversion.tesseract.usable", return_value=True),
              mock.patch("rag_search.core.conversion.tesseract.read_page", return_value=GOOD)):
            res = routed.convert_image(png, self.tmp / "o.md", prof, cache=pagecache.PageCache(self.paths.workspace), scan_reader=v)
        rec = res["records"][0]
        self.assertEqual((rec["route"]["runway"], rec["route"]["engine"], rec["route"]["final"]), ("b", "tesseract", "b"))
        self.assertEqual(v.calls, [])

    def test_without_the_switch_an_image_goes_to_the_document_reader(self):
        png = self.tmp / "letter.png"
        self.write_letter(png)
        v = StubVlm()
        res = routed.convert_image(png, self.tmp / "o.md", profiler.profile_file(png), cache=None, scan_reader=v)
        self.assertEqual(res["records"][0]["route"]["final"], "d")


@unittest.skipUnless(HAVE_PDF, "needs pypdfium2 and Pillow")
class ReviewFixTests(TempHome):
    """What the review of 0.9.24 found: pages reported one by one, no second OCR read, steps, a fair resolution check."""

    def scans(self, n=3):
        from PIL import Image, ImageDraw

        pages = []
        for i in range(n):
            im = Image.new("RGB", (1240, 1754), "white")
            d = ImageDraw.Draw(im)
            for y in range(150, 1600, 40):
                d.rectangle((120, y, 1000 - 30 * i, y + 14), fill="black")
            pages.append(im)
        pdf = self.tmp / "s.pdf"
        pages[0].save(pdf, "PDF", resolution=150.0, save_all=True, append_images=pages[1:])
        prof = profiler.profile_file(pdf)
        for pg in prof["pages"]:
            pg["ink"] = 0.1
        return pdf, prof

    def test_the_document_reader_reads_and_reports_one_page_at_a_time(self):
        pdf, prof = self.scans(3)
        v, order, steps = StubVlm(), [], []
        res = routed.convert_pdf(pdf, self.tmp / "o.md", prof, cache=pagecache.PageCache(self.paths.workspace),
                                 reader=FakeReader(), scan_reader=v, on_page=lambda rec, total: order.append(("page", rec["page"])),
                                 on_step=lambda page, what: (steps.append((page, what)), order.append(("step", page))))
        self.assertEqual(v.calls, [(1, 1), (2, 2), (3, 3)])                       # not one call for the run of three
        self.assertEqual(order, [("step", 1), ("page", 1), ("step", 2), ("page", 2), ("step", 3), ("page", 3)])
        self.assertEqual(steps[0], (1, "document reader"))
        self.assertEqual(len(res["records"]), 3)

    def test_when_the_reader_dies_the_other_pages_are_not_asked_of_it(self):
        pdf, prof = self.scans(3)

        class Dying(StubVlm):
            def read(self, src, first, last, mode):
                self.calls.append((first, last))
                self.dead = "the reader process stopped too often in this run"
                return {"pages": {}, "stats": {}, "failed": {first: "crashed: gone"}}

            def usable(self):
                return not self.dead

        v, r = Dying(), FakeReader()
        res = routed.convert_pdf(pdf, self.tmp / "o.md", prof, cache=None, reader=r, scan_reader=v)
        self.assertEqual(v.calls, [(1, 1)])
        self.assertEqual([x["branch"] for x in res["records"]], ["fallback"] * 3)   # docling's OCR read them
        self.assertEqual(r.calls, [(1, 3, "scan")])

    def test_ocr_text_that_the_reader_could_not_replace_is_kept_without_reading_the_page_again(self):
        pdf, prof = self.scans(1)
        os.environ["RAG_SEARCH_OCR_FIRST"] = "auto"
        self.addCleanup(os.environ.pop, "RAG_SEARCH_OCR_FIRST", None)
        r, v = FakeReader(bad={1: GIBBERISH}), StubVlm(fail=True)
        with (mock.patch("rag_search.core.conversion.router.decide_scan", return_value=("b", ["clean print"])),
              mock.patch("rag_search.core.conversion.scanfacts.pages_facts", side_effect=lambda s, pages: {n: dict(CLEAN) for n in pages}),
              mock.patch("rag_search.core.conversion.tesseract.why_not", return_value="not used by tier A")):
            res = routed.convert_pdf(pdf, self.tmp / "o.md", prof, cache=None, reader=r, scan_reader=v)
        rec = res["records"][0]
        self.assertEqual(r.calls, [(1, 1, "scan")])                               # once, not again after the reader failed
        self.assertEqual((rec["route"]["final"], rec["route"]["engine"], rec["outcome"]), ("b", "docling", "low"))
        self.assertIn("docling OCR's text kept", rec["note"])

    def test_a_page_drawn_from_vectors_is_not_judged_by_the_resolution_of_a_logo_on_it(self):
        e = {"mode": "scan", "profile": {"chars": 900, "image_cover": 0.05, "dpi": 72, "text_ok": False}}
        c = routed._Converter.__new__(routed._Converter)
        self.assertNotIn("dpi", c.gate_profile(e))
        scan = {"mode": "scan", "profile": {"chars": 0, "image_cover": 1.0, "dpi": 72}}
        self.assertEqual(c.gate_profile(scan)["dpi"], 72)                         # a real scan is judged by its own resolution
        g = gate.check_page(GOOD, branch_kind="scan", profile=c.gate_profile(e))
        self.assertNotIn("low_resolution", gate.failed(g))

    def test_a_page_the_repair_model_read_again_gets_the_one_table_format_too(self):
        pdf, prof = self.scans(1)
        html = "<table><tr><td>Date</td><td>Amount</td></tr><tr><td>1 May</td><td>10.00</td></tr><tr><td>2 May</td><td>20.00</td></tr></table>"
        bad = "| a | b | c |\n|---|---|---|\n| 1 | 2 |\n| 3 | 4 | 5 | 6 |\n"

        class Rep:
            tag, reread = "r|model|-", True
            reader = type("R", (), {"model": "m"})()

            def usable(self):
                return True

            def run(self, src, n, is_image, md, violations, page_ok=None):
                return {"md": html + "\n\n" + GOOD, "cells": [], "fixed": 0, "tried": 0, "tier": "page", "tokens": 0, "gpu_s": 0.0,
                        "seconds": 0.1, "model": "m", "second": "", "note": "page read again", "complete": True}

        class TableVlm(StubVlm):
            def read(self, src, first, last, mode):
                return {"pages": {first: bad + GOOD}, "stats": {first: {"read_s": 1.0}}, "failed": {}}

        routed.convert_pdf(pdf, self.tmp / "o.md", prof, cache=None, reader=FakeReader(), scan_reader=TableVlm(), repairer=Rep())
        text = (self.tmp / "o.md").read_text()
        self.assertNotIn("<table", text)
        self.assertIn("| 1 May | 10.00 |", text)

    def test_a_repair_that_could_not_run_is_tried_again_next_time(self):
        pdf, prof = self.scans(1)

        class Rep:
            tag, reread = "r|model|-", True
            reader = type("R", (), {"model": "m"})()
            runs = 0

            def usable(self):
                return True

            def run(self, src, n, is_image, md, violations, page_ok=None):
                Rep.runs += 1
                return {"md": md, "cells": [], "fixed": 0, "tried": 0, "tier": "", "tokens": 0, "gpu_s": 0.0, "seconds": 0.1,
                        "model": "m", "second": "", "note": "page not read again (unavailable: no memory)", "complete": Rep.runs > 1}

        table = "| a | b | c |\n|---|---|---|\n| 1 | 2 |\n| 3 | 4 | 5 | 6 |\n"

        class TableVlm(StubVlm):
            def read(self, src, first, last, mode):
                self.calls.append((first, last))
                return {"pages": {n: table + GOOD for n in range(first, last + 1)}, "stats": {n: {"read_s": 1.0} for n in range(first, last + 1)}, "failed": {}}

        cache = pagecache.PageCache(self.paths.workspace)
        for _ in range(3):
            routed.convert_pdf(pdf, self.tmp / "o.md", prof, cache=cache, reader=FakeReader(), scan_reader=TableVlm(), repairer=Rep())
        self.assertEqual(Rep.runs, 2)                 # run 1 could not (not remembered), run 2 did, run 3 knows it was tried


if __name__ == "__main__":
    unittest.main()


class ProfileTests(unittest.TestCase):
    def test_a_lane_that_is_switched_on_changes_the_document_profile_and_the_default_does_not(self):
        from rag_search.core.docling_convert import convert_profile

        names = ("RAG_SEARCH_OCR_FIRST", "RAG_SEARCH_RESIDUE", "RAG_SEARCH_ESCALATE_DIGITAL", "RAG_SEARCH_LAYER_FILL")
        saved = {k: os.environ.pop(k, None) for k in names}
        try:
            base = convert_profile()
            for var, tag in (("RAG_SEARCH_OCR_FIRST", "ocrfirst=auto"), ("RAG_SEARCH_RESIDUE", "residue=auto"),
                             ("RAG_SEARCH_ESCALATE_DIGITAL", "updigital=auto")):
                os.environ[var] = "auto"
                self.assertEqual(convert_profile(), base + "|" + tag)
                self.assertEqual(convert_profile(readers=False), convert_profile(readers=False))     # the page cache key is not touched
                del os.environ[var]
            os.environ["RAG_SEARCH_OCR_FIRST"] = "off"
            self.assertEqual(convert_profile(), base)
        finally:
            for k, v in saved.items():
                os.environ.pop(k, None)
                if v is not None:
                    os.environ[k] = v
