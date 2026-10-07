"""Per-page routed conversion (phase P2): plan, runs, page cache, gate, indexer integration."""

from __future__ import annotations

import json
import os
import time
import unittest
from pathlib import Path
from unittest import mock

from tests import corpus
from tests.helpers import TempHome, no_real_reader
from tests.portable.test_cli_api import run
from tests.portable.test_ui import Dash
from tests.portable.test_conversion import HAVE_PDF, TEXT, ConversionBase, write_text_pdf

from rag_search.core.conversion import gate, pagecache, pagemd, profiler, routed, runview, trace
from rag_search.core.docling_convert import NoTextError


def write_routing_pdf(path: Path) -> None:
    """Two text pages, a scanned page with dark content, and a blank scanned page (the corpus file)."""
    corpus.copy("pdf/text-scan-blank.pdf", path)


class FakeReader:
    """Stands in for docling: text per page, recording what it was asked to read."""

    id = "fake"

    def __init__(self, bad: dict[int, str] | None = None, fail: bool = False) -> None:
        self.calls: list[tuple[int, int, str]] = []
        self.bad = bad or {}
        self.fail = fail

    def read(self, src, first, last, mode):
        self.calls.append((first, last, mode))
        if self.fail:
            raise RuntimeError("the reader fell over")
        pages, stats = {}, {}
        for n in range(first, last + 1):
            md = self.bad.get(n, f"{TEXT} page {n} read as {mode}")
            pages[n] = md
            stats[n] = {"page": n, "chars": len("".join(md.split())), "script": "Latin", "tables": 0,
                        "pictures": 0, "big_pictures": 0}
        return {"pages": pages, "stats": stats, "seconds": 0.2 * (last - first + 1)}


@unittest.skipUnless(HAVE_PDF, "needs pypdfium2 and Pillow")
class RoutedUnitTests(TempHome):
    def setUp(self):
        super().setUp()
        self.pdf = self.tmp / "r.pdf"
        write_routing_pdf(self.pdf)
        self.profile = profiler.profile_file(self.pdf)
        self.cache = pagecache.PageCache(self.paths.workspace)

    def convert(self, reader, **kw):
        out = self.tmp / "out.md"
        return routed.convert_pdf(self.pdf, out, self.profile, cache=self.cache, reader=reader, **kw), out

    def test_plan_pages(self):
        plan = routed.plan_pages(self.profile)
        self.assertEqual([(e["page"], e["branch"], e["mode"], e["blank"]) for e in plan],
                         [(1, "digital", "digital", False), (2, "digital", "digital", False),
                          (3, "raster", "scan", False), (4, "raster", "", True)])   # fallback is decided when reading
        self.assertIn("blank", plan[3]["why"])
        self.assertTrue(all(e["hash"] for e in plan))
        self.assertNotEqual(plan[0]["hash"], plan[1]["hash"])           # different text, different page

    def test_light_writing_on_a_dark_ground_is_not_a_blank_page(self):
        from PIL import Image, ImageDraw

        from rag_search.core.conversion import profiler

        dark = Image.new("L", (400, 300), 20)
        d = ImageDraw.Draw(dark)
        for y in range(40, 260, 30):
            d.rectangle((40, y, 360, y + 8), fill=235)                 # lines of light "writing"
        facts = profiler.ink_and_hash(dark)
        self.assertLess(facts["ink"], profiler.BLANK_INK)              # nothing is darker than the ground ...
        self.assertGreater(facts["light"], 0.05)                       # ... but a lot is lighter
        white = profiler.ink_and_hash(Image.new("L", (400, 300), 252))
        self.assertEqual((white["ink"], white["light"]), (0.0, 0.0))
        page = {"page": 1, "branch": "image", "chars": 0, "hash": "r1"}
        for facts_, blank in ((facts, False), (white, True), ({"ink": 0.0}, True)):      # a profile without the figure: as before
            prof = {"kind": "image", "pages": [{**page, **facts_, "hash": "r1"}], "page_count": 1}
            with mock.patch.object(profiler, "route_pages", return_value=[(1, "image", "an image file")]):
                self.assertEqual(routed.plan_pages(prof)[0]["blank"], blank, facts_)

    def test_a_file_of_blank_pages_says_so(self):
        blank = {1: {"md": "", "cache": "none", "via": ""}, 2: {"md": "", "cache": "none", "via": ""}}
        msg = routed.no_text_message(Path("empty.jpg"), blank)
        self.assertIn("every page is blank", msg)
        self.assertNotIn("install", msg)                                # not the advice for a missing reader
        msg = routed.no_text_message(Path("photo.jpg"), {1: {"md": "", "cache": "miss", "via": "vlm"}})
        self.assertIn("found no text", msg)

    def test_runs(self):
        self.assertEqual(routed._runs([1, 2, 3, 7, 8, 12]), [(1, 3), (7, 8), (12, 12)])
        self.assertEqual(routed._runs(list(range(1, 46)), 20), [(1, 20), (21, 40), (41, 45)])
        self.assertEqual(routed._runs([]), [])

    def test_pages_are_read_by_kind_in_runs_and_recorded(self):
        r = FakeReader()
        res, out = self.convert(r)
        self.assertEqual(r.calls, [(1, 2, "digital"), (3, 3, "scan")])    # the blank page is never read
        recs = {x["page"]: x for x in res["records"]}
        self.assertEqual([recs[n]["branch"] for n in (1, 2, 3, 4)], ["digital", "digital", "fallback", "fallback"])
        self.assertEqual([recs[n]["outcome"] for n in (1, 2, 3, 4)], ["pass", "pass", "low", "no_text"])
        self.assertEqual([c["name"] for c in recs[3]["gate"]["checks"]], ["low_resolution"])   # the test scan is ~94 dpi
        self.assertEqual(recs[1]["reader"]["mode"], "ocr-auto")
        self.assertEqual(recs[3]["reader"]["mode"], "ocr-full-page")
        self.assertEqual(recs[3]["time_s"]["read"], 0.2)
        self.assertEqual(recs[4]["reader"], {"tool": "none"})
        self.assertEqual(recs[1]["cache"], "miss")
        self.assertEqual(res["cache"], {"hit": 0, "miss": 3})
        pages = pagemd.split_pages(out.read_text())
        self.assertEqual(sorted(pages), [1, 2, 3])
        self.assertIn("page 3 read as scan", pages[3])

    def test_second_run_is_served_from_the_cache(self):
        self.convert(FakeReader())
        r2 = FakeReader()
        res, _ = self.convert(r2)
        self.assertEqual(r2.calls, [])
        recs = {x["page"]: x for x in res["records"]}
        self.assertEqual([recs[n]["branch"] for n in (1, 2, 3)], ["cached"] * 3)
        self.assertEqual(recs[3]["was"], "fallback")
        self.assertEqual(recs[3]["cache"], "hit")
        self.assertEqual(res["cache"], {"hit": 3, "miss": 0})
        self.assertEqual(self.cache.stats()["entries"], 3)

    def test_a_different_setting_is_a_different_cache_entry(self):
        self.convert(FakeReader())
        os.environ["RAG_SEARCH_TABLE_MODE"] = "fast"
        try:
            r = FakeReader()
            self.convert(r)
            self.assertEqual(len(r.calls), 2)
        finally:
            del os.environ["RAG_SEARCH_TABLE_MODE"]

    def test_the_gate_marks_bad_pages_low(self):
        r = FakeReader(bad={2: "�" * 60 + " x" * 30})
        res, _ = self.convert(r)
        recs = {x["page"]: x for x in res["records"]}
        self.assertEqual(recs[2]["outcome"], "low")
        self.assertEqual([c["name"] for c in recs[2]["gate"]["checks"]], ["script"])
        self.assertEqual(recs[1]["outcome"], "pass")
        self.assertNotIn("gate", recs[1])

    def test_balance_violation_is_recorded_with_its_hypothesis(self):
        from tests.portable.test_conv_bench import PASSBOOK
        r = FakeReader(bad={3: PASSBOOK.replace("281.00", "2,81.00")})
        res, _ = self.convert(r)
        rec = {x["page"]: x for x in res["records"]}[3]
        self.assertEqual(rec["outcome"], "low")
        self.assertEqual(rec["gate"]["violations"][0]["expected"], "281.00")

    def test_nothing_read_raises_no_text(self):
        r = FakeReader(bad={1: "", 2: "", 3: ""})
        with self.assertRaises(NoTextError):
            self.convert(r)

    def test_placeholder_only_pages_are_no_text_with_an_actionable_message(self):
        """docling's export of a photographed page: an image marker and a classification label."""
        r = FakeReader(bad={n: "<!-- image -->\n\nOther" for n in (1, 2, 3)})
        with self.assertRaises(NoTextError) as cm:
            self.convert(r)
        self.assertIn("document reader", str(cm.exception))
        self.assertFalse((self.tmp / "out.md").exists())

    def test_a_scanned_page_docling_cannot_read_is_read_by_apple_vision(self):
        """The passbook case: docling's OCR returns a placeholder for a photographed page; Apple Vision reads it."""
        from rag_search.core.conversion import applevision
        r = FakeReader(bad={3: "<!-- image -->\n\nOther"})
        with mock.patch.object(applevision, "why_not", return_value=""), \
                mock.patch.object(applevision, "read_page", return_value="Account Type : PPF\n\nPassbook No : 1") as rp:
            res, out = self.convert(r)
        rec = {x["page"]: x for x in res["records"]}[3]
        self.assertEqual(rec["reader"], {"tool": "apple-vision", "mode": "page"})
        self.assertIn(rec["outcome"], ("pass", "low"))          # low: the test scan is ~94 dpi
        self.assertIn("Apple Vision", rec["note"])
        self.assertIn("Passbook No", pagemd.split_pages(out.read_text())[3])
        self.assertEqual(rp.call_args.args[1], 3)
        # the next run is served from the page cache, without docling or Apple Vision
        r2 = FakeReader()
        with mock.patch.object(applevision, "why_not", return_value=""), \
                mock.patch.object(applevision, "read_page", side_effect=AssertionError("read again")):
            res2, _ = self.convert(r2)
        self.assertEqual({x["page"]: x for x in res2["records"]}[3]["cache"], "hit")
        self.assertNotIn((3, 3, "scan"), r2.calls)

    def test_apple_vision_is_not_used_where_it_cannot_run_and_the_note_says_why(self):
        from rag_search.core.conversion import applevision
        r = FakeReader(bad={3: "<!-- image -->\n\nOther"})
        with mock.patch.object(applevision, "why_not", return_value="Apple Vision needs a Mac"):
            res, _ = self.convert(r)
        rec = {x["page"]: x for x in res["records"]}[3]
        self.assertEqual(rec["outcome"], "no_text")
        self.assertIn("Apple Vision not used (Apple Vision needs a Mac)", rec["note"])

    def test_all_pages_unreadable_by_both_gives_no_text_that_names_apple_vision(self):
        from rag_search.core.conversion import applevision
        r = FakeReader(bad={n: "<!-- image -->\n\nOther" for n in (1, 2, 3)})
        with mock.patch.object(applevision, "why_not", return_value=""), \
                mock.patch.object(applevision, "read_page", return_value=""), \
                self.assertRaises(NoTextError) as cm:
            self.convert(r)
        self.assertIn("Apple Vision", str(cm.exception))

    def test_a_document_converted_whole_is_rescued_by_apple_vision(self):
        from rag_search.core import indexer
        from rag_search.core.conversion import applevision
        out, info = self.tmp / "whole.md", {}
        with mock.patch.object(applevision, "why_not", return_value=""), \
                mock.patch.object(applevision, "read_page", side_effect=lambda src, n, **k: f"Passbook page {n} text"):
            how = indexer._rescue_whole(self.pdf, out, self.profile, "pdf", src_sha="abc", ocr=None, info=info,
                                        failure="no text extracted")
        self.assertEqual(how, "converted")
        pages = pagemd.split_pages(out.read_text())
        self.assertEqual(sorted(pages), [1, 2, 3])                       # the blank page 4 is not read
        recs = {x["page"]: x for x in info["page_records"]}
        self.assertEqual(recs[1]["reader"]["tool"], "apple-vision")
        self.assertEqual(recs[4]["reader"], {"tool": "none"})
        self.assertTrue(out.with_name("whole.md.sha256").read_text().startswith("abc\n"))
        with mock.patch.object(applevision, "why_not", return_value="Apple Vision needs a Mac"):
            self.assertEqual(indexer._rescue_whole(self.pdf, out, self.profile, "pdf", src_sha="abc", ocr=None,
                                                   info={}, failure="x"), "")

    def test_a_reader_error_propagates(self):
        with self.assertRaises(RuntimeError):
            self.convert(FakeReader(fail=True))
        self.assertEqual(self.cache.stats()["entries"], 0)

    def test_on_page_is_called_for_every_page_in_order(self):
        seen = []
        self.convert(FakeReader(), on_page=lambda rec, total: seen.append((rec["page"], total)))
        self.assertEqual(sorted(seen), [(1, 4), (2, 4), (3, 4), (4, 4)])


class AppleVisionTextTests(unittest.TestCase):
    def test_phrases_become_lines_in_reading_order(self):
        from rag_search.core.conversion import applevision
        # Vision boxes: (x, y, w, h) as fractions, origin bottom-left
        items = [("Passbook No", 0.9, (0.10, 0.50, 0.12, 0.03)),
                 ("Account No", 0.9, (0.10, 0.60, 0.12, 0.03)),
                 ("00112233445", 0.9, (0.50, 0.605, 0.15, 0.03)),
                 (": 1", 0.9, (0.50, 0.50, 0.05, 0.03)),
                 ("   ", 0.9, (0.1, 0.1, 0.1, 0.03)),
                 ("Branch", 0.9, (0.10, 0.90, 0.10, 0.03))]
        text = applevision.lines_to_text(items)
        self.assertEqual(text.split("\n\n"), ["Branch", "Account No  00112233445", "Passbook No  : 1"])

    def test_why_not_on_this_machine(self):
        from rag_search.core.conversion import applevision
        with mock.patch("sys.platform", "linux"):
            self.assertIn("needs a Mac", applevision.why_not())
        with mock.patch("sys.platform", "darwin"), mock.patch("importlib.util.find_spec", return_value=None):
            self.assertIn("ocrmac is not installed", applevision.why_not())
        with mock.patch("sys.platform", "darwin"), mock.patch("importlib.util.find_spec", return_value=object()):
            self.assertEqual(applevision.why_not(), "")


class PictureTextTests(unittest.TestCase):
    """docling drops the text inside a picture from its Markdown; the conversion takes it from the document."""

    class Prov:
        def __init__(self, page_no):
            self.page_no = page_no

    class Item:
        def __init__(self, text, page_no):
            self.text, self.prov = text, [PictureTextTests.Prov(page_no)]

    class Doc:
        def __init__(self, md, items):
            self.md, self.items, self.asked = md, items, []

        def export_to_markdown(self, page_no=None):
            return self.md

        def iterate_items(self, page_no=None, traverse_pictures=False):
            self.asked.append((page_no, traverse_pictures))
            return iter([(i, 1) for i in self.items])

    def test_real_text_counts_letters_not_markers(self):
        from rag_search.core.docling_convert import has_real_text, real_chars

        self.assertFalse(has_real_text("<!-- page 1 -->\n\n<!-- image -->\n\nOther"))
        self.assertEqual(real_chars("<!-- image -->"), 0)
        # the Markdown of a photographed passbook as docling exported it (three pages, nothing but placeholders)
        passbook = ("<!-- page 1 -->\n\n<!-- image -->\n\nOther\n\n<!-- page 2 -->\n\n<!-- image -->\n\n"
                    "<!-- page 3 -->\n\n<!-- image -->\n")
        self.assertFalse(has_real_text(passbook))
        self.assertTrue(has_real_text("# Hi"))                      # a short page is still a page
        self.assertTrue(has_real_text("Other than that, the total is 40"))
        self.assertTrue(has_real_text("Statement of account for the period ending March 2024"))
        self.assertTrue(has_real_text("हिन्दी में खाता विवरण और शेष राशि"))

    def test_text_inside_the_picture_is_used_when_the_export_is_a_placeholder(self):
        from rag_search.core.docling_convert import export_page

        doc = self.Doc("<!-- image -->\n\nOther", [self.Item("Savings account passbook, branch entries", 2),
                                                    self.Item("Opening balance 1,000.00", 2),
                                                    self.Item("a line of page 3 only, not for this page", 3)])
        md, note = export_page(doc, 2)
        self.assertIn("Savings account passbook", md)
        self.assertIn("Opening balance", md)
        self.assertNotIn("page 3 only", md)
        self.assertIn("inside the page's picture", note)
        self.assertEqual(doc.asked[0], (2, True))

    def test_normal_text_is_left_alone_and_a_dumb_docling_does_not_break(self):
        from rag_search.core.docling_convert import export_page

        doc = self.Doc("A page with plenty of real text on it, as docling exports it.", [self.Item("x" * 40, 1)])
        self.assertEqual(export_page(doc, 1)[1], "")
        self.assertEqual(doc.asked, [])

        class Broken(self.Doc):
            def iterate_items(self, *a, **k):
                raise RuntimeError("no such thing")

        md, note = export_page(Broken("<!-- image -->", []), 1)
        self.assertEqual((md, note), ("<!-- image -->", ""))


class GateTests(unittest.TestCase):
    def test_indic_text_is_not_garbled(self):
        from rag_search.core.docling_convert import page_text_ok
        self.assertTrue(page_text_ok("यह एक परीक्षण पृष्ठ है जिसमें पर्याप्त पाठ है। मराठी: हे एक चाचणी पान आहे."))
        self.assertFalse(page_text_ok("\ufffd\ufffd\ufffd abc \ufffd\ufffd"))

    def test_coverage_and_empty(self):
        g = gate.check_page("", branch_kind="digital", profile={"chars": 500})
        self.assertEqual(gate.failed(g), ["coverage"])
        self.assertEqual(gate.check_page("", branch_kind="digital", profile={"chars": 30})["verdict"], "empty")
        self.assertEqual(gate.check_page("", branch_kind="scan", profile={"ink": 0.0})["verdict"], "empty")
        self.assertEqual(gate.failed(gate.check_page("", branch_kind="scan", profile={"ink": 0.05})), ["coverage"])
        self.assertEqual(gate.failed(gate.check_page("word " * 30, branch_kind="digital", profile={"chars": 140})), [])

    def test_script_mismatch_and_noise(self):
        deva = "यह एक परीक्षण पृष्ठ है जिसमें पर्याप्त पाठ है"
        self.assertEqual(gate.failed(gate.check_page(deva, branch_kind="digital", profile={"chars": 40, "script": "Devanagari"})), [])
        latin = "this page should have been devanagari but came out as plain latin letters"
        self.assertEqual(gate.failed(gate.check_page(latin, branch_kind="digital", profile={"chars": 70, "script": "Devanagari"})), ["script"])
        noise = " ".join("a b c d e f g h".split() * 6)
        self.assertEqual(gate.failed(gate.check_page(noise, branch_kind="scan")), ["script"])

    def test_docling_grade_and_table_shape(self):
        self.assertEqual(gate.failed(gate.check_page("fine words here", branch_kind="digital", confidence={"grade": "poor", "low": 0.2})), ["docling_grade"])
        self.assertEqual(gate.failed(gate.check_page("| a | b |\n|---|---|\n| 1 |\n| 2 | 3 | 4 |\n", branch_kind="digital")), ["table_shape"])
        sparse = "| a | b |\n|---|---|\n| | |\n| | |\n| 1 | |\n"
        self.assertEqual(gate.failed(gate.check_page(sparse, branch_kind="digital")), ["table_shape"])
        self.assertEqual(gate.failed(gate.check_page("| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n", branch_kind="digital")), [])


class PageCacheTests(TempHome):
    def test_put_get_gc_and_references(self):
        c = pagecache.PageCache(self.paths.workspace)
        k1, k2 = pagecache.cache_key("h1", "docling:digital", "s"), pagecache.cache_key("h2", "docling:digital", "s")
        self.assertNotEqual(k1, pagecache.cache_key("h1", "docling:scan", "s"))
        self.assertNotEqual(k1, pagecache.cache_key("h1", "docling:digital", "s2"))
        c.put(k1, {"md": "one"})
        c.put(k2, {"md": "two"})
        self.assertEqual(c.get(k1)["md"], "one")
        self.assertIsNone(c.get("0" * 40))
        self.assertEqual(c.stats()["entries"], 2)
        mdf = self.paths.markup / "a" / "d.md"
        mdf.parent.mkdir(parents=True)
        trace.write_trace(trace.trace_path_for(mdf), source="d", src_sha="x",
                          pages=[{"page": 1, "key": k1}], summary={})
        keep = pagecache.referenced_keys(self.paths.markup)
        self.assertEqual(keep, {k1})
        self.assertEqual(c.gc(keep)["removed"], 0)                     # k2 is young: protected by the grace period
        self.assertEqual(c.gc(keep, now=time.time() + 7 * 3600)["removed"], 1)
        self.assertIsNotNone(c.get(k1))
        self.assertIsNone(c.get(k2))
        self.assertEqual(c.clear(), 1)


@unittest.skipUnless(HAVE_PDF, "needs pypdfium2 and Pillow")
class RoutedBase(ConversionBase):
    def setUp(self):
        super().setUp()
        os.environ["RAG_SEARCH_ROUTING"] = "pages"
        no_real_reader(self)                 # scanned pages take the docling fallback here, on a Mac too
        self.reader = FakeReader()
        p = mock.patch("rag_search.core.conversion.routed.default_reader", lambda: self.reader)
        p.start()
        self.addCleanup(p.stop)
        self.log = self.tmp / "events.jsonl"
        self.pdf = self.sdir / "reports" / "r.pdf"
        self.pdf.parent.mkdir(parents=True)
        write_routing_pdf(self.pdf)

    def go(self, **kw):
        return self.run_index(stage_log=self.log, **kw)

    def page_events(self):
        return [json.loads(x) for x in self.log.read_text().splitlines() if '"event": "page"' in x]


@unittest.skipUnless(HAVE_PDF, "needs pypdfium2 and Pillow")
class RoutedIndexTests(RoutedBase):
    def test_pdf_is_read_per_page_and_traced(self):
        summary = self.go()
        self.assertEqual(summary["indexed"], 1)
        self.assertEqual(self.reader.calls, [(1, 2, "digital"), (3, 3, "scan")])
        t = trace.read_trace(self.trace_file("reports/r"))
        self.assertEqual([p["branch"] for p in t["pages"]], ["digital", "digital", "fallback", "fallback"])
        self.assertEqual([p["outcome"] for p in t["pages"]], ["pass", "pass", "low", "no_text"])
        s = t["summary"]
        self.assertEqual(s["strip"], "d2f2")
        self.assertEqual(s["step_s"]["read"], 0.6)
        self.assertNotIn("cached_pages", s)
        self.assertEqual(self.meta("reports/r")["conversion"]["branches"], {"digital": 2, "fallback": 2})
        evs = self.page_events()
        self.assertEqual(sorted(e["page"] for e in evs), [1, 2, 3, 4])
        self.assertEqual({e["of"] for e in evs}, {4})
        self.assertEqual(summary["page_cache"]["entries"], 3)           # three pages were read and kept
        self.assertEqual(summary["conversion"]["step_s"]["read"], 0.6)

    def test_rerun_reads_nothing_twice(self):
        self.go()
        calls = list(self.reader.calls)
        summary = self.go(force_md=True)
        self.assertEqual(self.reader.calls, calls)
        s = trace.read_trace(self.trace_file("reports/r"))["summary"]
        self.assertEqual(s["strip"], "d2f2")                       # counted by what the pages are, not by how they came
        self.assertEqual(s["cached_pages"], 3)
        self.assertEqual(summary["conversion"]["branches"], {"digital": 2, "fallback": 2})
        self.assertEqual(summary["conversion"]["cached_pages"], 3)
        recs = trace.read_trace(self.trace_file("reports/r"))["pages"]
        self.assertEqual([p["branch"] for p in recs], ["cached", "cached", "cached", "fallback"])   # the record keeps how

    def test_events_say_what_kind_of_pages_a_document_has_and_what_each_page_is(self):
        self.go()
        first = [json.loads(x) for x in self.log.read_text().splitlines() if '"event": "plan"' in x]
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0]["pages"], 4)
        self.assertEqual(sum(first[0]["branches"].values()), 4)
        self.assertEqual(first[0]["branches"].get("digital"), 2)
        self.log.write_text("")
        self.go(force_md=True)                                     # everything from the page cache
        pages = self.page_events()
        self.assertEqual({e["cache"] for e in pages if e["page"] <= 3}, {"hit"})
        self.assertEqual(sorted(e["kind"] for e in pages if e["cache"] == "hit"), ["digital", "digital", "fallback"])
        self.assertTrue(all(e["branch"] == "cached" for e in pages if e["cache"] == "hit"))

    def test_unchanged_pages_are_not_read_again_when_the_file_changes(self):
        txt = self.sdir / "reports" / "t.pdf"
        write_text_pdf(txt, [TEXT + " one", TEXT + " two"])
        self.go()
        self.reader.calls.clear()
        write_text_pdf(txt, [TEXT + " one", TEXT + " TWO changed"])
        self.go()
        self.assertEqual(self.reader.calls, [(2, 2, "digital")])
        recs = trace.read_trace(self.trace_file("reports/t"))["pages"]
        self.assertEqual([p["branch"] for p in recs], ["cached", "digital"])

    def test_cache_gc_keeps_what_traces_refer_to(self):
        self.go()
        old = list(pagecache.PageCache(self.paths.workspace).keys())
        self.assertEqual(len(old), 3)
        for f in pagecache.PageCache(self.paths.workspace).root.rglob("*.json"):
            os.utime(f, (1, 1))                                         # old entries ...
        stray = pagecache.PageCache(self.paths.workspace)
        stray.put("ab" + "0" * 38, {"md": "nobody refers to me"})
        os.utime(stray._file("ab" + "0" * 38), (1, 1))
        summary = self.go(force_md=True)
        self.assertEqual(summary["page_cache"]["removed"], 1)           # ... only the unreferenced one goes
        self.assertEqual(summary["page_cache"]["entries"], 3)

    def test_document_routing_is_the_escape_hatch(self):
        os.environ["RAG_SEARCH_ROUTING"] = "document"
        self.go()
        self.assertEqual(self.reader.calls, [])
        t = trace.read_trace(self.trace_file("reports/r"))
        self.assertIn("route=document", t["convert"].split("|"))
        self.assertFalse(any(p.get("cache") for p in t["pages"]))

    def test_a_failing_reader_falls_back_to_whole_document_conversion(self):
        self.reader.fail = True
        summary = self.go()
        self.assertEqual(summary["indexed"], 1)
        self.assertEqual(summary["errors"], [])
        t = trace.read_trace(self.trace_file("reports/r"))
        self.assertIn("page routing failed", t["pages"][0]["note"])
        self.assertIn("page routing failed", t["summary"]["note"])

    def test_changing_the_routing_converts_again(self):
        self.go()
        self.assertEqual(self.go()["skipped_fresh"], 1)
        os.environ["RAG_SEARCH_ROUTING"] = "document"
        self.assertEqual(self.go()["indexed"], 1)

    def test_a_pdf_with_no_text_anywhere_is_reported_as_no_text(self):
        self.reader.bad = {1: "", 2: "", 3: ""}
        summary = self.go()
        self.assertEqual(summary["indexed"], 0)
        self.assertEqual(len(summary["no_text"]), 1)


@unittest.skipUnless(HAVE_PDF, "needs pypdfium2 and Pillow")
class RoutedInterfaceTests(RoutedBase):
    def setUp(self):
        super().setUp()
        self.go()

    def test_trace_command_shows_cache_time_and_failed_checks(self):
        rc, out, err = run("trace", "reports/r")
        self.assertEqual(rc, 0, err)
        self.assertIn("p.1", out)
        self.assertIn("0.2 s", out)                                   # a run of two pages took 0.4 s: 0.2 s each
        self.assertIn("no_text", out)
        self.go(force_md=True)
        rc, out, _ = run("trace", "reports/r")
        self.assertIn("page cache", out)
        rc, out, _ = run("trace", "reports/r", "--page", "3")
        data = json.loads(out[out.index("{"):])
        self.assertEqual(data["cache"], "hit")
        self.assertEqual(data["was"], "fallback")

    def test_index_status_shows_run_totals_with_cache_and_steps(self):
        self.go(force_md=True)
        from rag_search.cli import _fmt_conv_totals
        lines = "\n".join(_fmt_conv_totals({"pages": 4, "docs": 1, "branches": {"cached": 3}, "step_s": {"read": 1.2, "gate": 0.1},
                                             "cached_pages": 3, "gate_failed": {"script": 1}}))
        self.assertIn("3 from the page cache", lines)
        self.assertIn("failed checks: script 1", lines)

    def test_page_cache_command_and_route(self):
        rc, out, _ = run("index", "cache")
        self.assertEqual(rc, 0)
        self.assertIn("3 page(s)", out)
        d = Dash(self.paths, read_only=True)
        self.addCleanup(d.close)
        st, js, _, _ = d.req("GET", "/api/conversion/page-cache")
        self.assertEqual((st, js["result"]["entries"]), (200, 3))
        rc, out, _ = run("index", "cache", "--clear")
        self.assertIn("removed 3", out)
        self.assertIn("0 page(s)", out)
        rc, out, _ = run("index", "cache", "--json")
        self.assertEqual(json.loads(out)["entries"], 0)


class LiveViewTests(TempHome):
    def test_page_events_give_live_counts_and_lane_progress(self):
        log = self.paths.jobs / "j9.events.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        now = time.time()
        rows = [{"ts": now - 30, "event": "stage", "file": "a/x.pdf", "stage": "convert", "status": "start", "pid": 7}]
        for i, (branch, cache) in enumerate([("digital", "miss"), ("digital", "miss"), ("cached", "hit"), ("fallback", "miss")], 1):
            rows.append({"ts": now - 30 + i * 5, "event": "page", "file": "a/x.pdf", "pid": 7, "page": i, "of": 10,
                         "branch": branch, "outcome": "low" if i == 4 else "pass", "cache": cache})
        log.write_text("".join(json.dumps(r) + "\n" for r in rows))
        st = runview._read_new(log)
        v = runview.live_view(st, True)
        self.assertEqual((v["pages"], v["cached"]), (4, 1))
        self.assertEqual(v["branches"], {"digital": 2, "cached": 1, "fallback": 1})
        self.assertEqual(v["outcomes"], {"pass": 3, "low": 1})
        self.assertEqual(v["open"], {"a/x.pdf": {"done": 4, "of": 10}})
        self.assertGreater(v["pages_per_min"], 0)
        lane = runview.lanes_for(self.paths, "j9", True, now=now, state=st)[0]
        self.assertEqual(lane["progress"], {"done": 4, "of": 10})
        with open(log, "a") as fh:                                    # the document finishes
            fh.write(json.dumps({"ts": now, "event": "stage", "file": "a/x.pdf", "stage": "convert", "status": "done", "pid": 7}) + "\n")
        v = runview.live_view(runview._read_new(log), True)
        self.assertEqual(v["open"], {})
        self.assertEqual(v["pages"], 4)                                # the counts stay for the run


if __name__ == "__main__":
    unittest.main()
