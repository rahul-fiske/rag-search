"""How core/docling_convert.py reads what docling hands back: page numbering of a converted range, per-page
facts, the text inside pictures, confidence scores, the exit code of the command.  Tier A: docling's document is a
small fake with the attributes the code reads, so every version difference it guards against can be reproduced.
(That the real docling gives such documents is tier B's job.)"""

from __future__ import annotations

import json
import types
import unittest
from pathlib import Path
from unittest import mock

from tests.helpers import TempHome

from rag_search.core import docling_convert as dc


def prov(page, l=0.0, r=1.0, t=1.0, b=0.0):          # noqa: E741
    return types.SimpleNamespace(page_no=page, bbox=types.SimpleNamespace(l=l, r=r, t=t, b=b))


class FakeDoc:
    """A docling document: pages by number, tables and pictures with provenance, per-page Markdown."""

    def __init__(self, pages, tables=(), pictures=(), text_items=(), sizes=None, md_error_on=()):
        self.pages = {n: types.SimpleNamespace(size=(sizes or {}).get(n, types.SimpleNamespace(width=100, height=100)))
                      for n in pages}
        self.tables, self.pictures, self._items, self._md_error_on = list(tables), list(pictures), list(text_items), set(md_error_on)
        self.markdown = {n: f"text of page {n}" for n in pages}

    def num_pages(self):
        return len(self.pages)

    def export_to_markdown(self, page_no=None):
        if page_no in self._md_error_on:
            raise RuntimeError("export failed")
        return self.markdown[page_no] if page_no else "\n\n".join(self.markdown.values())

    def iterate_items(self, page_no=None, traverse_pictures=False):
        return iter([(i, 1) for i in self._items])


class ConvertRangeTests(TempHome):
    def read(self, doc, first, last, mode="digital"):
        with mock.patch.object(dc, "_convert_document", return_value=doc):
            return dc.convert_range(Path("x.pdf"), first, last, mode)

    def test_a_range_keeps_the_original_page_numbers_when_docling_does(self):
        res = self.read(FakeDoc([4, 5, 6]), 4, 6)
        self.assertEqual(sorted(res["pages"]), [4, 5, 6])
        self.assertEqual(res["pages"][5], "text of page 5")
        self.assertEqual(sorted(res["stats"]), [4, 5, 6])

    def test_a_range_that_docling_renumbered_from_one_is_mapped_back(self):
        res = self.read(FakeDoc([1, 2, 3]), 7, 9)
        self.assertEqual(sorted(res["pages"]), [7, 8, 9])
        self.assertEqual(res["pages"][8], "text of page 2")
        self.assertEqual([res["stats"][n]["page"] for n in (7, 8, 9)], [7, 8, 9])

    def test_a_docling_that_ignores_the_range_or_numbers_pages_strangely_is_reported_as_unsupported(self):
        with self.assertRaises(dc.RangeUnsupported) as cm:
            self.read(FakeDoc(list(range(1, 11))), 3, 4)               # all ten pages came back
        self.assertIn("returned 10 pages", str(cm.exception))
        with self.assertRaises(dc.RangeUnsupported) as cm:
            self.read(FakeDoc([20, 21]), 3, 4)                         # numbers that match neither scheme
        self.assertIn("numbered its pages", str(cm.exception))

    def test_a_format_that_does_not_number_its_pages_is_taken_as_asked(self):
        doc = FakeDoc([])
        doc.pages = None
        doc.markdown = {3: "only page three"}
        res = self.read(doc, 3, 3)
        self.assertEqual(res["pages"], {3: "only page three"})

    def test_the_mode_decides_the_ocr_and_an_unknown_mode_is_refused(self):
        seen = []

        def spy(src, suffix, cfg, rng=None):
            seen.append((cfg["ocr"], rng))
            return FakeDoc([1])

        with mock.patch.object(dc, "_convert_document", side_effect=spy):
            dc.convert_range(Path("x.pdf"), 1, 1, "digital")
            dc.convert_range(Path("x.pdf"), 1, 1, "scan")
        self.assertEqual(seen, [("auto", (1, 1)), ("force", (1, 1))])
        with self.assertRaises(ValueError):
            dc.convert_range(Path("x.pdf"), 1, 1, "photo")


class PageFactsTests(TempHome):
    def test_tables_pictures_and_big_pictures_are_counted_per_page(self):
        doc = FakeDoc([1, 2],
                      tables=[types.SimpleNamespace(prov=[prov(1)]), types.SimpleNamespace(prov=[prov(1)])],
                      pictures=[types.SimpleNamespace(prov=[prov(2, r=90, t=90)]),             # 81 % of a 100 x 100 page
                                types.SimpleNamespace(prov=[prov(2, r=10, t=10)])])            # 1 %
        stats = {s["page"]: s for s in dc.collect_page_stats(doc, {1: "a b", 2: "c"}, {2: {"grade": "good"}})}
        self.assertEqual((stats[1]["tables"], stats[1]["pictures"], stats[1]["chars"]), (2, 0, 2))
        self.assertEqual((stats[2]["pictures"], stats[2]["big_pictures"], stats[2]["confidence"]), (2, 1, {"grade": "good"}))

    def test_a_picture_without_a_size_is_counted_but_not_called_big_and_a_broken_doc_gives_plain_stats(self):
        doc = FakeDoc([1], pictures=[types.SimpleNamespace(prov=[prov(1)])])
        doc.pages = {}                                                   # no page sizes at all
        s = dc.collect_page_stats(doc, {1: "x"}, {})[0]
        self.assertEqual((s["pictures"], s["big_pictures"]), (1, 0))
        broken = types.SimpleNamespace(tables=[types.SimpleNamespace(prov=[object()])])    # no page_no: must not raise
        self.assertEqual(dc.collect_page_stats(broken, {1: "x"}, {})[0]["tables"], 0)

    def test_docling_confidence_scores_are_read_whatever_shape_the_version_gives(self):
        grade = types.SimpleNamespace(name="GOOD")
        score = types.SimpleNamespace(parse_score=0.91234, layout_score=float("nan"), table_score=None,
                                      ocr_score="0.5", mean_score=0.8, low_score=0.7,
                                      mean_grade=grade, low_grade=None)
        as_dict = types.SimpleNamespace(confidence=types.SimpleNamespace(pages={1: score}))
        got = dc.extract_confidence(as_dict)
        self.assertEqual(got[1]["parse"], 0.912)
        self.assertNotIn("layout", got[1])                               # NaN is left out
        self.assertNotIn("table", got[1])
        self.assertEqual((got[1]["ocr"], got[1]["grade"]), (0.5, "good"))
        as_list = types.SimpleNamespace(confidence=types.SimpleNamespace(pages=[score]))
        self.assertIn(1, dc.extract_confidence(as_list))
        self.assertEqual(dc.extract_confidence(types.SimpleNamespace()), {})
        self.assertEqual(dc.extract_confidence(types.SimpleNamespace(confidence=types.SimpleNamespace(pages=3))), {})


class PictureTextTests(TempHome):
    def test_text_inside_a_picture_is_used_only_when_the_export_has_none_and_a_failing_export_is_an_empty_page(self):
        doc = FakeDoc([1], text_items=[types.SimpleNamespace(text="Inside the picture", prov=[prov(1)]),
                                       types.SimpleNamespace(text="Other page", prov=[prov(2)]),
                                       types.SimpleNamespace(text="  ", prov=[prov(1)]),
                                       types.SimpleNamespace(text=None, prov=[])])
        doc.markdown[1] = "<!-- image -->\n\nOther"
        md, note = dc.export_page(doc, 1)
        self.assertEqual(md, "Inside the picture")
        self.assertIn("inside the page's picture", note)
        broken = FakeDoc([1], md_error_on=[1])
        self.assertEqual(dc.export_page(broken, 1), ("", ""))

    def test_an_older_docling_without_the_page_arguments_is_still_read(self):
        class Old(FakeDoc):
            calls = 0

            def iterate_items(self, *a, **k):
                Old.calls += 1
                if k:
                    raise TypeError("unexpected keyword argument 'page_no'")
                return super().iterate_items()

        doc = Old([1], text_items=[types.SimpleNamespace(text="Found anyway", prov=[])])
        doc.markdown[1] = "<!-- image -->"
        self.assertEqual(dc.export_page(doc, 1)[0], "Found anyway")
        self.assertEqual(Old.calls, 2)

        class Hopeless(FakeDoc):
            def iterate_items(self, *a, **k):
                raise TypeError("no")

        hopeless = Hopeless([1])
        hopeless.markdown[1] = "<!-- image -->"
        self.assertEqual(dc._picture_text(hopeless, 1), "")


class ConvertFileTests(TempHome):
    def convert(self, doc, name="a.pdf"):
        src = self.tmp / name
        src.write_bytes(b"%PDF-1.4")
        with mock.patch.object(dc, "_convert_document", return_value=doc), \
                mock.patch.object(dc, "resolve_ocr_mode", side_effect=lambda s, cfg: (cfg, "scanned: reading every page")):
            return dc.convert_file(src, self.tmp / "out" / "a.md"), self.tmp / "out" / "a.md"

    def test_a_pages_note_and_the_ocr_choice_are_printed_and_the_markdown_is_page_marked(self):
        doc = FakeDoc([1, 2])
        doc.markdown[1] = "<!-- image -->"
        doc._items = [types.SimpleNamespace(text="Words inside a scan", prov=[prov(1)])]
        import contextlib
        import io

        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            info, out = self.convert(doc)
        self.assertIn("note: a.pdf: scanned: reading every page", err.getvalue())
        self.assertIn("note: a.pdf page 1:", err.getvalue())
        text = out.read_text()
        self.assertIn("<!-- page 1 -->\n\nWords inside a scan", text)
        self.assertEqual(info["pages"], 2)

    def test_a_document_without_page_numbers_is_exported_whole_as_page_one(self):
        class Flat(FakeDoc):
            def num_pages(self):
                raise AttributeError("no page model")

        doc = Flat([1])
        doc.markdown = {1: "A whole document without pages"}
        info, out = self.convert(doc, "a.docx")
        self.assertEqual(info["pages"], 1)
        self.assertIn("<!-- page 1 -->", out.read_text())

    def test_asciidoc_that_docling_cannot_read_is_indexed_as_text_but_a_pdf_is_not(self):
        src = self.tmp / "deploy.adoc"
        src.write_text("= Deploy\n\nNeedle: azure quarry deployment.\n\n* one\n* two\n", encoding="utf-8")
        out = self.tmp / "out" / "deploy.md"
        with mock.patch.object(dc, "_convert_document", side_effect=RuntimeError("Pipeline SimplePipeline failed")):
            info = dc.convert_file(src, out)
            self.assertEqual(info["pages"], 1)
            self.assertTrue(out.read_text(encoding="utf-8").startswith("<!-- page 1 -->\n\n= Deploy"))
            self.assertIn("azure quarry deployment", out.read_text(encoding="utf-8"))
            empty = self.tmp / "empty.adoc"
            empty.write_text("  \n", encoding="utf-8")
            with self.assertRaises(dc.NoTextError):
                dc.convert_file(empty, self.tmp / "out" / "empty.md")
            pdf = self.tmp / "x.pdf"
            pdf.write_bytes(b"%PDF-1.4")
            with self.assertRaises(RuntimeError):                          # no fallback for a format that is not text
                dc.convert_file(pdf, self.tmp / "out" / "x.md")

    def test_pages_of_placeholders_only_are_not_text(self):
        doc = FakeDoc([1, 2])
        doc.markdown = {1: "<!-- image -->\n\nOther", 2: "<!-- image -->"}
        with mock.patch.object(dc, "damaged_pdf_reason", return_value=""), self.assertRaises(dc.NoTextError) as cm:
            self.convert(doc)                                        # a PDF that opens: probably a scan
        self.assertIn("document reader", str(cm.exception))
        with self.assertRaises(dc.NoTextError) as cm:
            self.convert(doc)                                        # the stand-in file is not a PDF at all
        self.assertIn("damaged or incomplete", str(cm.exception))
        self.assertNotIn("scanned PDF", str(cm.exception))

    def test_the_command_writes_the_facts_file_and_says_no_text_by_its_exit_code(self):
        src = self.tmp / "a.pdf"
        src.write_bytes(b"%PDF-1.4")
        info = {"pages": 2, "seconds": 0.5, "page_stats": []}
        with mock.patch.object(dc, "convert_file", return_value=info):
            self.assertEqual(dc.main([str(src), str(self.tmp / "o.md"), "--ocr", "--info", str(self.tmp / "i.json")]), 0)
        self.assertEqual(json.loads((self.tmp / "i.json").read_text())["pages"], 2)
        with mock.patch.object(dc, "convert_file", side_effect=dc.NoTextError("nothing")):
            self.assertEqual(dc.main([str(src), str(self.tmp / "o.md")]), dc.EXIT_NO_TEXT)
        with mock.patch.object(dc, "convert_file", side_effect=RuntimeError("boom")):
            self.assertEqual(dc.main([str(src), str(self.tmp / "o.md")]), 1)


class SettingsAndProbeTests(TempHome):
    def test_an_unknown_routing_mode_is_refused_with_the_choices(self):
        with self.assertRaises(ValueError) as cm:
            dc.convert_settings(env={"RAG_SEARCH_ROUTING": "sideways"})
        self.assertIn("RAG_SEARCH_ROUTING must be one of", str(cm.exception))

    def test_a_text_layer_probe_that_cannot_run_says_why(self):
        pdf = self.tmp / "a.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        import sys

        with mock.patch.dict(sys.modules, {"pypdfium2": None}):
            self.assertEqual(dc.text_layer_report(pdf)["reason"], "pypdfium2 is not installed")
        self.assertFalse(dc.text_layer_report(pdf)["reliable"])             # a PDF that does not parse

    def test_a_mixed_script_page_and_a_page_of_nothing_are_named_and_judged(self):
        self.assertEqual(dc.dominant_script("abcdefghij αβγδε абвгд אבגדה"), "Mixed")      # no script dominates
        self.assertEqual(dc.dominant_script("English heading " + "यह एक परीक्षण पृष्ठ है " * 3), "Devanagari")
        self.assertEqual(dc.dominant_script("too few"), "")
        self.assertFalse(dc._page_text_ok("   "))


if __name__ == "__main__":
    unittest.main()
