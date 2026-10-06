"""A digital page against its own text layer (conversion/layer.py): the comparison, the fill, headers and footers,
the gate's coverage check and notes, and the routed read step (RAG_SEARCH_LAYER_FILL)."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from tests import corpus
from tests.helpers import TempHome
from tests.portable.test_conv_routed import FakeReader
from tests.portable.test_conversion import HAVE_PDF

from rag_search.core.conversion import gate, layer, pagecache, profiler, routed

PAGE = ("The switch accepts a login request from a port and returns an accept frame to the originator. "
        "Each frame carries a sequence count, a destination identifier and a payload of up to 2112 bytes.")
CELLS = "Word 0\nReserved 31\nPort 4096\nLength 512\nStatus 7\nBuffer 128"


class TokenTests(unittest.TestCase):
    def test_tokens_normalise_a_layer_and_markdown_alike(self):
        words, nums = layer.tokens("The ﬁnal exam-\r\nple costs 1,234.50 in snake\\_case, inter­\r\npreted")
        self.assertEqual(set(words), {"the", "final", "example", "costs", "in", "snake", "case", "interpreted"})
        self.assertEqual(set(nums), {"1234.50"})

    def test_verdicts(self):
        self.assertEqual(layer.verdict(0.99, 1.0, 50), "intact")
        self.assertEqual(layer.verdict(0.99, 0.5, 50), "lost text")
        self.assertEqual(layer.verdict(0.95, 1.0, 50), "uncertain")
        self.assertEqual(layer.verdict(1.0, 1.0, 5), "uncertain")                     # too short to judge
        self.assertEqual(layer.verdict(None, None, 0), "uncertain")

    def test_compare_and_missing_lines(self):
        layer_text = f"{PAGE}\r\n{CELLS.replace(chr(10), chr(13) + chr(10))}"
        md = f"## Frames\n\n{PAGE}\n"
        cmp = layer.compare(layer_text, md)
        self.assertEqual(cmp["verdict"], "lost text")
        self.assertLess(cmp["word_recall"], 0.9)
        lines = layer.missing_lines(layer_text, md)
        self.assertEqual(lines, CELLS.split("\n"))                                   # the page's own lines are found
        new, added = layer.fill(md, lines)
        self.assertEqual(added, 6)
        self.assertIn(layer.FILL_MARKER, new)
        self.assertEqual(layer.compare(layer_text, new)["verdict"], "intact")
        self.assertEqual(layer.missing_lines(layer_text, new), [])                   # nothing left to add

    def test_a_repeated_value_counts_as_missing_when_the_result_has_fewer(self):
        text = "Value 0\nValue 0\nValue 0\nValue 0"
        self.assertEqual(len(layer.missing_lines(text, "Value 0 Value 0")), 2)

    def test_the_amount_added_to_a_page_is_capped(self):
        new, added = layer.fill("x", [f"line {i} " + "y" * 90 for i in range(1000)])
        self.assertLess(len(new), layer.MAX_FILL_CHARS + 200)
        self.assertLess(added, 1000)
        self.assertEqual(layer.fill("x", []), ("x", 0))

    def test_running_headers_footers_and_page_numbers_are_not_text_of_the_page(self):
        parts = [("Acme Confidential", f"body {i}", f"Revision 4.86 June 16, 2024 page {i}") for i in range(1, 5)]
        bands = layer.repeated_bands(parts)
        self.assertEqual(bands, {"acme confidential", "revision #.# june #, # page #"})
        self.assertEqual(layer.repeated_bands(parts[:2]), set())
        page = "Acme Confidential\nThe port logs in.\nRevision 4.86 June 16, 2024 page 3"
        out = layer.without_bands("Acme Confidential", page, "Revision 4.86 June 16, 2024 page 3", bands)
        self.assertEqual(out.split(), "The port logs in.".split())
        self.assertTrue(layer.page_number_line(" 12 ") and layer.page_number_line("Page 3 of 10"))
        self.assertFalse(layer.page_number_line("Table 12"))
        words = "ports fabric login zoning frames credits switches links buffers timers".split()
        layers = [f"ACME Spec Rev 1.{i}\n" + "\n".join(f"{words[(i + k) % 10]} section {i}.{k} explains {words[k]}"
                                                     for k in range(5)) + f"\nPage {i} of 9\n" for i in range(1, 10)]
        boiler = layer.boilerplate(layers)
        clean, dropped = layer.clean_layer(layers[3], boiler)
        self.assertEqual((dropped, len(clean.strip().splitlines())), (2, 5))
        self.assertEqual(layer.boilerplate(layers[:3]), set())


class GateLayerTests(unittest.TestCase):
    TABLE = "| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 | 5 |\n"                         # a shifted row

    def test_coverage_is_the_layers_recall_and_layout_findings_become_notes_when_the_text_is_there(self):
        md = PAGE + "\n\n" + self.TABLE
        intact = {"verdict": "intact", "layer_words": 40, "word_recall": 1.0, "number_recall": 1.0}
        g = gate.check_page(md, branch_kind="digital", profile={"chars": 400}, layer=intact)
        self.assertEqual(g["verdict"], "ok")
        self.assertEqual([c["name"] for c in g["notes"]], ["table_shape"])
        lost = {"verdict": "lost text", "layer_words": 40, "word_recall": 0.6, "number_recall": 0.5}
        g = gate.check_page(md, branch_kind="digital", profile={"chars": 400}, layer=lost)
        self.assertEqual(g["verdict"], "suspect")
        self.assertEqual(sorted(gate.failed(g)), ["coverage", "table_shape"])        # real loss: the layout finding stays
        self.assertIn("60 % of the words", g["checks"][0]["detail"] if g["checks"][0]["name"] == "coverage" else g["checks"][1]["detail"])

    def test_without_a_layer_the_old_rule_and_scans_are_unchanged(self):
        md = PAGE
        self.assertEqual(gate.check_page(md, branch_kind="digital", profile={"chars": 2000})["verdict"], "suspect")
        g = gate.check_page(self.TABLE + PAGE, branch_kind="scan", profile={"ink": 0.1},
                            layer={"verdict": "intact", "layer_words": 40})
        self.assertEqual(gate.failed(g), ["table_shape"])                            # a layer means nothing for a scan
        short = {"verdict": "lost text", "layer_words": 3, "word_recall": 0.0}
        self.assertEqual(gate.check_page(PAGE, branch_kind="digital", profile={"chars": 50}, layer=short)["verdict"], "ok")


@unittest.skipUnless(HAVE_PDF, "needs pypdfium2 and Pillow")
class RoutedLayerTests(TempHome):
    """The text.pdf corpus file has three digital pages; the fake docling drops part of what each page says."""

    def setUp(self):
        super().setUp()
        self.pdf = corpus.copy("pdf/text.pdf", self.tmp / "t.pdf")
        self.profile = profiler.profile_file(self.pdf)
        self.cache = pagecache.PageCache(self.paths.workspace)
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(str(self.pdf))
        self.layers = [pdf[i].get_textpage().get_text_range().replace("\r\n", "\n") for i in range(len(pdf))]
        pdf.close()
        self.half = {n: " ".join(t.split()[: len(t.split()) // 2]) for n, t in enumerate(self.layers, 1)}

    def convert(self, reader, mode):
        out = self.tmp / f"out-{mode}.md"
        with mock.patch.dict(os.environ, {"RAG_SEARCH_LAYER_FILL": mode}):
            res = routed.convert_pdf(self.pdf, out, self.profile, cache=pagecache.PageCache(self.tmp / f"ws-{mode}"),
                                     reader=reader)
        return res, out.read_text(encoding="utf-8")

    def test_a_page_that_lost_text_is_completed_from_its_layer(self):
        res, text = self.convert(FakeReader(bad=self.half), "fill")
        recs = res["records"]
        self.assertTrue(all(r["layer"]["verdict"] == "intact" and r["layer"]["added_lines"] >= 1 for r in recs), recs)
        self.assertEqual({r["layer"]["before"] for r in recs}, {"lost text"})
        self.assertEqual([r["outcome"] for r in recs], ["pass"] * 3)
        self.assertIn(layer.FILL_MARKER, text)
        self.assertIn("extension four hundred", text)                                # page 3's own words are in the result
        self.assertTrue(all("text layer" in r["note"] for r in recs))

    def test_report_records_and_changes_nothing_off_does_not_look(self):
        res, text = self.convert(FakeReader(bad=self.half), "report")
        self.assertTrue(all(r["layer"]["would_add_lines"] >= 1 and r["layer"]["mode"] == "report" for r in res["records"]))
        self.assertNotIn(layer.FILL_MARKER, text)
        res, text = self.convert(FakeReader(bad=self.half), "off")
        self.assertTrue(all("layer" not in r for r in res["records"]))

    def test_an_intact_page_gets_nothing_and_a_hidden_layer_is_never_filled(self):
        class Whole(FakeReader):
            def read(s, src, first, last, mode):
                res = super().read(src, first, last, mode)
                res["pages"] = {n: self.layers[n - 1] for n in range(first, last + 1)}
                return res
        res, text = self.convert(Whole(), "fill")
        self.assertTrue(all(r["layer"]["verdict"] == "intact" and "added_lines" not in r["layer"] for r in res["records"]))
        self.assertNotIn(layer.FILL_MARKER, text)
        profile = {**self.profile, "pages": [{**p, "hidden_ocr_layer": True} for p in self.profile["pages"]]}
        out = self.tmp / "hidden.md"
        with mock.patch.dict(os.environ, {"RAG_SEARCH_LAYER_FILL": "fill"}):
            res = routed.convert_pdf(self.pdf, out, profile, cache=pagecache.PageCache(self.tmp / "ws-h"),
                                     reader=FakeReader(bad=self.half))
        self.assertTrue(all("layer" not in r for r in res["records"]))
        self.assertNotIn(layer.FILL_MARKER, out.read_text())

    def test_the_mode_setting(self):
        for value, want in (("", "fill"), ("fill", "fill"), ("REPORT", "report"), ("off", "off"), ("nonsense", "fill")):
            with mock.patch.dict(os.environ, {"RAG_SEARCH_LAYER_FILL": value}):
                self.assertEqual(routed.layer_fill_mode(), want)
