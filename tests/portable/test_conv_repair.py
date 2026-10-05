"""Repair of suspect table cells (phase P4): replacing one cell, finding it among a second reader's
words, the accept rule, and the whole path through the converter.  Readers are fakes
(``FakeVlmBackend`` for the model, ``FakeSecondReader`` for Apple Vision): no MLX, no ocrmac."""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

from unittest import mock

from tests.portable.test_conv_vlm import FAKE, HAVE_PDF, VlmBase
from tests.portable.test_conversion import ConversionBase
from tests.portable.test_conv_routed import FakeReader

from rag_search.core.conversion import gate, pagecache, profiler, repair, routed, tables, trace, validators, vlm
from rag_search.core.conversion.repair import Word

if HAVE_PDF:
    from PIL import Image, ImageDraw

GOOD = """| Date | Description | Debit | Credit | Balance |
|---|---|---|---|---|
| 01-03-2024 | Opening | | | 10,000.00 |
| 02-03-2024 | Salary | | 2,819.00 | 12,819.00 |
| 03-03-2024 | Rent | 5,000.00 | | 7,819.00 |
| 04-03-2024 | Grocery | 450.00 | | 7,369.00 |
"""
BAD = GOOD.replace("2,819.00 | 12,819.00", "2,81.00 | 12,819.00")          # one digit lost

ROWS = [("01-03-2024", "Opening", "", "", "10,000.00"),
        ("02-03-2024", "Salary", "", "2,819.00", "12,819.00"),
        ("03-03-2024", "Rent", "5,000.00", "", "7,819.00"),
        ("04-03-2024", "Grocery", "450.00", "", "7,369.00")]
COLX = [(0.05, 0.16), (0.20, 0.32), (0.42, 0.54), (0.60, 0.72), (0.78, 0.90)]


def statement_words(rows=ROWS, y0=0.10, pitch=0.05, h=0.02) -> list[Word]:
    """What a second reader sees on a statement page: a header line, then one line per row."""
    out: list[Word] = []
    for ri, cells in enumerate([("Date", "Description", "Debit", "Credit", "Balance"), *rows]):
        top = y0 + ri * pitch
        for text, (l, r) in zip(cells, COLX):                 # noqa: E741
            for k, tok in enumerate(text.split()):
                n = len(text.split())
                w = (r - l) / n
                out.append(Word(tok, l + k * w, top, l + (k + 1) * w - 0.005, top + h))
    return out


class NumberTests(unittest.TestCase):
    def test_indian_amounts_with_a_slash_dash_are_not_negative(self):
        from decimal import Decimal
        for text, want in (("500/-", 500), ("Rs. 1,500/-", 1500), ("1,23,456/=", 123456), ("(500)", -500),
                           ("500-", -500), ("-500", -500), ("500 Dr", -500), ("500 Cr", 500)):
            self.assertEqual(tables.parse_number(text), Decimal(want), text)


class ReplaceCellTests(unittest.TestCase):
    def test_a_pipe_cell_is_replaced_and_nothing_else_changes(self):
        new = tables.replace_cell("before\n\n" + BAD + "\nafter", 0, 2, 3, "2,819.00")
        self.assertEqual(new, "before\n\n" + GOOD + "\nafter")

    def test_an_html_cell_is_replaced(self):
        md = "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2,81.00</td></tr></table>"
        self.assertEqual(tables.replace_cell(md, 0, 1, 1, "2,819.00"),
                         "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2,819.00</td></tr></table>")

    def test_the_header_row_and_the_second_table_can_be_addressed(self):
        md = "| A | B |\n|---|---|\n| 1 | 2 |\n\n| C | D |\n|---|---|\n| 3 | 4 |\n"
        self.assertIn("| X | D |", tables.replace_cell(md, 1, 0, 0, "X"))
        self.assertIn("| 3 | 9 |", tables.replace_cell(md, 1, 1, 1, "9"))

    def test_cells_that_cannot_be_addressed_are_refused(self):
        self.assertIsNone(tables.replace_cell(BAD, 3, 1, 1, "x"))
        self.assertIsNone(tables.replace_cell(BAD, 0, 99, 1, "x"))
        self.assertIsNone(tables.replace_cell(BAD, 0, 1, 99, "x"))
        merged = "<table><tr><th>A</th><th>B</th></tr><tr><td colspan=\"2\">1</td></tr></table>"
        self.assertIsNone(tables.replace_cell(merged, 0, 1, 0, "x"))

    def test_a_pipe_in_the_new_text_is_escaped(self):
        out = tables.replace_cell(BAD, 0, 1, 1, "a|b")
        self.assertEqual(tables.find_tables(out)[0].rows[1][1], "a|b")

    def test_violations_carry_their_table(self):
        g = gate.check_page("| A | B |\n|---|---|\n| 1 | 2 |\n\n" + BAD, branch_kind="scan")
        self.assertEqual([v["table"] for v in g["violations"]], [1])
        v = g["violations"][0]
        self.assertEqual((v["row"], v["col"], v["found"], v["expected"]), (2, 3, "2,81.00", "2819.00"))


class LocateTests(unittest.TestCase):
    rows = tables.find_tables(BAD)[0].rows

    def test_the_cell_is_found_between_its_neighbours(self):
        loc = repair.locate(self.rows, 2, 3, "2,81.00", statement_words())
        self.assertEqual(loc["second"], "2,819.00")
        l, t, r, b = loc["box"]                               # noqa: E741
        self.assertGreaterEqual(l, 0.32)                  # not into the description
        self.assertLessEqual(r, 0.78)                     # nor the balance
        self.assertTrue(0.60 <= (l + r) / 2 <= 0.72)
        self.assertTrue(0.19 < t < 0.21 and 0.21 < b < 0.23)

    def test_a_row_that_is_not_on_the_page_is_not_found(self):
        words = [w for w in statement_words() if w.text not in ("Salary", "12,819.00", "02-03-2024")]
        self.assertIn("not found", repair.locate(self.rows, 2, 3, "2,81.00", words)["why"])
        self.assertIn("no text", repair.locate(self.rows, 2, 3, "2,81.00", [])["why"])

    def test_repeated_rows_are_ambiguous(self):
        dup = [("05-03-2024", "Rent", "", "", "1.00")]
        words = statement_words() + [Word(w.text, w.l, w.t + 0.5, w.r, w.b + 0.5) for w in statement_words()]
        self.assertIn("more than one line", repair.locate(self.rows, 2, 3, "2,81.00", words)["why"])
        del dup

    def test_two_texts_where_the_cell_should_be_are_told_apart_by_similarity(self):
        words = statement_words() + [Word("(note)", 0.60, 0.2, 0.64, 0.22), Word("2,819.00", 0.66, 0.2, 0.72, 0.22)]
        words = [w for w in words if not (w.t == 0.2 and w.text == "2,819.00" and w.l == 0.60)]
        loc = repair.locate(self.rows, 2, 3, "2,81.00", words)
        self.assertEqual(loc.get("second"), "2,819.00")

    def test_nothing_between_the_neighbours(self):
        words = [w for w in statement_words() if not (abs(w.t - 0.2) < 1e-9 and w.text == "2,819.00")]
        self.assertIn("nothing between", repair.locate(self.rows, 2, 3, "2,81.00", words)["why"])


class PageRereadTests(unittest.TestCase):
    def test_a_page_that_lost_its_table_is_not_an_improvement(self):
        heading_only = "Statement of account\n\nClosing balance carried forward to the next month."
        self.assertFalse(repair.not_smaller(BAD, heading_only))
        self.assertTrue(repair.not_smaller(BAD, GOOD))
        no_numbers = GOOD.replace("12,819.00", "x").replace("7,819.00", "y").replace("7,369.00", "z")
        self.assertFalse(repair.not_smaller(GOOD, no_numbers))
        shorter = "| a | b |\n|---|---|\n| 1.00 | 2.00 |\n"
        self.assertFalse(repair.not_smaller("long text " * 40 + shorter, shorter))


class RepairCellsTests(unittest.TestCase):
    def go(self, reads, md=BAD, words=None, **kw):
        asked: list = []

        def read_cell(box):
            asked.append(box)
            return reads.pop(0) if isinstance(reads, list) else reads
        res = repair.repair_cells(md, gate.check_page(md, branch_kind="scan")["violations"],
                                  words if words is not None else statement_words(), read_cell, **kw)
        res["asked"] = asked
        return res

    def test_two_reads_and_the_arithmetic_agree_so_the_cell_is_replaced(self):
        res = self.go("2,819.00")
        self.assertEqual((res["fixed"], res["tried"]), (1, 1))
        self.assertEqual(res["md"], GOOD)
        c = res["cells"][0]
        self.assertEqual((c["status"], c["before"], c["after"], c["second"]), ("fixed", "2,81.00", "2,819.00", "2,819.00"))
        self.assertEqual((c["row"], c["col"], c["role"]), (2, 3, "credit"))
        self.assertEqual(gate.check_page(res["md"], branch_kind="scan")["verdict"], "ok")

    def test_the_same_reading_again_is_not_a_fix(self):
        res = self.go("2,81.00")
        self.assertEqual((res["fixed"], res["cells"][0]["status"]), (0, "unchanged"))
        self.assertEqual(res["md"], BAD)

    def test_a_model_the_second_reader_contradicts_is_rejected(self):
        res = self.go("2,918.00")
        self.assertEqual(res["cells"][0]["status"], "disagree")
        self.assertEqual(res["md"], BAD)

    def test_a_reading_that_is_not_a_number_is_rejected(self):
        res = self.go("two thousand")
        self.assertEqual(res["cells"][0]["status"], "not_a_number")

    def test_two_agreeing_reads_that_do_not_fix_the_arithmetic_are_not_enough(self):
        words = [Word("2,918.00", w.l, w.t, w.r, w.b) if (w.text == "2,819.00") else w for w in statement_words()]
        res = self.go("2,918.00", words=words)
        self.assertEqual(res["cells"][0]["status"], "not_confirmed")
        self.assertEqual(res["md"], BAD)

    def test_a_cell_that_cannot_be_found_is_left_alone(self):
        res = self.go("2,819.00", words=[Word("hello", 0.1, 0.1, 0.2, 0.12)])
        self.assertEqual((res["fixed"], res["cells"][0]["status"]), (0, "not_located"))
        self.assertEqual(res["asked"], [])                      # the model is not asked about a guess

    def test_a_reader_that_raises_is_logged_and_the_page_is_kept(self):
        def boom(box):
            raise vlm.ReaderTimeout("timed out after 5 s")
        res = repair.repair_cells(BAD, validators.page_violations(BAD), statement_words(), boom)
        self.assertEqual(res["cells"][0]["status"], "error")
        self.assertIn("timed out", res["cells"][0]["why"])

    def test_the_next_suspect_is_looked_at_again_after_a_fix(self):
        # two separate errors; both are found, the second one after the first is fixed
        md = BAD.replace("450.00", "4.50")
        viol = validators.page_violations(md)
        self.assertEqual(len(viol), 2)
        answers = iter(["2,819.00", "450.00"])
        res = repair.repair_cells(md, viol, statement_words(), lambda box: next(answers))
        self.assertEqual((res["fixed"], res["tried"]), (2, 2))
        self.assertEqual(res["md"], GOOD)

    def test_at_most_max_cells_are_tried(self):
        res = self.go("2,81.00", max_cells=1)
        self.assertEqual(res["tried"], 1)

    def test_an_html_table_is_repaired_too(self):
        html = "<table><tr><th>Date</th><th>Description</th><th>Debit</th><th>Credit</th><th>Balance</th></tr>" + "".join(
            "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
            for row in [("01-03-2024", "Opening", "", "", "10,000.00"), ("02-03-2024", "Salary", "", "2,81.00", "12,819.00"),
                        ("03-03-2024", "Rent", "5,000.00", "", "7,819.00"), ("04-03-2024", "Grocery", "450.00", "", "7,369.00")]
        ) + "</table>"
        res = self.go("2,819.00", md=html)
        self.assertEqual(res["fixed"], 1)
        self.assertIn("<td>2,819.00</td>", res["md"])
        self.assertNotIn("2,81.00", res["md"])

    def test_a_balance_cell_is_repaired(self):
        md = GOOD.replace("12,819.00", "12,181.00")
        viol = validators.page_violations(md)
        self.assertEqual(viol[0]["role"], "balance")
        res = repair.repair_cells(md, viol, statement_words(), lambda box: "12,819.00")
        self.assertEqual(res["md"], GOOD)


class SecondReaderTests(VlmBase):
    def test_second_reader_choices(self):
        self.assertIsNone(repair.second_reader() if repair.second_why_not() else None)
        os.environ["RAG_SEARCH_REPAIR_SECOND"] = "off"
        self.assertIsNone(repair.second_reader())
        self.assertIn("switched off", repair.second_why_not())
        os.environ["RAG_SEARCH_REPAIR_SECOND"] = "tests.helpers:FakeSecondReader"
        self.assertEqual(repair.second_reader().id, "fake-second")

    def test_ocrmac_needs_a_mac(self):
        r = repair.OcrMacSecond()
        if os.uname().sysname != "Darwin":
            self.assertIn("needs a Mac", r.why_not())

    def test_the_mode_and_the_builder(self):
        self.assertEqual(repair.mode(), "auto")
        os.environ["RAG_SEARCH_REPAIR"] = "off"
        self.assertEqual(repair.mode(), "off")
        self.assertIsNone(repair.build())
        del os.environ["RAG_SEARCH_REPAIR"]
        os.environ["RAG_SEARCH_VLM"] = "off"
        self.assertIsNone(repair.build())

    def test_the_repair_model_is_the_8b_unless_another_is_chosen_and_the_reader_s_own_when_it_is_the_same(self):
        from rag_search import models

        self.assertEqual(models.vlm_selection(models.REPAIR)[0], "mlx-community/Qwen3-VL-8B-Instruct-4bit")
        self.assertNotEqual(models.vlm_selection(models.REPAIR)[0], models.vlm_selection(models.READER)[0])
        os.environ["RAG_SEARCH_REPAIR_MODEL"] = vlm.shared().model
        a = vlm.repair_shared()
        self.assertIs(a, vlm.shared())
        os.environ["RAG_SEARCH_REPAIR_MODEL"] = "someorg/another-vlm"
        b = vlm.repair_shared()
        self.assertIsNot(b, vlm.shared())
        self.assertEqual(b.model, "someorg/another-vlm")
        self.assertIs(vlm.repair_shared(), b)


def write_statement_pdf(path: Path, dpi: int = 200) -> None:
    """A one-page scan (an image), dark enough to count as ink, at *dpi*."""
    w, h = int(8.27 * dpi), int(11.69 * dpi)
    im = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(im)
    for y in range(int(0.1 * h), int(0.4 * h), int(0.05 * h)):
        d.rectangle((int(0.05 * w), y, int(0.9 * w), y + int(0.02 * h)), fill=(0, 0, 0))
    im.save(path, resolution=dpi)


@unittest.skipUnless(HAVE_PDF, "needs pypdfium2 and Pillow")
class RepairInConverterTests(VlmBase):
    def setUp(self):
        super().setUp()
        os.environ["RAG_SEARCH_REPAIR_SECOND"] = "tests.helpers:FakeSecondReader"
        self.second_plan = self.tmp / "second.json"
        self.words(statement_words())
        os.environ["RAG_TEST_SECOND_PLAN"] = str(self.second_plan)
        self.pdf = self.tmp / "stmt.pdf"
        write_statement_pdf(self.pdf)
        self.profile = profiler.profile_file(self.pdf)
        self.cache = pagecache.PageCache(self.paths.workspace)
        self.out = self.tmp / "out.md"
        self.plan(text="<!-- reader -->\n" + BAD, cell="2,819.00")

    def words(self, words):
        self.second_plan.write_text(json.dumps({"words": [[w.text, w.l, w.t, w.r, w.b] for w in words]}))

    def convert(self, repairer="auto", reader=None):
        r = reader or self.reader()
        rp = repair.Repairer(r, repair.second_reader(), reader_model=r.model) if repairer == "auto" else repairer
        return routed.convert_pdf(self.pdf, self.out, self.profile, cache=self.cache, reader=FakeReader(),
                                  scan_reader=r, repairer=rp), r

    def page(self, res):
        return res["records"][0]

    def test_the_profile_of_the_fixture_is_a_sharp_scan(self):
        self.assertEqual([p["page"] for p in self.profile["pages"]], [1])
        self.assertGreaterEqual(self.profile["pages"][0].get("dpi") or 0, 150)

    def test_a_suspect_cell_is_repaired(self):
        res, _ = self.convert()
        rec = self.page(res)
        self.assertEqual((rec["branch"], rec["outcome"]), ("raster", "repaired"))
        self.assertEqual((rec["repair"]["tried"], rec["repair"]["fixed"], rec["repair"]["tier"]), (1, 1, "cells"))
        cell = rec["repair"]["cells"][0]
        self.assertEqual((cell["before"], cell["after"], cell["status"]), ("2,81.00", "2,819.00", "fixed"))
        self.assertEqual(rec["repair"]["second"], "fake-second")
        self.assertIn("repair", rec["time_s"])
        self.assertNotIn("gate", rec)                         # the page passes after the repair
        text = self.out.read_text()
        self.assertIn("2,819.00", text)
        self.assertNotIn("2,81.00 ", text)
        s = trace.summarize(res["records"])
        self.assertEqual((s["repaired_cells"], s["repair_tried"]), (1, 1))
        self.assertEqual(s["outcomes"], {"repaired": 1})

    def test_tokens_and_gpu_time_of_the_cell_reads_are_counted(self):
        res, r = self.convert()
        self.assertEqual(self.page(res)["tokens"], r.tokens)       # page read + one cell read
        self.assertEqual(r.pages_read, 2)

    def test_a_repaired_page_is_cached_and_not_repaired_again(self):
        self.convert()
        before = self.calls()
        res, _ = self.convert()
        self.assertEqual(self.calls(), before)                  # neither the page nor the cell is read again
        rec = self.page(res)
        self.assertEqual((rec["branch"], rec["cache"], rec["outcome"]), ("cached", "hit", "repaired"))
        self.assertEqual(rec["repair"]["fixed"], 1)
        self.assertIn("2,819.00", self.out.read_text())

    def test_a_failed_attempt_is_cached_too_and_not_repeated_with_the_same_models(self):
        self.plan(text="<!-- reader -->\n" + BAD, cell="2,918.00")
        res, _ = self.convert()
        rec = self.page(res)
        self.assertEqual(rec["outcome"], "low")
        self.assertEqual(rec["repair"]["cells"][0]["status"], "disagree")
        self.assertIn("2,81.00", self.out.read_text())            # the first reading is kept
        before = self.calls()
        self.convert()
        self.assertEqual(self.calls(), before)
        os.environ["RAG_SEARCH_REPAIR_SECOND"] = "off"             # another repair setup: tried again
        res2, _ = self.convert(repairer=repair.Repairer(self.reader(), None, reader_model="fake/model"))
        self.assertEqual(self.page(res2)["outcome"], "low")

    def test_no_second_reader_means_no_cell_repair_and_says_why(self):
        r = self.reader()
        res, _ = self.convert(repairer=repair.Repairer(r, None, reader_model=r.model), reader=r)
        rec = self.page(res)
        self.assertEqual(rec["outcome"], "low")
        self.assertIn("cells not repaired", rec["note"])
        self.assertNotIn("repair", {k for k in rec if k == "repair" and rec[k].get("fixed")})
        self.assertIn("2,81.00", self.out.read_text())

    def test_a_second_reader_that_fails_does_not_fail_the_page(self):
        self.second_plan.write_text(json.dumps({"error": "vision crashed"}))
        res, _ = self.convert()
        rec = self.page(res)
        self.assertEqual(rec["outcome"], "low")
        self.assertIn("vision crashed", rec["note"])

    def test_a_low_resolution_scan_is_repaired_but_stays_flagged(self):
        write_statement_pdf(self.pdf, dpi=100)
        self.profile = profiler.profile_file(self.pdf)
        res, _ = self.convert()
        rec = self.page(res)
        self.assertEqual(rec["repair"]["fixed"], 1)
        self.assertEqual(rec["outcome"], "low")
        self.assertEqual([c["name"] for c in rec["gate"]["checks"]], ["low_resolution"])

    def test_repair_off_leaves_the_page_as_read(self):
        res, _ = self.convert(repairer=None)
        rec = self.page(res)
        self.assertEqual(rec["outcome"], "low")
        self.assertNotIn("repair", rec)
        self.assertIn("2,81.00", self.out.read_text())

    def test_a_digital_page_is_not_repaired(self):
        # a born-digital page with a failing statement is a statement that does not add up, not a misreading
        from tests.portable.test_conversion import write_text_pdf

        write_text_pdf(self.pdf, ["Statement"])
        prof = profiler.profile_file(self.pdf)
        r = self.reader()

        class Docling(FakeReader):
            def read(self, src, first, last, mode):
                res = super().read(src, first, last, mode)
                res["pages"][1] = BAD
                return res
        res = routed.convert_pdf(self.pdf, self.out, prof, cache=None, reader=Docling(), scan_reader=r,
                                 repairer=repair.Repairer(r, repair.second_reader(), reader_model=r.model))
        self.assertNotIn("repair", res["records"][0])
        self.assertEqual(self.calls(), 0)

    def test_the_repair_model_reads_the_page_again_when_cells_cannot_be_fixed(self):
        os.environ["RAG_SEARCH_REPAIR_MODEL"] = "fake/repair-model"
        self.plan(text="<!-- reader -->\n" + BAD, cell="2,918.00",
                  text_by_model={"fake/repair-model": "<!-- repair model -->\n" + GOOD})
        reader = self.reader()
        rep_reader = vlm.VlmReader("fake/repair-model", style="instruct", need_gb=1.0)
        self.addCleanup(rep_reader.close)
        rp = repair.Repairer(rep_reader, repair.second_reader(), reader_model=reader.model)
        self.assertTrue(rp.reread)
        res, _ = self.convert(repairer=rp, reader=reader)
        rec = self.page(res)
        self.assertEqual((rec["outcome"], rec["repair"]["tier"]), ("repaired", "page"))
        self.assertIn("repair model", self.out.read_text())
        self.assertIn("read again by the repair model", rec["note"])
        self.assertEqual(rec["repair"]["model"], "fake/repair-model")
        self.assertEqual(trace.summarize(res["records"])["models"], ["fake/model", "fake/repair-model"])

    SHIFTED = ("| Date | Description | Cheque No | Debit | Credit | Balance |\n|---|---|---|---|---|---|\n"
               "| 01-03-2024 | Opening | 10,000.00 | 10,000.00 | 10,000.00 | 10,000.00 |\n"
               "| 02-03-2024 | Salary | 2,819.00 | 12,819.00 | 12,819.00 | 12,819.00 |\n"
               "| 03-03-2024 | Rent | 5,000.00 | 7,819.00 | 7,819.00 | 7,819.00 |\n"
               "| 04-03-2024 | Grocery | 450.00 | 7,369.00 | 7,369.00 | 7,369.00 |\n")
    LOOP = "\n".join(["The said property shall be conveyed to the purchaser free of all charges."] * 70)

    def escalating(self, reader_text, repair_text):
        os.environ["RAG_SEARCH_REPAIR_MODEL"] = "fake/repair-model"
        self.plan(text=reader_text, cell="x", text_by_model={"fake/repair-model": repair_text})
        reader = self.reader()
        rep_reader = vlm.VlmReader("fake/repair-model", style="instruct", need_gb=1.0)
        self.addCleanup(rep_reader.close)
        return reader, repair.Repairer(rep_reader, repair.second_reader(), reader_model=reader.model)

    def test_a_flagged_page_without_a_suspect_cell_goes_to_the_repair_model(self):
        reader, rp = self.escalating(self.SHIFTED, GOOD)             # columns mixed up: no cell to repair, the page is read again
        res, _ = self.convert(repairer=rp, reader=reader)
        rec = self.page(res)
        self.assertEqual((rec["outcome"], rec["repair"]["tier"]), ("repaired", "page"))
        self.assertIn("Salary", self.out.read_text())
        self.assertIn("read again by the repair model", rec["note"])

    def test_a_runaway_is_replaced_by_a_shorter_clean_reading(self):
        reader, rp = self.escalating(self.LOOP, GOOD)
        res, _ = self.convert(repairer=rp, reader=reader)
        rec = self.page(res)
        self.assertEqual(rec["repair"]["tier"], "page")
        self.assertNotIn("shall be conveyed", self.out.read_text())
        self.assertIn("Salary", self.out.read_text())

    def test_nothing_is_escalated_when_the_repair_model_is_the_reader(self):
        reader = self.reader()
        flagged_only = ("| Date | Item | Value 1 | Value 2 | Value 3 |\n|---|---|---|---|---|\n"
                        + "\n".join(f"| 0{i}-03-2024 | Item {i} | {i}00.00 | {i}00.00 | {i}00.00 |" for i in range(1, 6)))
        self.plan(text=flagged_only)                                  # flagged (one amount in three columns), no suspect cell
        rp = repair.Repairer(reader, repair.second_reader(), reader_model=reader.model)    # the same model
        self.assertFalse(rp.reread)
        res, _ = self.convert(repairer=rp, reader=reader)
        self.assertEqual(self.page(res)["outcome"], "low")
        self.assertNotIn("repair", self.page(res))

    def test_a_clean_page_is_not_sent_to_the_repair_model(self):
        reader, rp = self.escalating(GOOD, BAD)
        res, _ = self.convert(repairer=rp, reader=reader)
        self.assertEqual(self.page(res)["outcome"], "pass")
        self.assertNotIn("repair", self.page(res))
        self.assertEqual(self.calls(), 1)                            # the reader's own call only

    def test_a_page_read_again_that_is_no_better_is_not_taken(self):
        os.environ["RAG_SEARCH_REPAIR_MODEL"] = "fake/repair-model"
        self.plan(text="<!-- reader -->\n" + BAD, cell="2,918.00",
                  text_by_model={"fake/repair-model": "<!-- repair model -->\n" + BAD})
        reader = self.reader()
        rep_reader = vlm.VlmReader("fake/repair-model", style="instruct", need_gb=1.0)
        self.addCleanup(rep_reader.close)
        rp = repair.Repairer(rep_reader, repair.second_reader(), reader_model=reader.model)
        res, _ = self.convert(repairer=rp, reader=reader)
        rec = self.page(res)
        self.assertEqual(rec["outcome"], "low")
        self.assertIn("still suspect", rec["note"])
        self.assertNotIn("repair model", self.out.read_text())

    def test_an_unexpected_error_in_repair_does_not_fail_the_page_or_the_document(self):
        r = self.reader()

        class Broken(repair.Repairer):
            def run(self, *a, **kw):
                raise RuntimeError("something nobody expected")
        res, _ = self.convert(repairer=Broken(r, repair.second_reader(), reader_model=r.model), reader=r)
        rec = self.page(res)
        self.assertEqual(rec["outcome"], "low")
        self.assertIn("repair not run (RuntimeError", rec["note"])
        self.assertIn("2,81.00", self.out.read_text())

    def test_a_page_read_again_from_an_unreadable_source_is_an_error_in_the_log_not_in_the_run(self):
        os.environ["RAG_SEARCH_REPAIR_MODEL"] = "fake/repair-model"
        rep_reader = vlm.VlmReader("fake/repair-model", style="instruct", need_gb=1.0)
        self.addCleanup(rep_reader.close)
        rp = repair.Repairer(rep_reader, None, reader_model="fake/model")
        out = rp.run(self.tmp / "missing.pdf", 1, False, BAD, validators.page_violations(BAD), page_ok=lambda md: True)
        self.assertEqual(out["md"], BAD)
        self.assertIn("page not read again", out["note"])

    def test_a_second_attempt_on_a_cached_page_keeps_the_earlier_fixes_on_record(self):
        two = BAD.replace("450.00", "4.50")                      # two separate errors
        self.plan(text="<!-- reader -->\n" + two, cell="2,819.00")
        only_credit = [w for w in statement_words() if w.text != "450.00"]
        self.words(only_credit)                                   # the second reader misses the second cell
        first, _ = self.convert()
        rec1 = self.page(first)
        self.assertEqual((rec1["outcome"], rec1["repair"]["fixed"], rec1["repair"]["tried"]), ("low", 1, 2))

        class Other:                                              # a different second reader: tried again
            id = "fake-second-2"

            def words(self, image):
                return statement_words()
        self.plan(text="<!-- reader -->\n" + two, cell="450.00")
        r = self.reader()
        res, _ = self.convert(repairer=repair.Repairer(r, Other(), reader_model=r.model), reader=r)
        rec = self.page(res)
        self.assertEqual((rec["branch"], rec["outcome"]), ("cached", "repaired"))
        self.assertEqual((rec["repair"]["fixed"], rec["repair"]["tried"]), (2, 3))
        self.assertEqual([c["status"] for c in rec["repair"]["cells"]], ["fixed", "not_located", "fixed"])
        self.assertIn("450.00", self.out.read_text())
        self.assertNotIn("4.50", self.out.read_text())

    def test_a_dead_repair_reader_is_skipped_quietly(self):
        r = self.reader()
        rp = repair.Repairer(r, repair.second_reader(), reader_model=r.model)
        r.dead = "the model is not downloaded"
        res, _ = self.convert(repairer=rp, reader=r)
        rec = self.page(res)
        self.assertEqual(rec["branch"], "fallback")               # the page reader is dead too: docling read it
        self.assertNotIn("repair", rec)


@unittest.skipUnless(HAVE_PDF, "needs pypdfium2 and Pillow")
class StatementIndexBase(ConversionBase):
    """Environment for indexing a scanned statement (``bank/stmt.pdf``) with the fake readers."""

    def setUp(self):
        super().setUp()
        os.environ["RAG_SEARCH_ROUTING"] = "pages"
        os.environ["RAG_SEARCH_VLM_BACKEND"] = FAKE
        os.environ["RAG_SEARCH_VLM_FREE_GB"] = "64"
        os.environ["RAG_SEARCH_REPAIR_SECOND"] = "tests.helpers:FakeSecondReader"
        self.plan_file = self.tmp / "plan.json"
        self.calls_log = self.tmp / "calls.log"
        os.environ["RAG_TEST_VLM_PLAN"] = str(self.plan_file)
        self.second = self.tmp / "second.json"
        self.second.write_text(json.dumps({"words": [[w.text, w.l, w.t, w.r, w.b] for w in statement_words()]}))
        os.environ["RAG_TEST_SECOND_PLAN"] = str(self.second)
        p = mock.patch("rag_search.core.conversion.routed.default_reader", lambda: FakeReader())
        p.start()
        self.addCleanup(p.stop)
        self.addCleanup(vlm.close_shared)
        self.pdf = self.sdir / "bank" / "stmt.pdf"
        self.pdf.parent.mkdir(parents=True)
        write_statement_pdf(self.pdf)

    def plan(self, d):
        self.plan_file.write_text(json.dumps({**d, "log": str(self.calls_log)}))

    def calls(self) -> int:
        return len(self.calls_log.read_text().splitlines()) if self.calls_log.exists() else 0

    def nodes(self):
        d = self.paths.index / "bank" / "stmt"
        return json.loads((d / "nodes.json").read_text())["nodes"]


class RepairIndexTests(StatementIndexBase):
    """A whole indexing run of a scanned statement: the repaired cell is in the index; a page that
    stays suspect is flagged on its chunks (search and MCP show it)."""

    def test_a_repaired_cell_reaches_the_index(self):
        self.plan({"text": BAD, "cell": "2,819.00"})
        summary = self.run_index()
        self.assertEqual(summary["indexed"], 1)
        conv = self.meta("bank/stmt")["conversion"]
        self.assertEqual((conv["outcomes"], conv["repaired_cells"], conv["repair_tried"]), ({"repaired": 1}, 1, 1))
        nodes = self.nodes()
        self.assertIn("2,819.00", "".join(n["text"] for n in nodes))
        self.assertNotIn("2,81.00 ", "".join(n["text"] for n in nodes))
        self.assertTrue(all("confidence" not in n["metadata"] for n in nodes))
        self.assertEqual(summary["conversion"]["repaired_cells"], 1)

    def test_a_page_that_stays_suspect_is_flagged_on_its_chunks(self):
        self.plan({"text": BAD, "cell": "2,918.00"})
        self.run_index()
        conv = self.meta("bank/stmt")["conversion"]
        self.assertEqual((conv["outcomes"], conv["low_pages"]), ({"low": 1}, [1]))
        self.assertTrue(all(n["metadata"].get("confidence") == "low" for n in self.nodes()))
        self.assertIn("2,81.00", "".join(n["text"] for n in self.nodes()))        # kept as read
        from rag_search import api

        t = api.conversion_trace(self.paths, "bank", "stmt")["result"]
        self.assertEqual(t["pages"][0]["repair"], {"tried": 1, "fixed": 0, "tier": None})

    def test_repair_off_by_setting_flags_the_page(self):
        os.environ["RAG_SEARCH_REPAIR"] = "off"
        self.plan({"text": BAD, "cell": "2,819.00"})
        self.run_index()
        self.assertEqual(self.meta("bank/stmt")["conversion"]["outcomes"], {"low": 1})

    def test_the_repair_setting_is_a_validated_tunable(self):
        from rag_search import spec

        t = [x for x in spec.TUNABLES if x.key == "repair"][0]
        self.assertEqual((t.section, t.env, tuple(t.choices)), ("indexer", "RAG_SEARCH_REPAIR", ("auto", "off")))


if __name__ == "__main__":
    unittest.main()
