"""Tables that continue across a page break (phase P4): ``reconcile.merge`` and its use in the converter."""

from __future__ import annotations

import unittest

from tests.helpers import TempHome
from tests.portable.test_conv_routed import FakeReader
from tests.portable.test_conversion import HAVE_PDF, write_text_pdf

from rag_search.core.conversion import pagecache, profiler, reconcile, routed, tables, trace

HEAD = "| Date | Description | Debit | Credit | Balance |\n|---|---|---|---|---|\n"
P3 = ("Statement of account\n\n" + HEAD +
      "| 01-03 | Opening | | | 10,000.00 |\n| 02-03 | Salary | | 2,819.00 | 12,819.00 |\n"
      "| 03-03 | Rent | 5,000.00 | | 7,819.00 |\n")
ROWS4 = ("| 04-03 | Grocery | 450.00 | | 7,369.00 |\n| 05-03 | Fuel | 300.00 | | 7,069.00 |\n"
         "| 06-03 | Cash | | 100.00 | 7,169.00 |\n")


class MergeTests(unittest.TestCase):
    def test_a_headerless_pipe_continuation_gets_the_header(self):
        res = reconcile.merge({3: P3, 4: ROWS4 + "\nend of statement"})
        self.assertEqual(res["tables"], [{"pages": [3, 4], "rows": 6, "header": "added", "ok": True, "violations": []}])
        self.assertEqual(res["pages"][3], P3)
        t = tables.find_tables(res["pages"][4])[0]
        self.assertEqual(t.header[0], "Date")
        self.assertEqual(len(t.body), 3)
        self.assertTrue(res["pages"][4].endswith("end of statement"))

    def test_docling_style_continuation_whose_first_data_row_became_the_header(self):
        p4 = "| 04-03 | Grocery | 450.00 | | 7,369.00 |\n|---|---|---|---|---|\n| 05-03 | Fuel | 300.00 | | 7,069.00 |\n"
        res = reconcile.merge({3: P3, 4: p4})
        self.assertEqual(res["tables"][0]["header"], "added")
        t = tables.find_tables(res["pages"][4])[0]
        self.assertEqual((t.header[0], len(t.body)), ("Date", 2))             # no row was lost

    def test_an_html_continuation(self):
        p4 = "<table>" + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in [
            ("04-03", "Grocery", "450.00", "", "7,369.00"), ("05-03", "Fuel", "300.00", "", "7,069.00")]) + "</table>"
        res = reconcile.merge({3: P3, 4: p4})
        self.assertEqual((res["tables"][0]["header"], res["tables"][0]["ok"]), ("added", True))
        self.assertEqual(len(tables.find_tables(res["pages"][4])[0].body), 2)

    def test_a_repeated_header_leaves_the_text_alone(self):
        p4 = HEAD + ROWS4
        res = reconcile.merge({3: P3, 4: p4})
        self.assertEqual(res["tables"][0]["header"], "repeated")
        self.assertEqual(res["tables"][0]["rows"], 6)                       # the repeated header is not counted twice
        self.assertEqual(res["pages"][4], p4)

    def test_the_balance_is_checked_across_the_page_break(self):
        bad = ROWS4.replace("7,369.00", "7,339.00").replace("7,069.00", "7,039.00").replace("7,169.00", "7,139.00")
        res = reconcile.merge({3: P3, 4: bad})
        t = res["tables"][0]
        self.assertFalse(t["ok"])
        v = t["violations"][0]
        self.assertEqual(v["page"], 4)                                      # the first row of page 4 disagrees with page 3
        self.assertEqual((v["validator"], v["found"]), ("running_balance", "450.00"))
        self.assertIn("across the page break", v["why"])
        self.assertEqual(v["row"], 1)                                       # row 1 of the continuation, header is row 0

    def test_a_suspect_that_each_page_shows_alone_is_not_reported_twice(self):
        bad = P3.replace("12,819.00", "12,181.00")                          # wrong inside page 3 itself
        res = reconcile.merge({3: bad, 4: ROWS4})
        self.assertEqual(res["tables"][0]["violations"], [])

    def test_totals_that_span_the_break_are_checked(self):
        head = "| Item | Amount |\n|---|---|\n"
        p1 = head + "| a | 10.00 |\n| b | 20.00 |\n"
        p2 = "| c | 5.00 |\n| Total | 36.00 |\n"
        res = reconcile.merge({1: p1, 2: p2})
        t = res["tables"][0]
        self.assertFalse(t["ok"])
        self.assertEqual(t["violations"][0]["validator"], "totals")
        self.assertEqual(t["violations"][0]["page"], 2)
        ok = reconcile.merge({1: p1, 2: "| c | 6.00 |\n| Total | 36.00 |\n"})
        self.assertTrue(ok["tables"][0]["ok"])

    def test_things_that_are_not_continuations(self):
        cases = {
            "other width": {3: P3, 4: "| 04-03 | x | 1.00 |\n| 05-03 | y | 2.00 |\n"},
            "text after the table": {3: P3 + "\nNotes follow.", 4: ROWS4},
            "text before the table": {3: P3, 4: "Page two intro\n\n" + ROWS4},
            "a table with a header of its own": {3: P3, 4: "| Name | City | Qty | Price | Note |\n|---|---|---|---|---|\n| a | b | c | d | e |\n"},
            "pages that are not next to each other": {3: P3, 5: ROWS4},
            "a year header": {3: P3, 4: "| Item | Note | 2023 | 2024 | 2025 |\n|---|---|---|---|---|\n| a | b | 1 | 2 | 3 |\n"},
            "no table on the first page": {3: "plain text", 4: ROWS4},
        }
        for name, pages in cases.items():
            res = reconcile.merge(pages)
            self.assertEqual(res["tables"], [], name)
            self.assertEqual(res["pages"], pages, name)

    def test_a_year_row_is_a_header_not_data(self):
        self.assertFalse(reconcile._data_like(["Item", "2023", "2024"]))
        self.assertTrue(reconcile._data_like(["04-03", "Grocery", "450.00"]))
        self.assertFalse(reconcile._data_like(["", "Balance", "Debit"]))

    def test_the_blank_line_after_a_rewritten_table_is_kept(self):
        p4 = ROWS4 + "\nClosing balance | 7,100.00\n"
        res = reconcile.merge({3: P3, 4: p4})
        self.assertIn("|\n\nClosing balance | 7,100.00", res["pages"][4])
        self.assertEqual(len(tables.find_tables(res["pages"][4])), 1)         # the next line is not swallowed as a row

    def test_a_row_of_column_numbers_is_not_data(self):
        p4 = "| 1 | 2 | 3 | 4 | 5 |\n|---|---|---|---|---|\n| a | b | c | d | e |\n"
        self.assertEqual(reconcile.merge({3: P3, 4: p4})["tables"], [])

    def test_a_suspect_at_the_break_with_the_same_figure_as_a_local_one_is_still_reported(self):
        p3 = ("| Date | Description | Debit | Credit | Balance |\n|---|---|---|---|---|\n"
              "| 01 | Opening | | | 1,000.00 |\n| 02 | Gift | | 5.00 | 1,010.00 |\n| 03 | Fee | 1.00 | | 1,009.00 |\n")
        p4 = "| 04 | Gift | | 5.00 | 2,000.00 |\n| 05 | Fee | 1.00 | | 1,999.00 |\n"
        res = reconcile.merge({3: p3, 4: p4})
        self.assertTrue(any(v["page"] == 4 for v in res["tables"][0]["violations"]), res["tables"])

    def test_three_pages_make_two_merges(self):
        res = reconcile.merge({3: P3, 4: ROWS4, 5: "| 07-03 | Gift | 50.00 | | 7,119.00 |\n| 08-03 | Tea | 19.00 | | 7,100.00 |\n"})
        self.assertEqual([t["pages"] for t in res["tables"]], [[3, 4], [4, 5]])
        self.assertTrue(all(t["ok"] for t in res["tables"]))


@unittest.skipUnless(HAVE_PDF, "needs pypdfium2 and Pillow")
class ConverterReconcileTests(TempHome):
    def setUp(self):
        super().setUp()
        self.pdf = self.tmp / "two.pdf"
        write_text_pdf(self.pdf, ["Statement page one " * 10, "Statement page two " * 10])
        self.profile = profiler.profile_file(self.pdf)
        self.out = self.tmp / "out.md"

    def convert(self, pages):
        pages = {n: md + ("\n\n" + "Statement page two " * 10 if n == 2 else "") for n, md in pages.items()}
        return routed.convert_pdf(self.pdf, self.out, self.profile, cache=pagecache.PageCache(self.paths.workspace),
                                  reader=FakeReader(bad=pages))

    def test_the_continuation_is_written_with_its_header_and_recorded(self):
        res = self.convert({1: P3, 2: ROWS4})
        text = self.out.read_text()
        self.assertEqual(text.count("| Date | Description |"), 2)
        recs = {r["page"]: r for r in res["records"]}
        self.assertEqual(recs[1]["reconcile"], {"role": "starts", "with": 2, "header": "added", "rows": 6, "ok": True})
        self.assertEqual(recs[2]["reconcile"]["role"], "continues")
        self.assertEqual((recs[1]["outcome"], recs[2]["outcome"]), ("pass", "pass"))
        s = trace.summarize(res["records"])
        self.assertEqual(s["merged_tables"], 1)
        self.assertEqual(trace.aggregate([s])["merged_tables"], 1)

    def test_a_wrong_figure_at_the_page_break_flags_the_page(self):
        bad = ROWS4.replace("7,369.00", "7,339.00").replace("7,069.00", "7,039.00").replace("7,169.00", "7,139.00")
        res = self.convert({1: P3, 2: bad})
        recs = {r["page"]: r for r in res["records"]}
        self.assertEqual((recs[1]["outcome"], recs[2]["outcome"]), ("pass", "low"))
        self.assertIn("table_across_pages", [c["name"] for c in recs[2]["gate"]["checks"]])
        self.assertEqual(recs[2]["reconcile"]["violations"][0]["found"], "450.00")
        self.assertNotIn("violations", recs[1]["reconcile"])
        self.assertEqual(trace.summarize(res["records"])["gate_failed"]["table_across_pages"], 1)

    def test_a_page_whose_checks_pass_once_it_has_the_header_is_no_longer_low(self):
        from unittest import mock

        from rag_search.core.conversion import gate

        real = gate.check_page

        def picky(md, **kw):                           # a gate that dislikes a table without its header
            g = real(md, **kw)
            if "| 04-03" in md and "| Date |" not in md:
                return {"verdict": "suspect", "checks": [{"name": "table_shape", "ok": False, "detail": "no header"}]}
            return g
        with mock.patch.object(gate, "check_page", picky):
            res = self.convert({1: P3, 2: ROWS4})
        recs = {r["page"]: r for r in res["records"]}
        self.assertEqual(recs[2]["outcome"], "pass")
        self.assertNotIn("gate", recs[2])

    def test_the_page_cache_keeps_what_the_reader_wrote_not_the_merge(self):
        cache = pagecache.PageCache(self.paths.workspace)
        routed.convert_pdf(self.pdf, self.out, self.profile, cache=cache, reader=FakeReader(bad={1: P3, 2: ROWS4 + "\n\n" + "Statement page two " * 10}))
        for k in cache.keys():                       # the continuation is cached as the reader wrote it: no header added
            md = cache.get(k)["md"]
            self.assertFalse("| 04-03" in md and "| Date |" in md)
        again = routed.convert_pdf(self.pdf, self.out, self.profile, cache=cache, reader=FakeReader())
        self.assertEqual(self.out.read_text().count("| Date | Description |"), 2)    # merged again from the cached pages
        self.assertEqual([r["cache"] for r in again["records"]], ["hit", "hit"])


if __name__ == "__main__":
    unittest.main()
