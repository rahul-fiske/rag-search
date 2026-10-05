"""HTML tables written by a document reader become Markdown pipe tables -- without losing a cell."""

from __future__ import annotations

import unittest

from rag_search.core.conversion import tables
from rag_search.core.conversion.tables import find_tables, normalize_html_tables, parse_html_tables


def html(rows, head=None):
    th = "<thead><tr>" + "".join(f"<th>{c}</th>" for c in head) + "</tr></thead>" if head else ""
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table border=\"1\">{th}{body}</table>"


class NormalizeTests(unittest.TestCase):
    def test_a_table_with_a_header_becomes_a_pipe_table_with_the_same_cells(self):
        src = html([["01/02/2020", "UPI/IMPS/123", "1,234.50 Cr"], ["03/02/2020", "BY INTT", "99.00"]],
                   head=["Date", "Particulars", "Amount"])
        out, n = normalize_html_tables(f"Intro\n\n{src}\n\nOutro\n")
        self.assertEqual(n, 1)
        self.assertNotIn("<table", out)
        self.assertIn("Intro", out)
        self.assertIn("Outro", out)
        before = parse_html_tables(src)[0].rows
        after = tables.parse_pipe_tables(out)[0].rows
        self.assertEqual(after, before)

    def test_a_table_without_th_takes_its_first_row_as_the_header(self):
        out, n = normalize_html_tables(html([["Date", "Amount"], ["01/02/2020", "5.00"]]))
        self.assertEqual(n, 1)
        t = tables.parse_pipe_tables(out)[0]
        self.assertEqual((t.header, t.body), (["Date", "Amount"], [["01/02/2020", "5.00"]]))

    def test_stacked_header_lines_stay_as_rows(self):
        rows = [["तारीख", "रकम"], ["DATE", "AMOUNT"], ["01/02/2020", "5.00"]]
        out, _ = normalize_html_tables(html(rows))
        self.assertEqual(tables.parse_pipe_tables(out)[0].rows, rows)

    def test_devanagari_numbers_and_pipes_inside_cells_are_kept_exactly(self):
        rows = [["शिल्लक", "a | b"], ["१२३", "9,99,999.00 Cr"]]
        out, _ = normalize_html_tables(html(rows))
        self.assertEqual(tables.parse_pipe_tables(out)[0].rows, rows)

    def test_merged_nested_unclosed_and_empty_tables_are_left_alone(self):
        merged = '<table><tr><td rowspan="2">x</td><td>y</td></tr><tr><td>z</td></tr></table>'
        nested = "<table><tr><td><table><tr><td>in</td></tr></table></td></tr></table>"
        unclosed = "<table><tr><td>a</td></tr>"
        empty = "<table><tr><td></td><td></td></tr></table>"
        for s in (merged, nested, unclosed, empty, "no table here | a | b"):
            self.assertEqual(normalize_html_tables(s), (s, 0), s)

    def test_only_the_plain_table_changes_when_a_page_has_two(self):
        merged = '<table><tr><td colspan="2">x</td></tr><tr><td>1</td><td>2</td></tr></table>'
        out, n = normalize_html_tables(f"{html([['a', 'b']])}\n\ntext\n\n{merged}\n")
        self.assertEqual(n, 1)
        self.assertIn(merged, out)
        self.assertEqual([t.kind for t in find_tables(out)], ["pipe", "html"])

    def test_idempotent_and_prose_untouched(self):
        md = "# Title\n\nSome prose with a < b and 5 > 3.\n\n" + html([["a", "b"], ["1", "2"]]) + "\n"
        once, n = normalize_html_tables(md)
        self.assertEqual(n, 1)
        self.assertEqual(normalize_html_tables(once), (once, 0))
        self.assertTrue(once.startswith("# Title\n\nSome prose with a < b and 5 > 3.\n\n|"))

    def test_blank_lines_inside_the_html_do_not_matter(self):
        src = "<table>\n\n  <tr>\n    <td>a</td>\n\n    <td>b</td>\n  </tr>\n\n  <tr><td>1</td><td>2</td></tr>\n</table>"
        out, n = normalize_html_tables(src)
        self.assertEqual((n, tables.parse_pipe_tables(out)[0].rows), (1, [["a", "b"], ["1", "2"]]))


if __name__ == "__main__":
    unittest.main()


class JoinSplitTableTests(unittest.TestCase):
    HEAD = "| Date | Narration | Amount |\n|---|---|---|\n| DATE | PARTICULARS | AMOUNT |\n"
    ROWS = "| 01/02/2020 | UPI/IMPS/1 | 5.00 |\n| 02/02/2020 | UPI/IMPS/2 | 6.00 |\n"

    def test_data_rows_after_a_blank_line_join_the_table_above(self):
        md = "Intro\n\n" + self.HEAD + "\n" + self.ROWS + "\nText after.\n"
        out, n = tables.join_split_pipe_tables(md)
        self.assertEqual(n, 1)                                      # one place was cut
        self.assertEqual([len(t.rows) for t in find_tables(out)], [4])
        self.assertIn("Text after.", out)
        self.assertEqual(tables.join_split_pipe_tables(out), (out, 0))

    def test_a_second_table_with_its_own_header_is_left_alone(self):
        md = self.HEAD + self.ROWS + "\n| A | B |\n|---|---|\n| 1 | 2 |\n"
        out, n = tables.join_split_pipe_tables(md)
        self.assertEqual(n, 0)
        self.assertEqual(out, md)

    def test_rows_of_another_width_or_after_prose_are_not_joined(self):
        md = self.HEAD + "\n| only |\n\nSome text\n\n" + self.ROWS
        self.assertEqual(tables.join_split_pipe_tables(md)[1], 0)
        wide = self.HEAD + "\n| a | b | c | d | e |\n"
        self.assertEqual(tables.join_split_pipe_tables(wide)[1], 0)

    def test_rows_that_leave_out_trailing_empty_cells_still_join(self):
        md = self.HEAD + "\n| 01/02/2020 | UPI |\n| 02/02/2020 | UPI |\n"
        out, n = tables.join_split_pipe_tables(md)
        self.assertEqual(n, 1)
        self.assertEqual([len(t.rows) for t in find_tables(out)], [4])

    def test_the_arithmetic_of_a_cut_statement_is_checked_after_joining(self):
        from rag_search.core.conversion import validators

        head = ("| Date | Particulars | Debit | Credit | Balance |\n|---|---|---|---|---|\n"
                "| DATE | PARTICULARS | DEBIT | CREDIT | BALANCE |\n")
        rows = ("| 01/02/2020 | a | | 100.00 | 1,100.00 Cr |\n| 02/02/2020 | b | 50.00 | | 1,050.00 Cr |\n"
                "| 03/02/2020 | c | | 25.00 | 1,075.00 Cr |\n| 04/02/2020 | d | 5.00 | | 1,070.00 Cr |\n")
        cut = head + "\n" + rows
        before = [validators.running_balance(t).get("checked") for t in find_tables(cut)]
        out, _ = tables.join_split_pipe_tables(cut)
        after = [validators.running_balance(t).get("checked") for t in find_tables(out)]
        self.assertEqual(before, [0])
        self.assertGreater(after[0], 0)
