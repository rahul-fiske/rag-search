"""The runaway detector, the column checks and the HTML-table chunking (synthetic pages only)."""

from __future__ import annotations

import random
import unittest

from rag_search.core import chunker
from rag_search.core.conversion import gate
from rag_search.core.conversion.degenerate import assess, looping

WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa "
         "quebec romeo sierra tango uniform victor whiskey xray yankee zulu").split()


def prose(lines=40, seed=1):
    rnd = random.Random(seed)
    return "\n\n".join(" ".join(rnd.choice(WORDS) for _ in range(12)) + "." for _ in range(lines))


def statement(rows=40):
    head = ("| Line | Date | Particulars | Cheque No | Amount Withdrawn | Amount Deposited | Balance | For use |\n"
            "|---|---|---|---|---|---|---|---|\n")
    return head + "\n".join(
        f"| {i} | 0{i % 9 + 1}/02/2020 | UPI/IMPS/{1000 + i * 7}/ref{i * 13} | | {i * 37 + 100}.50 | | "
        f"{200000 - i * 937}.25 Cr | |" for i in range(1, rows + 1))


class DetectorTests(unittest.TestCase):
    def test_ordinary_pages_are_not_runaways(self):
        for md in (prose(), statement(), "", "short page", "<table><tr><td>a</td></tr></table>",
                   "\n".join(f"Clause {i}: the party shall pay {i * 100} rupees by the {i}th day." for i in range(60))):
            self.assertFalse(assess(md)["bad"], md[:40])

    def test_a_line_repeated_many_times_is_a_runaway(self):
        a = assess("\n".join(["<tr><td>The same long narration text here</td></tr>"] * 60))
        self.assertTrue(a["bad"])
        self.assertIn("repeated", a["why"])

    def test_a_phrase_repeated_in_running_text_is_a_runaway(self):
        a = assess(" ".join(["the said property shall be conveyed"] * 120))
        self.assertTrue(a["bad"])

    def test_empty_table_rows_filling_the_page_are_a_runaway(self):
        a = assess("<table>\n" + "<tr><td></td><td></td><td></td></tr>\n" * 200 + "</table>")
        self.assertTrue(a["bad"])
        self.assertIn("empty table row", a["why"])

    def test_a_script_that_is_not_on_the_page_is_a_runaway(self):
        rnd = random.Random(2)
        bengali = "\n".join("".join(chr(0x0985 + rnd.randrange(30)) for _ in range(20)) for _ in range(30))
        a = assess(bengali)
        self.assertTrue(a["bad"])
        self.assertIn("Bengali", a["why"])
        devanagari = "\n".join("".join(chr(0x0915 + rnd.randrange(30)) for _ in range(20)) for _ in range(30))
        self.assertFalse(assess(devanagari)["bad"])             # Marathi / Hindi is expected

    def test_looping_sees_a_repeating_tail_only(self):
        self.assertTrue(looping("intro. " + "the said property shall be conveyed. " * 80))
        self.assertTrue(looping("<table>\n" + "<tr><td></td></tr>\n" * 320))
        self.assertFalse(looping(prose(60) * 2))
        self.assertFalse(looping("short"))

    def test_looping_does_not_stop_a_page_with_a_repeated_structure(self):
        fields = "\n".join(f"Name: person {i}; Plot: {i * 7}; Share: {i}%; Remarks: nil" for i in range(12))
        self.assertFalse(looping(prose(30) + "\n\n" + fields))
        grid = "<table>\n" + "<tr><td></td><td></td><td></td></tr>\n" * 40 + "</table>\n" + prose(40)
        self.assertFalse(looping(grid))

    def test_the_gate_flags_a_runaway_only_for_pages_read_as_images(self):
        loop = "\n".join(["The same long narration text here for every row"] * 60)
        scan = gate.check_page(loop, branch_kind="scan")
        self.assertEqual(scan["verdict"], "suspect")
        self.assertIn("degenerate", gate.failed(scan))
        self.assertNotIn("degenerate", gate.failed(gate.check_page(loop, branch_kind="digital")))


class ColumnCheckTests(unittest.TestCase):
    HEAD = ("| Line | Date | Particulars | Cheque No | Amount Withdrawn | Amount Deposited | Balance | For use |\n"
            "|---|---|---|---|---|---|---|---|\n")

    def check(self, rows):
        return gate.check_page(self.HEAD + "\n".join(rows), branch_kind="scan")

    def test_one_amount_repeated_in_four_columns_of_every_row_is_flagged(self):
        rows = [f"| {i} | 0{i}/02/2020 | UPI/IMPS/{500 + i} | {i * 91 + 10}.00 | {i * 7 + 100000}.50 Cr | "
                f"{i * 7 + 100000}.50 Cr | {i * 7 + 100000}.50 Cr | {i * 7 + 100000}.50 Cr |" for i in range(1, 9)]
        res = self.check(rows)
        self.assertEqual(res["verdict"], "suspect")
        self.assertIn("same amount", res["checks"][0]["detail"])

    def test_balances_in_the_deposit_column_are_flagged(self):
        rows = [f"| {i} | 0{i}/02/2020 | UPI/IMPS/{500 + i} | | {i * 31 + 10}.00 | {i * 7 + 100000}.50 Cr | | |"
                for i in range(1, 7)]
        self.assertEqual(self.check(rows)["verdict"], "suspect")

    def test_amounts_in_the_cheque_column_are_flagged(self):
        rows = [f"| {i} | 0{i}/02/2020 | UPI/IMPS/{500 + i} | {i * 91 + 10}.00 | | {i * 40 + 5}.00 | "
                f"{9000 + i * 40}.00 | |" for i in range(1, 9)]
        res = self.check(rows)
        self.assertEqual(res["verdict"], "suspect")
        self.assertIn("cheque", res["checks"][0]["detail"])

    def test_a_clean_statement_and_a_balance_brought_forward_pass(self):
        clean = [f"| {i} | 0{i}/02/2020 | UPI/IMPS/{500 + i} | | {i * 10}.00 | | "
                 f"{5000 - sum(range(1, i + 1)) * 10}.00 Cr | |" for i in range(1, 9)]
        self.assertEqual(self.check(clean)["verdict"], "ok")
        carried = ["| 1 | 01/02/2020 | Brought Forward | | | 5000.00 | 5000.00 | |"] + [
            f"| {i} | 0{i}/02/2020 | UPI/IMPS/{500 + i} | | {i * 10}.00 | | "
            f"{5000 - sum(range(2, i + 1)) * 10}.00 | |" for i in range(2, 9)]
        self.assertEqual(self.check(carried)["verdict"], "ok")

    def test_rows_may_leave_out_the_last_empty_column_but_not_a_middle_one(self):
        head = "| Date | Particulars | Place | Remarks | For use |\n|---|---|---|---|---|\n"
        short = head + "\n".join(f"| 0{i}/02/2020 | UPI/{i} | Pune | note {i} |" for i in range(1, 7))
        self.assertEqual(gate.check_page(short, branch_kind="scan")["verdict"], "ok")
        shifted = short + "\n| 07/02/2020 | Pune | note 7 |"
        res = gate.check_page(shifted, branch_kind="scan")
        self.assertIn("different numbers of cells", res["checks"][0]["detail"])

    def test_a_cheque_number_that_is_a_number_is_fine(self):
        rows = [f"| {i} | 0{i}/02/2020 | CHEQUE | 00{i}4711 | {i * 10}.00 | | {5000 - i * 10}.00 | |"
                for i in range(1, 9)]
        self.assertEqual(self.check(rows)["verdict"], "ok")


class HtmlChunkingTests(unittest.TestCase):
    def table(self, rows, blank=False):
        sep = "\n\n" if blank else "\n"
        head = f"<thead><tr><th>Date</th><th>Text</th></tr></thead>{sep}"
        body = sep.join(f"<tr><td>0{i % 9 + 1}/02/2020</td><td>{' '.join(WORDS[:8])} {i}</td></tr>" for i in range(rows))
        return f"<table border=\"1\">{sep}{head}{body}{sep}</table>"

    def test_the_chunker_version_changed_with_this_behaviour(self):
        self.assertEqual(chunker.CHUNKER_VERSION, "v2")

    def test_a_small_html_table_with_blank_lines_is_one_chunk(self):
        md = "<!-- page 1 -->\n\nIntro text.\n\n" + self.table(4, blank=True) + "\n\nOutro.\n"
        chunks = chunker.markdown_to_nodes(md, 512, 64)
        holding = [c for c in chunks if "<table" in c["text"]]
        self.assertEqual(len(holding), 1)
        self.assertIn("</table>", holding[0]["text"])
        self.assertEqual(holding[0]["text"].count("<tr>"), 5)               # header row + 4 rows, none cut off

    def test_an_oversized_html_table_is_cut_by_rows_with_the_header_in_every_piece(self):
        md = "<!-- page 1 -->\n\n" + self.table(120) + "\n"
        chunks = chunker.markdown_to_nodes(md, 200, 20)
        tables = [c["text"] for c in chunks if "<table" in c["text"]]
        self.assertGreater(len(tables), 3)
        rows = 0
        for t in tables:
            self.assertTrue(t.lstrip().startswith("<table"))
            self.assertTrue(t.rstrip().endswith("</table>"))
            self.assertIn("<th>Date</th>", t)
            rows += t.count("<td>0")
        self.assertEqual(rows, 120)                                        # every data row, once

    def test_pipe_tables_are_chunked_as_before(self):
        md = "<!-- page 1 -->\n\n| A | B |\n|---|---|\n" + "\n".join(f"| {i} | {' '.join(WORDS)} |" for i in range(80)) + "\n"
        chunks = chunker.markdown_to_nodes(md, 200, 20)
        self.assertTrue(all(c["text"].startswith("| A | B |") for c in chunks))


if __name__ == "__main__":
    unittest.main()
