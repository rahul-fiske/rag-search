"""Multi-line table headers and column-layout checks (validators.stack_header / layout_problem, gate)."""

from __future__ import annotations

import unittest

from rag_search.core.conversion import gate, tables, validators

HEAD = ("| ओल्ड क्र. | तारीख | तपशील | चेक | रकम काटली | रकम देवली | शिलुक | ग्राहक |\n"
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
        "| LINE | DATE | PARTICULARS | CHEQ. NO. | AMOUNT | AMOUNT | BALANCE | FOR USE OF |\n"
        "| NO. | | | | WITHDRAWN | DEPOSITED | | CUSTOMER |\n")
GOOD = ("| 1 | 29/04/17 | BY TRF | | | 1000.00 | 1000.00 Cr | |\n"
        "| 2 | 08/03/18 | BY TRF | | | 3000.00 | 4000.00 Cr | |\n"
        "| 3 | 31/03/18 | BY INTT | | | 71.00 | 4071.00 Cr | |\n"
        "| 4 | 02/03/19 | BY TRF | | | 1000.00 | 5071.00 Cr | |\n")
SHIFTED = ("| 1 | 29/04/17 | BY TRF | FRM X | 1000.00 | 1000.00 Cr | | |\n"
           "| 2 | 08/03/18 | BY TRF | FRM X | 3000.00 | 4000.00 Cr | | |\n"
           "| 3 | 31/03/18 | BY INTT | | 71.00 | 4071.00 Cr | | |\n"
           "| 4 | 02/03/19 | BY TRF | FRM Y | 1000.00 | 5071.00 Cr | | |\n")


class StackedHeaderTests(unittest.TestCase):
    def test_title_lines_are_stacked(self):
        t = tables.find_tables(HEAD + GOOD)[0]
        head, k = validators.stack_header(t)
        self.assertEqual(k, 3)
        self.assertIn("AMOUNT WITHDRAWN", head[4])
        self.assertIn("BALANCE", head[6])

    def test_running_balance_now_applies_and_catches_an_error(self):
        bad = GOOD.replace("4071.00", "4701.00")
        t = tables.find_tables(HEAD + bad)[0]
        res = validators.running_balance(t)
        self.assertTrue(res["applicable"])
        self.assertFalse(res["ok"])
        t = tables.find_tables(HEAD + GOOD)[0]
        res = validators.running_balance(t)
        self.assertTrue(res["applicable"] and res["ok"])


class LayoutTests(unittest.TestCase):
    def test_balance_column_mostly_empty(self):
        t = tables.find_tables(HEAD + SHIFTED)[0]
        self.assertIn("balance column", validators.layout_problem(t))
        g = gate.check_page(HEAD + SHIFTED, branch_kind="scan")
        self.assertEqual(g["verdict"], "suspect")
        self.assertIn("table_shape", gate.failed(g))

    def test_good_table_has_no_problem(self):
        t = tables.find_tables(HEAD + GOOD)[0]
        self.assertEqual(validators.layout_problem(t), "")
        self.assertEqual(gate.check_page(HEAD + GOOD, branch_kind="scan")["verdict"], "ok")

    def test_repeated_rows(self):
        rows = ("| 1 | 31/03/23 | BY INTT | | | 1533.00 | 25789.00 Cr | |\n"
                "| 2 | 31/03/24 | BY INTT | | | 1831.00 | 27620.00 Cr | |\n"
                "| 3 | 31/03/25 | BY INTT | | | 2079.00 | 29699.00 Cr | |\n"
                "| 4 | 31/03/25 | BY INTT | | | 2079.00 | 29699.00 Cr | |\n")
        t = tables.find_tables(HEAD + rows)[0]
        self.assertIn("repeat", validators.layout_problem(t))


if __name__ == "__main__":
    unittest.main()
