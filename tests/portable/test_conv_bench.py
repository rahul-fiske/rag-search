"""Conversion measurement (P1): tables, numbers, validators, metrics, gold sets, bench runs."""

from __future__ import annotations

import json
import os
import unittest
from decimal import Decimal

from tests.helpers import TempHome
from tests.portable.test_cli_api import run
from tests.portable.test_conversion import HAVE_PDF, write_text_pdf

from rag_search.core.conversion import bench, engines, metrics, pagemd, tables, trace, validators

PASSBOOK = """| Date | Particulars | Debit | Credit | Balance |
|---|---|---|---|---|
| 01-01-2020 | Opening | | | 1,000.00 |
| 02-01-2020 | Cash | | 500.00 | 1,500.00 |
| 03-01-2020 | ATM | 200.00 | | 1,300.00 |
| 04-01-2020 | NEFT | | 281.00 | 1,581.00 |
| 05-01-2020 | Chq | 81.00 | | 1,500.00 |
"""


def table(md: str) -> tables.Table:
    return tables.find_tables(md)[0]


class NumberTests(unittest.TestCase):
    def test_numbers(self):
        p = tables.parse_number
        self.assertEqual(p("1,234.50"), Decimal("1234.50"))
        self.assertEqual(p("1,23,456.78"), Decimal("123456.78"))
        self.assertEqual(p("₹ 2,500"), Decimal(2500))
        self.assertEqual(p("(1,234.50)"), Decimal("-1234.50"))
        self.assertEqual(p("500.00 Dr"), Decimal("-500.00"))
        self.assertEqual(p("500.00Cr"), Decimal("500.00"))
        self.assertEqual(p("100-"), Decimal(-100))
        self.assertEqual(p("-12.5"), Decimal("-12.5"))
        for bad in ("", "abc", "12,34", "1,2345", "12 apples", "2,81.00", None):
            self.assertIsNone(p(bad), bad)

    def test_norm_number_text(self):
        n = tables.norm_number_text
        self.assertEqual(n("(1,234.50)"), "-1234.5")
        self.assertEqual(n("-1234.50"), "-1234.5")
        self.assertEqual(n("1,000.00"), "1000")
        self.assertEqual(n("  Hello   World "), "hello world")

    def test_dates(self):
        self.assertTrue(tables.looks_like_date("01-01-2020"))
        self.assertTrue(tables.looks_like_date("5 Jan 2020"))
        self.assertFalse(tables.looks_like_date("1,000.00"))


class TableParseTests(unittest.TestCase):
    def test_pipe_table(self):
        t = table("Intro\n\n" + PASSBOOK + "\nAfter")
        self.assertEqual(t.kind, "pipe")
        self.assertEqual(t.header[4], "Balance")
        self.assertEqual(len(t.body), 5)
        self.assertEqual(t.body[1][3], "500.00")
        self.assertEqual(t.width, 5)

    def test_html_table_with_spans(self):
        md = ("<table><tr><th rowspan=2>Item</th><th colspan=2>Qty</th></tr>"
              "<tr><th>A</th><th>B</th></tr>"
              "<tr><td>x</td><td>1</td><td>2</td></tr></table>")
        t = table(md)
        self.assertEqual(t.kind, "html")
        self.assertEqual(t.n_header, 2)
        self.assertEqual(t.rows[0], ["Item", "Qty", ""])
        self.assertEqual(t.rows[2], ["x", "1", "2"])

    def test_find_tables_in_order_and_roundtrip(self):
        md = "a\n\n" + PASSBOOK + "\n<table><tr><td>1</td></tr></table>\n"
        ts = tables.find_tables(md)
        self.assertEqual([t.kind for t in ts], ["pipe", "html"])
        again = tables.find_tables(tables.to_markdown(ts[0]))[0]
        self.assertEqual(again.rows, ts[0].rows)

    def test_plain_text_drops_furniture(self):
        s = tables.plain_text("<!-- page 1 -->\n# Title\n\n| a | b |\n|---|---|\n| 1 | 2 |\n")
        self.assertEqual(s, "Title a b 1 2")


class RunningBalanceTests(unittest.TestCase):
    def test_clean_table_passes(self):
        r = validators.running_balance(table(PASSBOOK))
        self.assertTrue(r["applicable"])
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["checked"], 4)

    def test_damaged_amount_fails_one_row_with_hypothesis(self):
        r = validators.running_balance(table(PASSBOOK.replace("281.00", "2,81.00")))
        self.assertFalse(r["ok"])
        self.assertEqual(len(r["violations"]), 1)
        v = r["violations"][0]
        self.assertEqual((v["col"], v["role"], v["found"], v["expected"]), (3, "credit", "2,81.00", "281.00"))
        self.assertEqual(v["row"], 4)            # header = row 0

    def test_wrong_balance_points_at_the_balance_cell(self):
        r = validators.running_balance(table(PASSBOOK.replace("1,300.00", "1,350.00")))
        self.assertEqual(len(r["violations"]), 1)
        v = r["violations"][0]
        self.assertEqual((v["role"], v["col"], v["row"], v["expected"]), ("balance", 4, 3, "1300.00"))

    def test_newest_first_statement(self):
        rows = PASSBOOK.strip().split("\n")
        rev = "\n".join(rows[:2] + rows[2:][::-1])
        r = validators.running_balance(table(rev))
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["order"], "up")

    def test_not_applicable_without_balance_column_or_enough_rows(self):
        self.assertFalse(validators.running_balance(table("| a | b |\n|---|---|\n| 1 | 2 |\n"))["applicable"])
        short = "\n".join(PASSBOOK.strip().split("\n")[:4])
        self.assertFalse(validators.running_balance(table(short))["applicable"])

    def test_signed_amount_column(self):
        md = ("| Date | Amount | Balance |\n|---|---|---|\n| a | | 100 |\n| b | 50 | 150 |\n"
              "| c | -30 | 120 |\n| d | 10 | 130 |\n")
        r = validators.running_balance(table(md))
        self.assertTrue(r["applicable"] and r["ok"], r)

    def test_unrelated_layout_is_not_flagged(self):
        md = "| Name | Debit | Balance |\n|---|---|---|\n" + "".join(f"| n{i} | {i * 7} | {i * 13 % 11} |\n" for i in range(8))
        r = validators.running_balance(table(md))
        self.assertTrue(r["ok"] or not r["applicable"])


class TotalsTests(unittest.TestCase):
    MD = ("| Item | Qty | Amount |\n|---|---|---|\n| a | 2 | 10.00 |\n| b | 3 | 20.50 |\n"
          "| c | 1 | 5.25 |\n| Total | 6 | 35.75 |\n")

    def test_ok(self):
        r = validators.totals(table(self.MD))
        self.assertTrue(r["applicable"] and r["ok"], r)

    def test_wrong_total(self):
        r = validators.totals(table(self.MD.replace("35.75", "36.75")))
        self.assertEqual(len(r["violations"]), 1)
        v = r["violations"][0]
        self.assertEqual((v["col"], v["expected"], v["role"]), (2, "35.75", "total"))

    def test_subtotal_blocks(self):
        md = ("| Item | Amount |\n|---|---|\n| a | 1 |\n| b | 2 |\n| Subtotal | 3 |\n| c | 4 |\n| d | 5 |\n"
              "| Total | 9 |\n")
        r = validators.totals(table(md))
        self.assertTrue(r["ok"], r)


class RecheckTests(unittest.TestCase):
    def test_fix_is_recognised(self):
        t = table(PASSBOOK.replace("281.00", "2,81.00"))
        good = validators.recheck_with(t, 4, 3, "281.00")
        self.assertTrue(good["fixed"], good)
        bad = validators.recheck_with(t, 4, 3, "282.00")
        self.assertFalse(bad["fixed"], bad)


class MetricTests(unittest.TestCase):
    def test_levenshtein_and_cer(self):
        self.assertEqual(metrics.levenshtein("kitten", "sitting"), 3)
        self.assertEqual(metrics.levenshtein("", "abc"), 3)
        self.assertEqual(metrics.levenshtein(["a", "b"], ["a", "c"]), 1)
        self.assertEqual(metrics.cer("Hello world", "Hello world"), 0.0)
        self.assertAlmostEqual(metrics.cer("Hello wrld", "Hello world"), 1 / 11, places=3)
        self.assertEqual(metrics.cer("", "text"), 1.0)
        self.assertEqual(metrics.cer("", ""), 0.0)

    def test_cell_scores_exact_and_bag(self):
        damaged = PASSBOOK.replace("281.00", "2,81.00").replace("1,500.00 |\n| 03", "1,550.00 |\n| 03")
        sc = metrics.score_page(damaged, PASSBOOK)
        c = sc["cells"]
        self.assertEqual(c["total"], 9)
        self.assertEqual(c["exact"], 7)           # one wrong amount, one wrong balance
        self.assertEqual(c["bag"], 7)
        self.assertFalse(sc["balance"]["ok"] == sc["balance"]["applicable"])
        self.assertLess(sc["table_sim"], 1.0)
        self.assertEqual(metrics.score_page(PASSBOOK, PASSBOOK)["table_sim"], 1.0)

    def test_shifted_columns_count_in_bag_not_exact(self):
        shifted = PASSBOOK.replace("| Debit | Credit |", "| Credit | Debit |")
        sc = metrics.score_page(shifted, PASSBOOK)
        self.assertEqual(sc["cells"]["bag"], sc["cells"]["total"])

    def test_missing_row_does_not_misalign_the_rest(self):
        lines = PASSBOOK.strip().split("\n")
        missing = "\n".join(lines[:3] + lines[4:])        # the 02-01 row is gone
        c = metrics.score_page(missing, PASSBOOK)["cells"]
        self.assertEqual(c["exact"], c["total"] - 2)      # the credit and the balance of the lost row

    def test_truth_tables_override_and_query_hits(self):
        sc = metrics.score_page("| a | b |\n|---|---|\n| 1,000 | 2 |\n", "ignored",
                                queries=["1000", "not here"], truth_tables=[[["a", "b"], ["1000", "2"]]])
        self.assertEqual(sc["cells"], {"total": 2, "exact": 2, "bag": 2})
        self.assertEqual(sc["queries"], {"total": 2, "hit": 1})

    def test_aggregate(self):
        a = metrics.score_page(PASSBOOK, PASSBOOK)
        a["seconds"] = 1.0
        b = metrics.score_page("nothing", PASSBOOK)
        b["seconds"] = 3.0
        agg = metrics.aggregate([a, b])
        self.assertEqual(agg["pages"], 2)
        self.assertEqual(agg["cell_exact"], 0.5)
        self.assertEqual(agg["s_per_page"], 2.0)
        self.assertEqual(agg["balance_ok"], 1.0)         # only the page that had a table to check
        self.assertEqual(metrics.aggregate([])["pages"], 0)


class PageMarkdownTests(unittest.TestCase):
    def test_split_join_roundtrip(self):
        md = "<!-- page 1 -->\n\nA\n\n<!-- page 3 -->\n\nC\n"
        self.assertEqual(pagemd.split_pages(md), {1: "A", 3: "C"})
        self.assertEqual(pagemd.split_pages(pagemd.join_pages({1: "A", 2: "", 3: "C"})), {1: "A", 3: "C"})
        self.assertIn("<!-- page 2 -->", pagemd.join_pages({1: "A", 2: ""}, keep_empty=True))
        self.assertEqual(pagemd.split_pages("no markers"), {1: "no markers"})
        self.assertEqual(pagemd.split_pages(""), {})
        self.assertEqual(pagemd.split_pages("intro\n<!-- page 2 -->\nB"), {1: "intro", 2: "B"})


class ClassifyTests(unittest.TestCase):
    def test_classes(self):
        rec = trace.page_record
        self.assertEqual(bench.classify(rec(1, "digital", "", out={"tables": 1})), "digital-table")
        self.assertEqual(bench.classify(rec(1, "digital", "", out={"chars": 9})), "digital-text")
        self.assertEqual(bench.classify(rec(1, "raster", "", out={"tables": 2})), "scan-table")
        self.assertEqual(bench.classify(rec(1, "fallback", "")), "scan-text")
        self.assertEqual(bench.classify(rec(1, "raster", "", out={"script": "Devanagari"})), "devanagari-text")
        self.assertEqual(bench.classify(rec(1, "image", "")), "image")
        self.assertEqual(bench.classify(rec(1, "raster", "", outcome="no_text")), "empty")


class FakeEngineTests(unittest.TestCase):
    def test_resolve(self):
        self.assertEqual(engines.resolve("current").name, "current")
        self.assertEqual(engines.resolve("tests.helpers:FakeEngine").name, "fake")
        with self.assertRaises(engines.EngineError):
            engines.resolve("nonsense")
        with self.assertRaises(engines.EngineError):
            engines.resolve("tests.helpers:Missing")


@unittest.skipUnless(HAVE_PDF, "needs pypdfium2 and Pillow")
class BenchRunTests(TempHome):
    TRUTH = PASSBOOK

    def setUp(self):
        super().setUp()
        self.src = self.sdir / "bank" / "pass.pdf"
        self.src.parent.mkdir(parents=True)
        write_text_pdf(self.src, ["x" * 60, "y" * 60])
        self.register_tree()
        md = pagemd.join_pages({1: "Intro text of page one", 2: self.TRUTH})
        mdf = self.paths.markup / "bank" / "pass.md"
        mdf.parent.mkdir(parents=True)
        mdf.write_text(md, encoding="utf-8")
        pages = [trace.page_record(1, "digital", "text", out={"chars": 20}),
                 trace.page_record(2, "raster", "scan", out={"tables": 1, "chars": 90})]
        from rag_search.paths import sha256_file
        trace.write_trace(trace.trace_path_for(mdf), source="pass.pdf", src_sha=sha256_file(self.src),
                          pages=pages, summary=trace.summarize(pages))
        meta = self.paths.index / "bank" / "pass" / "index.meta.json"
        meta.parent.mkdir(parents=True)
        meta.write_text(json.dumps({"src_path": str(self.src)}))

    def fake_pages(self, pages: dict) -> None:
        f = self.tmp / "fake.json"
        f.write_text(json.dumps({"pass.pdf": pages}))
        os.environ["RAG_TEST_FAKE_PAGES"] = str(f)

    def init(self, **kw):
        return bench.gold_init(self.paths, "g1", **kw)

    def verify_all(self):
        g = bench.read_gold(self.paths, "g1")
        for e in g["pages"]:
            e["verified"] = True
            if e["page"] == 2:
                e["queries"] = ["NEFT 281.00"]
        bench.write_json_atomic(bench.gold_dir(self.paths, "g1") / "gold.json", g)

    def test_gold_init_picks_pages_by_class_and_prefills(self):
        res = self.init()
        self.assertEqual(res["added"], {"digital-text": 1, "scan-table": 1})
        g = bench.read_gold(self.paths, "g1")
        e = {x["page"]: x for x in g["pages"]}
        self.assertEqual(e[2]["class"], "scan-table")
        self.assertIn("Opening", e[2]["truth"])
        self.assertFalse(e[2]["verified"])
        self.assertEqual(e[2]["rel"], "bank/pass.pdf")
        self.assertTrue((bench.gold_dir(self.paths, "g1") / e[2]["image"]).read_bytes().startswith(b"\x89PNG"))
        again = self.init()                                   # nothing new the second time
        self.assertEqual(again["added_total"], 0)
        self.assertEqual(again["total"], 2)
        self.assertEqual([g["name"] for g in bench.list_gold(self.paths)], ["g1"])
        self.assertEqual(bench.list_gold(self.paths)[0]["verified"], 0)

    def test_run_needs_verified_pages(self):
        self.init()
        with self.assertRaises(bench.BenchError):
            bench.run_bench(self.paths, "g1", engine="tests.helpers:FakeEngine")

    def test_run_scores_stores_and_compares(self):
        self.init()
        self.verify_all()
        self.fake_pages({"1": "Intro text of page one", "2": self.TRUTH})
        a = bench.run_bench(self.paths, "g1", engine="tests.helpers:FakeEngine", name="good")
        self.assertEqual(a["summary"]["all"]["cell_exact"], 1.0)
        self.assertEqual(a["summary"]["all"]["cer"], 0.0)
        self.assertEqual(a["summary"]["all"]["query_hit"], 1.0)
        self.assertEqual(a["summary"]["all"]["balance_ok"], 1.0)
        self.assertEqual(a["summary"]["all"]["s_per_page"], 0.5)
        self.assertEqual(sorted(a["summary"]["by_class"]), ["digital-text", "scan-table"])
        self.fake_pages({"1": "Intro text of page one", "2": self.TRUTH.replace("281.00", "2,81.00")})
        b = bench.run_bench(self.paths, "g1", engine="tests.helpers:FakeEngine", name="worse")
        self.assertLess(b["summary"]["all"]["cell_exact"], 1.0)
        self.assertEqual(b["summary"]["all"]["balance_ok"], 0.0)
        runs = bench.list_runs(self.paths, "g1")
        self.assertEqual({r["name"] for r in runs}, {"good", "worse"})
        cmp = bench.compare(self.paths, "g1", a["id"], b["id"])
        self.assertIs(cmp["all"]["cell_exact"]["better"], False)
        self.assertEqual(cmp["worse"][0]["id"], "p002")
        self.assertIs(cmp["by_class"]["digital-text"]["cer"]["better"], None)

    def test_reader_failure_and_changed_source_are_reported_not_fatal(self):
        self.init()
        self.verify_all()
        f = self.tmp / "fake.json"
        f.write_text(json.dumps({"other.pdf": {}}))
        os.environ["RAG_TEST_FAKE_PAGES"] = str(f)
        r = bench.run_bench(self.paths, "g1", engine="tests.helpers:FakeEngine")
        self.assertTrue(all(row.get("error") for row in r["pages"]))
        self.assertEqual(r["summary"]["all"]["cell_exact"], 0.0)
        write_text_pdf(self.src, ["z" * 60])               # the source changes after the gold was made
        with self.assertRaises(bench.BenchError) as cm:
            bench.run_bench(self.paths, "g1", engine="tests.helpers:FakeEngine")
        self.assertIn("changed", str(cm.exception))

    def test_source_outside_docs_is_refused(self):
        self.init()
        self.verify_all()
        g = bench.read_gold(self.paths, "g1")
        outside = self.tmp / "elsewhere.pdf"
        write_text_pdf(outside, ["q" * 60])
        for e in g["pages"]:
            e["path"], e["rel"], e["src_sha256"] = str(outside), "", ""
        bench.write_json_atomic(bench.gold_dir(self.paths, "g1") / "gold.json", g)
        self.fake_pages({})
        with self.assertRaises(bench.BenchError):
            bench.run_bench(self.paths, "g1", engine="tests.helpers:FakeEngine")

    def test_names_are_validated(self):
        for bad in ("../x", "a/b", "", ".hidden", "x" * 60):
            with self.assertRaises(bench.BenchError):
                bench.gold_dir(self.paths, bad)


@unittest.skipUnless(HAVE_PDF, "needs pypdfium2 and Pillow")
class BenchCliApiTests(BenchRunTests):
    def test_cli_flow(self):
        rc, out, err = run("bench", "gold", "init", "g1")
        self.assertEqual(rc, 0, err)
        self.assertIn("added 2 page(s)", out)
        self.assertIn("scan-table", out)
        rc, out, _ = run("bench", "gold", "list")
        self.assertIn("g1: 2 page(s), 0 verified", out)
        self.assertIn("current", out)
        rc, out, _ = run("bench", "gold", "show", "g1")
        self.assertIn("p002", out)
        rc, out, err = run("bench", "run", "g1", "--engine", "tests.helpers:FakeEngine")
        self.assertNotEqual(rc, 0)                                    # nothing verified yet
        self.assertIn("verified", err)
        self.verify_all()
        self.fake_pages({"1": "Intro text of page one", "2": self.TRUTH})
        rc, out, err = run("bench", "run", "g1", "--engine", "tests.helpers:FakeEngine", "--name", "one")
        self.assertEqual(rc, 0, err)
        self.assertIn("numeric cells exact 100.0%", out)
        self.fake_pages({"1": "Intro text of page one", "2": self.TRUTH.replace("281.00", "2,81.00")})
        run("bench", "run", "g1", "--engine", "tests.helpers:FakeEngine", "--name", "two")
        rc, out, _ = run("bench", "list", "--json")
        runs = json.loads(out)
        self.assertEqual(len(runs), 2)
        ids = {r["name"]: r["id"] for r in runs}
        rc, out, _ = run("bench", "show", "g1", ids["two"])
        self.assertIn("weakest pages", out)
        rc, out, err = run("bench", "compare", "g1", ids["one"], ids["two"])
        self.assertEqual(rc, 0, err)
        self.assertIn("WORSE", out)
        self.assertNotEqual(run("bench", "show", "g1", "nope")[0], 0)
        self.assertNotEqual(run("bench", "gold", "show", "../x")[0], 0)

    def test_api_envelopes(self):
        from rag_search import api
        r = api.bench_gold_init(self.paths, "g2", per_class=1)
        self.assertTrue(r["ok"])
        self.assertFalse(api.bench_run(self.paths, "g2", engine="bogus")["ok"])
        self.assertFalse(api.bench_gold_show(self.paths, "missing")["ok"])
        self.assertTrue(api.bench_list(self.paths, "")["ok"])


@unittest.skipUnless(HAVE_PDF, "needs pypdfium2 and Pillow")
class BenchDashboardTests(BenchRunTests):
    def test_listing_and_compare_routes(self):
        from tests.portable.test_ui import Dash

        self.init()
        self.verify_all()
        self.fake_pages({"1": "Intro text of page one", "2": self.TRUTH})
        a = bench.run_bench(self.paths, "g1", engine="tests.helpers:FakeEngine", name="a")
        self.fake_pages({"1": "Intro", "2": self.TRUTH})
        b = bench.run_bench(self.paths, "g1", engine="tests.helpers:FakeEngine", name="b")
        d = Dash(self.paths, read_only=True)
        self.addCleanup(d.close)
        st, js, _, _ = d.req("GET", "/api/conversion/bench")
        self.assertEqual(st, 200)
        self.assertEqual([s["name"] for s in js["sets"]], ["g1"])
        self.assertEqual(len(js["runs"]), 2)
        st, js, _, _ = d.req("GET", f"/api/conversion/bench-compare?set=g1&a={a['id']}&b={b['id']}")
        self.assertEqual(st, 200)
        self.assertIs(js["result"]["all"]["cer"]["better"], False)
        st, js, _, _ = d.req("GET", f"/api/conversion/bench-run?set=g1&run={a['id']}")
        self.assertEqual(js["result"]["id"], a["id"])
        self.assertEqual(d.req("GET", "/api/conversion/bench-run?set=g1&run=..%2Fx")[0], 400)
        self.assertEqual(d.req("GET", "/api/conversion/bench")[0], 200)
        self.assertEqual(d.req("GET", "/api/conversion/bench", auth=False)[0], 401)


if __name__ == "__main__":
    unittest.main()
