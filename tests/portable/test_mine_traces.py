"""scripts/mine_traces.py (routing plan step R0a): the report it builds from stored traces, and that it only reads."""

from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import json
import unittest
from pathlib import Path

from tests.helpers import TempHome

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "mine_traces.py"
TABLE = "| Date | Amount | Balance |\n|---|---|---|\n| 1 | 10 | 10 |\n| 2 | 5 | 15 |\n"


def load():
    spec = importlib.util.spec_from_file_location("mine_traces", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class MineTracesTests(TempHome):
    def setUp(self):
        super().setUp()
        from rag_search.core.conversion import trace

        def page(n, branch, outcome="pass", failed=(), was="", gate_s=0.01, read_s=1.0, reader=None, tokens=0):
            p = trace.page_record(n, branch, "why", outcome=outcome, reader=reader or {"tool": "docling"},
                                  profile={"chars": 400 if branch == "digital" else 0, "ink": 0.05})
            p["time_s"] = {"read": read_s, "gate": gate_s}
            p["cache"] = "hit" if branch == "cached" else "miss"
            if was:
                p["was"] = was
            if failed:
                p["gate"] = {"verdict": "suspect", "checks": [{"name": c, "ok": False, "detail": f"{c} detail"} for c in failed]}
            if tokens:
                p["tokens"] = tokens
            return p

        vlm = {"tool": "vlm", "model": "qwen-4b"}
        docs = {
            ("bank", "2024/pass book"): [page(1, "digital"), page(2, "digital", "low", ["table_shape"]),
                                         page(3, "raster", "low", ["coverage", "running_balance"], reader=vlm, tokens=900, read_s=60),
                                         page(4, "cached", "pass", was="raster", read_s=0.0, reader=vlm)],
            ("notes", "memo"): [page(1, "digital", "low", ["table_shape"], gate_s=2.5), page(2, "image", reader=vlm, tokens=300)],
        }
        for (coll, doc), pages in docs.items():
            md = self.paths.markup / coll / (doc + ".md")
            md.parent.mkdir(parents=True, exist_ok=True)
            md.write_text("".join(f"<!-- page {p['page']} -->\n\n{TABLE}\ntext of page {p['page']}\n\n" for p in pages))
            trace.write_trace(trace.trace_path_for(md), source=doc + ".pdf", src_sha="x", pages=pages,
                              summary=trace.summarize(pages))

    def run_script(self, *args):
        out = self.tmp / "out"
        buf = io.StringIO()
        before = sorted(str(p) for p in self.paths.home.rglob("*"))
        with contextlib.redirect_stdout(buf):
            rc = load().main(["--home", str(self.paths.home), "--out", str(out), *args])
        self.assertEqual(rc, 0, buf.getvalue())
        self.assertEqual(sorted(str(p) for p in self.paths.home.rglob("*")), before)     # nothing written in the data folder
        return out, buf.getvalue()

    def test_counts_kinds_checks_times_and_readers(self):
        out, said = self.run_script("--time-gate", "5")
        self.assertIn("2 documents, 6 pages, 3 low pages", said)
        rep = json.loads((out / "report.json").read_text())
        self.assertEqual(rep["by_kind_outcome"]["digital"], {"pass": 1, "low": 2})
        self.assertEqual(rep["by_kind_outcome"]["raster"], {"low": 1, "pass": 1})            # the cached scan counts as a scan
        self.assertEqual(rep["checks_by_kind"]["table_shape"], {"digital": 2})
        self.assertEqual(rep["only_reason_low"], {"table_shape": {"digital": 2}})             # the scan failed two checks
        self.assertEqual(rep["times_by_kind"]["raster"]["read_read"]["total_s"], 60.0)
        self.assertEqual(rep["times_by_kind"]["raster"]["gate_cached"]["pages"], 1)
        self.assertEqual(rep["tokens"], {"vlm qwen-4b": 1200})
        self.assertEqual(rep["slowest_gate"][0]["doc"], "memo")
        self.assertIn("table_shape", rep["gate_timing"]["digital"]["checks_ms"])
        self.assertIn("whole gate", rep["gate_timing"]["raster"]["checks_ms"])
        text = (out / "report.md").read_text()
        for heading in ("Pages by kind and outcome", "Gate checks that failed", "Time by kind", "Readers",
                        "Slowest pages", "re-timed"):
            self.assertIn(heading, text)

    def test_low_pages_and_samples_tell_how_to_open_each_page(self):
        out, _ = self.run_script("--time-gate", "0", "--sample", "1")
        with (out / "low_pages.csv").open() as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 3)
        self.assertEqual({r["checks"] for r in rows}, {"table_shape", "coverage;running_balance"})
        samples = (out / "samples.md").read_text()
        self.assertIn("## table_shape (2 low pages)", samples)
        self.assertIn("### digital pages (2)", samples)
        self.assertEqual(samples.count("rag-search trace"), 3 * 2)       # one page per check and group, two commands each
        self.assertIn("'bank/2024/pass book'", samples)                  # a path with a space is quoted for the shell
        self.assertNotIn("re-timed", (out / "report.md").read_text())

    def test_one_collection_and_an_empty_data_folder(self):
        out, said = self.run_script("--collection", "notes", "--time-gate", "0")
        self.assertIn("1 documents, 2 pages", said)
        import shutil
        shutil.rmtree(self.paths.markup)
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(load().main(["--home", str(self.paths.home), "--out", str(self.tmp / "o2")]), 1)


try:
    import pypdfium2  # noqa: F401
    HAVE_PDFIUM = True
except ImportError:
    HAVE_PDFIUM = False


@unittest.skipUnless(HAVE_PDFIUM, "needs pypdfium2")
class SourceCheckTests(TempHome):
    """R0b: a digital page's text layer compared with the Markdown that was indexed."""

    def setUp(self):
        super().setUp()
        import pypdfium2 as pdfium
        from tests import corpus
        from rag_search.core.conversion import trace

        src = corpus.copy("pdf/text.pdf", self.source("docs") / "plan.pdf")
        pdf = pdfium.PdfDocument(str(src))
        layers = [pdf[i].get_textpage().get_text_range() for i in range(len(pdf))]
        pdf.close()
        half = " ".join(layers[1].split()[: len(layers[1].split()) // 2])
        md_pages = {1: layers[0], 2: half, 3: layers[2]}            # page 2 lost half its text
        pages = [trace.page_record(1, "digital", "why", outcome="low"),
                 trace.page_record(2, "digital", "why", outcome="low"),
                 trace.page_record(3, "digital", "why", outcome="pass")]
        pages[0]["gate"] = {"verdict": "suspect", "checks": [{"name": "table_shape", "ok": False}]}
        pages[1]["gate"] = {"verdict": "suspect", "checks": [{"name": "coverage", "ok": False}]}
        md = self.paths.markup / "docs" / "plan.md"
        md.parent.mkdir(parents=True)
        md.write_text("".join(f"<!-- page {n} -->\n\n{t}\n\n" for n, t in md_pages.items()))
        trace.write_trace(trace.trace_path_for(md), source="plan.pdf", src_sha="x", pages=pages,
                          summary=trace.summarize(pages))
        self.src = src

    def run_script(self, *args):
        out = self.tmp / "out"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(load().main(["--home", str(self.paths.home), "--out", str(out), "--time-gate", "0", *args]), 0)
        return out, buf.getvalue()

    def test_low_pages_are_judged_by_the_text_layer(self):
        before = self.src.read_bytes()
        out, said = self.run_script("--check-sources")
        self.assertIn("text layer check: 2 pages, 1 with lost text", said)
        with (out / "source_check.csv").open() as fh:
            rows = {int(r["page"]): r for r in csv.DictReader(fh)}
        self.assertEqual(rows[1]["verdict"], "intact, table shape only")      # the flag was about layout only
        self.assertEqual(rows[2]["verdict"], "lost text")
        self.assertLess(float(rows[2]["word_recall"]), 0.6)
        rep = json.loads((out / "report.json").read_text())
        self.assertEqual(rep["source_check"]["by_check"]["coverage"], {"lost text": 1})
        self.assertIn("Text layer against the Markdown", (out / "report.md").read_text())
        self.assertEqual(self.src.read_bytes(), before)                      # the source is only read

    def test_all_pages_and_a_missing_source(self):
        out, _ = self.run_script("--check-sources", "all")
        rep = json.loads((out / "report.json").read_text())
        self.assertEqual(rep["source_check"]["by_outcome"]["pass"], {"intact": 1})
        self.src.unlink()
        out, said = self.run_script("--check-sources")
        rep = json.loads((out / "report.json").read_text())
        self.assertEqual(rep["source_check"]["by_outcome"]["low"], {"no source": 2})

    def test_tokens_compare_a_layer_with_markdown(self):
        m = load()
        words, nums = m._tokens("The ﬁnal exam-\r\nple costs 1,234.50 in snake\\_case, inter\u00ad\r\npreted")
        self.assertEqual(set(words), {"the", "final", "example", "costs", "in", "snake", "case", "interpreted"})
        self.assertEqual(set(nums), {"1234.50"})
        self.assertEqual(m._verdict(0.99, 1.0, 50, True, ["coverage"]), "intact")
        self.assertEqual(m._verdict(0.99, 0.5, 50, True, []), "lost text")
        self.assertEqual(m._verdict(0.95, 1.0, 50, True, []), "uncertain")
        self.assertEqual(m._verdict(1.0, 1.0, 5, True, []), "uncertain")            # too short to judge
        self.assertEqual(m._verdict(1.0, 1.0, 50, False, []), "uncertain")          # a garbled layer proves nothing

    def test_running_headers_footers_and_page_numbers_are_not_lost_text(self):
        m = load()
        words = "ports fabric login zoning frames credits switches links buffers timers".split()
        layers = [f"ACME Spec Rev 1.{i}\n" + "\n".join(f"{words[(i + k) % 10]} section {i}.{k} explains {words[k]}"
                                                     for k in range(5)) + f"\nPage {i} of 9\n" for i in range(1, 10)]
        boiler = m._boilerplate(layers)
        clean, dropped = m._clean_layer(layers[3], boiler)
        self.assertEqual(dropped, 2)
        self.assertEqual(len(clean.strip().splitlines()), 5)                 # the body stays
        self.assertEqual(m._boilerplate(layers[:3]), set())                 # too few pages to tell
        self.assertTrue(m._page_number_line(" 12 ") and m._page_number_line("Page 3 of 10"))
        parts = [("Acme Confidential", f"body {i}", f"Revision 4.86 June 16, 2024 page {i}") for i in range(1, 5)]
        self.assertEqual(m._repeated_bands(parts), {"acme confidential", "revision #.# june #, # page #"})
        self.assertEqual(m._repeated_bands(parts[:2]), set())
        page = "Acme Confidential\nThe port logs in.\nRevision 4.86 June 16, 2024 page 3"
        out = m._without_bands("Acme Confidential", page, "Revision 4.86 June 16, 2024 page 3", m._repeated_bands(parts))
        self.assertEqual(out.split(), "The port logs in.".split())
        self.assertFalse(m._page_number_line("Table 12"))
