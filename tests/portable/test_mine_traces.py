"""scripts/mine_traces.py (routing plan step R0a): the report it builds from stored traces, and that it only reads."""

from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import json
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
