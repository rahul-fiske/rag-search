"""Conversion tracking (phase P0): page records, run totals, trace files, API, CLI, dashboard routes."""

from __future__ import annotations

import json
import os
import time
import unittest
from pathlib import Path
from unittest import mock

from tests import corpus
from tests.helpers import FakeEmbedder, TempHome
from tests.portable.test_cli_api import run
from tests.portable.test_ui import Dash

from rag_search import api, jobs
from rag_search.core import indexer
from rag_search.core.conversion import (costs, estimate, profiler, records, router, runview,
                                        trace)

try:
    import pypdfium2  # noqa: F401
    from PIL import Image  # noqa: F401
    import PIL.JpegImagePlugin  # noqa: F401  (the PDF writer needs it)
    HAVE_PDF = True
except ImportError:                                    # pragma: no cover
    HAVE_PDF = False

TEXT = "Page text that is comfortably longer than the forty character limit for a usable layer."


def write_text_pdf(path: Path, pages: list[str]) -> None:
    """A minimal PDF with one line of Helvetica text per page (no libraries needed)."""
    n = len(pages)
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(n))
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", f"<< /Type /Pages /Kids [{kids}] /Count {n} >>",
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    for i, t in enumerate(pages):
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {5 + 2 * i} 0 R "
                    "/Resources << /Font << /F1 3 0 R >> >> >>")
        stream = f"BT /F1 12 Tf 72 700 Td ({t}) Tj ET"
        objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
    out, offs = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += f"{i} 0 obj\n{o}\nendobj\n".encode()
    x = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offs).encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{x}\n%%EOF\n".encode()
    path.write_bytes(out)


def write_mixed_pdf(path: Path) -> None:
    """Two text pages followed by one scanned (full-page picture) page (the corpus file)."""
    corpus.copy("pdf/mixed.pdf", path)


def fake_convert(src, md_path, ocr=None):
    """Stands in for docling: Markdown with page markers + the per-page facts docling reports."""
    if Path(src).suffix.lower() in (".md", ".txt"):                   # docling copies these: no page facts
        md_path.write_text(Path(src).read_text(), encoding="utf-8")
        return {}
    n = profiler.profile_file(Path(src)).get("page_count") or 1
    md = "\n\n".join(f"<!-- page {i} -->\n\n{TEXT} {i}" for i in range(1, n + 1))
    md_path.write_text(md, encoding="utf-8")
    stats = [{"page": i, "chars": 80, "script": "Latin", "tables": 0, "pictures": 0, "big_pictures": 0,
              "confidence": {"parse": 0.9, "layout": 0.8, "table": None, "ocr": 0.7, "mean": 0.8,
                             "low": 0.7, "grade": "good" if i != 3 else "poor"}} for i in range(1, n + 1)]
    return {"page_stats": stats, "pages": n, "ocr": bool(ocr)}


@unittest.skipUnless(HAVE_PDF, "needs pypdfium2 and Pillow")
class ConversionBase(TempHome):
    def setUp(self):
        super().setUp()
        self.use_fake_backends()
        os.environ["RAG_SEARCH_ROUTING"] = "document"      # these tests are about whole-document conversion
        patcher = mock.patch("rag_search.core.docling_convert.convert_file", fake_convert)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.events: list[dict] = []

    def add_pdf(self, rel: str = "reports/mixed.pdf") -> Path:
        p = self.sdir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        write_mixed_pdf(p)
        return p

    def run_index(self, **kw):
        srcs = indexer.scan_sources(self.sdir, indexer.exclude_dirs(self.paths))
        return indexer.run_index(self.paths, srcs, self.roots(), jobs=1,
                                 embedder=FakeEmbedder(), progress=self.events.append, **kw)

    def trace_file(self, rel="reports/mixed") -> Path:
        return self.paths.markup / (rel + ".trace.json")

    def meta(self, rel="reports/mixed") -> dict:
        return json.loads((self.paths.index / rel / "index.meta.json").read_text())


class TraceUnitTests(unittest.TestCase):
    def test_strip_roundtrip_and_unknown_letters(self):
        branches = ["digital"] * 3 + ["raster"] * 2 + ["digital"]
        s = trace.make_strip(branches)
        self.assertEqual(s, "d3r2d1")
        self.assertEqual(trace.parse_strip(s), [("digital", 3), ("raster", 2), ("digital", 1)])
        self.assertEqual(trace.parse_strip("z4"), [("unknown", 4)])
        self.assertEqual(trace.make_strip([]), "")

    def test_summarize_counts_pages_scripts_and_grades(self):
        pages = [
            trace.page_record(1, "digital", "text", out={"chars": 9, "script": "Latin", "tables": 1},
                              confidence={"grade": "Good"}),
            trace.page_record(2, "raster", "scan", outcome="low", out={"script": "Latin"},
                              confidence={"grade": "poor"}),
            trace.page_record(3, "raster", "scan", outcome="no_text"),
        ]
        s = trace.summarize(pages, time_s={"convert": 1.234}, cost={"cpu_s": 2.0}, trace="c/d.trace.json")
        self.assertEqual((s["pages"], s["branches"], s["strip"]), (3, {"digital": 1, "raster": 2}, "d1r2"))
        self.assertEqual(s["outcomes"], {"pass": 1, "low": 1, "no_text": 1})
        self.assertEqual((s["low_pages"], s["poor_pages"], s["tables"]), ([2], [2], 1))
        self.assertEqual(s["scripts"], {"Latin": 2})
        self.assertEqual(s["time_s"]["convert"], 1.23)
        self.assertEqual(s["trace"], "c/d.trace.json")

    def test_run_totals_add_and_snapshot(self):
        tot = trace.RunTotals({"pdf": 2})
        s1 = trace.summarize([trace.page_record(1, "digital", "x")], time_s={"convert": 1},
                             cost={"cpu_s": 1.5, "peak_mb": 300})
        s2 = trace.summarize([trace.page_record(1, "raster", "x", outcome="low")], cost={"cpu_s": 0.5, "peak_mb": 500})
        tot.add(s1), tot.add(s2), tot.add(None)
        snap = tot.snapshot()
        self.assertEqual((snap["docs"], snap["pages"], snap["no_record"], snap["low_docs"]), (2, 2, 1, 1))
        self.assertEqual(snap["branches"], {"digital": 1, "raster": 1})
        self.assertEqual(snap["cost"], {"cpu_s": 2.0, "peak_mb": 500})
        agg = trace.aggregate([s1, s2])
        self.assertEqual((agg["documents"], agg["pages"]), (2, 2))
        self.assertNotIn("pages_per_min", agg)

    def test_trace_path_for(self):
        self.assertEqual(trace.trace_path_for(Path("/m/c/a.md")), Path("/m/c/a.trace.json"))

    def test_router_decisions_have_reasons(self):
        self.assertEqual(router.decide("text")[0], "copy")
        self.assertEqual(router.decide("office")[0], "office")
        self.assertEqual(router.decide("image")[0], "image")
        self.assertEqual(router.decide("pdf", None)[0], "unknown")
        self.assertEqual(router.decide("pdf", {"chars": 500, "image_cover": 0.0})[0], "digital")
        self.assertEqual(router.decide("pdf", {"chars": 0, "image_cover": 1.0})[0], "raster")
        self.assertEqual(router.decide("pdf", {"chars": 500, "text_ok": False})[0], "raster")
        b, why = router.decide("pdf", {"chars": 500, "image_cover": 1.0, "hidden_ocr_layer": True})
        self.assertEqual(b, "digital")
        self.assertIn("hidden OCR", why)

    def test_meter_reports_cpu_and_peak(self):
        with costs.Meter() as m:
            sum(i * i for i in range(200_000))
        r = m.result()
        self.assertGreaterEqual(r["cpu_s"], 0)
        self.assertGreater(costs.peak_rss_mb(), 0)

    def test_layout_of_light_modules(self):
        import subprocess
        import sys
        from tests.helpers import SUBPROC_PYTHONPATH
        import os
        for mod in ("trace", "costs", "router", "profiler", "runview", "estimate", "records",
                    "pageimage", "tables", "validators", "metrics", "pagemd", "bench", "engines",
                    "pagecache", "gate", "routed", "vlm", "vlm_worker", "repair", "reconcile"):
            code = (f"import sys; sys.path[:0] = {SUBPROC_PYTHONPATH.split(os.pathsep)!r}; "
                    f"import rag_search.core.conversion.{mod}; "
                    "print(','.join(m for m in ('numpy', 'torch', 'docling') if m in sys.modules))")
            out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
            self.assertEqual(out.stdout.strip(), "", mod)


@unittest.skipUnless(HAVE_PDF, "needs pypdfium2 and Pillow")
class ProfilerTests(TempHome):
    def test_records_merge_profile_and_docling_facts(self):
        p = self.sdir / "m.pdf"
        write_mixed_pdf(p)
        prof = profiler.profile_file(p)
        stats = fake_convert(p, self.tmp / "o.md")["page_stats"]
        pages = records.build_pages("pdf", prof, stats, reader={"tool": "docling"})
        self.assertEqual([r["branch"] for r in pages], ["digital", "digital", "raster"])
        self.assertEqual(pages[2]["docling"]["grade"], "poor")
        self.assertGreater(pages[0]["profile"]["chars"], 40)
        # no docling facts (Markdown reused without a record): still one record per profiled page
        again = records.build_pages("pdf", prof, None, reader={"tool": "docling"})
        self.assertEqual(len(again), 3)
        # unprofilable PDF: one "unknown" page
        lone = records.build_pages("pdf", {"kind": "pdf", "pages": [], "error": "boom"}, None, reader={})
        self.assertEqual((lone[0]["branch"], lone[0]["why"]), ("unknown", "boom"))


class PrepareAndRunTests(ConversionBase):
    def test_run_records_branches_and_costs(self):
        self.add_pdf()
        self.write_doc("notes/todo.md", "# Todo\n\nBuy bread and write the report this week, please. " * 3)
        summary = self.run_index()
        self.assertEqual(summary["indexed"], 2)
        conv = summary["conversion"]
        self.assertEqual(conv["docs"], 2)
        self.assertEqual(conv["pages"], 4)                               # 3 pdf pages + 1 text document
        self.assertEqual(conv["branches"], {"digital": 2, "raster": 1, "copy": 1})
        self.assertEqual(conv["outcomes"], {"pass": 4})
        self.assertIn("cpu_s", conv["cost"])
        self.assertEqual(conv["docling_grades"], {"good": 2, "poor": 1})

    def test_meta_trace_and_doc_events(self):
        self.add_pdf()
        self.run_index()
        meta = self.meta()["conversion"]
        self.assertEqual((meta["pages"], meta["strip"]), (3, "d2r1"))
        self.assertEqual(meta["trace"], "reports/mixed.trace.json")
        self.assertEqual(meta["poor_pages"], [3])
        for key in ("profile", "convert", "chunk", "embed"):
            self.assertIn(key, meta["time_s"])
        tr = trace.read_trace(self.trace_file())
        self.assertEqual(tr["source"].endswith("mixed.pdf"), True)
        self.assertEqual([p["branch"] for p in tr["pages"]], ["digital", "digital", "raster"])
        self.assertEqual(tr["pages"][2]["why"][:14], "no usable text")
        docs = [e["doc"] for e in self.events if "doc" in e and e["doc"]["status"] == "indexed"]
        self.assertEqual(docs[0]["conversion"]["strip"], "d2r1")
        # live progress carries the totals once documents have finished
        phases = [e for e in self.events if e.get("phase") == "embed" and "conversion" in e]
        self.assertTrue(phases)
        self.assertEqual(phases[-1]["conversion"]["pages"], 3)

    def test_unchanged_document_keeps_its_trace(self):
        self.add_pdf()
        self.run_index()
        before = self.trace_file().read_text()
        s = self.run_index()
        self.assertEqual(s["skipped_fresh"], 1)
        self.assertEqual(self.trace_file().read_text(), before)
        self.assertEqual(s["conversion"]["pages"], 0)

    def test_rebuild_with_current_markdown_reuses_trace_without_profiling(self):
        self.add_pdf()
        self.run_index()
        with mock.patch.object(profiler, "profile_file", side_effect=AssertionError("profiled again")):
            s = self.run_index(rebuild=True)
        self.assertEqual(s["indexed"], 1)
        self.assertEqual(self.meta()["conversion"]["pages"], 3)

    def test_trace_is_removed_with_its_document(self):
        self.add_pdf()
        self.run_index()
        self.assertTrue(self.trace_file().exists())
        (self.sdir / "reports/mixed.pdf").unlink()
        srcs = indexer.scan_sources(self.sdir, indexer.exclude_dirs(self.paths))
        indexer.run_index(self.paths, srcs, self.roots(), jobs=1, embedder=FakeEmbedder(),
                          prune=["reports"])
        self.assertFalse(self.trace_file().exists())

    def test_wipe_removes_old_traces(self):
        self.add_pdf()
        self.run_index()
        self.run_index(wipe=True)
        self.assertTrue(self.trace_file().exists())          # rewritten by the rebuild
        self.assertEqual(self.meta()["conversion"]["pages"], 3)

    def test_failed_conversion_still_counts_its_pages(self):
        self.add_pdf()

        def boom(src, md_path, ocr=None):
            raise RuntimeError("docling blew up")
        with mock.patch("rag_search.core.docling_convert.convert_file", boom):
            s = self.run_index()
        self.assertEqual(s["error_count"] if "error_count" in s else len(s["errors"]), 1)
        self.assertFalse(self.trace_file().exists())

    def test_documents_filter_by_branch_and_outcome(self):
        self.add_pdf()
        self.write_doc("notes/todo.md", "# Todo\n\nBuy bread and write the report this week, please. " * 3)
        log = self.paths.jobs / "j1.events.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        self.run_index()
        with open(log, "w", encoding="utf-8") as fh:
            for e in self.events:
                if "doc" in e:
                    fh.write(json.dumps({"ts": 1, "event": "doc", **e["doc"]}) + "\n")
        out = jobs.documents(self.paths, "j1")
        self.assertEqual(out["by_branch"], {"digital": 1, "raster": 1, "copy": 1})
        self.assertEqual(out["by_outcome"], {"pass": 2})
        self.assertEqual([i["source"] for i in jobs.documents(self.paths, "j1", branch="raster")["items"]],
                         ["mixed.pdf"])
        self.assertEqual(jobs.documents(self.paths, "j1", branch="copy")["matched"], 1)
        self.assertEqual(jobs.documents(self.paths, "j1", outcome="low")["matched"], 0)
        self.assertEqual(jobs.documents(self.paths, "j1", outcome="pass", branch="copy")["matched"], 1)

    def test_stage_events_carry_the_process_id(self):
        self.add_pdf()
        log = self.paths.jobs / "j2.events.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        srcs = indexer.scan_sources(self.sdir, indexer.exclude_dirs(self.paths))
        indexer.run_index(self.paths, srcs, self.roots(), jobs=1, embedder=FakeEmbedder(),
                          stage_log=log)
        evs = [json.loads(line) for line in log.read_text().splitlines()]
        self.assertTrue(all("pid" in e for e in evs if e["event"] == "stage"))
        view = runview.lanes_for(self.paths, "j2", running=False)
        self.assertEqual([(lane["kind"], lane["docs"]) for lane in view], [("cpu", 1)])
        # incremental reading: a second call with nothing new gives the same answer
        self.assertEqual(runview.lanes_for(self.paths, "j2", running=False), view)


class RunViewTests(TempHome):
    def test_open_stage_shows_a_working_lane(self):
        log = self.paths.jobs / "j3.events.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        now = time.time()
        rows = [{"ts": now - 20, "event": "stage", "file": "a/x.pdf", "stage": "convert", "status": "start", "pid": 11},
                {"ts": now - 10, "event": "stage", "file": "a/x.pdf", "stage": "convert", "status": "done", "pid": 11},
                {"ts": now - 9, "event": "stage", "file": "a/y.pdf", "stage": "convert", "status": "start", "pid": 11},
                {"ts": now - 8, "event": "stage", "file": "a/z.pdf", "stage": "convert", "status": "start", "pid": 12}]
        log.write_text("".join(json.dumps(r) + "\n" for r in rows))
        lanes = runview.lanes_for(self.paths, "j3", running=True, now=now)
        self.assertEqual([lane["name"] for lane in lanes], ["docling worker 1", "docling worker 2"])
        self.assertEqual(lanes[0]["state"], "working")
        self.assertEqual(lanes[0]["file"], "a/y.pdf")
        self.assertEqual(lanes[0]["docs"], 1)
        self.assertGreater(lanes[0]["busy_pct"], 50)

    def log(self, name, rows):
        f = self.paths.jobs / f"{name}.events.jsonl"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return f

    def test_a_process_is_a_lane_whatever_phase_it_works_in(self):
        t = time.time() - 100
        w = lambda ts, pid, phase, file, status, **kw: {"ts": t + ts, "event": "work", "pid": pid, "phase": phase,  # noqa: E731
                                                       "file": file, "status": status, **kw}
        self.log("g1", [
            {"ts": t, "event": "phase", "pid": 1, "phase": "convert", "status": "start", "total": 3, "workers": 2},
            w(1, 2, "convert", "a/x.md", "start"),
            {"ts": t + 2, "event": "stage", "pid": 2, "file": "a/x.md", "stage": "convert", "id": "3", "status": "start"},
            w(1, 3, "convert", "a/bad.md", "start"),
            w(3, 3, "convert", "a/bad.md", "done", outcome="error"),        # a failure closes the lane's work
            w(4, 2, "convert", "a/x.md", "done", outcome="prepared"),
            {"ts": t + 5, "event": "phase", "pid": 1, "phase": "convert", "status": "done", "total": 3},
            {"ts": t + 6, "event": "phase", "pid": 1, "phase": "embed", "status": "start", "total": 1, "chunks": 8, "model": "m"},
            w(6, 1, "embed", "a/x.md", "start", chunks=8),
        ])
        st = runview._read_new(self.paths.jobs / "g1.events.jsonl")
        lanes = runview.lanes_for(self.paths, "g1", True, now=t + 10, state=st)
        by = {x["name"]: x for x in lanes}
        self.assertEqual(sorted(by), ["main process", "worker 1", "worker 2"])
        self.assertEqual((by["main process"]["state"], by["main process"]["phase"], by["main process"]["kind"]),
                         ("working", "embed", "gpu"))
        self.assertEqual(by["main process"]["file"], "a/x.md")              # the same spelling as the pool's
        self.assertEqual((by["worker 1"]["state"], by["worker 1"]["docs_by_phase"]), ("idle", {"convert": 1}))
        self.assertEqual((by["worker 2"]["state"], by["worker 2"]["file"]), ("idle", ""))   # not stuck on the failed file
        ph = {x["phase"]: x for x in runview.phases_view(st, {"progress": {"phase": "embed"}}, True)}
        self.assertEqual((ph["convert"]["status"], ph["convert"]["done"], ph["convert"]["outcomes"]),
                         ("done", 2, {"error": 1, "prepared": 1}))
        self.assertEqual((ph["embed"]["status"], ph["embed"]["chunks"], ph["embed"]["model"]), ("running", 8, "m"))
        self.assertEqual((ph["merge"]["status"], ph["publish"]["status"]), ("pending", "pending"))
        now = runview.now_view(st, lanes, {"progress": {"phase": "embed"}}, True)
        self.assertEqual((now["phase"], now["file"]), ("embed", "a/x.md"))

    def test_merge_work_and_the_publish_phase_are_reported_like_the_others(self):
        t = time.time() - 50
        self.log("g2", [
            {"ts": t, "event": "phase", "pid": 1, "phase": "merge", "status": "start"},
            {"ts": t + 1, "event": "work", "pid": 1, "phase": "merge", "file": "hr", "status": "start"},
            {"ts": t + 2, "event": "work", "pid": 1, "phase": "merge", "file": "hr", "status": "done", "outcome": "merged"},
            {"ts": t + 3, "event": "phase", "pid": 1, "phase": "merge", "status": "done", "total": 1}])
        st = runview._read_new(self.paths.jobs / "g2.events.jsonl")
        rec = {"progress": {"phase": "publish", "since": t + 4}}
        ph = {x["phase"]: x for x in runview.phases_view(st, rec, True)}
        self.assertEqual((ph["merge"]["status"], ph["merge"]["done"]), ("done", 1))
        self.assertEqual(ph["publish"]["status"], "running")
        rec = {"progress": {"phase": "done"}, "publish": {"generation": 4, "changed": True, "documents": 9},
               "publish_s": 1.5, "search_reload": {"ok": True}}
        pub = runview.phases_view(st, rec, False)[-1]
        self.assertEqual((pub["status"], pub["generation"], pub["seconds"]), ("done", 4, 1.5))
        rec["publish"] = {"error": "disk full"}
        self.assertEqual(runview.phases_view(st, rec, False)[-1]["status"], "failed")

    def test_a_run_writes_phase_and_work_events_and_all_name_the_file_alike(self):
        self.write_doc("reports/a.md", "# T\n\n<!-- page 1 -->\n\nquarterly figures\n")
        from rag_search.core.worker import EventWriter
        f = self.paths.jobs / "g3.events.jsonl"
        f.parent.mkdir(parents=True, exist_ok=True)
        w = EventWriter(f)
        srcs = indexer.scan_sources(self.sdir, indexer.exclude_dirs(self.paths))
        indexer.run_index(self.paths, srcs, self.roots(), jobs=1, embedder=FakeEmbedder(), stage_log=f,
                          progress=w.progress)
        w.close()
        evs = [json.loads(x) for x in f.read_text().splitlines()]
        phases = [(e["phase"], e["status"]) for e in evs if e["event"] == "phase"]
        self.assertEqual(phases, [("convert", "start"), ("convert", "done"), ("embed", "start"), ("embed", "done"),
                                  ("merge", "start"), ("merge", "done")])
        work = [(e["phase"], e["status"], e["file"]) for e in evs if e["event"] == "work"]
        files = {x[2] for x in work if x[0] in ("convert", "embed")}
        self.assertEqual(len(files), 1)                                    # one spelling for one document
        self.assertIn(("merge", "start", "reports"), work)
        self.assertEqual({e["file"] for e in evs if e["event"] == "stage"}, files)
        st = runview._read_new(f)
        lanes = runview.lanes_for(self.paths, "g3", False, state=st)
        self.assertEqual([x["name"] for x in lanes], ["main process"])      # jobs=1: one process does every phase
        self.assertEqual(lanes[0]["docs_by_phase"], {"convert": 1, "embed": 1, "merge": 1})

    def test_a_failing_conversion_closes_its_work(self):
        from unittest import mock
        f = self.paths.jobs / "g4.events.jsonl"
        f.parent.mkdir(parents=True, exist_ok=True)
        self.write_doc("c/a.md", "# T\n\ntext\n")
        srcs = indexer.scan_sources(self.sdir, indexer.exclude_dirs(self.paths))
        with mock.patch.object(indexer, "convert_source", side_effect=RuntimeError("boom")):
            indexer.run_index(self.paths, srcs, self.roots(), jobs=1, embedder=FakeEmbedder(), stage_log=f)
        evs = [json.loads(x) for x in f.read_text().splitlines() if '"work"' in x]
        self.assertEqual([(e["status"], e.get("outcome")) for e in evs], [("start", None), ("done", "error")])

    def test_run_view_without_jobs(self):
        self.assertEqual(runview.run_view(self.paths)["job"], None)
        self.assertEqual(runview.run_view(self.paths, "../etc")["job"], None)


@unittest.skipUnless(HAVE_PDF, "needs pypdfium2 and Pillow")
class ApiTests(ConversionBase):
    def setUp(self):
        super().setUp()
        self.add_pdf()
        self.run_index()

    def test_trace_summary_and_page_detail(self):
        r = api.conversion_trace(self.paths, "reports", "mixed")
        self.assertTrue(r["ok"], r)
        res = r["result"]
        self.assertEqual(res["summary"]["strip"], "d2r1")
        self.assertEqual([p["branch"] for p in res["pages"]], ["digital", "digital", "raster"])
        self.assertEqual(res["pages"][2]["grade"], "poor")
        one = api.conversion_trace(self.paths, "reports", "mixed", page=3)["result"]["pages"]
        self.assertEqual(len(one), 1)
        self.assertEqual(one[0]["profile"]["image_cover"], 1.0)
        self.assertFalse(api.conversion_trace(self.paths, "reports", "mixed", page=9)["ok"])

    def test_trace_is_confined_to_the_workspace(self):
        for doc in ("../../x", "/etc/passwd", "", "a/../../b", "."):
            self.assertFalse(api.conversion_trace(self.paths, "reports", doc)["ok"], doc)
        self.assertFalse(api.conversion_trace(self.paths, "../reports", "mixed")["ok"])

    def test_trace_falls_back_to_the_summary_in_the_meta(self):
        self.trace_file().unlink()
        r = api.conversion_trace(self.paths, "reports", "mixed")
        self.assertTrue(r["ok"])
        self.assertEqual(r["result"]["pages"], [])
        self.assertIn("only the summary", r["result"]["note"])

    def test_trace_missing_everywhere_explains_itself(self):
        meta = self.paths.index / "reports/mixed/index.meta.json"
        data = json.loads(meta.read_text())
        data.pop("conversion")
        meta.write_text(json.dumps(data))
        self.trace_file().unlink()
        r = api.conversion_trace(self.paths, "reports", "mixed")
        self.assertFalse(r["ok"])
        self.assertIn("--force-md", r["error"])

    def test_page_image_renders_a_png_and_refuses_foreign_sources(self):
        r = api.conversion_page_image(self.paths, "reports", "mixed", 3, 300)
        self.assertTrue(r["ok"], r)
        self.assertTrue(r["png"].startswith(b"\x89PNG"))
        self.assertFalse(api.conversion_page_image(self.paths, "reports", "mixed", 99)["ok"])
        meta = self.paths.index / "reports/mixed/index.meta.json"
        data = json.loads(meta.read_text())
        outside = self.tmp / "elsewhere.pdf"
        write_text_pdf(outside, [TEXT])
        data["src_path"] = str(outside)
        meta.write_text(json.dumps(data))
        r = api.conversion_page_image(self.paths, "reports", "mixed", 1)
        self.assertFalse(r["ok"])
        self.assertIn("not available", r["error"])

    def test_page_image_before_the_document_is_embedded(self):
        """A document that is converted but not embedded yet has no index.meta.json: the source is
        found from the trace (its file name) inside the collection's own folder."""
        (self.paths.index / "reports/mixed/index.meta.json").unlink()
        r = api.conversion_page_image(self.paths, "reports", "mixed", 1, 300)
        self.assertTrue(r["ok"], r)
        self.assertTrue(r["png"].startswith(b"\x89PNG"))
        # a trace that names a file outside the collection's folder is not followed
        data = json.loads(self.trace_file().read_text())
        outside = self.tmp / "elsewhere.pdf"
        write_text_pdf(outside, [TEXT])
        data["source"] = "../../../elsewhere.pdf"
        self.trace_file().write_text(json.dumps(data))
        self.assertFalse(api.conversion_page_image(self.paths, "reports", "mixed", 1)["ok"])

    def test_converted_markdown_of_a_document_and_of_one_page(self):
        r = api.conversion_markdown(self.paths, "reports", "mixed")
        self.assertTrue(r["ok"], r)
        whole = r["result"]
        self.assertIn("<!-- page", whole["markdown"])
        self.assertFalse(whole["truncated"])
        self.assertTrue(whole["pages"])
        n = whole["pages"][0]
        one = api.conversion_markdown(self.paths, "reports", "mixed", page=n)["result"]
        self.assertEqual(one["page"], n)
        self.assertNotIn("<!-- page", one["markdown"])
        self.assertIn(one["markdown"], whole["markdown"])
        self.assertFalse(api.conversion_markdown(self.paths, "reports", "mixed", page=99)["ok"])
        self.assertFalse(api.conversion_markdown(self.paths, "reports", "nothere")["ok"])
        self.assertFalse(api.conversion_markdown(self.paths, "reports", "../../x")["ok"])

    def test_collection_info_has_a_conversion_block(self):
        self.write_doc("reports/other.md", "# Other\n\nSome more text that is long enough to be indexed. " * 3)
        self.run_index()
        info = api.collection_info(self.paths, "reports")["result"]["conversion"]
        self.assertEqual(info["documents"], 2)
        self.assertEqual(info["pages"], 4)
        self.assertEqual(info["poor_documents"]["count"], 1)
        self.assertEqual(info["poor_documents"]["items"][0], {"doc": "mixed", "pages": [3]})
        self.assertEqual(info["low_documents"]["count"], 0)

    def test_estimate_counts_pages_per_branch_without_converting(self):
        self.write_doc("reports/other.md", "# Other\n\ntext")
        r = api.conversion_estimate(self.paths, "reports")
        self.assertTrue(r["ok"], r)
        e = r["result"]
        self.assertEqual(e["pages"], 4)
        self.assertEqual(e["branches"], {"digital": 2, "raster": 1, "copy": 1})
        self.assertEqual(e["planned_vlm"]["pages"], 1)
        self.assertFalse(e["partial"])
        self.assertFalse(api.conversion_estimate(self.paths, "nope/none")["ok"])

    def test_estimate_profile_budget_marks_the_rest(self):
        e = estimate.estimate(self.paths, [self.sdir / "reports/mixed.pdf"] * 3, budget_s=1e-9)
        self.assertTrue(e["partial"])
        self.assertLess(e["profiled"], 3)


@unittest.skipUnless(HAVE_PDF, "needs pypdfium2 and Pillow")
class CliTests(ConversionBase):
    def setUp(self):
        super().setUp()
        self.add_pdf()
        self.run_index()

    def test_trace_command(self):
        rc, out, err = run("trace", "reports/mixed")
        self.assertEqual(rc, 0, err)
        self.assertIn("3 pages: digital 2, scanned 1", out)
        self.assertIn("p.3", out)
        self.assertIn("docling poor", out)
        rc, out, _ = run("trace", "reports/mixed", "--page", "3")
        self.assertEqual(rc, 0)
        self.assertIn('"why": "no usable text layer', out)
        rc, out, _ = run("trace", "reports/mixed", "--json")
        self.assertEqual(json.loads(out)["summary"]["strip"], "d2r1")
        self.assertNotEqual(run("trace", "reports")[0], 0)
        self.assertNotEqual(run("trace", "reports/missing")[0], 0)

    def test_collection_info_shows_conversion(self):
        rc, out, err = run("collection", "info", "reports")
        self.assertEqual(rc, 0, err)
        self.assertIn("conversion (pages as last converted):", out)
        self.assertIn("pages 3 in 1 document(s): digital 2, scanned 1", out)
        self.assertIn("docling graded poor: 1 document(s)", out)

    def test_index_estimate_command(self):
        rc, out, err = run("index", "estimate", "reports")
        self.assertEqual(rc, 0, err)
        self.assertIn("pages: 3  (digital 2, scanned 1)", out)
        self.assertIn("docling today:", out)
        rc, out, _ = run("index", "estimate", "--json")
        self.assertEqual(json.loads(out)["pages"], 3)

    def test_index_status_filters_and_summary(self):
        from rag_search import cli
        doc = {"collection": "reports", "source": "mixed.pdf", "status": "indexed", "chunks": 3,
               "convert_s": 1.0, "chunk_s": 0.1, "embed_s": 0.2, "total_s": 1.3,
               "conversion": self.meta()["conversion"]}
        line = cli._fmt_doc(doc)
        self.assertIn("[3 pages: digital 2, scanned 1]", line)
        txt = cli._fmt_docs({"total": 1, "by_status": {"indexed": 1}, "by_branch": {"raster": 1},
                             "by_outcome": {"pass": 1}, "matched": 1, "items": [doc]})
        self.assertIn("documents with pages of each branch: scanned 1", txt)
        job = {"id": "j", "status": "succeeded", "summary": {
            "indexed": 1, "conversion": {"docs": 1, "pages": 3, "branches": {"digital": 2, "raster": 1},
                                         "outcomes": {"pass": 3}, "time_s": {"convert": 2.0},
                                         "cost": {"cpu_s": 1.5, "peak_mb": 300}}}}
        out = cli._fmt_job(job)
        self.assertIn("pages 3 in 1 document(s): digital 2, scanned 1", out)
        self.assertIn("CPU 1.5s", out)
        self.assertIn("--doc-branch", run("index", "status", "--help")[1])


@unittest.skipUnless(HAVE_PDF, "needs pypdfium2 and Pillow")
class DashboardRouteTests(ConversionBase):
    def setUp(self):
        super().setUp()
        self.add_pdf()
        self.run_index()
        self.dash = Dash(self.paths)
        self.addCleanup(self.dash.close)
        # a finished job whose event log holds the run's document events
        rec = {"id": "job1", "status": "succeeded", "created_at": 1, "started_at": 1, "finished_at": 5,
               "spec": {"mode": "new"}, "mode": "new", "summary": {"indexed": 1}}
        (self.paths.jobs / "job1.json").write_text(json.dumps(rec))
        with open(self.paths.jobs / "job1.events.jsonl", "w", encoding="utf-8") as fh:
            for e in self.events:
                if "doc" in e:
                    fh.write(json.dumps({"ts": 2, "event": "doc", **e["doc"]}) + "\n")

    def get(self, path):
        return self.dash.req("GET", path)

    def test_documents_endpoint_filters_by_branch(self):
        st, js, _, _ = self.get("/api/conversion/documents?job_id=job1&branch=raster")
        self.assertEqual(st, 200, js)
        self.assertEqual(js["documents"]["matched"], 1)
        self.assertEqual(js["documents"]["by_branch"], {"digital": 1, "raster": 1})
        st, js, _, _ = self.get("/api/conversion/documents?job_id=job1&branch=image")
        self.assertEqual(js["documents"]["matched"], 0)
        st, js, _, _ = self.get("/api/index/documents?job_id=job1&outcome=pass")
        self.assertEqual(js["documents"]["matched"], 1)

    def test_run_trace_and_page_image_endpoints(self):
        st, js, _, _ = self.get("/api/conversion/run?job_id=job1")
        self.assertEqual(st, 200)
        self.assertEqual(js["job"], "job1")
        st, js, _, _ = self.get("/api/conversion/trace?collection=reports&doc=mixed")
        self.assertEqual((st, js["result"]["summary"]["pages"]), (200, 3))
        st, js, _, _ = self.get("/api/conversion/trace?collection=reports&doc=..%2F..%2Fx")
        self.assertEqual(st, 400)
        st, _, raw, hdrs = self.get("/api/conversion/page-image?collection=reports&doc=mixed&page=1&width=200")
        self.assertEqual(st, 200)
        self.assertTrue(raw.startswith(b"\x89PNG"))
        self.assertEqual(hdrs.get("Content-Type"), "image/png")
        st, _, _, _ = self.get("/api/conversion/page-image?collection=reports&doc=mixed&page=50")
        self.assertEqual(st, 400)
        st, js, _, _ = self.get("/api/conversion/markdown?collection=reports&doc=mixed")
        self.assertEqual(st, 200, js)
        self.assertIn("<!-- page", js["result"]["markdown"])
        st, _, raw, hdrs = self.get("/api/conversion/markdown?collection=reports&doc=mixed&raw=1")
        self.assertEqual((st, hdrs.get("Content-Type")), (200, "text/plain; charset=utf-8"))
        self.assertIn(b"<!-- page", raw)
        st, _, _, _ = self.get("/api/conversion/markdown?collection=reports&doc=..%2F..%2Fx")
        self.assertEqual(st, 400)
        st, _, _, _ = self.dash.req("GET", "/api/conversion/markdown?collection=reports&doc=mixed", auth=False)
        self.assertEqual(st, 401)

    def test_requests_need_the_token(self):
        st, _, _, _ = self.dash.req("GET", "/api/conversion/trace?collection=reports&doc=mixed", auth=False)
        self.assertEqual(st, 401)
        st, _, _, _ = self.dash.req("GET", "/api/conversion/page-image?collection=reports&doc=mixed&page=1", auth=False)
        self.assertEqual(st, 401)

    def test_estimate_works_and_is_allowed_read_only(self):
        st, js, _, _ = self.dash.req("POST", "/api/conversion/estimate", {"target": "reports"})
        self.assertEqual(st, 200, js)
        self.assertEqual(js["result"]["pages"], 3)
        ro = Dash(self.paths, read_only=True)
        self.addCleanup(ro.close)
        st, js, _, _ = ro.req("POST", "/api/conversion/estimate", {"target": ""})
        self.assertEqual(st, 200)
        st, _, _, _ = ro.req("POST", "/api/index/start", {"mode": "new"})
        self.assertEqual(st, 403)

    def test_live_state_carries_the_conversion_view(self):
        live = ui_server_live(self.paths)
        self.assertIn("conversion", live)

    def test_static_files_are_served(self):
        for name in ("conversion.js", "indexing.js", "style.css"):
            st, _, raw, _ = self.get("/static/" + name)
            self.assertEqual(st, 200, name)
        st, _, raw, _ = self.get("/")
        self.assertIn(b"/static/conversion.js", raw)


def ui_server_live(paths):
    from rag_search.ui import server as ui_server
    return ui_server.Live(paths, read_only=False).build_live()


if __name__ == "__main__":
    unittest.main()
