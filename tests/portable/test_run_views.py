"""The two bars of the Convert panel: pages in the files being converted, and pages in the files that converted
successfully -- both by what the pages are (any format), a page reused from the cache keeping its kind."""

from __future__ import annotations

import json
import unittest

from rag_search.core.conversion import runview, trace


def rec(page, branch, **kw):
    return {"page": page, "branch": branch, "outcome": "pass", **kw}


class KindTests(unittest.TestCase):
    def test_a_cached_page_keeps_its_kind_in_the_summary(self):
        pages = [rec(1, "cached", was="raster", cache="hit"), rec(2, "cached", was="digital", cache="hit"),
                 rec(3, "digital"), rec(4, "cached", cache="hit")]          # an old record without `was`
        self.assertEqual([trace.page_kind(p) for p in pages], ["raster", "digital", "digital", "cached"])
        s = trace.summarize(pages)
        self.assertEqual(s["branches"], {"raster": 1, "digital": 2, "cached": 1})
        self.assertEqual(s["strip"], "r1d2k1")
        self.assertEqual(s["cached_pages"], 3)

    def test_the_totals_of_successful_files_leave_out_failed_and_empty_ones(self):
        tot = trace.RunTotals()
        tot.add({"pages": 3, "branches": {"raster": 2, "digital": 1}, "cached_pages": 2})
        tot.add({"pages": 5, "branches": {"digital": 5}}, ok=True)
        tot.add({"pages": 4, "branches": {"raster": 4}}, ok=False)           # failed: counted in the run, not in ok_*
        snap = tot.snapshot()
        self.assertEqual((snap["docs"], snap["pages"]), (3, 12))
        self.assertEqual((snap["ok_docs"], snap["ok_pages"]), (2, 8))
        self.assertEqual(snap["ok_branches"], {"raster": 2, "digital": 6})
        self.assertEqual(snap["branches"], {"raster": 6, "digital": 6})

    def test_the_rate_counts_only_pages_that_were_read(self):
        tot = trace.RunTotals()
        tot.add({"pages": 100, "branches": {"digital": 100}, "cached_pages": 100})
        self.assertEqual(tot.snapshot()["pages_per_min"], 0.0)               # nothing was read
        tot.add({"pages": 10, "branches": {"raster": 10}})
        self.assertGreater(tot.snapshot()["pages_per_min"], 0.0)


def ev(**kw):
    return json.dumps(kw)


class ActiveFilesTests(unittest.TestCase):
    def setUp(self):
        self.st = runview._fresh()
        self.t = 1000.0

    def feed(self, **kw):
        self.t += 10
        runview._feed(self.st, ev(ts=self.t, **kw))

    def start(self, pid, file):
        self.feed(event="work", phase="convert", status="start", pid=pid, file=file)

    def test_active_files_count_every_format_and_follow_the_plan(self):
        self.start(11, "c/deed.pdf")
        self.feed(event="plan", pid=11, file="c/deed.pdf", pages=5, branches={"digital": 3, "raster": 2})
        self.start(12, "c/report.docx")
        self.feed(event="plan", pid=12, file="c/report.docx", pages=1, branches={"office": 1})
        self.start(13, "c/new.pdf")                                              # not profiled yet
        a = runview.live_view(self.st, running=True)["active"]
        self.assertEqual(a, {"files": 3, "pages": 6, "done": 0, "branches": {"digital": 3, "raster": 2, "office": 1},
                             "unprofiled": 1})
        for n in (1, 2):
            self.feed(event="page", pid=11, file="c/deed.pdf", page=n, of=5, branch="digital", kind="digital",
                      outcome="pass", cache="")
        a = runview.live_view(self.st, running=True)["active"]
        self.assertEqual((a["files"], a["done"]), (3, 2))
        self.assertEqual(a["branches"]["digital"], 3)                            # the plan, not what is left

    def test_a_finished_file_leaves_the_active_bar(self):
        self.start(11, "c/deed.pdf")
        self.feed(event="plan", pid=11, file="c/deed.pdf", pages=2, branches={"digital": 2})
        self.feed(event="work", phase="convert", status="end", pid=11, file="c/deed.pdf", outcome="converted")
        self.assertEqual(runview.live_view(self.st, running=True)["active"]["files"], 0)
        self.assertEqual(runview.live_view(self.st, running=False)["active"]["pages"], 0)

    def test_finished_pages_are_counted_by_kind_and_the_rate_ignores_reused_pages(self):
        self.start(11, "c/deed.pdf")
        for n, (kind, cache) in enumerate([("raster", "hit"), ("raster", "hit"), ("digital", "hit"),
                                           ("digital", ""), ("raster", "")], 1):
            self.feed(event="page", pid=11, file="c/deed.pdf", page=n, of=5,
                      branch="cached" if cache else kind, kind=kind, outcome="pass", cache=cache)
        live = runview.live_view(self.st, running=True)
        self.assertEqual(live["branches"], {"raster": 3, "digital": 2})
        self.assertEqual((live["pages"], live["cached"], live["read"]), (5, 3, 2))
        self.assertEqual(live["pages_per_min"], 12.0)                            # 2 read pages 10 s apart; the 3 reused ones are not a rate

    def test_events_of_older_runs_without_a_kind_still_count(self):
        self.start(11, "c/a.pdf")
        self.feed(event="page", pid=11, file="c/a.pdf", page=1, of=1, branch="digital", outcome="pass", cache="")
        self.assertEqual(runview.live_view(self.st, running=True)["branches"], {"digital": 1})


class LaneCountTests(unittest.TestCase):
    """Pages are also counted by the lane (a text layer, b OCR, c text layer + pictures, d document reader) whose reader
    finished them, and by the lane that handed them on."""

    def test_the_summary_totals_and_the_live_view_count_lanes_and_moves(self):
        pages = [rec(1, "digital", route={"runway": "a", "final": "a", "reasons": []}),
                 rec(2, "embedded", route={"runway": "c", "final": "c", "reasons": []}),
                 rec(3, "fallback", route={"runway": "b", "final": "b", "engine": "tesseract", "reasons": []}),
                 rec(4, "raster", route={"runway": "b", "final": "d", "reasons": [], "escalated_from": {"runway": "b", "checks": ["plausibility"]}}),
                 rec(5, "raster", route={"runway": "d", "final": "d", "reasons": []}), rec(6, "digital")]
        s = trace.summarize(pages)
        self.assertEqual(s["runways"], {"a": 1, "c": 1, "b": 1, "d": 2})
        self.assertEqual(s["moves"], {"b>d": 1})
        self.assertEqual(s["b_engines"], {"tesseract": 1})
        tot = trace.RunTotals()
        tot.add(s)
        tot.add(s, ok=False)
        snap = tot.snapshot()
        self.assertEqual(snap["runways"]["d"], 4)
        self.assertEqual(snap["ok_runways"]["d"], 2)
        self.assertEqual(snap["moves"], {"b>d": 2})

    def test_page_events_carry_the_lane_and_the_live_view_counts_them_with_their_outcomes(self):
        st, t = runview._fresh(), 1000.0
        evs = [dict(runway="a", outcome="pass"), dict(runway="b", outcome="pass", engine="docling"),
               dict(runway="d", outcome="low", moved="b"), dict(runway="d", outcome="repaired")]
        for i, e in enumerate(evs, 1):
            runview._feed(st, ev(ts=t + i, event="page", pid=1, file="c/a.pdf", page=i, of=4, kind="raster", cache="", **e))
        live = runview.live_view(st, running=True)
        self.assertEqual(live["runways"], {"a": 1, "b": 1, "d": 2})
        self.assertEqual(live["moves"], {"b>d": 1})
        self.assertEqual(live["runway_outcomes"]["d"], {"low": 1, "repaired": 1})
        self.assertEqual(live["engines"], {"docling": 1})


class LogReaderTests(unittest.TestCase):
    """The same events through the reader of the real log file (which picks the lines it parses)."""

    def test_plan_and_step_events_reach_the_view_and_a_quiet_worker_shows(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "job.events.jsonl"
            t = 1000.0
            lines = [ev(ts=t, event="phase", phase="convert", status="start", pid=1, total=2),
                     ev(ts=t + 1, event="work", pid=11, phase="convert", file="c/deed.pdf", status="start"),
                     ev(ts=t + 2, event="plan", pid=11, file="c/deed.pdf", pages=5, branches={"digital": 3, "raster": 2}),
                     ev(ts=t + 3, event="page", pid=11, file="c/deed.pdf", page=1, of=5, kind="digital", outcome="pass", cache="", runway="a"),
                     ev(ts=t + 4, event="step", pid=11, file="c/deed.pdf", page=4, what="document reader")]
            f.write_text("\n".join(lines) + "\n")
            st = runview._read_new(f)
            live = runview.live_view(st, running=True)
            self.assertEqual(live["active"]["branches"], {"digital": 3, "raster": 2})     # the plan event was read
            self.assertEqual(live["active"]["unprofiled"], 0)
            lane = [x for x in runview._proc_lanes(st, True, t + 904) if x["pid"] == 11][0]
            self.assertEqual(lane["step"], {"page": 4, "what": "document reader", "since": t + 4})
            self.assertEqual(lane["quiet_s"], 900.0)
            self.assertEqual(lane["state"], "working")
            self.assertEqual([x["state"] for x in runview._proc_lanes(st, False, t + 904)], ["done"])

    def test_an_interrupted_document_is_not_counted_until_its_new_process_finishes_it(self):
        st = runview._fresh()
        for line in (ev(ts=1, event="work", pid=11, phase="convert", file="c/a.pdf", status="start"),
                     ev(ts=2, event="work", pid=11, phase="convert", file="c/a.pdf", status="done", outcome="interrupted"),
                     ev(ts=3, event="work", pid=12, phase="convert", file="c/a.pdf", status="start"),
                     ev(ts=4, event="work", pid=12, phase="convert", file="c/a.pdf", status="done", outcome="prepared")):
            runview._feed(st, line)
        self.assertEqual(st["phases"]["convert"]["done"], 1)
        self.assertEqual(st["phases"]["convert"]["outcomes"], {"prepared": 1})
        self.assertEqual([x["state"] for x in runview._proc_lanes(st, True, 10)], ["gone", "idle"])


class DocumentListTests(unittest.TestCase):
    def test_two_documents_with_the_same_file_name_are_two_documents(self):
        import tempfile
        from pathlib import Path

        from rag_search import jobs
        from rag_search.paths import get_paths

        with tempfile.TemporaryDirectory() as tmp:
            paths = get_paths(Path(tmp))
            paths.jobs.mkdir(parents=True, exist_ok=True)
            jid = "20260101-000000-abcd"
            lines = [ev(ts=1, event="doc", collection="c", source="statement.pdf", path="2023/statement.pdf", status="converted"),
                     ev(ts=2, event="doc", collection="c", source="statement.pdf", path="2024/statement.pdf", status="error", message="x"),
                     ev(ts=3, event="doc", collection="c", source="statement.pdf", path="2023/statement.pdf", status="indexed"),
                     ev(ts=4, event="doc", collection="c", source="old.pdf", status="indexed")]      # a log from before paths
            jobs.events_file(paths, jid).write_text("\n".join(lines) + "\n")
            d = jobs.documents(paths, jid, limit=10)
            self.assertEqual(d["total"], 3)
            self.assertEqual(d["by_status"], {"indexed": 2, "error": 1})
            self.assertEqual(jobs.documents(paths, jid, limit=10, q="2024")["matched"], 1)


if __name__ == "__main__":
    unittest.main()
