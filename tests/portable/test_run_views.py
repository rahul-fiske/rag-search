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


if __name__ == "__main__":
    unittest.main()
