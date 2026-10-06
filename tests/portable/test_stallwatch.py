"""The stall watch: a conversion process that goes silent is found, stopped and reported, and the run goes on; a pool
whose process ended abruptly does not lose the documents that were still waiting."""

from __future__ import annotations

import json
import os
import signal
import time
import unittest
from pathlib import Path
from unittest import mock

from tests.helpers import TempHome

from rag_search.core import indexer, stallwatch
from rag_search.core.conversion import runview


def ev(**kw):
    return json.dumps(kw) + "\n"


class WatchTests(TempHome):
    def setUp(self):
        super().setUp()
        self.log = self.tmp / "events.jsonl"
        self.log.write_text("")
        self.w = stallwatch.StallWatch(self.log, 600, poll_s=20)

    def add(self, **kw):
        with open(self.log, "a") as fh:
            fh.write(ev(ts=time.time(), **kw))

    def test_a_process_that_keeps_reporting_is_never_stalled(self):
        self.add(event="work", pid=11, phase="convert", file="c/a.pdf", status="start")
        for n in range(1, 40):
            self.add(event="step", pid=11, file="c/a.pdf", page=n, what="document reader")
            self.assertEqual(self.w.poll(elapsed=60), [])

    def test_a_silent_process_is_reported_once_with_what_it_was_doing(self):
        self.add(event="work", pid=11, phase="convert", file="c/a.pdf", status="start")
        self.add(event="step", pid=11, file="c/a.pdf", page=7, what="document reader")
        self.assertEqual(self.w.poll(elapsed=30), [])
        self.assertEqual(self.w.poll(elapsed=300), [])
        got = self.w.poll(elapsed=300)
        self.assertEqual([(s["pid"], s["file"], s["page"], s["what"]) for s in got], [(11, "c/a.pdf", 7, "document reader")])
        self.assertEqual(self.w.poll(elapsed=3000), [])                       # once
        text = stallwatch.describe(got[0])
        self.assertIn("page 7", text)
        self.assertIn("document reader", text)
        self.assertIn("10 min", text)

    def test_a_process_between_documents_and_another_phase_are_not_watched(self):
        self.add(event="work", pid=11, phase="convert", file="c/a.pdf", status="start")
        self.add(event="work", pid=11, phase="convert", file="c/a.pdf", status="done", outcome="prepared")
        self.add(event="work", pid=12, phase="embed", file="c/b.pdf", status="start")
        self.assertEqual(self.w.poll(elapsed=5000), [])
        self.assertEqual(self.w.in_flight(), {})

    def test_time_asleep_is_not_time_stalled(self):
        self.add(event="work", pid=11, phase="convert", file="c/a.pdf", status="start")
        self.w.poll()
        self.w._last -= 8 * 3600                                               # the laptop slept for the night
        self.assertEqual(self.w.poll(), [])                                    # one capped interval, not eight hours

    def test_the_limit_is_never_below_the_document_timeout_and_zero_switches_it_off(self):
        self.assertEqual(stallwatch.limit_s({}), 3600.0)
        self.assertEqual(stallwatch.limit_s({"RAG_SEARCH_STALL_TIMEOUT": "0"}), 0.0)
        self.assertEqual(stallwatch.limit_s({"RAG_SEARCH_STALL_TIMEOUT": "60"}), 2700.0 + 900.0)
        self.assertEqual(stallwatch.limit_s({"RAG_SEARCH_STALL_TIMEOUT": "60", "RAG_SEARCH_DOC_TIMEOUT": "100"}), 1000.0)
        self.assertEqual(stallwatch.limit_s({"RAG_SEARCH_STALL_TIMEOUT": "60", "RAG_SEARCH_DOC_TIMEOUT": "0"}), 60.0)
        self.assertEqual(stallwatch.limit_s({"RAG_SEARCH_STALL_TIMEOUT": "x"}), 3600.0)

    def test_the_default_document_timeout_is_the_converter_s(self):
        import re

        src = (Path(indexer.__file__).parent / "docling_convert.py").read_text()
        self.assertEqual(float(re.search(r"^DEFAULT_DOC_TIMEOUT = (\d+)", src, re.M).group(1)), stallwatch.DEFAULT_DOC_TIMEOUT)


def _fake_prepare(task):
    """Stands in for prepare_document in a pool process: what the file's name says happens."""
    name = Path(task["src"]).name
    slog = task["stage_log"]
    indexer.work_event(slog, name, "convert", "start")
    marker = Path(task["src"] + ".tried")
    tries = int(marker.read_text() or 0) if marker.exists() else 0
    marker.write_text(str(tries + 1))
    if name.startswith("hang"):
        time.sleep(600)
    if name.startswith("crash-once") and tries == 0 or name.startswith("crash-always"):
        time.sleep(0.4)                                    # the others are open by now
        os.kill(os.getpid(), signal.SIGKILL)
    if name.startswith("slow"):
        time.sleep(1.5)
    indexer.work_event(slog, name, "convert", "done", outcome="prepared")
    return {"status": "prepared", "src": task["src"]}


class PoolTests(TempHome):
    def setUp(self):
        super().setUp()
        self.log = self.tmp / "events.jsonl"
        self.log.write_text("")
        self.got: dict[str, dict] = {}
        p = mock.patch.object(indexer, "prepare_document", _fake_prepare)
        p.start()
        self.addCleanup(p.stop)

    def tasks(self, *names):
        out = []
        for n in names:
            f = self.tmp / n
            f.write_text("x")
            out.append({"src": str(f), "stage_log": str(self.log)})
        return out

    def run_pool(self, tasks, watch=None, stalled=None):
        rel = {Path(t["src"]).name: t["src"] for t in tasks}
        indexer._convert_in_pool(tasks, 2, lambda r: self.got.__setitem__(Path(r["src"]).name, r), watch=watch,
                                 stalled=stalled if stalled is not None else {}, rel_src=rel, stage_log=str(self.log))

    def test_a_process_that_dies_once_costs_nothing(self):
        w = stallwatch.StallWatch(self.log, 0)
        self.run_pool(self.tasks("crash-once.pdf", "slow-1.pdf", "ok-2.pdf", "ok-3.pdf"), watch=w)
        self.assertEqual({k: v["status"] for k, v in self.got.items()},
                         {"crash-once.pdf": "prepared", "slow-1.pdf": "prepared", "ok-2.pdf": "prepared", "ok-3.pdf": "prepared"})

    def test_a_document_that_kills_its_process_twice_is_left_out_and_the_rest_is_converted(self):
        w = stallwatch.StallWatch(self.log, 0)
        self.run_pool(self.tasks("crash-always.pdf", "slow-1.pdf", "ok-2.pdf", "ok-3.pdf", "ok-4.pdf"), watch=w)
        self.assertEqual(self.got["crash-always.pdf"]["status"], "error")
        self.assertIn("ended abruptly twice", self.got["crash-always.pdf"]["message"])
        self.assertEqual([self.got[n]["status"] for n in ("slow-1.pdf", "ok-2.pdf", "ok-3.pdf", "ok-4.pdf")], ["prepared"] * 4)
        self.assertEqual(int((self.tmp / "crash-always.pdf.tried").read_text()), 2)

    def test_a_stalled_process_is_stopped_reported_and_the_run_goes_on(self):
        stalled: dict[str, str] = {}
        w = stallwatch.StallWatch(self.log, 1.0, poll_s=0.2)

        def on_stall(st):
            stalled[st["file"]] = stallwatch.describe(st)
            indexer.work_event(str(self.log), st["file"], "convert", "done", outcome="stalled", of_pid=st["pid"])
            os.kill(st["pid"], signal.SIGKILL)

        w.start(on_stall)
        self.addCleanup(w.stop)
        t0 = time.time()
        self.run_pool(self.tasks("hang.pdf", "ok-1.pdf", "ok-2.pdf", "ok-3.pdf"), watch=w, stalled=stalled)
        self.assertLess(time.time() - t0, 60)
        self.assertEqual(self.got["hang.pdf"]["status"], "error")
        self.assertIn("stalled: no progress", self.got["hang.pdf"]["message"])
        self.assertEqual([self.got[n]["status"] for n in ("ok-1.pdf", "ok-2.pdf", "ok-3.pdf")], ["prepared"] * 3)
        st = runview._fresh()                                   # the dashboard: the stopped process has nothing open
        for line in self.log.read_text().splitlines():
            runview._feed(st, line)
        self.assertTrue(all(p["open"] is None for p in st["procs"].values()))
        self.assertEqual(st["phases"]["convert"]["outcomes"].get("stalled"), 1)

    def test_without_an_event_log_a_broken_pool_still_ends(self):
        tasks = self.tasks("crash-always.pdf", "ok-1.pdf")
        self.run_pool(tasks, watch=None)
        self.assertEqual(set(self.got), {"crash-always.pdf", "ok-1.pdf"})
        self.assertEqual(self.got["crash-always.pdf"]["status"], "error")


if __name__ == "__main__":
    unittest.main()
