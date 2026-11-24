"""Retention: the records and logs of old runs are deleted (after every run, from the CLI and the dashboard), the run in
progress and the newest finished run never; the event log is written whole lines at a time from several threads."""

from __future__ import annotations

import json
import os
import threading
import time
import unittest
from unittest import mock

from rag_search import api, jobs
from rag_search.core import events
from rag_search.core.worker import EventWriter

from ..helpers import TempHome


def make_run(paths, jid, status="succeeded", created=0.0, log_bytes=10):
    jobs.job_file(paths, jid).write_text(json.dumps({"id": jid, "status": status, "created_at": created}))
    jobs.events_file(paths, jid).write_text("x" * log_bytes)
    (paths.jobs / f"{jid}.log").write_text("log")


class PruneTests(TempHome):
    def runs(self):
        return sorted(r["id"] for r in jobs.all_records(self.paths))

    def test_only_the_last_n_runs_stay_and_their_three_files_go_together(self):
        for i in range(6):
            make_run(self.paths, f"r{i}", created=float(i))
        res = jobs.prune(self.paths, 2)
        self.assertEqual(self.runs(), ["r4", "r5"])
        self.assertEqual(sorted(res["removed"]), ["r0", "r1", "r2", "r3"])
        self.assertEqual(res["files"], 12)
        self.assertFalse(any(self.paths.jobs.glob("r0*")))

    def test_a_running_run_and_the_newest_finished_one_are_never_removed(self):
        make_run(self.paths, "old-done", created=1.0)
        make_run(self.paths, "running", status="running", created=2.0)
        make_run(self.paths, "x1", status="failed", created=3.0)
        make_run(self.paths, "queued", status="queued", created=4.0)
        jobs.prune(self.paths, 1)
        self.assertEqual(self.runs(), ["queued", "running", "x1"])     # x1 is the newest finished run

    def test_a_dry_run_removes_nothing_and_says_what_it_would_free(self):
        for i in range(3):
            make_run(self.paths, f"r{i}", created=float(i), log_bytes=1000)
        res = jobs.prune(self.paths, 1, dry_run=True)
        self.assertEqual(len(res["removed"]), 2)
        self.assertGreater(res["freed_bytes"], 2000)
        self.assertEqual(len(self.runs()), 3)

    def test_logs_without_a_record_go_once_they_are_quiet(self):
        make_run(self.paths, "r1", created=1.0)
        orphan = jobs.events_file(self.paths, "gone-1")
        orphan.write_text("x")
        fresh = jobs.events_file(self.paths, "gone-2")
        fresh.write_text("x")
        old = time.time() - 2 * jobs.ORPHAN_GRACE_S
        os.utime(orphan, (old, old))
        jobs.prune(self.paths, 5)
        self.assertFalse(orphan.exists())
        self.assertTrue(fresh.exists())                                 # perhaps a run that is starting

    def test_the_setting_and_its_default(self):
        self.assertEqual(jobs.keep_setting({}), jobs.DEFAULT_KEEP)
        self.assertEqual(jobs.keep_setting({"indexer": {"keep_runs": 0}}), 0)
        self.assertEqual(jobs.keep_setting({"indexer": {"keep_runs": "x"}}), jobs.DEFAULT_KEEP)

    def test_the_api_works_without_the_daemon_and_the_status_reports_what_is_on_disk(self):
        for i in range(4):
            make_run(self.paths, f"r{i}", created=float(i))
        r = api.index_prune(self.paths, keep=2)
        self.assertTrue(r["ok"])
        self.assertEqual(len(r["removed"]), 2)
        st = api.index_status(self.paths)
        self.assertEqual(st["runs"]["runs"], 2)
        self.assertEqual(st["runs"]["keep"], jobs.DEFAULT_KEEP)

    def test_the_cli_prunes_and_a_dry_run_says_would(self):
        from rag_search import cli

        for i in range(3):
            make_run(self.paths, f"r{i}", created=float(i))
        def run(*argv):
            with self.assertRaises(SystemExit) as ex:
                cli.main(list(argv))
            return ex.exception.code
        with mock.patch("builtins.print") as out:
            self.assertEqual(run("index", "prune", "--keep", "1", "--dry-run"), 0)
        self.assertIn("would remove 2 run(s)", out.call_args[0][0])
        with mock.patch("builtins.print"):
            self.assertEqual(run("index", "prune", "--keep", "1"), 0)
        self.assertEqual(self.runs(), ["r2"])

    def test_the_daemon_prunes_after_a_run_with_the_setting(self):
        from rag_search.config import update_config
        from rag_search.core.indexer_daemon import IndexerDaemon

        for i in range(5):
            make_run(self.paths, f"r{i}", created=float(i))
        update_config(self.paths, "indexer", {"keep_runs": 3})
        d = IndexerDaemon(self.paths)
        d._gc_jobs()
        self.assertEqual(self.runs(), ["r2", "r3", "r4"])
        self.assertEqual(d.dispatch({"action": "prune", "keep": 0}, None)["kept"], 3)   # 0: the setting
        update_config(self.paths, "indexer", {"keep_runs": 0})                       # 0 in the setting: keep every run
        d._gc_jobs()
        self.assertEqual(len(self.runs()), 3)


class EventLogTests(TempHome):
    def test_lines_written_from_many_threads_and_processes_stay_whole(self):
        path = self.paths.jobs / "t.events.jsonl"
        w = EventWriter(path)
        big = "y" * 20000                                          # longer than any buffer: one write all the same
        def run(n):
            for i in range(50):
                w.emit("doc", n=n, i=i, pad=big)
                events.append(path, "stage", n=n, i=i)
        ts = [threading.Thread(target=run, args=(n,)) for n in range(4)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        w.close()
        lines = path.read_text().splitlines()
        self.assertEqual(len(lines), 400)
        for ln in lines:
            json.loads(ln)

    def test_empty_fields_are_left_out_and_a_missing_log_is_no_error(self):
        path = self.paths.jobs / "e.events.jsonl"
        events.append(path, "page", page=1, model=None, gate=None)
        self.assertEqual(set(json.loads(path.read_text())), {"ts", "event", "page"})
        events.append("", "page", page=1)
        events.append(self.paths.jobs / "no" / "such" / "dir.jsonl", "page")


if __name__ == "__main__":
    unittest.main()
