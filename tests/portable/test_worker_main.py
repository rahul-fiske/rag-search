"""The indexing worker's entry point, run in this process (the daemon runs it as a child that leaves with
`os._exit`, which hides it from a coverage tool): what it writes to the job's event log and which exit code it
ends with.  Tier A: fake embedder, documents from the corpus."""

from __future__ import annotations

import json
import unittest
from unittest import mock

from tests import corpus
from tests.helpers import TempHome

from rag_search.core import worker
from rag_search.jobs import events_file, job_file
from rag_search.paths import index_lock, write_json_atomic


class WorkerMainTests(TempHome):
    def setUp(self):
        super().setUp()
        self.use_fake_backends()
        corpus.copy("text/notes.md", self.sdir / "team" / "notes.md")
        self.register_tree()

    def job(self, spec, jid="20260101-000000-wrkr"):
        write_json_atomic(job_file(self.paths, jid), {"id": jid, "spec": spec})
        return jid

    def events(self, jid):
        return [json.loads(x) for x in events_file(self.paths, jid).read_text().splitlines()]

    def test_a_run_writes_its_start_progress_documents_and_result_and_exits_zero(self):
        jid = self.job({"mode": "new"})
        self.assertEqual(worker.main([jid]), worker.EXIT_OK)
        kinds = [e["event"] for e in self.events(jid)]
        self.assertEqual(kinds[0], "start")
        self.assertIn("doc", kinds)
        self.assertEqual(kinds[-1], "result")
        res = self.events(jid)[-1]["summary"]
        self.assertEqual((res["indexed"], res["errors"]), (1, []))

    def test_a_bad_request_a_busy_workspace_and_a_crash_end_with_their_own_codes(self):
        bad = self.job({"mode": "new", "path": str(self.tmp / "outside")}, "20260101-000000-bad0")
        self.assertEqual(worker.main([bad]), worker.EXIT_FAILED)
        err = [e for e in self.events(bad) if e["event"] == "error"][0]
        self.assertIn("must be inside a registered location", err["error"])
        busy = self.job({"mode": "new"}, "20260101-000000-busy")
        with index_lock(self.paths):
            self.assertEqual(worker.main([busy]), worker.EXIT_BUSY)
        self.assertTrue([e for e in self.events(busy) if e["event"] == "error"][0]["busy"])
        boom = self.job({"mode": "new"}, "20260101-000000-boom")
        with mock.patch.object(worker, "run_spec", side_effect=RuntimeError("the disk went away")):
            self.assertEqual(worker.main([boom]), worker.EXIT_FAILED)
        self.assertIn("RuntimeError: the disk went away", [e for e in self.events(boom) if e["event"] == "error"][0]["error"])

    def test_the_worker_needs_exactly_one_job_id(self):
        self.assertEqual(worker.main([]), 2)
        self.assertEqual(worker.main(["a", "b"]), 2)

    def test_the_worker_always_leaves_through_os_exit_after_flushing(self):
        with mock.patch.object(worker.os, "_exit") as leave, mock.patch.object(worker.logging, "shutdown"):
            worker.leave(3)
        leave.assert_called_once_with(3)

    def test_chatty_progress_is_thinned_but_a_phase_change_and_the_last_step_are_kept(self):
        w = worker.EventWriter(self.tmp / "e.jsonl")
        w.progress({"phase": "embed", "done": 1, "total": 100})
        w.progress({"phase": "embed", "done": 2, "total": 100})                 # same phase, too soon: dropped
        w.progress({"phase": "merge", "done": 0, "total": 1})                   # a new phase: kept
        w.progress({"phase": "merge", "done": 1, "total": 1})                   # the last step: kept
        w.progress({"doc": {"source": "a.md", "status": "indexed"}})
        w.progress({"stage": {"file": "a.md", "stage": "convert", "status": "start"}})
        w.close()
        rows = [json.loads(x) for x in (self.tmp / "e.jsonl").read_text().splitlines()]
        self.assertEqual([r["event"] for r in rows], ["progress", "progress", "progress", "doc", "stage"])
        self.assertIn("pid", rows[-1])

    def test_a_change_of_document_is_never_throttled_and_work_and_phase_events_pass(self):
        w = worker.EventWriter(self.tmp / "t.jsonl")
        w.progress({"phase": "embed", "done": 1, "total": 9, "current": "a/x.md"})
        w.progress({"phase": "embed", "done": 1, "total": 9, "current": "a/x.md"})      # same document, too soon: dropped
        w.progress({"phase": "embed", "done": 1, "total": 9, "current": "a/y.md"})      # another document: kept at once
        w.progress({"work": {"phase": "embed", "file": "a/y.md", "status": "start"}})
        w.progress({"phase_event": {"phase": "embed", "status": "start", "total": 9}})
        w.close()
        rows = [json.loads(x) for x in (self.tmp / "t.jsonl").read_text().splitlines()]
        self.assertEqual([(r["event"], r.get("current")) for r in rows],
                         [("progress", "a/x.md"), ("progress", "a/y.md"), ("work", None), ("phase", None)])
        self.assertTrue(all("pid" in r for r in rows[2:]))


if __name__ == "__main__":
    unittest.main()
