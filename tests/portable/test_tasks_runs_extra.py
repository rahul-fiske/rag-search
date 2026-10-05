"""Model tasks and Playground runs: how a task or a run ends when something goes wrong, the supervisor's
bookkeeping, and the model smoke test's refusals.  Tier A: the model is a fake backend that misbehaves in one
chosen way, the runner and the detached child are replaced where they would start a process."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import unittest
from unittest import mock

import numpy as np

from tests.helpers import FakeEmbedder, FakeReranker, TempHome

from rag_search import model_tasks as mt
from rag_search import models, playground_runs as pr
from rag_search.paths import get_playground_paths, write_json_atomic

HERE = "tests.portable.test_tasks_runs_extra"


class FlatEmbedder(FakeEmbedder):
    def encode(self, texts, progress=None):
        return np.zeros(len(texts), dtype=np.float32)               # one dimension too few


class NanEmbedder(FakeEmbedder):
    def encode(self, texts, progress=None):
        return np.full((len(texts), 8), np.nan, dtype=np.float32)


class NanReranker(FakeReranker):
    def score(self, query, texts):
        return [float("nan")] * len(texts)


class BackwardsReranker(FakeReranker):
    def score(self, query, texts):
        return [-s for s in super().score(query, texts)]


class SmokeTestRefusalTests(TempHome):
    """`models verify` loads a model and checks it ranks the relevant passage first."""

    def verify(self, kind, model, **env):
        os.environ.update(env)
        self.addCleanup(lambda: [os.environ.pop(k, None) for k in env])
        os.environ["PYTHONPATH"] = os.pathsep.join(sys.path)
        return mt._verify_here(kind, model)

    def test_a_model_with_unusable_output_is_named_and_refused(self):
        for emb, text in ((f"{HERE}:FlatEmbedder", "unexpected output shape"), (f"{HERE}:NanEmbedder", "NaN/inf")):
            with self.subTest(emb), self.assertRaises(mt.TaskError) as cm:
                self.verify("embedding", "x/y", RAG_SEARCH_EMBEDDER=emb)
            self.assertIn(text, str(cm.exception))
        with self.assertRaises(mt.TaskError) as cm:             # a catalogue model has a known size: 1024, not 64
            self.verify("embedding", "BAAI/bge-m3", RAG_SEARCH_EMBEDDER="tests.helpers:FakeEmbedder")
        self.assertIn("expected 1024 dimensions but got 64", str(cm.exception))

    def test_a_reranker_that_gives_nonsense_or_ranks_backwards_is_refused(self):
        with self.assertRaises(mt.TaskError) as cm:
            self.verify("reranker", "x/y", RAG_SEARCH_RERANKER=f"{HERE}:NanReranker")
        self.assertIn("unusable scores", str(cm.exception))
        with self.assertRaises(mt.TaskError) as cm:
            self.verify("reranker", "x/y", RAG_SEARCH_RERANKER=f"{HERE}:BackwardsReranker")
        self.assertIn("did not put the relevant passage first", str(cm.exception))
        self.assertIn("reranking model", str(cm.exception))

    def test_the_verify_subprocess_wrapper_reports_a_timeout_or_a_crash(self):
        with mock.patch.object(mt.subprocess, "run", side_effect=subprocess.TimeoutExpired("x", 5)):
            self.assertIn("did not finish within 5 s", mt.verify_model("embedding", "x/y", timeout=5)["error"])
        crash = subprocess.CompletedProcess([], 3, "", "Traceback\nboom\nMemoryError")
        with mock.patch.object(mt.subprocess, "run", return_value=crash):
            res = mt.verify_model("embedding", "x/y")
        self.assertFalse(res["ok"])
        self.assertIn("exit 3", res["error"])
        self.assertIn("MemoryError", res["error"])

    def test_the_verify_entry_point_prints_one_result_line(self):
        with mock.patch.object(mt, "_verify_here", return_value={"ok": True, "dim": 8}):
            self.assertEqual(mt.main(["verify", "embedding", "x/y"]), 0)
        with mock.patch.object(mt, "_verify_here", side_effect=RuntimeError("no weights")):
            self.assertEqual(mt.main(["verify", "embedding", "x/y"]), 1)
        self.assertEqual(mt.main(["nonsense"]), 64)


class TaskRunnerTests(TempHome):
    def runner(self, fn):
        return mock.patch.dict(mt.RUNNERS, {"verify": fn})

    def test_how_a_task_ends_is_recorded_whatever_went_wrong(self):
        def cancelled(paths, task, req):
            raise mt.Cancelled()

        def bad_model(paths, task, req):
            raise models.ModelError("no such model")

        def crash(paths, task, req):
            raise KeyError("oops")

        for fn, status, text in ((cancelled, "cancelled", ""), (bad_model, "failed", "no such model"),
                                 (crash, "failed", "KeyError")):
            with self.subTest(status + text), self.runner(fn):
                rec = mt.run(self.paths, {"op": "verify", "targets": []})
            self.assertEqual(rec["status"], status)
            self.assertIn(text, rec.get("error", ""))
        with self.assertRaises(mt.TaskError):
            mt.run(self.paths, {"op": "nonsense"})

    def test_the_detached_runner_exits_zero_on_success_and_two_when_busy(self):
        with mock.patch.object(mt, "run", return_value={"status": "succeeded"}):
            self.assertEqual(mt.main(["run", json.dumps({"op": "verify"})]), 0)
        with mock.patch.object(mt, "run", return_value={"status": "failed"}):
            self.assertEqual(mt.main(["run", json.dumps({"op": "verify"})]), 1)
        with mock.patch.object(mt, "run", side_effect=mt.TaskBusy("one at a time")):
            self.assertEqual(mt.main(["run", json.dumps({"op": "verify"})]), 2)

    def test_starting_a_task_in_the_background_writes_its_record_first(self):
        with mock.patch.object(mt.subprocess, "Popen") as popen:
            queued = mt.start_detached(self.paths, {"op": "verify", "kind": "embedding", "model": "x/y"})
        self.assertEqual((queued["status"], queued["op"], queued["model"]), ("queued", "verify", "x/y"))
        self.assertEqual(mt.read_task(self.paths)["id"], queued["id"])
        self.assertEqual(json.loads(popen.call_args.args[0][-1])["task_id"], queued["id"])
        with self.assertRaises(mt.TaskError):
            mt.start_detached(self.paths, {"op": "nonsense"})
        with mock.patch.object(mt, "is_active", return_value=True), self.assertRaises(mt.TaskBusy):
            mt.start_detached(self.paths, {"op": "verify"})

    def test_cancel_signals_the_running_task_and_only_that(self):
        self.assertFalse(mt.cancel(self.paths))                              # nothing recorded
        write_json_atomic(mt.task_file(self.paths), {"id": "t0", "status": "succeeded", "pid": 5})
        self.assertFalse(mt.cancel(self.paths))                              # finished
        write_json_atomic(mt.task_file(self.paths), {"id": "t1", "status": "running", "pid": 4321})
        self.assertFalse(mt.cancel(self.paths))                              # its runner is gone: reported failed
        import fcntl

        fd = os.open(mt.lock_file(self.paths), os.O_RDWR | os.O_CREAT)       # what a live runner holds
        self.addCleanup(os.close, fd)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with mock.patch.object(mt.os, "kill") as kill:
            self.assertTrue(mt.cancel(self.paths))
        kill.assert_called_once_with(4321, signal.SIGTERM)
        with mock.patch.object(mt.os, "kill", side_effect=ProcessLookupError):
            self.assertFalse(mt.cancel(self.paths))

    def test_streamed_output_is_logged_line_by_line_and_a_failed_log_stops_the_child(self):
        class T:
            def __init__(self):
                self.lines = []

            def log(self, line):
                self.lines.append(line)

        t = T()
        code = mt._stream([sys.executable, "-c", "print('first'); print(); print('second')"], t)
        self.assertEqual((code, t.lines), (0, ["first", "second"]))

        class Broken(T):
            def log(self, line):
                raise OSError("disk full")

        with self.assertRaises(OSError):
            mt._stream([sys.executable, "-c", "import time; print('x', flush=True); time.sleep(30)"], Broken())

    def test_the_reader_runtime_install_reports_a_failed_installer_and_a_missing_package(self):
        task = mock.Mock()
        with mock.patch.object(models, "_apple_silicon", return_value=True), \
                mock.patch.object(models, "runtime_requirements", return_value=["mlx-vlm"]), \
                mock.patch.object(mt, "install_command", return_value=["pip", "install", "mlx-vlm"]):
            with mock.patch.object(mt, "_stream", return_value=2), self.assertRaises(mt.TaskError) as cm:
                mt.run_runtime(self.paths, task, {})
            self.assertIn("installer exited with code 2", str(cm.exception))
            with mock.patch.object(mt, "_stream", return_value=0), \
                    mock.patch.object(models, "runtime_state", return_value={"ready": False, "packages": []}), \
                    self.assertRaises(mt.TaskError) as cm:
                mt.run_runtime(self.paths, task, {})
            self.assertIn("cannot be found by rag-search", str(cm.exception))

    def test_a_single_model_download_task_downloads_that_model(self):
        task = mock.Mock()
        with mock.patch.object(mt, "download_model", return_value={"path": "/p"}) as dl:
            res = mt.run_download(self.paths, task, {"model": "x/y"})
        self.assertEqual(dl.call_args.args[0], "x/y")
        self.assertEqual(res, {"downloaded": ["x/y"]})

    def test_a_task_lock_nobody_holds_is_not_active(self):
        self.assertFalse(mt.is_active(self.paths))
        mt.lock_file(self.paths).write_text("")
        self.assertFalse(mt.is_active(self.paths))


class PlaygroundRunRecordTests(TempHome):
    def setUp(self):
        super().setUp()
        self.exp = get_playground_paths(self.paths, "demo")
        self.exp.home.mkdir(parents=True)
        self.exp.jobs.mkdir(parents=True)

    def test_a_run_that_cannot_start_says_why(self):
        with self.assertRaises(pr.RunError) as cm:
            pr.start(self.paths, "demo", "nonsense")
        self.assertIn("unknown run kind", str(cm.exception))
        with self.assertRaises(pr.RunError) as cm:
            pr.start(self.paths, "ghost", "index")
        self.assertIn("no such experiment", str(cm.exception))
        with self.assertRaises(pr.RunError):
            pr.view(self.paths, "ghost")
        with self.assertRaises(pr.RunError):
            pr.view(self.paths, "demo", "../escape")

    def test_cancelling_when_nothing_runs_is_harmless_and_a_run_can_be_cancelled(self):
        self.assertEqual(pr.cancel(self.paths, "demo"), {"cancelled": False})
        write_json_atomic(self.exp.jobs / "20260101-000000-abcd.json",
                          {"id": "20260101-000000-abcd", "kind": "index", "status": "running",
                           "pid": os.getpid(), "created_at": time.time()})
        with mock.patch.object(pr.os, "killpg", side_effect=OSError("gone")):
            res = pr.cancel(self.paths, "demo")
        self.assertEqual(res, {"cancelled": True, "job": "20260101-000000-abcd"})
        self.assertEqual(json.loads((self.exp.jobs / "20260101-000000-abcd.json").read_text())["status"], "cancelled")

    def test_a_recorder_follows_a_run_from_start_to_a_summary_or_to_the_error(self):
        def work(rec):
            rec.progress({"phase": "convert", "done": 1, "total": 4, "current": "a.pdf"})
            rec.progress({"query": {"query": "q", "hit_rank": 1}})
            rec.progress({"phase": "embed", "done": 2, "total": 4})
            return {"indexed": 4}

        res = pr.run_in_child(self.paths, "demo", "20260101-000000-aaaa", "index", work)
        self.assertEqual(res, {"indexed": 4})
        rec = json.loads((self.exp.jobs / "20260101-000000-aaaa.json").read_text())
        self.assertEqual((rec["status"], rec["files"], rec["summary"], rec["progress"]["phase"]),
                         ("done", 4, {"indexed": 4}, "embed"))
        events = (self.exp.jobs / "20260101-000000-aaaa.events.jsonl").read_text()
        self.assertIn('"event": "query"', events)

        def broken(rec):
            raise RuntimeError("the model fell over")

        with self.assertRaises(RuntimeError):
            pr.run_in_child(self.paths, "demo", "20260101-000000-bbbb", "bench", broken)
        rec = json.loads((self.exp.jobs / "20260101-000000-bbbb.json").read_text())
        self.assertEqual((rec["status"], rec["error"]), ("failed", "RuntimeError: the model fell over"))

    def test_the_timeline_skips_unreadable_lines_and_caps_the_pages_it_keeps(self):
        jid = "20260101-000000-cccc"
        lines = ['{"event": "stage", "file": "c/a.pdf", "stage": "convert", "status": "start"}', "not json {",
                 '{"event": "stage", "file": "", "stage": "convert"}', '{"event": "other", "file": "x"}',
                 '{"event": "query", "query": "q", "pid": 1}']
        lines += [json.dumps({"event": "page", "file": "c/a.pdf", "page": n, "branch": "digital"})
                  for n in range(pr.MAX_PAGES_PER_DOC + 5)]
        (self.exp.jobs / f"{jid}.events.jsonl").write_text("\n".join(lines) + "\n")
        tl = pr.timeline(self.exp, jid)
        self.assertEqual(len(tl["docs"]), 1)
        self.assertEqual(len(tl["docs"][0]["pages"]), pr.MAX_PAGES_PER_DOC)
        self.assertEqual(tl["queries"], [{"query": "q"}])
        self.assertEqual(pr.timeline(self.exp, "20260101-000000-none"), {"docs": [], "queries": []})

    def test_a_failed_run_shows_its_error_in_the_view_and_an_empty_experiment_has_no_job(self):
        self.assertEqual(pr.view(self.paths, "demo"), {"ok": True, "job": None})
        write_json_atomic(self.exp.jobs / "20260101-000000-dddd.json",
                          {"id": "20260101-000000-dddd", "kind": "bench", "status": "failed",
                           "error": "no queries", "created_at": 1.0})
        v = pr.view(self.paths, "demo")
        self.assertEqual((v["job"]["error"], v["job"]["kind"]), ("no queries", "bench"))

    def test_a_process_that_never_wrote_a_record_but_left_a_pid_file_is_found(self):
        jid = "20260101-000000-eeee"
        write_json_atomic(self.exp.jobs / f"{jid}.json", {"id": jid, "kind": "index", "status": "queued",
                                                           "created_at": time.time() - 100})
        (self.exp.jobs / f"{jid}.pid").write_text("not a number")
        self.assertEqual(pr._pid(self.exp, {"id": jid}), 0)
        (self.exp.jobs / f"{jid}.log").write_text("Traceback ... boom")
        v = pr.view(self.paths, "demo", jid)
        self.assertEqual(v["job"]["status"], "failed")
        self.assertIn("boom", v["job"]["error"])


class DaemonProtocolTests(TempHome):
    """The socket server shared by both daemons answers a bad request instead of dying."""

    def test_a_request_that_is_not_json_or_that_makes_the_daemon_raise_gets_an_error_reply(self):
        import socket
        import threading

        from rag_search import client, protocol
        from rag_search.core.daemon_base import DaemonBase

        class Boom(DaemonBase):
            kind = "search"

            def dispatch(self, req, conn):
                if req.get("action") == "boom":
                    raise RuntimeError("it broke")
                return super().dispatch(req, conn)

        d = Boom(self.paths, idle_exit_seconds=0)
        t = threading.Thread(target=d.run, daemon=True)
        t.start()
        for _ in range(100):
            if self.paths.socket("search").exists():
                break
            time.sleep(0.05)
        self.addCleanup(lambda: (d.stop.set(), t.join(5)))

        def raw(data: bytes):
            s = socket.socket(socket.AF_UNIX)
            s.connect(str(self.paths.socket("search")))
            s.sendall(data)
            out = protocol.read_line(s)
            s.close()
            return json.loads(out)

        self.assertEqual(raw(b"this is not json\n")["code"], protocol.BAD_REQUEST)
        self.assertEqual(raw(protocol.encode(protocol.make_request("boom", client="cli")))["code"], protocol.INTERNAL)
        self.assertEqual(raw(protocol.encode(protocol.make_request("nonsense", client="cli")))["code"],
                         protocol.BAD_REQUEST)
        self.assertTrue(client.ping(self.paths, "search") is not None)


if __name__ == "__main__":
    unittest.main()
