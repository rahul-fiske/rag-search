import fcntl
import json
import os
import subprocess
import sys
import threading
import time
import unittest

from tests.helpers import SUBPROC_PYTHONPATH, TempHome
from rag_search import client, protocol
from rag_search.core.indexer_daemon import IndexerDaemon

DOC = "# Title {n}\n\n<!-- page 1 -->\n\nUnique content about topic{n} with plenty of words."


class IndexerCase(TempHome):
    def setUp(self):
        super().setUp()
        self.use_fake_backends()
        self.daemons = []

    def tearDown(self):
        for d, t in self.daemons:
            d.stop.set()
            t.join(20)
        client.stop(self.paths, "search")
        super().tearDown()

    def start_daemon(self):
        d = IndexerDaemon(self.paths)
        d.idle_poll, d.accept_timeout = 0.2, 0.1
        t = threading.Thread(target=d.run, daemon=True)
        t.start()
        self.daemons.append((d, t))
        end = time.monotonic() + 5
        while time.monotonic() < end and client.ping(self.paths, "indexer", 0.5) is None:
            time.sleep(0.05)
        self.assertIsNotNone(client.ping(self.paths, "indexer"))
        return d

    def call(self, action, **f):
        return client.roundtrip(self.paths, "indexer", protocol.make_request(action, "cli", **f), 60)

    def docs(self, n=3, coll="c"):
        for i in range(n):
            self.write_doc(f"{coll}/d{i}.md", DOC.format(n=i))

    def wait_idle(self, timeout=60):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            st = self.call("status")
            if not st["running"]:
                return st["job"]
            time.sleep(0.2)
        self.fail("indexing did not finish")

    def slow(self, seconds="1.0"):
        os.environ["RAG_SEARCH_EMBEDDER"] = "tests.helpers:SlowEmbedder"
        os.environ["TEST_EMBED_SLEEP"] = seconds


class IndexerDaemonTests(IndexerCase):
    def test_run_publishes_and_search_daemon_serves_new_docs(self):
        self.docs()
        d = self.start_daemon()
        r = self.call("start")
        self.assertTrue(r["ok"] and r["started"], r)
        job = self.wait_idle()
        self.assertEqual(job["status"], "succeeded", job)
        self.assertEqual(job["summary"]["indexed"], 3)
        self.assertEqual(job["publish"]["generation"], 1)
        self.assertIn("search_reload", job)
        self.assertEqual(job["search_reload"]["ok"], True, job["search_reload"])
        self.assertFalse(d.is_busy())
        # search daemon was started by the reload step and serves the new generation
        res = client.request_sync(self.paths, "search", "search", query="topic1", wait_s=40)
        self.assertTrue(res["ok"], res)
        self.assertEqual(res["result"]["results"][0]["file"], "d1")
        # a second run finds everything fresh, publishes nothing new
        self.call("start")
        job2 = self.wait_idle()
        self.assertEqual((job2["summary"]["indexed"], job2["summary"]["skipped_fresh"]), (0, 3))
        self.assertFalse(job2["publish"]["changed"])

    def test_second_search_daemon_reload_when_running(self):
        self.docs(1)
        self.start_daemon()
        self.call("start")
        self.wait_idle()
        self.assertTrue(client.request_sync(self.paths, "search", "list", wait_s=40)["ok"])
        self.write_doc("c/new.md", DOC.format(n=77))
        self.call("start")
        job = self.wait_idle()
        self.assertEqual(job["search_reload"], {"ok": True, "generation": 2, "changed": True,
                                              "reused": [], "loaded": ["c"]})

    def test_start_is_idempotent_while_running(self):
        self.docs()
        self.slow("1.0")
        self.start_daemon()
        a = self.call("start")
        b = self.call("start")
        self.assertTrue(a["started"])
        self.assertEqual((b["started"], b["already_running"], b["job"]["id"]),
                         (False, True, a["job"]["id"]))
        self.assertEqual(self.wait_idle()["status"], "succeeded")

    def test_restart_kills_worker_group_and_starts_over(self):
        self.docs()
        self.slow("1.5")
        self.start_daemon()
        a = self.call("start")
        end = time.monotonic() + 10
        pid = None
        while time.monotonic() < end and not pid:
            pid = self.call("status")["job"]["pid"]
            time.sleep(0.1)
        self.assertTrue(pid)
        b = self.call("start", restart=True)
        self.assertTrue(b["started"] and b["restarted"], b)
        self.assertNotEqual(b["job"]["id"], a["job"]["id"])
        with self.assertRaises(OSError):
            os.killpg(pid, 0)  # old worker group is gone
        old = self.call("status", job_id=a["job"]["id"])["job"]
        self.assertEqual(old["status"], "cancelled")
        os.environ["TEST_EMBED_SLEEP"] = "0"
        final = self.wait_idle()
        self.assertEqual(final["id"], b["job"]["id"])
        self.assertEqual(final["status"], "succeeded")

    def test_the_daemon_log_follows_each_file_through_the_pipeline(self):
        self.docs(2)
        self.start_daemon()
        with self.assertLogs("rag_search.indexer_daemon", level="INFO") as cm:
            self.call("start")
            self.assertEqual(self.wait_idle()["status"], "succeeded")
            time.sleep(0.6)
        lines = "\n".join(cm.output)
        for name in ("c/d0.md", "c/d1.md"):
            for what in ("2 Fingerprint +changed", "3 Convert +started", "3 Convert +done in", "3.1 Profile",
                         "4 Chunk +\\d+ chunks", "5 Embed +started", "5 Embed +done in",
                         "6 Write +embeddings", "INDEXED"):
                self.assertRegex(lines, rf"{name}\s+{what}", (name, what))
        self.assertIn("looking at 2 file(s)", lines)
        self.assertRegex(lines, r"run \S+ finished: 2 indexed")

    def test_a_failed_file_is_logged_with_its_cause(self):
        from rag_search.core.indexer_daemon import _log_event
        with self.assertLogs("rag_search.indexer_daemon", level="INFO") as cm:
            _log_event("j1", {"event": "doc", "collection": "c", "source": "bad.pdf", "status": "error",
                              "message": "UnicodeDecodeError: bad\n [at x.py:1 in f]"})
        self.assertIn("c/bad.pdf  FAILED   UnicodeDecodeError: bad [at x.py:1 in f]", cm.output[0])
        self.assertTrue(cm.output[0].startswith("WARNING"))
        with self.assertLogs("rag_search.indexer_daemon", level="INFO") as cm:      # with its folder, as the stage lines
            _log_event("j1", {"event": "doc", "collection": "c", "source": "a.md", "path": "sub/a.md", "status": "indexed"})
        self.assertIn("c/sub/a.md  INDEXED", cm.output[0])

    def test_cancel_stops_and_does_not_publish(self):
        self.docs()
        self.slow("2.0")
        self.start_daemon()
        self.call("start")
        time.sleep(1.0)
        c = self.call("cancel")
        self.assertTrue(c["cancelled"])
        st = self.call("status")
        self.assertFalse(st["running"])
        self.assertEqual(st["job"]["status"], "cancelled")
        self.assertIsNone(self.paths.current_gen())
        self.assertFalse(self.call("cancel")["cancelled"])  # nothing left to cancel
        self.assertFalse(client.alive_lock_held(self.paths, "search"))  # no reload happened

    def test_a_run_that_reports_nothing_for_twice_the_stall_limit_is_stopped_and_says_so(self):
        from unittest import mock

        self.docs(1)
        self.slow("30")                                    # the embedder sleeps: the event log stays silent
        with mock.patch("rag_search.core.stallwatch.limit_s", return_value=0.6):
            self.start_daemon()
            self.call("start")
            job = self.wait_idle(timeout=30)
        self.assertEqual(job["status"], "failed")
        self.assertIn("stalled: the run reported nothing", job["error"])
        self.assertIsNone(self.paths.current_gen())        # nothing was published

    def test_work_resumes_after_cancel_because_finished_docs_are_skipped(self):
        self.docs(4)
        self.slow("1.0")
        self.start_daemon()
        self.call("start")
        time.sleep(2.5)
        self.call("cancel")
        os.environ["TEST_EMBED_SLEEP"] = "0"
        self.call("start")
        job = self.wait_idle()
        self.assertEqual(job["status"], "succeeded")
        s = job["summary"]
        self.assertEqual(s["indexed"] + s["skipped_fresh"], 4)
        self.assertGreaterEqual(s["skipped_fresh"], 1)

    def test_follow_streams_progress_then_end(self):
        self.docs()
        self.slow("0.6")
        self.start_daemon()
        self.call("start")
        events = list(client.stream(self.paths, "indexer",
                                    protocol.make_request("follow", "cli"), timeout=60))
        self.assertEqual(events[0]["event"], "status")
        self.assertEqual(events[-1]["event"], "end")
        self.assertEqual(events[-1]["job"]["status"], "succeeded")
        self.assertTrue(any(e.get("event") == "progress" for e in events[1:-1]))
        # following a finished job returns immediately
        again = list(client.stream(self.paths, "indexer",
                                   protocol.make_request("follow", "cli"), timeout=10))
        self.assertEqual([e["event"] for e in again], ["status", "end"])

    def test_mode_all_and_path_scope(self):
        self.docs(2, "a")
        self.docs(2, "b")
        self.start_daemon()
        self.call("start", path="a")
        self.assertEqual(self.wait_idle()["summary"]["scanned"], 2)
        self.call("start", mode="all")
        job = self.wait_idle()
        self.assertEqual((job["summary"]["indexed"], job["summary"]["scanned"]), (4, 4))
        self.call("start", path="a/d0.md", mode="all")
        self.assertEqual(self.wait_idle()["summary"]["scanned"], 1)

    def test_bad_requests(self):
        self.start_daemon()
        self.assertEqual(self.call("start", mode="weird")["code"], protocol.BAD_REQUEST)
        self.assertEqual(self.call("start", path="/etc")["code"], protocol.BAD_REQUEST)
        self.assertEqual(self.call("start", path="nope/x")["code"], protocol.BAD_REQUEST)
        self.assertEqual(self.call("status", job_id="zzz")["code"], protocol.BAD_REQUEST)
        st = self.call("status")
        self.assertEqual((st["ok"], st["running"], st["job"]), (True, False, None))

    def test_publish_refused_while_running_allowed_after(self):
        self.docs()
        self.slow("1.0")
        self.start_daemon()
        self.call("start")
        self.assertEqual(self.call("publish")["code"], protocol.BUSY)
        self.wait_idle()
        self.assertTrue(self.call("publish")["ok"])

    def test_auto_publish_can_be_disabled(self):
        (self.paths.home / "config.json").write_text(json.dumps({"indexer": {"auto_publish": False}}))
        self.docs(1)
        self.start_daemon()
        self.call("start")
        job = self.wait_idle()
        self.assertEqual(job["status"], "succeeded")
        self.assertIsNone(job["publish"])
        self.assertIsNone(self.paths.current_gen())
        pub = self.call("publish")
        self.assertTrue(pub["publish"]["changed"])

    def test_external_index_lock_makes_run_fail_cleanly(self):
        self.docs(1)
        self.start_daemon()
        fd = os.open(self.paths.index_lock, os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            self.call("start")
            job = self.wait_idle()
        finally:
            os.close(fd)
        self.assertEqual(job["status"], "failed")
        self.assertIn("already in progress", job["error"])

    def test_orphaned_records_are_marked_interrupted_on_start(self):
        rec = {"id": "20200101-000000-aaaa", "status": "running", "mode": "new", "path": "",
               "spec": {}, "created_at": 1.0, "pid": 999999, "progress": {}}
        (self.paths.jobs / f"{rec['id']}.json").write_text(json.dumps(rec))
        self.start_daemon()
        got = self.call("status", job_id=rec["id"])["job"]
        self.assertEqual(got["status"], "interrupted")

    def test_orphaned_live_worker_is_killed_on_start(self):
        # a worker that outlived its daemon: same module name in its command line
        stub = self.tmp / "stub"
        (stub / "rag_search" / "core").mkdir(parents=True)
        for p in (stub / "rag_search", stub / "rag_search" / "core"):
            (p / "__init__.py").write_text("")
        (stub / "rag_search" / "core" / "worker.py").write_text("import time\ntime.sleep(60)\n")
        proc = subprocess.Popen([sys.executable, "-m", "rag_search.core.worker", "x"],
                                cwd=str(stub), start_new_session=True,
                                env=dict(os.environ, PYTHONPATH=str(stub)))
        threading.Thread(target=proc.wait, daemon=True).start()  # reap: no zombie group
        try:
            rec = {"id": "20200101-000000-bbbb", "status": "running", "mode": "new", "path": "",
                   "spec": {}, "created_at": 1.0, "pid": proc.pid, "progress": {}}
            (self.paths.jobs / f"{rec['id']}.json").write_text(json.dumps(rec))
            self.start_daemon()
            self.assertIsNotNone(proc.wait(timeout=15))
        finally:
            if proc.poll() is None:
                proc.kill()

    def test_stopping_the_daemon_interrupts_the_run(self):
        self.docs()
        self.slow("2.0")
        d = self.start_daemon()
        a = self.call("start")
        time.sleep(1.0)
        pid = self.call("status")["job"]["pid"]
        self.assertTrue(client.stop(self.paths, "indexer"))
        time.sleep(0.5)
        rec = json.loads((self.paths.jobs / f"{a['job']['id']}.json").read_text())
        self.assertEqual(rec["status"], "interrupted")
        self.assertIn("shutdown requested by a client", rec["error"])
        with self.assertRaises(OSError):
            os.killpg(pid, 0)
        self.assertTrue(d.stop.is_set())

    def test_worker_killed_by_a_signal_is_named(self):
        from rag_search.core.indexer_daemon import _describe_exit
        msg = _describe_exit(-9, "job1")
        self.assertIn("SIGKILL", msg)
        self.assertIn("RAG_SEARCH_JOBS=1", msg)
        self.assertIn("SIGTERM", _describe_exit(-15, "job1"))
        self.assertNotIn("RAG_SEARCH_JOBS", _describe_exit(-15, "job1"))
        self.assertEqual(_describe_exit(3, "job1"), "worker exited with code 3 (see job1.log)")

    def test_always_on_by_default(self):
        d = self.start_daemon()
        self.assertEqual(d.idle_exit_seconds, 0)

    def test_real_subprocess_daemon_via_client_autostart(self):
        self.docs(1)
        try:
            r = client.request_sync(self.paths, "indexer", "start", wait_s=30)
            self.assertTrue(r["ok"] and r["started"], r)
            end = time.monotonic() + 60
            while time.monotonic() < end:
                st = client.request_sync(self.paths, "indexer", "status", autostart=False)
                if not st["running"]:
                    break
                time.sleep(0.3)
            self.assertEqual(st["job"]["status"], "succeeded", st)
        finally:
            client.stop(self.paths, "indexer")


class ConfigTunablesTests(IndexerCase):
    """config.json's `indexer`/`models` tunables become the worker subprocess's environment
    (chunk_size/chunk_overlap instead go into the job spec -- see test_indexer_extra.py /
    core/worker.py, which already reads them from there)."""

    def test_config_tunables_become_env_vars_but_never_override_an_existing_one(self):
        from rag_search import config

        config.update_config(self.paths, "indexer",
                             {"ocr": "smart", "table_mode": "fast", "doc_timeout": 900})
        config.update_config(self.paths, "models", {"embed_batch": 16, "device": "cpu"})
        d = IndexerDaemon(self.paths)
        os.environ["RAG_SEARCH_TABLE_MODE"] = "accurate"  # a real env var: config must not win
        self.addCleanup(os.environ.pop, "RAG_SEARCH_TABLE_MODE", None)
        env = d._worker_env()
        self.assertEqual(env["RAG_SEARCH_OCR"], "smart")
        self.assertEqual(env["RAG_SEARCH_TABLE_MODE"], "accurate")  # the real env var won
        self.assertEqual(env["RAG_SEARCH_DOC_TIMEOUT"], "900")
        self.assertEqual(env["RAG_SEARCH_EMBED_BATCH"], "16")
        self.assertEqual(env["RAG_SEARCH_DEVICE"], "cpu")
        self.assertEqual(env["RAG_SEARCH_HOME"], str(self.paths.home))
        self.assertNotIn("RAG_SEARCH_PDF_BACKEND", env)  # unset in config -- no override at all

    def test_publishing_is_a_phase_of_the_run_with_its_own_duration(self):
        from unittest import mock
        d = IndexerDaemon(self.paths)
        seen = []
        with mock.patch.object(d, "_update", side_effect=lambda rec, **kw: seen.append(kw)), \
                mock.patch("rag_search.api.publish_and_reload", return_value={"publish": {"generation": 3}}):
            d._auto_publish({"id": "x"})
        self.assertEqual(seen[0]["progress"]["phase"], "publish")
        self.assertEqual((seen[1]["publish"], seen[1]["progress"]), ({"generation": 3}, {"phase": "done"}))
        self.assertIn("publish_s", seen[1])

    def test_a_run_with_nothing_registered_is_refused(self):
        from rag_search import locations
        spec, err = IndexerDaemon(self.paths)._validate_spec({"mode": "new"})
        self.assertEqual(err, locations.NO_LOCATIONS)

    def test_chunk_size_and_overlap_flow_into_the_job_spec(self):
        from rag_search import config

        config.update_config(self.paths, "indexer", {"chunk_size": 256, "chunk_overlap": 32})
        self.write_doc("c/a.md", "x")
        d = IndexerDaemon(self.paths)
        spec, err = d._validate_spec({"mode": "new"})
        self.assertEqual(err, "")
        self.assertEqual((spec["chunk_size"], spec["chunk_overlap"]), (256, 32))
        # a request that supplies its own value still wins over the configured default:
        spec2, _ = d._validate_spec({"mode": "new", "chunk_size": 999})
        self.assertEqual(spec2["chunk_size"], 999)


class LayeringTests(unittest.TestCase):
    def _imports(self, module):
        code = (f"import sys; sys.path[:0] = {SUBPROC_PYTHONPATH.split(os.pathsep)!r}; "
                f"import {module}; "
                "bad = [m for m in ('numpy', 'torch', 'docling', 'sentence_transformers', 'mcp') "
                "if m in sys.modules]; print(','.join(bad))")
        return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                              check=True).stdout.strip()

    def test_light_modules_do_not_pull_heavy_dependencies(self):
        for mod in ("rag_search.core.indexer_daemon", "rag_search.client", "rag_search.publish",
                    "rag_search.grep", "rag_search.catalog", "rag_search.config",
                    "rag_search.policy", "rag_search.protocol", "rag_search.paths",
                    "rag_search.models", "rag_search.model_tasks", "rag_search.spec",
                    "rag_search.locations", "rag_search.bundle", "rag_search.lifecycle",
                    "rag_search.descriptions", "rag_search.access", "rag_search.api",
                    "rag_search.inventory"):
            self.assertEqual(self._imports(mod), "", mod)


if __name__ == "__main__":
    unittest.main()


class DetachedStartTests(unittest.TestCase):
    """Long-lived children start in the data folder, not in the caller's folder (which may be deleted)."""

    def test_cwd_is_the_data_folder_and_relative_pythonpath_becomes_absolute(self):
        import os
        import tempfile
        from pathlib import Path

        from rag_search.paths import detached_start

        with tempfile.TemporaryDirectory() as d:
            kw = detached_start(Path(d), {"PYTHONPATH": "src" + os.pathsep + "/abs/x", "A": "1"})
            self.assertEqual(kw["cwd"], d)
            self.assertEqual(kw["env"]["PYTHONPATH"], os.path.abspath("src") + os.pathsep + "/abs/x")
            self.assertEqual(kw["env"]["A"], "1")
            gone = Path(d) / "gone"
        self.assertTrue(os.path.isdir(detached_start(gone, {})["cwd"]))     # a missing folder: the root instead
