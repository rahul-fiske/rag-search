"""Dashboard server branches no other test reached: odd requests, the playground / model / doctor routes'
arguments and errors, and the helpers behind `rag-search ui`.  Tier A: processes and the model tasks are
replaced at the seam (`subprocess.run`, `playground_runs`, `model_tasks`)."""

from __future__ import annotations

import http.server
import json
import subprocess
import threading
import unittest
from unittest import mock

from tests.helpers import TempHome
from tests.portable.test_ui import UiBase

from rag_search import model_tasks, models
from rag_search.ui import server as ui_server


class RequestEdgeTests(UiBase):
    def test_head_is_answered_like_get_and_a_post_with_a_foreign_host_or_path_is_refused(self):
        st, _, body, _ = self.dash.req("HEAD", "/")
        self.assertEqual((st, body), (200, b""))
        st, _, _, _ = self.dash.req("POST", "/api/daemon", {"action": "stop"}, host="evil.example")
        self.assertEqual(st, 403)
        st, _, _, _ = self.dash.req("POST", "/elsewhere", {})
        self.assertEqual(st, 404)

    def test_pipeline_and_publish_and_doctor_routes(self):
        st, js, _, _ = self.dash.req("GET", "/api/pipeline")
        self.assertEqual(st, 200)
        self.assertTrue(js)
        with mock.patch.object(ui_server.api, "index_publish", return_value={"ok": True, "publish": {}}) as pub:
            st, js, _, _ = self.dash.req("POST", "/api/index/publish", {})
        self.assertEqual((st, js["ok"], pub.call_args.kwargs["client"]), (200, True, "cli"))
        rows = [{"status": "ok", "check": "python", "detail": "3.12"}]
        done = subprocess.CompletedProcess([], 1, json.dumps(rows), "")
        with mock.patch.object(ui_server.subprocess, "run", return_value=done):
            st, js, _, _ = self.dash.req("POST", "/api/doctor", {})
        self.assertEqual((js["ok"], js["rows"], js["failed"]), (True, rows, True))     # exit 1: a check failed
        with mock.patch.object(ui_server.subprocess, "run", return_value=subprocess.CompletedProcess([], 2, "x", "boom")):
            st, js, _, _ = self.dash.req("POST", "/api/doctor", {})
        self.assertEqual((js["ok"], js["error"]), (False, "boom"))
        with mock.patch.object(ui_server.subprocess, "run", side_effect=subprocess.TimeoutExpired("doctor", 120)):
            st, js, _, _ = self.dash.req("POST", "/api/doctor", {})
        self.assertIn("timed out", js["error"])


class PlaygroundRouteTests(UiBase):
    """Each route turns the request into `rag-search playground ...` arguments; the child process is replaced."""

    def post(self, action, body):
        calls = []

        def fake(handler, args, timeout):
            calls.append((args, timeout))
            return {"ok": True, "result": {}}

        with mock.patch.object(ui_server.Handler, "_playground_cli", autospec=True, side_effect=fake):
            st, js, _, _ = self.dash.req("POST", f"/api/playground/{action}", body)
        return st, js, calls

    def test_create_config_search_compare_preview_promote_settings_and_remove(self):
        st, _, calls = self.post("create", {"name": "e", "collection": "docs", "from_production": True})
        self.assertEqual(calls[0][0], ["create", "e", "--json", "--collection", "docs", "--from-production"])
        _, _, calls = self.post("config", {"name": "e", "embedding_model": "a/b", "chunk_size": 300, "rerank": False,
                                           "reader_model": "", "ocr": "smart"})
        args = calls[0][0]
        self.assertEqual(args[:3], ["config", "e", "--json"])
        for pair in (["--embedding-model", "a/b"], ["--chunk-size", "300"], ["--reader-model", "production"],
                     ["--ocr", "smart"]):
            self.assertIn(pair[0], args)
            self.assertEqual(args[args.index(pair[0]) + 1], pair[1])
        self.assertIn("--no-rerank", args)
        self.assertIn("--rerank", self.post("config", {"name": "e", "rerank": True})[2][0][0])
        _, _, calls = self.post("search", {"name": "e", "query": "q", "top_k": 4, "stages": ["bm25", "dense"],
                                           "retrieval_pool": 30, "rerank_pool": 20, "rrf_k": 50, "no_rerank": True})
        self.assertEqual(calls[0][0], ["search", "e", "q", "--top-k", "4", "--stages", "bm25,dense",
                                       "--retrieval-pool", "30", "--rerank-pool", "20", "--rrf-k", "50",
                                       "--no-rerank", "--json"])
        self.assertEqual(self.post("compare", {"name": "e"})[2][0][0], ["compare", "e", "--json"])
        self.assertEqual(self.post("preview", {"name": "e"})[2][0][0], ["promote", "e", "--dry-run", "--json"])
        self.assertEqual(self.post("promote", {"name": "e", "confirm": True})[2][0][0],
                         ["promote", "e", "--json", "--confirm"])
        self.assertEqual(self.post("settings", {"name": "e"})[2][0][0], ["settings", "e", "--json"])
        self.assertEqual(self.post("list", {})[2][0][0], ["list", "--json"])
        self.assertEqual(self.post("rm", {"name": "e", "confirm": True})[2][0][0], ["rm", "e", "--yes", "--json"])

    def test_bad_requests_are_400_or_404(self):
        self.assertEqual(self.post("rm", {"name": "e"})[0], 400)               # needs confirm: true
        self.assertEqual(self.post("search", {})[0], 400)                      # needs a name
        self.assertEqual(self.post("bogus", {"name": "e"})[0], 404)

    def test_index_bench_cancel_view_and_history_go_through_playground_runs(self):
        pr = ui_server.playground_runs
        with mock.patch.object(pr, "start", return_value={"id": "r1"}) as start:
            self.post("index", {"name": "e", "rebuild": True, "force_md": True})
            self.assertEqual(start.call_args.args[2:4], ("index", ["--rebuild", "--force-md"]))
            self.post("bench", {"name": "e", "k": 3, "label": "x", "stages": "bm25", "rrf_k": 10, "no_rerank": True})
            self.assertEqual(start.call_args.args[3],
                             ["-k", "3", "--label", "x", "--stages", "bm25", "--rrf-k", "10", "--no-rerank"])
        with mock.patch.object(pr, "cancel", return_value={"cancelled": True}) as cancel:
            st, js, _ = self.post("cancel", {"name": "e"})
        self.assertEqual((js["result"], cancel.call_args.args[1]), ({"cancelled": True}, "e"))
        with mock.patch.object(pr, "view", return_value={"status": "done"}) as view:
            self.post("run", {"name": "e", "run": "r1"})
        self.assertEqual(view.call_args.args[1:], ("e", "r1"))
        with mock.patch.object(pr, "history", return_value=[{"id": "r1"}]):
            self.assertEqual(self.post("runs", {"name": "e"})[1]["result"], [{"id": "r1"}])
        for name in ("start", "view", "history"):
            with mock.patch.object(pr, name, side_effect=ValueError("bad run id")):
                action = {"start": "index", "view": "run", "history": "runs"}[name]
                st, js, _ = self.post(action, {"name": "e"})
                self.assertEqual(st, 400, name)

    def test_the_child_process_failing_or_timing_out_is_an_error_reply_not_a_crash(self):
        handler = ui_server.Handler
        app = mock.Mock()
        app.paths.home = self.paths.home
        self_ = mock.Mock(app=app)
        with mock.patch.object(ui_server.subprocess, "run", side_effect=subprocess.TimeoutExpired("x", 5)):
            res = handler._playground_cli(self_, ["list"], timeout=5)
        self.assertEqual((res["ok"], "timed out after 5s" in res["error"]), (False, True))
        with mock.patch.object(ui_server.subprocess, "run",
                               return_value=subprocess.CompletedProcess([], 2, "not json", "error: no such experiment")):
            res = handler._playground_cli(self_, ["list"], timeout=5)
        self.assertEqual((res["ok"], res["error"]), (False, "error: no such experiment"))


class ModelRouteTests(UiBase):
    def post(self, action, body):
        st, js, _, _ = self.dash.req("POST", f"/api/models/{action}", body)
        return st, js

    def test_validation_and_busy_and_error_replies(self):
        self.assertEqual(self.post("reader", {})[0], 400)
        self.assertEqual(self.post("repair", {"model": "  "})[0], 400)
        self.assertEqual(self.post("switch", {"kind": "embedding"})[0], 400)
        self.assertEqual(self.post("download", {"models": ["not a model id"]})[0], 400)
        self.assertEqual(self.post("bogus", {})[0], 404)
        st, js = self.post("limit", {"gb": 12})
        self.assertEqual((st, js["memory_limit_gb"]), (200, 12.0))
        with mock.patch.object(model_tasks, "cancel", return_value=False):
            self.assertFalse(self.post("cancel", {})[1]["cancelled"])
        with mock.patch.object(model_tasks, "start_detached", side_effect=model_tasks.TaskBusy("one at a time")):
            self.assertEqual(self.post("verify", {})[0], 409)
        with mock.patch.object(model_tasks, "start_detached", side_effect=models.ModelError("no such model")):
            self.assertEqual(self.post("verify", {})[0], 400)

    def test_verify_targets_one_kind_or_both(self):
        with mock.patch.object(model_tasks, "start_detached", return_value={"id": "t"}) as sd:
            self.post("verify", {"kind": "reranker"})
            self.assertEqual([t[0] for t in sd.call_args.args[1]["targets"]], ["reranker"])
            self.post("verify", {})
            self.assertEqual([t[0] for t in sd.call_args.args[1]["targets"]], ["embedding", "reranker"])


class HelperTests(TempHome):
    def test_tail_file_returns_the_last_lines_and_nothing_for_a_missing_file(self):
        f = self.tmp / "log.txt"
        f.write_text("".join(f"line {i}\n" for i in range(1000)))
        self.assertEqual(ui_server.tail_file(f, 3), ["line 997", "line 998", "line 999"])
        self.assertLess(len(ui_server.tail_file(f, 5000, max_bytes=200)), 40)         # only the last bytes are read
        self.assertEqual(ui_server.tail_file(self.tmp / "none.txt", 3), [])

    def test_the_installed_version_is_empty_when_it_cannot_be_read(self):
        with mock.patch.dict(ui_server._INSTALLED, {"at": -1e9, "version": "x"}), \
                mock.patch.object(ui_server.importlib.metadata, "version", side_effect=Exception("no metadata")):
            self.assertEqual(ui_server.installed_version(), "")

    def test_only_the_newest_fifty_exports_stay_downloadable(self):
        app = ui_server.UiApp(self.paths, "token")
        ids = [app.remember_download(self.tmp / f"{i}.tgz") for i in range(55)]
        self.assertEqual(len(app.downloads), 50)
        self.assertNotIn(ids[0], app.downloads)
        self.assertIn(ids[-1], app.downloads)
        app.live.stop()

    def test_ping_tells_nothing_listening_from_something_else_listening(self):
        class Plain(http.server.BaseHTTPRequestHandler):
            def do_GET(self):                                      # noqa: N802
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"hello, not json")

            def log_message(self, *a):
                pass

        srv = http.server.HTTPServer(("127.0.0.1", 0), Plain)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.shutdown)
        self.addCleanup(srv.server_close)
        self.assertEqual(ui_server._ping(srv.server_address[1]), {})
        self.assertEqual(ui_server._probe(srv.server_address[1], self.paths), "other")
        free = http.server.HTTPServer(("127.0.0.1", 0), Plain)
        port = free.server_address[1]
        free.server_close()
        self.assertIsNone(ui_server._ping(port))
        self.assertEqual(ui_server._probe(port, self.paths), "")

    def test_stopping_a_dashboard_by_its_recorded_process_id(self):
        pid_file = ui_server.pid_file(self.paths)
        self.assertIn("no dashboard process id", ui_server.stop_running(self.paths))
        pid_file.write_text("garbage")
        self.assertIn("no dashboard process id", ui_server.stop_running(self.paths))
        pid_file.write_text("4242")
        with mock.patch.object(ui_server.os, "kill", side_effect=ProcessLookupError):
            self.assertEqual(ui_server.stop_running(self.paths), "the dashboard was not running")
        self.assertFalse(pid_file.exists())
        pid_file.write_text("4242")
        with mock.patch.object(ui_server.os, "kill", side_effect=PermissionError):
            self.assertEqual(ui_server.stop_running(self.paths), "cannot signal process 4242")
        calls = []

        def kill(pid, sig):
            calls.append(sig)
            if sig == 0:
                raise ProcessLookupError                        # gone after the first signal
        ui_server.port_file(self.paths).write_text("8765")
        with mock.patch.object(ui_server.os, "kill", side_effect=kill), mock.patch.object(ui_server.time, "sleep"):
            self.assertEqual(ui_server.stop_running(self.paths), "stopped the dashboard (pid 4242)")
        self.assertEqual((pid_file.exists(), ui_server.port_file(self.paths).exists()), (False, False))

    def test_run_can_print_the_url_or_stop_a_dashboard_or_say_there_is_none(self):
        with mock.patch.object(ui_server, "stop_running", return_value="stopped it"):
            self.assertEqual(ui_server.run(self.paths, stop=True), 0)
        with mock.patch.object(ui_server, "running_port", return_value=0), \
                mock.patch.object(ui_server, "_probe", return_value=""):
            self.assertEqual(ui_server.run(self.paths, port=8765, url_only=True), 1)
        with mock.patch.object(ui_server, "running_port", return_value=8123):
            self.assertEqual(ui_server.run(self.paths, url_only=True), 0)

    def test_run_refuses_a_port_that_something_else_uses_and_one_it_cannot_bind(self):
        with mock.patch.object(ui_server, "running_port", return_value=0), \
                mock.patch.object(ui_server, "_probe", return_value="other"):
            self.assertEqual(ui_server.run(self.paths, port=8765, open_browser=False), 1)
        with mock.patch.object(ui_server, "running_port", return_value=0), \
                mock.patch.object(ui_server, "_probe", return_value=""), \
                mock.patch.object(ui_server, "make_server", side_effect=OSError("address in use")):
            self.assertEqual(ui_server.run(self.paths, port=8765, open_browser=False), 1)

    def test_run_serves_until_interrupted_and_cleans_up_its_files(self):
        opened = []
        with mock.patch.object(ui_server, "_open", side_effect=opened.append), \
                mock.patch.object(ui_server.UiServer, "serve_forever", side_effect=KeyboardInterrupt):
            rc = ui_server.run(self.paths, port=0, open_browser=True)
        self.assertEqual((rc, len(opened)), (0, 1))
        self.assertFalse(ui_server.port_file(self.paths).exists())
        self.assertFalse(ui_server.pid_file(self.paths).exists())

    def test_run_replaces_a_dashboard_left_over_from_another_version_or_reuses_a_current_one(self):
        with mock.patch.object(ui_server, "running_port", return_value=8765), \
                mock.patch.object(ui_server, "_probe", return_value="ours"), \
                mock.patch.object(ui_server, "_ping", return_value={"version": ui_server.__version__}), \
                mock.patch.object(ui_server, "_open") as opened:
            self.assertEqual(ui_server.run(self.paths, port=8765, open_browser=True), 0)
        opened.assert_called_once()

    def test_detach_reports_a_dashboard_that_exits_at_once_or_never_comes_up(self):
        proc = mock.Mock(pid=77)
        proc.poll.return_value = 1
        with mock.patch.object(ui_server.subprocess, "Popen", return_value=proc), \
                mock.patch.object(ui_server.time, "sleep"):
            self.assertEqual(ui_server._detach(self.paths, 8765, False, False), 1)
        proc.poll.return_value = None
        with mock.patch.object(ui_server.subprocess, "Popen", return_value=proc), \
                mock.patch.object(ui_server.time, "sleep"), \
                mock.patch.object(ui_server, "_probe", return_value=""), \
                mock.patch.object(ui_server, "running_port", return_value=0):
            self.assertEqual(ui_server._detach(self.paths, 8765, True, False), 1)
        with mock.patch.object(ui_server.subprocess, "Popen", return_value=proc), \
                mock.patch.object(ui_server.time, "sleep"), \
                mock.patch.object(ui_server, "_probe", return_value="ours"), \
                mock.patch.object(ui_server, "_open") as opened:
            self.assertEqual(ui_server._detach(self.paths, 8765, False, True), 0)
        opened.assert_called_once()


if __name__ == "__main__":
    unittest.main()
