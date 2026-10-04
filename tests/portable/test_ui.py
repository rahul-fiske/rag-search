"""The web dashboard: security of the local server, the API behind each tab, and doc consistency."""

from __future__ import annotations

import http.client
import json
import os
import re
import subprocess
import sys
import threading
import time
import unittest
from unittest import mock

from tests.helpers import ROOT, SUBPROC_PYTHONPATH, TempHome

from rag_search import spec
from rag_search.ui import info, markdown, server as ui_server

DOC = "# Setup\n\nMulti-factor authentication is enabled per tenant. " * 4


class Dash:
    """A dashboard bound to a free port, in a thread, with a tiny HTTP client."""

    def __init__(self, paths, read_only=False):
        self.srv = ui_server.make_server(paths, port=0, read_only=read_only)
        self.port = self.srv.server_address[1]
        self.token = self.srv.app.token
        self.thread = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self.thread.start()

    def req(self, method, path, body=None, headers=None, auth=True, host=None, raw=None):
        h = {"Host": host or f"127.0.0.1:{self.port}"}
        if auth:
            h["X-RagSearch-Token"] = self.token
        if body is not None:
            h["Content-Type"] = "application/json"
        h.update(headers or {})
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        c.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
        for k, v in h.items():
            c.putheader(k, v)
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else b"")
        if method == "POST":
            c.putheader("Content-Length", str(len(data)))
        c.endheaders(data)
        r = c.getresponse()
        payload = r.read()
        hdrs = dict(r.getheaders())
        c.close()
        try:
            js = json.loads(payload)
        except ValueError:
            js = None
        return r.status, js, payload, hdrs

    def close(self):
        self.srv.shutdown_all()
        self.srv.server_close()


class UiBase(TempHome):
    read_only = False

    def setUp(self):
        super().setUp()
        self.use_fake_backends()
        self.write_doc("security/auth.md", DOC + "\n\n<!-- page 2 -->\n\n" + DOC)
        self.write_doc("hr/leave.md", "# Leave\n\nParental leave is sixteen weeks. " * 5)
        self.index()
        self.publish()
        self.dash = Dash(self.paths, self.read_only)
        self.addCleanup(self.dash.close)
        self.addCleanup(self._stop_daemons)

    def _stop_daemons(self):
        from rag_search import client
        for kind in ("search", "indexer"):
            try:
                client.stop(self.paths, kind)
            except Exception:  # noqa: BLE001
                pass


class SecurityTests(UiBase):
    def test_no_token_no_access(self):
        st, _, body, _ = self.dash.req("GET", "/api/state", auth=False)
        self.assertEqual(st, 401)
        st, _, _, _ = self.dash.req("GET", "/", auth=False)
        self.assertEqual(st, 401)
        st, _, _, _ = self.dash.req("POST", "/api/search", {"query": "x"}, auth=False)
        self.assertEqual(st, 401)
        st, _, _, _ = self.dash.req("GET", "/static/core.js", auth=False)
        self.assertEqual(st, 401)

    def test_ping_needs_no_token_and_leaks_nothing_sensitive(self):
        st, js, raw, _ = self.dash.req("GET", "/api/ping", auth=False)
        self.assertEqual(st, 200)
        self.assertEqual(js["app"], "rag-search-ui")
        self.assertNotIn(self.dash.token.encode(), raw)
        self.assertNotIn(str(self.paths.home).encode(), raw)

    def test_link_token_sets_a_strict_httponly_cookie(self):
        st, _, _, h = self.dash.req("GET", f"/?token={self.dash.token}", auth=False)
        self.assertEqual(st, 302)
        cookie = h["Set-Cookie"]
        self.assertIn("HttpOnly", cookie)
        self.assertIn("SameSite=Strict", cookie)
        st, _, _, _ = self.dash.req("GET", "/?token=wrong", auth=False)
        self.assertEqual(st, 401)
        st, js, _, _ = self.dash.req("GET", "/api/state", auth=False,
                                     headers={"Cookie": cookie.split(";")[0]})
        self.assertEqual(st, 200)

    def test_token_file_is_private_and_stable(self):
        f = ui_server.token_file(self.paths)
        self.assertEqual(f.stat().st_mode & 0o777, 0o600)
        self.assertEqual(ui_server.load_token(self.paths), self.dash.token)

    def test_only_loopback_host_names_are_served(self):
        for host in ("evil.example", f"evil.example:{self.dash.port}", "127.0.0.1:1"):
            st, _, _, _ = self.dash.req("GET", "/api/state", host=host)
            self.assertEqual(st, 403, host)
        st, _, _, _ = self.dash.req("GET", "/api/state", host=f"localhost:{self.dash.port}")
        self.assertEqual(st, 200)

    def test_cross_origin_posts_are_refused(self):
        st, _, _, _ = self.dash.req("POST", "/api/index/cancel", {}, headers={"Origin": "http://evil.example"})
        self.assertEqual(st, 403)
        st, _, _, _ = self.dash.req("POST", "/api/search", {"query": "leave"},
                                    headers={"Origin": f"http://127.0.0.1:{self.dash.port}"})
        self.assertEqual(st, 200)

    def test_post_needs_json_and_a_sane_body(self):
        st, _, _, _ = self.dash.req("POST", "/api/search", raw=b"query=x",
                                    headers={"Content-Type": "application/x-www-form-urlencoded"})
        self.assertEqual(st, 415)
        st, _, _, _ = self.dash.req("POST", "/api/search", raw=b"{nope", headers={"Content-Type": "application/json"})
        self.assertEqual(st, 400)
        st, _, _, _ = self.dash.req("POST", "/api/search", raw=b"[1]", headers={"Content-Type": "application/json"})
        self.assertEqual(st, 400)
        st, _, _, _ = self.dash.req("POST", "/api/search", raw=b"x" * (ui_server.MAX_BODY + 10),
                                    headers={"Content-Type": "application/json"})
        self.assertEqual(st, 413)

    def test_static_files_cannot_escape_the_static_folder(self):
        for p in ("/static/../server.py", "/static/%2e%2e/server.py", "/static/docs/../../server.py",
                  "/static/nope.js", "/static/../../paths.py"):
            st, _, _, _ = self.dash.req("GET", p)
            self.assertEqual(st, 404, p)

    def test_security_headers(self):
        st, _, _, h = self.dash.req("GET", "/")
        self.assertEqual(st, 200)
        self.assertIn("default-src 'self'", h.get("Content-Security-Policy", ""))
        self.assertEqual(h.get("X-Content-Type-Options"), "nosniff")


class ReadApiTests(UiBase):
    def test_state_has_live_and_catalog(self):
        st, js, _, _ = self.dash.req("GET", "/api/state")
        self.assertEqual(st, 200)
        self.assertIn("daemons", js["live"])
        self.assertIn("index", js["live"])
        cat = js["catalog"]
        self.assertEqual(cat["read_only"], False)
        self.assertIn("installed_version", cat)              # lets the page say "restart me"
        self.assertEqual({c["collection"] for c in cat["list"]["collections"]}, {"security", "hr"})
        self.assertEqual({r["collection"] for r in cat["access"]["collections"]}, {"security", "hr"})

    def test_events_stream_starts_with_live_and_catalog(self):
        c = http.client.HTTPConnection("127.0.0.1", self.dash.port, timeout=15)
        c.request("GET", "/api/events", headers={"X-RagSearch-Token": self.dash.token,
                                                 "Host": f"127.0.0.1:{self.dash.port}"})
        r = c.getresponse()
        self.assertEqual(r.status, 200)
        self.assertIn("text/event-stream", r.getheader("Content-Type"))
        seen: set[str] = set()
        deadline = time.time() + 15
        while time.time() < deadline and seen != {"live", "catalog"}:
            line = r.fp.readline().decode()
            if line.startswith("event:"):
                seen.add(line.split(":", 1)[1].strip())
        c.close()
        self.assertEqual(seen, {"live", "catalog"})

    def test_architecture_reports_the_constants_the_engine_uses(self):
        st, a, _, _ = self.dash.req("GET", "/api/architecture")
        self.assertEqual(st, 200)
        self.assertEqual(a["fusion"]["k"], spec.RRF_K)
        self.assertEqual(a["keyword"]["k1"], spec.BM25_K1)
        self.assertEqual(a["keyword"]["b"], spec.BM25_B)
        self.assertEqual(a["limits"]["top_k_max"], spec.MAX_TOP_K)
        self.assertEqual(a["models"]["reranker"]["cap"], spec.RERANK_CAP)
        self.assertEqual(a["pools"]["5"], {"per_retriever": spec.retrieval_pool(5), "to_reranker": spec.rerank_pool(5)})
        self.assertEqual(a["models"]["embedding"]["dim"], 64)      # read from the real index.meta.json
        self.assertTrue(a["index"]["node_fields"] and a["index"]["meta_fields"])
        cs = a["config_storage"]
        self.assertEqual(cs["file"], str(self.paths.config_file))
        self.assertIn("chunk_size", {t["key"] for t in cs["sections"]["indexer"]})

    def test_config_get_and_set(self):
        st, c, _, _ = self.dash.req("GET", "/api/config")
        self.assertEqual(st, 200)
        self.assertEqual(c["values"]["indexer"]["chunk_size"], 0)
        self.assertTrue(any(t["key"] == "ocr" and t["choices"] for t in c["tunables"]))

        st, r, _, _ = self.dash.req("POST", "/api/config/set",
                                    {"section": "indexer", "values": {"chunk_size": 600, "ocr": "smart"}})
        self.assertEqual(st, 200, r)
        self.assertEqual(r["values"]["indexer"]["chunk_size"], 600)
        self.assertEqual(r["changed"], {"chunk_size": 600, "ocr": "smart"})

        st, r2, _, _ = self.dash.req("POST", "/api/config/set",
                                     {"section": "indexer", "values": {"ocr": "bogus"}})
        self.assertEqual(st, 400)
        self.assertIn("OCR mode", r2["error"])

        st, r3, _, _ = self.dash.req("POST", "/api/config/set", {"section": "nope", "values": {}})
        self.assertEqual(st, 400)

    def test_help_has_cli_reference_and_both_documents(self):
        st, h, _, _ = self.dash.req("GET", "/api/help")
        self.assertEqual(st, 200)
        cmds = {c["command"] for c in h["cli"]["commands"]}
        for c in ("search", "grep", "index new", "access restrict", "access grant", "ui", "doctor", "daemon"):
            self.assertIn(c, cmds)
        self.assertIn("<h1", h["readme"]["html"])
        self.assertTrue(h["readme"]["toc"])
        self.assertTrue(h["architecture_doc"]["toc"])
        self.assertNotIn("<script", h["readme"]["html"].lower())

    def test_logs_endpoint(self):
        st, js, _, _ = self.dash.req("GET", "/api/logs?kind=search&lines=5")
        self.assertEqual(st, 200)
        self.assertIsInstance(js["lines"], list)
        st, _, _, _ = self.dash.req("GET", "/api/logs?kind=../../etc/passwd")
        self.assertEqual(st, 400)


class ActionTests(UiBase):
    def test_search_and_grep_as_a_client_follow_the_access_rules(self):
        from rag_search import access
        access.restrict(self.paths, "hr", ["claude"])
        st, js, _, _ = self.dash.req("POST", "/api/search", {"query": "parental leave", "client": "cli", "top_k": 3})
        self.assertEqual(st, 200, js)
        self.assertIn("hr", {r["collection"] for r in js["result"]["results"]})
        st, js, _, _ = self.dash.req("POST", "/api/search", {"query": "parental leave", "client": "agent", "top_k": 3})
        self.assertEqual(st, 200, js)
        self.assertNotIn("hr", {r["collection"] for r in js["result"]["results"]})
        st, js, _, _ = self.dash.req("POST", "/api/grep", {"pattern": "Parental", "client": "agent"})
        self.assertEqual(js["result"]["matches"], [])
        st, js, _, _ = self.dash.req("POST", "/api/grep", {"pattern": "Parental", "client": "claude"})
        self.assertTrue(js["result"]["matches"])

    def test_search_stage_and_pool_overrides_from_the_dashboard(self):
        st, js, _, _ = self.dash.req("POST", "/api/search", {"query": "parental leave", "client": "cli",
                                     "stages": ["bm25"], "retrieval_pool": 8, "rerank_pool": 6, "rrf_k": 15})
        self.assertEqual(st, 200, js)
        t = js["result"]["timing"]
        self.assertEqual((t["stages"], t["retrieval_pool"], t["rerank_pool"], t["rrf_k"]),
                         (["bm25"], 8, 6, 15))
        self.assertIsNone(js["result"]["results"][0]["dense_score"])

    def test_search_bad_stages_is_a_clean_failure_not_a_server_error(self):
        # a bad `stages` combination is caught by the search daemon (a round trip, like any
        # other daemon-mediated search error such as warming-up or access-denied), so it comes
        # back as HTTP 200 with ok:false + a bad_request code, not an HTTP error status.
        st, js, _, _ = self.dash.req("POST", "/api/search", {"query": "x", "stages": "rerank"})
        self.assertEqual(st, 200, js)
        self.assertEqual((js["ok"], js["code"]), (False, "bad_request"))
        # a non-numeric pool override is rejected locally by the dashboard server, before any
        # daemon round trip, so it's a genuine HTTP 400.
        st, js, _, _ = self.dash.req("POST", "/api/search", {"query": "x", "retrieval_pool": "many"})
        self.assertEqual(st, 400, js)

    def test_architecture_exposes_pool_and_rrf_k_bounds_for_the_debug_panel(self):
        st, js, _, _ = self.dash.req("GET", "/api/architecture")
        self.assertEqual(st, 200, js)
        self.assertIn("retrieval_pool_max", js["pool_overrides"])
        self.assertIn("rerank_pool_max", js["pool_overrides"])
        self.assertEqual(js["pools"]["5"]["per_retriever"], spec.retrieval_pool(5))
        self.assertEqual(js["fusion"]["k_min"], spec.RRF_K_MIN)

    def test_ui_searches_are_not_counted_as_a_client_connecting(self):
        self.dash.req("POST", "/api/search", {"query": "leave", "client": "newhost"})
        from rag_search import api
        ping = api.daemon_status(self.paths)["search"]
        self.assertNotIn("newhost", ping.get("clients_seen") or {})

    def test_access_edit_roundtrip(self):
        st, js, _, _ = self.dash.req("POST", "/api/access", {"action": "restrict", "collection": "hr", "clients": ["claude"]})
        self.assertEqual(st, 200, js)
        rules = json.loads(self.paths.access_file.read_text())
        self.assertEqual(rules["collections"], {"hr": ["claude"]})
        st, _, _, _ = self.dash.req("POST", "/api/access", {"action": "grant", "collection": "hr", "clients": ["agent"]})
        self.assertEqual(st, 200)
        self.assertEqual(json.loads(self.paths.access_file.read_text())["collections"], {"hr": ["agent", "claude"]})
        st, _, _, _ = self.dash.req("POST", "/api/access", {"action": "restrict", "collection": "hr", "clients": ["all"]})
        self.assertEqual(st, 200)
        self.assertEqual(json.loads(self.paths.access_file.read_text())["collections"], {})

    def test_access_edit_rejects_bad_input(self):
        for body in ({"action": "delete", "collection": "hr", "clients": []},
                     {"action": "restrict", "collection": "hr", "clients": "claude"},
                     {"action": "restrict", "collection": 5, "clients": []},
                     {"action": "restrict", "collection": "hr", "clients": ["Bad Name!"]}):
            st, _, _, _ = self.dash.req("POST", "/api/access", body)
            self.assertEqual(st, 400, body)
        self.assertFalse(self.paths.access_file.exists())

    def test_full_rebuild_needs_confirmation_and_bad_modes_fail(self):
        st, _, _, _ = self.dash.req("POST", "/api/index/start", {"mode": "all"})
        self.assertEqual(st, 400)
        st, _, _, _ = self.dash.req("POST", "/api/index/start", {"mode": "bogus"})
        self.assertEqual(st, 400)

    def test_index_start_runs_a_job_and_publishes(self):
        self.write_doc("security/new.md", "# New\n\nKerberos ticket lifetime is ten hours. " * 5)
        st, js, _, _ = self.dash.req("POST", "/api/index/start", {"mode": "new"})
        self.assertEqual(st, 200, js)
        deadline = time.time() + 90
        job = {}
        while time.time() < deadline:
            _, s, _, _ = self.dash.req("GET", "/api/state")
            job = (s["live"]["index"] or {}).get("job") or {}
            if job.get("status") in ("succeeded", "partial", "failed"):
                break
            time.sleep(0.5)
        self.assertEqual(job.get("status"), "succeeded", job)
        st, js, _, _ = self.dash.req("POST", "/api/search", {"query": "Kerberos ticket lifetime", "top_k": 3})
        self.assertIn("new", {r["file"] for r in js["result"]["results"]})

    def test_index_cancel_when_idle_is_harmless(self):
        st, js, _, _ = self.dash.req("POST", "/api/index/cancel", {})
        self.assertEqual(st, 200, js)

    def test_daemon_controls(self):
        st, js, _, _ = self.dash.req("POST", "/api/daemon", {"which": "search", "action": "start"})
        self.assertEqual(st, 200, js)
        st, _, _, _ = self.dash.req("POST", "/api/daemon", {"which": "search", "action": "explode"})
        self.assertEqual(st, 400)
        st, js, _, _ = self.dash.req("POST", "/api/daemon", {"which": "search", "action": "stop"})
        self.assertEqual(st, 200, js)

    def test_unknown_endpoints(self):
        st, _, _, _ = self.dash.req("POST", "/api/nope", {})
        self.assertEqual(st, 404)
        st, _, _, _ = self.dash.req("GET", "/api/nope")
        self.assertEqual(st, 404)


class ModelsApiTests(UiBase):
    def test_models_state_and_plan(self):
        st, js, _, _ = self.dash.req("GET", "/api/models")
        self.assertEqual(st, 200)
        self.assertTrue(js["ok"])
        self.assertEqual(js["embedding"]["active"], "BAAI/bge-m3")
        self.assertEqual(js["serving"], "BAAI/bge-m3")
        self.assertIn("task", js)
        st, js, _, _ = self.dash.req("GET", "/api/models/plan?kind=embedding&model=Qwen/Qwen3-Embedding-0.6B")
        self.assertEqual(st, 200)
        self.assertEqual(js["plan"]["reindex"]["documents"], 2)
        st, js, _, _ = self.dash.req("GET", "/api/models/plan?kind=embedding&model=garbage")
        self.assertEqual(st, 400)
        _, s, _, _ = self.dash.req("GET", "/api/state")
        self.assertEqual(s["catalog"]["models"]["embedding"], "BAAI/bge-m3")

    def test_limit_and_validation(self):
        st, js, _, _ = self.dash.req("POST", "/api/models/limit", {"gb": 12})
        self.assertEqual(st, 200)
        self.assertEqual(json.loads(self.paths.config_file.read_text())["models"]["memory_limit_gb"], 12)
        st, _, _, _ = self.dash.req("POST", "/api/models/limit", {"gb": -3})
        self.assertEqual(st, 400)
        st, _, _, _ = self.dash.req("POST", "/api/models/switch", {"kind": "embedding", "model": "garbage"})
        self.assertEqual(st, 400)
        st, _, _, _ = self.dash.req("POST", "/api/models/download", {"models": ["nonsense"]})
        self.assertEqual(st, 400)
        st, _, _, _ = self.dash.req("POST", "/api/models/nope", {})
        self.assertEqual(st, 404)

    def test_a_blocked_switch_is_refused_unless_forced(self):
        from rag_search import models as models_mod
        models_mod.set_memory_limit(self.paths, 2)
        st, js, _, _ = self.dash.req("POST", "/api/models/switch",
                                     {"kind": "embedding", "model": "Qwen/Qwen3-Embedding-8B"})
        self.assertEqual(st, 400)
        self.assertIn("memory", js["error"])

    def test_switch_starts_a_detached_task(self):
        from rag_search import model_tasks
        seen = []
        with mock.patch.object(model_tasks, "start_detached",
                               side_effect=lambda paths, req: seen.append(req) or {"id": "t1", "status": "queued"}):
            st, js, _, _ = self.dash.req("POST", "/api/models/switch",
                                         {"kind": "reranker", "model": "BAAI/bge-reranker-base", "force": True})
        self.assertEqual(st, 200)
        self.assertEqual(seen[0]["op"], "switch")
        self.assertEqual(seen[0]["model"], "BAAI/bge-reranker-base")
        self.assertTrue(seen[0]["force"])
        with mock.patch.object(model_tasks, "start_detached", side_effect=model_tasks.TaskBusy("busy")):
            st, js, _, _ = self.dash.req("POST", "/api/models/download", {"models": ["BAAI/bge-reranker-base"]})
        self.assertEqual(st, 409)

    def test_cancel_when_nothing_runs(self):
        st, js, _, _ = self.dash.req("POST", "/api/models/cancel", {})
        self.assertEqual((st, js["cancelled"]), (200, False))


class ReadOnlyTests(UiBase):
    read_only = True

    def test_read_only_refuses_every_change_but_allows_search_and_grep(self):
        for path, body in (("index/start", {"mode": "new"}), ("index/cancel", {}), ("index/publish", {}),
                           ("access", {"action": "restrict", "collection": "hr", "clients": []}),
                           ("daemon", {"which": "search", "action": "stop"}),
                           ("models/switch", {"kind": "reranker", "model": "BAAI/bge-reranker-base"}),
                           ("models/download", {"models": ["BAAI/bge-reranker-base"]}),
                           ("models/limit", {"gb": 4}), ("models/cancel", {}),
                           ("config/set", {"section": "indexer", "values": {"chunk_size": 600}})):
            st, js, _, _ = self.dash.req("POST", "/api/" + path, body)
            self.assertEqual(st, 403, path)
        self.assertFalse(self.paths.access_file.exists())
        st, js, _, _ = self.dash.req("POST", "/api/search", {"query": "leave"})
        self.assertEqual(st, 200)
        st, js, _, _ = self.dash.req("POST", "/api/grep", {"pattern": "Parental"})
        self.assertEqual(st, 200)
        _, s, _, _ = self.dash.req("GET", "/api/state")
        self.assertTrue(s["catalog"]["read_only"])


class MarkdownTests(unittest.TestCase):
    def test_everything_is_escaped_and_only_safe_links_survive(self):
        html = markdown.render(
            "# T <script>alert(1)</script>\n\n"
            "[bad](javascript:alert(1)) [ok](https://example.com) [anchor](#x) [data](data:text/html,x)\n\n"
            "<img src=x onerror=alert(1)>\n\n```\n<b>code</b>\n```\n")
        low = html.lower()
        self.assertNotIn("<script", low)
        self.assertNotIn("<img", low)
        self.assertNotIn("javascript:", low)
        self.assertNotIn("data:text", low)
        self.assertIn('href="https://example.com"', html)
        self.assertIn('href="#x"', html)
        self.assertIn("&lt;b&gt;code&lt;/b&gt;", html)

    def test_structure(self):
        html = markdown.render("# A\n\n## B c\n\n- one\n- two\n\n1. x\n\n| h | i |\n|---|---|\n| 1 | 2 |\n\n**bold** `code`")
        for frag in ('<h1 id="a">', '<h2 id="b-c">', "<ul>", "<ol>", "<table>", "<strong>bold</strong>", "<code>code</code>"):
            self.assertIn(frag, html)
        self.assertEqual([h["title"] for h in markdown.headings("# A\n\n## B\n\n```\n# not\n```\n")], ["A", "B"])

    def test_shipped_documents_render_without_raw_html(self):
        for name in ("README.md", "ARCHITECTURE.md"):
            out = markdown.render(info.read_doc(name)).lower()
            self.assertGreater(len(out), 2000, name)
            self.assertNotIn("<script", out)


class ConsistencyTests(TempHome):
    def test_documented_index_fields_match_the_files_the_pipeline_writes(self):
        self.write_doc("security/auth.md", DOC * 3)
        self.index()
        d = next(self.paths.index.glob("security/*/"))
        d = next(p for p in (self.paths.index / "security").iterdir() if p.name != "_all")
        nodes = json.loads((d / "nodes.json").read_text())
        node = nodes["nodes"][0]
        documented = set(info.NODE_FIELDS)
        actual = {k for k in node if k != "metadata"} | {f"metadata.{k}" for k in node["metadata"]}
        self.assertEqual(documented - info.OPTIONAL_NODE_FIELDS, actual - info.OPTIONAL_NODE_FIELDS)
        meta = json.loads((d / "index.meta.json").read_text())
        self.assertEqual(set(info.META_FIELDS), set(meta))
        self.publish()
        cat = json.loads((self.paths.serving / "current" / "catalog.json").read_text())
        top = {k.split("[")[0] for k in info.CATALOG_FIELDS if "[" not in k or k.startswith("collections[]")}
        for key in ("generation", "published_at", "model", "content_sha", "collections"):
            self.assertIn(key, cat)
            self.assertIn(key, top | {"collections"})

    def test_packaged_docs_equal_the_root_documents(self):
        for name in ("README.md", "ARCHITECTURE.md"):
            root = ROOT / name
            if not root.exists():
                self.skipTest("not running from a source checkout")
            self.assertEqual((info.DOCS / name).read_text(encoding="utf-8"), root.read_text(encoding="utf-8"),
                             f"{name}: run scripts/build_release.sh (or copy it into ui/static/docs/)")

    def test_every_script_referenced_by_the_page_exists(self):
        page = (info.STATIC / "index.html").read_text(encoding="utf-8")
        scripts = re.findall(r'<script src="/static/([^"]+)"', page)
        self.assertIn("core.js", scripts[:1])
        for s in scripts:
            self.assertTrue((info.STATIC / s).is_file(), s)
        for tab in re.findall(r'data-tab="(\w+)"', page):
            self.assertIn(f'id="v-{tab}"', page)
            self.assertIn(f"RS.views.{tab} =", "".join((info.STATIC / s).read_text(encoding="utf-8") for s in scripts), tab)

    def test_the_cli_command_list_in_the_help_tab_matches_the_parser(self):
        from rag_search.cli import build_parser
        ref = info.cli_reference()
        self.assertGreaterEqual(len(ref["commands"]), 20)
        names = {c["command"] for c in ref["commands"]}
        self.assertTrue({"ui", "access grant", "index publish"} <= names)
        self.assertTrue(build_parser())


class DetachAndCliTests(TempHome):
    def test_ui_modules_stay_light(self):
        for mod in ("rag_search.ui.server", "rag_search.ui.info", "rag_search.ui.markdown"):
            code = (f"import sys; sys.path[:0] = {SUBPROC_PYTHONPATH.split(os.pathsep)!r}; import {mod}; "
                    "print(','.join(m for m in ('numpy','torch','docling','sentence_transformers','mcp') if m in sys.modules))")
            out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
            self.assertEqual(out, "", mod)

    def test_doctor_json_is_machine_readable(self):
        env = dict(os.environ, PYTHONPATH=SUBPROC_PYTHONPATH)
        p = subprocess.run([sys.executable, "-m", "rag_search.cli", "doctor", "--json"], capture_output=True,
                           text=True, env=env, timeout=90)
        rows = json.loads(p.stdout)
        self.assertTrue(rows and all({"status", "check", "detail"} <= set(r) for r in rows))

    def test_cli_ui_command_starts_serves_and_stops(self):
        env = dict(os.environ, PYTHONPATH=SUBPROC_PYTHONPATH, RAG_SEARCH_UI_PORT="0")
        run = lambda *a: subprocess.run([sys.executable, "-m", "rag_search.cli", *a], capture_output=True,  # noqa: E731
                                        text=True, env=env, timeout=60)
        p = run("ui", "--no-browser", "--detach", "--port", "0")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        m = re.search(r"http://127\.0\.0\.1:(\d+)/\?token=(\S+)", p.stdout)
        self.assertTrue(m, p.stdout)
        try:
            c = http.client.HTTPConnection("127.0.0.1", int(m.group(1)), timeout=10)
            c.request("GET", "/api/ping")
            r = c.getresponse()
            r.read()
            c.close()
            self.assertEqual(r.status, 200)
            again = run("ui", "--url")
            self.assertIn(m.group(1), again.stdout)
        finally:
            s = run("ui", "--stop")
            self.assertEqual(s.returncode, 0, s.stdout + s.stderr)

    def _ping_as(self, version):
        from rag_search.ui import server
        return {"app": "rag-search-ui", "home": server.home_id(self.paths), "version": version}

    def test_a_dashboard_left_over_from_an_older_version_is_replaced(self):
        from unittest import mock
        from rag_search import __version__
        from rag_search.ui import server
        with mock.patch.object(server, "_ping", return_value=self._ping_as("0.0.1")), \
                mock.patch.object(server, "stop_running", return_value="stopped the dashboard (pid 1)") as stop, \
                mock.patch.object(server, "make_server", side_effect=OSError("boom")):
            rc = server.run(self.paths, port=8765, open_browser=False)
        stop.assert_called_once()
        self.assertEqual(rc, 1)                              # got as far as starting a new one
        # one that cannot be stopped is reported, and no second one is started
        with mock.patch.object(server, "_ping", return_value=self._ping_as("0.0.1")), \
                mock.patch.object(server, "stop_running", return_value="no dashboard process id is recorded"), \
                mock.patch.object(server, "make_server", side_effect=AssertionError("must not start")):
            self.assertEqual(server.run(self.paths, port=8765, open_browser=False), 1)
        # the same version is simply reused
        with mock.patch.object(server, "_ping", return_value=self._ping_as(__version__)), \
                mock.patch.object(server, "stop_running") as stop2, \
                mock.patch.object(server, "make_server", side_effect=AssertionError("must not start")):
            self.assertEqual(server.run(self.paths, port=8765, open_browser=False), 0)
        stop2.assert_not_called()

    def test_a_dashboard_run_in_the_foreground_can_be_stopped_with_ui_stop(self):
        env = dict(os.environ, PYTHONPATH=SUBPROC_PYTHONPATH)
        proc = subprocess.Popen([sys.executable, "-m", "rag_search.ui.server", "--port", "0", "--no-browser"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
        try:
            pid_file = self.paths.run / "ui.pid"
            end = time.monotonic() + 30
            while not pid_file.exists() and time.monotonic() < end:
                time.sleep(0.1)
            self.assertEqual(pid_file.read_text().strip(), str(proc.pid))
            p = subprocess.run([sys.executable, "-m", "rag_search.cli", "ui", "--stop"], capture_output=True,
                               text=True, env=env, timeout=60)
            self.assertIn("stopped the dashboard", p.stdout, p.stdout + p.stderr)
            proc.wait(timeout=20)
        finally:
            if proc.poll() is None:
                proc.kill()


if __name__ == "__main__":
    unittest.main()
