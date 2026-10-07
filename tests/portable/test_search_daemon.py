import json
import os
import threading
import time
import unittest

from tests.helpers import FakeEmbedder, FakeReranker, TempHome
from rag_search import client, protocol
from rag_search.core.search import SearchEngine
from rag_search.core.search_daemon import SearchDaemon

AUTH = ("# Authentication\n\n<!-- page 1 -->\n\nSSH access uses public key authentication. "
        "Administrators create accounts.\n\n<!-- page 2 -->\n\n## Roles\n\n"
        "Role based access control limits what each account can do.")
COOK = "# Cooking\n\n<!-- page 1 -->\n\nSourdough bread needs a starter, flour, water and salt."
DIARY = "# Diary\n\n<!-- page 1 -->\n\nSecret bread thoughts."


class SlowEngine(SearchEngine):
    """Model loading blocks until released (to observe the warm-up state)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.release = threading.Event()

    def load_models(self):
        self.release.wait(10)
        super().load_models()


class FlakyEngine(SearchEngine):
    """First model load fails (e.g. no network for the download), later ones work."""

    attempts = 0

    def load_models(self):
        FlakyEngine.attempts += 1
        if FlakyEngine.attempts == 1:
            raise RuntimeError("download failed")
        super().load_models()


class BrokenEngine(SearchEngine):
    def load_models(self):
        raise RuntimeError("no weights")


class DaemonCase(TempHome):
    def start_daemon(self, engine=None, retry=None, **kw):
        engine = engine or SearchEngine(self.paths, embedder=FakeEmbedder(),
                                        reranker=FakeReranker())
        d = SearchDaemon(self.paths, engine=engine, **kw)
        d.idle_poll, d.accept_timeout, d.watch_seconds = 0.2, 0.1, 0.2
        if retry is not None:
            d.retry_seconds = retry
        t = threading.Thread(target=d.run, daemon=True)
        t.start()
        self.addCleanup(self.stop_daemon, d, t)
        end = time.monotonic() + 5
        while time.monotonic() < end and client.ping(self.paths, "search", 0.5) is None:
            time.sleep(0.05)
        self.assertIsNotNone(client.ping(self.paths, "search"), "daemon did not come up")
        return d

    def stop_daemon(self, d, t):
        d.stop.set()
        t.join(5)

    def call(self, action, cli="cli", **f):
        return client.roundtrip(self.paths, "search",
                                protocol.make_request(action, cli, **f), 10)

    def wait_ready(self, d):
        self.assertTrue(d.ready.wait(5))

    def build(self, **docs):
        for rel, text in docs.items():
            self.write_doc(rel.replace("__", "/") + ".md", text)
        self.index()
        self.publish()


class SearchDaemonTests(DaemonCase):
    def test_ping_search_list_grep(self):
        self.write_doc("security/auth.md", AUTH)
        self.write_doc("kitchen/bread.md", COOK)
        self.index()
        self.publish()
        d = self.start_daemon()
        self.wait_ready(d)
        p = client.ping(self.paths, "search")
        self.assertEqual((p["role"], p["state"], p["generation"], p["collections"]),
                         ("search", "ready", 1, 2))
        r = self.call("search", query="role based access control", top_k=3)
        self.assertTrue(r["ok"])
        top = r["result"]["results"][0]
        self.assertEqual((top["file"], top["page"], top["collection"]), ("auth", "2", "security"))
        self.assertTrue(self.call("list")["result"])
        g = self.call("grep", pattern="Sourdough")
        self.assertTrue(g["ok"] and g["result"]["matches"], g)

    def test_hot_reload_and_watcher(self):
        self.build(c__a="# T\n\n<!-- page 1 -->\n\nalpha")
        d = self.start_daemon()
        self.wait_ready(d)
        self.assertEqual(self.call("search", query="zulu")["result"]["results"][0]["file"], "a")
        self.write_doc("c/b.md", "# T\n\n<!-- page 1 -->\n\nzulu unique")
        self.index()
        self.publish()
        rl = self.call("reload")
        self.assertEqual((rl["ok"], rl["changed"], rl["generation"]), (True, True, 2))
        self.assertEqual(self.call("search", query="zulu")["result"]["results"][0]["file"], "b")
        again = self.call("reload")
        self.assertFalse(again["changed"])
        # a further publish without any reload message is noticed by the watcher
        self.write_doc("c/c.md", "# T\n\n<!-- page 1 -->\n\nyankee unique")
        self.index()
        self.publish()
        end = time.monotonic() + 5
        while time.monotonic() < end and d.engine.generation != 3:
            time.sleep(0.1)
        self.assertEqual(d.engine.generation, 3)

    def test_reload_reports_which_collections_were_reused(self):
        self.build(security__auth=AUTH, kitchen__bread=COOK)
        d = self.start_daemon()
        self.wait_ready(d)
        self.write_doc("kitchen/rye.md", "# R\n\n<!-- page 1 -->\n\ncaraway rye")
        self.index()
        self.publish()
        r = self.call("reload")
        self.assertEqual((r["reused"], r["loaded"]), (["security"], ["kitchen"]))
        self.assertEqual(self.call("search", query="caraway")["result"]["results"][0]["file"],
                         "rye")

    def test_access_rules_per_client_apply_live(self):
        from rag_search import policy
        self.build(kitchen__bread=COOK, personal__diary=DIARY)
        d = self.start_daemon()
        self.wait_ready(d)

        def names(cli):
            return [c["collection"] for c in self.call("list", cli)["result"]["collections"]]

        # open by default: everybody sees everything
        self.assertEqual(sorted(names("agent")), ["kitchen", "personal"])
        hit = self.call("search", "agent", query="secret bread", collections="personal")
        self.assertTrue(hit["ok"] and hit["result"]["results"])

        policy.save_rules(self.paths, {"personal": ["claude"]})   # no daemon restart needed
        self.assertEqual(names("agent"), ["kitchen"])
        self.assertEqual(sorted(names("claude")), ["kitchen", "personal"])
        self.assertEqual(sorted(names("cli")), ["kitchen", "personal"])
        r = self.call("search", "agent", query="secret bread thoughts")
        self.assertTrue(all(x["collection"] != "personal" for x in r["result"]["results"]))
        ok = self.call("search", "claude", query="secret bread", collections="personal")
        self.assertTrue(ok["ok"] and ok["result"]["results"])
        bad = self.call("search", "agent", query="secret bread", collections="personal")
        self.assertEqual((bad["ok"], bad["code"]), (False, protocol.BAD_REQUEST))
        self.assertIn("unknown collection", bad["error"])
        badg = self.call("grep", "agent", pattern="Secret", collections="personal")
        self.assertEqual(badg["code"], protocol.BAD_REQUEST)
        self.assertEqual(self.call("grep", "agent", pattern="Secret")["result"]["matches"], [])
        self.assertTrue(self.call("grep", "claude", pattern="Secret")["result"]["matches"])
        # an unlisted (new) client is treated like any other client: only open collections
        self.assertEqual(names("some-future-host"), ["kitchen"])

        policy.save_rules(self.paths, {})                          # opened again
        self.assertEqual(sorted(names("agent")), ["kitchen", "personal"])
        info = client.ping(self.paths, "search")
        self.assertEqual(sorted(info["clients_seen"]), ["agent", "claude", "cli", "some-future-host"])

    def test_protocol_and_bad_requests(self):
        d = self.start_daemon()
        self.wait_ready(d)
        old = self.call("ping", v=99)
        self.assertEqual(old["code"], protocol.PROTOCOL_MISMATCH)
        self.assertEqual(self.call("nope")["code"], protocol.BAD_REQUEST)
        raw = client.roundtrip(self.paths, "search", {"no": "action"}, 5)
        self.assertEqual(raw["code"], protocol.BAD_REQUEST)

    def test_list_and_grep_work_while_models_load(self):
        self.build(kitchen__bread=COOK)
        eng = SlowEngine(self.paths, embedder=FakeEmbedder(), reranker=FakeReranker())
        self.addCleanup(eng.release.set)
        d = self.start_daemon(engine=eng)
        p = client.ping(self.paths, "search")
        self.assertEqual(p["state"], "loading_models")
        self.assertTrue(self.call("list")["ok"])
        self.assertTrue(self.call("grep", pattern="Sourdough")["result"]["matches"])
        r = self.call("search", query="bread", wait_s=0.3)
        self.assertEqual((r["ok"], r["code"]), (False, protocol.WARMING_UP))
        eng.release.set()
        self.wait_ready(d)
        self.assertTrue(self.call("search", query="bread")["ok"])

    def test_status_reports_warming_up_then_warm_with_timing_and_memory(self):
        self.build(kitchen__bread=COOK, security__auth=AUTH)
        eng = SlowEngine(self.paths, embedder=FakeEmbedder(), reranker=FakeReranker())
        self.addCleanup(eng.release.set)
        d = self.start_daemon(engine=eng)
        p = client.ping(self.paths, "search")
        self.assertFalse(p["warm"])
        self.assertEqual((p["warmup"]["status"], p["warmup"]["phase"]), ("warming_up", "loading_models"))
        self.assertIn("elapsed_s", p["warmup"])
        time.sleep(0.3)
        eng.release.set()
        self.wait_ready(d)
        p = client.ping(self.paths, "search")
        self.assertTrue(p["warm"])
        w = p["warmup"]
        self.assertEqual(w["status"], "warm")
        self.assertGreaterEqual(w["total_s"], 0.3)                   # includes the blocked model load
        self.assertGreaterEqual(w["total_s"], w["models_s"])
        self.assertIn("index_s", w)
        self.assertGreater(w["ready_at"], w["started_at"])
        m = p["memory"]
        self.assertGreater(m["rss_bytes"], 0)
        self.assertGreaterEqual(m["peak_rss_bytes"], m["rss_bytes"])
        self.assertGreater(m["embeddings_bytes"], 0)
        self.assertGreater(m["text_bytes"], 0)
        self.assertEqual(set(m["collections"]), {"kitchen", "security"})
        self.assertEqual(p["last_reload"]["generation"], 1)

    def test_search_and_grep_carry_timing(self):
        self.build(security__auth=AUTH)
        d = self.start_daemon()
        self.wait_ready(d)
        t = self.call("search", query="public key")["result"]["timing"]
        for key in ("total_ms", "embed_query_ms", "keyword_ms", "dense_ms", "retrieve_ms",
                    "rerank_ms", "queue_ms", "server_ms", "candidates"):
            self.assertIn(key, t)
        self.assertTrue(t["reranked"])
        self.assertGreaterEqual(t["server_ms"], t["total_ms"])
        g = self.call("grep", pattern="public key")["result"]["timing"]
        self.assertTrue({"scan_ms", "files_scanned", "total_ms", "server_ms"} <= set(g))
        self.assertEqual(g["files_scanned"], 1)

    def test_reload_from_the_indexer_while_models_load_is_not_an_error(self):
        from rag_search import api
        self.build(kitchen__bread=COOK)
        eng = SlowEngine(self.paths, embedder=FakeEmbedder(), reranker=FakeReranker())
        self.addCleanup(eng.release.set)
        d = self.start_daemon(engine=eng)
        r = api.reload_search(self.paths)
        self.assertTrue(r["ok"], r)
        self.assertIn("still loading", r["note"])
        eng.release.set()
        self.wait_ready(d)
        self.assertEqual(d.engine.generation, 1)  # picked up at the end of boot

    def test_model_error_is_reported(self):
        self.build(kitchen__bread=COOK)
        d = self.start_daemon(engine=BrokenEngine(self.paths, embedder=FakeEmbedder()))
        self.wait_ready(d)
        self.assertEqual(client.ping(self.paths, "search")["state"], "error")
        r = self.call("search", query="bread")
        self.assertEqual(r["code"], protocol.MODEL_ERROR)
        self.assertIn("no weights", r["error"])
        self.assertTrue(self.call("list")["ok"])

    def test_model_load_failure_is_retried_and_recovers(self):
        self.build(kitchen__bread=COOK)
        FlakyEngine.attempts = 0
        d = self.start_daemon(engine=FlakyEngine(self.paths, embedder=FakeEmbedder(),
                                                 reranker=FakeReranker()), retry=0.3)
        end = time.monotonic() + 5
        first = None
        while time.monotonic() < end and first is None:
            p = client.ping(self.paths, "search")
            first = p["error"] or None
            time.sleep(0.02)
        self.assertIn("download failed", first or "")
        end = time.monotonic() + 10
        while time.monotonic() < end and client.ping(self.paths, "search")["state"] != "ready":
            time.sleep(0.1)
        p = client.ping(self.paths, "search")
        self.assertEqual((p["state"], p["error"]), ("ready", ""))
        self.assertTrue(self.call("search", query="bread")["result"]["results"])
        self.assertTrue(d.ready.is_set())

    def test_model_mismatch_refuses_new_generation(self):
        self.build(kitchen__bread=COOK)
        d = self.start_daemon()
        self.wait_ready(d)
        d.engine.model = "some/other-model"
        self.write_doc("kitchen/x.md", COOK + " extra")
        self.index()
        self.publish()
        r = self.call("reload")
        self.assertEqual(r["code"], protocol.MODEL_MISMATCH)
        self.assertEqual(d.engine.generation, 1)  # still serving the old one
        self.assertTrue(self.call("search", query="bread")["ok"])

    def test_empty_home_serves_nothing_gracefully(self):
        d = self.start_daemon()
        self.wait_ready(d)
        r = self.call("search", query="anything")
        self.assertTrue(r["ok"])
        self.assertEqual(r["result"]["results"], [])

    def test_second_instance_exits_and_stale_socket_is_replaced(self):
        self.start_daemon()
        second = SearchDaemon(self.paths, engine=SearchEngine(
            self.paths, embedder=FakeEmbedder(), reranker=FakeReranker()))
        self.assertEqual(second.run(), 0)  # returns immediately: lock is held
        self.assertIsNotNone(client.ping(self.paths, "search"))
        self.assertTrue(client.alive_lock_held(self.paths, "search"))
        self.assertTrue(client.stop(self.paths, "search"))
        self.assertFalse(client.alive_lock_held(self.paths, "search"))
        self.paths.socket("search").write_text("stale")  # leftover file, no daemon
        self.start_daemon()  # replaces it

    def test_always_on_by_default_and_idle_exit_when_configured(self):
        d = self.start_daemon()
        self.assertEqual(d.idle_exit_seconds, 0)
        time.sleep(0.8)
        self.assertIsNotNone(client.ping(self.paths, "search"))
        self.assertFalse(d.stop.is_set())

    def test_configured_idle_exit(self):
        d = self.start_daemon(idle_exit_seconds=1)
        end = time.monotonic() + 6
        while time.monotonic() < end and not d.stop.is_set():
            time.sleep(0.1)
        self.assertTrue(d.stop.is_set())

    def test_config_file_controls_defaults(self):
        (self.paths.home / "config.json").write_text(json.dumps(
            {"search": {"idle_exit_seconds": 7, "prewarm": False}}))
        d = self.start_daemon()
        self.assertEqual((d.idle_exit_seconds, d.prewarm), (7, False))

    def test_lazy_indexes_load_on_first_search_when_prewarm_off(self):
        self.build(kitchen__bread=COOK)
        d = self.start_daemon(prewarm=False)
        self.wait_ready(d)
        self.assertEqual(d.engine.gen.indexes, {})
        self.assertTrue(self.call("search", query="bread")["result"]["results"])
        self.assertIn("kitchen", d.engine.gen.indexes)


class SearchDebugParamsTests(DaemonCase):
    """The stages/retrieval_pool/rerank_pool/rrf_k troubleshooting overrides, at the socket
    boundary the CLI and the dashboard both actually go through."""

    def test_stages_and_pool_overrides_reach_the_engine(self):
        self.build(security__auth=AUTH)
        d = self.start_daemon()
        self.wait_ready(d)
        r = self.call("search", query="public key authentication", stages="bm25",
                      retrieval_pool=7, rerank_pool=5, rrf_k=12)
        self.assertTrue(r["ok"], r)
        t = r["result"]["timing"]
        self.assertEqual((t["stages"], t["retrieval_pool"], t["rerank_pool"], t["rrf_k"]),
                         (["bm25"], 7, 5, 12))

    def test_bad_stages_is_a_clean_bad_request_before_touching_the_engine(self):
        self.build(security__auth=AUTH)
        d = self.start_daemon()
        self.wait_ready(d)
        r = self.call("search", query="x", stages="rerank")   # no retriever at all
        self.assertEqual((r["ok"], r["code"]), (False, protocol.BAD_REQUEST))
        r2 = self.call("search", query="x", stages="not-a-stage")
        self.assertEqual((r2["ok"], r2["code"]), (False, protocol.BAD_REQUEST))

    def test_non_numeric_pool_override_is_a_bad_request(self):
        self.build(security__auth=AUTH)
        d = self.start_daemon()
        self.wait_ready(d)
        r = self.call("search", query="x", retrieval_pool="lots")
        self.assertEqual((r["ok"], r["code"]), (False, protocol.BAD_REQUEST))

    def test_a_request_field_of_the_wrong_kind_is_a_bad_request_not_a_crash(self):
        self.build(security__auth=AUTH)
        d = self.start_daemon()
        self.wait_ready(d)
        for action, fields in (("search", {"query": "x", "wait_s": "soon"}), ("reload", {"wait_s": [1]}),
                               ("grep", {"pattern": "x", "context_lines": "two"}),
                               ("grep", {"pattern": "x", "max_matches": None})):
            r = self.call(action, **fields)
            self.assertEqual((r["ok"], r["code"]), (False, protocol.BAD_REQUEST), (action, fields))
        self.assertTrue(self.call("search", query="token")["ok"])                     # and it still answers

    def test_a_collection_name_that_cannot_exist_is_a_bad_request_from_the_api(self):
        from rag_search import api

        for call in (lambda: api.search(self.paths, "x", collections="../etc"),
                     lambda: api.grep(self.paths, "x", collections="a b")):
            r = call()
            self.assertEqual((r["ok"], r["code"]), (False, protocol.BAD_REQUEST))

    def test_overrides_are_clamped_even_from_a_client_request(self):
        from rag_search import spec
        self.build(security__auth=AUTH)
        d = self.start_daemon()
        self.wait_ready(d)
        r = self.call("search", query="public key authentication",
                      retrieval_pool=999999, rerank_pool=999999, rrf_k=999999)
        t = r["result"]["timing"]
        self.assertEqual((t["retrieval_pool"], t["rerank_pool"], t["rrf_k"]),
                         (spec.RETRIEVAL_POOL_MAX, spec.RERANK_POOL_MAX, spec.RRF_K_MAX))

    def test_config_json_defaults_apply_when_a_request_omits_the_field(self):
        """`rag-search config set` (or the dashboard's Settings tab) changes what "the default
        search" means for every client, including ones (like the MCP tool) that never ask for a
        specific value -- and takes effect on the very next search, no restart."""
        from rag_search import config
        self.build(security__auth=AUTH, kitchen__bread=COOK)
        config.update_config(self.paths, "search",
                             {"stages": "bm25", "retrieval_pool": 7, "rerank_pool": 5, "rrf_k": 12,
                              "top_k": 1})
        d = self.start_daemon()
        self.wait_ready(d)
        r = self.call("search", query="public key authentication")
        self.assertTrue(r["ok"], r)
        t = r["result"]["timing"]
        self.assertEqual((t["stages"], t["retrieval_pool"], t["rerank_pool"], t["rrf_k"]),
                         (["bm25"], 7, 5, 12))
        self.assertLessEqual(len(r["result"]["results"]), 1)
        # a caller-supplied value still wins over the configured default:
        r2 = self.call("search", query="public key authentication", stages="dense", top_k=2)
        t2 = r2["result"]["timing"]
        self.assertEqual(t2["stages"], ["dense"])
        self.assertLessEqual(len(r2["result"]["results"]), 2)


class ModelEnvTests(unittest.TestCase):
    """config.json's `models` tunables become environment variables before the embedder/reranker
    are constructed -- applied once, at daemon construction (a config change here needs
    `rag-search daemon restart` since the models are built once and kept warm)."""

    def test_config_tunables_become_env_vars_but_never_override_an_existing_one(self):
        from rag_search.core.search_daemon import apply_model_env

        os.environ["RAG_SEARCH_DEVICE"] = "mps"  # a real env var: config must not win
        self.addCleanup(os.environ.pop, "RAG_SEARCH_DEVICE", None)
        self.addCleanup(os.environ.pop, "RAG_SEARCH_EMBED_BATCH", None)
        self.addCleanup(os.environ.pop, "RAG_SEARCH_DTYPE", None)
        apply_model_env({"embed_batch": 16, "device": "cpu", "dtype": "", "max_seq": 0})
        self.assertEqual(os.environ["RAG_SEARCH_EMBED_BATCH"], "16")
        self.assertEqual(os.environ["RAG_SEARCH_DEVICE"], "mps")  # the real env var won
        self.assertNotIn("RAG_SEARCH_DTYPE", os.environ)   # blank in config -- no override
        self.assertNotIn("RAG_SEARCH_MAX_SEQ", os.environ)  # 0 in config -- no override


class ClientTests(DaemonCase):
    def test_unavailable_without_autostart(self):
        r = client.request_sync(self.paths, "search", "list", autostart=False)
        self.assertEqual((r["ok"], r["code"]), (False, protocol.UNAVAILABLE))

    def test_autostart_spawns_real_daemon_and_second_call_reuses_it(self):
        self.use_fake_backends()
        self.build(kitchen__bread=COOK)
        try:
            r = client.request_sync(self.paths, "search", "search", query="sourdough bread",
                                    wait_s=40)
            self.assertTrue(r["ok"], r)
            self.assertEqual(r["result"]["results"][0]["file"], "bread")
            pid1 = client.ping(self.paths, "search")["pid"]
            r2 = client.request_sync(self.paths, "search", "list")
            self.assertTrue(r2["ok"])
            self.assertEqual(client.ping(self.paths, "search")["pid"], pid1)
            self.assertFalse(client.spawn(self.paths, "search"))  # already running
        finally:
            client.stop(self.paths, "search")

    def test_concurrent_autostart_yields_one_daemon(self):
        self.use_fake_backends()
        self.build(kitchen__bread=COOK)
        results = []

        def go():
            results.append(client.request_sync(self.paths, "search", "list", wait_s=40))

        ts = [threading.Thread(target=go) for _ in range(5)]
        try:
            [t.start() for t in ts]
            [t.join(60) for t in ts]
            self.assertEqual(len(results), 5)
            self.assertTrue(all(r["ok"] for r in results), results)
            pid = client.ping(self.paths, "search")["pid"]
            os.kill(pid, 0)
            log = self.paths.log_file("search").read_text()
            self.assertEqual(log.count("listening"), 1, log)  # exactly one process was started
            self.assertNotIn("already holds the lock", log)
        finally:
            client.stop(self.paths, "search")

    def test_warming_up_when_daemon_never_binds(self):
        # a spawn that cannot start (lock held elsewhere) must end in a clear error, not hang
        import fcntl
        fd = os.open(self.paths.alive_lock("search"), os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            r = client.request_sync(self.paths, "search", "list", wait_s=1.5)
            self.assertEqual((r["ok"], r["code"]), (False, protocol.WARMING_UP))
        finally:
            os.close(fd)


if __name__ == "__main__":
    unittest.main()
