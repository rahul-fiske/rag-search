import contextlib
import io
import json
import os
import plistlib
import sys
import unittest
from unittest import mock
from pathlib import Path

from tests.helpers import TempHome
from rag_search import api, cli, client, config, register, service

AUTH = ("# Authentication\n\n<!-- page 1 -->\n\nSSH access uses public key authentication.\n\n"
        "<!-- page 2 -->\n\n## Roles\n\nRole based access control limits each account.")
DIARY = "# Diary\n\n<!-- page 1 -->\n\nSecret bread thoughts."


def run(*argv):
    os.environ.pop("RAG_SEARCH_CLIENT", None)   # cli.main sets it for --client; do not leak
    out, err = io.StringIO(), io.StringIO()
    rc = 0
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            cli.main(list(argv))
        except SystemExit as e:
            rc = e.code if isinstance(e.code, int) else 1
    return rc, out.getvalue(), err.getvalue()



class FakeHost:
    """Stands in for an optional ``hosts_*.py`` module."""
    NAME, LABEL, EXTRA_ENTRY = "demo", "Demo Host", {"disabled": False}

    def __init__(self, tmp):
        self.config = Path(tmp) / "demo-host" / "mcp.json"

    def installed(self):
        return self.config.parent.is_dir()

    def default_config(self):
        return self.config

    def find_config(self):
        return self.config if self.config.is_file() else None

    def register(self, home=None, path=None, tool_prefix=""):
        target = Path(path) if path else self.config
        return register._add(target, register.server_entry(home, self.NAME, tool_prefix,
                                                           self.EXTRA_ENTRY),
                             self.LABEL, "Reload it.", "")

    def unregister(self, path=None):
        return register._remove(Path(path) if path else self.config, self.LABEL)


class ApiFallbackTests(TempHome):
    """No daemons running: list and grep read serving/current directly."""

    def setUp(self):
        super().setUp()
        self.write_doc("security/auth.md", AUTH)
        self.write_doc("personal/diary.md", DIARY)
        self.index()
        self.publish()

    def test_list_and_grep_fall_back_to_local_reads(self):
        r = api.list_collections(self.paths, client="claude")
        self.assertEqual((r["ok"], r["source"]), (True, "local"))
        self.assertEqual(sorted(c["collection"] for c in r["result"]["collections"]),
                         ["personal", "security"])
        g = api.grep(self.paths, "public key", client="claude")
        self.assertTrue(g["ok"] and g["result"]["matches"])
        self.assertFalse(client.alive_lock_held(self.paths, "search"))  # nothing was started

    def test_local_fallback_enforces_access_rules(self):
        from rag_search import policy
        # open by default
        r = api.list_collections(self.paths, client="agent")
        self.assertEqual(sorted(c["collection"] for c in r["result"]["collections"]),
                         ["personal", "security"])
        policy.save_rules(self.paths, {"personal": ["claude"]})
        r = api.list_collections(self.paths, client="agent")
        self.assertEqual([c["collection"] for c in r["result"]["collections"]], ["security"])
        g = api.grep(self.paths, "Secret", collections="personal", client="agent")
        self.assertEqual((g["ok"], g["code"]), (False, "bad_request"))
        self.assertIn("unknown collection", g["error"])
        g = api.grep(self.paths, "Secret", client="agent")  # unscoped: only what agent may use
        self.assertEqual(g["result"]["matches"], [])
        g = api.grep(self.paths, "Secret", collections="personal", client="claude")
        self.assertEqual(len(g["result"]["matches"]), 1)
        g = api.grep(self.paths, "Secret", collections="personal", client="cli")
        self.assertEqual(len(g["result"]["matches"]), 1)

    def test_restricted_names_are_matched_exactly_and_never_revealed(self):
        from rag_search import policy
        policy.save_rules(self.paths, {"personal": ["claude"]})
        g = api.grep(self.paths, "Secret", collections="PERSONAL", client="agent")
        self.assertEqual((g["ok"], g["code"]), (False, "bad_request"))
        # a client that may read it gets the collection under its real spelling (the folder is
        # always reached by the canonical name, never by whatever casing was typed)
        g = api.grep(self.paths, "Secret", collections="PERSONAL", client="claude")
        self.assertTrue(g["ok"], g)
        self.assertTrue(g["result"]["matches"])
        self.assertEqual({m["collection"] for m in g["result"]["matches"]}, {"personal"})

    def test_index_status_hides_documents_and_paths_of_restricted_collections(self):
        import time
        from rag_search import jobs, policy
        self.paths.jobs.mkdir(parents=True, exist_ok=True)
        jobs.job_file(self.paths, "j1").write_text(json.dumps({
            "id": "j1", "status": "failed", "mode": "new", "path": "personal",
            "created_at": time.time(), "error": "cannot read personal/diary.pdf",
            "summary": {"errors": ["personal/diary.pdf: boom", "security/x.pdf: bad"]}}))
        jobs.events_file(self.paths, "j1").write_text("\n".join([
            '{"ts": 1.0, "event": "doc", "collection": "personal", "source": "diary.md", "status": "indexed"}',
            '{"ts": 2.0, "event": "doc", "collection": "security", "source": "auth.md", "status": "indexed"}',
        ]) + "\n")
        full = api.index_status(self.paths, client="agent")
        self.assertTrue(any(i["collection"] == "personal" for i in full["documents"]["items"]))
        policy.save_rules(self.paths, {"personal": ["claude"]})
        seen = api.index_status(self.paths, client="agent")
        self.assertTrue(seen["documents"]["items"])
        self.assertTrue(all(i["collection"] != "personal" for i in seen["documents"]["items"]))
        self.assertEqual(seen["documents"]["total"], len(seen["documents"]["items"]))
        self.assertNotIn("personal", json.dumps(seen))         # path, error text and messages too
        self.assertEqual(seen["job"]["path"], "[restricted]")
        self.assertEqual(seen["job"]["summary"]["errors"], ["[restricted]", "security/x.pdf: bad"])
        again = api.index_status(self.paths, client="claude")
        self.assertTrue(any(i["collection"] == "personal" for i in again["documents"]["items"]))

    def test_pathological_regex_cannot_freeze_the_caller(self):
        import time
        self.write_doc("security/long.md", "# T\n\n<!-- page 1 -->\n\n" + "a" * 40 + "b\n")
        self.index()
        self.publish()
        from rag_search import grep as grep_mod
        t0 = time.monotonic()
        res = grep_mod.grep_isolated(self.paths.live_markup(), "(a+)+$", ["security"],
                                     time_budget_s=1.0)
        self.assertLess(time.monotonic() - t0, 10)
        self.assertIn("timed out", res.get("error", ""))
        ok = grep_mod.grep_isolated(self.paths.live_markup(), "public key", ["security"])
        self.assertEqual(len(ok["matches"]), 1)

    def test_index_status_and_cancel_without_daemon_do_not_start_it(self):
        st = api.index_status(self.paths)
        self.assertEqual((st["ok"], st["running"], st["job"]), (True, False, None))
        self.assertTrue(api.index_cancel(self.paths)["ok"])
        self.assertFalse(client.alive_lock_held(self.paths, "indexer"))

    def test_daemon_status_and_unknown_kind(self):
        st = api.daemon_status(self.paths)
        self.assertEqual(st["search"]["running"], False)
        with self.assertRaises(ValueError):
            api.daemon_stop(self.paths, "bogus")
        self.assertEqual(api.daemon_stop(self.paths, "all"),
                         {"search": "not running", "indexer": "not running"})


class CliFlowTests(TempHome):
    def setUp(self):
        super().setUp()
        self.use_fake_backends()

    def tearDown(self):
        api.daemon_stop(self.paths, "all")
        super().tearDown()

    def test_full_cli_flow(self):
        self.write_doc("security/auth.md", AUTH)
        rc, out, err = run("index", "new", "--follow")
        self.assertEqual(rc, 0, err)
        self.assertIn("succeeded", out)
        self.assertIn("published generation 1", out)
        rc, out, _ = run("search", "role based access control", "-k", "2")
        self.assertEqual(rc, 0)
        self.assertIn("auth (security) p.2", out)
        rc, out, _ = run("search", "role based", "--json")
        self.assertEqual(json.loads(out)["results"][0]["page"], "2")
        rc, out, _ = run("list")
        self.assertIn("security: 1 doc(s)", out)
        rc, out, _ = run("grep", "public key")
        self.assertIn("security/auth  p.1", out)
        rc, out, _ = run("index", "status")
        self.assertIn("succeeded", out)
        rc, out, _ = run("daemon", "status")
        self.assertIn("search: ready", out)
        self.assertIn("indexer: ready", out)
        rc, out, _ = run("daemon", "restart", "search")
        self.assertEqual(rc, 0)
        self.assertIn("start search: started", out)
        rc, out, _ = run("search", "SSH", "--json")  # autostart again, index reloaded from disk
        self.assertEqual(rc, 0)
        self.assertTrue(json.loads(out)["results"])
        rc, out, _ = run("daemon", "stop")
        self.assertEqual(rc, 0)
        rc, out, _ = run("index", "status")
        self.assertIn("succeeded", out.replace("indexer daemon is not running (nothing is being "
                                               "indexed)\n", ""))

    def test_search_stage_and_pool_debug_flags(self):
        self.write_doc("security/auth.md", AUTH)
        rc, out, err = run("index", "new", "--follow")
        self.assertEqual(rc, 0, err)
        rc, out, _ = run("search", "role based access control", "--stages", "bm25", "--explain")
        self.assertEqual(rc, 0)
        self.assertIn("BM25 #", out)
        self.assertIn("Dense —", out)
        self.assertIn("Rerank —", out)
        self.assertIn("stages=bm25", out)
        self.assertIn("retrieval_pool=", out)
        rc, out, _ = run("search", "role based access control", "--json",
                         "--stages", "bm25,dense", "--rrf-k", "10")
        res = json.loads(out)
        self.assertFalse(res["timing"]["reranked"])
        self.assertEqual(res["timing"]["rrf_k"], 10)
        self.assertIsNone(res["results"][0]["rerank_score"])
        self.assertIsNotNone(res["results"][0]["rrf_score"])
        # a stage combination with no retriever at all is a clean, non-zero-exit failure
        rc, out, err = run("search", "x", "--stages", "rerank")
        self.assertNotEqual(rc, 0)

    def test_sizes_timings_and_daemon_details_in_the_cli(self):
        self.write_doc("security/auth.md", AUTH)
        rc, out, err = run("index", "new", "--follow")
        self.assertEqual(rc, 0, err)
        # index status: when it started, how long it took, and a line per document
        rc, out, _ = run("index", "status")
        self.assertIn("started 20", out)
        self.assertIn("took ", out)
        self.assertIn("time by phase: convert", out)
        self.assertIn("documents: 1 handled (indexed 1, unchanged 0, failed 0)", out)
        self.assertRegex(out, r"\[ ok \] security/auth\.md  \d+ chunks  3 convert .*5 embed .*total ")
        rc, out, _ = run("index", "status", "--json")
        docs = json.loads(out)["documents"]
        self.assertEqual(docs["by_status"], {"indexed": 1})
        self.assertIn("total_s", docs["items"][0])
        rc, out, _ = run("index", "status", "--docs", "0")
        self.assertNotIn("[ ok ]", out)
        # list: index size, when built, how long it took
        rc, out, _ = run("list")
        self.assertRegex(out, r"total: 1 collection\(s\), 1 doc\(s\), \d+ chunks, index [\d.]+ (B|KB|MB)")
        self.assertRegex(out, r"security: 1 doc\(s\), \d+ chunks, index [\d.]+ (B|KB|MB)")
        self.assertRegex(out, r"built 20\d\d-\d\d-\d\d \d\d:\d\d:\d\d, took ")
        self.assertRegex(out, r"auth  \(\d+ chunks, [\d.]+ (B|KB|MB), built in ")
        rc, out, _ = run("list", "--json")
        c = json.loads(out)["collections"][0]
        self.assertGreater(c["index_bytes"], 0)
        self.assertIsNotNone(c["build_seconds"])
        self.assertTrue(c["built_at"])
        # search and grep say how long they took
        rc, out, _ = run("search", "role based access control")
        self.assertRegex(out, r"\d+ result\(s\) in \d+ ms \(embed query \d+ ms, keyword \d+ ms, "
                              r"vectors \d+ ms, rerank \d+ ms; 1 collection\(s\), generation 1")
        self.assertIn("round trip", out)
        t = json.loads(run("search", "SSH", "--json")[1])["timing"]
        self.assertIn("total_ms", t)
        self.assertIn("round_trip_ms", t)
        rc, out, _ = run("grep", "public key")
        self.assertRegex(out, r"1 match\(es\) in \d+ ms \(scan \d+ ms, 1 files; round trip \d+ ms\)")
        # daemon status: warmed up, how long the warm-up took, memory
        rc, out, _ = run("daemon", "status")
        self.assertIn("search: ready - warmed up", out)
        self.assertRegex(out, r"warm-up: took [\d.]+s? \(models .*indexes .*\), ready since 20")
        self.assertRegex(out, r"memory: [\d.]+ (KB|MB|GB) resident \(peak [\d.]+ (KB|MB|GB)\); "
                              r"index data [\d.]+ (B|KB|MB)")
        st = json.loads(run("daemon", "status", "--json")[1])["search"]
        self.assertTrue(st["warm"])
        self.assertEqual(st["warmup"]["status"], "warm")
        self.assertGreater(st["memory"]["rss_bytes"], 0)

    def test_home_flag_works_before_and_after_subcommand(self):
        other = self.tmp / "other"
        rc, out, _ = run("--home", str(other), "paths", "home")
        self.assertEqual(out.strip(), str(other))
        rc, out, _ = run("paths", "home", "--home", str(other))
        self.assertEqual(out.strip(), str(other))
        rc, out, _ = run("index", "--home", str(other), "status", "--json")
        self.assertEqual(json.loads(out)["running"], False)

    def test_errors_have_exit_codes(self):
        rc, _, err = run("index", "new", "does/not/exist")
        self.assertEqual(rc, 1)
        self.assertIn("no source folders are registered", err)
        self.write_doc("c/a.md", "x")
        rc, _, err = run("index", "new", "does/not/exist")
        self.assertEqual(rc, 1)
        self.assertIn("neither a registered location", err)
        rc, _, _ = run("paths", "nonsense")
        self.assertEqual(rc, 2)
        rc, _, _ = run()
        self.assertEqual(rc, 2)

    def test_foreground_index_publishes_without_the_indexer_daemon(self):
        self.write_doc("c/a.md", AUTH)
        rc, out, err = run("index", "foreground")
        self.assertEqual(rc, 0, err)
        self.assertIn("published generation 1", out)
        self.assertFalse(client.alive_lock_held(self.paths, "indexer"))
        rc, out, _ = run("grep", "Roles")
        self.assertIn("c/a", out)

    def test_restart_flag_and_second_start_report(self):
        self.write_doc("c/a.md", AUTH)
        os.environ["RAG_SEARCH_EMBEDDER"] = "tests.helpers:SlowEmbedder"
        os.environ["TEST_EMBED_SLEEP"] = "2"
        rc, out, _ = run("index", "new")
        self.assertIn("indexing started", out)
        rc, out, _ = run("index", "new")
        self.assertIn("already active", out)
        rc, out, _ = run("index", "new", "--restart")
        self.assertIn("restarted indexing", out)
        run("index", "cancel")
        rc, out, _ = run("index", "status", "--history", "3")
        self.assertIn("cancelled", out)

    def test_config_commands(self):
        rc, out, _ = run("config", "init")
        self.assertTrue(self.paths.config_file.exists())
        rc, out, _ = run("config", "show")
        self.assertEqual(json.loads(out)["search"]["idle_exit_seconds"], 0)


class DoctorTests(TempHome):
    def setUp(self):
        super().setUp()
        self.use_fake_backends()

    def test_checks_report_without_crashing_and_roundtrip_passes_with_fake_backends(self):
        from rag_search.core import diagnostics

        rows = diagnostics.run_checks(self.paths)
        labels = [r[1] for r in rows]
        self.assertIn("search daemon", labels)
        self.assertIn("config.json", labels)
        home = os.environ["RAG_SEARCH_HOME"]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = diagnostics.roundtrip()
        self.assertEqual(rc, 0, buf.getvalue())
        self.assertIn("roundtrip PASSED", buf.getvalue())
        self.assertEqual(os.environ["RAG_SEARCH_HOME"], home)  # restored


class RegisterTests(TempHome):
    def setUp(self):
        super().setUp()
        self._home = os.environ.get("HOME")
        os.environ["HOME"] = str(self.tmp / "userhome")
        (self.tmp / "userhome").mkdir()

    def tearDown(self):
        if self._home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = self._home
        super().tearDown()

    def test_desktop_registration_uses_adapter_and_profile(self):
        cfg = register.desktop_config_path()
        cfg.parent.mkdir(parents=True)
        cfg.write_text(json.dumps({"mcpServers": {"other": {"command": "x"}}, "theme": 1}))
        msg = register.register_desktop("/data/rag", tool_prefix="kb_")
        data = json.loads(cfg.read_text())
        entry = data["mcpServers"]["rag-search"]
        self.assertIn("other", data["mcpServers"])
        self.assertEqual(data["theme"], 1)
        self.assertEqual(entry["args"][-4:], ["--profile", "claude", "--tool-prefix", "kb_"])
        self.assertTrue(entry["command"].endswith("rag-search-mcp")
                        or entry["args"][:2] == ["-m", "rag_search.mcp"])
        self.assertEqual(entry["env"], {"RAG_SEARCH_HOME": "/data/rag"})
        self.assertIn("backup", msg)
        self.assertIn("removed", register.unregister_desktop())
        self.assertNotIn("rag-search", json.loads(cfg.read_text())["mcpServers"])
        self.assertIn("not registered", register.unregister_desktop())

    def test_invalid_json_is_left_untouched(self):
        cfg = register.desktop_config_path()
        cfg.parent.mkdir(parents=True)
        cfg.write_text("{ not json")
        msg = register.register_desktop()
        self.assertEqual(cfg.read_text(), "{ not json")
        self.assertIn("left untouched", msg)
        self.assertIn("mcpServers", msg)

    def test_registration_status_for_doctor(self):
        with mock.patch.object(register, "_EXTRAS", []):                # Claude Desktop alone
            self._registration_status_for_doctor()

    def _registration_status_for_doctor(self):
        rows = {h: (s, d) for h, s, d in register.registration_status()}
        self.assertEqual(list(rows), ["Claude Desktop"])                # nothing else is listed
        self.assertEqual(rows["Claude Desktop"][0], "warn")
        register.register_desktop()
        rows = {h: (s, d) for h, s, d in register.registration_status()}
        self.assertEqual(rows["Claude Desktop"][0], "ok")
        self.assertIn("profile claude", rows["Claude Desktop"][1])
        # a command that no longer exists is a failure that says how to fix it
        cfg = register.desktop_config_path()
        data = json.loads(cfg.read_text())
        data["mcpServers"]["rag-search"]["command"] = "/nonexistent/rag-search-mcp"
        cfg.write_text(json.dumps(data))
        status = {h: (s, d) for h, s, d in register.registration_status()}
        self.assertEqual(status["Claude Desktop"][0], "fail")
        self.assertIn("rag-search register --desktop", status["Claude Desktop"][1])

    def test_register_never_touches_an_optional_host_unless_asked(self):
        h = FakeHost(self.tmp)
        with mock.patch.object(register, "_EXTRAS", [h]):
            rc, out, _ = run("register", "--home", str(self.tmp / "d"))
            self.assertEqual(rc, 0)
            self.assertNotIn(h.LABEL, out)
            self.assertFalse(h.config.exists())
            self.assertNotIn(h.LABEL, {r[0] for r in register.registration_status()})
            self.assertEqual(register.host_clients(), ("claude",))
            rc, out, _ = run("register", "--demo", "--home", str(self.tmp / "d"))
            self.assertIn(f"{h.LABEL}: registered", out)
            self.assertTrue(h.config.exists())
            self.assertEqual(register.host_clients(), ("claude", "demo"))
            self.assertIn(h.LABEL, {r[0] for r in register.registration_status()})
            rc, out, _ = run("unregister")
            self.assertIn(f"{h.LABEL}: removed", out)
            self.assertEqual(register.host_clients(), ("claude",))

    def test_optional_host_flags_are_not_advertised(self):
        h = FakeHost(self.tmp)
        with mock.patch.object(register, "_EXTRAS", [h]):
            rc, out, _ = run("register", "--help")
            self.assertEqual(rc, 0)
            self.assertNotIn("--demo", out)
            self.assertNotIn(h.LABEL, out)
            rc, out, _ = run("mcp-config", "--profile", "demo")
            self.assertEqual(json.loads(out)["mcpServers"]["rag-search"]["disabled"], False)
        rc, _, err = run("mcp-config", "--profile", "demo")             # this build has no such host
        self.assertNotEqual(rc, 0)
        self.assertIn("unknown profile", err)

    def test_access_lists_an_optional_host_only_once_registered(self):
        h = FakeHost(self.tmp)
        with mock.patch.object(register, "_EXTRAS", [h]):
            rc, out, _ = run("access")
            self.assertIn("Clients: cli (admin), claude", out)
            self.assertNotIn("demo", out)
            run("register", "--demo", "--home", str(self.tmp / "d"))
            rc, out, _ = run("access")
            self.assertIn("Clients: cli (admin), claude, demo", out)

    def test_switches_auto_register_always_list_and_advertise(self):
        h = FakeHost(self.tmp)
        h.AUTO_REGISTER = h.ALWAYS_LISTED = h.ADVERTISE = True
        with mock.patch.object(register, "_EXTRAS", [h]):
            self.assertEqual(register.host_clients(), ("claude", "demo"))          # listed while unregistered
            self.assertEqual({r[0]: r[1] for r in register.registration_status()}[h.LABEL], "ok")   # not installed
            rc, out, _ = run("register", "--help")
            self.assertIn("--demo", out)
            self.assertIn("--no-demo", out)
            rc, out, _ = run("register", "--home", str(self.tmp / "d"))
            self.assertNotIn(f"{h.LABEL}: registered", out)                        # not installed: skipped
            h.config.parent.mkdir(parents=True)
            self.assertEqual({r[0]: r[1] for r in register.registration_status()}[h.LABEL], "warn")
            rc, out, _ = run("register", "--home", str(self.tmp / "d"), "--no-demo")
            self.assertFalse(h.config.exists())
            rc, out, _ = run("register", "--home", str(self.tmp / "d"))
            self.assertIn(f"{h.LABEL}: registered", out)                           # installed: together with Claude
            self.assertEqual({r[0]: r[1] for r in register.registration_status()}[h.LABEL], "ok")

    def test_a_build_without_optional_hosts_has_none(self):
        register._EXTRAS = None
        try:
            names = [m.NAME for m in register.extra_hosts()]
        finally:
            register._EXTRAS = None
        self.assertEqual(names, sorted(names))                           # discovered by file name
        self.assertNotIn("claude", names)


class ServiceTests(TempHome):
    def test_plists(self):
        for kind, mod in (("search", "rag_search.core.search_daemon"),
                          ("indexer", "rag_search.core.indexer_daemon")):
            d = service.plist_dict(self.paths, kind, python="/opt/py/bin/python3")
            self.assertEqual(d["Label"], f"io.rag-search.{kind}")
            self.assertEqual(d["ProgramArguments"], ["/opt/py/bin/python3", "-m", mod])
            self.assertEqual(d["EnvironmentVariables"]["RAG_SEARCH_HOME"], str(self.paths.home))
            self.assertTrue(d["RunAtLoad"])
            self.assertEqual(d["KeepAlive"], {"SuccessfulExit": False})
            self.assertEqual(d["StandardOutPath"], str(self.paths.log_file(kind)))
            plistlib.loads(plistlib.dumps(d))  # serialisable

    def test_env_passthrough(self):
        os.environ["RAG_SEARCH_DEVICE"] = "cpu"
        try:
            d = service.plist_dict(self.paths, "search")
            self.assertEqual(d["EnvironmentVariables"]["RAG_SEARCH_DEVICE"], "cpu")
        finally:
            del os.environ["RAG_SEARCH_DEVICE"]

    @unittest.skipIf(sys.platform == "darwin", "would touch real launchd")
    def test_install_is_macos_only(self):
        self.assertEqual(service.install(self.paths), ["launchd services are macOS-only"])
        self.assertEqual(service.uninstall(self.paths), ["launchd services are macOS-only"])
        st = service.status(self.paths)
        self.assertFalse(st["search"]["installed"])


if __name__ == "__main__":
    unittest.main()


class AccessCliTests(TempHome):
    """`rag-search access ...`: the only place where access is managed."""

    def setUp(self):
        super().setUp()
        self.write_doc("security/auth.md", AUTH)
        self.write_doc("personal/diary.md", DIARY)
        self.index()
        self.publish()

    def rules(self):
        from rag_search import policy
        return policy.load_rules(self.paths)[0].by_name

    def test_default_lists_everything_as_open_to_everyone(self):
        with mock.patch.object(register, "_EXTRAS", []):                # Claude alone
            rc, out, _ = run("access")
        self.assertEqual(rc, 0, out)
        self.assertIn("open to all clients", out)
        for name in ("security", "personal"):
            self.assertRegex(out, rf"{name}\s+everyone")
        self.assertIn("Clients: cli (admin), claude", out)
        rc, out, _ = run("access", "list", "--json")
        data = json.loads(out)
        self.assertEqual({r["collection"]: r["access"] for r in data["collections"]},
                         {"personal": "everyone", "security": "everyone"})
        self.assertTrue(all(r["indexed"] and r["exists"] for r in data["collections"]))
        self.assertFalse(self.paths.access_file.exists())      # reading never creates rules

    def test_restrict_and_grant(self):
        rc, out, _ = run("access", "restrict", "personal", "claude")
        self.assertEqual(rc, 0, out)
        self.assertIn("personal: restricted to claude", out)
        self.assertEqual(self.rules(), {"personal": ["claude"]})
        rc, out, _ = run("access", "grant", "PERSONAL", "Agent", "mybot")     # names are normalised
        self.assertEqual(rc, 0, out)
        self.assertIn("'mybot' has not been seen before", out)
        self.assertEqual(self.rules(), {"personal": ["agent", "claude", "mybot"]})
        # restrict states the whole list, so it also removes clients
        run("access", "restrict", "personal", "claude")
        self.assertEqual(self.rules(), {"personal": ["claude"]})
        rc, out, _ = run("access")
        self.assertRegex(out, r"personal\s+restricted\s+claude")
        self.assertRegex(out, r"security\s+everyone")
        rc, out, _ = run("access", "restrict", "personal")                # no names = nobody
        self.assertIn("nobody", out)
        self.assertEqual(self.rules(), {"personal": []})                 # still closed
        rc, out, _ = run("access", "restrict", "personal", "all")        # 'all' = everyone
        self.assertIn("open to every client", out)
        self.assertEqual(self.rules(), {})
        rc, out, _ = run("access", "grant", "personal", "all")
        self.assertIn("already was", out)
        run("access", "restrict", "personal", "claude")
        rc, out, _ = run("access", "grant", "personal", "all")           # grant everyone = open
        self.assertIn("open to every client", out)
        self.assertEqual(self.rules(), {})

    def test_the_three_commands_are_all_there_is(self):
        for sub in ("open", "revoke", "clients"):
            rc, _, err = run("access", sub, "personal")
            self.assertEqual(rc, 2, sub)
        rc, out, _ = run("access", "--help")
        for sub in ("restrict", "grant"):
            self.assertIn(sub, out)
        self.assertNotIn("revoke", out)

    def test_what_each_client_gets_is_visible_with_client_flag(self):
        run("access", "restrict", "personal", "claude")
        run("access", "restrict", "security", "claude", "agent")     # a client named in a rule is listed
        rc, out, _ = run("list", "--client", "agent")
        self.assertIn("security", out)
        self.assertNotIn("personal", out)
        rc, out, _ = run("list", "--client", "claude")
        self.assertIn("personal", out)
        rc, out, _ = run("list")                     # no --client: the administrator
        self.assertIn("personal", out)
        rc, out, _ = run("access", "--json")
        by = {c["client"]: c for c in json.loads(out)["clients"]}
        self.assertEqual(by["claude"]["can_use"], ["personal", "security"])
        self.assertEqual(by["agent"]["blocked_from"], ["personal"])
        self.assertEqual(by["claude"]["restricted_to"], ["personal", "security"])

    def test_bad_input_is_refused_and_changes_nothing(self):
        for argv in (("access", "restrict", "personal", "cli"),
                     ("access", "restrict", "personal", "unknown"),
                     ("access", "restrict", "personal", "Bad Name!"),
                     ("access", "restrict", "../etc", "claude"),
                     ("access", "grant", "personal", "agent")):      # not restricted yet
            rc, out, err = run(*argv)
            self.assertEqual(rc, 2, (argv, out, err))
            self.assertIn("error:", err)
        self.assertEqual(self.rules(), {})

    def test_a_collection_can_be_restricted_before_it_is_indexed(self):
        self.write_doc("finance/plan.md", "# P\n\n<!-- page 1 -->\n\nbudget")
        rc, out, _ = run("access", "restrict", "finance", "claude")
        self.assertEqual(rc, 0, out)
        rc, out, _ = run("access", "list")
        self.assertRegex(out, r"finance\s+restricted\s+claude\s+not indexed yet")
        run("access", "restrict", "ghost", "agent")
        rc, out, _ = run("access", "list")
        self.assertRegex(out, r"ghost\s+restricted\s+agent\s+rule only")

    def test_a_damaged_rules_file_blocks_changes_and_keeps_collections_closed(self):
        run("access", "restrict", "personal", "claude")
        self.paths.access_file.write_text('{"collections": {"personal": ["claude"], ')
        rc, out, err = run("access", "grant", "personal", "agent")
        self.assertEqual(rc, 2)
        self.assertIn("fix or remove that file", err)
        rc, out, _ = run("list", "--client", "agent")
        self.assertNotIn("personal", out)
        rc, out, _ = run("list", "--client", "claude")
        self.assertNotIn("personal", out)                      # closed for everyone but the admin
        rc, out, _ = run("list")
        self.assertIn("personal", out)

    def test_doctor_lines_cover_rules_and_legacy_leftovers(self):
        from rag_search.core import diagnostics
        rows = {label: (st, detail) for st, label, detail in diagnostics._access_lines(
            self.paths, config.ConfigStore(self.paths))}
        self.assertIn("access rules", rows)
        self.assertIn("access: personal", rows)              # 'personal' used to be private
        run("access", "restrict", "personal", "claude")
        self.paths.config_file.write_text(json.dumps({"policy": {"private_collections": ["x"]}}))
        rows = {label: (st, detail) for st, label, detail in diagnostics._access_lines(
            self.paths, config.ConfigStore(self.paths))}
        self.assertIn("1 restricted", rows["access rules"][1])
        self.assertNotIn("access: personal", rows)
        self.assertIn("config.json policy", rows)


class DescribeCliTests(TempHome):
    """`rag-search describe ...`: the human override/audit path for collection descriptions."""

    def setUp(self):
        super().setUp()
        self.write_doc("security/auth.md", AUTH)
        self.write_doc("personal/diary.md", DIARY)
        self.index()
        self.publish()

    def test_no_arguments_lists_nothing_set_yet(self):
        rc, out, _ = run("describe")
        self.assertEqual(rc, 0, out)
        self.assertIn("No collections have a description yet", out)

    def test_set_show_and_clear(self):
        rc, out, _ = run("describe", "security", "Auth and roles docs")
        self.assertEqual(rc, 0, out)
        self.assertIn("security: Auth and roles docs", out)

        rc, out, _ = run("describe", "security")
        self.assertEqual(rc, 0, out)
        self.assertIn("security: Auth and roles docs", out)

        rc, out, _ = run("describe")
        self.assertEqual(rc, 0, out)
        self.assertIn("security", out)
        self.assertIn("Auth and roles docs", out)
        self.assertNotIn("personal", out)

        rc, out, _ = run("describe", "security", "--clear")
        self.assertEqual(rc, 0, out)
        self.assertIn("description cleared", out)
        rc, out, _ = run("describe", "security")
        self.assertIn("no description set", out)

    def test_json_output(self):
        run("describe", "security", "Auth and roles docs")
        rc, out, _ = run("describe", "security", "--json")
        self.assertEqual(rc, 0, out)
        self.assertEqual(json.loads(out),
                         {"collection": "security", "description": "Auth and roles docs"})
        rc, out, _ = run("describe", "--json")
        self.assertEqual(json.loads(out), {"collections": {"security": "Auth and roles docs"}})

    def test_bad_collection_name_is_a_usage_error(self):
        rc, out, err = run("describe", "../etc", "text")
        self.assertNotEqual(rc, 0)
        self.assertIn("not a collection name", err)

    def test_overlong_description_is_a_usage_error(self):
        rc, out, err = run("describe", "security", "x" * 501)
        self.assertNotEqual(rc, 0)
        self.assertIn("500", err)

    def test_list_shows_descriptions_and_document_counts(self):
        run("describe", "security", "Auth and roles docs")
        rc, out, _ = run("list")
        self.assertEqual(rc, 0, out)
        self.assertIn("Auth and roles docs", out)
