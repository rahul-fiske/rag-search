"""CLI commands that no other test drove: JSON and error variants, local commands, the text of the output.

Tier A: in process, fake embedder, documents from the corpus; anything that would start a daemon or touch
launchd is replaced at the `api` / `service` seam."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest import mock

from tests import corpus
from tests.helpers import TempHome
from tests.portable.test_cli_api import run

from rag_search import api, service
from rag_search.core.conversion import trace
from rag_search.paths import sha256_file


class LocalCommandTests(TempHome):
    def test_paths_prints_one_name_or_all_and_refuses_an_unknown_one(self):
        rc, out, _ = run("paths", "config")
        self.assertEqual((rc, out.strip()), (0, str(self.paths.config_file)))
        rc, out, _ = run("paths")
        self.assertEqual(json.loads(out)["home"], str(self.paths.home))
        rc, out, err = run("paths", "nonsense")
        self.assertNotEqual(rc, 0)
        self.assertIn("unknown name", err)

    def test_config_path_init_set_and_show(self):
        rc, out, _ = run("config", "path")
        self.assertEqual(out.strip(), str(self.paths.config_file))
        rc, out, _ = run("config", "init")
        self.assertIn("config file:", out)
        self.assertTrue(self.paths.config_file.exists())
        rc, out, _ = run("config", "set", "--chunk-size", "300", "--rrf-k", "40")
        self.assertEqual(rc, 0)
        self.assertIn("indexer.chunk_size", out)
        self.assertIn("search.rrf_k", out)
        rc, out, _ = run("config", "set", "--chunk-size", "300", "--json")      # JSON: the whole file
        self.assertEqual(json.loads(out[out.index("{"):])["indexer"]["chunk_size"], 300)
        rc, out, _ = run("config")
        self.assertEqual(json.loads(out)["search"]["rrf_k"], 40)

    def test_config_set_refuses_nothing_to_set_and_bad_values_and_changes_nothing(self):
        rc, _, err = run("config", "set")
        self.assertNotEqual(rc, 0)
        self.assertIn("at least one --flag", err)
        before = self.paths.config_file.read_text() if self.paths.config_file.exists() else ""
        rc, _, err = run("config", "set", "--stages", "bogus")
        self.assertNotEqual(rc, 0)
        self.assertIn("error:", err)
        after = self.paths.config_file.read_text() if self.paths.config_file.exists() else ""
        self.assertEqual(before, after)

    def test_a_broken_config_file_is_reported_as_a_warning_and_defaults_are_shown(self):
        self.paths.config_file.write_text("{ not json")
        rc, out, err = run("config")
        self.assertEqual(rc, 0)
        self.assertIn("warning:", err)
        self.assertIn("indexer", json.loads(out))


class DaemonAndServiceCommandTests(TempHome):
    def test_daemon_status_says_what_each_daemon_is_doing(self):
        st = {"search": {"state": "ready", "pid": 11, "uptime_s": 75, "generation": 3, "collections": 2,
                         "chunks": 40, "warmup": {"status": "warm"}},
              "indexer": {"state": "ready", "pid": 12, "uptime_s": 5, "running": True, "error": "disk is full"}}
        with mock.patch.object(api, "daemon_status", return_value=st):
            rc, out, _ = run("daemon", "status")
        self.assertEqual(rc, 0)
        self.assertIn("search: ready - warmed up (pid 11", out)
        self.assertIn("generation 3, 2 collection(s), 40 chunks", out)
        self.assertIn("indexer: ready (pid 12", out)
        self.assertIn("run active", out)
        self.assertIn("error: disk is full", out)
        with mock.patch.object(api, "daemon_status", return_value={"search": {"starting": True}, "indexer": {}}):
            rc, out, _ = run("daemon", "status")
        self.assertIn("search: starting", out)
        self.assertIn("indexer: not running", out)

    def test_daemon_start_stop_restart_print_each_result_or_json(self):
        with mock.patch.object(api, "daemon_start", return_value={"search": "started"}):
            rc, out, _ = run("daemon", "start", "search")
        self.assertEqual((rc, out.strip()), (0, "search: started"))
        with mock.patch.object(api, "daemon_stop", return_value={"search": "stopped"}):
            rc, out, _ = run("daemon", "stop", "search", "--json")
        self.assertEqual(json.loads(out), {"search": "stopped"})
        res = {"stop": {"search": "stopped"}, "start": {"search": "started"}}
        with mock.patch.object(api, "daemon_restart", return_value=res):
            rc, out, _ = run("daemon", "restart", "search")
        self.assertIn("stop search: stopped", out)
        self.assertIn("start search: started", out)
        with mock.patch.object(api, "daemon_start", side_effect=ValueError("unknown daemon 'x'")):
            rc, _, err = run("daemon", "start", "search")
        self.assertNotEqual(rc, 0)
        self.assertIn("unknown daemon", err)

    def test_daemon_run_needs_one_daemon_and_runs_its_module_in_this_process(self):
        rc, _, err = run("daemon", "run")
        self.assertNotEqual(rc, 0)
        self.assertIn("needs 'search' or 'indexer'", err)
        with mock.patch("runpy.run_module") as rm:
            rc, _, _ = run("daemon", "run", "indexer")
        self.assertEqual(rc, 0)
        self.assertEqual(rm.call_args.args[0], "rag_search.core.indexer_daemon")

    def test_service_install_uninstall_and_status(self):
        with mock.patch.object(service, "install", return_value=["wrote a.plist"]), \
                mock.patch.object(service, "uninstall", return_value=["removed a.plist"]), \
                mock.patch.object(service, "status", return_value={
                    "search": {"installed": True, "loaded": True, "running": False}}):
            self.assertIn("wrote a.plist", run("service", "install")[1])
            self.assertIn("removed a.plist", run("service", "uninstall")[1])
            rc, out, _ = run("service", "status")
            self.assertIn("search: installed=True loaded=True running=False", out)
            self.assertTrue(json.loads(run("service", "status", "--json")[1])["search"]["installed"])


class CollectionAndTraceCommandTests(TempHome):
    def setUp(self):
        super().setUp()
        corpus.copy("text/notes.md", self.sdir / "team" / "notes.md")
        corpus.copy("text/readme.txt", self.sdir / "team" / "readme.txt")
        self.index()
        self.publish()

    def test_collection_info_export_import_delete_as_json(self):
        info = json.loads(run("collection", "info", "team", "--json")[1])
        self.assertEqual((info["collection"], info["state"]), ("team", "ok"))
        bundle = self.tmp / "team.rag.tgz"
        res = json.loads(run("collection", "export", "team", "-o", str(bundle), "--json")[1])
        self.assertEqual((res["collection"], res["documents"], Path(res["file"])), ("team", 2, bundle))
        out = run("collection", "import", str(bundle), "--as", "team-copy", "--json")[1]
        self.assertTrue(json.loads(out[out.index("{"):])["ok"])
        out = run("collection", "delete", "team-copy", "-y", "--json")[1]
        self.assertEqual(json.loads(out[out.index("{"):])["result"]["collection"], "team-copy")

    def test_collection_commands_fail_cleanly_for_a_name_or_file_that_does_not_exist(self):
        for argv in (("info", "nope"), ("export", "nope"), ("import", str(self.tmp / "missing.rag.tgz")),
                     ("delete", "nope", "-y")):
            rc, _, err = run("collection", *argv)
            self.assertNotEqual(rc, 0, argv)
            self.assertTrue(err.strip(), argv)
        self.assertNotEqual(run("collection")[0], 0)                  # no sub-command: help, not a traceback

    def test_trace_md_prints_the_converted_text_or_one_page_or_json(self):
        rc, out, _ = run("trace", "team/notes", "--md")
        self.assertEqual(rc, 0)
        self.assertIn("olive horizon notes", out)
        rc, out, _ = run("trace", "team/notes", "--md", "--page", "1")
        self.assertIn("Decisions", out)
        self.assertIn("markdown", json.loads(run("trace", "team/notes", "--md", "--json")[1]))
        cut = {"markdown": "text", "truncated": True, "file": "/x/notes.md", "pages": [1]}
        with mock.patch.object(api, "conversion_markdown", return_value={"ok": True, "result": cut}):
            rc, out, err = run("trace", "team/notes", "--md")
        self.assertIn("text", out)
        self.assertIn("the whole text is in /x/notes.md", err)
        rc, _, err = run("trace", "team/absent", "--md")
        self.assertNotEqual(rc, 0)
        rc, _, err = run("trace", "team")
        self.assertIn("COLLECTION/DOCUMENT", err)

    def test_trace_lists_what_each_page_cost_and_what_was_repaired(self):
        self.source("bank")                                   # a collection is a registered location
        md = self.paths.markup / "bank" / "stmt.md"
        md.parent.mkdir(parents=True)
        md.write_text("<!-- page 1 -->\n\ntext\n\n<!-- page 2 -->\n\nmore", encoding="utf-8")
        src = self.tmp / "stmt.pdf"
        src.write_bytes(b"%PDF-1.4")
        pages = [trace.page_record(1, "raster", "scan", outcome="repaired", out={"chars": 90}),
                 trace.page_record(2, "digital", "text", outcome="low", out={"chars": 30})]
        pages[0].update(tokens=120, repair={"tried": 3, "fixed": 2}, time_s={"read": 1.5}, cache="hit")
        pages[1].update(across="continues", across_with=1, reconcile={"role": "continues", "with": 1})
        meta = self.paths.index / "bank" / "stmt" / "index.meta.json"          # the index of "bank"
        meta.parent.mkdir(parents=True)
        meta.write_text(json.dumps({"conversion": trace.summarize(pages)}))
        trace.write_trace(trace.trace_path_for(md), source="stmt.pdf", src_sha=sha256_file(src), pages=pages,
                          summary=trace.summarize(pages), settings="c5|ocr=force")
        rc, out, err = run("trace", "bank/stmt")
        self.assertEqual(rc, 0, err)
        self.assertIn("120 tokens", out)
        self.assertIn("repair 2/3 cells", out)
        self.assertIn("page cache", out)
        self.assertIn("table continues p.1", out)
        self.assertIn("converted with c5|ocr=force", out)
        # the full record is gone but the summary in index.meta.json is not: the command says so
        trace.trace_path_for(md).unlink()
        rc, out, _ = run("trace", "bank/stmt")
        self.assertEqual(rc, 0)
        self.assertIn("only the summary", out)


class PlaygroundTextOutputTests(TempHome):
    """The human-readable output of every playground command (the JSON forms are in test_playground_cli)."""

    def setUp(self):
        super().setUp()
        self.use_fake_backends()
        self.src = corpus.copy("text/notes.md", self.tmp / "notes" / "notes.md").parent

    def test_an_experiment_from_creation_to_comparison_reads_well(self):
        self.assertIn("no playground experiments yet", run("playground", "list")[1])
        run("playground", "create", "demo", "--from", str(self.src))
        rc, out, _ = run("playground", "list")
        self.assertIn("demo: 1 source folder(s)", out)
        self.assertIn("model: BAAI/bge-m3", out)
        self.assertIn("no runs yet for 'demo'", run("playground", "status", "demo")[1])
        run("playground", "index", "demo")
        rc, out, err = run("playground", "status", "demo")
        self.assertEqual(rc, 0, err)
        self.assertIn("demo", out)
        rc, out, _ = run("playground", "search", "demo", "rotate the keys", "--explain")
        self.assertEqual(rc, 0)
        self.assertIn("1. [", out)
        self.assertIn("notes (notes)", out)
        self.assertIn("models: embedding=", out)
        rc, out, _ = run("playground", "settings", "demo")
        self.assertIn("settings in effect, by pipeline stage", out)
        self.assertIn("[", out)
        self.assertIn("stages", json.loads(run("playground", "settings", "demo", "--json")[1]))
        self.assertIn("no bench runs yet for 'demo'", run("playground", "compare", "demo")[1])
        queries = self.paths.home / "playground" / "demo" / "bench" / "queries.jsonl"
        queries.write_text(json.dumps({"query": "rotate the keys", "relevant": [{"file": "notes.md", "page": "1"}]})
                           + "\n" + json.dumps({"query": "zzz nothing", "relevant": [{"file": "none.md", "page": "9"}]}) + "\n")
        rc, out, err = run("playground", "bench", "demo", "--label", "first")
        self.assertEqual(rc, 0, err)
        self.assertIn("(first)", out)
        self.assertIn("recall@", out)
        self.assertIn("✓ 'rotate the keys'", out)
        self.assertIn("✗ 'zzz nothing'", out)
        rc, out, _ = run("playground", "compare", "demo")
        self.assertIn("RUN", out)
        self.assertIn("first", out)

    def test_unknown_experiments_are_usage_errors_not_tracebacks(self):
        for argv in (("settings", "nope"), ("status", "nope"), ("bench", "nope"), ("search", "nope", "q")):
            rc, _, err = run("playground", *argv)
            self.assertNotEqual(rc, 0, argv)
            self.assertTrue(err.strip(), argv)

    def test_promote_says_what_changes_and_whether_a_reindex_follows(self):
        from rag_search import config

        config.update_config(self.paths, "models", {"embedding": "prod/embed"})
        run("playground", "create", "demo", "--from-production")
        self.assertIn("already matches production", run("playground", "promote", "demo")[1])
        run("playground", "config", "demo", "--rrf-k", "30")
        rc, out, _ = run("playground", "promote", "demo", "--dry-run")
        self.assertIn("would change", out)
        self.assertIn("rrf_k", out)
        self.assertIn("no reindex needed", run("playground", "promote", "demo", "--confirm")[1]
                      .replace("changed:", "changed:"))


class IndexAndLocationTextTests(TempHome):
    def setUp(self):
        super().setUp()
        self.use_fake_backends()

    def test_index_foreground_reports_what_it_could_not_index(self):
        corpus.copy("text/notes.md", self.sdir / "mix" / "notes.md")
        corpus.copy("pdf/damaged.pdf", self.sdir / "mix" / "damaged.pdf")
        corpus.copy("unsupported/notes.rtf", self.sdir / "mix" / "notes.rtf")
        corpus.copy("text/empty.txt", self.sdir / "mix" / "empty.txt")
        self.register_tree()
        rc, out, _ = run("index", "foreground")
        self.assertIn("! ", out)                                       # the damaged PDF: an error line
        self.assertIn("damaged.pdf", out)
        self.assertIn("unsupported format (.rtf)", out)
        rc, out, _ = run("index", "foreground", "--json")
        res = json.loads(out[out.index("{"):])
        self.assertEqual(res["summary"]["indexed"], 0)                 # nothing changed since the first run
        self.assertTrue(res["summary"]["unsupported_extension"])

    def test_index_foreground_refuses_a_path_that_is_nothing(self):
        rc, _, err = run("index", "foreground", str(self.tmp / "nowhere"))
        self.assertNotEqual(rc, 0)
        self.assertIn("error:", err)

    def test_index_publish_prints_the_generation_or_says_nothing_changed(self):
        corpus.copy("text/notes.md", self.sdir / "mix" / "notes.md")
        self.index()
        with mock.patch.object(api, "index_publish", return_value={
                "ok": True, "publish": {"changed": True, "generation": 4}, "search_reload": {"ok": True}}):
            rc, out, _ = run("index", "publish")
        self.assertIn("published generation 4", out)
        self.assertIn("search daemon:", out)
        with mock.patch.object(api, "index_publish", return_value={"ok": True, "publish": {"changed": False}}):
            self.assertEqual(json.loads(run("index", "publish", "--json")[1])["publish"]["changed"], False)
        with mock.patch.object(api, "index_publish", return_value={"ok": False, "error": "a run is active"}):
            rc, _, err = run("index", "publish")
        self.assertNotEqual(rc, 0)
        self.assertIn("a run is active", err)

    def test_location_list_add_remove_in_text_and_json(self):
        self.assertIn("no registered locations", run("location")[1])
        folder = self.tmp / "shared"
        corpus.copy("text/notes.md", folder / "notes.md")
        rc, out, err = run("location", "add", "shared", str(folder))
        self.assertEqual(rc, 0, err)
        self.assertIn("now indexes", out)
        self.assertIn("read only", out)
        rc, out, _ = run("location", "list")
        self.assertIn("shared", out)
        self.assertIn("[ok]", out)
        listed = json.loads(run("location", "list", "--json")[1])
        self.assertEqual(listed["locations"][0]["collection"], "shared")
        rc, out, _ = run("location", "remove", "shared", "-y")
        self.assertIn("removed location 'shared'", out)
        rc, out, err = run("location", "add", "x", str(self.tmp / "missing-folder"), "--json")
        self.assertNotEqual(rc, 0)

    def test_a_damaged_locations_file_is_a_warning_in_the_listing(self):
        (self.paths.home / "locations.json").write_text("{ nope")
        rc, out, err = run("location", "list")
        self.assertEqual(rc, 0)
        self.assertIn("warning:", err)


class ConvertAndMiscCommandTests(TempHome):
    def test_convert_writes_markdown_next_to_the_current_folder_and_says_how_long(self):
        out_file = self.tmp / "out.md"
        info = {"pages": 3, "seconds": 1.5, "ocr": "auto"}
        with mock.patch("rag_search.core.docling_convert.convert_file", return_value=info) as cf:
            rc, out, _ = run("convert", str(corpus.path("pdf/text.pdf")), "-o", str(out_file), "--ocr",
                             "--mode", "smart", "--backend", "docling-parse")
        self.assertEqual(rc, 0)
        self.assertIn(f"wrote {out_file} (3 page(s), 1.5s, OCR auto)", out)
        self.assertTrue(cf.call_args.kwargs["ocr"])
        self.assertEqual(os.environ.get("RAG_SEARCH_OCR"), "smart")

    def test_convert_compare_times_the_variants_and_says_which_text_differs(self):
        import subprocess

        calls = []

        def fake_run(cmd, **kw):
            out = Path(cmd[-1])
            calls.append(out.name)
            out.write_text("same text\n" if len(calls) < 3 else "different text\n", encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0 if len(calls) != 2 else 1, "", "boom: it failed")

        with mock.patch("subprocess.run", side_effect=fake_run):
            rc, out, err = run("convert", str(corpus.path("pdf/text.pdf")), "--compare", "-o", str(self.tmp / "cmp"))
        self.assertEqual(rc, 0, err)
        self.assertIn("current settings", out)
        self.assertIn("failed: boom: it failed", out)
        self.assertIn("Outputs are in", out)
        with mock.patch("subprocess.run", side_effect=fake_run):
            rc, out, _ = run("convert", str(corpus.path("pdf/text.pdf")), "--compare", "-o", str(self.tmp / "cmp2"),
                             "--json")
        self.assertIn("variants", json.loads(out))

    def test_doctor_json_lists_every_check_and_fails_only_on_a_failed_one(self):
        from rag_search.core import diagnostics

        rows = [(diagnostics.OK, "python", "3.12"), (diagnostics.WARN, "daemon", "not running")]
        with mock.patch.object(diagnostics, "run_checks", return_value=rows):
            rc, out, _ = run("doctor", "--json")
        self.assertEqual((rc, [r["check"] for r in json.loads(out)]), (0, ["python", "daemon"]))
        with mock.patch.object(diagnostics, "run_checks", return_value=rows + [(diagnostics.FAIL, "x", "bad")]):
            self.assertEqual(run("doctor", "--json")[0], 1)

    def test_serve_says_when_the_mcp_adapter_is_missing(self):
        import builtins

        real = builtins.__import__

        def no_mcp(name, *a, **k):
            if name.endswith("mcp.server"):
                raise ImportError("No module named 'mcp'")
            return real(name, *a, **k)

        with mock.patch("builtins.__import__", side_effect=no_mcp):
            rc, _, err = run("serve")
        self.assertNotEqual(rc, 0)
        self.assertIn("MCP adapter is not installed", err)

    def test_ui_takes_its_port_from_the_environment_when_none_is_given(self):
        os.environ["RAG_SEARCH_UI_PORT"] = "8123"
        self.addCleanup(os.environ.pop, "RAG_SEARCH_UI_PORT", None)
        with mock.patch("rag_search.ui.server.run", return_value=0) as srv:
            rc, _, _ = run("ui", "--no-browser")
        self.assertEqual((rc, srv.call_args.kwargs["port"], srv.call_args.kwargs["open_browser"]),
                         (0, 8123, False))
