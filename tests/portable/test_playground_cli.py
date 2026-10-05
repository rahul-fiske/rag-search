"""`rag-search playground ...`: CLI surface over core/playground.py."""

from __future__ import annotations

import json

from tests.helpers import TempHome
from tests.portable.test_cli_api import run


class PlaygroundCliTests(TempHome):
    def setUp(self):
        super().setUp()
        self.use_fake_backends()
        self.src = self.tmp / "docs"
        self.src.mkdir()
        (self.src / "doc.txt").write_text("<!-- page 1 -->\nhow is a session token refreshed\n"
                            "<!-- page 2 -->\nhow to reset a forgotten password\n",
                            encoding="utf-8")

    def test_create_index_search_json_flow(self):
        rc, out, err = run("playground", "create", "demo", "--from", str(self.src), "--json")
        self.assertEqual(rc, 0, err)
        self.assertEqual([x["collection"] for x in json.loads(out)["sources"]], ["docs"])

        rc, out, err = run("playground", "index", "demo", "--json")
        self.assertEqual(rc, 0, err)
        self.assertEqual(json.loads(out)["indexed"], 1)

        rc, out, err = run("playground", "search", "demo", "session token refreshed", "--json")
        self.assertEqual(rc, 0, err)
        res = json.loads(out)
        self.assertTrue(res["results"])
        self.assertEqual(res["results"][0]["source"], "doc.txt")

    def test_search_without_index_fails_cleanly(self):
        run("playground", "create", "demo", "--from", str(self.src))
        rc, out, err = run("playground", "search", "demo", "anything")
        self.assertNotEqual(rc, 0)
        self.assertIn("no index built yet", err)

    def test_config_show_and_update(self):
        run("playground", "create", "demo", "--from", str(self.src))
        rc, out, _ = run("playground", "config", "demo", "--json")
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(out)["embedding_model"], "BAAI/bge-m3")

        rc, out, _ = run("playground", "config", "demo", "--embedding-model", "custom/id",
                         "--no-rerank", "--json")
        self.assertEqual(rc, 0)
        cfg = json.loads(out)
        self.assertEqual(cfg["embedding_model"], "custom/id")
        self.assertFalse(cfg["rerank"])

    def test_config_docling_tunables_show_and_update(self):
        """`playground config` exposes the same docling/OCR/table/PDF-backend flags as
        `rag-search config set`, generated from the same spec.py registry (cli.py)."""
        run("playground", "create", "demo", "--from", str(self.src))
        rc, out, err = run("playground", "config", "demo", "--ocr", "smart",
                           "--table-mode", "fast", "--doc-timeout", "120", "--json")
        self.assertEqual(rc, 0, err)
        cfg = json.loads(out)
        self.assertEqual(cfg["ocr"], "smart")
        self.assertEqual(cfg["table_mode"], "fast")
        self.assertEqual(cfg["doc_timeout"], 120)
        self.assertEqual(cfg["ocr_engine"], "")   # untouched -- still the blank "no override" default

    def test_config_invalid_docling_choice_is_a_clean_error(self):
        run("playground", "create", "demo", "--from", str(self.src))
        rc, out, err = run("playground", "config", "demo", "--ocr", "bogus")
        self.assertNotEqual(rc, 0)
        self.assertIn("OCR mode must be one of", err)

    def test_bad_experiment_name_is_a_clean_error(self):
        rc, out, err = run("playground", "create", "../evil")
        self.assertNotEqual(rc, 0)
        self.assertIn("invalid experiment name", err)

    def test_bench_and_compare(self):
        run("playground", "create", "demo", "--from", str(self.src))
        run("playground", "index", "demo")
        exp_home = self.paths.home / "playground" / "demo"
        (exp_home / "bench" / "queries.jsonl").write_text(
            json.dumps({"query": "session token refreshed",
                       "relevant": [{"file": "doc.txt", "page": "1"}]}) + "\n",
            encoding="utf-8")

        rc, out, err = run("playground", "bench", "demo", "--label", "baseline", "--json")
        self.assertEqual(rc, 0, err)
        rec = json.loads(out)
        self.assertEqual(rec["metrics"]["recall_at_k"], 1.0)

        rc, out, err = run("playground", "compare", "demo", "--json")
        self.assertEqual(rc, 0, err)
        runs = json.loads(out)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["label"], "baseline")
        self.assertNotIn("per_query", runs[0])

    def test_rm_requires_yes(self):
        run("playground", "create", "demo", "--from", str(self.src))
        rc, _, err = run("playground", "rm", "demo")
        self.assertNotEqual(rc, 0)
        self.assertIn("--yes", err)

        rc, out, _ = run("playground", "list", "--json")
        self.assertEqual(len(json.loads(out)), 1)

        rc, _, _ = run("playground", "rm", "demo", "--yes")
        self.assertEqual(rc, 0)
        rc, out, _ = run("playground", "list", "--json")
        self.assertEqual(json.loads(out), [])

    def test_no_subcommand_shows_help(self):
        rc, out, err = run("playground")
        self.assertEqual(rc, 0)
        self.assertIn("playground", out + err)

    def test_production_search_is_a_separate_command_from_playground_search(self):
        """`rag-search search` (production, via the daemon) and `rag-search playground search`
        (sandbox, in-process) must never be confused with each other at the CLI layer."""
        rc, _, err = run("playground", "search", "no-such-experiment", "q")
        self.assertNotEqual(rc, 0)
        self.assertIn("no such experiment", err)

    def test_create_from_production_and_promote_round_trip(self):
        from rag_search import config

        config.update_config(self.paths, "models", {"embedding": "prod/embed"})
        config.update_config(self.paths, "indexer", {"chunk_size": 700})

        rc, out, err = run("playground", "create", "demo", "--from-production", "--json")
        self.assertEqual(rc, 0, err)
        cfg = json.loads(out)["config"]
        self.assertEqual(cfg["embedding_model"], "prod/embed")
        self.assertEqual(cfg["chunk_size"], 700)

        run("playground", "config", "demo", "--chunk-size", "900")

        rc, out, err = run("playground", "promote", "demo", "--dry-run", "--json")
        self.assertEqual(rc, 0, err)
        preview = json.loads(out)
        self.assertEqual(set(preview["changes"]), {"chunk_size"})
        self.assertTrue(preview["needs_reindex"])

        rc, out, err = run("playground", "promote", "demo")
        self.assertNotEqual(rc, 0)
        self.assertIn("confirm=true", err)

        rc, out, err = run("playground", "promote", "demo", "--confirm", "--json")
        self.assertEqual(rc, 0, err)
        res = json.loads(out)
        self.assertEqual(set(res["changes"]), {"chunk_size"})

        cfg2, _ = config.load_config(self.paths)
        self.assertEqual(cfg2["indexer"]["chunk_size"], 900)

    def test_promote_with_no_differences_is_a_clean_no_op(self):
        run("playground", "create", "demo")
        rc, out, err = run("playground", "promote", "demo", "--json")
        self.assertEqual(rc, 0, err)
        self.assertEqual(json.loads(out)["changes"], {})
