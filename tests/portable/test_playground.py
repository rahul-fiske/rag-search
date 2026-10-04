"""core/playground.py: structural isolation from production, build/search/bench correctness."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest import mock

from rag_search.paths import META_FILE, index_dir_for, read_json
from tests.helpers import TempHome


class PlaygroundBase(TempHome):
    def setUp(self):
        super().setUp()
        self.use_fake_backends()

    def make_source(self, name: str, text: str) -> Path:
        p = self.tmp / name
        p.write_text(text, encoding="utf-8")
        return p


class LifecycleTests(PlaygroundBase):
    def test_create_index_search_roundtrip(self):
        from rag_search.core import playground as pg

        src = self.make_source("doc1.txt", "<!-- page 1 -->\nhow is a session token refreshed\n"
                                            "<!-- page 2 -->\nsomething unrelated about pandas\n")
        info = pg.create_experiment(self.paths, "exp1", sources=[str(src)])
        self.assertEqual(info["documents_added"], 1)

        summary = pg.build_index(self.paths, "exp1")
        self.assertEqual(summary["indexed"], 1)
        self.assertEqual(summary["errors"], [])

        res = pg.search(self.paths, "exp1", "session token refreshed", top_k=3)
        self.assertTrue(res["results"])
        self.assertEqual(res["results"][0]["source"], "doc1.txt")
        self.assertEqual(res["models"]["embedding"], "BAAI/bge-m3")   # the experiment's default

    def test_invalid_experiment_name_rejected(self):
        from rag_search.core import playground as pg

        for bad in ("../evil", "a/b", "", ".", ".."):
            with self.assertRaises(ValueError):
                pg.create_experiment(self.paths, bad)

    def test_build_without_docs_raises(self):
        from rag_search.core import playground as pg

        pg.create_experiment(self.paths, "empty")
        with self.assertRaises(pg.PlaygroundError):
            pg.build_index(self.paths, "empty")

    def test_search_without_index_raises(self):
        from rag_search.core import playground as pg

        pg.create_experiment(self.paths, "noindex")
        with self.assertRaises(pg.PlaygroundError):
            pg.search(self.paths, "noindex", "anything")

    def test_list_and_remove(self):
        from rag_search.core import playground as pg

        src = self.make_source("doc2.txt", "<!-- page 1 -->\nhello world\n")
        pg.create_experiment(self.paths, "exp2", sources=[str(src)])
        pg.build_index(self.paths, "exp2")

        listed = pg.list_experiments(self.paths)
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["name"], "exp2")
        self.assertEqual(listed[0]["documents"], 1)
        self.assertIn("sample", listed[0]["indexed_collections"])

        pg.remove_experiment(self.paths, "exp2")
        self.assertEqual(pg.list_experiments(self.paths), [])
        with self.assertRaises(pg.PlaygroundError):
            pg.remove_experiment(self.paths, "exp2")


class RebuildVsForceMdTests(PlaygroundBase):
    """build_index()'s `rebuild` and `force_md` are two independent knobs, same as production's
    `run_index()` (see tests.portable.test_indexer_extra's test_force_md_reconverts_even_when_the_index_is_
    fresh): `rebuild` alone only bypasses the chunk/embed freshness check -- step 1 (docling
    conversion) still reuses the cached Markdown via core/indexer.py's convert_source() sidecar
    check whenever the source file and the current docling settings profile both still match what
    produced it. Only `force_md` (the dashboard's separate "re-convert to Markdown" checkbox,
    added to playground.js alongside "re-embed even if unchanged") forces that step to redo too."""

    def test_rebuild_alone_reuses_the_converted_markdown(self):
        from rag_search.core import playground as pg
        from rag_search.paths import get_playground_paths

        src = self.make_source("a.txt", "<!-- page 1 -->\noriginal text\n")
        pg.create_experiment(self.paths, "reb", sources=[str(src)])
        pg.build_index(self.paths, "reb")
        exp = get_playground_paths(self.paths, "reb")
        md = exp.markup / "sample" / "a.md"
        md.write_text("<!-- page 1 -->\nstale text (simulating an out-of-date conversion)\n",
                      encoding="utf-8")

        s = pg.build_index(self.paths, "reb", rebuild=True)
        self.assertEqual((s["indexed"], s["skipped_fresh"]), (1, 0))
        self.assertEqual(md.read_text(encoding="utf-8"),
                         "<!-- page 1 -->\nstale text (simulating an out-of-date conversion)\n")

    def test_force_md_reconverts_even_with_rebuild_alone_would_reuse(self):
        from rag_search.core import playground as pg
        from rag_search.paths import get_playground_paths

        src = self.make_source("b.txt", "<!-- page 1 -->\noriginal text\n")
        pg.create_experiment(self.paths, "fmd", sources=[str(src)])
        pg.build_index(self.paths, "fmd")
        exp = get_playground_paths(self.paths, "fmd")
        md = exp.markup / "sample" / "b.md"
        md.write_text("<!-- page 1 -->\nstale text\n", encoding="utf-8")

        s = pg.build_index(self.paths, "fmd", rebuild=True, force_md=True)
        self.assertEqual((s["indexed"], s["skipped_fresh"]), (1, 0))
        self.assertEqual(md.read_text(encoding="utf-8"), src.read_text(encoding="utf-8"))


class IsolationTests(PlaygroundBase):
    def test_production_untouched_by_playground(self):
        """Building and searching a playground experiment must never write to serving/, run/,
        or the production config.json -- the core promise of the feature."""
        from rag_search.core import playground as pg

        # a real production index + publish, so there's something to disturb
        self.write_doc("prod/real.txt", "<!-- page 1 -->\nproduction content\n")
        self.index()
        self.publish()
        before_serving = sorted(str(p) for p in self.paths.serving.rglob("*"))
        before_current = (self.paths.serving / "current").resolve()
        self.paths.config_file.write_text(json.dumps({"models": {}}), encoding="utf-8")
        before_config = self.paths.config_file.read_text(encoding="utf-8")
        before_run = sorted(str(p) for p in self.paths.run.rglob("*")) if self.paths.run.exists() else []

        src = self.make_source("pgdoc.txt", "<!-- page 1 -->\nplayground content\n")
        pg.create_experiment(self.paths, "sandbox", sources=[str(src)])
        pg.update_config(self.paths, "sandbox", embedding_model="Qwen/Qwen3-Embedding-0.6B")
        pg.build_index(self.paths, "sandbox")
        pg.search(self.paths, "sandbox", "playground content")

        after_serving = sorted(str(p) for p in self.paths.serving.rglob("*"))
        after_current = (self.paths.serving / "current").resolve()
        after_run = sorted(str(p) for p in self.paths.run.rglob("*")) if self.paths.run.exists() else []
        self.assertEqual(before_serving, after_serving)
        self.assertEqual(before_current, after_current)
        self.assertEqual(before_config, self.paths.config_file.read_text(encoding="utf-8"))
        self.assertEqual(before_run, after_run)   # not one new socket/lock/pid file in production run/
        # and the playground data is its own tree, not mixed into production dirs
        self.assertTrue((self.paths.home / "playground" / "sandbox").is_dir())
        self.assertNotIn("playground", [p.name for p in self.paths.index.iterdir()]
                          if self.paths.index.is_dir() else [])


class ConfigTests(PlaygroundBase):
    def test_per_experiment_model_is_recorded_and_does_not_leak(self):
        from rag_search.core import playground as pg

        src1 = self.make_source("a.txt", "<!-- page 1 -->\nalpha\n")
        src2 = self.make_source("b.txt", "<!-- page 1 -->\nbeta\n")
        pg.create_experiment(self.paths, "modelA", sources=[str(src1)])
        pg.update_config(self.paths, "modelA", embedding_model="model-a")
        pg.create_experiment(self.paths, "modelB", sources=[str(src2)])
        pg.update_config(self.paths, "modelB", embedding_model="model-b")

        pg.build_index(self.paths, "modelA")
        pg.build_index(self.paths, "modelB")

        exp_a = pg.get_config(self.paths, "modelA")
        exp_b = pg.get_config(self.paths, "modelB")
        self.assertEqual(exp_a["embedding_model"], "model-a")
        self.assertEqual(exp_b["embedding_model"], "model-b")

        from rag_search.paths import get_playground_paths
        pa = get_playground_paths(self.paths, "modelA")
        meta_a = read_json(index_dir_for(pa.docs / "sample" / "a.txt", pa.docs, pa.index)
                            / META_FILE)
        self.assertEqual(meta_a["model"], "model-a")
        pb = get_playground_paths(self.paths, "modelB")
        meta_b = read_json(index_dir_for(pb.docs / "sample" / "b.txt", pb.docs, pb.index)
                            / META_FILE)
        self.assertEqual(meta_b["model"], "model-b")

    def test_invalid_stage_override_rejected(self):
        from rag_search.core import playground as pg

        src = self.make_source("c.txt", "<!-- page 1 -->\nsomething\n")
        pg.create_experiment(self.paths, "badstage", sources=[str(src)])
        with self.assertRaises(ValueError):
            pg.update_config(self.paths, "badstage", stages="rerank")


class SearchDebugTests(PlaygroundBase):
    def setUp(self):
        super().setUp()
        from rag_search.core import playground as pg

        self.pg = pg
        src = self.make_source("doc.txt", "<!-- page 1 -->\nfirst page about apples\n"
                                           "<!-- page 2 -->\nsecond page about oranges\n")
        pg.create_experiment(self.paths, "dbg", sources=[str(src)])
        pg.build_index(self.paths, "dbg")

    def test_stage_override_threads_through(self):
        res = self.pg.search(self.paths, "dbg", "apples", stages="bm25")
        self.assertTrue(res["results"])
        for r in res["results"]:
            self.assertIsNone(r["dense_score"])

    def test_pool_and_rrf_overrides_accepted(self):
        res = self.pg.search(self.paths, "dbg", "oranges", retrieval_pool_n=50, rerank_pool_n=10,
                              rrf_k=30)
        self.assertEqual(res["timing"]["rrf_k"], 30)


class BenchTests(PlaygroundBase):
    def setUp(self):
        super().setUp()
        from rag_search.core import playground as pg

        self.pg = pg
        src = self.make_source("kb.txt", "<!-- page 1 -->\nhow is a session token refreshed\n"
                                          "<!-- page 2 -->\nhow to reset a forgotten password\n")
        pg.create_experiment(self.paths, "bench1", sources=[str(src)])
        pg.build_index(self.paths, "bench1")

    def _write_queries(self, name: str, rows: list[dict]) -> Path:
        from rag_search.paths import get_playground_paths
        exp = get_playground_paths(self.paths, name)
        qpath = exp.home / "bench" / "queries.jsonl"
        qpath.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        return qpath

    def test_bench_missing_queries_raises(self):
        with self.assertRaises(self.pg.PlaygroundError):
            self.pg.bench(self.paths, "bench1")

    def test_bench_perfect_recall(self):
        self._write_queries("bench1", [
            {"query": "session token refreshed", "relevant": [{"file": "kb.txt", "page": "1"}]},
            {"query": "forgotten password reset", "relevant": [{"file": "kb.txt", "page": "2"}]},
        ])
        record = self.pg.bench(self.paths, "bench1", k=3, label="baseline")
        self.assertEqual(record["metrics"]["recall_at_k"], 1.0)
        self.assertEqual(record["metrics"]["mrr"], 1.0)
        self.assertEqual(record["n_queries"], 2)
        self.assertEqual(record["label"], "baseline")

        runs = self.pg.compare(self.paths, "bench1")
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["run_id"], record["run_id"])
        self.assertNotIn("per_query", runs[0])

    def test_bench_records_zero_when_nothing_matches(self):
        self._write_queries("bench1", [
            {"query": "session token refreshed", "relevant": [{"file": "nope.pdf", "page": "9"}]},
        ])
        record = self.pg.bench(self.paths, "bench1", k=3)
        self.assertEqual(record["metrics"]["recall_at_k"], 0.0)
        self.assertEqual(record["metrics"]["mrr"], 0.0)


class ProductionBridgeTests(PlaygroundBase):
    """"copy from production" (create_experiment(from_production=True)) and its reverse,
    promote_to_production() -- both go through production_snapshot() so they agree on what
    "production's settings" means."""

    def setUp(self):
        super().setUp()
        from rag_search import config
        from rag_search.core import playground as pg

        self.pg = pg
        self.config = config
        config.update_config(self.paths, "models", {"embedding": "prod/embed", "reranker": "prod/rerank"})
        config.update_config(self.paths, "indexer", {"chunk_size": 700, "chunk_overlap": 70})
        config.update_config(self.paths, "search",
                             {"stages": "bm25,dense", "retrieval_pool": 40, "rerank_pool": 12, "rrf_k": 50})

    def test_snapshot_reflects_effective_production_settings(self):
        snap = self.pg.production_snapshot(self.paths)
        self.assertEqual(snap["embedding_model"], "prod/embed")
        self.assertEqual(snap["rerank_model"], "prod/rerank")
        self.assertEqual(snap["chunk_size"], 700)
        self.assertEqual(snap["chunk_overlap"], 70)
        self.assertEqual(snap["stages"], "bm25,dense")
        self.assertEqual(snap["retrieval_pool"], 40)
        self.assertEqual(snap["rerank_pool"], 12)
        self.assertEqual(snap["rrf_k"], 50)

    def test_create_from_production_seeds_the_experiment(self):
        info = self.pg.create_experiment(self.paths, "seeded", from_production=True)
        self.assertTrue(info["from_production"])
        cfg = info["config"]
        self.assertEqual(cfg["embedding_model"], "prod/embed")
        self.assertEqual(cfg["chunk_size"], 700)
        self.assertEqual(cfg["stages"], "bm25,dense")
        self.assertEqual(self.pg.get_config(self.paths, "seeded"), cfg)

    def test_create_without_from_production_keeps_module_defaults(self):
        info = self.pg.create_experiment(self.paths, "plain")
        self.assertFalse(info["from_production"])
        self.assertEqual(info["config"]["embedding_model"], "BAAI/bge-m3")

    def test_preview_reports_only_actual_differences(self):
        self.pg.create_experiment(self.paths, "exp", from_production=True)
        preview = self.pg.promotion_preview(self.paths, "exp")
        self.assertEqual(preview["changes"], {})
        self.assertFalse(preview["needs_reindex"])

        self.pg.update_config(self.paths, "exp", rerank_pool=99)
        preview = self.pg.promotion_preview(self.paths, "exp")
        self.assertEqual(set(preview["changes"]), {"rerank_pool"})
        self.assertFalse(preview["needs_reindex"])   # a search tunable, not a model/chunk change

    def test_embedding_model_change_needs_reindex_and_a_cost_estimate(self):
        self.pg.create_experiment(self.paths, "exp2", from_production=True)
        self.pg.update_config(self.paths, "exp2", embedding_model="new/model")
        preview = self.pg.promotion_preview(self.paths, "exp2")
        self.assertTrue(preview["needs_reindex"])
        self.assertIn("documents", preview["reindex_estimate"])

    def test_promotion_needing_reindex_is_refused_without_confirm(self):
        self.pg.create_experiment(self.paths, "exp3", from_production=True)
        self.pg.update_config(self.paths, "exp3", chunk_size=999)
        with self.assertRaises(self.pg.PlaygroundError):
            self.pg.promote_to_production(self.paths, "exp3")
        # nothing was written
        cfg, _ = self.config.load_config(self.paths)
        self.assertEqual(cfg["indexer"]["chunk_size"], 700)

    def test_confirmed_promotion_writes_production_config(self):
        self.pg.create_experiment(self.paths, "exp4", from_production=True)
        self.pg.update_config(self.paths, "exp4", embedding_model="new/model", chunk_size=999,
                              rrf_k=77)
        res = self.pg.promote_to_production(self.paths, "exp4", confirm=True)
        self.assertEqual(set(res["changes"]), {"embedding_model", "chunk_size", "rrf_k"})
        cfg, _ = self.config.load_config(self.paths)
        self.assertEqual(cfg["models"]["embedding"], "new/model")
        self.assertEqual(cfg["models"]["reranker"], "prod/rerank")   # untouched: didn't change
        self.assertEqual(cfg["indexer"]["chunk_size"], 999)
        self.assertEqual(cfg["indexer"]["chunk_overlap"], 70)        # untouched: didn't change
        self.assertEqual(cfg["search"]["rrf_k"], 77)
        self.assertEqual(cfg["search"]["stages"], "bm25,dense")      # untouched: didn't change

    def test_promotion_with_no_differences_is_a_no_op(self):
        self.pg.create_experiment(self.paths, "exp5", from_production=True)
        res = self.pg.promote_to_production(self.paths, "exp5")
        self.assertEqual(res["changes"], {})
        self.assertFalse(res["needs_reindex"])

    def test_snapshot_includes_effective_docling_tunables(self):
        from rag_search import config

        config.update_config(self.paths, "indexer", {"ocr": "smart", "table_mode": "fast"})
        os.environ["RAG_SEARCH_DOC_TIMEOUT"] = "900"   # an ambient env var beats config.json
        try:
            snap = self.pg.production_snapshot(self.paths)
        finally:
            os.environ.pop("RAG_SEARCH_DOC_TIMEOUT", None)
        self.assertEqual(snap["ocr"], "smart")
        self.assertEqual(snap["table_mode"], "fast")
        self.assertEqual(snap["doc_timeout"], "900")
        self.assertEqual(snap["ocr_engine"], "")   # never set anywhere -- stays blank

    def test_create_from_production_seeds_docling_tunables(self):
        from rag_search import config

        config.update_config(self.paths, "indexer", {"ocr": "off", "pdf_backend": "docling-parse"})
        info = self.pg.create_experiment(self.paths, "docling_seeded", from_production=True)
        self.assertEqual(info["config"]["ocr"], "off")
        self.assertEqual(info["config"]["pdf_backend"], "docling-parse")


class DoclingConfigTests(PlaygroundBase):
    """update_config() validates/normalises the docling/OCR/table/PDF-backend tunables through
    spec.py's shared registry -- same rules `rag-search config set` and the Settings tab use."""

    def test_defaults_are_blank_or_zero(self):
        from rag_search.core import playground as pg

        pg.create_experiment(self.paths, "blank")
        cfg = pg.get_config(self.paths, "blank")
        for key in ("ocr", "ocr_engine", "ocr_lang", "table_mode", "pdf_backend", "pipeline"):
            self.assertEqual(cfg[key], "", key)
        self.assertEqual(cfg["doc_timeout"], 0)
        self.assertEqual(cfg["docling_batch"], 0)

    def test_accepts_and_stores_valid_values(self):
        from rag_search.core import playground as pg

        pg.create_experiment(self.paths, "valid")
        cfg = pg.update_config(self.paths, "valid", ocr="smart", ocr_engine="rapidocr",
                               ocr_lang="en,fr", table_mode="fast", pdf_backend="docling-parse",
                               pipeline="vlm", doc_timeout=120, docling_batch=16)
        self.assertEqual(cfg["ocr"], "smart")
        self.assertEqual(cfg["ocr_engine"], "rapidocr")
        self.assertEqual(cfg["ocr_lang"], "en,fr")
        self.assertEqual(cfg["table_mode"], "fast")
        self.assertEqual(cfg["pdf_backend"], "docling-parse")
        self.assertEqual(cfg["pipeline"], "vlm")
        self.assertEqual(cfg["doc_timeout"], 120)
        self.assertEqual(cfg["docling_batch"], 16)
        # persisted, not just returned
        self.assertEqual(pg.get_config(self.paths, "valid"), cfg)

    def test_choice_values_are_case_insensitive(self):
        from rag_search.core import playground as pg

        pg.create_experiment(self.paths, "caseit")
        cfg = pg.update_config(self.paths, "caseit", ocr="SMART")
        self.assertEqual(cfg["ocr"], "smart")

    def test_invalid_choice_rejected(self):
        from rag_search.core import playground as pg

        pg.create_experiment(self.paths, "badocr")
        with self.assertRaises(ValueError):
            pg.update_config(self.paths, "badocr", ocr="bogus")

    def test_invalid_int_rejected(self):
        from rag_search.core import playground as pg

        pg.create_experiment(self.paths, "badtimeout")
        with self.assertRaises(ValueError):
            pg.update_config(self.paths, "badtimeout", doc_timeout=-5)

    def test_blank_clears_back_to_no_override(self):
        from rag_search.core import playground as pg

        pg.create_experiment(self.paths, "clearit")
        pg.update_config(self.paths, "clearit", ocr="smart", doc_timeout=60)
        cfg = pg.update_config(self.paths, "clearit", ocr="", doc_timeout=0)
        self.assertEqual(cfg["ocr"], "")
        self.assertEqual(cfg["doc_timeout"], 0)


class DoclingEnvTests(PlaygroundBase):
    """build_index() applies an experiment's docling tunables as environment variables scoped to
    that one call -- same precedence core/indexer_daemon.py's _worker_env uses for production."""

    def setUp(self):
        super().setUp()
        from rag_search.core import playground as pg

        self.pg = pg
        self.src = self.make_source("d.txt", "<!-- page 1 -->\nsome content\n")

    def test_env_applied_during_build_and_restored_after(self):
        pg = self.pg
        pg.create_experiment(self.paths, "envtest", sources=[str(self.src)])
        pg.update_config(self.paths, "envtest", ocr="smart", table_mode="fast", doc_timeout=45)
        self.assertNotIn("RAG_SEARCH_OCR", os.environ)

        seen = {}

        def fake_run_index(*a, **kw):
            seen["OCR"] = os.environ.get("RAG_SEARCH_OCR")
            seen["TABLE_MODE"] = os.environ.get("RAG_SEARCH_TABLE_MODE")
            seen["DOC_TIMEOUT"] = os.environ.get("RAG_SEARCH_DOC_TIMEOUT")
            return {"indexed": 0, "errors": []}

        with mock.patch.object(pg, "run_index", side_effect=fake_run_index):
            pg.build_index(self.paths, "envtest")

        self.assertEqual(seen["OCR"], "smart")
        self.assertEqual(seen["TABLE_MODE"], "fast")
        self.assertEqual(seen["DOC_TIMEOUT"], "45")
        # scoped to that one call -- nothing leaks into the ambient environment afterward
        self.assertNotIn("RAG_SEARCH_OCR", os.environ)
        self.assertNotIn("RAG_SEARCH_TABLE_MODE", os.environ)
        self.assertNotIn("RAG_SEARCH_DOC_TIMEOUT", os.environ)

    def test_blank_config_touches_no_env_vars(self):
        pg = self.pg
        pg.create_experiment(self.paths, "notunables", sources=[str(self.src)])

        seen = {}

        def fake_run_index(*a, **kw):
            seen["OCR"] = os.environ.get("RAG_SEARCH_OCR")
            return {"indexed": 0, "errors": []}

        with mock.patch.object(pg, "run_index", side_effect=fake_run_index):
            pg.build_index(self.paths, "notunables")

        self.assertIsNone(seen["OCR"])

    def test_ambient_env_var_always_wins_over_experiment_config(self):
        """Same precedence as production's _worker_env: a real ambient env var this process was
        already started with is never overridden by an experiment's own config.json."""
        pg = self.pg
        pg.create_experiment(self.paths, "ambient", sources=[str(self.src)])
        pg.update_config(self.paths, "ambient", ocr="smart")

        seen = {}

        def fake_run_index(*a, **kw):
            seen["OCR"] = os.environ.get("RAG_SEARCH_OCR")
            return {"indexed": 0, "errors": []}

        os.environ["RAG_SEARCH_OCR"] = "off"
        try:
            with mock.patch.object(pg, "run_index", side_effect=fake_run_index):
                pg.build_index(self.paths, "ambient")
        finally:
            os.environ.pop("RAG_SEARCH_OCR", None)

        self.assertEqual(seen["OCR"], "off")
