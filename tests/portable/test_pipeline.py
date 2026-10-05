import time
import unittest
from pathlib import Path

from tests.helpers import FakeEmbedder, TempHome
from rag_search.core import indexer
from rag_search.core.search import ModelMismatch, SearchEngine
from rag_search.grep import grep_markup
from rag_search.paths import ALL_DIR, EMB_FILE, NODES_FILE

AUTH = ("# Authentication\n\n<!-- page 1 -->\n\nSSH access uses public key authentication. "
        "Administrators create accounts with the security login create command.\n\n"
        "<!-- page 2 -->\n\n## Roles\n\nRole based access control limits what each account can do.")
COOK = "# Cooking\n\n<!-- page 1 -->\n\nSourdough bread needs a starter, flour, water and salt."


class PipelineTests(TempHome):
    def test_scan_skips_hidden_unsupported_and_own_dirs(self):
        self.write_doc("a/x.md", "hi")
        self.write_doc("a/.hidden.md", "hi")
        self.write_doc("a/notes.exe", "hi")
        self.write_doc(".git/y.md", "hi")
        self.write_doc("a/~$lock.docx", "hi")
        (self.paths.markup / "z").mkdir(parents=True)
        (self.paths.markup / "z" / "old.md").write_text("x")
        found = indexer.scan_sources(self.tmp / "home", indexer.exclude_dirs(self.paths))
        self.assertEqual([p.name for p in found], ["x.md"])

    def test_end_to_end_index_publish_search_page_citation(self):
        self.write_doc("security/auth.md", AUTH)
        self.write_doc("kitchen/bread.md", COOK)
        summary = self.index()
        self.assertEqual((summary["indexed"], summary["errors"]), (2, []))
        for c in ("security", "kitchen"):
            self.assertTrue((self.paths.index / c / ALL_DIR / EMB_FILE).exists())
        self.assertTrue(self.publish()["changed"])
        eng = self.engine()
        res = eng.search("role based access control", top_k=3)["results"]
        self.assertEqual((res[0]["file"], res[0]["page"], res[0]["heading"], res[0]["collection"]),
                         ("auth", "2", "Roles", "security"))
        only = eng.search("bread", collections=["kitchen"])["results"]
        self.assertTrue(only and all(r["collection"] == "kitchen" for r in only))

    def test_search_reads_only_the_published_generation(self):
        self.write_doc("c/a.md", "# T\n\n<!-- page 1 -->\n\nalpha")
        self.index()
        self.publish()
        eng = self.engine()
        self.write_doc("c/b.md", "# T\n\n<!-- page 1 -->\n\nzulu unique")
        self.index()  # workspace changes, nothing published
        self.assertEqual(eng.search("zulu")["results"] and eng.search("zulu")["results"][0]["file"],
                         "a")  # only doc a exists in the served generation
        self.publish()
        eng.install(eng.prepare_generation())  # hot swap
        self.assertEqual(eng.search("zulu")["results"][0]["file"], "b")

    def test_reload_reuses_unchanged_collections_and_loads_only_changed_ones(self):
        self.write_doc("security/auth.md", AUTH)
        self.write_doc("kitchen/bread.md", COOK)
        self.index()
        self.publish()
        eng = self.engine()
        before = dict(eng.gen.indexes)
        self.write_doc("kitchen/rye.md", "# Rye\n\n<!-- page 1 -->\n\nRye flour and caraway.")
        self.index()
        self.publish()
        gen = eng.prepare_generation()
        self.assertEqual((gen.reused, gen.loaded), (["security"], ["kitchen"]))
        self.assertIs(gen.indexes["security"], before["security"])       # shared, not re-read
        self.assertIsNot(gen.indexes["kitchen"], before["kitchen"])
        eng.install(gen)
        self.assertEqual(eng.search("caraway")["results"][0]["file"], "rye")
        self.assertEqual(eng.search("role based access control")["results"][0]["file"], "auth")
        # nothing changed at all: every collection is reused
        self.publish(force=True)
        again = eng.prepare_generation()
        self.assertEqual((sorted(again.reused), again.loaded), (["kitchen", "security"], []))

    def test_no_reuse_across_models_or_when_the_previous_index_was_not_loaded(self):
        self.write_doc("security/auth.md", AUTH)
        self.index()
        self.publish()
        lazy = SearchEngine(self.paths, embedder=FakeEmbedder())
        lazy.install(lazy.prepare_generation(prewarm=False))  # nothing loaded yet
        self.publish(force=True)
        gen = lazy.prepare_generation(prewarm=True)
        self.assertEqual((gen.reused, gen.loaded), ([], ["security"]))

    def test_second_run_is_all_fresh_and_change_reindexes(self):
        p = self.write_doc("c/a.md", AUTH)
        self.assertEqual(self.index()["indexed"], 1)
        emb = FakeEmbedder()
        s2 = self.index(embedder=emb)
        self.assertEqual((s2["indexed"], s2["skipped_fresh"], emb.calls), (0, 1, 0))
        p.write_text(AUTH + "\n\nNew paragraph about multifactor tokens.")
        self.assertEqual(self.index()["indexed"], 1)
        self.publish()
        self.assertIn("multifactor", self.engine().search("multifactor tokens")["results"][0]["text"])

    def test_stale_markdown_is_not_reused(self):
        p = self.write_doc("c/a.md", "# T\n\n<!-- page 1 -->\n\nalpha content only")
        self.index()
        p.write_text("# T\n\n<!-- page 1 -->\n\nbravo content only")
        self.index()
        md = (self.paths.markup / "c" / "a.md").read_text()
        self.assertIn("bravo", md)
        self.assertNotIn("alpha", md)

    def test_merge_reuses_embeddings_not_reembedding(self):
        self.write_doc("c/a.md", AUTH)
        self.index()
        self.write_doc("c/b.md", COOK)
        emb = FakeEmbedder()
        self.index(embedder=emb)
        self.assertEqual(emb.calls, 1)  # only the new document was embedded

    def test_removed_source_leaves_search_after_publish(self):
        a = self.write_doc("c/a.md", AUTH)
        self.write_doc("c/b.md", COOK)
        self.index()
        self.publish()
        a.unlink()
        self.index()
        self.publish()
        res = self.engine().search("authentication access control")["results"]
        self.assertTrue(res)
        self.assertTrue(all(r["file"] == "b" for r in res))

    def test_top_level_files_and_name_collisions(self):
        self.write_doc("top.md", COOK)
        self.write_doc("c/same.md", AUTH)
        self.write_doc("c/same.txt", "other text")
        s = self.index()
        self.assertTrue((self.paths.index / "default" / ALL_DIR / NODES_FILE).exists())
        self.assertEqual(len(s["errors"]), 1)
        self.assertIn("same document name", s["errors"][0]["message"])

    def test_failed_document_progress_carries_the_cause(self):
        from unittest import mock
        from rag_search.core import indexer
        self.write_doc("c/bad.md", AUTH)
        events = []
        with mock.patch.object(indexer, "convert_source", side_effect=RuntimeError("boom cause")):
            s = self.index(progress=events.append)
        self.assertEqual(len(s["errors"]), 1)
        msgs = [e.get("message", "") for e in events if e.get("phase") == "convert" and e.get("current")]
        self.assertTrue(any(m.startswith("error: ") and "boom cause" in m for m in msgs), msgs)

    def test_index_all_wipes_and_rebuilds(self):
        self.write_doc("c/a.md", AUTH)
        self.index()
        s = self.index(wipe=True)
        self.assertEqual((s["indexed"], s["wiped"]), (1, 1))

    def test_search_without_publication(self):
        r = SearchEngine(self.paths, embedder=FakeEmbedder()).search("anything")
        self.assertEqual(r["results"], [])
        self.assertIn("nothing is published", r["note"])

    def test_job_documents_reads_the_event_log(self):
        from rag_search import jobs
        (self.paths.jobs).mkdir(parents=True, exist_ok=True)
        lines = [
            '{"ts": 1.5, "event": "progress", "phase": "convert"}',
            '{"ts": 2.0, "event": "doc", "collection": "c", "source": "a.pdf", "status": "indexed", "total_s": 3.2}',
            "not json",
            '{"ts": 3.0, "event": "doc", "collection": "c", "source": "b.pdf", "status": "error", "message": "x"}',
            '{"ts": 4.0, "event": "doc", "collection": "c", "source": "c.pdf", "status": "skipped"}',
        ]
        jobs.events_file(self.paths, "j1").write_text("\n".join(lines) + "\n")
        d = jobs.documents(self.paths, "j1", limit=2)
        self.assertEqual((d["total"], d["by_status"]), (3, {"indexed": 1, "error": 1, "skipped": 1}))
        self.assertEqual([i["source"] for i in d["items"]], ["b.pdf", "c.pdf"])   # the last two
        self.assertEqual(d["items"][0]["finished_at"], 3.0)
        self.assertEqual(jobs.documents(self.paths, "j1", limit=0)["items"], [])
        self.assertEqual(jobs.documents(self.paths, "nope")["total"], 0)
        self.assertEqual(jobs.documents(self.paths, "../etc")["total"], 0)

    def test_job_documents_lists_a_document_once_with_its_latest_status(self):
        from rag_search import jobs
        (self.paths.jobs).mkdir(parents=True, exist_ok=True)
        lines = [
            '{"ts": 1.0, "event": "doc", "collection": "c", "source": "a.pdf", "status": "converted", "chunks": 7, "convert_s": 5.0, "chunk_s": 0.1, "total_s": 5.1}',
            '{"ts": 2.0, "event": "doc", "collection": "c", "source": "b.pdf", "status": "converted", "chunks": 3, "convert_s": 2.0, "chunk_s": 0.1, "total_s": 2.1}',
            '{"ts": 3.0, "event": "doc", "collection": "c", "source": "c.pdf", "status": "error", "message": "bad"}',
            '{"ts": 4.0, "event": "doc", "collection": "c", "source": "a.pdf", "status": "indexed", "chunks": 7, "embed_s": 4.0, "total_s": 9.1}',
            '{"ts": 5.0, "event": "doc", "collection": "c", "source": "b.pdf", "status": "error", "message": "embedding failed: x"}',
        ]
        jobs.events_file(self.paths, "j2").write_text("\n".join(lines) + "\n")
        d = jobs.documents(self.paths, "j2")
        self.assertEqual(d["total"], 3)
        self.assertEqual(d["by_status"], {"error": 2, "indexed": 1})
        self.assertEqual([i["source"] for i in d["items"]], ["c.pdf", "a.pdf", "b.pdf"])
        a = d["items"][1]
        self.assertEqual((a["status"], a["chunks"], a["convert_s"], a["embed_s"], a["total_s"]),
                         ("indexed", 7, 5.0, 4.0, 9.1))            # timings carried over
        self.assertEqual(d["items"][2]["message"], "embedding failed: x")
        jobs.events_file(self.paths, "j3").write_text(lines[0] + "\n" + lines[1] + "\n")
        self.assertEqual(jobs.documents(self.paths, "j3")["by_status"], {"converted": 2})

    def test_files_without_text_are_skipped_not_failed(self):
        self.write_doc("c/a.md", AUTH)
        self.write_doc("c/blank.md", "   \n\n")
        events = []
        summary = self.index(progress=events.append)
        self.assertEqual(summary["errors"], [])
        self.assertEqual([Path(n["src"]).name for n in summary["no_text"]], ["blank.md"])
        self.assertEqual(summary["indexed"], 1)
        st = {e["doc"]["source"]: e["doc"]["status"] for e in events if "doc" in e}
        self.assertEqual(st["blank.md"], "no_text")
        self.assertEqual(st["a.md"], "indexed")
        self.assertIn("no text", summary["no_text"][0]["message"])
        msgs = [e.get("message", "") for e in events if e.get("phase") == "convert" and e.get("current")]
        self.assertIn("skipped: no text", msgs)

    def test_document_events_timings_and_catalog_sizes(self):
        self.write_doc("c/a.md", AUTH)
        self.write_doc("c/b.md", "# B\n\n<!-- page 1 -->\n\nsecond document about bread")
        events = []
        self.index(progress=events.append)
        all_docs = [e["doc"] for e in events if "doc" in e]
        # each document is announced as converted (waiting to embed) before it is indexed
        converted = [d for d in all_docs if d["status"] == "converted"]
        self.assertEqual(sorted(d["source"] for d in converted), ["a.md", "b.md"])
        self.assertTrue(all(d["chunks"] and "convert_s" in d and "chunk_s" in d for d in converted))
        first_indexed = next(i for i, d in enumerate(all_docs) if d["status"] == "indexed")
        self.assertTrue(all(d["status"] == "converted" for d in all_docs[:first_indexed]))
        docs = [d for d in all_docs if d["status"] != "converted"]
        self.assertEqual(sorted(d["source"] for d in docs), ["a.md", "b.md"])
        for d in docs:
            self.assertEqual((d["status"], d["collection"]), ("indexed", "c"))
            for k in ("convert_s", "chunk_s", "embed_s", "total_s", "chunks"):
                self.assertIn(k, d)
            self.assertAlmostEqual(d["total_s"], d["convert_s"] + d["chunk_s"] + d["embed_s"], delta=0.02)
        # every progress event of the long phases says when the phase began
        phased = [e for e in events if e.get("phase") in ("convert", "embed", "merge")]
        self.assertTrue(phased and all("phase_started_at" in e for e in phased))
        # per-document meta and the published catalog keep sizes and build time
        self.publish()
        from rag_search.catalog import list_view
        from rag_search.policy import Rules
        view = list_view(self.paths, Rules(), "cli", full=True)
        c = view["collections"][0]
        self.assertGreater(c["index_bytes"], 0)
        self.assertGreater(c["markdown_bytes"], 0)
        self.assertGreater(c["source_bytes"], 0)
        self.assertRegex(c["built_at"], r"^\d{4}-\d\d-\d\dT")
        self.assertIsNotNone(c["build_seconds"])
        self.assertEqual(c["build_seconds_documents"], 2)
        self.assertTrue(all(d["build_s"] is not None and d["index_bytes"] > 0 for d in c["documents"]))
        self.assertEqual(view["totals"]["documents"], 2)
        self.assertEqual(view["totals"]["index_bytes"], c["index_bytes"])
        # a second run only reports the documents as unchanged
        events.clear()
        summary = self.index(progress=events.append)
        self.assertEqual([e["doc"]["status"] for e in events if "doc" in e], ["skipped", "skipped"])
        self.assertEqual(set(summary["phase_s"]), {"convert", "embed", "merge"})

    def test_rerank_off_uses_rrf(self):
        self.write_doc("c/a.md", AUTH)
        self.index()
        self.publish()
        r = self.engine(rerank=False, reranker=None).search("public key")
        self.assertFalse(r["timing"]["reranked"])
        self.assertTrue(r["results"])

    def test_model_mismatch_refuses_generation(self):
        self.write_doc("c/a.md", AUTH)
        self.index()
        self.publish()
        eng = SearchEngine(self.paths, embedder=FakeEmbedder())
        eng.model = "some/other-model"
        with self.assertRaises(ModelMismatch):
            eng.prepare_generation()


class SearchDebugTests(TempHome):
    """Per-stage score retention, stage on/off, and pool/RRF-k overrides (troubleshooting)."""

    def setUp(self):
        super().setUp()
        self.write_doc("c/a.md", AUTH)
        self.index()
        self.publish()

    def test_hybrid_default_carries_every_stage_score(self):
        eng = self.engine()
        r = eng.search("public key authentication")
        self.assertEqual(r["timing"]["stages"], ["bm25", "dense", "rerank"])
        hit = r["results"][0]
        self.assertIsNotNone(hit["bm25_score"])
        self.assertIsNotNone(hit["bm25_rank"])
        self.assertIsNotNone(hit["dense_score"])
        self.assertIsNotNone(hit["dense_rank"])
        self.assertIsNotNone(hit["rrf_score"])
        self.assertIsNotNone(hit["rerank_score"])
        for key in ("bm25_candidates", "dense_candidates", "overlap_count",
                    "bm25_only_count", "dense_only_count"):
            self.assertIn(key, r["timing"])

    def test_bm25_only_stage_skips_dense_and_fusion(self):
        eng = self.engine()
        r = eng.search("public key authentication", stages=("bm25",))
        self.assertEqual(r["timing"]["stages"], ["bm25"])
        self.assertEqual(r["timing"]["dense_candidates"], 0)
        self.assertLessEqual(r["timing"]["embed_query_ms"], 2)   # the embedder was never called
        hit = r["results"][0]
        self.assertIsNotNone(hit["bm25_score"])
        self.assertIsNone(hit["dense_score"])
        self.assertIsNone(hit["dense_rank"])
        self.assertIsNone(hit["rrf_score"])          # nothing to fuse with a single retriever
        self.assertEqual(hit["score"], hit["bm25_score"])   # ranked by the raw stage score
        self.assertFalse(r["timing"]["reranked"])

    def test_dense_only_stage_skips_bm25(self):
        eng = self.engine()
        r = eng.search("public key authentication", stages="dense")   # comma-string form too
        self.assertEqual(r["timing"]["stages"], ["dense"])
        self.assertEqual(r["timing"]["bm25_candidates"], 0)
        hit = r["results"][0]
        self.assertIsNone(hit["bm25_score"])
        self.assertIsNone(hit["bm25_rank"])
        self.assertIsNotNone(hit["dense_score"])
        self.assertIsNone(hit["rrf_score"])

    def test_dropping_rerank_stage_is_not_an_error(self):
        eng = self.engine()   # reranking is on by default
        r = eng.search("public key authentication", stages=("bm25", "dense"))
        self.assertFalse(r["timing"]["reranked"])
        self.assertNotIn("rerank_error", r["timing"])
        self.assertIsNotNone(r["results"][0]["rrf_score"])   # still fused, just not reranked

    def test_requesting_rerank_when_disabled_reports_a_note_and_never_loads_a_model(self):
        eng = self.engine(rerank=False, reranker=None)
        self.assertIsNone(eng.reranker)
        r = eng.search("public key authentication")   # default stages include "rerank"
        self.assertFalse(r["timing"]["reranked"])
        self.assertIn("rerank_error", r["timing"])
        self.assertIsNone(eng.reranker)   # a debug request must never trigger a model load

    def test_pool_and_rrf_k_overrides_are_clamped(self):
        from rag_search import spec
        eng = self.engine()
        r = eng.search("public key authentication",
                       retrieval_pool_n=10_000, rerank_pool_n=10_000, rrf_k=10_000)
        self.assertEqual(r["timing"]["retrieval_pool"], spec.RETRIEVAL_POOL_MAX)
        self.assertEqual(r["timing"]["rerank_pool"], spec.RERANK_POOL_MAX)
        self.assertEqual(r["timing"]["rrf_k"], spec.RRF_K_MAX)
        r2 = eng.search("public key authentication", rrf_k=-5)
        self.assertEqual(r2["timing"]["rrf_k"], spec.RRF_K_MIN)

    def test_unset_overrides_reproduce_default_pool_sizes(self):
        from rag_search import spec
        eng = self.engine()
        r = eng.search("public key authentication", top_k=3)
        self.assertEqual(r["timing"]["retrieval_pool"], spec.retrieval_pool(3))
        self.assertEqual(r["timing"]["rerank_pool"], spec.rerank_pool(3))
        self.assertEqual(r["timing"]["rrf_k"], spec.RRF_K)

    def test_invalid_stage_combination_is_rejected(self):
        eng = self.engine()
        with self.assertRaises(ValueError):
            eng.search("x", stages=("rerank",))         # no retriever at all
        with self.assertRaises(ValueError):
            eng.search("x", stages=("not-a-stage",))


class GrepTests(TempHome):
    def setUp(self):
        super().setUp()
        md = self.paths.markup / "manuals"
        md.mkdir(parents=True)
        (md / "g.md").write_text("<!-- page 1 -->\n\nintro\n\n<!-- page 2 -->\n\n"
                                 "line before\nThe Vserver Create command\nline after\n")
        (self.tmp / "secret.md").write_text("vserver outside markup")
        (md / "link.md").symlink_to(self.tmp / "secret.md")
        self.root = self.paths.markup

    def test_match_has_page_and_context(self):
        r = grep_markup(self.root, r"vserver\s+create", ["manuals"])
        self.assertEqual(len(r["matches"]), 1)
        m = r["matches"][0]
        self.assertEqual((m["collection"], m["doc"], m["page"]), ("manuals", "g", "2"))
        self.assertIn("line before", m["context"])

    def test_symlinks_and_traversal_are_not_followed(self):
        self.assertEqual(grep_markup(self.root, "outside", ["manuals"])["matches"], [])
        self.assertIn("error", grep_markup(self.root, "x", [".."]))
        self.assertIn("error", grep_markup(self.root, "x", ["a/b"]))

    def test_bad_regex_limits_and_no_root(self):
        self.assertIn("invalid regex", grep_markup(self.root, "(", ["manuals"])["error"])
        self.assertIn("error", grep_markup(self.root, "a" * 600, ["manuals"]))
        self.assertTrue(grep_markup(self.root, ".", ["manuals"], max_matches=1)["truncated"])
        self.assertEqual(grep_markup(None, "x", ["manuals"])["matches"], [])

    def test_timing_budget(self):
        t0 = time.perf_counter()
        r = grep_markup(self.root, "x", ["manuals"], time_budget_s=-1)
        self.assertTrue(r["timed_out"])
        self.assertLess(time.perf_counter() - t0, 1)


if __name__ == "__main__":
    unittest.main()
