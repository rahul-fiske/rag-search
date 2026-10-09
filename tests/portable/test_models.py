"""Model management: catalogue, selection, fit, cache state, switching, and the engine following
the published index.  No real model is loaded and nothing is downloaded (fakes and stubs)."""

import contextlib
import io
import json
import os
import sys
import types
import unittest
from unittest import mock

from tests.helpers import FakeEmbedder, FakeReranker, TempHome
from rag_search import api, cli, config, model_tasks as mt, models, paths as paths_mod
from rag_search.core import embedding
from rag_search.core.search import ModelMismatch, SearchEngine
from rag_search.paths import META_FILE

DOC_A = "# A\n\n<!-- page 1 -->\n\nSourdough bread rises with wild yeast."
DOC_B = "# B\n\n<!-- page 1 -->\n\nTCP uses a three-way handshake."


def run_cli(*argv):
    out, err = io.StringIO(), io.StringIO()
    rc = 0
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            cli.main(list(argv))
        except SystemExit as e:
            rc = e.code if isinstance(e.code, int) else 1
    return rc, out.getvalue(), err.getvalue()


class Named:
    """Mixin: fakes that carry the model name like the real backends do."""

    def __init__(self, name=""):
        super().__init__()
        self.name = name


class NamedEmbedder(Named, FakeEmbedder):
    pass


class NamedReranker(Named, FakeReranker):
    pass


class CatalogTests(TempHome):
    def test_catalogue_is_consistent(self):
        ids = [m.id for m in models.CATALOG]
        self.assertEqual(len(ids), len(set(ids)))
        for m in models.CATALOG:
            self.assertIn(m.kind, models.KINDS)
            self.assertTrue(m.license and m.note and m.mem_gb > 0, m.id)
            if m.kind == models.EMBEDDING:
                self.assertGreater(m.dim, 0, m.id)
        for name, (emb, rer) in models.PRESETS.items():
            self.assertEqual(models.find(emb).kind, models.EMBEDDING, name)
            self.assertEqual(models.find(rer).kind, models.RERANKER, name)
        self.assertEqual(models.PRESETS["default"], (paths_mod.DEFAULT_MODEL, paths_mod.DEFAULT_RERANK_MODEL))
        self.assertEqual({m.tier for m in models.CATALOG if m.tier == "default"}, {"default"})

    def test_spec_for(self):
        self.assertFalse(models.spec_for("embedding", "BAAI/bge-m3").custom)
        custom = models.spec_for("reranker", "someone/their-reranker")
        self.assertTrue(custom.custom)
        self.assertEqual(custom.backend, models.CROSS_ENCODER)
        with self.assertRaises(models.ModelError):
            models.spec_for("embedding", "BAAI/bge-reranker-base")      # wrong kind
        for bad in ("nonsense", "a/b/c", "../x/y", "", "x/ y"):
            with self.assertRaises(models.ModelError, msg=bad):
                models.spec_for("embedding", bad)
        with self.assertRaises(models.ModelError):
            models.spec_for("colour", "a/b")

    def test_resolve_preset(self):
        self.assertEqual(models.resolve_preset("qwen3-small")[0], "Qwen/Qwen3-Embedding-0.6B")
        with self.assertRaises(models.ModelError):
            models.resolve_preset("nope")


class SelectionTests(TempHome):
    def test_default_config_environment(self):
        self.assertEqual(models.selection("embedding"), ("BAAI/bge-m3", "default"))
        self.assertEqual(models.selection("reranker"), ("BAAI/bge-reranker-v2-m3", "default"))
        models.set_selection(self.paths, "embedding", "Qwen/Qwen3-Embedding-0.6B")
        self.assertEqual(models.selection("embedding"), ("Qwen/Qwen3-Embedding-0.6B", "config"))
        self.assertEqual(paths_mod.model_name(), "Qwen/Qwen3-Embedding-0.6B")
        os.environ["RAG_SEARCH_MODEL"] = "env/model"
        self.assertEqual(models.selection("embedding"), ("env/model", "environment"))
        self.assertEqual(paths_mod.model_name(), "env/model")

    def test_set_selection_keeps_other_settings(self):
        self.paths.config_file.write_text(json.dumps({"indexer": {"jobs": 3}, "models": {"x": 1}}))
        models.set_selection(self.paths, "reranker", "BAAI/bge-reranker-base")
        models.set_memory_limit(self.paths, 8)
        data = json.loads(self.paths.config_file.read_text())
        self.assertEqual(data["indexer"], {"jobs": 3})
        self.assertEqual(data["models"], {"x": 1, "reranker": "BAAI/bge-reranker-base",
                                          "memory_limit_gb": 8})
        self.assertEqual(paths_mod.rerank_model_name(), "BAAI/bge-reranker-base")
        self.assertEqual(models.memory_limit_gb(self.paths), 8.0)
        with self.assertRaises(models.ModelError):
            models.set_selection(self.paths, "reranker", "not a model")
        with self.assertRaises(models.ModelError):
            models.set_memory_limit(self.paths, -1)

    def test_broken_config_is_never_overwritten(self):
        self.paths.config_file.write_text("{ broken")
        with self.assertRaises(ValueError):
            models.set_selection(self.paths, "embedding", "BAAI/bge-m3")
        self.assertEqual(self.paths.config_file.read_text(), "{ broken")
        self.assertEqual(models.selection("embedding")[1], "default")   # readers fall back

    def test_loaded_config_has_the_models_section(self):
        cfg, err = config.load_config(self.paths)
        self.assertEqual(err, "")
        self.assertEqual(cfg["models"], config.DEFAULTS["models"])


class FitTests(unittest.TestCase):
    MAC = {"ram_gb": 32.0, "device": "mps", "weight_bytes": 2}
    INTEL = {"ram_gb": 16.0, "device": "cpu", "weight_bytes": 4}

    def spec(self, model_id):
        return models.find(model_id)

    def test_estimate_and_levels(self):
        bge, rer = self.spec("BAAI/bge-m3"), self.spec("BAAI/bge-reranker-v2-m3")
        # fp16 embedder + fp32 cross-encoder + 1.5 GB overhead
        self.assertAlmostEqual(models.estimate_gb(bge, rer, self.MAC), 1.14 + 2.28 + 1.5, places=1)
        self.assertAlmostEqual(models.estimate_gb(bge, rer, self.INTEL), 2.28 * 2 + 1.5, places=1)
        big = self.spec("Qwen/Qwen3-Embedding-8B")
        est = models.estimate_gb(big, rer, self.MAC)
        self.assertEqual(models.fit_level(est, models.budget_gb(self.MAC, 8)), "too_large")
        self.assertEqual(models.fit_level(4.0, 8), "ok")
        self.assertEqual(models.fit_level(7.0, 8), "tight")
        self.assertEqual(models.fit_level(9.0, 8), "too_large")
        self.assertEqual(models.fit_level(99, 0), "ok")            # unknown budget

    def test_budget(self):
        self.assertEqual(models.budget_gb(self.MAC), 19.2)
        self.assertEqual(models.budget_gb(self.MAC, 8), 8.0)

    def test_reranker_can_be_switched_off(self):
        bge, rer = self.spec("BAAI/bge-m3"), self.spec("BAAI/bge-reranker-v2-m3")
        with mock.patch.dict(os.environ, {"RAG_SEARCH_RERANK": "0"}):
            self.assertAlmostEqual(models.estimate_gb(bge, rer, self.MAC), 1.14 + 1.5, places=1)

    def test_requirements(self):
        spec = models.find("Qwen/Qwen3-Embedding-0.6B")
        with mock.patch.object(models.md, "version", return_value="4.44.0"):
            self.assertTrue(models.missing_requirements(spec))
        with mock.patch.object(models.md, "version", return_value="4.51.3"):
            self.assertEqual(models.missing_requirements(spec), [])
        with mock.patch.object(models.md, "version", side_effect=models.md.PackageNotFoundError("x")):
            self.assertIn("not installed", models.missing_requirements(spec)[0])
        self.assertEqual(models.parse_version("4.51.3+cpu"), (4, 51, 3))


class CacheTests(TempHome):
    def setUp(self):
        super().setUp()
        os.environ["HF_HOME"] = str(self.tmp / "hf")

    def make(self, model_id, files, incomplete=False):
        repo = models.repo_dir(model_id)
        (repo / "blobs").mkdir(parents=True)
        snap = repo / "snapshots" / "abc123"
        snap.mkdir(parents=True)
        for name, data in files.items():
            (snap / name).parent.mkdir(parents=True, exist_ok=True)
            (snap / name).write_bytes(data)
            (repo / "blobs" / name.replace("/", "_")).write_bytes(data)
        if incomplete:
            (repo / "blobs" / "deadbeef.incomplete").write_bytes(b"x" * 10)

    def test_missing_partial_and_complete(self):
        st0 = models.cache_state("a/b")
        self.assertEqual({k: st0[k] for k in ("cached", "partial", "bytes")}, {"cached": False, "partial": False, "bytes": 0})
        self.assertTrue(st0["why"])
        self.make("a/b", {"config.json": b"{}"}, incomplete=True)
        st = models.cache_state("a/b")
        self.assertFalse(st["cached"])
        self.assertTrue(st["partial"])
        self.make("c/d", {"config.json": b"{}", "model.safetensors": b"w" * 100})
        st = models.cache_state("c/d")
        self.assertTrue(st["cached"])
        self.assertFalse(st["partial"])
        self.assertGreaterEqual(st["bytes"], 100)

    def test_sharded_model_needs_every_shard(self):
        index = json.dumps({"weight_map": {"a": "m-1.safetensors", "b": "m-2.safetensors"}}).encode()
        self.make("e/f", {"config.json": b"{}", "model.safetensors.index.json": index,
                          "m-1.safetensors": b"1"})
        self.assertFalse(models.cache_state("e/f")["cached"])
        snap = models.repo_dir("e/f") / "snapshots" / "abc123"
        (snap / "m-2.safetensors").write_bytes(b"2")
        self.assertTrue(models.cache_state("e/f")["cached"])

    def test_a_stale_index_from_the_unquantised_model_is_ignored(self):
        """mlx-community/Qwen3-VL-4B-Instruct-4bit ships one model.safetensors next to the two-shard
        index of the unquantised model.  The download is complete; it must read as downloaded."""
        index = json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors",
                                           "b": "model-00002-of-00002.safetensors"}}).encode()
        self.make("q/w", {"config.json": b"{}", "model.safetensors.index.json": index,
                          "model.safetensors": b"w" * 50})
        st = models.cache_state("q/w")
        self.assertTrue(st["cached"], st)
        self.assertEqual(st["why"], "")

    def test_the_reason_a_copy_is_not_usable_is_reported(self):
        index = json.dumps({"weight_map": {"a": "m-1.safetensors", "b": "m-2.safetensors"}}).encode()
        self.make("e/f", {"config.json": b"{}", "model.safetensors.index.json": index, "m-1.safetensors": b"1"})
        self.assertIn("m-2.safetensors", models.cache_state("e/f")["why"])
        self.make("n/c", {"model.safetensors": b"w"})
        self.assertEqual(models.cache_state("n/c")["why"], "no config.json")

    def test_cache_location_follows_the_environment(self):
        self.assertEqual(models.hub_cache(), self.tmp / "hf" / "hub")
        os.environ["HF_HUB_CACHE"] = str(self.tmp / "elsewhere")
        self.assertEqual(models.hub_cache(), self.tmp / "elsewhere")

    def test_stray_newer_snapshot_does_not_mask_a_complete_one(self):
        """A repo directory can hold more than one snapshot -- e.g. a complete download plus a
        later, narrower fetch that only grabbed one file (a stray revision peek, an interrupted
        re-verify, ...). The newest snapshot by mtime is not necessarily the complete one; the one
        `refs/main` points to should still be found."""
        repo = models.repo_dir("g/h")
        (repo / "blobs").mkdir(parents=True)
        complete = repo / "snapshots" / "complete123"
        complete.mkdir(parents=True)
        (complete / "config.json").write_bytes(b"{}")
        (complete / "model.safetensors").write_bytes(b"w" * 100)
        (repo / "refs").mkdir(parents=True)
        (repo / "refs" / "main").write_text("complete123", encoding="utf-8")
        # a sparse, incomplete-looking snapshot with a LATER mtime than the complete one
        sparse = repo / "snapshots" / "sparse456"
        sparse.mkdir(parents=True)
        (sparse / "model.safetensors").write_bytes(b"w" * 100)
        os.utime(complete, (1_000_000, 1_000_000))
        os.utime(sparse, (2_000_000, 2_000_000))
        st = models.cache_state("g/h")
        self.assertTrue(st["cached"], st)
        self.assertFalse(st["partial"])


class WorkspaceTests(TempHome):
    def index_docs(self, model="BAAI/bge-m3"):
        self.write_doc("c/a.md", DOC_A)
        self.write_doc("c/b.md", DOC_B)
        if model != "BAAI/bge-m3":
            os.environ["RAG_SEARCH_MODEL"] = model
        self.index()
        os.environ.pop("RAG_SEARCH_MODEL", None)

    def test_reindex_estimate(self):
        self.index_docs()
        models._WS["at"] = -1e9
        est = models.reindex_estimate(self.paths, "BAAI/bge-m3")
        self.assertEqual(est["documents"], 0)
        est = models.reindex_estimate(self.paths, "Qwen/Qwen3-Embedding-0.6B")
        self.assertEqual((est["documents"], est["total_documents"]), (2, 2))
        self.assertGreater(est["chunks"], 0)

    def test_estimate_scales_with_model_size(self):
        self.index_docs()
        for d in (self.paths.index / "c").iterdir():
            if d.name == "_all":
                continue
            meta = d / META_FILE
            data = json.loads(meta.read_text())
            data["embed_s"] = 100.0
            meta.write_text(json.dumps(data))
        small = models.reindex_estimate(self.paths, "Qwen/Qwen3-Embedding-0.6B")["estimated_s"]
        big = models.reindex_estimate(self.paths, "Qwen/Qwen3-Embedding-4B")["estimated_s"]
        self.assertGreater(big, small * 5)

    def test_state(self):
        self.index_docs()
        self.publish()
        models.set_selection(self.paths, "embedding", "Qwen/Qwen3-Embedding-0.6B")
        models._WS["at"] = -1e9
        st = models.state(self.paths)
        self.assertEqual(st["serving"], "BAAI/bge-m3")
        self.assertEqual(st["embedding"]["active"], "Qwen/Qwen3-Embedding-0.6B")
        self.assertTrue(st["reindex"]["needed"])
        self.assertEqual(st["reindex"]["documents_on_other_model"], 2)
        rows = {r["id"]: r for r in st["embedding"]["models"]}
        self.assertTrue(rows["Qwen/Qwen3-Embedding-0.6B"]["active"])
        self.assertTrue(rows["BAAI/bge-m3"]["serving"])
        self.assertIn(rows["Qwen/Qwen3-Embedding-8B"]["fit"], ("ok", "tight", "too_large"))
        self.assertEqual({r["id"] for r in st["reranker"]["models"]} >= {m.id for m in models.by_kind("reranker")}, True)
        json.dumps(st)                                   # the dashboard sends it as JSON

    def test_custom_choice_stays_visible_and_bad_ids_do_not_break_state(self):
        models.set_selection(self.paths, "reranker", "someone/their-reranker")
        st = models.state(self.paths)
        self.assertIn("someone/their-reranker", [r["id"] for r in st["reranker"]["models"]])
        self.paths.config_file.write_text(json.dumps({"models": {"embedding": "garbage id"}}))
        st = models.state(self.paths)                    # must not raise
        self.assertEqual(st["embedding"]["active"], "garbage id")

    def test_publish_refuses_a_half_finished_switch_with_a_helpful_message(self):
        from rag_search import publish

        self.index_docs()
        meta = self.paths.index / "c" / "b" / META_FILE
        data = json.loads(meta.read_text())
        data["model"] = "Qwen/Qwen3-Embedding-0.6B"
        meta.write_text(json.dumps(data))
        with self.assertRaises(publish.PublishError) as cm:
            self.publish()
        text = str(cm.exception)
        self.assertIn("rag-search index new", text)
        self.assertIn("Qwen/Qwen3-Embedding-0.6B", text)


class BackendChoiceTests(TempHome):
    def test_make_reranker_picks_the_backend_from_the_catalogue(self):
        self.assertIsInstance(embedding.make_reranker("BAAI/bge-reranker-v2-m3"), embedding.Reranker)
        self.assertIsInstance(embedding.make_reranker("Qwen/Qwen3-Reranker-0.6B"), embedding.Qwen3Reranker)
        self.assertIsInstance(embedding.make_reranker("someone/custom"), embedding.Reranker)

    def test_configured_names_reach_the_backends(self):
        models.set_selection(self.paths, "embedding", "Qwen/Qwen3-Embedding-0.6B")
        models.set_selection(self.paths, "reranker", "Qwen/Qwen3-Reranker-0.6B")
        self.assertEqual(embedding.make_embedder().name, "Qwen/Qwen3-Embedding-0.6B")
        self.assertIsInstance(embedding.make_reranker(), embedding.Qwen3Reranker)

    def test_query_prefix_only_for_instruction_models(self):
        q = embedding.Embedder("Qwen/Qwen3-Embedding-0.6B")
        self.assertTrue(q.query_prefix.startswith("Instruct:"))
        self.assertEqual(embedding.Embedder("BAAI/bge-m3").query_prefix, "")
        seen = []
        q.encode = lambda texts, progress=None: (seen.append(texts), [[0.0]])[1]
        q.encode_query("hello")
        self.assertEqual(seen[0], [q.query_prefix + "hello"])

    def test_qwen3_prompt_follows_the_model_card(self):
        r = embedding.Qwen3Reranker("Qwen/Qwen3-Reranker-0.6B")
        pair = r._pair("what is x", "x is y")
        self.assertEqual(pair, "<Instruct>: Given a web search query, retrieve relevant passages "
                               "that answer the query\n<Query>: what is x\n<Document>: x is y")
        self.assertTrue(embedding.QWEN3_RERANK_PREFIX.startswith("<|im_start|>system\nJudge whether"))
        self.assertTrue(embedding.QWEN3_RERANK_SUFFIX.endswith("<think>\n\n</think>\n\n"))

    def test_dtype_override(self):
        torch = types.SimpleNamespace(float16="f16", bfloat16="bf16", float32="f32")
        self.assertEqual(embedding.torch_dtype(torch, "mps"), "f16")
        self.assertIsNone(embedding.torch_dtype(torch, "cpu"))
        self.assertIsNone(embedding.torch_dtype(torch, "mps", default_half=False))
        os.environ["RAG_SEARCH_DTYPE"] = "bfloat16"
        self.assertEqual(embedding.torch_dtype(torch, "cpu"), "bf16")
        os.environ["RAG_SEARCH_DTYPE"] = "wat"
        with self.assertRaises(ValueError):
            embedding.torch_dtype(torch, "cpu")

    def test_nan_guard(self):
        import numpy as np

        embedding._finite_or_raise(np.ones(3), "x", "m")
        with self.assertRaises(RuntimeError) as cm:
            embedding._finite_or_raise(np.array([1.0, np.nan]), "x", "m")
        self.assertIn("RAG_SEARCH_DTYPE", str(cm.exception))


class EngineFollowsPublishedModelTests(TempHome):
    """Switching the setting must not disturb search until the re-embedded index is published."""

    def setUp(self):
        super().setUp()
        self.made = []

        def make_embedder(name=None):
            self.made.append(("embedder", name))
            return NamedEmbedder(name or paths_mod.model_name())

        def make_reranker(name=None):
            self.made.append(("reranker", name))
            return NamedReranker(name or paths_mod.rerank_model_name())

        for target, fn in (("make_embedder", make_embedder), ("make_reranker", make_reranker)):
            p = mock.patch.object(embedding, target, fn)
            p.start()
            self.addCleanup(p.stop)
        self.write_doc("c/a.md", DOC_A)
        self.write_doc("c/b.md", DOC_B)
        self.index()
        self.publish()

    def test_engine_serves_the_model_the_index_was_built_with(self):
        models.set_selection(self.paths, "embedding", "Qwen/Qwen3-Embedding-0.6B")   # not re-embedded yet
        eng = SearchEngine(self.paths)
        eng.load_models()
        self.assertIn(("embedder", "BAAI/bge-m3"), self.made)
        self.assertEqual(eng.models_info()["embedding"], "BAAI/bge-m3")
        eng.install(eng.prepare_generation())
        self.assertTrue(eng.search("sourdough bread", 3, ["c"])["results"])

    def test_new_generation_brings_its_own_embedder(self):
        eng = SearchEngine(self.paths)
        eng.load_models()
        eng.install(eng.prepare_generation())
        old = eng.embedder
        # re-embed everything with another model and publish
        os.environ["RAG_SEARCH_MODEL"] = "Qwen/Qwen3-Embedding-0.6B"
        self.index()
        self.publish()
        os.environ.pop("RAG_SEARCH_MODEL")
        gen = eng.prepare_generation()
        self.assertIsNotNone(gen.embedder)
        self.assertEqual(gen.embedder.name, "Qwen/Qwen3-Embedding-0.6B")
        self.assertIs(eng.embedder, old)                  # nothing changed until install
        eng.install(gen)
        self.assertEqual(eng.models_info()["embedding"], "Qwen/Qwen3-Embedding-0.6B")
        self.assertEqual(eng.model, "Qwen/Qwen3-Embedding-0.6B")
        self.assertTrue(eng.search("three-way handshake", 3, ["c"])["results"])

    def test_injected_embedder_keeps_the_strict_rule(self):
        os.environ["RAG_SEARCH_MODEL"] = "other/model"
        self.index()
        self.publish()
        os.environ.pop("RAG_SEARCH_MODEL")
        eng = SearchEngine(self.paths, embedder=FakeEmbedder(), reranker=FakeReranker())
        with self.assertRaises(ModelMismatch):
            eng.prepare_generation()

    def test_reranker_switch_is_prepared_aside_and_swapped(self):
        eng = SearchEngine(self.paths)
        eng.load_models()
        self.assertIsNone(eng.prepare_reranker())          # nothing changed
        models.set_selection(self.paths, "reranker", "BAAI/bge-reranker-base")
        new = eng.prepare_reranker()
        self.assertEqual(new.name, "BAAI/bge-reranker-base")
        self.assertEqual(eng.models_info()["reranker"], "BAAI/bge-reranker-v2-m3")
        eng.install_reranker(new)
        self.assertEqual(eng.models_info()["reranker"], "BAAI/bge-reranker-base")
        self.assertIsNone(eng.prepare_reranker())

    def test_injected_reranker_is_never_replaced(self):
        eng = SearchEngine(self.paths, embedder=FakeEmbedder(), reranker=FakeReranker())
        models.set_selection(self.paths, "reranker", "BAAI/bge-reranker-base")
        self.assertIsNone(eng.prepare_reranker())


class DaemonRerankerSwapTests(TempHome):
    class Engine:
        generation = 3

        def __init__(self, fail=False):
            self.fail, self.installed = fail, []
            self.gen = types.SimpleNamespace(catalog={"collections": []})

        def prepare_reranker(self):
            if self.fail:
                raise RuntimeError("out of memory")
            return types.SimpleNamespace(name="new/reranker")

        def install_reranker(self, new):
            self.installed.append(new.name)

        def prepare_generation(self, prewarm=True):
            return types.SimpleNamespace(number=3)

        def models_info(self):
            return {"embedding": "e", "reranker": "r"}

    def daemon(self, engine):
        from rag_search.core.search_daemon import SearchDaemon

        return SearchDaemon(self.paths, engine=engine, prewarm=False)

    def test_reload_swaps_the_reranker_and_reports_it(self):
        eng = self.Engine()
        r = self.daemon(eng).reload()
        self.assertEqual(eng.installed, ["new/reranker"])
        self.assertEqual(r["reranker"], "new/reranker")
        self.assertFalse(r["changed"])                    # same generation

    def test_failed_reranker_load_keeps_the_old_one_and_says_so(self):
        eng = self.Engine(fail=True)
        r = self.daemon(eng).reload()
        self.assertEqual(eng.installed, [])
        self.assertIn("out of memory", r["reranker_error"])

    def test_ping_lists_the_loaded_models(self):
        self.assertEqual(self.daemon(self.Engine()).ping_info()["models"],
                         {"embedding": "e", "reranker": "r"})

    def test_api_passes_the_reranker_result_through(self):
        reply = {"ok": True, "generation": 3, "changed": False, "reranker": "new/reranker"}
        with mock.patch("rag_search.client.request_sync", return_value=reply):
            out = api.reload_search(self.paths)
        self.assertEqual(out["reranker"], "new/reranker")


class TaskTests(TempHome):
    """The switch runner, with download and verify replaced."""

    def setUp(self):
        super().setUp()
        self.calls = []
        # results must not depend on the machine the tests run on
        for target, value in (("machine_info", lambda: {"ram_gb": 64.0, "system": "Darwin",
                                                        "arch": "arm64", "device": "mps",
                                                        "weight_bytes": 2}),
                              ("missing_requirements", lambda spec: [])):
            p = mock.patch.object(models, target, value)
            p.start()
            self.addCleanup(p.stop)
        for name, fn in (("download_model", lambda mid, task: self.calls.append(("download", mid))),
                         ("verify_model", self.fake_verify)):
            p = mock.patch.object(mt, name, fn)
            p.start()
            self.addCleanup(p.stop)
        self.verify_result = {"ok": True, "dim": 8, "load_s": 0.1}

    def fake_verify(self, kind, model_id, timeout=0):
        self.calls.append(("verify", model_id))
        return self.verify_result

    def switch(self, kind, model, **kw):
        return mt.run(self.paths, {"op": "switch", "kind": kind, "model": model, **kw})

    def test_plan_reports_blocking_reasons_and_reindex(self):
        self.write_doc("c/a.md", DOC_A)
        self.index()
        models._WS["at"] = -1e9
        pl = mt.plan_switch(self.paths, "embedding", "Qwen/Qwen3-Embedding-8B")
        self.assertEqual(pl["reindex"]["documents"], 1)
        models.set_memory_limit(self.paths, 4)
        pl = mt.plan_switch(self.paths, "embedding", "Qwen/Qwen3-Embedding-8B")
        self.assertTrue(pl["blocking"])
        self.assertFalse(mt.plan_switch(self.paths, "embedding", "Qwen/Qwen3-Embedding-0.6B",
                                        force=True)["same"])
        os.environ["RAG_SEARCH_RERANK_MODEL"] = "x/y"
        pl = mt.plan_switch(self.paths, "reranker", "BAAI/bge-reranker-base")
        self.assertTrue(any("RAG_SEARCH_RERANK_MODEL" in w for w in pl["warnings"]))

    def test_reranker_switch_without_a_daemon_just_saves(self):
        rec = self.switch("reranker", "BAAI/bge-reranker-base")
        self.assertEqual(rec["status"], "succeeded", rec)
        self.assertEqual(models.selection("reranker"), ("BAAI/bge-reranker-base", "config"))
        self.assertEqual(self.calls, [("download", "BAAI/bge-reranker-base"),
                                      ("verify", "BAAI/bge-reranker-base")])
        self.assertEqual(rec["result"]["search_daemon"], "not running")

    def test_reranker_switch_reloads_a_running_daemon_and_undoes_on_failure(self):
        with mock.patch("rag_search.client.ping", return_value={"state": "ready"}), \
                mock.patch.object(api, "reload_search", return_value={"ok": True, "reranker": "BAAI/bge-reranker-base"}):
            rec = self.switch("reranker", "BAAI/bge-reranker-base")
        self.assertEqual(rec["status"], "succeeded")
        self.assertEqual(rec["result"]["search_daemon"], "reloaded")
        with mock.patch("rag_search.client.ping", return_value={"state": "ready"}), \
                mock.patch.object(api, "reload_search",
                                  return_value={"ok": True, "reranker_error": "OOM"}):
            rec = self.switch("reranker", "mixedbread-ai/mxbai-rerank-base-v2")
        self.assertEqual(rec["status"], "failed")
        self.assertIn("OOM", rec["error"])
        self.assertEqual(models.selection("reranker")[0], "BAAI/bge-reranker-base")   # restored

    def test_failed_test_changes_nothing(self):
        self.verify_result = {"ok": False, "error": "did not rank"}
        rec = self.switch("embedding", "Qwen/Qwen3-Embedding-0.6B")
        self.assertEqual(rec["status"], "failed")
        self.assertIn("Nothing was changed", rec["error"])
        self.assertEqual(models.selection("embedding")[1], "default")

    def test_embedding_switch_starts_reembedding(self):
        self.write_doc("c/a.md", DOC_A)
        self.index()
        models._WS["at"] = -1e9
        started = []
        with mock.patch.object(api, "index_start", side_effect=lambda *a, **k: started.append(k) or
                               {"ok": True, "job": {"id": "j1"}}):
            rec = self.switch("embedding", "Qwen/Qwen3-Embedding-0.6B", force=True)
        self.assertEqual(rec["status"], "succeeded", rec)
        self.assertEqual(rec["result"]["reindex"], "started")
        self.assertEqual(started[0]["mode"], "new")
        self.assertTrue(started[0]["restart"])
        self.assertEqual(models.selection("embedding")[0], "Qwen/Qwen3-Embedding-0.6B")

    def test_no_reindex_only_saves(self):
        self.write_doc("c/a.md", DOC_A)
        self.index()
        models._WS["at"] = -1e9
        with mock.patch.object(api, "index_start", side_effect=AssertionError("must not start")):
            rec = self.switch("embedding", "Qwen/Qwen3-Embedding-0.6B", force=True, reindex=False)
        self.assertEqual(rec["result"]["reindex"], "not started")

    def test_blocked_without_force(self):
        models.set_memory_limit(self.paths, 2)
        rec = self.switch("embedding", "Qwen/Qwen3-Embedding-8B")
        self.assertEqual(rec["status"], "failed")
        self.assertIn("--force", rec["error"])
        self.assertEqual(self.calls, [])

    def test_already_in_use(self):
        models.set_selection(self.paths, "reranker", "BAAI/bge-reranker-base")
        with mock.patch.object(models, "cache_state", return_value={"cached": True, "partial": False, "bytes": 1}):
            rec = self.switch("reranker", "BAAI/bge-reranker-base")
        self.assertTrue(rec["result"]["unchanged"])
        self.assertEqual(self.calls, [])

    def test_only_one_task_at_a_time(self):
        holder = mt.Task(self.paths, {"op": "download", "model": "a/b"}, None)
        try:
            self.assertTrue(mt.is_active(self.paths))
            with self.assertRaises(mt.TaskBusy):
                mt.run(self.paths, {"op": "download", "model": "a/b"})
            with self.assertRaises(mt.TaskBusy):
                mt.start_detached(self.paths, {"op": "download", "model": "a/b"})
        finally:
            holder.finish("cancelled")
        self.assertFalse(mt.is_active(self.paths))

    def test_a_dead_runner_is_reported_as_failed(self):
        from rag_search.paths import write_json_atomic

        write_json_atomic(mt.task_file(self.paths), {"id": "x", "status": "running", "op": "switch",
                                                      "updated_at": 1.0})
        rec = mt.read_task(self.paths)
        self.assertEqual(rec["status"], "failed")
        self.assertIn("interrupted", rec["error"])
        self.assertFalse(mt.cancel(self.paths))

    def test_download_task_runs_every_model(self):
        rec = mt.run(self.paths, {"op": "download", "models": ["a/b", "c/d"]})
        self.assertEqual(rec["status"], "succeeded")
        self.assertEqual(self.calls, [("download", "a/b"), ("download", "c/d")])

    def test_verify_task_reports_failures(self):
        self.verify_result = {"ok": False, "error": "nope"}
        rec = mt.run(self.paths, {"op": "verify", "targets": [["embedding", "a/b"]]})
        self.assertEqual(rec["status"], "failed")
        self.assertIn("nope", rec["error"])

    def test_unknown_task(self):
        with self.assertRaises(mt.TaskError):
            mt.run(self.paths, {"op": "explode"})


class DownloadTests(TempHome):
    """download_model with a stand-in huggingface_hub."""

    def setUp(self):
        super().setUp()
        os.environ["HF_HOME"] = str(self.tmp / "hf")
        self.task = mt.Task(self.paths, {"op": "download", "model": "a/b"}, None)
        self.addCleanup(self.task.finish, "cancelled")

    def fake_hub(self, snapshot):
        mod = types.ModuleType("huggingface_hub")
        mod.snapshot_download = snapshot

        class HfApi:
            def model_info(self, *a, **k):
                sib = [types.SimpleNamespace(rfilename="model.safetensors", size=100),
                       types.SimpleNamespace(rfilename="onnx/model.onnx", size=999)]
                return types.SimpleNamespace(siblings=sib)

        mod.HfApi = HfApi
        return mock.patch.dict(sys.modules, {"huggingface_hub": mod})

    def test_expected_size_skips_ignored_files(self):
        with self.fake_hub(lambda **k: None):
            self.assertEqual(mt.expected_bytes("a/b"), 100)

    def test_download_reports_and_succeeds(self):
        def snap(repo_id, ignore_patterns):
            self.assertIn("*.onnx", ignore_patterns)
            repo = models.repo_dir(repo_id)
            (repo / "blobs").mkdir(parents=True)
            (repo / "snapshots" / "s").mkdir(parents=True)
            (repo / "snapshots" / "s" / "config.json").write_text("{}")
            (repo / "snapshots" / "s" / "model.safetensors").write_bytes(b"w" * 100)
            (repo / "blobs" / "w").write_bytes(b"w" * 100)
            return str(repo)

        with self.fake_hub(snap):
            mt.download_model("a/b", self.task)
        self.assertTrue(models.cache_state("a/b")["cached"])
        with self.fake_hub(lambda **k: self.fail("already downloaded")):
            mt.download_model("a/b", self.task)

    def test_a_finished_download_that_is_not_usable_is_not_reported_as_downloaded(self):
        def snap(repo_id, ignore_patterns):
            repo = models.repo_dir(repo_id)
            (repo / "blobs").mkdir(parents=True)
            (repo / "snapshots" / "s").mkdir(parents=True)
            (repo / "snapshots" / "s" / "model.safetensors").write_bytes(b"w" * 100)   # no config.json
            (repo / "blobs" / "w").write_bytes(b"w" * 100)
            return str(repo)

        with self.fake_hub(snap), self.assertRaises(mt.TaskError) as cm:
            mt.download_model("a/b", self.task)
        self.assertIn("not complete", str(cm.exception))
        self.assertIn("no config.json", str(cm.exception))
        self.assertNotIn("a/b: downloaded", str(self.task.rec.get("log", [])))

    def test_errors_are_explained(self):
        class GatedRepoError(Exception):
            pass

        def boom(**k):
            raise GatedRepoError("401 Client Error")

        with self.fake_hub(boom), self.assertRaises(mt.TaskError) as cm:
            mt.download_model("a/gated", self.task)
        self.assertIn("gated", str(cm.exception))
        self.assertIn("not found", mt.explain_download_error("a/b", type("RepositoryNotFoundError", (Exception,), {})("x")))
        self.assertIn("cannot reach", mt.explain_download_error("a/b", type("ConnectionError", (Exception,), {})("dns")))
        self.assertIn("disk is full", mt.explain_download_error("a/b", OSError(28, "No space left")))


class InstallCommandTests(TempHome):
    """The runtime installer must work in a `uv tool` environment (no pip) under launchd's short PATH."""

    def test_uv_in_a_known_folder_is_found_when_it_is_not_on_the_path(self):
        uv = self.tmp / ".local" / "bin" / "uv"
        uv.parent.mkdir(parents=True)
        uv.write_text("#!/bin/sh\n")
        uv.chmod(0o755)
        with mock.patch("shutil.which", return_value=None), mock.patch.object(models.Path, "home", return_value=self.tmp):
            self.assertEqual(models.find_uv(), str(uv))
            cmd = mt.install_command(["mlx-vlm>=0.3.4"])
        self.assertEqual(cmd[:3], [str(uv), "pip", "install"])
        self.assertIn("--python", cmd)

    def test_without_uv_pip_is_used_when_the_environment_has_it(self):
        with mock.patch.object(models, "find_uv", return_value=""), \
                mock.patch("importlib.util.find_spec", return_value=object()):
            self.assertEqual(mt.install_command(["x"])[1:3], ["-m", "pip"])

    def test_neither_uv_nor_pip_gives_the_command_to_run_by_hand(self):
        with mock.patch.object(models, "find_uv", return_value=""), \
                mock.patch("importlib.util.find_spec", return_value=None), \
                self.assertRaises(mt.TaskError) as cm:
            mt.install_command(["mlx-vlm>=0.3.4"])
        self.assertIn("uv pip install", str(cm.exception))
        self.assertIn("mlx-vlm>=0.3.4", str(cm.exception))


class VerifyTests(TempHome):
    def test_verify_in_a_subprocess_with_fake_backends(self):
        self.use_fake_backends()
        emb = mt.verify_model("embedding", "some/fake-embedder", timeout=120)
        self.assertTrue(emb["ok"], emb)
        self.assertEqual(emb["dim"], 64)
        rer = mt.verify_model("reranker", "some/fake-reranker", timeout=120)
        self.assertTrue(rer["ok"], rer)

    def test_a_model_that_ranks_badly_is_rejected(self):
        self.use_fake_backends()
        os.environ["RAG_SEARCH_EMBEDDER"] = "tests.portable.test_models:RandomEmbedder"
        res = mt.verify_model("embedding", "some/random", timeout=120)
        self.assertFalse(res["ok"])
        self.assertIn("did not put the relevant passage first", res["error"])

    def test_a_crash_is_reported_not_raised(self):
        self.use_fake_backends()
        os.environ["RAG_SEARCH_EMBEDDER"] = "tests.portable.test_models:CrashingEmbedder"
        res = mt.verify_model("embedding", "some/crash", timeout=120)
        self.assertFalse(res["ok"])
        self.assertIn("boom", res["error"])


class RandomEmbedder(FakeEmbedder):
    """Every text gets the same vector: cannot rank anything."""

    def _vec(self, text):
        import numpy as np

        v = np.ones(64, dtype=np.float32)
        return v / np.linalg.norm(v)


class CrashingEmbedder(FakeEmbedder):
    def load(self):
        raise RuntimeError("boom")


class CliTests(TempHome):
    def setUp(self):
        super().setUp()
        for target, value in (("machine_info", lambda: {"ram_gb": 64.0, "system": "Darwin",
                                                        "arch": "arm64", "device": "mps",
                                                        "weight_bytes": 2}),
                              ("missing_requirements", lambda spec: [])):
            p = mock.patch.object(models, target, value)
            p.start()
            self.addCleanup(p.stop)
        for name, fn in (("download_model", lambda mid, task: None),
                         ("verify_model", lambda kind, mid, timeout=0: {"ok": True})):
            p = mock.patch.object(mt, name, fn)
            p.start()
            self.addCleanup(p.stop)

    def test_list(self):
        rc, out, _ = run_cli("models")
        self.assertEqual(rc, 0)
        self.assertIn("EMBEDDING MODEL", out)
        self.assertIn("BAAI/bge-m3", out)
        self.assertIn("Qwen/Qwen3-Reranker-0.6B", out)
        rc, out, _ = run_cli("models", "--json")
        data = json.loads(out)
        self.assertEqual(data["embedding"]["active"], "BAAI/bge-m3")
        self.assertIn("machine", data)

    def test_set_needs_confirmation_when_not_a_terminal(self):
        rc, out, err = run_cli("models", "set", "reranker", "BAAI/bge-reranker-base", "--force")
        self.assertNotEqual(rc, 0)
        self.assertIn("--yes", err)
        self.assertEqual(models.selection("reranker")[1], "default")

    def test_set_yes(self):
        rc, out, err = run_cli("models", "set", "reranker", "BAAI/bge-reranker-base", "--yes", "--force")
        self.assertEqual(rc, 0, out + err)
        self.assertEqual(models.selection("reranker")[0], "BAAI/bge-reranker-base")
        self.assertIn("no re-indexing", out)

    def test_use_preset_without_reindex(self):
        rc, out, err = run_cli("models", "use", "qwen3-small", "--yes", "--force", "--no-reindex")
        self.assertEqual(rc, 0, out + err)
        self.assertEqual(models.selection("embedding")[0], "Qwen/Qwen3-Embedding-0.6B")
        self.assertEqual(models.selection("reranker")[0], "Qwen/Qwen3-Reranker-0.6B")

    def test_bad_input(self):
        self.assertEqual(run_cli("models", "set", "embedding", "nonsense", "--yes")[0], 2)
        self.assertEqual(run_cli("models", "use", "nope", "--yes")[0], 2)
        self.assertEqual(run_cli("models", "download", "nonsense")[0], 2)

    def test_too_large_is_blocked_unless_forced(self):
        run_cli("models", "limit", "3")
        rc, out, err = run_cli("models", "set", "embedding", "Qwen/Qwen3-Embedding-8B", "--yes")
        self.assertEqual(rc, 1)
        self.assertIn("BLOCKED", out)
        self.assertEqual(models.selection("embedding")[1], "default")

    def test_limit(self):
        run_cli("models", "limit", "12")
        self.assertEqual(models.memory_limit_gb(self.paths), 12.0)
        self.assertIn("12.0 GB", run_cli("models", "limit")[1])
        run_cli("models", "limit", "off")
        self.assertEqual(models.memory_limit_gb(self.paths), 0.0)

    def test_download_and_verify_commands(self):
        rc, out, err = run_cli("models", "download")
        self.assertEqual(rc, 0, out + err)
        rc, out, err = run_cli("models", "verify", "reranker")
        self.assertEqual(rc, 0, out + err)
        rc, out, _ = run_cli("models", "status")
        self.assertIn("verify", out)

    def test_setup_runs_every_step_and_each_one_can_be_left_out(self):
        from rag_search.core import diagnostics
        from rag_search.core.conversion import tesseract

        chained = []

        def fake_main(argv):
            chained.append(list(argv))
            raise SystemExit(0)

        def run(*flags):
            chained.clear()
            with mock.patch.object(diagnostics, "download_models", return_value=["x"]), \
                    mock.patch.object(tesseract, "ensure_installed") as tess, \
                    mock.patch.object(cli, "main", fake_main), \
                    mock.patch("platform.system", return_value="Darwin"), mock.patch("platform.machine", return_value="arm64"):
                rc = cli._cmd_setup(cli.build_parser().parse_args(["setup", *flags]))
            return rc, tess

        rc, tess = run()
        self.assertEqual(rc, 0)
        self.assertEqual(chained, [["models", "download", "--reader", "--repair"], ["doctor"], ["daemon", "start"], ["register"]])
        tess.assert_called_once()
        rc, tess = run("--service", "--tool-prefix", "x_", "--no-tesseract")
        self.assertEqual(chained, [["models", "download", "--reader", "--repair"], ["doctor"], ["service", "install"],
                                   ["register", "--tool-prefix", "x_"]])
        tess.assert_not_called()
        rc, tess = run("--skip-reader", "--no-start", "--no-register")
        self.assertEqual(chained, [["doctor"]])
        rc, tess = run("--skip-docling")                          # imported collections only: no reader, no Tesseract
        self.assertEqual(chained, [["doctor"], ["daemon", "start"], ["register"]])
        tess.assert_not_called()
        rc, tess = run("--minimal")
        self.assertEqual(chained, [])

    def test_setup_numbers_its_steps_and_summarises_them(self):
        import contextlib
        import io

        from rag_search.core import diagnostics
        from rag_search.core.conversion import tesseract

        def run(*flags):
            out = io.StringIO()
            with mock.patch.object(diagnostics, "download_models", return_value=["x"]), \
                    mock.patch.object(tesseract, "ensure_installed"), \
                    mock.patch.object(cli, "main", side_effect=SystemExit(0)), \
                    mock.patch("platform.system", return_value="Darwin"), mock.patch("platform.machine", return_value="arm64"), \
                    contextlib.redirect_stdout(out):
                rc = cli._cmd_setup(cli.build_parser().parse_args(["setup", *flags]))
            return rc, out.getvalue()

        rc, out = run()
        self.assertEqual(rc, 0)
        steps = [line for line in out.splitlines() if line.startswith("[")]
        self.assertEqual([line[:5] for line in steps], [f"[{i}/7]" for i in range(1, 8)])
        self.assertRegex(steps[1], r"search models .*GB")                  # what is downloaded and how much
        self.assertRegex(steps[2], r"document reader and the repair model, about \d+\.\d GB")
        self.assertIn("Summary:", out)
        rc, out = run("--skip-models", "--no-tesseract", "--no-start", "--no-register")
        steps = [line for line in out.splitlines() if line.startswith("[")]
        self.assertEqual([line[:5] for line in steps], ["[1/2]", "[2/2]"])    # the data folder and the check
        for left in ("models: left out (--skip-models)", "Tesseract: left out (--no-tesseract)",
                     "daemons: not started (--no-start)", "Claude: not registered (--no-register)"):
            self.assertIn(left, out)

    def test_a_step_that_fails_does_not_stop_the_setup(self):
        from rag_search.core import diagnostics
        from rag_search.core.conversion import tesseract

        def failing(argv):
            raise SystemExit(1)

        with mock.patch.object(diagnostics, "download_models", return_value=[]), \
                mock.patch.object(tesseract, "ensure_installed"), mock.patch.object(cli, "main", failing), \
                mock.patch("platform.system", return_value="Linux"):
            rc = cli._cmd_setup(cli.build_parser().parse_args(["setup"]))
        self.assertEqual(rc, 0)

    def test_setup_with_a_preset(self):
        from rag_search.core import diagnostics

        with mock.patch.object(diagnostics, "download_models", return_value=["x"]) as dl:
            rc, out, err = run_cli("setup", "--minimal", "--models", "qwen3-large")
        self.assertEqual(rc, 0, out + err)
        dl.assert_called_once()
        self.assertEqual(models.selection("embedding")[0], "Qwen/Qwen3-Embedding-4B")
        rc, _, err = run_cli("setup", "--minimal", "--models", "nope", "--skip-models")
        self.assertEqual(rc, 2)

class DoctorTests(TempHome):
    def test_doctor_shows_the_models_in_use_and_a_pending_switch(self):
        from rag_search.core import diagnostics

        os.environ["HF_HOME"] = str(self.tmp / "hf")
        self.write_doc("c/a.md", DOC_A)
        self.index()
        self.publish()
        models.set_selection(self.paths, "embedding", "someone/other-embedder")
        rows = diagnostics.run_checks(self.paths)
        labels = {r[1]: r for r in rows}
        self.assertIn("model someone/other-embedder", labels)
        self.assertIn("embedding model switch", labels)
        self.assertEqual(labels["embedding model switch"][0], diagnostics.WARN)


if __name__ == "__main__":
    unittest.main()
