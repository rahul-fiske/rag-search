"""Every setting a stage owns really reaches the code that reads it (src/rag_search/effective.py)."""

from __future__ import annotations

import copy
import importlib
import inspect
import os
import unittest
from unittest import mock

from rag_search import config, effective, spec, stages
from rag_search.core import docling_convert, indexer
from rag_search.core.conversion import repair, vlm
from tests.helpers import TempHome

SAMPLE = {"ocr": "smart", "ocr_engine": "tesseract", "ocr_lang": "en,hi", "table_mode": "fast",
          "pdf_backend": "docling-parse", "pipeline": "vlm", "routing": "document", "vlm": "off", "repair": "off",
          "doc_timeout": 99, "docling_batch": 4, "chunk_size": 300, "chunk_overlap": 30,
          "embed_batch": 5, "max_seq": 256, "dtype": "bfloat16", "rerank_batch": 3, "rerank_max_len": 128,
          "device": "cpu"}


def cfg_with(**kv):
    cfg = copy.deepcopy(config.DEFAULTS)
    for k, v in kv.items():
        for section in ("indexer", "models", "search"):
            if k in cfg[section]:
                cfg[section][k] = v
    return cfg


class ReadByTests(unittest.TestCase):
    def test_every_owned_setting_names_a_reader_that_exists(self):
        for s in stages.ALL:
            for setting in s.settings:
                self.assertIn(setting, effective.READ_BY, setting)
                mod, fn, when = effective.READ_BY[setting]
                obj = getattr(importlib.import_module(mod), fn, None)
                self.assertIsNotNone(obj, f"{setting}: {mod}.{fn} does not exist")
                self.assertTrue(when, setting)

    def test_the_environment_variable_of_a_setting_is_read_by_its_module_family(self):
        # the variable is parsed in the reader's module, or in the module that parses the whole family
        family = {"rag_search.core.indexer": "rag_search.core.docling_convert"}
        for t in spec.TUNABLES:
            if not t.env:
                continue
            mod, _fn, _when = effective.READ_BY[f"{t.section}.{t.key}"]
            texts = [inspect.getsource(importlib.import_module(m)) for m in {mod, family.get(mod, mod),
                                                                              "rag_search.core.docling_convert",
                                                                              "rag_search.core.embedding",
                                                                              "rag_search.models"}]
            self.assertTrue(any(t.env in x for x in texts), f"{t.env} is not read where {t.key} says")


class MergeTests(unittest.TestCase):
    def test_config_becomes_the_environment_and_an_ambient_variable_wins(self):
        cfg = cfg_with(**SAMPLE)
        env = effective.settings_env(cfg, {"RAG_SEARCH_OCR": "off"})
        self.assertEqual(env["RAG_SEARCH_OCR"], "off")                         # ambient wins
        self.assertEqual(env["RAG_SEARCH_OCR_ENGINE"], "tesseract")
        self.assertEqual(env["RAG_SEARCH_EMBED_BATCH"], "5")
        self.assertEqual(env["RAG_SEARCH_DOC_TIMEOUT"], "99")
        self.assertNotIn("RAG_SEARCH_CHUNK_SIZE", env)                         # chunk size has no variable
        blank = effective.settings_env(config.DEFAULTS, {})
        self.assertEqual(blank, {})                                            # 0 / "" add nothing

    def test_sections_are_respected(self):
        cfg = cfg_with(**SAMPLE)
        env = effective.settings_env(cfg, {}, ("models",))
        self.assertIn("RAG_SEARCH_DTYPE", env)
        self.assertNotIn("RAG_SEARCH_OCR", env)


class ResolvedValueTests(unittest.TestCase):
    def rows(self, cfg, base=None):
        out = effective.resolve(cfg, base or {})
        return {x["id"]: x for s in out["stages"] for x in s["settings"]}

    def test_each_setting_resolves_to_what_the_parser_reads(self):
        rows = self.rows(cfg_with(**SAMPLE))
        want = {"indexer.ocr": "smart", "indexer.ocr_engine": "tesseract", "indexer.ocr_lang": "en, hi",
                "indexer.table_mode": "fast", "indexer.pdf_backend": "docling-parse", "indexer.pipeline": "vlm",
                "indexer.routing": "document", "indexer.vlm": "off", "indexer.repair": "off",
                "indexer.doc_timeout": 99, "indexer.docling_batch": 4, "indexer.chunk_size": 300,
                "indexer.chunk_overlap": 30, "models.embed_batch": 5, "models.max_seq": 256,
                "models.dtype": "bfloat16", "models.rerank_batch": 3, "models.rerank_max_len": 128,
                "models.device": "cpu"}
        for k, v in want.items():
            self.assertEqual(rows[k]["value"], v, k)
            self.assertEqual(rows[k]["source"], "config", k)

    def test_defaults_say_default(self):
        rows = self.rows(copy.deepcopy(config.DEFAULTS))
        self.assertEqual((rows["indexer.ocr"]["value"], rows["indexer.ocr"]["source"]), ("force", "default"))
        self.assertEqual(rows["indexer.chunk_size"]["value"], 512)
        self.assertEqual(rows["models.embed_batch"]["value"], spec.EMBED_BATCH)
        self.assertEqual(rows["indexer.doc_timeout"]["value"], 2700)

    def test_an_environment_variable_of_the_daemon_wins_and_is_labelled(self):
        rows = self.rows(cfg_with(ocr="smart"), {"RAG_SEARCH_OCR": "off", "RAG_SEARCH_EMBED_BATCH": "7"})
        self.assertEqual((rows["indexer.ocr"]["value"], rows["indexer.ocr"]["source"]), ("off", "environment"))
        self.assertEqual((rows["models.embed_batch"]["value"], rows["models.embed_batch"]["source"]), (7, "environment"))

    def test_an_invalid_value_is_reported_not_raised(self):
        out = effective.resolve(cfg_with(), {"RAG_SEARCH_TABLE_MODE": "slow"})
        self.assertTrue(out["errors"])
        rows = {x["id"]: x for s in out["stages"] for x in s["settings"]}
        self.assertTrue(str(rows["indexer.ocr"]["value"]).startswith("invalid"))

    def test_the_model_choices(self):
        rows = self.rows(cfg_with(embedding="x/embed", reranker="x/rank", reader="x/read"),
                         {"RAG_SEARCH_REPAIR_MODEL": "x/fix"})
        self.assertEqual(rows["models.embedding"]["value"], "x/embed")
        self.assertEqual(rows["models.reranker"]["value"], "x/rank")
        self.assertEqual(rows["models.reader"]["value"], "x/read")
        self.assertEqual((rows["models.repair"]["value"], rows["models.repair"]["source"]), ("x/fix", "environment"))

    def test_every_stage_setting_is_reported_once(self):
        out = effective.resolve(copy.deepcopy(config.DEFAULTS), {})
        ids = [x["id"] for s in out["stages"] for x in s["settings"]]
        self.assertEqual(len(ids), len(set(ids)))
        for s in stages.ALL:
            self.assertEqual(set(s.settings), {x["id"] for st in out["stages"] if st["id"] == s.id
                                               for x in st["settings"]})


class BehaviourTests(TempHome):
    """The value changes what the code does."""

    def test_routing_and_pipeline_decide_whether_pages_are_routed(self):
        prof = {"pages": [{"page": 1}], "kind": "pdf"}
        self.assertTrue(indexer._wants_routing("pdf", prof, None))
        os.environ["RAG_SEARCH_ROUTING"] = "document"
        self.assertFalse(indexer._wants_routing("pdf", prof, None))
        os.environ["RAG_SEARCH_ROUTING"] = "pages"
        os.environ["RAG_SEARCH_PIPELINE"] = "vlm"
        self.assertFalse(indexer._wants_routing("pdf", prof, None))
        self.assertIn("RAG_SEARCH_PIPELINE=vlm", indexer._why_not_routed("pdf", prof, None))
        del os.environ["RAG_SEARCH_PIPELINE"]
        os.environ["RAG_SEARCH_ROUTING"] = "document"
        self.assertIn("RAG_SEARCH_ROUTING=document", indexer._why_not_routed("pdf", prof, None))
        os.environ["RAG_SEARCH_ROUTING"] = "pages"
        self.assertEqual(indexer._why_not_routed("pdf", prof, None), "")
        self.assertIn("no pages", indexer._why_not_routed("pdf", {"pages": []}, None))
        os.environ["RAG_SEARCH_DOCLING_PYTHON"] = "/x/python"
        self.assertIn("RAG_SEARCH_DOCLING_PYTHON", indexer._why_not_routed("pdf", prof, None))
        del os.environ["RAG_SEARCH_DOCLING_PYTHON"]

    def test_the_document_reader_switch(self):
        self.assertEqual(vlm.mode(), "auto")
        os.environ["RAG_SEARCH_VLM"] = "off"
        self.assertEqual(vlm.mode(), "off")
        self.assertIn("|vlm=off", docling_convert.convert_profile())
        self.assertIsNone(vlm.shared())
        self.assertIsNone(repair.build())

    def test_the_repair_switch(self):
        os.environ["RAG_SEARCH_REPAIR"] = "off"
        self.assertEqual(repair.mode(), "off")
        self.assertIsNone(repair.build())
        self.assertIn("|repair=off", docling_convert.convert_profile())

    def test_conversion_settings_reach_the_docling_settings_and_the_fingerprint(self):
        cfg = cfg_with(**SAMPLE)
        env = effective.settings_env(cfg, {})
        with mock.patch.dict(os.environ, env):
            s = docling_convert.convert_settings()
            profile = docling_convert.convert_profile()
        self.assertEqual((s["ocr"], s["engine"], s["lang"], s["table"], s["pipeline"], s["pdf_backend"],
                          s["routing"], s["timeout"], s["batch"]),
                         ("smart", "tesseract", ["en", "hi"], "fast", "vlm", "docling-parse", "document", 99.0, 4))
        for part in ("ocr=smart", "engine=tesseract", "lang=en,hi", "table=fast", "pipeline=vlm", "pdf=docling-parse",
                     "route=document"):
            self.assertIn(part, profile)

    def test_the_device_and_the_precision_are_honoured(self):
        from rag_search.core import embedding

        os.environ["RAG_SEARCH_DEVICE"] = "cpu"
        self.assertEqual(embedding.pick_device(), "cpu")

        class T:
            float16, bfloat16, float32 = "f16", "bf16", "f32"
        os.environ["RAG_SEARCH_DTYPE"] = "bfloat16"
        self.assertEqual(embedding.torch_dtype(T, "cpu"), "bf16")

    def test_the_embedding_and_reranking_batches_are_read_from_the_environment(self):
        from rag_search.core import embedding

        text = inspect.getsource(embedding)
        for var in ("RAG_SEARCH_EMBED_BATCH", "RAG_SEARCH_MAX_SEQ", "RAG_SEARCH_RERANK_BATCH",
                    "RAG_SEARCH_RERANK_MAX_LEN"):
            self.assertIn(var, text)

    def test_the_search_daemon_applies_the_model_settings_before_it_loads_them(self):
        from rag_search.core import search_daemon

        search_daemon.apply_model_env({"embed_batch": 6, "device": "cpu", "dtype": ""})
        try:
            self.assertEqual(os.environ["RAG_SEARCH_EMBED_BATCH"], "6")
            self.assertEqual(os.environ["RAG_SEARCH_DEVICE"], "cpu")
            self.assertNotIn("RAG_SEARCH_DTYPE", os.environ)
        finally:
            os.environ.pop("RAG_SEARCH_EMBED_BATCH", None)

    def test_the_daemons_report_the_overrides_they_were_started_with(self):
        from rag_search.core.indexer_daemon import IndexerDaemon

        os.environ["RAG_SEARCH_OCR"] = "off"
        with mock.patch.dict(os.environ):
            os.environ.pop("RAG_SEARCH_LAYER_FILL", None)          # the test helpers switch the text-layer fill off for every test
            self.assertEqual(IndexerDaemon(self.paths).ping_info()["env_overrides"], {"RAG_SEARCH_OCR": "off"})


if __name__ == "__main__":
    unittest.main()
