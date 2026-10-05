import json
import os
import unittest
from pathlib import Path

from tests.helpers import TempHome  # noqa: F401  (sets sys.path)
from rag_search.core import bm25, chunker
from rag_search import config, policy, protocol
from rag_search.paths import (
    DEFAULT_COLLECTION, collection_of, get_paths, index_dir_for, markup_path_for,
    mirror_rel, parse_collections,
)


class PathsTests(TempHome):
    def test_mirror_and_default_collection(self):
        d = self.paths.docs
        a = self.write_doc("manuals/net/guide.pdf", "x")
        b = self.write_doc("top.pdf", "x")
        self.assertEqual(mirror_rel(a, d), Path("manuals/net/guide.pdf"))
        self.assertEqual(mirror_rel(b, d), Path(DEFAULT_COLLECTION) / "top.pdf")
        self.assertEqual(collection_of(a, d), "manuals")
        self.assertEqual(index_dir_for(a, d, self.paths.index),
                         self.paths.index / "manuals/net/guide")
        self.assertEqual(markup_path_for(b, d, self.paths.markup),
                         self.paths.markup / "default/top.md")
        self.assertEqual(self.paths.index, self.paths.workspace / "index")

    def test_outside_docs_root(self):
        with self.assertRaises(ValueError):
            mirror_rel(self.tmp / "elsewhere.pdf", self.paths.docs)

    def test_parse_collections(self):
        self.assertEqual(parse_collections(" a, b ,,"), ["a", "b"])
        self.assertEqual(parse_collections(""), [])
        for bad in ("../x", "a/b", ".."):
            with self.assertRaises(ValueError):
                parse_collections(bad)

    def test_long_home_socket_falls_back_to_tmp(self):
        long_home = self.tmp / ("x" * 120)
        p = get_paths(long_home)
        for kind in ("search", "indexer"):
            self.assertLessEqual(len(str(p.socket(kind))), 100)
            self.assertNotIn(str(long_home), str(p.socket(kind)))
        self.assertNotEqual(p.socket("search"), p.socket("indexer"))

    def test_layout_and_no_current_generation(self):
        p = self.paths
        self.assertEqual(p.socket("search").name, "search.sock")
        self.assertIsNone(p.current_gen())
        self.assertIsNone(p.live_index())
        (p.serving / "gen-000001").mkdir(parents=True)
        p.current_link.symlink_to("gen-000001")
        self.assertEqual(p.current_gen(), (p.serving / "gen-000001").resolve())
        self.assertEqual(p.live_markup(), (p.serving / "gen-000001").resolve() / "markup")

    def test_env_docs_override(self):
        os.environ["RAG_SEARCH_DOCS"] = str(self.tmp / "mydocs")
        self.assertEqual(get_paths().docs, self.tmp / "mydocs")


class ConfigPolicyProtocolTests(TempHome):
    def test_defaults_file_and_env_override(self):
        cfg, err = config.load_config(self.paths)
        self.assertEqual(err, "")
        self.assertEqual(cfg["search"]["idle_exit_seconds"], 0)  # always on
        self.paths.config_file.write_text(json.dumps({"search": {"prewarm": False}}))
        cfg, _ = config.load_config(self.paths)
        self.assertFalse(cfg["search"]["prewarm"])
        self.assertTrue(cfg["indexer"]["auto_publish"])  # merged, not replaced
        self.assertNotIn("policy", cfg)                  # access lives in access.json
        os.environ["RAG_SEARCH_IDLE_SECONDS"] = "60"
        cfg, _ = config.load_config(self.paths)
        self.assertEqual(cfg["search"]["idle_exit_seconds"], 60)

    def test_broken_config_falls_back(self):
        self.paths.config_file.write_text("{ nope")
        cfg, err = config.load_config(self.paths)
        self.assertIn("using defaults", err)
        self.assertEqual(cfg["indexer"]["auto_publish"], True)

    def test_config_store_rereads_on_change(self):
        store = config.ConfigStore(self.paths)
        self.assertTrue(store.get()["search"]["prewarm"])
        self.paths.config_file.write_text(json.dumps({"search": {"prewarm": False}}))
        self.assertFalse(store.get()["search"]["prewarm"])

    def test_effective_jobs(self):
        self.assertEqual(config.effective_jobs({"indexer": {"jobs": 3}}), 3)
        self.assertIn(config.effective_jobs({"indexer": {"jobs": 0}}), (1, 2))

    def test_everything_is_open_without_rules(self):
        rules, err = policy.load_rules(self.paths)
        self.assertEqual(err, "")
        ex = ["manuals", "personal"]
        for client in ("claude", "agent", "cli", "brand-new", "unknown"):
            self.assertEqual(policy.resolve_scope(rules, client, [], ex), (ex, ""))
            self.assertEqual(rules.visible(client, ex), ex)

    def test_restricted_collection_rules(self):
        policy.save_rules(self.paths, {"personal": ["claude"], "hr": []})
        rules, err = policy.load_rules(self.paths)
        self.assertEqual(err, "")
        ex = ["hr", "manuals", "personal"]
        self.assertEqual(policy.resolve_scope(rules, "claude", [], ex), (["manuals", "personal"], ""))
        self.assertEqual(policy.resolve_scope(rules, "claude", ["personal"], ex), (["personal"], ""))
        self.assertEqual(policy.resolve_scope(rules, "agent", [], ex), (["manuals"], ""))
        self.assertEqual(policy.resolve_scope(rules, "weird-new-client", [], ex), (["manuals"], ""))
        self.assertEqual(policy.resolve_scope(rules, "agent", ["manuals"], ex), (["manuals"], ""))
        self.assertEqual(rules.visible("cli", ex), ex)            # the admin's terminal sees all
        self.assertEqual(rules.visible("claude", ex), ["manuals", "personal"])  # hr: nobody
        self.assertEqual(sorted(rules.hidden_from("agent")), ["hr", "personal"])
        self.assertEqual(rules.hidden_from("cli"), [])
        self.assertEqual(policy.normalize_client("Bad Client!"), "unknown")
        self.assertEqual(policy.normalize_client("Claude"), "claude")

    def test_a_restricted_name_looks_exactly_like_a_missing_one(self):
        policy.save_rules(self.paths, {"personal": ["claude"]})
        rules, _ = policy.load_rules(self.paths)
        ex = ["manuals", "personal"]
        hidden = policy.resolve_scope(rules, "agent", ["personal"], ex)
        missing = policy.resolve_scope(rules, "agent", ["nothing-here"], ex)
        self.assertEqual(hidden[0], [])
        self.assertEqual(hidden[1].replace("personal", "X"), missing[1].replace("nothing-here", "X"))
        self.assertNotIn("personal", hidden[1].split("(available")[1])   # not in the hints either
        # a differently-cased spelling resolves to the real name -- for a client allowed to use it
        self.assertEqual(policy.resolve_scope(rules, "claude", ["PERSONAL"], ex), (["personal"], ""))
        self.assertEqual(policy.resolve_scope(rules, "agent", ["PERSONAL"], ex)[0], [])
        self.assertFalse(rules.allows("agent", "PERSONAL"))          # case-folded restriction

    def test_scope_dedupes_and_prefers_exact_spelling(self):
        rules, _ = policy.load_rules(self.paths)
        ex = ["Docs", "docs", "hr"]
        self.assertEqual(policy.resolve_scope(rules, "x", ["hr", "HR", "hr"], ex), (["hr"], ""))
        self.assertEqual(policy.resolve_scope(rules, "x", ["docs"], ex), (["docs"], ""))
        # two collections that differ only in case: an inexact spelling is ambiguous -> error
        self.assertEqual(policy.resolve_scope(rules, "x", ["DOCS"], ex)[0], [])
        from rag_search.paths import parse_collections
        self.assertEqual(parse_collections("a, b,a"), ["a", "b"])

    def test_access_store_rereads_and_a_damaged_file_fails_closed(self):
        store = policy.AccessStore(self.paths)
        self.assertEqual(store.get().by_name, {})
        policy.save_rules(self.paths, {"hr": ["claude"]})
        self.assertEqual(store.get().by_name, {"hr": ["claude"]})
        self.paths.access_file.write_text('{"collections": {"hr": ["claude"], "x": [')
        rules = store.get()
        self.assertTrue(store.error)
        self.assertFalse(rules.allows("claude", "hr"))   # unreadable: closed, not open
        self.assertTrue(rules.allows("agent", "manuals"))
        self.paths.access_file.write_text('{"collections": {"hr": "claude"}}')
        self.assertTrue(policy.load_rules(self.paths)[1])
        self.assertEqual(oct(os.stat(policy.save_rules(self.paths, {"a": []})).st_mode & 0o777), "0o600")

    def test_scrub_hides_names_in_free_text_only_at_word_boundaries(self):
        out = policy.scrub({"a": "indexing docs/hr/pay.pdf", "b": ["chrome ok", "the HR folder"],
                            "n": 3}, ["hr"])
        self.assertEqual(out, {"a": "[restricted]", "b": ["chrome ok", "[restricted]"], "n": 3})
        self.assertEqual(policy.scrub({"a": "x"}, []), {"a": "x"})

    def test_protocol_validation(self):
        req, err = protocol.validate_request({"v": 1, "action": "ping", "client": "CLI"})
        self.assertIsNone(err)
        self.assertEqual(req["client"], "cli")
        _, err = protocol.validate_request({"v": 99, "action": "ping"})
        self.assertEqual(err["code"], protocol.PROTOCOL_MISMATCH)
        _, err = protocol.validate_request({"v": 1})
        self.assertEqual(err["code"], protocol.BAD_REQUEST)
        _, err = protocol.validate_request([1])
        self.assertEqual(err["code"], protocol.BAD_REQUEST)


class ChunkerTests(unittest.TestCase):
    def test_pages_and_headings(self):
        md = ("<!-- page 1 -->\n\n# Intro\n\nHello world.\n\n"
              "<!-- page 2 -->\n\nMore text on page two.\n\n## Details\n\nDetail text.")
        nodes = chunker.markdown_to_nodes(md, 512, 64)
        pages = [n["page"] for n in nodes]
        self.assertEqual(pages, ["1", "2"])
        self.assertEqual(nodes[0]["heading"], "Intro")
        self.assertEqual(nodes[1]["heading"], "Intro")  # inherited until a new heading
        self.assertIn("Detail text", nodes[1]["text"])

    def test_sizes_respected(self):
        para = " ".join(f"Sentence number {i} is here." for i in range(400))
        chunks = chunker.split_text(para, 100, 10)
        self.assertGreater(len(chunks), 5)
        for c in chunks:
            self.assertLessEqual(chunker.est_tokens(c), 130)

    def test_table_split_repeats_header(self):
        rows = "\n".join(f"| row{i} | value {i} |" for i in range(200))
        table = "| name | value |\n|---|---|\n" + rows
        chunks = chunker.split_text(table, 80, 0)
        self.assertGreater(len(chunks), 3)
        for c in chunks:
            self.assertTrue(c.startswith("| name | value |"), c[:40])

    def test_empty(self):
        self.assertEqual(chunker.markdown_to_nodes("  \n", 512, 64), [])


class Bm25Tests(unittest.TestCase):
    def test_tokenize_compounds(self):
        t = bm25.tokenize("Use svm-name with ONTAP 9.16.1 and a/b")
        for want in ("svm-name", "svm", "name", "9.16.1", "9", "16", "a/b", "ontap"):
            self.assertIn(want, t)

    def test_ranking(self):
        docs = ["the cat sat", "dogs chase cats", "authentication and access control"]
        b = bm25.BM25([bm25.tokenize(d) for d in docs])
        s = b.scores(bm25.tokenize("access control"))
        self.assertEqual(int(s.argmax()), 2)
        self.assertEqual(float(s[0]), 0.0)

    def test_empty_index(self):
        b = bm25.BM25([])
        self.assertEqual(b.scores(["x"]).size, 0)


if __name__ == "__main__":
    unittest.main()


class LegacyTorchLoadTests(unittest.TestCase):
    """Intel Macs (PyTorch 2.2.2): transformers refuses torch.load unless we relax its check."""

    def setUp(self):
        import sys
        import types
        from rag_search.core import embedding

        self.embedding = embedding
        self.mod = types.ModuleType("transformers.fake_for_test")

        def refuse():
            raise ValueError("Due to a serious vulnerability issue in `torch.load`")

        self.mod.check_torch_load_is_safe = refuse
        self.refuse = refuse
        # Not mock.patch.dict(sys.modules): undoing that also forgets every module imported during the
        # test, and importing torch a second time in one process crashes the interpreter.
        sys.modules["transformers.fake_for_test"] = self.mod
        self.addCleanup(sys.modules.pop, "transformers.fake_for_test", None)
        self.addCleanup(lambda: os.environ.pop("RAG_SEARCH_ALLOW_UNSAFE_TORCH_LOAD", None))

    def _with_torch(self, ver):
        from unittest import mock
        return mock.patch.object(self.embedding, "_torch_version", return_value=ver)

    def test_old_torch_and_official_model_is_relaxed(self):
        with self._with_torch((2, 2)):
            self.assertTrue(self.embedding.allow_trusted_legacy_torch_load("BAAI/bge-m3"))
        self.mod.check_torch_load_is_safe()          # no longer raises

    def test_unknown_model_is_not_relaxed(self):
        with self._with_torch((2, 2)):
            self.assertFalse(self.embedding.allow_trusted_legacy_torch_load("someone/else"))
        with self.assertRaises(ValueError):
            self.mod.check_torch_load_is_safe()

    def test_opt_in_env_relaxes_any_model(self):
        os.environ["RAG_SEARCH_ALLOW_UNSAFE_TORCH_LOAD"] = "1"
        with self._with_torch((2, 2)):
            self.assertTrue(self.embedding.allow_trusted_legacy_torch_load("someone/else"))

    def test_new_torch_is_left_alone(self):
        with self._with_torch((2, 6)):
            self.assertFalse(self.embedding.allow_trusted_legacy_torch_load("BAAI/bge-m3"))
        with self.assertRaises(ValueError):
            self.mod.check_torch_load_is_safe()


class ConvertSettingsTests(unittest.TestCase):
    """RAG_SEARCH_OCR & co: parsing, and the profile that invalidates converted documents."""

    KEYS = ("RAG_SEARCH_OCR", "RAG_SEARCH_OCR_ENGINE", "RAG_SEARCH_OCR_LANG",
            "RAG_SEARCH_TABLE_MODE", "RAG_SEARCH_PIPELINE", "RAG_SEARCH_PDF_BACKEND")

    def setUp(self):
        from rag_search.core import docling_convert
        self.dc = docling_convert
        saved = {k: os.environ.pop(k, None) for k in self.KEYS}

        def restore():
            for k, v in saved.items():
                os.environ.pop(k, None)
                if v is not None:
                    os.environ[k] = v
        self.addCleanup(restore)

    def test_defaults_are_force_ocr_with_pypdfium2(self):
        # = `docling convert --force-ocr --pdf-backend pypdfium2`, the best setup for our tables
        s = self.dc.convert_settings()
        self.assertEqual((s["ocr"], s["engine"], s["table"], s["pipeline"], s["pdf_backend"]),
                         ("force", "auto", "accurate", "standard", "pypdfium2"))

    def test_pdf_backend_spellings_and_profile(self):
        base = self.dc.convert_profile()
        self.assertIn("|pdf=pypdfium2", base)
        for raw, want in [("PyPdfium2", "pypdfium2"), ("docling", "docling-parse"), ("docling_parse", "docling-parse"),
                          ("auto", "default"), ("default", "default"), ("", "pypdfium2")]:
            os.environ["RAG_SEARCH_PDF_BACKEND"] = raw
            self.assertEqual(self.dc.convert_settings()["pdf_backend"], want, raw)
        os.environ["RAG_SEARCH_PDF_BACKEND"] = "default"
        self.assertNotEqual(base, self.dc.convert_profile())      # changing it reconverts documents
        os.environ["RAG_SEARCH_PDF_BACKEND"] = "mupdf"
        with self.assertRaises(ValueError):
            self.dc.convert_settings()

    def test_ocr_spellings(self):
        for raw, want in [("0", "off"), ("off", "off"), ("1", "auto"), ("on", "auto"),
                          ("force", "force"), ("FORCE", "force"), ("auto", "auto")]:
            os.environ["RAG_SEARCH_OCR"] = raw
            self.assertEqual(self.dc.convert_settings()["ocr"], want, raw)

    def test_legacy_ocr_flag_overrides_off(self):
        os.environ["RAG_SEARCH_OCR"] = "off"
        self.assertEqual(self.dc.convert_settings(ocr=True)["ocr"], "auto")

    def test_engine_lang_table(self):
        os.environ.update({"RAG_SEARCH_OCR_ENGINE": "OcrMac", "RAG_SEARCH_OCR_LANG": "en-US, de-DE",
                           "RAG_SEARCH_TABLE_MODE": "fast"})
        s = self.dc.convert_settings()
        self.assertEqual((s["engine"], s["lang"], s["table"]), ("ocrmac", ["en-US", "de-DE"], "fast"))

    def test_bad_values_are_rejected(self):
        for key, val in [("RAG_SEARCH_OCR_ENGINE", "abbyy"), ("RAG_SEARCH_TABLE_MODE", "slow"),
                         ("RAG_SEARCH_PIPELINE", "magic"), ("RAG_SEARCH_PDF_BACKEND", "mupdf")]:
            os.environ[key] = val
            with self.assertRaises(ValueError):
                self.dc.convert_settings()
            self.assertTrue(self.dc.convert_profile().endswith("invalid"))
            os.environ.pop(key)

    def test_profile_changes_with_settings(self):
        base = self.dc.convert_profile()
        os.environ["RAG_SEARCH_TABLE_MODE"] = "fast"
        self.assertNotEqual(base, self.dc.convert_profile())
        os.environ.pop("RAG_SEARCH_TABLE_MODE")
        self.assertEqual(base, self.dc.convert_profile())


class SmartOcrTests(unittest.TestCase):
    """RAG_SEARCH_OCR=smart: trust a clean text layer, OCR everything else."""

    def setUp(self):
        from rag_search.core import docling_convert
        self.dc = docling_convert
        saved = {k: os.environ.pop(k, None) for k in ("RAG_SEARCH_OCR", "RAG_SEARCH_THREADS",
                                                       "RAG_SEARCH_DOC_TIMEOUT",
                                                       "RAG_SEARCH_DOCLING_BATCH")}
        def restore():
            for k, v in saved.items():
                os.environ.pop(k, None)
                if v is not None:
                    os.environ[k] = v
        self.addCleanup(restore)

    def test_settings(self):
        os.environ["RAG_SEARCH_OCR"] = "smart"
        self.assertEqual(self.dc.convert_settings()["ocr"], "smart")
        os.environ["RAG_SEARCH_OCR"] = "adaptive"
        self.assertEqual(self.dc.convert_settings()["ocr"], "smart")
        self.assertIn("ocr=smart", self.dc.convert_profile())

    def test_threads_and_timeout_settings_do_not_change_the_profile(self):
        before = self.dc.convert_profile()
        os.environ["RAG_SEARCH_THREADS"] = "6"
        os.environ["RAG_SEARCH_DOC_TIMEOUT"] = "60"
        cs = self.dc.convert_settings()
        self.assertEqual((cs["threads"], cs["timeout"]), (6, 60.0))
        self.assertEqual(self.dc.convert_profile(), before)
        self.assertEqual(self.dc.DEFAULT_DOC_TIMEOUT, 2700)      # 45 minutes
        self.assertEqual(self.dc.convert_settings()["batch"], 8)
        os.environ["RAG_SEARCH_DOCLING_BATCH"] = "16"
        try:
            self.assertEqual(self.dc.convert_settings()["batch"], 16)
            self.assertEqual(self.dc.convert_profile(), before)
        finally:
            del os.environ["RAG_SEARCH_DOCLING_BATCH"]

    def test_batch_sizes_are_applied_where_docling_has_them(self):
        class Opts:                       # a docling version with two of the three fields
            layout_batch_size = 4
            ocr_batch_size = 4

        o = Opts()
        self.dc._apply_batch_sizes(o, 8)
        self.assertEqual((o.layout_batch_size, o.ocr_batch_size), (8, 8))
        self.assertFalse(hasattr(o, "table_batch_size"))
        self.dc._apply_batch_sizes(o, 0)  # 0 = leave docling's own values
        self.assertEqual(o.layout_batch_size, 8)

    def test_bad_numbers_are_rejected(self):
        for var, val in (("RAG_SEARCH_THREADS", "many"), ("RAG_SEARCH_DOC_TIMEOUT", "-1")):
            os.environ[var] = val
            with self.assertRaises(ValueError):
                self.dc.convert_settings()
            del os.environ[var]

    def test_page_text_judgement(self):
        ok = self.dc._page_text_ok
        self.assertTrue(ok("• To enable “multi-factor” authentication — open Settings and select Security. " * 3))
        self.assertFalse(ok(""))
        self.assertFalse(ok("\ufffd" * 50 + "abc"))                       # replacement characters
        self.assertFalse(ok("\ue001\ue002 abcdef " * 30))                 # private-use glyphs
        self.assertFalse(ok("Toenablemultifactorauthenticationandthenselectsecurity " * 6))  # lost spaces

    def test_resolve_follows_the_probe(self):
        from unittest import mock
        cfg = dict(self.dc.convert_settings(), ocr="smart")
        good = {"reliable": True, "reason": "clean text layer"}
        bad = {"reliable": False, "reason": "scanned"}
        with mock.patch.object(self.dc, "text_layer_report", return_value=good):
            c, why = self.dc.resolve_ocr_mode(Path("a.pdf"), cfg)
        self.assertEqual(c["ocr"], "auto")
        self.assertIn("clean text layer", why)
        with mock.patch.object(self.dc, "text_layer_report", return_value=bad):
            c, why = self.dc.resolve_ocr_mode(Path("a.pdf"), cfg)
        self.assertEqual(c["ocr"], "force")
        self.assertIn("scanned", why)
        c, why = self.dc.resolve_ocr_mode(Path("a.docx"), cfg)     # no text layer to judge
        self.assertEqual((c["ocr"], why), ("auto", ""))
        c, why = self.dc.resolve_ocr_mode(Path("a.pdf"), dict(cfg, ocr="force"))
        self.assertEqual((c["ocr"], why), ("force", ""))         # other modes are left alone

class ConvertReuseTests(unittest.TestCase):
    """One converter serves many documents; a time-out is not retried with another backend."""

    def setUp(self):
        from unittest import mock
        from rag_search.core import docling_convert
        self.dc = docling_convert
        self.dc._CONVERTERS.clear()
        self.addCleanup(self.dc._CONVERTERS.clear)
        self.built = []
        self.status = "SUCCESS"
        outer = self
        Doc = type("Doc", (), {"num_pages": lambda s: 1,
                               "export_to_markdown": lambda s, page_no=None: "# hi"})

        def build(suffix, cfg):
            outer.built.append(cfg["pdf_backend"])
            def convert(path):
                return type("R", (), {"document": Doc(), "status": types_status(outer.status)})()
            return type("C", (), {"convert": staticmethod(convert)})()

        def types_status(name):
            return type("S", (), {"name": name})()

        p = mock.patch.object(docling_convert, "_build_converter", build)
        p.start()
        self.addCleanup(p.stop)
        self.tmp = Path(__import__("tempfile").mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))
        saved = {k: os.environ.pop(k, None) for k in ("RAG_SEARCH_OCR", "RAG_SEARCH_PDF_BACKEND")}
        self.addCleanup(lambda: [os.environ.__setitem__(k, v) for k, v in saved.items() if v is not None])

    def test_converter_is_built_once_per_setting(self):
        for i in range(3):
            self.dc.convert_file(Path(f"d{i}.pdf"), self.tmp / f"d{i}.md")
        self.assertEqual(self.built, ["pypdfium2"])
        os.environ["RAG_SEARCH_PDF_BACKEND"] = "docling-parse"
        self.dc.convert_file(Path("e.pdf"), self.tmp / "e.md")
        self.assertEqual(self.built, ["pypdfium2", "docling-parse"])

    def test_cache_holds_at_most_two_converters(self):
        cfg = self.dc.convert_settings()
        for backend in ("pypdfium2", "docling-parse", "default"):
            self.dc._cached_converter(".pdf", dict(cfg, pdf_backend=backend))
        self.assertEqual(len(self.dc._CONVERTERS), 2)

    def test_partial_result_after_the_time_limit_is_an_error_without_retry(self):
        from unittest import mock
        self.status = "PARTIAL_SUCCESS"
        os.environ["RAG_SEARCH_DOC_TIMEOUT"] = "5"
        try:
            with mock.patch.object(self.dc.time, "perf_counter", side_effect=[0.0, 0.0, 6.0, 6.0, 6.0, 6.0]):
                with self.assertRaises(self.dc.ConversionTimeout) as cm:
                    self.dc.convert_file(Path("slow.pdf"), self.tmp / "slow.md")
        finally:
            os.environ.pop("RAG_SEARCH_DOC_TIMEOUT", None)
        self.assertIn("RAG_SEARCH_DOC_TIMEOUT", str(cm.exception))
        self.assertEqual(self.built, ["pypdfium2"])            # the other backend was not tried

    def test_partial_result_within_the_limit_is_kept(self):
        self.status = "PARTIAL_SUCCESS"              # e.g. one page failed: still usable
        r = self.dc.convert_file(Path("p.pdf"), self.tmp / "p.md")
        self.assertEqual(r["pages"], 1)


class ConvertFallbackTests(unittest.TestCase):
    """A PDF one backend cannot read is retried once with the other backend."""

    def setUp(self):
        from unittest import mock
        from rag_search.core import docling_convert
        self.dc = docling_convert
        self.tmp = Path(__import__("tempfile").mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))
        self.calls = []
        self.dc._CONVERTERS.clear()
        self.addCleanup(self.dc._CONVERTERS.clear)
        saved = {k: os.environ.pop(k, None) for k in ("RAG_SEARCH_PDF_BACKEND", "RAG_SEARCH_OCR")}
        self.addCleanup(lambda: [os.environ.__setitem__(k, v) for k, v in saved.items() if v is not None])
        Doc = type("Doc", (), {"num_pages": lambda s: 1,
                               "export_to_markdown": lambda s, page_no=None: "# hello"})
        outer = self

        def build(suffix, cfg):
            outer.calls.append(cfg["pdf_backend"])
            ok = cfg["pdf_backend"] in outer.works
            def convert(path):
                if not ok:
                    raise UnicodeDecodeError("utf-8", b"\xe4", 0, 1, "invalid continuation byte")
                return type("R", (), {"document": Doc()})()
            return type("C", (), {"convert": staticmethod(convert)})()
        p = mock.patch.object(docling_convert, "_build_converter", build)
        p.start()
        self.addCleanup(p.stop)

    def run_convert(self, works):
        self.works, self.calls[:] = works, []
        return self.dc.convert_file(Path("x.pdf"), self.tmp / "x.md")

    def test_default_backend_works_no_retry(self):
        self.run_convert({"pypdfium2"})
        self.assertEqual(self.calls, ["pypdfium2"])

    def test_falls_back_to_the_other_backend_once(self):
        r = self.run_convert({"docling-parse"})
        self.assertEqual(self.calls, ["pypdfium2", "docling-parse"])
        self.assertEqual(r["pages"], 1)
        self.assertIn("<!-- page 1 -->", (self.tmp / "x.md").read_text())

    def test_both_failing_reports_both_causes_with_location(self):
        with self.assertRaises(RuntimeError) as cm:
            self.run_convert(set())
        msg = str(cm.exception)
        self.assertEqual(self.calls, ["pypdfium2", "docling-parse"])
        self.assertIn("UnicodeDecodeError", msg)
        self.assertIn("docling-parse backend also failed", msg)
        self.assertIn("[at portable/test_core.py:", msg)

    def test_other_formats_are_not_retried(self):
        self.works = set()
        self.calls[:] = []
        with self.assertRaises(UnicodeDecodeError):
            self.dc.convert_file(Path("x.docx"), self.tmp / "y.md")
        self.assertEqual(len(self.calls), 1)

    def test_an_encrypted_pdf_is_reported_as_such(self):
        pdf = self.tmp / "locked.pdf"
        pdf.write_bytes(b"%PDF-1.6\n1 0 obj<<>>endobj\ntrailer<</Root 1 0 R/Encrypt 9 0 R>>\n%%EOF\n")
        self.works = set()
        with self.assertRaises(self.dc.ProtectedPdfError) as cm:
            self.dc.convert_file(pdf, self.tmp / "locked.md")
        self.assertRegex(str(cm.exception), r"(?i)(password-protected|encrypted) PDF")
        self.assertIn("unprotected copy", str(cm.exception))
        plain = self.tmp / "plain.pdf"
        plain.write_bytes(b"%PDF-1.6\ntrailer<</Root 1 0 R>>\n%%EOF\n")
        with self.assertRaises(RuntimeError) as cm2:      # a different failure keeps its own message
            self.dc.convert_file(plain, self.tmp / "plain.md")
        self.assertNotIsInstance(cm2.exception, self.dc.ProtectedPdfError)

    def test_a_document_without_text_raises_no_text_error(self):
        self.dc._CONVERTERS.clear()
        Empty = type("Doc", (), {"num_pages": lambda s: 1,
                                 "export_to_markdown": lambda s, page_no=None: "  "})
        from unittest import mock
        with mock.patch.object(self.dc, "_convert_document", lambda *a, **k: Empty()):
            with self.assertRaises(self.dc.NoTextError) as cm:
                self.dc.convert_file(Path("photo.png"), self.tmp / "photo.md")
        self.assertIn("no text extracted from photo.png", str(cm.exception))
        self.assertTrue(issubclass(self.dc.NoTextError, RuntimeError))


class ConvertSidecarTests(TempHome):
    """A converted .md is reused only for the same source AND the same conversion settings."""

    def test_reuse_and_invalidation(self):
        from rag_search.core import indexer
        src = Path(self.tmp) / "a.md"
        src.write_text("# T\n\nhello world\n")
        md = Path(self.tmp) / "out" / "a.md"
        os.environ.pop("RAG_SEARCH_TABLE_MODE", None)
        self.addCleanup(lambda: os.environ.pop("RAG_SEARCH_TABLE_MODE", None))
        self.assertEqual(indexer.convert_source(src, md, force=False, src_sha="s1"), "converted")
        self.assertEqual(indexer.convert_source(src, md, force=False, src_sha="s1"), "reused")
        self.assertEqual(indexer.convert_source(src, md, force=False, src_sha="s2"), "converted")
        os.environ["RAG_SEARCH_TABLE_MODE"] = "fast"
        self.assertEqual(indexer.convert_source(src, md, force=False, src_sha="s2"), "converted")
        # an old-format sidecar (source SHA only) is treated as stale
        md.with_name("a.md.sha256").write_text("s2\n")
        os.environ.pop("RAG_SEARCH_TABLE_MODE")
        self.assertEqual(indexer.convert_source(src, md, force=False, src_sha="s2"), "converted")


class BuildConverterTests(unittest.TestCase):
    """_build_converter against a stub 'docling' (the real one is not installed in CI)."""

    def setUp(self):
        import sys
        import types
        from unittest import mock

        class Opt:                                   # any attribute is settable, like a pydantic model
            def __init__(self, **kw):
                self.__dict__.update(kw)

        class Ocr(Opt):
            force_full_page_ocr = False
            lang = ["auto"]

        class OcrMacOptions(Ocr):
            pass

        class Table(Opt):
            mode = None

        class Pdf(Opt):
            def __init__(self):
                self.do_ocr = True
                self.ocr_options = Ocr()
                self.table_structure_options = Table()

        class Mode:
            ACCURATE, FAST = "accurate", "fast"

        po = types.ModuleType("docling.datamodel.pipeline_options")
        po.PdfPipelineOptions, po.TableFormerMode, po.OcrMacOptions = Pdf, Mode, OcrMacOptions
        self.po = po
        bm = types.ModuleType("docling.datamodel.base_models")
        bm.InputFormat = types.SimpleNamespace(PDF="pdf", IMAGE="image")
        dc = types.ModuleType("docling.document_converter")
        dc.DocumentConverter = lambda format_options: format_options
        dc.PdfFormatOption = Opt
        dc.ImageFormatOption = Opt
        dm = types.ModuleType("docling.datamodel")
        dm.pipeline_options = po
        mods = {"docling": types.ModuleType("docling"), "docling.datamodel": dm,
                "docling.datamodel.pipeline_options": po, "docling.datamodel.base_models": bm,
                "docling.document_converter": dc}
        # the backend modules of docling (only the classes this test asks for exist)
        self.backends = {}
        for mod, cls in [("docling.backend.pypdfium2_backend", "PyPdfiumDocumentBackend"),
                         ("docling.backend.docling_parse_v2_backend", "DoclingParseV2DocumentBackend")]:
            m = types.ModuleType(mod)
            setattr(m, cls, type(cls, (), {}))
            self.backends[cls] = getattr(m, cls)
            mods[mod] = m
        mods["docling.backend"] = types.ModuleType("docling.backend")
        p = mock.patch.dict(sys.modules, mods)
        p.start()
        self.addCleanup(p.stop)
        from rag_search.core import docling_convert
        self.dc = docling_convert

    def cfg(self, **kw):
        base = {"ocr": "auto", "engine": "auto", "lang": [], "table": "accurate", "pipeline": "standard"}
        base.update(kw)
        return base

    def test_default_pdf_and_image_share_accurate_ocr_options(self):
        fmt = self.dc._build_converter(".pdf", self.cfg())
        opts = fmt["pdf"].pipeline_options
        self.assertTrue(opts.do_ocr)
        self.assertEqual(opts.table_structure_options.mode, "accurate")
        self.assertIs(fmt["image"].pipeline_options, opts)

    def test_threads_and_time_limit_reach_docling(self):
        import sys
        import types
        ao = types.ModuleType("docling.datamodel.accelerator_options")
        ao.AcceleratorOptions = lambda **kw: types.SimpleNamespace(**kw)
        with __import__("unittest").mock.patch.dict(sys.modules, {"docling.datamodel.accelerator_options": ao}):
            opts = self.dc._build_converter(".pdf", self.cfg(threads=6, timeout=90.0))["pdf"].pipeline_options
        self.assertEqual((opts.accelerator_options.num_threads, opts.accelerator_options.device), (6, "auto"))
        self.assertEqual(opts.document_timeout, 90.0)

    def test_no_threads_no_limit_leaves_docling_alone(self):
        opts = self.dc._build_converter(".pdf", self.cfg(threads=0, timeout=0.0))["pdf"].pipeline_options
        self.assertFalse(hasattr(opts, "accelerator_options"))
        self.assertFalse(hasattr(opts, "document_timeout"))

    def test_off_still_ocrs_images_but_not_pdfs(self):
        self.assertFalse(self.dc._build_converter(".pdf", self.cfg(ocr="off"))["pdf"].pipeline_options.do_ocr)
        self.assertTrue(self.dc._build_converter(".png", self.cfg(ocr="off"))["pdf"].pipeline_options.do_ocr)

    def test_force_engine_lang_and_fast_tables(self):
        opts = self.dc._build_converter(
            ".pdf", self.cfg(ocr="force", engine="ocrmac", lang=["en-US"], table="fast")
        )["pdf"].pipeline_options
        self.assertEqual(type(opts.ocr_options).__name__, "OcrMacOptions")
        self.assertTrue(opts.ocr_options.force_full_page_ocr)
        self.assertEqual(opts.ocr_options.lang, ["en-US"])
        self.assertEqual(opts.table_structure_options.mode, "fast")

    def test_pypdfium2_backend_is_passed_to_the_pdf_format_only(self):
        fmt = self.dc._build_converter(".pdf", self.cfg(pdf_backend="pypdfium2"))
        self.assertIs(fmt["pdf"].backend, self.backends["PyPdfiumDocumentBackend"])
        self.assertFalse(hasattr(fmt["image"], "backend"))       # images keep docling's image backend

    def test_default_backend_leaves_docling_alone(self):
        fmt = self.dc._build_converter(".pdf", self.cfg(pdf_backend="default"))
        self.assertFalse(hasattr(fmt["pdf"], "backend"))

    def test_docling_parse_uses_the_newest_installed_generation(self):
        fmt = self.dc._build_converter(".pdf", self.cfg(pdf_backend="docling-parse"))
        self.assertIs(fmt["pdf"].backend, self.backends["DoclingParseV2DocumentBackend"])

    def test_backend_missing_in_this_docling_is_a_clear_error(self):
        import sys
        from unittest import mock
        with mock.patch.dict(sys.modules, {"docling.backend.pypdfium2_backend": None}):
            with self.assertRaises(RuntimeError) as cm:
                self.dc._build_converter(".pdf", self.cfg(pdf_backend="pypdfium2"))
        self.assertIn("RAG_SEARCH_PDF_BACKEND=default", str(cm.exception))

    def test_force_uses_ocr_mode_on_new_docling_and_the_flag_on_old(self):
        class Mode:
            FULL_PAGE = "full_page"
        self.po.OcrMode = Mode
        try:
            opts = self.dc._build_converter(".pdf", self.cfg(ocr="force"))["pdf"].pipeline_options
            self.assertEqual(opts.ocr_options.mode, "full_page")
            self.assertFalse(opts.ocr_options.force_full_page_ocr)
        finally:
            del self.po.OcrMode
        opts = self.dc._build_converter(".pdf", self.cfg(ocr="force"))["pdf"].pipeline_options
        self.assertTrue(opts.ocr_options.force_full_page_ocr)

    def test_engine_missing_in_this_docling_is_a_clear_error(self):
        with self.assertRaises(RuntimeError) as cm:
            self.dc._build_converter(".pdf", self.cfg(engine="rapidocr"))
        self.assertIn("RAG_SEARCH_OCR_ENGINE=auto", str(cm.exception))


class CloudFileTests(unittest.TestCase):
    def test_allowing_cloud_files_is_harmless_and_repeatable(self):
        import sys

        from rag_search import paths

        first = paths.allow_cloud_files()
        self.assertEqual(paths.allow_cloud_files(), first)
        if sys.platform != "darwin":
            self.assertFalse(first)                  # nothing to switch on outside macOS

    def test_a_file_the_cloud_app_could_not_fetch_is_explained(self):
        from unittest import mock

        from rag_search.core import docling_convert

        exc = OSError(11, "Resource deadlock avoided")
        with mock.patch.object(docling_convert.sys, "platform", "darwin"):
            self.assertIn("online-only", docling_convert.describe_error(exc))
        with mock.patch.object(docling_convert.sys, "platform", "linux"):
            self.assertNotIn("online-only", docling_convert.describe_error(exc))
