"""Every tunable, end to end: what may be typed, what is refused, how a value travels from config.json to the process
that reads it, which of default / config.json / environment wins, and that the documents name every variable."""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from tests.helpers import TempHome

from rag_search import config, effective, service, spec, stages
from rag_search.core import chunker, docling_convert
from rag_search.core.conversion import tables

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_INT = {"retrieval_pool": 50, "rerank_pool": 20, "rrf_k": 30, "top_k": 7, "chunk_size": 400, "chunk_overlap": 40,
              "stall_timeout": 7200, "doc_timeout": 600, "docling_batch": 4, "embed_batch": 16, "max_seq": 512,
              "rerank_batch": 4, "rerank_max_len": 512}


class ValidationTests(unittest.TestCase):
    def test_every_whole_number_tunable_has_a_range_and_keeps_to_it(self):
        ints = [t for t in spec.TUNABLES if t.kind == "int"]
        self.assertEqual(sorted(t.key for t in ints), sorted(spec.LIMITS), "a whole-number tunable without a range (or a range without a tunable)")
        for t in ints:
            lo, hi = spec.LIMITS[t.key]
            with self.subTest(t.key):
                self.assertEqual(spec.validate_tunable(t, 0), 0)                       # 0 = the built-in default
                self.assertEqual(spec.validate_tunable(t, ""), 0)
                self.assertEqual(spec.validate_tunable(t, str(SAMPLE_INT[t.key])), SAMPLE_INT[t.key])
                self.assertEqual(spec.validate_tunable(t, lo), lo)
                self.assertEqual(spec.validate_tunable(t, hi), hi)
                for bad in (-1, hi + 1, "many", "1.5"):
                    with self.assertRaises(ValueError, msg=f"{t.key}={bad!r}"):
                        spec.validate_tunable(t, bad)
                if lo > 1:
                    with self.assertRaises(ValueError):
                        spec.validate_tunable(t, lo - 1)
                self.assertTrue(lo <= SAMPLE_INT[t.key] <= hi)

    def test_every_choice_tunable_takes_its_choices_and_nothing_else(self):
        for t in (t for t in spec.TUNABLES if t.kind == "choice"):
            with self.subTest(t.key):
                self.assertTrue(t.choices)
                self.assertEqual(spec.validate_tunable(t, ""), "")
                for c in t.choices:
                    self.assertEqual(spec.validate_tunable(t, c.upper()), c)           # case does not matter
                    self.assertIn(c, t.choice_help, f"{t.key}: the choice {c!r} is not explained")
                with self.assertRaises(ValueError):
                    spec.validate_tunable(t, "no-such-choice")
                self.assertTrue(t.default_label and t.what and t.impact and t.cli_flag.startswith("--"))

    def test_the_search_stages_tunable_needs_a_retriever(self):
        t = spec.TUNABLES_BY_KEY["stages"]
        self.assertEqual(spec.validate_tunable(t, "bm25,rerank"), "bm25,rerank")
        for bad in ("rerank", "bm25,nothing"):
            with self.assertRaises(ValueError):
                spec.validate_tunable(t, bad)

    def test_the_overlap_is_at_most_half_a_chunk(self):
        self.assertEqual(spec.validate_section("indexer", {"chunk_size": 400, "chunk_overlap": 200})["chunk_overlap"], 200)
        with self.assertRaises(ValueError):
            spec.validate_section("indexer", {"chunk_size": 400, "chunk_overlap": 201})
        with self.assertRaises(ValueError):
            spec.validate_section("indexer", {"chunk_overlap": 300})                   # against the default size of 512
        with self.assertRaises(ValueError):
            spec.validate_section("indexer", {"chunk_size": 100})                      # against the default overlap of 64
        text = " ".join(f"word{i}" for i in range(4000))
        for chunk in chunker.split_text(text, 100, 5000):                              # a value that got past validation
            self.assertLessEqual(chunker.est_tokens(chunk), 100 + 50 + 20)

    def test_an_unknown_setting_is_refused(self):
        with self.assertRaises(ValueError):
            spec.validate_section("indexer", {"ocr_frist": "auto"})


class TravelTests(unittest.TestCase):
    """config.json -> the environment of the process that does the work -> the value the code reads."""

    def sample(self, t):
        if t.kind == "int":
            return SAMPLE_INT[t.key]
        if t.kind == "choice":
            return next(c for c in t.choices if c != (t.default_label or "").split(" ")[0]) if len(t.choices) > 1 else t.choices[0]
        return "bm25,dense" if t.kind == "stages" else "eng"

    def test_a_configured_value_reaches_the_environment_and_a_set_variable_wins(self):
        for t in (t for t in spec.TUNABLES if t.env):
            with self.subTest(t.key):
                v = self.sample(t)
                env = effective.settings_env({t.section: {t.key: v}}, {})
                self.assertEqual(env.get(t.env), str(v))
                self.assertEqual(effective.settings_env({t.section: {t.key: v}}, {t.env: "from-the-shell"})[t.env], "from-the-shell")
                self.assertNotIn(t.env, effective.settings_env({t.section: {t.key: t.default}}, {}))   # 0 / "" = no override

    def test_the_value_in_effect_and_where_it_comes_from(self):
        for t in spec.TUNABLES:
            with self.subTest(t.key):
                v = self.sample(t)
                cfg = json.loads(json.dumps(config.DEFAULTS))
                rows = {r["id"]: r for s in effective.resolve(cfg, {})["stages"] for r in s["settings"]}
                self.assertEqual(rows[f"{t.section}.{t.key}"]["source"], "default")
                cfg[t.section][t.key] = v
                if t.key == "chunk_size":
                    cfg[t.section]["chunk_overlap"] = 40
                rows = {r["id"]: r for s in effective.resolve(cfg, {})["stages"] for r in s["settings"]}
                row = rows[f"{t.section}.{t.key}"]
                self.assertEqual(row["source"], "config", row)
                got = row["value"]
                self.assertEqual(str(got).replace(" ", "") if t.kind == "stages" else got,
                                 v if t.kind != "text" else got, f"{t.key}: {got!r}")
                if t.env:
                    other = self.sample(t)
                    rows = {r["id"]: r for s in effective.resolve(cfg, {t.env: str(other)})["stages"] for r in s["settings"]}
                    self.assertEqual(rows[f"{t.section}.{t.key}"]["source"], "environment")

    def test_every_tunable_is_in_the_defaults_of_config_json(self):
        for t in spec.TUNABLES:
            self.assertIn(t.key, config.DEFAULTS[t.section], t.key)
            self.assertEqual(config.DEFAULTS[t.section][t.key], t.default, t.key)


class ConfigFileTests(TempHome):
    def write(self, data):
        self.paths.config_file.parent.mkdir(parents=True, exist_ok=True)
        self.paths.config_file.write_text(json.dumps(data))

    def test_a_value_of_the_wrong_kind_costs_that_setting_only(self):
        self.write({"indexer": {"jobs": "two", "chunk_size": 300, "auto_publish": "yes", "ocr": 5, "stall_timeout": -4,
                                "docling_batch": 2.0, "note_to_self": "kept"},
                    "search": 7, "models": {"memory_limit_gb": 12.5, "embed_batch": True}})
        cfg, err = config.load_config(self.paths)
        self.assertEqual(cfg["indexer"]["jobs"], 0)
        self.assertEqual(cfg["indexer"]["chunk_size"], 300)
        self.assertIs(cfg["indexer"]["auto_publish"], True)
        self.assertEqual(cfg["indexer"]["ocr"], "")
        self.assertEqual(cfg["indexer"]["stall_timeout"], 0)
        self.assertEqual(cfg["indexer"]["docling_batch"], 2)
        self.assertEqual(cfg["indexer"]["note_to_self"], "kept")
        self.assertEqual(cfg["search"], config.DEFAULTS["search"])                    # a section that is not an object
        self.assertEqual(cfg["models"]["memory_limit_gb"], 12.5)
        self.assertEqual(cfg["models"]["embed_batch"], 0)
        for part in ("indexer.jobs", "indexer.auto_publish", "indexer.ocr", "indexer.stall_timeout", "search is not an object",
                     "models.embed_batch"):
            self.assertIn(part, err)
        self.assertEqual(config.effective_jobs(cfg) in (1, 2), True)                  # the daemon can start a run

    def test_a_zero_in_a_text_setting_is_the_old_way_to_say_not_set(self):
        self.write({"search": {"stages": 0, "top_k": 7}, "indexer": {"ocr": 0}})
        cfg, err = config.load_config(self.paths)
        self.assertEqual((cfg["search"]["stages"], cfg["indexer"]["ocr"], cfg["search"]["top_k"], err), ("", "", 7, ""))

    def test_a_good_file_reports_nothing(self):
        self.write({"indexer": {"jobs": 2, "ocr_first": "auto"}})
        cfg, err = config.load_config(self.paths)
        self.assertEqual((cfg["indexer"]["jobs"], cfg["indexer"]["ocr_first"], err), (2, "auto", ""))

    def test_setting_a_value_keeps_the_rest(self):
        self.write({"indexer": {"jobs": 2}, "mine": {"x": 1}})
        config.update_config(self.paths, "indexer", {"ocr_first": "auto"})
        data = json.loads(self.paths.config_file.read_text())
        self.assertEqual(data, {"indexer": {"jobs": 2, "ocr_first": "auto"}, "mine": {"x": 1}})


class EnvironmentTests(unittest.TestCase):
    def test_a_bad_conversion_variable_is_named_in_the_error_and_an_empty_one_is_the_default(self):
        base = docling_convert.convert_settings(env={})
        for var in ("RAG_SEARCH_OCR_ENGINE", "RAG_SEARCH_TABLE_MODE", "RAG_SEARCH_PIPELINE", "RAG_SEARCH_ROUTING",
                    "RAG_SEARCH_PDF_BACKEND", "RAG_SEARCH_THREADS", "RAG_SEARCH_DOC_TIMEOUT", "RAG_SEARCH_DOCLING_BATCH"):
            with self.subTest(var):
                with self.assertRaises(ValueError) as cm:
                    docling_convert.convert_settings(env={var: "nonsense"})
                self.assertIn(var, str(cm.exception))
                self.assertEqual(docling_convert.convert_settings(env={var: ""}), base)
        for var in ("RAG_SEARCH_THREADS", "RAG_SEARCH_DOC_TIMEOUT", "RAG_SEARCH_DOCLING_BATCH"):
            with self.assertRaises(ValueError):
                docling_convert.convert_settings(env={var: "-1"})

    def test_the_documents_name_every_variable_the_code_reads(self):
        code: set[str] = set()
        for f in (ROOT / "src" / "rag_search").rglob("*.py"):
            code |= set(re.findall(r"RAG_SEARCH_[A-Z0-9_]+", f.read_text(encoding="utf-8")))
        readme = set(re.findall(r"RAG_SEARCH_[A-Z0-9_]+", (ROOT / "README.md").read_text(encoding="utf-8")))
        self.assertEqual(sorted(code - readme), [], "read by the code, not in README.md")

    def test_a_service_keeps_every_variable_a_stage_or_tunable_reads(self):
        registry = {t.env for t in spec.TUNABLES if t.env} | {v for s in stages.ALL for v in s.env_only}
        self.assertEqual(sorted(registry - set(service.PASS_ENV)), [])
        for var in ("RAG_SEARCH_VLM", "RAG_SEARCH_OCR_FIRST", "RAG_SEARCH_STALL_TIMEOUT", "RAG_SEARCH_VLM_MODEL", "RAG_SEARCH_JOBS"):
            self.assertIn(var, service.PASS_ENV)
        self.assertNotIn("RAG_SEARCH_CLIENT", service.PASS_ENV)


class SmallFixTests(TempHome):
    def test_a_text_file_in_another_encoding_is_read_as_text(self):
        for name, data in (("u8.txt", "Grüße – ₹ 500".encode("utf-8")), ("bom.txt", b"\xef\xbb\xbf" + "Grüße".encode("utf-8")),
                           ("u16.txt", "Grüße".encode("utf-16")), ("cp.txt", "Grüße".encode("cp1252"))):
            f = self.tmp / name
            f.write_bytes(data)
            self.assertIn("Grüße", docling_convert.read_text_file(f), name)
            out = self.tmp / (name + ".md")
            docling_convert.convert_file(f, out)
            self.assertIn("Grüße", out.read_text(encoding="utf-8"))
            self.assertEqual(f.read_bytes(), data)                                    # the source is only read

    def test_a_pdf_that_cannot_be_opened_is_called_damaged_not_a_scan(self):
        cut = self.tmp / "cut.pdf"
        cut.write_bytes(b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\n" + bytes(range(256)) * 20)
        self.assertIn("damaged or incomplete", docling_convert.damaged_pdf_reason(cut))
        from tests import corpus
        self.assertEqual(docling_convert.damaged_pdf_reason(corpus.copy("pdf/text.pdf", self.tmp / "ok.pdf")), "")

    def test_a_browser_that_goes_away_is_not_a_traceback_in_the_dashboard_log(self):
        import contextlib
        import io

        from rag_search.ui.server import UiServer

        srv = UiServer.__new__(UiServer)
        for exc, logged in ((ConnectionResetError(54, "reset"), False), (BrokenPipeError(), False), (KeyError("x"), True)):
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                try:
                    raise exc
                except Exception:  # noqa: BLE001
                    srv.handle_error(None, ("127.0.0.1", 1))
            self.assertEqual("Traceback" in err.getvalue(), logged, type(exc).__name__)

    def test_a_separator_row_is_recognised_and_a_long_odd_line_costs_nothing(self):
        import time

        for good in ("|---|---|", "| :--- | ---: |", "--- | ---", "  |--|--|  "):
            self.assertTrue(tables._SEP_RE.match(good), good)
        for bad in ("| a | b |", "", "|   |   |", "- item"):
            self.assertFalse(tables._SEP_RE.match(bad), bad)
        t0 = time.perf_counter()
        self.assertFalse(tables._SEP_RE.match("|--" + " " * 200_000 + "x"))
        self.assertFalse(tables._SEP_RE.match("|--" + " |" * 100_000))               # only blanks and bars, but too long
        self.assertLess(time.perf_counter() - t0, 0.5)


if __name__ == "__main__":
    unittest.main()
