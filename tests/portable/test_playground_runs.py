"""Playground runs as jobs: started in the background, watched through the same job record / event log
machinery as production, with the numbered stages; per-experiment reader/repair pins and the settings
the experiment really uses."""

from __future__ import annotations

import json
import os
import time
import unittest

from rag_search import playground_runs as runs
from rag_search.core import playground as pg
from tests.helpers import TempHome
from tests.portable.test_cli_api import run
from tests.portable.test_ui import UiBase


def wait_done(base, name, jid="", timeout=120):
    end = time.time() + timeout
    while time.time() < end:
        v = runs.view(base, name, jid)
        if v["job"] and v["job"]["status"] not in ("queued", "running"):
            return v
        time.sleep(0.4)
    raise AssertionError("the run did not finish: " + json.dumps(runs.view(base, name, jid), default=str)[:800])


class _Base(TempHome):
    def setUp(self):
        super().setUp()
        self.use_fake_backends()
        self.src = self.tmp / "docs"
        self.src.mkdir()
        (self.src / "doc.txt").write_text("<!-- page 1 -->\nhow is a session token refreshed\n"
                            "<!-- page 2 -->\nhow to reset a forgotten password\n", encoding="utf-8")
        pg.create_experiment(self.paths, "demo", sources=[str(self.src)])


class RunLifecycleTests(_Base):
    def test_index_run_records_numbered_stages_and_documents(self):
        started = runs.start(self.paths, "demo", "index")
        self.assertEqual(started["kind"], "index")
        v = wait_done(self.paths, "demo", started["job"])
        self.assertEqual(v["job"]["status"], "done", v["job"].get("error"))
        self.assertEqual(v["job"]["kind"], "index")
        self.assertEqual(v["job"]["summary"]["indexed"], 1)
        item = v["documents"]["items"][0]
        self.assertEqual(item["status"], "indexed")
        stages = item["timeline"]["stages"]
        # the same numbers as production: 2 fingerprint, 3 convert, 4 chunk, 5 embed, 6 write
        self.assertEqual({k: s["id"] for k, s in stages.items() if k in ("fingerprint", "convert", "chunk", "embed", "write")},
                         {"fingerprint": "2", "convert": "3", "chunk": "4", "embed": "5", "write": "6"})
        self.assertEqual(stages["embed"]["status"], "done")

    def test_second_run_is_skipped_at_stage_2(self):
        wait_done(self.paths, "demo", runs.start(self.paths, "demo", "index")["job"])
        v = wait_done(self.paths, "demo", runs.start(self.paths, "demo", "index")["job"])
        item = v["documents"]["items"][0]
        self.assertEqual(item["status"], "skipped")
        self.assertTrue(item["timeline"]["stages"]["fingerprint"]["unchanged"])

    def test_only_one_run_at_a_time(self):
        first = runs.start(self.paths, "demo", "index", ["--rebuild"])
        with self.assertRaises(runs.RunError) as cm:
            runs.start(self.paths, "demo", "bench")
        self.assertIn("already running", str(cm.exception))
        wait_done(self.paths, "demo", first["job"])
        self.assertIsNone(runs.active(self.paths, "demo"))

    def test_a_failing_run_says_why(self):
        pg.create_experiment(self.paths, "empty")
        v = wait_done(self.paths, "empty", runs.start(self.paths, "empty", "index")["job"])
        self.assertEqual(v["job"]["status"], "failed")
        self.assertIn("no source folders", v["job"]["error"])

    def test_a_dead_process_is_not_left_running(self):
        exp = pg.get_playground_paths(self.paths, "demo")
        exp.jobs.mkdir(parents=True, exist_ok=True)
        from rag_search.jobs import job_file
        from rag_search.paths import write_json_atomic

        write_json_atomic(job_file(exp, "20200101-000000-dead"), {
            "id": "20200101-000000-dead", "kind": "index", "status": "running", "pid": 2 ** 22 + 12345,
            "created_at": time.time() - 100, "started_at": time.time() - 90})
        v = runs.view(self.paths, "demo", "20200101-000000-dead")
        self.assertEqual(v["job"]["status"], "failed")
        self.assertIn("ended without finishing", v["job"]["error"])
        self.assertIsNone(runs.active(self.paths, "demo"))

    def test_bench_run_reports_each_query(self):
        wait_done(self.paths, "demo", runs.start(self.paths, "demo", "index")["job"])
        (self.paths.home / "playground" / "demo" / "bench" / "queries.jsonl").write_text(
            json.dumps({"query": "session token refreshed", "relevant": [{"file": "doc.txt", "page": 1}]}) + "\n"
            + json.dumps({"query": "reset forgotten password", "relevant": [{"file": "doc.txt", "page": 2}]}) + "\n",
            encoding="utf-8")
        v = wait_done(self.paths, "demo", runs.start(self.paths, "demo", "bench", ["-k", "3"])["job"])
        self.assertEqual(v["job"]["status"], "done", v["job"].get("error"))
        self.assertEqual([q["i"] for q in v["queries"]], [1, 2])
        self.assertTrue(all("total_ms" in q for q in v["queries"]))
        self.assertEqual(len(runs.history(self.paths, "demo")), 2)

    def test_cancel_stops_the_run(self):
        started = runs.start(self.paths, "demo", "index", ["--rebuild"])
        res = runs.cancel(self.paths, "demo")
        # either it was still running (cancelled) or it had already finished: never left "running"
        v = runs.view(self.paths, "demo", started["job"])
        self.assertNotIn(v["job"]["status"], ("queued", "running"), res)

    def test_cli_status_prints_the_stages(self):
        wait_done(self.paths, "demo", runs.start(self.paths, "demo", "index")["job"])
        rc, out, err = run("playground", "status", "demo")
        self.assertEqual(rc, 0, err)
        self.assertIn("2 done", out)
        self.assertIn("5 done", out)


class ModelPinTests(_Base):
    def test_pins_are_validated_and_can_be_cleared(self):
        cfg = pg.update_config(self.paths, "demo", reader_model="mlx-community/Qwen3-VL-4B-Instruct-4bit")
        self.assertEqual(cfg["reader_model"], "mlx-community/Qwen3-VL-4B-Instruct-4bit")
        with self.assertRaises(pg.PlaygroundError):
            pg.update_config(self.paths, "demo", repair_model="not a model id")
        cfg = pg.update_config(self.paths, "demo", reader_model="production")
        self.assertEqual(cfg["reader_model"], "")

    def test_a_pinned_model_reaches_the_reading_code_and_is_restored(self):
        from rag_search import models

        pg.update_config(self.paths, "demo", reader_model="mlx-community/PaddleOCR-VL-1.5-8bit",
                         repair_model="mlx-community/Qwen3-VL-4B-Instruct-4bit")
        cfg = pg.get_config(self.paths, "demo")
        os.environ.pop("RAG_SEARCH_VLM_MODEL", None)
        os.environ.pop("RAG_SEARCH_REPAIR_MODEL", None)
        with pg._experiment_env(self.paths, cfg):
            self.assertEqual(models.reader_choice()[0], "mlx-community/PaddleOCR-VL-1.5-8bit")
            self.assertEqual(models.repair_choice()[0], "mlx-community/Qwen3-VL-4B-Instruct-4bit")
        self.assertNotIn("RAG_SEARCH_VLM_MODEL", os.environ)
        self.assertNotEqual(models.reader_choice()[0], "mlx-community/PaddleOCR-VL-1.5-8bit")

    def test_an_environment_variable_still_wins(self):
        pg.update_config(self.paths, "demo", reader_model="mlx-community/PaddleOCR-VL-1.5-8bit")
        os.environ["RAG_SEARCH_VLM_MODEL"] = "org/from-environment"
        self.addCleanup(os.environ.pop, "RAG_SEARCH_VLM_MODEL", None)
        added = pg.experiment_env(self.paths, pg.get_config(self.paths, "demo"))
        self.assertNotIn("RAG_SEARCH_VLM_MODEL", added)
        res = pg.effective_settings(self.paths, "demo")
        row = next(r for s in res["stages"] for r in s["settings"] if r["id"] == "models.reader")
        self.assertEqual((row["value"], row["source"]), ("org/from-environment", "environment"))

    def test_promote_carries_a_pinned_reader_and_ignores_a_blank_one(self):
        pg.update_config(self.paths, "demo", reader_model="mlx-community/PaddleOCR-VL-1.5-8bit")
        prev = pg.promotion_preview(self.paths, "demo")
        self.assertIn("reader_model", prev["changes"])
        self.assertNotIn("repair_model", prev["changes"])
        self.assertFalse(prev["needs_reindex"])

    def test_cli_config_shows_production_choice_when_blank(self):
        rc, out, err = run("playground", "config", "demo")
        self.assertEqual(rc, 0, err)
        self.assertIn("reader_model: (production's choice)", out)
        rc, out, _ = run("playground", "config", "demo", "--reader-model", "org/model", "--json")
        self.assertEqual(json.loads(out)["reader_model"], "org/model")


class EffectiveSettingsTests(_Base):
    def test_every_indexing_stage_but_publish(self):
        res = pg.effective_settings(self.paths, "demo")
        ids = [s["id"] for s in res["stages"]]
        self.assertEqual(ids, ["1", "2", "3", "3.1", "3.2", "3.3", "3.4", "3.5", "4", "5", "6", "7"])
        self.assertIn("8 Publish does not exist", res["note"])

    def test_values_and_sources_come_from_the_experiment(self):
        pg.update_config(self.paths, "demo", chunk_size=333, ocr_lang="en")
        res = pg.effective_settings(self.paths, "demo")
        rows = {r["id"]: r for s in res["stages"] for r in s["settings"]}
        self.assertEqual((rows["indexer.chunk_size"]["value"], rows["indexer.chunk_size"]["source"]), (333, "experiment"))
        self.assertEqual(rows["indexer.ocr_lang"]["source"], "experiment")
        self.assertEqual(rows["indexer.table_mode"]["source"], "default")
        self.assertNotIn("indexer.jobs", rows)
        self.assertNotIn("indexer.auto_publish", rows)

    def test_the_value_shown_is_the_value_the_run_applies(self):
        pg.update_config(self.paths, "demo", ocr_engine="rapidocr", doc_timeout=77)
        cfg = pg.get_config(self.paths, "demo")
        added = pg.experiment_env(self.paths, cfg)
        for t in pg.DOCLING_TUNABLES:
            if cfg.get(t.key):
                self.assertEqual(added.get(t.env), str(cfg[t.key]), t.key)


class PlaygroundRunApiTests(UiBase):
    def setUp(self):
        super().setUp()
        src = self.tmp / "pgdocs"
        src.mkdir()
        (src / "pg.txt").write_text("<!-- page 1 -->\nhow is a session token refreshed\n", encoding="utf-8")
        self.post("create", {"name": "demo"})
        # the API has no upload: the sample is copied in the way the CLI does it
        run("playground", "create", "withdoc", "--from", str(src))

    def post(self, action, body):
        st, js, _, _ = self.dash.req("POST", f"/api/playground/{action}", body)
        return st, js

    def test_index_returns_at_once_and_the_run_can_be_watched(self):
        st, js = self.post("index", {"name": "withdoc"})
        self.assertEqual(st, 200)
        self.assertTrue(js["ok"], js)
        jid = js["result"]["job"]
        end = time.time() + 120
        while time.time() < end:
            st, js = self.post("run", {"name": "withdoc", "run": jid})
            if js["result"]["job"]["status"] not in ("queued", "running"):
                break
            time.sleep(0.4)
        self.assertEqual(js["result"]["job"]["status"], "done", js)
        st, js = self.post("runs", {"name": "withdoc"})
        self.assertEqual(js["result"][0]["id"], jid)
        st, js = self.post("settings", {"name": "withdoc"})
        self.assertTrue(js["ok"], js)
        self.assertEqual([s["id"] for s in js["result"]["stages"]][:3], ["1", "2", "3"])

    def test_no_documents_is_reported_by_the_run(self):
        st, js = self.post("index", {"name": "demo"})
        self.assertTrue(js["ok"], js)
        jid = js["result"]["job"]
        end = time.time() + 60
        while time.time() < end:
            st, js = self.post("run", {"name": "demo", "run": jid})
            if js["result"]["job"]["status"] not in ("queued", "running"):
                break
            time.sleep(0.3)
        self.assertEqual(js["result"]["job"]["status"], "failed")
        self.assertIn("no source folders", js["result"]["job"]["error"])

    def test_a_bad_run_id_is_a_400(self):
        st, js = self.post("run", {"name": "demo", "run": "../../x"})
        self.assertEqual(st, 400)


if __name__ == "__main__":
    unittest.main()
