"""The stage registry (src/rag_search/stages.py) matches the settings and the code that reads them."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from rag_search import spec, stages

SRC = Path(__file__).resolve().parents[2] / "src" / "rag_search"


def _source_text() -> str:
    parts = []
    for p in SRC.rglob("*.py"):
        if p.name in ("spec.py", "stages.py"):
            continue
        parts.append(p.read_text(encoding="utf-8"))
    return "\n".join(parts)


class RegistryTests(unittest.TestCase):
    def test_ids_are_unique_and_children_have_parents(self):
        ids = [s.id for s in stages.ALL]
        self.assertEqual(len(ids), len(set(ids)))
        for s in stages.ALL:
            if s.parent:
                self.assertIn(s.parent, stages.BY_ID, s.id)
        self.assertEqual([s.id for s in stages.INDEXING if not s.parent], [str(i) for i in range(1, 9)])
        self.assertEqual([s.id for s in stages.children("3")], ["3.0", "3.1", "3.2", "3.3", "3.4", "3.5", "3.6", "3.7", "3.8", "3.9", "3.10", "3.11"])
        self.assertEqual([s.id for s in stages.SEARCH], ["S1", "S2", "S3", "S3b", "S4", "S5", "S6"])

    def test_every_tunable_belongs_to_exactly_one_stage(self):
        for t in spec.TUNABLES:
            key = f"{t.section}.{t.key}"
            owners = stages.owners_of(key)
            self.assertEqual(len(owners), 1, f"{key} is owned by {[o.id for o in owners]}")

    def test_a_stage_with_parts_puts_every_setting_and_variable_in_exactly_one_part(self):
        seen = 0
        for s in stages.ALL:
            if not s.groups:
                continue
            seen += 1
            listed = [x for g in s.groups for x in g.settings]
            self.assertEqual(sorted(listed), sorted(s.settings), s.id)             # all of them, none twice
            envs = [x for g in s.groups for x in g.env_only]
            self.assertEqual(sorted(envs), sorted(s.env_only), s.id)
            self.assertEqual(len({g.id for g in s.groups}), len(s.groups))
            self.assertTrue(all(g.id.startswith(s.id) and g.name and g.what for g in s.groups))
        self.assertGreaterEqual(seen, 1)
        read = stages.BY_ID["3.2"]
        self.assertEqual([g.id for g in read.groups], ["3.2", "3.2a", "3.2b", "3.2c", "3.2d"])   # the router, then the lanes
        d = [x for x in stages.describe() if x["id"] == "3.2"][0]
        self.assertEqual([g["id"] for g in d["groups"]], ["3.2", "3.2a", "3.2b", "3.2c", "3.2d"])

    def test_every_listed_setting_exists(self):
        known = {f"{t.section}.{t.key}" for t in spec.TUNABLES} | set(stages.CONFIG_KEYS)
        for s in stages.ALL:
            for setting in s.settings:
                self.assertIn(setting, known, f"{s.id} lists {setting}")
            for setting, others in s.also_used_by.items():
                self.assertIn(setting, s.settings)
                for o in others:
                    self.assertIn(o, stages.BY_ID)

    def test_the_indexer_and_model_settings_sit_in_the_stage_that_reads_them(self):
        want = {"indexer.chunk_size": "4", "indexer.chunk_overlap": "4", "indexer.ocr": "3.2", "indexer.vlm": "3.2",
                "indexer.repair": "3.4", "models.repair": "3.4", "models.embed_batch": "5", "models.device": "5",
                "search.rrf_k": "S4", "models.rerank_batch": "S5", "search.top_k": "S6", "models.describe": "3.10"}
        for setting, sid in want.items():
            self.assertEqual(stages.owner_of(setting).id, sid, setting)

    def test_environment_only_variables_are_read_somewhere(self):
        text = _source_text()
        for s in stages.ALL:
            for var in s.env_only:
                self.assertIn(var, text, f"{s.id}: {var} is not read by any module")

    def test_tunable_environment_variables_are_read_where_the_stage_runs(self):
        text = _source_text()
        for t in spec.TUNABLES:
            if t.env:
                self.assertRegex(text, re.escape(t.env), t.env)

    def test_the_browser_copy_of_the_keys_equals_the_registry(self):
        js = (SRC / "ui" / "static" / "pipeline.js").read_text(encoding="utf-8")
        block = js.split("const STAGE_KEYS = {", 1)[1].split("};", 1)[0]
        found = {k: (i, n) for k, i, n in re.findall(r"(\w+): \['([\w.]+)', '([^']+)'\]", block)}
        want = {s.key: (s.id, s.name) for s in stages.ALL}
        self.assertEqual(found, want)

    def test_every_kind_takes_steps_of_3_convert_and_has_a_converter_and_the_browser_knows_it(self):
        from rag_search import kinds
        from rag_search.core import indexer
        from rag_search.core.conversion import trace

        for k in kinds.KINDS:
            for sid in k.steps:
                self.assertTrue(sid.startswith("3.") and sid in stages.BY_ID, (k.key, sid))
        self.assertEqual(set(indexer.CONVERT), {k.key for k in kinds.KINDS})
        self.assertEqual(trace.WHOLE_FILES, {k.branch for k in kinds.KINDS if k.branch})
        self.assertEqual(kinds.route_of_branches({"photo": 1}), "photograph")
        self.assertEqual(kinds.route_of_branches({"digital": 3, "raster": 1}), "pages")
        js = (SRC / "ui" / "static" / "pipeline.js").read_text(encoding="utf-8")
        block = js.split("const KINDS = {", 1)[1].split("};", 1)[0]
        found = {key: (b, one, many, tuple(st.split())) for key, b, one, many, st in
                 re.findall(r"(\w+): \['([\w]*)', '([^']+)', '([^']+)', '([^']+)'\]", block)}
        self.assertEqual(found, {k.key: (k.branch, k.label, k.plural, k.steps) for k in kinds.KINDS})

    def test_the_architecture_view_draws_a_band_for_every_kind_with_its_own_steps(self):
        from rag_search import kinds

        js = (SRC / "ui" / "static" / "architecture.js").read_text(encoding="utf-8")
        for k in kinds.KINDS:
            for sid in k.steps:                                    # each of the kind's steps is a box with its number
                self.assertIn(f"'{sid} · ", js, (k.key, sid))
        for heading in ("DOCUMENTS OF PAGES", "PHOTOGRAPHS", "VIDEOS", "GOOGLE MAPS TIMELINES"):
            self.assertIn(f"'{heading} ·", js, heading)
        self.assertEqual(js.count("      band("), len(kinds.KINDS))              # one band per kind

    def test_describe_is_json_ready(self):
        import json

        d = stages.describe()
        json.dumps(d)
        self.assertEqual(d[0]["id"], "1")
        self.assertTrue(any(x["id"] == "3.4" and x["optional"] for x in d))


if __name__ == "__main__":
    unittest.main()
