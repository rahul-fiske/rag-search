"""Descriptions of photographs: the model's reply, the word filter, the cache, the lines in a photograph's record, and that
neither a map refresh nor a rebuild runs the model again.  The model is replaced by a stand-in."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

from tests.portable.test_maps import fake_reader as fake_overture
from tests.portable.test_photo import IndexedPhotoTests, jpeg
from tests.portable.test_ui import Dash

from rag_search import maps, models
from rag_search.core import describe, indexer, photo

REPLY = ("Description: A tiger walks through a dry, wooded area with bare trees.\n"
         "Tags: tiger, tree, forest, indoor, winter, Dry, tree\n"
         "People: 0")


class StandIn:
    """Looks like a started ``vlm.VlmReader``; answers with REPLY and counts the questions."""

    def __init__(self, reply=REPLY):
        self.reply, self.calls = reply, 0
        self.worker = self

    def ensure_started(self):
        pass

    def read(self, image, prompt, *, timeout, max_tokens):
        assert Path(image).is_file() and "Describe this photograph" in prompt
        self.calls += 1
        return {"md": self.reply}

    def close(self):
        pass


class ParseTests(IndexedPhotoTests):
    def test_labelled_and_unlabelled_replies(self):
        rec = describe.parse(REPLY)
        self.assertEqual(rec["description"], "A tiger walks through a dry, wooded area with bare trees.")
        self.assertEqual(rec["tags"], ["tiger", "tree", "forest", "dry"])         # indoor and winter dropped, no repeats
        self.assertEqual(rec["people"], 0)
        plain = describe.parse("A boy smiles.\n\nboy, smile, Sunny, red\n\n3")
        self.assertEqual((plain["description"], plain["tags"], plain["people"]), ("A boy smiles.", ["boy", "smile", "red"], 3))
        broken = describe.parse("Description: Two camels.")
        self.assertEqual((broken["tags"], broken["people"]), ([], None))
        self.assertEqual(describe.parse("")["description"], "")

    def test_lines_are_marked_as_machine_written(self):
        out = describe.lines(describe.parse(REPLY), "Qwen3-VL 8B (4-bit), prompt d1")
        self.assertEqual(out[0], "Looks like (model-written): A tiger walks through a dry, wooded area with bare trees.")
        self.assertEqual(out[1], "Things seen (model-written): tiger, tree, forest, dry")
        self.assertEqual(out[2:], ["People in the picture (model count): 0", "Described by: Qwen3-VL 8B (4-bit), prompt d1"])
        self.assertEqual(describe.lines({"description": "", "tags": [], "people": None}, "x"), [])

    def test_the_describe_model_is_one_of_two_and_defaults_to_the_8b(self):
        self.assertEqual(models.describe_selection(), (models.DESCRIBE_MODELS[0], "default"))
        models.set_describe_model(self.paths, models.DESCRIBE_MODELS[1])
        self.assertEqual(models.describe_selection(), (models.DESCRIBE_MODELS[1], "config"))
        self.assertIn("4B", describe.current_id())
        with self.assertRaises(models.ModelError):
            models.set_describe_model(self.paths, "someone/else")


class SectionTests(IndexedPhotoTests):
    def test_lines_carry_over_a_renewed_section_and_set_extra_replaces_by_label(self):
        src = jpeg(self.tmp / "a.jpg")
        md = self.tmp / "a.md"
        md.write_text("<!-- page 1 -->\n\n", encoding="utf-8")
        photo.attach(md, photo.block(src, self.paths, "EXIT 4"))
        extra = describe.lines(describe.parse(REPLY), "M, prompt d1") + ["Faces detected: 2"]
        self.assertTrue(photo.set_extra(md, extra))
        text = md.read_text(encoding="utf-8")
        self.assertIn("- Looks like (model-written): A tiger", text)
        self.assertIn("- Faces detected: 2", text)
        self.assertTrue(text.index("Described by") < text.index("Reference data") if "Reference data" in text else True)
        photo.attach(md, photo.block(src, self.paths))                 # a map refresh renews the section ...
        again = md.read_text(encoding="utf-8")
        for kept in ("Looks like (model-written)", "Things seen", "People in the picture", "Faces detected: 2", "Described by: M, prompt d1",
                     "Text in the picture: EXIT 4"):
            self.assertIn(kept, again)                                 # ... and keeps what others wrote
        self.assertEqual(again.count("## Photo details"), 1)
        photo.set_extra(md, ["Faces detected: 5", *describe.lines(describe.parse("Description: A boy.\nTags: boy\nPeople: 1"), "N, prompt d1")])
        text = md.read_text(encoding="utf-8")
        self.assertEqual((text.count("Faces detected:"), text.count("Looks like"), "A tiger" in text), (1, 1, False))
        self.assertIn("Faces detected: 5", text)
        self.assertFalse(photo.set_extra(md, []))


class DescribeRunTests(IndexedPhotoTests):
    def photo_md(self):
        (self.sdir / "c").mkdir(exist_ok=True)
        src = self.sdir / "c" / "p.jpg"
        if not src.exists():
            jpeg(src)
        with mock.patch.object(photo, "visible_text", return_value="EXIT 4"):
            self.run_index()
        return next(self.paths.markup.rglob("p.md"))

    def test_describe_writes_the_lines_caches_them_and_never_repeats(self):
        md = self.photo_md()
        st = describe.status(self.paths)
        self.assertEqual((st["photographs"], st["described"], st["pending"]), (1, 0, 1))
        model = StandIn()
        res = describe.describe_all(self.paths, reader=model)
        self.assertEqual((res["candidates"], res["described"], res["updated"]), (1, 1, 1))
        text = md.read_text(encoding="utf-8")
        self.assertIn("- Looks like (model-written): A tiger walks", text)
        self.assertIn(f"- Described by: {describe.current_id()}", text)
        self.assertIn("Text in the picture: EXIT 4", text)
        if photo.faces_in(Path(next(Path(self.tmp).rglob("*.jpg")))) is not None:
            self.assertRegex(text, r"- Faces detected: \d")
        self.assertFalse((self.paths.index / "c" / "p" / "index.meta.json").exists())   # marked for embedding
        self.assertEqual(describe.status(self.paths)["described"], 1)
        again = describe.describe_all(self.paths, reader=model)
        self.assertEqual((again["candidates"], model.calls), (0, 1))                    # nothing to do, no question asked
        self.run_index()                                                                # the next run embeds it
        nodes = json.loads(next(self.paths.index.rglob("p/nodes.json")).read_text())["nodes"]
        self.assertTrue(any("A tiger walks" in n["text"] for n in nodes))

    def test_a_map_refresh_keeps_the_description_and_does_not_ask_the_model(self):
        md = self.photo_md()
        model = StandIn()
        describe.describe_all(self.paths, reader=model)
        self.run_index()
        maps.write_stamp(self.paths, md, md.read_text(encoding="utf-8"))
        maps.download_overture(self.paths, ["19.05,78.30"], release="r1", reader=fake_overture())
        res = maps.refresh(self.paths)
        self.assertEqual((res["refreshed"], res["in_place"]), (1, 1))
        with mock.patch.object(photo, "visible_text", side_effect=AssertionError("the picture must not be read")):
            self.run_index()
        text = md.read_text(encoding="utf-8")
        self.assertIn("- Nearby: Hotel Sai Krishna", text)                              # the new place lines
        self.assertIn("- Looks like (model-written): A tiger walks", text)              # the description, untouched
        self.assertEqual(model.calls, 1)
        self.assertEqual(describe.describe_all(self.paths, reader=model)["candidates"], 0)

    def test_a_rebuild_puts_the_cached_description_back_without_the_model(self):
        md = self.photo_md()
        model = StandIn()
        describe.describe_all(self.paths, reader=model)
        srcs = indexer.scan_sources(self.sdir, indexer.exclude_dirs(self.paths))
        md.unlink()
        with mock.patch.object(photo, "visible_text", return_value="EXIT 4"):
            indexer.run_index(self.paths, srcs, self.roots(), jobs=1, embedder=__import__("tests.helpers", fromlist=["x"]).FakeEmbedder(),
                              progress=lambda e: None, rebuild=True, force_md=True)
        text = next(self.paths.markup.rglob("p.md")).read_text(encoding="utf-8")
        self.assertIn("- Looks like (model-written): A tiger walks", text)
        self.assertEqual(model.calls, 1)

    def test_a_new_model_or_prompt_describes_again_and_redo_forces_it(self):
        self.photo_md()
        model = StandIn()
        describe.describe_all(self.paths, reader=model)
        models.set_describe_model(self.paths, models.DESCRIBE_MODELS[1])
        st = describe.status(self.paths)
        self.assertEqual((st["described"], st["described_by_another_model"], st["pending"]), (0, 1, 1))
        res = describe.describe_all(self.paths, reader=model)
        self.assertEqual((res["candidates"], res["described"]), (1, 1))
        self.assertEqual(model.calls, 2)
        text = next(self.paths.markup.rglob("p.md")).read_text(encoding="utf-8")
        self.assertEqual(text.count("Described by:"), 1)
        self.assertIn("4B", text)
        self.assertEqual(describe.describe_all(self.paths, reader=model, redo=True)["described"], 0)   # same id: the cache answers
        self.assertEqual(model.calls, 2)

    def test_a_photographed_page_is_not_described(self):
        (self.sdir / "c").mkdir()
        jpeg(self.sdir / "c" / "page.jpg")

        def convert(src, out, ocr=None):
            out.write_text("<!-- page 1 -->\n\n" + "The agreement is made between the parties. " * 10 + "\n", encoding="utf-8")
            return {"pages": 1, "seconds": 0.0}
        with mock.patch.object(photo, "visible_text", return_value="word " * 120), \
                mock.patch("rag_search.core.docling_convert.convert_file", convert):
            self.run_index()
        self.assertEqual(describe.status(self.paths)["photographs"], 0)
        self.assertEqual(describe.describe_all(self.paths, reader=StandIn())["candidates"], 0)

    def test_a_model_that_writes_nothing_leaves_the_photograph_as_it_was(self):
        md = self.photo_md()
        before = md.read_text(encoding="utf-8")
        res = describe.describe_all(self.paths, reader=StandIn(reply=""))
        self.assertEqual((res["described"], res["skipped"]), (0, 1))
        self.assertEqual(md.read_text(encoding="utf-8"), before)

    def test_limit_and_collection(self):
        self.photo_md()
        self.assertEqual(describe.describe_all(self.paths, reader=StandIn(), collection="nowhere")["candidates"], 0)
        self.assertEqual(describe.describe_all(self.paths, reader=StandIn(), collection="c", limit=1)["described"], 1)


class RouteTests(IndexedPhotoTests):
    def test_dashboard_routes(self):
        dash = Dash(self.paths)
        try:
            status, js, _, _ = dash.req("GET", "/api/photos")
            self.assertEqual(status, 200)
            self.assertEqual((js["photographs"], js["pending"]), (0, 0))
            self.assertEqual([c["id"] for c in js["choices"]], list(models.DESCRIBE_MODELS))
            status, js, _, _ = dash.req("POST", "/api/photos/model", {"model": "someone/else"})
            self.assertEqual(status, 400)
            status, js, _, _ = dash.req("POST", "/api/photos/model", {"model": models.DESCRIBE_MODELS[1]})
            self.assertEqual(status, 200)
            self.assertEqual(models.describe_selection()[0], models.DESCRIBE_MODELS[1])
            status, js, _, _ = dash.req("POST", "/api/photos/describe", {"limit": "many"})
            self.assertEqual(status, 400)
        finally:
            dash.close()

    def test_read_only_dashboard_cannot_describe(self):
        dash = Dash(self.paths, read_only=True)
        try:
            status, _, _, _ = dash.req("POST", "/api/photos/describe", {})
            self.assertEqual(status, 403)
        finally:
            dash.close()
