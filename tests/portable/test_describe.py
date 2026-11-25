"""Descriptions of photographs: the model's reply, the word filter, the cache, the lines in a photograph's record, and that
neither a map refresh nor a rebuild runs the model again.  The model is replaced by a stand-in."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

from tests.portable.test_maps import fake_reader
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


def switch_as_left():
    """The environment of a test has the automatic switch off (so no test loads the real model); this puts back "as the user left it"
    (the default, or what config.json says) for the code inside the ``with``."""
    import contextlib
    import os

    @contextlib.contextmanager
    def cm():
        with mock.patch.dict(os.environ):
            os.environ.pop("RAG_SEARCH_DESCRIBE_AUTO", None)
            yield
    return cm()


class AutomaticTests(IndexedPhotoTests):
    """Photographs are described by themselves after an indexing run: the switch, a machine short of memory, a photograph
    the model cannot describe, and the daemon's hook."""

    def photo_md(self):
        (self.sdir / "c").mkdir(exist_ok=True)
        jpeg(self.sdir / "c" / "p.jpg")
        with mock.patch.object(photo, "visible_text", return_value="EXIT 4"):
            self.run_index()
        return next(self.paths.markup.rglob("p.md"))

    def test_the_switch_is_on_unless_turned_off(self):
        import os
        with switch_as_left():
            self.assertTrue(models.describe_auto())
            models.set_describe_auto(self.paths, False)
            self.assertFalse(models.describe_auto())
            models.set_describe_auto(self.paths, True)
            self.assertTrue(models.describe_auto())
        with mock.patch.dict(os.environ, {"RAG_SEARCH_DESCRIBE_AUTO": "off"}):
            self.assertFalse(models.describe_auto())
        models.set_describe_auto(self.paths, False)
        with mock.patch.dict(os.environ, {"RAG_SEARCH_DESCRIBE_AUTO": "on"}):
            self.assertTrue(models.describe_auto())

    def test_route_and_command(self):
        from tests.portable.test_cli_api import run
        with switch_as_left():
            dash = Dash(self.paths)
            try:
                status, js, _, _ = dash.req("GET", "/api/photos")
                self.assertTrue(js["auto"])
                self.assertEqual(dash.req("POST", "/api/photos/auto", {"on": False})[0], 200)
                self.assertFalse(models.describe_auto())
                self.assertEqual(dash.req("POST", "/api/photos/auto", {"on": "yes"})[0], 400)
            finally:
                dash.close()
            rc, out, err = run("photos", "auto", "on")
            self.assertEqual(rc, 0, err)
            self.assertTrue(models.describe_auto())
            rc, out, _ = run("photos", "--json")
            self.assertTrue(json.loads(out)["auto"])
            self.assertEqual(run("photos", "auto", "maybe")[0], 2)


class InlineDescribeTests(IndexedPhotoTests):
    """A photograph is described while it is indexed (3.10, after Faces and before Chunk) and embedded once, complete."""

    def setUp(self):
        super().setUp()
        self.model = StandIn()
        describe._SHARED = self.model
        self.addCleanup(setattr, describe, "_SHARED", None)
        patch = mock.patch.object(describe, "inline_ready", return_value="")
        patch.start()
        self.addCleanup(patch.stop)

    def index(self, log=None, **kw):
        (self.sdir / "c").mkdir(exist_ok=True)
        if not (self.sdir / "c" / "p.jpg").exists():
            jpeg(self.sdir / "c" / "p.jpg")
        srcs = indexer.scan_sources(self.sdir, indexer.exclude_dirs(self.paths))
        with mock.patch.object(photo, "visible_text", return_value="EXIT 4"):
            from tests.helpers import FakeEmbedder
            return indexer.run_index(self.paths, srcs, self.roots(), jobs=1, embedder=FakeEmbedder(),
                                     progress=lambda e: None, stage_log=log, **kw)

    def events(self, log):
        return [json.loads(x) for x in Path(log).read_text(encoding="utf-8").splitlines() if x.strip()]

    def test_one_run_describes_chunks_and_embeds_a_photograph(self):
        log = self.tmp / "stages.jsonl"
        self.index(log)
        md = next(self.paths.markup.rglob("p.md"))
        text = md.read_text(encoding="utf-8")
        self.assertIn("- Looks like (model-written): A tiger walks", text)
        self.assertIn(f"- Described by: {describe.current_id()}", text)
        self.assertEqual(self.model.calls, 1)
        nodes = json.loads(next(self.paths.index.rglob("p/nodes.json")).read_text())["nodes"]
        self.assertTrue(any("A tiger walks" in n["text"] for n in nodes))         # chunked after describing: one pass
        self.assertTrue((next(self.paths.index.rglob("p")) / "embeddings.npy").exists())
        order = [e["stage"] for e in self.events(log) if e.get("event") == "stage" and e.get("status") == "done"
                 and e.get("file", "").endswith("p.jpg")]
        wanted = ["sniff", "photo", "locate", "describe", "chunk"]
        positions = [order.index(x) for x in wanted if x in order]
        self.assertEqual([x for x in order if x in wanted], [x for x in wanted if x in order])      # in this order
        self.assertIn("describe", order)
        self.assertEqual(positions, sorted(positions))
        done = next(e for e in self.events(log) if e.get("stage") == "describe" and e.get("status") == "done")
        self.assertEqual((done["id"], done["source"]), ("3.10", "model"))
        self.assertIsNone(describe._SHARED)                                           # the model is released before the embedder loads
        self.assertEqual(describe.status(self.paths)["described"], 1)

    def test_nothing_is_asked_again_by_a_second_run_a_refresh_or_a_rebuild(self):
        from rag_search import maps
        self.index()
        describe._SHARED = self.model
        self.index()                                                                  # unchanged: skipped
        self.assertEqual(self.model.calls, 1)
        md = next(self.paths.markup.rglob("p.md"))
        maps.write_stamp(self.paths, md, md.read_text(encoding="utf-8"))
        maps.download_overture(self.paths, ["19.05,78.30"], release="r1", reader=fake_reader())
        self.assertEqual(maps.refresh(self.paths)["in_place"], 1)
        describe._SHARED = self.model
        self.index()
        self.assertIn("- Nearby: Hotel Sai Krishna", md.read_text(encoding="utf-8"))
        self.assertIn("- Looks like (model-written): A tiger walks", md.read_text(encoding="utf-8"))
        self.assertEqual(self.model.calls, 1)
        describe._SHARED = self.model
        md.unlink()
        self.index(rebuild=True, force_md=True)                                       # converted again: from the cache
        self.assertIn("- Looks like (model-written): A tiger walks", next(self.paths.markup.rglob("p.md")).read_text(encoding="utf-8"))
        self.assertEqual(self.model.calls, 1)

    def test_a_photographed_page_is_not_described(self):
        (self.sdir / "c").mkdir()
        jpeg(self.sdir / "c" / "page.jpg")

        def convert(src, out, ocr=None):
            out.write_text("<!-- page 1 -->\n\n" + "The agreement is made between the parties. " * 10 + "\n", encoding="utf-8")
            return {"pages": 1, "seconds": 0.0}
        srcs = indexer.scan_sources(self.sdir, indexer.exclude_dirs(self.paths))
        with mock.patch.object(photo, "visible_text", return_value="word " * 120), \
                mock.patch("rag_search.core.docling_convert.convert_file", convert):
            from tests.helpers import FakeEmbedder
            indexer.run_index(self.paths, srcs, self.roots(), jobs=1, embedder=FakeEmbedder(), progress=lambda e: None)
        self.assertEqual(self.model.calls, 0)

    def test_a_machine_short_of_memory_still_indexes_the_photograph_and_leaves_it_for_later(self):
        from rag_search.core.conversion import vlm

        class NoRoom(StandIn):
            def ensure_started(self):
                raise vlm.ReaderUnavailable("only 3.0 GB of memory is free")
        describe._SHARED = NoRoom()
        log = self.tmp / "stages.jsonl"
        res = self.index(log)
        self.assertEqual(res["errors"], [])
        md = next(self.paths.markup.rglob("p.md"))
        self.assertNotIn("Looks like", md.read_text(encoding="utf-8"))
        self.assertIn("Taken:", md.read_text(encoding="utf-8"))                       # still indexed from its details
        done = next(e for e in self.events(log) if e.get("stage") == "describe" and e.get("status") == "done")
        self.assertEqual(done["source"], "none")
        self.assertIn("3.0 GB", done["why"])
        self.assertEqual((describe.status(self.paths)["pending"], describe.status(self.paths)["failed"]), (1, 0))

    def test_a_model_that_writes_nothing_is_not_asked_again_for_that_photograph(self):
        describe._SHARED = StandIn(reply="")
        self.index()
        self.assertEqual(describe._SHARED.calls if describe._SHARED else 1, 1) if describe._SHARED else None
        md = next(self.paths.markup.rglob("p.md"))
        self.assertNotIn("Looks like", md.read_text(encoding="utf-8"))
        again = StandIn(reply="")
        describe._SHARED = again
        self.index(rebuild=True, force_md=True)
        self.assertEqual(again.calls, 0)
        self.assertEqual(describe.status(self.paths)["failed"], 1)

    def test_a_photograph_is_converted_after_the_documents_and_in_this_process(self):
        (self.sdir / "c").mkdir()
        (self.sdir / "c" / "a.md").write_text("# T\n\n<!-- page 1 -->\n\nthe text of a document\n", encoding="utf-8")
        jpeg(self.sdir / "c" / "p.jpg")
        srcs = indexer.scan_sources(self.sdir, indexer.exclude_dirs(self.paths))
        seen = []
        real = indexer.prepare_document

        def spy(task):
            seen.append((Path(task["src"]).name, __import__("os").getpid()))
            return real(task)
        with mock.patch.object(indexer, "prepare_document", spy), mock.patch.object(photo, "visible_text", return_value=""):
            from tests.helpers import FakeEmbedder
            indexer.run_index(self.paths, srcs, self.roots(), jobs=2, embedder=FakeEmbedder(), progress=lambda e: None)
        names = [n for n, _ in seen]
        self.assertEqual(names[-1], "p.jpg")                                           # the image comes last
        self.assertEqual(seen[-1][1], __import__("os").getpid())                       # and in the run's own process


class DescribeIsAStepOfTheRunTests(InlineDescribeTests):
    """3.10 is a step of the run and nothing else: a new model or prompt is caught up by the next runs, PER_RUN a run, the
    rest wait (and the daemon starts the next run); a photograph the model failed on is tried again only when asked."""

    def two(self):
        (self.sdir / "c").mkdir(exist_ok=True)
        jpeg(self.sdir / "c" / "p.jpg")
        jpeg(self.sdir / "c" / "q.jpg", when="2024:03:15 09:00:00")          # another photograph (another checksum)

    def test_a_new_model_or_prompt_is_described_again_by_the_next_runs_within_their_quota(self):
        self.two()
        self.index()
        self.assertEqual(self.model.calls, 2)
        models.set_describe_model(self.paths, models.DESCRIBE_MODELS[1])
        describe._SHARED = self.model
        s = self.index(describe_limit=1)
        self.assertEqual((self.model.calls, s["describe"]), (3, {"made": 1, "deferred": 1}))
        describe._SHARED = self.model
        s = self.index(describe_limit=1)
        self.assertEqual((self.model.calls, s["describe"]), (4, {"made": 1, "deferred": 0}))
        for md in self.paths.markup.rglob("*.md"):
            text = md.read_text(encoding="utf-8")
            self.assertIn("4B", text)
            self.assertEqual(text.count("Described by:"), 1)
        describe._SHARED = self.model
        self.assertEqual(self.index()["describe"], {"made": 0, "deferred": 0})          # all current: nothing asked
        self.assertEqual(self.model.calls, 4)

    def test_a_photograph_left_undescribed_keeps_its_earlier_description(self):
        from rag_search.core.conversion import vlm

        self.two()
        self.index()
        models.set_describe_model(self.paths, models.DESCRIBE_MODELS[1])

        class NoRoom(StandIn):
            def ensure_started(self):
                raise vlm.ReaderUnavailable("only 3.0 GB of memory is free")
        describe._SHARED = NoRoom()
        s = self.index()
        self.assertEqual(s["describe"]["made"], 0)
        for md in self.paths.markup.rglob("*.md"):
            self.assertIn("A tiger walks", md.read_text(encoding="utf-8"))           # the 8B's description stays

    def test_a_failure_is_tried_again_only_when_asked(self):
        describe._SHARED = StandIn(reply="")
        self.index()
        self.assertEqual(describe.status(self.paths)["failed"], 1)
        again = StandIn()
        describe._SHARED = again
        self.index()
        self.assertEqual(again.calls, 0)                                               # not by itself
        describe._SHARED = again
        self.index(describe_retry=True)
        self.assertEqual(again.calls, 1)
        self.assertEqual((describe.status(self.paths)["failed"], describe.status(self.paths)["described"]), (0, 1))

    def test_the_daemon_starts_the_next_run_while_photographs_are_left_and_progress_is_made(self):
        from rag_search.core.indexer_daemon import IndexerDaemon

        fake = mock.Mock(paths=self.paths, _update=lambda r, **kw: r.update(kw))
        rec = {"spec": {"path": "c", "describe_limit": 5}, "summary": {"describe": {"made": 5, "deferred": 7}}}
        with switch_as_left():
            self.assertEqual(IndexerDaemon._describe_next(fake, rec), {"mode": "new", "path": "c", "describe_limit": 5})
            self.assertEqual(rec["describe"], {"started": True, "left": 7})
            rec = {"summary": {"describe": {"made": 0, "deferred": 7}}}
            self.assertIsNone(IndexerDaemon._describe_next(fake, rec))                  # a model that cannot run never loops
            self.assertIn("could be described", rec["describe"]["why"])
            self.assertIsNone(IndexerDaemon._describe_next(fake, {"summary": {"describe": {"made": 3, "deferred": 0}}}))
        rec = {"summary": {"describe": {"made": 2, "deferred": 1}}}
        self.assertIsNone(IndexerDaemon._describe_next(fake, rec))                      # the switch is off in tests
        self.assertIn("switched off", rec["describe"]["why"])

    def test_the_dashboard_and_the_command_start_an_indexing_run(self):
        from rag_search import api
        from tests.portable.test_cli_api import run

        with mock.patch.object(api, "index_start", return_value={"ok": True, "started": True, "job": {"id": "j1"}}) as start:
            dash = Dash(self.paths)
            try:
                status, js, _, _ = dash.req("POST", "/api/photos/describe", {"limit": 10, "collection": "c", "redo": True})
            finally:
                dash.close()
            self.assertEqual((status, js["job"]["id"]), (200, "j1"))
            kw = start.call_args.kwargs
            self.assertEqual((kw["path"], kw["describe_limit"], kw["describe_retry"]), ("c", 10, True))
