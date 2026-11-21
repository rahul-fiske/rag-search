"""Every kind of file the same way: 3.0 Sniff decides the kind before a file is scheduled, a photograph takes its own way
to a result (never the failure path), and the run counts documents by kind -- photographs are not pages."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from unittest import mock

from tests.helpers import FakeEmbedder
from tests.portable.test_photo import IndexedPhotoTests, jpeg

from rag_search import jobs
from rag_search.core import describe, indexer, photo
from rag_search.core.conversion import runview, trace


class ByKindTests(IndexedPhotoTests):
    def task(self, src, **kw):
        return {"src": str(src), "roots": self.roots().to_dict(), "markup_root": str(self.paths.markup),
                "index_root": str(self.paths.index), "chunk_size": 1000, "chunk_overlap": 100, "rebuild": False,
                "force_md": False, "model": "fake", "ocr": None, "stage_log": None, **kw}

    def test_photographs_and_pictures_of_pages_go_to_the_model_process_and_the_rest_to_the_pool(self):
        tasks = [{"src": "/x/a.jpg", "route": "photograph"}, {"src": "/x/b.jpg", "route": "pages"},
                 {"src": "/x/c.pdf", "route": "pages"}, {"src": "/x/d.json", "route": "timeline"},
                 {"src": "/x/e.jpg", "route": ""}, {"src": "/x/f.jpg"}]
        pool, main = indexer.by_kind(tasks)
        self.assertEqual([t["src"] for t in main], ["/x/a.jpg", "/x/b.jpg", "/x/f.jpg"])     # unsniffed: by its format
        self.assertEqual([t["src"] for t in pool], ["/x/c.pdf", "/x/d.json", "/x/e.jpg"])    # unchanged: a quick skip

    def test_documents_convert_while_the_images_are_being_sniffed(self):
        (self.sdir / "c").mkdir()
        jpeg(self.sdir / "c" / "p.jpg")
        (self.sdir / "c" / "a.md").write_text("# T\n\n<!-- page 1 -->\n\nsome text\n", encoding="utf-8")
        (self.sdir / "c" / "b.md").write_text("# T\n\n<!-- page 1 -->\n\nmore text\n", encoding="utf-8")
        order = []
        real_prepare, real_sniff = indexer.prepare_document, indexer._sniff_all

        def sniff(tasks, workers, progress, on_sorted=None):
            order.append(("sniff", sorted(Path(t["src"]).suffix for t in tasks)))
            for t in tasks:                                   # (no real pool in the test)
                t.update(route="photograph", sniffed=True, seen="", fill=False)
                if on_sorted:
                    on_sorted(t)
        with mock.patch.object(indexer, "_sniff_all", sniff), mock.patch.object(photo, "visible_text", return_value=""):
            srcs = indexer.scan_sources(self.sdir, indexer.exclude_dirs(self.paths))
            s = indexer.run_index(self.paths, srcs, self.roots(), jobs=2, embedder=FakeEmbedder(), progress=lambda e: None)
        self.assertEqual(order, [("sniff", [".jpg"])])         # only the image is sniffed
        self.assertEqual((s["indexed"], s["errors"]), (3, []))

    def test_a_photograph_is_converted_as_soon_as_it_is_sorted_not_when_every_file_is(self):
        (self.sdir / "c").mkdir()
        jpeg(self.sdir / "c" / "a.jpg")
        jpeg(self.sdir / "c" / "b.jpg", when="2024:03:15 09:00:00")
        (self.sdir / "c" / "d.md").write_text("# T\n\n<!-- page 1 -->\n\nsome text\n", encoding="utf-8")
        first_done = threading.Event()
        real_prepare = indexer.prepare_document
        seen_order = []

        def prepare(task):
            res = real_prepare(task)
            if Path(task["src"]).suffix == ".jpg":
                seen_order.append(Path(task["src"]).name)
                first_done.set()
            return res

        def sniff(tasks, workers, progress, on_sorted=None):
            ts = sorted(tasks, key=lambda t: t["src"])
            for k, t in enumerate(ts):
                t.update(route="photograph", sniffed=True, seen="", fill=False)
                on_sorted(t)
                if k == 0:                                   # the second file is not sorted until the first is converted
                    self.assertTrue(first_done.wait(30), "the first photograph was not converted while sniffing went on")
        with mock.patch.object(indexer, "_sniff_all", sniff), mock.patch.object(indexer, "prepare_document", prepare), \
                mock.patch.object(photo, "visible_text", return_value=""):
            srcs = indexer.scan_sources(self.sdir, indexer.exclude_dirs(self.paths))
            s = indexer.run_index(self.paths, srcs, self.roots(), jobs=2, embedder=FakeEmbedder(), progress=lambda e: None)
        self.assertEqual((s["indexed"], s["errors"]), (3, []))
        self.assertEqual(seen_order, ["a.jpg", "b.jpg"])

    def test_the_text_check_of_a_picture_is_remembered_by_its_checksum(self):
        (self.sdir / "c").mkdir()
        src = jpeg(self.sdir / "c" / "p.jpg")
        with mock.patch.object(photo, "visible_text", return_value="EXIT 4") as once:
            first = indexer.sniff_document(self.task(src))
        self.assertEqual((first["route"], once.call_count), ("photograph", 1))
        with mock.patch.object(photo, "visible_text", side_effect=AssertionError("looked again")):
            again = indexer.sniff_document(self.task(src))
        self.assertEqual((again["route"], again["seen"]), ("photograph", "EXIT 4"))
        self.assertTrue((self.paths.workspace / "sniff.sqlite").exists())

    def test_the_sniff_ahead_of_the_run_finds_a_photograph_and_its_conversion_does_not_look_again(self):
        (self.sdir / "c").mkdir()
        src = jpeg(self.sdir / "c" / "p.jpg")
        with mock.patch.object(photo, "visible_text", return_value="EXIT 4"):
            got = indexer.sniff_document(self.task(src))
        self.assertEqual((got["route"], got["seen"], got["fill"]), ("photograph", "EXIT 4", False))
        with mock.patch.object(photo, "visible_text", side_effect=AssertionError("sniffed already")), \
                mock.patch.object(indexer, "NoTextError", side_effect=AssertionError("not the failure path")):
            res = indexer.prepare_document(self.task(src, route="photograph", seen="EXIT 4"))
        self.assertEqual((res["status"], res["markdown"]), ("prepared", "photo details"))
        self.assertEqual((res["understood"]["kind"], res["understood"]["track"]), ("photograph", "photograph"))
        self.assertIn("Text in the picture: EXIT 4", next(self.paths.markup.rglob("p.md")).read_text(encoding="utf-8"))

    def test_a_photograph_with_no_details_at_all_is_a_genuine_no_text(self):
        (self.sdir / "c").mkdir()
        src = jpeg(self.sdir / "c" / "p.jpg")
        with mock.patch.object(photo, "block", return_value=""), \
                mock.patch.object(describe, "for_indexing", return_value=([], {"described": "", "why": "", "seconds": 0})):
            res = indexer.prepare_document(self.task(src, route="photograph", seen=""))
        self.assertEqual(res["status"], "no_text")
        self.assertIn("no details", res["message"])
        self.assertEqual(res["understood"]["grade"], "no_text")

    def test_a_photograph_is_a_document_of_the_run_not_a_page(self):
        (self.sdir / "c").mkdir()
        jpeg(self.sdir / "c" / "p.jpg")
        with mock.patch.object(photo, "visible_text", return_value=""):
            s = self.run_index()
        self.assertEqual((s["indexed"], s["conversion"]["pages"], s["conversion"]["whole"]), (1, 0, {"photo": 1}))

    def test_a_missing_description_is_filled_in_by_a_later_run(self):
        (self.sdir / "c").mkdir()
        src = jpeg(self.sdir / "c" / "p.jpg")
        with mock.patch.object(photo, "visible_text", return_value=""):
            self.run_index()                                  # no model here: indexed without a description
        md = next(self.paths.markup.rglob("p.md"))
        sha = indexer.sha256_file(src)
        self.assertFalse(indexer._fills_description(self.task(src), "image", md, sha))          # cannot describe now
        self.assertTrue(indexer._fills_description(self.task(src, describe_ready=True), "image", md, sha))
        line = f"- {describe.LOOKS} a road\n"
        with mock.patch.object(describe, "inline_ready", return_value=""), \
                mock.patch.object(describe, "for_indexing", return_value=([line[2:].strip()], {"described": "model",
                                                                                                 "why": "", "seconds": 1.0})), \
                mock.patch.object(photo, "visible_text", return_value=""):
            s = self.run_index()
        self.assertEqual((s["indexed"], s.get("skipped_fresh", 0)), (1, 0))
        self.assertIn(describe.LOOKS, md.read_text(encoding="utf-8"))
        self.assertFalse(indexer._fills_description(self.task(src, describe_ready=True), "image", md, sha))


class TotalsByKindTests(IndexedPhotoTests):
    def test_photographs_and_timelines_are_counted_apart_from_pages(self):
        t = trace.RunTotals()
        t.add({"pages": 1, "branches": {"photo": 1}, "outcomes": {"pass": 1}, "time_s": {"convert": 2.0}})
        t.add({"pages": 3, "branches": {"digital": 3}, "outcomes": {"pass": 3}})
        snap = t.snapshot()
        self.assertEqual((snap["pages"], snap["branches"], snap["outcomes"], snap["whole"], snap["docs"]),
                         (3, {"digital": 3}, {"pass": 3}, {"photo": 1}, 1))
        self.assertEqual(snap["time_s"]["convert"], 2.0)

    def test_the_tracks_count_documents_by_kind_with_what_is_left_and_a_rate(self):
        st = runview._fresh()
        feed = lambda **kw: runview._feed(st, json.dumps(kw))
        feed(event="phase", phase="convert", status="start", ts=0, pid=1, total=10)
        feed(event="phase", phase="convert", status="kinds", ts=1, pid=1,
             kinds={"photograph": 4, "pages": 2, "unchanged": 4})
        for i in range(3):
            feed(event="work", phase="convert", status="start", ts=10 + 10 * i, pid=1, file=f"c/{i}.jpg", route="photograph")
            feed(event="work", phase="convert", status="done", ts=20 + 10 * i, pid=1, file=f"c/{i}.jpg", outcome="prepared",
                 understood={"kind": "photograph", "track": "photograph", "grade": "partial" if i else "complete"})
        feed(event="work", phase="convert", status="start", ts=40, pid=2, file="c/x.pdf", route="pages")
        feed(event="work", phase="convert", status="start", ts=40, pid=3, file="c/old.pdf")
        feed(event="work", phase="convert", status="done", ts=41, pid=3, file="c/old.pdf", outcome="skipped")
        v = runview.tracks_view(st, running=True, now=70)
        p = v["photograph"]
        self.assertEqual((p["planned"], p["done"], p["running"], p["left"], p["grades"]),
                         (4, 3, 0, 1, {"complete": 1, "partial": 2}))
        self.assertEqual(p["per_min"], 3.0)                 # 3 in the minute since the first began
        self.assertEqual(p["eta_s"], 20)
        self.assertEqual((v["pages"]["running"], v["pages"]["left"]), (1, 1))
        self.assertEqual(v["unchanged"], {"planned": 4, "done": 1})
        self.assertNotIn("timeline", v)
        self.assertEqual(runview._active(st)["files"], 1)  # the photographs are not files of pages


class DocumentsByKindTests(IndexedPhotoTests):
    def test_the_documents_list_filters_by_the_kind_and_grade_of_the_understood_table(self):
        paths = self.paths
        paths.jobs.mkdir(parents=True, exist_ok=True)
        jid = "20260101-000000-abcf"
        ev = lambda n, p, st, **kw: json.dumps({"ts": n, "event": "doc", "collection": "c", "source": p, "path": p,
                                                "status": st, **kw})
        lines = [ev(1, "a.jpg", "converted", understood={"kind": "photograph", "grade": "partial", "missing": ["position"]}),
                 ev(2, "b.jpg", "converted", understood={"kind": "photograph", "grade": "complete", "missing": []}),
                 ev(3, "c.pdf", "converted", understood={"kind": "PDF with a text layer", "grade": "complete"}),
                 ev(4, "a.jpg", "indexed"),                    # embedding done: the kind and grade stay
                 ev(5, "d.pdf", "error", message="x"), ev(6, "e.md", "skipped")]
        jobs.events_file(paths, jid).write_text("\n".join(lines) + "\n")
        d = jobs.documents(paths, jid, limit=10)
        self.assertEqual(d["by_kind"], {"photograph": 2, "PDF with a text layer": 1, "other document": 1})
        self.assertEqual(d["by_grade"], {"partial": 1, "complete": 2, "error": 1})
        got = jobs.documents(paths, jid, limit=10, kind="photograph", grade="partial")
        self.assertEqual([(i["path"], i["status"]) for i in got["items"]], [("a.jpg", "indexed")])


if __name__ == "__main__":
    import unittest

    unittest.main()
