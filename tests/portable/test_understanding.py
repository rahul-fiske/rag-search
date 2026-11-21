"""What indexing understood of each document (core/understanding.py), the totals per kind in the run view, and the
text check and face count read ahead for the images of a run (photo.read_ahead)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest import mock

from rag_search.core import photo, understanding
from rag_search.core.conversion import runview

FULL = ("- Taken: 2024-03-14 12:52\n- Position: Nirmal (geo:19.09,78.34)\n- Area: Nirmal, Telangana\n"
        "- Nearby: Hotel (restaurant, 6 m)\n- Faces detected: 2\n- Looks like (model-written): a lunch table\n")


class OfDocumentTests(unittest.TestCase):
    def test_a_photograph_with_everything_is_complete(self):
        u = understanding.of_document(fmt="image", suffix=".jpg", status="prepared", text=FULL, route="photograph")
        self.assertEqual((u["kind"], u["grade"], u["missing"]), ("photograph", "complete", []))
        self.assertEqual(u["found"], ["time", "position", "place", "nearby", "faces", "description"])

    def test_a_photograph_without_a_position_misses_only_the_position_and_what_it_lacks_besides(self):
        u = understanding.of_document(fmt="image", suffix=".jpg", status="prepared", text="- Taken: 2003-01-01\n",
                                      route="photograph")
        self.assertEqual((u["grade"], u["missing"]), ("partial", ["position", "description"]))

    def test_a_face_count_of_nought_is_not_a_face_found(self):
        self.assertNotIn("faces", understanding.photo_facts("- Faces detected: 0\n"))
        self.assertIn("faces", understanding.photo_facts("- Faces detected: 12\n"))

    def test_a_position_without_map_data_misses_the_place(self):
        u = understanding.of_document(fmt="image", suffix=".jpg", status="prepared", route="photograph",
                                      text="- Taken: x\n- Position: geo:1,2\n- Looks like (model-written): a road\n")
        self.assertEqual(u["missing"], ["place"])

    def test_pdfs_are_named_by_their_pages_and_weak_pages_make_them_partial(self):
        of = lambda b, o: understanding.of_document(fmt="pdf", suffix=".pdf", status="prepared", route="pages",
                                                    text="x", branches=b, outcomes=o)
        self.assertEqual(of({"digital": 3}, {"pass": 3})["kind"], "PDF with a text layer")
        self.assertEqual(of({"raster": 2}, {"pass": 2})["kind"], "PDF, scanned")
        u = of({"digital": 3, "raster": 1}, {"pass": 3, "low": 1})
        self.assertEqual((u["kind"], u["grade"], u["missing"], u["weak_pages"]), ("PDF, partly scanned", "partial", ["low pages"], 1))

    def test_a_photographed_page_and_a_timeline_and_a_failure(self):
        u = understanding.of_document(fmt="image", suffix=".jpg", status="prepared", route="pages", branches={"image": 1},
                                      outcomes={"pass": 1}, text="## Photo details\n\n- Taken: x\n")
        self.assertEqual((u["kind"], u["found"]), ("photographed page", ["text", "time"]))
        tl = understanding.of_document(fmt="text", suffix=".json", status="prepared", route="timeline",
                                       text="3 visits and 2 journeys\n  - Area: Pune\n")
        self.assertEqual((tl["kind"], tl["grade"], tl["visits"]), ("Google Maps Timeline", "complete", 3))
        bad = understanding.of_document(fmt="pdf", suffix=".pdf", status="error")
        self.assertEqual((bad["kind"], bad["grade"]), ("PDF", "error"))


class RunViewKindsTests(unittest.TestCase):
    def test_documents_are_totalled_by_kind_with_their_facts_gaps_and_examples(self):
        st = runview._fresh()
        feed = lambda i, **kw: runview._feed(st, json.dumps({"event": "work", "phase": "convert", "pid": 1, "ts": 10 + i,
                                                             "file": f"c/{i}.jpg", **kw}))
        full = understanding.of_document(fmt="image", suffix=".jpg", status="prepared", text=FULL, route="photograph")
        bare = understanding.of_document(fmt="image", suffix=".jpg", status="prepared", text="- Taken: x\n", route="photograph")
        for i, u in enumerate([full, bare, bare]):
            feed(i, status="start")
            feed(i, status="done", outcome="prepared", understood=u)
        feed(5, status="start")
        feed(5, status="done", outcome="error")                       # no record: named by its format
        feed(6, status="start")
        feed(6, status="done", outcome="error", known=True)           # not tried again: not counted
        k = st["kinds"]
        p = k["photograph"]
        self.assertEqual((p["n"], p["grades"], p["found"]["time"], p["found"].get("description")),
                         (3, {"complete": 1, "partial": 2}, 3, 1))
        self.assertEqual(p["missing"], {"position": 2, "description": 2})
        self.assertEqual(p["examples"]["position"], ["c/1.jpg", "c/2.jpg"])
        self.assertEqual(k["image"]["grades"], {"error": 1})
        self.assertEqual(sum(v["n"] for v in k.values()), 4)


class ReadAheadTests(unittest.TestCase):
    def tearDown(self):
        photo.stop_ahead()

    def test_the_text_check_and_face_count_are_done_once_and_taken_from_what_was_read_ahead(self):
        calls = []
        with mock.patch.object(photo, "read_exif", return_value={"camera": "x"}), \
                mock.patch.object(photo, "_visible_text", side_effect=lambda p: calls.append(("t", p.name)) or ""), \
                mock.patch.object(photo, "_faces_in", side_effect=lambda p: calls.append(("f", p.name)) or 2):
            photo.read_ahead([Path("a.jpg"), Path("b.jpg")])
            self.assertEqual(photo.visible_text(Path("a.jpg")), "")
            self.assertEqual(photo.faces_in(Path("a.jpg")), 2)
            self.assertEqual(photo.visible_text(Path("c.jpg")), "")          # not read ahead: read now
            photo.stop_ahead()
        self.assertEqual(sorted(c for c in calls if c[1] == "a.jpg"), [("f", "a.jpg"), ("t", "a.jpg")])
        self.assertIn(("t", "c.jpg"), calls)


if __name__ == "__main__":
    unittest.main()
