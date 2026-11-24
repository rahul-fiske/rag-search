"""The Files tab's backend: listing a source folder lazily, the per-file dossier and the original, served read-only."""

from __future__ import annotations

import json
import unittest

from tests.helpers import TempHome
from tests.portable.test_ui import Dash

from rag_search import locations
from rag_search.core import dossier


def jpeg(path):
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (60, 40), (200, 30, 30)).save(path, "JPEG")


class DossierCase(TempHome):
    def setUp(self):
        super().setUp()
        self.root = self.sdir / "pics"
        jpeg(self.root / "trip" / "a.jpg")
        jpeg(self.root / "trip" / "b.jpg")
        (self.root / "notes.txt").write_text("hello", encoding="utf-8")
        locations.add(self.paths, "pics", str(self.root))
        idx = self.paths.index / "pics" / "trip" / "a"
        idx.mkdir(parents=True)
        (idx / "index.meta.json").write_text(json.dumps({
            "src_sha256": "ab" * 32, "src_bytes": (self.root / "trip" / "a.jpg").stat().st_size, "model": "m", "dim": 4, "nodes": 1,
            "built_at": "2026-11-20T10:00:00Z", "conversion": {"branches": {"photo": 1}, "pages": 1, "readers": ["photo"]}}), encoding="utf-8")
        md = self.paths.markup / "pics" / "trip" / "a.md"
        md.parent.mkdir(parents=True)
        md.write_text("<!-- page 1 -->\n\n## Photo details\n\n- Taken: 2018-12-27 09:32:50\n"
                      "- Position: 17.75 N, 73.17 E (geo:17.75,73.17)\n- Camera: X\n", encoding="utf-8")


class ListingTests(DossierCase):
    def test_a_folder_is_listed_one_level_at_a_time_with_what_is_indexed(self):
        top = dossier.listing(self.paths, "pics")
        self.assertEqual([f["name"] for f in top["folders"]], ["trip"])
        self.assertEqual([f["name"] for f in top["files"]], ["notes.txt"])
        inner = {f["name"]: f for f in dossier.listing(self.paths, "pics", "trip")["files"]}
        self.assertEqual((inner["a.jpg"]["state"], inner["a.jpg"]["kind"]), ("indexed", "photo"))
        self.assertEqual(inner["b.jpg"]["state"], "not indexed")

    def test_a_search_walks_the_collection_and_is_capped(self):
        got = dossier.listing(self.paths, "pics", q="B.JP")
        self.assertEqual([f["path"] for f in got["files"]], ["trip/b.jpg"])
        self.assertTrue(dossier.listing(self.paths, "pics", q="jpg", limit=1)["more"])

    def test_nothing_outside_the_source_folder_can_be_named(self):
        for bad in ("../x", "trip/../../x"):
            with self.assertRaises(dossier.FileError):
                dossier.listing(self.paths, "pics", bad)
        with self.assertRaises(dossier.FileError):
            dossier.listing(self.paths, "nope")


class DossierTests(DossierCase):
    def test_the_dossier_puts_the_records_of_each_step_side_by_side(self):
        d = dossier.dossier(self.paths, "pics", "trip/a.jpg")
        self.assertEqual((d["kind"], d["indexed"], d["sha"][:2]), ("photo", True, "ab"))
        self.assertEqual(d["geo"], {"lat": 17.75, "lon": 73.17})
        self.assertIn("Camera", [f["label"] for f in d["facts"]])
        steps = {r["step"]: r for r in d["ledger"]}
        self.assertEqual(steps["convert"]["state"], "done")
        self.assertEqual(steps["locate"]["state"], "missing")          # a position, but no stamp of the places
        self.assertEqual(steps["faces"]["state"] in ("missing", "n/a"), True)
        self.assertEqual(steps["describe"]["state"], "missing")
        self.assertEqual(steps["embed"]["by"], "m")
        self.assertEqual({r["name"] for r in d["records"] if r["group"] == "index folder"}, {"index.meta.json"})
        self.assertIn("Photo details", d["markdown"])

    def test_a_file_that_was_never_indexed_says_so(self):
        d = dossier.dossier(self.paths, "pics", "trip/b.jpg")
        self.assertFalse(d["indexed"])
        self.assertEqual(d["ledger"][0]["state"], "missing")


class EndpointTests(DossierCase):
    def setUp(self):
        super().setUp()
        self.dash = Dash(self.paths)
        self.addCleanup(self.dash.close)

    def test_list_dossier_and_original_are_served_and_stay_inside_the_folder(self):
        st, js, _p, _h = self.dash.req("GET", "/api/files/list?collection=pics&prefix=trip")
        self.assertEqual((st, len(js["files"])), (200, 2))
        st, js, _p, _h = self.dash.req("GET", "/api/files/dossier?collection=pics&path=trip/a.jpg")
        self.assertEqual((st, js["kind"]), (200, "photo"))
        st, _j, body, hdrs = self.dash.req("GET", "/api/files/original?collection=pics&path=trip/a.jpg&w=256")
        self.assertEqual((st, body[:2]), (200, b"\xff\xd8"))
        st, _j, _p, _h = self.dash.req("GET", "/api/files/original?collection=pics&path=../../etc/passwd")
        self.assertEqual(st, 404)
        st, _j, _p, _h = self.dash.req("GET", "/api/files/raw?collection=pics&path=trip/a.jpg")
        self.assertEqual(st, 404)                                       # only videos are sent as they are
        st, _j, _p, _h = self.dash.req("GET", "/api/files/dossier?collection=pics&path=trip/a.jpg", auth=False)
        self.assertEqual(st, 401)

    def test_a_video_is_sent_in_ranges(self):
        (self.root / "v.mov").write_bytes(bytes(range(256)) * 4)
        st, _j, body, hdrs = self.dash.req("GET", "/api/files/raw?collection=pics&path=v.mov", headers={"Range": "bytes=10-19"})
        self.assertEqual((st, len(body), hdrs.get("Content-Range")), (206, 10, "bytes 10-19/1024"))


if __name__ == "__main__":
    unittest.main()
