"""Photographs: camera metadata as a "Photo details" section, place names from reference data, and the registry
that fetches the lookup tables (no network: the download is replaced by a function that writes the files)."""

from __future__ import annotations

import io
import json
import os
import zipfile
from pathlib import Path
from unittest import mock

from tests.helpers import FakeEmbedder, TempHome

from rag_search import reference
from rag_search.core import geo, indexer, photo
from rag_search.core.docling_convert import NoTextError

try:
    from PIL import Image
except ImportError:                                    # pragma: no cover
    Image = None

CITIES = "\n".join([
    "1\tNirmal\tNirmal\t\t19.09\t78.34\tP\tPPL\tIN\t\t40\t\t\t\t90000",
    "2\tHyderabad\tHyderabad\t\t17.38\t78.48\tP\tPPLA\tIN\t\t40\t\t\t\t6800000",
    "3\tNagpur\tNagpur\t\t21.15\t79.09\tP\tPPLA2\tIN\t\t16\t\t\t\t2400000",
]) + "\n"
ADMIN1 = "IN.40\tTelangana\tTelangana\t1\nIN.16\tMaharashtra\tMaharashtra\t2\n"
COUNTRIES = "# comment\nIN\tIND\t356\tIN\tIndia\tNew Delhi\n"


def fake_fetch(url: str, dest: Path) -> None:
    if url.endswith("cities1000.zip"):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("cities1000.txt", CITIES)
        dest.write_bytes(buf.getvalue())
    elif url.endswith("admin1CodesASCII.txt"):
        dest.write_text(ADMIN1, encoding="utf-8")
    else:
        dest.write_text(COUNTRIES, encoding="utf-8")


def jpeg(path: Path, *, when="2024:03:14 13:12:05", offset="+05:30", gps=True, camera=True) -> Path:
    ex = Image.Exif()
    if camera:
        ex[0x010F], ex[0x0110] = "Apple", "iPhone 13 Pro"
    ifd = ex.get_ifd(0x8769)
    if when:
        ifd[0x9003] = when
    if offset:
        ifd[0x9011] = offset
    if gps:
        g = ex.get_ifd(0x8825)
        g[1], g[2], g[3], g[4], g[5], g[6] = "N", (19, 5, 26.0), "E", (78, 20, 28.0), 0, 312.0
    Image.new("RGB", (64, 48), "red").save(path, exif=ex)
    return path


class ReferenceTests(TempHome):
    def test_download_installs_and_gives_a_ref(self):
        self.assertFalse(reference.installed(self.paths, "geonames"))
        self.assertEqual(reference.ref(self.paths, "geonames"), "")
        man = reference.download(self.paths, "geonames", fetch=fake_fetch)
        self.assertTrue(reference.installed(self.paths, "geonames"))
        self.assertEqual(reference.ref(self.paths, "geonames"), f"geonames@{man['fetched']}")
        self.assertTrue(reference.path(self.paths, "geonames", "cities1000.txt").is_file())
        self.assertEqual(len(man["files"]["cities1000.txt"]["sha256"]), 64)
        self.assertTrue(reference.status(self.paths)[0]["installed"])

    def test_a_failed_download_keeps_the_installed_copy(self):
        reference.download(self.paths, "geonames", fetch=fake_fetch)
        before = reference.manifest(self.paths, "geonames")

        def broken(url, dest):
            if url.endswith("countryInfo.txt"):
                raise OSError("offline")
            fake_fetch(url, dest)
        with self.assertRaises(reference.ReferenceDataError):
            reference.download(self.paths, "geonames", fetch=broken)
        self.assertEqual(reference.manifest(self.paths, "geonames"), before)
        self.assertTrue(reference.installed(self.paths, "geonames"))
        self.assertEqual([p.name for p in reference.root(self.paths).iterdir()], ["geonames"])

    def test_unknown_dataset(self):
        with self.assertRaises(reference.ReferenceDataError):
            reference.download(self.paths, "faces", fetch=fake_fetch)


class PlacesTests(TempHome):
    def test_nothing_without_the_dataset(self):
        self.assertIsNone(geo.near(19.09, 78.34, self.paths))

    def test_nearest_place_with_state_country_and_direction(self):
        reference.download(self.paths, "geonames", fetch=fake_fetch)
        near = geo.near(19.12, 78.40, self.paths)
        self.assertEqual((near.name, near.admin1, near.country), ("Nirmal", "Telangana", "India"))
        self.assertTrue(near.text().startswith("near Nirmal, Telangana, India (about "))
        self.assertTrue(geo.near(19.0901, 78.3401, self.paths).text().startswith("at Nirmal"))
        self.assertIsNone(geo.near(10.0, 20.0, self.paths))          # nothing within 50 km
        self.assertIsNone(geo.near(95.0, 0.0, self.paths))


class PhotoTests(TempHome):
    def test_details_with_and_without_reference_data(self):
        src = jpeg(self.tmp / "a.jpg")
        text = photo.block(src, self.paths)
        for line in ("- Taken: 2024-03-14 13:12:05 +05:30 (Thursday)", "- Position: 19.09056 N, 78.34111 E, altitude 312 m",
                     "- Camera: Apple iPhone 13 Pro", "- Size: 64 x 48 pixels"):
            self.assertIn(line, text)
        self.assertNotIn("Town:", text)
        reference.download(self.paths, "geonames", fetch=fake_fetch)
        text = photo.block(src, self.paths)
        self.assertIn("- Town: at Nirmal, Telangana, India", text)
        self.assertIn("- Reference data: geonames@", text)

    def test_no_zone_no_gps_and_no_metadata(self):
        text = photo.block(jpeg(self.tmp / "b.jpg", offset="", gps=False), self.paths)
        self.assertIn("2024-03-14 13:12:05 (time zone not recorded)", text)
        self.assertNotIn("Position", text)
        plain = self.tmp / "plain.png"
        Image.new("RGB", (8, 8)).save(plain)
        self.assertEqual(photo.block(plain, self.paths), "")
        self.assertEqual(photo.block(self.tmp / "missing.jpg", self.paths), "")
        unknown = jpeg(self.tmp / "c.jpg", when="0000:00:00 00:00:00", gps=False)
        self.assertNotIn("Taken", photo.block(unknown, self.paths))

    def test_attach_replaces_the_section_and_is_repeatable(self):
        md = self.tmp / "a.md"
        md.write_text("<!-- page 1 -->\n\nText read from the picture.\n", encoding="utf-8")
        one = "## Photo details\n\n- Taken: 2024-01-01 10:00:00 +00:00 (Monday)\n"
        two = "## Photo details\n\n- Taken: 2024-02-02 10:00:00 +00:00 (Friday)\n"
        self.assertTrue(photo.attach(md, one))
        self.assertFalse(photo.attach(md, one))
        self.assertTrue(photo.attach(md, two))
        text = md.read_text(encoding="utf-8")
        self.assertEqual(text.count("## Photo details"), 1)
        self.assertIn("2024-02-02", text)
        self.assertTrue(text.endswith("Text read from the picture.\n"))
        self.assertTrue(text.startswith("<!-- page 1 -->\n\n## Photo details"))

    def test_a_failing_annotator_does_not_lose_the_photo(self):
        def boom(meta, paths):
            raise RuntimeError("x")
        with mock.patch.object(photo, "ANNOTATORS", [boom]):
            self.assertIn("Taken:", photo.block(jpeg(self.tmp / "d.jpg"), self.paths))


class IndexedPhotoTests(TempHome):
    def setUp(self):
        super().setUp()
        self.use_fake_backends()
        os.environ["RAG_SEARCH_ROUTING"] = "document"
        reference.download(self.paths, "geonames", fetch=fake_fetch)

    def run_index(self):
        srcs = indexer.scan_sources(self.sdir, indexer.exclude_dirs(self.paths))
        return indexer.run_index(self.paths, srcs, self.roots(), jobs=1, embedder=FakeEmbedder(),
                                 progress=lambda e: None)

    def test_a_photo_without_text_is_indexed_from_its_details(self):
        (self.sdir / "trip").mkdir()
        jpeg(self.sdir / "trip" / "IMG_1.jpg")
        plain = self.sdir / "trip" / "scan.png"
        Image.new("RGB", (8, 8)).save(plain)               # no metadata, no text: still nothing to index
        with mock.patch("rag_search.core.docling_convert.convert_file", side_effect=NoTextError("no text")):
            self.run_index()
        md = next(self.paths.markup.rglob("IMG_1.md"))
        text = md.read_text(encoding="utf-8")
        self.assertIn("Taken: 2024-03-14 13:12:05 +05:30", text)
        self.assertIn("Town: at Nirmal, Telangana, India", text)
        nodes = json.loads(next(self.paths.index.rglob("IMG_1/nodes.json")).read_text())["nodes"]
        self.assertTrue(any("2024-03-14" in n["text"] for n in nodes))
        # a picture with no camera data and no text is a photograph all the same, but is indexed only when something is known of it
        # (a description, faces): then its section has a heading and its size, never loose lines
        for md in self.paths.markup.rglob("scan.md"):
            self.assertIn("## Photo details\n\n- Size: 8 x 8 pixels\n", md.read_text(encoding="utf-8"))


class PhotographOrPageTests(TempHome):
    def test_text_decides(self):
        self.assertTrue(photo.is_photograph(""))
        self.assertTrue(photo.is_photograph("Hotel Sai Krishna Veg Meals"))
        self.assertFalse(photo.is_photograph("lorem ipsum " * 40))
        self.assertFalse(photo.is_photograph(None))                  # no OCR on this machine: cannot tell

    def test_a_digit_or_two_is_not_text_in_the_picture(self):
        src = jpeg(self.tmp / "n.jpg")
        for noise in ("2", "2 102 30", "16000 goo"):
            self.assertNotIn("Text in the picture", photo.block(src, self.paths, noise))
        self.assertIn("Text in the picture: RJ26TA 0769", photo.block(src, self.paths, "RJ26TA 0769"))

    def test_the_text_found_in_a_picture_survives_a_renewed_section(self):
        src = jpeg(self.tmp / "e.jpg")
        md = self.tmp / "e.md"
        md.write_text("<!-- page 1 -->\n\n", encoding="utf-8")
        photo.attach(md, photo.block(src, self.paths, "Hotel   Sai Krishna\nVeg Meals"))
        self.assertIn("- Text in the picture: Hotel Sai Krishna Veg Meals\n", md.read_text(encoding="utf-8"))
        reference.download(self.paths, "geonames", fetch=fake_fetch)
        photo.attach(md, photo.block(src, self.paths))                # places data arrived; the text is not read again
        text = md.read_text(encoding="utf-8")
        self.assertEqual(text.count("Text in the picture"), 1)
        self.assertIn("- Town: at Nirmal", text)
        self.assertTrue(text.index("Text in the picture") < text.index("Reference data"))


class IndexedPhotographTests(IndexedPhotoTests):
    def test_a_photograph_never_reaches_the_document_reader(self):
        (self.sdir / "c").mkdir()
        jpeg(self.sdir / "c" / "p.jpg")
        with mock.patch.object(photo, "visible_text", return_value="EXIT 4"), \
                mock.patch("rag_search.core.docling_convert.convert_file",
                           side_effect=AssertionError("the reader must not run")):
            self.run_index()
        text = next(self.paths.markup.rglob("p.md")).read_text(encoding="utf-8")
        self.assertIn("Text in the picture: EXIT 4", text)
        self.assertNotIn("illegible", text)

    def test_a_photographed_page_is_read_and_gets_the_details_too(self):
        (self.sdir / "c").mkdir()
        jpeg(self.sdir / "c" / "page.jpg")

        def convert(src, out, ocr=None):
            out.write_text("<!-- page 1 -->\n\n" + "The agreement is made between the parties. " * 10 + "\n", encoding="utf-8")
            return {"pages": 1, "seconds": 0.0}
        with mock.patch.object(photo, "visible_text", return_value="word " * 120), \
                mock.patch("rag_search.core.docling_convert.convert_file", convert):
            self.run_index()
        text = next(self.paths.markup.rglob("page.md")).read_text(encoding="utf-8")
        self.assertIn("## Photo details", text)
        self.assertIn("The agreement is made", text)


class PhotoRefreshTests(IndexedPhotoTests):
    def test_new_map_data_rewrites_the_section_in_place_without_reading_the_picture(self):
        from rag_search import maps
        from tests.portable.test_maps import fake_reader

        (self.sdir / "c").mkdir()
        jpeg(self.sdir / "c" / "p.jpg")
        with mock.patch.object(photo, "visible_text", return_value="EXIT 4"):
            self.run_index()
        md = next(self.paths.markup.rglob("p.md"))
        self.assertTrue(maps.stamp_path(md).exists())
        self.assertNotIn("Nearby:", md.read_text(encoding="utf-8"))
        maps.download_overture(self.paths, ["19.05,78.30"], release="r1", reader=fake_reader())
        self.assertEqual(maps.refresh(self.paths)["in_place"], 1)
        with mock.patch.object(photo, "visible_text", side_effect=AssertionError("the picture must not be read")), \
                mock.patch("rag_search.core.docling_convert.convert_file", side_effect=AssertionError("nor converted")):
            self.run_index()
        text = md.read_text(encoding="utf-8")
        self.assertIn("- Nearby: Hotel Sai Krishna (restaurant, ", text)
        self.assertIn("- Text in the picture: EXIT 4", text)
        self.assertEqual(text.count("## Photo details"), 1)
        self.assertEqual(maps.refresh(self.paths)["refreshed"], 0)          # stamped again: current


class PhotoPipelineRecordTests(IndexedPhotoTests):
    """The dashboard draws the photograph path from what a run records: stage events 3.6 and 3.7, and the page branch."""

    def events(self, log):
        return [json.loads(x) for x in Path(log).read_text(encoding="utf-8").splitlines() if x.strip()]

    def test_a_photograph_records_photo_and_locate_stages_and_a_photo_branch(self):
        (self.sdir / "c").mkdir()
        jpeg(self.sdir / "c" / "p.jpg")
        log = self.tmp / "stages.jsonl"
        srcs = indexer.scan_sources(self.sdir, indexer.exclude_dirs(self.paths))
        with mock.patch.object(photo, "visible_text", return_value="EXIT 4"):
            indexer.run_index(self.paths, srcs, self.roots(), jobs=1, embedder=FakeEmbedder(), progress=lambda e: None,
                              stage_log=log)
        stages = {(e["stage"], e["id"]): e for e in self.events(log) if e.get("event") == "stage" and e.get("status") == "done"}
        self.assertEqual(stages[("photo", "3.6")]["photograph"], True)
        self.assertIn(("locate", "3.7"), stages)                              # it has a position and the town table is installed
        self.assertIn(("convert", "3"), stages)
        self.assertEqual(stages[("sniff", "3.0")]["route"], "photograph")      # decided first, from the file itself ...
        self.assertNotIn(("profile", "3.1"), stages)                          # ... so no page is profiled
        plans = [e for e in self.events(log) if e.get("event") == "plan"]
        self.assertEqual(plans, [])                                           # no pages are planned: it is not a page
        trace_file = next(self.paths.markup.rglob("p.trace.json"))
        summary = json.loads(trace_file.read_text(encoding="utf-8"))["summary"]
        self.assertEqual(summary["branches"], {"photo": 1})
        self.assertEqual(summary["readers"], ["photo"])

    def test_a_photographed_page_records_the_photo_stage_but_keeps_its_page_branch(self):
        (self.sdir / "c").mkdir()
        jpeg(self.sdir / "c" / "page.jpg")

        def convert(src, out, ocr=None):
            out.write_text("<!-- page 1 -->\n\n" + "The agreement is made between the parties. " * 10 + "\n", encoding="utf-8")
            return {"pages": 1, "seconds": 0.0}
        log = self.tmp / "stages2.jsonl"
        srcs = indexer.scan_sources(self.sdir, indexer.exclude_dirs(self.paths))
        with mock.patch.object(photo, "visible_text", return_value="word " * 120), \
                mock.patch("rag_search.core.docling_convert.convert_file", convert):
            indexer.run_index(self.paths, srcs, self.roots(), jobs=1, embedder=FakeEmbedder(), progress=lambda e: None,
                              stage_log=log)
        stages = {e["stage"]: e for e in self.events(log) if e.get("event") == "stage" and e.get("status") == "done"}
        self.assertEqual(stages["photo"]["photograph"], False)
        self.assertEqual(stages["sniff"]["route"], "pages")                   # a page: profiled and read down the lanes
        self.assertIn("profile", stages)
        summary = json.loads(next(self.paths.markup.rglob("page.trace.json")).read_text(encoding="utf-8"))["summary"]
        self.assertNotIn("photo", summary["branches"])
