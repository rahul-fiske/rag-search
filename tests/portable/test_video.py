"""Videos: 3.11 Video details from the QuickTime / MP4 header atoms, and the Live Photo videos 1 Discover leaves out."""

from __future__ import annotations

import json
import struct
from datetime import UTC, datetime
from pathlib import Path
from unittest import mock

from rag_search import locations, reference
from rag_search.core import indexer, understanding, video
from rag_search.core.conversion import runview
from tests.helpers import FakeEmbedder, TempHome
from tests.portable.test_photo import fake_fetch, jpeg


def atom(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I4s", 8 + len(body), kind) + body


def quicktime(path: Path, *, keys: dict[str, str] | None = None, udta: dict[bytes, str] | None = None,
              created: int = 0, seconds: float = 32.0, size=(1920, 1080), rotated=False) -> Path:
    """A minimal QuickTime file: a movie header, one video track header, Apple's keys / ilst metadata, and a few bytes of
    frames."""
    mvhd = atom(b"mvhd", bytes(4) + struct.pack(">IIII", created, created, 600, int(seconds * 600)) + bytes(80))
    matrix = (0, 65536, 0, -65536, 0, 0, 0, 0, 1 << 30) if rotated else (65536, 0, 0, 0, 65536, 0, 0, 0, 1 << 30)
    tkhd = atom(b"tkhd", bytes(4) + struct.pack(">IIII", 0, 0, 1, 0) + struct.pack(">I", 0) + bytes(8)
                + struct.pack(">hhhh", 0, 0, 0, 0) + struct.pack(">9i", *matrix)
                + struct.pack(">II", size[0] << 16, size[1] << 16))
    parts = [mvhd, atom(b"trak", tkhd)]
    if keys:
        names = list(keys)
        kbody = struct.pack(">II", 0, len(names)) + b"".join(
            struct.pack(">I4s", 8 + len(n), b"mdta") + n.encode() for n in names)
        items = b"".join(atom(struct.pack(">I", i + 1), atom(b"data", struct.pack(">II", 1, 0) + keys[n].encode()))
                         for i, n in enumerate(names))
        parts.append(atom(b"meta", atom(b"hdlr", bytes(24)) + atom(b"keys", kbody) + atom(b"ilst", items)))
    if udta:
        parts.append(atom(b"udta", b"".join(atom(k, struct.pack(">HH", len(v.encode()), 0) + v.encode())
                                            for k, v in udta.items())))
    path.write_bytes(atom(b"ftyp", b"qt  \x00\x00\x00\x00qt  ") + atom(b"mdat", b"\x00" * 64) + atom(b"moov", b"".join(parts)))
    return path


IPHONE = {"com.apple.quicktime.creationdate": "2024-03-14T13:12:05+0530",
          "com.apple.quicktime.location.ISO6709": "+19.0905+078.3412+312.000/",
          "com.apple.quicktime.make": "Apple", "com.apple.quicktime.model": "iPhone 13 Pro"}


class VideoDetailsTests(TempHome):
    def test_an_iphone_video_has_its_local_time_position_camera_size_and_duration(self):
        meta = video.read_meta(quicktime(self.tmp / "IMG_1.MOV", keys=IPHONE, seconds=64.0, rotated=True))
        self.assertEqual(str(meta["when"]), "2024-03-14 13:12:05")
        self.assertEqual(meta["offset"], "+05:30")
        self.assertAlmostEqual(meta["lat"], 19.0905)
        self.assertAlmostEqual(meta["lon"], 78.3412)
        self.assertEqual(meta["altitude"], 312.0)
        self.assertEqual(meta["camera"], "Apple iPhone 13 Pro")
        self.assertEqual(meta["size"], (1080, 1920))          # held upright: shown as it was filmed
        self.assertEqual(meta["duration"], 64.0)

    def test_an_mp4_without_metadata_has_the_movie_header_time_in_utc(self):
        utc = UTC
        created = int((datetime(2020, 12, 22, 1, tzinfo=utc) - datetime(1904, 1, 1, tzinfo=utc)).total_seconds())
        meta = video.read_meta(quicktime(self.tmp / "a.mp4", created=created))
        self.assertEqual(str(meta["when"]), "2020-12-22 01:00:00")
        self.assertEqual(meta["offset"], "+00:00")
        self.assertNotIn("lat", meta)

    def test_an_android_position_in_udta(self):
        meta = video.read_meta(quicktime(self.tmp / "b.mp4", udta={b"\xa9xyz": "-33.8688+151.2093/"}))
        self.assertAlmostEqual(meta["lat"], -33.8688)
        self.assertAlmostEqual(meta["lon"], 151.2093)

    def test_a_file_that_is_not_a_video_has_no_details(self):
        (self.tmp / "x.mov").write_bytes(b"not a movie at all")
        self.assertEqual(video.read_meta(self.tmp / "x.mov"), {})
        self.assertEqual(video.block(self.tmp / "x.mov", self.paths), "")

    def test_the_section_uses_the_photograph_labels(self):
        text = video.block(quicktime(self.tmp / "IMG_2.MOV", keys=IPHONE, seconds=3725), self.paths)
        self.assertTrue(text.startswith("## Video details\n\n- Taken: 2024-03-14 13:12:05 +05:30 (Thursday)"))
        self.assertIn("- Position: 19.09050 N, 78.34120 E, altitude 312 m (geo:19.09050,78.34120)", text)
        self.assertIn("- Camera: Apple iPhone 13 Pro", text)
        self.assertIn("- Duration: 1 h 2 min", text)

    def test_durations_read_naturally(self):
        self.assertEqual([video.duration_text(s) for s in (0.4, 2.9, 59, 64, 3725)], ["1 s", "3 s", "59 s", "1 min 4 s", "1 h 2 min"])

    def test_a_live_photo_video_belongs_to_its_picture(self):
        names = ["IMG_7.HEIC", "IMG_7.MOV", "IMG_8.MOV", "clip.mp4", "clip.txt"]
        self.assertEqual(video.companion_of(Path("IMG_7.MOV"), names), "IMG_7.HEIC")
        self.assertEqual(video.companion_of(Path("IMG_8.MOV"), names), "")
        self.assertEqual(video.companion_of(Path("clip.mp4"), names), "")      # a text file is not a picture

    def test_understood_as_a_video(self):
        text = video.block(quicktime(self.tmp / "IMG_3.MOV", keys=IPHONE), self.paths)
        u = understanding.of_document(fmt="video", suffix=".mov", status="prepared", text=text, route="video")
        self.assertEqual((u["kind"], u["track"]), ("video", "video"))
        self.assertEqual(u["missing"], ["place"])                # no map data installed in this test
        bare = video.block(quicktime(self.tmp / "c.mp4"), self.paths)
        u = understanding.of_document(fmt="video", suffix=".mp4", status="prepared", text=bare, route="video")
        self.assertEqual((u["grade"], u["missing"]), ("partial", ["time", "position"]))
        self.assertEqual(runview._format_of("x/IMG_3.MOV"), "video")


class IndexedVideoTests(TempHome):
    def setUp(self):
        super().setUp()
        self.use_fake_backends()
        reference.download(self.paths, "geonames", fetch=fake_fetch)

    def test_a_video_is_indexed_from_its_details_and_a_live_photo_video_is_left_out(self):
        (self.sdir / "trip").mkdir()
        quicktime(self.sdir / "trip" / "IMG_9.MOV", keys=IPHONE)
        jpeg(self.sdir / "trip" / "IMG_10.jpg")
        quicktime(self.sdir / "trip" / "IMG_10.MOV", keys=IPHONE, seconds=2.9)
        with mock.patch("rag_search.core.docling_convert.convert_file", side_effect=AssertionError("docling")):
            found, unsup = locations.scan_tree(self.sdir, indexer.exclude_dirs(self.paths))
            summary = indexer.run_index(self.paths, found, self.roots(), unsupported=unsup, jobs=1, embedder=FakeEmbedder(),
                                        progress=lambda e: None)
        text = next(self.paths.markup.rglob("IMG_9.md")).read_text(encoding="utf-8")
        self.assertTrue(text.startswith("<!-- page 1 -->\n\n## Video details"))
        self.assertIn("Town: at Nirmal, Telangana, India", text)
        nodes = json.loads(next(self.paths.index.rglob("IMG_9/nodes.json")).read_text())["nodes"]
        self.assertTrue(any("2024-03-14" in n["text"] for n in nodes))
        meta = json.loads(next(self.paths.index.rglob("IMG_9/index.meta.json")).read_text())
        self.assertEqual(meta["conversion"]["branches"], {"video": 1})
        self.assertEqual([u["companion"] for u in summary["live_photo"]], ["IMG_10.jpg"])
        self.assertEqual(summary["unsupported_extension"], [])
        self.assertFalse(any(e["src"].endswith("IMG_10.MOV") for e in summary["errors"]))   # not "the same name as"
        self.assertEqual(summary["indexed"], 2)                                               # the video and the photo
