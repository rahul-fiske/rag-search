"""rag_get_photo: an indexed photograph comes back as an upright JPEG (HEIC too), with the errors a client can act on."""

from __future__ import annotations

import asyncio
import io
import json
import os
from unittest import mock

from PIL import Image

from tests.helpers import FakeEmbedder, TempHome
from tests.portable.test_photo import jpeg

from rag_search import api, policy
from rag_search.core import indexer, photo
from rag_search.mcp import server as mcp_server

try:
    import pillow_heif
    HAVE_HEIF = True
except ImportError:  # pragma: no cover
    HAVE_HEIF = False


class Base(TempHome):
    def setUp(self):
        super().setUp()
        self.use_fake_backends()
        os.environ["RAG_SEARCH_ROUTING"] = "document"
        (self.sdir / "pics" / "trip").mkdir(parents=True)
        self.big = self.sdir / "pics" / "trip" / "big.jpg"
        ex = Image.Exif()
        ex[0x010F], ex[0x0110] = "Apple", "iPhone 13 Pro"      # a photograph needs camera details to be indexed
        ex[0x0112] = 6                                    # rotate 90 clockwise to display
        Image.new("RGB", (3000, 2000), "red").save(self.big, exif=ex)
        jpeg(self.sdir / "pics" / "small.jpg")
        self.write_doc("pics/note.md", "# Note\n\nnot a picture\n")
        (self.sdir / "other").mkdir()
        jpeg(self.sdir / "other" / "secret.jpg")
        policy.save_rules(self.paths, {"other": ["claude"]})
        self.reindex()

    def reindex(self):
        srcs = indexer.scan_sources(self.sdir, indexer.exclude_dirs(self.paths))
        with mock.patch.object(photo, "visible_text", return_value=""):
            indexer.run_index(self.paths, srcs, self.roots(), jobs=1, embedder=FakeEmbedder(), progress=lambda e: None)
        self.publish()

    def get(self, doc, collection="pics", client="claude", **kw):
        return api.get_photo(self.paths, collection, doc, client=client, **kw)

    def decode(self, r):
        self.assertTrue(r["ok"], r)
        return Image.open(io.BytesIO(r["result"]["jpeg_bytes"]))


class GetPhotoTests(Base):
    def test_a_jpeg_is_shrunk_and_turned_upright(self):
        r = self.get("trip/big", max_px=800)
        im = self.decode(r)
        self.assertEqual(im.format, "JPEG")
        self.assertEqual(im.size, (533, 800))                       # 3000x2000 turned to portrait, long side 800
        m = r["result"]["meta"]
        self.assertEqual((m["collection"], m["doc"], m["width"], m["height"]), ("pics", "trip/big", 533, 800))
        self.assertEqual(m["bytes"], len(r["result"]["jpeg_bytes"]))

    def test_a_small_photo_is_not_enlarged_and_carries_when_it_was_taken(self):
        m = self.get("small")["result"]["meta"]
        self.assertEqual((m["width"], m["height"]), (64, 48))
        self.assertTrue(m["taken"].startswith("2024-03-14 13:12:05"))

    def test_the_source_file_name_works_too(self):
        self.assertTrue(self.get("big.jpg")["ok"])

    def test_max_px_and_quality_are_clamped(self):
        self.assertEqual(max(self.decode(self.get("trip/big", max_px=99999)).size), 2048)
        self.assertEqual(max(self.decode(self.get("trip/big", max_px=1)).size), 256)
        lo, hi = len(self.get("trip/big", quality=1)["result"]["jpeg_bytes"]), len(self.get("trip/big", quality=100)["result"]["jpeg_bytes"])
        self.assertEqual(lo, len(self.get("trip/big", quality=50)["result"]["jpeg_bytes"]))
        self.assertEqual(hi, len(self.get("trip/big", quality=95)["result"]["jpeg_bytes"]))

    def test_errors(self):
        self.assertEqual(self.get("secret", collection="other", client="agent")["code"], "bad_request")
        self.assertEqual(self.get("nope")["code"], "bad_request")
        r = self.get("note")
        self.assertEqual((r["code"], r["source"]), ("not_an_image", "note.md"))
        (self.sdir / "pics" / "small.jpg").unlink()
        r = self.get("small")
        self.assertEqual((r["code"], r["source"]), ("source_missing", "small.jpg"))
        (self.sdir / "pics" / "trip" / "big.jpg").write_bytes(b"not a jpeg")
        r = self.get("trip/big")
        self.assertEqual(r["code"], "decode_failed")
        self.assertTrue(r["detail"])

    def test_a_heic_comes_back_as_jpeg(self):
        if not HAVE_HEIF:
            self.skipTest("pillow-heif is not installed")
        pillow_heif.register_heif_opener()
        ex = Image.Exif()
        ex[0x010F], ex[0x0110] = "Apple", "iPhone 13 Pro"
        Image.new("RGB", (400, 300), "blue").save(self.sdir / "pics" / "phone.heic", exif=ex)
        self.reindex()
        im = self.decode(self.get("phone"))
        self.assertEqual((im.format, im.size), ("JPEG", (400, 300)))


class McpTests(Base):
    def test_the_tool_returns_json_then_the_image_and_errors_as_json(self):
        tools = mcp_server.make_tools("claude")
        out = asyncio.run(tools["rag_get_photo"]("pics", "small"))
        self.assertEqual(json.loads(out[0])["doc"], "small")
        self.assertEqual(out[1].to_image_content().mimeType, "image/jpeg")
        err = json.loads(asyncio.run(tools["rag_get_photo"]("pics", "note")))
        self.assertEqual((err["code"], err["source"]), ("not_an_image", "note.md"))
