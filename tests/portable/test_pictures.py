"""5.2 Picture vectors and S3b: a photograph gets a vector from its pixels with a key of its own (a new picture model or
the first run after picture vectors exist embeds only the photographs), the merge keeps which passage each picture
belongs to, and search finds a photograph by its picture and shows the multimodal reranker the picture."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np

from tests.helpers import FakeEmbedder, FakePictureReranker, FakeReranker, TempHome
from tests.portable.test_photo import fake_fetch, jpeg

from rag_search import models, reference
from rag_search.core import embedding, indexer, photo, pictures
from rag_search.spec import parse_stages

BASE, LARGE = models.PICTURE_CATALOG[0].id, models.PICTURE_CATALOG[1].id


class PhotoHome(TempHome):
    def setUp(self):
        super().setUp()
        self.use_fake_backends()
        os.environ["RAG_SEARCH_ROUTING"] = "document"
        reference.download(self.paths, "geonames", fetch=fake_fetch)

    def run_index(self):
        srcs = indexer.scan_sources(self.sdir, indexer.exclude_dirs(self.paths))
        with mock.patch.object(photo, "visible_text", return_value=""):
            return indexer.run_index(self.paths, srcs, self.roots(), jobs=1, embedder=FakeEmbedder(),
                                     progress=lambda e: None)


class PictureVectorTests(PhotoHome):
    def setUp(self):
        super().setUp()
        os.environ["RAG_SEARCH_PICTURE_EMBEDDER"] = "tests.helpers:FakePictureEmbedder"
        os.environ["RAG_SEARCH_PICTURE_MODEL"] = BASE
        (self.sdir / "trip").mkdir()
        jpeg(self.sdir / "trip" / "golden_gate_bridge.jpg")
        jpeg(self.sdir / "trip" / "snowy_mountain.jpg", when="2024:03:15 09:00:00")
        self.write_doc("docs/invoice.md", "# Invoice\n\nInvoice total 2023 for the bridge repair: 400 rupees.\n")

    def doc_dir(self, name: str) -> Path:
        return next(p.parent for p in self.paths.index.rglob(f"{name}/index.meta.json"))

    def test_each_photograph_gets_a_picture_vector_with_its_own_key_and_a_document_none(self):
        s = self.run_index()
        self.assertEqual(s["pictures"]["done"], 2)
        d = self.doc_dir("golden_gate_bridge")
        key = json.loads((d / pictures.META).read_text())
        self.assertEqual((key["model"], key["src_sha256"]), (BASE, indexer.sha256_file(self.sdir / "trip" / "golden_gate_bridge.jpg")))
        self.assertEqual(np.load(d / pictures.FILE).shape, (1, 64))
        self.assertFalse((self.doc_dir("invoice") / pictures.FILE).exists())
        all_dir = self.paths.index / "trip" / "_all"
        rows = json.loads((all_dir / pictures.ROWS).read_text())
        nodes = json.loads((all_dir / "nodes.json").read_text())["nodes"]
        self.assertEqual(rows["model"], BASE)
        self.assertEqual(sorted(Path(nodes[r]["metadata"]["src_path"]).stem for r in rows["rows"]),
                         ["golden_gate_bridge", "snowy_mountain"])
        self.assertFalse((self.paths.index / "docs" / "_all" / pictures.FILE).exists())

    def test_the_first_run_with_picture_vectors_embeds_only_the_photographs(self):
        os.environ["RAG_SEARCH_PICTURE_MODEL"] = "off"
        self.run_index()
        self.assertFalse(list(self.paths.index.rglob(pictures.FILE)))
        os.environ["RAG_SEARCH_PICTURE_MODEL"] = BASE
        before = (self.doc_dir("snowy_mountain") / "embeddings.npy").stat().st_mtime_ns
        emb = FakeEmbedder()
        with mock.patch.object(photo, "visible_text", return_value=""):
            s = indexer.run_index(self.paths, indexer.scan_sources(self.sdir, indexer.exclude_dirs(self.paths)),
                                  self.roots(), jobs=1, embedder=emb, progress=lambda e: None)
        self.assertEqual((s["indexed"], emb.calls, s["pictures"]["done"]), (0, 0, 2))       # nothing converted or embedded
        self.assertEqual((self.doc_dir("snowy_mountain") / "embeddings.npy").stat().st_mtime_ns, before)
        self.assertTrue((self.paths.index / "trip" / "_all" / pictures.FILE).exists())     # and merged again
        self.assertEqual(self.run_index()["pictures"]["done"], 0)                        # then they are current

    def test_a_photograph_without_camera_details_is_caught_up_by_what_its_conversion_says(self):
        idx, md = self.tmp / "print", self.tmp / "print.md"
        idx.mkdir()
        md.write_text("- Looks like (model-written): two people at a counter\n- Faces detected: 2\n")   # no Photo details
        task = {"picture_model": BASE}
        for branches, wanted in (({"photo": 1}, True), ({"image": 1}, False), ({"photo": 1, "digital": 1}, False)):
            (idx / "index.meta.json").write_text(json.dumps({"conversion": {"branches": branches}}))
            self.assertEqual(indexer._wants_picture(task, "image", md, idx, "s"), wanted, branches)

    def test_another_picture_model_embeds_the_photographs_again(self):
        self.run_index()
        os.environ["RAG_SEARCH_PICTURE_MODEL"] = LARGE
        s = self.run_index()
        self.assertEqual((s["indexed"], s["pictures"]["done"]), (0, 2))
        self.assertEqual(json.loads((self.doc_dir("snowy_mountain") / pictures.META).read_text())["model"], LARGE)

    def test_a_picture_model_that_is_not_downloaded_leaves_the_step_out(self):
        del os.environ["RAG_SEARCH_PICTURE_EMBEDDER"]
        with tempfile.TemporaryDirectory() as hub, mock.patch.dict(os.environ, {"HF_HUB_CACHE": hub}):
            self.assertIn("not downloaded", pictures.ready(BASE))
            s = self.run_index()
        self.assertEqual(s["pictures"]["done"], 0)
        self.assertIn("not downloaded", s["pictures"]["why_not"])
        self.assertFalse(list(self.paths.index.rglob(pictures.FILE)))

    def test_a_photograph_is_found_by_its_picture_and_the_reranker_sees_it(self):
        self.run_index()
        self.publish()
        cat = json.loads((self.paths.current_gen() / "catalog.json").read_text())
        self.assertEqual(cat["picture_model"], BASE)
        self.assertTrue((self.paths.current_gen() / "index" / "trip" / "_all" / pictures.FILE).exists())
        rr = FakePictureReranker()
        eng = self.engine(reranker=rr)
        only = eng.search("golden gate", stages="pictures")["results"]
        self.assertEqual(only[0]["file"], "golden_gate_bridge")
        self.assertEqual(only[0]["picture_rank"], 1)
        res = eng.search("golden gate", top_k=3)
        self.assertEqual(res["results"][0]["file"], "golden_gate_bridge")
        self.assertEqual(res["results"][0]["rerank_score"], 1.0)
        shown = [Path(p).name for p in rr.seen[-1] if p]
        self.assertEqual(sorted(shown), ["golden_gate_bridge.jpg", "snowy_mountain.jpg"])     # photographs only
        self.assertIn("picture_candidates", res["timing"])

    def test_a_text_only_reranker_is_given_text_and_a_collection_without_pictures_is_searched_as_before(self):
        self.run_index()
        self.publish()
        eng = self.engine(reranker=FakeReranker())
        res = eng.search("invoice total", collections=["docs"])
        self.assertEqual(res["results"][0]["file"], "invoice")
        self.assertIsNone(res["results"][0]["picture_score"])
        self.assertNotIn("picture_candidates", res["timing"])

    def test_the_people_a_query_names_keep_the_pictures_to_their_photographs(self):
        self.run_index()
        self.publish()
        eng = self.engine()
        src = str(self.sdir / "trip" / "snowy_mountain.jpg")
        got = eng.search("golden gate", stages="pictures", only={src})["results"]
        self.assertEqual([r["file"] for r in got], ["snowy_mountain"])


class PictureBitsTests(PhotoHome):
    def test_stages_take_pictures_alone_but_never_no_retriever(self):
        self.assertEqual(parse_stages("pictures"), ("pictures",))
        self.assertIn("pictures", parse_stages(None))
        with self.assertRaises(ValueError):
            parse_stages("rerank")

    def test_the_reranker_reads_a_photograph_without_its_position_camera_and_size(self):
        text = "- Taken: 2024-03-14\n- Position: 19.1, 78.3\n- Camera: Apple iPhone\n- Size: 64 x 48\n- Town: at Nirmal"
        self.assertEqual(embedding.trimmed_photo_text(text), "- Taken: 2024-03-14\n- Town: at Nirmal")

    def test_the_multimodal_reranker_is_made_for_its_model_id(self):
        with mock.patch.dict(os.environ, {"RAG_SEARCH_RERANKER": ""}):
            self.assertIsInstance(embedding.make_reranker("Qwen/Qwen3-VL-Reranker-2B"), embedding.Qwen3VLReranker)

    def test_the_picture_card_says_whether_the_reranker_can_see_pictures(self):
        os.environ.pop("RAG_SEARCH_PICTURE_MODEL", None)
        st = models.picture_state(self.paths)
        self.assertEqual(st["active"], BASE)
        self.assertEqual([m["id"] for m in st["models"]], [BASE, LARGE])
        self.assertFalse(st["reranker_sees_pictures"])
        models.set_selection(self.paths, models.RERANKER, "Qwen/Qwen3-VL-Reranker-2B")
        self.assertTrue(models.picture_state(self.paths)["reranker_sees_pictures"])
        models.set_picture_selection(self.paths, "off")
        self.assertTrue(models.picture_state(self.paths)["off"])
        with self.assertRaises(models.ModelError):
            models.set_picture_selection(self.paths, "someone/else")

    def test_merge_keeps_one_picture_model(self):
        a, b, c = (self.tmp / n for n in "abc")
        for d, model in ((a, BASE), (b, BASE), (c, LARGE)):
            d.mkdir()
            pictures.write(d, np.ones(4 if model == BASE else 6, np.float32), "s", model)
        mat, rows = pictures.merged([(a, 0), (b, 3), (c, 5)])
        self.assertEqual((mat.shape, rows), ((2, 4), {"model": BASE, "rows": [0, 3]}))
        self.assertEqual(pictures.merged([]), (None, {}))


if __name__ == "__main__":
    import unittest

    unittest.main()
