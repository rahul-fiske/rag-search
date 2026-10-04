"""The document VLM reader (phase P3): child process, protocol, memory guard, fallback, pictures,
image files, model catalogue.  The model is a fake backend (``tests.helpers:FakeVlmBackend``): no
MLX, no weights; the child process and its JSON-lines protocol are real."""

from __future__ import annotations

import json
import os
import time
import unittest
from pathlib import Path
from unittest import mock

from tests.helpers import TempHome, no_real_reader
from tests.portable.test_conversion import HAVE_PDF, TEXT, ConversionBase, write_text_pdf
from tests.portable.test_conv_routed import FakeReader, write_routing_pdf

from rag_search import models
from rag_search.core.conversion import gate, pagecache, profiler, routed, trace, vlm
from rag_search.core.docling_convert import NoTextError

if HAVE_PDF:
    import pypdfium2 as pdfium
    from PIL import Image, ImageDraw

FAKE = "tests.helpers:FakeVlmBackend"


class VlmBase(TempHome):
    """Environment for a fake reader; a plan file steers the fake (see FakeVlmBackend)."""

    def setUp(self):
        super().setUp()
        os.environ["RAG_SEARCH_VLM_BACKEND"] = FAKE
        os.environ["RAG_SEARCH_VLM_FREE_GB"] = "64"
        self.plan_file = self.tmp / "plan.json"
        self.calls_log = self.tmp / "calls.log"
        self.plan()
        os.environ["RAG_TEST_VLM_PLAN"] = str(self.plan_file)
        self.addCleanup(vlm.close_shared)

    def plan(self, **plan):
        plan = dict(plan, log=str(self.calls_log))
        self.plan_file.write_text(json.dumps(plan))

    def calls(self) -> int:
        return len(self.calls_log.read_text().splitlines()) if self.calls_log.exists() else 0

    def reader(self, **kw) -> "vlm.VlmReader":
        r = vlm.VlmReader("fake/model", style="instruct", need_gb=1.0, **kw)
        self.addCleanup(r.close)
        return r

    def png(self, name="a.png", size=(400, 500), dark=True):
        im = Image.new("RGB", size, (255, 255, 255))
        if dark:
            d = ImageDraw.Draw(im)
            for y in range(40, size[1] - 40, 30):
                d.rectangle((40, y, size[0] - 40, y + 12), fill=(0, 0, 0))
        p = self.tmp / name
        im.save(p)
        return p


@unittest.skipUnless(HAVE_PDF, "needs pypdfium2 and Pillow")
class ReaderProcessTests(VlmBase):
    def test_reads_an_image_through_the_child_process(self):
        r = self.reader()
        res = r.read_image(self.png())
        self.assertIn("Scanned statement page", res["md"])
        self.assertGreater(res["tokens"], 0)
        self.assertGreater(res["rss_mb"], 0)
        self.assertEqual(r.pages_read, 1)
        self.assertTrue(r.worker.alive())
        r.close()
        self.assertIsNone(r.worker)

    def test_the_prompt_depends_on_the_model_style(self):
        self.assertIn("HTML <table>", vlm.prompt_for("instruct"))
        self.assertEqual(vlm.prompt_for("paddleocr"), "OCR:")
        self.assertEqual(vlm.prompt_for("paddleocr", "table"), "Table Recognition:")

    def test_a_model_that_does_not_load_is_unavailable_for_the_run(self):
        self.plan(load_error="weights are corrupt")
        r = self.reader()
        with self.assertRaises(vlm.ReaderUnavailable) as cm:
            r.read_image(self.png())
        self.assertIn("weights are corrupt", str(cm.exception))
        self.assertIn("weights are corrupt", r.dead)
        started = self.calls()
        with self.assertRaises(vlm.ReaderUnavailable):          # no new process for the next page
            r.read_image(self.png())
        self.assertEqual(self.calls(), started)

    def test_not_enough_free_memory_is_reported_but_not_held_against_the_reader(self):
        os.environ["RAG_SEARCH_VLM_FREE_GB"] = "1.5"
        r = self.reader()                                      # needs 1.0 + 2.0 headroom
        with self.assertRaises(vlm.ReaderUnavailable) as cm:
            r.read_image(self.png())
        self.assertIn("1.5 GB of memory is free", str(cm.exception))
        self.assertEqual(r.dead, "")
        os.environ["RAG_SEARCH_VLM_FREE_GB"] = "64"            # memory came back
        self.assertIn("Scanned", r.read_image(self.png())["md"])

    def test_a_crash_costs_one_page_and_the_reader_restarts(self):
        self.plan(crash_on=[2])
        r = self.reader()
        r.read_image(self.png("1.png"))
        with self.assertRaises(vlm.ReaderCrashed):
            r.read_image(self.png("2.png"))
        self.assertIn("Scanned", r.read_image(self.png("3.png"))["md"])
        self.assertEqual(r.restarts, 1)

    def test_native_output_on_fd_1_does_not_break_the_protocol(self):
        self.plan(native_write=True)
        r = self.reader()
        t0 = time.monotonic()
        self.assertIn("Scanned", r.read_image(self.png())["md"])
        self.assertLess(time.monotonic() - t0, 20)                # not a page timeout followed by a restart
        self.assertEqual(r.restarts, 0)

    def test_a_hang_is_cut_off_by_the_page_time_limit(self):
        self.plan(hang_on=[1])
        os.environ["RAG_SEARCH_VLM_PAGE_TIMEOUT"] = "2"
        r = self.reader()
        t0 = time.monotonic()
        with self.assertRaises(vlm.ReaderTimeout):
            r.read_image(self.png())
        self.assertLess(time.monotonic() - t0, 30)
        self.assertFalse(r.worker.alive())

    def test_a_reader_that_keeps_dying_is_given_up(self):
        self.plan(crash_on=[1, 2, 3, 4, 5])
        r = self.reader()
        outcomes = []
        for i in range(5):
            try:
                r.read_image(self.png(f"{i}.png"))
            except vlm.ReaderError as exc:
                outcomes.append(type(exc).__name__)
        self.assertTrue(r.dead)
        self.assertIn("stopped too often", r.dead)
        self.assertEqual(outcomes[0], "ReaderCrashed")
        self.assertEqual(outcomes[-1], "ReaderUnavailable")

    def test_an_error_on_one_page_does_not_kill_the_process(self):
        self.plan(error_on=[1])
        r = self.reader()
        with self.assertRaises(vlm.ReaderError):
            r.read_image(self.png("1.png"))
        self.assertIn("Scanned", r.read_image(self.png("2.png"))["md"])
        self.assertEqual(r.restarts, 0)

    def test_the_real_backend_needs_apple_silicon_and_a_downloaded_model(self):
        no_real_reader(self)
        r = vlm.VlmReader("mlx-community/Qwen3-VL-4B-Instruct-4bit", backend="mlx")
        r.check()
        self.assertTrue(r.dead)                                 # this machine is not an Apple Silicon Mac
        self.assertTrue("Apple Silicon" in r.dead or "mlx-vlm" in r.dead or "not downloaded" in r.dead)

    def test_pdf_pages_and_tiff_frames_are_rendered_and_read(self):
        pdf = self.tmp / "s.pdf"
        write_routing_pdf(pdf)
        r = self.reader()
        res = r.read(pdf, 3, 3, "scan")
        self.assertEqual(sorted(res["pages"]), [3])
        self.assertEqual(res["failed"], {})
        self.assertGreater(res["stats"][3]["tokens"], 0)
        tif = self.tmp / "m.tif"
        a, b = Image.new("RGB", (300, 400), (255, 255, 255)), Image.new("RGB", (300, 400), (200, 200, 200))
        ImageDraw.Draw(b).rectangle((20, 20, 200, 60), fill=(0, 0, 0))
        a.save(tif, save_all=True, append_images=[b])
        res = r.read(tif, 1, 2, "scan")
        self.assertEqual(sorted(res["pages"]), [1, 2])
        self.assertNotEqual(res["pages"][1], res["pages"][2])    # different frames, different images

    def test_failed_pages_are_listed_with_the_reason(self):
        pdf = self.tmp / "s.pdf"
        write_routing_pdf(pdf)
        self.plan(error_on=[1])
        res = self.reader().read(pdf, 3, 4, "scan")
        self.assertEqual(sorted(res["pages"]), [4])
        self.assertIn("could not read this page", res["failed"][3])


@unittest.skipUnless(HAVE_PDF, "needs pypdfium2 and Pillow")
class RoutedVlmTests(VlmBase):
    def setUp(self):
        super().setUp()
        self.pdf = self.tmp / "r.pdf"
        write_routing_pdf(self.pdf)                             # pages: text, text, scan, blank
        self.profile = profiler.profile_file(self.pdf)
        self.cache = pagecache.PageCache(self.paths.workspace)
        self.docling = FakeReader()
        self.out = self.tmp / "out.md"

    def convert(self, scan_reader):
        return routed.convert_pdf(self.pdf, self.out, self.profile, cache=self.cache, reader=self.docling,
                                  scan_reader=scan_reader)

    def test_scanned_pages_go_to_the_document_reader(self):
        r = self.reader()
        res = self.convert(r)
        self.assertEqual(self.docling.calls, [(1, 2, "digital")])
        recs = {x["page"]: x for x in res["records"]}
        self.assertEqual([recs[n]["branch"] for n in (1, 2, 3, 4)], ["digital", "digital", "raster", "raster"])
        p3 = recs[3]
        self.assertEqual(p3["reader"], {"tool": "vlm", "model": "fake/model", "mode": "page"})
        self.assertGreater(p3["tokens"], 0)
        self.assertGreater(p3["gpu_s"], 0)
        self.assertGreater(p3["time_s"]["read"], 0)
        self.assertEqual(p3["outcome"], "low")                  # the fixture scan is ~94 dpi
        self.assertEqual([c["name"] for c in p3["gate"]["checks"]], ["low_resolution"])
        self.assertEqual(recs[4]["outcome"], "no_text")         # the blank page was not read at all
        self.assertEqual(self.calls(), 1)
        self.assertIn("Scanned statement page", self.out.read_text())
        s = trace.summarize(res["records"], readers=["docling", "vlm"])
        self.assertEqual(s["models"], ["fake/model"])
        self.assertGreater(s["tokens"], 0)
        self.assertGreater(s["gpu_s"], 0)

    def test_a_page_the_reader_cannot_read_falls_back_to_docling_and_says_why(self):
        self.plan(error_on=[1])
        res = self.convert(self.reader())
        p3 = [x for x in res["records"] if x["page"] == 3][0]
        self.assertEqual(p3["branch"], "fallback")
        self.assertEqual(p3["reader"]["tool"], "fake")             # "fake" stands in for docling
        self.assertIn("could not read this page", p3["note"])
        self.assertEqual(self.docling.calls, [(1, 2, "digital"), (3, 3, "scan")])
        self.assertIn("read as scan", self.out.read_text())

    def test_a_reader_that_cannot_start_is_ruled_out_before_any_page(self):
        no_real_reader(self)
        r = vlm.VlmReader("mlx-community/Qwen3-VL-4B-Instruct-4bit", backend="mlx")
        res = self.convert(r)
        p3 = [x for x in res["records"] if x["page"] == 3][0]
        self.assertEqual(p3["branch"], "fallback")
        self.assertTrue("Apple Silicon" in p3["note"] or "mlx-vlm" in p3["note"] or "downloaded" in p3["note"])
        self.assertEqual(self.calls(), 0)

    def test_no_reader_at_all_is_the_old_behaviour(self):
        res = self.convert(None)
        p3 = [x for x in res["records"] if x["page"] == 3][0]
        self.assertEqual(p3["branch"], "fallback")
        self.assertIn("switched off", p3["note"])

    def test_results_are_cached_per_reader_and_model(self):
        r = self.reader()
        self.convert(r)
        before = self.calls()
        res = self.convert(r)                                    # nothing is read twice
        self.assertEqual(self.calls(), before)
        p3 = [x for x in res["records"] if x["page"] == 3][0]
        self.assertEqual((p3["branch"], p3["was"], p3["cache"]), ("cached", "raster", "hit"))
        self.assertEqual(p3["reader"]["tool"], "vlm")
        other = vlm.VlmReader("fake/other-model", need_gb=1.0)   # another model: the page is read again
        self.addCleanup(other.close)
        self.convert(other)
        self.assertEqual(self.calls(), before + 1)

    def test_a_fallback_result_is_cached_for_docling_not_for_the_reader(self):
        self.plan(error_on=[1])
        self.convert(self.reader())
        self.plan()                                              # the reader works again
        self.docling.calls.clear()
        res = self.convert(self.reader())
        p3 = [x for x in res["records"] if x["page"] == 3][0]
        self.assertEqual(p3["branch"], "raster")                 # read by the reader now, not taken from the cache
        self.assertEqual(self.docling.calls, [])

    def test_a_crash_in_the_middle_of_a_run_only_costs_that_page(self):
        pdf = self.tmp / "many.pdf"
        parts = []
        for i in range(3):
            f = self.tmp / f"s{i}.pdf"
            im = Image.new("RGB", (400, 500), (255, 255, 255))
            ImageDraw.Draw(im).rectangle((40, 40, 300, 60 + i * 20), fill=(0, 0, 0))
            im.save(f)
            parts.append(f)
        docs = [pdfium.PdfDocument(str(f)) for f in parts]
        base = docs[0]
        for d in docs[1:]:
            base.import_pages(d)
        base.save(str(pdf))
        prof = profiler.profile_file(pdf)
        self.plan(crash_on=[2])
        res = routed.convert_pdf(pdf, self.tmp / "m.md", prof, cache=self.cache, reader=self.docling,
                                 scan_reader=self.reader())
        self.assertEqual([x["branch"] for x in res["records"]], ["raster", "fallback", "raster"])
        self.assertIn("crashed", [x for x in res["records"] if x["page"] == 2][0]["note"])

    def test_a_large_picture_on_a_text_page_is_read_by_the_reader(self):
        base = self.tmp / "t.pdf"
        write_text_pdf(base, [TEXT])
        pdf = pdfium.PdfDocument(str(base))
        page = pdf[0]
        im = Image.new("RGB", (600, 400), (240, 240, 240))
        d = ImageDraw.Draw(im)
        for y in range(40, 360, 40):
            d.rectangle((40, y, 500, y + 16), fill=(0, 0, 0))
        obj = pdfium.PdfImage.new(pdf)
        obj.set_bitmap(pdfium.PdfBitmap.from_pil(im.convert("RGBA")))
        obj.set_matrix(pdfium.PdfMatrix().scale(450, 380).translate(80, 280))
        page.insert_obj(obj)
        page.gen_content()
        withpic = self.tmp / "p.pdf"
        pdf.save(str(withpic))
        prof = profiler.profile_file(withpic)
        self.assertEqual(len(prof["pages"][0]["big_pics"]), 1)
        self.plan(text="Receipt from the shop: total paid 4,512.00 rupees on the fifth of March.")
        res = routed.convert_pdf(withpic, self.tmp / "p.md", prof, cache=self.cache, reader=self.docling,
                                 scan_reader=self.reader())
        rec = res["records"][0]
        self.assertEqual(rec["branch"], "embedded")
        self.assertEqual(rec["reader"]["mode"], "picture")
        text = (self.tmp / "p.md").read_text()
        self.assertIn("read as digital", text)                   # docling's page text is kept
        self.assertIn("4,512.00 rupees", text)                   # and the picture's text is added
        self.assertEqual(self.calls(), 1)
        routed.convert_pdf(withpic, self.tmp / "p.md", prof, cache=self.cache, reader=self.docling,
                           scan_reader=self.reader())            # cached together
        self.assertEqual(self.calls(), 1)
        # no reader: the page is just a digital page again
        res = routed.convert_pdf(withpic, self.tmp / "p2.md", prof, cache=self.cache, reader=self.docling,
                                 scan_reader=None)
        self.assertEqual(res["records"][0]["branch"], "digital")

    def test_text_already_on_the_page_is_not_added_twice(self):
        page = "Receipt from the shop: total paid 4,512.00 rupees.\n\nThanks for shopping."
        self.assertEqual(routed._new_text(page, "Receipt from the shop: total paid 4,512.00 rupees."), "")
        self.assertEqual(routed._new_text(page, "Thanks for shopping.\nOpen daily 9 to 5."),
                         "Open daily 9 to 5.")
        tab = "<table><tr><td>a</td><td>1</td></tr></table>"
        self.assertEqual(routed._new_text(page, tab), tab)


@unittest.skipUnless(HAVE_PDF, "needs pypdfium2 and Pillow")
class ImageFileTests(VlmBase):
    def setUp(self):
        super().setUp()
        self.cache = pagecache.PageCache(self.paths.workspace)
        self.out = self.tmp / "i.md"

    def convert(self, src, reader):
        prof = profiler.profile_file(src)
        return routed.convert_image(src, self.out, prof, cache=self.cache, scan_reader=reader), prof

    def test_an_image_is_one_page_read_by_the_reader(self):
        src = self.png("scan.png", (1200, 1600))
        res, _ = self.convert(src, self.reader())
        self.assertEqual(len(res["records"]), 1)
        rec = res["records"][0]
        self.assertEqual((rec["branch"], rec["reader"]["tool"]), ("image", "vlm"))
        self.assertIn("Scanned statement page", self.out.read_text())

    def test_a_multi_page_tiff_has_one_page_per_frame(self):
        tif = self.tmp / "m.tif"
        frames = []
        for i in range(3):
            im = Image.new("RGB", (900, 1200), (255, 255, 255))
            ImageDraw.Draw(im).rectangle((50, 50, 500, 80 + 30 * i), fill=(0, 0, 0))
            frames.append(im)
        frames[0].save(tif, save_all=True, append_images=frames[1:])
        res, prof = self.convert(tif, self.reader())
        self.assertEqual(prof["page_count"], 3)
        self.assertEqual([x["page"] for x in res["records"]], [1, 2, 3])
        text = self.out.read_text()
        self.assertEqual(text.count("<!-- page "), 3)

    def test_a_low_resolution_image_is_read_but_flagged(self):
        src = self.png("small.png", (500, 700))
        res, _ = self.convert(src, self.reader())
        rec = res["records"][0]
        self.assertEqual(rec["outcome"], "low")
        self.assertEqual([c["name"] for c in rec["gate"]["checks"]], ["low_resolution"])
        self.assertIn("500 x 700", rec["gate"]["checks"][0]["detail"])

    def test_dpi_in_the_file_decides_when_it_is_given(self):
        good, bad = self.tmp / "good.png", self.tmp / "bad.png"
        base = Image.new("RGB", (1500, 2000), (255, 255, 255))
        ImageDraw.Draw(base).rectangle((100, 100, 900, 140), fill=(0, 0, 0))
        base.save(good, dpi=(300, 300))
        base.save(bad, dpi=(96, 96))
        self.assertEqual(self.convert(good, self.reader())[0]["records"][0]["outcome"], "pass")
        rec = self.convert(bad, self.reader())[0]["records"][0]
        self.assertIn("96 dpi", rec["gate"]["checks"][0]["detail"])

    def test_a_photograph_without_text_is_no_text(self):
        self.plan(empty=True)
        src = self.png("photo.png", (1600, 1200))
        with self.assertRaises(NoTextError):
            self.convert(src, self.reader())

    def test_a_reader_that_cannot_read_raises_so_docling_can_take_over(self):
        self.plan(load_error="no weights")
        with self.assertRaises(vlm.ReaderError):
            self.convert(self.png("scan.png", (1200, 1600)), self.reader())

    def test_exif_orientation_is_applied_before_reading(self):
        src = self.tmp / "rot.jpg"
        im = Image.new("RGB", (1200, 800), (255, 255, 255))
        ImageDraw.Draw(im).rectangle((50, 50, 600, 90), fill=(0, 0, 0))
        exif = Image.Exif()
        exif[0x0112] = 6                                          # rotate 90 clockwise to display
        im.save(src, exif=exif)
        prof = profiler.profile_file(src)
        self.assertEqual(prof["pages"][0]["exif_orientation"], 6)
        out = vlm.render_image_frame(src, 1, self.tmp / "o.png")
        with Image.open(out) as o:
            self.assertEqual(o.size, (800, 1200))

    def test_heic_is_a_supported_extension(self):
        from rag_search.paths import SUPPORTED_EXTENSIONS

        self.assertTrue({".heic", ".heif"} <= SUPPORTED_EXTENSIONS)
        self.assertEqual(profiler.kind_of(Path("x.HEIC")), "image")


class ModelCatalogueTests(TempHome):
    def test_the_catalogue_is_consistent_and_ids_are_real_looking(self):
        ids = [(m.kind, m.id) for m in models.VLM_CATALOG]
        self.assertEqual(len(ids), len(set(ids)))
        for m in models.VLM_CATALOG:
            self.assertIn(m.kind, models.VLM_KINDS)
            self.assertTrue(models._ID_RE.match(m.id), m.id)
            self.assertTrue(m.mem_gb > 0 and m.license and m.note and m.style, m.id)
        for kind, mid in models.VLM_DEFAULTS.items():
            self.assertIsNotNone(models.vlm_find(kind, mid))

    def test_selection_default_config_and_environment(self):
        self.assertEqual(models.vlm_selection(models.READER),
                         (models.VLM_DEFAULTS[models.READER], "default"))
        models.set_vlm_selection(self.paths, models.READER, "mlx-community/PaddleOCR-VL-1.5-8bit")
        self.assertEqual(models.vlm_selection(models.READER),
                         ("mlx-community/PaddleOCR-VL-1.5-8bit", "config"))
        self.assertEqual(models.reader_choice()[1], "paddleocr")
        os.environ["RAG_SEARCH_VLM_MODEL"] = "someone/other-mlx-model"
        self.assertEqual(models.vlm_selection(models.READER), ("someone/other-mlx-model", "environment"))
        mid, style, need = models.reader_choice()
        self.assertEqual((mid, style), ("someone/other-mlx-model", "instruct"))
        self.assertGreater(need, 4.0)
        with self.assertRaises(models.ModelError):
            models.set_vlm_selection(self.paths, models.READER, "not an id")
        with self.assertRaises(models.ModelError):
            models.set_vlm_selection(self.paths, "embedding", "a/b")

    def test_state_lists_readers_with_download_and_fit(self):
        os.environ["RAG_SEARCH_VLM_FREE_GB"] = "2"
        os.environ["HF_HOME"] = str(self.tmp / "no-models")       # an empty model cache, whatever this machine has
        st = models.state(self.paths)["vlm"]
        self.assertEqual(st["mode"], "auto")
        self.assertEqual(st["reader"]["active"], models.VLM_DEFAULTS[models.READER])
        rows = {r["id"]: r for r in st["reader"]["models"]}
        d = rows[models.VLM_DEFAULTS[models.READER]]
        self.assertTrue(d["active"])
        self.assertFalse(d["cached"])
        self.assertEqual(d["fit"], "tight")                     # 3.1 + 2 GB wanted, 2 free
        os.environ["RAG_SEARCH_VLM_FREE_GB"] = "30"
        self.assertEqual(models.state(self.paths)["vlm"]["reader"]["models"][0]["fit"], "ok")
        self.assertFalse(st["apple_silicon"] and not st["mlx_vlm"] and False)

    def test_a_custom_reader_stays_visible(self):
        models.set_vlm_selection(self.paths, models.READER, "someone/custom-mlx")
        rows = models.state(self.paths)["vlm"]["reader"]["models"]
        self.assertTrue([r for r in rows if r["id"] == "someone/custom-mlx" and r["custom"] and r["active"]])

    def test_switching_the_reader_off(self):
        os.environ["RAG_SEARCH_VLM"] = "off"
        self.assertEqual(vlm.mode(), "off")
        self.assertIsNone(vlm.shared())
        os.environ["RAG_SEARCH_VLM"] = ""
        self.assertEqual(vlm.mode(), "auto")


class GateResolutionTests(unittest.TestCase):
    def test_only_scans_are_judged_by_resolution(self):
        prof = {"dpi": 100, "size_px": [800, 1000]}
        self.assertEqual(gate.check_page("Some text that is long enough to read as a page.",
                                         branch_kind="digital", profile=prof)["verdict"], "ok")
        g = gate.check_page("Some text that is long enough to read as a page.", branch_kind="scan", profile=prof)
        self.assertEqual(gate.failed(g), ["low_resolution"])
        ok = gate.check_page("Some text that is long enough to read as a page.", branch_kind="scan",
                             profile={"dpi": 300, "size_px": [2480, 3508]})
        self.assertEqual(ok["verdict"], "ok")


@unittest.skipUnless(HAVE_PDF, "needs pypdfium2 and Pillow")
class VlmIndexTests(ConversionBase):
    """Whole indexing runs with the fake reader (docling replaced by the fake page reader)."""

    def setUp(self):
        super().setUp()
        os.environ["RAG_SEARCH_ROUTING"] = "pages"
        os.environ["RAG_SEARCH_VLM_BACKEND"] = FAKE
        os.environ["RAG_SEARCH_VLM_FREE_GB"] = "64"
        os.environ["RAG_TEST_VLM_PLAN"] = str(self.tmp / "plan.json")
        (self.tmp / "plan.json").write_text("{}")
        self.reader = FakeReader()
        p = mock.patch("rag_search.core.conversion.routed.default_reader", lambda: self.reader)
        p.start()
        self.addCleanup(p.stop)
        self.addCleanup(vlm.close_shared)
        self.pdf = self.paths.docs / "reports" / "r.pdf"
        self.pdf.parent.mkdir(parents=True)
        write_routing_pdf(self.pdf)

    def test_scanned_pages_and_images_are_read_by_the_reader_in_a_real_run(self):
        img = self.paths.docs / "reports" / "scan.png"
        im = Image.new("RGB", (1500, 2000), (255, 255, 255))
        ImageDraw.Draw(im).rectangle((100, 100, 900, 140), fill=(0, 0, 0))
        im.save(img, dpi=(300, 300))
        log = self.tmp / "events.jsonl"
        summary = self.run_index(stage_log=log)
        self.assertEqual(summary["indexed"], 2)
        t = trace.read_trace(self.trace_file("reports/r"))
        self.assertEqual([p["branch"] for p in t["pages"]], ["digital", "digital", "raster", "raster"])
        self.assertEqual(t["summary"]["readers"], ["fake", "vlm"])      # "fake" stands in for docling
        self.assertEqual(t["summary"]["models"], [models.VLM_DEFAULTS[models.READER]])
        ti = trace.read_trace(self.trace_file("reports/scan"))
        self.assertEqual([p["branch"] for p in ti["pages"]], ["image"])
        conv = summary["conversion"]
        self.assertEqual(conv["branches"]["raster"] + conv["branches"]["image"], 3)
        self.assertGreater(conv["tokens"], 0)
        self.assertIn("gpu_s", conv)
        evs = [json.loads(x) for x in log.read_text().splitlines() if '"event": "page"' in x]
        self.assertEqual(len(evs), 5)

    def test_an_image_the_reader_cannot_read_is_converted_by_docling(self):
        (self.tmp / "plan.json").write_text(json.dumps({"load_error": "no weights"}))
        img = self.paths.docs / "reports" / "scan.png"
        im = Image.new("RGB", (1500, 2000), (255, 255, 255))
        ImageDraw.Draw(im).rectangle((100, 100, 900, 300), fill=(0, 0, 0))
        im.save(img)

        def fake_convert(src, out, **kw):
            Path(out).write_text("<!-- page 1 -->\n\nDocling read this picture of a receipt for tea and sugar.",
                                 encoding="utf-8")
            return {"pages": 1}

        with mock.patch("rag_search.core.docling_convert.convert_file", fake_convert):
            summary = self.run_index()
        self.assertEqual(summary["indexed"], 2)
        meta = self.meta("reports/scan")
        self.assertIn("document reader not used", meta["conversion"]["note"])

    def test_switching_the_reader_off_keeps_scanned_pages_on_docling(self):
        os.environ["RAG_SEARCH_VLM"] = "off"
        self.run_index()
        t = trace.read_trace(self.trace_file("reports/r"))
        self.assertEqual([p["branch"] for p in t["pages"]], ["digital", "digital", "fallback", "fallback"])
        self.assertEqual(t["summary"]["readers"], ["fake"])


@unittest.skipUnless(HAVE_PDF, "needs pypdfium2 and Pillow")
class BenchEngineTests(VlmBase):
    def setUp(self):
        super().setUp()
        self.pdf = self.tmp / "r.pdf"
        write_routing_pdf(self.pdf)

    def test_vlm_engine_reads_the_wanted_pages_with_the_reader_only(self):
        from rag_search.core.conversion import engines

        eng = engines.resolve("vlm")
        res = eng.read_pages(self.pdf, [3])
        self.assertIn("Scanned statement page", res["pages"][3])
        self.assertEqual((res["pages_read"], res["page_count"]), (1, 1))
        self.assertGreater(res["page_s"][3], 0)
        d = eng.describe()
        self.assertEqual((d["name"], d["model"]), ("vlm", models.VLM_DEFAULTS[models.READER]))
        self.assertGreater(d["tokens"], 0)
        self.plan(error_on=[2])                                  # the second call overall
        with self.assertRaises(engines.EngineError):             # no silent fallback in a measurement
            eng.read_pages(self.pdf, [3])
        eng.close()
        self.assertIn("vlm", engines.available())
        self.assertIn("routed", engines.available())

    def test_vlm_engine_refuses_when_the_reader_is_off(self):
        from rag_search.core.conversion import engines

        os.environ["RAG_SEARCH_VLM"] = "off"
        with self.assertRaises(engines.EngineError):
            engines.resolve("vlm")

    def test_routed_engine_runs_the_production_path_without_the_cache(self):
        from rag_search.core.conversion import engines

        docling = FakeReader()
        with mock.patch("rag_search.core.conversion.routed.default_reader", lambda: docling):
            eng = engines.resolve("routed")
            res = eng.read_pages(self.pdf, [1, 3])
            self.assertIn("read as digital", res["pages"][1])
            self.assertIn("Scanned statement page", res["pages"][3])
            self.assertEqual(res["page_count"], 4)
            eng.read_pages(self.pdf, [1])
            self.assertEqual(len([c for c in docling.calls if c[2] == "digital"]), 2)   # nothing cached
            eng.close()

    def test_a_benchmark_run_with_the_vlm_engine_records_model_and_closes_the_reader(self):
        from rag_search.core.conversion import bench

        shutil_dest = self.paths.docs / "c" / "r.pdf"
        shutil_dest.parent.mkdir(parents=True)
        shutil_dest.write_bytes(self.pdf.read_bytes())
        gold = bench.gold_dir(self.paths, "g")
        gold.mkdir(parents=True)
        (gold / "gold.json").write_text(json.dumps({
            "format": bench.GOLD_FORMAT if hasattr(bench, "GOLD_FORMAT") else 1, "name": "g",
            "pages": [{"id": "r-3", "rel": "c/r.pdf", "page": 3, "class": "scan-text", "verified": True,
                       "truth": "Scanned statement page", "queries": ["Scanned statement"]}]}))
        rec = bench.run_bench(self.paths, "g", engine="vlm")
        self.assertEqual(rec["engine_info"]["model"], models.VLM_DEFAULTS[models.READER])
        self.assertEqual(rec["summary"]["all"]["pages"], 1)
        self.assertEqual(rec["summary"]["all"]["query_hit"], 1.0)
        self.assertFalse(vlm._SHARED)                           # the reader process was stopped
        self.assertGreater(rec["cost"]["peak_mb"], 0)


class VlmInterfaceTests(VlmBase):
    """CLI, HTTP, live view and estimate."""

    def test_models_command_lists_readers_and_chooses_one(self):
        from tests.portable.test_cli_api import run

        rc, out, err = run("models")
        self.assertEqual(rc, 0, err)
        self.assertIn("DOCUMENT READER", out)
        self.assertIn("mlx-community/Qwen3-VL-4B-Instruct-4bit", out)
        rc, out, err = run("models", "reader", "mlx-community/PaddleOCR-VL-1.5-8bit")
        self.assertEqual(rc, 0, err)
        self.assertIn("not downloaded yet", out)
        rc, out, _ = run("models", "reader", "--json")
        self.assertEqual(json.loads(out)["model"], "mlx-community/PaddleOCR-VL-1.5-8bit")
        rc, out, err = run("models", "reader", "not an id")
        self.assertNotEqual(rc, 0)
        rc, out, _ = run("models", "--json")
        self.assertEqual(json.loads(out)["vlm"]["reader"]["active"], "mlx-community/PaddleOCR-VL-1.5-8bit")

    def test_dashboard_chooses_the_reader_and_lists_it(self):
        from tests.portable.test_ui import Dash

        d = Dash(self.paths)
        self.addCleanup(d.close)
        st, js, _, _ = d.req("GET", "/api/models")
        self.assertEqual(st, 200)
        self.assertEqual(js["vlm"]["mode"], "auto")
        st, js, _, _ = d.req("POST", "/api/models/reader", {"model": "mlx-community/PaddleOCR-VL-1.5-bf16"})
        self.assertEqual((st, js["ok"]), (200, True))
        st, js, _, _ = d.req("GET", "/api/models")
        self.assertEqual(js["vlm"]["reader"]["active"], "mlx-community/PaddleOCR-VL-1.5-bf16")
        st, js, _, _ = d.req("POST", "/api/models/reader", {"model": "bad id"})
        self.assertEqual(st, 400)
        ro = Dash(self.paths, read_only=True)
        self.addCleanup(ro.close)
        st, _, _, _ = ro.req("POST", "/api/models/reader", {"model": "mlx-community/PaddleOCR-VL-1.5-8bit"})
        self.assertIn(st, (403, 405))

    def test_the_vlm_setting_is_a_validated_tunable(self):
        from rag_search import spec

        t = [x for x in spec.TUNABLES if x.key == "vlm"][0]
        self.assertEqual((t.section, t.env, tuple(t.choices)), ("indexer", "RAG_SEARCH_VLM", ("auto", "off")))

    def test_live_view_counts_tokens(self):
        from rag_search.core.conversion import runview

        log = self.paths.jobs / "jv.events.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        now = time.time()
        rows = [{"ts": now - 20 + i, "event": "page", "file": "a/x.pdf", "pid": 7, "page": i, "of": 3,
                 "branch": "raster", "outcome": "pass", "cache": "miss", "tokens": 100, "gpu_s": 2.0,
                 "model": "m"} for i in range(1, 4)]
        log.write_text("".join(json.dumps(r) + "\n" for r in rows))
        v = runview.live_view(runview._read_new(log), True)
        self.assertEqual((v["tokens"], v["gpu_s"], v["tokens_per_s"]), (300, 6.0, 50.0))

    @unittest.skipUnless(HAVE_PDF, "needs pypdfium2 and Pillow")
    def test_estimate_says_whether_the_reader_can_run(self):
        from rag_search.core.conversion import estimate

        src = self.paths.docs / "c" / "s.pdf"
        src.parent.mkdir(parents=True)
        write_routing_pdf(src)
        e = estimate.estimate(self.paths, [src])
        self.assertEqual(e["planned_vlm"]["pages"], 2)        # the scan and the blank page
        self.assertTrue(e["planned_vlm"]["reader"]["usable"])
        os.environ["RAG_SEARCH_VLM"] = "off"
        e = estimate.estimate(self.paths, [src])
        self.assertFalse(e["planned_vlm"]["reader"]["usable"])
        self.assertIn("switched off", e["planned_vlm"]["reader"]["why"])
        os.environ.pop("RAG_SEARCH_VLM")
        os.environ["RAG_SEARCH_VLM_BACKEND"] = "mlx"
        no_real_reader(self)
        vlm.close_shared()
        e = estimate.estimate(self.paths, [src])
        self.assertFalse(e["planned_vlm"]["reader"]["usable"])     # not an Apple Silicon Mac


if __name__ == "__main__":
    unittest.main()


class ReaderReadinessTests(VlmBase):
    """The Models tab never says "in use" for a model that cannot be used (the QWEN3-VL screenshot)."""

    def vlm(self, *, apple=True, installed=("mlx_vlm", "ocrmac", "pillow_heif"), cached=()):
        def cache(mid):
            ok = mid in cached
            return {"cached": ok, "partial": False, "bytes": 1 if ok else 0}

        with mock.patch.object(models, "_apple_silicon", return_value=apple), \
                mock.patch.object(models, "_installed", side_effect=lambda m: m in installed), \
                mock.patch.object(models, "cache_state", side_effect=cache):
            return models.vlm_state(self.paths)

    def chosen(self, v, kind="reader"):
        return next(r for r in v[kind]["models"] if r["active"])

    def test_chosen_but_not_downloaded_is_not_in_use(self):
        v = self.vlm()
        self.assertEqual(self.chosen(v)["state"], "selected_download")
        self.assertEqual(v["reading"]["by"], "docling")
        self.assertIn("not downloaded", v["reading"]["text"])

    def test_in_use_needs_weights_runtime_and_a_mac(self):
        mid = models.VLM_DEFAULTS[models.READER]
        v = self.vlm(cached=(mid,))
        self.assertEqual(self.chosen(v)["state"], "in_use")
        self.assertEqual(v["reading"]["by"], "reader")
        self.assertEqual(self.chosen(self.vlm(cached=(mid,), installed=()))["state"], "selected_runtime")
        self.assertEqual(self.chosen(self.vlm(cached=(mid,), apple=False))["state"], "selected_platform")
        os.environ["RAG_SEARCH_VLM"] = "off"
        self.assertEqual(self.chosen(self.vlm(cached=(mid,)))["state"], "selected_off")

    def test_the_checklist_names_each_fix(self):
        v = self.vlm(installed=("pillow_heif",))
        checks = {c["id"]: c for c in v["checks"]}
        self.assertEqual(checks["runtime"]["fix"], "install_runtime")
        self.assertEqual(checks["ocrmac"]["fix"], "install_runtime")
        self.assertTrue(checks["weights_reader"]["fix"].startswith("download:"))
        self.assertTrue(checks["runtime"]["required"] and not checks["ocrmac"]["required"])
        self.assertEqual(self.vlm(apple=False)["checks"][2]["fix"], "")        # nothing to install on a PC

    def test_embedding_and_reranker_rows_have_the_same_states(self):
        st = models.state(self.paths)
        for kind in ("embedding", "reranker"):
            row = next(r for r in st[kind]["models"] if r["active"])
            self.assertEqual(row["state"], "in_use" if row["cached"] else "selected_download")
            self.assertTrue(all(r["state"] == "available" for r in st[kind]["models"] if not r["active"]))

    def test_runtime_requirements_come_from_the_package_metadata(self):
        reqs = models.runtime_requirements()
        self.assertTrue(any(r.startswith("mlx-vlm") for r in reqs))
        self.assertTrue(all(";" not in r for r in reqs))
        with mock.patch.object(models.md, "requires", return_value=["torch>=2", 'x>=1; extra == "mac-vlm"']):
            self.assertEqual(models.runtime_requirements(), ["x>=1"])

    def test_runtime_install_task(self):
        from rag_search import model_tasks as mt

        lines = []

        def fake_stream(cmd, task):
            lines.append(cmd)
            task.log("Installed mlx-vlm")
            return 0

        with mock.patch.object(models, "_apple_silicon", return_value=True), \
                mock.patch.object(mt, "_stream", side_effect=fake_stream), \
                mock.patch.object(models, "_installed", return_value=True):
            rec = mt.run(self.paths, {"op": "runtime"})
        self.assertEqual(rec["status"], "succeeded")
        self.assertIn("mlx-vlm", rec["result"]["installed"])
        self.assertIn("install", lines[0])
        with mock.patch.object(models, "_apple_silicon", return_value=True), \
                mock.patch.object(mt, "_stream", return_value=1):
            self.assertEqual(mt.run(self.paths, {"op": "runtime"})["status"], "failed")
        with mock.patch.object(models, "_apple_silicon", return_value=False):
            rec = mt.run(self.paths, {"op": "runtime"})
        self.assertEqual(rec["status"], "failed")
        self.assertIn("Apple Silicon", rec["error"])

    def test_cli_runtime_status_and_dashboard_route(self):
        from tests.portable.test_cli_api import run
        from tests.portable.test_ui import Dash

        rc, out, err = run("models", "runtime", "--json")
        self.assertEqual(rc, 0, err)
        self.assertIn("packages", json.loads(out))
        d = Dash(self.paths)
        self.addCleanup(d.close)
        with mock.patch("rag_search.model_tasks.start_detached", return_value={"id": "t"}) as sd:
            st, js, _, _ = d.req("POST", "/api/models/runtime", {})
        self.assertEqual((st, js["ok"]), (200, True))
        self.assertEqual(sd.call_args[0][1]["op"], "runtime")


class CleanReplyTests(unittest.TestCase):
    def test_a_fence_around_the_whole_answer_is_removed(self):
        from rag_search.core.conversion.vlm import clean_reply

        self.assertEqual(clean_reply("```markdown\n# Title\n\ntext\n```"), "# Title\n\ntext")
        self.assertEqual(clean_reply("  ```\nplain\n```  \n"), "plain")
        self.assertEqual(clean_reply("```md\nA | B\n```\n"), "A | B")
        self.assertEqual(clean_reply("```markdown\ncut off at the token limit"), "cut off at the token limit")

    def test_code_inside_the_page_and_ordinary_text_are_left_alone(self):
        from rag_search.core.conversion.vlm import clean_reply

        keep = "Intro\n\n```python\nprint(1)\n```\n\nOutro"
        self.assertEqual(clean_reply(keep), keep)
        self.assertEqual(clean_reply(None), "")
        both = "```\nA\n```\n\nB\n\n```py\nx\n```"       # a bare fence that closes early: not a wrapper
        self.assertEqual(clean_reply(both), both)
