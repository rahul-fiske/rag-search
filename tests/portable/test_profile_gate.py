"""The profiler and the gate on the cases a review of 0.9.28 found them wrong or blind on: pictures on a rotated page,
pictures with a transparent ground, the frames of a photograph, the text layer that is not garbled (identifiers, bullets,
zero-width spaces), visible text over a background picture, the skew beyond the search range.  Every document is made
here with pypdfium2 and Pillow."""

from __future__ import annotations

import ctypes
import unittest
from pathlib import Path

from tests.helpers import TempHome

from rag_search.core import docling_convert as dc
from rag_search.core.conversion import gate, profiler, residue, routed, router, scanfacts, vlm

try:
    import numpy as np
    import pypdfium2 as pdfium
    import pypdfium2.raw as raw
    from PIL import Image, ImageDraw
    HAVE = True
except ImportError:                                           # pragma: no cover - tier A has them
    HAVE = False

LINES = [f"Line {k} of the statement of account number 4096" for k in range(6)]


def add_text(pdf, page, lines=LINES, mode=None, x=40, top=520, font=b"Helvetica", size=12, leading=30):
    font = raw.FPDFText_LoadStandardFont(pdf.raw, font)
    for k, line in enumerate(lines):
        obj = raw.FPDFPageObj_CreateTextObj(pdf.raw, font, size)
        text = ctypes.create_string_buffer((line + "\x00").encode("utf-16-le"))
        raw.FPDFText_SetText(obj, ctypes.cast(text, raw.FPDF_WIDESTRING))
        raw.FPDFPageObj_Transform(obj, 1, 0, 0, 1, x, top - leading * k)
        if mode is not None:
            raw.FPDFTextObj_SetTextRenderMode(obj, mode)
        raw.FPDFPage_InsertObject(page.raw, obj)


def add_picture(pdf, page, box, colour=(200, 30, 30)):
    """A picture in PDF coordinates (left, bottom, width, height)."""
    img = pdfium.PdfImage.new(pdf)
    img.set_bitmap(pdfium.PdfBitmap.from_pil(Image.new("RGB", (120, 150), colour)))
    img.set_matrix(pdfium.PdfMatrix().scale(box[2], box[3]).translate(box[0], box[1]))
    page.insert_obj(img)


def writing(size=(600, 800), ink=0, paper=255, mode="L"):
    im = Image.new(mode, size, paper)
    d = ImageDraw.Draw(im)
    for y in range(60, size[1] - 60, 22):
        d.rectangle((50, y, size[0] - 50, y + 7), fill=ink)
    return im


@unittest.skipUnless(HAVE, "needs pypdfium2, Pillow and numpy")
class PictureBoxTests(TempHome):
    """A large picture on a text page is cut out and read by the document reader: the box has to be where the picture
    is seen."""

    def make(self, name, rotation=0, box=None):
        pdf = pdfium.PdfDocument.new()
        page = pdf.new_page(400, 600)
        add_picture(pdf, page, (20, 30, 240, 300))               # lower left of the page as it is written
        add_text(pdf, page, x=40, top=560)
        page.gen_content()
        if rotation:
            page.set_rotation(rotation)
        if box:
            page.set_mediabox(*box)
            page.set_cropbox(*box)
        path = self.tmp / name
        pdf.save(str(path))
        pdf.close()
        return path

    def seen(self, path):
        """Where the red picture is in the page as the readers get it: (left, top, right, bottom) fractions."""
        png = self.tmp / (path.name + ".png")
        vlm.render_pdf_page(path, 1, png)
        a = np.asarray(Image.open(png).convert("RGB")).astype(int)
        ys, xs = np.nonzero((a[..., 0] > 150) & (a[..., 1] < 90))
        h, w = a.shape[:2]
        return [xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h]

    def test_the_box_of_a_picture_follows_the_rotation_and_the_origin_of_the_page(self):
        for name, rotation, box in (("r0.pdf", 0, None), ("r90.pdf", 90, None), ("r180.pdf", 180, None),
                                    ("r270.pdf", 270, None), ("shifted.pdf", 0, (-100, -50, 300, 550))):
            with self.subTest(name):
                path = self.make(name, rotation, box)
                page = profiler.profile_file(path)["pages"][0]
                self.assertEqual(page["rotation"], rotation)
                (got,) = page["big_pics"]
                for a, b in zip(got, self.seen(path)):
                    self.assertAlmostEqual(a, b, delta=0.01, msg=f"{got} against {self.seen(path)}")
                self.assertAlmostEqual(page["image_cover"], 0.3, delta=0.01)

    def test_a_picture_that_hangs_over_the_edge_counts_for_the_part_on_the_page(self):
        pdf = pdfium.PdfDocument.new()
        page = pdf.new_page(400, 600)
        add_picture(pdf, page, (-300, 0, 400, 600))              # three quarters of it are left of the page
        add_text(pdf, page)
        page.gen_content()
        path = self.tmp / "edge.pdf"
        pdf.save(str(path))
        pdf.close()
        prof = profiler.profile_file(path)["pages"][0]
        self.assertAlmostEqual(prof["image_cover"], 0.25, delta=0.01)   # not 1.0: the page is not "a full-page picture"
        self.assertFalse(prof["hidden_ocr_layer"])

    def test_the_text_of_a_rotated_page_is_not_taken_for_residue(self):
        from unittest import mock

        block = [f"THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG {k:02d}" for k in range(22)]     # a dense block of bold text
        for rotation in (0, 90, 270):
            pdf = pdfium.PdfDocument.new()
            page = pdf.new_page(400, 600)
            add_text(pdf, page, lines=block, x=30, top=560, font=b"Helvetica-Bold", size=13, leading=14)
            page.gen_content()
            if rotation:
                page.set_rotation(rotation)
            path = self.tmp / f"text{rotation}.pdf"
            pdf.save(str(path))
            pdf.close()
            self.assertEqual(residue.page_regions(path, 1), [], f"rotation {rotation}")      # the text explains all the ink
            if rotation:                    # by the page's own coordinates the text mask lies beside the text
                with mock.patch.object(profiler, "page_box", return_value=None):
                    self.assertTrue(residue.page_regions(path, 1), f"rotation {rotation}")


@unittest.skipUnless(HAVE, "needs pypdfium2, Pillow and numpy")
class HiddenLayerTests(TempHome):
    def make(self, name, mode, picture=True):
        pdf = pdfium.PdfDocument.new()
        page = pdf.new_page(400, 600)
        if picture:
            add_picture(pdf, page, (0, 0, 400, 600), colour=(240, 240, 235))
        add_text(pdf, page, mode=mode)
        page.gen_content()
        path = self.tmp / name
        pdf.save(str(path))
        pdf.close()
        return profiler.profile_file(path)["pages"][0]

    def test_only_invisible_text_over_a_picture_is_a_scanners_layer(self):
        hidden = self.make("scan.pdf", 3)                        # text render mode 3: what a scanner's OCR writes
        self.assertTrue(hidden["hidden_ocr_layer"])
        self.assertIn("scanner's hidden OCR layer", router.decide("pdf", hidden)[1])
        letterhead = self.make("statement.pdf", None)            # visible text on a full-page background picture
        self.assertEqual((letterhead["image_cover"], letterhead["hidden_ocr_layer"]), (1.0, False))
        self.assertEqual(router.decide("pdf", letterhead), ("digital", f"good text layer ({letterhead['chars']} characters)"))
        self.assertFalse(self.make("plain.pdf", None, picture=False)["hidden_ocr_layer"])

    def test_a_scanners_layer_is_checked_as_an_ocr_reading_and_other_text_is_not(self):
        junk = ("Tbe qvick brwn fx jmps ovr tbe lzy dg. " * 2 + "Xqzt wrtk plmn bvcx zzrt qwrt mnbv lkjh gfds. " * 6)
        table = "| Date | Amount |\n|---|---|\n| 1 Jan | 105 |\n| 2 Jan | 1O5 |\n| 3 Jan | 220 |\n| 4 Jan | 310 |\n"
        hidden, plain = {"chars": 400, "hidden_ocr_layer": True}, {"chars": 400, "hidden_ocr_layer": False}
        self.assertEqual(gate.failed(gate.check_page(junk, branch_kind="digital", profile=hidden)), ["plausibility"])
        self.assertEqual(gate.failed(gate.check_page(junk, branch_kind="digital", profile=plain)), [])
        self.assertIn("column_types", gate.failed(gate.check_page("Statement of account\n\n" + table, branch_kind="digital", profile=hidden)))
        good = "The seller shall deliver the property to the buyer on the date named in the schedule. " * 4
        self.assertEqual(gate.check_page(good, branch_kind="digital", profile=hidden)["verdict"], "ok")
        self.assertTrue({"plausibility", "column_types"} <= set(routed.UP_DIGITAL))     # and can send the page to the reader


@unittest.skipUnless(HAVE, "needs Pillow and numpy")
class ImageFileTests(TempHome):
    def ground(self, path):
        png = self.tmp / (path.name + ".seen.png")
        vlm.render_image_frame(path, 1, png)
        a = np.asarray(Image.open(png).convert("L")).astype(int)
        return int(np.median(a)), float((a < 100).mean()), float((a > 180).mean())

    def test_dark_writing_on_a_transparent_ground_is_put_on_paper(self):
        ink = writing()
        rgba = Image.new("RGBA", ink.size, (0, 0, 0, 0))
        rgba.paste(Image.new("RGBA", ink.size, (0, 0, 0, 255)), mask=Image.eval(ink, lambda v: 255 - v))
        cases = {"rgba.png": rgba, "la.png": rgba.convert("LA")}
        for name, im in cases.items():
            im.save(self.tmp / name)
        rgba.convert("P").save(self.tmp / "palette.png", transparency=0)
        for name in (*cases, "palette.png"):
            with self.subTest(name):
                prof = profiler.profile_file(self.tmp / name)["pages"][0]
                self.assertTrue(0.05 < prof["ink"] < 0.5, prof)             # it was 1.0: the whole ground counted as ink
                median, dark, _light = self.ground(self.tmp / name)
                self.assertEqual(median, 255)                              # what a reader gets: white paper ...
                self.assertTrue(0.05 < dark < 0.5)                         # ... with the writing on it (it was all black)

    def test_light_writing_on_a_transparent_ground_is_put_on_a_dark_one(self):
        ink = writing()
        rgba = Image.new("RGBA", ink.size, (255, 255, 255, 0))
        rgba.paste(Image.new("RGBA", ink.size, (250, 250, 250, 255)), mask=Image.eval(ink, lambda v: 255 - v))
        rgba.save(self.tmp / "white.png")
        median, _dark, light = self.ground(self.tmp / "white.png")
        self.assertEqual(median, 0)
        self.assertTrue(0.05 < light < 0.5)
        plan = routed.plan_pages(profiler.profile_file(self.tmp / "white.png"))
        self.assertFalse(plan[0]["blank"])

    def test_an_opaque_picture_is_left_as_it_is(self):
        im = writing().convert("RGB")
        self.assertIs(profiler.opaque(im), im)
        full = im.convert("RGBA")                                          # an alpha channel that hides nothing
        self.assertEqual(profiler.opaque(full).tobytes(), im.tobytes())

    def test_only_a_tiff_has_pages(self):
        page = writing().convert("RGB")
        page.save(self.tmp / "scan.tif", save_all=True, append_images=[page.rotate(180)])
        page.save(self.tmp / "anim.png", save_all=True, append_images=[page.rotate(180)])
        page.save(self.tmp / "phone.jpg", format="MPO", save_all=True, append_images=[page.resize((150, 200))])
        with Image.open(self.tmp / "phone.jpg") as im:
            self.assertEqual((im.format, im.n_frames), ("MPO", 2))         # a main picture and a second one that is no page
        for name, pages in (("scan.tif", 2), ("anim.png", 1), ("phone.jpg", 1)):
            prof = profiler.profile_file(self.tmp / name)
            self.assertEqual((len(prof["pages"]), prof["page_count"], prof["truncated"]), (pages, pages, False), name)

    def test_the_coarser_direction_of_a_fax_is_its_resolution(self):
        writing().save(self.tmp / "fax.tif", dpi=(204, 98))
        prof = profiler.profile_file(self.tmp / "fax.tif")["pages"][0]
        self.assertEqual(prof["dpi"], 98)
        self.assertEqual(gate.failed(gate.check_page("text " * 20, branch_kind="scan", profile=prof)), ["low_resolution"])


class TextLayerTests(unittest.TestCase):
    """``page_text_ok`` decides whether a page keeps its own text layer or is read as an image."""

    PROSE = "The adapter returns the status of the mailbox command in the completion queue entry. " * 6

    def test_identifiers_and_values_are_not_words_run_together(self):
        uuid = "3f2a9c1e-77b4-4d21-9a6e-0c5d8e1f4b7a "
        ident = "sli4_config_special_wcqe_mailbox_command_status:lpfc_mbx_cmd_read_config "
        row = "000000000012abcd,00000000000014,00000000000002 "
        for extra in (uuid * 12, ident * 12, row * 12):
            self.assertTrue(dc.page_text_ok(self.PROSE + extra), extra[:30])
        glued = "Theadapterreturnsthestatusofthemailboxcommandinthecompletionqueueentry "
        self.assertFalse(dc.page_text_ok(self.PROSE + glued * 12))          # the spaces of real words are gone

    def test_a_sentence_in_a_script_without_spaces_is_not_run_together(self):
        thai = "เครื่องพิมพ์นี้รองรับการพิมพ์สองหน้าโดยอัตโนมัติและการสแกนเอกสารหลายหน้า " * 8
        chinese = "本设备支持自动双面打印和多页文档扫描功能并且可以通过无线网络连接到计算机 " * 8
        self.assertTrue(dc.page_text_ok(thai))
        self.assertTrue(dc.page_text_ok(chinese))

    def test_bullets_and_icons_from_a_symbol_font_are_not_damage(self):
        items = "".join(f" Item {k}: connect the cable and switch the unit on.\n" for k in range(12))
        self.assertTrue(dc.page_text_ok(items))                              # 12 private-use bullets among 600 characters
        self.assertTrue(dc.page_text_ok(items.replace("", "\x95")))    # the bullet of a Windows code page
        broken = self.PROSE.replace("fi", "").replace("st", "")  # ligatures the font map lost, inside words
        self.assertFalse(dc.page_text_ok(broken))
        self.assertFalse(dc.page_text_ok(" ".join("" for _ in range(200))))       # a page in a symbol font
        self.assertFalse(dc.page_text_ok(self.PROSE + " �����" * 12))  # runs of unknown glyphs

    def test_invisible_format_characters_are_part_of_the_text(self):
        zw = self.PROSE.replace(" ", " ​")                              # a word-break opportunity after every space
        self.assertGreater(zw.count("​") / len(zw), 0.1)
        self.assertTrue(dc.page_text_ok(zw))
        hindi = "यह मशीन दो‍तरफ़ा छपाई और कई पन्नों की स्कैनिंग का समर्थन कर‌ती है। " * 6
        self.assertTrue(dc.page_text_ok(hindi))
        self.assertFalse(dc.page_text_ok("​ ​"))

    def test_the_script_of_a_garbled_layer_is_not_held_against_the_reading(self):
        read = "The seller shall deliver the property to the buyer on the date named in the schedule. " * 3
        garbled = {"chars": 300, "text_ok": False, "script": "Greek"}        # a broken font map that came out as Greek letters
        self.assertEqual(gate.failed(gate.check_page(read, branch_kind="scan", profile=garbled)), [])
        sound = {"chars": 300, "text_ok": True, "script": "Devanagari"}      # a layer that can be trusted still counts
        self.assertEqual(gate.failed(gate.check_page(read, branch_kind="digital", profile=sound)), ["script"])


@unittest.skipUnless(HAVE, "needs Pillow and numpy")
class SkewTests(unittest.TestCase):
    def test_the_search_reaches_beyond_what_is_straightened(self):
        self.assertGreater(scanfacts.SKEW_RANGE, router.THRESHOLDS["max_deskew"])
        page = writing((660, 900)).convert("RGB")
        for true in (0, 3, 6, 8):
            im = page.rotate(true, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))
            self.assertAlmostEqual(abs(scanfacts.facts_of_image(im)["skew"]), true, delta=0.6, msg=f"{true} degrees")
        far = page.rotate(14, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))
        facts = scanfacts.facts_of_image(far)
        runway, why = router.decide_scan(facts, {"dpi": 300}, {"ocr": True, "deskew": True})
        self.assertEqual(runway, "d", why)                                   # not a page for the straighten-and-OCR lane


class MessageTests(unittest.TestCase):
    def test_a_page_from_the_cache_was_read_by_the_reader_recorded_with_it(self):
        cached = {1: {"md": "<!-- image -->", "cache": "hit", "via": "cache", "reader": {"tool": "vlm", "model": "m"}}}
        msg = routed.no_text_message(Path("photo.jpg"), cached)
        self.assertIn("the document reader read the pages and found no text", msg)
        self.assertNotIn("did not read any page", msg)                       # which also kept it from being remembered
        from rag_search.core import indexer
        self.assertEqual(indexer.lasting_reason("no_text", None, msg), "no_text")
        docling = {1: {"md": "", "cache": "hit", "via": "cache", "reader": {"tool": "docling"}}}
        self.assertIn("did not read any page", routed.no_text_message(Path("photo.jpg"), docling))


if __name__ == "__main__":
    unittest.main()
