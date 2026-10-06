"""Lane b with the real tools: a clean page of text read by docling's OCR, and a skewed one straightened and read by
Tesseract, through the router and the gate.  The document reader is a stub that is only there to switch routing on (it
must not be called for a clean page); what is tested is the framework's wiring with real OCR, not the OCR's quality."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.real import missing_tools

from rag_search.core.conversion import pagecache, profiler, routed

LINES = ["The storage array replicates every volume to the second site.",
         "Customers receive a statement of the account balance each month.",
         "The payment schedule follows the agreement signed in March.",
         "Every pool has a quota and a policy that sets its limits.",
         "A volume can be moved to another pool without any downtime.",
         "The administrator receives an alert when a quota is reached.",
         "Backups are written to tape once a week and kept for a year.",
         "All changes to the configuration are recorded in a log file.",
         "Users can ask for a larger quota through the service desk.",
         "The service desk answers every request within two working days."] * 2


class StubVlm:
    """A document reader that is only there to switch routing on: it answers any page with a short text."""

    id, model, dead = "stub", "stub/model", ""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def usable(self):
        return True

    def check(self):
        pass

    def close(self):
        pass

    def read(self, src, first, last, mode):
        self.calls.append((first, last))
        text = "The document reader's text for this page, a stand-in that the test never judges."
        return {"pages": {n: text for n in range(first, last + 1)},
                "stats": {n: {"read_s": 1.0, "tokens": 10} for n in range(first, last + 1)}, "failed": {}}


def write_page(path: Path, angle: float = 0.0) -> None:
    from PIL import Image, ImageDraw, ImageFont

    im = Image.new("RGB", (1654, 2339), "white")                    # A4 at 200 dpi
    d = ImageDraw.Draw(im)
    font = ImageFont.load_default(size=34)
    for i, line in enumerate(LINES):
        d.text((140, 160 + i * 90), line, fill="black", font=font)
    if angle:
        im = im.rotate(angle, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
    im.save(path, "PDF", resolution=200.0)


@unittest.skipIf(missing_tools(), missing_tools())
class RealLaneBTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="rag-lanes-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        os.environ["RAG_SEARCH_OCR_FIRST"] = "auto"
        self.addCleanup(os.environ.pop, "RAG_SEARCH_OCR_FIRST", None)

    def convert(self, pdf: Path) -> dict:
        prof = profiler.profile_file(pdf)
        v = StubVlm()
        res = routed.convert_pdf(pdf, self.tmp / "o.md", prof, cache=pagecache.PageCache(self.tmp / "ws"), scan_reader=v)
        return {"rec": res["records"][0], "vlm_calls": v.calls, "md": (self.tmp / "o.md").read_text()}

    def test_a_clean_page_is_routed_and_read_with_real_ocr(self):
        pdf = self.tmp / "clean.pdf"
        write_page(pdf)
        got = self.convert(pdf)
        rt = got["rec"]["route"]
        self.assertEqual((rt["runway"], rt["final"], rt["engine"]), ("b", "b", "docling"), rt)
        self.assertEqual(got["vlm_calls"], [])                       # the document reader was not needed
        self.assertIn("storage array", got["md"].replace("\n", " "))
        self.assertEqual(got["rec"]["outcome"], "pass")

    @unittest.skipUnless(shutil.which("tesseract"), "needs tesseract")
    def test_a_skewed_page_is_straightened_and_read_by_tesseract(self):
        pdf = self.tmp / "skew.pdf"
        write_page(pdf, angle=4.0)
        with mock.patch("rag_search.core.conversion.tesseract.languages", return_value="eng"):
            got = self.convert(pdf)
        rt = got["rec"]["route"]
        self.assertEqual((rt["runway"], rt["final"], rt["engine"]), ("b", "b", "tesseract"), rt)
        self.assertEqual(got["vlm_calls"], [])
        self.assertIn("storage array", got["md"].replace("\n", " "))
        self.assertIn("straightening", got["rec"]["note"])

if __name__ == "__main__":
    unittest.main()
