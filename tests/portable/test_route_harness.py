"""Synthetic scans, page-image facts, the scan router and the routing harness (plan steps R0, R1, R3)."""

from __future__ import annotations

import unittest

try:
    import numpy  # noqa: F401
    import PIL  # noqa: F401
    HAVE = True
except ImportError:
    HAVE = False


@unittest.skipUnless(HAVE, "needs numpy and Pillow")
class ScanRoutingTests(unittest.TestCase):
    def facts(self, **kw):
        from PIL import Image

        from rag_search.core.conversion import scanfacts, synth

        kw.setdefault("dpi", 200)
        im = synth.make_page(synth.text_lines(1), **kw)
        small = im.resize((round(im.width * 100 / kw["dpi"]), round(im.height * 100 / kw["dpi"])), Image.LANCZOS)
        return scanfacts.facts_of_image(small)

    def decide(self, **kw):
        from rag_search.core.conversion import router

        return router.decide_scan(self.facts(**kw), {"dpi": kw.get("dpi", 200)})

    def test_a_clean_page_goes_to_ocr_and_says_why(self):
        runway, why = self.decide()
        self.assertEqual(runway, "b")
        self.assertIn("clean print", why[0])

    def test_damage_sends_a_page_to_the_document_reader(self):
        for kw, word in (({"skew": 4.0}, "skew"), ({"table": True}, "ruled"), ({"photo": True}, "texture"),
                         ({"noise": 0.02}, "speckle"), ({"dpi": 100}, "dpi")):
            runway, why = self.decide(**kw)
            self.assertEqual(runway, "d", kw)
            self.assertTrue(any(word in x for x in why), (kw, why))

    def test_no_ocr_engine_or_no_facts_means_the_document_reader(self):
        from rag_search.core.conversion import router

        self.assertEqual(router.decide_scan(self.facts(), {"dpi": 200}, {"ocr": False})[0], "d")
        self.assertEqual(router.decide_scan(None, {"dpi": 200})[0], "d")
        self.assertEqual(router.decide_scan(self.facts(), {})[0], "d")           # resolution unknown

    def test_synthetic_pdf_has_no_text_layer(self):
        import tempfile
        from pathlib import Path

        from rag_search.core.conversion import layer, scanfacts, synth

        with tempfile.TemporaryDirectory() as t:
            pdf = Path(t) / "s.pdf"
            synth.write_pdf(pdf, [synth.make_page(synth.text_lines(2), dpi=100)], 100)
            self.assertFalse((layer.LayerSource(pdf).page(1) or "").strip())
            self.assertEqual(set(scanfacts.pages_facts(pdf, [1, 5])) , {1, 5})
            self.assertIsNotNone(scanfacts.page_facts(pdf, 1))
            self.assertIsNone(scanfacts.page_facts(Path(t) / "none.pdf", 1))


@unittest.skipUnless(HAVE, "needs numpy and Pillow")
class HarnessTests(unittest.TestCase):
    def test_the_report_classifies_every_page(self):
        from rag_search.core.conversion import routeharness, synth

        def ocr(pdf):                                   # a perfect OCR on clean pages, and noise on the photograph
            if pdf.name.startswith("photo"):
                return "xkqzrtp b3rn7 qwrtpsd " * 30
            return "\n".join(synth.text_lines(int(pdf.stem.rsplit("-", 1)[1])))

        rep = routeharness.run(ocr, seeds=1, levels=["clean", "table", "photo"])
        out = {c["level"]: c["outcome"] for c in rep["cases"]}
        self.assertEqual(out, {"clean": "saved", "table": "missed_saving", "photo": "right"})
        self.assertEqual(rep["false_pass"], [])
        self.assertEqual(rep["gate_alone"]["bad_pages"], 1)
        self.assertEqual(rep["gate_alone"]["caught"], 1)
        text = routeharness.markdown(rep)
        self.assertIn("saved", text)
        self.assertIn("The gate alone", text)

    def test_a_gross_loss_is_caught_and_a_partial_loss_is_the_known_gap(self):
        from rag_search.core.conversion import routeharness, synth

        def first(share):
            def ocr(pdf):
                lines = synth.text_lines(int(pdf.stem.rsplit("-", 1)[1]))
                return "\n".join(lines[: int(len(lines) * share)])
            return ocr

        self.assertEqual(routeharness.run(first(0.2), seeds=1, levels=["clean"])["cases"][0]["outcome"], "caught")
        # 60 % of the text for all the ink looks like a lighter page: only the text layer (digital) or a second reader can tell
        rep = routeharness.run(first(0.6), seeds=1, levels=["clean"])
        self.assertEqual(rep["cases"][0]["outcome"], "false_pass")
        self.assertEqual(len(rep["false_pass"]), 1)


if __name__ == "__main__":
    unittest.main()
