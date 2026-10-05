"""scripts/try_readers.py: compare the page readers on one document (the readers are stand-ins here)."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import unittest
from pathlib import Path
from unittest import mock

from tests.helpers import TempHome
from tests.portable.test_conv_routed import HAVE_PDF, write_routing_pdf

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "try_readers.py"


def load():
    spec = importlib.util.spec_from_file_location("try_readers", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@unittest.skipUnless(HAVE_PDF, "needs pypdfium2 and Pillow")
class TryReadersTests(TempHome):
    def test_default_results_folder_is_not_next_to_the_document(self):
        from rag_search.core.conversion import applevision

        m = load()
        pdf = self.tmp / "docs" / "p.pdf"
        pdf.parent.mkdir()
        write_routing_pdf(pdf)
        buf = io.StringIO()
        with mock.patch.object(applevision, "why_not", return_value="x"), \
                mock.patch.object(m.Path, "home", return_value=self.tmp / "home"), contextlib.redirect_stdout(buf):
            m.main([str(pdf), "--readers", "apple-vision"])
        self.assertEqual(sorted(x.name for x in pdf.parent.iterdir()), ["p.pdf"])          # nothing written beside it
        self.assertIn(str(self.tmp / "home" / ".cache"), buf.getvalue())

    def test_pages_spec(self):
        m = load()
        self.assertEqual(m.parse_pages("", 3), [1, 2, 3])
        self.assertEqual(m.parse_pages("2-3,9", 3), [2, 3])

    def test_a_reader_that_works_and_one_that_cannot_run(self):
        from rag_search.core.conversion import applevision

        m = load()
        pdf = self.tmp / "p.pdf"
        write_routing_pdf(pdf)
        out = self.tmp / "res"
        buf = io.StringIO()

        def no_docling(src, pages, is_image):                 # the same on every machine, docling installed or not
            yield None, "docling is not importable here (stubbed by the test)"

        with mock.patch.object(applevision, "why_not", return_value=""), \
                mock.patch.object(applevision, "read_page", side_effect=lambda s, n, **k: f"Account Type PPF page {n}"), \
                mock.patch.dict(m.RUNNERS, {"docling": no_docling}), contextlib.redirect_stdout(buf):
            code = m.main([str(pdf), "--readers", "apple-vision,docling", "--pages", "1-2", "--out", str(out)])
        self.assertEqual(code, 0)
        text = buf.getvalue()
        self.assertIn("== apple-vision", text)
        self.assertIn("page 2:", text)
        self.assertIn("Account Type PPF page 2", (out / "apple-vision-p2.md").read_text())
        self.assertIn("== docling", text)
        self.assertIn("== summary", text)


if __name__ == "__main__":
    unittest.main()
