"""Tesseract as the last-resort reader: the module (with a stand-in binary) and the router."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

from tests.helpers import TempHome
from tests.portable.test_conv_vlm import HAVE_PDF, RoutedVlmBase

from rag_search.core.conversion import tesseract, vlm

FAKE_BIN = """#!/bin/sh
if [ "$1" = "--list-langs" ]; then
  echo 'List of available languages in "/fake/tessdata/" (3):'
  printf 'eng\\nhin\\nosd\\n'
  exit 0
fi
if [ -n "$FAKE_TESS_FAIL" ]; then echo "Error: could not open the image" >&2; exit 1; fi
echo "language argument: $4 | psm $6"
echo "Page text read by the stand-in binary with plenty of ordinary words in it."
"""


class ModuleTests(TempHome):
    def setUp(self):
        super().setUp()
        self.bindir = self.tmp / "bin"
        self.bindir.mkdir()
        exe = self.bindir / "tesseract"
        exe.write_text(FAKE_BIN)
        exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
        self.old_path = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{self.bindir}{os.pathsep}{self.old_path}"
        for k in ("RAG_SEARCH_TESSERACT", "RAG_SEARCH_TESSERACT_LANG", "FAKE_TESS_FAIL"):
            os.environ.pop(k, None)
        tesseract._langs_cache.clear()
        self.addCleanup(self.restore)

    def restore(self):
        os.environ["PATH"] = self.old_path
        for k in ("RAG_SEARCH_TESSERACT", "RAG_SEARCH_TESSERACT_LANG", "FAKE_TESS_FAIL"):
            os.environ.pop(k, None)
        tesseract._langs_cache.clear()

    def test_the_requested_languages_are_cut_down_to_the_installed_ones(self):
        self.assertEqual(tesseract.installed_languages(), {"eng", "hin", "osd"})
        self.assertEqual(tesseract.languages(), "hin+eng")                # mar is not installed
        os.environ["RAG_SEARCH_TESSERACT_LANG"] = "eng,fra"
        self.assertEqual(tesseract.languages(), "eng")
        self.assertEqual(tesseract.why_not(), "")

    def test_it_says_why_it_cannot_be_used(self):
        os.environ["RAG_SEARCH_TESSERACT"] = "off"
        self.assertIn("switched off", tesseract.why_not())
        del os.environ["RAG_SEARCH_TESSERACT"]
        os.environ["RAG_SEARCH_TESSERACT_LANG"] = "fra"
        self.assertIn("none of the requested languages", tesseract.why_not())
        os.environ["PATH"] = str(self.tmp / "nowhere")
        tesseract._langs_cache.clear()
        self.assertIn("not installed", tesseract.why_not())

    def test_an_image_is_read_with_the_languages_and_page_segmentation_mode(self):
        text = tesseract.read_image(self.tmp / "x.png")
        self.assertIn("language argument: hin+eng | psm 4", text)
        self.assertIn("ordinary words", text)

    def test_a_failing_binary_is_an_error(self):
        os.environ["FAKE_TESS_FAIL"] = "1"
        with self.assertRaises(RuntimeError):
            tesseract.read_image(self.tmp / "x.png")


@unittest.skipUnless(HAVE_PDF, "needs pypdfium2 and Pillow")
class EnsureInstalledTests(TempHome):
    """``rag-search setup``'s Tesseract step, with stand-in programs and no network."""

    def setUp(self):
        super().setUp()
        self.bindir = self.tmp / "bin"
        self.bindir.mkdir()
        self.said: list[str] = []
        self.old_path = os.environ.get("PATH", "")
        os.environ["PATH"] = str(self.bindir)                     # no tesseract, brew or curl from this machine
        tesseract._langs_cache.clear()
        self.addCleanup(self.restore)

    def restore(self):
        os.environ["PATH"] = self.old_path
        tesseract._langs_cache.clear()

    def exe(self, name: str, text: str) -> Path:
        path = self.bindir / name
        path.write_text(text)
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
        return path

    def fake_tesseract(self, langs: str, data_dir: Path) -> None:
        self.exe("tesseract", '#!/bin/sh\nif [ "$1" = "--list-langs" ]; then\n'
                 f'  echo \'List of available languages in "{data_dir}/" (2):\'\n'
                 + "".join(f"  echo {x}\n" for x in langs.split()) + "fi\n")

    def run_step(self, **kw):
        tesseract.ensure_installed(self.said.append, **kw)
        return "\n".join(self.said)

    def test_nothing_is_done_when_the_languages_are_there(self):
        self.fake_tesseract("eng hin mar osd", self.tmp)
        out = self.run_step()
        self.assertEqual(out, "")

    def test_the_missing_language_data_is_downloaded_into_a_writable_folder(self):
        data = self.tmp / "tessdata"
        data.mkdir()
        self.fake_tesseract("eng", data)
        asked = []

        class Resp:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *a):
                return False

            def read(self_inner):
                return b"traineddata"

        def opener(url, timeout=0):
            asked.append(url)
            return Resp()

        with mock.patch.object(tesseract.urllib.request, "urlopen", opener):
            out = self.run_step()
        self.assertEqual(sorted(p.name for p in data.iterdir()), ["hin.traineddata", "mar.traineddata"])
        self.assertEqual(len(asked), 2)
        self.assertTrue(all("tessdata_best" in u for u in asked))
        self.assertIn("adding the mar language data", out)

    def test_a_failed_download_leaves_no_file_and_says_so(self):
        data = self.tmp / "tessdata"
        data.mkdir()
        self.fake_tesseract("eng hin", data)
        with mock.patch.object(tesseract.urllib.request, "urlopen", side_effect=OSError("offline")):
            out = self.run_step()
        self.assertEqual(list(data.iterdir()), [])
        self.assertIn("could not download mar.traineddata", out)

    def test_a_folder_that_cannot_be_written_is_reported(self):
        self.fake_tesseract("eng", Path("/nonexistent/tessdata"))
        out = self.run_step()
        self.assertIn("no 'mar' data", out)
        self.assertIn("no 'hin' data", out)

    def test_only_looking_changes_nothing(self):
        data = self.tmp / "tessdata"
        data.mkdir()
        self.fake_tesseract("eng", data)
        with mock.patch.object(tesseract.urllib.request, "urlopen", side_effect=AssertionError("no network")):
            out = self.run_step(install=False)
        self.assertEqual(list(data.iterdir()), [])
        self.assertIn("no 'mar' data", out)

    def test_a_missing_tesseract_is_installed_with_homebrew_when_it_is_there(self):
        calls = []
        self.exe("brew", "#!/bin/sh\n")

        def fake_run(cmd, **kw):
            calls.append(cmd)
            if cmd[:2] == ["brew", "install"]:
                self.fake_tesseract("eng hin mar", self.tmp)
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with mock.patch.object(tesseract.subprocess, "run", fake_run):
            out = self.run_step()
        self.assertEqual(calls[0], ["brew", "install", "tesseract"])
        self.assertIn("Installing Tesseract with Homebrew", out)
        self.assertNotIn("warning", out)

    def test_without_homebrew_it_says_how_to_get_tesseract(self):
        with mock.patch.object(sys, "platform", "darwin"):
            out = self.run_step()
        self.assertIn("Tesseract is not installed", out)
        self.assertIn("brew install tesseract", out)
        self.said.clear()
        with mock.patch.object(sys, "platform", "linux"):
            self.assertIn("apt-get install -y tesseract-ocr", self.run_step())


class LastResortTests(RoutedVlmBase):
    TEXT = ("Clause one binds the seller to deliver the property. Clause two binds the buyer to pay the sum "
            "agreed on the date named in the schedule below. " * 3)

    def reader(self):
        return vlm.VlmReader("fake/model", style="instruct", need_gb=1.0)

    def page3(self, res):
        return [x for x in res["records"] if x["page"] == 3][0]

    def test_a_page_the_reader_cannot_stop_repeating_is_read_by_tesseract(self):
        self.plan(always_loop=True)
        with mock.patch.object(tesseract, "why_not", return_value=""), \
                mock.patch.object(tesseract, "read_page", return_value=self.TEXT) as read:
            res = self.convert(self.reader())
            rec = self.page3(res)
            self.assertEqual((rec["reader"]["tool"], rec["branch"]), ("tesseract", "fallback"))
            self.assertEqual([c["name"] for c in rec["gate"]["checks"]], ["low_resolution"])   # the fixture is 72 dpi; the runaway is gone
            self.assertIn("read by Tesseract", rec["note"])
            self.assertIn("Clause one binds", self.out.read_text())
            self.assertEqual(read.call_count, 1)
            calls = self.calls()
            self.convert(self.reader())                                   # the next run: from the cache
            self.assertEqual(read.call_count, 1)
            self.assertEqual(self.calls(), calls)

    def test_without_tesseract_the_page_stays_flagged_and_says_so(self):
        self.plan(always_loop=True)
        with mock.patch.object(tesseract, "why_not", return_value="tesseract is not installed"):
            rec = self.page3(self.convert(self.reader()))
        self.assertEqual(rec["outcome"], "low")
        self.assertIn("Tesseract not used (tesseract is not installed)", rec["note"])

    def test_a_runaway_from_tesseract_is_not_taken(self):
        self.plan(always_loop=True)
        loop = "The said property shall be conveyed to the purchaser free of all charges.\n" * 80
        with mock.patch.object(tesseract, "why_not", return_value=""), \
                mock.patch.object(tesseract, "read_page", return_value=loop):
            rec = self.page3(self.convert(self.reader()))
        self.assertEqual(rec["outcome"], "low")
        self.assertIn("found no usable text either", rec["note"])

    def test_letters_that_are_not_words_are_not_taken(self):
        self.plan(always_loop=True)                         # what Tesseract writes on a page in a script it was not given
        junk = "Xqzt wrtk plmn bvcx zzrt qwrt mnbv lkjh gfds trwq pljk hgfd szxc vbnm.\n" * 12
        with mock.patch.object(tesseract, "why_not", return_value=""), \
                mock.patch.object(tesseract, "read_page", return_value=junk):
            rec = self.page3(self.convert(self.reader()))
        self.assertEqual((rec["outcome"], rec["reader"]["tool"]), ("low", "vlm"))
        self.assertIn("found no usable text either", rec["note"])
        self.assertNotIn("Xqzt", self.out.read_text())

    def test_a_healthy_page_never_reaches_tesseract(self):
        with mock.patch.object(tesseract, "read_page", side_effect=AssertionError("must not be called")):
            rec = self.page3(self.convert(self.reader()))
        self.assertEqual(rec["reader"]["tool"], "vlm")


if __name__ == "__main__":
    unittest.main()
