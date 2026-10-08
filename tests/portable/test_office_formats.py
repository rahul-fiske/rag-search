"""RTF and macro-enabled Word files: read with the standard library and docling's own Word reader, no other tool.

(Tier A: the RTF reader is plain Python; that docling really reads the .docm is checked by tests/real through the
corpus, which has both files.)"""

from __future__ import annotations

import unittest
import zipfile

from tests import corpus
from tests.helpers import TempHome

from rag_search.core import docling_convert as dc
from rag_search.paths import SUPPORTED_EXTENSIONS


def rtf(text: str) -> bytes:
    """RTF source written with @ for the backslash, so the tests stay readable."""
    return text.replace("@", "\\").encode("ascii")


class RtfReaderTests(unittest.TestCase):
    def test_paragraphs_and_the_groups_that_hold_no_text(self):
        md = dc.rtf_to_markdown(rtf(
            "{@rtf1@ansi{@fonttbl{@f0 Times;}}{@colortbl;@red0@green0@blue0;}{@*@generator Writer;}"
            "{@info{@title Hidden title}}{@stylesheet{@s1 Heading;}}"
            "First paragraph@par Second with {@b bold} and {@i italic} words.@par}"))
        self.assertEqual(md, "First paragraph\n\nSecond with bold and italic words.\n")

    def test_accents_unicode_and_the_fallback_characters(self):
        self.assertEqual(dc.rtf_to_markdown(rtf("{@rtf1@ansi@ansicpg1252@uc1 caf@'e9 @u8364? and @u1087?@u1088?@par}")),
                         "café € and пр\n")
        # @uc0: no fallback character follows; @uc2: two do
        self.assertEqual(dc.rtf_to_markdown(rtf("{@rtf1@uc0 a@u233 b@par}")), "aéb\n")
        self.assertEqual(dc.rtf_to_markdown(rtf("{@rtf1@uc2 a@u233 ??b@par}")), "aéb\n")
        # a negative number is a 16-bit value, and a pair of surrogates is one character
        self.assertEqual(dc.rtf_to_markdown(rtf("{@rtf1@u-10179?@u-8704? ok@par}")), "\U0001f600 ok\n")

    def test_the_code_page_of_the_document_decodes_the_bytes(self):
        self.assertEqual(dc.rtf_to_markdown(rtf("{@rtf1@mac Caf@'8e@par}")), "Café\n")
        self.assertEqual(dc.rtf_to_markdown(rtf("{@rtf1@ansi@ansicpg1251 @'cf@'f0@'e8@'e2@'e5@'f2@par}")), "Привет\n")

    def test_symbols_and_special_characters(self):
        md = dc.rtf_to_markdown(rtf("{@rtf1 a@~b@_c@{d@}@@e@emdash f@endash g@bullet h@lquote i@rdblquote@par}"))
        self.assertEqual(md, "a b-c{d}\\e—f–g•h‘i”\n")      # the space after a control word is its delimiter, not text

    def test_a_table_becomes_a_pipe_table_between_the_paragraphs(self):
        md = dc.rtf_to_markdown(rtf(
            "{@rtf1@ansi Before@par@trowd@cellx3000@cellx6000@pard@intbl Item@cell Cost | EUR@cell@row"
            "@trowd@cellx3000@cellx6000@pard@intbl Pump@cell 105@par@cell@row@pard After the table.@par}"))
        self.assertEqual(md, "Before\n\n| Item | Cost \\| EUR |\n|---|---|\n| Pump | 105 |\n\nAfter the table.\n")

    def test_a_hyperlink_keeps_its_text_and_not_its_address(self):
        md = dc.rtf_to_markdown(rtf('{@rtf1 See {@field{@*@fldinst HYPERLINK "https://example.org"}{@fldrslt the site}}.@par}'))
        self.assertEqual(md, "See the site.\n")

    def test_a_picture_and_binary_data_are_skipped(self):
        md = dc.rtf_to_markdown(rtf("{@rtf1 a{@pict@pngblip 89504e470d0a}b@bin3 {}}c@par}"))
        self.assertEqual(md, "abc\n")

    def test_nothing_in_the_document_is_nothing(self):
        self.assertEqual(dc.rtf_to_markdown(rtf("{@rtf1@ansi{@fonttbl{@f0 Arial;}}@par}")), "")
        self.assertEqual(dc.rtf_to_markdown(b""), "")

    def test_a_truncated_or_unbalanced_file_is_still_read(self):
        self.assertEqual(dc.rtf_to_markdown(rtf("{@rtf1 half a paragraph")), "half a paragraph\n")
        self.assertEqual(dc.rtf_to_markdown(rtf("{@rtf1 text}}}} more@par")), "text more\n")

    def test_a_long_document_is_read_in_linear_time(self):
        import time

        big = rtf("{@rtf1@ansi " + "word @b bold@b0 caf@'e9 @u8364?@par " * 40000 + "}")
        t0 = time.perf_counter()
        md = dc.rtf_to_markdown(big)
        self.assertLess(time.perf_counter() - t0, 5.0)
        self.assertEqual(md.count("café"), 40000)


class FileTests(TempHome):
    def test_both_extensions_are_indexed_formats(self):
        self.assertLessEqual({".rtf", ".docm"}, SUPPORTED_EXTENSIONS)

    def test_an_rtf_file_is_converted_without_docling(self):
        out = self.tmp / "notes.md"
        info = dc.convert_file(corpus.copy("office/notes.rtf", self.tmp / "notes.rtf"), out)
        self.assertEqual(info["pages"], 1)
        text = out.read_text(encoding="utf-8")
        self.assertTrue(text.startswith("<!-- page 1 -->\n\nSite notes"))
        for part in ("Needle: coral meadow notes", "café budget is 40 € per week", "| Item | Cost |", "| Tables | 120.00 |",
                     "See the site."):
            self.assertIn(part, text)
        self.assertNotIn("Hidden title", text)                    # document properties are not content
        self.assertNotIn("example.org", text)

    def test_an_rtf_without_text_is_no_text_and_a_text_file_with_that_name_is_read_as_text(self):
        empty = self.tmp / "empty.rtf"
        empty.write_bytes(rtf("{@rtf1@ansi{@fonttbl{@f0 Arial;}}@par}"))
        with self.assertRaises(dc.NoTextError):
            dc.convert_file(empty, self.tmp / "empty.md")
        plain = self.tmp / "plain.rtf"
        plain.write_text("just a text file with the wrong extension", encoding="utf-8")
        dc.convert_file(plain, self.tmp / "plain.md")
        self.assertIn("just a text file", (self.tmp / "plain.md").read_text(encoding="utf-8"))
        self.assertEqual(plain.read_text(encoding="utf-8"), "just a text file with the wrong extension")   # the source is only read

    def test_a_macro_enabled_word_file_is_offered_to_docling_as_a_docx(self):
        src = corpus.copy("office/policy.docm", self.tmp / "policy.docm")
        before = src.read_bytes()
        out = dc._word_macro_as_docx(src, self.tmp)
        self.assertEqual(out.name, "policy.docx")
        with zipfile.ZipFile(src) as a, zipfile.ZipFile(out) as b:
            self.assertEqual(a.namelist(), b.namelist())             # the same package: nothing added or dropped
            self.assertIn(b"macroEnabled.main+xml", a.read("[Content_Types].xml"))
            types = b.read("[Content_Types].xml")
            self.assertNotIn(b"macroEnabled", types)
            self.assertIn(b"wordprocessingml.document.main+xml", types)
            self.assertEqual(a.read("word/document.xml"), b.read("word/document.xml"))
            self.assertIn("word/vbaProject.bin", b.namelist())       # still there, and never read
        self.assertEqual(src.read_bytes(), before)                   # the source is only read


if __name__ == "__main__":
    unittest.main()
