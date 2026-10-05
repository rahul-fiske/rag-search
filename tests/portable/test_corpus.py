"""Every file of the test corpus, without docling: which files the scan lists, skips or reports, and the
branch the profiler and router choose for each page (``route`` in ``tests/data/corpus.json``)."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from tests import corpus
from tests.helpers import TempHome

from rag_search.core import docling_convert, indexer
from rag_search.core.conversion import profiler

HAVE_PDF = all(importlib.util.find_spec(m) for m in ("pypdfium2", "PIL"))


class ManifestTests(unittest.TestCase):
    def test_every_file_is_described_once_and_exists(self):
        listed = [e["path"] for e in corpus.manifest()]
        self.assertEqual(len(listed), len(set(listed)))
        on_disk = {p.relative_to(corpus.CORPUS).as_posix() for p in corpus.CORPUS.rglob("*") if p.is_file()}
        self.assertEqual(set(listed), on_disk)
        for e in corpus.manifest():
            self.assertTrue(e.get("shows"), e["path"])
            self.assertTrue(("skip" in e) != ("route" in e), f"{e['path']}: either skipped or routed")

    def test_needles_are_unique_to_their_file(self):
        texts = {p: p.read_bytes() for p in corpus.CORPUS.rglob("*") if p.is_file() and p.suffix in (".md", ".txt", ".html", ".htm", ".csv", ".adoc")}
        for e in corpus.entries():
            needle = e.get("needle")
            if not needle:
                continue
            others = [p for p, b in texts.items() if needle.encode() in b and p != corpus.path(e["path"])]
            self.assertEqual(others, [], f"{needle!r} also occurs elsewhere")


class ScanTests(TempHome):
    def test_the_scan_lists_reads_skips_and_reports_as_described(self):
        docs = corpus.copy_tree(self.paths.docs)
        found, unsupported = indexer.scan_sources_with_skips(docs, indexer.exclude_dirs(self.paths))
        rel = {p.relative_to(docs).as_posix() for p in found}
        self.assertEqual(rel, {e["path"] for e in corpus.entries(has="route")})
        reported = {Path(u["src"]).relative_to(docs).as_posix() for u in unsupported}
        self.assertEqual(reported, {e["path"] for e in corpus.entries(has="skip") if e["skip"] == "unsupported"})
        never = {e["path"] for e in corpus.entries(has="skip") if e["skip"] in ("hidden", "lock")}
        self.assertTrue(never and not never & (rel | reported))
        self.assertIn({"src": str(docs / "unsupported" / "no-extension"), "extension": ""}, unsupported)


@unittest.skipUnless(HAVE_PDF, "needs pypdfium2 and Pillow")
class RouteTests(unittest.TestCase):
    def test_every_file_is_profiled_and_routed_as_described(self):
        for e in corpus.entries(has="route"):
            r = e["route"]
            with self.subTest(e["path"]):
                if r.get("needs") and not importlib.util.find_spec(r["needs"]):
                    self.skipTest(f"{r['needs']} is not installed")
                src = corpus.path(e["path"])
                self.assertEqual(profiler.kind_of(src), e["kind"])
                prof = profiler.profile_file(src)
                if "error" in r:
                    self.assertIn(r["error"].lower(), (prof.get("error") or "").lower())
                    self.assertFalse(prof.get("pages"))
                    continue
                if e["kind"] in ("pdf", "image"):
                    self.assertEqual(prof["page_count"], r["pages"])
                branches = [b for _n, b, _why in profiler.route_pages(prof)]
                self.assertEqual(branches, corpus.expand_branches(r["branches"]))
                first = (prof.get("pages") or [{}])[0]
                for key, want in (r.get("profile") or {}).items():
                    got = first.get(key)
                    self.assertEqual(len(got or []) if key == "big_pics" else got, want, key)

    def test_the_text_layer_probe(self):
        """``smart`` OCR trusts a clean text layer and OCRs a scan, a garbled layer or a damaged file."""
        rep = docling_convert.text_layer_report
        self.assertTrue(rep(corpus.path("pdf/text.pdf"))["reliable"])
        for rel in ("pdf/scan.pdf", "pdf/garbled-text-layer.pdf", "pdf/damaged.pdf"):
            self.assertFalse(rep(corpus.path(rel))["reliable"], rel)

    def test_a_password_protected_pdf_is_recognised(self):
        why = docling_convert.protected_pdf_reason(corpus.path("pdf/password.pdf"))
        self.assertIn("password-protected PDF", why)
        self.assertEqual(docling_convert.protected_pdf_reason(corpus.path("pdf/damaged.pdf")), "")


if __name__ == "__main__":
    unittest.main()
