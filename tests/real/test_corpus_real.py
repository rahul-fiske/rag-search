"""The whole test corpus through the real pipeline: one indexing run (docling with OCR, a small real
embedder and reranker, the real worker, publish and search daemon), then every file is checked against
``real`` in ``tests/data/corpus.json`` and every needle is searched for."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from tests import corpus
from tests.real import RealCase

from rag_search import api, machine


def needle_of(e: dict) -> str:
    """The needle of a corpus entry on this machine: text inside a large picture is found by the document reader only."""
    if e.get("needle_needs_reader") and not machine.mlx_possible():
        return ""
    return e.get("needle", "")


class CorpusRunTests(RealCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.sdir = cls.tmp / "sources"
        corpus.copy_tree(cls.sdir)
        from rag_search import locations
        locations._save(cls.paths, {d.name: str(d) for d in sorted(cls.sdir.iterdir()) if d.is_dir()})
        out = cls.cli_json("index", "foreground", ok_codes=(1,))
        cls.summary, cls.published = out["summary"], out["publish"]

    # what the run concluded about one file ----------------------------------------------------
    def observed(self, rel: str) -> dict:
        src = str(self.sdir / rel)
        for key, status in (("errors", "error"), ("no_text", "no_text")):
            for e in self.summary.get(key) or []:
                if e["src"] == src:
                    return {"status": status, "message": e.get("message", "")}
        coll, doc = corpus.doc_name(rel)
        idx = self.paths.index / coll / doc
        meta_f = idx / "index.meta.json"
        if not meta_f.exists():
            return {"status": "missing"}
        meta = json.loads(meta_f.read_text())
        conv = meta.get("conversion") or {}
        tr = self.paths.markup / coll / (doc + ".trace.json")
        pages = json.loads(tr.read_text())["pages"] if tr.exists() else []
        return {"status": "indexed", "strip": conv.get("strip"), "outcomes": conv.get("outcomes"),
                "tables": conv.get("tables"), "src": meta.get("src_path"),
                "notes": [p.get("note") for p in pages if p.get("note")],
                "gate": sorted({c["name"] for p in pages for c in (p.get("gate") or {}).get("checks", [])}),
                "reconcile": [p["reconcile"]["role"] for p in pages if p.get("reconcile")],
                "markdown": (self.paths.markup / coll / (doc + ".md")).read_text(encoding="utf-8")}

    def mismatches(self, e: dict) -> list[str]:
        want, got = e["real"], self.observed(e["path"])
        bad = []
        if got["status"] != want["status"]:
            return [f"status {got['status']!r} ({got.get('message', '')[:200]}), expected {want['status']!r}"]
        if "message" in want and not any(m.lower() in got.get("message", "").lower() for m in want["message"].split("|")):
            bad.append(f"message {got.get('message', '')[:200]!r} lacks {want['message']!r}")
        for key in ("strip", "outcomes", "tables", "reconcile"):
            if key in want and got.get(key) != want[key]:
                bad.append(f"{key} {got.get(key)!r}, expected {want[key]!r}")
        if "gate" in want and got.get("gate") != sorted(want["gate"]):
            bad.append(f"failed checks {got.get('gate')!r}, expected {want['gate']!r}")
        if got["status"] == "indexed" and needle_of(e) and e["needle"].lower() not in got["markdown"].lower():
            bad.append(f"the Markdown lacks the needle {e['needle']!r}; it holds {got['markdown'][:400]!r}; "
                       f"notes {got.get('notes')!r}")
        if want.get("remembered"):
            coll, doc = corpus.doc_name(e["path"])
            if not (self.paths.index / coll / doc / "outcome.json").exists():
                bad.append("the failure was not remembered (no outcome.json)")
        return bad

    # the tests -------------------------------------------------------------------------------
    def test_every_file_ends_as_described(self):
        for e in corpus.entries(has="real"):
            with self.subTest(e["path"]):
                bad = self.mismatches(e)
                if e.get("known_issue"):
                    self.assertTrue(bad, f"known issue fixed? update corpus.json: {e['known_issue']}")
                else:
                    self.assertEqual(bad, [])

    def test_the_run_reports_unsupported_files_and_publishes_the_rest(self):
        reported = {Path(u["src"]).relative_to(self.sdir).as_posix()
                    for u in self.summary["unsupported_extension"]}
        self.assertEqual(reported, {e["path"] for e in corpus.entries(has="skip") if e["skip"] == "unsupported"})
        indexed = [e for e in corpus.entries(has="real") if e.get("today_status", e["real"]["status"]) == "indexed"]
        self.assertEqual(self.summary["indexed"], len(indexed))
        self.assertEqual(self.published["documents"], len(indexed))
        self.assertTrue(self.published["changed"])

    def test_every_needle_is_found_by_search_in_its_own_file(self):
        """Keyword stage only: an exact phrase must rank its own file first whatever the (small) models
        think, so this tests the index and the search path, not model quality.  A file whose text is
        also in another file (manual.docx repeats report.docx) is found in either."""
        for e in corpus.entries(has="real"):
            if not needle_of(e) or e["real"]["status"] != "indexed" or e.get("known_issue"):
                continue
            with self.subTest(e["path"]):
                r = api.search(self.paths, e["needle"], top_k=3, wait_s=300, stages=["bm25"])
                self.assertTrue(r.get("ok"), r)
                coll, doc = corpus.doc_name(e["path"])
                hits = [(h["collection"], h["file"]) for h in r["result"]["results"]]
                self.assertIn((coll, Path(doc).name), hits[:2], hits)
                if e["path"] != "office/report.docx":
                    self.assertEqual(hits[0], (coll, Path(doc).name), hits)

    def test_hybrid_search_with_the_real_models_returns_scored_hits(self):
        r = api.search(self.paths, "quarterly plan to move the archive", top_k=3, wait_s=300)
        self.assertTrue(r.get("ok"), r)
        t, hits = r["result"]["timing"], r["result"]["results"]
        self.assertEqual(t["stages"], ["bm25", "dense", "rerank"])
        self.assertTrue(t["reranked"] and hits)
        self.assertTrue(all(h["dense_score"] is not None and h["rerank_score"] is not None for h in hits))

    def test_grep_finds_exact_text_with_its_page(self):
        r = api.grep(self.paths, "silver fjord handbook", collections=["pdf"])
        self.assertTrue(r.get("ok"), r)
        m = r["result"]["matches"]
        self.assertEqual([(x["doc"], x["page"]) for x in m], [("many-pages", "22")])

    def test_a_second_run_converts_nothing_and_does_not_retry_lasting_failures(self):
        again = self.cli_json("index", "foreground", ok_codes=(0, 1))["summary"]
        self.assertEqual(again["indexed"], 0)
        self.assertEqual(again["skipped_fresh"], self.summary["indexed"])
        rel = lambda rows: {Path(e["src"]).relative_to(self.sdir).as_posix() for e in rows}  # noqa: E731
        lasting = {"pdf/password.pdf", "pdf/damaged.pdf"}                 # protected; bytes that are not a PDF
        self.assertTrue(lasting <= rel(again["known"]), again["known"])   # listed as not tried again ...
        self.assertFalse(lasting & rel(again["errors"]))                  # ... and not as errors of this run
        self.assertEqual(again["not_retried"], len(again["known"]))
        # (an .heic photo stays an error here: no reader for it in this tier, which a later run may have)

    def test_the_cli_views_of_a_real_run(self):
        out = self.cli("trace", "pdf/mixed").stdout
        self.assertIn("3 pages", out)
        out = self.cli("collection", "info", "pdf").stdout
        self.assertIn("conversion (pages as last converted):", out)
        est = self.cli_json("index", "estimate", "pdf")
        self.assertGreater(est["pages"], 30)
        res = self.cli_json("search", "violet glacier memo", "-k", "2")
        self.assertEqual(res["results"][0]["file"], "scan")
        listing = self.cli_json("list")
        self.assertTrue({"pdf", "office", "text", "images", "collision"}
                        <= {c["collection"] for c in listing["collections"]})

    def test_page_image_and_markdown_of_a_real_document(self):
        img = api.conversion_page_image(self.paths, "pdf", "mixed", 3, 300)
        self.assertTrue(img["ok"], img)
        self.assertTrue(img["png"].startswith(b"\x89PNG"))
        md = api.conversion_markdown(self.paths, "pdf", "mixed", page=1)
        self.assertIn("indigo orchard", md["result"]["markdown"].lower())

    def test_export_and_import_a_real_collection(self):
        bundle = self.tmp / "office.rag.tgz"
        self.cli("collection", "export", "office", "-o", str(bundle))
        self.cli("collection", "import", str(bundle), "--as", "office-copy")
        r = api.search(self.paths, "navy orchard hours", collections=["office-copy"], top_k=1, wait_s=300)
        self.assertTrue(r.get("ok"), r)
        self.assertEqual(r["result"]["results"][0]["file"], "hours")

    def test_the_mcp_tools_answer_from_the_real_index(self):
        import asyncio

        from rag_search.mcp.server import make_tools

        tools = make_tools("claude")
        out = json.loads(asyncio.run(tools["rag_search"]("crimson lantern onboarding", collection="office")))
        self.assertEqual(out["results"][0]["file"], "slides")
        listed = json.loads(asyncio.run(tools["rag_list_collections"]()))
        self.assertIn("pdf", {c["collection"] for c in listed["collections"]})


if __name__ == "__main__":
    unittest.main()
