"""Release checks for the conversion work (phase P5): the upgrade path, export/import of the new
metadata, and the flag from the page to a search hit."""

from __future__ import annotations

import json
import os
import unittest

from tests.portable.test_conv_repair import BAD, GOOD, StatementIndexBase
from tests.portable.test_conversion import HAVE_PDF

from rag_search import api, bundle
from rag_search.core import docling_convert
from rag_search.paths import ALL_DIR, ensure_dirs, get_paths


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.pop(k, None) for k in ("RAG_SEARCH_VLM", "RAG_SEARCH_REPAIR")}

    def tearDown(self):
        for k, v in self._saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v

    def test_the_defaults_add_nothing_to_the_profile(self):
        p = docling_convert.convert_profile()
        self.assertTrue(p.endswith("|route=pages"), p)

    def test_a_reader_or_repair_that_is_off_is_part_of_the_profile(self):
        os.environ["RAG_SEARCH_VLM"] = "off"
        self.assertTrue(docling_convert.convert_profile().endswith("|route=pages|vlm=off"))
        os.environ["RAG_SEARCH_REPAIR"] = "off"
        self.assertTrue(docling_convert.convert_profile().endswith("|vlm=off|repair=off"))
        os.environ["RAG_SEARCH_VLM"] = "auto"
        self.assertTrue(docling_convert.convert_profile().endswith("|route=pages|repair=off"))

    def test_the_page_cache_does_not_see_them(self):
        base = docling_convert.convert_profile(readers=False)
        os.environ["RAG_SEARCH_VLM"] = "off"
        os.environ["RAG_SEARCH_REPAIR"] = "off"
        self.assertEqual(docling_convert.convert_profile(readers=False), base)
        self.assertNotEqual(docling_convert.convert_profile(), base)


@unittest.skipUnless(HAVE_PDF, "needs pypdfium2 and Pillow")
class UpgradeTests(StatementIndexBase):
    def setUp(self):
        super().setUp()
        self.plan({"text": BAD, "cell": "2,819.00"})

    def test_a_current_document_is_skipped_and_switching_the_reader_off_converts_it_again(self):
        self.assertEqual(self.run_index()["indexed"], 1)
        self.assertEqual(self.run_index().get("indexed", 0), 0)               # nothing changed: nothing redone
        os.environ["RAG_SEARCH_VLM"] = "off"
        self.assertEqual(self.run_index()["indexed"], 1)                      # the page is read by docling now
        t = json.loads((self.paths.markup / "bank" / "stmt.trace.json").read_text())
        self.assertEqual([p["branch"] for p in t["pages"]], ["fallback"])
        self.assertTrue(t["convert"].endswith("|vlm=off"))

    def test_switching_repair_off_converts_again_but_a_page_repaired_earlier_stays_repaired(self):
        self.run_index()
        before = self.calls()
        os.environ["RAG_SEARCH_REPAIR"] = "off"
        self.assertEqual(self.run_index()["indexed"], 1)
        conv = self.meta("bank/stmt")["conversion"]
        self.assertEqual((conv["outcomes"], conv.get("cached_pages")), ({"repaired": 1}, 1))
        self.assertEqual(self.calls(), before)                                # nothing was read again

    def test_a_page_cached_before_repair_existed_is_repaired_when_repair_is_switched_on(self):
        os.environ["RAG_SEARCH_REPAIR"] = "off"
        self.run_index()                                                      # an "old" cache entry: no repair info
        self.assertEqual(self.meta("bank/stmt")["conversion"]["outcomes"], {"low": 1})
        pages_read = self.calls()
        os.environ["RAG_SEARCH_REPAIR"] = "auto"
        self.assertEqual(self.run_index()["indexed"], 1)
        conv = self.meta("bank/stmt")["conversion"]
        self.assertEqual((conv["outcomes"], conv["repaired_cells"], conv.get("cached_pages")), ({"repaired": 1}, 1, 1))
        self.assertEqual(self.calls(), pages_read + 1)                        # only the cell was read, not the page


@unittest.skipUnless(HAVE_PDF, "needs pypdfium2 and Pillow")
class FlagTests(StatementIndexBase):
    def test_a_search_hit_on_a_flagged_page_says_so(self):
        self.plan({"text": BAD, "cell": "2,918.00"})
        self.run_index()
        self.publish()
        hits = self.engine().search("running balance credit", 5, None)["results"]
        self.assertTrue(hits)
        self.assertTrue(all(h.get("confidence") == "low" for h in hits))

    def test_a_repaired_page_is_not_flagged(self):
        self.plan({"text": BAD, "cell": "2,819.00"})
        self.run_index()
        self.publish()
        hits = self.engine().search("Salary", 5, None)["results"]
        self.assertTrue(hits)
        self.assertTrue(all("confidence" not in h for h in hits))
        self.assertIn("2,819.00", hits[0]["text"])


@unittest.skipUnless(HAVE_PDF, "needs pypdfium2 and Pillow")
class ExportImportTests(StatementIndexBase):
    def roundtrip(self):
        self.run_index()
        self.publish()
        f = bundle.export_collection(self.paths, "bank", self.tmp)["file"]
        os.environ["RAG_SEARCH_HOME"] = str(self.tmp / "home2")
        self.paths = get_paths()
        ensure_dirs(self.paths)
        r = api.collection_import(self.paths, f)
        self.assertTrue(r["ok"], r)
        self.assertTrue((self.paths.index / "bank" / "stmt" / "index.meta.json").is_file())
        return json.loads((self.paths.index / "bank" / ALL_DIR / "nodes.json").read_text())["nodes"]

    def test_the_low_confidence_flag_and_the_conversion_record_survive_an_export(self):
        self.plan({"text": BAD, "cell": "2,918.00"})
        nodes = self.roundtrip()
        self.assertTrue(nodes and all(n["metadata"].get("confidence") == "low" for n in nodes))
        meta = json.loads((self.paths.index / "bank" / "stmt" / "index.meta.json").read_text())
        conv = meta["conversion"]
        self.assertEqual((conv["outcomes"], conv["low_pages"], conv["repair_tried"]), ({"low": 1}, [1], 1))
        info = api.collection_info(self.paths, "bank")["result"]
        self.assertEqual(info["conversion"]["low_documents"]["count"], 1)

    def test_a_repaired_collection_exports_with_its_repair_counts(self):
        self.plan({"text": BAD, "cell": "2,819.00"})
        nodes = self.roundtrip()
        self.assertTrue(all("confidence" not in n["metadata"] for n in nodes))
        self.assertIn("2,819.00", "".join(n["text"] for n in nodes))
        info = api.collection_info(self.paths, "bank")["result"]
        self.assertEqual((info["conversion"]["repaired_cells"], info["conversion"]["outcomes"]), (1, {"repaired": 1}))

    def test_an_export_without_the_new_fields_imports_and_reads(self):
        self.plan({"text": GOOD})                # a clean page: no repair, no flag
        nodes = self.roundtrip()
        self.assertTrue(nodes and all("confidence" not in n["metadata"] for n in nodes))
        info = api.collection_info(self.paths, "bank")["result"]
        self.assertNotIn("repaired_cells", {k: v for k, v in info["conversion"].items() if v})
        self.assertEqual(info["conversion"]["low_documents"]["count"], 0)


if __name__ == "__main__":
    unittest.main()
