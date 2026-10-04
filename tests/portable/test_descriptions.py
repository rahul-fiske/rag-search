"""descriptions.py (storage) and its use from catalog.list_view / api.describe_collection."""

from __future__ import annotations

import json

from tests.helpers import TempHome
from rag_search import api, descriptions, policy
from rag_search.catalog import list_view


class DescriptionsStorageTests(TempHome):
    def test_round_trip_and_clear(self):
        by_name, error = descriptions.load_descriptions(self.paths)
        self.assertEqual((by_name, error), ({}, ""))

        res = descriptions.set_description(self.paths, "manuals", "Product manuals")
        self.assertEqual(res, {"collection": "manuals", "description": "Product manuals",
                               "changed": True})
        self.assertEqual(descriptions.get_description(self.paths, "manuals"), "Product manuals")
        # case-insensitive lookup
        self.assertEqual(descriptions.get_description(self.paths, "MANUALS"), "Product manuals")

        cleared = descriptions.set_description(self.paths, "manuals", "")
        self.assertEqual(cleared["description"], "")
        self.assertEqual(descriptions.get_description(self.paths, "manuals"), "")
        by_name, _ = descriptions.load_descriptions(self.paths)
        self.assertEqual(by_name, {})

    def test_updating_an_existing_case_folded_name_keeps_its_original_case(self):
        descriptions.set_description(self.paths, "Manuals", "first")
        res = descriptions.set_description(self.paths, "manuals", "second")
        self.assertEqual(res["collection"], "Manuals")
        by_name, _ = descriptions.load_descriptions(self.paths)
        self.assertEqual(by_name, {"Manuals": "second"})

    def test_rejects_bad_names(self):
        for bad in ("", "  ", "..", "a/b", ".hidden"):
            with self.assertRaises(descriptions.DescriptionError):
                descriptions.set_description(self.paths, bad, "text")

    def test_rejects_overlong_text(self):
        with self.assertRaises(descriptions.DescriptionError):
            descriptions.set_description(self.paths, "manuals", "x" * 501)

    def test_broken_file_reports_error_and_refuses_writes(self):
        self.paths.descriptions_file.parent.mkdir(parents=True, exist_ok=True)
        self.paths.descriptions_file.write_text("not json", encoding="utf-8")
        by_name, error = descriptions.load_descriptions(self.paths)
        self.assertEqual(by_name, {})
        self.assertIn("descriptions.json", error)
        with self.assertRaises(descriptions.DescriptionError):
            descriptions.set_description(self.paths, "manuals", "text")

    def test_stored_file_shape(self):
        descriptions.set_description(self.paths, "manuals", "Product manuals")
        data = json.loads(self.paths.descriptions_file.read_text(encoding="utf-8"))
        self.assertEqual(data, {"version": 1, "collections": {"manuals": "Product manuals"}})


class ListViewCompactTests(TempHome):
    def _publish_one(self):
        self.write_doc("manuals/a.md", "# A\n\n<!-- page 1 -->\n\nhello world\n")
        self.index()
        self.publish()

    def test_full_false_by_default_and_document_count(self):
        self._publish_one()
        view = list_view(self.paths, policy.Rules(), "cli")
        c = view["collections"][0]
        self.assertNotIn("documents", c)
        self.assertEqual(c["document_count"], 1)
        self.assertEqual(c["description"], "")
        self.assertEqual(view["totals"]["documents"], 1)
        self.assertIn("hint", view)

    def test_full_true_includes_documents(self):
        self._publish_one()
        view = list_view(self.paths, policy.Rules(), "cli", full=True)
        c = view["collections"][0]
        self.assertEqual([d["source"] for d in c["documents"]], ["a.md"])
        self.assertEqual(c["document_count"], 1)

    def test_description_surfaces_once_set(self):
        self._publish_one()
        descriptions.set_description(self.paths, "manuals", "How-to guides")
        view = list_view(self.paths, policy.Rules(), "cli")
        self.assertEqual(view["collections"][0]["description"], "How-to guides")


class ApiDescribeCollectionTests(TempHome):
    def _publish_one(self):
        self.write_doc("manuals/a.md", "# A\n\n<!-- page 1 -->\n\nhello world\n")
        self.index()
        self.publish()

    def test_set_and_clear_round_trip_through_list_collections(self):
        self._publish_one()
        r = api.describe_collection(self.paths, "manuals", "How-to guides", client="cli")
        self.assertTrue(r.get("ok"), r)
        r = api.list_collections(self.paths, client="cli")
        self.assertTrue(r.get("ok"), r)
        self.assertEqual(r["result"]["collections"][0]["description"], "How-to guides")

        r = api.describe_collection(self.paths, "manuals", "", client="cli")
        self.assertTrue(r.get("ok"), r)
        r = api.list_collections(self.paths, client="cli")
        self.assertEqual(r["result"]["collections"][0]["description"], "")

    def test_refuses_a_collection_the_client_cannot_see(self):
        self._publish_one()
        policy.save_rules(self.paths, {"manuals": ["claude"]})
        r = api.describe_collection(self.paths, "manuals", "nope", client="agent")
        self.assertFalse(r.get("ok"))
        self.assertEqual(r.get("code"), "bad_request")
        self.assertIn("unknown collection", r.get("error", ""))
        by_name, _ = descriptions.load_descriptions(self.paths)
        self.assertEqual(by_name, {})

    def test_refuses_an_unknown_collection(self):
        self._publish_one()
        r = api.describe_collection(self.paths, "nope-at-all", "text", client="cli")
        self.assertFalse(r.get("ok"))
        self.assertEqual(r.get("code"), "bad_request")
