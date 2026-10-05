"""inventory.collection_info: the Collections tab's per-collection details and
`rag-search collection info`."""

from __future__ import annotations

import os
import time

from tests.helpers import FakeEmbedder, TempHome
from tests.portable.test_cli_api import run
from tests.portable.test_ui import UiBase
from rag_search import api, bundle, inventory, locations
from rag_search.core import indexer
from rag_search.paths import get_paths

DOC = "# Title\n\n<!-- page 1 -->\n\nPublic key authentication protects every account.\n"


class InventoryTests(TempHome):
    def plan_index(self, raw=""):
        return indexer.run_plan(self.paths, locations.plan_scan(self.paths, raw), jobs=1,
                                embedder=FakeEmbedder())

    def test_a_published_up_to_date_collection(self):
        self.write_doc("manuals/a.md", DOC)
        self.write_doc("manuals/sub/b.md", DOC + "More text.\n")
        self.write_doc("manuals/old.xls", "x")
        self.plan_index()
        self.publish()
        i = inventory.collection_info(self.paths, "MANUALS")
        self.assertEqual((i["collection"], i["kind"], i["state"]), ("manuals", "location", "ok"))
        self.assertEqual((i["source"]["files"], i["source"]["unsupported"]), (2, 1))
        self.assertEqual(i["source"]["folder"], str(self.sdir / "manuals"))
        ws = i["workspace"]
        self.assertEqual(ws["documents"], 2)
        self.assertEqual(ws["markdown_files"], 2)
        self.assertEqual(ws["index_folder"], str(self.paths.index / "manuals"))
        self.assertEqual(ws["markdown_folder"], str(self.paths.markup / "manuals"))
        self.assertGreater(ws["index_bytes"], ws["merged_index_bytes"] > 0)
        self.assertEqual(i["disk"]["total_bytes"], ws["index_bytes"] + ws["markdown_bytes"])
        self.assertEqual(i["published"]["documents"], 2)
        self.assertIn("gen-", i["published"]["index_folder"])
        self.assertEqual(i["build"]["model"], "BAAI/bge-m3")
        self.assertTrue(i["build"]["last_indexed"])
        self.assertIsNone(i["last_run"])                  # indexed in-process: no job record
        self.assertEqual(i["attention"]["not_indexed"]["count"], 0)

    def test_new_and_changed_documents_are_pending(self):
        self.write_doc("manuals/a.md", DOC)
        self.plan_index()
        self.publish()
        self.write_doc("manuals/new.md", DOC)
        p = self.sdir / "manuals" / "a.md"
        later = time.time() + 30
        os.utime(p, (later, later))
        i = inventory.collection_info(self.paths, "manuals")
        self.assertEqual(i["state"], "pending")
        self.assertEqual(i["attention"]["not_indexed"]["names"], ["new.md"])
        self.assertEqual(i["attention"]["modified_since_indexed"]["names"], ["a"])

    def test_indexed_but_not_published(self):
        self.write_doc("manuals/a.md", DOC)
        self.plan_index()
        self.assertEqual(inventory.collection_info(self.paths, "manuals")["state"], "unpublished")

    def test_failed_documents_show_their_reason(self):
        self.write_doc("manuals/a.md", DOC)
        self.write_doc("manuals/broken.pdf", "this is not a pdf")
        from rag_search.core import worker

        class Events(worker.EventWriter):
            def __init__(self, path):
                super().__init__(path)

        from rag_search import jobs
        self.paths.jobs.mkdir(parents=True, exist_ok=True)
        jobs.job_file(self.paths, "j1").write_text('{"id": "j1", "status": "succeeded", '
                                                    '"created_at": 1.0}')
        ev = Events(jobs.events_file(self.paths, "j1"))
        try:
            s = indexer.run_plan(self.paths, locations.plan_scan(self.paths), jobs=1,
                                 embedder=FakeEmbedder(), progress=ev.progress)
        finally:
            ev.close()
        self.assertTrue(s["errors"])
        i = inventory.collection_info(self.paths, "manuals")
        self.assertIn("broken.pdf", i["attention"]["not_indexed"]["names"])
        self.assertIn("broken.pdf", i["attention"]["not_indexed"]["reasons"])
        self.assertEqual(i["last_run"]["job"], "j1")
        self.assertEqual(i["last_run"]["errors"][0]["source"], "broken.pdf")

    def test_location_unreachable_and_imported(self):
        folder = self.tmp / "vault"
        folder.mkdir()
        (folder / "n.md").write_text(DOC)
        locations.add(self.paths, "vault", str(folder))
        self.plan_index()
        self.publish()
        i = inventory.collection_info(self.paths, "vault")
        self.assertEqual((i["kind"], i["state"], i["source"]["folder"]),
                         ("location", "ok", str(folder.resolve())))
        folder.rename(self.tmp / "gone")
        i = inventory.collection_info(self.paths, "vault")
        self.assertEqual((i["state"], i["source"]["reachable"], i["source"]["files"]),
                         ("unreachable", False, None))
        self.assertEqual(i["workspace"]["documents"], 1)
        (self.tmp / "gone").rename(folder)
        f = bundle.export_collection(self.paths, "vault", self.tmp)["file"]
        os.environ["RAG_SEARCH_HOME"] = str(self.tmp / "home2")
        self.paths = get_paths()
        api.collection_import(self.paths, f)
        i = inventory.collection_info(self.paths, "vault")
        self.assertEqual((i["kind"], i["state"]), ("imported", "imported"))
        self.assertEqual(i["origin"]["source_collection"], "vault")
        self.assertIsNone(i["source"]["folder"])

    def test_unknown_names_and_cli(self):
        r = api.collection_info(self.paths, "nope")
        self.assertFalse(r["ok"])
        self.assertIn("no collection named", r["error"])
        self.assertFalse(api.collection_info(self.paths, "../etc")["ok"])
        self.write_doc("manuals/a.md", DOC)
        self.plan_index()
        self.publish()
        rc, out, err = run("collection", "info", "manuals")
        self.assertEqual(rc, 0, err)
        for text in ("state: ok", str(self.paths.markup / "manuals"),
                     str(self.paths.index / "manuals"), "last run"):
            self.assertIn(text, out)


class DashboardInfoTests(UiBase):
    def test_collection_info_endpoint(self):
        st, js, _, _ = self.dash.req("GET", "/api/collection/info?name=hr")
        self.assertEqual(st, 200, js)
        self.assertEqual(js["result"]["collection"], "hr")
        self.assertEqual(js["result"]["workspace"]["documents"], 1)
        st, js, _, _ = self.dash.req("GET", "/api/collection/info?name=nope")
        self.assertEqual(st, 400)

    def test_architecture_lists_sources_and_runtime_settings(self):
        folder = self.tmp / "vault"
        folder.mkdir()
        (folder / "v.md").write_text(DOC)
        locations.add(self.paths, "vault", str(folder))
        st, js, _, _ = self.dash.req("GET", "/api/architecture")
        self.assertEqual(st, 200, js)
        self.assertIn({"collection": "vault", "folder": str(folder.resolve())}, js["sources"]["locations"])
        self.assertEqual(js["sources"]["unregistered"], [])
        self.assertEqual(js["sources"]["imported"], [])
        self.assertIn("max_len", js["models"]["reranker"])
        self.assertGreater(js["chunking"]["size"], js["chunking"]["overlap"])


class DashboardCollectionActionsTests(UiBase):
    def test_add_export_download_import_delete(self):
        folder = self.tmp / "notes"
        folder.mkdir()
        (folder / "n.md").write_text(DOC)
        st, js, _, _ = self.dash.req("POST", "/api/collection/add-location",
                                     {"name": "garden", "folder": str(folder)})
        self.assertEqual(st, 200, js)
        self.assertIn("garden", locations.names(self.paths))
        st, js, _, _ = self.dash.req("POST", "/api/collection/add-location",
                                     {"name": "x", "folder": str(self.tmp / "missing")})
        self.assertEqual(st, 400)

        # export hr into the default exports folder, then download it
        st, js, _, _ = self.dash.req("POST", "/api/collection/export", {"name": "hr"})
        self.assertEqual(st, 200, js)
        f = js["result"]["file"]
        self.assertTrue(f.startswith(str(self.paths.home / "exports")))
        st, _, payload, hdrs = self.dash.req("GET", "/api/" + js["result"]["download"])
        self.assertEqual(st, 200)
        self.assertEqual(payload[:2], b"\x1f\x8b")                   # gzip
        self.assertIn("hr.rag.tgz", dict(hdrs).get("Content-Disposition", ""))
        st, _, _, _ = self.dash.req("GET", "/api/download?id=made-up")
        self.assertEqual(st, 404)

        # delete needs the name typed back; it removes only the workspace data
        st, js, _, _ = self.dash.req("POST", "/api/collection/delete", {"name": "hr", "confirm": "h"})
        self.assertEqual(st, 400)
        self.assertTrue((self.paths.index / "hr").exists())
        st, js, _, _ = self.dash.req("POST", "/api/collection/delete", {"name": "hr", "confirm": "HR"})
        self.assertEqual(st, 200, js)
        self.assertFalse((self.paths.index / "hr").exists())
        self.assertTrue((self.sdir / "hr" / "leave.md").is_file())
        self.assertTrue(js["result"]["sources_remain"])

        # import the export under another name
        st, js, _, _ = self.dash.req("POST", "/api/collection/import",
                                     {"file": f, "as_name": "hr-copy"})
        self.assertEqual(st, 200, js)
        self.assertTrue(locations.is_imported(self.paths, "hr-copy"))

        st, js, _, _ = self.dash.req("POST", "/api/location/remove",
                                     {"name": "garden", "confirm": "garden"})
        self.assertEqual(st, 200, js)
        self.assertNotIn("garden", locations.names(self.paths))
        self.assertTrue((folder / "n.md").is_file())


class ReadOnlyDashboardTests(UiBase):
    read_only = True

    def test_collection_actions_are_refused(self):
        for route, body in (("collection/add-location", {"name": "x", "folder": "/tmp"}),
                            ("collection/delete", {"name": "hr", "confirm": "hr"}),
                            ("collection/export", {"name": "hr"})):
            st, _, _, _ = self.dash.req("POST", "/api/" + route, body)
            self.assertEqual(st, 403, route)
        self.assertTrue((self.paths.index / "hr").exists())
