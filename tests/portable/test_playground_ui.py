"""Dashboard /api/playground/*: every action shells out to `rag-search playground ...` (like
doctor) so the dashboard's own process never loads a playground experiment's model."""

from __future__ import annotations

import re
import time
import unittest

from tests.portable.test_ui import UiBase

from rag_search.ui import info


class PlaygroundApiTests(UiBase):
    def setUp(self):
        super().setUp()
        self.src = self.tmp / "pgdocs"
        self.src.mkdir()
        (self.src / "pg.txt").write_text("<!-- page 1 -->\nhow is a session token refreshed\n",
                            encoding="utf-8")

    def _post(self, action, body):
        st, js, _, _ = self.dash.req("POST", f"/api/playground/{action}", body)
        return st, js

    def test_sources_are_the_users_folders_and_documents_open_like_production(self):
        from tests.portable.test_cli_api import run
        st, js = self._post("create", {"name": "demo", "folders": [str(self.src)]})
        self.assertEqual(st, 200, js)
        self.assertEqual([x["collection"] for x in js["result"]["sources"]], ["pgdocs"])
        st, js = self._post("sources", {"name": "demo", "op": "list"})
        self.assertEqual([(x["collection"], x["folder"]) for x in js["result"]["status"]], [("pgdocs", str(self.src.resolve()))])
        other = self.tmp / "more"
        other.mkdir()
        st, js = self._post("sources", {"name": "demo", "op": "add", "folder": str(other), "collection": "extra"})
        self.assertEqual((st, js["result"].get("collection")), (200, "extra"), js)
        st, js = self._post("sources", {"name": "demo", "op": "remove", "collection": "extra"})
        self.assertEqual(js["result"].get("removed"), "extra", js)
        # the same functions that serve production's document list serve the experiment's
        rc, out, err = run("playground", "index", "demo", "--json")
        self.assertEqual(rc, 0, err)
        st, js, _, _ = self.dash.req("GET", "/api/conversion/markdown?exp=demo&collection=pgdocs&doc=pg")
        self.assertEqual(st, 200, js)
        self.assertIn("session token", js["result"]["markdown"])
        st, js, _, _ = self.dash.req("GET", "/api/conversion/trace?exp=demo&collection=pgdocs&doc=pg")
        self.assertEqual((st, js["result"]["collection"]), (200, "pgdocs"))
        st, _, _, _ = self.dash.req("GET", "/api/conversion/markdown?collection=pgdocs&doc=pg")   # not production's
        self.assertEqual(st, 400)
        st, _, _, _ = self.dash.req("GET", "/api/conversion/markdown?exp=nope&collection=pgdocs&doc=pg")
        self.assertEqual(st, 400)

    def test_full_flow(self):
        st, js = self._post("create", {"name": "demo"})
        self.assertEqual(st, 200)
        self.assertTrue(js["ok"], js)

        st, js = self._post("config", {"name": "demo", "embedding_model": "custom/id"})
        self.assertTrue(js["ok"], js)
        self.assertEqual(js["result"]["embedding_model"], "custom/id")

        # no docs were added through the API (no upload endpoint yet): the run starts in the background
        # and fails there with the CLI's clean "no source folders" message -- the dashboard does not crash
        st, js = self._post("index", {"name": "demo"})
        self.assertEqual(st, 200)
        self.assertTrue(js["ok"], js)
        for _ in range(100):
            st, js = self._post("run", {"name": "demo"})
            if js["result"]["job"]["status"] not in ("queued", "running"):
                break
            time.sleep(0.3)
        self.assertEqual(js["result"]["job"]["status"], "failed")
        self.assertIn("no source folders", js["result"]["job"]["error"])

    def test_isolated_from_production_dashboard_session(self):
        """The playground subprocess must never disturb the production daemons/sockets the
        rest of this dashboard session is using."""
        st, js = self._post("create", {"name": "sandbox"})
        self.assertTrue(js["ok"], js)
        st, js = self._post("list", {})
        self.assertTrue(js["ok"], js)
        names = [e["name"] for e in js["result"]]
        self.assertIn("sandbox", names)
        # production search (via the real daemon) still works exactly as before
        st, js, _, _ = self.dash.req("POST", "/api/search", {"query": "leave"})
        self.assertEqual(st, 200)
        self.assertTrue(js["ok"])

    def test_missing_name_is_a_clean_400(self):
        st, js = self._post("index", {})
        self.assertEqual(st, 400)
        self.assertFalse(js["ok"])

    def test_rm_requires_confirm(self):
        self._post("create", {"name": "gone"})
        st, js = self._post("rm", {"name": "gone"})
        self.assertEqual(st, 400)
        st, js = self._post("rm", {"name": "gone", "confirm": True})
        self.assertEqual(st, 200)
        self.assertTrue(js["ok"])

    def test_unknown_action_404(self):
        st, js, _, _ = self.dash.req("POST", "/api/playground/nonsense", {"name": "x"})
        self.assertEqual(st, 404)

    def test_config_action_accepts_docling_tunables(self):
        """The dashboard's /api/playground/config passes the docling/OCR/table/PDF-backend knobs
        through to the CLI the same way it already does for embedding_model & co (server.py's
        _playground_post, sourced from spec.py's shared registry -- see its own comment)."""
        st, js = self._post("create", {"name": "demo"})
        self.assertTrue(js["ok"], js)

        st, js = self._post("config", {"name": "demo", "ocr": "smart", "table_mode": "fast",
                                       "doc_timeout": 90})
        self.assertEqual(st, 200)
        self.assertTrue(js["ok"], js)
        self.assertEqual(js["result"]["ocr"], "smart")
        self.assertEqual(js["result"]["table_mode"], "fast")
        self.assertEqual(js["result"]["doc_timeout"], 90)

        st, js = self._post("config", {"name": "demo", "ocr": "bogus"})
        self.assertEqual(st, 200)
        self.assertFalse(js["ok"])


class PlaygroundReadOnlyTests(UiBase):
    read_only = True

    def test_read_only_allows_search_list_compare_but_refuses_writes(self):
        st, js, _, _ = self.dash.req("POST", "/api/playground/list", {})
        self.assertEqual(st, 200)
        st, js, _, _ = self.dash.req("POST", "/api/playground/create", {"name": "x"})
        self.assertEqual(st, 403)
        st, js, _, _ = self.dash.req("POST", "/api/playground/index", {"name": "x"})
        self.assertEqual(st, 403)

    def test_read_only_allows_a_promotion_preview_but_refuses_the_real_thing(self):
        st, js, _, _ = self.dash.req("POST", "/api/playground/preview", {"name": "x"})
        self.assertEqual(st, 200)   # a dry-run reads config only; the CLI call reports its own error
        st, js, _, _ = self.dash.req("POST", "/api/playground/promote", {"name": "x"})
        self.assertEqual(st, 403)


class ProductionBridgeApiTests(UiBase):
    def _post(self, action, body):
        st, js, _, _ = self.dash.req("POST", f"/api/playground/{action}", body)
        return st, js

    def test_create_from_production_and_promote_round_trip(self):
        from rag_search import config

        config.update_config(self.paths, "models", {"embedding": "prod/embed"})
        config.update_config(self.paths, "indexer", {"chunk_size": 700})

        st, js = self._post("create", {"name": "bridge", "from_production": True})
        self.assertTrue(js["ok"], js)
        self.assertEqual(js["result"]["config"]["embedding_model"], "prod/embed")
        self.assertEqual(js["result"]["config"]["chunk_size"], 700)

        st, js = self._post("config", {"name": "bridge", "chunk_size": 900})
        self.assertTrue(js["ok"], js)

        st, js = self._post("preview", {"name": "bridge"})
        self.assertTrue(js["ok"], js)
        self.assertEqual(set(js["result"]["changes"]), {"chunk_size"})
        self.assertTrue(js["result"]["needs_reindex"])

        st, js = self._post("promote", {"name": "bridge"})
        self.assertFalse(js["ok"])   # needs confirm: true, same as the CLI needing --confirm

        st, js = self._post("promote", {"name": "bridge", "confirm": True})
        self.assertTrue(js["ok"], js)
        self.assertEqual(set(js["result"]["changes"]), {"chunk_size"})

        cfg, _ = config.load_config(self.paths)
        self.assertEqual(cfg["indexer"]["chunk_size"], 900)

class PlaygroundJsApiCallsTests(unittest.TestCase):
    """Regression guard: core.js's api(path, body) only sends a POST when *body* is passed --
    omit it and it sends a plain GET instead, which 404s against every /api/playground/* route
    (all of them are POST-only, see server.py's _api_post). refreshExperiments() used to call
    api('playground/list') with no second argument, so the Experiments list silently rendered
    empty ("no experiments yet") even when experiments existed on disk and every sibling
    playground call (which does pass a body) worked fine. This statically checks every
    api('playground/...') call site in the JS passes a body, so that bug can't come back
    unnoticed -- a JS-level unit test isn't available in this stdlib-only project."""

    def test_every_playground_api_call_passes_a_body(self):
        js = (info.STATIC / "playground.js").read_text(encoding="utf-8")
        bodyless = re.findall(r"api\(\s*'playground/[^']*'\s*\)", js)
        self.assertEqual(bodyless, [], f"api(...) call(s) with no body argument -- these "
                                       f"silently send a GET and 404: {bodyless}")


class PlaygroundJsForceMdTests(unittest.TestCase):
    """Regression guard: "Build index" used to send only `rebuild` -- there was no way, from the
    dashboard, to make it redo step 1 (docling conversion) of an already-converted document, since
    `rebuild` alone only bypasses the chunk/embed freshness check (core/indexer.py's
    convert_source() separately reuses the cached Markdown whenever the source and the current
    docling settings profile still match). server.py's /api/playground/index and
    core/playground.py's build_index already supported `force_md` end to end; only the UI never
    exposed or sent it. This statically checks playground.js both renders the control and sends it,
    so the gap can't come back unnoticed."""

    def test_build_index_sends_force_md(self):
        js = (info.STATIC / "playground.js").read_text(encoding="utf-8")
        self.assertIn("refs.idxForceMd = h('input', { type: 'checkbox' })", js)
        self.assertIn("force_md: refs.idxForceMd.checked", js)
