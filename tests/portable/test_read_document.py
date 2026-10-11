"""rag_read_document and rag_list_documents: a whole document, byte for byte, found by the names search and grep return."""

from __future__ import annotations

import asyncio
import json
import os
import unittest

from tests.helpers import TempHome

from rag_search import api, policy, readdoc
from rag_search.mcp import server as mcp_server

TABLE = ("---\ntitle: Weekly rollup\nweek: 2026-W41\n---\n\n# Rollup\n\n"
         "| day | km | pace |\n|---|---:|---|\n" + "".join(f"| 2026-10-{d:02d} | {d}.{d % 10}5 | 5:{d:02d} |\n" for d in range(1, 29))
         + "\nSee [the run](../activities/2026-10-11-123.md).\n")
RUN = "# Run 123\n\n| split | time |\n|---|---|\n| 1 | 5:01 |\n| 2 | 4:59 |\n"
CRLF = "a,b\r\n1,2\r\n"


class Base(TempHome):
    """A published collection of Markdown tables and a second, restricted one."""

    def setUp(self):
        super().setUp()
        self.files = {"garmin/rollups/2026-W41.md": TABLE, "garmin/activities/2026-10-11-123.md": RUN,
                      "garmin/log/2026-10-11-123.md": "# Log of the same day\n\nrested\n",
                      "garmin/index.md": "# Index\n\nstart here\n", "garmin/crlf.txt": CRLF,
                      "secret/diary.md": "# Diary\n\nthe Secret table\n"}
        for rel, text in self.files.items():
            p = self.write_doc(rel, "")
            p.write_bytes(text.encode("utf-8"))
        self.index()
        self.publish()

    def read(self, doc, collection="garmin", client="claude", **kw):
        return api.read_document(self.paths, collection, doc, client=client, **kw)

    def text_of(self, doc, **kw):
        r = self.read(doc, **kw)
        self.assertTrue(r["ok"], r)
        return r["result"]


class ReadTests(Base):
    # -- byte-identical and paging ----------------------------------------------------------------
    def test_a_markdown_file_comes_back_exactly_as_written(self):
        res = self.text_of("rollups/2026-W41", max_chars=readdoc.MAX_CHARS)
        self.assertEqual(res["text"], TABLE)
        self.assertEqual(res["text"].encode("utf-8"), TABLE.encode("utf-8"))
        self.assertEqual((res["next_offset"], res["truncated"], res["total_chars"]), (None, False, len(TABLE)))
        self.assertEqual((res["collection"], res["doc"], res["source"]), ("garmin", "rollups/2026-W41", "rollups/2026-W41.md"))
        self.assertTrue(res["indexed_at"])

    def test_line_endings_are_not_normalised(self):
        self.assertEqual(self.text_of("crlf")["text"], CRLF)

    def test_pages_of_any_size_join_into_the_whole_file_with_no_gap_or_overlap(self):
        for size in (1, 7, 50, 999, len(TABLE), len(TABLE) + 5):
            with self.subTest(size=size):
                parts, offset, calls = [], 0, 0
                while offset is not None:
                    res = self.text_of("rollups/2026-W41", offset=offset, max_chars=size)
                    self.assertEqual(res["offset"], offset)
                    self.assertLessEqual(len(res["text"]), size)
                    parts.append(res["text"])
                    offset, calls = res["next_offset"], calls + 1
                    self.assertLess(calls, len(TABLE) + 3)
                self.assertEqual("".join(parts), TABLE)

    def test_the_slice_is_limited_and_a_read_past_the_end_is_empty(self):
        res = self.text_of("rollups/2026-W41", max_chars=10**9)
        self.assertEqual(len(res["text"]), len(TABLE))            # the upper bound still covers this small file
        self.assertEqual(self.text_of("index", offset=10**6)["text"], "")
        self.assertEqual(self.text_of("index", offset=-5)["offset"], 0)
        self.assertEqual(readdoc.MAX_CHARS, 40_000)

    # -- the names ----------------------------------------------------------------------------------
    def test_every_name_the_other_tools_return_works(self):
        want = RUN
        for ident in ("activities/2026-10-11-123",                   # rag_grep doc, catalogue name
                      "activities/2026-10-11-123.md",                # the source path
                      "garmin/activities/2026-10-11-123",            # with the collection in front
                      "./activities/2026-10-11-123.md", "ACTIVITIES/2026-10-11-123"):
            with self.subTest(ident):
                self.assertEqual(self.text_of(ident)["text"], want)
        self.assertEqual(self.text_of("index.md")["text"], "# Index\n\nstart here\n")        # rag_search `source`
        self.assertEqual(self.text_of("index")["doc"], "index")                                # rag_search `file`

    def test_a_bare_name_that_two_documents_share_is_refused_with_the_candidates(self):
        r = self.read("2026-10-11-123.md")
        self.assertFalse(r["ok"])
        self.assertIn("matches 2 documents", r["error"])
        self.assertIn("activities/2026-10-11-123", r["error"])
        self.assertIn("log/2026-10-11-123", r["error"])

    def test_search_results_carry_the_name_to_pass_in(self):
        hits = self.engine().search("Run 123 split time", top_k=5)["results"]
        mine = [h for h in hits if h["collection"] == "garmin"]
        self.assertTrue(mine and all(h["doc"] for h in mine))
        for h in mine:
            self.assertTrue(self.read(h["doc"])["ok"], h)

    def test_an_unknown_document_suggests_close_names(self):
        r = self.read("activities/2026-10-11-124")
        self.assertFalse(r["ok"])
        self.assertIn("Did you mean", r["error"])
        self.assertIn("activities/2026-10-11-123", r["error"])
        self.assertEqual(self.read("")["code"], "bad_request")

    # -- access and confinement ----------------------------------------------------------------------
    def test_a_collection_the_client_may_not_use_is_refused_like_an_unknown_one(self):
        policy.save_rules(self.paths, {"secret": ["claude"]})
        self.assertTrue(self.read("diary", collection="secret", client="claude")["ok"])
        denied = self.read("diary", collection="secret", client="agent")
        unknown = self.read("diary", collection="nonexistent", client="agent")
        self.assertFalse(denied["ok"])
        self.assertEqual(denied["code"], "bad_request")
        self.assertNotIn("secret", denied["error"].split("(available")[0].replace("collection 'secret'", ""))
        self.assertEqual(denied["error"].split("(available")[1], unknown["error"].split("(available")[1])
        listed = api.list_documents(self.paths, "secret", client="agent")
        self.assertFalse(listed["ok"])
        self.assertTrue(api.list_documents(self.paths, "secret", client="claude")["ok"])

    def test_one_collection_exactly(self):
        self.assertIn("one collection", self.read("index", collection="garmin,secret")["error"])
        self.assertIn("which collection", self.read("index", collection="")["error"])

    def test_names_that_try_to_leave_the_collection_are_refused(self):
        for ident in ("../secret/diary", "../../etc/passwd", "/etc/passwd", "activities/../../secret/diary",
                      "..", "a/../b", "index\x00.md", "..\\secret\\diary", "secret/diary", "~/.ssh/id_rsa"):
            with self.subTest(ident):
                r = self.read(ident)
                self.assertFalse(r["ok"], r)
                self.assertNotIn("the Secret table", json.dumps(r))

    def test_a_link_inside_the_markdown_is_never_followed(self):
        gen = self.paths.live_markup() / "garmin"
        outside = self.tmp / "outside.md"
        outside.write_text("OUTSIDE", encoding="utf-8")
        link = gen / "linked.md"
        link.symlink_to(outside)
        docs = [{"name": "linked", "source": "linked.md"}]
        with self.assertRaises(readdoc.DocError):
            readdoc.read(self.paths.live_markup(), "garmin", docs, "linked")
        folder_link = gen / "viaFolder"
        folder_link.symlink_to(self.paths.live_markup() / "secret", target_is_directory=True)
        docs = [{"name": "viaFolder/diary", "source": "diary.md"}]
        with self.assertRaises(readdoc.DocError):
            readdoc.read(self.paths.live_markup(), "garmin", docs, "viaFolder/diary")

    def test_nothing_published_is_said_plainly(self):
        os.environ["RAG_SEARCH_HOME"] = str(self.tmp / "empty-home")
        from rag_search.paths import get_paths
        r = api.read_document(get_paths(), "garmin", "index", client="claude")
        self.assertFalse(r["ok"])

    # -- pages --------------------------------------------------------------------------------------
    def test_a_paged_document_can_be_read_by_page_range(self):
        doc = "<!-- page 1 -->\n\nfirst\n\n<!-- page 2 -->\n\nsecond\n\n<!-- page 3 -->\n\nthird\n"
        self.assertEqual(readdoc.page_slice(doc, "2")[0], "<!-- page 2 -->\n\nsecond\n\n")
        self.assertEqual(readdoc.page_slice(doc, "2-3")[0], "<!-- page 2 -->\n\nsecond\n\n<!-- page 3 -->\n\nthird\n")
        self.assertEqual(readdoc.page_slice(doc, "3-")[0], "<!-- page 3 -->\n\nthird\n")
        self.assertEqual(readdoc.page_slice(doc, "1")[0], "<!-- page 1 -->\n\nfirst\n\n")
        self.assertEqual(readdoc.page_slice(doc, "2")[1], {"pages": "2", "total_pages": 3})
        for bad in ("0", "4", "3-2", "x", "2-4-5"):
            with self.subTest(bad), self.assertRaises(readdoc.DocError):
                readdoc.page_slice(doc, bad)

    def test_a_markdown_source_is_one_page(self):
        res = self.text_of("index", pages="1")
        self.assertEqual((res["text"], res["total_pages"]), ("# Index\n\nstart here\n", 1))
        self.assertFalse(self.read("index", pages="2")["ok"])

    # -- listing ------------------------------------------------------------------------------------
    def listing(self, **kw):
        r = api.list_documents(self.paths, "garmin", client="claude", **kw)
        self.assertTrue(r["ok"], r)
        return r["result"]

    def test_documents_are_listed_by_prefix_with_their_size(self):
        res = self.listing(prefix="rollups/")
        self.assertEqual([d["name"] for d in res["documents"]], ["rollups/2026-W41"])
        d = res["documents"][0]
        self.assertEqual((d["source"], d["chars"]), ("rollups/2026-W41.md", len(TABLE)))
        self.assertTrue(d["indexed_at"])
        self.assertEqual(self.listing(prefix="garmin/activities/")["total_matches"], 1)
        self.assertEqual(self.listing()["total_matches"], 5)

    def test_a_glob_a_limit_and_an_offset_page_through_the_matches(self):
        self.assertEqual([d["name"] for d in self.listing(glob="*2026-10-11-123")["documents"]],
                         ["activities/2026-10-11-123", "log/2026-10-11-123"])
        first = self.listing(limit=2)
        self.assertEqual((first["returned"], first["next_offset"], first["total_matches"]), (2, 2, 5))
        rest = self.listing(limit=2, offset=first["next_offset"])
        last = self.listing(limit=2, offset=4)
        self.assertEqual((rest["returned"], last["returned"], last["next_offset"]), (2, 1, None))
        names = [d["name"] for r in (first, rest, last) for d in r["documents"]]
        self.assertEqual(names, sorted(names))
        self.assertEqual(len(set(names)), 5)

    def test_a_prefix_cannot_leave_the_collection(self):
        r = api.list_documents(self.paths, "garmin", prefix="../secret", client="claude")
        self.assertFalse(r["ok"])


class CliTests(Base):
    def run_cli(self, *argv):
        from tests.portable.test_cli_api import run
        return run(*argv)

    def test_read_prints_the_document_exactly(self):
        rc, out, _err = self.run_cli("read", "garmin", "rollups/2026-W41", "--max-chars", "40000")
        self.assertEqual((rc, out), (0, TABLE))

    def test_a_sliced_read_says_where_to_continue_and_all_prints_everything(self):
        rc, out, err = self.run_cli("read", "garmin", "rollups/2026-W41", "--max-chars", "300")
        self.assertEqual((rc, out), (0, TABLE[:300]))
        self.assertIn("--offset 300", err)
        rc, out, _err = self.run_cli("read", "garmin", "rollups/2026-W41", "--max-chars", "300", "--all")
        self.assertEqual((rc, out), (0, TABLE))

    def test_json_carries_the_documented_fields(self):
        rc, out, _err = self.run_cli("read", "garmin", "index", "--json")
        res = json.loads(out)
        self.assertEqual((rc, res["text"], res["next_offset"], res["doc"]), (0, "# Index\n\nstart here\n", None, "index"))

    def test_a_bad_name_fails_with_a_message(self):
        rc, out, err = self.run_cli("read", "garmin", "../secret/diary")
        self.assertNotEqual(rc, 0)
        self.assertEqual(out, "")
        self.assertIn("not a document", err + out)

    def test_docs_lists_by_prefix(self):
        rc, out, _err = self.run_cli("docs", "garmin", "--prefix", "rollups/")
        self.assertEqual(rc, 0)
        self.assertIn("rollups/2026-W41", out)
        self.assertIn("1 of 1 document(s)", out)
        rc, out, _err = self.run_cli("docs", "garmin", "--glob", "*123", "--json")
        self.assertEqual([d["name"] for d in json.loads(out)["documents"]], ["activities/2026-10-11-123", "log/2026-10-11-123"])


class McpToolTests(Base):
    """The same through the MCP adapter (the tools the host sees)."""

    def call(self, tool, **kw):
        tools = mcp_server.make_tools("claude")
        return json.loads(asyncio.run(tools[tool](**kw)))

    def test_the_tools_are_registered_and_described(self):
        srv = mcp_server.build_server("claude")
        names = {t.name for t in asyncio.run(srv.list_tools())}
        self.assertTrue({"rag_read_document", "rag_list_documents"} <= names, names)
        from rag_search.mcp import profiles
        self.assertIn("rag_read_document", profiles.INSTRUCTIONS)
        self.assertIn("search or grep to locate", profiles.INSTRUCTIONS)

    def test_the_tool_returns_the_documented_fields_and_the_text_exactly(self):
        r = self.call("rag_read_document", collection="garmin", doc="rollups/2026-W41")
        self.assertEqual(set(r), {"collection", "doc", "source", "text", "offset", "next_offset", "total_chars",
                                  "truncated", "indexed_at"})
        self.assertEqual(r["text"], TABLE)
        r = self.call("rag_read_document", collection="garmin", doc="rollups/2026-W41", max_chars=100)
        self.assertEqual((r["offset"], r["next_offset"], r["truncated"], len(r["text"])), (0, 100, True, 100))

    def test_errors_come_back_as_json_errors(self):
        r = self.call("rag_read_document", collection="garmin", doc="../secret/diary")
        self.assertIn("error", r)
        r = self.call("rag_list_documents", collection="garmin", prefix="rollups/")
        self.assertEqual([d["name"] for d in r["documents"]], ["rollups/2026-W41"])


if __name__ == "__main__":
    unittest.main()
