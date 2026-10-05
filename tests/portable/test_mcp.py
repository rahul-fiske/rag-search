import asyncio
import json
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

from tests.helpers import SRC, SUBPROC_PYTHONPATH, TempHome
from rag_search import api
from rag_search.mcp import server as mcp_server

try:
    import mcp  # noqa: F401
    HAVE_MCP = True
except ImportError:  # pragma: no cover
    HAVE_MCP = False

AUTH = ("# Authentication\n\n<!-- page 1 -->\n\nSSH access uses public key authentication.\n\n"
        "<!-- page 2 -->\n\n## Roles\n\nRole based access control limits each account.")
DIARY = "# Diary\n\n<!-- page 1 -->\n\nSecret bread thoughts."


class McpCase(TempHome):
    def setUp(self):
        super().setUp()
        self.use_fake_backends()

    def tearDown(self):
        api.daemon_stop(self.paths, "all")
        super().tearDown()


class ToolFunctionTests(McpCase):
    def run_async(self, coro):
        return asyncio.run(coro)

    def test_tools_end_to_end_and_client_identity(self):
        self.write_doc("security/auth.md", AUTH)
        self.write_doc("personal/diary.md", DIARY)
        from rag_search import policy
        policy.save_rules(self.paths, {"personal": ["claude"]})    # agent may not use it
        claude = mcp_server.make_tools("claude")
        agent = mcp_server.make_tools("agent")
        self.assertEqual(sorted(claude), sorted([
            "rag_list_collections", "rag_describe_collection", "rag_search", "rag_grep",
            "rag_index_update", "rag_index_rebuild", "rag_index_status", "rag_index_cancel"]))

        async def scenario():
            started = json.loads(await claude["rag_index_update"]())
            self.assertTrue(started["started"], started)
            again = json.loads(await claude["rag_index_update"]())
            self.assertTrue(again["already_running"] or again["started"])
            st = {}
            for _ in range(60):
                st = json.loads(await claude["rag_index_status"](wait_seconds=10))
                if not st["running"]:
                    break
            self.assertEqual(st["job"]["status"], "succeeded", st)
            self.assertEqual(st["job"]["publish"]["generation"], 1)

            hits = json.loads(await claude["rag_search"]("role based access control"))
            self.assertEqual(hits["results"][0]["page"], "2")
            lst_c = json.loads(await claude["rag_list_collections"]())
            lst_b = json.loads(await agent["rag_list_collections"]())
            self.assertEqual(sorted(c["collection"] for c in lst_c["collections"]),
                             ["personal", "security"])
            self.assertEqual([c["collection"] for c in lst_b["collections"]], ["security"])
            # compact by default: no per-document listing, but a document_count and an
            # (empty, until described) description are always there
            for c in lst_c["collections"]:
                self.assertNotIn("documents", c)
                self.assertEqual(c["document_count"], 1)
                self.assertEqual(c["description"], "")
            full = json.loads(await claude["rag_list_collections"](documents=True))
            sec = next(c for c in full["collections"] if c["collection"] == "security")
            self.assertEqual([d["name"] for d in sec["documents"]], ["auth"])

            # rag_describe_collection: agent may not describe a collection it cannot see
            denied_desc = json.loads(await agent["rag_describe_collection"]("personal", "diary stuff"))
            self.assertEqual(denied_desc["code"], "bad_request")
            self.assertIn("unknown collection", denied_desc["error"])
            # claude may, and rag_list_collections then reports it back (any host, any call)
            set_desc = json.loads(await claude["rag_describe_collection"](
                "security", "Security docs: auth and roles"))
            self.assertEqual(set_desc["description"], "Security docs: auth and roles")
            lst_after = json.loads(await agent["rag_list_collections"]())
            self.assertEqual(lst_after["collections"][0]["description"],
                             "Security docs: auth and roles")
            # clearing with an empty string removes it again
            cleared = json.loads(await claude["rag_describe_collection"]("security", ""))
            self.assertEqual(cleared["description"], "")
            # agent may not use the restricted collection, claude may
            denied = json.loads(await agent["rag_search"]("secret bread", collection="personal"))
            self.assertEqual(denied["code"], "bad_request")
            self.assertIn("unknown collection", denied["error"])
            self.assertNotIn("personal", denied["error"].split("(available")[1])
            st_b = json.loads(await agent["rag_index_status"]())
            self.assertNotIn("personal", json.dumps(st_b))     # progress does not leak it either
            st_c = json.loads(await claude["rag_index_status"]())
            self.assertIn("personal", json.dumps(st_c))
            ok = json.loads(await claude["rag_search"]("secret bread", collection="personal"))
            self.assertTrue(ok["results"])
            g = json.loads(await claude["rag_grep"]("public key"))
            self.assertEqual(g["matches"][0]["doc"], "auth")
            gd = json.loads(await agent["rag_grep"]("Secret", collection="personal"))
            self.assertEqual(gd["code"], "bad_request")
            # rebuild requires confirmation
            refuse = json.loads(await claude["rag_index_rebuild"]())
            self.assertFalse(refuse["started"])
            bad = json.loads(await claude["rag_index_update"](path="nope"))
            self.assertIn("neither a registered location", bad["error"])
            cancel = json.loads(await claude["rag_index_cancel"]())
            self.assertFalse(cancel["cancelled"])

        self.run_async(scenario())

    def test_status_without_daemon_and_wait_is_bounded(self):
        tools = mcp_server.make_tools("claude")
        st = json.loads(asyncio.run(tools["rag_index_status"](wait_seconds=5)))
        self.assertEqual((st["running"], st["job"]), (False, None))


@unittest.skipUnless(HAVE_MCP, "mcp package not installed")
class StdioTests(McpCase):
    def test_stdio_server_prefix_profile_and_calls(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        self.write_doc("security/auth.md", AUTH)
        self.write_doc("personal/diary.md", DIARY)
        from rag_search import policy
        policy.save_rules(self.paths, {"personal": ["claude"]})
        env = {k: os.environ[k] for k in ("RAG_SEARCH_HOME", "RAG_SEARCH_EMBEDDER",
                                          "RAG_SEARCH_RERANKER", "PYTHONPATH")}
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "rag_search.mcp", "--profile", "agent", "--tool-prefix", "kb_"],
            env=env)

        async def scenario():
            async with stdio_client(params) as (r, w):
                async with ClientSession(r, w) as s:
                    await s.initialize()
                    names = sorted(t.name for t in (await s.list_tools()).tools)
                    self.assertIn("kb_rag_search", names)
                    self.assertEqual(len(names), 8)

                    async def call(name, **kw):
                        res = await s.call_tool(name, kw)
                        return json.loads(res.content[0].text)

                    started = await call("kb_rag_index_update")
                    self.assertTrue(started["started"], started)
                    st = {}
                    for _ in range(40):
                        st = await call("kb_rag_index_status", wait_seconds=10)
                        if not st["running"]:
                            break
                    self.assertEqual(st["job"]["status"], "succeeded", st)
                    hits = await call("kb_rag_search", query="public key authentication")
                    self.assertEqual(hits["results"][0]["file"], "auth")
                    lst = await call("kb_rag_list_collections")
                    self.assertEqual([c["collection"] for c in lst["collections"]], ["security"])

        asyncio.run(asyncio.wait_for(scenario(), 120))


class LayeringTests(unittest.TestCase):
    def test_only_the_adapter_imports_mcp(self):
        pat = re.compile(r"^\s*(import mcp\b|from mcp\b)", re.M)
        offenders = []
        for f in (SRC / "rag_search").rglob("*.py"):
            rel = f.relative_to(SRC / "rag_search")
            if rel.parts[0] == "mcp":
                continue
            if pat.search(f.read_text(encoding="utf-8")):
                offenders.append(str(rel))
        self.assertEqual(offenders, [])

    def test_the_adapter_cannot_manage_access(self):
        # hosts only see the outcome; the code that edits rules is never imported by them
        for f in (SRC / "rag_search" / "mcp").glob("*.py"):
            text = f.read_text(encoding="utf-8")
            self.assertNotRegex(text, r"(import|from)\s+\.*(rag_search)?\.?access\b", f.name)
            self.assertNotIn("save_rules", text, f.name)
        names = mcp_server.make_tools("claude")
        self.assertFalse([n for n in names if "access" in n or "restrict" in n or "grant" in n])

    def test_any_profile_name_is_a_client_but_cli_and_unknown_are_not(self):
        from rag_search.mcp.profiles import get_profile
        self.assertEqual(get_profile("newhost").client, "newhost")
        self.assertEqual(get_profile("agent").client, "agent")
        import argparse
        self.assertEqual(mcp_server._profile_name(" NewHost "), "newhost")
        for bad in ("cli", "unknown", "all", "Bad Name!", ""):
            with self.assertRaises(argparse.ArgumentTypeError):
                mcp_server._profile_name(bad)

    def test_adapter_does_not_pull_ml_dependencies(self):
        code = (f"import sys; sys.path[:0] = {SUBPROC_PYTHONPATH.split(os.pathsep)!r}; "
                "import rag_search.mcp.profiles, rag_search.api, rag_search.cli; "
                "try:\n import rag_search.mcp.server\nexcept ImportError:\n pass\n"
                "bad = [m for m in ('numpy', 'torch', 'docling', 'sentence_transformers') "
                "if m in sys.modules]; print(','.join(bad))")
        out = subprocess.run([sys.executable, "-c", code.replace("try:\\n", "try:\n")],
                             capture_output=True, text=True)
        self.assertEqual(out.stdout.strip(), "", out.stderr)

    def test_readme_mentions_every_tool(self):
        # keeps docs honest: tool names documented in the README exist in the adapter
        readme = (Path(__file__).resolve().parents[2] / "README.md")
        if not readme.exists():
            self.skipTest("no README")
        text = readme.read_text(encoding="utf-8")
        for name in mcp_server.make_tools("cli"):
            self.assertIn(name, text, f"{name} missing from README")


if __name__ == "__main__":
    unittest.main()
