"""launchd service, MCP registration helpers, the MCP adapter's entry points and error replies.  Tier A:
`launchctl`, the `claude` CLI and the `mcp` server object are the seams; files are real, in a temporary folder."""

from __future__ import annotations

import asyncio
import json
import subprocess
import unittest
from unittest import mock

from tests.helpers import TempHome

from rag_search import api, register, service
from rag_search.mcp import server as mcp_server


def done(rc=0, out="", err=""):
    return subprocess.CompletedProcess([], rc, out, err)


class ServiceTests(TempHome):
    """The launchd agents, on a pretend Mac: the plist files are real, `launchctl` is a recorder."""

    def setUp(self):
        super().setUp()
        self.agents = self.tmp / "LaunchAgents"
        self.calls: list[tuple[str, ...]] = []
        self.replies: dict[str, subprocess.CompletedProcess] = {}

        def launchctl(*args):
            self.calls.append(args)
            return self.replies.get(args[0], done())

        for p in (mock.patch.object(service, "agents_dir", return_value=self.agents),
                  mock.patch.object(service.sys, "platform", "darwin"),
                  mock.patch.object(service, "_launchctl", side_effect=launchctl),
                  mock.patch.object(api, "daemon_stop", return_value={})):
            p.start()
            self.addCleanup(p.stop)

    def test_install_writes_one_plist_per_daemon_and_boots_it(self):
        out = service.install(self.paths, python="/venv/bin/python")
        self.assertEqual([line.split(":")[0] for line in out], list(api.KINDS))
        self.assertTrue(all("installed and started" in line for line in out))
        for kind in api.KINDS:
            plist = service.plistlib.loads(service.plist_path(kind).read_bytes())
            self.assertEqual(plist["ProgramArguments"][0], "/venv/bin/python")
            self.assertEqual(plist["EnvironmentVariables"]["RAG_SEARCH_HOME"], str(self.paths.home))
        self.assertIn("bootstrap", {c[0] for c in self.calls})

    def test_an_older_macos_falls_back_to_load_and_a_failure_is_reported_with_its_reason(self):
        self.replies["bootstrap"] = done(1, "", "no such verb")
        out = service.install(self.paths)
        self.assertIn("load", {c[0] for c in self.calls})
        self.assertTrue(all("installed and started" in line for line in out))       # `load` worked
        self.replies["load"] = done(5, "", "Load failed: 5")
        out = service.install(self.paths)
        self.assertTrue(all("FAILED" in line and "Load failed: 5" in line for line in out))

    def test_uninstall_removes_what_is_installed_and_says_what_is_not(self):
        self.assertEqual(service.uninstall(self.paths), [f"{k}: not installed" for k in api.KINDS])
        service.install(self.paths)
        self.replies["bootout"] = done(3)                      # not loaded any more: unload the file instead
        out = service.uninstall(self.paths)
        self.assertEqual(out, [f"{k}: removed" for k in api.KINDS])
        self.assertIn("unload", {c[0] for c in self.calls})
        self.assertFalse(any(service.plist_path(k).exists() for k in api.KINDS))

    def test_status_says_installed_loaded_and_running_with_the_pid(self):
        service.install(self.paths)
        self.replies["print"] = done(0)
        with mock.patch.object(service.client, "ping", side_effect=lambda p, kind: {"pid": 321} if kind == "search" else None):
            st = service.status(self.paths)
        self.assertEqual((st["search"]["installed"], st["search"]["loaded"], st["search"]["running"], st["search"]["pid"]),
                         (True, True, True, 321))
        self.assertEqual((st["indexer"]["running"], "pid" in st["indexer"]), (False, False))

    def test_the_launchd_domain_is_the_users(self):
        self.assertTrue(service._domain().startswith("gui/"))


class RealLaunchctlWrapperTests(unittest.TestCase):
    def test_it_runs_launchctl_with_the_given_arguments(self):
        with mock.patch.object(service.subprocess, "run", return_value=done()) as run:
            service._launchctl("print", "gui/1/x")
        self.assertEqual(run.call_args.args[0], ["launchctl", "print", "gui/1/x"])


class RegisterHelperTests(TempHome):
    def test_the_adapter_command_prefers_the_sibling_then_the_path_then_python_dash_m(self):
        bin_dir = self.tmp / "venv" / "bin"
        bin_dir.mkdir(parents=True)
        py = bin_dir / "python"
        with mock.patch.object(register.sys, "executable", str(py)):
            (bin_dir / "rag-search-mcp").write_text("#!/bin/sh\n")
            cmd, args = register.adapter_command("agent", "pre_")
            self.assertEqual((cmd, args), (str(bin_dir / "rag-search-mcp"), ["--profile", "agent", "--tool-prefix", "pre_"]))
            (bin_dir / "rag-search-mcp").unlink()
            with mock.patch.object(register.shutil, "which", return_value="/usr/local/bin/rag-search-mcp"):
                self.assertEqual(register.adapter_command()[0], "/usr/local/bin/rag-search-mcp")
            with mock.patch.object(register.shutil, "which", return_value=None):
                self.assertEqual(register.adapter_command(), (str(py), ["-m", "rag_search.mcp", "--profile", "claude"]))

    def test_optional_host_modules_are_found_and_a_broken_or_incomplete_one_is_skipped(self):
        good = mock.Mock(NAME="zeta", spec=["NAME", *register._REQUIRED])
        for attr in register._REQUIRED:
            setattr(good, attr, lambda *a, **k: None)
        incomplete = mock.Mock(spec=["NAME"], NAME="alpha")

        def import_module(name):
            if name == "rag_search":
                import rag_search
                return rag_search
            if name.endswith("hosts_good"):
                return good
            if name.endswith("hosts_incomplete"):
                return incomplete
            raise ImportError("broken optional module")

        infos = [mock.Mock(), mock.Mock(), mock.Mock(), mock.Mock()]
        for i, n in zip(infos, ("hosts_good", "hosts_incomplete", "hosts_broken", "unrelated")):
            i.name = n
        saved = register._EXTRAS
        register._EXTRAS = None
        self.addCleanup(setattr, register, "_EXTRAS", saved)
        with mock.patch.object(register.pkgutil, "iter_modules", return_value=infos), \
                mock.patch.object(register.importlib, "import_module", side_effect=import_module):
            self.assertEqual(register.extra_hosts(), [good])

    def test_has_entry_registered_hosts_and_the_json_loader(self):
        self.assertFalse(register.has_entry(None))
        self.assertFalse(register.has_entry(self.tmp / "missing.json"))
        bad = self.tmp / "bad.json"
        bad.write_text("{ nope")
        self.assertFalse(register.has_entry(bad))
        blank = self.tmp / "blank.json"
        blank.write_text("  \n")
        self.assertEqual(register._load(blank), {})
        self.assertEqual(register._load(self.tmp / "absent.json"), {})
        listed = self.tmp / "list.json"
        listed.write_text("[1]")
        with self.assertRaises(ValueError):
            register._load(listed)
        ok = self.tmp / "ok.json"
        ok.write_text(json.dumps({"mcpServers": {register.SERVER_NAME: {}}}))
        self.assertTrue(register.has_entry(ok))
        host = mock.Mock()
        host.find_config.side_effect = OSError("denied")
        with mock.patch.object(register, "extra_hosts", return_value=[host]):
            self.assertEqual(register.registered_hosts(), [])

    def test_unregistering_from_a_damaged_config_leaves_it_alone(self):
        bad = self.tmp / "bad.json"
        bad.write_text("{ nope")
        msg = register._remove(bad, "Demo")
        self.assertIn("not valid JSON", msg)
        self.assertEqual(bad.read_text(), "{ nope")

    def test_the_desktop_config_lives_where_each_platform_keeps_it(self):
        with mock.patch.object(register.sys, "platform", "darwin"):
            self.assertIn("Application Support", str(register.desktop_config_path()))
        with mock.patch.object(register.sys, "platform", "win32"), mock.patch.dict("os.environ", {"APPDATA": str(self.tmp)}):
            self.assertEqual(register.desktop_config_path(), self.tmp / "Claude" / "claude_desktop_config.json")
        with mock.patch.object(register.sys, "platform", "linux"):
            self.assertIn(".config", str(register.desktop_config_path()))

    def test_claude_code_registration_reports_a_missing_cli_a_failure_and_success(self):
        with mock.patch.object(register.shutil, "which", return_value=None):
            self.assertIn("not found on PATH", register.register_code(str(self.tmp)))
            self.assertIn("nothing to do", register.unregister_code())
        with mock.patch.object(register.shutil, "which", return_value="/bin/claude"), \
                mock.patch.object(register.subprocess, "run", side_effect=[done(), done(1, "", "boom")]):
            msg = register.register_code()
        self.assertIn("registration failed: boom", msg)
        with mock.patch.object(register.shutil, "which", return_value="/bin/claude"), \
                mock.patch.object(register.subprocess, "run", return_value=done()) as run:
            self.assertIn("registered", register.register_code(str(self.tmp)))
        self.assertIn("-e", run.call_args.args[0])                   # the data folder travels with it
        with mock.patch.object(register.shutil, "which", return_value="/bin/claude"), \
                mock.patch.object(register.subprocess, "run", return_value=done(0, "Removed", "")):
            self.assertEqual(register.unregister_code(), "Claude Code: Removed")

    def test_a_status_row_for_a_damaged_config_is_a_failure(self):
        bad = self.tmp / "bad.json"
        bad.write_text("{ nope")
        self.assertEqual(register._status_row("Demo", bad, "rag-search register")[1], "fail")


class McpAdapterTests(TempHome):
    def setUp(self):
        super().setUp()
        self.use_fake_backends()

    def call(self, tool, **kw):
        tools = mcp_server.make_tools("claude")
        return json.loads(asyncio.run(tools[tool](**kw)))

    def test_a_daemon_that_is_still_starting_says_so_and_other_errors_carry_their_code(self):
        r = mcp_server._error({"code": "warming_up", "error": "loading", "state": "loading_models"}, query="q")
        self.assertEqual((json.loads(r)["status"], json.loads(r)["query"]), ("warming_up", "q"))
        r = json.loads(mcp_server._error({"code": "forbidden", "error": "no", "log": "/x.log"}))
        self.assertEqual((r["code"], r["error"], r["log"]), ("forbidden", "no", "/x.log"))

    def test_rebuilding_needs_confirmation_and_a_status_error_is_reported(self):
        r = self.call("rag_index_rebuild")
        self.assertFalse(r["started"])
        self.assertIn("confirm=true", r["error"])
        with mock.patch.object(api, "index_start", return_value={"ok": True, "job": {"id": "j"}}) as start:
            r = self.call("rag_index_rebuild", confirm=True, path="docs/x")
        self.assertIn("rag_index_status", r["next"])
        self.assertEqual(start.call_args.kwargs.get("mode"), "all")
        with mock.patch.object(api, "index_status", return_value={"ok": False, "error": "nope", "code": "internal"}):
            self.assertEqual(self.call("rag_index_status")["code"], "internal")

    def test_the_server_registers_every_tool_under_the_prefix(self):
        srv = mcp_server.build_server("agent", tool_prefix="rs_")
        names = {t.name for t in asyncio.run(srv.list_tools())}
        self.assertTrue(names and all(n.startswith("rs_") for n in names), names)
        self.assertIn("rs_rag_search", names)

    def test_main_sets_the_data_folder_and_runs_on_stdio_or_says_the_adapter_is_missing(self):
        fake = mock.Mock()
        with mock.patch.object(mcp_server, "build_server", return_value=fake) as build:
            mcp_server.main(["--home", str(self.tmp / "h"), "--profile", "agent", "--tool-prefix", "p_",
                             "--client-id", "me"])
        fake.run.assert_called_once_with(transport="stdio")
        self.assertEqual(build.call_args.args, ("agent", "p_", "me"))
        import os
        self.assertEqual(os.environ["RAG_SEARCH_HOME"], str(self.tmp / "h"))
        with mock.patch.object(mcp_server, "build_server", side_effect=ImportError("no mcp")), \
                self.assertRaises(SystemExit) as cm:
            mcp_server.main([])
        self.assertEqual(cm.exception.code, 1)
        with self.assertRaises(SystemExit) as cm:
            mcp_server.main(["--version"])
        self.assertEqual(cm.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
