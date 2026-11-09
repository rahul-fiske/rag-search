"""The menu-bar item's decisions (the icon, the menu, what each item does, staleness) and its login service.
AppKit is not used here: ``menubar.run`` is only the thin layer that draws what these functions decide."""

from __future__ import annotations

import plistlib
import unittest
from unittest import mock

from tests.helpers import TempHome
from tests.portable.test_platform_profiles import on

from rag_search import menubar, service

READY = {"state": "ready", "chunks": 1234, "version": "9.9.9"}
IDLE = {"version": "9.9.9"}


def snap(search=READY, indexer=IDLE, job=None, installed="9.9.9"):
    return {"search": search, "indexer": indexer, "job": job, "dashboard_port": 0, "dashboard_version": "",
            "installed": installed}


class StateTests(unittest.TestCase):
    def test_the_icon_says_what_is_going_on(self):
        self.assertEqual(menubar.icon_state(snap()), "ready")
        self.assertEqual(menubar.icon_state(snap(search={"state": "loading_models", "version": "9.9.9"})), "warming")
        self.assertEqual(menubar.icon_state(snap(job={"phase": "embed", "done": 3, "total": 10})), "indexing")
        self.assertEqual(menubar.icon_state(snap(search=None)), "down")
        self.assertEqual(menubar.icon_state(snap(indexer=None)), "down")
        self.assertTrue(set(menubar.ICONS) == {"ready", "warming", "indexing", "down"})

    def test_the_status_lines(self):
        self.assertEqual(menubar.status_lines(snap()), ["Search ready · 1,234 chunks", "Indexer idle"])
        lines = menubar.status_lines(snap(job={"phase": "convert", "done": 3, "total": 10}))
        self.assertEqual(lines[1], "Indexing: convert 3 of 10")
        self.assertEqual(menubar.status_lines(snap(search=None, indexer=None)),
                         ["Search daemon: not running", "Indexer: not running"])
        warming = menubar.status_lines(snap(search={"state": "loading_models", "version": "9.9.9"}))
        self.assertEqual(warming[0], "Search loading models…")

    def test_daemons_older_than_the_install_are_pointed_out(self):
        lines = menubar.status_lines(snap(installed="10.0.0"))
        self.assertIn("Running 9.9.9; installed 10.0.0: use Restart everything", lines)
        self.assertEqual(len(menubar.status_lines(snap())), 2)


class MenuTests(unittest.TestCase):
    def items(self, s):
        return {m["id"]: m for m in menubar.menu_model(s) if m != "-" and m["id"]}

    def test_the_menu_has_the_common_operations(self):
        ids = set(self.items(snap()))
        self.assertEqual(ids, {"open_dashboard", "index_new", "cancel_index", "restart_all", "restart_dashboard",
                               "start_daemons", "stop_daemons", "open_data", "open_logs", "quit"})

    def test_items_are_enabled_when_they_make_sense(self):
        idle = self.items(snap())
        self.assertTrue(idle["index_new"]["enabled"] and not idle["cancel_index"]["enabled"])
        self.assertFalse(idle["start_daemons"]["enabled"])
        busy = self.items(snap(job={"phase": "embed"}))
        self.assertTrue(busy["cancel_index"]["enabled"] and not busy["index_new"]["enabled"])
        down = self.items(snap(search=None, indexer=None))
        self.assertTrue(down["start_daemons"]["enabled"] and not down["stop_daemons"]["enabled"]
                        and not down["index_new"]["enabled"])
        self.assertTrue(down["open_dashboard"]["enabled"] and down["restart_dashboard"]["enabled"])

    def test_what_would_cancel_a_run_asks_first(self):
        busy = self.items(snap(job={"phase": "embed"}))
        for key in ("restart_all", "stop_daemons"):
            self.assertIn("cancels it", busy[key]["confirm"])
            self.assertFalse(self.items(snap())[key].get("confirm"))
        self.assertFalse(busy["restart_dashboard"].get("confirm"))

    def test_restart_everything_includes_the_dashboard(self):
        self.assertIn("dashboard", self.items(snap())["restart_all"]["title"])


class StalenessTests(unittest.TestCase):
    def test_a_process_older_than_the_files_on_disk_relaunches(self):
        self.assertTrue(menubar.needs_relaunch("1.0.0", "1.0.1"))
        self.assertFalse(menubar.needs_relaunch("1.0.1", "1.0.1"))
        self.assertEqual(menubar.installed_version(), __import__("rag_search").__version__)


class RelaunchTests(TempHome):
    def test_under_launchd_it_exits_non_zero_so_launchd_starts_it_again(self):
        left = []
        with mock.patch.dict("os.environ", {"XPC_SERVICE_NAME": "io.rag-search.menubar"}):
            menubar.relaunch(self.paths, popen=mock.Mock(), leave=left.append)
        self.assertEqual(left, [menubar.RELAUNCH_EXIT])
        self.assertNotEqual(menubar.RELAUNCH_EXIT, 0)

    def test_started_by_hand_a_new_detached_process_takes_over(self):
        left, popen = [], mock.Mock()
        env = {k: v for k, v in __import__("os").environ.items() if k != "XPC_SERVICE_NAME"}
        with mock.patch.dict("os.environ", env, clear=True):
            menubar.relaunch(self.paths, popen=popen, leave=left.append)
        self.assertEqual(left, [0])
        args, kw = popen.call_args
        self.assertEqual(args[0][1:], ["-m", "rag_search.menubar"])
        self.assertTrue(kw["start_new_session"])

    def test_the_process_is_never_replaced_in_place(self):
        import inspect
        self.assertNotIn("execv", inspect.getsource(menubar))


class ActionTests(TempHome):
    def test_restart_everything_restarts_the_daemons_and_then_the_dashboard(self):
        calls = []
        with mock.patch.object(menubar.api, "daemon_restart", side_effect=lambda *a: calls.append("daemons")), \
                mock.patch.object(menubar, "restart_dashboard", side_effect=lambda *a, **k: calls.append("ui") or "dashboard restarted"):
            msg = menubar.act("restart_all", self.paths)
        self.assertEqual(calls, ["daemons", "ui"])
        self.assertIn("dashboard restarted", msg)

    def test_a_stale_dashboard_is_replaced_before_it_is_opened(self):
        from rag_search.ui import server

        opened = []
        with mock.patch.object(server, "running_port", side_effect=[8765, 8765]), \
                mock.patch.object(server, "_ping", return_value={"version": "0.0.1"}), \
                mock.patch.object(server, "stop_running", return_value="stopped") as stop, \
                mock.patch.object(server, "run", return_value=0) as run, \
                mock.patch.object(server, "load_token", return_value="tok"):
            msg = menubar.open_dashboard(self.paths, opener=opened.append)
        stop.assert_called_once()
        self.assertTrue(run.call_args.kwargs["detach"] and not run.call_args.kwargs["open_browser"])
        self.assertEqual(opened, ["http://127.0.0.1:8765/?token=tok"])
        self.assertIn("restarted", msg)

    def test_a_current_dashboard_is_just_opened(self):
        from rag_search import __version__
        from rag_search.ui import server

        opened = []
        with mock.patch.object(server, "running_port", return_value=8765), \
                mock.patch.object(server, "_ping", return_value={"version": __version__}), \
                mock.patch.object(server, "stop_running") as stop, mock.patch.object(server, "run") as run, \
                mock.patch.object(server, "load_token", return_value="tok"):
            msg = menubar.open_dashboard(self.paths, opener=opened.append)
        stop.assert_not_called()
        run.assert_not_called()
        self.assertEqual((msg, opened), ("opened", ["http://127.0.0.1:8765/?token=tok"]))

    def test_no_dashboard_is_started_and_opened(self):
        from rag_search.ui import server

        opened = []
        with mock.patch.object(server, "running_port", side_effect=[0, 8765]), \
                mock.patch.object(server, "run", return_value=0) as run, mock.patch.object(server, "load_token", return_value="t"):
            msg = menubar.open_dashboard(self.paths, opener=opened.append)
        self.assertTrue(run.call_args.kwargs["detach"])
        self.assertEqual((msg, len(opened)), ("started and opened", 1))

    def test_index_and_daemon_items_call_the_same_api_as_the_cli(self):
        with mock.patch.object(menubar.api, "index_start", return_value={"ok": True}) as start:
            self.assertEqual(menubar.act("index_new", self.paths), "indexing started")
        self.assertEqual(start.call_args.kwargs["mode"], "new")
        with mock.patch.object(menubar.api, "index_cancel", return_value={"ok": True}):
            self.assertEqual(menubar.act("cancel_index", self.paths), "indexing cancelled")
        with mock.patch.object(menubar.api, "daemon_stop") as stop:
            menubar.act("stop_daemons", self.paths)
        stop.assert_called_once()
        with mock.patch.object(menubar.api, "daemon_start") as start_d:
            menubar.act("start_daemons", self.paths)
        start_d.assert_called_once()

    def test_a_second_instance_is_not_started_and_stop_finds_the_first(self):
        self.assertEqual(menubar.running_pid(self.paths), 0)
        self.paths.run.mkdir(parents=True, exist_ok=True)
        menubar.pid_file(self.paths).write_text("999999")                  # no such process
        self.assertEqual(menubar.running_pid(self.paths), 0)
        self.assertIn("not running", menubar.stop_running(self.paths))


class ServiceTests(TempHome):
    def test_every_mac_gets_the_menubar_service_and_linux_does_not(self):
        for profile, expect in (("apple_silicon", True), ("intel_mac", True), ("linux_cpu", False)):
            with self.subTest(profile), on(profile):
                self.assertEqual(service.MENUBAR in service.kinds(), expect)
                self.assertEqual(service.kinds()[:2], ("search", "indexer"))

    def test_the_menubar_agent_runs_the_menubar_module_in_the_login_session(self):
        d = service.plist_dict(self.paths, "menubar", "/opt/py/bin/python")
        self.assertEqual(d["Label"], "io.rag-search.menubar")
        self.assertEqual(d["ProgramArguments"], ["/opt/py/bin/python", "-m", "rag_search.menubar"])
        self.assertEqual(d["LimitLoadToSessionType"], "Aqua")
        self.assertTrue(d["RunAtLoad"])
        self.assertNotIn("LimitLoadToSessionType", service.plist_dict(self.paths, "search"))

    def test_the_services_run_the_named_commands_when_the_install_has_them(self):
        bindir = self.tmp / "venv" / "bin"
        bindir.mkdir(parents=True)
        py = bindir / "python"
        py.write_text("")
        for kind, name in (("search", "rag-search-searchd"), ("indexer", "rag-search-indexerd"), ("menubar", "rag-search-menubar")):
            self.assertEqual(service.program(kind, str(py)), [str(py), "-m", service._module(kind)])      # not installed yet
            (bindir / name).write_text("#!/bin/sh\n")
            self.assertEqual(service.program(kind, str(py)), [str(bindir / name)])
            self.assertEqual(service.plist_dict(self.paths, kind, str(py))["ProgramArguments"], [str(bindir / name)])
        self.assertIn(f'ExecStart="{bindir / "rag-search-searchd"}"', service.unit_text(self.paths, "search", str(py)))

    def test_the_commands_are_declared_by_the_package(self):
        import tomllib
        from pathlib import Path as P

        scripts = tomllib.loads((P(__file__).parents[2] / "pyproject.toml").read_text())["project"]["scripts"]
        for kind, name in service.SCRIPTS.items():
            module, _, func = scripts[name].partition(":")
            self.assertEqual((module, func), (service._module(kind), "main"))

    def test_install_writes_three_agents_on_a_mac(self):
        agents = self.tmp / "LaunchAgents"
        fake = mock.Mock(returncode=0, stderr="", stdout="")
        with on("apple_silicon"), mock.patch.object(service, "agents_dir", return_value=agents), \
                mock.patch.object(service, "_launchctl", return_value=fake), \
                mock.patch.object(service.api, "daemon_stop"), mock.patch.object(menubar, "stop_running"):
            out = service.install(self.paths, "/opt/py/bin/python")
        self.assertEqual([line.split(":")[0] for line in out], ["search", "indexer", "menubar"])
        for kind in ("search", "indexer", "menubar"):
            self.assertTrue((agents / f"io.rag-search.{kind}.plist").exists())
        plist = plistlib.loads((agents / "io.rag-search.menubar.plist").read_bytes())
        self.assertEqual(plist["ProgramArguments"][-1], "rag_search.menubar")


if __name__ == "__main__":
    unittest.main()
