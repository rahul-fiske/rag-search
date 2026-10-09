"""The systemd user-service backend of ``rag-search service`` (a Linux machine; the commands are faked)."""

from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path
from unittest import mock

from tests.helpers import TempHome
from tests.portable.test_platform_profiles import on

from rag_search import machine, service


def ok(*a, **k):
    return subprocess.CompletedProcess(a, 0, "", "")


class SystemdTests(TempHome):
    def setUp(self):
        super().setUp()
        self.cfg = self.tmp / "xdgconf"
        self.env = mock.patch.dict(os.environ, {"XDG_CONFIG_HOME": str(self.cfg)})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_the_unit_runs_the_daemon_with_its_environment(self):
        text = service.unit_text(self.paths, "search", "/opt/py/bin/python")
        self.assertIn('ExecStart="/opt/py/bin/python" -m rag_search.core.search_daemon', text)
        self.assertIn(f'Environment="RAG_SEARCH_HOME={self.paths.home}"', text)
        self.assertIn("Restart=on-failure", text)
        self.assertIn("WantedBy=default.target", text)

    def test_special_characters_are_escaped(self):
        self.assertEqual(service._q('a"b%c$d', True), 'a\\"b%%c$$d')
        self.assertEqual(service._q("a$b"), "a$b")

    def test_install_and_uninstall_write_and_remove_the_units(self):
        calls = []

        def fake(*args):
            calls.append(args)
            return ok()

        with on("linux_cpu"), mock.patch.object(machine, "_which", return_value="/usr/bin/systemctl"), \
                mock.patch.object(service, "_systemctl", side_effect=fake), mock.patch.object(service.api, "daemon_stop"):
            out = service.install(self.paths, "/opt/py/bin/python")
            self.assertTrue(all("installed and started" in line for line in out), out)
            for kind in service.api.KINDS:
                self.assertTrue(service.unit_path(kind).exists())
            self.assertIn(("daemon-reload",), calls)
            self.assertTrue(any(c[0] == "enable" and c[1] == "--now" for c in calls))
            out = service.uninstall(self.paths)
            self.assertTrue(all(line.endswith("removed") for line in out), out)
            for kind in service.api.KINDS:
                self.assertFalse(service.unit_path(kind).exists())

    def test_the_unit_folder_follows_xdg(self):
        self.assertEqual(service.unit_dir(), Path(self.cfg) / "systemd" / "user")


if __name__ == "__main__":
    unittest.main()
