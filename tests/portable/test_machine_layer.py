"""rag_search.machine: what machine this is and what it can do; and that nothing else asks the operating system."""

from __future__ import annotations

import os
import re
import subprocess
import types
import unittest
from pathlib import Path
from unittest import mock

from rag_search import machine
from tests.portable.test_platform_profiles import PROFILES, fake_torch, on

SRC = Path(__file__).resolve().parents[2] / "src" / "rag_search"
# the only files that may ask: this layer, and the conversion script that must not import rag_search
ALLOWED = {"machine.py", "core/docling_convert.py"}
ASKING = re.compile(r"sys\.platform|platform\.(system|machine|mac_ver|processor|uname|architecture)\(|os\.uname\(")


class KindTests(unittest.TestCase):
    def test_the_kind_of_each_profile(self):
        for name, kind in (("apple_silicon", machine.APPLE_SILICON), ("intel_mac", machine.INTEL_MAC),
                           ("linux_cpu", machine.LINUX), ("linux_cuda", machine.LINUX)):
            with self.subTest(name), on(name):
                self.assertEqual(machine.kind(), kind)

    def test_what_each_profile_can_run(self):
        expect = {  # reader backends, apple vision, service manager, package manager
            "apple_silicon": (["mlx"], True, "launchd", "brew"),
            "intel_mac": ([], True, "launchd", "brew"),
            "linux_cpu": ([], False, "", "apt"),
            "linux_cuda": ([], False, "", "apt"),
        }
        for name, (readers, vision, services, packages) in expect.items():
            with self.subTest(name), on(name):
                self.assertEqual((machine.reader_backends(), machine.apple_vision_possible(),
                                  machine.service_manager(), machine.package_manager()), (readers, vision, services, packages))

    def test_the_device(self):
        with on("linux_cuda"):
            self.assertEqual(machine.probe_device(fake_torch("cuda")), "cuda")
            self.assertEqual(machine.light_device(), "cpu")
        with on("intel_mac"):
            self.assertEqual(machine.probe_device(fake_torch("mps")), "cpu")      # an AMD GPU is not used
        with on("apple_silicon"):
            self.assertEqual((machine.light_device(), machine.unified_memory()), ("mps", True))
        with on("linux_cpu"), mock.patch.dict(os.environ, {"RAG_SEARCH_DEVICE": "cuda"}):
            self.assertEqual((machine.light_device(), machine.probe_device(fake_torch(""))), ("cuda", "cuda"))
        self.assertEqual((machine.weight_bytes("mps"), machine.weight_bytes("cuda"), machine.weight_bytes("cpu")), (2, 2, 4))

    def test_describe_is_complete_for_every_profile(self):
        for name in PROFILES:
            with self.subTest(name), on(name):
                d = machine.describe()
                for key in ("kind", "system", "arch", "ram_gb", "device", "weight_bytes", "unified_memory", "cpu_cores",
                            "reader_backends", "apple_vision", "service_manager", "package_manager", "note"):
                    self.assertIn(key, d)
                self.assertTrue(d["note"])
                self.assertGreaterEqual(d["cpu_cores"], 1)


class UnitsAndMemoryTests(unittest.TestCase):
    def test_the_peak_memory_unit(self):
        with on("apple_silicon"):
            self.assertEqual(machine.rss_mb(5 * 1024 * 1024), 5.0)             # bytes on a Mac
        with on("linux_cpu"):
            self.assertEqual(machine.rss_mb(5 * 1024), 5.0)                    # kilobytes elsewhere

    def test_the_free_memory_on_linux_and_on_a_mac(self):
        meminfo = "MemTotal: 16000000 kB\nMemAvailable: 8388608 kB\n"
        with on("linux_cpu"), mock.patch("builtins.open", mock.mock_open(read_data=meminfo)):
            self.assertEqual(machine.available_memory_gb(), 8.0)
        vm = ("Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 65536.\nPages inactive: 65536.\n"
              "Pages speculative: 0.\nPages purgeable: 0.\n")
        with on("apple_silicon"), mock.patch("builtins.open", side_effect=OSError), \
                mock.patch.object(subprocess, "run", return_value=types.SimpleNamespace(stdout=vm)):
            self.assertEqual(machine.available_memory_gb(), 2.0)
        with on("linux_cpu"), mock.patch("builtins.open", side_effect=OSError):
            self.assertIsNone(machine.available_memory_gb())
        with mock.patch.dict(os.environ, {"RAG_SEARCH_VLM_FREE_GB": "3.5"}):
            self.assertEqual(machine.available_memory_gb(), 3.5)

    def test_the_cores_that_do_work(self):
        self.assertGreaterEqual(machine.physical_cores(), 1)
        with on("linux_cpu"), mock.patch("builtins.open", mock.mock_open(
                read_data="physical id\t: 0\ncore id\t: 0\nphysical id\t: 0\ncore id\t: 0\nphysical id\t: 0\ncore id\t: 1\n")):
            self.assertEqual(machine.physical_cores(), 2)                      # three hardware threads, two cores
        with on("linux_cpu"), mock.patch("builtins.open", side_effect=OSError), mock.patch.object(os, "cpu_count", return_value=8):
            self.assertEqual(machine.physical_cores(), 4)


class NobodyElseAsksTests(unittest.TestCase):
    def test_the_operating_system_is_asked_in_machine_py_only(self):
        offenders = []
        for path in sorted(SRC.rglob("*.py")):
            rel = path.relative_to(SRC).as_posix()
            if rel in ALLOWED:
                continue
            for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if ASKING.search(line) and not line.lstrip().startswith("#"):
                    offenders.append(f"{rel}:{n}: {line.strip()[:90]}")
        self.assertEqual(offenders, [], "ask rag_search.machine (add the question there) instead of the operating system")


if __name__ == "__main__":
    unittest.main()
