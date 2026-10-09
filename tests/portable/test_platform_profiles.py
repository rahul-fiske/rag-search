"""What each kind of machine gets: the platform decisions of rag-search, pinned for four machine profiles.

The profiles are an Apple Silicon Mac, an Intel Mac, a Linux machine without a GPU and a Linux machine with an NVIDIA
card.  The expected values are what the code did before the platform layer existed (``platform.py``): the table is the
guarantee that moving the decisions into one place changed nothing for the Apple Silicon Mac.  Nothing here needs the
machine to be the one it pretends to be: the platform is mocked.
"""

from __future__ import annotations

import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

from tests.helpers import TempHome

from rag_search import machine, models, paths, register, service
from rag_search.core import embedding
from rag_search.core.conversion import applevision, repair, vlm

# name: (sys.platform, platform.system(), platform.machine(), has a GPU torch can use)
PROFILES = {
    "apple_silicon": ("darwin", "Darwin", "arm64", "mps"),
    "intel_mac": ("darwin", "Darwin", "x86_64", "mps"),       # torch reports MPS for an AMD or Intel GPU there
    "linux_cpu": ("linux", "Linux", "x86_64", ""),
    "linux_cuda": ("linux", "Linux", "x86_64", "cuda"),
}


def fake_torch(gpu: str):
    return types.SimpleNamespace(
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: gpu == "mps")),
        cuda=types.SimpleNamespace(is_available=lambda: gpu == "cuda"))


class on:
    """Context manager: the code sees the machine of *profile*."""

    def __init__(self, profile: str, ram_gb: float = 32.0):
        self.sys_platform, self.system, self.machine, self.gpu = PROFILES[profile]
        self.ram_gb = ram_gb
        self.stack = []

    def __enter__(self):
        page = 4096
        sysconf = {"SC_PAGE_SIZE": page, "SC_PHYS_PAGES": int(self.ram_gb * 1024 ** 3 / page)}
        patches = [
            mock.patch.object(sys, "platform", self.sys_platform),
            mock.patch("platform.system", return_value=self.system),
            mock.patch("platform.machine", return_value=self.machine),
            mock.patch.object(os, "sysconf", side_effect=lambda name: sysconf[name]),
            mock.patch.dict(sys.modules, {"torch": fake_torch(self.gpu)}),
            mock.patch.dict(os.environ, {}, clear=False),
        ]
        for p in patches:
            p.start()
            self.stack.append(p)
        for var in ("RAG_SEARCH_DEVICE", "RAG_SEARCH_HOME", "XDG_DATA_HOME"):
            os.environ.pop(var, None)
        return self

    def __exit__(self, *exc):
        for p in reversed(self.stack):
            p.stop()
        return False


class MachineTests(unittest.TestCase):
    """The machine as the light processes (the dashboard, the daemons' status) see it: no torch is imported."""

    EXPECT = {                                    # device, bytes per weight, apple silicon
        "apple_silicon": ("mps", 2, True),
        "intel_mac": ("cpu", 4, False),
        "linux_cpu": ("cpu", 4, False),
        "linux_cuda": ("cpu", 4, False),          # a CUDA card is not assumed without RAG_SEARCH_DEVICE=cuda
    }

    def test_the_machine_of_each_profile(self):
        for name, (device, weight_bytes, apple) in self.EXPECT.items():
            with self.subTest(name), on(name, ram_gb=32):
                info = models.machine_info()
                self.assertEqual((info["device"], info["weight_bytes"], info["ram_gb"]), (device, weight_bytes, 32.0))
                self.assertEqual(models._apple_silicon(), apple)

    def test_a_forced_device_wins(self):
        with on("linux_cuda"), mock.patch.dict(os.environ, {"RAG_SEARCH_DEVICE": "cuda"}):
            info = models.machine_info()
            self.assertEqual((info["device"], info["weight_bytes"]), ("cuda", 2))
        with on("apple_silicon"), mock.patch.dict(os.environ, {"RAG_SEARCH_DEVICE": "cpu"}):
            info = models.machine_info()
            self.assertEqual((info["device"], info["weight_bytes"]), ("cpu", 4))

    def test_the_memory_the_models_may_use_and_what_they_need(self):
        emb, rer = models.find("BAAI/bge-m3"), models.find("BAAI/bge-reranker-v2-m3")
        for name, budget, estimate in (("apple_silicon", 19.2, 4.9), ("intel_mac", 19.2, 6.1)):
            with self.subTest(name), on(name, ram_gb=32):
                m = models.machine_info()
                self.assertEqual(models.budget_gb(m), budget)
                self.assertEqual(models.budget_gb(m, limit_gb=8), 8.0)
                self.assertEqual(models.estimate_gb(emb, rer, m), estimate)


class DeviceTests(unittest.TestCase):
    """The device torch is told to use (the processes that load models)."""

    EXPECT = {"apple_silicon": "mps", "intel_mac": "cpu", "linux_cpu": "cpu", "linux_cuda": "cuda"}

    def test_the_device_of_each_profile(self):
        for name, device in self.EXPECT.items():
            with self.subTest(name), on(name):
                self.assertEqual(embedding.pick_device(), device)

    def test_a_forced_device_wins(self):
        with on("apple_silicon"), mock.patch.dict(os.environ, {"RAG_SEARCH_DEVICE": "cpu"}):
            self.assertEqual(embedding.pick_device(), "cpu")

    def test_half_precision_only_on_a_gpu(self):
        torch = types.SimpleNamespace(float16="fp16", bfloat16="bf16", float32="fp32")
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RAG_SEARCH_DTYPE", None)
            self.assertEqual(embedding.torch_dtype(torch, "mps"), "fp16")
            self.assertEqual(embedding.torch_dtype(torch, "cuda"), "fp16")
            self.assertIsNone(embedding.torch_dtype(torch, "cpu"))


class ReaderTests(TempHome):
    """The document reader (MLX) and the second reader (Apple Vision): who can have them."""

    def test_the_reader_runtime_is_for_apple_silicon_only(self):
        for name, apple in (("apple_silicon", True), ("intel_mac", False), ("linux_cpu", False), ("linux_cuda", False)):
            with self.subTest(name), on(name):
                self.assertEqual(models.runtime_state()["apple_silicon"], apple)

    def test_what_blocks_the_reader_on_each_machine(self):
        for name, blocker in (("intel_mac", "platform"), ("linux_cpu", "platform"), ("linux_cuda", "platform")):
            with self.subTest(name), on(name), mock.patch.dict(os.environ, {"RAG_SEARCH_VLM": "auto"}):
                self.assertEqual(models.vlm_state(self.paths)["runtime"]["apple_silicon"], False)
                row = next(r for r in models.vlm_state(self.paths)["reader"]["models"] if r["active"])
                self.assertEqual(row["state"], "selected_" + blocker)
        with on("apple_silicon"), mock.patch.dict(os.environ, {"RAG_SEARCH_VLM": "auto"}), \
                mock.patch.object(models, "_installed", return_value=True):
            row = next(r for r in models.vlm_state(self.paths)["reader"]["models"] if r["active"])
            self.assertNotEqual(row["state"], "selected_platform")

    def test_the_reader_process_refuses_to_start_off_apple_silicon(self):
        reader = vlm.VlmReader("fake/model", style="instruct", need_gb=1.0)
        for name, uname in (("intel_mac", "x86_64"), ("linux_cpu", "x86_64"), ("linux_cuda", "x86_64")):
            with self.subTest(name), on(name), mock.patch.object(vlm.os, "uname", return_value=types.SimpleNamespace(machine=uname)):
                with self.assertRaises(vlm.ReaderUnavailable) as cm:
                    reader.preflight()
                self.assertIn("Apple Silicon", str(cm.exception))

    def test_apple_vision_needs_a_mac(self):
        for name, mac in (("apple_silicon", True), ("intel_mac", True), ("linux_cpu", False), ("linux_cuda", False)):
            with self.subTest(name), on(name):
                self.assertEqual("needs a Mac" in applevision.why_not(), not mac)
                self.assertEqual("needs a Mac" in repair.OcrMacSecond().why_not(), not mac)


class PlacesTests(unittest.TestCase):
    """Where things live and which service manager exists."""

    def test_the_data_folder(self):
        home = Path.home()
        with on("apple_silicon"):
            self.assertEqual(paths.default_home(), home / "Library" / "Application Support" / "rag-search")
        with on("intel_mac"):
            self.assertEqual(paths.default_home(), home / "Library" / "Application Support" / "rag-search")
        with on("linux_cpu"):
            self.assertEqual(paths.default_home(), home / ".local" / "share" / "rag-search")
        with on("linux_cuda"), mock.patch.dict(os.environ, {"XDG_DATA_HOME": "/data/xdg"}):
            self.assertEqual(paths.default_home(), Path("/data/xdg/rag-search"))

    def test_the_claude_desktop_config(self):
        home = Path.home()
        with on("apple_silicon"):
            self.assertEqual(register.desktop_config_path(), home / "Library/Application Support/Claude/claude_desktop_config.json")
        with on("linux_cpu"):
            self.assertEqual(register.desktop_config_path(), home / ".config/Claude/claude_desktop_config.json")

    def test_start_at_login_services_exist_on_macos_only(self):
        for name, ok in (("apple_silicon", True), ("intel_mac", True), ("linux_cpu", False), ("linux_cuda", False)):
            with self.subTest(name), on(name):
                self.assertEqual(service._require_macos() is None, ok)

    def test_linux_with_systemctl_has_a_service_manager(self):
        for name, manager in (("linux_cpu", "systemd"), ("apple_silicon", "launchd")):
            with self.subTest(name), on(name), mock.patch("shutil.which", return_value="/usr/bin/systemctl"):
                self.assertEqual(machine.service_manager(), manager)
                self.assertIsNone(service._require_macos())


class SetupPlanTests(unittest.TestCase):
    """The steps ``rag-search setup`` takes on each machine."""

    def keys(self, apple: bool, *flags: str) -> list[str]:
        from rag_search import cli

        args = cli.build_parser().parse_args(["setup", *flags])
        return [k for k, _t in cli._setup_plan(args, apple)[0]]

    def test_the_reader_is_downloaded_on_apple_silicon_only(self):
        self.assertEqual(self.keys(True), ["folder", "models", "reader", "tesseract", "doctor", "daemons", "register"])
        self.assertEqual(self.keys(False), ["folder", "models", "tesseract", "doctor", "daemons", "register"])


if __name__ == "__main__":
    unittest.main()
