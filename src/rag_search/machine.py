"""What machine is this, and what can it do?  The one place that asks (standard library only; imports nothing heavy).

rag-search runs on an Apple Silicon Mac, an Intel Mac and (as it is made ready for them) on Linux.  The differences are
few and they were once decided where they were needed -- ``sys.platform == "darwin"`` here, ``platform.machine() ==
"arm64"`` there.  They are decided here now, in terms of *what the machine has* (an Apple GPU, MLX, Apple Vision, a
CUDA card, a service manager, a place for the data), so that a new kind of machine is one new answer in this file and
not a hunt through the code.  A test (``tests/portable/test_machine_layer.py``) keeps the rest of the code from asking
the operating system directly.

The answers are worked out at every call, never cached: they are cheap, and the tests make the machine pretend to be
another one.  ``RAG_SEARCH_DEVICE`` (cpu | mps | cuda) overrides the device everywhere.

Machine kinds:  ``apple_silicon`` (M1 and newer: Apple GPU through Metal and MLX), ``intel_mac`` (CPU only, an older
software stack), ``linux`` (CPU, or an NVIDIA card with ``cuda``), ``other``.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

APPLE_SILICON, INTEL_MAC, LINUX, OTHER = "apple_silicon", "intel_mac", "linux", "other"
DEVICES = ("cpu", "mps", "cuda")


# ── what kind of machine ─────────────────────────────────────────────────────────────────────────────────────────

def system() -> str:
    """``Darwin``, ``Linux``, ``Windows`` ..."""
    return platform.system()


def arch() -> str:
    """``arm64`` or ``x86_64`` (on Linux, ``aarch64`` is reported as it is)."""
    return platform.machine()


def is_mac() -> bool:
    return sys.platform == "darwin"


def is_linux() -> bool:
    return sys.platform.startswith("linux")


def is_windows() -> bool:
    return sys.platform.startswith("win")


def apple_silicon() -> bool:
    return system() == "Darwin" and arch() == "arm64"


def intel_mac() -> bool:
    return system() == "Darwin" and arch() == "x86_64"


def kind() -> str:
    if apple_silicon():
        return APPLE_SILICON
    if intel_mac():
        return INTEL_MAC
    if system() == "Linux":
        return LINUX
    return OTHER


def ram_gb() -> float:
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3, 1)
    except (ValueError, OSError, AttributeError):
        return 0.0


def physical_cores() -> int:
    """Cores that do work at full speed (not hyper-threads): what a CPU-bound library should be told to use."""
    if is_mac():
        try:
            out = subprocess.run(["sysctl", "-n", "hw.physicalcpu"], capture_output=True, text=True, timeout=5).stdout
            return max(1, int(out.strip()))
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    if is_linux():
        try:
            cores = set()
            with open("/proc/cpuinfo", encoding="utf-8") as fh:
                package = core = None
                for line in fh:
                    key, _, val = line.partition(":")
                    if key.strip() == "physical id":
                        package = val.strip()
                    elif key.strip() == "core id":
                        core = val.strip()
                        cores.add((package, core))
            if cores:
                return len(cores)
        except (OSError, ValueError):
            pass
    return max(1, (os.cpu_count() or 2) // 2)


# ── the device the models run on ─────────────────────────────────────────────────────────────────────────────────

def forced_device() -> str:
    return os.environ.get("RAG_SEARCH_DEVICE", "")


def light_device() -> str:
    """The device, without importing torch (the dashboard, the status of a daemon): the Apple GPU on Apple Silicon, else
    the CPU -- a CUDA card is not assumed unless ``RAG_SEARCH_DEVICE=cuda`` says so."""
    forced = forced_device()
    if forced:
        return forced
    return "mps" if apple_silicon() else "cpu"


def probe_device(torch: Any) -> str:
    """The device the models are loaded on, asked of *torch*.  Intel Macs report MPS for an AMD or Intel GPU, but half
    precision there is slow or wrong: the CPU is used."""
    forced = forced_device()
    if forced:
        return forced
    if not intel_mac() and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def weight_bytes(device: str) -> int:
    """Bytes per weight of a model on *device*: half precision on a GPU, full precision on the CPU."""
    return 2 if device in ("mps", "cuda") else 4


def quantize_default() -> str:
    """Should a model that runs on the CPU use 8-bit weights unless told otherwise?  ``int8`` or ``off``.  Measured
    (scripts/retrieval_parity.py, docs/design/platform-support-plan.md): rankings move and the speed gain is unproven
    on real Intel hardware, so every machine keeps full precision; ``models.quantize=int8`` is an opt-in."""
    return "off"


def unified_memory() -> bool:
    """Do the CPU and the GPU share one memory (so the models' size counts against the same RAM)?"""
    return apple_silicon()


# ── what the machine can run ─────────────────────────────────────────────────────────────────────────────────────

def mlx_possible() -> bool:
    """The document reader runs on MLX, which exists for Apple Silicon only."""
    return apple_silicon()


def apple_vision_possible() -> bool:
    """Apple's text recognition (through ocrmac) exists on every Mac, Intel included."""
    return is_mac()


def service_manager() -> str:
    """The program that starts the daemons at login: ``launchd`` on a Mac, ``""`` where none is supported yet."""
    return "launchd" if is_mac() else ""


def package_manager() -> str:
    """The tool that installs system programs (Tesseract): ``brew`` on a Mac, ``apt`` on Linux."""
    return "brew" if is_mac() else ("apt" if is_linux() else "")


def reader_backends() -> list[str]:
    """Reader backends that can work on this machine, best first (see core/conversion/vlm.py)."""
    return ["mlx"] if mlx_possible() else []


# ── memory ───────────────────────────────────────────────────────────────────────────────────────────────────────

def available_memory_gb() -> float | None:
    """Memory that can be used now without swapping, in GB (None when it cannot be told).  ``RAG_SEARCH_VLM_FREE_GB``
    overrides it (tests, and a machine that lies)."""
    env = os.environ.get("RAG_SEARCH_VLM_FREE_GB")
    if env:
        try:
            return float(env)
        except ValueError:
            pass
    try:                                                      # Linux
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024 / 1024
    except (OSError, ValueError, IndexError):
        pass
    if is_mac():                                              # macOS: free + inactive + speculative + purgeable
        try:
            out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
            page = int(out.split("page size of")[1].split()[0])
            n = 0
            for line in out.splitlines():
                key, _, val = line.partition(":")
                if key.strip() in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"):
                    n += int(val.strip().rstrip("."))
            return n * page / 1024 ** 3
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            pass
    return None


def rss_mb(ru_maxrss: float) -> float:
    """Megabytes of a ``resource.getrusage().ru_maxrss`` value (bytes on a Mac, kilobytes elsewhere)."""
    return ru_maxrss / (1024 * 1024) if is_mac() else ru_maxrss / 1024


# ── where things live ────────────────────────────────────────────────────────────────────────────────────────────

def data_home() -> Path:
    """The default data folder (``RAG_SEARCH_HOME`` is read by the caller)."""
    if is_mac():
        return Path.home() / "Library" / "Application Support" / "rag-search"
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
    return base / "rag-search"


def claude_desktop_config() -> Path:
    if is_mac():
        return Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    if is_windows():
        return Path(os.environ.get("APPDATA", "~")).expanduser() / "Claude" / "claude_desktop_config.json"
    return Path.home() / ".config" / "Claude" / "claude_desktop_config.json"


# ── all of it, for people ────────────────────────────────────────────────────────────────────────────────────────

def describe() -> dict[str, Any]:
    """The machine and what it can do, as the dashboard and ``doctor`` show it."""
    device = light_device()
    k = kind()
    notes = {
        APPLE_SILICON: "Apple GPU (Metal) for the models, MLX for the document reader, Apple Vision for OCR.",
        INTEL_MAC: "CPU only. An older software stack (PyTorch 2.2, docling 2.7x): no MLX document reader; Apple Vision, "
                   "docling's OCR and Tesseract read scans.",
        LINUX: "CPU, or an NVIDIA card when RAG_SEARCH_DEVICE=cuda. No Apple Vision, no MLX.",
    }.get(k, "")
    return {"kind": k, "system": system(), "arch": arch(), "ram_gb": ram_gb(), "device": device,
            "weight_bytes": weight_bytes(device), "unified_memory": unified_memory(),
            "cpu_cores": physical_cores(), "reader_backends": reader_backends(),
            "apple_vision": apple_vision_possible(), "service_manager": service_manager(),
            "package_manager": package_manager(), "note": notes}
