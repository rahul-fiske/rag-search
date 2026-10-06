"""Run both daemons at login via launchd (macOS, stdlib only).

`rag-search service install` writes one LaunchAgent per daemon.  The daemons are
always-on by design; launchd just makes sure they are up after a login or a crash
(KeepAlive restarts on failure, not on a clean `daemon stop`).  Without the service
the daemons still start on demand the first time anything talks to them.
"""

from __future__ import annotations

import os
import plistlib
import subprocess
import sys
from pathlib import Path
from typing import Any

from . import api, client
from .paths import Paths

LABEL_PREFIX = "io.rag-search"
PASS_ENV = ("RAG_SEARCH_MODEL", "RAG_SEARCH_RERANK_MODEL", "RAG_SEARCH_RERANK",
            "RAG_SEARCH_DEVICE", "RAG_SEARCH_OCR", "RAG_SEARCH_OCR_ENGINE", "RAG_SEARCH_OCR_LANG",
            "RAG_SEARCH_TABLE_MODE", "RAG_SEARCH_PIPELINE", "RAG_SEARCH_ROUTING", "RAG_SEARCH_PDF_BACKEND", "RAG_SEARCH_THREADS", "RAG_SEARCH_DOC_TIMEOUT",
            "RAG_SEARCH_DOCLING_PYTHON", "HF_HOME",
            "RAG_SEARCH_EMBED_BATCH", "RAG_SEARCH_MAX_SEQ",
            "RAG_SEARCH_ALLOW_UNSAFE_TORCH_LOAD",
            "RAG_SEARCH_EMBEDDER", "RAG_SEARCH_RERANKER", "RAG_SEARCH_DTYPE",
            "RAG_SEARCH_RERANK_MAX_LEN", "RAG_SEARCH_RERANK_BATCH", "RAG_SEARCH_DOCLING_BATCH",
            "RAG_SEARCH_CONVERT_TIMEOUT")


def label(kind: str) -> str:
    return f"{LABEL_PREFIX}.{kind}"


def agents_dir() -> Path:
    return Path.home() / "Library" / "LaunchAgents"


def plist_path(kind: str) -> Path:
    return agents_dir() / f"{label(kind)}.plist"


def plist_dict(paths: Paths, kind: str, python: str | None = None) -> dict[str, Any]:
    env = {"RAG_SEARCH_HOME": str(paths.home),
           "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"}
    env.update({k: os.environ[k] for k in PASS_ENV if os.environ.get(k)})
    return {
        "Label": label(kind),
        "ProgramArguments": [python or sys.executable, "-m", client.DAEMON_MODULES[kind]],
        "EnvironmentVariables": env,
        "WorkingDirectory": str(paths.home),
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ProcessType": "Interactive",
        "StandardOutPath": str(paths.log_file(kind)),
        "StandardErrorPath": str(paths.log_file(kind)),
    }


def _domain() -> str:
    return f"gui/{os.getuid()}"


def _launchctl(*args: str) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(["launchctl", *args], capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(["launchctl", *args], 124, "", "launchctl did not answer within 60 seconds")


def _require_macos() -> str | None:
    return None if sys.platform == "darwin" else "launchd services are macOS-only"


def install(paths: Paths, python: str | None = None) -> list[str]:
    if (msg := _require_macos()):
        return [msg]
    from .paths import ensure_dirs

    ensure_dirs(paths)
    agents_dir().mkdir(parents=True, exist_ok=True)
    out = []
    api.daemon_stop(paths, "all")  # hand over from on-demand daemons to launchd
    for kind in api.KINDS:
        pf = plist_path(kind)
        _launchctl("bootout", f"{_domain()}/{label(kind)}")  # ignore "not loaded"
        pf.write_bytes(plistlib.dumps(plist_dict(paths, kind, python)))
        proc = _launchctl("bootstrap", _domain(), str(pf))
        if proc.returncode != 0:  # older macOS
            proc = _launchctl("load", "-w", str(pf))
        out.append(f"{kind}: {'installed and started' if proc.returncode == 0 else 'FAILED'}"
                   f" ({pf})" + ("" if proc.returncode == 0 else f"\n  {proc.stderr.strip()}"))
    return out


def uninstall(paths: Paths) -> list[str]:
    if (msg := _require_macos()):
        return [msg]
    out = []
    for kind in api.KINDS:
        pf = plist_path(kind)
        if not pf.exists():
            out.append(f"{kind}: not installed")
            continue
        proc = _launchctl("bootout", f"{_domain()}/{label(kind)}")
        if proc.returncode != 0:
            _launchctl("unload", str(pf))
        pf.unlink()
        out.append(f"{kind}: removed")
    api.daemon_stop(paths, "all")
    return out


def status(paths: Paths) -> dict[str, Any]:
    res: dict[str, Any] = {}
    for kind in api.KINDS:
        entry: dict[str, Any] = {"plist": str(plist_path(kind)),
                                 "installed": plist_path(kind).exists()}
        if sys.platform == "darwin":
            entry["loaded"] = _launchctl("print", f"{_domain()}/{label(kind)}").returncode == 0
        info = client.ping(paths, kind)
        entry["running"] = bool(info)
        if info:
            entry["pid"] = info.get("pid")
        res[kind] = entry
    return res
