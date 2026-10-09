"""Run both daemons at login via launchd (macOS) or a systemd user service (Linux); stdlib only.

`rag-search service install` writes one LaunchAgent (or one systemd user unit) per daemon.  The daemons are
always-on by design; launchd just makes sure they are up after a login or a crash
(KeepAlive restarts on failure, not on a clean `daemon stop`).  Without the service
the daemons still start on demand the first time anything talks to them.
"""

from __future__ import annotations

import os
import plistlib
import subprocess
import sys
import shutil
from pathlib import Path
from typing import Any

from . import api, client, machine
from .paths import Paths

LABEL_PREFIX = "io.rag-search"
def _pass_env() -> tuple[str, ...]:
    """The environment variables a daemon installed as a service keeps from the shell that installed it: every
    variable a tunable or a pipeline stage reads (taken from the registries, so a new one cannot be forgotten), and
    the few that are not tunables.  ``RAG_SEARCH_HOME`` is always set; the client name and the dashboard's port are
    not a daemon's business."""
    from . import spec, stages

    names = {t.env for t in spec.TUNABLES if t.env} | {v for s in stages.ALL for v in s.env_only}
    names |= {"RAG_SEARCH_MODEL", "RAG_SEARCH_RERANK_MODEL", "RAG_SEARCH_RERANK", "RAG_SEARCH_VLM_MODEL",
              "RAG_SEARCH_REPAIR_MODEL", "RAG_SEARCH_DOCLING_PYTHON", "RAG_SEARCH_ALLOW_UNSAFE_TORCH_LOAD",
              "RAG_SEARCH_EMBEDDER", "RAG_SEARCH_RERANKER", "RAG_SEARCH_CONVERT_TIMEOUT", "RAG_SEARCH_JOBS",
              "RAG_SEARCH_IDLE_SECONDS", "RAG_SEARCH_PREWARM", "HF_HOME", "HF_HUB_CACHE"}
    return tuple(sorted(names))


PASS_ENV = _pass_env()


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
    """None when the machine has a service manager rag-search can use (launchd, systemd), else the reason."""
    return None if machine.service_manager() else "start-at-login services need launchd (macOS) or systemd (Linux)"


# ── systemd (a user service per daemon) ──────────────────────────────────────────────────────────────────────────

def unit_name(kind: str) -> str:
    return f"rag-search-{kind}.service"


def unit_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME")
    return (Path(base).expanduser() if base else Path.home() / ".config") / "systemd" / "user"


def unit_path(kind: str) -> Path:
    return unit_dir() / unit_name(kind)


def _q(value: str, command: bool = False) -> str:
    """A value inside double quotes in a unit file (``%`` is expanded everywhere, ``$`` in a command line only)."""
    value = value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")
    return value.replace("$", "$$") if command else value


def unit_text(paths: Paths, kind: str, python: str | None = None) -> str:
    env = {"RAG_SEARCH_HOME": str(paths.home)}
    env.update({k: os.environ[k] for k in PASS_ENV if os.environ.get(k)})
    lines = ["[Unit]", f"Description=rag-search {kind} daemon", "", "[Service]",
             f'ExecStart="{_q(python or sys.executable, True)}" -m {client.DAEMON_MODULES[kind]}',
             f'WorkingDirectory={paths.home}', "Restart=on-failure", "RestartSec=5",
             f"StandardOutput=append:{paths.log_file(kind)}", f"StandardError=append:{paths.log_file(kind)}"]
    lines += [f'Environment="{_q(k)}={_q(v)}"' for k, v in sorted(env.items())]
    return "\n".join(lines + ["", "[Install]", "WantedBy=default.target", ""])


def _systemctl(*args: str) -> subprocess.CompletedProcess:
    cmd = [shutil.which("systemctl") or "systemctl", "--user", *args]
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return subprocess.CompletedProcess(cmd, 124, "", f"systemctl did not answer: {exc}")


def install(paths: Paths, python: str | None = None) -> list[str]:
    if (msg := _require_macos()):
        return [msg]
    from .paths import ensure_dirs

    ensure_dirs(paths)
    out = []
    api.daemon_stop(paths, "all")  # hand over from on-demand daemons to the service manager
    if machine.service_manager() == "systemd":
        unit_dir().mkdir(parents=True, exist_ok=True)
        for kind in api.KINDS:
            unit_path(kind).write_text(unit_text(paths, kind, python), encoding="utf-8")
        _systemctl("daemon-reload")
        for kind in api.KINDS:
            proc = _systemctl("enable", "--now", unit_name(kind))
            out.append(f"{kind}: {'installed and started' if proc.returncode == 0 else 'FAILED'} ({unit_path(kind)})"
                       + ("" if proc.returncode == 0 else f"\n  {proc.stderr.strip()}"))
        return out
    agents_dir().mkdir(parents=True, exist_ok=True)
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
    if machine.service_manager() == "systemd":
        for kind in api.KINDS:
            if not unit_path(kind).exists():
                out.append(f"{kind}: not installed")
                continue
            _systemctl("disable", "--now", unit_name(kind))
            unit_path(kind).unlink()
            out.append(f"{kind}: removed")
        _systemctl("daemon-reload")
        api.daemon_stop(paths, "all")
        return out
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
        if machine.service_manager() == "systemd":
            entry: dict[str, Any] = {"unit": str(unit_path(kind)), "installed": unit_path(kind).exists()}
            entry["loaded"] = _systemctl("is-active", unit_name(kind)).returncode == 0
        else:
            entry = {"plist": str(plist_path(kind)), "installed": plist_path(kind).exists()}
            if machine.service_manager() == "launchd":
                entry["loaded"] = _launchctl("print", f"{_domain()}/{label(kind)}").returncode == 0
        info = client.ping(paths, kind)
        entry["running"] = bool(info)
        if info:
            entry["pid"] = info.get("pid")
        res[kind] = entry
    return res
