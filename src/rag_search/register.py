"""Register the MCP adapter with Claude Desktop and Claude Code (stdlib only), and with any
other host that has an optional ``hosts_*.py`` module next to this file.

The adapter (`rag-search-mcp`) is launched by the host as a stdio server; it is only a
thin client of the daemons, so every host shares the same data folder, indexes and
warm models.  Each host gets its own identity (`--profile`) so `rag-search access` can give each host its
own set of collections.
"""

from __future__ import annotations

import importlib
import json
import os
import pkgutil
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import machine

SERVER_NAME = "rag-search"


def adapter_command(profile: str = "claude", tool_prefix: str = "") -> tuple[str, list[str]]:
    """Absolute command that launches the MCP adapter.

    GUI apps do not inherit the shell PATH, so a bare 'rag-search-mcp' would not
    resolve; always register an absolute path.
    """
    args = ["--profile", profile]
    if tool_prefix:
        args += ["--tool-prefix", tool_prefix]
    sibling = Path(sys.executable).parent / "rag-search-mcp"
    if sibling.exists():
        return str(sibling), args
    found = shutil.which("rag-search-mcp")
    if found:
        return str(Path(found).absolute()), args
    return sys.executable, ["-m", "rag_search.mcp", *args]


def server_entry(home: str | None = None, profile: str = "claude", tool_prefix: str = "",
                 extra: dict | None = None) -> dict:
    cmd, args = adapter_command(profile, tool_prefix)
    entry: dict = {"command": cmd, "args": args}
    env = {}
    if home:
        env["RAG_SEARCH_HOME"] = str(Path(home).expanduser().absolute())
    if env:
        entry["env"] = env
    if extra:                       # host-specific keys (an optional host module supplies them)
        entry.update(extra)
    return entry


def mcp_config_snippet(home: str | None = None, profile: str = "claude",
                       tool_prefix: str = "", extra: dict | None = None) -> str:
    return json.dumps({"mcpServers": {SERVER_NAME: server_entry(home, profile, tool_prefix, extra)}},
                      indent=2)


# ── optional hosts ───────────────────────────────────────────────────────────
# A module ``rag_search/hosts_<name>.py`` adds one more MCP host.  It is discovered by its file
# name, so nothing else in the package needs to know it exists (a build without the file simply
# has no such host).  The module provides: NAME (profile / client identity), LABEL, EXTRA_ENTRY
# (extra keys for the server entry), installed(), default_config(), find_config(),
# register(home, path, tool_prefix) and unregister(path).  Three optional switches (all False
# when absent) say how the host behaves:
#   AUTO_REGISTER  a plain `rag-search register` also registers it when installed() (skip it
#                  with --no-<name>)
#   ALWAYS_LISTED  it is always shown (doctor, access lists, dashboard), like Claude; otherwise
#                  only once it is registered
#   ADVERTISE      its flags appear in --help

_REQUIRED = ("NAME", "LABEL", "installed", "default_config", "find_config", "register", "unregister")
_EXTRAS: list | None = None


def extra_hosts() -> list:
    global _EXTRAS
    if _EXTRAS is None:
        found = []
        pkg = importlib.import_module(__package__ or "rag_search")
        for info in pkgutil.iter_modules(pkg.__path__):
            if not info.name.startswith("hosts_"):
                continue
            try:
                mod = importlib.import_module(f"{pkg.__name__}.{info.name}")
            except Exception:  # noqa: BLE001 - a broken optional module must not break the CLI
                continue
            if all(hasattr(mod, a) for a in _REQUIRED):
                found.append(mod)
        _EXTRAS = sorted(found, key=lambda m: m.NAME)
    return _EXTRAS


def has_entry(path: Path | None) -> bool:
    """Is rag-search registered in this JSON config file?"""
    if path is None or not path.is_file():
        return False
    try:
        return SERVER_NAME in (_load(path).get("mcpServers") or {})
    except ValueError:
        return False


def registered_hosts() -> list:
    """Optional host modules that are registered right now."""
    out = []
    for h in extra_hosts():
        try:
            if has_entry(h.find_config()):
                out.append(h)
        except OSError:
            continue
    return out


def shown_hosts() -> list:
    """Optional hosts to show: those that are always listed, and the others once registered."""
    reg = {h.NAME for h in registered_hosts()}
    return [h for h in extra_hosts() if getattr(h, "ALWAYS_LISTED", False) or h.NAME in reg]


def host_clients() -> tuple[str, ...]:
    """Client names of the hosts to show: Claude, plus the optional hosts that are shown."""
    return ("claude", *(h.NAME for h in shown_hosts()))


def known_profiles() -> tuple[str, ...]:
    return ("claude", *(h.NAME for h in extra_hosts()))


# ── JSON config files (Claude Desktop and the optional hosts) ────────────────

def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        return {}
    data = json.loads(raw)  # ValueError -> caller reports, file untouched
    if not isinstance(data, dict):
        raise ValueError("config root is not a JSON object")
    return data


def _write(path: Path, data: dict) -> str:
    """Write *data* atomically, keeping permissions, symlinks and every earlier backup."""
    real = path.resolve()  # a symlinked (dotfiles) config stays a symlink
    real.parent.mkdir(parents=True, exist_ok=True)
    backup = ""
    if real.exists():
        stamp = time.strftime("%Y%m%d-%H%M%S")
        backup = f"{real}.bak-{stamp}"
        n = 1
        while os.path.exists(backup):  # never overwrite an earlier backup
            n += 1
            backup = f"{real}.bak-{stamp}-{n}"
        shutil.copy2(real, backup)
    tmp = real.with_name(real.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    if real.exists():
        shutil.copymode(real, tmp)  # e.g. keep 0600 on files that hold other servers' tokens
    os.replace(tmp, real)
    return backup


def _add(path: Path, entry: dict, label: str, restart_hint: str, snippet: str) -> str:
    try:
        data = _load(path)
    except ValueError as exc:
        return (f"! {path} is not valid JSON ({exc}); left untouched. Add this by hand:\n"
                + snippet)
    data.setdefault("mcpServers", {})[SERVER_NAME] = entry
    backup = _write(path, data)
    msg = f"{label}: registered '{SERVER_NAME}' in {path}"
    if backup:
        msg += f" (backup: {backup})"
    return msg + "\n  " + restart_hint


def _remove(path: Path, label: str) -> str:
    try:
        data = _load(path)
    except ValueError as exc:
        return f"! {path} is not valid JSON ({exc}); left untouched."
    servers = data.get("mcpServers", {})
    if SERVER_NAME not in servers:
        return f"{label}: not registered."
    del servers[SERVER_NAME]
    backup = _write(path, data)
    return f"{label}: removed '{SERVER_NAME}' (backup: {backup})."


# ── Claude Desktop ───────────────────────────────────────────────────────────

def desktop_config_path() -> Path:
    return machine.claude_desktop_config()


def register_desktop(home: str | None = None, tool_prefix: str = "") -> str:
    return _add(desktop_config_path(), server_entry(home, "claude", tool_prefix),
                "Claude Desktop", "Fully quit and reopen Claude Desktop to load it.",
                mcp_config_snippet(home, "claude", tool_prefix))


def unregister_desktop() -> str:
    return _remove(desktop_config_path(), "Claude Desktop")


# ── Claude Code ──────────────────────────────────────────────────────────────

def _claude(cmd: list[str]) -> subprocess.CompletedProcess:
    """Run the ``claude`` CLI without a terminal to wait on and with a time limit (it is another program: it may ask
    a question, or hang).  A timeout is an ordinary failure, with a message."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=90)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, 124, "", "the claude command did not answer within 90 seconds")
    except OSError as exc:
        return subprocess.CompletedProcess(cmd, 127, "", str(exc))


def register_code(home: str | None = None, tool_prefix: str = "") -> str:
    cmd, args = adapter_command("claude", tool_prefix)
    env_args = ["-e", f"RAG_SEARCH_HOME={Path(home).expanduser().absolute()}"] if home else []
    full = ["claude", "mcp", "add", "--scope", "user", SERVER_NAME, *env_args, "--", cmd, *args]
    if not shutil.which("claude"):
        return "Claude Code CLI ('claude') not found on PATH. To register later run:\n  " \
               + " ".join(full)
    _claude(["claude", "mcp", "remove", "--scope", "user", SERVER_NAME])
    proc = _claude(full)
    if proc.returncode != 0:
        return (f"! Claude Code registration failed: {(proc.stderr or proc.stdout).strip()}\n"
                f"  (any previous '{SERVER_NAME}' entry was removed first)  Retry with:\n  "
                + " ".join(full))
    return f"Claude Code: registered '{SERVER_NAME}' (user scope). Start a new session to use it."


def unregister_code() -> str:
    if not shutil.which("claude"):
        return "Claude Code CLI not found; nothing to do."
    proc = _claude(["claude", "mcp", "remove", "--scope", "user", SERVER_NAME])
    return "Claude Code: " + ((proc.stdout or proc.stderr).strip() or "removed")


# ── status (used by `rag-search doctor`) ─────────────────────────────────────

def _status_row(label: str, path: Path, hint: str) -> tuple[str, str, str]:
    """Health of the entry registered in the config file *path* (which exists)."""
    try:
        entry = (_load(path).get("mcpServers") or {}).get(SERVER_NAME)
    except ValueError as exc:
        return (label, "fail", f"{path} is not valid JSON ({exc})")
    if not entry:
        return (label, "warn", f"not registered in {path}; run `{hint}`")
    cmd = str(entry.get("command", ""))
    if not (Path(cmd).is_file() or shutil.which(cmd)):
        return (label, "fail", f"registered in {path} but its command {cmd!r} does not "
                f"exist; run `{hint}` again")
    if entry.get("disabled"):
        return (label, "warn", f"registered in {path} but disabled")
    args = entry.get("args") or []
    prof = args[args.index("--profile") + 1] if "--profile" in args and \
        args.index("--profile") + 1 < len(args) else "?"
    return (label, "ok", f"registered in {path} (profile {prof})")


def registration_status() -> list[tuple[str, str, str]]:
    """[(host, "ok"|"warn"|"fail", detail)] for the hosts configured through JSON files.
    Optional hosts appear as their module says (see above)."""
    out: list[tuple[str, str, str]] = []
    path = desktop_config_path()
    if not path.is_file():
        out.append(("Claude Desktop", "warn", "no config file yet; run `rag-search register --desktop`"))
    else:
        out.append(_status_row("Claude Desktop", path, "rag-search register --desktop"))
    reg = {h.NAME for h in registered_hosts()}
    for h in shown_hosts():
        hint = f"rag-search register --{h.NAME}"
        if h.NAME in reg:
            out.append(_status_row(h.LABEL, h.find_config(), hint))
        elif not h.installed():
            out.append((h.LABEL, "ok", "not installed"))
        else:
            out.append((h.LABEL, "warn", f"installed but not registered; run `{hint}`"))
    return out
