"""Which collections a given client may use (stdlib only).

Every collection is open to every client unless it has an entry in ``<home>/access.json``:

    {"version": 1, "collections": {"hr": ["claude"], "api-docs": ["claude", "agent"]}}

A collection with an entry is restricted to the listed clients (an empty list means
"nobody except the administrator's CLI").  The file is written only by ``rag-search
access ...``; the MCP tools can neither read nor change it, they only see the effect:
each host gets exactly the collections it is authorised for, and a collection it may not
use looks the same as one that does not exist.

Client identity is the name a host passes when it starts the adapter
(``rag-search-mcp --profile agent``); the adapter puts it into every request.  ``cli`` is the
administrator at the terminal and can use everything (``--client agent`` on a CLI command
shows what that client would get).  This is a guard against an agent reaching documents it should
not, not a security boundary: any process of the same macOS user can read the files.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from .paths import CachedFile, Paths, write_json_atomic

CLIENT_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,31}")
ADMIN_CLIENT = "cli"                 # the person at the terminal: never restricted
UNKNOWN_CLIENT = "unknown"           # a caller that did not identify itself
EVERYONE = "all"                     # in `rag-search access ...`: every client (never a client name)
RESERVED_CLIENTS = (ADMIN_CLIENT, UNKNOWN_CLIENT, EVERYONE)
ACCESS_VERSION = 1


def normalize_client(value: Any) -> str:
    v = str(value or "").strip().lower()
    return v if CLIENT_RE.fullmatch(v) else UNKNOWN_CLIENT


def valid_client_name(value: str) -> bool:
    return bool(CLIENT_RE.fullmatch(str(value or "").strip().lower()))


class Rules:
    """The parsed access rules.  ``by_name`` maps collection name -> allowed client names."""

    def __init__(self, by_name: dict[str, list[str]] | None = None):
        self.by_name: dict[str, list[str]] = {
            n: sorted(set(c)) for n, c in (by_name or {}).items()}
        # macOS volumes are usually case-insensitive: match restricted names case-folded
        self._fold = {n.casefold(): c for n, c in self.by_name.items()}

    # -- questions -----------------------------------------------------------
    def restricted(self, name: str) -> bool:
        return name.casefold() in self._fold

    def clients_for(self, name: str) -> list[str] | None:
        """The allowed clients, or None when the collection is open to everyone."""
        return self._fold.get(name.casefold())

    def allows(self, client: str, name: str) -> bool:
        if client == ADMIN_CLIENT:
            return True
        allowed = self._fold.get(name.casefold())
        return allowed is None or client in allowed

    def visible(self, client: str, existing: Iterable[str]) -> list[str]:
        """Collections *client* may see, in the order given."""
        return [c for c in existing if self.allows(client, c)]

    def hidden_from(self, client: str) -> list[str]:
        """Restricted collection names *client* may not use."""
        return [n for n in self.by_name if not self.allows(client, n)]

    def any_for(self, client: str) -> bool:
        return bool(self.hidden_from(client))


def resolve_scope(rules: Rules, client: str, requested: list[str],
                  existing: Iterable[str]) -> tuple[list[str], str]:
    """Return (collections to use, error).  No names requested = everything *client* may use.

    Names are matched the way the rest of rag-search matches collection names: exactly, else
    case-insensitively (macOS volumes are case-insensitive, and access rules and descriptions
    already fold case) -- the result always carries the collection's real spelling, and a name
    asked for twice is used once.  A name that does not exist or is not available to this client
    gives the same error, so the reply never reveals a restricted collection.
    """
    visible = rules.visible(client, existing)
    if not requested:
        return visible, ""
    exact = set(visible)
    folded: dict[str, list[str]] = {}
    for v in visible:
        folded.setdefault(v.casefold(), []).append(v)
    out: list[str] = []
    for name in requested:
        if name in exact:
            canon = name
        else:
            matches = folded.get(name.casefold(), [])
            canon = matches[0] if len(matches) == 1 else ""   # ambiguous spelling: no guess
        if not canon:
            avail = ", ".join(visible) if visible else "none"
            return [], f"unknown collection {name!r} (available: {avail})"
        if canon not in out:
            out.append(canon)
    return out, ""


def scrub(value: Any, hidden: Iterable[str]) -> Any:
    """Replace strings that mention a hidden collection (e.g. a path in an error message)."""
    names = [h for h in hidden if h]
    if not names:
        return value
    rx = re.compile(r"(?<![\w-])(?:" + "|".join(re.escape(n) for n in names) + r")(?![\w-])",
                    re.IGNORECASE)

    def walk(v: Any) -> Any:
        if isinstance(v, str):
            return "[restricted]" if rx.search(v) else v
        if isinstance(v, list):
            return [walk(x) for x in v]
        if isinstance(v, dict):
            return {k: walk(x) for k, x in v.items()}
        return v

    return walk(value)


# ── storage ─────────────────────────────────────────────────────────────────

def load_rules(paths: Paths) -> tuple[Rules, str]:
    """Return (rules, error).  A broken file restricts nothing new and reports why."""
    return _load_file(paths.access_file)


def _load_file(f: Path) -> tuple[Rules, str]:
    try:
        raw = f.read_text(encoding="utf-8")
    except FileNotFoundError:
        return Rules(), ""
    except OSError as exc:
        return Rules(), f"{f}: {exc}"
    try:
        data = json.loads(raw)
        coll = data.get("collections", {}) if isinstance(data, dict) else None
        if not isinstance(coll, dict):
            raise ValueError('expected {"collections": {name: [clients]}}')
        by_name: dict[str, list[str]] = {}
        for name, clients in coll.items():
            if not isinstance(clients, list):
                raise ValueError(f"{name!r}: the value must be a list of client names")
            by_name[str(name)] = [normalize_client(c) for c in clients
                                  if normalize_client(c) != UNKNOWN_CLIENT]
        return Rules(by_name), ""
    except ValueError as exc:
        # fail closed: a file we cannot understand must not silently open everything
        return _closed_rules(raw), f"{f}: {exc}"


def _closed_rules(raw: str) -> Rules:
    """Best effort for a damaged access.json: keep every collection name we can still see
    restricted to nobody, so a typo does not expose what was protected."""
    names = re.findall(r'"([^"\\]+)"\s*:\s*\[', raw)
    return Rules({n: [] for n in names})


class AccessStore(CachedFile):
    """Cached rules that re-read ``access.json`` when it changes (daemons and the other
    long-lived processes keep one; see ``store()``)."""

    def __init__(self, paths: Paths):
        super().__init__(paths.access_file, _load_file)
        self.paths = paths

    def get(self) -> Rules:
        return super().get()


_STORES: dict[str, AccessStore] = {}


def store(paths: Paths) -> AccessStore:
    """The process-wide cached store for *paths*' ``access.json``."""
    key = str(paths.access_file)
    s = _STORES.get(key)
    if s is None:
        s = _STORES[key] = AccessStore(paths)
    return s


def current_rules(paths: Paths) -> Rules:
    """The rules now, read through the process-wide cache (re-read only after a change)."""
    return store(paths).get()


def save_rules(paths: Paths, by_name: dict[str, list[str]]) -> Path:
    """Atomically write ``access.json`` (mode 0600).  Only the CLI calls this; callers doing a
    read-modify-write hold ``file_lock(paths.access_file)`` around it (see access.py)."""
    f = paths.access_file
    body = {"version": ACCESS_VERSION,
            "collections": {n: sorted(set(c)) for n, c in sorted(by_name.items())}}
    write_json_atomic(f, body, mode=0o600, newline=True)
    return f
