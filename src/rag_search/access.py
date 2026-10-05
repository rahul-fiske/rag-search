"""Managing which clients may use which collections (stdlib only).

Used by ``rag-search access ...`` only.  The MCP adapter never imports this module: hosts
can see the outcome (their own list of collections) but cannot inspect or change the rules.
The rules themselves and how they are applied live in ``policy.py``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from . import locations, policy, register
from .catalog import canonical_name, live_catalog
from .paths import Paths, file_lock


class AccessError(ValueError):
    """A request the user can fix (bad name, unknown client, wrong state)."""


# ── what exists ─────────────────────────────────────────────────────────────

def known_collections(paths: Paths) -> dict[str, dict[str, Any]]:
    """Every collection we know of: published ones, registered source locations and imported
    collections -- with where each one's documents come from (``kind``: location | imported |
    and ``folder`` for a location): a collection is one or the other."""
    out: dict[str, dict[str, Any]] = {}
    locs = locations.load(paths)[0]
    imported = set(locations.imported_names(paths))

    def kind(name: str) -> dict[str, Any]:
        if name in locs:
            return {"kind": "location", "folder": locs[name]}
        if name in imported:
            return {"kind": "imported", "folder": None}
        return {"kind": "imported", "folder": None}

    for c in live_catalog(paths).get("collections", []):
        if c["collection"] not in locs and c["collection"] not in imported:
            continue                          # a leftover index nobody registered: not a collection
        out[c["collection"]] = {"indexed": True, "documents": len(c.get("documents", [])),
                                "chunks": c.get("chunks", 0), **kind(c["collection"])}
    for name in (*locs, *imported):
        out.setdefault(name, {"indexed": False, "documents": 0, "chunks": 0, **kind(name)})
    return out


def _canonical(paths: Paths, rules: policy.Rules, name: str) -> str:
    """The name to store: the existing spelling if the collection exists, else as typed."""
    try:
        return canonical_name(paths, name, extra=(*rules.by_name, *known_collections(paths)))
    except ValueError as exc:
        raise AccessError(str(exc)) from None


def _client(name: str) -> str:
    c = (name or "").strip().lower()
    if c == policy.ADMIN_CLIENT:
        raise AccessError("'cli' is the administrator's own terminal and can always use every "
                          "collection; it does not need to be listed")
    if c in policy.RESERVED_CLIENTS or not policy.valid_client_name(c):
        raise AccessError(f"{name!r} is not a usable client name (letters, digits, '_', '.', "
                          "'-'; the name a host passes as `rag-search-mcp --profile NAME`)")
    return c


def _clients(names: Iterable[str]) -> list[str]:
    return sorted({_client(n) for n in names})


# ── views ───────────────────────────────────────────────────────────────────

def overview(paths: Paths, seen: dict[str, float] | None = None) -> dict[str, Any]:
    """Every collection with who may use it, plus every client name we know about."""
    rules, error = policy.load_rules(paths)
    known = known_collections(paths)
    rows = []
    for name in sorted(set(known) | set(rules.by_name), key=str.casefold):
        allowed = rules.clients_for(name)
        info = known.get(name)
        rows.append({
            "collection": name,
            "access": "everyone" if allowed is None else "restricted",
            "clients": None if allowed is None else list(allowed),
            "exists": info is not None,
            "indexed": bool(info and info["indexed"]),
            "documents": info["documents"] if info else 0,
            "chunks": info["chunks"] if info else 0,
            "kind": info["kind"] if info else "",
            "folder": info["folder"] if info else None,
        })
    return {"collections": rows, "clients": clients(paths, seen), "access_file": str(paths.access_file),
            "error": error}


def clients(paths: Paths, seen: dict[str, float] | None = None) -> list[dict[str, Any]]:
    """Client identities: Claude and the registered hosts, any named in a rule, and any seen by the daemon."""
    rules = policy.load_rules(paths)[0]
    names: dict[str, dict[str, Any]] = {}

    def add(n: str, **kw: Any) -> None:
        names.setdefault(n, {"client": n, "last_seen": None, "restricted_to": []}).update(kw)

    add(policy.ADMIN_CLIENT, note="administrator's terminal: can use every collection")
    for n in register.host_clients():
        add(n)
    for coll, allowed in rules.by_name.items():
        for n in allowed:
            add(n)
    for n, ts in (seen or {}).items():
        if n != policy.ADMIN_CLIENT:
            add(n, last_seen=ts)
    every = sorted(set(known_collections(paths)) | set(rules.by_name), key=str.casefold)
    for n, info in names.items():
        info["can_use"] = [c for c in every if rules.allows(n, c)]
        info["blocked_from"] = [c for c in every if not rules.allows(n, c)]
        info["restricted_to"] = sorted(c for c, allowed in rules.by_name.items() if n in allowed)
    return sorted(names.values(), key=lambda i: (i["client"] != policy.ADMIN_CLIENT, i["client"]))


# ── changes ─────────────────────────────────────────────────────────────────

def _save(paths: Paths, by_name: dict[str, list[str]]) -> None:
    policy.save_rules(paths, by_name)


def _current(paths: Paths) -> tuple[policy.Rules, dict[str, list[str]]]:
    rules, error = policy.load_rules(paths)
    if error:
        raise AccessError(f"cannot change access rules while {error}; fix or remove that file")
    return rules, {n: list(c) for n, c in rules.by_name.items()}


def _wants_everyone(client_names: Iterable[str]) -> bool:
    return any(str(n).strip().lower() == policy.EVERYONE for n in client_names)


def restrict(paths: Paths, collection: str, client_names: Iterable[str]) -> dict[str, Any]:
    """Only *client_names* may use *collection*.

    This replaces any earlier list, so it also removes clients; no names = nobody; ``all`` =
    every client again (the restriction is removed).
    """
    client_names = list(client_names)
    if _wants_everyone(client_names):
        return _open(paths, collection)
    with file_lock(paths.access_file):
        rules, by_name = _current(paths)
        name = _canonical(paths, rules, collection)
        by_name[name] = _clients(client_names)
        _save(paths, by_name)
    return _result(paths, name, exists=name in known_collections(paths))


def grant(paths: Paths, collection: str, client_names: Iterable[str]) -> dict[str, Any]:
    """Add clients to a restricted collection (``all`` = every client: removes the restriction)."""
    client_names = list(client_names)
    if _wants_everyone(client_names):
        return _open(paths, collection)
    with file_lock(paths.access_file):
        rules, by_name = _current(paths)
        name = _canonical(paths, rules, collection)
        if name not in by_name:
            raise AccessError(f"{name!r} is open to every client, so there is nothing to grant. "
                              f"To limit it to specific clients use: rag-search access restrict "
                              f"{name} CLIENT [CLIENT ...]")
        by_name[name] = sorted(set(by_name[name]) | set(_clients(client_names)))
        _save(paths, by_name)
    return _result(paths, name)


def _open(paths: Paths, collection: str) -> dict[str, Any]:
    """Remove the restriction: every client may use *collection* again."""
    with file_lock(paths.access_file):
        rules, by_name = _current(paths)
        name = _canonical(paths, rules, collection)
        if name not in by_name:
            return {"collection": name, "access": "everyone", "clients": None, "changed": False,
                    "exists": name in known_collections(paths)}
        del by_name[name]
        _save(paths, by_name)
    out = _result(paths, name)
    out["changed"] = True
    return out


def _result(paths: Paths, name: str, exists: bool | None = None) -> dict[str, Any]:
    rules = policy.load_rules(paths)[0]
    allowed = rules.clients_for(name)
    return {"collection": name, "access": "everyone" if allowed is None else "restricted",
            "clients": None if allowed is None else list(allowed), "changed": True,
            "exists": name in known_collections(paths) if exists is None else exists}
