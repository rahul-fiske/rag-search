"""Where each collection's documents come from (stdlib only).

A collection is one of:

* **a docs-folder collection** -- a first-level folder of the docs folder (the original model;
  nothing to register).  Files directly in the docs folder form the ``default`` collection.
* **a registered location** -- any folder elsewhere (a notes vault, a synced drive, a network
  share); its whole tree is one collection, under the name it was registered with::

      <home>/locations.json   {"version": 1, "locations": {"vault": "/Users/me/Notes"}}

  written only by ``rag-search location add/remove``.
* **an imported collection** -- unpacked from a collection export (see bundle.py).  It has no
  source anywhere; a ``collection.origin.json`` file in its workspace index folder marks it, so
  indexing never scans, prunes, merges or rebuilds it.

Source folders are only ever *read*: rag-search never writes, moves or deletes anything in the
docs folder or in a registered location (``read_source``; a test checks no other module opens
files under them for writing).  What indexing deletes when a document disappears is its own
derived data (converted Markdown, per-document index).

A registered location can be temporarily unreachable (an unmounted drive, a share that is down):
``reachable`` tells that apart from "empty" before anything is concluded about its documents,
and an unreachable location is skipped -- its collection keeps its last index untouched.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .paths import (
    DEFAULT_COLLECTION,
    SUPPORTED_EXTENSIONS,
    CachedFile,
    Paths,
    SourceRoots,
    file_lock,
    is_plain_name,
    is_within,
    read_json,
    write_json_atomic,
)

LOCATIONS_VERSION = 1
ORIGIN_FILE = "collection.origin.json"
REACH_TIMEOUT_S = 10.0


class LocationError(ValueError):
    """A request the user can fix (bad name, folder missing, overlap, ...)."""


# ── reading ─────────────────────────────────────────────────────────────────

def _load_file(f: Path) -> tuple[dict[str, str], str]:
    try:
        raw = f.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}, ""
    except OSError as exc:
        return {}, f"{f}: {exc}"
    try:
        data = json.loads(raw)
        locs = data.get("locations", {}) if isinstance(data, dict) else None
        if not isinstance(locs, dict):
            raise ValueError('expected {"locations": {name: "/folder"}}')
        out = {}
        for name, folder in locs.items():
            if not is_plain_name(str(name)) or not isinstance(folder, str) or not folder.strip():
                raise ValueError(f"bad entry {name!r}: {folder!r}")
            out[str(name)] = folder
        return out, ""
    except ValueError as exc:
        return {}, f"{f}: {exc}"


_STORES: dict[str, CachedFile] = {}


def load(paths: Paths) -> tuple[dict[str, str], str]:
    """(collection name -> folder, error), through a process-wide cache."""
    key = str(paths.locations_file)
    s = _STORES.get(key)
    if s is None:
        s = _STORES[key] = CachedFile(paths.locations_file, _load_file)
    return dict(s.get()), s.error


def names(paths: Paths) -> list[str]:
    return sorted(load(paths)[0])


def source_roots(paths: Paths) -> SourceRoots:
    """The docs folder plus every registered location, for indexing and path mirroring."""
    locs = load(paths)[0]
    return SourceRoots(str(paths.docs), tuple(sorted((n, str(Path(f).expanduser()))
                                                     for n, f in locs.items())))


def source_root(paths: Paths, collection: str) -> Path:
    """Where *collection*'s sources live (whether or not that folder exists right now)."""
    return source_roots(paths).root_of(collection)


# ── imported collections ────────────────────────────────────────────────────

def workspace_collections(paths: Paths) -> list[str]:
    """Collection folders in the indexer workspace (generated and imported)."""
    try:
        return sorted(p.name for p in paths.index.iterdir()
                      if p.is_dir() and not p.name.startswith("."))
    except OSError:
        return []


def origin(paths: Paths, collection: str) -> dict[str, Any]:
    """The origin record of an imported collection, or {} for a generated one."""
    data = read_json(paths.index / collection / ORIGIN_FILE)
    return data if isinstance(data, dict) and data.get("origin") == "imported" else {}


def is_imported(paths: Paths, collection: str) -> bool:
    return bool(origin(paths, collection))


def imported_names(paths: Paths) -> list[str]:
    return [c for c in workspace_collections(paths) if is_imported(paths, c)]


def index_is_imported(index_root: Path, collection: str) -> bool:
    """Same as ``is_imported`` for code that only has the workspace index folder."""
    data = read_json(index_root / collection / ORIGIN_FILE)
    return isinstance(data, dict) and data.get("origin") == "imported"


# ── reachability ────────────────────────────────────────────────────────────

def reachable(folder: Path, timeout: float = REACH_TIMEOUT_S) -> bool:
    """True when *folder* exists and can be listed within *timeout* seconds.

    Run in a worker thread: on an unresponsive network share a plain ``listdir`` can hang for
    minutes.  A folder that is missing (an unmounted drive's mount point) or cannot be listed
    is unreachable -- never "empty"."""
    result: dict[str, bool] = {}

    def probe() -> None:
        try:
            result["ok"] = folder.is_dir() and (os.listdir(folder) is not None)
        except OSError:
            result["ok"] = False

    th = threading.Thread(target=probe, name="reach", daemon=True)
    th.start()
    th.join(timeout)
    return bool(result.get("ok"))


def status(paths: Paths) -> list[dict[str, Any]]:
    """Every registered location with whether it can be read right now."""
    locs, _ = load(paths)
    out = []
    for name, folder in sorted(locs.items()):
        p = Path(folder).expanduser()
        out.append({"collection": name, "folder": str(p), "reachable": reachable(p)})
    return out


def sources(paths: Paths) -> dict[str, Any]:
    """Where the documents of every collection come from, for the dashboard (nothing is probed beyond
    listing the docs folder): the docs folder and the collections made of its sub-folders, the registered
    locations, and the imported collections (which have no source)."""
    from .catalog import docs_folder_names

    try:
        locs, _err = load(paths)
        return {"docs_folder": str(paths.docs), "docs_collections": docs_folder_names(paths),
                "locations": [{"collection": n, "folder": f} for n, f in sorted(locs.items())],
                "imported": imported_names(paths)}
    except Exception as exc:  # noqa: BLE001 - a status call must always answer
        return {"docs_folder": str(paths.docs), "docs_collections": [], "locations": [], "imported": [],
                "error": str(exc)}


# ── reading sources ─────────────────────────────────────────────────────────

def read_source(path: Path) -> bytes:
    """Read a source document.  Sources are opened read-only, always -- this is the one place
    indexing code is expected to open a file under a source folder itself (the converters
    receive the path and open it the same way)."""
    with open(path, "rb") as fh:
        return fh.read()


def scan_tree(scan_root: Path, exclude: Iterable[Path] = (),
              errors: list[str] | None = None) -> tuple[list[Path], list[dict[str, str]]]:
    """One walk of *scan_root* (hidden files/dirs, ``~$`` lock files and *exclude* skipped).

    -> (supported files, unsupported files) -- the second list is every file this walk saw and
    passed over only because its extension isn't one indexing reads, as ``{"src", "extension"}``
    (``extension`` is ``""`` for a file with none at all).  A file skipped for any other reason
    (hidden, a lock file, inside *exclude*) is in neither list -- it was never really "seen".

    Folders that could not be listed (permissions, a cloud or network folder timing out) are
    appended to *errors*: what is inside them is unknown, not absent."""
    ex = [Path(e).resolve() for e in exclude]
    found: list[Path] = []
    unsupported: list[dict[str, str]] = []

    def failed(exc: OSError) -> None:
        if errors is not None:
            errors.append(str(getattr(exc, "filename", "") or scan_root))

    for dirpath, dirnames, filenames in os.walk(scan_root, onerror=failed, followlinks=False):
        dp = Path(dirpath)
        keep = []
        for d in sorted(dirnames):
            if d.startswith("."):
                continue
            if any(is_within(dp / d, e) for e in ex):
                continue
            keep.append(d)
        dirnames[:] = keep
        for fn in sorted(filenames):
            if fn.startswith(".") or fn.startswith("~$"):
                continue
            p = dp / fn
            if not p.is_file():
                continue
            ext = p.suffix.lower()
            if ext in SUPPORTED_EXTENSIONS:
                found.append(p)
            else:
                unsupported.append({"src": str(p), "extension": ext})
    return found, unsupported


def _is_empty_dir(folder: Path) -> bool:
    try:
        with os.scandir(folder) as it:
            return next(it, None) is None
    except OSError:
        return False


def workspace_excludes(paths: Paths) -> list[Path]:
    """rag-search's own folders, never scanned even when they sit inside the docs folder."""
    return [paths.markup, paths.index, paths.run, paths.jobs]


# ── what one indexing run looks at ──────────────────────────────────────────

@dataclass
class ScanPlan:
    """The documents one indexing run covers and what it may conclude from them.

    ``covered`` lists collections whose *whole* source folder was walked and readable: only for
    those may a document that was not found be treated as deleted (its Markdown and index are
    removed).  ``unreachable`` are registered locations that could not be read: their
    collections are left exactly as they are (no pruning, no re-merge, still searchable)."""

    roots: SourceRoots
    sources: list[Path] = field(default_factory=list)
    unsupported: list[dict[str, str]] = field(default_factory=list)
    covered: list[str] = field(default_factory=list)
    unreachable: list[str] = field(default_factory=list)
    frozen: list[str] = field(default_factory=list)    # collections to leave as they are
    shadowed: list[str] = field(default_factory=list)
    target: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"covered": self.covered, "unreachable": self.unreachable,
                "frozen": self.frozen, "shadowed": self.shadowed}


def _match(name: str, candidates: Iterable[str]) -> str:
    for c in candidates:
        if c == name:
            return c
    for c in candidates:
        if c.casefold() == name.casefold():
            return c
    return ""


def resolve_target(paths: Paths, raw: str) -> tuple[Path, str]:
    """(folder or file to index, the collection it belongs to or "") for a user-given path:
    absolute, relative to the docs folder, or starting with a registered location's name
    (``vault`` or ``vault/projects``).  Raises LocationError with a message for the user."""
    roots = source_roots(paths)
    locs = dict(roots.locations)
    raw = (raw or "").strip()
    docs = paths.docs.resolve()
    if not raw:
        return docs, ""
    p = Path(raw).expanduser()
    if not p.is_absolute():
        first = p.parts[0] if p.parts else ""
        loc = _match(first, locs)
        if loc:
            root = Path(locs[loc])
            if not reachable(root):
                raise LocationError(f"location {loc!r} ({root}) is not reachable right now "
                                    "(unmounted drive or share?); nothing to index there")
            p = root.joinpath(*p.parts[1:])
        else:
            imp = _match(first, imported_names(paths))
            if imp:
                raise LocationError(f"{imp!r} is an imported collection: it has no source "
                                    "documents to index (delete it and import a newer export "
                                    "to update it)")
            p = docs / p
    p = p.resolve()
    for name, root in locs.items():
        r = Path(root).resolve()
        if is_within(p, r):
            if not reachable(r):
                raise LocationError(f"location {name!r} ({r}) is not reachable right now")
            if not p.exists():
                raise LocationError(f"not found: {p}")
            return p, name if p == r else ""
    if not is_within(p, docs):
        raise LocationError(f"path must be inside the docs folder {docs} or a registered "
                            "location (rag-search location list)")
    if not p.exists():
        raise LocationError(f"not found: {p}")
    if p != docs:
        first = p.relative_to(docs).parts[0]
        reserved = _match(first, [*locs, *imported_names(paths)])
        if reserved and (docs / first).is_dir():
            raise LocationError(f"{docs / first} is ignored: {reserved!r} is the name of a "
                                "registered location or an imported collection")
        if p.parent == docs and p.is_dir() and first != DEFAULT_COLLECTION:
            return p, first
    return p, ""


def plan_scan(paths: Paths, raw: str = "") -> ScanPlan:
    """Decide what an indexing run over *raw* (empty = everything) reads and may prune.

    Nothing is concluded from a source folder that could not be read completely: a registered
    location or docs folder that is missing or cannot be listed, a sub-folder the walk could not
    open, or a source root that is completely empty while its collections have an index (an
    unmounted mount point looks exactly like that) -- those collections are *frozen*: not
    pruned, not re-merged, still served as indexed before."""
    _, error = load(paths)
    if error:
        # without the registry, a location's collection would look like an emptied docs folder
        raise LocationError(f"cannot index while {error}; fix or remove that file "
                            "(rag-search location list)")
    roots = source_roots(paths)
    locs = dict(roots.locations)
    imported = imported_names(paths)
    reserved = {n.casefold() for n in (*locs, *imported)}
    docs = paths.docs.resolve()
    shadowed = [f for f in _docs_folders(paths) if f.casefold() in reserved]
    excludes = workspace_excludes(paths) + [docs / f for f in shadowed]
    plan = ScanPlan(roots=roots, shadowed=sorted(shadowed), target=raw)
    have = set(workspace_collections(paths))

    def docs_collections() -> list[str]:
        names = have | set(_docs_folders(paths)) | {DEFAULT_COLLECTION}
        return sorted(n for n in names if n.casefold() not in reserved)

    def freeze(names: Iterable[str], why: str) -> None:
        for n in names:
            if n not in plan.frozen:
                plan.frozen.append(n)
        if why not in plan.unreachable:
            plan.unreachable.append(why)

    # Every source folder is probed, whatever this run's scope: a collection whose folder cannot
    # be read must not be re-merged either (its documents would all look deleted).
    docs_ok = reachable(docs)
    if not docs_ok:
        freeze(docs_collections(), f"(docs folder {docs})")
    elif _is_empty_dir(docs) and any(c in have for c in docs_collections()):
        docs_ok = False
        freeze(docs_collections(), f"(docs folder {docs} is empty: treated as not mounted)")
    loc_ok = {}
    for name, root in sorted(locs.items()):
        r = Path(root)
        loc_ok[name] = reachable(r)
        if not loc_ok[name]:
            freeze([name], name)
        elif _is_empty_dir(r) and name in have:
            loc_ok[name] = False
            freeze([name], f"{name} (folder is empty: treated as not mounted)")

    def walk(root: Path, owner: str | None) -> None:
        errs: list[str] = []
        found, unsup = scan_tree(root, excludes, errs)
        plan.sources += found
        plan.unsupported += unsup
        for e in errs:                      # which collection lost part of its tree?
            if owner is not None:
                freeze([owner], f"{owner} (could not read {e})")
                continue
            try:
                parts = Path(e).resolve().relative_to(docs).parts
            except ValueError:
                parts = ()
            freeze(docs_collections() if not parts else [parts[0]],
                   f"{parts[0] if parts else 'docs folder'} (could not read {e})")

    target, coll = resolve_target(paths, raw) if raw else (docs, "")
    if raw and target.is_file():
        if target.suffix.lower() in SUPPORTED_EXTENSIONS:
            plan.sources = [target]
        else:
            plan.unsupported = [{"src": str(target), "extension": target.suffix.lower()}]
        return plan
    if not raw:
        if docs_ok:
            walk(docs, None)
            plan.covered = docs_collections()
        for name, root in sorted(locs.items()):
            if loc_ok[name]:
                walk(Path(root), name)
                plan.covered.append(name)
    else:
        owner = next((n for n, r in locs.items() if is_within(target, Path(r).resolve())), None)
        walk(target, owner)
        if target == docs:
            plan.covered = docs_collections() if docs_ok else []
        elif coll:
            plan.covered = [coll]
    frozen = {f.casefold() for f in plan.frozen}
    plan.covered = [c for c in plan.covered if c.casefold() not in frozen]
    return plan


# ── changes (CLI only) ──────────────────────────────────────────────────────

def _save(paths: Paths, locs: dict[str, str]) -> None:
    write_json_atomic(paths.locations_file,
                      {"version": LOCATIONS_VERSION, "locations": dict(sorted(locs.items()))},
                      newline=True)


def add(paths: Paths, name: str, folder: str) -> dict[str, Any]:
    """Register *folder* as the source of collection *name*."""
    name = (name or "").strip()
    if not is_plain_name(name):
        raise LocationError(f"{name!r} is not a usable collection name (one folder-name-like "
                            "word: letters, digits, '-', '_')")
    if name.casefold() == DEFAULT_COLLECTION:
        raise LocationError(f"{DEFAULT_COLLECTION!r} is reserved for files directly in the docs "
                            "folder; choose another name")
    raw = (folder or "").strip()
    if not raw:
        raise LocationError("give the folder to index")
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    p = p.resolve()
    if not reachable(p):
        raise LocationError(f"{p} is not a folder that can be read right now")
    home, docs = paths.home.resolve(), paths.docs.resolve()
    if is_within(p, docs):
        raise LocationError(f"{p} is inside the docs folder ({docs}); its first-level folders "
                            "are collections already -- no need to register it")
    if is_within(p, home) or is_within(home, p) or is_within(docs, p):
        raise LocationError(f"{p} overlaps rag-search's own data folder ({home}) or the docs "
                            "folder; choose a folder outside both")
    with file_lock(paths.locations_file):
        locs, error = _load_file(paths.locations_file)
        if error:
            raise LocationError(f"cannot change locations while {error}; fix or remove that file")
        for other, f in locs.items():
            if other.casefold() == name.casefold():
                raise LocationError(f"{other!r} is already registered ({f}); remove it first")
            o = Path(f).expanduser().resolve()
            if is_within(p, o) or is_within(o, p):
                raise LocationError(f"{p} overlaps the folder of {other!r} ({o}); one folder "
                                    "can belong to one collection only")
        for existing in workspace_collections(paths):
            if existing.casefold() == name.casefold() and is_imported(paths, existing):
                raise LocationError(f"{existing!r} is an imported collection; delete it first "
                                    "or choose another name")
        clash = [e for e in _docs_folders(paths) if e.casefold() == name.casefold()]
        if clash:
            raise LocationError(f"the docs folder already has a {clash[0]!r} folder (collection "
                                f"{clash[0]!r}); rename one of them")
        locs[name] = str(p)
        _save(paths, locs)
    return {"collection": name, "folder": str(p), "changed": True,
            "note": "run `rag-search index new` to index it"}


def _docs_folders(paths: Paths) -> list[str]:
    try:
        with os.scandir(paths.docs) as it:
            return [e.name for e in it
                    if e.is_dir(follow_symlinks=False) and not e.name.startswith(".")]
    except OSError:
        return []


def remove_entry(paths: Paths, name: str) -> str:
    """Unregister *name* (the stored spelling is matched case-insensitively).  Returns the name
    as it was registered, or "" when there was no such location."""
    with file_lock(paths.locations_file):
        locs, error = _load_file(paths.locations_file)
        if error:
            raise LocationError(f"cannot change locations while {error}; fix or remove that file")
        hit = next((n for n in locs if n.casefold() == name.casefold()), "")
        if hit:
            del locs[hit]
            _save(paths, locs)
    return hit


def write_origin(index_coll_dir: Path, record: dict[str, Any]) -> None:
    write_json_atomic(index_coll_dir / ORIGIN_FILE,
                      {"origin": "imported", "imported_at": time.strftime(
                          "%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **record})
