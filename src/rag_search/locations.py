"""Where each collection's documents come from (stdlib only).

A collection is one of:

* **a registered location** -- any folder (a notes vault, a synced drive, a network share, a project
  folder); its whole tree is one collection, under the name it was registered with::

      <home>/locations.json   {"version": 1, "locations": {"vault": "/Users/me/Notes"}}

  written only by ``rag-search location add/remove``.  The documents stay where they are.
* **an imported collection** -- unpacked from a collection export (see bundle.py).  It has no
  source anywhere; a ``collection.origin.json`` file in its workspace index folder marks it, so
  indexing never scans, prunes, merges or rebuilds it.

There is no third kind: registering a collection means giving its folder, and removing a location
deletes its index with it.  An index folder that is neither (a leftover of an older layout or a
hand-edited ``locations.json``) is removed by the next full indexing run.

Source folders are only ever *read*: rag-search never writes, moves or deletes anything in a
registered location (``read_source``; a test checks no other module opens files under them for
writing).  What indexing deletes when a document disappears is its own derived data (converted
Markdown, per-document index).

A registered location can be temporarily unreachable (an unmounted drive, a share that is down):
``reachable`` tells that apart from "empty" before anything is concluded about its documents,
and an unreachable location is skipped -- its collection keeps its last index untouched.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .paths import (
    SUPPORTED_EXTENSIONS,
    CachedFile,
    Paths,
    SourceRoots,
    file_lock,
    is_plain_name,
    is_within,
    name_from_folder,
    pasted_path,
    read_json,
    write_json_atomic,
)

LOCATIONS_VERSION = 1
ORIGIN_FILE = "collection.origin.json"
REACH_TIMEOUT_S = 10.0


class LocationError(ValueError):
    """A request the user can fix (bad name, folder missing, overlap, ...)."""


NO_LOCATIONS = ("no source folders are registered, so there is nothing to index: add one with "
                "`rag-search location add NAME FOLDER`")


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
    """Every registered location, for indexing and path mirroring."""
    locs = load(paths)[0]
    return SourceRoots(tuple(sorted((n, str(Path(f).expanduser())) for n, f in locs.items())))


def source_root(paths: Paths, collection: str) -> Path | None:
    """Where *collection*'s sources live (whether or not that folder exists right now), or None when
    no location of that name is registered."""
    return source_roots(paths).root_of(collection)


def leftover_names(paths: Paths) -> list[str]:
    """Index folders that are neither a registered location nor imported: nothing can ever update them
    (``location remove`` deletes the index with the registration), so a full indexing run removes them."""
    locs, imported = set(load(paths)[0]), set(imported_names(paths))
    return [c for c in workspace_collections(paths) if c not in locs and c not in imported]


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

def why_unreachable(folder: Path, timeout: float = REACH_TIMEOUT_S) -> str:
    """"" when *folder* exists and can be listed within *timeout* seconds, otherwise the reason.

    Run in a worker thread: on an unresponsive network share a plain ``listdir`` can hang for
    minutes.  A folder that is missing (an unmounted drive's mount point) or cannot be listed
    is unreachable -- never "empty".  A listing that fails is tried again until the time is up: a
    folder kept by a cloud-storage app (Box, Google Drive, iCloud under ``~/Library/CloudStorage``)
    can refuse the first listing while the app fetches it, and answer a moment later."""
    result: dict[str, str] = {}
    deadline = time.monotonic() + max(0.0, timeout)

    def probe() -> None:
        while True:
            try:
                if not folder.is_dir():
                    result["why"] = "it is not a folder" if folder.exists() else "it does not exist"
                    return                            # nothing to wait for
                else:
                    os.listdir(folder)
                    result["why"] = ""
                    return
            except OSError as exc:
                result["why"] = f"it cannot be listed ({exc.strerror or type(exc).__name__})"
            if time.monotonic() + 0.5 >= deadline:
                return
            time.sleep(0.5)

    th = threading.Thread(target=probe, name="reach", daemon=True)
    th.start()
    th.join(timeout + 1.0)
    return result.get("why", f"it did not answer within {timeout:.0f} s")


def reachable(folder: Path, timeout: float = REACH_TIMEOUT_S) -> bool:
    """True when *folder* exists and can be listed within *timeout* seconds (see ``why_unreachable``)."""
    return not why_unreachable(folder, timeout)


def status(paths: Paths) -> list[dict[str, Any]]:
    """Every registered location with whether it can be read right now."""
    locs, _ = load(paths)
    out = []
    for name, folder in sorted(locs.items()):
        p = Path(folder).expanduser()
        out.append({"collection": name, "folder": str(p), "reachable": reachable(p)})
    return out


def sources(paths: Paths) -> dict[str, Any]:
    """Where the documents of every collection come from, for the dashboard (nothing is probed): the
    registered locations, the imported collections (which have no source) and the indexed collections
    that have no registered folder."""
    try:
        locs, err = load(paths)
        return {"locations": [{"collection": n, "folder": f} for n, f in sorted(locs.items())],
                "imported": imported_names(paths),
                **({"error": err} if err else {})}
    except Exception as exc:  # noqa: BLE001 - a status call must always answer
        return {"locations": [], "imported": [], "error": str(exc)}


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
    """rag-search's own folders, never scanned even when a registered folder contains them."""
    return [paths.markup, paths.index, paths.run, paths.jobs]


# ── what one indexing run looks at ──────────────────────────────────────────

@dataclass
class ScanPlan:
    """The documents one indexing run covers and what it may conclude from them.

    ``covered`` lists collections whose *whole* source folder was walked and readable: only for
    those may a document that was not found be treated as deleted (its Markdown and index are
    removed).  ``unreachable`` are registered locations that could not be read: their
    collections are left exactly as they are (no pruning, no re-merge, still searchable).
    """

    roots: SourceRoots
    sources: list[Path] = field(default_factory=list)
    unsupported: list[dict[str, str]] = field(default_factory=list)
    covered: list[str] = field(default_factory=list)
    unreachable: list[str] = field(default_factory=list)
    frozen: list[str] = field(default_factory=list)    # collections to leave as they are
    orphans: list[str] = field(default_factory=list)   # index folders nobody registered (a full run removes them)
    target: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"covered": self.covered, "unreachable": self.unreachable, "frozen": self.frozen,
                "orphans": self.orphans}


def _match(name: str, candidates: Iterable[str]) -> str:
    for c in candidates:
        if c == name:
            return c
    for c in candidates:
        if c.casefold() == name.casefold():
            return c
    return ""


def resolve_target(paths: Paths, raw: str) -> tuple[Path | None, str]:
    """(folder or file to index, the collection it is the whole of or "") for a user-given path:
    absolute, or starting with a registered location's name (``vault`` or ``vault/projects``).
    An empty *raw* means everything: ``(None, "")``.  Raises LocationError with a message for the user."""
    roots = source_roots(paths)
    locs = dict(roots.locations)
    raw = pasted_path(raw)
    if not raw:
        return None, ""
    if not locs:
        raise LocationError(NO_LOCATIONS)
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
            raise LocationError(f"{raw!r} is neither a registered location nor a path inside one "
                                "(see `rag-search location list`)")
    p = p.resolve()
    for name, root in locs.items():
        r = Path(root).resolve()
        if is_within(p, r):
            if not reachable(r):
                raise LocationError(f"location {name!r} ({r}) is not reachable right now")
            if not p.exists():
                raise LocationError(f"not found: {p}")
            return p, name if p == r else ""
    raise LocationError(f"path must be inside a registered location (see `rag-search location "
                        f"list`): {p}")


def plan_scan(paths: Paths, raw: str = "") -> ScanPlan:
    """Decide what an indexing run over *raw* (empty = everything) reads and may prune.

    Nothing is concluded from a source folder that could not be read completely: a registered
    location that is missing or cannot be listed, a sub-folder the walk could not open, or a
    location that is completely empty while its collection has an index (an unmounted mount point
    looks exactly like that) -- those collections are *frozen*: not pruned, not re-merged, still
    served as indexed before."""
    _, error = load(paths)
    if error:
        # without the registry, every location's collection would look like an emptied folder
        raise LocationError(f"cannot index while {error}; fix or remove that file "
                            "(rag-search location list)")
    roots = source_roots(paths)
    locs = dict(roots.locations)
    excludes = workspace_excludes(paths)
    if not locs:
        raise LocationError(NO_LOCATIONS)
    plan = ScanPlan(roots=roots, target=raw)
    have = set(workspace_collections(paths))

    def freeze(names: Iterable[str], why: str) -> None:
        for n in names:
            if n not in plan.frozen:
                plan.frozen.append(n)
        if why not in plan.unreachable:
            plan.unreachable.append(why)

    # Every location is probed, whatever this run's scope: a collection whose folder cannot be read
    # must not be re-merged either (its documents would all look deleted).
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

    target, coll = resolve_target(paths, raw)
    if target is not None and target.is_file():
        if target.suffix.lower() in SUPPORTED_EXTENSIONS:
            plan.sources = [target]
        else:
            plan.unsupported = [{"src": str(target), "extension": target.suffix.lower()}]
        return plan
    if target is None:
        for name, root in sorted(locs.items()):
            if loc_ok[name]:
                walk(Path(root), name)
                plan.covered.append(name)
    else:
        owner = next((n for n, r in locs.items() if is_within(target, Path(r).resolve())), None)
        walk(target, owner)
        if coll:
            plan.covered = [coll]
    frozen = {f.casefold() for f in plan.frozen}
    plan.covered = [c for c in plan.covered if c.casefold() not in frozen]
    if target is None:                                  # the whole run: nothing registered can own these
        plan.orphans = leftover_names(paths)
    return plan


# ── changes (CLI only) ──────────────────────────────────────────────────────

def _save(paths: Paths, locs: dict[str, str]) -> None:
    write_json_atomic(paths.locations_file,
                      {"version": LOCATIONS_VERSION, "locations": dict(sorted(locs.items()))},
                      newline=True)


def add(paths: Paths, name: str, folder: str) -> dict[str, Any]:
    """Register *folder* as the source of collection *name*."""
    name, folder = pasted_path(name), pasted_path(folder)
    if not folder and ("/" in name or name.startswith("~")):
        name, folder = "", name                     # the path was put where the name goes
    if not name and folder:                         # no name given: the folder's own name
        name = name_from_folder(Path(folder).expanduser())
        if not name:
            raise LocationError(f"give a collection name: none can be made from {folder!r}")
    if not is_plain_name(name):
        raise LocationError(f"{name!r} is not a usable collection name (one folder-name-like "
                            "word: letters, digits, '-', '_'); the folder goes in the other field")
    raw = (folder or "").strip()
    if not raw:
        raise LocationError("give the folder to index")
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    p = p.resolve()
    why = why_unreachable(p)
    if why:
        raise LocationError(f"{p} is not a folder that can be read right now: {why}")
    home = paths.home.resolve()
    if is_within(p, home) or is_within(home, p):
        raise LocationError(f"{p} overlaps rag-search's own data folder ({home}); choose a folder "
                            "outside it")
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
        locs[name] = str(p)
        _save(paths, locs)
    return {"collection": name, "folder": str(p), "changed": True,
            "note": "run `rag-search index new` to index it"}


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
