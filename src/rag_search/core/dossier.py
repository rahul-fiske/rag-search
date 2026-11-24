"""The file inspector's backend (the dashboard's Files tab): browse the folders of a collection and, for one file, put together
everything rag-search recorded about it.

Read-only: nothing is recomputed.  Every fact comes from a record a step wrote when it ran -- the index folder (``index.meta.json``,
``nodes.json``, ``pictures.json``), the Markdown and its sidecars, the describe cache, the People store, the map stamp -- and the
ledger puts them side by side, one row per step, saying what produced each and whether it is still current.

A file is named by its collection and its path inside the collection's source folder.  Folders are listed from the source folder
itself, a level at a time, so a collection of 20 000 files is never read whole; only a search walks it.
"""

from __future__ import annotations

import calendar
import contextlib
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

from .. import locations, maps, models
from ..paths import META_FILE, Paths, read_json
from . import describe, faces, people, photo, pictures

MAX_LIST = 400                  # entries of one listing or search
MARKDOWN_CHARS = 200_000        # characters of Markdown sent
SECTION_RE = re.compile(r"^## (Photo|Video) details\s*$", re.M)
FACT_RE = re.compile(r"^- ([^:]+):\s*(.*)$")
GEO_RE = re.compile(r"geo:(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)")
PHOTO_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".heic", ".heif"}
VIDEO_EXT = {".mov", ".mp4", ".m4v", ".3gp", ".avi"}


class FileError(ValueError):
    """The collection or the file asked for is not one this can show."""


# ── naming a file ────────────────────────────────────────────────────────────

def collections(paths: Paths) -> list[dict[str, Any]]:
    """The collections that can be browsed: registered source folders that exist."""
    out = []
    for name, root in locations.source_roots(paths).locations:
        if Path(root).is_dir():
            out.append({"name": name, "root": root, "indexed": (paths.index / name).is_dir()})
    return out


def root_of(paths: Paths, collection: str) -> Path:
    root = locations.source_root(paths, collection)
    if root is None or not root.is_dir():
        raise FileError(f"{collection!r} is not a collection with a source folder here")
    return root.resolve()


def _clean(rel: str) -> str:
    rel = (rel or "").replace("\\", "/").strip("/")
    if "\x00" in rel or ".." in rel.split("/"):
        raise FileError(f"{rel!r} is not a path inside the collection")
    return rel


def resolve(paths: Paths, collection: str, rel: str) -> tuple[Path, Path, str]:
    """(source folder, the path in it, the cleaned relative path); the path is inside the folder or this raises."""
    root = root_of(paths, collection)
    rel = _clean(rel)
    src = (root / rel).resolve() if rel else root
    if not src.is_relative_to(root):
        raise FileError(f"{rel!r} is not inside the collection")
    return root, src, rel


def source_file(paths: Paths, collection: str, rel: str) -> Path:
    """An existing file of the collection (for showing the original), or FileError."""
    _root, src, _rel = resolve(paths, collection, rel)
    if not src.is_file():
        raise FileError("the file is not in the source folder any more")
    return src


def locate(paths: Paths, src: str) -> str:
    """``collection/path`` of an absolute source path ("" when it is in no registered folder)."""
    try:
        return locations.source_roots(paths).rel(Path(src)).as_posix()
    except ValueError:
        return ""


def _dirs(paths: Paths, collection: str, rel: str) -> tuple[Path, Path, Path]:
    """(index folder, Markdown file, rel-less parent) of a file of the collection."""
    p = Path(rel)
    return (paths.index / collection / p.parent / p.stem, paths.markup / collection / p.parent / (p.stem + ".md"), p.parent)


def _guess(name: str) -> str:
    ext = Path(name).suffix.lower()
    return "photo" if ext in PHOTO_EXT else "video" if ext in VIDEO_EXT else "pages"


def kind_of(idx: Path) -> str:
    """What the last conversion recorded: photo | video | timeline | pages | "" (not indexed)."""
    meta = read_json(idx / META_FILE)
    if not meta:
        return ""
    branches = set(((meta.get("conversion") or {}).get("branches")) or {})
    for k in ("photo", "video", "timeline"):
        if branches == {k}:
            return k
    return "pages"


# ── browsing ─────────────────────────────────────────────────────────────────

def _entry(paths: Paths, collection: str, rel: str, st: os.stat_result) -> dict[str, Any]:
    idx, _md, _par = _dirs(paths, collection, rel)
    kind = kind_of(idx)
    if not kind and (idx / "outcome.json").exists():
        state = "failed"
    else:
        state = "indexed" if kind else "not indexed"
    return {"name": Path(rel).name, "path": rel, "size": st.st_size, "mtime": st.st_mtime,
            "kind": kind or _guess(rel), "state": state}


def listing(paths: Paths, collection: str, prefix: str = "", q: str = "", limit: int = MAX_LIST) -> dict[str, Any]:
    """The folders and files directly inside *prefix* (``{"folders": [{name, path}], "files": [entry]}``), or, with *q*, the files
    anywhere in the collection whose path contains it (case-insensitive), at most *limit*."""
    root, base, prefix = resolve(paths, collection, prefix)
    limit = max(1, min(int(limit or MAX_LIST), MAX_LIST))
    q = (q or "").strip().lower()
    folders: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    more = False
    if q:
        for dp, dn, fn in os.walk(root):
            dn[:] = sorted(d for d in dn if not d.startswith("."))
            for f in sorted(fn):
                if f.startswith("."):
                    continue
                full = Path(dp) / f
                rel = full.relative_to(root).as_posix()
                if q in rel.lower():
                    if len(files) >= limit:
                        more = True
                        break
                    with contextlib.suppress(OSError):
                        files.append(_entry(paths, collection, rel, full.stat()))
            if more:
                break
    else:
        try:
            with os.scandir(base) as it:
                items = sorted(it, key=lambda e: e.name.lower())
        except OSError as exc:
            raise FileError(f"cannot list the folder: {exc.strerror or exc}") from exc
        for e in items:
            if e.name.startswith("."):
                continue
            rel = f"{prefix}/{e.name}" if prefix else e.name
            try:
                if e.is_dir(follow_symlinks=False):
                    folders.append({"name": e.name, "path": rel})
                elif e.is_file():
                    if len(files) >= limit:
                        more = True
                        break
                    files.append(_entry(paths, collection, rel, e.stat()))
            except OSError:
                continue
    return {"collection": collection, "prefix": prefix, "q": q, "folders": folders, "files": files, "more": more}


# ── one file ─────────────────────────────────────────────────────────────────

def _epoch(value: Any) -> float | None:
    """Seconds since 1970 of a recorded time: a number, ``2026-11-20 19:48`` (local) or ``2026-11-23T17:39:31Z`` (UTC)."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    for fmt, utc in (("%Y-%m-%dT%H:%M:%SZ", True), ("%Y-%m-%d %H:%M", False), ("%Y-%m-%d %H:%M:%S", False)):
        with contextlib.suppress(ValueError):
            t = time.strptime(text, fmt)
            return float(calendar.timegm(t) if utc else time.mktime(t))
    return None


def _mtime(p: Path) -> float | None:
    with contextlib.suppress(OSError):
        return p.stat().st_mtime
    return None


def _sha_of(md: Path, meta: dict[str, Any]) -> str:
    sha = str(meta.get("src_sha256") or "")
    if not sha:
        with contextlib.suppress(OSError):
            sha = md.with_name(md.name + ".sha256").read_text(encoding="utf-8").split("\n")[0].strip()
    return sha


def _facts(text: str) -> list[dict[str, str]]:
    """The ``- Label: value`` lines of the Photo / Video details section of the Markdown, in order."""
    m = SECTION_RE.search(text)
    if not m:
        return []
    out = []
    for line in text[m.end():].split("\n"):
        if line.startswith("## "):
            break
        f = FACT_RE.match(line)
        if f:
            out.append({"label": f.group(1).strip(), "value": f.group(2).strip()})
    return out


def _row(step: str, result: str, by: str = "", when: Any = None, current: bool | None = None, note: str = "",
         state: str = "done") -> dict[str, Any]:
    """One line of the ledger: ``state`` is done | missing | n/a | failed; ``current`` says whether what produced it is still what
    would produce it now (None when that does not apply)."""
    return {"step": step, "state": state, "result": result, "by": by, "when": _epoch(when), "current": current, "note": note}


def _described(paths: Paths, sha: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(every description kept for the checksum, every failure kept), the current model's first."""
    con = describe._cache(paths)
    if con is None or not sha:
        return [], []
    cur = describe.current_id(models.describe_selection()[0])
    try:
        with contextlib.closing(con):
            d = [{"id": i, "description": desc, "tags": _tags(tags), "people": ppl, "at": at, "current": i == cur}
                 for i, desc, tags, ppl, at in con.execute("SELECT id, description, tags, people, at FROM d WHERE sha = ?", (sha,))]
            f = [{"id": i, "why": why, "at": at, "current": i == cur}
                 for i, why, at in con.execute("SELECT id, why, at FROM failed WHERE sha = ?", (sha,))]
    except sqlite3.Error:
        return [], []
    d.sort(key=lambda r: not r["current"])
    return d, f


def _tags(raw: Any) -> list[str]:
    import json

    with contextlib.suppress(ValueError, TypeError):
        v = json.loads(raw)
        if isinstance(v, list):
            return [str(x) for x in v]
    return []


def _records(paths: Paths, idx: Path, md: Path, collection: str, rel: str) -> list[dict[str, Any]]:
    """Every derived file on disk for this file: the index folder, the Markdown and its sidecars, and the published copies."""
    out: list[dict[str, Any]] = []

    def add(group: str, p: Path, note: str = "") -> None:
        with contextlib.suppress(OSError):
            st = p.stat()
            out.append({"group": group, "name": p.name, "path": str(p), "size": st.st_size, "mtime": st.st_mtime, "note": note})
    if idx.is_dir():
        for f in sorted(idx.iterdir()):
            if f.is_file():
                add("index folder", f)
    for suffix in (".md", ".md.sha256", ".md.geo.json", ".trace.json"):
        add("markup", md.with_name(md.stem + suffix))
    live_md, live_idx = paths.live_markup(), paths.live_index()
    gen = paths.current_gen()
    if live_md is not None and gen is not None:
        p = live_md / md.relative_to(paths.markup)
        add(f"published ({gen.name})", p, "same file as the workspace copy" if _same(p, md) else "")
        d = live_idx / idx.relative_to(paths.index) if live_idx is not None else None
        if d is not None and d.is_dir():
            for f in sorted(d.iterdir()):
                if f.is_file():
                    add(f"published ({gen.name})", f, "same file as the workspace copy" if _same(f, idx / f.name) else "")
    return out


def _same(a: Path, b: Path) -> bool:
    with contextlib.suppress(OSError):
        return os.path.samefile(a, b)
    return False


def dossier(paths: Paths, collection: str, rel: str) -> dict[str, Any]:
    """Everything recorded about one file (see the module docstring)."""
    root, src, rel = resolve(paths, collection, rel)
    if not rel:
        raise FileError("name a file")
    idx, md, _par = _dirs(paths, collection, rel)
    exists = src.is_file()
    st = src.stat() if exists else None
    meta = read_json(idx / META_FILE)
    kind = kind_of(idx)
    sha = _sha_of(md, meta)
    try:
        text = md.read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""
    conv = meta.get("conversion") or {}
    facts = _facts(text)
    geo = GEO_RE.search(text)
    out: dict[str, Any] = {
        "collection": collection, "path": rel, "name": Path(rel).name, "exists": exists, "kind": kind or _guess(rel),
        "indexed": bool(kind), "size": st.st_size if st else None, "mtime": st.st_mtime if st else None,
        "sha": sha, "changed": bool(st and meta.get("src_bytes") not in (None, st.st_size)),
        "outcome": read_json(idx / "outcome.json") or None, "pages": conv.get("pages"),
        "facts": facts, "geo": {"lat": float(geo.group(1)), "lon": float(geo.group(2))} if geo else None,
        "shares_folder": _shares(src, exists),
    }
    ledger: list[dict[str, Any]] = []
    # convert
    if kind:
        trace = paths.markup / str(conv.get("trace") or "")
        when = _epoch((read_json(trace) or {}).get("written_at")) if conv.get("trace") else None
        by = ", ".join(conv.get("readers") or []) or kind
        ledger.append(_row("convert", f"{kind}" + (f", {conv['pages']} page(s)" if kind == "pages" and conv.get("pages") else ""),
                           by, when or _mtime(md), not out["changed"], "the file changed since it was converted" if out["changed"] else ""))
    else:
        failed = out["outcome"]
        ledger.append(_row("convert", (failed or {}).get("error") or (failed or {}).get("reason") or "not converted", "", None, None,
                           "remembered failure" if failed else "", "failed" if failed else "missing"))
    if kind == "photo":
        ledger += _photo_rows(paths, out, idx, md, text, sha, facts, geo is not None)
    elif kind == "video":
        ledger += _geo_row(paths, md, text, geo is not None)
    if kind:
        n = meta.get("nodes")
        serving = ""
        with contextlib.suppress(Exception):
            serving = models.serving_model(paths)
        ledger.append(_row("embed", f"{n} passage(s), {meta.get('dim')} dimensions", str(meta.get("model") or ""), meta.get("built_at"),
                           (meta.get("model") == serving) if serving else None))
        gen = paths.current_gen()
        pub = paths.live_markup()
        in_live = bool(pub and (pub / md.relative_to(paths.markup)).is_file())
        ledger.append(_row("publish", f"in {gen.name}" if in_live and gen else "not in the published generation",
                           gen.name if gen else "", _mtime(pub / md.relative_to(paths.markup)) if in_live and pub else None,
                           in_live or None, "", "done" if in_live else "missing"))
    out["ledger"] = ledger
    out["index"] = {k: meta.get(k) for k in ("model", "dim", "nodes", "built_at", "convert", "src_path", "src_bytes",
                                              "build_s", "convert_s", "embed_s")} if meta else None
    out["records"] = _records(paths, idx, md, collection, rel)
    out["markdown"] = text[:MARKDOWN_CHARS]
    out["markdown_cut"] = len(text) > MARKDOWN_CHARS
    return out


def _shares(src: Path, exists: bool) -> list[str]:
    """Other files of the folder with the same name but another extension: they share one index folder (a known limit)."""
    if not exists:
        return []
    with contextlib.suppress(OSError):
        return sorted(f.name for f in src.parent.iterdir() if f.stem == src.stem and f.name != src.name and f.is_file())
    return []


def _geo_row(paths: Paths, md: Path, text: str, has_geo: bool) -> list[dict[str, Any]]:
    if not has_geo:
        return [_row("locate", "no position recorded", state="n/a")]
    stamp = maps.stamp_path(md)
    refs = ""
    m = re.search(r"^- Reference data:\s*(.*)$", text, re.M)
    if m:
        refs = m.group(1).strip()
    if not stamp.exists():
        return [_row("locate", "position found, places not looked up", state="missing")]
    return [_row("locate", "area, town and places around", refs, _mtime(stamp), maps.is_current(md, text, maps.map_state(paths)))]


def _photo_rows(paths: Paths, out: dict[str, Any], idx: Path, md: Path, text: str, sha: str, facts: list[dict[str, str]],
                has_geo: bool) -> list[dict[str, Any]]:
    rows = _geo_row(paths, md, text, has_geo)
    # faces
    found = people.faces_of_photo(paths, sha)
    out["faces"] = found["faces"] if found else []
    if found:
        rows.append(_row("faces", f"{found['found']} found", found["detector"], found["at"], found["detector"].startswith(faces.DETECTOR)))
    else:
        reason = faces.available(paths)
        rows.append(_row("faces", reason or "not looked at yet", state="missing" if not reason else "n/a"))
    # describe
    descs, bad = _described(paths, sha)
    out["descriptions"] = descs
    cur = next((d for d in descs if d["current"]), None) or (descs[0] if descs else None)
    if cur:
        rows.append(_row("describe", f"{len(cur['tags'])} tags", cur["id"], cur["at"], cur["current"],
                         "" if cur["current"] else "made by another model or prompt than the current one"))
    elif bad:
        rows.append(_row("describe", bad[0]["why"] or "the model could not describe it", bad[0]["id"], bad[0]["at"], bad[0]["current"],
                         "not tried again by the automatic step", "failed"))
    else:
        rows.append(_row("describe", "not described", state="missing"))
    # picture vector
    pj = read_json(idx / pictures.META)
    sel = pictures.selected()
    if pj and (idx / pictures.FILE).exists():
        rows.append(_row("picture vector", f"{pj.get('dim')} dimensions", str(pj.get("model") or ""), _mtime(idx / pictures.META),
                         pictures.is_current(idx, sha, sel) if sel else None))
    else:
        rows.append(_row("picture vector", "no picture vector" if sel else "picture vectors are off", state="missing" if sel else "n/a"))
    return rows
