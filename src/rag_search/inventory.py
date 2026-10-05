"""Everything about one collection, for the administrator (stdlib only).

``collection_info(paths, name)`` answers "what is this collection, where does it live on disk,
how big is it, when was it indexed and is it up to date?" for the dashboard's Collections tab
and ``rag-search collection info``.  It reads rag-search's own files (workspace, published
generation, job event logs) and walks the collection's source folder once (counting files and
bytes; sources are only listed and stat'ed, never opened).

It is an administrator view: it shows folder paths, so it is not an MCP tool and the dashboard
asks for it as ``cli``.  It is computed on demand (one collection at a time), never in the
dashboard's polling loop.
"""

from __future__ import annotations

import calendar
import os
import time
from pathlib import Path
from typing import Any

from . import descriptions, jobs, locations, policy
from .catalog import live_catalog
from .paths import (
    ALL_DIR,
    EMB_FILE,
    MERGE_MANIFEST,
    META_FILE,
    NODES_FILE,
    Paths,
    index_dir_for,
    is_plain_name,
    read_json,
)

LIST_LIMIT = 25          # names listed per "needs attention" group
RUNS_LOOKED_AT = 10      # how far back to look for the last run that touched the collection


class InventoryError(ValueError):
    """Unknown or invalid collection name."""


def _du(folder: Path) -> tuple[int, int]:
    """(files, bytes) under *folder* (0, 0 when it does not exist)."""
    n = size = 0
    for dp, _dirs, files in os.walk(folder):
        for f in files:
            try:
                size += os.lstat(os.path.join(dp, f)).st_size
                n += 1
            except OSError:
                pass
    return n, size


def _epoch(stamp: str) -> float | None:
    try:
        return float(calendar.timegm(time.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ")))
    except (TypeError, ValueError):
        return None


def _find(paths: Paths, name: str) -> str:
    name = (name or "").strip()
    if not is_plain_name(name):
        raise InventoryError(f"{name!r} is not a collection name")
    from .catalog import known_names

    known = known_names(paths)
    hit = name if name in known else next((k for k in known if k.casefold() == name.casefold()), "")
    if not hit:
        raise InventoryError(f"no collection named {name!r} (known: {', '.join(known) or 'none'})")
    return hit


def _kind(paths: Paths, coll: str, locs: dict[str, str]) -> str:
    if coll in locs:
        return "location"
    if locations.is_imported(paths, coll):
        return "imported"
    return "imported"


def _source(paths: Paths, coll: str, kind: str, locs: dict[str, str]) -> dict[str, Any]:
    """Where the documents are, whether that can be read now, and what is in it."""
    if kind != "location":                 # imported, or indexed earlier with no registered folder
        return {"folder": None, "reachable": None, "files": None, "bytes": None,
                "unsupported": None, "list": []}
    roots = locations.source_roots(paths)
    folder = Path(locs[coll])
    out: dict[str, Any] = {"folder": str(folder), "reachable": locations.reachable(folder)}
    files: list[Path] = []
    unsupported = 0
    if out["reachable"]:
        found, unsup = locations.scan_tree(folder, locations.workspace_excludes(paths))
        files, unsupported = found, len(unsup)
    size = 0
    stats: list[tuple[Path, float]] = []
    for f in files:
        try:
            st = f.stat()
        except OSError:
            continue
        size += st.st_size
        stats.append((f, st.st_mtime))
    out.update({"files": len(files) if out["reachable"] else None,
                "bytes": size if out["reachable"] else None,
                "unsupported": unsupported if out["reachable"] else None,
                "list": stats, "roots": roots})
    return out


def _supported() -> frozenset[str]:
    from .paths import SUPPORTED_EXTENSIONS

    return SUPPORTED_EXTENSIONS


def _workspace(paths: Paths, coll: str) -> dict[str, Any]:
    idx, md = paths.index / coll, paths.markup / coll
    all_dir = idx / ALL_DIR
    manifest = read_json(all_dir / MERGE_MANIFEST)
    metas: dict[str, dict[str, Any]] = {}
    incomplete: list[str] = []
    if idx.is_dir():
        for dp, dirnames, filenames in os.walk(idx):
            dirnames[:] = [d for d in dirnames if d != ALL_DIR and not d.startswith(".")]
            rel = Path(dp).relative_to(idx).as_posix()
            if META_FILE in filenames:
                meta = read_json(Path(dp) / META_FILE)
                if meta:
                    metas[rel] = meta
            elif NODES_FILE in filenames or EMB_FILE in filenames:
                incomplete.append(rel)
    md_files, md_bytes = _du(md)
    idx_files, idx_bytes = _du(idx)
    all_files, all_bytes = _du(all_dir)
    return {"index_folder": str(idx), "markdown_folder": str(md),
            "index_exists": idx.is_dir(), "markdown_exists": md.is_dir(),
            "index_bytes": idx_bytes, "merged_index_bytes": all_bytes,
            "per_document_index_bytes": idx_bytes - all_bytes,
            "markdown_files": sum(1 for _ in md.rglob("*.md")) if md.is_dir() else 0,
            "markdown_bytes": md_bytes,
            "documents": len(metas), "incomplete": sorted(incomplete),
            "merged": {"documents": len(manifest.get("docs", [])),
                       "chunks": manifest.get("nodes", 0),
                       "built_at": manifest.get("built_at", ""),
                       "sha": manifest.get("sha256", "")} if manifest else None,
            "_metas": metas, "_files": idx_files + md_files}


def _conversion(metas: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Pages per branch and outcome, time and cost over the documents' stored conversion
    summaries, plus the documents with pages docling itself graded poor (``poor_documents``) or
    that ended low-confidence (``low_documents``)."""
    from .core.conversion import trace

    out = trace.aggregate(m.get("conversion") for m in metas.values())
    poor, low = [], []
    for rel in sorted(metas):
        conv = metas[rel].get("conversion") or {}
        if conv.get("poor_pages"):
            poor.append({"doc": rel, "pages": conv["poor_pages"][:10]})
        if conv.get("low_pages"):
            low.append({"doc": rel, "pages": conv["low_pages"][:10]})
    out["poor_documents"] = {"count": len(poor), "items": poor[:LIST_LIMIT]}
    out["low_documents"] = {"count": len(low), "items": low[:LIST_LIMIT]}
    return out


def _build(metas: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not metas:
        return {}
    vals = list(metas.values())

    def one(key: str) -> Any:
        seen = {str(m.get(key, "")) for m in vals}
        return vals[0].get(key) if len(seen) == 1 else "mixed"

    built = sorted(str(m.get("built_at", "")) for m in vals if m.get("built_at"))
    timed = [m for m in vals if "build_s" in m]
    return {"model": one("model"), "model_revision": one("model_revision"), "dim": one("dim"),
            "chunk_size": one("chunk_size"), "chunk_overlap": one("chunk_overlap"),
            "chunks": sum(int(m.get("nodes", 0) or 0) for m in vals),
            "first_indexed": built[0] if built else "", "last_indexed": built[-1] if built else "",
            "build_seconds": round(sum(float(m.get("build_s", 0) or 0) for m in timed), 1)
            if timed else None,
            "convert_seconds": round(sum(float(m.get("convert_s", 0) or 0) for m in timed), 1)
            if timed else None,
            "embed_seconds": round(sum(float(m.get("embed_s", 0) or 0) for m in timed), 1)
            if timed else None}


def _published(paths: Paths, coll: str) -> dict[str, Any] | None:
    cat = live_catalog(paths)
    c = next((x for x in cat.get("collections", []) if x.get("collection") == coll), None)
    if c is None:
        return None
    gen = paths.current_gen()
    return {"generation": cat.get("generation"), "published_at": cat.get("published_at", ""),
            "documents": len(c.get("documents", [])), "chunks": c.get("chunks", 0),
            "sha": c.get("manifest_sha", ""), "origin": c.get("origin", "generated"),
            "index_folder": str(gen / "index" / coll) if gen else None,
            "markdown_folder": str(gen / "markup" / coll) if gen else None,
            "index_bytes": c.get("index_bytes"), "markdown_bytes": c.get("markdown_bytes"),
            "documents_list": c.get("documents", [])}


def _last_run(paths: Paths, coll: str) -> dict[str, Any] | None:
    """The newest indexing run that handled at least one document of *coll*."""
    for rec in jobs.all_records(paths)[:RUNS_LOOKED_AT]:
        docs = jobs.documents(paths, rec["id"], 0, collection=coll)
        counts = docs["by_collection"].get(coll)
        if not counts:
            continue
        errors = jobs.documents(paths, rec["id"], LIST_LIMIT, status="error", collection=coll)
        return {"job": rec["id"], "status": rec.get("status", ""), "mode": rec.get("mode", ""),
                "path": rec.get("path", ""), "started_at": rec.get("started_at"),
                "finished_at": rec.get("finished_at"), "counts": counts,
                "errors": [{"source": i.get("source", ""), "message": i.get("message", "")}
                           for i in errors["items"]]}
    return None


def collection_info(paths: Paths, name: str) -> dict[str, Any]:
    """Counts, folders, sizes, dates and state of collection *name* (see the module docstring)."""
    coll = _find(paths, name)
    locs, _ = locations.load(paths)
    kind = _kind(paths, coll, locs)
    src = _source(paths, coll, kind, locs)
    ws = _workspace(paths, coll)
    metas = ws.pop("_metas")
    ws.pop("_files")
    build = _build(metas)
    pub = _published(paths, coll)
    run = _last_run(paths, coll)

    # documents found in the source folder but not indexed, and indexed but changed since
    not_indexed: list[str] = []
    modified: list[str] = []
    roots = src.pop("roots", None)
    stats = src.pop("list")
    if roots is not None:
        for f, mtime in stats:
            try:
                rel_dir = index_dir_for(f, roots, paths.index).relative_to(paths.index / coll)
            except ValueError:
                continue
            meta = metas.get(rel_dir.as_posix())
            if meta is None:
                not_indexed.append(_rel(f, src["folder"]))
                continue
            built = _epoch(str(meta.get("built_at", "")))
            if built is not None and mtime > built + 1:
                modified.append(rel_dir.as_posix())
    reasons: dict[str, str] = {}
    if run:
        for e in run["errors"]:
            reasons[e["source"]] = e["message"]

    state, detail = _state(kind, src, ws, pub, not_indexed, modified)
    origin = locations.origin(paths, coll) if kind == "imported" else None
    rules = policy.current_rules(paths)
    allowed = rules.clients_for(coll)
    return {
        "collection": coll, "kind": kind,
        "description": descriptions.get_description(paths, coll),
        "state": state, "state_detail": detail,
        "access": "everyone" if allowed is None else allowed,
        "source": src,
        "workspace": ws,
        "published": {k: v for k, v in pub.items() if k != "documents_list"} if pub else None,
        "build": build,
        "disk": {"markdown_bytes": ws["markdown_bytes"], "index_bytes": ws["index_bytes"],
                 "total_bytes": ws["markdown_bytes"] + ws["index_bytes"],
                 "source_bytes": src.get("bytes"),
                 "note": "published generations share these files (hard links): no extra space"},
        "conversion": _conversion(metas),
        "attention": {
            "not_indexed": {"count": len(not_indexed), "names": not_indexed[:LIST_LIMIT],
                            "reasons": {n: reasons[Path(n).name] for n in not_indexed[:LIST_LIMIT]
                                        if Path(n).name in reasons}},
            "modified_since_indexed": {"count": len(modified), "names": modified[:LIST_LIMIT]},
            "incomplete": {"count": len(ws["incomplete"]), "names": ws["incomplete"][:LIST_LIMIT]},
            "errors_last_run": len(run["errors"]) if run else 0,
        },
        "last_run": run,
        "origin": origin,
    }


def _rel(f: Path, folder: str) -> str:
    """*f* relative to the collection's source folder (just the name when outside it)."""
    try:
        return f.relative_to(folder).as_posix()
    except ValueError:
        return f.name


def _state(kind: str, src: dict[str, Any], ws: dict[str, Any], pub: dict[str, Any] | None,
           not_indexed: list[str], modified: list[str]) -> tuple[str, str]:
    """(state, explanation): ok | pending | unpublished | unreachable | not_indexed | imported |
    unregistered."""
    merged = ws.get("merged")
    if kind == "imported":
        return ("imported", "imported from a collection export; it has no source documents and "
                "is never re-indexed" + ("" if pub else " (not published yet)"))
    if src.get("reachable") is False:
        return ("unreachable", "the source folder cannot be read right now; indexing leaves this "
                "collection as it was" + (" and search keeps serving it" if pub else ""))
    if not merged and not pub:
        return ("not_indexed", "nothing indexed yet: run `rag-search index new "
                "COLLECTION` (or the Index button)")
    if merged and (not pub or pub.get("sha") != merged.get("sha")):
        return ("unpublished", "the workspace holds changes search does not serve yet; they are "
                "published when the next indexing run finishes (or: rag-search index publish)")
    if not_indexed or modified:
        parts = []
        if not_indexed:
            parts.append(f"{len(not_indexed)} new document(s) not indexed")
        if modified:
            parts.append(f"{len(modified)} changed since they were indexed")
        return "pending", "; ".join(parts) + " -- run `rag-search index new` to catch up"
    return "ok", "published and up to date with its source folder"
