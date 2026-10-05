"""Read-only views of the live generation (stdlib only): listings and collection names."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from . import descriptions, policy
from .paths import Paths, is_plain_name
from .publish import read_catalog


def live_catalog(paths: Paths) -> dict[str, Any]:
    return read_catalog(paths.current_gen())


def collection_names(catalog: dict[str, Any]) -> list[str]:
    return [c["collection"] for c in catalog.get("collections", [])]


def known_names(paths: Paths) -> list[str]:
    """Every collection name rag-search knows of: the registered source locations and the imported
    collections -- a collection has no other origin.  (An index that is neither is a leftover; a full
    indexing run removes it.)"""
    from . import locations

    names: list[str] = []
    for n in (*locations.names(paths), *locations.imported_names(paths)):
        if n not in names:
            names.append(n)
    return names


def canonical_name(paths: Paths, name: str, extra: Iterable[str] = ()) -> str:
    """The spelling to use for *name*: an existing collection's own spelling when one matches
    case-insensitively (exact match first), else *name* as typed.  The one implementation of
    "which collection did they mean" shared by access rules, descriptions and collection
    management.  Raises ValueError for something that cannot be a collection name."""
    name = (name or "").strip()
    if not is_plain_name(name):
        raise ValueError(f"{name!r} is not a collection name (the name a folder was registered "
                         "with, e.g. 'manuals')")
    known = [*known_names(paths), *extra]
    if name in known:
        return name
    for k in known:
        if k.casefold() == name.casefold():
            return k
    return name


def list_view(paths: Paths, rules: policy.Rules, client: str, *, full: bool = False) -> dict[str, Any]:
    """What `rag_list_collections` returns for *client*: only what it is authorised to use.

    By default each collection is summarized (`document_count` + a human/agent-set
    `description`, if any) instead of enumerating every document -- a collection can hold
    hundreds of files, and the full per-document listing used to be sent unconditionally,
    bloating the calling LLM's context on every call. Pass `full=True` to also include the
    per-document `documents` list (used by the CLI and the dashboard, which display it).
    """
    cat = live_catalog(paths)
    visible = set(rules.visible(client, collection_names(cat)))
    by_name = descriptions.cached_descriptions(paths)
    desc_folded = {n.casefold(): d for n, d in by_name.items()}
    colls = []
    for c in cat.get("collections", []):
        if c["collection"] not in visible:
            continue
        docs = c.get("documents", [])
        entry = {
            "collection": c["collection"],
            "description": desc_folded.get(c["collection"].casefold(), ""),
            "chunks": c.get("chunks", 0),
            "document_count": len(docs),
            "index_bytes": c.get("index_bytes"),
            "markdown_bytes": c.get("markdown_bytes"),
            "source_bytes": c.get("source_bytes"),
            "built_at": c.get("built_at") or None,
            "build_seconds": c.get("build_seconds"),
            "build_seconds_documents": c.get("build_seconds_documents", 0),
            "model": c.get("model", ""),
            "origin": c.get("origin", "generated"),
        }
        if full:
            entry["documents"] = docs
        colls.append(entry)
    known = [c["build_seconds"] for c in colls if c["build_seconds"] is not None]
    out: dict[str, Any] = {
        "generation": cat.get("generation"),
        "published_at": cat.get("published_at"),
        "model": cat.get("model", ""),
        "collections": colls,
        "totals": {
            "collections": len(colls),
            "documents": sum(c["document_count"] for c in colls),
            "chunks": sum(c["chunks"] for c in colls),
            "index_bytes": sum(c["index_bytes"] or 0 for c in colls),
            "markdown_bytes": sum(c["markdown_bytes"] or 0 for c in colls),
            "source_bytes": sum(c["source_bytes"] or 0 for c in colls),
            "build_seconds": round(sum(known), 1) if known else None,
        },
    }
    if not colls:
        out["hint"] = ("Nothing is published yet. Register a folder of documents (rag-search "
                       "location add NAME FOLDER), then start indexing (rag_index_update / "
                       "rag-search index new).")
    elif not full:
        out["hint"] = ("Document listings are omitted by default -- pass documents=true to "
                        "rag_list_collections (or --full to `rag-search list`) to see them.")
    return out
