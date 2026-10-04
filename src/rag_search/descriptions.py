"""Short, human- or agent-written descriptions of collections (stdlib only), used by
`rag_list_collections` / `rag-search list` to summarize a collection instead of (or
alongside) enumerating every document -- descriptions cannot be generated automatically,
since a collection is just whatever folder of documents someone indexed, so this is how
a description gets supplied (normally by the calling LLM itself, via the
`rag_describe_collection` MCP tool, after it has examined a collection once; the
`rag-search describe` CLI command and the dashboard are the human override/audit paths).
Every writer goes through `api.describe_collection`.

    <home>/descriptions.json:  {"version": 1, "collections": {"manuals": "Product manuals..."}}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .paths import CachedFile, Paths, file_lock, write_json_atomic

DESCRIPTIONS_VERSION = 1
MAX_LENGTH = 500


class DescriptionError(ValueError):
    """A request the user (or calling LLM) can fix (bad name, too long, ...)."""


def _load_file(f: Path) -> tuple[dict[str, str], str]:
    try:
        raw = f.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}, ""
    except OSError as exc:
        return {}, f"{f}: {exc}"
    try:
        data = json.loads(raw)
        coll = data.get("collections", {}) if isinstance(data, dict) else None
        if not isinstance(coll, dict):
            raise ValueError('expected {"collections": {name: "text"}}')
        return {str(k): str(v) for k, v in coll.items() if str(v).strip()}, ""
    except ValueError as exc:
        return {}, f"{f}: {exc}"


def load_descriptions(paths: Paths) -> tuple[dict[str, str], str]:
    """Return (collection name -> description, error). A broken file yields {} and reports why."""
    return _load_file(paths.descriptions_file)


_STORES: dict[str, CachedFile] = {}


def cached_descriptions(paths: Paths) -> dict[str, str]:
    """Descriptions through a process-wide cache (re-read only when the file changed): what
    `catalog.list_view` uses, since it runs on every listing and every dashboard refresh."""
    key = str(paths.descriptions_file)
    s = _STORES.get(key)
    if s is None:
        s = _STORES[key] = CachedFile(paths.descriptions_file, _load_file)
    return dict(s.get())


def save_descriptions(paths: Paths, by_name: dict[str, str]) -> Path:
    f = paths.descriptions_file
    body = {"version": DESCRIPTIONS_VERSION,
            "collections": {n: d for n, d in sorted(by_name.items()) if d.strip()}}
    write_json_atomic(f, body, newline=True)
    return f


def get_description(paths: Paths, name: str) -> str:
    folded = {n.casefold(): d for n, d in cached_descriptions(paths).items()}
    return folded.get(name.casefold(), "")


def set_description(paths: Paths, name: str, text: str) -> dict[str, Any]:
    """Set (or, with blank *text*, clear) one collection's description."""
    from .catalog import canonical_name

    text = (text or "").strip()
    if len(text) > MAX_LENGTH:
        raise DescriptionError(f"description is {len(text)} characters; keep it to "
                                f"{MAX_LENGTH} or fewer -- it is returned on every "
                                "rag_list_collections call, so a long one defeats the point")
    with file_lock(paths.descriptions_file):
        by_name, error = load_descriptions(paths)
        if error:
            raise DescriptionError(f"cannot change descriptions while {error}; fix or remove "
                                   "that file")
        try:
            name_canon = canonical_name(paths, name, extra=by_name)
        except ValueError as exc:
            raise DescriptionError(str(exc)) from None
        # one entry per collection, whatever spelling an older write used
        for old in [n for n in by_name if n.casefold() == name_canon.casefold()]:
            del by_name[old]
        if text:
            by_name[name_canon] = text
        save_descriptions(paths, by_name)
    return {"collection": name_canon, "description": text, "changed": True}


def forget(paths: Paths, name: str) -> bool:
    """Drop *name*'s description (used when a collection is deleted).  True if one existed."""
    with file_lock(paths.descriptions_file):
        by_name, error = load_descriptions(paths)
        if error:
            return False
        gone = [n for n in by_name if n.casefold() == name.casefold()]
        for n in gone:
            del by_name[n]
        if gone:
            save_descriptions(paths, by_name)
    return bool(gone)
