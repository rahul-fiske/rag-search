"""Read a whole converted document, and list documents by name (stdlib only; no models, works while the daemons warm up).

Search and grep return passages and matching lines; a table split into passages loses its header and most of its rows.
This is the other half of "search, then read": given a collection and a document name (what ``rag_search``, ``rag_grep``
and ``rag_list_collections`` return), it hands back the document's converted Markdown exactly as the index holds it
(a ``.md`` or ``.txt`` source is copied as it is: tables, front matter and relative links intact), in slices of at most
``MAX_CHARS`` characters that can be put back together with ``offset`` / ``next_offset``.

The text comes from the published generation's Markdown (the same files grep scans), never from the source folder, so a
read is consistent with what search finds.  A name is only ever looked up in that generation's catalogue; the file path
is built from the catalogue's own entry, and refused when it is a link or leaves the collection's folder, so no
identifier can reach another file.  Access rules are applied by the caller (``api.read_document`` / ``api.list_documents``).
"""

from __future__ import annotations

import difflib
import fnmatch
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any

DEFAULT_MAX_CHARS = 20_000           # one call stays well inside an MCP result
MAX_CHARS = 40_000
MAX_FILE_BYTES = 64 * 1024 * 1024
DEFAULT_LIST_LIMIT = 200
MAX_LIST_LIMIT = 1000
_PAGE_RE = re.compile(r"<!-- page (\d+) -->")


class DocError(ValueError):
    """The request cannot be served; the message says why (and what was meant, when that can be guessed)."""


def source_path(doc: dict[str, Any]) -> str:
    """Where the source file was inside the collection (the catalogue keeps the file name; the folder is in the name)."""
    parent = PurePosixPath(doc["name"]).parent
    src = doc.get("source") or ""
    return src if str(parent) == "." else f"{parent}/{src}"


def _keys(doc: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    """The spellings by which a document can be named, from the most to the least specific."""
    name = doc["name"]
    rel_src = source_path(doc)
    return {"exact": (name, rel_src),
            "short": tuple(x for x in (doc.get("source") or "", PurePosixPath(name).name) if x)}


def _plain(ident: str) -> str:
    ident = (ident or "").strip().replace("\\", "/")
    while ident.startswith("./"):
        ident = ident[2:]
    return ident


def resolve(docs: list[dict[str, Any]], collection: str, ident: str) -> dict[str, Any]:
    """The catalogue entry that *ident* names.  Accepted: the document ``name`` (rag_grep ``doc``, rag_list_collections
    ``name``), the same with the source's extension (``folder/file.md``), either of those with the collection in front,
    and -- when only one document has it -- a bare file name or file name without extension (rag_search ``source`` /
    ``file``).  Anything with ``..``, an absolute path or a NUL is refused."""
    raw = _plain(ident)
    if not raw:
        raise DocError("no document given: pass the name rag_search, rag_grep or rag_list_documents returned")
    if "\x00" in raw or raw.startswith("/") or ".." in PurePosixPath(raw).parts:
        raise DocError(f"{ident!r} is not a document of collection {collection!r}")
    wanted = [raw]
    if raw.startswith(collection + "/"):
        wanted.append(raw[len(collection) + 1:])
    for tier in ("exact", "short"):
        for fold in (False, True):
            hits = []
            for d in docs:
                keys = _keys(d)[tier]
                keys = tuple(k.casefold() for k in keys) if fold else keys
                if any((w.casefold() if fold else w) in keys for w in wanted):
                    hits.append(d)
            if len(hits) == 1:
                return hits[0]
            if len(hits) > 1:
                names = ", ".join(sorted(d["name"] for d in hits)[:8])
                raise DocError(f"{ident!r} matches {len(hits)} documents in {collection!r} ({names}"
                               f"{', ...' if len(hits) > 8 else ''}): use the full name from rag_grep or rag_list_documents")
    names = [d["name"] for d in docs]
    close = difflib.get_close_matches(raw, names, n=5, cutoff=0.6)
    hint = f" Did you mean: {', '.join(close)}?" if close else " Use rag_list_documents to see the names."
    raise DocError(f"no document {ident!r} in collection {collection!r}.{hint}")


def markup_file(markup_root: Path | None, collection: str, doc: dict[str, Any]) -> Path:
    """The converted Markdown of a catalogue entry, refused when it is a link or not inside the collection's folder."""
    if markup_root is None:
        raise DocError("nothing is published yet")
    base = os.path.realpath(markup_root / collection)
    path = markup_root / collection / (doc["name"] + ".md")
    if path.is_symlink() or not os.path.realpath(path).startswith(base + os.sep):
        raise DocError(f"{doc['name']!r} is not a document of collection {collection!r}")
    if not path.is_file():
        raise DocError(f"the converted text of {doc['name']!r} is not available (the collection may be rebuilding)")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise DocError(f"{doc['name']!r} is larger than {MAX_FILE_BYTES // 2**20} MB: use rag_grep")
    return path


def _text(path: Path) -> str:
    """The file's characters, nothing normalised (line endings stay as they are)."""
    return path.read_bytes().decode("utf-8", errors="replace")


def page_slice(text: str, pages: str) -> tuple[str, dict[str, Any]]:
    """The part of *text* on pages ``N``, ``N-M`` or ``N-`` (markers ``<!-- page N -->``; text without markers is page 1)."""
    marks = [(int(m.group(1)), m.start()) for m in _PAGE_RE.finditer(text)]
    total = max((n for n, _ in marks), default=1)
    m = re.fullmatch(r"\s*(\d+)\s*(?:(-)\s*(\d*))?\s*", pages or "")
    if not m:
        raise DocError(f"pages must look like 3, 2-4 or 3- (got {pages!r})")
    first = int(m.group(1))
    last = int(m.group(3)) if m.group(3) else (total if m.group(2) else first)
    if first < 1 or last < first or first > total:
        raise DocError(f"pages {pages!r} are outside this document (it has pages 1-{total})")
    last = min(last, total)
    label = f"{first}-{last}" if last != first else str(first)
    if not marks:
        return text, {"pages": label, "total_pages": 1}
    start = 0 if first == 1 else next((pos for n, pos in marks if n >= first), len(text))
    end = next((pos for n, pos in marks if n > last), len(text))
    return text[start:end], {"pages": label, "total_pages": total}


def read(markup_root: Path | None, collection: str, docs: list[dict[str, Any]], ident: str, *, offset: int = 0,
         max_chars: int = DEFAULT_MAX_CHARS, pages: str = "") -> dict[str, Any]:
    entry = resolve(docs, collection, ident)
    text = _text(markup_file(markup_root, collection, entry))
    extra: dict[str, Any] = {}
    if pages and pages.strip():
        text, extra = page_slice(text, pages)
        if not text:
            extra["note"] = "no text was read from these pages (a blank page, or a picture without text)"
    try:
        offset, max_chars = int(offset), int(max_chars)
    except (TypeError, ValueError) as exc:
        raise DocError("offset and max_chars must be whole numbers") from exc
    offset = max(0, offset)
    max_chars = min(max(1, max_chars), MAX_CHARS)
    total = len(text)
    piece = text[offset:offset + max_chars]
    end = offset + len(piece)
    nxt = end if end < total else None
    return {"collection": collection, "doc": entry["name"], "source": source_path(entry), "text": piece,
            "offset": offset, "next_offset": nxt, "total_chars": total, "truncated": nxt is not None,
            "indexed_at": entry.get("indexed_at") or None, **extra}


def list_documents(markup_root: Path | None, collection: str, docs: list[dict[str, Any]], *, prefix: str = "",
                   glob: str = "", limit: int = DEFAULT_LIST_LIMIT, offset: int = 0) -> dict[str, Any]:
    """Documents whose name starts with *prefix* and matches *glob* (``*`` crosses folders), by name, a page at a time."""
    prefix = _plain(prefix)
    if prefix.startswith(collection + "/"):
        prefix = prefix[len(collection) + 1:]
    if "\x00" in prefix or ".." in PurePosixPath(prefix).parts:
        raise DocError(f"{prefix!r} is not a folder of collection {collection!r}")
    try:
        limit = min(max(1, int(limit)), MAX_LIST_LIMIT)
        offset = max(0, int(offset))
    except (TypeError, ValueError) as exc:
        raise DocError("limit and offset must be whole numbers") from exc
    glob = (glob or "").strip()
    hits = [d for d in docs
            if d["name"].startswith(prefix)
            and (not glob or fnmatch.fnmatchcase(d["name"], glob) or fnmatch.fnmatchcase(source_path(d), glob))]
    hits.sort(key=lambda d: d["name"])
    page = hits[offset:offset + limit]
    out = []
    for d in page:
        try:
            chars = len(_text(markup_file(markup_root, collection, d)))
        except (DocError, OSError):
            chars = None
        out.append({"name": d["name"], "source": source_path(d), "chars": chars, "indexed_at": d.get("indexed_at") or None})
    nxt = offset + len(page)
    return {"collection": collection, "prefix": prefix, "glob": glob, "total_matches": len(hits), "returned": len(out),
            "next_offset": nxt if nxt < len(hits) else None, "documents": out}
