"""Page cache: what a reader produced for one page, kept so no reading is ever repeated (stdlib only).

``<workspace>/page_cache/<hh>/<key>.json`` holds the Markdown and the facts of one page.  The key is
a hash of *what was read* (the page's content hash from the profiler), *who read it* (reader id) and
*how* (the conversion settings): any change of source page, tool or setting is a different key, so a
hit is always valid and nothing needs invalidating by hand.

It makes an interrupted or cancelled run resumable at page granularity, lets a changed document
re-read only its changed pages, and lets two documents that contain the same page share the work.
Entries are written atomically (one file per page, so parallel workers never touch the same file) and
garbage-collected after a run: an entry that no stored trace refers to and that is older than a grace
period (a run in progress must not lose what it just read) is removed.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterable

from ...paths import read_json, write_json_atomic

CACHE_DIR = "page_cache"
GRACE_S = 6 * 3600


def cache_key(content_hash: str, reader: str, settings: str) -> str:
    """The key of one page read by *reader* with *settings*."""
    return hashlib.sha256(f"{content_hash}\x00{reader}\x00{settings}".encode()).hexdigest()[:40]


class PageCache:
    def __init__(self, workspace: Path) -> None:
        self.root = Path(workspace) / CACHE_DIR

    def _file(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        data = read_json(self._file(key))
        return data if isinstance(data, dict) and "md" in data else None

    def put(self, key: str, entry: dict[str, Any]) -> None:
        """Store *entry* (needs ``md``).  Never raises: a cache that cannot be written only means the
        page is read again next time."""
        try:
            write_json_atomic(self._file(key), {**entry, "created": round(time.time())}, indent=None)
        except (OSError, ValueError, TypeError):
            pass

    def keys(self) -> Iterable[str]:
        if self.root.is_dir():
            for sub in self.root.iterdir():
                if sub.is_dir():
                    for f in sub.glob("*.json"):
                        yield f.stem

    def stats(self) -> dict[str, int]:
        n = size = 0
        if self.root.is_dir():
            for sub in self.root.iterdir():
                if sub.is_dir():
                    for f in sub.glob("*.json"):
                        n += 1
                        with contextlib.suppress(OSError):
                            size += f.stat().st_size
        return {"entries": n, "bytes": size}

    def gc(self, keep: set[str], *, grace_s: float = GRACE_S, now: float | None = None) -> dict[str, int]:
        """Remove entries whose key is not in *keep* and that are older than *grace_s*."""
        now = now or time.time()
        removed = freed = 0
        if self.root.is_dir():
            for sub in list(self.root.iterdir()):
                if not sub.is_dir():
                    continue
                for f in sub.glob("*.json"):
                    if f.stem in keep:
                        continue
                    try:
                        st = f.stat()
                        if now - st.st_mtime < grace_s:
                            continue
                        f.unlink()
                        removed += 1
                        freed += st.st_size
                    except OSError:
                        continue
                with contextlib.suppress(OSError):
                    sub.rmdir()                           # only succeeds when empty
        return {"removed": removed, "freed_bytes": freed}

    def clear(self) -> int:
        n = 0
        for k in list(self.keys()):
            with contextlib.suppress(OSError):
                self._file(k).unlink()
                n += 1
        return n


def referenced_keys(markup_root: Path) -> set[str]:
    """Cache keys that the stored traces still refer to (``page.key``)."""
    keys: set[str] = set()
    if not markup_root.is_dir():
        return keys
    for tf in markup_root.rglob("*.trace.json"):
        try:
            data = json.loads(tf.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for p in data.get("pages", []) if isinstance(data, dict) else []:
            if isinstance(p, dict) and p.get("key"):
                keys.add(str(p["key"]))
    return keys
