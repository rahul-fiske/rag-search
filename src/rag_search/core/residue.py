"""Residue: what rag-search keeps about a photograph after the photograph has left every source folder.

Indexing removes the Markdown, the index and the pictures vector of a document whose file was deleted (``indexer.prune_orphans``).
Three more stores are keyed by a photograph's checksum and live outside the index, so they used to outlive it:

    people.sqlite          the faces found in it, their fingerprints, a small picture of each, and the names given to them
    describe_cache.sqlite  the description a vision model wrote (and a note that the model could not)
    sniff.sqlite           the text the quick check saw in the picture

``purge`` forgets what none of the photographs still present need.  A photograph is *present* when its checksum is in an index
folder (``index.meta.json``) or beside a Markdown (``.md.sha256``) anywhere in the workspace -- that is also true of a collection
whose folder cannot be read right now (it is left as it is) and of a document in the middle of being made again.  A face of the
People store goes only when its photograph's file is gone from where it was seen *and* the photograph is not present; the names
given to such faces (and the "not this person" answers about them) go with them.  A photograph that is present but whose
remembered path is gone (its folder was renamed) has the path corrected.  Nothing is removed when no photograph is present at all (the workspace is unreadable, not
empty), and a purge that would remove more than a fifth of what a store holds stops and says so (``force`` goes on).
"""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from ..paths import META_FILE, Paths
from . import describe, faces, sniffcache

MAX_SHARE = 0.20            # a purge removing more than this share of a store (and more than MIN_ROWS rows) needs ``force``
MIN_ROWS = 100


def present(paths: Paths) -> dict[str, str]:
    """Every document still in the workspace, by checksum: the path of its source file where the index folder records it ("" when
    only a Markdown record knows it)."""
    out: dict[str, str] = {}
    for root, name in ((paths.index, META_FILE), (paths.markup, None)):
        if not root.is_dir():
            continue
        for dp, _dn, fn in os.walk(root):
            if name and name in fn:
                with contextlib.suppress(OSError, ValueError):
                    meta = json.loads((Path(dp) / name).read_text(encoding="utf-8"))
                    if meta.get("src_sha256"):
                        out[str(meta["src_sha256"])] = str(meta.get("src_path") or "")
            if not name:
                for f in fn:
                    if f.endswith(".md.sha256"):
                        with contextlib.suppress(OSError):
                            first = (Path(dp) / f).read_text(encoding="utf-8").split("\n", 1)[0].strip()
                            if first:
                                out.setdefault(first, "")
    return out


def _limited(n: int, of: int, force: bool) -> bool:
    return not force and n > MIN_ROWS and n > MAX_SHARE * of


def purge(paths: Paths, *, force: bool = False, dry_run: bool = False) -> dict[str, Any]:
    """Forget what is left of photographs that are gone.  -> ``{"photographs", "faces", "labelled", "descriptions", "text_checks",
    "repaired", "skipped"}``: what was (or with *dry_run* would be) removed, how many photographs' remembered paths were corrected
    (their folder was renamed or moved), and ``skipped`` naming a store left alone and why."""
    res: dict[str, Any] = {"photographs": 0, "faces": 0, "labelled": 0, "descriptions": 0, "text_checks": 0, "repaired": 0,
                           "skipped": {}}
    alive = present(paths)
    if not alive:
        res["skipped"]["all"] = "no photograph is present in the workspace (it cannot be read, or is empty): nothing is forgotten"
        return res
    store = faces.store_file(paths)
    if store.exists():
        con = sqlite3.connect(store, timeout=30)
        try:
            rows = con.execute("SELECT sha, src FROM photos").fetchall()
            for sha, src in rows:                      # the file is where the index says now (a folder was renamed or moved)
                now = alive.get(sha, "")
                if now and now != src and not Path(src).exists() and Path(now).exists():
                    res["repaired"] += 1
                    if not dry_run:
                        with con:
                            con.execute("UPDATE photos SET src = ? WHERE sha = ?", (now, sha))
            gone = [sha for sha, src in rows if sha not in alive and not Path(src).exists()]
            if _limited(len(gone), len(rows), force):
                res["skipped"]["people"] = f"{len(gone)} of {len(rows)} photographs would be forgotten: more than a fifth, so none is"
            elif gone:
                for i in range(0, len(gone), 500):
                    part = gone[i:i + 500]
                    q = ",".join("?" * len(part))
                    ids = f"SELECT id FROM faces WHERE sha IN ({q})"
                    res["faces"] += con.execute(f"SELECT COUNT(*) FROM faces WHERE sha IN ({q})", part).fetchone()[0]
                    res["labelled"] += con.execute(f"SELECT COUNT(DISTINCT face) FROM labels WHERE face IN ({ids})", part).fetchone()[0]
                    if not dry_run:
                        with con:
                            con.execute(f"DELETE FROM labels WHERE face IN ({ids})", part)
                            con.execute(f"DELETE FROM vectors WHERE face IN ({ids})", part)
                            con.execute(f"DELETE FROM faces WHERE sha IN ({q})", part)
                            con.execute(f"DELETE FROM photos WHERE sha IN ({q})", part)
                res["photographs"] = len(gone)
        finally:
            con.close()
    cache = describe.cache_path(paths)
    if cache.exists():
        con = sqlite3.connect(cache, timeout=30)
        try:
            shas = {r[0] for r in con.execute("SELECT DISTINCT sha FROM d")} | {r[0] for r in con.execute("SELECT DISTINCT sha FROM failed")}
            gone = [s for s in shas if s not in alive]
            if _limited(len(gone), len(shas), force):
                res["skipped"]["descriptions"] = f"{len(gone)} of {len(shas)} descriptions would be forgotten: more than a fifth, so none is"
            else:
                res["descriptions"] = len(gone)
                if gone and not dry_run:
                    with con:
                        for i in range(0, len(gone), 500):
                            part = gone[i:i + 500]
                            q = ",".join("?" * len(part))
                            con.execute(f"DELETE FROM d WHERE sha IN ({q})", part)
                            con.execute(f"DELETE FROM failed WHERE sha IN ({q})", part)
        finally:
            con.close()
    sniff = paths.workspace / sniffcache.FILE
    if sniff.exists():
        con = sqlite3.connect(sniff, timeout=30)
        try:
            shas = [r[0] for r in con.execute("SELECT sha FROM seen")]
            gone = [s for s in shas if s not in alive]
            if _limited(len(gone), len(shas), force):
                res["skipped"]["text_checks"] = f"{len(gone)} of {len(shas)} text checks would be forgotten: more than a fifth, so none is"
            else:
                res["text_checks"] = len(gone)
                if gone and not dry_run:
                    with con:
                        for i in range(0, len(gone), 500):
                            part = gone[i:i + 500]
                            con.execute(f"DELETE FROM seen WHERE sha IN ({','.join('?' * len(part))})", part)
        finally:
            con.close()
    return res
