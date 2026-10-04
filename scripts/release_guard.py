#!/usr/bin/env python3
"""Fail if a denied word appears anywhere in a release.

    release_guard.py PATH [PATH ...]

PATH is a folder, a file, or an archive (.zip, .whl, .tar.gz, .tgz; archives inside archives are
opened too).  The words come from the environment variable RAG_SEARCH_RELEASE_DENY (comma
separated) or, when that is empty, from the file ``.release-deny`` in the repository root (one
word per line; that file is never shipped).  File names, file contents and archive member names
are all checked, case-insensitively.  Exit status 1 lists every hit; 2 means no words were given.
"""

from __future__ import annotations

import io
import os
import re
import sys
import tarfile
import zipfile
from pathlib import Path

MAX_DEPTH = 4
_RECORD_HASH = re.compile(rb"sha256=[A-Za-z0-9_=-]+")   # base64 digests in a wheel's RECORD can spell any word
ARCHIVE_SUFFIXES = (".zip", ".whl", ".tar.gz", ".tgz", ".tar")


def load_words() -> list[str]:
    words = [w.strip() for w in os.environ.get("RAG_SEARCH_RELEASE_DENY", "").split(",")]
    if not any(words):
        root = Path(__file__).resolve().parent.parent
        try:
            words = (root / ".release-deny").read_text(encoding="utf-8").splitlines()
        except OSError:
            words = []
    return sorted({w.strip().lower() for w in words if w.strip() and not w.strip().startswith("#")})


def _scan_bytes(label: str, data: bytes, words: list[str], hits: list[str], depth: int) -> None:
    low = label.lower()
    for w in words:
        if w in low:
            hits.append(f"{label}: file name contains {w!r}")
    if depth < MAX_DEPTH and low.endswith(ARCHIVE_SUFFIXES):
        try:
            if low.endswith((".zip", ".whl")):
                with zipfile.ZipFile(io.BytesIO(data)) as z:
                    for info in z.infolist():
                        if not info.is_dir():
                            _scan_bytes(f"{label}!{info.filename}", z.read(info), words, hits, depth + 1)
                return
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as t:
                for m in t.getmembers():
                    if m.isfile():
                        f = t.extractfile(m)
                        if f is not None:
                            _scan_bytes(f"{label}!{m.name}", f.read(), words, hits, depth + 1)
            return
        except (zipfile.BadZipFile, tarfile.TarError, OSError):
            pass                                 # not really an archive: scan it as bytes
    if low.endswith(".dist-info/record"):
        data = _RECORD_HASH.sub(b"sha256=", data)
    lowdata = data.lower()
    for w in words:
        n = lowdata.count(w.encode())
        if n:
            hits.append(f"{label}: contains {w!r} {n}x")


def scan(paths: list[Path], words: list[str]) -> list[str]:
    hits: list[str] = []
    for p in paths:
        files = [p] if p.is_file() else sorted(q for q in p.rglob("*") if q.is_file())
        for f in files:
            if "__pycache__" in f.parts:
                continue
            _scan_bytes(str(f), f.read_bytes(), words, hits, 0)
    return hits


def main(argv: list[str]) -> int:
    words = load_words()
    if not words:
        print("release_guard: no words to look for (set RAG_SEARCH_RELEASE_DENY or create .release-deny)",
              file=sys.stderr)
        return 2
    paths = [Path(a) for a in argv]
    missing = [str(p) for p in paths if not p.exists()]
    if missing or not paths:
        print(f"release_guard: nothing to scan: {missing or 'no path given'}", file=sys.stderr)
        return 2
    hits = scan(paths, words)
    if hits:
        print(f"release_guard: FOUND {len(hits)} hit(s) for {len(words)} denied word(s):", file=sys.stderr)
        for h in hits[:60]:
            print("  " + h, file=sys.stderr)
        if len(hits) > 60:
            print(f"  ... and {len(hits) - 60} more", file=sys.stderr)
        return 1
    print(f"release_guard: clean ({len(words)} denied word(s), {len(paths)} path(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
