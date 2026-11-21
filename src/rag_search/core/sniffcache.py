"""The text check of 3.0 Sniff, remembered by the file's checksum.

Telling a photograph from a photographed page takes a quick OCR (Apple Vision, about a second a picture).  What it sees is a
fact of the picture's bytes, so it is kept in ``<workspace>/sniff.sqlite`` under the checksum: a later run (a restart, the
next update, a photograph that was converted but not yet embedded) reads it instead of looking again.  Which side of the
line the text falls on is decided when it is read, so changing that line needs no new cache.  The sniff processes write
to it side by side (SQLite, one short connection each); a cache that cannot be opened or written only means the check runs.
"""

from __future__ import annotations

import contextlib
import sqlite3
from pathlib import Path

FILE = "sniff.sqlite"


def _open(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(path, timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("CREATE TABLE IF NOT EXISTS seen (sha TEXT PRIMARY KEY, text TEXT NOT NULL)")
    return con


def get(path: Path, sha: str) -> str | None:
    """The text seen in the picture with this checksum, or None when it was not looked at (or the cache is unreadable)."""
    try:
        with contextlib.closing(_open(path)) as con:
            row = con.execute("SELECT text FROM seen WHERE sha = ?", (sha,)).fetchone()
        return None if row is None else str(row[0])
    except (sqlite3.Error, OSError):
        return None


def put(path: Path, sha: str, text: str) -> None:
    with contextlib.suppress(sqlite3.Error, OSError), contextlib.closing(_open(path)) as con, con:
        con.execute("INSERT OR REPLACE INTO seen (sha, text) VALUES (?, ?)", (sha, text))
