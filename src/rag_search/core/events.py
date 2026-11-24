"""The job's event log (``jobs/<id>.events.jsonl``): one JSON object per line, appended by several processes and threads.

Every writer goes through ``append``: the line is encoded first and written with one ``os.write`` on a descriptor opened with
``O_APPEND``, so lines from the run's main process, its threads and the conversion processes never interleave inside one
another (a buffered text file splits a long line into several writes).  Stdlib only.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any


def line(event: str, **fields: Any) -> bytes:
    """One encoded event line: ``ts`` and ``event`` first, then *fields* without the ones that are None."""
    rec = {"ts": round(time.time(), 3), "event": event, **{k: v for k, v in fields.items() if v is not None}}
    return (json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8")


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        n = os.write(fd, view)
        view = view[n:]


def append(path: str | os.PathLike | None, event: str, **fields: Any) -> None:
    """Append one event to the log at *path* (nothing when *path* is empty).  Never raises: an event log that cannot be
    written must not fail the document it describes."""
    if not path:
        return
    try:
        data = line(event, **fields)
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            _write_all(fd, data)
        finally:
            os.close(fd)
    except (OSError, ValueError, TypeError):
        pass


class Writer:
    """The run's own writer: one descriptor kept open, one lock, one ``os.write`` per line (threads of the run share it)."""

    def __init__(self, path: Any) -> None:
        self.path = path
        self._fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        self._lock = threading.Lock()

    def write(self, event: str, **fields: Any) -> None:
        data = line(event, **fields)
        with self._lock:
            _write_all(self._fd, data)

    def close(self) -> None:
        with self._lock:
            if self._fd >= 0:
                os.close(self._fd)
                self._fd = -1
