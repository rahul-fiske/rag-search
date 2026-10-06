"""The stall watch of an indexing run: a conversion process that has gone silent is found and stopped (stdlib only).

Every reader call in the pipeline has a time limit of its own -- docling's per-document timeout, the document reader's
page timeout, Tesseract's -- but a call into native code that never returns (seen: an Apple Vision text request, a PDF
render of a file a cloud-storage app never delivers) has none, and a run then waits for that document for ever: the
pool never gets its result, embedding never starts, and the dashboard shows a run that is "working".

The conversion processes already say what they do in the run's event log (``work`` start / done, ``stage``, ``step``,
``page``, each with the process id).  The watch reads that log: a process with a document open that has written
nothing for ``RAG_SEARCH_STALL_TIMEOUT`` seconds (default one hour; 0 switches the watch off) is reported.  ``run_index``
then stops that one process, records the document as failed with what it was doing, and carries on with the others.

Time is counted in observed poll intervals (each capped), not by the clock on the wall: a laptop that slept for the
night has not stalled.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

DEFAULT_STALL_S = 3600.0
DEFAULT_DOC_TIMEOUT = 2700.0         # docling_convert.DEFAULT_DOC_TIMEOUT (a test keeps the two equal)
RUN_FACTOR = 2.0                     # the daemon stops a whole run whose log is silent for this many limits
POLL_S = 20.0
MARGIN_S = 900.0                     # the limit is at least this much above the per-document timeout


def limit_s(env: Any = None) -> float:
    """Seconds of silence after which a conversion process counts as stalled; 0 = never.  Never below the
    per-document timeout plus a margin: a whole-document conversion is silent for as long as that allows."""
    e = os.environ if env is None else env
    try:
        v = float(e.get("RAG_SEARCH_STALL_TIMEOUT", "") or DEFAULT_STALL_S)
    except ValueError:
        v = DEFAULT_STALL_S
    if v <= 0:
        return 0.0
    try:
        doc = float(e.get("RAG_SEARCH_DOC_TIMEOUT", "") or 0)
    except ValueError:
        doc = 0.0
    if doc <= 0:                          # not set: docling_convert's default (not imported: the daemon stays light)
        doc = DEFAULT_DOC_TIMEOUT if not e.get("RAG_SEARCH_DOC_TIMEOUT") else 0.0
    return max(v, doc + MARGIN_S) if doc > 0 else v


class StallWatch:
    """Follows one event log.  ``poll()`` returns the processes that are stalled now:
    ``[{"pid", "file", "phase", "page", "what", "idle_s"}]``; each is reported once."""

    def __init__(self, path: Path | str, limit: float, *, poll_s: float = POLL_S, phases: tuple[str, ...] = ("convert",)) -> None:
        self.path, self.limit, self.poll_s, self.phases = Path(path), float(limit), float(poll_s), phases
        self.offset = 0
        self.open: dict[int, dict[str, Any]] = {}       # pid -> the work it has open
        self.idle: dict[int, float] = {}                # pid -> seconds observed without a word from it
        self._seen: set[int] = set()
        self._last = time.monotonic()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- the log
    def _read(self) -> None:
        try:
            with open(self.path, "rb") as fh:
                fh.seek(self.offset)
                data = fh.read()
        except OSError:
            return
        end = data.rfind(b"\n")
        if end < 0:
            return
        self.offset += end + 1
        for line in data[:end].split(b"\n"):
            if b'"pid"' not in line:
                continue
            try:
                ev = json.loads(line)
                pid = int(ev.get("pid") or 0)
            except (ValueError, TypeError):
                continue
            if not pid:
                continue
            self._seen.add(pid)
            kind = ev.get("event")
            if kind == "work":
                if ev.get("status") == "start" and str(ev.get("phase")) in self.phases:
                    self.open[pid] = {"file": str(ev.get("file", "")), "phase": str(ev.get("phase", "")), "page": None, "what": ""}
                    self.idle[pid] = 0.0
                elif ev.get("status") == "done":
                    self.open.pop(pid, None)
                    self.idle.pop(pid, None)
            elif kind == "step" and pid in self.open:
                self.open[pid].update(page=ev.get("page"), what=str(ev.get("what") or ""))
            elif kind == "stage" and pid in self.open and not self.open[pid]["what"]:
                self.open[pid]["what"] = str(ev.get("stage") or "")

    def poll(self, elapsed: float | None = None) -> list[dict[str, Any]]:
        """Read what is new and return the processes that have just crossed the limit.  *elapsed* (tests) replaces the
        time since the last poll."""
        with self._lock:
            now = time.monotonic()
            dt = min(now - self._last, 2 * self.poll_s) if elapsed is None else float(elapsed)
            self._last = now
            self._read()
            out = []
            for pid, work in list(self.open.items()):
                self.idle[pid] = 0.0 if pid in self._seen else self.idle.get(pid, 0.0) + dt
                if self.limit > 0 and self.idle[pid] >= self.limit:
                    out.append({"pid": pid, **work, "idle_s": round(self.idle[pid])})
                    self.open.pop(pid, None)
                    self.idle.pop(pid, None)
            self._seen.clear()
            return out

    def in_flight(self) -> dict[int, str]:
        """pid -> the document it has open (after reading what is new)."""
        with self._lock:
            self._read()
            return {pid: w["file"] for pid, w in self.open.items()}

    def forget_all(self) -> None:
        with self._lock:
            self.open.clear()
            self.idle.clear()

    # -- a thread that polls
    def start(self, on_stall: Callable[[dict[str, Any]], None]) -> "StallWatch":
        def loop() -> None:
            while not self._stop.wait(self.poll_s):
                try:
                    for st in self.poll():
                        on_stall(st)
                except Exception:  # noqa: BLE001 - the watch never breaks a run
                    pass

        self._thread = threading.Thread(target=loop, name="stall-watch", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()


def describe(st: dict[str, Any]) -> str:
    """What a stalled process was doing, in words for the run's error list."""
    mins = max(1, round(float(st.get("idle_s") or 0) / 60))
    where = ""
    if st.get("page"):
        where = f" on page {st['page']}"
    if st.get("what"):
        where += f" ({st['what']})"
    return (f"stalled: no progress for {mins} min{where}; the conversion process was stopped and the run went on. "
            "The pages already read are kept; the document is tried again on the next run "
            "(RAG_SEARCH_STALL_TIMEOUT sets the limit, 0 switches the watch off)")
