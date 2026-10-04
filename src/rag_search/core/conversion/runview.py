"""Live view of a run's conversion work: totals and what each worker is doing (stdlib only).

The totals come from the job record (``progress.conversion``, frozen into ``summary.conversion``).
The *lanes* -- one per docling worker process, one for embedding -- are derived from the ``stage``
events the pool processes append to the job's event log, read incrementally (the file offset is
remembered per job, so a dashboard tick costs only the new lines).
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any

from ...jobs import JOB_ID_RE, all_records, events_file, read_record
from ...paths import Paths

_LOCK = threading.Lock()
_STATE: dict[str, dict[str, Any]] = {}        # events file -> parsed state
_MAX_STATES = 8


def _fresh() -> dict[str, Any]:
    return {"offset": 0, "lanes": {}, "order": 0,
            "live": {"pages": 0, "cached": 0, "branches": {}, "outcomes": {}, "open": {},
                     "first": 0.0, "last": 0.0, "tokens": 0, "gpu_s": 0.0}}


def _lane(st: dict[str, Any], pid: int, role: str, ts: float) -> dict[str, Any]:
    key = f"{role}:{pid}"
    lane = st["lanes"].get(key)
    if lane is None:
        st["order"] += 1
        lane = st["lanes"][key] = {"role": role, "pid": pid, "n": st["order"], "first": ts,
                                   "open": None, "busy_s": 0.0, "docs": 0, "last": ts}
    return lane


def _feed(st: dict[str, Any], line: str) -> None:
    try:
        ev = json.loads(line)
    except ValueError:
        return
    if ev.get("event") == "page" and "pid" in ev:
        _feed_page(st, ev)
        return
    if ev.get("event") != "stage" or "pid" not in ev:
        return
    stage, status, ts = ev.get("stage"), ev.get("status"), float(ev.get("ts") or 0)
    role = {"convert": "convert", "embed": "embed"}.get(str(stage))
    if role is None:
        return
    lane = _lane(st, int(ev["pid"]), role, ts)
    lane["last"] = ts
    if status == "start":
        lane["open"] = {"file": ev.get("file", ""), "since": ts}
        lane["progress"] = None
        if role == "convert":
            st["live"]["open"].pop(str(ev.get("file", "")), None)    # a fresh attempt at this file
    elif status == "done" and lane["open"]:
        lane["busy_s"] += max(0.0, ts - lane["open"]["since"])
        lane["docs"] += 1
        lane["open"] = None
        lane["progress"] = None
        if role == "convert":
            st["live"]["open"].pop(str(ev.get("file", "")), None)


def _feed_page(st: dict[str, Any], ev: dict[str, Any]) -> None:
    """One finished page: counts per branch and outcome, progress of the document it belongs to."""
    live, ts = st["live"], float(ev.get("ts") or 0)
    live["pages"] += 1
    live["first"] = live["first"] or ts
    live["last"] = max(live["last"], ts)
    b, o = str(ev.get("branch") or "unknown"), str(ev.get("outcome") or "pass")
    live["branches"][b] = live["branches"].get(b, 0) + 1
    live["outcomes"][o] = live["outcomes"].get(o, 0) + 1
    if ev.get("cache") == "hit":
        live["cached"] += 1
    live["tokens"] += int(ev.get("tokens") or 0)
    live["gpu_s"] += float(ev.get("gpu_s") or 0.0)
    f = str(ev.get("file", ""))
    d = live["open"].setdefault(f, {"done": 0, "of": int(ev.get("of") or 0)})
    d["done"] += 1
    d["of"] = int(ev.get("of") or d["of"])
    lane = _lane(st, int(ev["pid"]), "convert", ts)
    lane["last"] = ts
    lane["progress"] = {"done": d["done"], "of": d["of"]}


def live_view(st: dict[str, Any], running: bool) -> dict[str, Any]:
    """Pages read so far in the run, per branch / outcome, and the documents being read now."""
    live = st["live"]
    span = live["last"] - live["first"]
    return {"pages": live["pages"], "cached": live["cached"], "branches": dict(live["branches"]),
            "outcomes": dict(live["outcomes"]),
            "open": {k: dict(v) for k, v in list(live["open"].items())[-8:]} if running else {},
            "pages_per_min": round(60.0 * live["pages"] / span, 1) if span >= 5 and live["pages"] > 1 else 0.0,
            "tokens": live["tokens"], "gpu_s": round(live["gpu_s"], 1),
            "tokens_per_s": round(live["tokens"] / live["gpu_s"], 1) if live["gpu_s"] >= 1 else 0.0}


def _read_new(path: Any) -> dict[str, Any]:
    key = str(path)
    with _LOCK:
        st = _STATE.get(key)
        try:
            size = path.stat().st_size
        except OSError:
            _STATE.pop(key, None)
            return _fresh()
        if st is None or size < st["offset"]:                  # new job, or the log was rewritten
            st = _STATE[key] = _fresh()
            while len(_STATE) > _MAX_STATES:
                _STATE.pop(next(iter(_STATE)))
        if size > st["offset"]:
            try:
                with open(path, "rb") as fh:
                    fh.seek(st["offset"])
                    data = fh.read()
            except OSError:
                return st
            end = data.rfind(b"\n")
            if end >= 0:
                st["offset"] += end + 1
                for line in data[:end].decode("utf-8", "replace").split("\n"):
                    if '"stage"' in line or '"event": "page"' in line:
                        _feed(st, line)
        return st


def lanes_for(paths: Paths, job_id: str, running: bool, now: float | None = None,
              state: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """The workers of a run: ``[{"name", "kind", "state", "file", "since", "busy_pct", "docs"}]``.
    ``kind`` is ``cpu`` for docling workers and ``gpu`` for embedding (the embedding lane appears when
    the log holds ``embed`` stage events; today only the conversion processes write stage events)."""
    st = state or _read_new(events_file(paths, job_id))
    now = now or time.time()
    out = []
    convert = sorted((x for x in st["lanes"].values() if x["role"] == "convert"), key=lambda x: x["n"])
    embed = sorted((x for x in st["lanes"].values() if x["role"] == "embed"), key=lambda x: x["n"])
    for i, lane in enumerate(convert, 1):
        out.append(_view(lane, f"docling worker {i}" if len(convert) > 1 else "docling", "cpu",
                         running, now))
    for lane in embed:
        out.append(_view(lane, "embedding", "gpu", running, now))
    return out


def _view(lane: dict[str, Any], name: str, kind: str, running: bool, now: float) -> dict[str, Any]:
    busy = lane["busy_s"]
    open_ = lane["open"] if running else None
    end = now if running else lane["last"]
    if open_:
        busy += max(0.0, now - open_["since"])
    span = max(0.001, end - lane["first"])
    return {"name": name, "kind": kind, "pid": lane["pid"],
            "state": "working" if open_ else "idle",
            "file": open_["file"] if open_ else "", "since": open_["since"] if open_ else None,
            "busy_pct": round(min(100.0, 100.0 * busy / span)), "busy_s": round(busy, 1),
            "docs": lane["docs"], "progress": lane.get("progress") if open_ else None}


def run_view(paths: Paths, job_id: str = "") -> dict[str, Any]:
    """Totals and lanes of a run (the latest when *job_id* is empty)."""
    rec = read_record(paths, job_id) if job_id else next(iter(all_records(paths)), None)
    if not rec or not JOB_ID_RE.fullmatch(str(rec.get("id", ""))):
        return {"ok": True, "job": None, "totals": {}, "lanes": []}
    running = rec.get("status") in ("queued", "running")
    prog = rec.get("progress") or {}
    totals = (rec.get("summary") or {}).get("conversion") or prog.get("conversion") or {}
    st = _read_new(events_file(paths, rec["id"]))
    return {"ok": True, "job": rec["id"], "status": rec.get("status"), "running": running,
            "phase": prog.get("phase", ""), "done": prog.get("done"), "total": prog.get("total"),
            "totals": totals, "lanes": lanes_for(paths, rec["id"], running, state=st),
            "live": live_view(st, running),
            "started_at": rec.get("started_at"), "finished_at": rec.get("finished_at")}
