"""Live view of a run: what each phase has done, what each process is doing, and the one thing the run is
working on now (stdlib only).

Everything comes from the job's event log, read incrementally (the file offset is remembered per job, so a
dashboard tick costs only the new lines), plus the totals in the job record (``progress.conversion``, frozen
into ``summary.conversion``).  The log carries four kinds of line, written by whichever process does the work:

* ``phase`` -- a phase of the run (``convert``, ``embed``, ``merge``) starts or ends, with its totals;
* ``work``  -- a process starts or ends one unit of work (a document in a phase, a collection being merged),
  with its outcome: this is what makes a *process* generic, the same worker can be converting now and
  merging later;
* ``stage`` -- a numbered pipeline stage (2 Fingerprint, 3.1 Profile, ..., 6 Write) reached inside that work;
* ``page``  -- one converted page.

A *lane* is a process: ``worker N`` for the conversion pool, ``main process`` for the one that embeds and
merges.  The view is one snapshot (``now``, ``phases``, ``lanes``, ``live``), so every card of the dashboard
shows the same file, spelt the same way (its path inside the collection), at the same instant.
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
    return {"offset": 0, "lanes": {}, "order": 0, "procs": {}, "phases": {}, "main_pid": 0,
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


def _proc(st: dict[str, Any], pid: int, ts: float) -> dict[str, Any]:
    p = st["procs"].get(pid)
    if p is None:
        p = st["procs"][pid] = {"pid": pid, "n": len(st["procs"]) + 1, "first": ts, "last": ts, "open": None,
                                "busy_s": 0.0, "docs": {}, "phase": ""}
    return p


def _phase(st: dict[str, Any], name: str) -> dict[str, Any]:
    return st["phases"].setdefault(name, {"phase": name, "status": "", "total": None, "done": 0, "started": None,
                                          "finished": None, "outcomes": {}})


def _feed_phase(st: dict[str, Any], ev: dict[str, Any]) -> None:
    ts = float(ev.get("ts") or 0)
    ph = _phase(st, str(ev.get("phase") or ""))
    if ev.get("status") == "start":
        ph.update(status="running", started=ts, finished=None, done=0, outcomes={})
        st["main_pid"] = int(ev.get("pid") or 0) or st["main_pid"]
    else:
        ph.update(status="done", finished=ts)
    for k, v in ev.items():
        if k not in ("ts", "event", "phase", "status", "pid"):
            ph[k] = v


def _feed_work(st: dict[str, Any], ev: dict[str, Any]) -> None:
    ts, pid, phase = float(ev.get("ts") or 0), int(ev.get("pid") or 0), str(ev.get("phase") or "")
    proc, ph = _proc(st, pid, ts), _phase(st, phase)
    proc["last"] = ts
    if ev.get("status") == "start":
        proc["open"] = {"phase": phase, "file": str(ev.get("file", "")), "since": ts, "stage": "", "stage_id": "",
                        "progress": None, "chunks": ev.get("chunks")}
        proc["phase"] = phase
        if ph["status"] == "":
            ph.update(status="running", started=ts)
        return
    if proc["open"]:
        proc["busy_s"] += max(0.0, ts - proc["open"]["since"])
        if phase == "embed" and ev.get("outcome") == "indexed":
            ph["chunks_done"] = ph.get("chunks_done", 0) + int(proc["open"].get("chunks") or 0)
        proc["open"] = None
    proc["docs"][phase] = proc["docs"].get(phase, 0) + 1
    ph["done"] += 1
    out = str(ev.get("outcome") or "done")
    ph["outcomes"][out] = ph["outcomes"].get(out, 0) + 1
    if st["phases"].get(phase, {}).get("status") == "" and phase not in ("convert", "embed"):
        ph["status"] = "running"


def _feed_stage_of_proc(st: dict[str, Any], ev: dict[str, Any]) -> None:
    proc = st["procs"].get(int(ev.get("pid") or 0))
    if proc and proc["open"] and proc["open"]["file"] == str(ev.get("file", "")):
        proc["open"]["stage"] = str(ev.get("stage") or "")
        proc["open"]["stage_id"] = str(ev.get("id") or "")
        proc["open"]["stage_done"] = ev.get("status") == "done"


def _feed(st: dict[str, Any], line: str) -> None:
    try:
        ev = json.loads(line)
    except ValueError:
        return
    kind = ev.get("event")
    if kind == "phase":
        _feed_phase(st, ev)
        return
    if kind == "work":
        _feed_work(st, ev)
        return
    if kind == "page" and "pid" in ev:
        proc = st["procs"].get(int(ev["pid"]))
        if proc and proc["open"] and proc["open"]["file"] == str(ev.get("file", "")):
            prog = proc["open"]["progress"] or {"done": 0, "of": 0}
            proc["open"]["progress"] = {"done": prog["done"] + 1, "of": int(ev.get("of") or prog["of"])}
    if kind == "stage" and "pid" in ev:
        _feed_stage_of_proc(st, ev)
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
                    if '"stage"' in line or '"event": "page"' in line or '"event": "work"' in line \
                            or '"event": "phase"' in line:
                        _feed(st, line)
        return st


def lanes_for(paths: Paths, job_id: str, running: bool, now: float | None = None,
              state: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """The workers of a run: ``[{"name", "kind", "state", "file", "since", "busy_pct", "docs"}]``.
    ``kind`` is ``cpu`` for docling workers and ``gpu`` for embedding (the embedding lane appears when
    the log holds ``embed`` stage events; today only the conversion processes write stage events)."""
    st = state or _read_new(events_file(paths, job_id))
    now = now or time.time()
    if st["procs"]:
        return _proc_lanes(st, running, now)
    out = []
    convert = sorted((x for x in st["lanes"].values() if x["role"] == "convert"), key=lambda x: x["n"])
    embed = sorted((x for x in st["lanes"].values() if x["role"] == "embed"), key=lambda x: x["n"])
    for i, lane in enumerate(convert, 1):
        out.append(_view(lane, f"docling worker {i}" if len(convert) > 1 else "docling", "cpu",
                         running, now))
    for lane in embed:
        out.append(_view(lane, "embedding", "gpu", running, now))
    return out


def _proc_lanes(st: dict[str, Any], running: bool, now: float) -> list[dict[str, Any]]:
    """One lane per process, whatever phase it is in: ``main process`` (the one that announced the phases)
    first, then the conversion pool's ``worker N`` in the order they appeared."""
    out = []
    main = st["main_pid"]
    pool = sorted((p for p in st["procs"].values() if p["pid"] != main), key=lambda p: p["n"])
    ordered = [p for p in st["procs"].values() if p["pid"] == main] + pool
    for p in ordered:
        name = "main process" if p["pid"] == main else f"worker {pool.index(p) + 1}"
        o = p["open"] if running else None
        busy = p["busy_s"] + (max(0.0, now - o["since"]) if o else 0.0)
        end = now if running else p["last"]
        span = max(0.001, end - p["first"])
        phase = (o or {}).get("phase") or p["phase"]
        out.append({
            "name": name,
            "kind": "gpu" if phase == "embed" else "cpu", "pid": p["pid"],
            "state": "working" if o else "idle", "phase": phase if o else "",
            "stage": (o or {}).get("stage_id", "") if o else "", "stage_name": (o or {}).get("stage", "") if o else "",
            "file": o["file"] if o else "", "since": o["since"] if o else None,
            "busy_pct": round(min(100.0, 100.0 * busy / span)), "busy_s": round(busy, 1),
            "docs": sum(p["docs"].values()), "docs_by_phase": dict(p["docs"]),
            "progress": o.get("progress") if o else None})
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


def phases_view(st: dict[str, Any], rec: dict[str, Any], running: bool) -> list[dict[str, Any]]:
    """Convert, Embed, Merge and Publish as equal citizens: status, counts, timing and what each reports
    about its own work.  A phase the log has not announced yet is ``pending``."""
    out = []
    prog = rec.get("progress") or {}
    now = time.time()
    for name in ("convert", "embed", "merge"):
        ph = st["phases"].get(name)
        if ph is None:
            out.append({"phase": name, "status": "pending"})
            continue
        total = ph.get("total")
        v = {k: x for k, x in ph.items() if k not in ("started", "finished")}
        v["started"], v["finished"] = ph.get("started"), ph.get("finished")
        v["elapsed_s"] = round(((ph.get("finished") or (now if running else ph.get("started") or 0)) -
                                (ph.get("started") or 0)), 1) if ph.get("started") else 0.0
        if total:
            v["pct"] = min(100, round(100 * ph["done"] / total))
        if name == "embed" and ph.get("chunks_done") and v["elapsed_s"] >= 3:
            v["chunks_per_s"] = round(ph["chunks_done"] / v["elapsed_s"], 1)
        out.append(v)
    publish = {"phase": "publish", "status": "pending"}
    if rec.get("publish") is not None or rec.get("publish_s") is not None:
        pub = rec.get("publish") or {}
        publish = {"phase": "publish", "status": "failed" if pub.get("error") else "done", "seconds": rec.get("publish_s"),
                   "generation": pub.get("generation"), "changed": pub.get("changed"), "error": pub.get("error", ""),
                   "documents": pub.get("documents"), "reload": rec.get("search_reload")}
    elif prog.get("phase") == "publish":
        publish = {"phase": "publish", "status": "running", "started": prog.get("since")}
    out.append(publish)
    return out


def now_view(st: dict[str, Any], lanes: list[dict[str, Any]], rec: dict[str, Any], running: bool) -> dict[str, Any] | None:
    """The one thing the run is working on now: the longest-running work of the job's current phase (any
    work when the job record has not caught up), spelt as the document's path inside its collection."""
    if not running:
        return None
    phase = (rec.get("progress") or {}).get("phase", "")
    busy = [x for x in lanes if x["state"] == "working"]
    pick = [x for x in busy if x["phase"] == phase] or busy
    if not pick:
        return {"phase": phase, "file": "", "since": None, "workers": 0} if phase else None
    first = min(pick, key=lambda x: x["since"] or 0)
    return {"phase": first["phase"], "file": first["file"], "since": first["since"], "stage": first["stage"],
            "stage_name": first["stage_name"], "progress": first["progress"], "worker": first["name"],
            "workers": len(pick)}


def run_view(paths: Paths, job_id: str = "") -> dict[str, Any]:
    """Totals and lanes of a run (the latest when *job_id* is empty)."""
    rec = read_record(paths, job_id) if job_id else next(iter(all_records(paths)), None)
    if not rec or not JOB_ID_RE.fullmatch(str(rec.get("id", ""))):
        return {"ok": True, "job": None, "totals": {}, "lanes": []}
    running = rec.get("status") in ("queued", "running")
    prog = rec.get("progress") or {}
    totals = (rec.get("summary") or {}).get("conversion") or prog.get("conversion") or {}
    st = _read_new(events_file(paths, rec["id"]))
    lanes = lanes_for(paths, rec["id"], running, state=st)
    return {"ok": True, "job": rec["id"], "status": rec.get("status"), "running": running,
            "now": now_view(st, lanes, rec, running), "phases": phases_view(st, rec, running),
            "phase": prog.get("phase", ""), "done": prog.get("done"), "total": prog.get("total"),
            "totals": totals, "lanes": lanes,
            "live": live_view(st, running),
            "started_at": rec.get("started_at"), "finished_at": rec.get("finished_at")}
