"""Playground runs as jobs (stdlib only): start one in the background, watch it, cancel it.

A playground *index* or *bench* run takes minutes, so the dashboard starts it detached and polls it, the
same way production watches an indexing run -- and with the same machinery: the run is a job record
``<experiment>/jobs/<id>.json`` plus an event log ``<id>.events.jsonl`` that the pipeline writes while it
works (numbered ``stage`` events, one ``page`` event per converted page, ``progress`` and ``doc`` events,
for a bench one ``query`` event per query).  ``jobs.documents`` and ``conversion.runview`` read them
exactly as they do for production, so the Playground shows the same stages 1-8 and the same page
branches/outcomes as the Indexing tab.

Parent side (the dashboard / CLI): :func:`start`, :func:`view`, :func:`cancel`.
Child side (the ``rag-search playground index|bench --job ID`` process): :class:`Recorder`.
Nothing here loads a model; nothing here ever touches production's jobs or serving state.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from typing import Any

from . import stages
from .jobs import ACTIVE, JOB_ID_RE, documents, events_file, job_file, now, read_record, view as job_view
from .paths import Paths, detached_start, get_playground_paths, read_json, write_json_atomic

KINDS = ("index", "bench")
MAX_PAGES_PER_DOC = 400          # page lines kept per document in the view (a sandbox: documents are small)
MAX_QUERIES = 500


class RunError(ValueError):
    """A user-facing problem starting or reading a playground run."""


def _alive(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
    except (OSError, TypeError, ValueError):
        return False
    return True


def _records(exp: Paths) -> list[dict[str, Any]]:
    out = []
    if exp.jobs.is_dir():
        for f in exp.jobs.glob("*.json"):
            rec = read_json(f)
            if isinstance(rec, dict) and rec.get("id"):
                out.append(rec)
    out.sort(key=lambda r: r.get("created_at", 0), reverse=True)
    return out


def _pid(exp: Paths, rec: dict[str, Any]) -> int:
    """The run's process: the child's own record says so once it has begun; before that, the parent's pid file."""
    if rec.get("pid"):
        return int(rec["pid"])
    try:
        return int((exp.jobs / f"{rec['id']}.pid").read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return 0


def _settle(exp: Paths, rec: dict[str, Any]) -> dict[str, Any]:
    """A record that says "running" whose process is gone did not finish: mark it failed, with the log's tail."""
    if rec.get("status") in ACTIVE:
        pid = _pid(exp, rec)
        young = now() - float(rec.get("created_at") or 0) < 30      # the pid file is written right after the start
        if (pid and not _alive(pid)) or (not pid and not young):
            rec = {**rec, "status": "failed", "finished_at": now(),
                   "error": rec.get("error") or ("the process ended without finishing"
                                                 + _log_tail(exp, rec["id"]))}
            write_json_atomic(job_file(exp, rec["id"]), rec)
    return rec


def _log_tail(exp: Paths, jid: str, n: int = 600) -> str:
    try:
        text = (exp.jobs / f"{jid}.log").read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""
    return (": " + text[-n:]) if text else ""


def active(base: Paths, name: str) -> dict[str, Any] | None:
    """The experiment's run that is still going, if any."""
    exp = get_playground_paths(base, name)
    for rec in _records(exp):
        rec = _settle(exp, rec)
        if rec.get("status") in ACTIVE:
            return rec
    return None


def start(base: Paths, name: str, kind: str, args: list[str] | None = None, opts: dict[str, Any] | None = None) -> dict[str, Any]:
    """Start ``rag-search playground KIND NAME --job ID --json ARGS`` detached and return at once with the job id."""
    if kind not in KINDS:
        raise RunError(f"unknown run kind {kind!r}")
    exp = get_playground_paths(base, name)
    if not exp.home.is_dir():
        raise RunError(f"no such experiment: {name}")
    busy = active(base, name)
    if busy:
        raise RunError(f"experiment {name!r} is already running a {busy.get('kind', 'job')} ({busy['id']}); "
                       "wait for it or cancel it")
    exp.jobs.mkdir(parents=True, exist_ok=True)
    jid = time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + uuid.uuid4().hex[:4]
    write_json_atomic(job_file(exp, jid), {"id": jid, "kind": kind, "experiment": name, "status": "queued",
                                           "created_at": now(), "opts": opts or {}})
    cmd = [sys.executable, "-m", "rag_search.cli", "playground", kind, name, "--job", jid, "--json", *(args or [])]
    log = open(exp.jobs / f"{jid}.log", "ab")
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True,
                                close_fds=True, **detached_start(base.home, dict(os.environ, RAG_SEARCH_HOME=str(base.home))))
    finally:
        log.close()
    threading.Thread(target=proc.wait, daemon=True, name=f"playground-{jid}").start()   # reap it: no zombies
    (exp.jobs / f"{jid}.pid").write_text(str(proc.pid), encoding="utf-8")   # the record is the child's to write
    return {"job": jid, "kind": kind, "pid": proc.pid}


def cancel(base: Paths, name: str) -> dict[str, Any]:
    rec = active(base, name)
    if not rec:
        return {"cancelled": False}
    exp = get_playground_paths(base, name)
    pid = _pid(exp, rec)
    if pid:
        try:
            os.killpg(pid, signal.SIGTERM)             # the child leads its own process group
        except OSError:
            pass
    write_json_atomic(job_file(exp, rec["id"]), {**rec, "status": "cancelled", "finished_at": now()})
    return {"cancelled": True, "job": rec["id"]}


# ── reading a run ──────────────────────────────────────────────────────────────

def timeline(exp: Paths, jid: str) -> dict[str, Any]:
    """Per document: its stage events (numbers from stages.py) and its pages; per run: the queries of a bench.

    ``docs[file] = {"stages": {key: {id, status, seconds, ...}}, "pages": [{page, of, branch, outcome, ...}]}``"""
    docs: dict[str, dict[str, Any]] = {}
    queries: list[dict[str, Any]] = []
    try:
        raw = events_file(exp, jid).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {"docs": [], "queries": queries}
    for line in raw.splitlines():
        if '"stage"' not in line and '"page"' not in line and '"query"' not in line:
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        kind = ev.get("event")
        if kind == "query":
            if len(queries) < MAX_QUERIES:
                queries.append({k: v for k, v in ev.items() if k not in ("event", "pid")})
            continue
        f = str(ev.get("file", ""))
        if not f or kind not in ("stage", "page"):
            continue
        d = docs.setdefault(f, {"file": f, "stages": {}, "pages": []})
        if kind == "stage":
            key = str(ev.get("stage"))
            slot = d["stages"].setdefault(key, {"id": stages.id_of(key), "key": key})
            slot.update({k: v for k, v in ev.items() if k not in ("event", "stage", "file", "pid")})
            slot["status"] = ev.get("status")
        else:
            pages = d["pages"]
            if len(pages) < MAX_PAGES_PER_DOC:
                pages.append({k: ev.get(k) for k in ("page", "of", "branch", "outcome", "cache", "read_s", "chars",
                                                     "tokens", "gpu_s", "model", "gate", "ts")})
    return {"docs": list(docs.values()), "queries": queries}


def view(base: Paths, name: str, jid: str = "", *, docs: int = 100) -> dict[str, Any]:
    """The run *jid* (the latest when empty) with everything the Playground shows: the job, the live conversion view
    (pages per branch/outcome, workers), the documents with their stage timeline and pages, a bench's queries."""
    from .core.conversion import runview

    exp = get_playground_paths(base, name)
    if not exp.home.is_dir():
        raise RunError(f"no such experiment: {name}")
    if jid and not JOB_ID_RE.fullmatch(jid):
        raise RunError("bad job id")
    rec = read_record(exp, jid) if jid else next(iter(_records(exp)), None)
    if not rec:
        return {"ok": True, "job": None}
    rec = _settle(exp, rec)
    out = job_view(rec)
    out["kind"] = rec.get("kind", "index")
    out["opts"] = rec.get("opts") or {}
    out["files"] = rec.get("files")
    if rec.get("status") == "failed" and not out.get("error"):
        out["error"] = rec.get("error")
    run = runview.run_view(exp, rec["id"]) if rec.get("kind", "index") == "index" else {}
    tl = timeline(exp, rec["id"])
    by_file = {d["file"]: d for d in tl["docs"]}
    doc_list = documents(exp, rec["id"], docs)
    for item in doc_list["items"]:
        rel = f"{item.get('collection', '')}/{item.get('path') or item.get('source', '')}".strip("/")
        item["timeline"] = by_file.pop(rel, None) or by_file.pop(str(item.get("source", "")), None)
    # documents that have stage events but no finished `doc` event yet: still being worked on
    in_flight = list(by_file.values())
    return {"ok": True, "job": out, "run": run, "documents": doc_list, "in_flight": in_flight,
            "queries": tl["queries"], "stage_ids": [s.id for s in stages.INDEXING]}


def history(base: Paths, name: str, limit: int = 15) -> list[dict[str, Any]]:
    exp = get_playground_paths(base, name)
    out = []
    for rec in _records(exp)[:limit]:
        rec = _settle(exp, rec)
        v = job_view(rec)
        v["kind"] = rec.get("kind", "index")
        out.append({k: v.get(k) for k in ("id", "kind", "status", "created_at", "started_at", "finished_at",
                                          "elapsed_s", "error")} |
                   {"summary": {k: (v.get("summary") or {}).get(k) for k in
                                ("indexed", "skipped_fresh", "error_count", "no_text_count", "elapsed_s")}})
    return out


# ── the child process ──────────────────────────────────────────────────────────

class Recorder:
    """Keeps the job record and the event log of the run this process is."""

    def __init__(self, base: Paths, name: str, jid: str, kind: str):
        from .core.worker import EventWriter

        self.exp = get_playground_paths(base, name)
        self.jid = jid
        self.exp.jobs.mkdir(parents=True, exist_ok=True)
        self.rec = read_record(self.exp, jid) or {"id": jid, "kind": kind, "experiment": name,
                                                  "created_at": now()}
        self.events = EventWriter(events_file(self.exp, jid))
        self.path = self.events.path
        self._saved = 0.0

    def _save(self, force: bool = False) -> None:
        if force or time.monotonic() - self._saved > 0.5:
            self._saved = time.monotonic()
            write_json_atomic(job_file(self.exp, self.jid), self.rec)

    def begin(self) -> None:
        self.rec.update(status="running", started_at=now(), pid=os.getpid())
        self._save(True)
        self.events.emit("start", kind=self.rec.get("kind"))

    def progress(self, ev: dict[str, Any]) -> None:
        """The ``progress`` callback of ``run_index``: writes the event, keeps the record's progress current."""
        if "query" in ev:
            self.events.emit("query", **ev["query"])
            return
        self.events.progress(ev)
        if "phase" in ev:
            if ev["phase"] == "convert" and "files" not in self.rec and ev.get("total"):
                self.rec["files"] = ev["total"]           # how many documents the run found (stage 1)
            keep = ("phase", "done", "total", "current", "message", "conversion", "phase_started_at", "current_since")
            self.rec["progress"] = {k: ev[k] for k in keep if k in ev}
            self._save()

    def finish(self, summary: dict[str, Any] | None = None, error: str = "") -> None:
        if error:
            self.rec.update(status="failed", error=error)
        else:
            self.rec.update(status="done", **({"summary": summary} if summary else {}))
        self.rec["finished_at"] = now()
        self._save(True)
        self.events.close()

    def failed(self, exc: BaseException) -> None:
        self.finish(error=f"{type(exc).__name__}: {exc}")


def run_in_child(base: Paths, name: str, jid: str, kind: str, fn) -> dict[str, Any]:
    """Run ``fn(recorder)`` as job *jid* in this process: record start, result or failure; return the result."""
    rec = Recorder(base, name, jid, kind)
    rec.begin()
    try:
        result = fn(rec)
    except BaseException as exc:             # noqa: BLE001 - recorded, then re-raised for the exit code
        rec.failed(exc)
        raise
    rec.finish(summary=result if isinstance(result, dict) else None)
    return result
