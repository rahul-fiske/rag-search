"""Indexing job records on disk (stdlib only): read side shared by daemon, API and CLI.

``jobs/<id>.json`` is written only by the indexer daemon; the worker writes only
``jobs/<id>.events.jsonl``.  Records stay readable when the daemon is not running.
"""

from __future__ import annotations

from collections import Counter

import json
import re
import threading
import time
from pathlib import Path
from typing import Any

from .paths import Paths, read_json

ACTIVE = ("queued", "running")
JOB_ID_RE = re.compile(r"[0-9a-z-]{1,40}")


def now() -> float:
    return round(time.time(), 3)


def job_file(paths: Paths, job_id: str):
    return paths.jobs / f"{job_id}.json"


def events_file(paths: Paths, job_id: str):
    return paths.jobs / f"{job_id}.events.jsonl"


def read_record(paths: Paths, job_id: str) -> dict[str, Any] | None:
    if not JOB_ID_RE.fullmatch(job_id or ""):
        return None
    rec = read_json(job_file(paths, job_id))
    return rec if isinstance(rec, dict) and rec.get("id") else None


def all_records(paths: Paths) -> list[dict[str, Any]]:
    """Every job record, newest first."""
    recs = []
    if paths.jobs.is_dir():
        for f in paths.jobs.glob("*.json"):
            rec = read_json(f)
            if isinstance(rec, dict) and rec.get("id"):
                recs.append(rec)
    recs.sort(key=lambda r: r.get("created_at", 0), reverse=True)
    return recs


_DOC_LOCK = threading.Lock()
_DOC_STATE: dict[str, dict[str, Any]] = {}       # events file -> {"ino", "offset", "latest"}: parsed once, then only what is appended
_DOC_MAX = 4


def _doc_items(path: Path) -> list[dict[str, Any]] | None:
    """Every document of a run's event log with its latest state, in order of last change (None when there is no log).
    The log is read incrementally: the dashboard asks every second, and a finished run's log never changes."""
    key = str(path)
    try:
        st = path.stat()
    except OSError:
        with _DOC_LOCK:
            _DOC_STATE.pop(key, None)
        return None
    with _DOC_LOCK:
        cur = _DOC_STATE.get(key)
        if cur is None or cur["ino"] != st.st_ino or st.st_size < cur["offset"]:      # new, replaced or rewritten
            cur = _DOC_STATE[key] = {"ino": st.st_ino, "offset": 0, "latest": {}}
            while len(_DOC_STATE) > _DOC_MAX:
                _DOC_STATE.pop(next(iter(_DOC_STATE)))
        if st.st_size > cur["offset"]:
            try:
                with open(path, "rb") as fh:
                    fh.seek(cur["offset"])
                    data = fh.read(st.st_size - cur["offset"])
            except OSError:
                return list(cur["latest"].values())
            end = data.rfind(b"\n")
            if end >= 0:
                cur["offset"] += end + 1
                _feed_docs(cur["latest"], data[:end].decode("utf-8", "replace"))
        return list(cur["latest"].values())


def _feed_docs(latest: dict[tuple[str, str], dict[str, Any]], raw: str) -> None:
    for line in raw.splitlines():
        if '"doc"' not in line:
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("event") != "doc":
            continue
        item = {("finished_at" if k == "ts" else k): v for k, v in ev.items() if k != "event"}
        # by the document's path inside its collection (older logs have only the file name)
        key = (str(item.get("collection", "")), str(item.get("path") or item.get("source", "")))
        earlier = latest.pop(key, None)          # re-inserted: the list stays in order of last change
        if earlier and item.get("status") != "skipped":
            item = {**{k: v for k, v in earlier.items() if k not in ("status", "message")}, **item}
        latest[key] = item


SKIPPED = ("skipped", "known", "unsupported", "live_photo")   # what a run did not convert: unchanged, not tried again,
                                                               # unsupported, the video of a Live Photo


def _kind_grade(item: dict[str, Any]) -> tuple[str, str]:
    """A listed document's kind and grade (from its "understood" record), ``("", "")`` when the run did not convert it.
    A failure without a record still has a grade: it failed."""
    u = item.get("understood") if isinstance(item.get("understood"), dict) else {}
    st = str(item.get("status", ""))
    if u.get("kind"):
        return str(u["kind"]), "error" if st == "error" else str(u.get("grade") or "complete")
    if st in ("error", "no_text"):
        return "other document", st
    return "", ""


def documents(paths: Paths, job_id: str, limit: int = 50, *, status: str = "",
             collection: str = "", q: str = "", hidden: "set[str] | None" = None,
             branch: str = "", outcome: str = "", kind: str = "", grade: str = "") -> dict[str, Any]:
    """Per-document results of a run, read from its event log (works while it is running).

    ``kind`` / ``grade`` keep the documents of that kind and grade -- the vocabulary of the "understood" table
    (``core/understanding.py``: a converted document's kind, and complete | partial | no_text | error); ``by_kind`` /
    ``by_grade`` count them.  A document the run did not convert (unchanged, not tried again) has neither.

    -> {"total": n, "by_status": {...}, "by_collection": {coll: {status: n, ...}, ...},
        "matched": n, "items": [the last *limit* MATCHING documents, oldest first]}.
    Each item has collection, source, status and, for converted / indexed documents, chunks and
    convert_s / chunk_s / embed_s / total_s.  A document is listed once, with its latest status:
    "converted" (converted and chunked, waiting for the embed phase) becomes "indexed" (or
    "error") when embedding is done.

    Converted documents carry ``conversion``, their conversion summary (pages per branch and
    outcome, a run-length branch strip, times, cost -- see core/conversion/trace.py).  ``branch``
    / ``outcome`` keep the documents with at least one page of that branch / outcome;
    ``by_branch`` / ``by_outcome`` count documents the same way.

    ``status``/``collection``/``q`` narrow which documents are counted and returned -- ``total``,
    ``by_status`` and ``by_collection`` are always computed before filtering (so the stat cards
    stay accurate regardless of what the document list is currently filtered to); ``matched`` is
    the count *after* filtering, before ``limit`` truncates ``items``.  ``by_status_filtered`` counts the statuses
    after every filter but the status itself: the numbers on the status filters, which then agree with the list under
    them whatever is typed in the search box.  The status ``skipped_all`` stands for everything the run did not
    convert (``SKIPPED``: unchanged, not tried again, unsupported format).  ``hidden`` (collection
    names, case-insensitive) excludes documents from every count and from ``items`` -- this runs
    before anything else, so a client that cannot see a collection never sees it in a total either.
    """
    out: dict[str, Any] = {"total": 0, "by_status": {}, "by_collection": {}, "by_branch": {},
                           "by_outcome": {}, "by_kind": {}, "by_grade": {}, "by_status_filtered": {}, "matched": 0,
                           "items": []}
    if not JOB_ID_RE.fullmatch(job_id or ""):
        return out
    items = _doc_items(events_file(paths, job_id))
    if items is None:
        return out
    if hidden:
        hidden_fold = {h.casefold() for h in hidden}
        items = [i for i in items if str(i.get("collection", "")).casefold() not in hidden_fold]
    for item in items:
        st = str(item.get("status", ""))
        c = str(item.get("collection", ""))
        out["by_status"][st] = out["by_status"].get(st, 0) + 1
        slot = out["by_collection"].setdefault(c, {})
        slot[st] = slot.get(st, 0) + 1
        conv = item.get("conversion") or {}
        for b, n in (conv.get("branches") or {}).items():
            if n:
                out["by_branch"][b] = out["by_branch"].get(b, 0) + 1
        for o, n in (conv.get("outcomes") or {}).items():
            if n:
                out["by_outcome"][o] = out["by_outcome"].get(o, 0) + 1
        k, g = _kind_grade(item)
        if k:
            out["by_kind"][k] = out["by_kind"].get(k, 0) + 1
            out["by_grade"][g] = out["by_grade"].get(g, 0) + 1
    out["total"] = len(items)
    filtered = items
    if collection:
        filtered = [i for i in filtered if str(i.get("collection", "")) == collection]
    if kind:
        filtered = [i for i in filtered if _kind_grade(i)[0] == kind]
    if grade:
        filtered = [i for i in filtered if _kind_grade(i)[1] == grade]
    if branch:
        filtered = [i for i in filtered if ((i.get("conversion") or {}).get("branches") or {}).get(branch)]
    if outcome:
        filtered = [i for i in filtered if ((i.get("conversion") or {}).get("outcomes") or {}).get(outcome)]
    if q:
        ql = q.casefold()
        filtered = [i for i in filtered if ql in str(i.get("path") or i.get("source", "")).casefold()]
    for i in filtered:
        st = str(i.get("status", ""))
        out["by_status_filtered"][st] = out["by_status_filtered"].get(st, 0) + 1
    out["by_status_filtered"][""] = len(filtered)
    out["by_status_filtered"]["skipped_all"] = sum(out["by_status_filtered"].get(s, 0) for s in SKIPPED)
    if status:
        wanted = SKIPPED if status == "skipped_all" else (status,)
        filtered = [i for i in filtered if str(i.get("status", "")) in wanted]
    out["matched"] = len(filtered)
    out["items"] = [dict(i) for i in filtered[-max(0, int(limit)):]] if limit else []
    return out


def view(rec: dict[str, Any]) -> dict[str, Any]:
    """A job record trimmed for API output."""
    out = {k: rec.get(k) for k in ("id", "status", "mode", "path", "created_at", "started_at",
                                   "finished_at", "progress", "error", "publish",
                                   "search_reload", "pid")}
    end = rec.get("finished_at") or now()
    if rec.get("started_at"):
        out["elapsed_s"] = round(end - rec["started_at"], 1)
    summ = rec.get("summary")
    if summ:
        errs = summ.get("errors", [])
        empty = summ.get("no_text") or []
        uns = summ.get("unsupported_extension") or []
        removed = summ.get("removed") or []
        known = summ.get("known") or []
        live = summ.get("live_photo") or []
        out["summary"] = {
            **{k: v for k, v in summ.items()
               if k not in ("errors", "no_text", "unsupported_extension", "removed", "known", "live_photo")},
            "errors": errs[:20], "error_count": len(errs),
            "no_text": empty[:20], "no_text_count": len(empty),
            "unsupported_extension": uns[:20], "unsupported_count": len(uns),
            "unsupported_by_extension": dict(Counter(u.get("extension") or "" for u in uns).most_common()),
            "removed": removed[:20], "removed_count": len(removed),
            "known": known[:20], "known_count": len(known),
            "live_photo": live[:20], "live_photo_count": len(live),
        }
    return out


# ── retention: the records and logs of old runs ──────────────────────────────

DEFAULT_KEEP = 10            # runs whose records and logs are kept (config ``indexer.keep_runs``; 0 = keep every run)
ORPHAN_GRACE_S = 3600.0      # a log without a record is removed only when it has not been written for this long


def run_files(paths: Paths, job_id: str) -> list[Path]:
    """The files of one run: its record, its event log and its worker log."""
    return [job_file(paths, job_id), events_file(paths, job_id), paths.jobs / f"{job_id}.log"]


def disk_use(paths: Paths) -> dict[str, int]:
    """How many runs are on record and how many bytes their files take."""
    n = size = 0
    if paths.jobs.is_dir():
        for f in paths.jobs.iterdir():
            try:
                size += f.stat().st_size
            except OSError:
                continue
            n += f.suffix == ".json" and not f.name.startswith(".")
    return {"runs": n, "bytes": size}


def prune(paths: Paths, keep: int, *, active: str = "", dry_run: bool = False) -> dict[str, Any]:
    """Delete the records and logs of every run but the newest *keep* (at least 1).  Never touched: a run that is queued or
    running (or *active*, the one the daemon holds), and the newest finished run.  A log whose record is gone is removed too
    once it has been quiet for an hour.  -> ``{"keep", "removed": [ids], "files", "freed_bytes", "kept"}``."""
    keep = max(1, int(keep))
    recs = all_records(paths)                                    # newest first
    newest_done = next((r["id"] for r in recs if r.get("status") not in ACTIVE), "")
    safe = {r["id"] for r in recs if r.get("status") in ACTIVE} | {active, newest_done} - {""}
    doomed = [r["id"] for r in recs[keep:] if r["id"] not in safe]
    files: list[Path] = [f for jid in doomed for f in run_files(paths, jid) if f.exists()]
    known = {r["id"] for r in recs}
    now_t = time.time()
    if paths.jobs.is_dir():                                      # logs left behind by a record removed earlier
        for f in paths.jobs.iterdir():
            jid = f.name.split(".", 1)[0]
            if (f.name.endswith((".events.jsonl", ".log")) and JOB_ID_RE.fullmatch(jid) and jid not in known
                    and jid not in safe and f not in files):
                try:
                    if now_t - f.stat().st_mtime > ORPHAN_GRACE_S:
                        files.append(f)
                except OSError:
                    continue
    freed = 0
    for f in files:
        try:
            size = f.stat().st_size
            if not dry_run:
                f.unlink()
            freed += size
        except OSError:
            continue
    return {"keep": keep, "removed": doomed, "files": len(files), "freed_bytes": freed,
            "kept": len(recs) - len(doomed), "dry_run": dry_run}


def keep_setting(cfg: dict[str, Any]) -> int:
    """``indexer.keep_runs`` from a config dict (0 = keep every run)."""
    try:
        return max(0, int((cfg.get("indexer") or {}).get("keep_runs", DEFAULT_KEEP)))
    except (TypeError, ValueError):
        return DEFAULT_KEEP
