"""Indexing job records on disk (stdlib only): read side shared by daemon, API and CLI.

``jobs/<id>.json`` is written only by the indexer daemon; the worker writes only
``jobs/<id>.events.jsonl``.  Records stay readable when the daemon is not running.
"""

from __future__ import annotations

import json
import re
import time
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


SKIPPED = ("skipped", "known", "unsupported")      # what a run did not convert: unchanged, not tried again, unsupported


def documents(paths: Paths, job_id: str, limit: int = 50, *, status: str = "",
             collection: str = "", q: str = "", hidden: "set[str] | None" = None,
             branch: str = "", outcome: str = "") -> dict[str, Any]:
    """Per-document results of a run, read from its event log (works while it is running).

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
                           "by_outcome": {}, "by_status_filtered": {}, "matched": 0, "items": []}
    if not JOB_ID_RE.fullmatch(job_id or ""):
        return out
    try:
        raw = events_file(paths, job_id).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    latest: dict[tuple[str, str], dict[str, Any]] = {}
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
    items = list(latest.values())
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
    out["total"] = len(items)
    filtered = items
    if collection:
        filtered = [i for i in filtered if str(i.get("collection", "")) == collection]
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
    out["items"] = filtered[-max(0, int(limit)):] if limit else []
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
        out["summary"] = {
            **{k: v for k, v in summ.items()
               if k not in ("errors", "no_text", "unsupported_extension", "removed", "known")},
            "errors": errs[:20], "error_count": len(errs),
            "no_text": empty[:20], "no_text_count": len(empty),
            "unsupported_extension": uns[:20], "unsupported_count": len(uns),
            "removed": removed[:20], "removed_count": len(removed),
            "known": known[:20], "known_count": len(known),
        }
    return out
