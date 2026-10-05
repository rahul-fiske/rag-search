"""Page and document records of the conversion pipeline, and their aggregation (stdlib only).

One vocabulary is used by the engine, the CLI, the HTTP API and the dashboard:

* a page takes one **branch** (which path of the flow chart read it) and ends with one
  **outcome** (what became of it);
* a document keeps a **summary** (counts, a run-length branch strip, times, cost) in its
  ``index.meta.json`` and on its ``doc`` events, and the full per-page **trace** next to its
  Markdown (``markup/<coll>/<doc>.trace.json``);
* a run keeps **totals** (``RunTotals``), live in the job's progress and frozen in its summary.

Phase P0 records what *would* be decided for every page (``profile`` -> branch) while every page
is still read by docling; later phases change the readers, not these records.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable

from ...paths import read_json, write_json_atomic

TRACE_VERSION = 1
TRACE_SUFFIX = ".trace.json"

BRANCHES = ("copy", "office", "digital", "raster", "image", "embedded", "fallback", "cached",
            "unknown")
BRANCH_LETTER = {"copy": "c", "office": "o", "digital": "d", "raster": "r", "image": "i",
                 "embedded": "e", "fallback": "f", "cached": "k", "unknown": "?"}
LETTER_BRANCH = {v: k for k, v in BRANCH_LETTER.items()}
BRANCH_TITLE = {
    "copy": "Markdown / text, used as is",
    "office": "Office / HTML file (docling reads the structure)",
    "digital": "PDF page with a text layer",
    "raster": "scanned or photographed PDF page, read as an image by the document VLM",
    "image": "image file (or one frame of a multi-page TIFF), read by the document VLM",
    "embedded": "large picture inside a PDF page, read by the document VLM",
    "fallback": "read by docling + OCR (the document VLM is off, not installed, or could not read this page)",
    "cached": "page result reused from the page cache",
    "unknown": "not profiled (the profiling time limit was reached)",
}

BRANCH_SHORT = {"copy": "text", "office": "office", "digital": "digital", "raster": "scanned",
                "image": "image", "embedded": "embedded", "fallback": "scanned (OCR)",
                "cached": "cached", "unknown": "unprofiled"}

OUTCOMES = ("pass", "repaired", "low", "no_text", "error")
GRADES = ("poor", "fair", "good", "excellent")
MAX_LISTED_PAGES = 50


def trace_path_for(md_path: Path) -> Path:
    """``markup/<coll>/<doc>.md`` -> ``markup/<coll>/<doc>.trace.json``."""
    return md_path.with_name(md_path.name[:-3] + TRACE_SUFFIX if md_path.name.endswith(".md")
                             else md_path.name + TRACE_SUFFIX)


def page_record(page: int, branch: str, why: str, *, profile: dict[str, Any] | None = None,
                reader: dict[str, Any] | None = None, outcome: str = "pass",
                out: dict[str, Any] | None = None, confidence: dict[str, Any] | None = None,
                note: str = "") -> dict[str, Any]:
    """One page's record (plan 4.2).  Empty parts are left out to keep traces small."""
    rec: dict[str, Any] = {"page": int(page), "branch": branch, "outcome": outcome, "why": why}
    for key, val in (("profile", profile), ("reader", reader), ("out", out),
                     ("docling", confidence)):
        if val:
            rec[key] = val
    if note:
        rec["note"] = note
    return rec


def _count(items: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for i in items:
        out[i] = out.get(i, 0) + 1
    return out


def _page_script(p: dict[str, Any]) -> str:
    out = (p.get("out") or {}).get("script") or ""
    prof = (p.get("profile") or {}).get("script") or ""
    return out or prof


def page_kind(p: dict[str, Any]) -> str:
    """What kind of page a page is (``digital``, ``raster``, ``image``, ...).  A page served from the page cache
    has the branch ``cached`` -- that says how this run got it, not what it is; its kind is the branch it was
    first read on (``was``).  Counting by kind keeps a cached scan a scan."""
    b = str(p.get("branch") or "unknown")
    return str(p.get("was") or b) if b == "cached" else b


def summarize(pages: list[dict[str, Any]], *, time_s: dict[str, float] | None = None,
              cost: dict[str, float] | None = None, readers: list[str] | None = None,
              trace: str = "", note: str = "") -> dict[str, Any]:
    """A document's summary from its page records (plan 4.3)."""
    branches = _count(page_kind(p) for p in pages)
    outcomes = _count(p.get("outcome", "pass") for p in pages)
    scripts = _count(s for s in (_page_script(p) for p in pages) if s and s != "none")
    grades = _count(str((p.get("docling") or {}).get("grade") or "").lower() for p in pages
                    if (p.get("docling") or {}).get("grade"))
    poor = [p["page"] for p in pages if str((p.get("docling") or {}).get("grade", "")).lower() == "poor"]
    low = [p["page"] for p in pages if p.get("outcome") == "low"]
    step_s: dict[str, float] = {}
    for p in pages:
        for k, v in (p.get("time_s") or {}).items():
            step_s[k] = step_s.get(k, 0.0) + float(v)
    cached = sum(1 for p in pages if p.get("cache") == "hit")
    tokens = sum(int(p.get("tokens") or 0) for p in pages)
    gpu_s = sum(float(p.get("gpu_s") or 0.0) for p in pages)
    models = sorted({str((p.get("reader") or {}).get("model")) for p in pages
                     if (p.get("reader") or {}).get("tool") == "vlm" and (p.get("reader") or {}).get("model")}
                    | {str((p.get("repair") or {}).get("model")) for p in pages
                       if (p.get("repair") or {}).get("model") and (p.get("repair") or {}).get("tried")})
    rep_tried = sum(int((p.get("repair") or {}).get("tried") or 0) for p in pages)
    rep_fixed = sum(int((p.get("repair") or {}).get("fixed") or 0) for p in pages)
    out = {
        "v": TRACE_VERSION, "pages": len(pages), "branches": branches, "outcomes": outcomes,
        "strip": make_strip(page_kind(p) for p in pages),
        "low_pages": low[:MAX_LISTED_PAGES], "scripts": scripts,
        "tables": sum(int((p.get("out") or {}).get("tables", 0)) for p in pages),
        "big_pictures": sum(int((p.get("out") or {}).get("big_pictures", 0)) for p in pages),
        "time_s": {k: round(float(v), 2) for k, v in (time_s or {}).items()},
        "cost": {k: round(float(v), 2) for k, v in (cost or {}).items()},
        "readers": list(readers or []),
    }
    if step_s:
        out["step_s"] = {k: round(v, 2) for k, v in step_s.items()}
    if cached:
        out["cached_pages"] = cached
    if tokens:
        out["tokens"] = tokens
    if gpu_s:
        out["gpu_s"] = round(gpu_s, 2)
    if models:
        out["models"] = models
    if rep_tried or rep_fixed:
        out["repair_tried"], out["repaired_cells"] = rep_tried, rep_fixed
    merged = sum(1 for p in pages if (p.get("reconcile") or {}).get("role") == "continues")
    if merged:
        out["merged_tables"] = merged
    gate_failed = _count(c.get("name", "?") for p in pages for c in (p.get("gate") or {}).get("checks", []))
    if gate_failed:
        out["gate_failed"] = gate_failed
    if grades:
        out["docling_grades"] = grades
        out["poor_pages"] = poor[:MAX_LISTED_PAGES]
    if trace:
        out["trace"] = trace
    if note:
        out["note"] = note
    return out


def write_trace(path: Path, *, source: str, src_sha: str, pages: list[dict[str, Any]],
                summary: dict[str, Any], profile: dict[str, Any] | None = None,
                settings: str = "") -> None:
    """Write ``<doc>.trace.json`` (atomic, like the Markdown)."""
    write_json_atomic(path, {
        "version": TRACE_VERSION, "source": source, "src_sha256": src_sha,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "convert": settings, "profile": profile or {}, "summary": summary, "pages": pages,
    }, indent=None)


def read_trace(path: Path) -> dict[str, Any]:
    data = read_json(path)
    return data if isinstance(data, dict) and data.get("version") else {}


def make_strip(branches: Iterable[str]) -> str:
    """Branches of consecutive pages as a run-length string: ``digital x3, raster x2`` is
    ``"d3r2"`` (one letter per branch, see ``BRANCH_LETTER``).  Compact enough for events."""
    out: list[str] = []
    last, n = "", 0
    for b in branches:
        letter = BRANCH_LETTER.get(b, "?")
        if letter == last:
            n += 1
            continue
        if last:
            out.append(f"{last}{n}")
        last, n = letter, 1
    if last:
        out.append(f"{last}{n}")
    return "".join(out)


def parse_strip(strip: str) -> list[tuple[str, int]]:
    """``"d3r2d1"`` -> ``[("digital", 3), ("raster", 2), ("digital", 1)]``."""
    runs: list[tuple[str, int]] = []
    i, n = 0, len(strip or "")
    while i < n:
        letter = strip[i]
        i += 1
        j = i
        while j < n and strip[j].isdigit():
            j += 1
        count = int(strip[i:j] or 1)
        runs.append((LETTER_BRANCH.get(letter, "unknown"), count))
        i = j
    return runs


class RunTotals:
    """What a run has converted so far: pages per branch and outcome, time and cost per step.

    ``add(summary)`` once per document that was converted in this run; ``snapshot()`` is the plain
    dict stored in the job's progress (live) and in its summary (frozen at the end).
    """

    def __init__(self, files: dict[str, int] | None = None) -> None:
        self.t0 = time.time()
        self.files = dict(files or {})
        self.docs = 0
        self.pages = 0
        self.branches: dict[str, int] = {}
        self.ok_docs = 0                      # documents that converted successfully (not no-text, not failed)
        self.ok_pages = 0
        self.ok_branches: dict[str, int] = {}
        self.outcomes: dict[str, int] = {}
        self.scripts: dict[str, int] = {}
        self.grades: dict[str, int] = {}
        self.time_s: dict[str, float] = {}
        self.step_s: dict[str, float] = {}
        self.gate_failed: dict[str, int] = {}
        self.cached_pages = 0
        self.tokens = 0
        self.gpu_s = 0.0
        self.cpu_s = 0.0
        self.peak_mb = 0.0
        self.big_pictures = 0
        self.tables = 0
        self.low_docs = 0
        self.no_record = 0
        self.repair_tried = 0
        self.repaired_cells = 0
        self.merged_tables = 0

    def add(self, summary: dict[str, Any] | None, ok: bool = True) -> None:
        """*ok*: the document converted successfully (a failed or empty one still counts in the totals, but
        not in ``ok_*``, the figures of the successfully converted files)."""
        if not summary:
            self.no_record += 1
            return
        self.docs += 1
        self.pages += int(summary.get("pages", 0))
        if ok:
            self.ok_docs += 1
            self.ok_pages += int(summary.get("pages", 0))
            for k, v in (summary.get("branches") or {}).items():
                self.ok_branches[k] = self.ok_branches.get(k, 0) + int(v)
        for key, dst in (("branches", self.branches), ("outcomes", self.outcomes),
                         ("scripts", self.scripts), ("docling_grades", self.grades)):
            for k, v in (summary.get(key) or {}).items():
                dst[k] = dst.get(k, 0) + int(v)
        for k, v in (summary.get("time_s") or {}).items():
            self.time_s[k] = self.time_s.get(k, 0.0) + float(v)
        for k, v in (summary.get("step_s") or {}).items():
            self.step_s[k] = self.step_s.get(k, 0.0) + float(v)
        for k, v in (summary.get("gate_failed") or {}).items():
            self.gate_failed[k] = self.gate_failed.get(k, 0) + int(v)
        self.cached_pages += int(summary.get("cached_pages", 0))
        self.tokens += int(summary.get("tokens", 0))
        self.gpu_s += float(summary.get("gpu_s", 0.0))
        cost = summary.get("cost") or {}
        self.cpu_s += float(cost.get("cpu_s", 0))
        self.peak_mb = max(self.peak_mb, float(cost.get("peak_mb", 0)))
        self.big_pictures += int(summary.get("big_pictures", 0))
        self.tables += int(summary.get("tables", 0))
        if summary.get("low_pages"):
            self.low_docs += 1
        self.repair_tried += int(summary.get("repair_tried", 0))
        self.repaired_cells += int(summary.get("repaired_cells", 0))
        self.merged_tables += int(summary.get("merged_tables", 0))

    def snapshot(self) -> dict[str, Any]:
        elapsed = max(0.001, time.time() - self.t0)
        return {
            "v": TRACE_VERSION, "files": self.files, "docs": self.docs, "pages": self.pages,
            "branches": self.branches, "ok_docs": self.ok_docs, "ok_pages": self.ok_pages,
            "ok_branches": self.ok_branches, "outcomes": self.outcomes, "scripts": self.scripts,
            "docling_grades": self.grades, "big_pictures": self.big_pictures,
            "tables": self.tables, "low_docs": self.low_docs, "no_record": self.no_record,
            "time_s": {k: round(v, 1) for k, v in self.time_s.items()},
            "step_s": {k: round(v, 1) for k, v in self.step_s.items()},
            "gate_failed": self.gate_failed, "cached_pages": self.cached_pages,
            "tokens": self.tokens, "gpu_s": round(self.gpu_s, 1),
            "repair_tried": self.repair_tried, "repaired_cells": self.repaired_cells,
            "merged_tables": self.merged_tables,
            "cost": {"cpu_s": round(self.cpu_s, 1), "peak_mb": round(self.peak_mb)},
            # pages actually read: a page served from the page cache takes no time and is not a rate
            "pages_per_min": round(60.0 * (self.pages - self.cached_pages) / elapsed, 1)
            if self.pages > self.cached_pages else 0.0,
        }


def aggregate(summaries: Iterable[dict[str, Any] | None]) -> dict[str, Any]:
    """Totals over stored per-document summaries (a collection's *Conversion* section)."""
    tot = RunTotals()
    items = list(summaries)
    for s in items:
        tot.add(s)
    snap = tot.snapshot()
    snap.pop("pages_per_min", None)
    snap["documents"] = len(items)
    return snap
