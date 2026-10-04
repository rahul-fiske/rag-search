"""Dry run: profile the source files and say how many pages would take which branch, and about
how long converting them takes (no conversion, no model; sources are opened read-only).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from ...paths import META_FILE, Paths, read_json
from . import profiler, vlm

DEFAULT_S_PER_PAGE = 0.5            # docling with forced OCR on this Mac (measured: 0.39-0.54)
PLANNED_VLM_S = (3.0, 10.0)         # document VLM, per scanned page on an M-series Mac (an estimate, analysis section 14)


def reader_status() -> dict[str, Any]:
    """Can the document reader run now?  ``{"mode", "usable", "model", "why"}`` (nothing is started)."""
    out: dict[str, Any] = {"mode": vlm.mode(), "usable": False, "model": "", "why": ""}
    r = vlm.shared()
    if r is None:
        out["why"] = "switched off (RAG_SEARCH_VLM=off)"
        return out
    r.check()
    out.update(model=r.model, usable=not r.dead, why=r.dead)
    return out


def recent_rate(paths: Paths) -> tuple[float, str]:
    """Seconds per page of docling conversion from the documents already indexed here, else the
    default.  Returns (rate, where it came from)."""
    secs = pages = 0.0
    if paths.index.is_dir():
        for meta_file in paths.index.rglob(META_FILE):
            meta = read_json(meta_file)
            conv = (meta or {}).get("conversion") or {}
            n = int(conv.get("pages", 0) or 0)
            t = float((conv.get("time_s") or {}).get("convert", 0) or 0)
            if n and t and (conv.get("branches") or {}).keys() - {"copy"}:
                secs += t
                pages += n
    if pages >= 20:
        return secs / pages, f"measured on {int(pages)} pages indexed here"
    return DEFAULT_S_PER_PAGE, "default (nothing measured here yet)"


def estimate(paths: Paths, sources: list[Path], *, budget_s: float = 60.0,
             progress: Callable[[int, int, str], None] | None = None) -> dict[str, Any]:
    """Profile *sources* (stopping when *budget_s* is used up: ``partial``) and add up the pages."""
    t0 = time.perf_counter()
    by_ext: dict[str, int] = {}
    pages_by_branch: dict[str, int] = {}
    scripts: dict[str, int] = {}
    errors: list[dict[str, str]] = []
    profiled = 0
    for i, src in enumerate(sources, 1):
        if budget_s and time.perf_counter() - t0 > budget_s:
            break
        if progress:
            progress(i, len(sources), src.name)
        prof = profiler.profile_file(src)
        if prof.get("error"):
            errors.append({"src": str(src), "message": prof["error"]})
        ext = src.suffix.lower().lstrip(".")
        by_ext[ext] = by_ext.get(ext, 0) + 1
        for n, branch, _why in profiler.route_pages(prof):
            pages_by_branch[branch] = pages_by_branch.get(branch, 0) + 1
        for p in prof.get("pages", []):
            if p.get("script"):
                scripts[p["script"]] = scripts.get(p["script"], 0) + 1
        profiled += 1
    pages = sum(pages_by_branch.values())
    rate, source = recent_rate(paths)
    raster = pages_by_branch.get("raster", 0) + pages_by_branch.get("image", 0)
    return {
        "files": len(sources), "profiled": profiled, "partial": profiled < len(sources),
        "pages": pages, "branches": pages_by_branch, "by_extension": by_ext, "scripts": scripts,
        "profile_s": round(time.perf_counter() - t0, 1), "errors": errors[:20],
        "error_count": len(errors),
        "docling": {"s_per_page": round(rate, 2), "basis": source, "seconds": round(pages * rate)},
        "planned_vlm": {"pages": raster, "seconds_low": round(raster * PLANNED_VLM_S[0]),
                        "seconds_high": round(raster * PLANNED_VLM_S[1]),
                        "note": "estimate for the document VLM on scanned pages and images "
                                "(3-10 s per page on an M-series Mac, not measured here)",
                        "reader": reader_status()},
    }
