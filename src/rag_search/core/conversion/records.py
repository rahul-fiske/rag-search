"""Build a document's page records from its profile and what docling reported (stdlib only).

Phase P0: the branch of a page is the routing decision its profile implies (``router.decide``)
while the *reader* is docling for every page.  ``reader.mode`` says how docling was run, so a later
phase that reads some pages with another reader changes ``reader`` and nothing else.
"""

from __future__ import annotations

from typing import Any

from . import profiler, router, trace

PROFILE_KEEP = ("chars", "text_ok", "image_cover", "dpi", "rotation", "size", "size_px", "script",
                "hidden_ocr_layer", "exif_orientation", "ink", "big_pics")


def _compact(profile_page: dict[str, Any] | None) -> dict[str, Any]:
    if not profile_page:
        return {}
    return {k: profile_page[k] for k in PROFILE_KEEP if k in profile_page and profile_page[k] not in (None, "")}


def build_pages(kind: str, profile: dict[str, Any] | None, page_stats: list[dict[str, Any]] | None,
                *, reader: dict[str, Any], outcome_if_empty: str = "no_text",
                note: str = "") -> list[dict[str, Any]]:
    """One record per page.  *page_stats* is docling's per-page facts (``collect_page_stats``) or
    None when the Markdown came from an earlier run or was copied."""
    profile = profile or {"kind": kind, "pages": [], "page_count": 0}
    routed = {n: (b, w) for n, b, w in profiler.route_pages(profile)} if profile.get("pages") else {}
    prof_by_page = {int(p.get("page", 0)): p for p in profile.get("pages", [])}
    stats = {int(s["page"]): s for s in (page_stats or [])}
    numbers = sorted(set(routed) | set(stats)) or [1]
    records = []
    for n in numbers:
        if n in routed:
            branch, why = routed[n]
        elif kind == "pdf":
            branch, why = "unknown", profile.get("error") or "no profile for this page"
        else:
            branch, why = router.decide(kind)
        st = stats.get(n)
        out: dict[str, Any] = {}
        conf = None
        outcome = "pass"
        if st:
            out = {k: st[k] for k in ("chars", "script", "tables", "pictures", "big_pictures") if st.get(k)}
            if not out and st.get("chars") == 0:
                out = {"chars": 0}
            conf = st.get("confidence")
            if not (st.get("chars") or st.get("tables")):
                outcome = outcome_if_empty
        records.append(trace.page_record(
            n, branch, why, profile=_compact(prof_by_page.get(n)),
            reader=reader if branch != "copy" else {"tool": "copy"}, outcome=outcome,
            out=out, confidence=conf, note=note if (n == numbers[0] and note) else ""))
    return records
