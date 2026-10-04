"""The quality gate: does what a reader produced for a page look right? (stdlib only)

Checks, each returning ``{"name", "ok", "detail"}``:

1. ``coverage``      a digital page lost text (much less comes out than its text layer holds), or a
                     scanned page with clear ink came out (nearly) empty
2. ``script``        the text is garbled (broken font map, run-together words, OCR noise), or its
                     script is not the one the page's own text layer shows
3. ``docling_grade`` docling's own confidence for the page is poor
4. ``table_shape``   a table is ragged, almost empty, or has no header text
5. ``low_resolution`` an image page at under 150 dpi (or a small image of unknown dpi)
6. ``running_balance`` / ``totals``  the arithmetic of a table holds (``validators.py``)

``verdict`` is ``ok``, ``suspect`` (at least one check failed) or ``empty`` (nothing was read and
nothing is expected).  ``violations`` are the validators' suspect cells (with a hypothesis each) for
the repair step.  Everything here is deterministic and cheap: milliseconds per page.
"""

from __future__ import annotations

from typing import Any

from ..docling_convert import dominant_script, page_text_ok
from . import tables, validators

MIN_LAYER_CHARS = 100                # below this the text layer is too short to judge coverage by
LOST_TEXT_RATIO = 0.5                # a digital page keeps at least this share of its layer's characters
INK_PAGE = 0.01                      # a scanned page with this much ink must yield some text
MIN_SCAN_CHARS = 20
NOISE_TOKEN_SHARE = 0.4              # share of one-character tokens that marks OCR noise
POOR_SCORE = 0.45
MAX_VIOLATIONS = 10


def _check(name: str, ok: bool, detail: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {"name": name, "ok": ok}
    if detail:
        out["detail"] = detail
    return out


def _chars(md: str) -> int:
    return len("".join(tables.plain_text(md).split()))


def _coverage(md: str, branch_kind: str, prof: dict[str, Any]) -> dict[str, Any]:
    n = _chars(md)
    if branch_kind == "digital":
        layer = int(prof.get("chars") or 0)
        if layer >= MIN_LAYER_CHARS and n < LOST_TEXT_RATIO * layer:
            return _check("coverage", False, f"{n} characters came out of a text layer of {layer}")
    elif branch_kind == "scan":
        ink = prof.get("ink")
        if isinstance(ink, (int, float)) and ink >= INK_PAGE and n < MIN_SCAN_CHARS:
            return _check("coverage", False, f"the page has {round(100 * ink, 1)} % ink but only {n} characters were read")
    return _check("coverage", True)


def _script(md: str, branch_kind: str, prof: dict[str, Any]) -> dict[str, Any]:
    text = tables.plain_text(md)
    if len("".join(text.split())) < 20:
        return _check("script", True)
    if not page_text_ok(text):
        return _check("script", False, "the text looks garbled (replacement characters, run-together words)")
    toks = text.split()
    if branch_kind == "scan" and len(toks) >= 30:
        short = sum(1 for t in toks if len(t) == 1 and not t.isdigit()) / len(toks)
        if short > NOISE_TOKEN_SHARE:
            return _check("script", False, f"{round(100 * short)} % of the words are single letters: OCR noise")
    want = str(prof.get("script") or "")
    if want and want not in ("Latin", "Mixed", "none"):
        got = dominant_script(text)
        if got and got != want and got != "Mixed":
            return _check("script", False, f"the page's text layer is {want} but the result is {got}")
    return _check("script", True)


LOW_DPI = 150                        # an image file below this resolution is read, but flagged
LOW_PX = 1000                        # ... and so is one whose long side is shorter than this (no dpi given)


def _resolution(branch_kind: str, prof: dict[str, Any]) -> dict[str, Any]:
    """A page that is an image (a scan, a photo) at a low resolution is read less reliably, and a
    small digit may be wrong without any check noticing: say so (the page is listed as needing a look)."""
    if branch_kind != "scan":
        return _check("low_resolution", True)
    dpi = prof.get("dpi")
    px = prof.get("size_px")
    if isinstance(dpi, (int, float)) and 0 < dpi < LOW_DPI:
        return _check("low_resolution", False, f"the image is only {round(dpi)} dpi (a scan should be {LOW_DPI} or more)")
    if not dpi and isinstance(px, (list, tuple)) and len(px) == 2 and 0 < max(px) < LOW_PX:
        return _check("low_resolution", False, f"the image is only {px[0]} x {px[1]} pixels")
    return _check("low_resolution", True)


def _grade(conf: dict[str, Any] | None) -> dict[str, Any]:
    conf = conf or {}
    grade = str(conf.get("grade") or "").lower()
    low = conf.get("low")
    if grade == "poor" or (isinstance(low, (int, float)) and low < POOR_SCORE):
        return _check("docling_grade", False, f"docling graded the page {grade or 'poor'}"
                      + (f" (lowest score {low})" if isinstance(low, (int, float)) else ""))
    return _check("docling_grade", True)


def _table_shape(md: str) -> dict[str, Any]:
    for t in tables.find_tables(md):
        if t.kind == "pipe":
            raw = md[t.start:t.end].strip("\n").split("\n")
            widths = {len(tables._split_pipe_row(r)) for i, r in enumerate(raw) if i != 1}
            if len(widths) > 1:
                return _check("table_shape", False, f"table rows have different numbers of cells ({sorted(widths)})")
        body = t.body
        cells = [c for r in body for c in r]
        if len(body) >= 3 and cells and sum(1 for c in cells if not c.strip()) / len(cells) > 0.6:
            return _check("table_shape", False, "more than 60 % of a table's cells are empty")
        if t.width >= 2 and t.header and not any(c.strip() for c in t.header) and len(body) >= 2:
            return _check("table_shape", False, "a table has no header text")
        why = validators.layout_problem(t)
        if why:
            return _check("table_shape", False, why)
    return _check("table_shape", True)


def check_page(md: str, *, branch_kind: str, profile: dict[str, Any] | None = None,
               confidence: dict[str, Any] | None = None, validate: bool = True) -> dict[str, Any]:
    """Gate result for one page's Markdown.  *branch_kind* is ``digital`` (a text layer exists),
    ``scan`` (the page was read as an image) or ``other``; *profile* the page's profile record."""
    prof = profile or {}
    n = _chars(md)
    has_table = bool(tables.find_tables(md))
    if n == 0 and not has_table:
        wanted = branch_kind == "digital" and int(prof.get("chars") or 0) >= MIN_LAYER_CHARS
        ink = prof.get("ink")
        if not wanted and not (branch_kind == "scan" and isinstance(ink, (int, float)) and ink >= INK_PAGE):
            return {"verdict": "empty", "checks": []}
    checks = [_coverage(md, branch_kind, prof), _script(md, branch_kind, prof), _grade(confidence),
              _table_shape(md), _resolution(branch_kind, prof)]
    violations: list[dict[str, Any]] = []
    if validate:
        violations = validators.page_violations(md)
        for name in ("running_balance", "totals"):
            vs = [v for v in violations if v["validator"] == name]
            if vs:
                checks.append(_check(name, False, vs[0]["why"]))
        if violations and not any(c["name"] in ("running_balance", "totals") for c in checks):
            checks.append(_check("running_balance", False, violations[0].get("why", "")))
    out: dict[str, Any] = {"verdict": "ok" if all(c["ok"] for c in checks) else "suspect",
                           "checks": [c for c in checks if not c["ok"]]}
    if violations:
        out["violations"] = violations[:MAX_VIOLATIONS]
    return out


def failed(gate: dict[str, Any] | None) -> list[str]:
    """Names of the checks that failed."""
    return [c["name"] for c in (gate or {}).get("checks", []) if not c.get("ok")]
