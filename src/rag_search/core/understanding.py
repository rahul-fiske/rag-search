"""What indexing understood of one document: its kind, the facts found in it and how complete that is.

The Indexing tab answers "what kind of documents were these, and how well was each understood?" from one line per
document (the ``work`` event that closes its conversion), so this is worked out once, where the document is at hand:

    kind     what the document turned out to be: a photograph, a photographed page, a Google Maps Timeline, a PDF with
             a text layer, a scanned PDF, ... (a file that failed is named by its format)
    grade    complete | partial | no_text | error
    found    the facts that were found (for a photograph: time, position, place, places nearby, faces, description)
    missing  the expected facts that were not, in the order they matter (a photograph without a position cannot have
             a place either: only "position" is missing then)

Facts come from the Markdown the document was indexed as (the labels of the Photo details section, the Timeline head
line, the outcomes of the pages), so what is reported is what is searchable.
"""

from __future__ import annotations

import re
from typing import Any

from . import describe

PHOTOGRAPH = "photograph"
PHOTOGRAPHED_PAGE = "photographed page"
TIMELINE = "Google Maps Timeline"

# a photograph: label in its Photo details section -> fact
_PHOTO_FACTS = (("time", "- Taken:"), ("position", "- Position:"), ("place", ("- Area:", "- Town:")),
                ("nearby", "- Nearby:"), ("faces", f"- {describe.FACES}"), ("description", f"- {describe.LOOKS}"))
# what a complete photograph has; "faces" and "nearby" are reported but not expected (a landscape has no face, a
# position in a field has no place within 100 m)
PHOTO_EXPECTED = ("time", "position", "place", "description")
_TIMELINE_HEAD = re.compile(r"(\d+) visits and (\d+) journeys")

FORMAT_NAMES = {"pdf": "PDF", "image": "image", "office": "Office or HTML document", "text": "text or Markdown file"}


_FACES = re.compile(rf"^- {re.escape(describe.FACES)} ([1-9]\d*)", re.M)


def photo_facts(text: str) -> list[str]:
    """The facts of a photograph found in its Photo details ("faces" only when at least one face was counted)."""
    out = []
    for name, label in _PHOTO_FACTS:
        labels = label if isinstance(label, tuple) else (label,)
        if name == "faces" and not _FACES.search(text):
            continue
        if any(lab in text for lab in labels):
            out.append(name)
    return out


def _missing_photo(found: list[str]) -> list[str]:
    miss = [x for x in PHOTO_EXPECTED if x not in found]
    if "position" in miss:                      # no position: no place to look up either
        miss = [x for x in miss if x != "place"]
    return miss


def _page_kind(fmt: str, suffix: str, branches: dict[str, int]) -> str:
    if suffix == ".json":
        return "JSON data"
    if fmt != "pdf":
        return {"image": "image read as a page", "office": "Office or HTML document",
                "text": "text or Markdown file"}.get(fmt, "other document")
    text_layer = sum(branches.get(k, 0) for k in ("digital", "copy", "office", "embedded"))
    scanned = sum(branches.get(k, 0) for k in ("raster", "image", "fallback"))
    if text_layer and scanned:
        return "PDF, partly scanned"
    return "PDF, scanned" if scanned else "PDF with a text layer"


def of_document(*, fmt: str, suffix: str, status: str, text: str = "", route: str = "",
                branches: dict[str, int] | None = None, outcomes: dict[str, int] | None = None) -> dict[str, Any]:
    """``{"kind", "grade", "found", "missing", ...}`` for one document.  *fmt* is the profiler's kind of file (pdf |
    image | office | text), *status* the conversion's (prepared | no_text | error | stalled), *text* the Markdown,
    *route* what 3.0 Sniff decided, *branches* / *outcomes* the page counts of its conversion summary."""
    branches, outcomes = branches or {}, outcomes or {}
    if status not in ("prepared",):
        kind = (PHOTOGRAPH if route == "photograph" else TIMELINE if route == "timeline"
                else _page_kind(fmt, suffix, branches) if branches else FORMAT_NAMES.get(fmt, "other document"))
        return {"kind": kind, "grade": "no_text" if status == "no_text" else "error", "found": [], "missing": []}
    if route == "timeline" or branches.get("timeline"):
        m = _TIMELINE_HEAD.search(text)
        visits, journeys = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
        found = ["visits"] if visits else []
        found += ["journeys"] if journeys else []
        found += ["place"] if "  - Area:" in text or "  - Town:" in text else []
        missing = [] if "place" in found or not visits else ["place"]
        return {"kind": TIMELINE, "grade": "partial" if missing or not visits else "complete", "found": found,
                "missing": missing if visits else ["visits"], "visits": visits, "journeys": journeys}
    if route == "photograph" or branches.get("photo"):
        found = photo_facts(text)
        missing = _missing_photo(found)
        return {"kind": PHOTOGRAPH, "grade": "partial" if missing else "complete", "found": found, "missing": missing}
    pages = sum(outcomes.values())
    weak = {k: v for k, v in outcomes.items() if k in ("low", "no_text", "error") and v}
    out: dict[str, Any] = {"kind": _page_kind(fmt, suffix, branches), "grade": "partial" if weak else "complete",
                           "found": ["text"], "missing": [f"{k.replace('_', ' ')} pages" for k in weak], "pages": pages,
                           "weak_pages": sum(weak.values())}
    if "## Photo details" in text:              # a picture of a page: read by the lanes, with its photo details
        out["kind"] = PHOTOGRAPHED_PAGE
        out["found"] += [x for x in photo_facts(text) if x != "description"]
    return out
