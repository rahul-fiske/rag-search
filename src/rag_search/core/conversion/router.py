"""Which branch of the flow chart a page takes, and why (stdlib only).

Phase P0 only *observes*: every page is still read by docling, and the branch is the decision the
profile implies.  The same function drives the real routing from P2 on, so the numbers collected
now are the baseline for it.  Every decision returns its reason, which is stored in the trace.
"""

from __future__ import annotations

from typing import Any

MIN_TEXT_CHARS = 40                  # fewer characters than this is "no usable text" (as in docling_convert)
SCAN_COVER = 0.30                    # a picture covering this much of the page is a scan candidate
FULL_PAGE_COVER = 0.85               # ... and this much is a full-page picture


def decide(kind: str, page: dict[str, Any] | None = None) -> tuple[str, str]:
    """(branch, reason) for a page of a file of *kind* (pdf | image | office | text)."""
    if kind == "text":
        return "copy", "Markdown or plain text, used as it is"
    if kind == "office":
        return "office", "Office or HTML file: docling reads its structure"
    if kind == "image":
        return "image", "image file: no text layer, read as one page"
    if kind != "pdf" or not page:
        return "unknown", "not profiled"
    chars = int(page.get("chars") or 0)
    cover = page.get("image_cover")
    pct = f"{round(100 * cover)} % picture" if isinstance(cover, (int, float)) else "picture coverage unknown"
    if chars < MIN_TEXT_CHARS and (cover is None or cover >= SCAN_COVER or chars == 0 and cover > 0.02):
        return "raster", f"no usable text layer ({chars} characters, {pct})"
    if chars < MIN_TEXT_CHARS:
        return "digital", f"short page ({chars} characters, {pct})"
    if page.get("text_ok") is False:
        return "raster", f"the text layer looks garbled ({chars} characters)"
    if page.get("hidden_ocr_layer"):
        return "digital", (f"text layer ({chars} characters) over a full-page picture: probably a "
                           "scanner's hidden OCR layer; kept as digital until it is compared with a read")
    return "digital", f"good text layer ({chars} characters)"
