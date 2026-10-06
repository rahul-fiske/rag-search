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


# ── routing v2: which runway reads a page that has no text layer ─────────────────────────────

# Versioned and in one place so the routing harness can sweep them: the page-image facts (scanfacts.py) that make a scan
# "clean print" -- sure enough for docling's conventional OCR (runway 3.2b) -- and everything else, which is in doubt
# and goes to the document reader (3.2d).  Wrong guesses cost one cheap OCR pass: the gate sends a failed 3.2b page on.
THRESHOLDS = {
    "version": 2,
    "min_dpi": 150,                  # effective resolution of the page's image
    "min_contrast": 0.60,            # (p98 - p2) of the paper-normalised grey levels
    "min_sharp": 1.0,                # strong strokes per unit of ink: a blurred page has almost none
    "max_skew": 2.0,                 # degrees; docling's OCR does not straighten a page
    "max_deskew": 8.0,               # a page skewed up to this is straightened and read by Tesseract (needs it installed)
    "max_speckle": 0.003,            # isolated dark pixels per ink pixel
    "max_bg_std": 20.0,              # texture of the paper: a photograph is not flat
    "min_text_lines": 8,             # regular short lines of ink: prose, not a drawing
    "max_h_rules": 2, "max_v_rules": 1,      # ruled lines: a table, whose structure OCR alone does not give
    "min_ink": 0.004, "max_ink": 0.30,
}


def decide_scan(facts: dict[str, Any] | None, profile: dict[str, Any] | None = None,
                capabilities: dict[str, Any] | None = None) -> tuple[str, list[str]]:
    """(runway, reasons) for a page without a text layer: ``b`` (docling + OCR) when the page-image *facts* say clean
    print and an OCR engine is available, otherwise ``d`` (the document reader).  *reasons* names every condition that
    sent the page to ``d``, or says why ``b`` is likely to work."""
    t = THRESHOLDS
    caps = capabilities or {}
    why: list[str] = []
    if not caps.get("ocr", True):
        why.append("no OCR engine available")
    if not facts:
        why.append("the page image could not be examined")
        return "d", why
    dpi = (profile or {}).get("dpi")
    if not isinstance(dpi, (int, float)) or dpi <= 0:
        why.append("resolution unknown")
    elif dpi < t["min_dpi"]:
        why.append(f"only {round(dpi)} dpi")
    checks = (
        (facts["contrast"] < t["min_contrast"], f"low contrast ({facts['contrast']})"),
        (facts["sharp"] < t["min_sharp"], f"soft strokes ({facts['sharp']})"),
        (abs(facts["skew"]) > (t["max_deskew"] if caps.get("deskew") else t["max_skew"]), f"skewed {facts['skew']} degrees"),
        (facts["speckle"] > t["max_speckle"], f"speckled ({facts['speckle']})"),
        (facts["bg_std"] > t["max_bg_std"], f"textured paper ({facts['bg_std']})"),
        (facts["text_lines"] < t["min_text_lines"], f"only {facts['text_lines']} text lines"),
        (facts["h_rules"] > t["max_h_rules"] or facts["v_rules"] > t["max_v_rules"],
         f"ruled lines ({facts['h_rules']} horizontal, {facts['v_rules']} vertical): a table"),
        (not t["min_ink"] <= facts["ink"] <= t["max_ink"], f"ink share {round(100 * facts['ink'], 1)} %"),
    )
    why += [msg for bad, msg in checks if bad]
    if why:
        return "d", why
    return "b", [f"clean print: {round(dpi)} dpi, contrast {facts['contrast']}, {facts['text_lines']} text lines, "
                 f"skew {facts['skew']}"]


def b_engine(facts: dict[str, Any] | None, capabilities: dict[str, Any] | None = None) -> str:
    """The engine that reads a page of lane b: ``docling`` (its own OCR, for a page that is straight and is a PDF page) or
    ``tesseract`` (a page that is skewed and has to be straightened first, and every image file: docling reads PDF pages
    by number)."""
    caps = capabilities or {}
    skew = abs((facts or {}).get("skew") or 0.0)
    if caps.get("deskew") and (skew > THRESHOLDS["max_skew"] or not caps.get("docling_pages", True)):
        return "tesseract"
    return "docling"


def decide_digital(page: dict[str, Any] | None, regions: list[list[float]] | None = None,
                   capabilities: dict[str, Any] | None = None) -> tuple[str, list[str]]:
    """(lane, reasons) for a page that has a text layer: ``a`` (docling reads the layer) or ``c`` (docling reads the layer
    and the document reader reads the pictures and the regions the layer does not explain).  *regions* are the residue
    boxes; the profile's big pictures count too.  Without a document reader a page is ``a``."""
    pics = list((page or {}).get("big_pics") or [])
    extra = list(regions or [])
    if not (capabilities or {}).get("vlm", True):
        return "a", ["text layer" + (" (no document reader for its pictures)" if pics or extra else "")]
    if not pics and not extra:
        return "a", ["text layer, nothing that the layer does not explain"]
    why = []
    if pics:
        why.append(f"{len(pics)} large picture(s)")
    if extra:
        why.append(f"{len(extra)} region(s) of ink outside the text layer")
    return "c", why
