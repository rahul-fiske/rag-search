"""Apple Vision as the last reader for a page nobody else could read (macOS, ``ocrmac``).

docling's OCR cannot read a *photographed* page: its layout model calls the whole photo a picture, and the
text found inside a picture is dropped from the result.  The document reader (a vision model) is the proper
tool for such pages; when it cannot run (not installed, not downloaded, out of memory) or finds nothing,
this module reads the page image directly with Apple's text recognition and returns plain text in reading
order.  It is a rescue, not a layout reader: tables come back as lines of text, never as Markdown tables, and
the page record says which reader produced the text (``apple-vision``).  Everything is stdlib at import;
``ocrmac`` and the page renderer are imported when a page is read.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path
from typing import Any

ID = "apple-vision"
LONG_SIDE = 2600                     # px: enough for dot-matrix and small print, still quick


def why_not() -> str:
    """"" when Apple Vision can read pages here, otherwise the reason."""
    if sys.platform != "darwin":
        return "Apple Vision needs a Mac"
    if importlib.util.find_spec("ocrmac") is None:
        return "ocrmac is not installed (Models tab: install the document reader runtime)"
    return ""


def usable() -> bool:
    return not why_not()


def lines_to_text(items: list[tuple[str, float, tuple[float, float, float, float]]]) -> str:
    """Recognised phrases ``(text, confidence, (x, y, w, h))`` -- boxes as fractions of the image with the
    origin at the *bottom* left, which is what Vision returns -- to text in reading order: phrases whose
    vertical centres are within half a line of each other form one line, left to right."""
    rows: list[dict[str, Any]] = []
    for text, _conf, box in sorted(items, key=lambda it: -(float(it[2][1]) + float(it[2][3]) / 2)):
        text = (text or "").strip()
        if not text:
            continue
        x, y, w, h = (float(v) for v in box)
        cy = y + h / 2
        for row in rows:
            if abs(row["cy"] - cy) <= max(row["h"], h) * 0.5:
                row["parts"].append((x, text))
                row["h"] = max(row["h"], h)
                break
        else:
            rows.append({"cy": cy, "h": h, "parts": [(x, text)]})
    rows.sort(key=lambda r: -r["cy"])
    return "\n\n".join("  ".join(t for _x, t in sorted(r["parts"])) for r in rows)


def read_image(image: Path, languages: list[str] | None = None) -> str:
    """Text of one image file, in reading order (``ocrmac`` imported here)."""
    from ocrmac import ocrmac                                   # type: ignore

    kw: dict[str, Any] = {"recognition_level": "accurate"}
    if languages:
        kw["language_preference"] = languages
    return lines_to_text(list(ocrmac.OCR(str(image), **kw).recognize()))


def read_page(src: Path, page: int, *, is_image: bool = False, languages: list[str] | None = None) -> str:
    """Text of page *page* (1-based) of a PDF, or of frame *page* of an image file."""
    from . import vlm

    with tempfile.TemporaryDirectory(prefix="rag-search-av-") as tmp:
        img = Path(tmp) / "page.png"
        if is_image:
            from PIL import Image

            with Image.open(src) as im:
                im.seek(page - 1)
                frame = im.convert("RGB")
            frame.save(img, "PNG")
        else:
            vlm.render_pdf_page(src, page, img, long_side=LONG_SIDE)
        return read_image(img, languages)
