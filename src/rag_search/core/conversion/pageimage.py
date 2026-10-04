"""Render one source page as a small PNG, for the dashboard's document drawer.

Only PDF pages and image files can be shown; the source is opened read-only.  pypdfium2 and Pillow
are imported inside ``render`` (the dashboard process stays light until a page is asked for).
"""

from __future__ import annotations

import io
from pathlib import Path

from . import profiler

MAX_PX = 1400


def render(src: Path, page: int, width_px: int = 900) -> bytes:
    """PNG bytes of page *page* (1-based) of *src*, about *width_px* wide.  Raises ValueError for a
    file kind that has no pages to show or a page out of range, ImportError without the libraries."""
    width_px = max(100, min(int(width_px or 900), MAX_PX))
    kind = profiler.kind_of(src)
    buf = io.BytesIO()
    if kind == "pdf":
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(src))
        try:
            if not 1 <= page <= len(pdf):
                raise ValueError(f"page {page} is outside this document (1-{len(pdf)})")
            pg = pdf[page - 1]
            try:
                scale = width_px / max(1.0, pg.get_size()[0])
                pg.render(scale=scale).to_pil().convert("RGB").save(buf, "PNG")
            finally:
                pg.close()
        finally:
            pdf.close()
    elif kind == "image":
        from PIL import Image

        with Image.open(src) as im:
            frames = int(getattr(im, "n_frames", 1) or 1)
            if not 1 <= page <= frames:
                raise ValueError(f"page {page} is outside this image (1-{frames})")
            im.seek(page - 1)
            frame = im.convert("RGB")
        frame.thumbnail((width_px, width_px * 3))
        frame.save(buf, "PNG")
    else:
        raise ValueError("only PDF pages and image files can be shown")
    return buf.getvalue()
