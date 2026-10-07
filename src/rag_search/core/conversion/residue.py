"""Residue of a page that has a text layer: regions with ink that the text layer does not explain (numpy + pypdfium2).

A born-digital page is read from its text layer (lane 3.2a).  What the layer cannot hold is the *residue*: a stamp, a
signature, a handwritten note, a photograph of a document pasted into the page, a drawn figure with lettering.  Lane
3.2c reads exactly those regions with the document reader and leaves the rest to docling.  ``profiler`` already finds the
big embedded pictures; this finds the regions drawn as vector graphics or small bitmaps too: ink that lies outside every
text rectangle and is not a ruled line or a table border.

One render at about 60 dpi, a mask of the text rectangles and of long straight lines taken away, then the cells of a
coarse grid that still hold ink are grouped.  Nothing here reads text.  The source file is only read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

RENDER_DPI = 60
CELL = 6                             # px of the coarse grid (0.1 inch)
CELL_INK = 0.06                      # share of ink that makes a cell part of a region
MIN_AREA = 0.04                      # a region is at least this share of the page (0.02 gave regions on 37 % of 99 text pages, 0.04 on 10 %)
MIN_SIDE = 0.10                      # ... and at least this share of the page's width and height (a stamp is not a mark)
MAX_REGIONS = 4
LINE_RUN = 0.06                      # a straight dark run this long (share of the page's side) is a rule, not residue
MARGIN = 0.01                        # regions are cut a little wider than the ink


def _runs_mask(mask: np.ndarray, n: int) -> np.ndarray:
    """True where *mask* holds an unbroken run of at least *n* True along each row (the whole run)."""
    if n < 2 or mask.shape[1] < n:
        return np.zeros_like(mask)
    c = np.concatenate((np.zeros((mask.shape[0], 1), np.int32), np.cumsum(mask, axis=1, dtype=np.int32)), axis=1)
    full = ((c[:, n:] - c[:, :-n]) == n).astype(np.int32)           # a window of n True starts at each column
    f = np.concatenate((np.zeros((mask.shape[0], 1), np.int32), np.cumsum(full, axis=1, dtype=np.int32)), axis=1)
    width = mask.shape[1]
    cols = np.arange(width)
    lo = np.clip(cols - n + 1, 0, full.shape[1])                     # windows starting in [col-n+1, col] cover the column
    hi = np.clip(cols + 1, 0, full.shape[1])
    return (f[:, hi] - f[:, lo]) > 0


def _components(grid: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    """4-connected groups of True cells as (row0, col0, row1, col1, cells)."""
    seen = np.zeros_like(grid, dtype=bool)
    h, w = grid.shape
    out = []
    for r0, c0 in zip(*np.nonzero(grid)):
        if seen[r0, c0]:
            continue
        stack, cells = [(int(r0), int(c0))], []
        seen[r0, c0] = True
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and grid[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        rs, cs = [x[0] for x in cells], [x[1] for x in cells]
        out.append((min(rs), min(cs), max(rs) + 1, max(cs) + 1, len(cells)))
    return out


def _overlap(a: list[float], b: list[float]) -> float:
    """Share of box *a* that lies inside box *b* (both left, top, right, bottom as fractions)."""
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = max(1e-9, (a[2] - a[0]) * (a[3] - a[1]))
    return (w * h) / area


def regions_of(ink: np.ndarray, text: np.ndarray, known: list[list[float]] | None = None) -> list[list[float]]:
    """The residue boxes of a page from its ink mask and the mask of its text rectangles (same shape)."""
    h, w = ink.shape
    rest = ink & ~text
    rest &= ~_runs_mask(rest, max(8, int(LINE_RUN * w)))                          # horizontal rules and table borders
    rest &= ~_runs_mask(rest.T, max(8, int(LINE_RUN * h))).T                      # vertical ones
    gh, gw = h // CELL, w // CELL
    if gh < 4 or gw < 4:
        return []
    cells = rest[:gh * CELL, :gw * CELL].reshape(gh, CELL, gw, CELL).mean(axis=(1, 3)) >= CELL_INK
    boxes: list[list[float]] = []
    for r0, c0, r1, c1, n in _components(cells):
        box = [max(0.0, c0 / gw - MARGIN), max(0.0, r0 / gh - MARGIN), min(1.0, c1 / gw + MARGIN), min(1.0, r1 / gh + MARGIN)]
        area = (box[2] - box[0]) * (box[3] - box[1])
        if area < MIN_AREA or box[2] - box[0] < MIN_SIDE or box[3] - box[1] < MIN_SIDE:
            continue
        if n / max(1, (r1 - r0) * (c1 - c0)) < 0.25:                              # a thin frame or a scatter, not a region
            continue
        if any(_overlap(box, k) > 0.6 for k in (known or [])):                    # a picture the profile has already
            continue
        boxes.append([round(x, 3) for x in box])
    boxes.sort(key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
    return boxes[:MAX_REGIONS]


def page_regions(src: Path, page: int, known: list[list[float]] | None = None) -> list[list[float]]:
    """Residue boxes of page *page* (1-based) of a PDF; ``[]`` when there are none or the page cannot be examined."""
    try:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(src))
    except Exception:  # noqa: BLE001 - no pypdfium2 / unreadable file: no regions
        return []
    try:
        pg = pdf[page - 1]
        try:
            scale = RENDER_DPI / 72.0
            im = pg.render(scale=scale).to_pil().convert("L")
            a = np.asarray(im, dtype=np.float32)
            med = float(np.median(a)) or 255.0
            ink = (np.clip(a * (255.0 / max(med, 1.0)), 0, 255) < 170)
            h, w = ink.shape
            text = np.zeros_like(ink)
            tp = pg.get_textpage()
            try:
                import pypdfium2.raw as raw

                from . import profiler

                pw, ph = pg.get_size()
                for i in range(tp.count_rects()):
                    left, bottom, right, top = tp.get_rect(i)
                    # where the text is on the page as displayed (rotation, box origin), as the render is
                    box = profiler.page_box(pg, raw, left, bottom, right, top) or [left / pw, 1 - top / ph, right / pw, 1 - bottom / ph]
                    x0, x1 = int(box[0] * w) - 1, int(box[2] * w) + 2
                    y0, y1 = int(box[1] * h) - 1, int(box[3] * h) + 2
                    text[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = True
            finally:
                tp.close()
        finally:
            pg.close()
        return regions_of(ink, text, known)
    except Exception:  # noqa: BLE001 - one bad page never stops the document
        return []
    finally:
        pdf.close()


def pages_regions(src: Path, pages: list[int], known: dict[int, list[list[float]]] | None = None) -> dict[int, list[list[float]]]:
    """``page_regions`` for several pages; pages without regions are left out."""
    out: dict[int, list[list[float]]] = {}
    for n in pages:
        boxes = page_regions(src, n, (known or {}).get(n))
        if boxes:
            out[n] = boxes
    return out


def facts(boxes: list[list[float]]) -> dict[str, Any]:
    return {"regions": len(boxes), "share": round(sum((b[2] - b[0]) * (b[3] - b[1]) for b in boxes), 3)}
