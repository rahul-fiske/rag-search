"""Page-image facts of a page that has no text layer (numpy + Pillow; the profile's layer 3.1B).

One render of the page at about 100 dpi gives what decides whether a conventional OCR engine is likely to read it well,
or the page is in doubt and belongs to the document reader (a vision model): how much ink, how much contrast, how sharp
the strokes are, how skewed the text lines are, how regular the lines are, whether long ruled lines make it a table,
whether it looks like a photograph.  Nothing here reads text; it costs about 50-150 ms a page.  ``router.decide_scan``
turns the facts into a runway.  The source file is only read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

RENDER_DPI = 100
SKEW_RANGE = 6.0                     # degrees searched on each side
SKEW_STEP = 0.5


def _gray(im: Any) -> np.ndarray:
    return np.asarray(im.convert("L"), dtype=np.uint8)


def _normalise(a: np.ndarray) -> np.ndarray:
    """Paper (the median grey) mapped to 255, so a yellowed or grey page compares with a white one."""
    median = float(np.median(a)) or 255.0
    return np.clip(a.astype(np.float32) * (255.0 / max(median, 1.0)), 0, 255)


def _skew(binary: Any) -> float:
    """The angle (degrees, positive = counter-clockwise rotation needed) at which the row profile of the ink is
    sharpest: text lines are horizontal then."""
    from PIL import Image

    small = Image.fromarray((np.asarray(binary) * 255).astype(np.uint8)).resize((400, round(400 * binary.shape[0] / binary.shape[1])))
    best, best_angle = -1.0, 0.0
    angle = -SKEW_RANGE
    while angle <= SKEW_RANGE + 1e-9:
        rot = np.asarray(small.rotate(angle, resample=Image.NEAREST, fillcolor=0)) > 127
        score = float(np.var(rot.sum(axis=1)))
        if score > best:
            best, best_angle = score, angle
        angle += SKEW_STEP
    return round(-best_angle, 2)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, length) of the runs of True in a 1-D boolean array."""
    if not mask.any():
        return []
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    return [(int(s), int(e - s)) for s, e in zip(starts, ends)]


def _long_run(mask: np.ndarray, share: float) -> np.ndarray:
    """Rows of *mask* that hold an unbroken run of True at least *share* of the row long."""
    w = mask.shape[1]
    n = max(2, int(share * w))
    c = np.concatenate((np.zeros((mask.shape[0], 1), np.int32), np.cumsum(mask, axis=1, dtype=np.int32)), axis=1)
    return ((c[:, n:] - c[:, :-n]) == n).any(axis=1)


def facts_of_image(im: Any, *, dpi: float = RENDER_DPI) -> dict[str, Any]:
    """The facts of a PIL page image rendered at *dpi*."""
    a = _gray(im)
    norm = _normalise(a)
    ink_mask = norm < 170
    ink = float(ink_mask.mean())
    p2, p98 = np.percentile(norm, (2, 98))
    contrast = float((p98 - p2) / 255.0)
    lap = (4 * norm[1:-1, 1:-1] - norm[:-2, 1:-1] - norm[2:, 1:-1] - norm[1:-1, :-2] - norm[1:-1, 2:])
    edge_pix = np.abs(lap) > 60                       # strokes: a strong second difference
    sharp = float(edge_pix.mean() / max(ink, 1e-6)) if ink > 0 else 0.0
    mid = float(((norm > 60) & (norm < 200)).mean())
    h, w = ink_mask.shape
    rows = ink_mask.mean(axis=1)
    h_lines = len(_runs(_long_run(ink_mask, 0.25)))               # ruled lines: an unbroken dark run of a quarter page
    v_lines = len(_runs(_long_run(ink_mask.T, 0.18)))
    line_runs = _runs(rows > 0.004)
    paper = norm[~ink_mask]
    bg_std = float(paper.std()) if paper.size else 0.0            # a clean page's paper is flat: texture means a photo
    nb = sum(np.roll(np.roll(ink_mask, dy, 0), dx, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)
    speckle = float((ink_mask & (nb == 0)).sum() / max(1, ink_mask.sum()))      # isolated dark pixels: scanner noise
    lengths = np.array([n for _s, n in line_runs], dtype=np.float32)
    # text lines are many, short and alike; a photograph or a drawing is a few big ones
    n_lines = int(np.sum((lengths >= 3) & (lengths <= 0.06 * h))) if len(lengths) else 0
    skew = _skew(ink_mask) if ink > 0.002 else 0.0
    return {"ink": round(ink, 4), "contrast": round(contrast, 3), "sharp": round(sharp, 3), "mid_tones": round(mid, 3),
            "skew": skew, "text_lines": n_lines, "h_rules": h_lines, "v_rules": v_lines, "bg_std": round(bg_std, 2),
            "speckle": round(speckle, 4), "render_dpi": dpi,
            "px": [w, h]}


def page_facts(src: Path, page: int, *, dpi: float = RENDER_DPI) -> dict[str, Any] | None:
    """The facts of page *page* (1-based) of a PDF, or None when it cannot be rendered."""
    try:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(src))
    except Exception:  # noqa: BLE001 - no pypdfium2 / unreadable file: no facts
        return None
    try:
        pg = pdf[page - 1]
        try:
            im = pg.render(scale=dpi / 72.0).to_pil()
        finally:
            pg.close()
        return facts_of_image(im, dpi=dpi)
    except Exception:  # noqa: BLE001
        return None
    finally:
        pdf.close()


def pages_facts(src: Path, pages: list[int], *, dpi: float = RENDER_DPI) -> dict[int, dict[str, Any] | None]:
    """``page_facts`` for several pages of one PDF (it is opened once); a page that cannot be examined maps to None."""
    out: dict[int, dict[str, Any] | None] = {n: None for n in pages}
    try:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(src))
    except Exception:  # noqa: BLE001
        return out
    try:
        for n in pages:
            try:
                pg = pdf[n - 1]
                try:
                    im = pg.render(scale=dpi / 72.0).to_pil()
                finally:
                    pg.close()
                out[n] = facts_of_image(im, dpi=dpi)
            except Exception:  # noqa: BLE001 - one bad page does not stop the others
                continue
    finally:
        pdf.close()
    return out


def straighten(im: Any, skew: float) -> Any:
    """The PIL page image *im* turned so that its text lines are horizontal.  *skew* is the page's measured angle; the
    direction is not trusted to a sign convention but tried both ways on a small copy, and the one that makes the ink
    rows sharper is used.  A page with no measurable skew is returned as it is."""
    from PIL import Image

    if abs(skew) < 0.3:
        return im
    gray = im.convert("L")
    small = gray.resize((400, max(1, round(400 * gray.height / max(1, gray.width)))))
    norm = _normalise(np.asarray(small, dtype=np.uint8))

    def sharp(angle: float) -> float:
        rot = np.asarray(Image.fromarray((norm < 170).astype(np.uint8) * 255).rotate(angle, resample=Image.NEAREST, fillcolor=0)) > 127
        return float(np.var(rot.sum(axis=1)))

    angle = abs(skew) if sharp(abs(skew)) >= sharp(-abs(skew)) else -abs(skew)
    return im.convert("RGB").rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))


def image_facts(src: Path, frame: int, *, dpi: float = RENDER_DPI) -> dict[str, Any] | None:
    """The facts of frame *frame* (1-based) of an image file, rendered with its long side at an A4 page's length at *dpi*
    (an image file's own dpi tag is rarely a scan's: 72 or 96 by default).  None when it cannot be examined."""
    import tempfile

    from . import vlm

    try:
        with tempfile.TemporaryDirectory(prefix="rag-search-facts-") as tmp:
            png = Path(tmp) / "f.png"
            vlm.render_image_frame(src, frame, png, long_side=int(11.7 * dpi))
            from PIL import Image

            with Image.open(png) as im:
                im.load()
                return facts_of_image(im.copy(), dpi=dpi)
    except Exception:  # noqa: BLE001
        return None
