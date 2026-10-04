"""Step 1 of the conversion flow: a per-page profile of a source file, without any model.

For a PDF, every page gets: the number of text characters, whether that text reads like text,
how much of the page is covered by pictures, the resolution of the largest picture, the page
rotation and the script of the text layer.  Images are one page each (a multi-page TIFF has one
per frame).  Office and text files are one "page".  The profile costs milliseconds per page
(pypdfium2 reads the text layer; nothing is rendered).

pypdfium2 and Pillow are imported inside the functions: importing this module costs nothing.
The source file is opened for reading only.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

from ..docling_convert import dominant_script, page_text_ok
from . import router

PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".heic", ".heif"}
TEXT_EXTENSIONS = {".md", ".txt"}
DEFAULT_BUDGET_S = 120.0             # profile time limit per file; the rest is marked "unknown"
MAX_IMAGE_FRAMES = 2000
BIG_PICTURE = 0.25                   # a picture covering this share of a page is read by the document VLM
MAX_PICTURES = 4                     # ... at most this many per page


def kind_of(src: Path) -> str:
    ext = src.suffix.lower()
    if ext in PDF_EXTENSIONS:
        return "pdf"
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in TEXT_EXTENSIONS:
        return "text"
    return "office"


INK_SCALE = 0.5                      # the small render used for ink and the content hash (~36 dpi)
BLANK_INK = 0.0003                   # less ink than this: a blank page


def ink_and_hash(im: Any) -> dict[str, Any]:
    """``{"ink", "hash"}`` of a small greyscale PIL image: the share of pixels clearly darker than the
    paper (the median grey minus 40 levels), and a hash of the pixels (the page-cache identity of a
    page that has no text layer)."""
    im = im.convert("L")
    hist = im.histogram()
    total = max(1, sum(hist))
    acc, median = 0, 255
    for level, n in enumerate(hist):
        acc += n
        if acc * 2 >= total:
            median = level
            break
    cut = max(1, median - 40)
    dark = sum(hist[:cut])
    return {"ink": round(dark / total, 5),
            "hash": "r" + hashlib.sha256(im.tobytes()).hexdigest()[:31]}


def _ink_and_hash(page: Any) -> dict[str, Any]:
    try:
        return ink_and_hash(page.render(scale=INK_SCALE).to_pil())
    except Exception:  # noqa: BLE001 - no Pillow, or a page that cannot be drawn: just no ink figure
        return {}


def _page_profile(pdf: Any, i: int, raw: Any) -> dict[str, Any]:
    page = pdf[i]
    try:
        w, h = page.get_size()
        textpage = page.get_textpage()
        try:
            text = textpage.get_text_range() or ""
        finally:
            textpage.close()
        chars = len("".join(text.split()))
        cover, dpi = 0.0, 0
        pics: list[list[float]] = []
        area = max(1.0, w * h)
        try:                                               # pictures: bounding boxes and pixels
            covered = 0.0
            for obj in page.get_objects(filter=[raw.FPDF_PAGEOBJ_IMAGE], max_depth=15):
                left, b, r, t = obj.get_bounds()
                covered += max(0.0, r - left) * max(0.0, t - b)
                share = max(0.0, r - left) * max(0.0, t - b) / area
                if BIG_PICTURE <= share < router.FULL_PAGE_COVER and len(pics) < MAX_PICTURES:
                    pics.append([round(max(0.0, left / w), 3), round(max(0.0, 1 - t / h), 3),
                                 round(min(1.0, r / w), 3), round(min(1.0, 1 - b / h), 3)])
                try:
                    px = obj.get_px_size()[0]
                    dpi = max(dpi, round(px / max(0.1, (r - left) / 72.0)))
                except Exception:  # noqa: BLE001 - another pypdfium2 version
                    pass
            cover = min(1.0, covered / area)
        except Exception:  # noqa: BLE001
            cover = None                                   # unknown, not "no pictures"
        prof: dict[str, Any] = {
            "page": i + 1, "chars": chars, "text_ok": page_text_ok(text) if chars else False,
            "image_cover": None if cover is None else round(cover, 3),
            "rotation": int(page.get_rotation() or 0), "size": [round(w), round(h)],
            "script": dominant_script(text),
        }
        if dpi:
            prof["dpi"] = int(dpi)
        if pics:                                           # large pictures inside a text page
            prof["big_pics"] = pics
        prof["hidden_ocr_layer"] = bool(chars >= router.MIN_TEXT_CHARS
                                        and (cover or 0) >= router.FULL_PAGE_COVER)
        if chars < router.MIN_TEXT_CHARS:                  # a candidate for reading as an image
            prof.update(_ink_and_hash(page))
        else:                                              # text pages: what the text layer says
            prof["hash"] = hashlib.sha256(
                f"t|{round(w)}x{round(h)}|{len(text)}|{text}".encode("utf-8", "replace")).hexdigest()[:32]
        return prof
    finally:
        page.close()


def _profile_pdf(src: Path, budget_s: float) -> dict[str, Any]:
    import pypdfium2 as pdfium
    import pypdfium2.raw as raw

    pdf = pdfium.PdfDocument(str(src))
    try:
        total = len(pdf)
        pages: list[dict[str, Any]] = []
        t0 = time.perf_counter()
        truncated = False
        for i in range(total):
            if budget_s and i and time.perf_counter() - t0 > budget_s:
                truncated = True
                break
            try:
                pages.append(_page_profile(pdf, i, raw))
            except Exception as exc:  # noqa: BLE001 - one bad page must not hide the others
                pages.append({"page": i + 1, "error": f"{type(exc).__name__}: {exc}"})
        return {"pages": pages, "page_count": total, "truncated": truncated}
    finally:
        pdf.close()


def _profile_image(src: Path) -> dict[str, Any]:
    from PIL import Image

    try:
        import pillow_heif                                 # type: ignore

        pillow_heif.register_heif_opener()
    except ImportError:
        pass
    pages: list[dict[str, Any]] = []
    with Image.open(src) as im:
        frames = min(int(getattr(im, "n_frames", 1) or 1), MAX_IMAGE_FRAMES)
        total = int(getattr(im, "n_frames", 1) or 1)
        for f in range(frames):
            if f:
                im.seek(f)
            dpi = im.info.get("dpi")
            prof: dict[str, Any] = {"page": f + 1, "chars": 0, "image_cover": 1.0,
                                    "size_px": [im.width, im.height]}
            if isinstance(dpi, (tuple, list)) and dpi and dpi[0]:
                prof["dpi"] = round(float(dpi[0]))
            try:
                orient = im.getexif().get(0x0112)
                if orient and int(orient) != 1:
                    prof["exif_orientation"] = int(orient)
            except Exception:  # noqa: BLE001
                pass
            try:
                small = im.convert("L")
                small.thumbnail((400, 400))
                prof.update(ink_and_hash(small))
            except Exception:  # noqa: BLE001
                pass
            pages.append(prof)
    return {"pages": pages, "page_count": total, "truncated": total > frames}


def profile_file(src: Path, *, budget_s: float = DEFAULT_BUDGET_S) -> dict[str, Any]:
    """Profile *src*: ``{"kind", "pages": [...], "page_count", "truncated", "seconds", "error"}``.

    Never raises: a file that cannot be profiled (encrypted PDF, damaged image, no pypdfium2)
    returns ``error`` and no pages -- conversion carries on without a profile.
    """
    t0 = time.perf_counter()
    kind = kind_of(src)
    out: dict[str, Any] = {"kind": kind, "pages": [], "page_count": 0, "truncated": False,
                           "error": ""}
    try:
        if kind == "pdf":
            out.update(_profile_pdf(src, budget_s))
        elif kind == "image":
            out.update(_profile_image(src))
        else:
            out.update({"pages": [{"page": 1}], "page_count": 1})
    except ImportError as exc:
        out["error"] = f"profiler unavailable: {exc}"
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    out["seconds"] = round(time.perf_counter() - t0, 3)
    return out


def route_pages(profile: dict[str, Any]) -> list[tuple[int, str, str]]:
    """``(page, branch, why)`` for every page of a profile, as ``router.decide`` sees it.  Pages
    beyond a truncated profile are ``unknown``."""
    kind = profile.get("kind", "office")
    out: list[tuple[int, str, str]] = []
    for p in profile.get("pages", []):
        if p.get("error"):
            out.append((p["page"], "unknown", f"profile failed: {p['error']}"))
            continue
        branch, why = router.decide(kind, p)
        out.append((int(p.get("page", len(out) + 1)), branch, why))
    for n in range(len(out) + 1, int(profile.get("page_count") or 0) + 1):
        out.append((n, "unknown", "not profiled: time limit reached"))
    return out
