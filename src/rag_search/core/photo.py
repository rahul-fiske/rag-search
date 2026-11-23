"""What a photograph's file says about itself, as a short section of its Markdown.

Every image gets one when it carries camera metadata (EXIF: when it was taken, where, with what).  The section is
facts only -- nothing is looked at or guessed from the pixels -- and reads like this::

    ## Photo details

    - Taken: 2024-03-14 13:12:05 +05:30 (Thursday)
    - Position: 19.09050 N, 78.34120 E, altitude 312 m (geo:19.09050,78.34120)
    - Area: Nirmal, Telangana, India
    - Town: near Nirmal, Telangana, India (about 4 km NE of it)
    - Nearby: Hotel Sai Krishna (restaurant, 25 m); HP Petrol Pump (gas station, 60 m)
    - Camera: Apple iPhone 13 Pro
    - Size: 4032 x 3024 pixels
    - Reference data: geonames@2026-11-20, overture-divisions@2026-11-18.0, overture-places@2026-11-18.0

The times are written as an ISO date and time with the offset the camera recorded, so ``rag_grep 2024-03-14`` finds a
day; the position is also written as ``geo:LAT,LON``, which is how ``maps.plan`` finds the parts of the map the
documents need.  What else is known about a photograph comes from *annotators*: functions that take the metadata and
return lines (and the reference data they used).  ``_where`` (``geo.py``: area, town, places around) is the first; a
face or object lookup is added to ``ANNOTATORS`` the same way.  The section is kept apart from the text read from the picture, so it can be replaced when the data
behind it changes, without reading the picture again.
"""

from __future__ import annotations

import contextlib
import os
import re
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from ..paths import META_FILE, Paths, read_json
from . import geo

HEADING = "## Photo details"
_PAGE_ONE = "<!-- page 1 -->"
_SEEN = "Text in the picture:"
# lines that something other than the file's own metadata wrote: kept when the section is renewed (a map refresh
# rewrites the place lines and nothing else); ``describe.py`` writes the model's lines and the face count
_KEEP = (_SEEN, "Looks like (model-written):", "Things seen (model-written):", "People in the picture (model count):",
         "Faces detected:", "Described by:")
# The section is a list of "- label: value" lines.  An earlier version wrote the lines of a description after a blank line
# (about a quarter of the photographs of a collection), which cut the section short here and made the photograph look like a
# photographed page (no faces, no description catch-up): a blank line before one of these labels belongs to the section too.
_BLOCK_RE = re.compile(r"^## Photo details\n\n(?:- .*\n?|\n+(?=- (?:" + "|".join(re.escape(k) for k in _KEEP) + r")))+\n?", re.M)

# EXIF tags (base IFD, Exif IFD 0x8769, GPS IFD 0x8825)
_MAKE, _MODEL, _DATETIME, _ORIENT = 0x010F, 0x0110, 0x0132, 0x0112
_TAKEN, _OFFSET, _LENS = 0x9003, 0x9011, 0xA434
_EXIF_IFD, _GPS_IFD = 0x8769, 0x8825


def _num(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _degrees(value: Any, ref: Any) -> float | None:
    try:
        d, m, s = (float(x) for x in value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    deg = d + m / 60 + s / 3600
    return -deg if str(ref).strip().upper() in ("S", "W") else deg


def _clean(v: Any) -> str:
    return " ".join(str(v).replace("\x00", " ").split()) if v is not None else ""


def register_heif() -> bool:
    """Teach Pillow to open iPhone photos (.heic/.heif) when pillow-heif is installed; says whether it is."""
    try:
        import pillow_heif                                    # type: ignore

        pillow_heif.register_heif_opener()
        return True
    except Exception:  # noqa: BLE001 - not installed (or unusable): other formats still open
        return False


def open_image(src: Path) -> Any:
    """The image file opened with Pillow, HEIC included (the caller closes it, or uses ``with``)."""
    from PIL import Image

    register_heif()
    return Image.open(src)


MIN_PX, MAX_PX = 256, 2048
MIN_QUALITY, MAX_QUALITY = 50, 95


def render_jpeg(src: Path, max_px: int = 1400, quality: int = 85) -> tuple[bytes, tuple[int, int]]:
    """The photograph as upright JPEG bytes no larger than *max_px* (256-2048) on its long side, and its size.
    Raises OSError when the file cannot be decoded (pillow-heif missing for a .heic, a damaged file)."""
    import io

    from PIL import ImageOps

    max_px = min(max(int(max_px), MIN_PX), MAX_PX)
    quality = min(max(int(quality), MIN_QUALITY), MAX_QUALITY)
    try:
        with open_image(src) as im:
            if im.format == "JPEG":
                im.draft("RGB", (max_px * 2, max_px * 2))     # decoded at a fraction of the size: quicker
            im = ImageOps.exif_transpose(im).convert("RGB")
    except Exception as exc:  # noqa: BLE001 - PIL raises many types for a file it cannot read
        hint = "" if src.suffix.lower() not in (".heic", ".heif") or register_heif() else " (pillow-heif is not installed)"
        raise OSError(f"{type(exc).__name__}: {exc}{hint}") from exc
    im.thumbnail((max_px, max_px))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True)
    return buf.getvalue(), im.size


def read_exif(src: Path) -> dict[str, Any]:
    """The metadata of an image file as a plain dict (empty when there is none or it cannot be read); the file is
    only read."""
    try:
        from PIL import Image
    except ImportError:
        return {}
    try:
        with open_image(src) as im:
            size = im.size
            exif = im.getexif()
            base = dict(exif)
            ifd = dict(exif.get_ifd(_EXIF_IFD)) if exif else {}
            gps = dict(exif.get_ifd(_GPS_IFD)) if exif else {}
    except Exception:  # noqa: BLE001 - an unreadable image has no metadata; the readers report it
        return {}
    meta: dict[str, Any] = {}
    taken = _clean(ifd.get(_TAKEN) or base.get(_DATETIME))
    if taken:
        try:
            when = datetime.strptime(taken, "%Y:%m:%d %H:%M:%S")
            if when.year > 1900:                          # 0000:00:00 is how some cameras say "unknown"
                meta["when"] = when
                offset = _clean(ifd.get(_OFFSET))
                if re.fullmatch(r"[+-]\d\d:\d\d", offset):
                    meta["offset"] = offset
        except ValueError:
            pass
    lat = _degrees(gps.get(2), gps.get(1))
    lon = _degrees(gps.get(4), gps.get(3))
    if lat is not None and lon is not None and not (lat == 0 and lon == 0):
        meta["lat"], meta["lon"] = lat, lon
        alt = _num(gps.get(6))
        if alt is not None:
            meta["altitude"] = -alt if gps.get(5) in (1, b"\x01") else alt
        heading = _num(gps.get(17))
        if heading is not None:
            meta["heading"] = heading
    make, model = _clean(base.get(_MAKE)), _clean(base.get(_MODEL))
    if make or model:
        meta["camera"] = model if make and model.lower().startswith(make.lower()) else f"{make} {model}".strip()
    lens = _clean(ifd.get(_LENS))
    if lens:
        meta["lens"] = lens
    if meta:                                              # a size alone is not metadata worth a section
        meta["size"] = size
    return meta


def recorded_photo(idx_dir: Path) -> bool | None:
    """Whether the last conversion of a file recorded it as a photograph (its page branches in ``index.meta.json`` are
    ``photo`` alone): True, False for another kind, None when nothing is recorded (not indexed yet).  The one answer to "is this
    a photograph?" for every step that works on photographs after the conversion -- faces, descriptions, picture vectors, the
    coverage report -- so a Markdown that looks odd (an old version wrote the description after a blank line, or without its
    heading) cannot make a photograph look like a page."""
    branches = ((read_json(idx_dir / META_FILE) or {}).get("conversion") or {}).get("branches") or {}
    return set(branches) == {"photo"} if branches else None


Annotator = Callable[[dict[str, Any], Paths], "tuple[list[str], str]"]


LAST: dict[str, float] = {}           # seconds the last Locate took, for the stage events of the indexer


def _where(meta: dict[str, Any], paths: Paths) -> tuple[list[str], str]:
    """Area, town and the places around (``geo.py``, from the map data that is installed)."""
    LAST["locate_s"] = 0.0
    if "lat" not in meta:
        return [], ""
    t0 = time.perf_counter()
    lines, refs = geo.describe(meta["lat"], meta["lon"], paths)
    LAST["locate_s"] = round(time.perf_counter() - t0, 3)
    return lines, ", ".join(refs)


ANNOTATORS: list[Annotator] = [_where]


DOCUMENT_CHARS = 200             # this many letters and digits in the picture: it is a page, not a photograph
TEXT_LINE_CHARS = 400            # most of the text of a photograph that is kept in its section
MIN_LETTERS = 4                  # fewer letters than this found in a picture are not worth a line


def visible_text(src: Path) -> str | None:
    """The text a quick OCR (Apple Vision, else Tesseract) finds in the picture; None when this machine has neither,
    so the caller cannot tell a photograph from a page."""
    from .conversion import applevision, tesseract

    try:
        if applevision.usable():
            return applevision.read_page(src, 1, is_image=True)
        if tesseract.usable():
            return tesseract.read_page(src, 1, is_image=True)
    except Exception:  # noqa: BLE001 - no answer is not "no text"
        return None
    return None


def amount(text: str) -> int:
    return sum(ch.isalnum() for ch in text)


def is_photograph(text: str | None) -> bool:
    """A picture with little text is a photograph: the document reader, made for pages, would only make up text for it."""
    return text is not None and amount(text) < DOCUMENT_CHARS


def lines(meta: dict[str, Any], paths: Paths, text: str = "", extra: Sequence[str] = ()) -> list[str]:
    out: list[str] = []
    if "when" in meta:
        when = meta["when"]
        stamp = when.strftime("%Y-%m-%d %H:%M:%S") + (f" {meta['offset']}" if "offset" in meta else
                                                      " (time zone not recorded)")
        out.append(f"Taken: {stamp} ({when.strftime('%A')})")
    if "lat" in meta:
        pos = (f"{abs(meta['lat']):.5f} {'N' if meta['lat'] >= 0 else 'S'}, "
               f"{abs(meta['lon']):.5f} {'E' if meta['lon'] >= 0 else 'W'}")
        if "altitude" in meta:
            pos += f", altitude {meta['altitude']:.0f} m"
        out.append(f"Position: {pos} (geo:{meta['lat']:.5f},{meta['lon']:.5f})")
    refs: list[str] = []
    for annotate in ANNOTATORS:
        try:
            more, ref = annotate(meta, paths)
        except Exception:  # noqa: BLE001 - a lookup that fails must not lose the photo
            continue
        out.extend(more)
        if ref:
            refs.append(ref)
    if "camera" in meta:
        out.append(f"Camera: {meta['camera']}" + (f", {meta['lens']}" if "lens" in meta else ""))
    if "size" in meta:
        out.append(f"Size: {meta['size'][0]} x {meta['size'][1]} pixels")
    text = " ".join(text.split())
    if sum(ch.isalpha() for ch in text) >= MIN_LETTERS:                # "2" or "2 102 30" is not text
        out.append(_SEEN + " " + (text if len(text) <= TEXT_LINE_CHARS else text[:TEXT_LINE_CHARS].rstrip() + " ..."))
    out.extend(extra)
    if refs:
        out.append(f"Reference data: {', '.join(refs)}")
    return out


def block(src: Path, paths: Paths, text: str = "", extra: Sequence[str] = ()) -> str:
    """The "Photo details" section for *src* (with the short *text* found in it, when given), or "" when the file has no
    camera metadata."""
    meta = read_exif(src)
    if not meta:
        return ""
    return HEADING + "\n\n" + "\n".join(f"- {x}" for x in lines(meta, paths, text, extra)) + "\n"


def _carry(old: str, new: str) -> str:
    """*new* (a section) with the ``_KEEP`` lines of *old* that it lacks, just before its Reference data line."""
    have = [ln for ln in new.splitlines() if ln.startswith("- ")]
    kept = [ln for ln in old.splitlines() if ln.startswith("- ")
            and any(ln.startswith(f"- {k}") for k in _KEEP)
            and not any(h.startswith(ln[:ln.index(":") + 1]) for h in have)]
    if not kept:
        return new
    head, sep, tail = new.rstrip("\n").rpartition("\n- Reference data")
    return head + "\n" + "\n".join(kept) + (sep + tail if sep else "") + "\n"


def headed(section: str, src: Path) -> str:
    """A Photo details section for a picture with no camera data, from the lines the faces and describe steps added: a heading, the
    picture's size (a fact of the file) and those lines, so the section is a section and not loose lines."""
    lines = [ln for ln in section.splitlines() if ln.startswith("- ")]
    if not lines:
        return ""
    try:
        with open_image(src) as im:
            lines.insert(0, f"- Size: {im.size[0]} x {im.size[1]} pixels")
    except Exception:  # noqa: BLE001 - the size is a courtesy
        pass
    return HEADING + "\n\n" + "\n".join(lines) + "\n"


def insert_extra(section: str, extra: Sequence[str]) -> str:
    """*section* with the lines of *extra* just before its Reference data line (at the end without one)."""
    if not extra:
        return section
    head, sep, tail = section.rstrip("\n").rpartition("\n- Reference data")
    body = "\n".join(f"- {x}" for x in extra)
    return (head + "\n" + body + sep + tail if sep else section.rstrip("\n") + "\n" + body) + "\n"


def set_extra(md_path: Path, extra: Sequence[str]) -> bool:
    """Put the lines of *extra* (each starts with one of the ``_KEEP`` labels) into the Photo details section of an
    existing Markdown file, replacing the lines that have the same label; nothing else in the file changes.  True when
    the file changed."""
    text = md_path.read_text(encoding="utf-8", errors="replace")
    m = _BLOCK_RE.search(text)
    if not m or not extra:
        return False
    labels = [x[:x.index(":") + 1] for x in extra if ":" in x]
    old = [ln for ln in m.group(0).splitlines() if ln.startswith("- ")]
    keep = [ln for ln in old if not any(ln.startswith(f"- {lab}") for lab in labels)]
    at = next((i for i, ln in enumerate(keep) if ln.startswith("- Reference data")), len(keep))
    new_lines = keep[:at] + [f"- {x}" for x in extra] + keep[at:]
    section = HEADING + "\n\n" + "\n".join(new_lines) + "\n"
    new = text[:m.start()] + section + ("\n" if m.group(0).endswith("\n\n") else "") + text[m.end():]
    if new == text:
        return False
    tmp = md_path.with_name(md_path.name + ".partial")
    tmp.write_text(new, encoding="utf-8")
    os.replace(tmp, md_path)
    return True


def attach(md_path: Path, section: str) -> bool:
    """Put *section* into the Markdown (under page 1), replacing the one that is there.  True when the file changed."""
    text = md_path.read_text(encoding="utf-8", errors="replace")
    old = _BLOCK_RE.search(text) if HEADING in text else None
    if old:                                          # what others wrote is kept when only the rest is renewed
        section = _carry(old.group(0), section)
    new = _BLOCK_RE.sub("", text).lstrip("\n") if old else text
    if new.startswith(_PAGE_ONE):
        body = new[len(_PAGE_ONE):].lstrip("\n")
        new = f"{_PAGE_ONE}\n\n{section}\n{body}" if body else f"{_PAGE_ONE}\n\n{section}"
    else:
        new = f"{_PAGE_ONE}\n\n{section}\n{new}" if new.strip() else f"{_PAGE_ONE}\n\n{section}"
    if new == text:
        return False
    tmp = md_path.with_name(md_path.name + ".partial")
    tmp.write_text(new, encoding="utf-8")
    os.replace(tmp, md_path)
    return True


def only(md_path: Path, section: str) -> None:
    """The Markdown of a picture that holds no text: its photo details alone."""
    md_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = md_path.with_name(md_path.name + ".partial")
    tmp.write_text(f"{_PAGE_ONE}\n\n{section}", encoding="utf-8")
    os.replace(tmp, md_path)
