"""3.11 Video details: what a video file says about itself, as a short section of its Markdown.

A video is indexed like a photograph is, from facts its file records (QuickTime / MP4 atoms), never from its frames or its
sound (yet)::

    ## Video details

    - Taken: 2024-03-14 13:12:05 +05:30 (Thursday)
    - Position: 19.09050 N, 78.34120 E, altitude 312 m (geo:19.09050,78.34120)
    - Area: Nirmal, Telangana, India
    - Town: near Nirmal, Telangana, India (about 4 km NE of it)
    - Camera: Apple iPhone 13 Pro
    - Size: 1080 x 1920 pixels
    - Duration: 1 min 4 s
    - Reference data: geonames@2026-11-20, overture-divisions@2026-11-18.0

The lines are the photograph's (``photo.lines``: the same labels, the same 3.7 Locate annotators), so ``rag_grep
2024-03-14`` and a search for a place find videos and photographs alike.  The time is the phone's own
``com.apple.quicktime.creationdate`` (local time with its offset) when there is one, else ``©day``, else the movie header's
creation time, which the format defines as UTC.  A file that is not a QuickTime / MP4 file (an .avi) gets its size and
duration from the decoder (OpenCV), and nothing else.

A Live Photo's video (an iPhone ``IMG_1234.MOV`` beside ``IMG_1234.HEIC``) is not a video of its own: ``companion_of``
names the picture it belongs to, and the scan leaves it out (``locations.scan_tree``).

Only the atom headers are read (``mdat``, the frames, is skipped by seeking), so a 2 GB video costs a few reads.
"""

from __future__ import annotations

import re
import struct
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, BinaryIO

from ..paths import IMAGE_EXTENSIONS, Paths
from . import photo

HEADING = "## Video details"
LIVE_PHOTO_EXTENSIONS = frozenset({".mov", ".mp4"})     # what an iPhone writes beside a Live Photo's picture
DURATION = "Duration:"

_CONTAINERS = {b"moov", b"trak", b"mdia", b"udta", b"meta", b"ilst"}
_EPOCH_1904 = datetime(1904, 1, 1, tzinfo=UTC)
_ISO6709 = re.compile(r"^([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)?")
_MAX_ATOM = 64 << 20                 # a header atom larger than this is not read (a moov is a few hundred KB)


def companion_of(src: Path, names: Iterable[str]) -> str:
    """The picture whose Live Photo video *src* is ("IMG_1234.HEIC"), or "": a .mov or .mp4 whose name without its
    extension is that of an image in the same folder.  *names* are the file names of the folder."""
    if src.suffix.lower() not in LIVE_PHOTO_EXTENSIONS:
        return ""
    stem = src.stem.lower()
    for n in names:
        p = Path(n)
        if p.stem.lower() == stem and p.suffix.lower() in IMAGE_EXTENSIONS:
            return n
    return ""


# ── the atoms ────────────────────────────────────────────────────────────────

def _atoms(fh: BinaryIO, start: int, end: int) -> Iterable[tuple[bytes, int, int]]:
    """(type, payload start, payload end) of each atom between *start* and *end*."""
    pos = start
    while pos + 8 <= end:
        fh.seek(pos)
        head = fh.read(8)
        if len(head) < 8:
            return
        size, kind = struct.unpack(">I4s", head)
        body = pos + 8
        if size == 1:                                  # a 64-bit size follows
            big = fh.read(8)
            if len(big) < 8:
                return
            size = struct.unpack(">Q", big)[0]
            body = pos + 16
        elif size == 0:                                # to the end of the file
            size = end - pos
        if size < body - pos or pos + size > end:
            return
        yield kind, body, pos + size
        pos += size


def _walk(fh: BinaryIO, start: int, end: int, out: dict[str, Any], path: tuple[bytes, ...] = ()) -> None:
    for kind, body, stop in _atoms(fh, start, end):
        if kind == b"mdat" or stop - body > _MAX_ATOM and kind not in _CONTAINERS:
            continue
        here = path + (kind,)
        if kind == b"meta":
            fh.seek(body)
            first = fh.read(8)
            # QuickTime's meta is a plain container; the ISO one (MP4) starts with a 4-byte version and flags
            skip = 4 if len(first) == 8 and first[4:8] != b"hdlr" and first[4:8] != b"keys" else 0
            _walk(fh, body + skip, stop, out, here)
        elif kind in _CONTAINERS:
            _walk(fh, body, stop, out, here)
        elif kind in (b"mvhd", b"tkhd", b"keys", b"hdlr") or (len(path) and path[-1] in (b"udta", b"ilst")):
            fh.seek(body)
            out.setdefault("atoms", []).append((here, fh.read(stop - body)))


def _text(data: bytes) -> str:
    return " ".join(data.decode("utf-8", errors="replace").replace("\x00", " ").split())


def _ilst_value(payload: bytes) -> str:
    """The text of an ``ilst`` item: its ``data`` atom (type 1 = UTF-8), after the 8-byte type and locale."""
    if len(payload) >= 16 and payload[4:8] == b"data":
        return _text(payload[16:struct.unpack(">I", payload[:4])[0]])
    return ""


def read_atoms(src: Path) -> dict[str, Any]:
    """The facts of the header atoms of *src*: ``{"keys": {name: text}, "udta": {type: text}, "mvhd": (created, timescale,
    duration), "tracks": [(width, height, rotated)]}`` (missing parts left out); empty when the file is not a QuickTime / MP4
    file or cannot be read."""
    try:
        with open(src, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            raw: dict[str, Any] = {}
            _walk(fh, 0, size, raw)
    except OSError:
        return {}
    out: dict[str, Any] = {}
    names: list[str] = []
    for path, data in raw.get("atoms", []):
        kind = path[-1]
        if kind == b"mvhd" and len(data) >= 20:
            if data[0] == 1 and len(data) >= 32:
                created, _mod, scale, dur = struct.unpack(">QQIQ", data[4:32])
            else:
                created, _mod, scale, dur = struct.unpack(">IIII", data[4:20])
            out["mvhd"] = (created, scale, dur)
        elif kind == b"tkhd" and len(data) >= 84:
            off = 4 + (32 if data[0] == 1 else 20) + 8 + 8   # times, track id, reserved, duration; then layer .. volume
            matrix = struct.unpack(">9i", data[off:off + 36])
            w, h = (struct.unpack(">I", data[off + 36 + i:off + 40 + i])[0] >> 16 for i in (0, 4))
            if w and h:
                out.setdefault("tracks", []).append((w, h, matrix[0] == 0 and matrix[1] != 0))
        elif kind == b"keys" and len(data) >= 8:
            pos = 8
            while pos + 8 <= len(data):
                n = struct.unpack(">I", data[pos:pos + 4])[0]
                if n < 8:
                    break
                names.append(data[pos + 8:pos + n].decode("utf-8", errors="replace"))
                pos += n
        elif len(path) >= 2 and path[-2] == b"ilst":
            idx = struct.unpack(">I", kind)[0]
            value = _ilst_value(data)
            if value and 1 <= idx <= len(names):
                out.setdefault("keys", {})[names[idx - 1]] = value
            elif value and kind.startswith(b"\xa9"):
                out.setdefault("udta", {})[kind.decode("latin-1")] = value
        elif len(path) >= 2 and path[-2] == b"udta" and kind.startswith(b"\xa9") and len(data) > 4:
            n = struct.unpack(">H", data[:2])[0]       # a QuickTime text item: length, language, the text
            out.setdefault("udta", {})[kind.decode("latin-1")] = _text(data[4:4 + n] if 0 < n <= len(data) - 4 else data[4:])
    return out


# ── the facts ────────────────────────────────────────────────────────────────

def _when(text: str) -> tuple[datetime, str] | None:
    """A date and time written as ISO 8601 ("2023-07-14T18:22:05+0530", "2019-05-03T10:00:00Z"), with its offset."""
    m = re.match(r"(\d{4})-(\d\d)-(\d\d)[T ](\d\d):(\d\d):(\d\d)(?:\.\d+)?\s*(Z|[+-]\d\d:?\d\d)?", text.strip())
    if not m:
        return None
    try:
        when = datetime(*(int(x) for x in m.groups()[:6]))
    except ValueError:
        return None
    zone = m.group(7) or ""
    offset = "+00:00" if zone == "Z" else (zone[:3] + ":" + zone[-2:] if zone else "")
    return (when, offset) if when.year > 1900 else None


def read_meta(src: Path) -> dict[str, Any]:
    """The metadata of a video as the dict ``photo.read_exif`` makes for a photograph (``when``, ``offset``, ``lat``, ``lon``,
    ``altitude``, ``camera``, ``size``), plus ``duration`` in seconds; empty when the file is not a QuickTime / MP4 file."""
    atoms = read_atoms(src)
    if not atoms:
        return _decoded(src)
    keys, udta = atoms.get("keys", {}), atoms.get("udta", {})
    meta: dict[str, Any] = {}
    stamp = (_when(keys.get("com.apple.quicktime.creationdate", "")) or _when(udta.get("©day", ""))
             or _when(keys.get("©day", "")))
    created, scale, dur = atoms.get("mvhd", (0, 0, 0))
    if stamp:
        meta["when"] = stamp[0]
        if stamp[1]:
            meta["offset"] = stamp[1]
    elif created:                                    # the movie header: UTC by the format (0 = not recorded)
        when = _EPOCH_1904 + timedelta(seconds=created)
        if 1990 <= when.year <= datetime.now().year + 1:
            meta["when"], meta["offset"] = when.replace(tzinfo=None), "+00:00"
    where = keys.get("com.apple.quicktime.location.ISO6709") or udta.get("©xyz", "")
    m = _ISO6709.match(where)
    if m:
        lat, lon = float(m.group(1)), float(m.group(2))
        if -90 <= lat <= 90 and -180 <= lon <= 180 and not (lat == 0 and lon == 0):
            meta["lat"], meta["lon"] = lat, lon
            if m.group(3):
                meta["altitude"] = float(m.group(3))
    make = keys.get("com.apple.quicktime.make") or udta.get("©mak", "")
    model = keys.get("com.apple.quicktime.model") or udta.get("©mod", "")
    if make or model:
        meta["camera"] = model if make and model.lower().startswith(make.lower()) else f"{make} {model}".strip()
    for w, h, rotated in atoms.get("tracks", []):
        meta["size"] = (h, w) if rotated else (w, h)  # the picture as it is shown (a phone held upright is rotated)
        break
    if scale and dur:
        meta["duration"] = dur / scale
    return meta


def _decoded(src: Path) -> dict[str, Any]:
    """Size and duration from the decoder (OpenCV), for a video that is not a QuickTime / MP4 file (an .avi): such a file
    records no time or place this can read."""
    try:
        import cv2
    except ImportError:
        return {}
    cap = cv2.VideoCapture(str(src))
    try:
        if not cap.isOpened():
            return {}
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()
    meta: dict[str, Any] = {"size": (w, h)} if w and h else {}
    if frames > 0 and fps > 0:
        meta["duration"] = frames / fps
    return meta


def duration_text(seconds: float) -> str:
    s = round(seconds)
    if s < 60:
        return f"{max(s, 1) if seconds > 0 else 0} s"
    h, rest = divmod(s, 3600)
    m, s = divmod(rest, 60)
    return (f"{h} h {m} min" if h else f"{m} min") + (f" {s} s" if s and not h else "")


def lines(meta: dict[str, Any], paths: Paths, extra: Sequence[str] = ()) -> list[str]:
    more = ([f"{DURATION} {duration_text(meta['duration'])}"] if "duration" in meta else []) + list(extra)
    return photo.lines(meta, paths, "", more)


def block(src: Path, paths: Paths, meta: dict[str, Any] | None = None) -> str:
    """The "Video details" section for *src*, or "" when it is not a video file this can read."""
    meta = read_meta(src) if meta is None else meta
    if not meta:
        return ""
    return HEADING + "\n\n" + "\n".join(f"- {x}" for x in lines(meta, paths)) + "\n"
