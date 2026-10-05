"""Tesseract: the plain-text reader for the pages the document reader cannot read (stdlib + the binary).

A classic OCR engine -- not a generative model -- so it cannot loop, hallucinate or run for minutes: a page
takes 1 to 13 seconds, on any machine, and it reads Marathi and Hindi (Devanagari) from its language data.
It writes plain text in reading order and no tables, which is why it is only the last resort, after the
document reader, its retries and the second reader have failed on a page (a runaway, a page it cannot read),
and next to Apple Vision, which cannot read Devanagari on every macOS.

``brew install tesseract`` (macOS) / ``apt-get install tesseract-ocr`` plus the language data
``mar.traineddata`` and ``hin.traineddata`` (``scripts/install.sh`` does it).  ``RAG_SEARCH_TESSERACT=off``
switches it off; ``RAG_SEARCH_TESSERACT_LANG`` (default ``mar+hin+eng``) picks the languages, and a language
that is not installed is left out.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ID = "tesseract"
DEFAULT_LANGS = "mar+hin+eng"
LONG_SIDE = 3000                     # px: Devanagari print is small
PSM = "4"                            # a single column of variable-size text: right for deeds and letters
TIMEOUT_S = 180.0

_langs_cache: dict[str, set[str]] = {}


def mode(env: dict[str, str] | None = None) -> str:
    v = (os.environ if env is None else env).get("RAG_SEARCH_TESSERACT", "auto").strip().lower() or "auto"
    return "off" if v in ("off", "0", "no", "false") else "auto"


def binary() -> str | None:
    return shutil.which("tesseract")


def installed_languages(exe: str | None = None) -> set[str]:
    """The languages the binary has data for (``tesseract --list-langs``), cached per binary."""
    exe = exe or binary()
    if not exe:
        return set()
    if exe not in _langs_cache:
        try:
            out = subprocess.run([exe, "--list-langs"], capture_output=True, text=True, timeout=30).stdout
        except (OSError, subprocess.SubprocessError):
            return set()
        _langs_cache[exe] = {ln.strip() for ln in out.splitlines()[1:] if ln.strip()}
    return _langs_cache[exe]


def languages() -> str:
    """The language argument (``mar+hin+eng``): the requested ones that are installed."""
    want = (os.environ.get("RAG_SEARCH_TESSERACT_LANG") or DEFAULT_LANGS).replace(",", "+").split("+")
    have = installed_languages()
    return "+".join(x for x in (w.strip() for w in want) if x in have)


def why_not() -> str:
    """"" when Tesseract can read pages here, otherwise the reason."""
    if mode() == "off":
        return "Tesseract is switched off (RAG_SEARCH_TESSERACT)"
    if not binary():
        return "tesseract is not installed (brew install tesseract; scripts/install.sh does it)"
    if not languages():
        return "tesseract has none of the requested languages installed (" + (
            os.environ.get("RAG_SEARCH_TESSERACT_LANG") or DEFAULT_LANGS) + ")"
    return ""


def usable() -> bool:
    return not why_not()


def read_image(image: Path, langs: str = "") -> str:
    """Plain text of one image file, in reading order."""
    exe = binary()
    if not exe:
        raise RuntimeError("tesseract is not installed")
    proc = subprocess.run([exe, str(image), "-", "-l", langs or languages() or "eng", "--psm", PSM],
                          capture_output=True, text=True, timeout=TIMEOUT_S)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or "tesseract failed").strip()[-200:])
    return proc.stdout.strip()


def read_page(src: Path, page: int, *, is_image: bool = False) -> str:
    """Text of page *page* (1-based) of a PDF, or of frame *page* of an image file."""
    from . import vlm

    with tempfile.TemporaryDirectory(prefix="rag-search-tess-") as tmp:
        img = Path(tmp) / "page.png"
        if is_image:
            vlm.render_image_frame(src, page, img, long_side=LONG_SIDE)
        else:
            vlm.render_pdf_page(src, page, img, long_side=LONG_SIDE)
        return read_image(img)
