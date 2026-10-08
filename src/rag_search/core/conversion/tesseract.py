"""Tesseract: the plain-text reader for the pages the document reader cannot read (stdlib + the binary).

A classic OCR engine -- not a generative model -- so it cannot loop, hallucinate or run for minutes: a page
takes 1 to 13 seconds, on any machine, and it reads Marathi and Hindi (Devanagari) from its language data.
It writes plain text in reading order and no tables, which is why it is only the last resort, after the
document reader, its retries and the second reader have failed on a page (a runaway, a page it cannot read),
and next to Apple Vision, which cannot read Devanagari on every macOS.

``brew install tesseract`` (macOS) / ``apt-get install tesseract-ocr`` plus the language data
``mar.traineddata`` and ``hin.traineddata`` (``rag-search setup`` does it: ``ensure_installed``).  ``RAG_SEARCH_TESSERACT=off``
switches it off; ``RAG_SEARCH_TESSERACT_LANG`` (default ``mar+hin+eng``) picks the languages, and a language
that is not installed is left out.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
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


SETUP_LANGS = ("mar", "hin")         # the language data `ensure_installed` adds (English comes with Tesseract)
TESSDATA_URL = "https://github.com/tesseract-ocr/tessdata_best/raw/main/{lang}.traineddata"


def _tessdata_dir(exe: str) -> str:
    """The folder Tesseract keeps its language data in: the quoted path in the first line of ``--list-langs``."""
    try:
        first = subprocess.run([exe, "--list-langs"], capture_output=True, text=True, timeout=30).stdout.splitlines()[:1]
    except (OSError, subprocess.SubprocessError):
        return ""
    m = re.search(r'"(.*)"', first[0]) if first else None
    return m.group(1) if m else ""


def ensure_installed(say=print, *, install: bool = True) -> None:
    """``rag-search setup``: Tesseract and the Marathi and Hindi language data, as far as it can be done without
    asking for a password.  Installs the program with Homebrew when it is missing (``install=False`` only looks);
    adds the two language files to the data folder when that is writable; says what is left for the person to do.
    Never raises: the last-resort reader is optional."""
    exe = binary()
    if not exe:
        if install and shutil.which("brew"):
            say("Installing Tesseract with Homebrew (the last-resort page reader) ...")
            try:
                subprocess.run(["brew", "install", "tesseract"], timeout=1800, check=False)
            except (OSError, subprocess.SubprocessError):
                pass
            exe = binary()
            if not exe:
                say("warning: 'brew install tesseract' failed; pages the reader cannot read will stay flagged")
        elif sys.platform.startswith("linux"):
            say("note: Tesseract is not installed; install it with:  sudo apt-get install -y tesseract-ocr   "
                "(then run: rag-search setup)")
        else:
            say("note: Tesseract is not installed and Homebrew was not found; install it "
                "(https://brew.sh, then: brew install tesseract)")
    if not exe:
        return
    _langs_cache.pop(exe, None)
    have = installed_languages(exe)
    folder = _tessdata_dir(exe)
    for lang in SETUP_LANGS:
        if lang in have:
            continue
        if install and folder and os.access(folder, os.W_OK):
            say(f"adding the {lang} language data to {folder}")
            target = Path(folder) / f"{lang}.traineddata"
            try:
                with urllib.request.urlopen(TESSDATA_URL.format(lang=lang), timeout=120) as resp:   # noqa: S310
                    data = resp.read()
                target.write_bytes(data)
            except (OSError, ValueError) as exc:
                target.unlink(missing_ok=True)
                say(f"warning: could not download {lang}.traineddata ({exc})")
        else:
            say(f"note: Tesseract has no '{lang}' data; put {lang}.traineddata "
                f"(github.com/tesseract-ocr/tessdata_best) into {folder or 'its tessdata folder'}")
    _langs_cache.pop(exe, None)


def why_not() -> str:
    """"" when Tesseract can read pages here, otherwise the reason."""
    if mode() == "off":
        return "Tesseract is switched off (RAG_SEARCH_TESSERACT)"
    if not binary():
        return "tesseract is not installed (brew install tesseract; rag-search setup does it)"
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


def read_page(src: Path, page: int, *, is_image: bool = False, skew: float = 0.0) -> str:
    """Text of page *page* (1-based) of a PDF, or of frame *page* of an image file.  A page whose measured *skew* is
    more than a quarter degree is straightened first."""
    from . import vlm

    with tempfile.TemporaryDirectory(prefix="rag-search-tess-") as tmp:
        img = Path(tmp) / "page.png"
        if is_image:
            vlm.render_image_frame(src, page, img, long_side=LONG_SIDE)
        else:
            vlm.render_pdf_page(src, page, img, long_side=LONG_SIDE)
        if abs(skew) >= 0.3:
            from PIL import Image

            from . import scanfacts

            with Image.open(img) as im:
                im.load()
                scanfacts.straighten(im.copy(), skew).save(img, "PNG")
        return read_image(img)
