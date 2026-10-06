"""A PDF page's own text layer against the text that was read from it (pypdfium2 + stdlib; read-only).

docling leaves things out of its Markdown: table cells it could not structure, sidebars, footnotes, labels inside
vector drawings, a page's running header and footer (on purpose).  The PDF's text layer has all of it, exactly.  This
module answers two questions per page, without a model and in milliseconds:

* ``compare``: how much of the layer's words and numbers are in the result?  (the gate's coverage check for digital
  pages: ``intact`` | ``lost text`` | ``uncertain``)
* ``missing_lines`` / ``fill``: which lines of the layer are not in the result, so that they can be appended and the
  page becomes searchable for them.

Running headers and footers (a line repeated at the top or bottom of at least three pages) and lone page numbers are
left out of the layer first: nobody searches for them, and docling drops them deliberately.
The source file is only read.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

MIN_LAYER_TOKENS = 15                # fewer words than this: too short to judge
INTACT_WORDS, INTACT_NUMBERS = 0.97, 0.98    # at least this share of the layer is in the result: intact
LOST = 0.90                          # below this share of the words or the numbers: lost text
BAND = 0.07                          # top and bottom share of a page where a running header or footer sits
LINE_COVERED = 0.6                   # a layer line is "there" when this share of its tokens is in the result
MAX_FILL_CHARS = 20_000              # most text added to one page
FILL_MARKER = "<!-- text layer: lines the conversion left out -->"


def tokens(text: str) -> tuple[Counter, Counter]:
    """(words, numbers) of a text, normalised so that a PDF layer and docling's Markdown compare: NFKC (ligatures),
    lower case, words broken at a line end joined, Markdown escapes and soft hyphens dropped, numbers without
    thousands separators."""
    t = unicodedata.normalize("NFKC", text or "").replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"(\w)[-­‐‑][ \t]*\n[ \t]*(\w)", r"\1\2", t)           # a word broken at a line end
    t = t.replace("­", "").replace("\\", "").lower()
    nums = Counter(re.sub(r"[,\s]", "", m) for m in re.findall(r"\d[\d,]*(?:\.\d+)?", t))
    words = Counter(re.findall(r"[^\W\d_]{2,}", t))
    return words, nums


def recall(layer: Counter, out: Counter) -> float | None:
    """The share of *layer*'s tokens (with multiplicity) that *out* has; None when the layer has none."""
    total = sum(layer.values())
    if not total:
        return None
    return sum(min(n, out.get(k, 0)) for k, n in layer.items()) / total


def verdict(w: float | None, n: float | None, layer_words: int, layer_ok: bool = True) -> str:
    """``intact`` | ``lost text`` | ``uncertain`` from the word and number recall of a page."""
    if w is None or layer_words < MIN_LAYER_TOKENS or not layer_ok:
        return "uncertain"
    nn = 1.0 if n is None else n
    if w < LOST or nn < LOST:
        return "lost text"
    if w >= INTACT_WORDS and nn >= INTACT_NUMBERS:
        return "intact"
    return "uncertain"


def compare(layer_text: str, md_text: str) -> dict[str, Any]:
    """Word and number recall of the layer in the converted text, and the verdict."""
    from .tables import plain_text

    lw, ln = tokens(layer_text)
    ow, on = tokens(plain_text(md_text))
    w, n = recall(lw, ow), recall(ln, on)
    return {"word_recall": None if w is None else round(w, 3), "number_recall": None if n is None else round(n, 3),
            "layer_words": sum(lw.values()), "layer_numbers": sum(ln.values()),
            "verdict": verdict(w, n, sum(lw.values()))}


def missing_lines(layer_text: str, md_text: str) -> list[str]:
    """The layer's lines that the converted text does not hold, in reading order.  Tokens of the converted text are
    used up as lines are matched, so a line that repeats (a table's ``0`` cells) counts as missing when the result has
    fewer of them."""
    from .tables import plain_text

    avail_w, avail_n = tokens(plain_text(md_text))
    out: list[str] = []
    for raw in (layer_text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw.strip()
        if not line:
            continue
        lw, ln = tokens(line)
        total = sum(lw.values()) + sum(ln.values())
        if not total:
            continue
        have = sum(min(n, avail_w.get(k, 0)) for k, n in lw.items()) + sum(min(n, avail_n.get(k, 0)) for k, n in ln.items())
        if have / total >= LINE_COVERED:
            for k, n in lw.items():
                avail_w[k] = max(0, avail_w.get(k, 0) - n)
            for k, n in ln.items():
                avail_n[k] = max(0, avail_n.get(k, 0) - n)
        else:
            out.append(line)
    return out


def fill(md: str, lines: list[str]) -> tuple[str, int]:
    """*md* with *lines* appended under a marker (at most ``MAX_FILL_CHARS`` of them); (new text, lines added)."""
    kept, size = [], 0
    for line in lines:
        if size + len(line) > MAX_FILL_CHARS:
            break
        kept.append(line)
        size += len(line) + 1
    if not kept:
        return md, 0
    return md.rstrip() + "\n\n" + FILL_MARKER + "\n\n" + "\n".join(kept) + "\n", len(kept)


# ── headers, footers, page numbers ───────────────────────────────────────────

def page_number_line(line: str) -> bool:
    return bool(re.fullmatch(r"\W*(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?\W*", line.strip(), re.I))


def _band_key(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", text.lower())).strip()


def layer_parts(pdf: Any, i: int) -> tuple[str, str, str]:
    """(top band, whole text, bottom band) of page *i* (0-based) of an open pypdfium2 document; empty strings when
    the page cannot be read."""
    try:
        page = pdf[i]
        tp = page.get_textpage()
        try:
            w, h = page.get_size()
            top = tp.get_text_bounded(0, h * (1 - BAND), w, h) or ""
            body = tp.get_text_range() or ""
            bottom = tp.get_text_bounded(0, 0, w, h * BAND) or ""
            return top, body, bottom
        finally:
            tp.close()
            page.close()
    except Exception:  # noqa: BLE001 - a page that cannot be read has no layer here
        return "", "", ""


def repeated_bands(parts: list[tuple[str, str, str]]) -> set[str]:
    """Top and bottom band texts (digits ignored) that occur on at least 3 pages: running headers and footers."""
    seen = Counter(k for top, _b, bottom in parts for k in {_band_key(top), _band_key(bottom)} if k)
    return {k for k, n in seen.items() if n >= 3}


def without_bands(top: str, body: str, bottom: str, bands: set[str]) -> str:
    """The page's text without its running header (first occurrence of the top band's text) and footer (last
    occurrence of the bottom band's text), when those repeat across the document."""
    out = body
    if top.strip() and _band_key(top) in bands:
        out = out.replace(top.strip(), " ", 1)
    if bottom.strip() and _band_key(bottom) in bands:
        k = out.rfind(bottom.strip())
        if k >= 0:
            out = out[:k] + " " + out[k + len(bottom.strip()):]
    return out


def boilerplate(layers: list[str]) -> set[str]:
    """Running headers and footers that sit outside the bands: lines among the first and last two of a page (of six
    lines or more) that repeat (digits ignored) on at least 3 pages."""
    if len(layers) < 4:
        return set()
    seen: Counter = Counter()
    for text in layers:
        lines = [x.strip() for x in text.splitlines() if x.strip()]
        if len(lines) >= 6:
            seen.update({_band_key(x) for x in lines[:2] + lines[-2:]})
    return {k for k, n in seen.items() if n >= 3}


def clean_layer(text: str, boiler: set[str]) -> tuple[str, int]:
    """*text* without lines in *boiler* and without lone page numbers; (text, lines dropped)."""
    keep, dropped = [], 0
    for line in text.splitlines():
        if line.strip() and (_band_key(line) in boiler or page_number_line(line)):
            dropped += 1
            continue
        keep.append(line)
    return "\n".join(keep), dropped


class LayerSource:
    """The text layers of one PDF, page by page, without running headers, footers and page numbers.  The document is
    opened and every page's layer read on first use (needed to know what repeats); a file that cannot be opened has
    no layers (``page`` returns None).  Close it when done."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.texts: list[str] | None = None
        self.raw: list[str] = []

    def _load(self) -> None:
        self.texts = []
        try:
            import pypdfium2 as pdfium

            pdf = pdfium.PdfDocument(str(self.path))
        except Exception:  # noqa: BLE001 - no pypdfium2, or a file it cannot open
            return
        try:
            parts = [layer_parts(pdf, i) for i in range(len(pdf))]
        finally:
            pdf.close()
        bands = repeated_bands(parts)
        self.raw = [body for _t, body, _b in parts]
        layers = [without_bands(t, b, bo, bands) for t, b, bo in parts]
        boiler = boilerplate(layers)
        self.texts = [clean_layer(x, boiler)[0] for x in layers]

    def page(self, n: int) -> str | None:
        """The layer of page *n* (1-based), or None when there is none to use."""
        if self.texts is None:
            self._load()
        assert self.texts is not None
        return self.texts[n - 1] if 0 < n <= len(self.texts) else None

    def close(self) -> None:
        self.texts = None
        self.raw = []
