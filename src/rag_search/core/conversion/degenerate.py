"""Is a page's text a runaway of the reader, not a reading of the page? (stdlib only)

A small generative reader that is decoding greedily can fall into a loop on a dense page (a long
annexure table, close Devanagari print): it writes the same line or phrase again and again until it hits
the token limit, or drifts into a script that is not on the page.  The text then looks like text, the page
takes minutes, and nothing in the table or coverage checks notices.  Three independent signals, each
conservative (a page of a statement has many *different* lines of the same shape; a loop has the *same*
words hundreds of times):

* one substantive line repeated many times and making up a large share of the lines,
* one four-word phrase making up a large share of the page,
* a long text that compresses to almost nothing,
* the same raw line (an empty table row, say) filling the page, with hardly any words, and
* most of the letters in a script that is neither Latin nor Devanagari.

``looping`` is the cheap tail test the reader runs while it generates, so a loop is stopped after a
few hundred tokens instead of at the limit.
"""

from __future__ import annotations

import collections
import re
import unicodedata
import zlib
from typing import Any

_TAG = re.compile(r"<[^>]+>|<!--.*?-->", re.S)
_WORD = re.compile(r"\w+", re.U)

MIN_LINE_CHARS = 10                 # a "substantive" line has this many characters (tags and bars removed)
LINE_REPEATS = 8                    # ... repeated this often
LINE_SHARE = 0.25                   # ... and at least this share of the substantive lines
PHRASE_REPEATS = 25                 # one four-word phrase this often
PHRASE_SHARE = 0.10                 # ... and at least this share of all four-word phrases
MIN_WORDS_FOR_RATIO = 150
RATIO_LIMIT = 0.10                  # compressed / raw size below this is a loop (ordinary text: 0.25 - 0.5)
RAW_REPEATS = 40                    # the same raw line this often, making up RAW_SHARE of the lines,
RAW_SHARE = 0.60                    # on a page with fewer than MIN_WORDS_FOR_RATIO words
FOREIGN_SHARE = 0.30
MIN_LETTERS = 40
EXPECTED_SCRIPTS = ("LATIN", "DEVANAGARI")
_ODD_SCRIPTS = ("BENGALI", "GURMUKHI", "GUJARATI", "ORIYA", "TAMIL", "TELUGU", "KANNADA", "MALAYALAM", "SINHALA",
                "THAI", "LAO", "TIBETAN", "MYANMAR", "ARABIC", "HEBREW", "CJK", "HANGUL", "HIRAGANA", "KATAKANA",
                "CYRILLIC", "GREEK")


def _plain(md: str) -> str:
    return _TAG.sub(" ", md or "")


def assess(md: str) -> dict[str, Any]:
    """``{"bad": bool, "why": str, ...measures}`` for a page's Markdown."""
    plain = _plain(md)
    lines = [re.sub(r"[\s|:\-]+", " ", ln).strip() for ln in plain.split("\n")]
    lines = [ln for ln in lines if sum(c.isalpha() for c in ln) >= MIN_LINE_CHARS - 2 and len(ln) >= MIN_LINE_CHARS]
    top_line = collections.Counter(lines).most_common(1)[0][1] if lines else 0
    words = _WORD.findall(plain)
    grams = [" ".join(words[i:i + 4]) for i in range(max(0, len(words) - 3))]
    top_gram = collections.Counter(grams).most_common(1)[0][1] if grams else 0
    flat = re.sub(r"\s+", " ", plain).strip().encode("utf-8")
    ratio = len(zlib.compress(flat)) / len(flat) if len(words) >= MIN_WORDS_FOR_RATIO else 1.0
    raw_lines = [ln.strip() for ln in (md or "").split("\n") if ln.strip()]
    top_raw = collections.Counter(raw_lines).most_common(1)[0][1] if raw_lines else 0
    letters = collections.Counter()
    for ch in plain:
        if ch.isalpha():
            letters[unicodedata.name(ch, "X").split()[0]] += 1
    n_letters = sum(letters.values())
    foreign = max(((s, letters[s] / n_letters) for s in _ODD_SCRIPTS if letters[s]), key=lambda x: x[1],
                  default=("", 0.0)) if n_letters >= MIN_LETTERS else ("", 0.0)
    why = ""
    if top_line >= LINE_REPEATS and top_line / max(1, len(lines)) >= LINE_SHARE:
        why = f"one line is repeated {top_line} times"
    elif top_gram >= PHRASE_REPEATS and top_gram / max(1, len(grams)) >= PHRASE_SHARE:
        why = f"one phrase is repeated {top_gram} times"
    elif ratio < RATIO_LIMIT:
        why = "the text is almost entirely repetition"
    elif len(words) < MIN_WORDS_FOR_RATIO and top_raw >= RAW_REPEATS and top_raw / len(raw_lines) >= RAW_SHARE:
        why = f"the same line (an empty table row?) is repeated {top_raw} times and the page has hardly any text"
    elif foreign[1] > FOREIGN_SHARE:
        why = f"most of the text is in {foreign[0].title()} script, which is not what the page is written in"
    return {"bad": bool(why), "why": why, "line_repeats": top_line, "phrase_repeats": top_gram,
            "compression": round(ratio, 3), "foreign": foreign[0] if foreign[1] > FOREIGN_SHARE else ""}


def looping(text: str) -> bool:
    """Cheap test while generating: has the end of *text* just repeated itself several times in a row?"""
    if len(text) < 700:
        return False
    tail = text[-100:]
    if not tail.strip() or len(set(tail)) < 4:
        return text[-400:].count(tail[-20:]) > 15 if tail.strip() else True
    return text[-2400:].count(tail) >= 4
