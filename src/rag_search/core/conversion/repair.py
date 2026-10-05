"""Repair of suspect table cells (stdlib only; the readers are reached through ``vlm`` and a second reader).

A scanned bank statement is read by the document VLM; one amount comes out as ``2,81.00``.  The
validators (``validators.py``) see that the running balance no longer adds up, and say *which cell*
is the suspect and what it would have to be (a hypothesis).  Repair then looks at that one cell:

1. an **independent second reader** (``ocrmac`` / Apple Vision on the Mac) reads the page image and
   gives its words with boxes; the row is found by matching the row's *other* cells against the
   words (the line that matches most of them), the cell is the unclaimed word between the row's
   neighbouring cells, and its box is the crop;
2. the **repair model** (``vlm.repair_shared``) reads that crop: just the cell, a number;
3. the new text is **accepted only when all three agree**: it is a number, it equals what the second
   reader saw in that place, and the validators agree (the cell is no longer a suspect and no new
   suspect appeared).  A candidate equal to the original, or one the second reader contradicts, is
   rejected: a wrong digit that two models agree on *and* that makes the arithmetic hold is as good
   as certain; anything less is left alone and flagged.

Cells are repaired one at a time (the suspects are recomputed after each accepted fix, because one
fix can explain the next row), at most ``MAX_CELLS`` per page.  When cells cannot be fixed and the
repair model is not the reader model, the whole page is read once more by the repair model and kept
only if it passes the gate.  Nothing here changes a source file; the repaired text replaces the
reader's text for that page, and every attempt (accepted or not) is recorded in the page's trace.

``Repairer.run`` is the entry point used by ``routed.py``; ``repair_cells`` is the pure part (words
and a cell reader in, new Markdown and a log out), which the tests drive directly.
"""

from __future__ import annotations

import difflib
import importlib
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from . import degenerate, tables, validators

MAX_CELLS = 10                       # cells tried per page
ALGO_VERSION = "r2"                  # bump when the repair method changes: cached pages are tried again
MIN_SIMILARITY = 0.5                 # candidate vs the cell's current text, to choose between words
VPAD, HPAD = 0.3, 0.6                # crop padding, in word heights


def mode(env: Any = None) -> str:
    """``auto`` or ``off`` (``$RAG_SEARCH_REPAIR``)."""
    v = (os.environ if env is None else env).get("RAG_SEARCH_REPAIR", "auto").strip().lower() or "auto"
    return "off" if v in ("off", "0", "false", "no", "none") else "auto"


# ── the second reader ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Word:
    text: str
    l: float                         # noqa: E741  box as fractions of the page, origin top-left
    t: float
    r: float
    b: float

    @property
    def h(self) -> float:
        return max(1e-6, self.b - self.t)

    @property
    def cy(self) -> float:
        return (self.t + self.b) / 2

    @property
    def cx(self) -> float:
        return (self.l + self.r) / 2


class OcrMacSecond:
    """Apple Vision text recognition through ``ocrmac`` (``pip install 'rag-search[mac-vlm]'``).  It
    returns phrases with boxes (origin bottom-left, fractions of the image); a phrase is cut into
    words in proportion to its characters, which is exact enough for finding a cell."""

    id = "ocrmac"

    def why_not(self) -> str:
        if sys.platform != "darwin":
            return "ocrmac (Apple Vision) needs a Mac"
        import importlib.util

        if importlib.util.find_spec("ocrmac") is None:
            return "ocrmac is not installed (pip install 'rag-search[mac-vlm]')"
        return ""

    def words(self, image: Path) -> list[Word]:
        from ocrmac import ocrmac                              # type: ignore

        out: list[Word] = []
        for text, _conf, box in ocrmac.OCR(str(image), recognition_level="accurate").recognize():
            x, y, w, h = (float(v) for v in box)
            top, bottom = 1.0 - (y + h), 1.0 - y
            n = max(1, len(text))
            for m in re.finditer(r"\S+", text):
                out.append(Word(m.group(0), x + w * m.start() / n, top, x + w * m.end() / n, bottom))
        return out


def second_reader() -> Any | None:
    """The independent reader that gives words with boxes: ``$RAG_SEARCH_REPAIR_SECOND`` is ``auto``
    (ocrmac when it can run), ``off``, or ``module:attr`` of a class with ``words(image)``."""
    spec = os.environ.get("RAG_SEARCH_REPAIR_SECOND", "auto").strip() or "auto"
    if spec.lower() in ("off", "0", "false", "no", "none"):
        return None
    if spec.lower() == "auto":
        r = OcrMacSecond()
        return None if r.why_not() else r
    mod, _, attr = spec.partition(":")
    if not attr:
        return None
    return getattr(importlib.import_module(mod), attr)()


def second_why_not() -> str:
    """Why there is no second reader (for the trace and the models list); empty when there is one."""
    spec = os.environ.get("RAG_SEARCH_REPAIR_SECOND", "auto").strip().lower() or "auto"
    if spec in ("off", "0", "false", "no", "none"):
        return "switched off (RAG_SEARCH_REPAIR_SECOND=off)"
    if spec == "auto":
        return OcrMacSecond().why_not()
    return ""


# ── finding a cell ────────────────────────────────────────────────────────────────────────

def _norm(text: str) -> str:
    n = tables.parse_number(text)
    if n is not None:
        return "n:" + tables.norm_number_text(text)
    return re.sub(r"[^\w]+", "", str(text).lower())


def _lines(words: list[Word]) -> list[list[int]]:
    """Indices of *words* grouped into text lines (by vertical position), each left to right."""
    order = sorted(range(len(words)), key=lambda i: words[i].cy)
    lines: list[list[int]] = []
    centre: list[float] = []
    height: list[float] = []
    for i in order:
        w = words[i]
        if lines and abs(w.cy - centre[-1]) <= 0.5 * max(w.h, height[-1]):
            lines[-1].append(i)
            centre[-1] = sum(words[j].cy for j in lines[-1]) / len(lines[-1])
            height[-1] = max(height[-1], w.h)
        else:
            lines.append([i])
            centre.append(w.cy)
            height.append(w.h)
    return [sorted(ln, key=lambda i: words[i].l) for ln in lines]


def _match_cell(text: str, line: list[int], words: list[Word], used: set[int]) -> list[int] | None:
    """The words of *line* that spell the cell's text, or None.  A multi-word cell needs most of its
    words (descriptions are often broken differently by two readers)."""
    toks = [_norm(t) for t in str(text).split() if _norm(t)]
    if not toks:
        return None
    found: list[int] = []
    taken = set(used)
    for tok in toks:
        for i in line:
            if i not in taken and _norm(words[i].text) == tok:
                found.append(i)
                taken.add(i)
                break
    need = len(toks) if len(toks) <= 2 else max(2, int(0.6 * len(toks) + 0.999))
    return found if len(found) >= need else None


def _similar(a: str, b: str) -> float:
    a, b = re.sub(r"\s+", "", a), re.sub(r"\s+", "", b)
    return difflib.SequenceMatcher(None, a, b).ratio() if a and b else 0.0


def locate(rows: list[list[str]], row: int, col: int, found: str, words: list[Word]) -> dict[str, Any]:
    """Find cell (*row*, *col*) of a table among the second reader's *words*.

    Returns ``{"box": (l, t, r, b), "second": text}`` or ``{"why": reason}``.  *found* is what the
    first reader wrote there (it may be a misread, so only its similarity is used)."""
    if not words:
        return {"why": "the second reader found no text on the page"}
    others = {c: str(v) for c, v in enumerate(rows[row]) if c != col and str(v).strip()}
    if not others:
        return {"why": "the row has no other cell to find it by"}
    lines = _lines(words)
    scored: list[tuple[int, int, dict[int, list[int]]]] = []
    for li, ln in enumerate(lines):
        matched: dict[int, list[int]] = {}
        used: set[int] = set()
        for c, text in others.items():
            m = _match_cell(text, ln, words, used)
            if m:
                matched[c] = m
                used.update(m)
        scored.append((len(matched), li, matched))
    scored.sort(key=lambda s: -s[0])
    best = scored[0]
    if best[0] < min(2, len(others)):
        return {"why": f"the row was not found on the page (best line matches {best[0]} of {len(others)} other cells)"}
    if len(scored) > 1 and scored[1][0] == best[0]:
        return {"why": "the row's other cells match more than one line equally (repeated rows)"}
    _, li, matched = best
    line = lines[li]
    claimed = {i for ids in matched.values() for i in ids}

    def extent(ids: list[int]) -> tuple[float, float]:
        return min(words[i].l for i in ids), max(words[i].r for i in ids)

    left = [c for c in matched if c < col]
    right = [c for c in matched if c > col]
    gap_l = extent(matched[max(left)])[1] if left else 0.0
    gap_r = extent(matched[min(right)])[0] if right else 1.0
    cand = [i for i in line if i not in claimed and gap_l - 1e-6 <= words[i].cx <= gap_r + 1e-6]
    if not cand:
        return {"why": "the second reader sees nothing between the neighbouring cells"}
    groups: list[list[int]] = [[cand[0]]]                  # words that touch form one cell
    for i in cand[1:]:
        prev = words[groups[-1][-1]]
        if words[i].l - prev.r <= prev.h:
            groups[-1].append(i)
        else:
            groups.append([i])
    texts = [" ".join(words[i].text for i in g) for g in groups]
    if len(groups) > 1:
        sims = [_similar(found, t) for t in texts]
        order = sorted(range(len(groups)), key=lambda k: -sims[k])
        if sims[order[0]] < MIN_SIMILARITY or sims[order[0]] == sims[order[1]]:
            return {"why": f"{len(groups)} separate texts sit where the cell should be"}
        pick = order[0]
    else:
        pick = 0
    ids = groups[pick]
    l, r = extent(ids)                                    # noqa: E741
    t, b = min(words[i].t for i in ids), max(words[i].b for i in ids)
    h = b - t
    box = (max(gap_l, l - HPAD * h), max(0.0, t - VPAD * h), min(gap_r, r + HPAD * h), min(1.0, b + VPAD * h))
    return {"box": box, "second": texts[pick]}


# ── repairing a page's cells ──────────────────────────────────────────────────────────────

def _reject(status: str, why: str, **kw: Any) -> dict[str, Any]:
    return {"status": status, "why": why, **kw}


def _attempt(md: str, v: dict[str, Any], words: list[Word],
             read_cell: Callable[[tuple[float, float, float, float]], str]) -> tuple[str | None, dict[str, Any]]:
    """Try to fix one suspect cell.  Returns (new Markdown or None, the log entry)."""
    ti, row, col = int(v.get("table", 0)), int(v["row"]), int(v["col"])
    entry: dict[str, Any] = {"table": ti, "row": row, "col": col, "role": v.get("role", ""),
                             "before": v.get("found", ""), "expected": v.get("expected", "")}
    tabs = tables.find_tables(md)
    if ti >= len(tabs):
        return None, {**entry, **_reject("not_located", "the table is gone")}
    t = tabs[ti]
    loc = locate(t.rows, row, col, str(v.get("found", "")), words)
    if "box" not in loc:
        return None, {**entry, **_reject("not_located", loc["why"])}
    entry["second"] = loc["second"]
    try:
        cand = str(read_cell(loc["box"]) or "").strip().strip("`").strip()
    except Exception as exc:  # noqa: BLE001 - a reader problem is a log entry, not a failure
        return None, {**entry, **_reject("error", f"{type(exc).__name__}: {str(exc)[:160]}")}
    cand = re.sub(r"\s+", " ", cand.splitlines()[0] if cand else "")
    entry["read"] = cand
    if tables.norm_number_text(cand) == tables.norm_number_text(str(v.get("found", ""))):
        return None, {**entry, **_reject("unchanged", "the model reads the same text again: the arithmetic is wrong, not the reading")}
    if tables.parse_number(cand) is None:
        return None, {**entry, **_reject("not_a_number", f"the model read {cand!r} in the crop, which is not a number")}
    if tables.norm_number_text(cand) != tables.norm_number_text(loc["second"]):
        return None, {**entry, **_reject("disagree", f"the model reads {cand!r} but the second reader sees {loc['second']!r}")}
    chk = validators.recheck_with(t, row, col, cand)
    if not chk["fixed"]:
        return None, {**entry, **_reject("not_confirmed", f"{cand!r} does not make the table's arithmetic hold")}
    new = tables.replace_cell(md, ti, row, col, cand)
    if new is None:
        return None, {**entry, **_reject("not_editable", "the cell cannot be replaced in this table (merged cells)")}
    return new, {**entry, "after": cand, "status": "fixed", "why": "two reads agree and the arithmetic holds"}


def repair_cells(md: str, violations: list[dict[str, Any]], words: list[Word],
                 read_cell: Callable[[tuple[float, float, float, float]], str], *,
                 max_cells: int = MAX_CELLS) -> dict[str, Any]:
    """Fix suspect cells of *md*.  *violations* come from the gate; *words* from the second reader;
    *read_cell(box)* returns the repair model's reading of the crop.  Returns ``{"md", "cells":
    [log entries], "fixed", "tried"}``."""
    cells: list[dict[str, Any]] = []
    tried: set[tuple[int, int, int]] = set()
    queue = list(violations)
    fixed = 0
    while queue and len(tried) < max_cells:
        v = queue.pop(0)
        key = (int(v.get("table", 0)), int(v["row"]), int(v["col"]))
        if key in tried:
            continue
        tried.add(key)
        new, entry = _attempt(md, v, words, read_cell)
        cells.append(entry)
        if new is not None:
            md, fixed = new, fixed + 1
            queue = [x for x in validators.page_violations(md)
                     if (int(x.get("table", 0)), int(x["row"]), int(x["col"])) not in tried]
    return {"md": md, "cells": cells, "fixed": fixed, "tried": len(tried)}


def _numbers(md: str) -> int:
    """How many different numbers the tables hold.  Distinct, not cells: a page whose columns are mixed up
    repeats one amount in several columns, and a correct re-read must not be turned down for having fewer."""
    return len({tables.norm_number_text(c) for t in tables.find_tables(md) for r in t.rows for c in r
                if tables.is_numeric(c)})


def _vocabulary(md: str) -> set[str]:
    return set(re.findall(r"\w{2,}", tables.plain_text(md).lower()))


def not_smaller(old: str, new: str) -> bool:
    """Is a page read again at least as complete as the first reading?  A re-read replaces the page
    when it passes the gate, and a page that lost its table passes it trivially: so it must keep the
    tables, nearly all the different numbers and most of the different words.  Different, not how many:
    a page whose columns were mixed up repeats its figures, and a correct reading is shorter."""
    if len(tables.find_tables(new)) < len(tables.find_tables(old)):
        return False
    if _numbers(new) < 0.9 * _numbers(old):
        return False
    return len(_vocabulary(new)) >= 0.8 * len(_vocabulary(old))


# ── the repairer used by the converter ────────────────────────────────────────────────────

class Repairer:
    """Repairs a page of a PDF or image file: renders it, asks the second reader for words, reads
    the crops with the repair model.  ``tag`` goes into the page cache next to a repaired page, so a
    page is not tried again with the same models and method."""

    def __init__(self, reader: Any, second: Any | None, *, reader_model: str = "") -> None:
        self.reader = reader                                  # a vlm.VlmReader (the repair model)
        self.second = second
        self.reread = bool(reader_model) and reader.model != reader_model
        self.tag = f"{ALGO_VERSION}|{reader.model}|{getattr(second, 'id', '-')}"
        self.cells_fixed = 0
        self.cells_tried = 0

    def usable(self) -> bool:
        self.reader.check()                                   # cheap: rules out a model that is not there
        return self.reader.usable()

    def _crop(self, src: Path, page: int, is_image: bool, tmp: Path) -> Callable[[tuple[float, float, float, float]], str]:
        from . import vlm

        def read_cell(box: tuple[float, float, float, float]) -> str:
            img = tmp / f"cell_{page}.png"
            try:
                if is_image:
                    vlm.render_image_frame(src, page, img, long_side=900, crop=box)
                else:
                    vlm.render_pdf_page(src, page, img, crop=box, long_side=900)
                res = self.reader.read_image(img, "cell")
            finally:
                img.unlink(missing_ok=True)
            self._tokens += int(res.get("tokens") or 0)
            self._gpu += float(res.get("seconds") or 0.0)
            return str(res.get("md") or "")
        return read_cell

    def run(self, src: Path, page: int, is_image: bool, md: str, violations: list[dict[str, Any]], *,
            page_ok: Callable[[str], bool] | None = None) -> dict[str, Any]:
        """Repair page *page*.  Returns ``{"md", "cells", "fixed", "tried", "tokens", "gpu_s",
        "seconds", "model", "second", "tier", "note"}``; ``md`` is the original text when nothing was
        fixed.  *page_ok(md)* tells whether a page's Markdown passes the gate (for the page re-read)."""
        from . import vlm

        t0 = time.perf_counter()
        self._tokens, self._gpu = 0, 0.0
        out: dict[str, Any] = {"md": md, "cells": [], "fixed": 0, "tried": 0, "model": self.reader.model,
                               "second": getattr(self.second, "id", ""), "tier": "", "note": ""}
        tmp = self.reader._tmp_dir()
        try:
            if not violations:
                pass                                           # a flagged page without a suspect cell: only the page re-read below
            elif self.second is None:
                out["note"] = "cells not repaired: " + (second_why_not() or "no second reader")
            else:
                img = tmp / f"page_{page}.png"
                try:
                    if is_image:
                        vlm.render_image_frame(src, page, img)
                    else:
                        vlm.render_pdf_page(src, page, img)
                    words = self.second.words(img)
                except Exception as exc:  # noqa: BLE001
                    words = []
                    out["note"] = f"cells not repaired: the second reader failed ({type(exc).__name__}: {str(exc)[:120]})"
                finally:
                    img.unlink(missing_ok=True)
                if words:
                    res = repair_cells(md, violations, words, self._crop(src, page, is_image, tmp))
                    out.update(md=res["md"], cells=res["cells"], fixed=res["fixed"], tried=res["tried"],
                               tier="cells" if res["fixed"] else "")
                elif not out["note"]:
                    out["note"] = "cells not repaired: the second reader found no text on the page"
            # tier 2: a page that still fails the gate (a suspect cell nobody could fix, shifted columns, a loop)
            # goes to the repair model as a whole
            remaining = validators.page_violations(out["md"])
            if (self.reread and page_ok is not None and (remaining or not page_ok(out["md"]))
                    and self.reader.usable()):
                try:                                           # tier 2: the repair model reads the whole page
                    img = tmp / f"reread_{page}.png"
                    try:
                        if is_image:
                            vlm.render_image_frame(src, page, img)
                        else:
                            vlm.render_pdf_page(src, page, img)
                        res2, loop_note = self.reader.read_page_guarded(img)
                    finally:
                        img.unlink(missing_ok=True)
                    self._tokens += int(res2.get("tokens") or 0)
                    self._gpu += float(res2.get("seconds") or 0.0)
                    md2 = str(res2.get("md") or "")
                    if loop_note:
                        out["note"] = (out["note"] + "; " if out["note"] else "") + loop_note
                    runaway = degenerate.assess(out["md"])["bad"]       # a loop is longer than the page: not "smaller"
                    if md2.strip() and page_ok(md2) and (runaway or not_smaller(out["md"], md2)):
                        out.update(md=md2, tier="page")
                        out["note"] = (out["note"] + "; " if out["note"] else "") + \
                            f"page read again by the repair model {self.reader.model}: it passes"
                    else:
                        out["note"] = (out["note"] + "; " if out["note"] else "") + \
                            "page read again by the repair model: still suspect or less complete, kept the first reading"
                except vlm.ReaderError as exc:
                    out["note"] = (out["note"] + "; " if out["note"] else "") + \
                        f"page not read again ({exc.reason}: {str(exc)[:100]})"
                except Exception as exc:  # noqa: BLE001 - an optional step never fails the page
                    out["note"] = (out["note"] + "; " if out["note"] else "") + \
                        f"page not read again ({type(exc).__name__}: {str(exc)[:100]})"
        finally:
            out.update(tokens=self._tokens, gpu_s=round(self._gpu, 3), seconds=round(time.perf_counter() - t0, 3))
        self.cells_fixed += out["fixed"]
        self.cells_tried += out["tried"]
        return out


def build(reader_model: str = "") -> "Repairer | None":
    """The repairer for this process: None when repair is switched off (``RAG_SEARCH_REPAIR=off``),
    or the document reader is (``RAG_SEARCH_VLM=off``)."""
    if mode() == "off":
        return None
    from . import vlm

    rd = vlm.repair_shared()
    if rd is None:
        return None
    sh = vlm.shared()
    return Repairer(rd, second_reader(), reader_model=sh.model if sh else reader_model)
