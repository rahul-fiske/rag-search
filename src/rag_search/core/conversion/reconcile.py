"""Tables that continue across a page break (stdlib only).

A statement or a ledger rarely fits one page: the table that ends page 3 goes on at the top of page 4,
and the next page often has no header row of its own.  Read page by page, that costs twice:

* the continuation has no header, so a chunk of it is a pile of numbers (the reader even takes its
  first data row for a header), and a search that matches the *columns* cannot find it;
* the arithmetic is only checked inside a page: the first row of page 4 is never compared with the
  last balance of page 3, so a wrong carried-forward figure goes unnoticed.

``merge`` looks at every pair of consecutive pages.  When the last thing on page N is a table and the
first thing on page N+1 is a table of the same width whose first row is either the same header again
or plainly data (numbers, dates), the two are one table: the continuation gets the header (so its
chunks are readable and its page-local checks work), and the **merged** table runs through the
validators.  A suspect cell that only shows up in the merged table (it sits at the page boundary) is
reported on the page it is on; nothing is repaired (the page images are not looked at again) but the
page is flagged low-confidence, and the trace says which two pages form the table.

Page text is changed in one way only: a header row added to a headerless continuation.  The pages
stay separate (``<!-- page N -->`` markers are untouched), so a chunk keeps one page number.
"""

from __future__ import annotations

import re
from typing import Any

from . import tables, validators

_AMOUNT_RE = re.compile(r"\d\.\d|\d,\d{2,3}")
HEADER_MATCH = 0.6                   # share of the non-empty header cells that must repeat


def _norm(cell: str) -> str:
    return re.sub(r"[^\w]+", "", str(cell).lower())


def _same_header(a: list[str], b: list[str]) -> bool:
    pairs = [(_norm(x), _norm(y)) for x, y in zip(a, b) if _norm(x) or _norm(y)]
    if not pairs:
        return False
    return sum(1 for x, y in pairs if x == y) / len(pairs) >= HEADER_MATCH


def _data_like(row: list[str]) -> bool:
    """A row that is data, not a header: at least one number or date cell, and no cell that reads
    like a column title of a statement (``Balance``, ``Debit``...)."""
    cells = [c for c in row if str(c).strip()]
    if not cells:
        return False
    if validators._roles(list(row)):
        return False
    nums = [c for c in cells if tables.is_numeric(c)]
    if nums and all(re.fullmatch(r"(19|20)\d\d", str(c).strip()) for c in nums) and not any(
            tables.looks_like_date(c) for c in cells):
        return False                                          # a row of years is a header ("Item | 2023 | 2024")
    # an amount (decimals or digit grouping) or a date: a row of 1, 2, 3 is column numbering, not data
    return any(tables.looks_like_date(c) for c in cells) or any(_AMOUNT_RE.search(str(c)) for c in nums)


def _last_table(md: str) -> tuple[int, tables.Table] | None:
    """(index, table) when the page ends with a table."""
    tabs = tables.find_tables(md)
    if tabs and not md[tabs[-1].end:].strip():
        return len(tabs) - 1, tabs[-1]
    return None


def _headerless_rows(md: str) -> tables.Table | None:
    """Pipe rows at the very top of a page with no header and separator line: the body of a table
    whose header is on the page before (a Table with ``n_header`` 0)."""
    text = md.lstrip("\n")
    lead = len(md) - len(text)
    lines: list[str] = []
    for ln in text.split("\n"):
        if not ln.strip():
            break
        if not ln.lstrip().startswith("|"):
            return None
        lines.append(ln)
    if len(lines) < 1 or any(tables._SEP_RE.match(ln) for ln in lines):
        return None
    rows = [tables._split_pipe_row(ln) for ln in lines]
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    end = lead + sum(len(ln) + 1 for ln in lines)
    return tables.Table(rows, 0, lead, min(end, len(md)), "pipe")


def _first_table(md: str) -> tuple[int, tables.Table] | None:
    tabs = tables.find_tables(md)
    if tabs and not md[:tabs[0].start].strip():
        return 0, tabs[0]
    hl = _headerless_rows(md)
    return (0, hl) if hl else None


def _keys(vs: list[dict[str, Any]], table: tables.Table) -> set[tuple]:
    """Identity of suspect cells: the cell's text *and its row*, so that a boundary suspect is not
    taken for a local one that happens to carry the same figure."""
    return {(v["validator"], v.get("role"), v.get("col"), str(v.get("found")),
             tuple(table.rows[int(v["row"])]) if 0 <= int(v["row"]) < len(table.rows) else ())
            for v in vs}


def merge(pages: dict[int, str]) -> dict[str, Any]:
    """Look for tables that continue across consecutive pages.

    Returns ``{"pages": {page: markdown}, "tables": [...]}``: the pages (a headerless continuation
    gets the previous table's header), and one entry per merged table::

        {"pages": [3, 4], "rows": 12, "header": "repeated" | "added", "ok": True,
         "violations": [{"page": 4, "row": 1, "col": 3, "found": ..., "why": ...}]}

    ``violations`` are the suspect cells of the merged table that neither page shows on its own."""
    out = dict(pages)
    merged: list[dict[str, Any]] = []
    nums = sorted(pages)
    for a, b in zip(nums, nums[1:]):
        if b != a + 1:
            continue
        prev = _last_table(out[a])
        nxt = _first_table(out[b])
        if not prev or not nxt:
            continue
        (_, tp), (_, tc) = prev, nxt
        if tp.width < 2 or tp.width != tc.width or len(tp.rows) < 2 or _data_like(tp.header):
            continue
        if tc.n_header and _same_header(tp.header, tc.header):
            how, body_c, new_c = "repeated", tc.rows[tc.n_header:], None
        elif _data_like(tc.rows[0]):
            how, body_c = "added", list(tc.rows)
            new_c = tables.to_markdown(tables.Table([list(tp.header)] + [list(r) for r in tc.rows], 1))
        else:
            continue
        if not body_c:
            continue
        local = _keys(validators.violations_of(validators.validate_table(tp)), tp)
        if how == "added":
            tc_h = tables.Table([list(tp.header)] + [list(r) for r in tc.rows], 1)
            local |= _keys(validators.violations_of(validators.validate_table(tc_h)), tc_h)
        else:
            local |= _keys(validators.violations_of(validators.validate_table(tc)), tc)
        whole = tables.Table([list(r) for r in tp.rows] + [list(r) for r in body_c], tp.n_header, kind=tp.kind)
        found = validators.violations_of(validators.validate_table(whole))
        new = [v for v in found if next(iter(_keys([v], whole))) not in local]
        np_ = len(tp.rows)
        entry: dict[str, Any] = {"pages": [a, b], "rows": len(whole.rows) - whole.n_header, "header": how,
                                 "ok": not new, "violations": []}
        for v in new:
            r = int(v["row"])
            on_next = r >= np_
            # the row as numbered on its own page (headers first, as in Table.rows)
            row = r - np_ + (1 if how == "added" else tc.n_header) if on_next else r
            entry["violations"].append({
                "page": b if on_next else a, "validator": v["validator"], "role": v.get("role"),
                "row": row, "col": v["col"], "found": v.get("found"), "expected": v.get("expected"),
                "why": "across the page break: " + str(v.get("why", ""))})
        merged.append(entry)
        if new_c is not None:
            tail = out[b][tc.end:]
            if out[b][tc.start:tc.end].endswith("\n") and not new_c.endswith("\n"):
                new_c += "\n"                                  # keep the blank line that ended the table
            out[b] = out[b][:tc.start] + new_c + tail
    return {"pages": out, "tables": merged}
