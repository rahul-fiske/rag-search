"""Tables and numbers in page Markdown (stdlib only).

Readers return Markdown: docling writes GFM pipe tables, document VLMs write HTML or pipe tables.
The gate, the validators, the reconciler and the benchmark all need the cells, so one parser lives
here.  A ``Table`` is a rectangular list of rows of cell strings; ``n_header`` says how many leading
rows are headers (pipe tables: 1; HTML: the rows made of ``<th>`` cells, else 1).
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from html.parser import HTMLParser

_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_NUM_STRIP = re.compile(r"[\s\u00a0\u202f]")
_CURRENCY = "₹$€£¥"
_NUMBER_RE = re.compile(r"^(\d+(\.\d+)?|\.\d+|\d{1,3}(,\d{3})+(\.\d+)?|\d{1,2}(,\d{2})+,\d{3}(\.\d+)?)$")


@dataclass
class Table:
    rows: list[list[str]]
    n_header: int = 1
    start: int = 0                    # offset in the Markdown where the table starts
    end: int = 0                      # ... and ends (exclusive)
    kind: str = "pipe"                # pipe | html
    meta: dict = field(default_factory=dict)

    @property
    def width(self) -> int:
        return max((len(r) for r in self.rows), default=0)

    @property
    def header(self) -> list[str]:
        return self.rows[self.n_header - 1] if self.rows and self.n_header else []

    @property
    def body(self) -> list[list[str]]:
        return self.rows[self.n_header:]


# ── numbers ───────────────────────────────────────────────────────────────────────────────

def parse_number(cell: str) -> Decimal | None:
    """The number in a cell, or None.  Understands Indian grouping (1,23,456.78), currency
    signs, parentheses and a trailing minus as negative, and Dr / Cr suffixes (Dr = negative).
    A cell that is not *only* a number (text around it) is None."""
    s = html.unescape(str(cell or "")).strip()
    if not s:
        return None
    s = _NUM_STRIP.sub("", s)
    s = re.sub(r"/[-=]+$", "", s)                  # 500/-  and  1,500/=  are plain amounts (rupees, no paise)
    sign = 1
    low = s.lower()
    for suffix, sg in (("dr.", -1), ("dr", -1), ("cr.", 1), ("cr", 1)):
        if low.endswith(suffix) and len(s) > len(suffix):
            s, sign = s[: -len(suffix)], sg
            break
    if s.startswith("(") and s.endswith(")"):
        s, sign = s[1:-1], -sign
    if s.endswith("-") and len(s) > 1:
        s, sign = s[:-1], -sign
    if s[:1] in "+-−–" and len(s) > 1:
        if s[0] in "-−–":
            sign = -sign
        s = s[1:]
    low = s.lower()
    for prefix in ("rs.", "rs", "inr"):
        if low.startswith(prefix) and len(s) > len(prefix) and (s[len(prefix)].isdigit() or s[len(prefix)] in ".-"):
            s = s[len(prefix):]
            break
    s = s.lstrip(_CURRENCY).rstrip("/")
    if s[:1] in "-−–" and len(s) > 1:
        sign, s = -sign, s[1:]
    if not _NUMBER_RE.match(s):
        return None
    try:
        return sign * Decimal(s.replace(",", ""))
    except InvalidOperation:
        return None


def is_numeric(cell: str) -> bool:
    return parse_number(cell) is not None


def norm_number_text(cell: str) -> str:
    """Canonical text of a number cell (for exact-match comparisons): ``(1,234.50)`` and
    ``-1234.5`` both become ``-1234.5``; anything else is returned stripped and lower-cased."""
    n = parse_number(cell)
    if n is None:
        return re.sub(r"\s+", " ", html.unescape(str(cell or ""))).strip().lower()
    t = format(n.normalize(), "f")
    return t[:-2] if t.endswith(".0") else t


_DATE_RES = (re.compile(r"^\d{1,2}[-/. ]\d{1,2}[-/. ]\d{2,4}$"),
             re.compile(r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}$"),
             re.compile(r"^\d{1,2}[- /]?[A-Za-z]{3,9}[- /,]*\d{2,4}$"),
             re.compile(r"^[A-Za-z]{3,9}[- /.]\d{1,2}[, -]+\d{2,4}$"))


def looks_like_date(cell: str) -> bool:
    s = html.unescape(str(cell or "")).strip()
    return bool(s) and any(r.match(s) for r in _DATE_RES)


# ── parsing ───────────────────────────────────────────────────────────────────────────────

def _split_pipe_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith("\\|"):
        s = s[:-1]
    cells, cur, esc = [], [], False
    for ch in s:
        if esc:
            cur.append(ch)
            esc = False
        elif ch == "\\":
            esc = True
            cur.append(ch)
        elif ch == "|":
            cells.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    cells.append("".join(cur).strip())
    return [c.replace("\\|", "|") for c in cells]


def parse_pipe_tables(md: str) -> list[Table]:
    lines = md.split("\n")
    offsets, pos = [], 0
    for ln in lines:
        offsets.append(pos)
        pos += len(ln) + 1
    out: list[Table] = []
    i = 0
    while i < len(lines):
        if "|" in lines[i] and i + 1 < len(lines) and _SEP_RE.match(lines[i + 1]) and "|" in lines[i + 1] + lines[i]:
            head = _split_pipe_row(lines[i])
            rows = [head]
            j = i + 2
            while j < len(lines) and "|" in lines[j] and lines[j].strip():
                rows.append(_split_pipe_row(lines[j]))
                j += 1
            width = max(len(r) for r in rows)
            rows = [r + [""] * (width - len(r)) for r in rows]
            out.append(Table(rows, 1, offsets[i], min(pos, offsets[j - 1] + len(lines[j - 1]) + 1), "pipe"))
            i = j
        else:
            i += 1
    return out


class _HtmlTables(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[dict] = []
        self.stack: list[dict] = []
        self.cell: dict | None = None
        self.row: list | None = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "table":
            t = {"rows": [], "th_rows": [], "start": self.getpos()}
            self.stack.append(t)
        elif tag == "tr" and self.stack:
            self.row = []
            self.stack[-1]["_row_th"] = True
        elif tag in ("td", "th") and self.stack and self.row is not None:
            def num(v):
                try:
                    return max(1, int(v))
                except (TypeError, ValueError):
                    return 1
            self.cell = {"text": [], "rs": num(a.get("rowspan")), "cs": num(a.get("colspan")), "th": tag == "th"}
        elif tag == "br" and self.cell is not None:
            self.cell["text"].append(" ")

    def handle_data(self, data):
        if self.cell is not None:
            self.cell["text"].append(data)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None and self.row is not None:
            self.row.append(self.cell)
            self.cell = None
        elif tag == "tr" and self.row is not None and self.stack:
            t = self.stack[-1]
            t["rows"].append(self.row)
            self.row = None
        elif tag == "table" and self.stack:
            self.tables.append(self.stack.pop())


def _expand_html(raw_rows: list[list[dict]]) -> tuple[list[list[str]], int]:
    grid: list[list[str | None]] = []
    header_rows = 0
    for r, cells in enumerate(raw_rows):
        while len(grid) <= r:
            grid.append([])
        c = 0
        for cell in cells:
            while c < len(grid[r]) and grid[r][c] is not None:
                c += 1
            text = re.sub(r"\s+", " ", "".join(cell["text"])).strip()
            for dr in range(cell["rs"]):
                while len(grid) <= r + dr:
                    grid.append([])
                row = grid[r + dr]
                while len(row) < c + cell["cs"]:
                    row.append(None)
                for dc in range(cell["cs"]):
                    row[c + dc] = text if (dr == 0 and dc == 0) else ""
            c += cell["cs"]
        if cells and all(x["th"] for x in cells) and header_rows == r:
            header_rows += 1
    width = max((len(r) for r in grid), default=0)
    rows = [[(x if x is not None else "") for x in r] + [""] * (width - len(r)) for r in grid]
    return rows, header_rows


def parse_html_tables(md: str) -> list[Table]:
    out: list[Table] = []
    for m in re.finditer(r"<table\b.*?</table>", md, flags=re.S | re.I):
        p = _HtmlTables()
        try:
            p.feed(m.group(0))
            p.close()
        except Exception:  # noqa: BLE001 - broken HTML from a model: skip this table
            continue
        if not p.tables:
            continue
        raw = p.tables[-1]["rows"] if len(p.tables) == 1 else max(p.tables, key=lambda t: len(t["rows"]))["rows"]
        rows, nh = _expand_html(raw)
        if rows:
            out.append(Table(rows, nh or 1, m.start(), m.end(), "html"))
    return out


def find_tables(md: str) -> list[Table]:
    """Every table in *md* (pipe and HTML), in reading order."""
    tabs = parse_pipe_tables(md) + parse_html_tables(md)
    tabs.sort(key=lambda t: t.start)
    return tabs




def _pipe_cell_spans(line: str) -> list[tuple[int, int]]:
    """(start, end) offsets of each cell's text in a pipe row, between its separators."""
    n = len(line)
    i = 0
    while i < n and line[i] in " \t":
        i += 1
    start = i + 1 if i < n and line[i] == "|" else i
    end = n
    j = n
    while j > 0 and line[j - 1] in " \t":
        j -= 1
    if j > start and line[j - 1] == "|" and not (j > 1 and line[j - 2] == "\\"):
        end = j - 1
    spans, cur, esc = [], start, False
    for pos in range(start, end):
        ch = line[pos]
        if esc:
            esc = False
        elif ch == "\\":
            esc = True
        elif ch == "|":
            spans.append((cur, pos))
            cur = pos + 1
    spans.append((cur, end))
    return spans


def replace_cell(md: str, table_index: int, row: int, col: int, new: str) -> str | None:
    """*md* with the cell (*row*, *col*) of table number *table_index* (reading order, rows counted
    as in ``Table.rows``, headers first) replaced by the text *new*.  Only that cell changes.
    Returns None when the cell cannot be addressed safely: a table with merged cells (row/colspan),
    a ragged row, a position that does not exist."""
    tabs = find_tables(md)
    if not 0 <= table_index < len(tabs):
        return None
    t = tabs[table_index]
    if not (0 <= row < len(t.rows) and 0 <= col < t.width):
        return None
    new = re.sub(r"\s+", " ", str(new)).strip()
    seg = md[t.start:t.end]
    if t.kind == "pipe":
        lines = seg.split("\n")
        k = 0 if row == 0 else row + 1
        if k >= len(lines):
            return None
        spans = _pipe_cell_spans(lines[k])
        if len(spans) != t.width:
            return None
        a, b = spans[col]
        old_cell = lines[k][a:b]
        pad_l = old_cell[: len(old_cell) - len(old_cell.lstrip())] or " "
        pad_r = old_cell[len(old_cell.rstrip()):] or " "
        lines[k] = lines[k][:a] + pad_l + new.replace("|", "\\|") + pad_r + lines[k][b:]
        return md[:t.start] + "\n".join(lines) + md[t.end:]
    if re.search(r"\b(rowspan|colspan)\s*=\s*[\"']?(?!1\b)\d", seg, flags=re.I):
        return None
    row_spans = [m for m in re.finditer(r"<tr\b[^>]*>.*?</tr>", seg, flags=re.S | re.I)]
    if len(row_spans) != len(t.rows) or len(re.findall(r"<table\b", seg, flags=re.I)) != 1:
        return None
    rm = row_spans[row]
    cells = list(re.finditer(r"(<t[dh]\b[^>]*>)(.*?)(</t[dh]>)", rm.group(0), flags=re.S | re.I))
    if len(cells) != t.width:
        return None
    cm = cells[col]
    body = html.escape(new, quote=False)
    cell_new = cm.group(1) + body + cm.group(3)
    row_new = rm.group(0)[:cm.start()] + cell_new + rm.group(0)[cm.end():]
    seg2 = seg[:rm.start()] + row_new + seg[rm.end():]
    return md[:t.start] + seg2 + md[t.end:]


def to_markdown(t: Table) -> str:
    """A pipe table for *t* (cells escaped)."""
    w = t.width or 1
    rows = [r + [""] * (w - len(r)) for r in t.rows] or [[""] * w]
    esc = lambda c: str(c).replace("|", "\\|").replace("\n", " ")        # noqa: E731
    head = rows[t.n_header - 1] if t.n_header else [""] * w
    lines = ["| " + " | ".join(esc(c) for c in head) + " |", "|" + "|".join(["---"] * w) + "|"]
    for r in rows[t.n_header:]:
        lines.append("| " + " | ".join(esc(c) for c in r) + " |")
    if t.n_header > 1:                        # keep extra header rows as body rows (GFM has one)
        extra = [("| " + " | ".join(esc(c) for c in r) + " |") for r in rows[: t.n_header - 1]]
        lines = [lines[0], lines[1], *extra, *lines[2:]]
    return "\n".join(lines)


_HTML_TABLE_RE = re.compile(r"<table\b.*?</table>", re.S | re.I)


def _plain_html_table(raw: str) -> Table | None:
    """The table of one ``<table>...</table>`` text, or None when it cannot be written as a pipe table
    without losing something: merged cells (a pipe table has none), a table inside a table, or markup
    that does not parse."""
    if len(re.findall(r"<table\b", raw, flags=re.I)) != 1:
        return None
    p = _HtmlTables()
    try:
        p.feed(raw)
        p.close()
    except Exception:  # noqa: BLE001 - broken HTML from a model: leave it as it is
        return None
    if len(p.tables) != 1 or not p.tables[0]["rows"]:
        return None
    if any(c["rs"] > 1 or c["cs"] > 1 for row in p.tables[0]["rows"] for c in row):
        return None
    rows, nh = _expand_html(p.tables[0]["rows"])
    return Table(rows, nh or 1, kind="html") if rows and any(c for r in rows for c in r) else None


def normalize_html_tables(md: str) -> tuple[str, int]:
    """*md* with every HTML table that has no merged cells written as a Markdown pipe table, and how many
    were changed.  A document VLM is asked for HTML tables (they can carry merged cells) but one table
    format is what the chunker, the viewers and the search results handle best; docling writes pipe
    tables.  Cell text is kept exactly (a ``|`` inside a cell is escaped); a table with merged cells,
    nested tables, an unclosed table and everything outside the tables are left untouched.  Idempotent."""
    spans = []
    for m in _HTML_TABLE_RE.finditer(md):
        t = _plain_html_table(m.group(0))
        if t is not None:
            spans.append((m.start(), m.end(), to_markdown(t)))
    if not spans:
        return md, 0
    out, last = [], 0
    for start, end, pipe in spans:
        before, after = md[last:start], md[end:]
        out.append(before)
        lead = "" if (not (md[:start].strip()) or md[:start].endswith("\n\n")) else ("\n" if md[:start].endswith("\n") else "\n\n")
        out.append(lead + pipe)
        trail = len(after) - len(after.lstrip("\n"))
        out.append("" if not after.strip() and trail else "\n" * max(0, 2 - trail) if after else "\n")
        last = end
    out.append(md[last:])
    return "".join(out), len(spans)


def join_split_pipe_tables(md: str) -> tuple[str, int]:
    """*md* with the blank lines removed that cut a pipe table in two, and how many places were joined.

    A reader sometimes writes the header lines of a table, a blank line, and then the data rows with no
    header of their own: the parser (and the chunker, which splits blocks at blank lines) then sees a
    table of three header rows and, apart from it, loose lines -- so the arithmetic of the statement is
    never checked.  Rows that follow a table after a blank line, start with ``|``, are not a separator row
    and have the header's number of cells (or up to two fewer: a reader leaves out trailing empty cells)
    belong to it.  A second table (it has its own separator row) is left alone.  Idempotent."""
    lines = md.split("\n")
    out: list[str] = []
    joined = 0
    width = 0                                   # cells per row of the table just above, 0 = not in a table
    pending: list[str] = []                     # blank lines seen since that table's last row
    for i, ln in enumerate(lines):
        s = ln.strip()
        is_row = s.startswith("|") and s.count("|") >= 2
        starts_a_table = i + 1 < len(lines) and bool(_SEP_RE.match(lines[i + 1])) and "|" in lines[i + 1]
        if (is_row and width and pending and not _SEP_RE.match(ln) and not starts_a_table
                and max(2, width - 2) <= len(_split_pipe_row(ln)) <= width):
            pending = []                        # a continuation: drop the blank lines between
            joined += 1
            out.append(ln)
        elif is_row:
            out.extend(pending)
            pending = []
            if _SEP_RE.match(ln):               # a separator row: the line above it is this table's header
                width = len(_split_pipe_row(out[-1])) if out and out[-1].strip().startswith("|") else 0
            out.append(ln)
        elif not s:
            if width:
                pending.append(ln)
            else:
                out.append(ln)
        else:
            out.extend(pending)
            pending = []
            width = 0
            out.append(ln)
    out.extend(pending)
    return "\n".join(out), joined


def table_text(t: Table) -> str:
    return " ".join(c for r in t.rows for c in r if c)


def plain_text(md: str) -> str:
    """Page Markdown as plain text for text metrics: tags, markers and table furniture removed,
    whitespace collapsed."""
    s = re.sub(r"<!--.*?-->", " ", md, flags=re.S)
    s = re.sub(r"</?(table|thead|tbody|tr|td|th|br|p|div|span|b|i|u)[^>]*>", " ", s, flags=re.I)
    s = html.unescape(s)
    # line-local whitespace only: with \s* a run of blank lines (a runaway reading) backtracked for hours
    s = re.sub(r"^[ \t]*\|?[ \t]*:?-{2,}:?[ \t]*(?:\|[ \t]*:?-{2,}:?[ \t]*)*\|?[ \t]*$", " ", s, flags=re.M)
    s = re.sub(r"[|#*_`>~]", " ", s)
    return re.sub(r"\s+", " ", s).strip()
