"""A small, safe Markdown -> HTML renderer for the dashboard's Help and Architecture tabs.

Handles what README.md / ARCHITECTURE.md use: headings, paragraphs, fenced code, pipe tables,
nested bullet/numbered lists, block quotes, rules, and inline code / bold / italic / links.
All text is HTML-escaped; only http(s) and in-page (#) links are kept as links.
"""

from __future__ import annotations

import html
import re

_FENCE = re.compile(r"^\s*```\s*([\w+-]*)\s*$")
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_BULLET = re.compile(r"^(\s*)([-*+])\s+(.*)$")
_NUMBER = re.compile(r"^(\s*)(\d+)[.)]\s+(.*)$")
_RULE = re.compile(r"^\s*([-*_])\s*(\1\s*){2,}$")
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_CODE = re.compile(r"`([^`\n]+)`")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<![\w*])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![\w*])")


def slug(text: str) -> str:
    s = re.sub(r"[`*_\[\]()]", "", text).strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    return re.sub(r"[\s]+", "-", s).strip("-") or "section"


def _inline(text: str) -> str:
    """Escape *text*, then turn code spans, bold, italic and safe links into tags."""
    codes: list[str] = []

    def stash(m: re.Match) -> str:
        codes.append(f"<code>{html.escape(m.group(1), quote=False)}</code>")
        return f"\x00{len(codes) - 1}\x00"

    text = _CODE.sub(stash, text)
    text = html.escape(text, quote=False)

    def link(m: re.Match) -> str:
        label, url = m.group(1), html.unescape(m.group(2))
        if url.startswith(("http://", "https://")):
            return (f'<a href="{html.escape(url)}" target="_blank" rel="noopener noreferrer">'
                    f"{label}</a>")
        if url.startswith("#"):
            return f'<a href="{html.escape(url)}">{label}</a>'
        return label  # relative file links point outside the dashboard: keep the words only

    text = _LINK.sub(link, text)
    text = _BOLD.sub(r"<strong>\1</strong>", text)
    text = _ITALIC.sub(r"<em>\1</em>", text)
    return re.sub(r"\x00(\d+)\x00", lambda m: codes[int(m.group(1))], text)


def _cells(row: str) -> list[str]:
    row = row.strip()
    if row.startswith("|"):
        row = row[1:]
    if row.endswith("|") and not row.endswith("\\|"):
        row = row[:-1]
    return [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", row)]


def _list(lines: list[str], i: int) -> tuple[str, int]:
    """Render the list starting at lines[i]; returns (html, next index).  Nesting by indent."""
    def item(line: str):
        m = _BULLET.match(line) or _NUMBER.match(line)
        if not m:
            return None
        return len(m.group(1).expandtabs(4)), ("ul" if _BULLET.match(line) else "ol"), m.group(3)

    def build(i: int, indent: int) -> tuple[str, int]:
        first = item(lines[i])
        tag = first[1]
        out = [f"<{tag}>"]
        while i < len(lines):
            it = item(lines[i])
            if it is None:
                if lines[i].strip() and lines[i].startswith(" " * (indent + 2)) and out[-1] != f"<{tag}>":
                    out[-1] = out[-1][:-5] + " " + _inline(lines[i].strip()) + "</li>"   # wrapped line
                    i += 1
                    continue
                break
            ind, t, body = it
            if ind < indent or (ind == indent and t != tag):
                break
            if ind > indent:
                sub, i = build(i, ind)
                out[-1] = out[-1][:-5] + sub + "</li>"
                continue
            out.append(f"<li>{_inline(body)}</li>")
            i += 1
        out.append(f"</{tag}>")
        return "".join(out), i

    return build(i, item(lines[i])[0])


def render(md: str) -> str:
    lines = md.replace("\r\n", "\n").split("\n")
    out: list[str] = []
    para: list[str] = []

    def flush() -> None:
        if para:
            out.append("<p>" + _inline(" ".join(s.strip() for s in para)) + "</p>")
            para.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        fm = _FENCE.match(line)
        if fm:
            flush()
            j = i + 1
            code: list[str] = []
            while j < len(lines) and not _FENCE.match(lines[j]):
                code.append(lines[j])
                j += 1
            lang = f' class="lang-{html.escape(fm.group(1))}"' if fm.group(1) else ""
            out.append(f"<pre><code{lang}>{html.escape(chr(10).join(code), quote=False)}</code></pre>")
            i = j + 1
            continue
        if not line.strip() or (line.strip().startswith("<!--") and line.strip().endswith("-->")):
            flush()                          # blank lines and one-line HTML comments
            i += 1
            continue
        hm = _HEADING.match(line)
        if hm:
            flush()
            n = len(hm.group(1))
            out.append(f'<h{n} id="{slug(hm.group(2))}">{_inline(hm.group(2))}</h{n}>')
            i += 1
            continue
        if _RULE.match(line):
            flush()
            out.append("<hr>")
            i += 1
            continue
        if "|" in line and i + 1 < len(lines) and _TABLE_SEP.match(lines[i + 1]) and "-" in lines[i + 1]:
            flush()
            head = _cells(line)
            j = i + 2
            rows = []
            while j < len(lines) and "|" in lines[j] and lines[j].strip():
                rows.append(_cells(lines[j]))
                j += 1
            t = ["<div class=\"table-wrap\"><table><thead><tr>"]
            t += [f"<th>{_inline(c)}</th>" for c in head]
            t.append("</tr></thead><tbody>")
            for r in rows:
                t.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>")
            t.append("</tbody></table></div>")
            out.append("".join(t))
            i = j
            continue
        if line.lstrip().startswith(">"):
            flush()
            quote = []
            while i < len(lines) and lines[i].lstrip().startswith(">"):
                quote.append(lines[i].lstrip()[1:].lstrip())
                i += 1
            out.append("<blockquote>" + render("\n".join(quote)) + "</blockquote>")
            continue
        if _BULLET.match(line) or _NUMBER.match(line):
            flush()
            block, i = _list(lines, i)
            out.append(block)
            continue
        para.append(line)
        i += 1
    flush()
    return "\n".join(out)


def headings(md: str) -> list[dict]:
    """[{level, title, id}] for the table of contents (fenced code is skipped)."""
    out, fenced = [], False
    for line in md.replace("\r\n", "\n").split("\n"):
        if _FENCE.match(line):
            fenced = not fenced
            continue
        m = None if fenced else _HEADING.match(line)
        if m and len(m.group(1)) <= 3:
            title = re.sub(r"[`*]", "", m.group(2))
            out.append({"level": len(m.group(1)), "title": title, "id": slug(m.group(2))})
    return out
