"""Page Markdown: one string per page, joined with ``<!-- page N -->`` markers (stdlib only).

This is the intermediate every reader produces and the chunker consumes (the chunker finds the
page of a chunk from these markers), so splitting and joining live in one place.
"""

from __future__ import annotations

import re
from typing import Mapping

MARKER_RE = re.compile(r"^[ \t]*<!--\s*page\s+(\d+)\s*-->[ \t]*$", re.M | re.I)


def marker(page: int) -> str:
    return f"<!-- page {int(page)} -->"


def split_pages(md: str) -> dict[int, str]:
    """``{page: markdown}`` from a marked-up document.  Text before the first marker is page 1; a
    document without markers is one page; a page number seen twice keeps the later text."""
    marks = list(MARKER_RE.finditer(md or ""))
    if not marks:
        text = (md or "").strip()
        return {1: text} if text else {}
    out: dict[int, str] = {}
    head = md[: marks[0].start()].strip()
    if head:
        out[1] = head
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(md)
        out[int(m.group(1))] = md[m.end():end].strip()
    return out


def join_pages(pages: Mapping[int, str], *, keep_empty: bool = False) -> str:
    """Marked-up document from ``{page: markdown}`` in page order; pages without text are left
    out unless *keep_empty*."""
    parts: list[str] = []
    for n in sorted(pages):
        text = (pages[n] or "").strip()
        if text or keep_empty:
            parts.append(marker(n))
            if text:
                parts.append(text)
    return ("\n\n".join(parts) + "\n") if parts else ""
