"""Markdown -> chunks with page + heading metadata.  Pure stdlib.

Chunk sizes are *estimated* tokens (max of words*1.3 and chars/4) rather than a
real tokenizer's count; the embedder's window is larger than chunk_size so
estimates never truncate.  Tables that must be split repeat their header rows.
"""

from __future__ import annotations

import re
from typing import Any

from ..spec import CHUNKER_VERSION  # noqa: E402,F401  (re-exported)
_PAGE_RE = re.compile(r"<!-- page (\d+) -->")
_HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def est_tokens(s: str) -> int:
    if not s:
        return 0
    return max(int(len(s.split()) * 1.3) + 1, len(s) // 4)


def _blocks(segment: str) -> list[str]:
    """Split into paragraph-ish blocks, keeping fenced code and tables whole."""
    blocks: list[str] = []
    cur: list[str] = []
    in_fence = False
    for line in segment.split("\n"):
        if line.strip().startswith("```"):
            in_fence = not in_fence
        if not in_fence and not line.strip():
            if cur:
                blocks.append("\n".join(cur))
                cur = []
            continue
        cur.append(line)
    if cur:
        blocks.append("\n".join(cur))
    return blocks


def _split_words(text: str, size: int) -> list[str]:
    words = text.split()
    out, cur, n = [], [], 0
    for w in words:
        wt = est_tokens(w)
        if cur and n + wt > size:
            out.append(" ".join(cur))
            cur, n = [], 0
        cur.append(w)
        n += wt
    if cur:
        out.append(" ".join(cur))
    return out


def _split_table(block: str, size: int) -> list[str]:
    lines = block.split("\n")
    header = lines[:2] if len(lines) > 2 and set(lines[1].strip()) <= set("|-: ") else []
    body = lines[len(header):]
    head_txt = "\n".join(header)
    out, cur, n = [], [], est_tokens(head_txt)
    for ln in body:
        lt = est_tokens(ln)
        if lt > size:  # one enormous row
            if cur:
                out.append("\n".join(header + cur))
                cur, n = [], est_tokens(head_txt)
            out.extend(_split_words(ln, size))
            continue
        if cur and n + lt > size:
            out.append("\n".join(header + cur))
            cur, n = [], est_tokens(head_txt)
        cur.append(ln)
        n += lt
    if cur:
        out.append("\n".join(header + cur))
    return out


def _pieces(block: str, size: int) -> list[str]:
    """Break one oversized block into <= size pieces."""
    if est_tokens(block) <= size:
        return [block]
    if block.lstrip().startswith("|"):
        return _split_table(block, size)
    sentences = _SENT_RE.split(block)
    out: list[str] = []
    cur: list[str] = []
    n = 0
    for s in sentences:
        st = est_tokens(s)
        if st > size:
            if cur:
                out.append(" ".join(cur))
                cur, n = [], 0
            out.extend(_split_words(s, size))
            continue
        if cur and n + st > size:
            out.append(" ".join(cur))
            cur, n = [], 0
        cur.append(s)
        n += st
    if cur:
        out.append(" ".join(cur))
    return out


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Greedy packing of blocks into chunks of ~chunk_size with block-level overlap."""
    pieces: list[str] = []
    for b in _blocks(text):
        pieces.extend(_pieces(b, chunk_size))
    chunks: list[str] = []
    cur: list[str] = []
    n = 0
    for p in pieces:
        pt = est_tokens(p)
        if cur and n + pt > chunk_size:
            chunks.append("\n\n".join(cur))
            # carry a tail of the previous chunk as overlap
            tail: list[str] = []
            tn = 0
            for prev in reversed(cur):
                t = est_tokens(prev)
                if tn + t > chunk_overlap:
                    break
                tail.insert(0, prev)
                tn += t
            cur, n = tail, tn
        cur.append(p)
        n += pt
    if cur:
        chunks.append("\n\n".join(cur))
    return [c for c in chunks if c.strip()]


def markdown_to_nodes(text: str, chunk_size: int, chunk_overlap: int) -> list[dict[str, Any]]:
    """Return [{"text", "page", "heading"}] for a page-annotated Markdown string."""
    nodes: list[dict[str, Any]] = []
    page = "1"
    heading = ""
    pos = 0
    segments: list[tuple[str, str]] = []  # (page, text)
    for m in _PAGE_RE.finditer(text):
        if m.start() > pos:
            segments.append((page, text[pos:m.start()]))
        page = m.group(1)
        pos = m.end()
    segments.append((page, text[pos:]))

    for page, seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        for chunk in split_text(seg, chunk_size, chunk_overlap):
            ch = _HEADING_RE.findall(chunk)
            starts_with_heading = bool(_HEADING_RE.match(chunk.lstrip().split("\n", 1)[0]))
            if starts_with_heading or not heading:
                label = ch[0].strip() if ch else heading
            else:
                label = heading  # heading in effect where the chunk begins
            if ch:
                heading = ch[-1].strip()
            nodes.append({"text": chunk, "page": page, "heading": label})
    return nodes
