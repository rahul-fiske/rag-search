"""Measures of how well a page was read (stdlib only).

Used by the benchmark (``bench.py``): a *prediction* (the Markdown a reader produced for one page)
is compared with a *truth* (Markdown a person checked).

* ``cer``         character error rate on the plain text (tables and Markdown furniture removed)
* ``cells``       numeric table cells: ``exact`` (same row, same column, same value), ``bag`` (same
                  value anywhere in the table), ``total`` (numeric cells in the truth)
* ``table_sim``   similarity of the table structure and cell texts (1 = identical).  Not the
                  published TEDS (no tree edit distance): the cells in reading order, with row ends
                  marked, are compared as a sequence, which punishes the same mistakes cheaply
* ``balance``     do the running-balance / totals validators hold on the predicted tables
* ``queries``     share of the search phrases that a plain text search would find
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

from . import tables, validators


def levenshtein(a: Sequence, b: Sequence) -> int:
    """Edit distance (insert, delete, substitute) between two strings or sequences.  Bit-parallel
    (Myers / Hyyrö) on Python integers: a 6,000-character page takes milliseconds."""
    if len(a) < len(b):
        a, b = b, a
    m = len(b)
    if m == 0:
        return len(a)
    peq: dict = {}
    for i, ch in enumerate(b):
        peq[ch] = peq.get(ch, 0) | (1 << i)
    mask = (1 << m) - 1
    last = 1 << (m - 1)
    pv, mv, score = mask, 0, m
    for ch in a:
        eq = peq.get(ch, 0)
        xv = eq | mv
        xh = (((eq & pv) + pv) ^ pv) | eq
        ph = (mv | ~(xh | pv)) & mask
        mh = pv & xh
        if ph & last:
            score += 1
        elif mh & last:
            score -= 1
        ph = ((ph << 1) | 1) & mask
        mh = (mh << 1) & mask
        pv = (mh | ~(xv | ph)) & mask
        mv = ph & xv
    return score


def cer(pred_md: str, truth_md: str) -> float:
    """Character error rate of the plain text, 0 (identical) to 1 (nothing right)."""
    t = tables.plain_text(truth_md)
    p = tables.plain_text(pred_md)
    if not t:
        return 0.0 if not p else 1.0
    return min(1.0, levenshtein(p, t) / len(t))


# ── tables ────────────────────────────────────────────────────────────────────────────────

def _numeric_bag(t: tables.Table) -> Counter:
    return Counter(tables.norm_number_text(c) for r in t.rows for c in r if tables.is_numeric(c))


def _row_key(row: list[str]) -> str:
    return "|".join(tables.norm_number_text(c) for c in row if c.strip() and not tables.is_numeric(c))


def _best_match(truth: tables.Table, preds: list[tables.Table]) -> tables.Table | None:
    bag = _numeric_bag(truth)
    best, best_score = None, 0
    for p in preds:
        score = sum((bag & _numeric_bag(p)).values())
        if score > best_score:
            best, best_score = p, score
    if best is None and preds:                       # no number in common: the closest in size
        best = min(preds, key=lambda p: abs(len(p.rows) - len(truth.rows)))
    return best


def cell_scores(pred: list[tables.Table], truth: list[tables.Table]) -> dict[str, int]:
    """Numeric cells of the truth tables: how many were read exactly (same place) and how many
    were read at all (same value somewhere in the matching table)."""
    total = exact = bag_hit = 0
    for t in truth:
        n_truth = sum(1 for r in t.rows for c in r if tables.is_numeric(c))
        total += n_truth
        if not n_truth:
            continue
        p = _best_match(t, pred)
        if p is None:
            continue
        bag_hit += sum((_numeric_bag(t) & _numeric_bag(p)).values())
        by_key: dict[str, list[int]] = {}
        for i, r in enumerate(p.rows):
            by_key.setdefault(_row_key(r), []).append(i)
        used: set[int] = set()
        truth_keys = {_row_key(r) for r in t.rows} - {""}
        for i, row in enumerate(t.rows):
            if not any(tables.is_numeric(c) for c in row):
                continue
            key = _row_key(row)
            cand = [j for j in by_key.get(key, []) if j not in used] if key else []
            if cand:
                j = cand[0]
            elif i < len(p.rows) and i not in used and _row_key(p.rows[i]) not in truth_keys:
                j = i                     # same place, text misread: still the same row
            else:
                j = None
            if j is None:
                continue
            used.add(j)
            prow = p.rows[j]
            for c, cell in enumerate(row):
                if tables.is_numeric(cell) and c < len(prow) and \
                        tables.norm_number_text(prow[c]) == tables.norm_number_text(cell):
                    exact += 1
    return {"total": total, "exact": exact, "bag": bag_hit}


def _serialise(t: tables.Table) -> list[str]:
    out: list[str] = []
    for r in t.rows:
        out.extend(tables.norm_number_text(c) for c in r)
        out.append("\x1e")                             # end of row
    return out


def table_sim(pred: list[tables.Table], truth: list[tables.Table]) -> float | None:
    """Mean over the truth tables of 1 - edit distance / length of their cell sequences; None when
    the truth has no table."""
    if not truth:
        return None
    scores = []
    for t in truth:
        p = _best_match(t, pred)
        a = _serialise(t)
        if p is None:
            scores.append(0.0)
            continue
        b = _serialise(p)
        scores.append(max(0.0, 1.0 - levenshtein(b, a) / max(1, len(a))))
    return sum(scores) / len(scores)


def balance_check(pred: list[tables.Table]) -> dict[str, int]:
    """Validators on the predicted tables: ``applicable`` (tables a validator recognised), ``ok``
    (of those, the invariant held) and ``violations``."""
    app = ok = viol = 0
    for t in pred:
        for r in validators.validate_table(t):
            if r.get("applicable"):
                app += 1
                ok += 1 if r.get("ok") else 0
                viol += len(r.get("violations", []))
    return {"applicable": app, "ok": ok, "violations": viol}


# ── search phrases ────────────────────────────────────────────────────────────────────────

def _tokens(text: str) -> list[str]:
    s = tables.plain_text(text).lower()
    return [tables.norm_number_text(w.strip(".,;:()[]{}'\"")) for w in s.split() if w.strip(".,;:()[]{}'\"")]


def query_hits(pred_md: str, queries: Sequence[str]) -> dict[str, int]:
    """How many of *queries* a plain-text search of the prediction would find: the query's tokens
    (numbers compared by value) must follow each other in the page."""
    toks = _tokens(pred_md)
    hit = 0
    for q in queries:
        qt = _tokens(q)
        if not qt:
            continue
        n = len(qt)
        if any(toks[i:i + n] == qt for i in range(0, len(toks) - n + 1)):
            hit += 1
    return {"total": sum(1 for q in queries if _tokens(q)), "hit": hit}


# ── one page ──────────────────────────────────────────────────────────────────────────────

def score_page(pred_md: str, truth_md: str, queries: Sequence[str] = (),
               truth_tables: list[list[list[str]]] | None = None) -> dict[str, Any]:
    """All measures for one page.  *truth_tables* overrides the tables parsed from *truth_md*
    (rows of cell strings, first row = header)."""
    p_tabs = tables.find_tables(pred_md)
    if truth_tables:
        t_tabs = [tables.Table([list(map(str, r)) for r in rows]) for rows in truth_tables if rows]
    else:
        t_tabs = tables.find_tables(truth_md)
    out: dict[str, Any] = {
        "cer": round(cer(pred_md, truth_md), 4),
        "cells": cell_scores(p_tabs, t_tabs),
        "table_sim": None,
        "balance": balance_check(p_tabs),
        "queries": query_hits(pred_md, queries),
        "chars": len(tables.plain_text(pred_md)),
    }
    sim = table_sim(p_tabs, t_tabs)
    out["table_sim"] = round(sim, 4) if sim is not None else None
    return out


def _ratio(num: float, den: float) -> float | None:
    return round(num / den, 4) if den else None


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summary of page scores (each row: the ``score_page`` result plus ``seconds``): means for
    CER and table similarity, pooled ratios for cells, balance and queries."""
    cers = [r["cer"] for r in rows if r.get("cer") is not None]
    sims = [r["table_sim"] for r in rows if r.get("table_sim") is not None]
    cells_total = sum(r["cells"]["total"] for r in rows)
    app = sum(r["balance"]["applicable"] for r in rows)
    q_total = sum(r["queries"]["total"] for r in rows)
    secs = [r["seconds"] for r in rows if r.get("seconds") is not None]
    return {
        "pages": len(rows),
        "cer": round(sum(cers) / len(cers), 4) if cers else None,
        "cell_exact": _ratio(sum(r["cells"]["exact"] for r in rows), cells_total),
        "cell_bag": _ratio(sum(r["cells"]["bag"] for r in rows), cells_total),
        "cells": cells_total,
        "table_sim": round(sum(sims) / len(sims), 4) if sims else None,
        "balance_ok": _ratio(sum(r["balance"]["ok"] for r in rows), app),
        "balance_tables": app,
        "query_hit": _ratio(sum(r["queries"]["hit"] for r in rows), q_total),
        "queries": q_total,
        "s_per_page": round(sum(secs) / len(secs), 3) if secs else None,
    }

