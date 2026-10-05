"""Domain validators for tables (stdlib only): does the arithmetic of the table hold?

A validator recognises its own kind of table from the header, checks an invariant and, when it
fails, says *which cells* are suspect so repair can look at a crop instead of the whole page.

* **running balance** (bank statements, passbooks): ``balance[i] = balance[i-1] + credit[i] - debit[i]``.
  One wrong amount fails one row; one wrong balance fails two consecutive rows (its own and the
  next), which is how the two are told apart.
* **totals**: a row labelled Total / Sub-total equals the sum of the rows above, per numeric column.

``Violation`` is a plain dict (JSON-able, stored in the page's trace)::

    {"validator": "running_balance", "row": 6, "col": 3, "role": "credit", "found": "2,81.00",
     "expected": "2819.00", "why": "row 6: 44,699 + 2,81.00 ≠ 47,518"}

``expected`` is a hypothesis that satisfies the invariant (what the cell would have to be); repair
only accepts a new read when it equals the hypothesis *and* an independent read agrees.
"""

from __future__ import annotations

import collections
import re
from decimal import Decimal
from typing import Any

from .tables import Table, norm_number_text, parse_number

TOL = Decimal("0.011")                    # rounding slack: amounts are to the paisa

_ROLE_PATTERNS = (
    ("balance", re.compile(r"\b(closing\s+)?balance\b|\bbal\.?\b|\bbalance\s*\(", re.I)),
    ("debit", re.compile(r"\bdebits?\b|withdraw|\bdr\.?\b|paid\s*out|\bpayments?\b|money\s*out|\bwdl\b", re.I)),
    ("credit", re.compile(r"\bcredits?\b|deposit|\bcr\.?\b|paid\s*in|\breceipts?\b|money\s*in", re.I)),
    ("amount", re.compile(r"\bamount\b|\btransaction\s*amount\b|\btxn\s*amt\b", re.I)),
    ("ref", re.compile(r"\bche?q\.?\s*(no|number)?\b|\bcheque\b|\binstrument\b|\bref(erence)?\.?\s*(no|number)?\b|चेक", re.I)),
)
_TOTAL_RE = re.compile(r"^\s*(grand\s+|sub[\s-]?|net\s+)?total\b|^\s*total\s*[:(]", re.I)


def _roles(header: list[str]) -> dict[int, str]:
    out: dict[int, str] = {}
    for c, text in enumerate(header):
        for role, rx in _ROLE_PATTERNS:
            if rx.search(text or ""):
                if role == "amount" and "balance" in (text or "").lower():
                    continue
                out[c] = role
                break
    return out


def _headerish(row: list[str]) -> bool:
    """A row of column titles: some text, and no number, date or empty-only row."""
    from .tables import looks_like_date

    cells = [c for c in row if (c or "").strip()]
    return bool(cells) and not any(_dec(c) is not None or looks_like_date(c) for c in cells)


def stack_header(t: Table) -> tuple[list[str], int]:
    """(header text per column, number of leading rows it takes).  Passbooks print their titles on two or
    three lines (Marathi, Hindi, English; "AMOUNT" over "WITHDRAWN"), but Markdown has one header row, so the
    later title lines arrive as body rows: the leading title-like rows are stacked column by column."""
    n = max(t.n_header, 1)
    if not t.rows:
        return [], 0
    k = n
    while k < min(len(t.rows), n + 3) and _headerish(t.rows[k]):
        k += 1
    w = max(len(r) for r in t.rows[:k])
    head = [" ".join((r[c] if c < len(r) else "").strip() for r in t.rows[:k]).strip() for c in range(w)]
    return head, k


def layout_problem(t: Table) -> str:
    """Why the columns of a statement-like table look wrong ("" when they do not): the balance column is
    mostly empty (its numbers slid into a neighbour), consecutive rows are exact copies, or figures sit in
    columns they do not belong in (``_column_problem``)."""
    head, h = stack_header(t)
    cols: dict[str, list[int]] = {}
    for col, role in _roles(head).items():
        cols.setdefault(role, []).append(col)
    body = t.rows[h:]
    if len(body) < 4:
        return ""
    if len(cols.get("balance", [])) == 1 and (cols.get("debit") or cols.get("credit") or cols.get("amount")):
        b = cols["balance"][0]
        used = [r for r in body if sum(1 for c in r if _dec(c) is not None) >= 2]
        if len(used) >= 4 and sum(1 for r in used if b < len(r) and _dec(r[b]) is not None) < 0.5 * len(used):
            return "the balance column is mostly empty: the numbers sit in the wrong columns"
    for a, b2 in zip(body, body[1:]):
        if a[1:] == b2[1:] and sum(1 for c in a if _dec(c) is not None) >= 2:
            return "two consecutive rows repeat each other (apart from the line number)"
    return _column_problem(body, cols)


_MONEY_RE = re.compile(r"\d[\d,]*\.\d{2}\b")
_DRCR_RE = re.compile(r"\b(dr|cr)\.?\s*$", re.I)


def _money(cell: str) -> bool:
    """An amount as a statement prints it: digits with two decimals (paise)."""
    return bool(_MONEY_RE.search(cell or "")) and _dec(cell) is not None


def _column_problem(body: list[list[str]], cols: dict[str, list[int]]) -> str:
    """Figures in columns they do not belong in -- what a reader that lost the column grid produces:
    the same amount in three or more columns of a row, an amount in the cheque / reference column, or
    balances (``... Cr``) in a debit / credit column.  Each needs several rows to agree, so one odd
    row (a balance brought forward that equals the deposit) is not enough."""
    rows = [r for r in body if sum(1 for c in r if _money(c)) >= 3]
    if len(rows) >= 3:
        same = sum(1 for r in rows if collections.Counter(
            norm_number_text(c) for c in r if _money(c)).most_common(1)[0][1] >= 3)
        if same >= max(3, 0.5 * len(rows)):
            return "the same amount appears in three or more columns of most rows: the columns are mixed up"
    for c in cols.get("ref", []):
        hits = sum(1 for r in body if c < len(r) and _money(r[c]))
        if hits >= max(3, 0.6 * len(body)):
            return "amounts sit in the cheque / reference column: the columns are shifted"
    for role in ("debit", "credit", "amount"):
        for c in cols.get(role, []):
            hits = sum(1 for r in body if c < len(r) and _money(r[c]) and _DRCR_RE.search(r[c] or ""))
            if hits >= max(3, 0.5 * len(body)):
                return f"balances (with Cr/Dr) sit in the {role} column: the columns are shifted"
    return ""


def _dec(cell: str) -> Decimal | None:
    return parse_number(cell)


def _fmt(d: Decimal) -> str:
    t = format(d.quantize(Decimal("0.01")), "f")
    return t


def _equation(prev: Decimal, row: list[str], cols: dict[str, list[int]], c: int | None = None) -> tuple[Decimal | None, bool]:
    """(expected balance, row has any amount) for one row given the previous balance."""
    credit = debit = Decimal(0)
    any_amount = False
    for col in cols.get("credit", []):
        v = _dec(row[col])
        if v is not None:
            credit += v
            any_amount = True
    for col in cols.get("debit", []):
        v = _dec(row[col])
        if v is not None:
            debit += abs(v)
            any_amount = True
    for col in cols.get("amount", []):
        v = _dec(row[col])
        if v is not None:                     # signed (or Dr/Cr suffix); positive = credit
            credit += v
            any_amount = True
    return prev + credit - debit, any_amount


def running_balance(t: Table) -> dict[str, Any]:
    """Check a statement-like table.  Returns {"name", "applicable", "ok", "checked", "violations"}."""
    res: dict[str, Any] = {"name": "running_balance", "applicable": False, "ok": True, "checked": 0,
                           "violations": []}
    head, h = stack_header(t)
    roles = _roles(head)
    cols: dict[str, list[int]] = {}
    for col, role in roles.items():
        cols.setdefault(role, []).append(col)
    if len(cols.get("balance", [])) != 1 or not (cols.get("debit") or cols.get("credit") or cols.get("amount")):
        return res
    if cols.get("amount") and (cols.get("debit") or cols.get("credit")):
        cols.pop("amount")
    bcol = cols["balance"][0]
    body = t.rows[h:]
    if len(body) < 3:
        return res
    best: dict[str, Any] | None = None
    for order in ("down", "up"):               # newest-first statements run the other way
        rows = list(range(len(body))) if order == "down" else list(range(len(body) - 1, -1, -1))
        seq = _check_sequence(body, rows, bcol, cols)
        if best is None or (seq["checked"] and seq["fail_rate"] < best["fail_rate"]):
            best = {**seq, "order": order}
        if seq["checked"] and seq["fail_rate"] == 0:
            break
    assert best is not None
    res["applicable"] = best["checked"] >= 2
    res["checked"] = best["checked"]
    res["order"] = best["order"]
    if not res["applicable"]:
        return res
    # a table that fails almost everywhere is not a running-balance table (a different layout)
    if best["fail_rate"] > 0.6 and best["checked"] >= 6:
        res["applicable"] = False
        res["note"] = "most rows fail: probably not a plain running-balance table"
        return res
    res["violations"] = [{"validator": "running_balance", **v, "row": v["row"] + h} for v in best["violations"]]
    res["ok"] = not res["violations"]
    return res


def _check_sequence(body: list[list[str]], order: list[int], bcol: int, cols: dict[str, list[int]]) -> dict[str, Any]:
    fails: dict[int, dict[str, Any]] = {}
    checked = 0
    prev_idx: int | None = None
    for pos, i in enumerate(order):
        row = body[i]
        bal = _dec(row[bcol])
        if bal is None:
            prev_idx = None if not row[bcol].strip() else prev_idx
            continue
        if prev_idx is not None:
            prev = _dec(body[prev_idx][bcol])
            assert prev is not None
            exp, any_amount = _equation(prev, row, cols)
            # Rows with no amount at all (e.g. a brought-forward line) only count when the balance moved
            if any_amount or bal != prev:
                checked += 1
                if abs(exp - bal) > TOL:
                    fails[pos] = {"row": i, "exp": exp, "bal": bal, "prev": prev, "prev_row": prev_idx,
                                  "any_amount": any_amount}
        prev_idx = i
    violations: list[dict[str, Any]] = []
    positions = sorted(fails)
    pos_set = set(positions)
    for pos in positions:
        f = fails[pos]
        row = body[f["row"]]
        nxt = pos + 1 in pos_set
        prv = pos - 1 in pos_set
        if nxt and not prv:
            # this row's balance feeds the next failing row: the balance cell itself is the suspect
            nf = fails[pos + 1]
            nrow = body[nf["row"]]
            # the balance this row would have if its own amounts are right; the one the NEXT row
            # needs is the cross-check (when both agree the balance cell is surely the culprit)
            hypo, _ = _equation(f["prev"], row, cols)
            cand, _ = _equation(Decimal(0), nrow, cols)
            agrees = abs((nf["bal"] - cand) - hypo) <= TOL
            violations.append({"row": f["row"], "col": bcol, "role": "balance", "found": row[bcol],
                               "expected": _fmt(hypo),
                               "why": f"balance in row {f['row'] + 1} breaks two consecutive rows"
                                      + ("" if agrees else " (amounts around it disagree too)")})
        elif prv and not nxt:
            continue                           # already explained by the balance cell above it
        elif prv and nxt:
            continue
        else:
            delta = f["bal"] - f["prev"]       # what the amounts should add up to
            amount_cols = [c for r in ("credit", "debit", "amount") for c in cols.get(r, [])]
            filled = [c for c in amount_cols if _dec(row[c]) is not None or row[c].strip()]
            col = filled[0] if filled else (amount_cols[0] if amount_cols else None)
            role = next((r for r in ("credit", "debit", "amount") if col in cols.get(r, [])), "amount")
            if col is None:
                continue
            others = Decimal(0)
            for r2 in ("credit", "debit", "amount"):
                for c2 in cols.get(r2, []):
                    if c2 == col:
                        continue
                    v = _dec(row[c2])
                    if v is not None:
                        others += (-abs(v) if r2 == "debit" else v)
            want = delta - others
            hypo = abs(want) if role in ("credit", "debit") else want
            if role == "credit" and want < 0:
                role_note = " (negative: the cell may belong in the debit column)"
            else:
                role_note = ""
            violations.append({"row": f["row"], "col": col, "role": role, "found": row[col],
                               "expected": _fmt(hypo),
                               "why": f"row {f['row'] + 1}: {_fmt(f['prev'])} + {row[col] or '(empty)'} "
                                      f"≠ {_fmt(f['bal'])}{role_note}"})
    rate = (len(fails) / checked) if checked else 0.0
    return {"checked": checked, "fail_rate": rate, "violations": violations}


def totals(t: Table) -> dict[str, Any]:
    """A Total row equals the sum of the rows above it, per numeric column."""
    res: dict[str, Any] = {"name": "totals", "applicable": False, "ok": True, "checked": 0, "violations": []}
    body = t.body
    for ri in range(len(body) - 1, 0, -1):
        if _TOTAL_RE.match(body[ri][0] if body[ri] else ""):
            break
    else:
        return res
    total_row = body[ri]
    # the block above the total row: back to the previous total / sub-total row (or the top)
    start = 0
    for k in range(ri - 1, -1, -1):
        if body[k] and _TOTAL_RE.match(body[k][0]):
            start = k + 1
            break
    block = body[start:ri]
    if len(block) < 2:
        return res
    for c in range(1, len(total_row)):
        tv = _dec(total_row[c])
        vals = [_dec(r[c]) for r in block if c < len(r)]
        nums = [v for v in vals if v is not None]
        if tv is None or len(nums) < 2 or len(nums) < 0.7 * len(block):
            continue
        res["applicable"] = True
        res["checked"] += 1
        s = sum(nums, Decimal(0))
        if abs(s - tv) > TOL:
            res["violations"].append({
                "validator": "totals", "row": ri + t.n_header, "col": c, "role": "total",
                "found": total_row[c], "expected": _fmt(s),
                "why": f"column {c + 1}: rows add up to {_fmt(s)}, the total says {total_row[c]}"})
    res["ok"] = not res["violations"]
    return res


VALIDATORS = (running_balance, totals)


def validate_table(t: Table) -> list[dict[str, Any]]:
    return [v(t) for v in VALIDATORS]


def violations_of(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [v for r in results for v in r.get("violations", [])]


def page_violations(md: str) -> list[dict[str, Any]]:
    """Every suspect cell of every table in a page's Markdown; each carries ``table`` (the table's
    number in reading order)."""
    from .tables import find_tables

    out: list[dict[str, Any]] = []
    for ti, t in enumerate(find_tables(md)):
        for res in validate_table(t):
            if res.get("applicable") and not res.get("ok"):
                out.extend({**v, "table": ti} for v in res["violations"])
    return out


def recheck_with(t: Table, row: int, col: int, value: str) -> dict[str, Any]:
    """Run every validator on a copy of *t* with one cell replaced.  Used by repair: a candidate
    read fixes the table when the cell is no longer a suspect and no new suspect appeared."""
    rows = [list(r) for r in t.rows]
    rows[row][col] = value
    t2 = Table(rows, t.n_header, kind=t.kind)
    before = {(v["row"], v["col"], v["validator"]) for v in violations_of(validate_table(t))}
    after = {(v["row"], v["col"], v["validator"]) for v in violations_of(validate_table(t2))}
    gone = not any(r == row and c == col for r, c, _ in after)
    return {"fixed": gone and after <= before and len(after) < len(before) or (gone and not after),
            "before": len(before), "after": len(after), "value": norm_number_text(value)}
