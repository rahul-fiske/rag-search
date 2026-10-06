#!/usr/bin/env python3
"""Measure the cheap lanes against what the document reader produced, on real pages of an existing index.

For a sample of scanned PDF pages that already have a trace (and the document reader's Markdown next to it) it
  * takes the page-image facts and the router's decision (lane b or d, with its reasons);
  * reads the page with docling's OCR and with Tesseract (straightened when skewed) and gates both texts;
  * scores each text against the document reader's Markdown (word and number recall);
and for a sample of text pages it looks for residue regions (lane c) and reports how many pages would go to the reader.

The source files are only read.  Nothing from a page's content is printed or written: pages are named S01, S02 ...
and only numbers, check names and reasons are reported (a private mapping goes to ``mapping.json`` in the output
folder, which is not part of the repository).

    rag-search's python scripts/measure_lanes.py --scans 40 --out /tmp/lanes     (in the uv tool environment)
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

GOOD_WORDS, GOOD_NUMBERS = 0.90, 0.95          # an OCR text is as good as the reference when it holds this much of it
MIN_REF_WORDS = 20
STRUCTURE = {"table_shape", "column_types", "totals", "running_balance"}


def _tokens(md: str) -> tuple[Counter, Counter]:
    from rag_search.core.conversion import tables

    text = tables.plain_text(md).lower()
    toks = re.findall(r"[^\W_]+", text)
    words = Counter(t for t in toks if not t.isdigit() and len(t) >= 3)
    nums = Counter(re.sub(r"[,\s]", "", t) for t in re.findall(r"\d[\d,.]*\d|\d", text))
    return words, nums


def recall(ref: Counter, got: Counter) -> float | None:
    total = sum(ref.values())
    if not total:
        return None
    return sum(min(n, got.get(t, 0)) for t, n in ref.items()) / total


def score(ref_md: str, got_md: str) -> dict[str, Any]:
    rw, rn = _tokens(ref_md)
    gw, gn = _tokens(got_md)
    return {"ref_words": sum(rw.values()), "word_recall": recall(rw, gw), "number_recall": recall(rn, gn)}


def good(sc: dict[str, Any]) -> bool:
    return ((sc["word_recall"] or 0) >= GOOD_WORDS) and (sc["number_recall"] is None or sc["number_recall"] >= GOOD_NUMBERS)


def candidates(paths: Any, api: Any, only: str) -> dict[str, list[dict[str, Any]]]:
    """Scanned and text pages of PDF documents that have a trace: {"scan": [...], "text": [...]}, one dict per page."""
    from rag_search.core.conversion.trace import TRACE_SUFFIX, page_kind, read_trace

    out: dict[str, list[dict[str, Any]]] = {"scan": [], "text": []}
    root = paths.markup / only if only else paths.markup
    for f in sorted(root.rglob("*" + TRACE_SUFFIX)):
        rel = f.relative_to(paths.markup).as_posix()
        coll, _, rest = rel.partition("/")
        doc = rest[:-len(TRACE_SUFFIX)]
        data = read_trace(f)
        if not data or not doc:
            continue
        pages = data.get("pages", [])
        if not any(page_kind(p) in ("scanned", "raster", "digital", "embedded") for p in pages):
            continue
        out_doc = {"coll": coll, "doc": doc, "trace": f, "pages": pages}
        for p in pages:
            kind = page_kind(p)
            if p.get("outcome") in ("pass", "repaired", "low") and kind in ("scanned", "raster"):
                out["scan"].append({**out_doc, "page": int(p["page"]), "rec": p})
            elif kind in ("digital", "embedded") and p.get("outcome") in ("pass", "repaired", "low"):
                out["text"].append({**out_doc, "page": int(p["page"]), "rec": p})
    return out


def pick(items: list[dict[str, Any]], n: int, rng: random.Random) -> list[dict[str, Any]]:
    """*n* pages spread over as many documents as possible."""
    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for it in items:
        by_doc[f"{it['coll']}/{it['doc']}"].append(it)
    docs = list(by_doc)
    rng.shuffle(docs)
    out: list[dict[str, Any]] = []
    i = 0
    while len(out) < n and any(by_doc[d] for d in docs):
        d = docs[i % len(docs)]
        i += 1
        if by_doc[d]:
            out.append(by_doc[d].pop(rng.randrange(len(by_doc[d]))))
    return out


def run_scans(items: list[dict[str, Any]], paths: Any, api: Any) -> list[dict[str, Any]]:
    from rag_search.core.conversion import gate, pagemd, router, routed, scanfacts, tesseract

    reader = routed.DoclingReader()
    caps = {"ocr": True, "deskew": tesseract.usable()}
    rows: list[dict[str, Any]] = []
    for i, it in enumerate(items, 1):
        sid = f"S{i:02d}"
        idx = paths.index.joinpath(it["coll"], *it["doc"].split("/"))
        src = api._doc_source(paths, it["coll"], it["doc"], idx, it["trace"])
        row: dict[str, Any] = {"id": sid, "outcome": it["rec"].get("outcome")}
        rows.append(row)
        if src is None or src.suffix.lower() != ".pdf":
            row["skip"] = "no source"
            continue
        md_path = it["trace"].with_name(it["trace"].name[:-len(".trace.json")] + ".md")
        try:
            ref = pagemd.split_pages(md_path.read_text(encoding="utf-8", errors="replace")).get(it["page"], "")
        except OSError:
            row["skip"] = "no reference"
            continue
        if sum(_tokens(ref)[0].values()) < MIN_REF_WORDS:
            row["skip"] = "reference too short"
            continue
        prof = dict(it["rec"].get("profile") or {})
        t0 = time.perf_counter()
        facts = scanfacts.page_facts(src, it["page"])
        row["facts_s"] = round(time.perf_counter() - t0, 3)
        row["facts"] = facts
        lane, why = router.decide_scan(facts, prof, caps)
        row.update(lane=lane, reasons=why, engine=router.b_engine(facts, caps) if lane == "b" else "")
        row["ref_words"] = sum(_tokens(ref)[0].values())
        md_d = md_t = None
        try:
            t0 = time.perf_counter()
            res = reader.read(src, it["page"], it["page"], "scan")
            md = md_d = (res.get("pages") or {}).get(it["page"], "")
            stats = (res.get("stats") or {}).get(it["page"]) or {}
            g = gate.check_page(md, branch_kind="scan", profile=prof, confidence=stats.get("confidence"), ocr=True)
            row["docling"] = {"s": round(time.perf_counter() - t0, 2), **score(ref, md), "escalate": bool(g.get("escalate")),
                              "checks": gate.failed(g), "chars": len("".join(md.split()))}
        except Exception as exc:  # noqa: BLE001 - a page that cannot be read is a row
            row["docling"] = {"error": f"{type(exc).__name__}: {str(exc)[:80]}"}
        if tesseract.usable():
            try:
                t0 = time.perf_counter()
                md = md_t = tesseract.read_page(src, it["page"], skew=float((facts or {}).get("skew") or 0.0))
                g = gate.check_page(md, branch_kind="scan", profile=prof, ocr=True)
                row["tesseract"] = {"s": round(time.perf_counter() - t0, 2), **score(ref, md), "escalate": bool(g.get("escalate")),
                                    "checks": gate.failed(g), "chars": len("".join(md.split()))}
            except Exception as exc:  # noqa: BLE001
                row["tesseract"] = {"error": f"{type(exc).__name__}: {str(exc)[:80]}"}
        if md_d is not None and md_t is not None:
            a, b = score(md_d, md_t), score(md_t, md_d)               # each engine's words found in the other's text
            nums = [x for x in (a["number_recall"], b["number_recall"]) if x is not None]
            row["agree"] = {"words": round(min(a["word_recall"] or 0, b["word_recall"] or 0), 3),
                            "numbers": round(min(nums), 3) if nums else None}
        print(f"  {sid} {lane} {row.get('docling', {}).get('word_recall')}", file=sys.stderr, flush=True)
    return rows


def ladder(row: dict[str, Any]) -> tuple[str, bool]:
    """Where the page would end with the lane ladder (docling OCR, then Tesseract unless the page is a table, then the
    reader) and whether the text kept there is as good as the reader's."""
    d, t = row.get("docling") or {}, row.get("tesseract") or {}
    if d and "error" not in d and not d["escalate"]:
        return "docling", good(d)
    if t and "error" not in t and not t["escalate"] and not (set(d.get("checks") or []) & STRUCTURE):
        return "tesseract", good(t)
    return "reader", True


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    used = [r for r in rows if "skip" not in r and r.get("docling")]
    out: dict[str, Any] = {"pages": len(rows), "measured": len(used), "skipped": dict(Counter(r.get("skip") for r in rows if "skip" in r))}
    routed_b = [r for r in used if r["lane"] == "b"]
    out["router"] = {"b": len(routed_b), "d": len(used) - len(routed_b)}
    by: dict[str, Any] = {}
    for eng in ("docling", "tesseract"):
        have = [r for r in used if eng in r and "error" not in r[eng]]
        kept = [r for r in have if not r[eng]["escalate"]]
        by[eng] = {"read": len(have), "gate_kept": len(kept), "kept_and_good": sum(1 for r in kept if good(r[eng])),
                   "FALSE_PASS": sum(1 for r in kept if not good(r[eng]) and r["outcome"] != "low"),
                   "good_but_escalated": sum(1 for r in have if r[eng]["escalate"] and good(r[eng])),
                   "median_s": round(statistics.median([r[eng]["s"] for r in have]), 1) if have else None,
                   "median_word_recall": round(statistics.median([r[eng]["word_recall"] or 0 for r in have]), 3) if have else None}
    out["engines"] = by
    # the ladder's result on every page, whatever the router said (what OCR-first would do if the router let every page in)
    lad = Counter()
    false_pass = 0
    for r in used:
        where, ok = ladder(r)
        lad[where] += 1
        if where != "reader" and not ok and r["outcome"] != "low":
            false_pass += 1
    out["ladder_all_pages"] = {**lad, "false_pass": false_pass}
    # with the router in front: routed b pages end where?
    lr = Counter()
    fp = 0
    for r in routed_b:
        where, ok = ladder(r)
        lr[where] += 1
        if where != "reader" and not ok and r["outcome"] != "low":
            fp += 1
    out["ladder_routed_b"] = {**lr, "false_pass": fp}
    # two cheap readers that agree: accept the page without the document reader (whatever the router said)
    pol = {}
    for wmin in (0.80, 0.85, 0.90, 0.95):
        acc = [r for r in used if r.get("agree") and r["agree"]["words"] >= wmin
               and (r["agree"]["numbers"] is None or r["agree"]["numbers"] >= wmin)
               and not r["docling"]["escalate"]]
        pol[f"agree>={wmin}"] = {"accepted": len(acc), "good": sum(1 for r in acc if good(r["docling"])),
                                 "FALSE_ACCEPT": sum(1 for r in acc if not good(r["docling"]) and r["outcome"] != "low"),
                                 "of_which_router_b": sum(1 for r in acc if r["lane"] == "b")}
    out["agreement"] = pol
    # which fact keeps a page from lane b although the ladder would have kept a good text
    blocked: Counter = Counter()
    only: Counter = Counter()
    missed = 0
    for r in used:
        where, ok = ladder(r)
        if r["lane"] == "d" and where != "reader" and ok:
            missed += 1
            names = [re.sub(r"[\d.]+", "#", x) for x in r["reasons"]]
            blocked.update(set(names))
            if len(set(names)) == 1:
                only[names[0]] += 1
    out["missed_saving"] = missed
    out["blocked_by"] = dict(blocked.most_common())
    out["blocked_only_by"] = dict(only.most_common())
    # time: reader-only baseline against the ladder with every page let in and with the router
    out["ocr_seconds_all"] = round(sum((r["docling"].get("s") or 0) + (r.get("tesseract") or {}).get("s", 0) for r in used), 1)
    return out


def run_text(items: list[dict[str, Any]], paths: Any, api: Any) -> dict[str, Any]:
    from rag_search.core.conversion import residue

    found = Counter()
    secs, shares, with_pics = [], [], 0
    for it in items:
        idx = paths.index.joinpath(it["coll"], *it["doc"].split("/"))
        src = api._doc_source(paths, it["coll"], it["doc"], idx, it["trace"])
        if src is None or src.suffix.lower() != ".pdf":
            continue
        prof = (it["rec"].get("profile") or {})
        known = prof.get("big_pics") or []
        t0 = time.perf_counter()
        boxes = residue.page_regions(src, it["page"], known)
        secs.append(time.perf_counter() - t0)
        found["pages"] += 1
        if known:
            with_pics += 1
        if boxes:
            found["with_regions"] += 1
            found["regions"] += len(boxes)
            shares.append(residue.facts(boxes)["share"])
    n = max(1, found["pages"])
    return {"pages": found["pages"], "with_big_picture": with_pics, "with_regions": found["with_regions"],
            "share_with_regions": round(found["with_regions"] / n, 3), "regions_per_page": round(found["regions"] / n, 2),
            "median_region_share": round(statistics.median(shares), 3) if shares else None,
            "median_ms": round(1000 * statistics.median(secs), 0) if secs else None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scans", type=int, default=40, help="scanned pages to read with OCR (default 40)")
    ap.add_argument("--text", type=int, default=100, help="text pages to look for residue regions on (default 100)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--only", default="", help="limit to one collection")
    ap.add_argument("--out", default="/tmp/lanes")
    a = ap.parse_args()
    from rag_search import api
    from rag_search.paths import get_paths

    paths = get_paths()
    rng = random.Random(a.seed)
    cand = candidates(paths, api, a.only)
    scans, text = pick(cand["scan"], a.scans, rng), pick(cand["text"], a.text, rng)
    print(f"{len(cand['scan'])} scanned and {len(cand['text'])} text pages in the traces; reading {len(scans)} scans, "
          f"looking at {len(text)} text pages", file=sys.stderr)
    rows = run_scans(scans, paths, api)
    report = {"scan": summarise(rows), "residue": run_text(text, paths, api)}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.json").write_text(json.dumps(rows, indent=1, default=str))
    (out / "summary.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
