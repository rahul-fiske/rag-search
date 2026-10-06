#!/usr/bin/env python3
"""Mine the conversion traces of a workspace: what kinds of page there are, what the gate flags, where the time goes.

Step R0a of docs/design/conversion-routing-plan.md.  It reads; it converts, reads with a model and changes nothing.
Run it with rag-search's own Python, so it sees the same packages and data folder the dashboard does:

    "$(uv tool dir)/rag-search/bin/python" scripts/mine_traces.py
    ... mine_traces.py --collection documents          # one collection
    ... mine_traces.py --sample 30 --seed 7                # more pages per check to look at
    ... mine_traces.py --time-gate 0                       # skip re-timing the gate checks
    ... mine_traces.py --home /path/to/data --out /tmp/mine

What it writes (default folder ``~/.cache/rag-search-mine/<date-time>/``, never inside the data folder):

  report.md       the tables below, readable
  report.json     the same numbers, for comparing runs
  low_pages.csv   every page whose outcome is ``low``, with the checks that failed and their details
  samples.md      for each failed check, a random sample of pages (digital and scanned separately) with the
                  commands that show the page's record and its converted text -- the pages to look at by hand

The report answers, from the traces stored next to the converted Markdown (``markup/<coll>/<doc>.trace.json``):

  1. pages by kind (digital, scanned, image, embedded, ...) and outcome (pass, low, repaired, no text)
  2. gate checks that failed, by kind of page, and how often a check is the *only* reason a page is low
  3. read, gate and repair time by kind, split into pages read in their run and pages reused from the page cache
  4. readers used (docling, the document reader and its model, Apple Vision, Tesseract) and their tokens
  5. the slowest pages at the gate
  6. (``--time-gate N``, default 200) the gate's checks re-timed one by one on up to N stored pages per kind, from the
     stored Markdown: which check takes the time
  7. (``--check-sources [low|all]``, step R0b) for pages read from a PDF text layer (digital and embedded), whether
     docling's Markdown still holds the text layer's words and numbers: the source PDF is opened read-only (only
     files inside a registered location) and each page's layer is compared with the stored Markdown.  ``low`` checks
     the pages the gate flagged; ``all`` every such page, which also shows text lost on pages the gate passed.
     Verdicts: ``intact`` (the flag is a false alarm for search), ``intact, table shape only`` (the text is there,
     only the table's layout was questioned), ``lost text``, ``uncertain`` (in between, or a layer too short or
     garbled to judge), ``no source`` (file not found).  Written to ``source_check.csv`` and section 7 of the report.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shlex
import statistics
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SHOW = {"raster": "scanned", "fallback": "scanned (OCR)", "embedded": "embedded", "digital": "digital",
        "image": "image", "office": "office", "copy": "text", "unknown": "unprofiled", "cached": "cached"}
SCAN_KINDS = ("raster", "fallback", "image")


def _show(kind: str) -> str:
    return SHOW.get(kind, kind)


def _read_traces(markup: Path, only: str):
    """(collection, document, trace) for every readable trace file under *markup*."""
    from rag_search.core.conversion.trace import TRACE_SUFFIX, read_trace

    root = markup / only if only else markup
    for f in sorted(root.rglob("*" + TRACE_SUFFIX)) if root.is_dir() else []:
        rel = f.relative_to(markup).as_posix()
        coll, _, rest = rel.partition("/")
        data = read_trace(f)
        if data and rest:
            yield coll, rest[:-len(TRACE_SUFFIX)], data, f


def _failed(p: dict[str, Any]) -> list[dict[str, Any]]:
    return [c for c in (p.get("gate") or {}).get("checks", []) if not c.get("ok", False)]


def _commands(coll: str, doc: str, page: int) -> tuple[str, str]:
    target = shlex.quote(f"{coll}/{doc}")
    return (f"rag-search trace {target} --page {page}", f"rag-search trace {target} --page {page} --md")


def mine(markup: Path, only: str = "") -> dict[str, Any]:
    by_kind_outcome: dict[str, Counter] = defaultdict(Counter)
    checks_by_kind: dict[str, Counter] = defaultdict(Counter)
    only_reason: dict[str, Counter] = defaultdict(Counter)
    times: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    readers: Counter = Counter()
    tokens: Counter = Counter()
    low_rows: list[dict[str, Any]] = []
    slow_gate: list[tuple[float, str, str, int, str]] = []
    docs = pages = 0
    for coll, doc, data, _f in _read_traces(markup, only):
        docs += 1
        from rag_search.core.conversion.trace import page_kind

        for p in data.get("pages", []):
            pages += 1
            kind = page_kind(p)
            outcome = str(p.get("outcome") or "pass")
            by_kind_outcome[kind][outcome] += 1
            failed = _failed(p)
            for c in failed:
                checks_by_kind[str(c.get("name"))][kind] += 1
            if outcome == "low" and len(failed) == 1:
                only_reason[str(failed[0].get("name"))][kind] += 1
            cached = p.get("cache") == "hit" or p.get("branch") == "cached"
            t = p.get("time_s") or {}
            for step in ("read", "gate", "repair"):
                if isinstance(t.get(step), (int, float)):
                    times[kind][f"{step}_{'cached' if cached else 'read'}"].append(float(t[step]))
            if isinstance(t.get("gate"), (int, float)):
                slow_gate.append((float(t["gate"]), coll, doc, int(p.get("page") or 0), kind))
            r = p.get("reader") or {}
            tool = str(r.get("tool") or "?")
            name = f"{tool} {r.get('model')}" if r.get("model") else tool
            readers[name] += 1
            if p.get("tokens"):
                tokens[name] += int(p["tokens"])
            if outcome == "low":
                low_rows.append({"collection": coll, "doc": doc, "page": int(p.get("page") or 0), "kind": _show(kind),
                                 "reader": name, "checks": ";".join(str(c.get("name")) for c in failed),
                                 "details": " | ".join(f"{c.get('name')}: {c.get('detail', '')}" for c in failed),
                                 "cached": cached})
    slow_gate.sort(reverse=True)

    def tsum(kind: str) -> dict[str, Any]:
        out = {}
        for key, vals in sorted(times[kind].items()):
            out[key] = {"pages": len(vals), "total_s": round(sum(vals), 1),
                        "mean_s": round(statistics.fmean(vals), 3) if vals else 0.0,
                        "max_s": round(max(vals), 2) if vals else 0.0}
        return out

    return {
        "documents": docs, "pages": pages,
        "by_kind_outcome": {k: dict(v) for k, v in sorted(by_kind_outcome.items())},
        "checks_by_kind": {k: dict(v) for k, v in sorted(checks_by_kind.items(), key=lambda kv: -sum(kv[1].values()))},
        "only_reason_low": {k: dict(v) for k, v in sorted(only_reason.items(), key=lambda kv: -sum(kv[1].values()))},
        "times_by_kind": {k: tsum(k) for k in sorted(times)},
        "readers": dict(readers.most_common()), "tokens": dict(tokens.most_common()),
        "slowest_gate": [{"seconds": round(s, 3), "collection": c, "doc": d, "page": n, "kind": _show(k)}
                         for s, c, d, n, k in slow_gate[:20]],
        "low_rows": low_rows,
    }


def time_gate(markup: Path, only: str, per_kind: int, seed: int) -> dict[str, Any]:
    """Re-run the gate's checks one by one on stored pages and time each: which check costs the time."""
    from rag_search.core.conversion import gate, pagemd, tables, validators
    from rag_search.core.conversion.trace import page_kind

    rng = random.Random(seed)
    pool: dict[str, list[tuple[Path, int, dict[str, Any]]]] = defaultdict(list)
    for _coll, _doc, data, f in _read_traces(markup, only):
        md = f.with_name(f.name[:-len(".trace.json")] + ".md")
        for p in data.get("pages", []):
            pool[page_kind(p)].append((md, int(p.get("page") or 0), p))
    checks = {
        "find_tables": lambda md, kind, prof: tables.find_tables(md),
        "coverage": lambda md, kind, prof: gate._coverage(md, kind, prof),
        "script": lambda md, kind, prof: gate._script(md, kind, prof),
        "table_shape": lambda md, kind, prof: gate._table_shape(md),
        "degenerate": lambda md, kind, prof: gate._degenerate(md, kind),
        "validators": lambda md, kind, prof: validators.page_violations(md),
        "whole gate": lambda md, kind, prof: gate.check_page(md, branch_kind=kind, profile=prof),
    }
    out: dict[str, Any] = {}
    texts: dict[Path, dict[int, str]] = {}
    for kind, items in sorted(pool.items()):
        sample = rng.sample(items, min(per_kind, len(items)))
        spent: dict[str, list[float]] = defaultdict(list)
        chars: list[int] = []
        for md_path, n, p in sample:
            if md_path not in texts:
                try:
                    texts[md_path] = pagemd.split_pages(md_path.read_text(encoding="utf-8", errors="replace"))
                except OSError:
                    texts[md_path] = {}
            md = texts[md_path].get(n)
            if md is None:
                continue
            chars.append(len(md))
            gkind = "digital" if kind in ("digital", "embedded") else ("scan" if kind in SCAN_KINDS else "other")
            prof = p.get("profile") or {}
            for name, fn in checks.items():
                t0 = time.perf_counter()
                try:
                    fn(md, gkind, prof)
                except Exception:  # noqa: BLE001 - a check that fails here is timed all the same
                    pass
                spent[name].append(time.perf_counter() - t0)
        if chars:
            out[kind] = {"pages": len(chars), "mean_chars": round(statistics.fmean(chars)),
                         "checks_ms": {k: {"mean": round(1000 * statistics.fmean(v), 2), "max": round(1000 * max(v), 1)}
                                       for k, v in spent.items()}}
    return out


INTACT_WORDS, INTACT_NUMBERS = 0.97, 0.98          # at least this share of the layer is in the Markdown: intact
LOST = 0.90                                        # below this share of words or numbers: lost text
MIN_LAYER_TOKENS = 15                              # fewer layer words than this: too short to judge


def _tokens(text: str) -> tuple[Counter, Counter]:
    """(words, numbers) of a text, normalised so that a PDF layer and docling's Markdown compare: NFKC (ligatures),
    lower case, words broken at a line end joined, Markdown escapes and soft hyphens dropped, numbers without
    thousands separators."""
    import re

    t = unicodedata.normalize("NFKC", text or "").replace("\u00ad", "").replace("\\", "")
    t = re.sub(r"(\w)-[ \t]*\n[ \t]*(\w)", r"\1\2", t).lower()
    nums = Counter(re.sub(r"[,\s]", "", m) for m in re.findall(r"\d[\d,]*(?:\.\d+)?", t))
    words = Counter(w for w in re.findall(r"[^\W\d_]{2,}", t))
    return words, nums


def _recall(layer: Counter, out: Counter) -> float | None:
    total = sum(layer.values())
    if not total:
        return None
    return sum(min(n, out.get(k, 0)) for k, n in layer.items()) / total


def _verdict(w: float | None, n: float | None, layer_words: int, layer_ok: bool, failed: list[str]) -> str:
    if w is None or layer_words < MIN_LAYER_TOKENS or not layer_ok:
        return "uncertain"
    nn = 1.0 if n is None else n
    if w < LOST or nn < LOST:
        return "lost text"
    if w >= INTACT_WORDS and nn >= INTACT_NUMBERS:
        return "intact, table shape only" if failed and set(failed) <= {"table_shape"} else "intact"
    return "uncertain"


def check_sources(paths: Any, only: str, which: str) -> dict[str, Any]:
    """Compare the text layer of every digital / embedded page (``which`` = low | all) with its stored Markdown."""
    from rag_search import api
    from rag_search.core.conversion import pagemd, tables
    from rag_search.core.conversion.trace import page_kind
    from rag_search.core.docling_convert import page_text_ok

    try:
        import pypdfium2 as pdfium
    except ImportError:
        return {"error": "pypdfium2 is not installed in this Python"}
    rows: list[dict[str, Any]] = []
    for coll, doc, data, f in _read_traces(paths.markup, only):
        want = [p for p in data.get("pages", []) if page_kind(p) in ("digital", "embedded")
                and (which == "all" or p.get("outcome") == "low")]
        if not want:
            continue
        idx = paths.index.joinpath(coll, *doc.split("/"))
        src = api._doc_source(paths, coll, doc, idx, f)
        md_path = f.with_name(f.name[:-len(".trace.json")] + ".md")
        base = {"collection": coll, "doc": doc}
        if src is None or src.suffix.lower() != ".pdf":
            rows += [{**base, "page": int(p.get("page") or 0), "outcome": p.get("outcome"), "verdict": "no source",
                      "checks": ";".join(str(c.get("name")) for c in _failed(p))} for p in want]
            continue
        try:
            pages_md = pagemd.split_pages(md_path.read_text(encoding="utf-8", errors="replace"))
            pdf = pdfium.PdfDocument(str(src))
        except Exception as exc:  # noqa: BLE001 - an unreadable file is a row, not a crash
            rows += [{**base, "page": int(p.get("page") or 0), "outcome": p.get("outcome"), "verdict": "no source",
                      "checks": ";".join(str(c.get("name")) for c in _failed(p)), "note": f"{type(exc).__name__}: {exc}"}
                     for p in want]
            continue
        try:
            for p in want:
                n = int(p.get("page") or 0)
                failed = [str(c.get("name")) for c in _failed(p)]
                try:
                    page = pdf[n - 1]
                    tp = page.get_textpage()
                    layer = tp.get_text_range() or ""
                    tp.close()
                    page.close()
                except Exception:  # noqa: BLE001
                    layer = ""
                lw, ln = _tokens(layer)
                ow, on = _tokens(tables.plain_text(pages_md.get(n, "")))
                w, num = _recall(lw, ow), _recall(ln, on)
                rows.append({**base, "page": n, "outcome": p.get("outcome"), "checks": ";".join(failed),
                             "layer_words": sum(lw.values()), "layer_numbers": sum(ln.values()),
                             "word_recall": None if w is None else round(w, 3),
                             "number_recall": None if num is None else round(num, 3),
                             "verdict": _verdict(w, num, sum(lw.values()), page_text_ok(layer), failed)})
        finally:
            pdf.close()
    by_check: dict[str, Counter] = defaultdict(Counter)
    by_outcome: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        by_outcome[str(r.get("outcome"))][r["verdict"]] += 1
        for name in (r.get("checks") or "").split(";"):
            if name:
                by_check[name][r["verdict"]] += 1
    return {"which": which, "pages": len(rows), "rows": rows,
            "by_check": {k: dict(v) for k, v in sorted(by_check.items(), key=lambda kv: -sum(kv[1].values()))},
            "by_outcome": {k: dict(v) for k, v in sorted(by_outcome.items())}}


def _table(rows: list[list[Any]], head: list[str]) -> str:
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def _sources_section(src: dict[str, Any], sample: int, seed: int) -> list[str]:
    if src.get("error"):
        return ["## 7. Text layer against the Markdown", "", f"Not run: {src['error']}", ""]
    verdicts = ["intact", "intact, table shape only", "uncertain", "lost text", "no source"]
    out = ["## 7. Text layer against the Markdown (digital and embedded pages, " + src["which"] + ")", "",
           f"{src['pages']} pages compared. `intact`: at least {round(100 * INTACT_WORDS)} % of the layer's words and "
           f"{round(100 * INTACT_NUMBERS)} % of its numbers are in the Markdown. `lost text`: under {round(100 * LOST)} %.", "",
           "By outcome:", "", _table([[o] + [c.get(v, 0) for v in verdicts] for o, c in src["by_outcome"].items()],
                                     ["outcome"] + verdicts), "",
           "By failed check (a page can count under several):", "",
           _table([[k] + [c.get(v, 0) for v in verdicts] for k, c in src["by_check"].items()], ["check"] + verdicts), ""]
    rng = random.Random(seed)
    for v in ("lost text", "uncertain", "intact, table shape only"):
        pick = [r for r in src["rows"] if r["verdict"] == v]
        if not pick:
            continue
        out += [f"Sample, {v} ({len(pick)}):", ""]
        for r in rng.sample(pick, min(sample, len(pick))):
            rec, text = _commands(r["collection"], r["doc"], r["page"])
            out.append(f"- `{r['collection']}/{r['doc']}` page {r['page']} ({r.get('outcome')}; words "
                       f"{r.get('word_recall')}, numbers {r.get('number_recall')}; {r.get('checks') or 'no check'}): `{text}`")
        out.append("")
    return out


def write_report(out_dir: Path, rep: dict[str, Any], timing: dict[str, Any], sample: int, seed: int,
                 home: Path, sources: dict[str, Any] | None = None) -> None:
    kinds = sorted(rep["by_kind_outcome"], key=lambda k: -sum(rep["by_kind_outcome"][k].values()))
    outcomes = ["pass", "low", "repaired", "no_text", "error"]
    md: list[str] = [f"# Conversion traces of {home}", "",
                     f"{rep['documents']} documents with a trace, {rep['pages']} pages. Written {time.strftime('%Y-%m-%d %H:%M')}.", ""]
    md += ["## 1. Pages by kind and outcome", "", _table(
        [[_show(k), sum(rep["by_kind_outcome"][k].values())] + [rep["by_kind_outcome"][k].get(o, 0) for o in outcomes]
         for k in kinds], ["kind", "pages"] + outcomes), ""]
    md += ["## 2. Gate checks that failed, by kind of page", "",
           "A page can fail several checks; `only reason` counts low pages where this check was the single failure.", "",
           _table([[name] + [counts.get(k, 0) for k in kinds] + [sum(rep["only_reason_low"].get(name, {}).values())]
                   for name, counts in rep["checks_by_kind"].items()],
                  ["check"] + [_show(k) for k in kinds] + ["only reason"]), ""]
    md += ["## 3. Time by kind (seconds; `read` = read in its run, `cached` = reused from the page cache)", ""]
    rows = []
    for k in kinds:
        for key, v in rep["times_by_kind"].get(k, {}).items():
            rows.append([_show(k), key, v["pages"], v["total_s"], v["mean_s"], v["max_s"]])
    md += [_table(rows, ["kind", "step", "pages", "total", "mean", "max"]), ""]
    md += ["## 4. Readers", "", _table([[name, n, rep["tokens"].get(name, "")] for name, n in rep["readers"].items()],
                                       ["reader", "pages", "tokens"]), ""]
    md += ["## 5. Slowest pages at the gate", "", _table(
        [[r["seconds"], r["kind"], f"{r['collection']}/{r['doc']}", r["page"]] for r in rep["slowest_gate"]],
        ["seconds", "kind", "document", "page"]), ""]
    if timing:
        names = list(next(iter(timing.values()))["checks_ms"])
        md += ["## 6. The gate's checks re-timed on stored pages (milliseconds, mean / max)", "", _table(
            [[_show(k), v["pages"], v["mean_chars"]] + [f"{v['checks_ms'][n]['mean']} / {v['checks_ms'][n]['max']}" for n in names]
             for k, v in timing.items()], ["kind", "pages", "chars"] + names), ""]
    if sources:
        md += _sources_section(sources, sample, seed)
    md += ["## Next", "", "Open `samples.md` and look at the pages listed for the big checks (table shape, coverage):",
           "for each, is the problem real (content lost or mangled) or a false alarm? That decides how gate v2 changes."]
    (out_dir / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out_dir / "report.json").write_text(json.dumps({**{k: v for k, v in rep.items() if k != "low_rows"},
                                                     "gate_timing": timing,
                                                     "source_check": {k: v for k, v in (sources or {}).items() if k != "rows"}},
                                                    indent=2), encoding="utf-8")
    if sources and sources.get("rows"):
        cols = ["collection", "doc", "page", "outcome", "checks", "verdict", "word_recall", "number_recall",
                "layer_words", "layer_numbers", "note"]
        with open(out_dir / "source_check.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(sources["rows"])
    with open(out_dir / "low_pages.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["collection", "doc", "page", "kind", "reader", "checks", "details", "cached"])
        w.writeheader()
        w.writerows(rep["low_rows"])
    rng = random.Random(seed)
    by_check: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rep["low_rows"]:
        for name in r["checks"].split(";"):
            if name:
                by_check[name].append(r)
    sm = ["# Low pages to look at", "",
          "For each page: is the problem real (text lost, columns shifted, wrong script) or a false alarm?",
          "Note the answer next to the page; the counts per check decide how the gate changes.", ""]
    for name, rows_ in sorted(by_check.items(), key=lambda kv: -len(kv[1])):
        sm += [f"## {name} ({len(rows_)} low pages)", ""]
        for group, keep in (("digital pages", lambda r: r["kind"] in ("digital", "embedded")),
                            ("scanned and image pages", lambda r: r["kind"] not in ("digital", "embedded"))):
            pick = [r for r in rows_ if keep(r)]
            if not pick:
                continue
            sm += [f"### {group} ({len(pick)})", ""]
            for r in rng.sample(pick, min(sample, len(pick))):
                rec, text = _commands(r["collection"], r["doc"], r["page"])
                detail = next((d.split(": ", 1)[1] for d in r["details"].split(" | ") if d.startswith(name + ":")), "")
                sm += [f"- [ ] `{r['collection']}/{r['doc']}` page {r['page']} ({r['kind']}, {r['reader']}): {detail}",
                       f"  `{rec}` · `{text}`"]
            sm.append("")
    (out_dir / "samples.md").write_text("\n".join(sm) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--home", type=Path, default=None, help="rag-search's data folder (default: the one in use)")
    ap.add_argument("--collection", default="", help="only this collection")
    ap.add_argument("--out", type=Path, default=None, help="where the report goes (default ~/.cache/rag-search-mine/...)")
    ap.add_argument("--sample", type=int, default=20, help="low pages listed per check and page group (default 20)")
    ap.add_argument("--seed", type=int, default=1, help="random seed of the samples")
    ap.add_argument("--time-gate", type=int, default=200, help="re-time the gate checks on up to N pages per kind (0 = off)")
    ap.add_argument("--check-sources", nargs="?", const="low", default="", choices=["low", "all"],
                    help="compare each digital page's PDF text layer with its Markdown: the low pages, or all (R0b)")
    a = ap.parse_args(argv)
    from rag_search.paths import get_paths

    paths = get_paths(a.home.expanduser()) if a.home else get_paths()
    if not paths.markup.is_dir():
        print(f"no converted documents under {paths.markup}", file=sys.stderr)
        return 1
    out_dir = (a.out or Path.home() / ".cache" / "rag-search-mine" / time.strftime("%Y%m%d-%H%M%S")).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    print(f"reading traces under {paths.markup} ...", flush=True)
    rep = mine(paths.markup, a.collection)
    timing = time_gate(paths.markup, a.collection, a.time_gate, a.seed) if a.time_gate > 0 else {}
    sources = None
    if a.check_sources:
        print(f"comparing text layers with the Markdown ({a.check_sources} pages) ...", flush=True)
        sources = check_sources(paths, a.collection, a.check_sources)
    write_report(out_dir, rep, timing, a.sample, a.seed, paths.home, sources)
    print(f"{rep['documents']} documents, {rep['pages']} pages, {len(rep['low_rows'])} low pages "
          f"in {time.perf_counter() - t0:.1f} s")
    if sources and not sources.get("error"):
        lost = sum(1 for r in sources["rows"] if r["verdict"] == "lost text")
        print(f"text layer check: {sources['pages']} pages, {lost} with lost text")
    print(f"report: {out_dir / 'report.md'}\nsamples to look at: {out_dir / 'samples.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
