"""Conversion benchmark: gold sets and runs (stdlib at import; readers and renderers are lazy).

A **gold set** is a folder ``<home>/conversion_gold/<set>/`` with ``gold.json`` and ``images/``: a
list of source pages, each with the text a person checked (``truth``), its tables, search phrases,
and a failure ``class`` (digital-table, scan-table, devanagari-text, ...).  ``gold_init`` builds a
draft from conversion traces of an index the owner already has -- it picks pages per failure class,
renders them, and pre-fills ``truth`` with what the pipeline read.  Pre-filled entries are
``"verified": false``: a benchmark compared with the pipeline's own output proves nothing, so runs
use verified entries only (``include_drafts`` to override).  The owner fixes ``truth`` by hand
against the PNG and sets ``"verified": true``.

A **run** reads the gold pages with an *engine* (``engines.py``) and stores the measures
(``metrics.py``) per page, per class and overall in ``<home>/conversion_bench/<set>/<run>.json``.
Runs never write to the index, the serving folder or the markup of any collection; the source
files are opened read-only and must lie in a registered location.
"""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ...paths import Paths, read_json, sha256_file, write_json_atomic
from . import costs, engines, metrics, pagemd, trace

GOLD_DIR = "conversion_gold"
BENCH_DIR = "conversion_bench"
FORMAT = 1
_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,47}")

# metrics where a smaller number is better
LOWER_IS_BETTER = ("cer", "s_per_page")
HEADLINE = ("cell_exact", "cell_bag", "cer", "table_sim", "balance_ok", "query_hit", "s_per_page")


class BenchError(ValueError):
    """A problem the caller can fix (bad name, missing set, ...): the message is for the user."""


def valid_name(name: str, what: str = "name") -> str:
    name = (name or "").strip()
    if not _NAME_RE.fullmatch(name):
        raise BenchError(f"invalid {what} {name!r}: letters, digits, '-' and '_' only (at most 48)")
    return name


def gold_dir(paths: Paths, set_name: str) -> Path:
    return paths.home / GOLD_DIR / valid_name(set_name, "set name")


def bench_dir(paths: Paths, set_name: str) -> Path:
    return paths.home / BENCH_DIR / valid_name(set_name, "set name")


# ── failure classes ───────────────────────────────────────────────────────────────────────

LATIN_LIKE = ("", "none", "Latin", "Mixed")


def classify(page: dict[str, Any]) -> str:
    """Failure class of a page record from a conversion trace.  The classes are the ways pages
    fail differently: scans vs digital text, with or without tables, other scripts."""
    out = page.get("out") or {}
    script = str(out.get("script") or (page.get("profile") or {}).get("script") or "")
    tab = bool(out.get("tables"))
    branch, outcome = page.get("branch"), page.get("outcome")
    if outcome in ("no_text", "error"):
        return "empty"
    base = {"digital": "digital", "office": "office", "image": "image", "copy": "text"}.get(str(branch), "scan")
    if script and script not in LATIN_LIKE:
        return f"{script.lower()}-{'table' if tab else 'text'}"
    if base in ("office", "image", "text"):
        return base
    return f"{base}-{'table' if tab else 'text'}"


def _pick_key(seed: int, ident: str) -> str:
    return hashlib.sha256(f"{seed}|{ident}".encode()).hexdigest()


# ── gold sets ─────────────────────────────────────────────────────────────────────────────

def read_gold(paths: Paths, set_name: str) -> dict[str, Any]:
    data = read_json(gold_dir(paths, set_name) / "gold.json")
    if not isinstance(data, dict) or not isinstance(data.get("pages"), list):
        raise BenchError(f"no gold set {set_name!r} (create one: rag-search bench gold init {set_name})")
    return data


def list_gold(paths: Paths) -> list[dict[str, Any]]:
    root = paths.home / GOLD_DIR
    out = []
    if root.is_dir():
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            g = read_json(d / "gold.json")
            if not isinstance(g, dict):
                continue
            pg = g.get("pages") or []
            classes: dict[str, int] = {}
            for e in pg:
                classes[e.get("class", "?")] = classes.get(e.get("class", "?"), 0) + 1
            out.append({"name": d.name, "pages": len(pg),
                        "verified": sum(1 for e in pg if e.get("verified")),
                        "classes": classes, "runs": len(_run_files(paths, d.name))})
    return out


def _source_roots(paths: Paths) -> list[Path]:
    from ... import locations

    locs, _err = locations.load(paths)
    return [Path(f) for f in (locs or {}).values()]


def resolve_source(paths: Paths, entry: dict[str, Any]) -> Path | None:
    """The source file of a gold entry, if it is still there and lies inside a
    registered location (the path in the file is never trusted on its own)."""
    roots = _source_roots(paths)
    cands = []
    if entry.get("path"):
        cands.append(Path(str(entry["path"])))
    rel = str(entry.get("rel") or "").strip("/")
    if rel:
        parts = rel.split("/", 1)
        if len(parts) == 2:
            from ... import locations

            locs, _err = locations.load(paths)
            if parts[0] in (locs or {}):
                cands.append(Path(locs[parts[0]]) / parts[1])
    for c in cands:
        try:
            real = c.resolve()
            if real.is_file() and any(real.is_relative_to(r.resolve()) for r in roots):
                return real
        except OSError:
            continue
    return None


def _trace_docs(paths: Paths, collection: str = "") -> list[tuple[str, str, Path]]:
    """(collection, document path without extension, trace file) for the stored traces."""
    out = []
    root = paths.markup
    if not root.is_dir():
        return out
    for tf in sorted(root.rglob("*" + trace.TRACE_SUFFIX)):
        rel = tf.relative_to(root).as_posix()[: -len(trace.TRACE_SUFFIX)]
        coll, _, doc = rel.partition("/")
        if not doc or (collection and coll != collection):
            continue
        out.append((coll, doc, tf))
    return out


def gold_init(paths: Paths, set_name: str, *, collection: str = "", per_class: int = 5,
              classes: list[str] | None = None, seed: int = 0, render: bool = True) -> dict[str, Any]:
    """Create (or extend) a draft gold set from the conversion traces of the current index.

    Picks up to *per_class* pages of every failure class, spread over different documents,
    renders each as ``images/<id>.png`` and pre-fills ``truth`` with the Markdown the pipeline
    produced (``verified: false`` until a person has checked it).  Pages already in the set are
    not added again.  Returns {"set", "added", "total", "by_class", "skipped"}."""
    gdir = gold_dir(paths, set_name)
    per_class = max(1, min(int(per_class), 200))
    gold = read_json(gdir / "gold.json") or {"format": FORMAT, "name": set_name, "pages": []}
    pages: list[dict[str, Any]] = gold.setdefault("pages", [])
    have = {(e.get("rel"), e.get("page")) for e in pages}
    n_have = len(pages)
    pool: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for coll, doc, tf in _trace_docs(paths, collection):
        data = trace.read_trace(tf)
        if not data:
            continue
        for p in data.get("pages", []):
            cls = classify(p)
            if classes and cls not in classes:
                continue
            pool.setdefault(cls, []).append((f"{coll}/{doc}", {"trace": tf, "data": data, "page": p,
                                                              "coll": coll, "doc": doc}))
    added: dict[str, int] = {}
    skipped: list[str] = []
    for cls in sorted(pool):
        cands = sorted(pool[cls], key=lambda c: _pick_key(seed, f"{c[0]}#{c[1]['page'].get('page')}"))
        seen_docs: set[str] = set()
        chosen = []
        for key, c in cands:            # first one page per document, then fill up with more
            if key in seen_docs:
                continue
            seen_docs.add(key)
            chosen.append(c)
            if len(chosen) >= per_class:
                break
        if len(chosen) < per_class:
            for key, c in cands:
                if c not in chosen:
                    chosen.append(c)
                if len(chosen) >= per_class:
                    break
        for c in chosen:
            idx = paths.index / c["coll"] / c["doc"] / "index.meta.json"
            meta = read_json(idx) or {}
            src_path = str(meta.get("src_path") or "")
            ext = Path(src_path).suffix or Path(str(c["data"].get("source") or "")).suffix
            rel = f"{c['coll']}/{c['doc']}{ext}"
            pg = int(c["page"].get("page") or 1)
            if (rel, pg) in have:
                continue
            md_file = paths.markup / c["coll"] / (c["doc"] + ".md")
            truth = ""
            try:
                truth = pagemd.split_pages(md_file.read_text(encoding="utf-8")).get(pg, "")
            except OSError:
                skipped.append(f"{rel} p{pg}: no Markdown to pre-fill from")
            ident = f"p{len(pages) + 1:03d}"
            entry = {"id": ident, "class": cls, "collection": c["coll"], "doc": c["doc"], "rel": rel,
                     "path": src_path, "page": pg, "src_sha256": c["data"].get("src_sha256", ""),
                     "script": str((c["page"].get("out") or {}).get("script") or
                                   (c["page"].get("profile") or {}).get("script") or ""),
                     "truth": truth, "truth_tables": None, "queries": [], "verified": False,
                     "note": "pre-filled from the pipeline's own output: check it against the image"}
            if render:
                src = resolve_source(paths, entry)
                if src is None:
                    skipped.append(f"{rel} p{pg}: source file not available, no image")
                else:
                    try:
                        from . import pageimage

                        png = pageimage.render(src, pg, 1200)
                        (gdir / "images").mkdir(parents=True, exist_ok=True)
                        (gdir / "images" / f"{ident}.png").write_bytes(png)
                        entry["image"] = f"images/{ident}.png"
                    except Exception as exc:  # noqa: BLE001 - libraries missing, damaged file
                        skipped.append(f"{rel} p{pg}: no image ({type(exc).__name__}: {exc})")
            pages.append(entry)
            have.add((rel, pg))
            added[cls] = added.get(cls, 0) + 1
    gold["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    gdir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(gdir / "gold.json", gold)
    return {"set": set_name, "dir": str(gdir), "added": added, "added_total": len(pages) - n_have,
            "total": len(pages), "skipped": skipped[:50],
            "classes_available": {k: len(v) for k, v in sorted(pool.items())}}


# ── runs ──────────────────────────────────────────────────────────────────────────────────

def _run_files(paths: Paths, set_name: str) -> list[Path]:
    d = bench_dir(paths, set_name)
    return sorted(d.glob("*.json")) if d.is_dir() else []


def run_bench(paths: Paths, set_name: str, *, engine: str = "current", name: str = "",
              include_drafts: bool = False, classes: list[str] | None = None, limit: int = 0,
              progress: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Read the gold pages with *engine*, score them, store and return the run record."""
    gold = read_gold(paths, set_name)
    label = valid_name(name, "run name") if name else ""
    entries = [e for e in gold["pages"]
               if (include_drafts or e.get("verified")) and (not classes or e.get("class") in classes)]
    if limit:
        entries = entries[: int(limit)]
    if not entries:
        raise BenchError("no verified pages in this gold set (check the truth text against the "
                         "images and set \"verified\": true, or pass --drafts to measure against "
                         "the unchecked pre-filled text)")
    eng = engines.resolve(engine)
    meter = costs.Meter().start()
    t_start = time.time()
    by_file: dict[str, list[dict[str, Any]]] = {}
    skipped: list[dict[str, str]] = []
    sources: dict[str, Path] = {}
    for e in entries:
        src = resolve_source(paths, e)
        if src is None:
            skipped.append({"id": e["id"], "why": "source file not found in a registered location"})
            continue
        sources[e["id"]] = src
        by_file.setdefault(str(src), []).append(e)
    rows: list[dict[str, Any]] = []
    sha_cache: dict[str, str] = {}
    try:
        _measure(eng, by_file, rows, skipped, sha_cache, progress)
        child_peak = float(getattr(eng, "peak_mb", 0.0) or 0.0)
        try:
            desc = eng.describe() if hasattr(eng, "describe") else {"name": engine}
        except Exception as exc:  # noqa: BLE001
            desc = {"name": engine, "error": str(exc)}
    finally:
        close = getattr(eng, "close", None)
        if callable(close):
            close()
    return _finish(paths, set_name, engine, label, include_drafts, rows, skipped, desc, meter, t_start,
                   child_peak)


def _measure(eng: Any, by_file: dict[str, list[dict[str, Any]]], rows: list[dict[str, Any]],
             skipped: list[dict[str, str]], sha_cache: dict[str, str],
             progress: Callable[[dict[str, Any]], None] | None) -> None:
    for fi, (fname, ents) in enumerate(sorted(by_file.items()), 1):
        src = Path(fname)
        stale = ""
        want = {e.get("src_sha256") for e in ents if e.get("src_sha256")}
        if want:
            sha_cache[fname] = sha_cache.get(fname) or sha256_file(src)
            if sha_cache[fname] not in want:
                stale = "the source file has changed since the gold page was made"
        if stale:
            skipped.extend({"id": e["id"], "why": stale} for e in ents)
            continue
        wanted = sorted({int(e["page"]) for e in ents})
        try:
            res = eng.read_pages(src, wanted)
            err = ""
        except Exception as exc:  # noqa: BLE001 - one bad file must not end the run
            res, err = {"pages": {}, "seconds": 0.0, "pages_read": len(wanted)}, f"{type(exc).__name__}: {exc}"
        per_page_s = (res.get("seconds") or 0.0) / max(1, int(res.get("pages_read") or len(wanted)))
        for e in ents:
            pred = (res.get("pages") or {}).get(int(e["page"]), "")
            row = {"id": e["id"], "class": e.get("class", ""), "file": e.get("rel") or src.name,
                   "page": int(e["page"]), "seconds": round(res.get("page_s", {}).get(int(e["page"]), per_page_s)
                                                            if isinstance(res.get("page_s"), dict) else per_page_s, 3),
                   "verified": bool(e.get("verified"))}
            if err:
                row["error"] = err
                row.update(metrics.score_page("", e.get("truth", ""), e.get("queries") or [],
                                              e.get("truth_tables")))
            else:
                row.update(metrics.score_page(pred, e.get("truth", ""), e.get("queries") or [],
                                              e.get("truth_tables")))
            rows.append(row)
        if progress:
            progress({"file": fi, "files": len(by_file), "pages": len(rows)})


def _finish(paths: Paths, set_name: str, engine: str, label: str, include_drafts: bool,
            rows: list[dict[str, Any]], skipped: list[dict[str, str]], desc: dict[str, Any],
            meter: Any, t_start: float, child_peak: float) -> dict[str, Any]:
    if not rows:
        raise BenchError("nothing to measure: " + "; ".join(f"{s['id']}: {s['why']}" for s in skipped[:3]))
    cls_names = sorted({r["class"] for r in rows})
    summary = {"all": metrics.aggregate(rows),
               "by_class": {c: metrics.aggregate([r for r in rows if r["class"] == c]) for c in cls_names}}
    run_id = time.strftime("%Y%m%d-%H%M%S", time.gmtime(t_start)) + (f"-{label}" if label else "")
    d = bench_dir(paths, set_name)
    base, n = run_id, 1
    while (d / f"{run_id}.json").exists():
        n += 1
        run_id = f"{base}-{n}"
    rec = {"format": FORMAT, "id": run_id, "name": label, "set": set_name, "engine": engine,
           "engine_info": desc, "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_start)),
           "seconds": round(time.time() - t_start, 2),
           "cost": {"cpu_s": round(meter.cpu_now(), 1), "peak_mb": round(max(costs.peak_rss_mb(), child_peak))},
           "drafts_included": include_drafts, "summary": summary, "pages": rows, "skipped": skipped}
    d.mkdir(parents=True, exist_ok=True)
    write_json_atomic(d / f"{run_id}.json", rec)
    return rec


def list_runs(paths: Paths, set_name: str = "") -> list[dict[str, Any]]:
    """Stored runs, newest first, without their per-page rows."""
    names = [set_name] if set_name else [g["name"] for g in list_gold(paths)]
    root = paths.home / BENCH_DIR
    if not set_name and root.is_dir():
        names = sorted({*names, *[p.name for p in root.iterdir() if p.is_dir()]})
    out = []
    for n in names:
        for f in _run_files(paths, n):
            r = read_json(f)
            if isinstance(r, dict) and r.get("id"):
                out.append({k: r.get(k) for k in ("id", "name", "set", "engine", "started_at", "seconds",
                                                  "cost", "drafts_included")}
                           | {"summary": (r.get("summary") or {}).get("all", {}),
                              "classes": sorted((r.get("summary") or {}).get("by_class", {}))})
    out.sort(key=lambda r: r.get("started_at") or "", reverse=True)
    return out


def read_run(paths: Paths, set_name: str, run_id: str) -> dict[str, Any]:
    rid = valid_name(run_id, "run id")
    r = read_json(bench_dir(paths, set_name) / f"{rid}.json")
    if not isinstance(r, dict) or not r.get("id"):
        raise BenchError(f"no run {run_id!r} in set {set_name!r}")
    return r


def compare(paths: Paths, set_name: str, a: str, b: str) -> dict[str, Any]:
    """Run *b* against run *a*: every headline measure overall and per class, with the change and
    whether it is better, and the pages that got worse."""
    ra, rb = read_run(paths, set_name, a), read_run(paths, set_name, b)

    def delta(x: dict[str, Any], y: dict[str, Any]) -> dict[str, Any]:
        out = {}
        for m in HEADLINE:
            va, vb = x.get(m), y.get(m)
            if va is None or vb is None:
                out[m] = {"a": va, "b": vb, "delta": None, "better": None}
                continue
            d = round(vb - va, 4)
            better = None if d == 0 else (d < 0) == (m in LOWER_IS_BETTER)
            out[m] = {"a": va, "b": vb, "delta": d, "better": better}
        return out

    sa, sb = ra["summary"], rb["summary"]
    classes = {c: delta(sa["by_class"].get(c, {}), sb["by_class"].get(c, {}))
               for c in sorted(set(sa["by_class"]) | set(sb["by_class"]))}
    pa = {r["id"]: r for r in ra["pages"]}
    worse = []
    for r in rb["pages"]:
        o = pa.get(r["id"])
        if not o:
            continue
        d_cer = (r.get("cer") or 0) - (o.get("cer") or 0)
        ca, cb = o["cells"], r["cells"]
        d_cells = (cb["exact"] - ca["exact"]) if ca["total"] else 0
        if d_cer > 0.02 or d_cells < 0:
            worse.append({"id": r["id"], "file": r["file"], "page": r["page"], "class": r["class"],
                          "cer": [o.get("cer"), r.get("cer")],
                          "cells_exact": [ca["exact"], cb["exact"]], "of": cb["total"]})
    worse.sort(key=lambda w: (w["cells_exact"][1] - w["cells_exact"][0], -(w["cer"][1] or 0) + (w["cer"][0] or 0)))
    return {"set": set_name, "a": {"id": ra["id"], "engine": ra["engine"], "name": ra.get("name", "")},
            "b": {"id": rb["id"], "engine": rb["engine"], "name": rb.get("name", "")},
            "all": delta(sa["all"], sb["all"]), "by_class": classes, "worse": worse[:20]}
