"""`rag-search` command line: a thin client of the two daemons (stdlib only).

Search / list / grep / index / daemon commands talk to the daemons over their sockets
(starting them on demand); doctor, setup, convert, paths, config, service and the
register commands run locally.  Add ``--json`` to any command for machine output.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from . import __version__, api, spec
from .paths import DEFAULT_TOP_K, allow_cloud_files, ensure_dirs, get_paths

EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_UNAVAILABLE = 0, 1, 2, 3


# ── output helpers ───────────────────────────────────────────────────────────

def _json(obj: Any) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def _err(msg: str) -> None:
    print(msg, file=sys.stderr)


def _fail(resp: dict[str, Any]) -> int:
    """Print a daemon error reply; return the exit code."""
    code = resp.get("code", "")
    _err(f"error: {resp.get('error', 'request failed')}")
    if code == "warming_up":
        _err("  the daemon is still starting; try again shortly "
             f"(log: {resp.get('log', 'see `rag-search paths run`')})")
    if code in ("warming_up", "unavailable"):
        return EXIT_UNAVAILABLE
    return EXIT_FAIL


def _bytes(n: Any) -> str:
    if n is None:
        return "?"
    n = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _dur(s: Any) -> str:
    if s is None:
        return "?"
    s = float(s)
    if s < 60:
        return f"{s:.1f}s"
    m, sec = divmod(int(round(s)), 60)
    if m < 60:
        return f"{m}m {sec:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


def _clock(ts: Any) -> str:
    """Local date and time for an epoch number or a UTC ISO string; '?' when unknown."""
    import datetime as dt

    if ts in (None, ""):
        return "?"
    try:
        if isinstance(ts, (int, float)):
            d = dt.datetime.fromtimestamp(ts)
        else:
            d = dt.datetime.strptime(str(ts), "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=dt.UTC).astimezone()
        return d.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, OSError, OverflowError):
        return str(ts)


def _since(epoch: Any) -> str:
    import time

    return _dur(max(0.0, time.time() - float(epoch))) if epoch else "?"


def _fmt_progress(p: dict[str, Any] | None) -> str:
    if not p:
        return ""
    bits = [str(p.get("phase", ""))]
    if p.get("total"):
        bits.append(f"{p.get('done', 0)}/{p['total']}")
    if p.get("current"):
        bits.append(str(p["current"]))
    if p.get("message"):
        bits.append(f"({p['message']})")
    return " ".join(b for b in bits if b)


def _fmt_progress_live(p: dict[str, Any] | None) -> str:
    """Progress plus how long the phase and the current document have been running."""
    line = _fmt_progress(p)
    if p and p.get("phase") in ("convert", "embed", "merge") and p.get("phase_started_at"):
        line += f" - {p['phase']} phase running for {_since(p['phase_started_at'])}"
    if p and p.get("current") and p.get("current_since"):
        line += f", {p['current']} for {_since(p['current_since'])}"
    conv = (p or {}).get("conversion")
    if conv and conv.get("pages"):
        line += f"\n  pages so far: {conv['pages']} ({_fmt_branches(conv.get('branches'))})"
    return line


def _fmt_branches(branches: dict[str, Any] | None) -> str:
    """``{"digital": 566, "raster": 2}`` -> ``digital 566, scanned 2`` (pages per branch)."""
    from .core.conversion.trace import BRANCH_SHORT

    return ", ".join(f"{BRANCH_SHORT.get(b, b)} {n}" for b, n in (branches or {}).items() if n)


def _fmt_outcomes(outcomes: dict[str, Any] | None) -> str:
    return ", ".join(f"{k.replace('_', ' ')} {n}" for k, n in (outcomes or {}).items() if n)


def _stage(key: str) -> str:
    """"3.1 profile" for a time key such as ``profile`` (the pipeline's numbering, stages.py)."""
    from . import stages

    return stages.named(key).lower() if key in stages.BY_KEY else key.replace("_", " ")


def _fmt_conv_doc(conv: dict[str, Any] | None) -> str:
    """One document's conversion in a few words: ``12 pages: digital 10, scanned 2``."""
    if not conv or not conv.get("pages"):
        return ""
    bits = f"{conv['pages']} page{'s' if conv['pages'] != 1 else ''}: {_fmt_branches(conv.get('branches'))}"
    bad = {k: v for k, v in (conv.get("outcomes") or {}).items() if k != "pass" and v}
    if bad:
        bits += f" ({_fmt_outcomes(bad)})"
    return bits


def _fmt_conv_totals(c: dict[str, Any] | None, indent: str = "  ") -> list[str]:
    """Run / collection totals: pages per branch and outcome, time per step, cost."""
    if not c or not c.get("pages"):
        return []
    lines = [f"{indent}pages {c['pages']} in {c.get('docs', c.get('documents', 0))} document(s): "
             f"{_fmt_branches(c.get('branches'))}"]
    if c.get("outcomes"):
        lines.append(f"{indent}outcomes: {_fmt_outcomes(c['outcomes'])}")
    if c.get("time_s"):
        lines.append(f"{indent}time: " + ", ".join(f"{_stage(k)} {_dur(v)}" for k, v in c["time_s"].items()))
    cost = c.get("cost") or {}
    extra = []
    if cost.get("cpu_s"):
        extra.append(f"CPU {_dur(cost['cpu_s'])}")
    if cost.get("peak_mb"):
        extra.append(f"peak memory {_bytes(float(cost['peak_mb']) * 1048576)}")
    if c.get("pages_per_min"):
        extra.append(f"{c['pages_per_min']} pages/min")
    if extra:
        lines.append(f"{indent}cost: " + ", ".join(extra))
    if c.get("step_s"):
        lines.append(f"{indent}pages: " + ", ".join(f"{_stage(k)} {_dur(v)}" for k, v in c["step_s"].items())
                     + (f"  ({c['cached_pages']} from the page cache)" if c.get("cached_pages") else ""))
    elif c.get("cached_pages"):
        lines.append(f"{indent}{c['cached_pages']} page(s) from the page cache")
    if c.get("tokens"):
        lines.append(f"{indent}document reader: {c['tokens']} tokens" + (f" in {_dur(c['gpu_s'])}" if c.get("gpu_s") else ""))
    if c.get("repair_tried"):
        lines.append(f"{indent}repair: {c.get('repaired_cells', 0)} of {c['repair_tried']} suspect cell(s) fixed")
    if c.get("gate_failed"):
        lines.append(f"{indent}quality gate, failed checks: "
                     + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in c["gate_failed"].items()))
    if c.get("low_docs"):
        lines.append(f"{indent}low-confidence documents: {c['low_docs']}")
    return lines


def _fmt_doc(d: dict[str, Any]) -> str:
    name = f"{d.get('collection', '')}/{d.get('source', '')}"
    st = d.get("status")
    conv = _fmt_conv_doc(d.get("conversion"))
    conv = f"  [{conv}]" if conv else ""
    if st == "indexed":
        return (f"[ ok ] {name}  {d.get('chunks', '?')} chunks  3 convert {_dur(d.get('convert_s'))}  "
                f"4 chunk {_dur(d.get('chunk_s'))}  5 embed {_dur(d.get('embed_s'))}  "
                f"total {_dur(d.get('total_s'))}{conv}")
    if st == "converted":
        return (f"[conv] {name}  {d.get('chunks', '?')} chunks  3 convert {_dur(d.get('convert_s'))}  "
                f"4 chunk {_dur(d.get('chunk_s'))}  (waiting for 5 embed){conv}")
    if st == "skipped":
        return f"[skip] {name}  unchanged"
    if st == "no_text":
        return f"[skip] {name}  no text to index"
    if st == "unsupported":
        return f"[skip] {name}  unsupported format ({d.get('extension') or 'no extension'})"
    if st == "removed":
        return f"[gone] {name}  source deleted: its Markdown and index were removed"
    return f"[FAIL] {name}  after {_dur(d.get('total_s'))}: {d.get('message', '')}"


def _fmt_docs(docs: dict[str, Any] | None) -> str:
    if not docs or not docs.get("total"):
        return ""
    by = docs.get("by_status", {})
    waiting = f", converted {by['converted']} (waiting for embedding)" if by.get("converted") else ""
    empty = f", no text {by['no_text']}" if by.get("no_text") else ""
    unsup = f", unsupported format {by['unsupported']}" if by.get("unsupported") else ""
    gone = f", removed {by['removed']}" if by.get("removed") else ""
    head = (f"documents: {docs['total']} handled (indexed {by.get('indexed', 0)}{waiting}, unchanged "
            f"{by.get('skipped', 0)}{empty}{unsup}{gone}, failed {by.get('error', 0)})")
    if docs.get("by_branch"):
        head += "\n  documents with pages of each branch: " + _fmt_branches(docs["by_branch"])
    if docs.get("by_outcome"):
        head += "\n  documents by page outcome: " + _fmt_outcomes(docs["by_outcome"])
    items = docs.get("items", [])
    matched = docs.get("matched", docs["total"])
    if matched != docs["total"]:
        head += f"; {matched} match the filter"
    if items and len(items) < matched:
        head += f"; last {len(items)}:"
    elif items:
        head += ":"
    return "\n".join([head, *("  " + _fmt_doc(d) for d in items)])


def _fmt_job(job: dict[str, Any] | None) -> str:
    if not job:
        return "no indexing run yet"
    line = f"job {job['id']}: {job['status']}"
    if job.get("mode"):
        line += f" [{job['mode']}{' ' + job['path'] if job.get('path') else ''}]"
    active = job["status"] in ("running", "queued")
    if job.get("started_at"):
        line += f"\n  started {_clock(job['started_at'])}"
        if job.get("elapsed_s") is not None:
            line += f", {'running for' if active else 'took'} {_dur(job['elapsed_s'])}"
    if active:
        line += "\n  " + _fmt_progress_live(job.get("progress"))
    summ = job.get("summary")
    if summ:
        line += (f"\n  indexed {summ.get('indexed', 0)}, unchanged {summ.get('skipped_fresh', 0)}, "
                 f"no text {summ.get('no_text_count', 0)}, errors {summ.get('error_count', 0)}")
        if summ.get("unsupported_count"):
            line += f", unsupported format {summ['unsupported_count']}"
        if summ.get("not_retried"):
            line += f"\n  {summ['not_retried']} of those are unchanged files that failed or had no text before: not tried again"
        if summ.get("removed_count"):
            line += f", removed {summ['removed_count']} (source deleted)"
        if summ.get("unreachable"):
            line += ("\n  skipped, not reachable (kept as indexed before): "
                     + ", ".join(summ["unreachable"]))
        if summ.get("phase_s"):
            line += "\n  time by phase: " + ", ".join(
                f"{k} {_dur(v)}" for k, v in summ["phase_s"].items())
        conv_lines = _fmt_conv_totals(summ.get("conversion"), "    ")
        if conv_lines:
            line += "\n  conversion:\n" + "\n".join(conv_lines)
        for e in summ.get("errors", [])[:5]:
            line += f"\n    ! {e.get('src')}: {e.get('message')}"
        for u in summ.get("unsupported_extension", [])[:5]:
            line += f"\n    ~ {u.get('src')}: unsupported format ({u.get('extension') or 'no extension'})"
    if job.get("error"):
        line += f"\n  error: {job['error']}"
    pub = job.get("publish")
    if pub:
        line += "\n  published generation " + str(pub.get("generation")) if pub.get("changed") \
            else "\n  nothing new to publish"
        if pub.get("error"):
            line += f"\n  publish error: {pub['error']}"
        if pub.get("incomplete"):
            line += (f"\n  warning: {len(pub['incomplete'])} document(s) not fully indexed and "
                     "missing from search until the next `index new`: "
                     + ", ".join(pub["incomplete"][:5]))
    rl = job.get("search_reload")
    if rl:
        line += "\n  search daemon: " + ("reloaded" if rl.get("ok") else f"FAILED {rl.get('error')}")
    return line


# ── search side ──────────────────────────────────────────────────────────────

def _cmd_search(a: argparse.Namespace) -> int:
    r = api.search(get_paths(), a.query, top_k=a.top_k, collections=a.collection,
                   client=a.client, wait_s=a.wait, stages=a.stages,
                   retrieval_pool=a.retrieval_pool, rerank_pool=a.rerank_pool, rrf_k=a.rrf_k)
    if not r.get("ok"):
        return _fail(r)
    res = r["result"]
    if a.json:
        _json(res)
        return EXIT_OK
    for hit in res.get("results", []):
        head = f" — {hit['heading']}" if hit.get("heading") else ""
        low = "  (low-confidence page: check the source)" if hit.get("confidence") == "low" else ""
        print(f"{hit['rank']}. [{hit['score']}] {hit['file']} ({hit['collection']}) "
              f"p.{hit['page']}{head}{low}")
        if a.explain:
            print("   " + _fmt_hit_breakdown(hit))
        text = " ".join(hit["text"].split())
        print("   " + (text[:300] + " …" if len(text) > 300 else text))
    if not res.get("results"):
        print(res.get("note") or "no results")
    print(_fmt_search_timing(res, explain=a.explain))
    return EXIT_OK


def _fmt_hit_breakdown(hit: dict[str, Any]) -> str:
    def part(label: str, score: Any, rank: Any) -> str:
        if score is None:
            return f"{label} —"
        return f"{label} #{rank} ({score})" if rank else f"{label} ({score})"
    parts = [part("BM25", hit.get("bm25_score"), hit.get("bm25_rank")),
             part("Dense", hit.get("dense_score"), hit.get("dense_rank"))]
    if hit.get("rrf_score") is not None:
        parts.append(f"RRF {hit['rrf_score']}")
    rerank_score = hit.get("rerank_score")
    parts.append(f"Rerank {rerank_score}" if rerank_score is not None else "Rerank —")
    return " · ".join(parts)


def _ms(v: Any) -> str:
    return f"{v} ms" if v is not None else "?"


def _fmt_search_timing(res: dict[str, Any], explain: bool = False) -> str:
    t = res.get("timing") or {}
    if not t:
        return ""
    parts = [f"embed query {_ms(t.get('embed_query_ms'))}", f"keyword {_ms(t.get('keyword_ms'))}",
             f"vectors {_ms(t.get('dense_ms'))}"]
    parts.append(f"rerank {_ms(t.get('rerank_ms'))}" if t.get("reranked") else "no rerank")
    extra = []
    if t.get("queue_ms"):
        extra.append(f"waited {_ms(t['queue_ms'])}")
    if t.get("round_trip_ms") is not None:
        extra.append(f"round trip {_ms(t['round_trip_ms'])}")
    line = (f"{len(res.get('results', []))} result(s) in {_ms(t.get('total_ms'))} "
            f"({', '.join(parts)}; {t.get('collections', '?')} collection(s), "
            f"generation {t.get('generation')}" + (f"; {', '.join(extra)}" if extra else "") + ")")
    if not explain or not t:
        return line
    stages = ",".join(t.get("stages") or [])
    diag = [f"stages={stages}", f"retrieval_pool={t.get('retrieval_pool')}",
            f"rerank_pool={t.get('rerank_pool')}", f"rrf_k={t.get('rrf_k')}",
            f"candidates={t.get('candidates')} "
            f"(bm25={t.get('bm25_candidates', 0)}, dense={t.get('dense_candidates', 0)}, "
            f"overlap={t.get('overlap_count', 0)}, bm25_only={t.get('bm25_only_count', 0)}, "
            f"dense_only={t.get('dense_only_count', 0)})"]
    for label in ("bm25", "dense", "rerank"):
        gap = t.get(f"{label}_gap")
        if gap is not None:
            diag.append(f"{label} top1→top2 gap={gap} (top1={t.get(f'{label}_top1')})")
    if t.get("rerank_error"):
        diag.append(f"rerank_error={t['rerank_error']!r}")
    return line + "\n   " + "; ".join(diag)


def _cmd_grep(a: argparse.Namespace) -> int:
    r = api.grep(get_paths(), a.pattern, collections=a.collection, context_lines=a.context,
                 max_matches=a.max, client=a.client)
    if not r.get("ok"):
        return _fail(r)
    res = r["result"]
    if a.json:
        _json(res)
        return EXIT_FAIL if res.get("error") else EXIT_OK
    if res.get("error"):
        _err(f"error: {res['error']}")
        return EXIT_FAIL
    for m in res["matches"]:
        print(f"{m['collection']}/{m['doc']}  p.{m['page']}  line {m['line']}")
        for c in m["context"]:
            print("    " + c)
    t = res.get("timing") or {}
    print(f"{res['total_matches']} match(es)" + (" (truncated)" if res.get("truncated") else "")
          + (f" in {_ms(t.get('total_ms'))} (scan {_ms(t.get('scan_ms'))}, "
             f"{t.get('files_scanned', '?')} files"
             + (f"; round trip {_ms(t['round_trip_ms'])}" if t.get("round_trip_ms") is not None else "")
             + ")" if t else ""))
    return EXIT_OK


def _cmd_list(a: argparse.Namespace) -> int:
    r = api.list_collections(get_paths(), client=a.client, full=True)
    if not r.get("ok"):
        return _fail(r)
    res = r["result"]
    if a.json:
        _json(res)
        return EXIT_OK
    print(f"generation {res.get('generation')}, published {_clock(res.get('published_at'))}")
    t = res.get("totals") or {}
    if t.get("collections"):
        print(f"  total: {t['collections']} collection(s), {t['documents']} doc(s), "
              f"{t['chunks']} chunks, index {_bytes(t.get('index_bytes'))}"
              + (f", built in {_dur(t['build_seconds'])}" if t.get("build_seconds") is not None else ""))
    for c in res["collections"]:
        print(f"  {c['collection']}: "
              f"{len(c['documents'])} doc(s), {c['chunks']} chunks, "
              f"index {_bytes(c.get('index_bytes'))}")
        if c.get("description"):
            print(f"      {c['description']}")
        built = f"built {_clock(c.get('built_at'))}"
        if c.get("build_seconds") is not None:
            n, total = c.get("build_seconds_documents", 0), len(c["documents"])
            built += f", took {_dur(c['build_seconds'])}" + (
                f" (timed for {n} of {total} docs)" if n < total else "")
        else:
            built += ", build time not recorded (indexed by an older version)"
        print(f"      {built}; markdown {_bytes(c.get('markdown_bytes'))}, "
              f"sources {_bytes(c.get('source_bytes'))}")
        for d in c["documents"]:
            bits = [f"{d['chunks']} chunks"]
            if d.get("index_bytes") is not None:
                bits.append(_bytes(d["index_bytes"]))
            if d.get("build_s") is not None:
                bits.append(f"built in {_dur(d['build_s'])}")
            print(f"      {d['name']}  ({', '.join(bits)})")
    if res.get("hint"):
        print(res["hint"])
    return EXIT_OK


# ── indexing ─────────────────────────────────────────────────────────────────

def _follow(paths, job_id: str, as_json: bool) -> int:
    last = ""
    final: dict[str, Any] | None = None
    try:
        for ev in api.index_follow(paths, job_id=job_id):
            if as_json:
                print(json.dumps(ev, ensure_ascii=False), flush=True)
            elif ev.get("event") == "progress":
                line = _fmt_progress(ev)
                if line != last:
                    print("  " + line, flush=True)
                    last = line
            elif ev.get("event") == "doc":
                print("  " + _fmt_doc(ev), flush=True)
            elif ev.get("event") in ("status", "end"):
                if ev["event"] == "status":
                    print(_fmt_job(ev.get("job")), flush=True)
                else:
                    final = ev.get("job")
            elif ev.get("ok") is False:
                return _fail(ev)
    except KeyboardInterrupt:
        _err("\n(stopped following; the indexing run continues in the background)")
        return EXIT_OK
    if final and not as_json:
        print(_fmt_job(final))
    return EXIT_FAIL if final and final.get("status") in ("failed", "partial") else EXIT_OK


def _cmd_index_start(a: argparse.Namespace) -> int:
    paths = get_paths()
    r = api.index_start(paths, mode=a.mode, path=a.path, rebuild=a.rebuild, force_md=a.force_md,
                        restart=a.restart, client=a.client)
    if not r.get("ok"):
        return _fail(r)
    if a.json and not a.follow:
        _json(r)
        return EXIT_OK
    if not a.json:
        if r.get("already_running"):
            print("an indexing run is already active (use --restart to start over):")
        elif r.get("restarted"):
            print("restarted indexing:")
        else:
            print("indexing started in the background:")
        print("  " + _fmt_job(r["job"]).replace("\n", "\n  "))
    if a.follow:
        return _follow(paths, r["job"]["id"], a.json)
    return EXIT_OK


def _cmd_index_status(a: argparse.Namespace) -> int:
    paths = get_paths()
    if a.follow:
        return _follow(paths, a.job_id, a.json)
    r = api.index_status(paths, job_id=a.job_id, history=a.history, docs=a.docs,
                        doc_status=a.doc_status, doc_collection=a.doc_collection, doc_q=a.doc_q,
                        doc_branch=a.doc_branch, doc_outcome=a.doc_outcome)
    if not r.get("ok"):
        return _fail(r)
    if a.json:
        _json(r)
        return EXIT_OK
    if r.get("daemon") == "not running":
        print("indexer daemon is not running (nothing is being indexed)")
    print(_fmt_job(r.get("job")))
    docs_txt = _fmt_docs(r.get("documents"))
    if docs_txt:
        print(docs_txt)
    for j in r.get("history", []):
        print(f"  - {j['id']}  {j['status']}")
    return EXIT_OK


def _cmd_index_estimate(a: argparse.Namespace) -> int:
    r = api.conversion_estimate(get_paths(), a.path)
    if not r.get("ok"):
        return _fail(r)
    e = r["result"]
    if a.json:
        _json(e)
        return EXIT_OK
    print(f"{e['profiled']} of {e['files']} file(s) profiled in {_dur(e['profile_s'])}"
          + (" (time limit reached: the rest is not counted)" if e.get("partial") else ""))
    print(f"pages: {e['pages']}  ({_fmt_branches(e['branches'])})")
    if e.get("by_extension"):
        print("files: " + ", ".join(f"{k or 'none'} {v}" for k, v in sorted(e["by_extension"].items())))
    if e.get("scripts"):
        print("scripts: " + ", ".join(f"{k} {v}" for k, v in sorted(e["scripts"].items())))
    d = e["docling"]
    print(f"docling today: about {_dur(d['seconds'])} ({d['s_per_page']} s/page, {d['basis']})")
    v = e["planned_vlm"]
    if v["pages"]:
        rd = v.get("reader") or {}
        state = (f"document reader {rd['model']}: {_dur(v['seconds_low'])} to {_dur(v['seconds_high'])}"
                 if rd.get("usable") else f"document reader cannot run now ({rd.get('why', '')}): docling OCR")
        print(f"scanned pages and images: {v['pages']}  ({state})")
    for x in e.get("errors", [])[:5]:
        print(f"  ! {x['src']}: {x['message']}")
    return EXIT_OK


def _cmd_trace(a: argparse.Namespace) -> int:
    coll, _, doc = a.target.partition("/")
    if not coll or not doc:
        return _fail({"error": "give COLLECTION/DOCUMENT, e.g. contracts/2024/lease"})
    if a.md:                                   # the converted text itself, not the record
        r = api.conversion_markdown(get_paths(), coll, doc, page=a.page)
        if not r.get("ok"):
            return _fail(r)
        if a.json:
            _json(r["result"])
        else:
            print(r["result"]["markdown"])
            if r["result"]["truncated"]:
                print(f"(cut: the whole text is in {r['result']['file']})", file=sys.stderr)
        return EXIT_OK
    r = api.conversion_trace(get_paths(), coll, doc, page=a.page)
    if not r.get("ok"):
        return _fail(r)
    t = r["result"]
    if a.json:
        _json(t)
        return EXIT_OK
    s = t.get("summary") or {}
    print(f"{t['collection']}/{t['doc']}  ({t.get('source') or '?'})")
    if t.get("convert"):
        print(f"  converted with {t['convert']}" + (f" at {t['written_at']}" if t.get("written_at") else ""))
    if s:
        print("  " + _fmt_conv_doc(s))
        for line in _fmt_conv_totals({**s, "docs": 1}, "  ")[1:]:
            print(line)
    if t.get("note"):
        print("  " + t["note"])
    if a.page:
        print(json.dumps(t["pages"][0], indent=2, ensure_ascii=False))
        return EXIT_OK
    for p in t["pages"]:
        bits = [f"p.{p['page']:<4} {p['branch']:<9} {p['outcome']}"]
        if p.get("chars") is not None:
            bits.append(f"{p['chars']} chars")
        if p.get("grade"):
            bits.append(f"docling {p['grade']}")
        if p.get("cache") == "hit":
            bits.append("page cache")
        if p.get("read_s"):
            bits.append(f"{p['read_s']} s")
        if p.get("tokens"):
            bits.append(f"{p['tokens']} tokens")
        if p.get("failed"):
            bits.append("failed: " + ", ".join(p["failed"]))
        if p.get("repair"):
            bits.append(f"repair {p['repair'].get('fixed', 0)}/{p['repair'].get('tried', 0)} cells")
        if p.get("across"):
            bits.append(f"table {p['across']} p.{p['across_with']}")
        print("  " + "  ".join(bits))
    return EXIT_OK


def _pct(v: Any) -> str:
    return "-" if v is None else f"{100 * v:.1f}%"


def _bench_summary_lines(s: dict[str, Any], indent: str = "  ") -> list[str]:
    return [f"{indent}pages {s.get('pages', 0)}   numeric cells exact {_pct(s.get('cell_exact'))} "
            f"(found anywhere {_pct(s.get('cell_bag'))}, {s.get('cells', 0)} cells)",
            f"{indent}CER {_pct(s.get('cer'))}   table similarity {_pct(s.get('table_sim'))}   "
            f"balance checks pass {_pct(s.get('balance_ok'))} ({s.get('balance_tables', 0)} tables)   "
            f"search phrases found {_pct(s.get('query_hit'))} ({s.get('queries', 0)})",
            f"{indent}time {s.get('s_per_page', '-')} s/page"]


def _cmd_bench_gold(a: argparse.Namespace) -> int:
    paths = get_paths()
    if a.gold_cmd == "init":
        r = api.bench_gold_init(paths, a.set, collection=a.collection, per_class=a.per_class,
                                classes=[c for c in a.classes.split(",") if c] or None, seed=a.seed)
    elif a.gold_cmd == "show":
        r = api.bench_gold_show(paths, a.set)
    else:
        r = api.bench_gold_list(paths)
    if not r.get("ok"):
        return _fail(r)
    res = r["result"]
    if a.json:
        _json(res)
        return EXIT_OK
    if a.gold_cmd == "init":
        print(f"gold set {res['set']!r}: added {res['added_total']} page(s), {res['total']} in total  ({res['dir']})")
        for k, v in sorted(res["added"].items()):
            print(f"  {k:<18} +{v}   (of {res['classes_available'].get(k, 0)} candidate pages)")
        for x in res["skipped"][:10]:
            print(f"  ! {x}")
        print("The text is pre-filled with what the pipeline read: open gold.json next to the images, correct "
              'each "truth" against the page, and set "verified": true. Runs measure verified pages only.')
    elif a.gold_cmd == "show":
        print(f"{res['name']}  ({res['dir']})")
        for p in res["pages"]:
            print(f"  {p['id']}  {p['class']:<18} {p['rel']} p.{p['page']:<4} "
                  f"{'verified' if p['verified'] else 'draft':<8} {p['chars']} chars, {p['queries']} phrase(s)")
    else:
        if not res["sets"]:
            print("no gold sets yet (rag-search bench gold init SET)")
        for g in res["sets"]:
            print(f"{g['name']}: {g['pages']} page(s), {g['verified']} verified, {g['runs']} run(s)  "
                  + ", ".join(f"{k} {v}" for k, v in sorted(g["classes"].items())))
        print("engines: " + ", ".join(res["engines"]))
    return EXIT_OK


def _cmd_bench_run(a: argparse.Namespace) -> int:
    def progress(p: dict[str, Any]) -> None:
        if not a.json:
            print(f"\r  file {p['file']}/{p['files']}, {p['pages']} page(s) read", end="", file=sys.stderr, flush=True)

    r = api.bench_run(get_paths(), a.set, engine=a.engine, name=a.name, include_drafts=a.drafts,
                      classes=[c for c in a.classes.split(",") if c] or None, limit=a.limit, progress=progress)
    if not a.json:
        print(file=sys.stderr)
    if not r.get("ok"):
        return _fail(r)
    rec = r["result"]
    if a.json:
        _json(rec)
        return EXIT_OK
    print(f"run {rec['id']}  engine {rec['engine']}  {_dur(rec['seconds'])}, peak {rec['cost']['peak_mb']} MB")
    for line in _bench_summary_lines(rec["summary"]["all"]):
        print(line)
    for c, s in rec["summary"]["by_class"].items():
        print(f"  {c}")
        for line in _bench_summary_lines(s, "      "):
            print(line)
    for x in rec["skipped"][:10]:
        print(f"  ! {x['id']} skipped: {x['why']}")
    for row in rec["pages"]:
        if row.get("error"):
            print(f"  ! {row['file']} p.{row['page']}: {row['error']}")
    if rec["drafts_included"]:
        print("  (drafts included: pages not yet verified are compared with the pipeline's own text)")
    return EXIT_OK


def _cmd_bench_list(a: argparse.Namespace) -> int:
    r = api.bench_list(get_paths(), a.set)
    if not r.get("ok"):
        return _fail(r)
    if a.json:
        _json(r["result"])
        return EXIT_OK
    if not r["result"]:
        print("no bench runs yet (rag-search bench run SET)")
    for x in r["result"]:
        s = x["summary"]
        print(f"{x['set']}/{x['id']}  {x['engine']:<14} cells {_pct(s.get('cell_exact'))}  CER {_pct(s.get('cer'))}  "
              f"balance {_pct(s.get('balance_ok'))}  {s.get('s_per_page', '-')} s/page  ({s.get('pages', 0)} pages)")
    return EXIT_OK


def _cmd_bench_show(a: argparse.Namespace) -> int:
    r = api.bench_show(get_paths(), a.set, a.run)
    if not r.get("ok"):
        return _fail(r)
    rec = r["result"]
    if a.json:
        _json(rec)
        return EXIT_OK
    print(f"{rec['set']}/{rec['id']}  engine {rec['engine']}  started {rec['started_at']}")
    for line in _bench_summary_lines(rec["summary"]["all"]):
        print(line)
    for c, s in rec["summary"]["by_class"].items():
        print(f"  {c}")
        for line in _bench_summary_lines(s, "      "):
            print(line)
    worst = sorted(rec["pages"], key=lambda p: (p["cells"]["exact"] - p["cells"]["total"], -(p.get("cer") or 0)))[:5]
    print("  weakest pages:")
    for p in worst:
        print(f"    {p['id']} {p['file']} p.{p['page']}  cells {p['cells']['exact']}/{p['cells']['total']}  CER {_pct(p.get('cer'))}")
    return EXIT_OK


def _cmd_bench_compare(a: argparse.Namespace) -> int:
    r = api.bench_compare(get_paths(), a.set, a.run_a, a.run_b)
    if not r.get("ok"):
        return _fail(r)
    c = r["result"]
    if a.json:
        _json(c)
        return EXIT_OK
    print(f"{c['a']['id']} ({c['a']['engine']})  ->  {c['b']['id']} ({c['b']['engine']})")

    def row(label: str, d: dict[str, Any]) -> None:
        bits = []
        for m, x in d.items():
            if x["delta"] is None:
                continue
            mark = "" if x["better"] is None else (" better" if x["better"] else " WORSE")
            val = (lambda v: f"{v:g}") if m == "s_per_page" else _pct
            bits.append(f"{m} {val(x['a'])} -> {val(x['b'])}{mark}")
        print(f"  {label:<18} " + "; ".join(bits))

    row("all", c["all"])
    for k, d in c["by_class"].items():
        row(k, d)
    for w in c["worse"][:10]:
        print(f"  worse: {w['id']} {w['file']} p.{w['page']} cells {w['cells_exact'][0]} -> {w['cells_exact'][1]} of {w['of']}")
    return EXIT_OK


def _cmd_index_cache(a: argparse.Namespace) -> int:
    r = api.page_cache(get_paths(), clear=a.clear)
    if not r.get("ok"):
        return _fail(r)
    res = r["result"]
    if a.json:
        _json(res)
        return EXIT_OK
    if a.clear:
        print(f"removed {res['removed']} cached page(s)")
    print(f"page cache: {res['entries']} page(s), {_bytes(res['bytes'])}  ({res['dir']})")
    return EXIT_OK


def _cmd_index_cancel(a: argparse.Namespace) -> int:
    r = api.index_cancel(get_paths(), client=a.client)
    if not r.get("ok"):
        return _fail(r)
    if a.json:
        _json(r)
    else:
        print("cancelled" if r.get("cancelled") else r.get("note", "nothing to cancel"))
    return EXIT_OK


def _cmd_index_publish(a: argparse.Namespace) -> int:
    r = api.index_publish(get_paths(), client=a.client)
    if not r.get("ok"):
        return _fail(r)
    if a.json:
        _json(r)
    else:
        pub = r["publish"]
        print(f"published generation {pub.get('generation')}" if pub.get("changed")
              else pub.get("note") or pub.get("error") or "nothing to publish")
        if r.get("search_reload"):
            print("search daemon:", r["search_reload"])
    return EXIT_OK


def _cmd_index_foreground(a: argparse.Namespace) -> int:
    """Index in this process (no daemon), then publish.  For scripts and debugging."""
    from .core import worker
    from .core.indexer import IndexBusyError

    paths = get_paths()
    ensure_dirs(paths)

    class _Print:
        last = ""

        def progress(self, ev: dict[str, Any]) -> None:
            line = _fmt_progress(ev)
            if line != self.last and ev.get("phase") in ("convert", "embed", "merge"):
                print("  " + line, file=sys.stderr, flush=True)
                self.last = line

    class _Events(worker.EventWriter):
        def __init__(self):
            self.p = _Print()

        def emit(self, event: str, **fields: Any) -> None:
            pass

        def progress(self, ev: dict[str, Any]) -> None:
            self.p.progress(ev)

        def close(self) -> None:
            pass

    from .config import ConfigStore, effective_jobs

    spec = {"mode": a.mode, "path": a.path, "rebuild": a.rebuild, "force_md": a.force_md,
            "jobs": effective_jobs(ConfigStore(paths).get())}
    from .core.embedding import prepare_environment

    prepare_environment()
    try:
        summary = worker.run_spec(paths, spec, _Events())
    except IndexBusyError as exc:
        _err(f"error: {exc}")
        return EXIT_UNAVAILABLE
    except ValueError as exc:
        _err(f"error: {exc}")
        return EXIT_USAGE
    pub = api.publish_and_reload(paths)
    out = {"summary": summary, **pub}
    if a.json:
        _json(out)
    else:
        uns = summary.get("unsupported_extension", [])
        print(f"indexed {summary['indexed']}, unchanged {summary['skipped_fresh']}, "
              f"no text {len(summary.get('no_text', []))}, "
              f"errors {len(summary['errors'])}"
              + (f", unsupported format {len(uns)}" if uns else "")
              + (f", removed {len(summary['removed'])} (source deleted)"
                 if summary.get("removed") else "")
              + f" in {summary['elapsed_s']}s")
        if summary.get("unreachable"):
            print("  skipped, not reachable (kept as indexed before): "
                  + ", ".join(summary["unreachable"]))
        for e in summary["errors"][:10]:
            print(f"  ! {e['src']}: {e['message']}")
        for u in uns[:10]:
            print(f"  ~ {u['src']}: unsupported format ({u['extension'] or 'no extension'})")
        p = pub.get("publish", {})
        print(f"published generation {p.get('generation')}" if p.get("changed")
              else p.get("error") or "nothing new to publish")
    return EXIT_FAIL if summary["errors"] else EXIT_OK


# ── daemons / service ────────────────────────────────────────────────────────

def _search_status_lines(info: dict[str, Any]) -> list[str]:
    """Warm-up and memory details of the search daemon (`rag-search daemon status`)."""
    out: list[str] = []
    w = info.get("warmup") or {}
    if w.get("status") == "warm":
        out.append(f"warm-up: took {_dur(w.get('total_s'))} (models {_dur(w.get('models_s'))}, "
                   f"indexes {_dur(w.get('index_s'))}), ready since {_clock(w.get('ready_at'))}")
    elif w.get("status") == "warming_up":
        out.append(f"warming up: {w.get('phase', '')} for {_dur(w.get('elapsed_s'))} so far"
                   + (f" (models loaded in {_dur(w['models_s'])})" if w.get("models_s") else ""))
    m = info.get("memory") or {}
    if m.get("rss_bytes") or m.get("embeddings_bytes"):
        bits = [f"{_bytes(m.get('rss_bytes'))} resident (peak {_bytes(m.get('peak_rss_bytes'))})"]
        idx = (m.get("embeddings_bytes") or 0) + (m.get("text_bytes") or 0)
        if info.get("collections"):
            bits.append(f"index data {_bytes(idx)} (vectors {_bytes(m.get('embeddings_bytes'))}, "
                        f"chunk text {_bytes(m.get('text_bytes'))})")
        mods = m.get("models_bytes") or {}
        if mods:
            bits.append("models " + _bytes(sum(mods.values())))
        out.append("memory: " + "; ".join(bits))
        for name, c in (m.get("collections") or {}).items():
            out.append(f"  {name}: {c['chunks']} chunks, vectors {_bytes(c['embeddings_bytes'])}, "
                       f"text {_bytes(c['text_bytes'])}")
    lr = info.get("last_reload") or {}
    if lr:
        out.append(f"last reload: generation {lr.get('generation')} at {_clock(lr.get('at'))}, "
                   f"{_dur(lr.get('seconds'))} (reused {len(lr.get('reused', []))}, "
                   f"loaded {len(lr.get('loaded', []))} collection(s))")
    if info.get("index_error"):
        out.append(f"index error: {info['index_error']}")
    return out


def _cmd_daemon(a: argparse.Namespace) -> int:
    paths = get_paths()
    which = a.which
    if a.action == "status":
        st = api.daemon_status(paths)
        if a.json:
            _json(st)
            return EXIT_OK
        for kind, info in st.items():
            if info.get("state"):
                extra = ""
                if kind == "search":
                    extra = (f", generation {info.get('generation')}, {info.get('collections')} "
                             f"collection(s), {info.get('chunks')} chunks")
                else:
                    extra = ", run active" if info.get("running") else ", idle"
                label = info["state"]
                if kind == "search" and info.get("warmup"):
                    label += {"warm": " - warmed up", "warming_up": " - warming up",
                              "error": " - NOT warm"}.get(info["warmup"].get("status"), "")
                print(f"{kind}: {label} (pid {info['pid']}, up {_dur(info['uptime_s'])}{extra})")
                if info.get("error"):
                    print(f"  error: {info['error']}")
                if kind == "search":
                    for line in _search_status_lines(info):
                        print("  " + line)
            else:
                print(f"{kind}: {'starting' if info.get('starting') else 'not running'}")
        return EXIT_OK
    if a.action == "run":
        if which not in ("search", "indexer"):
            _err("daemon run needs 'search' or 'indexer'")
            return EXIT_USAGE
        import runpy

        from .client import DAEMON_MODULES

        runpy.run_module(DAEMON_MODULES[which], run_name="__main__")
        return EXIT_OK
    try:
        res = {"start": api.daemon_start, "stop": api.daemon_stop,
               "restart": api.daemon_restart}[a.action](paths, which)
    except ValueError as exc:
        _err(str(exc))
        return EXIT_USAGE
    if a.json:
        _json(res)
    else:
        flat = res if a.action != "restart" else {**{f"stop {k}": v for k, v in res["stop"].items()},
                                                  **{f"start {k}": v for k, v in res["start"].items()}}
        for k, v in flat.items():
            print(f"{k}: {v}")
    return EXIT_OK


def _cmd_service(a: argparse.Namespace) -> int:
    from . import service

    paths = get_paths()
    if a.action == "install":
        for line in service.install(paths):
            print(line)
    elif a.action == "uninstall":
        for line in service.uninstall(paths):
            print(line)
    else:
        st = service.status(paths)
        if a.json:
            _json(st)
        else:
            for kind, e in st.items():
                print(f"{kind}: installed={e['installed']} loaded={e.get('loaded', '-')} "
                      f"running={e['running']}")
    return EXIT_OK


# ── local commands ───────────────────────────────────────────────────────────

def _cmd_paths(a: argparse.Namespace) -> int:
    p = get_paths()
    info = {"home": p.home, "locations": p.locations_file, "workspace": p.workspace, "index": p.index,
            "markup": p.markup, "serving": p.serving, "current": p.current_link, "run": p.run,
            "jobs": p.jobs, "config": p.config_file, "search_socket": p.socket("search"),
            "indexer_socket": p.socket("indexer"), "search_log": p.log_file("search"),
            "indexer_log": p.log_file("indexer")}
    if a.name:
        if a.name not in info:
            _err(f"unknown name; choose from {', '.join(info)}")
            return EXIT_USAGE
        print(info[a.name])
    else:
        _json({k: str(v) for k, v in info.items()})
    return EXIT_OK


def _cmd_config(a: argparse.Namespace) -> int:
    from .config import ConfigStore, update_config, write_default_config

    paths = get_paths()
    if a.action == "path":
        print(paths.config_file)
    elif a.action == "init":
        ensure_dirs(paths)
        f = write_default_config(paths)
        print(f"config file: {f}")
    elif a.action == "set":
        overrides: dict[str, dict[str, Any]] = {"search": {}, "indexer": {}, "models": {}}
        for t in spec.TUNABLES:
            v = getattr(a, t.key, None)
            if v is not None:
                overrides[t.section][t.key] = v
        if not any(overrides.values()):
            _err("error: config set needs at least one --flag; see `rag-search config set --help`")
            return EXIT_USAGE
        try:
            validated = {section: spec.validate_section(section, values)
                        for section, values in overrides.items() if values}
            for section, values in validated.items():
                update_config(paths, section, values)
        except ValueError as exc:
            _err(f"error: {exc}")
            return EXIT_USAGE
        for section, values in validated.items():
            for key, v in values.items():
                label = spec.TUNABLES_BY_KEY[key].label
                print(f"{section}.{key} ({label}) = {v if v != '' else '(built-in default)'}")
        if a.json:
            _json(ConfigStore(paths).get())
    else:
        store = ConfigStore(paths)
        _json(store.get())
        if store.error:
            _err(f"warning: {store.error}")
    return EXIT_OK


# ── access control ───────────────────────────────────────────────────────────

def _who(clients: Any) -> str:
    if clients is None:
        return "everyone"
    return ", ".join(clients) if clients else "nobody (only the admin CLI)"


def _access_seen() -> dict[str, float]:
    from . import client as client_mod

    info = client_mod.ping(get_paths(), "search")
    return dict(info.get("clients_seen") or {}) if info else {}


def _cmd_access(a: argparse.Namespace) -> int:
    from . import access

    paths = get_paths()
    cmd = getattr(a, "access_cmd", None) or "list"
    try:
        if cmd == "list":
            ov = access.overview(paths, _access_seen())
            if a.json:
                _json(ov)
                return EXIT_OK
            if ov["error"]:
                _err(f"warning: {ov['error']} (restricted collections stay closed until it is fixed)")
            rows = ov["collections"]
            print("Every collection is open to all clients unless it is restricted below.")
            if not rows:
                print("\n(no collections yet: register a folder of documents with "
                      "`rag-search location add NAME FOLDER`, then run `rag-search index new`)")
            else:
                w = max([len("COLLECTION")] + [len(r["collection"]) for r in rows])
                print(f"\n  {'COLLECTION'.ljust(w)}  {'ACCESS'.ljust(10)}  "
                      f"{'CLIENTS ALLOWED'.ljust(26)}  STATE")
                for r in rows:
                    state = (f"{r['documents']} doc(s), {r['chunks']} chunks" if r["indexed"]
                             else "not indexed yet" if r["exists"]
                             else "rule only: no folder or index of that name yet")
                    print(f"  {r['collection'].ljust(w)}  {r['access'].ljust(10)}  "
                          f"{_who(r['clients']).ljust(26)}  {state}")
            names = ", ".join(c["client"] + (" (admin)" if c["client"] == "cli" else "")
                              for c in ov["clients"])
            print(f"\nClients: {names}")
            print("  restrict COLLECTION CLIENT...  sets who may use it (re-run with fewer names to "
                  "remove one; no names = nobody; 'all' = everyone)")
            print("  grant COLLECTION CLIENT...     adds clients to a restricted collection")
            print(f"Rules file: {ov['access_file']}")
            return EXIT_OK

        known = {c["client"] for c in access.clients(paths, _access_seen())}
        if cmd == "restrict":
            res = access.restrict(paths, a.collection, a.clients)
        else:  # grant
            res = access.grant(paths, a.collection, a.clients)
    except access.AccessError as exc:
        _err(f"error: {exc}")
        return EXIT_USAGE
    if a.json:
        _json(res)
        return EXIT_OK
    if res["clients"] is None:
        print(f"{res['collection']}: open to every client"
              + ("" if res.get("changed") else " (it already was)"))
    else:
        print(f"{res['collection']}: restricted to {_who(res['clients'])}")
        if res["clients"] == []:
            print("  no client can use it; only this terminal (`rag-search search --client ...` "
                  "shows what a client sees)")
    if not res.get("exists"):
        print("  note: that collection does not exist yet; the rule applies as soon as a folder "
              "with that name is indexed")
    for c in (a.clients if cmd in ("restrict", "grant") else []):
        if c.strip().lower() not in known:
            print(f"  note: '{c}' has not been seen before; it takes effect when a host is "
                  f"started as `rag-search-mcp --profile {c.strip().lower()}`")
    print("  takes effect immediately (no restart needed)")
    return EXIT_OK


def _cmd_describe(a: argparse.Namespace) -> int:
    """View or set a collection's description.

    This is normally set by the calling LLM itself, via the rag_describe_collection MCP
    tool, right after it explores a collection -- descriptions can't be generated
    automatically, since a collection is just whatever folder of documents someone
    indexed. This command is the human override/audit path: read what an agent set, or
    fix one that's wrong or stale.
    """
    from . import descriptions

    paths = get_paths()
    if not a.collection:
        by_name, error = descriptions.load_descriptions(paths)
        if error:
            _err(f"warning: {error}")
        if a.json:
            _json({"collections": by_name})
            return EXIT_OK
        if not by_name:
            print("No collections have a description yet.")
            print('  rag-search describe COLLECTION "text"   sets one')
            print("  rag-search describe COLLECTION --clear  clears one")
        else:
            w = max(len(n) for n in by_name)
            for name, text in sorted(by_name.items()):
                print(f"  {name.ljust(w)}  {text}")
        return EXIT_OK

    if a.text is None and not a.clear:
        text = descriptions.get_description(paths, a.collection)
        if a.json:
            _json({"collection": a.collection, "description": text})
            return EXIT_OK
        print(f"{a.collection}: {text}" if text else f"{a.collection}: no description set")
        return EXIT_OK

    r = api.describe_collection(paths, a.collection, "" if a.clear else a.text, client=a.client)
    if not r.get("ok"):
        _err(f"error: {r.get('error', 'failed')}")
        return EXIT_USAGE
    res = r["result"]
    if a.json:
        _json(res)
        return EXIT_OK
    print(f"{res['collection']}: {res['description']}" if res["description"]
          else f"{res['collection']}: description cleared")
    return EXIT_OK


# ── source locations, collection export/import/delete (administrator only) ───

def _fmt_publish(r: dict[str, Any]) -> str:
    pub = r.get("publish") or {}
    if pub.get("error"):
        return f"  publish: {pub['error']}"
    if pub.get("changed"):
        return f"  published generation {pub.get('generation')}"
    return "  " + (pub.get("note") or "nothing to publish")


def _cmd_location(a: argparse.Namespace) -> int:
    paths = get_paths()
    cmd = getattr(a, "location_cmd", None) or "list"
    if cmd == "list":
        r = api.location_list(paths)
        if a.json:
            _json(r["result"])
            return EXIT_OK
        res = r["result"]
        if res.get("error"):
            _err(f"warning: {res['error']}")
        if not res["locations"]:
            print("no registered locations.  Add one with:  rag-search location add NAME FOLDER")
        for loc in res["locations"]:
            state = "ok" if loc["reachable"] else "UNREACHABLE (skipped by indexing)"
            print(f"  {loc['collection']:<20} {loc['folder']}  [{state}]")
        return EXIT_OK
    if cmd == "add":
        r = api.location_add(paths, a.name, a.folder)
        if not r.get("ok"):
            return _fail(r)
        if a.json:
            _json(r["result"])
            return EXIT_OK
        print(f"collection {r['result']['collection']!r} now indexes {r['result']['folder']}")
        print("  (read only: rag-search never changes or deletes anything in that folder)")
        print("  next: rag-search index new --follow")
        return EXIT_OK
    # remove
    if not _confirm(f"unregister location {a.name!r} and delete its index (the folder itself and "
                    "its documents are not touched)?", a):
        return EXIT_USAGE
    r = api.location_remove(paths, a.name)
    if not r.get("ok"):
        return _fail(r)
    if a.json:
        _json(r)
        return EXIT_OK
    res = r["result"]
    print(f"removed location {res['collection']!r}: unregistered, Markdown and index deleted, "
          "documents untouched")
    print(_fmt_publish(r))
    return EXIT_OK


def _fmt_info(i: dict[str, Any]) -> str:
    """`rag-search collection info` as text: a short summary, then the details by topic."""
    src, ws, pub, b, att = i["source"], i["workspace"], i["published"], i["build"], i["attention"]
    lines = [f"{i['collection']}  ({i['kind']})  state: {i['state']}",
             f"  {i['state_detail']}"]
    if i.get("description"):
        lines.append(f"  {i['description']}")
    docs = f"{ws['documents']} indexed"
    if src.get("files") is not None:
        docs += f" of {src['files']} in the source folder"
    if src.get("unsupported"):
        docs += f" (+{src['unsupported']} in formats that are not indexed)"
    lines.append(f"  documents: {docs}; chunks {b.get('chunks', 0) if b else 0}; "
                 f"on disk {_bytes(i['disk']['total_bytes'])} (Markdown "
                 f"{_bytes(ws['markdown_bytes'])} + index {_bytes(ws['index_bytes'])})"
                 + (f"; sources {_bytes(src['bytes'])}" if src.get("bytes") is not None else ""))
    lines.append("where:")
    if src.get("folder"):
        lines.append(f"  source    {src['folder']}"
                     + ("" if src.get("reachable") else "  (NOT REACHABLE NOW)"))
    lines.append(f"  markdown  {ws['markdown_folder']}  ({ws['markdown_files']} file(s))")
    lines.append(f"  index     {ws['index_folder']}")
    if pub and pub.get("index_folder"):
        lines.append(f"  published {pub['index_folder']}  (generation {pub['generation']}, "
                     "hard links: no extra space)")
    if b:
        lines.append("indexing:")
        lines.append(f"  model {b.get('model')}" + (f" @ {str(b['model_revision'])[:12]}"
                                                    if b.get("model_revision") else "")
                     + f", {b.get('dim')} dims, chunks {b.get('chunk_size')}/"
                       f"{b.get('chunk_overlap')} tokens")
        lines.append(f"  first indexed {b.get('first_indexed') or '-'}, last indexed "
                     f"{b.get('last_indexed') or '-'}"
                     + (f", total build time {_dur(b['build_seconds'])}"
                        if b.get("build_seconds") is not None else ""))
    if pub:
        lines.append(f"  published {pub.get('published_at') or '-'} (generation "
                     f"{pub.get('generation')}): {pub['documents']} document(s), {pub['chunks']} chunks")
    run = i.get("last_run")
    if run:
        counts = ", ".join(f"{k} {v}" for k, v in sorted(run["counts"].items()))
        lines.append(f"  last run {run['job']} ({run['status']}): {counts}")
        for e in run["errors"][:5]:
            lines.append(f"    ! {e['source']}: {e['message']}")
    elif i["kind"] != "imported":
        lines.append("  last run: none recorded for this collection")
    conv = _fmt_conv_totals(i.get("conversion"))
    if conv:
        lines.append("conversion (pages as last converted):")
        lines.extend(conv)
        for key, label in (("poor_documents", "docling graded poor"), ("low_documents", "low confidence")):
            grp = (i.get("conversion") or {}).get(key) or {}
            if grp.get("count"):
                lines.append(f"  {label}: {grp['count']} document(s)")
                for it in grp.get("items", [])[:5]:
                    lines.append(f"    - {it['doc']}: page(s) {', '.join(str(x) for x in it['pages'])}")
    if i.get("origin"):
        o = i["origin"]
        lines.append(f"origin: export of {o.get('source_collection')!r} ({o.get('file')}), exported "
                     f"{o.get('exported_at') or '-'}, imported {o.get('imported_at') or '-'}")
    for key, label in (("not_indexed", "not indexed yet"),
                       ("modified_since_indexed", "changed since indexed"),
                       ("incomplete", "incomplete (interrupted run)")):
        grp = att[key]
        if grp["count"]:
            lines.append(f"{label}: {grp['count']}")
            for n in grp["names"][:10]:
                why = (grp.get("reasons") or {}).get(n)
                lines.append(f"  - {n}" + (f": {why}" if why else ""))
    return "\n".join(lines)


def _cmd_collection(a: argparse.Namespace) -> int:
    paths = get_paths()
    cmd = getattr(a, "collection_cmd", None)
    if cmd == "info":
        r = api.collection_info(paths, a.name)
        if not r.get("ok"):
            return _fail(r)
        if a.json:
            _json(r["result"])
        else:
            print(_fmt_info(r["result"]))
        return EXIT_OK
    if cmd == "export":
        r = api.collection_export(paths, a.name, a.output or None)
        if not r.get("ok"):
            return _fail(r)
        res = r["result"]
        if a.json:
            _json(res)
            return EXIT_OK
        print(f"exported {res['collection']!r}: {res['documents']} document(s), {res['chunks']} "
              f"chunks, {res['bytes'] / 1e6:.1f} MB")
        print(f"  file:  {res['file']}")
        print(f"  model: {res['model']}" + (f" @ {res['model_revision'][:12]}"
                                            if res.get("model_revision") else ""))
        print("  access rules are not exported; source documents are not included")
        return EXIT_OK
    if cmd == "import":
        r = api.collection_import(paths, a.file, as_name=a.as_name or None, replace=a.replace)
        if not r.get("ok"):
            return _fail(r)
        if a.json:
            _json(r)
            return EXIT_OK
        res = r["result"]
        verb = "replaced" if res["replaced"] else "imported"
        print(f"{verb} {res['collection']!r} ({res['documents']} document(s), {res['chunks']} "
              f"chunks, {res['model']})" + (f" from export {res['from']!r}"
                                             if res["from"] != res["collection"] else ""))
        if res.get("note"):
            print(f"  note: {res['note']}")
        print(f"  access: {res['access']}"
              + ("  (limit it with: rag-search access restrict "
                 f"{res['collection']} CLIENT ...)" if res["access"] == "everyone" else ""))
        print(_fmt_publish(r))
        return EXIT_OK
    if cmd == "delete":
        if not _confirm(f"delete collection {a.name!r} from the workspace: its converted Markdown "
                        "and its index (documents, access rule and description are kept)?", a):
            return EXIT_USAGE
        r = api.collection_delete(paths, a.name)
        if not r.get("ok"):
            return _fail(r)
        if a.json:
            _json(r)
            return EXIT_OK
        res = r["result"]
        print(f"deleted {res['collection']!r} ({res['kind']} collection) from the workspace: "
              "Markdown and index removed, documents untouched")
        if res.get("note"):
            print(f"  note: {res['note']}")
        print(_fmt_publish(r))
        return EXIT_OK
    _err("usage: rag-search collection {info,export,import,delete} ...")
    return EXIT_USAGE


# ── playground: a sandbox for trying models/tunables and benchmarking ────────
# Structurally separate from production (see core/playground.py's docstring): every
# function here lazily imports core.playground so this module stays light, and every
# error it raises is a ValueError (PlaygroundError or spec.parse_stages) caught the same
# way index/access do.

def _cmd_playground_create(a: argparse.Namespace) -> int:
    from .core import playground as pg

    try:
        res = pg.create_experiment(get_paths(), a.name, sources=a.source or None,
                                   collection=a.collection, from_production=a.from_production)
    except ValueError as exc:
        _err(f"error: {exc}")
        return EXIT_USAGE
    if a.json:
        _json(res)
        return EXIT_OK
    print(f"created playground experiment {a.name!r} at {res['home']}")
    if res["from_production"]:
        c = res["config"]
        print(f"  seeded from production: embedding={c['embedding_model']}, "
              f"reranker={c['rerank_model']}, chunk_size={c['chunk_size']}, "
              f"chunk_overlap={c['chunk_overlap']}")
    for src in res["sources"]:
        print(f"  source {src['collection']!r}: {src['folder']} (read in place, never copied)")
    if not res["sources"]:
        print(f"  add a source folder: rag-search playground source {a.name} add FOLDER, "
              f"then: rag-search playground index {a.name}")
    print(f"  a labeled-query template for benchmarking is at {res['home']}/bench/"
          "queries.jsonl.example")
    return EXIT_OK


def _cmd_playground_source(a: argparse.Namespace) -> int:
    from .core import playground as pg

    try:
        if a.action == "add":
            if not a.arg:
                raise ValueError("give the folder: playground source NAME add FOLDER [--as COLLECTION]")
            res: Any = pg.add_source(get_paths(), a.name, a.arg, a.collection)
            text = f"source {res['collection']!r}: {res['folder']}"
        elif a.action == "remove":
            if not a.arg:
                raise ValueError("give the collection: playground source NAME remove COLLECTION")
            res = {"removed": pg.remove_source(get_paths(), a.name, a.arg)}
            text = f"removed source {res['removed']!r} (its index stays until the next build prunes it)"
        else:
            res = pg.list_sources(get_paths(), a.name)
            text = "\n".join(f"{x['collection']}  {x['folder']}" + ("" if x["reachable"] else "  (unreachable)")
                             for x in res["status"]) or "no sources (rag-search playground source NAME add FOLDER)"
    except ValueError as exc:
        _err(f"error: {exc}")
        return EXIT_USAGE
    if a.json:
        _json(res)
    else:
        print(text)
    return EXIT_OK


def _cmd_playground_promote(a: argparse.Namespace) -> int:
    from .core import playground as pg

    try:
        if a.dry_run:
            res = pg.promotion_preview(get_paths(), a.name)
        else:
            res = pg.promote_to_production(get_paths(), a.name, confirm=a.confirm)
    except ValueError as exc:
        _err(f"error: {exc}")
        return EXIT_USAGE
    if a.json:
        _json(res)
        return EXIT_OK
    changes = res["changes"]
    if not changes:
        print(f"{a.name!r} already matches production -- nothing to promote")
        return EXIT_OK
    verb = "would change" if a.dry_run else "changed"
    print(f"promoting {a.name!r} {verb}:")
    for key, d in changes.items():
        print(f"  {key}: {d['from']} -> {d['to']}")
    if res["needs_reindex"]:
        est = (res.get("reindex_estimate") or {})
        cost = f"~{est.get('documents', '?')} document(s)"
        if est.get("estimated_s") is not None:
            cost += f", about {est['estimated_s']}s"
        print(f"  this makes the existing production index stale ({cost}) -- "
              + ("run without --dry-run and pass --confirm to apply, then `rag-search index "
                 "--rebuild`" if a.dry_run else "run `rag-search index --rebuild` next"))
    elif not a.dry_run:
        print("  no reindex needed (reranker/search tunables only, or nothing model/chunk-related)")
    return EXIT_OK


def _cmd_playground_config(a: argparse.Namespace) -> int:
    from .core import playground as pg

    overrides = {"embedding_model": a.embedding_model, "rerank_model": a.rerank_model,
                "chunk_size": a.chunk_size, "chunk_overlap": a.chunk_overlap,
                "stages": a.stages, "retrieval_pool": a.retrieval_pool,
                "rerank_pool": a.rerank_pool, "rrf_k": a.rrf_k,
                "reader_model": a.reader_model, "repair_model": a.repair_model,
                "rerank": (False if a.no_rerank else (True if a.rerank else None))}
    # the docling/OCR/table/PDF-backend flags below are generated from the same spec.py
    # registry production's own `config set` uses -- see the argparse loop that adds them.
    for t in pg.DOCLING_TUNABLES:
        overrides[t.key] = getattr(a, t.key, None)
    try:
        if any(v is not None for v in overrides.values()):
            cfg = pg.update_config(get_paths(), a.name, **overrides)
        else:
            cfg = pg.get_config(get_paths(), a.name)
    except ValueError as exc:
        _err(f"error: {exc}")
        return EXIT_USAGE
    if a.json:
        _json(cfg)
        return EXIT_OK
    print(f"playground {a.name!r} config:")
    for key in ("embedding_model", "rerank_model", "reader_model", "repair_model", "rerank", "chunk_size",
               "chunk_overlap", "stages", "retrieval_pool", "rerank_pool", "rrf_k",
               *(t.key for t in pg.DOCLING_TUNABLES)):
        val = cfg.get(key)
        if val in ("", None) and key in pg.MODEL_PINS:
            val = "(production's choice)"
        print(f"  {key}: {val}")
    return EXIT_OK


def _cmd_playground_index(a: argparse.Namespace) -> int:
    from .core import playground as pg

    class _Print:
        last = ""

        def __call__(self, ev: dict[str, Any]) -> None:
            line = _fmt_progress(ev)
            if line != self.last and ev.get("phase") in ("convert", "embed", "merge"):
                print("  " + line, file=sys.stderr, flush=True)
                self.last = line

    def build(rec: Any = None) -> dict[str, Any]:
        return pg.build_index(get_paths(), a.name, jobs=a.jobs, rebuild=a.rebuild, wipe=a.wipe,
                              force_md=a.force_md,
                              progress=rec.progress if rec else (None if a.json else _Print()),
                              stage_log=rec.path if rec else None)

    try:
        if a.job:       # a run the dashboard started: its job record and event log feed the Playground tab
            from . import playground_runs

            summary = playground_runs.run_in_child(get_paths(), a.name, a.job, "index", build)
        else:
            summary = build()
    except ValueError as exc:
        _err(f"error: {exc}")
        return EXIT_USAGE
    if a.json:
        _json(summary)
        return EXIT_FAIL if summary["errors"] else EXIT_OK
    print(f"indexed {summary['indexed']}, unchanged {summary['skipped_fresh']}, "
          f"no text {len(summary.get('no_text', []))}, errors {len(summary['errors'])} "
          f"in {summary['elapsed_s']}s")
    for e in summary["errors"][:10]:
        print(f"  ! {e['src']}: {e['message']}")
    for c in summary.get("collections", []):
        print(f"  collection {c['collection']}: {c.get('docs', '?')} doc(s), "
              f"{c.get('nodes', '?')} chunks")
    return EXIT_FAIL if summary["errors"] else EXIT_OK


def _cmd_playground_settings(a: argparse.Namespace) -> int:
    from .core import playground as pg

    try:
        res = pg.effective_settings(get_paths(), a.name)
    except ValueError as exc:
        _err(f"error: {exc}")
        return EXIT_USAGE
    if a.json:
        _json(res)
        return EXIT_OK
    print(f"playground {a.name!r}: settings in effect, by pipeline stage")
    for st in res["stages"]:
        print(f"  {st['id']:<4}{st['name']}")
        for r in st["settings"]:
            print(f"        {r['label']}: {r['value']}   [{r['source']}]")
    print("  " + res["note"])
    return EXIT_OK


def _cmd_playground_status(a: argparse.Namespace) -> int:
    from . import playground_runs

    try:
        res = playground_runs.view(get_paths(), a.name, a.run)
    except ValueError as exc:
        _err(f"error: {exc}")
        return EXIT_USAGE
    if a.json:
        _json(res)
        return EXIT_OK
    job = res.get("job")
    if not job:
        print(f"no runs yet for {a.name!r} (rag-search playground index {a.name})")
        return EXIT_OK
    prog = job.get("progress") or {}
    print(f"run {job['id']} ({job['kind']}): {job['status']}"
          + (f", {prog.get('phase')} {prog.get('done')}/{prog.get('total')}" if prog else "")
          + (f", {job['elapsed_s']}s" if job.get("elapsed_s") is not None else ""))
    if job.get("error"):
        print(f"  ! {job['error']}")
    for d in [*(i.get("timeline") for i in res["documents"]["items"]), *res["in_flight"]]:
        if not d:
            continue
        marks = "  ".join(f"{s['id']} {s['status']}" for s in d["stages"].values())
        print(f"  {d['file']}: {marks}  ({len(d['pages'])} page(s) read)")
    return EXIT_OK


def _cmd_playground_search(a: argparse.Namespace) -> int:
    from .core import playground as pg
    from .paths import parse_collections

    try:
        res = pg.search(get_paths(), a.name, a.query, top_k=a.top_k,
                        collections=parse_collections(a.collection) or None,
                        stages=a.stages, retrieval_pool_n=a.retrieval_pool,
                        rerank_pool_n=a.rerank_pool, rrf_k=a.rrf_k,
                        rerank=(False if a.no_rerank else None))
    except ValueError as exc:
        _err(f"error: {exc}")
        return EXIT_USAGE
    if a.json:
        _json(res)
        return EXIT_OK
    for hit in res.get("results", []):
        head = f" — {hit['heading']}" if hit.get("heading") else ""
        low = "  (low-confidence page: check the source)" if hit.get("confidence") == "low" else ""
        print(f"{hit['rank']}. [{hit['score']}] {hit['file']} ({hit['collection']}) "
              f"p.{hit['page']}{head}{low}")
        if a.explain:
            print("   " + _fmt_hit_breakdown(hit))
        text = " ".join(hit["text"].split())
        print("   " + (text[:300] + " …" if len(text) > 300 else text))
    if not res.get("results"):
        print(res.get("note") or "no results")
    print(_fmt_search_timing(res, explain=a.explain))
    m = res.get("models") or {}
    print(f"  models: embedding={m.get('embedding')}, reranker={m.get('reranker') or 'off'}")
    return EXIT_OK


def _cmd_playground_bench(a: argparse.Namespace) -> int:
    from .core import playground as pg

    def run(recorder: Any = None) -> dict[str, Any]:
        return pg.bench(get_paths(), a.name, queries_path=a.queries or None, k=a.k,
                        stages=a.stages, retrieval_pool_n=a.retrieval_pool,
                        rerank_pool_n=a.rerank_pool, rrf_k=a.rrf_k,
                        rerank=(False if a.no_rerank else None), label=a.label,
                        progress=recorder.progress if recorder else None)

    try:
        if a.job:
            from . import playground_runs

            rec = playground_runs.run_in_child(get_paths(), a.name, a.job, "bench", run)
        else:
            rec = run()
    except ValueError as exc:
        _err(f"error: {exc}")
        return EXIT_USAGE
    if a.json:
        _json(rec)
        return EXIT_OK
    m = rec["metrics"]
    print(f"bench run {rec['run_id']}" + (f" ({rec['label']})" if rec["label"] else "")
          + f": {rec['n_queries']} quer(y/ies)")
    print(f"  recall@{a.k}={m['recall_at_k']}  mrr={m['mrr']}  ndcg@{a.k}={m['ndcg_at_k']}  "
          f"latency mean={m['latency_ms']['mean']}ms p50={m['latency_ms']['p50']}ms "
          f"p95={m['latency_ms']['p95']}ms")
    c = rec["combo"]
    print(f"  combo: embedding={c['embedding_model']}, reranker={c['rerank_model'] or 'off'}, "
          f"stages={','.join(c['stages'])}, retrieval_pool={c['retrieval_pool']}, "
          f"rerank_pool={c['rerank_pool']}, rrf_k={c['rrf_k']}")
    for q in rec["per_query"]:
        mark = "✓" if q["hit_rank"] else "✗"
        top = q.get("top_result") or {}
        print(f"  {mark} {q['query']!r}: hit_rank={q['hit_rank']} top={top.get('file')} "
              f"p.{top.get('page')}")
    print(f"  saved to {get_paths().home}/playground/{a.name}/bench/runs/{rec['run_id']}.json")
    return EXIT_OK


def _cmd_playground_compare(a: argparse.Namespace) -> int:
    from .core import playground as pg

    runs = pg.compare(get_paths(), a.name)
    key = {"recall": lambda r: r["metrics"]["recall_at_k"], "mrr": lambda r: r["metrics"]["mrr"],
          "latency": lambda r: r["metrics"]["latency_ms"]["mean"]}[a.sort]
    runs.sort(key=key, reverse=(a.sort != "latency"))
    if a.json:
        _json(runs)
        return EXIT_OK
    if not runs:
        print(f"no bench runs yet for {a.name!r} (rag-search playground bench {a.name})")
        return EXIT_OK
    print(f"{'RUN'.ljust(20)}  {'LABEL'.ljust(14)}  {'RECALL'.rjust(6)}  {'MRR'.rjust(6)}  "
          f"{'NDCG'.rjust(6)}  {'LAT(ms)'.rjust(8)}  COMBO")
    for r in runs:
        m, c = r["metrics"], r["combo"]
        combo = (f"{c['embedding_model']}/{c['rerank_model'] or 'no-rerank'} "
                f"stages={','.join(c['stages'])} pool={c['retrieval_pool']}/{c['rerank_pool']} "
                f"rrf_k={c['rrf_k']}")
        print(f"{r['run_id'].ljust(20)}  {(r['label'] or '-').ljust(14)}  "
              f"{m['recall_at_k']:>6}  {m['mrr']:>6}  {m['ndcg_at_k']:>6}  "
              f"{m['latency_ms']['mean']:>8}  {combo}")
    return EXIT_OK


def _cmd_playground_list(a: argparse.Namespace) -> int:
    from .core import playground as pg

    rows = pg.list_experiments(get_paths())
    if a.json:
        _json(rows)
        return EXIT_OK
    if not rows:
        print("no playground experiments yet (rag-search playground create NAME)")
        return EXIT_OK
    for r in rows:
        cfg = r["config"]
        print(f"{r['name']}: {len(r['sources'])} source folder(s), collections "
              f"{', '.join(r['indexed_collections']) or '(not indexed yet)'}, "
              f"{r['bench_runs']} bench run(s)")
        print(f"  model: {cfg['embedding_model']} + {cfg['rerank_model']}")
    return EXIT_OK


def _cmd_playground_rm(a: argparse.Namespace) -> int:
    from .core import playground as pg

    if not a.yes:
        _err(f"this permanently deletes the playground experiment {a.name!r} and all its "
             "bench runs. Re-run with --yes to confirm.")
        return EXIT_USAGE
    try:
        pg.remove_experiment(get_paths(), a.name)
    except ValueError as exc:
        _err(f"error: {exc}")
        return EXIT_USAGE
    if a.json:
        _json({"removed": a.name})
    else:
        print(f"removed playground experiment {a.name!r}")
    return EXIT_OK


def _cmd_ui(a: argparse.Namespace) -> int:
    from .ui import server

    port = a.port
    if port is None:
        env = os.environ.get("RAG_SEARCH_UI_PORT", "")
        port = int(env) if env.isdigit() else server.DEFAULT_PORT
    return server.run(get_paths(), port=port, open_browser=not a.no_browser,
                      read_only=a.read_only, detach=a.detach, url_only=a.url, stop=a.stop)


def _compare_variants(cur: dict[str, Any]) -> list[tuple[str, dict[str, str]]]:
    """The settings worth timing against the current ones (only those that differ)."""
    import importlib.util
    import platform

    variants: list[tuple[str, dict[str, str]]] = [("current settings", {})]
    smart: dict[str, str] = {"RAG_SEARCH_OCR": "smart"}
    if cur["ocr"] != "smart":
        variants.append(("OCR smart", dict(smart)))
    if platform.system() == "Darwin" and importlib.util.find_spec("ocrmac") and cur["engine"] != "ocrmac":
        variants.append(("OCR smart + Apple Vision", {**smart, "RAG_SEARCH_OCR_ENGINE": "ocrmac"}))
    if cur["table"] != "fast":
        variants.append(("OCR smart + fast tables", {**smart, "RAG_SEARCH_TABLE_MODE": "fast"}))
    return variants


def _compare_conversions(a: argparse.Namespace, src: Path) -> int:
    """Time the current conversion settings against faster ones on one file, and say how much the
    text differs, so the choice is made on your own documents.  Each variant runs in a fresh
    process (its time includes loading the models)."""
    import difflib
    import subprocess
    import time

    from .core.docling_convert import convert_settings

    out_dir = Path(a.output).expanduser() if a.output else Path.cwd() / f"{src.stem}-compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    base_lines: list[str] = []
    for i, (label, env) in enumerate(_compare_variants(convert_settings())):
        out = out_dir / f"{src.stem}.{i}-{label.replace(' + ', '-').replace(' ', '-').lower()}.md"
        print(f"[{i + 1}] {label} ...", flush=True, file=sys.stderr)
        t0 = time.perf_counter()
        proc = subprocess.run([sys.executable, "-m", "rag_search.core.docling_convert", str(src), str(out)],
                              capture_output=True, text=True, env={**os.environ, **env})
        secs = time.perf_counter() - t0
        row: dict[str, Any] = {"variant": label, "env": env, "seconds": round(secs, 1),
                               "file": str(out), "ok": proc.returncode == 0}
        if proc.returncode != 0:
            row["error"] = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ["failed"]
        else:
            text = out.read_text(encoding="utf-8")
            lines = text.splitlines()
            if not rows:
                base_lines = lines
            row.update(chars=len(text), same=round(100 * difflib.SequenceMatcher(
                None, base_lines, lines, autojunk=False).ratio(), 1))
        rows.append(row)
    base = next((r["seconds"] for r in rows if r["ok"]), None)
    if a.json:
        _json({"file": str(src), "variants": rows})
    else:
        print(f"\n{'variant':<28}{'time':>9}{'speed-up':>10}{'characters':>12}{'text same as current':>24}")
        for r in rows:
            if not r["ok"]:
                print(f"{r['variant']:<28}  failed: {' '.join(r['error'])}")
                continue
            sp = f"{base / r['seconds']:.1f}x" if base and r["seconds"] else "-"
            print(f"{r['variant']:<28}{_dur(r['seconds']):>9}{sp:>10}{r['chars']:>12}{r['same']:>23}%")
        print(f"\nOutputs are in {out_dir}. 'text same' compares line by line with the first row; "
              "100% means identical. Look at the differences before switching, for example:\n"
              f"  diff {rows[0]['file']} {rows[-1]['file']} | head -50\n"
              "then set the winner with RAG_SEARCH_OCR / RAG_SEARCH_OCR_ENGINE / "
              "RAG_SEARCH_TABLE_MODE and run `rag-search index new` (changed settings re-convert).")
    return EXIT_OK if any(r["ok"] for r in rows) else EXIT_FAIL


def _cmd_convert(a: argparse.Namespace) -> int:
    from .core.docling_convert import convert_file

    if a.compare:
        return _compare_conversions(a, Path(a.file).expanduser())

    # one-off overrides for A/B comparisons; they only apply to this command
    for var, val in (("RAG_SEARCH_OCR", a.mode), ("RAG_SEARCH_PDF_BACKEND", a.backend),
                     ("RAG_SEARCH_OCR_ENGINE", a.engine), ("RAG_SEARCH_TABLE_MODE", a.table)):
        if val:
            os.environ[var] = val
    os.environ.setdefault("RAG_SEARCH_THREADS", str(min(os.cpu_count() or 4, 8)))
    src = Path(a.file).expanduser()
    out = Path(a.output).expanduser() if a.output else Path.cwd() / (src.stem + ".md")
    info = convert_file(src, out, ocr=True if a.ocr else None)
    used = f", OCR {info['ocr']}" if info.get("ocr") else ""
    print(f"wrote {out} ({info['pages']} page(s), {info['seconds']}s{used})")
    return EXIT_OK


def _cmd_convert_legacy(a: argparse.Namespace) -> int:
    from .core.legacy_convert import LEGACY_FORMATS, convert_tree, find_soffice

    paths = get_paths()
    root = Path(a.path).expanduser().resolve()
    if not root.is_dir():
        _err(f"not a directory: {root}")
        return EXIT_USAGE

    raw_exts = [e.strip().lower().lstrip(".") for e in a.ext.split(",") if e.strip()]
    exts = {"." + e for e in raw_exts}
    unknown = exts - set(LEGACY_FORMATS)
    if unknown:
        _err(f"unsupported extension(s): {', '.join(sorted(unknown))} "
             f"(this command only handles {', '.join(sorted(LEGACY_FORMATS))})")
        return EXIT_USAGE

    soffice = find_soffice()
    if not soffice and not a.dry_run:
        _err("LibreOffice (soffice) was not found.")
        _err("Install it with:  brew install --cask libreoffice")
        _err("then confirm with:  soffice --version")
        return EXIT_UNAVAILABLE

    log_path = paths.run / "convert-legacy.jsonl"
    summary = convert_tree(root, exts=exts, soffice=soffice, dry_run=a.dry_run,
                           keep_originals=not a.delete_originals, force=a.force,
                           timeout=a.timeout, log_path=log_path)

    if a.json:
        _json({
            "root": str(root), "dry_run": a.dry_run,
            "converted": [{"src": str(r.src), "dest": str(r.dest), "deleted": r.deleted}
                          for r in summary.converted],
            "skipped": [str(r.src) for r in summary.skipped],
            "failed": [{"src": str(r.src), "error": r.error} for r in summary.failed],
        })
        return EXIT_FAIL if summary.failed else EXIT_OK

    total = len(summary.converted) + len(summary.skipped) + len(summary.failed)
    if not total:
        print(f"no {'/'.join(sorted(exts))} files found under {root}")
        return EXIT_OK

    if a.dry_run:
        print(f"would convert {len(summary.converted)} file(s) under {root} "
              f"({len(summary.skipped)} already done)")
    else:
        print(f"converting {len(summary.converted)} file(s) under {root} "
              f"({len(summary.skipped)} already done, {len(summary.failed)} failed)")
    for r in summary.skipped:
        print(f"  [skip]  {r.src.relative_to(root)}  "
              f"({r.dest.name} already exists; use --force to redo)")
    for r in summary.converted:
        if a.dry_run:
            print(f"  [dry-run] {r.src.relative_to(root)} -> {r.dest.name}")
        else:
            kept = "original kept" if not r.deleted else "original deleted"
            print(f"  [ok]    {r.src.relative_to(root)} -> {r.dest.name}  ({kept})")
    for r in summary.failed:
        print(f"  [failed] {r.src.relative_to(root)}  {r.error}")

    if not a.dry_run and total:
        print(f"\nlog: {log_path}")
    return EXIT_FAIL if summary.failed else EXIT_OK


def _cmd_setup(a: argparse.Namespace) -> int:
    from . import models
    from .core import diagnostics

    paths = get_paths()
    ensure_dirs(paths)
    print(f"Data folder: {paths.home}")
    if a.models:
        try:
            emb, rer = models.resolve_preset(a.models)
            models.set_selection(paths, models.EMBEDDING, emb)
            models.set_selection(paths, models.RERANKER, rer)
        except (models.ModelError, ValueError, OSError) as exc:
            _err(f"error: {exc}")
            return EXIT_USAGE
        print(f"Models: preset {a.models} ({emb} + {rer}), saved in {paths.config_file}")
        rx = models.reindex_estimate(paths, emb)
        if rx["documents"]:
            print(f"note: {rx['documents']} indexed document(s) use another embedding model; search "
                  "keeps using them until you run `rag-search index new` (it re-embeds them).")
    if not a.skip_models:
        for m in diagnostics.download_models(skip_docling=a.skip_docling):
            print("  " + m)
    print("\nSetup complete. Next: rag-search register   (then restart Claude Desktop)")
    return EXIT_OK


# ── models ───────────────────────────────────────────────────────────────────

def _gb(x: Any) -> str:
    return f"{float(x):.1f} GB"


def _model_status(row: dict[str, Any]) -> str:
    """Disk state, and for the chosen model whether it can actually be used."""
    disk = "partly downloaded" if row.get("partial") else ("downloaded" if row.get("cached") else "not downloaded")
    st = row.get("state", "")
    if row.get("active") and st.startswith("selected_") and st not in ("selected_download", "selected_partial"):
        return {"selected_off": "chosen, reader off", "selected_platform": "chosen, not a Mac",
                "selected_runtime": "chosen, no runtime"}.get(st, disk)
    return disk


def _print_models(st: dict[str, Any]) -> None:
    m = st["machine"]
    limit = st["memory_limit_gb"]
    print(f"This machine: {m['ram_gb']} GB memory, compute device {m['device']}. Memory the models "
          f"may use: {st['budget_gb']} GB "
          + ("(your limit)" if limit else "(60% of the memory; change it with `rag-search models limit GB`)"))
    titles = {"embedding": "EMBEDDING MODEL   (text -> vectors; a new one means every document is "
                           "embedded again)",
              "reranker": "RERANKER   (orders the best hits; switching needs no re-indexing)"}
    for kind in ("embedding", "reranker"):
        sec = st[kind]
        src = {"environment": " - set by an environment variable, which wins over config.json",
               "config": "", "default": " (default)"}[sec["source"]]
        print(f"\n{titles[kind]}\n  in use: {sec['active']}{src}")
        if kind == "embedding" and st["serving"] and st["serving"] != sec["active"]:
            print(f"  the published index was built with {st['serving']} and stays in use until "
                  "the re-embedding is finished")
        print(f"   {'model':<40} {'params':>6} {'licence':<11} {'on disk':<17}memory with the other model")
        for r in sec["models"]:
            flag = ("*" if r.get("state") == "in_use" else "!") if r["active"] else " "
            size = f"{r['params_m']}M" if r["params_m"] else "?"
            fit = {"ok": "", "tight": "  memory: tight", "too_large": "  memory: TOO LARGE",
                   "unknown": ""}.get(r["fit"], "")
            miss = f"  needs: {'; '.join(r['missing'])}" if r["missing"] else ""
            print(f" {flag} {r['id']:<40} {size:>6} {r['license']:<11} {_model_status(r):<17}"
                  f"~{r['estimate_gb']} GB total{fit}{miss}")
            if r["note"]:
                print(f"       {r['note']}")
    _print_readers(st.get("vlm"))
    print("\nPresets: " + "; ".join(f"{n} = {p['embedding'].split('/')[-1]} + "
                                    f"{p['reranker'].split('/')[-1]}"
                                    for n, p in st["presets"].items()))
    print("Switch:  rag-search models set embedding|reranker MODEL_ID     "
          "(any Hugging Face id works; it is tested first)\n"
          "         rag-search models use PRESET      rag-search models download [MODEL_ID ...]")
    print("Memory figures are estimates for the search daemon (weights + index + about "
          "1.5 GB); models other than the defaults have not been benchmarked on your documents.")


def _print_readers(v: dict[str, Any] | None) -> None:
    """The document reader section of `rag-search models`."""
    if not v:
        return
    free = f", {v['free_gb']:.1f} GB memory free now" if v.get("free_gb") is not None else ""
    print(f"\nDOCUMENT READER   (a vision model reads scanned pages, pictures and image files; "
          f"docling OCR is the fallback)\n  mode: {'off (RAG_SEARCH_VLM)' if v['mode'] == 'off' else 'auto'}{free}")
    print(f"  {v['reading']['text']}")
    for c in v.get("checks", []):
        mark = "ok     " if c["ok"] else ("MISSING" if c["required"] else "missing")
        print(f"   {mark} {c['label']:<28} {c['detail']}")
    if not v["runtime"]["ready"] and v["apple_silicon"]:
        print("   -> install the runtime: rag-search models runtime install   "
              "(optional extra rag-search[mac-vlm], Apple Silicon only)")
    print("   (* in use, ! chosen but not usable yet)")
    for kind, title in (("reader", "reads pages"), ("repair", "re-reads suspect table cells and pages")):
        sec = v[kind]
        src = {"environment": " - set by an environment variable", "config": "", "default": " (default)"}[sec["source"]]
        print(f"  {kind} ({title}): {sec['active']}{src}")
        for r in sec["models"]:
            flag = ("*" if r.get("state") == "in_use" else "!") if r["active"] else " "
            tight = "  memory: tight now" if r["fit"] == "tight" else ""
            print(f"   {flag} {r['id']:<46} {_gb(r['mem_gb']):>8} {r['license']:<11} {_model_status(r):<17}"
                  f"needs ~{r['need_gb']} GB{tight}")
            if r["note"]:
                print(f"       {r['note']}")
    print("  Choose:    rag-search models reader|repair MODEL_ID     "
          "Download: rag-search models download --reader   (nothing is downloaded by itself)")


def _download_bar():
    """Progress printer for a foreground download (a line that updates on a terminal)."""
    state = {"last": -1}
    tty = sys.stdout.isatty()

    def show(p: dict[str, Any]) -> None:
        done, total = p.get("done") or 0, p.get("total")
        if not total:
            if tty:
                print(f"\r  {p.get('model', '')}: {_bytes(done)} downloaded ", end="", flush=True)
            return
        pct = min(100, int(100 * done / total))
        if tty:
            print(f"\r  {p.get('model', '')}: {_bytes(done)} / {_bytes(total)} ({pct}%)   ",
                  end="", flush=True)
        elif pct // 10 != state["last"]:
            state["last"] = pct // 10
            print(f"  {p.get('model', '')}: {pct}%", flush=True)
        if tty and pct >= 100:
            print()
    return show


def _run_task_cli(a: argparse.Namespace, req: dict[str, Any]) -> int:
    from . import model_tasks as mt

    try:
        rec = mt.run(get_paths(), req, echo=(lambda m: print(("\n" if sys.stdout.isatty() else "")
                                                             + m, flush=True)) if not a.json
                     else None, progress_echo=None if a.json else _download_bar())
    except mt.TaskBusy as exc:
        _err(f"error: {exc}")
        return EXIT_FAIL
    except mt.TaskError as exc:
        _err(f"error: {exc}")
        return EXIT_FAIL
    if a.json:
        _json(rec)
    elif rec["status"] != "succeeded":
        _err(f"{rec['status']}: {rec.get('error', '')}".rstrip(": "))
    return EXIT_OK if rec["status"] == "succeeded" else EXIT_FAIL


def _describe_plan(pl: dict[str, Any]) -> list[str]:
    kind = "embedding model" if pl["kind"] == "embedding" else "reranker"
    lines = [f"Switch the {kind} from {pl['current']} to {pl['model']}"
             + (f" ({pl['label']}, {pl['license']})" if not pl["custom"] else "")]
    lines.append("  " + ("already downloaded" if pl["cached"] else
                         f"to download: about {pl['weights_gb']:.1f} GB of weights"
                         if pl["weights_gb"] else "to download: size unknown until it starts"))
    if pl["fit"] != "unknown":
        lines.append(f"  memory: about {pl['estimate_gb']} GB for the search daemon with the other "
                     f"model (budget {pl['budget_gb']} GB)")
    rx = pl.get("reindex")
    if pl["kind"] == "embedding":
        if rx and rx["documents"]:
            eta = (f", roughly {_dur(rx['estimated_s'])} of embedding" if rx["estimated_s"] else "")
            lines.append(f"  {rx['documents']} of {rx['total_documents']} document(s) will be "
                         f"embedded again{eta}. Search keeps using the current index until all of "
                         "it is done; conversion is not repeated.")
        else:
            lines.append("  no documents are indexed yet, so nothing needs re-embedding")
    else:
        lines.append("  no re-indexing; the running search daemon loads it and swaps it in")
    lines.extend(f"  note: {w}" for w in pl["warnings"])
    lines.extend(f"  BLOCKED: {b}" for b in pl["blocking"])
    return lines


def _confirm(text: str, a: argparse.Namespace) -> bool:
    if a.yes:
        return True
    if not sys.stdin.isatty():
        _err(f"{text}\nerror: not a terminal; repeat with --yes to go ahead")
        return False
    try:
        return input(f"{text} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def _switch_many(a: argparse.Namespace, wanted: list[tuple[str, str]]) -> int:
    """Plan, confirm once, then switch each (reranker first: it is the cheap one)."""
    from . import model_tasks as mt
    from . import models

    paths = get_paths()
    plans = []
    for kind, mid in sorted(wanted, key=lambda w: w[0] != "reranker"):
        try:
            pl = mt.plan_switch(paths, kind, mid, force=a.force)
        except models.ModelError as exc:
            _err(f"error: {exc}")
            return EXIT_USAGE
        if not pl["same"]:
            plans.append(pl)
        else:
            print(f"{mid} is already the {kind} model.")
    if not plans:
        return EXIT_OK
    lines = [line for pl in plans for line in _describe_plan(pl)]
    blocked = [pl for pl in plans if pl["blocking"] and not a.force]
    if a.json and not a.yes:
        _json({"plans": plans})
        return EXIT_OK
    if not a.json:
        print("\n".join(lines))
    if blocked:
        _err("error: " + "; ".join(b for pl in blocked for b in pl["blocking"])
             + " (use --force to try anyway)")
        return EXIT_FAIL
    if any(pl["kind"] == "embedding" for pl in plans) and not a.no_reindex:
        question = "Go ahead (download, test, switch, re-embed)?"
    else:
        question = "Go ahead (download, test, switch)?"
    if not _confirm(question, a):
        print("Nothing changed.")
        return EXIT_FAIL
    rc = EXIT_OK
    for pl in plans:
        rc = _run_task_cli(a, {"op": "switch", "kind": pl["kind"], "model": pl["model"],
                               "force": a.force, "no_verify": a.no_verify,
                               "reindex": not a.no_reindex})
        if rc != EXIT_OK:
            break
    if rc == EXIT_OK and not a.json and any(pl["kind"] == "embedding" for pl in plans):
        print("\nFollow the re-embedding with:  rag-search index status --follow")
    return rc


def _cmd_models(a: argparse.Namespace) -> int:
    from . import model_tasks as mt
    from . import models

    paths = get_paths()
    act = a.models_cmd or "list"
    try:
        if act == "list":
            st = models.state(paths)
            if a.json:
                _json({**st, "task": mt.read_task(paths)})
            else:
                _print_models(st)
            return EXIT_OK
        if act == "set":
            return _switch_many(a, [(a.kind, a.model)])
        if act == "use":
            emb, rer = models.resolve_preset(a.preset)
            return _switch_many(a, [("embedding", emb), ("reranker", rer)])
        if act in ("reader", "repair"):
            sel, source = models.vlm_selection(act)
            if a.model:
                models.set_vlm_selection(paths, act, a.model)
                sel, source = models.vlm_selection(act)
                if a.json:
                    _json({"kind": act, "model": sel, "source": source})
                else:
                    print(f"{act} model set to {sel}"
                          + ("" if source == "config" else f" (but {source} wins over config.json)")
                          + ("" if models.cache_state(sel)["cached"] else
                             f"\nIt is not downloaded yet: rag-search models download {sel}"))
            elif a.json:
                _json({"kind": act, "model": sel, "source": source})
            else:
                print(f"{act} model: {sel} ({source})")
            return EXIT_OK
        if act == "download":
            ids = list(a.models_ids)
            if getattr(a, "reader", False):
                ids.append(models.vlm_selection(models.READER)[0])
            if not ids:
                ids = [models.selection(k)[0] for k in models.KINDS]
            for mid in ids:
                if not models.find(mid) and not models._ID_RE.match(mid):
                    raise models.ModelError(f"{mid!r} is not a Hugging Face model id (ORG/NAME)")
            return _run_task_cli(a, {"op": "download", "models": ids})
        if act == "runtime":
            if a.what == "install":
                return _run_task_cli(a, {"op": "runtime", "model": "mlx-vlm"})
            rt = models.runtime_state()
            if a.json:
                _json(rt)
                return EXIT_OK
            print(f"Document reader runtime (optional extra rag-search[{rt['extra']}], Apple Silicon only; "
                  f"installed with {rt['installer']} into {rt['python']})")
            for x in rt["packages"]:
                print(f"  {'installed' if x['installed'] else 'missing  '}  {x['package']:<12} {x['what']}")
            if not rt["apple_silicon"]:
                print("This computer is not an Apple Silicon Mac: the reader cannot run here (docling OCR is used).")
            elif not rt["complete"]:
                print("Install it with: rag-search models runtime install")
            return EXIT_OK
        if act == "verify":
            kinds = [a.kind] if a.kind else list(models.KINDS)
            targets = [[k, models.selection(models.check_kind(k))[0]] for k in kinds]
            return _run_task_cli(a, {"op": "verify", "targets": targets})
        if act == "limit":
            if a.value is None:
                lim = models.memory_limit_gb(paths)
                print(f"memory limit for the models: {_gb(lim) if lim else 'none (60% of the memory)'}")
                return EXIT_OK
            gb = 0.0 if a.value.lower() in ("off", "none", "0") else float(a.value)
            models.set_memory_limit(paths, gb)
            print((f"memory limit set to {_gb(gb)}" if gb else "memory limit removed")
                  + ". It is used to judge which models fit; it does not cap running processes.")
            return EXIT_OK
        if act == "status":
            rec = mt.read_task(paths)
            if a.json:
                _json(rec or {})
            elif not rec:
                print("no model download or switch has been run yet")
            else:
                print(f"{rec['op']} {rec.get('model', '')}: {rec['status']} ({rec.get('phase', '')})")
                for e in rec.get("log", [])[-8:]:
                    print("  " + e["msg"])
                if rec.get("error"):
                    print("  error: " + rec["error"])
            return EXIT_OK
        if act == "cancel":
            print("cancelled" if mt.cancel(paths) else "no model task is running")
            return EXIT_OK
    except (models.ModelError, ValueError, OSError) as exc:
        _err(f"error: {exc}")
        return EXIT_USAGE
    return EXIT_USAGE


def _cmd_doctor(a: argparse.Namespace) -> int:
    from .core import diagnostics

    rows = diagnostics.run_checks(get_paths())
    if a.json:
        _json([{"status": st, "check": label, "detail": detail} for st, label, detail in rows])
        return 1 if any(st == diagnostics.FAIL for st, _, _ in rows) else EXIT_OK
    rc = diagnostics.print_checks(rows)
    if a.roundtrip:
        rc = max(rc, diagnostics.roundtrip())
    return rc


def _host_selected(a: argparse.Namespace, h) -> tuple[bool, str]:
    """(--<host> or --<host>-file given?, the file path or "")."""
    path = getattr(a, f"host_{h.NAME}_file", "") or ""
    return bool(getattr(a, f"host_{h.NAME}", False) or path), path


def _cmd_register(a: argparse.Namespace) -> int:
    from . import register

    home = a.home or os.environ.get("RAG_SEARCH_HOME") or None
    hosts = [(h, *_host_selected(a, h)) for h in register.extra_hosts()]
    explicit = a.desktop or a.code or any(sel for _, sel, _ in hosts)
    if home:
        ensure_dirs(get_paths(home))
    if a.desktop or not explicit:
        print(register.register_desktop(home, a.tool_prefix))
    if a.code or not explicit:
        print(register.register_code(home, a.tool_prefix))
    for h, selected, path in hosts:
        # an optional host is registered when asked for, or (if its module says so) together
        # with Claude when it is installed, unless --no-<name>
        auto = (not explicit and getattr(h, "AUTO_REGISTER", False) and h.installed()
                and not getattr(a, f"no_host_{h.NAME}", False))
        if selected or auto:
            print(h.register(home, path or None, a.tool_prefix))
    return EXIT_OK


def _cmd_unregister(a: argparse.Namespace) -> int:
    from . import register

    hosts = [(h, *_host_selected(a, h)) for h in register.extra_hosts()]
    explicit = a.desktop or a.code or any(sel for _, sel, _ in hosts)
    if a.desktop or not explicit:
        print(register.unregister_desktop())
    if a.code or not explicit:
        print(register.unregister_code())
    registered = {h.NAME for h in register.registered_hosts()}
    for h, selected, path in hosts:
        if selected or (not explicit and h.NAME in registered):
            print(h.unregister(path or None))
    return EXIT_OK


def _cmd_mcp_config(a: argparse.Namespace) -> int:
    from . import register

    known = register.known_profiles()
    profile = a.profile or "claude"
    if profile not in known:
        _err(f"unknown profile {profile!r}; this build knows: {', '.join(known)}")
        return EXIT_USAGE
    extra = next((getattr(h, "EXTRA_ENTRY", None) for h in register.extra_hosts()
                  if h.NAME == profile), None)
    print(register.mcp_config_snippet(a.home or os.environ.get("RAG_SEARCH_HOME") or None,
                                      profile, a.tool_prefix, extra))
    return EXIT_OK


def _cmd_serve(_a: argparse.Namespace) -> int:
    try:
        from .mcp.server import main as serve_main
    except ImportError as exc:
        _err(f"the MCP adapter is not installed ({exc}); install rag-search[mcp]")
        return EXIT_FAIL
    serve_main([])
    return EXIT_OK


# ── parser ───────────────────────────────────────────────────────────────────

def _common() -> argparse.ArgumentParser:
    c = argparse.ArgumentParser(add_help=False)
    c.add_argument("--home", default=argparse.SUPPRESS, help="data folder (= $RAG_SEARCH_HOME)")
    c.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                   help="machine-readable output")
    c.add_argument("--client", default=argparse.SUPPRESS,
                   help="act as this client, to see what it would get (default: $RAG_SEARCH_CLIENT or "
                        "cli, the administrator, who can use everything)")
    return c


def build_parser() -> argparse.ArgumentParser:
    # each parser gets its own copy: parents share Action objects, and set_defaults() on
    # the top-level parser would otherwise overwrite the SUPPRESS defaults of the others
    ap = argparse.ArgumentParser(
        prog="rag-search", parents=[_common()],
        description="Local document RAG: index documents once, search from the shell, "
                    "Claude Desktop or Claude Code.")
    ap.set_defaults(home="", json=False, client=None)
    ap.add_argument("--version", action="version", version=f"rag-search {__version__}")
    sub = ap.add_subparsers(dest="command", metavar="COMMAND")

    def add(name: str, help_: str, fn, **kw) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_, parents=[_common()], **kw)
        p.set_defaults(fn=fn)
        return p

    p = add("search", "hybrid search (BM25 + vectors + reranker) via the search daemon", _cmd_search)
    p.add_argument("query")
    p.add_argument("-c", "--collection", default="", help="comma-separated collection names")
    p.add_argument("-k", "--top-k", type=int, default=None,
                   help=f"results to return (default: the production default in config.json, "
                        f"else {DEFAULT_TOP_K})")
    p.add_argument("--wait", type=float, default=45.0,
                   help="seconds to wait for a starting daemon (default 45)")
    p.add_argument("--stages", default=None,
                   help="comma-separated subset of bm25,dense,rerank to run (default: all three) "
                        "-- troubleshooting: isolate one retriever or drop the reranker")
    p.add_argument("--retrieval-pool", type=int, default=None, dest="retrieval_pool",
                   help="override candidates each retriever contributes (default: a formula on -k)")
    p.add_argument("--rerank-pool", type=int, default=None, dest="rerank_pool",
                   help="override candidates handed to the reranker (default: a formula on -k)")
    p.add_argument("--rrf-k", type=int, default=None, dest="rrf_k",
                   help="override the reciprocal-rank-fusion constant (default: 60)")
    p.add_argument("--explain", action="store_true",
                   help="show each hit's BM25/dense/RRF/rerank scores and pipeline diagnostics")

    p = add("grep", "regex search over the converted Markdown", _cmd_grep)
    p.add_argument("pattern")
    p.add_argument("-c", "--collection", default="")
    p.add_argument("--context", type=int, default=2)
    p.add_argument("--max", type=int, default=20)

    add("list", "list published collections and documents", _cmd_list)

    p = add("trace", "how a document was converted: branch, outcome and checks per page", _cmd_trace)
    p.add_argument("target", help="COLLECTION/DOCUMENT (the path `list` shows, without extension)")
    p.add_argument("--page", type=int, default=0, help="the full record of this page")
    p.add_argument("--md", action="store_true",
                   help="print the converted Markdown (of the whole document, or of --page) instead of the record")

    bp = add("bench", "measure how well pages are read: gold sets and benchmark runs", None)
    bsub = bp.add_subparsers(dest="bench_cmd", metavar="ACTION")

    def badd(name: str, help_: str, fn) -> argparse.ArgumentParser:
        q = bsub.add_parser(name, help=help_, parents=[_common()])
        q.set_defaults(fn=fn)
        return q

    gp = badd("gold", "gold sets: pages with checked text to measure against", None)
    gsub = gp.add_subparsers(dest="gold_cmd", metavar="ACTION")
    q = gsub.add_parser("init", help="create or extend a draft gold set from the conversion traces", parents=[_common()])
    q.set_defaults(fn=_cmd_bench_gold)
    q.add_argument("set")
    q.add_argument("--collection", default="", help="only pages of this collection")
    q.add_argument("--per-class", type=int, default=5, dest="per_class", help="pages per failure class (default 5)")
    q.add_argument("--classes", default="", help="comma-separated classes (default: all found)")
    q.add_argument("--seed", type=int, default=0)
    q = gsub.add_parser("list", help="the gold sets and the engines available", parents=[_common()])
    q.set_defaults(fn=_cmd_bench_gold)
    q = gsub.add_parser("show", help="the pages of one gold set", parents=[_common()])
    q.set_defaults(fn=_cmd_bench_gold)
    q.add_argument("set")
    q = badd("run", "read the verified gold pages with an engine and score the result", _cmd_bench_run)
    q.add_argument("set")
    q.add_argument("--engine", default="current", help="current (docling as configured) or module:attr")
    q.add_argument("--name", default="", help="a label for this run")
    q.add_argument("--classes", default="")
    q.add_argument("--limit", type=int, default=0, help="only the first N pages")
    q.add_argument("--drafts", action="store_true", help="include pages that are not verified yet")
    q = badd("list", "stored runs", _cmd_bench_list)
    q.add_argument("set", nargs="?", default="")
    q = badd("show", "one run", _cmd_bench_show)
    q.add_argument("set")
    q.add_argument("run")
    q = badd("compare", "run B against run A of the same set", _cmd_bench_compare)
    q.add_argument("set")
    q.add_argument("run_a")
    q.add_argument("run_b")

    idx = add("index", "indexing (runs in the indexer daemon)", None)
    isub = idx.add_subparsers(dest="index_cmd", metavar="ACTION")

    def iadd(name: str, help_: str, fn) -> argparse.ArgumentParser:
        q = isub.add_parser(name, help=help_, parents=[_common()])
        q.set_defaults(fn=fn)
        return q

    for mode, text in (("new", "index new/changed documents (unchanged ones are skipped)"),
                       ("all", "wipe and rebuild every document in scope")):
        q = iadd(mode, text, _cmd_index_start)
        q.set_defaults(mode=mode)
        q.add_argument("path", nargs="?", default="",
                       help="a registered location's name (optionally /subfolder), or a file or "
                            "folder inside one (default: everything)")
        q.add_argument("--restart", action="store_true",
                       help="kill a running indexing run and start over")
        q.add_argument("-f", "--follow", action="store_true", help="stream progress until done")
        if mode == "new":
            q.add_argument("--rebuild", action="store_true", help="re-embed even if unchanged")
            q.add_argument("--force-md", action="store_true", help="re-convert to Markdown")
        else:
            q.set_defaults(rebuild=False, force_md=False)
    q = iadd("status", "status of the current/last run", _cmd_index_status)
    q.add_argument("job_id", nargs="?", default="")
    q.add_argument("-f", "--follow", action="store_true", help="stream progress until done")
    q.add_argument("--history", type=int, default=0, help="also list the last N runs")
    q.add_argument("--docs", type=int, default=20,
                   help="show the timings of the last N documents of the run (default 20, 0 = none)")
    q.add_argument("--doc-status", default="", dest="doc_status",
                   help="only documents with this status (e.g. error, indexed, skipped, no_text)")
    q.add_argument("--doc-collection", default="", dest="doc_collection",
                   help="only documents in this collection")
    q.add_argument("--doc-find", default="", dest="doc_q", help="only documents whose filename contains this text")
    q.add_argument("--doc-branch", default="", dest="doc_branch",
                   help="only documents with a page read by this branch (digital, raster, image, "
                        "office, embedded, copy, ...)")
    q.add_argument("--doc-outcome", default="", dest="doc_outcome",
                   help="only documents with a page with this outcome (pass, repaired, low, "
                        "no_text, error)")
    iadd("estimate", "dry run: profile the sources and estimate pages per branch and time",
         _cmd_index_estimate).add_argument("path", nargs="?", default="",
                                           help="file or folder or location (default: everything)")
    iadd("cache", "the page cache: pages already read, kept so no reading is repeated",
         _cmd_index_cache).add_argument("--clear", action="store_true",
                                        help="delete every cached page (they are read again next time)")
    iadd("cancel", "stop the running indexing run", _cmd_index_cancel)
    iadd("publish", "publish the workspace now (normally automatic after a run)",
         _cmd_index_publish)
    q = iadd("foreground", "index in this process, without the daemon, then publish",
             _cmd_index_foreground)
    q.add_argument("path", nargs="?", default="")
    q.add_argument("--mode", choices=["new", "all"], default="new")
    q.add_argument("--rebuild", action="store_true")
    q.add_argument("--force-md", action="store_true")

    acc = add("access", "which clients (claude, ...) may use which collections", _cmd_access,
              description="Collections are open to every client by default. Restrict a "
              "collection to specific clients, add clients to it, or list who can use what. Only this command "
              "manages access: MCP hosts just see the collections they are authorised for. "
              "A client is the name a host passes as `rag-search-mcp --profile NAME`.")
    asub = acc.add_subparsers(dest="access_cmd", metavar="ACTION")

    def aadd(name: str, help_: str) -> argparse.ArgumentParser:
        q = asub.add_parser(name, help=help_, parents=[_common()])
        q.set_defaults(fn=_cmd_access)
        return q

    aadd("list", "every collection and who may use it (default)")
    q = aadd("restrict", "only the given clients may use a collection (none = nobody, all = everyone)")
    q.add_argument("collection")
    q.add_argument("clients", nargs="*", metavar="CLIENT")
    q = aadd("grant", "add clients to a restricted collection (all = open it to everyone)")
    q.add_argument("collection")
    q.add_argument("clients", nargs="+", metavar="CLIENT")

    p = add("describe", "view or set a collection's short description (shown by rag_list_collections)",
           _cmd_describe,
           description="Normally set by the calling LLM itself via the rag_describe_collection "
           "MCP tool, right after it explores a collection. This command is a human override/"
           "audit path: run with no arguments to list every description, with just a "
           "collection to see its current one, or with text to set it.")
    p.add_argument("collection", nargs="?", default="",
                   help="collection name (default: list every description)")
    p.add_argument("text", nargs="?", default=None, help="new description to set")
    p.add_argument("--clear", action="store_true", help="clear the description")

    loc = add("location", "register the folders whose documents are indexed", _cmd_location,
              description="A collection is a folder registered here under a name (a notes vault, a "
              "synced drive, a share, a project folder): its whole tree becomes one collection.  "
              "Sources are only ever read.  A location that "
              "cannot be read during an indexing run (unmounted, offline) is skipped and keeps "
              "its last index; documents removed from a readable location are removed from the "
              "index.")
    lsub = loc.add_subparsers(dest="location_cmd", metavar="ACTION")

    def ladd(name: str, help_: str) -> argparse.ArgumentParser:
        q = lsub.add_parser(name, help=help_, parents=[_common()])
        q.set_defaults(fn=_cmd_location)
        return q

    ladd("list", "every registered location, and whether it is readable")
    q = ladd("add", "make FOLDER (and everything below it) the collection NAME")
    q.add_argument("name")
    q.add_argument("folder")
    q = ladd("remove", "unregister a location and delete its index (the folder is not touched)")
    q.add_argument("name")
    q.add_argument("-y", "--yes", action="store_true", help="do not ask for confirmation")

    col = add("collection", "export, import or delete a collection's index", None,
              description="export writes a collection's index, converted Markdown and build "
              "metadata to one .rag.tgz file (no source documents, no access rules); import "
              "unpacks such a file as a searchable collection -- only when it was embedded with "
              "this installation's embedding model; delete removes a collection's converted "
              "Markdown and index from the workspace (never its documents; access rule and "
              "description are kept).  Not available to MCP hosts.")
    csub = col.add_subparsers(dest="collection_cmd", metavar="ACTION")

    def cadd(name: str, help_: str) -> argparse.ArgumentParser:
        q = csub.add_parser(name, help=help_, parents=[_common()])
        q.set_defaults(fn=_cmd_collection)
        return q

    q = cadd("info", "everything about NAME: documents, folders, sizes, dates, state")
    q.add_argument("name")
    q = cadd("export", "write NAME's index to one .rag.tgz file")
    q.add_argument("name")
    q.add_argument("-o", "--output", default="",
                   help="file or folder to write (default: ./NAME.rag.tgz)")
    q = cadd("import", "add the collection in FILE (made by `collection export`)")
    q.add_argument("file")
    q.add_argument("--as", dest="as_name", default="", metavar="NAME",
                   help="import under another name (e.g. when the name is taken)")
    q.add_argument("--replace", action="store_true",
                   help="replace an earlier import of the same collection")
    q = cadd("delete", "delete NAME's Markdown and index from the workspace (documents are never "
                       "touched; indexed again next run while they are in place)")
    q.add_argument("name")
    q.add_argument("-y", "--yes", action="store_true", help="do not ask for confirmation")

    pg_p = add("playground", "a sandbox for trying models/tunables and benchmarking, "
              "structurally separate from your real collections", None,
              description="Everything here lives under <home>/playground/<name>/ -- its own "
              "sources, its own index, its own config.json -- and is never read by production "
              "search, indexing or publish. No daemon: each command loads the small index and "
              "the experiment's chosen models, runs, and exits.")
    pgsub = pg_p.add_subparsers(dest="playground_cmd", metavar="ACTION")

    def pgadd(name: str, help_: str, fn) -> argparse.ArgumentParser:
        q = pgsub.add_parser(name, help=help_, parents=[_common()])
        q.set_defaults(fn=fn)
        return q

    q = pgadd("create", "start a new experiment (optionally seeded with documents)",
              _cmd_playground_create)
    q.add_argument("name")
    q.add_argument("--from", dest="source", action="append", default=[], metavar="FOLDER",
                   help="a folder of sample documents to use as a source, read where it is (repeatable)")
    q.add_argument("--collection", default="",
                   help="collection name for a single --from folder (default: the folder's name)")
    q.add_argument("--from-production", dest="from_production", action="store_true",
                   help="seed the experiment's embedding/reranker/chunk/search settings from "
                   "production's current config.json, instead of this module's own defaults")

    q = pgadd("source", "list, add or remove the folders an experiment reads its documents from",
              _cmd_playground_source)
    q.add_argument("name")
    q.add_argument("action", choices=["list", "add", "remove"])
    q.add_argument("arg", nargs="?", default="", help="the folder (add) or the collection (remove)")
    q.add_argument("--as", dest="collection", default="", metavar="COLLECTION",
                   help="collection name for the folder (default: the folder's own name)")

    q = pgadd("config", "show, or change, an experiment's model/chunk/tunable choices",
              _cmd_playground_config)
    q.add_argument("name")
    q.add_argument("--embedding-model", dest="embedding_model", default=None, metavar="ID")
    q.add_argument("--rerank-model", dest="rerank_model", default=None, metavar="ID")
    q.add_argument("--reader-model", dest="reader_model", default=None, metavar="ID",
                   help="the document reader (stage 3.2b) for this experiment; 'production' = use production's choice")
    q.add_argument("--repair-model", dest="repair_model", default=None, metavar="ID",
                   help="the repair model (stage 3.4) for this experiment; 'production' = use production's choice")
    q.add_argument("--chunk-size", dest="chunk_size", type=int, default=None)
    q.add_argument("--chunk-overlap", dest="chunk_overlap", type=int, default=None)
    q.add_argument("--rerank", action="store_true", help="use a reranker (default)")
    q.add_argument("--no-rerank", dest="no_rerank", action="store_true",
                   help="skip reranking for this experiment")
    q.add_argument("--stages", default=None,
                   help="default comma-separated subset of bm25,dense,rerank for this experiment")
    q.add_argument("--retrieval-pool", dest="retrieval_pool", type=int, default=None)
    q.add_argument("--rerank-pool", dest="rerank_pool", type=int, default=None)
    q.add_argument("--rrf-k", dest="rrf_k", type=int, default=None)
    # Every docling/OCR/table/PDF-backend knob (everything in spec.py's "indexer" section that
    # has an environment variable -- chunk_size/chunk_overlap are handled by their own flags
    # above), generated the same way `rag-search config`'s `set` action generates its flags, so
    # this can never drift from what production and the dashboard's Settings tab offer for the
    # same knob. An empty value (e.g. --ocr '') clears it back to "no override" for this
    # experiment, exactly like `rag-search config set`.
    for t in (t for t in spec.TUNABLES_BY_SECTION["indexer"] if t.env):
        kw: dict[str, Any] = {"dest": t.key, "default": None,
                              "help": f"{t.what}" + (f" ({', '.join(t.choices)})" if t.choices else "")}
        if t.kind == "int":
            kw["type"] = int
        q.add_argument(t.cli_flag, **kw)

    q = pgadd("index", "build (or rebuild) the experiment's sample index -- no publish, "
              "no generations", _cmd_playground_index)
    q.add_argument("name")
    q.add_argument("--rebuild", action="store_true", help="re-embed even if unchanged")
    q.add_argument("--wipe", action="store_true", help="drop documents no longer in the sources first")
    q.add_argument("--force-md", dest="force_md", action="store_true", help="re-convert to Markdown")
    q.add_argument("--jobs", type=int, default=1)
    q.add_argument("--job", default="", help=argparse.SUPPRESS)       # set by the dashboard

    q = pgadd("search", "search the experiment's sample index (loads in-process, no daemon)",
              _cmd_playground_search)
    q.add_argument("name")
    q.add_argument("query")
    q.add_argument("-c", "--collection", default="")
    q.add_argument("-k", "--top-k", type=int, default=DEFAULT_TOP_K, dest="top_k")
    q.add_argument("--stages", default=None)
    q.add_argument("--retrieval-pool", dest="retrieval_pool", type=int, default=None)
    q.add_argument("--rerank-pool", dest="rerank_pool", type=int, default=None)
    q.add_argument("--rrf-k", dest="rrf_k", type=int, default=None)
    q.add_argument("--no-rerank", dest="no_rerank", action="store_true")
    q.add_argument("--explain", action="store_true",
                   help="show each hit's BM25/dense/RRF/rerank scores and pipeline diagnostics")

    q = pgadd("bench", "replay a labeled query set and record Recall@k/MRR/nDCG@k/latency",
              _cmd_playground_bench)
    q.add_argument("name")
    q.add_argument("--queries", default=None, metavar="PATH",
                   help="default: <experiment>/bench/queries.jsonl")
    q.add_argument("-k", type=int, default=5)
    q.add_argument("--stages", default=None)
    q.add_argument("--retrieval-pool", dest="retrieval_pool", type=int, default=None)
    q.add_argument("--rerank-pool", dest="rerank_pool", type=int, default=None)
    q.add_argument("--rrf-k", dest="rrf_k", type=int, default=None)
    q.add_argument("--no-rerank", dest="no_rerank", action="store_true")
    q.add_argument("--label", default=None, help="a short name for this run, for `compare`")
    q.add_argument("--job", default="", help=argparse.SUPPRESS)       # set by the dashboard

    q = pgadd("compare", "every recorded bench run for an experiment, side by side",
              _cmd_playground_compare)
    q.add_argument("name")
    q.add_argument("--sort", choices=["recall", "mrr", "latency"], default="recall")

    q = pgadd("promote", "write an experiment's embedding/reranker/chunk/search settings into "
              "production's config.json", _cmd_playground_promote)
    q.add_argument("name")
    q.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="show what would change, without writing anything")
    q.add_argument("--confirm", action="store_true",
                   help="required when the promotion changes the embedding model or chunk "
                   "size/overlap, since that makes the existing production index stale until "
                   "it is rebuilt")

    q = pgadd("settings", "what the experiment's next run really uses, by pipeline stage",
              _cmd_playground_settings)
    q.add_argument("name")

    q = pgadd("status", "the latest (or a given) run of an experiment: stage, documents, pages",
              _cmd_playground_status)
    q.add_argument("name")
    q.add_argument("--run", default="", help="a run id (default: the latest)")

    pgadd("list", "every playground experiment", _cmd_playground_list)

    q = pgadd("rm", "permanently delete an experiment and its bench runs", _cmd_playground_rm)
    q.add_argument("name")
    q.add_argument("-y", "--yes", action="store_true", help="do not ask for confirmation")

    p = add("ui", "open the web dashboard (status, indexing, search, access, architecture, help)",
            _cmd_ui, description="Local web dashboard on 127.0.0.1: live daemon and indexing status, "
            "collections and per-client access, a search playground, and architecture and CLI help. "
            "Everything you can do with the CLI you can do there. Only this computer can reach it.")
    p.add_argument("--port", type=int, default=None,
                   help="port to listen on (default 8765 or $RAG_SEARCH_UI_PORT; 0 = any free port)")
    p.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    p.add_argument("--read-only", action="store_true",
                   help="only look: refuse indexing, access changes and daemon control")
    p.add_argument("--detach", action="store_true", help="keep running in the background")
    p.add_argument("--stop", action="store_true", help="stop a dashboard started with --detach")
    p.add_argument("--url", action="store_true", help="print the dashboard URL (with its token) and exit")

    p = add("daemon", "control the search and indexer daemons", _cmd_daemon)
    p.add_argument("action", choices=["status", "start", "stop", "restart", "run"])
    p.add_argument("which", nargs="?", default="all", help="search | indexer | all")

    p = add("service", "start the daemons at login (macOS launchd)", _cmd_service)
    p.add_argument("action", choices=["install", "uninstall", "status"])

    p = add("config", "show, create, or change production tunables in config.json", _cmd_config,
            description="Without an action: print the effective config.json (built-in defaults "
            "merged with the file, merged with environment variables). `set` changes one or more "
            "tunables and prints what changed; an empty value (e.g. --ocr '') clears a tunable "
            "back to its built-in default. Every flag below is documented, with what it means "
            "and its impact, in the dashboard's Settings and Models tabs and in ARCHITECTURE.md.")
    p.add_argument("action", nargs="?", choices=["show", "init", "path", "set"], default="show")
    for t in spec.TUNABLES:
        kw: dict[str, Any] = {"dest": t.key, "default": None,
                              "help": f"[{t.section}] {t.what}" + (f" ({', '.join(t.choices)})"
                                                                   if t.choices else "")}
        if t.kind == "int":
            kw["type"] = int
        p.add_argument(t.cli_flag, **kw)

    p = add("paths", "print data and index folder locations", _cmd_paths)
    p.add_argument("name", nargs="?", default="")

    mp = add("models", "list, download and switch the embedding model and the reranker", _cmd_models,
             description="Without an action: what is available, what is in use, what fits this "
             "machine. Models are downloaded from Hugging Face into its cache. A new reranker is "
             "used at once (no re-indexing); a new embedding model means every document is "
             "embedded again, and the new index goes live only when it is complete.")
    msub = mp.add_subparsers(dest="models_cmd", metavar="ACTION")

    def madd(name: str, help_: str) -> argparse.ArgumentParser:
        q = msub.add_parser(name, help=help_, parents=[_common()])
        q.set_defaults(fn=_cmd_models)
        return q

    def switch_flags(q: argparse.ArgumentParser) -> None:
        q.add_argument("-y", "--yes", action="store_true", help="do not ask for confirmation")
        q.add_argument("--no-reindex", action="store_true",
                       help="save the embedding model but do not start re-embedding now "
                            "(the next `rag-search index new` does it)")
        q.add_argument("--no-verify", action="store_true", help="skip the quick test of the model")
        q.add_argument("--force", action="store_true",
                       help="go ahead although the model is too large for this machine's budget")

    madd("list", "the available models and which are in use (default)")
    q = madd("set", "switch the embedding model or the reranker (asks first)")
    q.add_argument("kind", choices=["embedding", "reranker"])
    q.add_argument("model", metavar="MODEL_ID", help="a catalogue id or any Hugging Face ORG/NAME")
    switch_flags(q)
    q = madd("use", "switch both to a preset: default, qwen3-small, qwen3-large")
    q.add_argument("preset")
    switch_flags(q)
    q = madd("download", "download models now (default: the two in use)")
    q.add_argument("models_ids", nargs="*", metavar="MODEL_ID")
    q.add_argument("--reader", action="store_true", help="also download the document reader model (about 3 GB)")
    for kind, text in (("reader", "choose the document reader (a vision model for scanned pages and images)"),
                       ("repair", "choose the model that re-reads suspect table cells")):
        q = madd(kind, text)
        q.add_argument("model", nargs="?", default="", metavar="MODEL_ID",
                       help="a catalogue id or any Hugging Face ORG/NAME; without it the current choice is shown")
    q = madd("runtime", "the document reader's runtime (mlx-vlm, ocrmac): status, or `install`")
    q.add_argument("what", nargs="?", choices=["status", "install"], default="status")
    q = madd("verify", "load the models in use and run a quick relevance test")
    q.add_argument("kind", nargs="?", choices=["embedding", "reranker"], default="")
    q = madd("limit", "memory the models may use, for the fit check (GB, or off)")
    q.add_argument("value", nargs="?", default=None, metavar="GB")
    madd("status", "the running or last download/switch")
    madd("cancel", "stop the running download/switch")

    p = add("setup", "create folders and download the models", _cmd_setup)
    p.add_argument("--models", default="", metavar="PRESET",
                   help="use a model preset first: default, qwen3-small or qwen3-large")
    p.add_argument("--skip-models", action="store_true")
    p.add_argument("--skip-docling", action="store_true")

    p = add("doctor", "check the installation", _cmd_doctor)
    p.add_argument("--roundtrip", action="store_true",
                   help="also index and search 3 tiny docs end to end (real daemons and models)")

    p = add("convert", "convert one document to page-annotated Markdown", _cmd_convert)
    p.add_argument("file")
    p.add_argument("-o", "--output", default="",
                   help="output file (with --compare: the folder for the outputs)")
    p.add_argument("--compare", action="store_true",
                   help="time the current settings against faster ones (OCR smart, Apple Vision, "
                        "fast tables) on this file, and show how much the text differs")
    p.add_argument("--ocr", action="store_true",
                   help="OCR PDFs even if RAG_SEARCH_OCR=off")
    p.add_argument("--mode", choices=("force", "smart", "auto", "off"), default="",
                   help="OCR mode for this run (default: RAG_SEARCH_OCR)")
    p.add_argument("--backend", choices=("pypdfium2", "docling-parse", "default"), default="",
                   help="PDF backend for this run (default: RAG_SEARCH_PDF_BACKEND)")
    p.add_argument("--engine", default="",
                   help="OCR engine for this run: auto, ocrmac, rapidocr, easyocr, tesseract "
                        "(default: RAG_SEARCH_OCR_ENGINE)")
    p.add_argument("--table", choices=("accurate", "fast"), default="",
                   help="table structure mode for this run (default: RAG_SEARCH_TABLE_MODE)")

    p = add("convert-legacy", "convert legacy .doc/.xls/.ppt/.rtf files to modern formats "
                              "(via LibreOffice) so they can be indexed",
           _cmd_convert_legacy,
           description="docling only reads modern Office formats reliably; this converts "
                       ".doc/.rtf -> .docx, .xls -> .xlsx and .ppt -> .pptx with LibreOffice, "
                       "one file at a time, writing each converted copy next to its original. "
                       "This is the one rag-search command that writes into a source folder, and "
                       "only because you ask it to: indexing itself never changes, moves or "
                       "deletes a source document.  Originals are kept unless you pass "
                       "--delete-originals.  Requires LibreOffice (brew install --cask "
                       "libreoffice).")
    p.add_argument("path", help="folder to scan recursively")
    p.add_argument("--ext", default="doc,xls,ppt,rtf",
                   help="comma-separated legacy extensions to convert (default: doc,xls,ppt,rtf)")
    p.add_argument("--dry-run", action="store_true",
                   help="show what would be converted; touches nothing")
    p.add_argument("--delete-originals", action="store_true", dest="delete_originals",
                   help="delete each legacy file once its converted copy is verified")
    p.add_argument("--keep-originals", action="store_true",
                   help="keep the legacy files (the default; accepted for older scripts)")
    p.add_argument("--force", action="store_true",
                   help="reconvert even if a modern copy already exists (overwriting it)")
    p.add_argument("--timeout", type=float, default=120.0,
                   help="seconds to wait for LibreOffice per file before giving up (default: 120)")

    from . import register as _register

    hosts = _register.extra_hosts()
    also = "".join(f", and {h.LABEL} when installed" if getattr(h, "AUTO_REGISTER", False) else ""
                   for h in hosts)
    p = add("register", f"register the MCP adapter (Claude Desktop and Claude Code{also})", _cmd_register)
    p.add_argument("--desktop", action="store_true")
    p.add_argument("--code", action="store_true")
    for h in hosts:                       # optional hosts: flags shown only if the module advertises them
        shown = (lambda t: t) if getattr(h, "ADVERTISE", False) else (lambda t: argparse.SUPPRESS)
        p.add_argument(f"--{h.NAME}", dest=f"host_{h.NAME}", action="store_true",
                       help=shown(f"{h.LABEL} only ({h.default_config()}; created if missing)"))
        p.add_argument(f"--{h.NAME}-file", dest=f"host_{h.NAME}_file", default="",
                       metavar="PATH", help=shown(f"path of {h.LABEL}'s MCP settings file"))
        p.add_argument(f"--no-{h.NAME}", dest=f"no_host_{h.NAME}", action="store_true",
                       help=shown(f"do not register with {h.LABEL}"))
    p.add_argument("--tool-prefix", default="", help="extra prefix for the MCP tool names (they already start with rag_)")

    p = add("unregister", "remove the registration", _cmd_unregister)
    p.add_argument("--desktop", action="store_true")
    p.add_argument("--code", action="store_true")
    for h in hosts:
        p.add_argument(f"--{h.NAME}", dest=f"host_{h.NAME}", action="store_true", help=argparse.SUPPRESS)
        p.add_argument(f"--{h.NAME}-file", dest=f"host_{h.NAME}_file", default="", help=argparse.SUPPRESS)

    p = add("mcp-config", "print an mcpServers JSON snippet", _cmd_mcp_config)
    p.add_argument("--profile", default="claude", metavar="NAME",
                   help="the host's name (default: claude)")
    p.add_argument("--tool-prefix", default="")

    add("serve", "run the MCP adapter on stdio (same as rag-search-mcp)", _cmd_serve)
    return ap


def main(argv: list[str] | None = None) -> None:
    ap = build_parser()
    args = ap.parse_args(argv)
    allow_cloud_files()
    if args.home:
        os.environ["RAG_SEARCH_HOME"] = str(Path(args.home).expanduser().absolute())
    if args.client:
        os.environ["RAG_SEARCH_CLIENT"] = args.client
    fn = getattr(args, "fn", None)
    if fn is None:
        if args.command in ("index", "playground", "bench"):
            ap.parse_args([args.command, "--help"])
        ap.print_help()
        raise SystemExit(EXIT_USAGE)
    try:
        raise SystemExit(fn(args))
    except BrokenPipeError:
        raise SystemExit(EXIT_OK) from None
    except KeyboardInterrupt:
        raise SystemExit(130) from None


if __name__ == "__main__":
    # `python -m rag_search.cli` is how child processes run (a Playground experiment, doctor): end
    # like the indexing worker does, without joining a thread docling abandoned (core/worker.leave).
    from .core.worker import leave

    try:
        main()
        _code: object = EXIT_OK
    except SystemExit as _exc:
        _code = _exc.code
    if _code is not None and not isinstance(_code, int):
        print(_code, file=sys.stderr)
        _code = 1
    leave(int(_code or 0))
