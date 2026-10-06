"""High-level operations shared by the CLI and the MCP adapter (stdlib only).

Everything here is a thin call over the daemon sockets (`client.py`); the two
read-only operations `list` and `grep` fall back to reading the published generation
directly when the search daemon is not running, applying the same per-client access rules.
All functions are synchronous and return the daemon-style reply dict:
``{"ok": True, ...}`` or ``{"ok": False, "code": ..., "error": ...}``.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

from . import client as client_mod
from . import descriptions, jobs, policy, protocol, publish
from .catalog import collection_names, known_names, list_view, live_catalog
from .grep import grep_isolated
from .paths import SUPPORTED_EXTENSIONS, Paths, is_plain_name, parse_collections, read_json

KINDS = ("search", "indexer")


def _client(client: str | None) -> str:
    return policy.normalize_client(client or client_mod.client_id())


# ── search side ──────────────────────────────────────────────────────────────

def search(paths: Paths, query: str, *, top_k: int | None = None,
           collections: list[str] | str | None = None,
           client: str | None = None, wait_s: float = 45.0, origin: str = "",
           stages: list[str] | str | None = None, retrieval_pool: int | None = None,
           rerank_pool: int | None = None, rrf_k: int | None = None) -> dict[str, Any]:
    """Hybrid search via the search daemon (started on demand).

    ``result["timing"]`` says where the time went (see core/search.py); ``round_trip_ms`` is
    what this call took including the socket, and starting the daemon if it was not running.
    ``top_k``/``stages``/``retrieval_pool``/``rerank_pool``/``rrf_k`` are optional overrides for
    this one search (see core/search.py and spec.py); leaving any of them unset (the default)
    falls back to the production default configured in ``config.json`` (``rag-search config
    set``, or the dashboard's Settings tab), and from there to the built-in default -- so an
    administrator can change what "the default search" means for every client, including the
    MCP tool, without every caller having to ask for it.
    """
    t0 = time.perf_counter()
    fields: dict[str, Any] = {}
    if top_k is not None:
        fields["top_k"] = top_k
    if stages is not None:
        fields["stages"] = stages
    if retrieval_pool is not None:
        fields["retrieval_pool"] = retrieval_pool
    if rerank_pool is not None:
        fields["rerank_pool"] = rerank_pool
    if rrf_k is not None:
        fields["rrf_k"] = rrf_k
    r = client_mod.request_sync(
        paths, "search", "search", client=_client(client), wait_s=wait_s, query=query,
        collections=parse_collections(collections), origin=origin, **fields)
    return _with_round_trip(r, t0)


def _with_round_trip(r: dict[str, Any], t0: float) -> dict[str, Any]:
    res = r.get("result") if r.get("ok") else None
    if isinstance(res, dict):
        res.setdefault("timing", {})["round_trip_ms"] = int((time.perf_counter() - t0) * 1000)
    return r


def list_collections(paths: Paths, *, client: str | None = None, full: bool = False) -> dict[str, Any]:
    """`full=True` also includes each collection's per-document listing (see catalog.list_view)."""
    r = client_mod.request_sync(paths, "search", "list", client=_client(client),
                                autostart=False, wait_s=0, request_timeout=15, full=full)
    if r.get("ok") or r.get("code") not in (protocol.UNAVAILABLE, protocol.WARMING_UP):
        return r
    rules = policy.current_rules(paths)
    return {"ok": True, "source": "local",
            "result": list_view(paths, rules, _client(client), full=full)}


def describe_collection(paths: Paths, name: str, description: str,
                        *, client: str | None = None) -> dict[str, Any]:
    """Set (or, with an empty *description*, clear) one collection's description.

    A local, file-based write (like `rag-search access`), not routed through the daemon, and the
    one writer of descriptions: the CLI (`rag-search describe`), the MCP tool and the dashboard
    all call it.  Refuses a name *client* is not authorised to see, using the same "unknown
    collection" wording as search/grep so a restricted collection is never confirmed to exist.
    Clients may describe published collections; the administrator's terminal (`cli`) also ones
    not published yet (a registered location, an import in progress).
    """
    c = _client(client)
    if not is_plain_name((name or "").strip()):
        return protocol.error(protocol.BAD_REQUEST, f"{name!r} is not a collection name (the name a "
                              "folder was registered with, e.g. 'manuals')")
    rules = policy.current_rules(paths)
    existing = collection_names(live_catalog(paths))
    if c == policy.ADMIN_CLIENT:
        existing = list(dict.fromkeys([*existing, *known_names(paths)]))
    requested = [name.strip()] if name.strip() else []
    scope, err = policy.resolve_scope(rules, c, requested, existing)
    if err:
        return protocol.error(protocol.BAD_REQUEST, err)
    try:
        result = descriptions.set_description(paths, scope[0] if scope else name, description)
    except descriptions.DescriptionError as exc:
        return protocol.error(protocol.BAD_REQUEST, str(exc))
    return {"ok": True, "result": result}


def grep(paths: Paths, pattern: str, *, collections: list[str] | str | None = None,
         context_lines: int = 2, max_matches: int = 20,
         client: str | None = None, origin: str = "") -> dict[str, Any]:
    t0 = time.perf_counter()
    r = client_mod.request_sync(
        paths, "search", "grep", client=_client(client), autostart=False, wait_s=0,
        request_timeout=60, pattern=pattern, collections=parse_collections(collections),
        context_lines=context_lines, max_matches=max_matches, origin=origin)
    if r.get("ok") or r.get("code") not in (protocol.UNAVAILABLE, protocol.WARMING_UP):
        return _with_round_trip(r, t0)
    return _with_round_trip(
        local_grep(paths, pattern, collections, context_lines, max_matches, _client(client)), t0)


def local_grep(paths: Paths, pattern: str, collections: list[str] | str | None,
               context_lines: int, max_matches: int, client: str) -> dict[str, Any]:
    """Same access rules and result shape as the daemon, read straight from `serving/current`."""
    rules = policy.current_rules(paths)
    try:
        requested = parse_collections(collections)
    except ValueError as exc:
        return protocol.error(protocol.BAD_REQUEST, str(exc))
    scope, err = policy.resolve_scope(rules, client, requested,
                                      collection_names(live_catalog(paths)))
    if err:
        return protocol.error(protocol.BAD_REQUEST, err)
    res = grep_isolated(paths.live_markup(), pattern, scope, context_lines, max_matches)
    return {"ok": True, "source": "local", "result": res}


# ── indexing side ────────────────────────────────────────────────────────────

def index_start(paths: Paths, *, mode: str = "new", path: str = "", rebuild: bool = False,
                force_md: bool = False, restart: bool = False,
                client: str | None = None) -> dict[str, Any]:
    return client_mod.request_sync(paths, "indexer", "start", client=_client(client), wait_s=30,
                                   request_timeout=120, mode=mode, path=path, rebuild=rebuild,
                                   force_md=force_md, restart=restart)


def index_status(paths: Paths, *, job_id: str = "", history: int = 0, docs: int = 50,
                 doc_status: str = "", doc_collection: str = "", doc_q: str = "",
                 doc_branch: str = "", doc_outcome: str = "",
                 client: str | None = None) -> dict[str, Any]:
    """Summary of the active/last run, with per-document timings (`documents`, the last *docs*
    matching ``doc_status``/``doc_collection``/``doc_q`` -- all optional; unset reproduces
    today's "the last *docs* documents touched" behaviour).

    Does not start the daemon just to say 'idle'.
    """
    c = _client(client)
    r = _index_status(paths, job_id, history, client)
    job = r.get("job") if r.get("ok") else None
    if job:
        hidden = policy.current_rules(paths).hidden_from(c)
        r["documents"] = jobs.documents(paths, job["id"], docs, status=doc_status,
                                        collection=doc_collection, q=doc_q, hidden=hidden,
                                        branch=doc_branch, outcome=doc_outcome)
    return _hide_restricted(paths, c, r)


def _hide_restricted(paths: Paths, client: str, r: dict[str, Any]) -> dict[str, Any]:
    """Indexing is shared, but a client must not learn about collections it may not use: blank
    strings that name them.  (Per-document results are already filtered to visible collections
    by `index_status` itself, before any total or count is computed.)"""
    hidden = policy.current_rules(paths).hidden_from(client)
    if not hidden or not r.get("ok"):
        return r
    return policy.scrub(r, hidden)


def _index_status(paths: Paths, job_id: str, history: int, client: str | None) -> dict[str, Any]:
    from . import locations

    r = client_mod.request_sync(paths, "indexer", "status", client=_client(client),
                                autostart=False, wait_s=0, request_timeout=15,
                                job_id=job_id, history=history)
    if r.get("code") == protocol.UNAVAILABLE:  # daemon down: the records are still on disk
        rec = jobs.read_record(paths, job_id) if job_id else next(iter(jobs.all_records(paths)),
                                                                   None)
        if job_id and rec is None:
            return protocol.error(protocol.BAD_REQUEST, f"no such job: {job_id}")
        out = {"ok": True, "running": False, "daemon": "not running",
               "job": jobs.view(rec) if rec else None,
               "sources": locations.sources(paths),
               "supported_extensions": sorted(SUPPORTED_EXTENSIONS)}
        if history:
            out["history"] = [jobs.view(x) for x in jobs.all_records(paths)[:history]]
        return out
    return r


def index_cancel(paths: Paths, *, client: str | None = None) -> dict[str, Any]:
    r = client_mod.request_sync(paths, "indexer", "cancel", client=_client(client),
                                autostart=False, wait_s=0, request_timeout=60)
    if r.get("code") == protocol.UNAVAILABLE:
        return {"ok": True, "cancelled": False, "note": "indexer daemon is not running"}
    return r


def index_publish(paths: Paths, *, client: str | None = None) -> dict[str, Any]:
    return client_mod.request_sync(paths, "indexer", "publish", client=_client(client),
                                   wait_s=30, request_timeout=700)


def index_follow(paths: Paths, *, job_id: str = "",
                 client: str | None = None) -> Iterator[dict[str, Any]]:
    """Yield status/progress events until the run ends.  Raises OSError if no daemon."""
    if client_mod.ping(paths, "indexer") is None:
        yield protocol.error(protocol.UNAVAILABLE, "indexer daemon is not running")
        return
    yield from client_mod.stream(paths, "indexer", protocol.make_request(
        "follow", _client(client), job_id=job_id))


def publish_and_reload(paths: Paths, *, allow_drop: bool = False) -> dict[str, Any]:
    """Publish the workspace as a new generation, then tell the search daemon.

    Used by the indexer daemon after a run, by `rag-search index foreground`, and after a
    collection import or deletion (*allow_drop*: a deleted collection may leave the index).
    """
    out: dict[str, Any] = {}
    try:
        out["publish"] = publish.publish(paths, allow_drop=allow_drop)
    except Exception as exc:  # noqa: BLE001
        out["publish"] = {"changed": False, "error": f"{type(exc).__name__}: {exc}"}
        return out
    if out["publish"].get("changed"):
        out["search_reload"] = reload_search(paths)
    return out


def reload_search(paths: Paths) -> dict[str, Any]:
    """Ask the search daemon to swap generations; start it if it is not running."""
    r = client_mod.request_sync(paths, "search", "reload", client="cli", autostart=False,
                                wait_s=0, request_timeout=600)
    if r.get("ok"):
        out = {"ok": True, "generation": r.get("generation"), "changed": r.get("changed"),
               "reused": r.get("reused", []), "loaded": r.get("loaded", [])}
        for k in ("reranker", "reranker_error"):
            if r.get(k):
                out[k] = r[k]
        return out
    if r.get("code") == protocol.UNAVAILABLE:
        started = client_mod.spawn(paths, "search")  # it loads the newest generation at boot
        return {"ok": True, "note": "search daemon was not running; "
                + ("started it" if started else "it is starting")}
    if r.get("code") == protocol.WARMING_UP:  # it loads the newest generation when it is ready
        return {"ok": True, "note": "search daemon is still loading its models; it will pick "
                "up the new generation as soon as it is ready"}
    return {"ok": False, "error": r.get("error", "reload failed"), "code": r.get("code")}


# ── document conversion: what each page took, how long, at what cost ────────────
# Read side of core/conversion (see docs/design/document-conversion-plan.md).  Run totals and
# document lists are readable by any client for the collections it may see; traces, page images
# and the dry run show folder paths and page content, so they are administrator views (CLI /
# dashboard), like collection_info.

def conversion_run(paths: Paths, job_id: str = "", *, client: str | None = None) -> dict[str, Any]:
    """Conversion totals (pages per branch and outcome, time, cost) and the workers' lanes of a
    run -- the latest one when *job_id* is empty.  Works while the run is going and afterwards."""
    from .core.conversion import runview

    return _hide_restricted(paths, _client(client), runview.run_view(paths, job_id))


def conversion_documents(paths: Paths, *, job_id: str = "", status: str = "", collection: str = "",
                         q: str = "", branch: str = "", outcome: str = "", limit: int = 200,
                         client: str | None = None) -> dict[str, Any]:
    """The documents of a run with their conversion summaries, filterable by branch (documents
    with at least one page of it) and outcome; ``by_branch`` / ``by_outcome`` count documents."""
    c = _client(client)
    rec = jobs.read_record(paths, job_id) if job_id else next(iter(jobs.all_records(paths)), None)
    if rec is None:
        if job_id:
            return protocol.error(protocol.BAD_REQUEST, f"no such job: {job_id}")
        return {"ok": True, "job": None, "documents": jobs.documents(paths, "")}
    hidden = policy.current_rules(paths).hidden_from(c)
    docs = jobs.documents(paths, rec["id"], limit, status=status, collection=collection, q=q,
                          hidden=hidden, branch=branch, outcome=outcome)
    return {"ok": True, "job": rec["id"], "documents": docs}


def _doc_files(paths: Paths, collection: str, doc: str):
    """(collection, trace file, index folder) of one document, confined to the workspace."""
    from pathlib import PurePosixPath

    from . import inventory
    from .core.conversion.trace import TRACE_SUFFIX

    coll = inventory._find(paths, collection)
    rel = PurePosixPath(str(doc or "").strip().strip("/"))
    if not rel.parts or rel.is_absolute() or any(x in ("", ".", "..") for x in rel.parts):
        raise inventory.InventoryError(f"{doc!r} is not a document of {coll!r}")
    tfile = paths.markup.joinpath(coll, *rel.parts[:-1], rel.parts[-1] + TRACE_SUFFIX)
    idx = paths.index.joinpath(coll, *rel.parts)
    return coll, tfile, idx


def conversion_trace(paths: Paths, collection: str, doc: str, *, page: int = 0) -> dict[str, Any]:
    """One document's conversion trace.  *doc* is its path inside the collection without the
    extension (as ``rag-search list`` shows it).  Without *page*: the summary and a compact line
    per page; with *page*: that page's full record (profile, reason, reader, checks)."""
    from . import inventory
    from .core.conversion import trace

    try:
        coll, tfile, idx = _doc_files(paths, collection, doc)
    except inventory.InventoryError as exc:
        return protocol.error(protocol.BAD_REQUEST, str(exc))
    data = trace.read_trace(tfile)
    meta = trace.read_json(idx / "index.meta.json") if (idx / "index.meta.json").exists() else {}
    if not data:
        summary = (meta or {}).get("conversion") or {}
        if not summary:
            return protocol.error(protocol.BAD_REQUEST, f"{coll}/{doc}: no conversion record "
                                  "(it was indexed before conversion tracking existed; index it "
                                  "again with --force-md to record one)")
        return {"ok": True, "result": {"collection": coll, "doc": doc, "summary": summary,
                                       "pages": [], "note": "only the summary is stored for this "
                                       "document (an imported collection, or the trace was deleted)"}}
    pages = data.get("pages", [])
    if page:
        pages = [p for p in pages if p.get("page") == int(page)]
        if not pages:
            return protocol.error(protocol.BAD_REQUEST, f"{coll}/{doc}: no page {page} in the trace")
    else:
        pages = [{"page": p.get("page"), "branch": p.get("branch"), "outcome": p.get("outcome"),
                  "runway": (p.get("route") or {}).get("final"),
                  "moved_from": (p.get("route") or {}).get("escalated_from", {}).get("runway"),
                  "engine": (p.get("route") or {}).get("engine"),
                  "chars": (p.get("out") or {}).get("chars"),
                  "grade": (p.get("docling") or {}).get("grade"),
                  "cache": p.get("cache"), "read_s": (p.get("time_s") or {}).get("read"),
                  "failed": [c.get("name") for c in (p.get("gate") or {}).get("checks", [])] or None,
                  "tokens": p.get("tokens"), "model": (p.get("reader") or {}).get("model"),
                  "repair": ({k: p["repair"].get(k) for k in ("tried", "fixed", "tier")} if p.get("repair") else None),
                  "across": (p.get("reconcile") or {}).get("role"), "across_with": (p.get("reconcile") or {}).get("with")}
                 for p in pages]
    return {"ok": True, "result": {
        "collection": coll, "doc": doc, "source": data.get("source"), "convert": data.get("convert"),
        "written_at": data.get("written_at"), "profile": data.get("profile"),
        "summary": data.get("summary"), "pages": pages}}


def _doc_source(paths: Paths, coll: str, doc: str, idx: Any, tfile: Any) -> Any:
    """The readable source file of a document, or None.  ``index.meta.json`` names it once the
    document is embedded; before that (a run still converting, or one that stopped before
    embedding) only the trace exists, which records the file's name: the file is then looked for
    next to the document's place in its collection's own folder.  Either way the result must lie
    inside a registered location (a stored path is never trusted blindly)."""
    from pathlib import Path, PurePosixPath

    from . import locations
    from .core.conversion import trace

    locs, _err = locations.load(paths)
    locs = locs or {}
    candidates = []
    meta = read_json(idx / "index.meta.json") or {}
    if meta.get("src_path"):
        candidates.append(Path(str(meta["src_path"])))
    else:
        name = Path(str((trace.read_trace(tfile) or {}).get("source") or "")).name
        if name:
            sub = PurePosixPath(doc.strip("/")).parts[:-1]
            homes = [Path(locs[coll])] if coll in locs else []
            candidates += [h.joinpath(*sub, name) for h in homes]
    roots = [Path(f) for f in locs.values()]
    for src in candidates:
        try:
            real = src.resolve()
            if real.is_file() and any(real.is_relative_to(r.resolve()) for r in roots):
                return real
        except OSError:
            continue
    return None


def conversion_page_image(paths: Paths, collection: str, doc: str, page: int,
                          width_px: int = 900) -> dict[str, Any]:
    """PNG bytes of a source page, ``{"ok": True, "png": bytes}``.  The source must be in a
    registered location (see ``_doc_source``)."""
    from . import inventory
    from .core.conversion import pageimage

    try:
        coll, tfile, idx = _doc_files(paths, collection, doc)
    except inventory.InventoryError as exc:
        return protocol.error(protocol.BAD_REQUEST, str(exc))
    real = _doc_source(paths, coll, str(doc), idx, tfile)
    if real is None:
        return protocol.error(protocol.BAD_REQUEST, "the source file is not available here (an "
                              "imported collection, a moved or deleted file, or outside the "
                              "registered locations)")
    try:
        return {"ok": True, "png": pageimage.render(real, int(page), width_px)}
    except ImportError as exc:
        return protocol.error(protocol.INTERNAL, f"cannot render pages here: {exc}")
    except ValueError as exc:
        return protocol.error(protocol.BAD_REQUEST, str(exc))
    except Exception as exc:  # noqa: BLE001 - a damaged PDF
        return protocol.error(protocol.BAD_REQUEST, f"cannot render this page: {type(exc).__name__}: {exc}")


MARKDOWN_VIEW_MAX = 2_000_000          # characters of converted Markdown sent in one reply


def conversion_markdown(paths: Paths, collection: str, doc: str, *, page: int = 0) -> dict[str, Any]:
    """The converted Markdown of a document (``markup/<coll>/<doc>.md``, what the chunker read), or
    of one *page* of it.  Confined to the workspace; a very long text is cut and says so."""
    from . import inventory
    from .core.conversion import pagemd
    from .core.conversion.trace import TRACE_SUFFIX

    try:
        coll, tfile, _idx = _doc_files(paths, collection, doc)
    except inventory.InventoryError as exc:
        return protocol.error(protocol.BAD_REQUEST, str(exc))
    md_file = tfile.with_name(tfile.name[:-len(TRACE_SUFFIX)] + ".md")
    try:
        text = md_file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return protocol.error(protocol.BAD_REQUEST, f"{coll}/{doc}: no converted Markdown in the workspace "
                              "(not converted yet, no text in it, or an imported collection)")
    pages = pagemd.split_pages(text)
    if page:
        if int(page) not in pages:
            return protocol.error(protocol.BAD_REQUEST, f"{coll}/{doc}: page {page} has no text in the "
                                  "converted Markdown" + (f" (pages with text: {_ranges(sorted(pages))})" if pages else ""))
        text = pages[int(page)]
    return {"ok": True, "result": {
        "collection": coll, "doc": doc, "page": int(page or 0), "pages": sorted(pages),
        "chars": len(text), "truncated": len(text) > MARKDOWN_VIEW_MAX,
        "markdown": text[:MARKDOWN_VIEW_MAX], "file": str(md_file)}}


def _ranges(nums: list[int]) -> str:
    """``[1, 2, 3, 7]`` -> ``"1-3, 7"``."""
    runs: list[list[int]] = []
    for n in nums:
        if runs and runs[-1][1] == n - 1:
            runs[-1][1] = n
        else:
            runs.append([n, n])
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)


def page_cache(paths: Paths, *, clear: bool = False) -> dict[str, Any]:
    """Size of the page cache (pages already read, kept for resuming and re-indexing); *clear*
    deletes every entry (they are read again next time).  Administrator view."""
    from .core.conversion import pagecache

    pc = pagecache.PageCache(paths.workspace)
    removed = pc.clear() if clear else 0
    return {"ok": True, "result": {**pc.stats(), "dir": str(pc.root), "removed": removed}}


def conversion_estimate(paths: Paths, target: str = "", *, budget_s: float = 60.0,
                        progress: Any = None) -> dict[str, Any]:
    """Dry run: profile the sources under *target* (empty = everything) and estimate pages per
    branch and conversion time.  Reads the sources, converts nothing."""
    from . import locations
    from .core.conversion import estimate as est

    try:
        plan = locations.plan_scan(paths, str(target or "").strip())
    except locations.LocationError as exc:
        return protocol.error(protocol.BAD_REQUEST, str(exc))
    return {"ok": True, "result": est.estimate(paths, plan.sources, budget_s=budget_s,
                                               progress=progress)}


# ── conversion benchmark: gold sets and runs (administrator views, not MCP tools) ──
# Read the sources, never the production index; results live in <home>/conversion_gold and
# <home>/conversion_bench (core/conversion/bench.py).

def _bench(fn, *args, **kw) -> dict[str, Any]:
    from .core.conversion import bench, engines

    try:
        return {"ok": True, "result": fn(*args, **kw)}
    except (bench.BenchError, engines.EngineError) as exc:
        return protocol.error(protocol.BAD_REQUEST, str(exc))


def bench_gold_list(paths: Paths) -> dict[str, Any]:
    from .core.conversion import bench, engines

    return {"ok": True, "result": {"sets": bench.list_gold(paths), "engines": engines.available()}}


def bench_gold_init(paths: Paths, set_name: str, *, collection: str = "", per_class: int = 5,
                    classes: list[str] | None = None, seed: int = 0) -> dict[str, Any]:
    from .core.conversion import bench

    return _bench(bench.gold_init, paths, set_name, collection=collection, per_class=per_class,
                  classes=classes, seed=seed)


def bench_gold_show(paths: Paths, set_name: str) -> dict[str, Any]:
    from .core.conversion import bench

    def show() -> dict[str, Any]:
        g = bench.read_gold(paths, set_name)
        return {"name": set_name, "dir": str(bench.gold_dir(paths, set_name)),
                "pages": [{k: e.get(k) for k in ("id", "class", "rel", "page", "verified", "script",
                                                 "image")} | {"chars": len(e.get("truth") or ""),
                                                              "queries": len(e.get("queries") or [])}
                          for e in g["pages"]]}

    return _bench(show)


def bench_run(paths: Paths, set_name: str, *, engine: str = "current", name: str = "",
              include_drafts: bool = False, classes: list[str] | None = None, limit: int = 0,
              progress: Any = None) -> dict[str, Any]:
    from .core.conversion import bench

    return _bench(bench.run_bench, paths, set_name, engine=engine, name=name,
                  include_drafts=include_drafts, classes=classes, limit=limit, progress=progress)


def bench_list(paths: Paths, set_name: str = "") -> dict[str, Any]:
    from .core.conversion import bench

    return _bench(bench.list_runs, paths, set_name)


def bench_show(paths: Paths, set_name: str, run_id: str) -> dict[str, Any]:
    from .core.conversion import bench

    return _bench(bench.read_run, paths, set_name, run_id)


def bench_compare(paths: Paths, set_name: str, a: str, b: str) -> dict[str, Any]:
    from .core.conversion import bench

    return _bench(bench.compare, paths, set_name, a, b)


# ── collections: source locations, export/import, deletion (CLI / admin only) ──
# None of these is an MCP tool: they change what exists, not just what one client sees.

def collection_info(paths: Paths, name: str) -> dict[str, Any]:
    """Counts, folders, sizes, dates and state of one collection (administrator view)."""
    from . import inventory

    try:
        return {"ok": True, "result": inventory.collection_info(paths, name)}
    except inventory.InventoryError as exc:
        return protocol.error(protocol.BAD_REQUEST, str(exc))


def location_list(paths: Paths) -> dict[str, Any]:
    from . import locations

    _, error = locations.load(paths)
    return {"ok": True, "result": {"locations": locations.status(paths),
                                   "file": str(paths.locations_file), "error": error}}


def location_add(paths: Paths, name: str, folder: str) -> dict[str, Any]:
    from . import locations

    try:
        return {"ok": True, "result": locations.add(paths, name, folder)}
    except (locations.LocationError, TimeoutError) as exc:
        return protocol.error(protocol.BAD_REQUEST, str(exc))


def location_remove(paths: Paths, name: str) -> dict[str, Any]:
    """Unregister a location and delete its collection's index (its folder is not touched)."""
    from . import locations

    known = {n.casefold(): n for n in locations.names(paths)}
    if (name or "").strip().casefold() not in known:
        return protocol.error(protocol.BAD_REQUEST, f"no registered location named {name!r}")
    return collection_delete(paths, known[name.strip().casefold()], unregister=True)


def collection_export(paths: Paths, name: str, out: str | None = None) -> dict[str, Any]:
    from . import bundle

    try:
        return {"ok": True, "result": bundle.export_collection(paths, name, out)}
    except bundle.BundleError as exc:
        return protocol.error(protocol.BAD_REQUEST, str(exc))


def collection_import(paths: Paths, archive: str, *, as_name: str | None = None,
                      replace: bool = False) -> dict[str, Any]:
    """Import an export file (model must match), then publish so it is searchable at once."""
    from . import bundle

    try:
        res = bundle.import_collection(paths, archive, as_name=as_name, replace=replace)
    except bundle.BundleError as exc:
        return protocol.error(protocol.BAD_REQUEST, str(exc))
    except (ValueError, TypeError, KeyError, AttributeError, OSError) as exc:
        return protocol.error(protocol.BAD_REQUEST,
                              f"this export could not be imported: {type(exc).__name__}: {exc}")
    return {"ok": True, "result": res, **publish_and_reload(paths)}


def collection_delete(paths: Paths, name: str, *, unregister: bool = False) -> dict[str, Any]:
    """Delete a collection's Markdown and index from the workspace (never its documents, access
    rule or description), then publish.  *unregister*: also unregister a location."""
    from . import lifecycle

    try:
        res = lifecycle.delete_collection(paths, name, unregister=unregister)
    except lifecycle.LifecycleError as exc:
        return protocol.error(protocol.BAD_REQUEST, str(exc))
    return {"ok": True, "result": res, **publish_and_reload(paths, allow_drop=True)}


# ── daemons ──────────────────────────────────────────────────────────────────

def _kinds(which: str) -> tuple[str, ...]:
    if which in ("", "all"):
        return KINDS
    if which not in KINDS:
        raise ValueError(f"unknown daemon {which!r}; choose search, indexer or all")
    return (which,)


def daemon_status(paths: Paths) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for kind in KINDS:
        info = client_mod.ping(paths, kind)
        out[kind] = info if info else {"running": False,
                                       "starting": client_mod.alive_lock_held(paths, kind)}
    return out


def pipeline(paths: Paths) -> dict[str, Any]:
    """Every stage of the indexing and the search pipeline with the settings it uses *now*: value, where
    the value comes from (default | config.json | environment variable of the daemon), when a change
    applies, and which function reads it (``effective.resolve``).  Indexing stages are resolved with the
    environment of the indexer daemon, search stages with that of the search daemon (what each really
    runs with); when a daemon is not running, with this process's environment."""
    from . import effective, stages
    from .config import load_config

    cfg, error = load_config(paths)
    out: dict[str, Any] = {"ok": True, "config_error": error, "stages": [], "daemons": {}}
    resolved: dict[str, dict[str, Any]] = {}
    for kind in KINDS:
        ping = client_mod.ping(paths, kind)
        env = (ping or {}).get("env_overrides") if ping else None
        out["daemons"][kind] = {"running": bool(ping), "env_overrides": env or {}}
        resolved[kind] = effective.resolve(cfg, env if ping else None)
    errors = []
    for st in stages.ALL:
        kind = "search" if st.id.startswith("S") else "indexer"
        row = next(r for r in resolved[kind]["stages"] if r["id"] == st.id)
        row = {**row, "daemon": kind}
        out["stages"].append(row)
    for kind in KINDS:
        errors += [e for e in resolved[kind]["errors"] if e not in errors]
    out["errors"] = errors
    out["overrides"] = {k: out["daemons"][k]["env_overrides"] for k in KINDS}
    return out


def daemon_start(paths: Paths, which: str = "all") -> dict[str, str]:
    out = {}
    for kind in _kinds(which):
        out[kind] = "started" if client_mod.spawn(paths, kind) else "already running/starting"
    return out


def daemon_stop(paths: Paths, which: str = "all") -> dict[str, str]:
    out = {}
    for kind in _kinds(which):
        if client_mod.ping(paths, kind) is None and not client_mod.alive_lock_held(paths, kind):
            out[kind] = "not running"
        else:
            out[kind] = "stopped" if client_mod.stop(paths, kind) else "could not stop"
    return out


def daemon_restart(paths: Paths, which: str = "all") -> dict[str, dict[str, str]]:
    stopped = daemon_stop(paths, which)
    return {"stop": stopped, "start": daemon_start(paths, which)}
