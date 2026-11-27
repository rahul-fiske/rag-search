"""MCP server (stdio): tools that call the rag-search daemons through `rag_search.api`.

This process imports only the standard library and `mcp`; it never loads a model.  The
tool functions are built by `make_tools(client)` so they can be tested without an MCP
runtime, and so that each host's identity is bound into every request.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from .. import __version__, api
from ..paths import get_paths
from ..policy import RESERVED_CLIENTS, normalize_client, valid_client_name
from .profiles import DEFAULT_PROFILE, get_profile, instructions

log = logging.getLogger("rag_search.mcp")

MAX_WAIT_SECONDS = 50


def _json(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


def _error(resp: dict[str, Any], **extra: Any) -> str:
    code = resp.get("code", "")
    if code == "warming_up":
        return _json({"status": "warming_up", "results": [], **extra,
                      "message": "The rag-search daemon is still starting (loading the "
                                 "embedding and reranking models; the very first start after "
                                 "install downloads them once, which can take minutes). "
                                 "Retry in about 30 seconds.",
                      "detail": resp.get("error", ""), "state": resp.get("state")})
    return _json({"error": resp.get("error", "request failed"), "code": code, "results": [],
                  **{k: v for k, v in resp.items() if k in ("log", "supported", "source", "detail")}, **extra})


def make_tools(client: str) -> dict[str, Callable[..., Awaitable[Any]]]:
    """The tool functions, bound to *client* (the identity that selects its collections)."""

    async def rag_list_collections(documents: bool = False) -> str:
        """List the indexed collections this host is authorised to use.

        A collection is a folder of documents registered under a name (rag-search location
        add). Returns JSON {generation, collections:
        [{collection, description, chunks, document_count, index_bytes, markdown_bytes,
        source_bytes, built_at, build_seconds}], totals}. `description` is a
        short summary set with rag_describe_collection, or "" if none has been set yet --
        after exploring an undescribed collection (a search or two, or documents=true below),
        call rag_describe_collection once to save a short summary for future calls, yours or
        another agent's. `built_at` is when the collection was last indexed and
        `build_seconds` how long building it took. Collections that this host is not
        authorised for are not listed and cannot be searched.

        Args:
            documents: If true, also include each collection's per-document listing
                (documents: [{name, source, chunks, indexed_at, build_s, index_bytes}]).
                This can be a lot of data for a large collection -- leave it false unless
                you need individual file names or per-document stats.
        """
        r = await asyncio.to_thread(api.list_collections, get_paths(), client=client,
                                    full=documents)
        return _json(r["result"]) if r.get("ok") else _error(r)

    async def rag_describe_collection(collection: str = "", description: str | None = None,
                                      clear: bool = False) -> str:
        """View, set or clear the short descriptions of collections -- one tool for all of it.

        A description is what `rag_list_collections` shows for a collection instead of just a
        name and a document count; agents read it to choose which collection to search.
        It can't be generated automatically, so after exploring a collection (a search, a
        grep, a rag_list_documents listing) save what it holds: dense keywords beat prose --
        document types, people, organisations, places, date range, typical search terms, and
        what it does NOT hold when that is easy to confuse with another collection.
        Descriptions are shared by every host that may see the collection.

        Calls:
            rag_describe_collection()                          -> every collection you may see,
                with its description ("" if none) and `undescribed`: the names still without one
            rag_describe_collection("manuals")                 -> that collection's description
            rag_describe_collection("manuals", "X100 manuals") -> set it (overwrites the old one)
            rag_describe_collection("manuals", clear=True)     -> remove it
                (description="" also removes it)

        Args:
            collection: Exact collection name as returned by rag_list_collections; empty lists all.
            description: New text, 500 characters at most. Leave unset to read the current one.
            clear: True removes the description (do not pass a description with it).

        Returns JSON: {collections, undescribed, max_length} for the listing, else
        {collection, description} (+ changed: true after a set or clear).
        """
        paths = get_paths()
        if clear and (description or "").strip():
            return _error({"code": "bad_request",
                           "error": "pass either a description or clear=true, not both"})
        if not collection.strip():
            if clear or description is not None:
                return _error({"code": "bad_request",
                               "error": "name the collection to set or clear its description"})
            r = await asyncio.to_thread(api.collection_descriptions, paths, client=client)
        elif clear or description is not None:
            r = await asyncio.to_thread(api.describe_collection, paths, collection,
                                        "" if clear else description, client=client)
        else:
            r = await asyncio.to_thread(api.collection_descriptions, paths, collection, client=client)
        return _json(r["result"]) if r.get("ok") else _error(r)

    async def rag_search(query: str, collection: str = "", top_k: int | None = None) -> str:
        """Semantic + keyword search over indexed documents, reranked by a cross-encoder.

        Args:
            query: Natural-language question or keywords.
            collection: Collection name(s), comma-separated, e.g. "manuals,policies".
                Empty searches every collection this host is authorised for.
            top_k: Number of passages to return (1-25). Leave unset to use the production
                default (`rag-search config set` / the dashboard's Settings tab; 5 otherwise).

        Returns JSON {results: [{rank, score, collection, file, source, doc, page, heading,
        text}], timing: {total_ms, embed_query_ms, keyword_ms, dense_ms, rerank_ms, queue_ms,
        server_ms, round_trip_ms, ...}}. `score` is the reranker relevance in [0, 1]. `page`
        is the page number in the source document - use it as the citation. A result with
        `"confidence": "low"` comes from a page the converter could not fully verify (a scan
        whose figures did not add up, say): quote it with that caveat and check the source. A photograph whose
        faces the user has named carries `people: [names]`. When the query names people, their aliases ("me") or groups
        ("family") the user set up on the People tab, the search keeps to the photographs that have them and the reply
        says so in `people: {people, groups, photos, searched_for}`; a query of names only lists those photographs.
        If the engine is still starting the reply has status "warming_up": wait ~30 s and call again.
        """
        r = await asyncio.to_thread(api.search, get_paths(), query, top_k=top_k,
                                    collections=collection, client=client, wait_s=40)
        return _json(r["result"]) if r.get("ok") else _error(r)

    async def rag_grep(pattern: str, collection: str = "", context_lines: int = 2,
                        max_matches: int = 20) -> str:
        """Regex search (case-insensitive) over the converted Markdown of indexed documents.

        Use for exact strings, identifiers, numbers or command names where semantic search
        is unreliable. Does not need the embedding models, so it works while the search
        engine is still starting.

        Args:
            pattern: Python regular expression, e.g. "vserver\\s+create" or "error code \\d+".
            collection: Collection name(s), comma-separated. Empty = every collection this host is authorised for.
            context_lines: Lines of context around each match (0-10).
            max_matches: Maximum matches returned (1-100).

        Returns JSON {matches: [{collection, doc, page, line, text, context}], truncated,
        timing: {total_ms, scan_ms, files_scanned, round_trip_ms}}.
        """
        r = await asyncio.to_thread(api.grep, get_paths(), pattern, collections=collection,
                                    context_lines=context_lines, max_matches=max_matches,
                                    client=client)
        return _json(r["result"]) if r.get("ok") else _error(r)

    async def rag_read_document(collection: str, doc: str, offset: int = 0, max_chars: int = 20000,
                                pages: str = "") -> str:
        """Read one whole document as Markdown, exactly as indexed -- after rag_search or rag_grep found it.

        Search and grep return passages and single lines, which break a table apart (the header row and most
        rows are lost). Use this to read the document whole and quote exact figures instead of estimating.
        A Markdown or text source comes back verbatim (tables, front matter and relative links intact); a
        converted one (PDF, Word ...) comes back as the converted Markdown, with `<!-- page N -->` markers.
        Does not need the embedding models, so it works while the search engine is still starting.

        Args:
            collection: Exact collection name.
            doc: The document's name: `doc` from rag_search or rag_grep, or `name` from rag_list_documents /
                rag_list_collections(documents=true), e.g. "activities/2026-10-11-123". A source file name
                ("2026-10-11-123.md") works when only one document has it.
            offset: Character to start at (0 = the beginning); pass the previous `next_offset` to continue.
            max_chars: Most characters to return (default 20000, at most 40000).
            pages: Optional page range for a paged document: "3", "2-4" or "3-". Offsets then count within it.

        Returns JSON {collection, doc, source, text, offset, next_offset, total_chars, truncated, indexed_at}
        (and `people: [names]` for a photograph whose faces the user has named)
        (+ pages, total_pages when `pages` is used). `next_offset` is null at the end; while it is a number,
        call again with offset=next_offset: the slices join exactly, with no gap or overlap.
        """
        r = await asyncio.to_thread(api.read_document, get_paths(), collection, doc, offset=offset,
                                    max_chars=max_chars, pages=pages, client=client)
        return _json(r["result"]) if r.get("ok") else _error(r)

    async def rag_get_photo(collection: str, doc: str, max_px: int = 1400, quality: int = 85) -> Any:
        """Show a photo from an indexed collection as a JPEG, after rag_search/rag_grep found it.

        Converts iPhone HEIC and fixes rotation on the server, so it works for any image format. Returns a short
        JSON description followed by the image. Use it when the user wants to see a photo, or to check what a
        caption describes.

        Args:
            collection: Exact collection name.
            doc: The photo's name: `doc` from rag_search or rag_grep, e.g. "trips/Ranthambore March 25/IMG_1641"
                (or a source file name that only one document has).
            max_px: Longest side in pixels (256-2048, default 1400).
            quality: JPEG quality (50-95, default 85).

        Returns JSON {collection, doc, source, taken, people, width, height, bytes} and the image.
        Errors: not_an_image, source_missing, decode_failed (detail says why), or the usual unknown
        collection / document.
        """
        r = await asyncio.to_thread(api.get_photo, get_paths(), collection, doc, max_px=max_px,
                                    quality=quality, client=client)
        if not r.get("ok"):
            return _error(r)
        from mcp.server.fastmcp import Image

        res = r["result"]
        return [_json(res["meta"]), Image(data=res["jpeg_bytes"], format="jpeg")]

    async def rag_list_documents(collection: str, prefix: str = "", glob: str = "", limit: int = 200,
                                 offset: int = 0) -> str:
        """List the documents of one collection by name -- a folder (prefix) or a pattern (glob) at a time.

        Lighter than rag_list_collections(documents=true) for a collection with many files.

        Args:
            collection: Exact collection name.
            prefix: Names starting with this, e.g. "rollups/" or "activities/2026-". Empty = all.
            glob: Optional pattern on the name or source path, e.g. "*2026-10*" ("*" matches across folders).
            limit: Most documents returned (1-1000, default 200).
            offset: Skip this many matches (use the previous `next_offset`).

        Returns JSON {collection, total_matches, returned, next_offset, documents: [{name, source, chars,
        indexed_at}]}. `name` is what rag_read_document takes; `chars` is its length.
        """
        r = await asyncio.to_thread(api.list_documents, get_paths(), collection, prefix=prefix, glob=glob,
                                    limit=limit, offset=offset, client=client)
        return _json(r["result"]) if r.get("ok") else _error(r)

    async def _start(mode: str, path: str, restart: bool, rebuild: bool = False) -> str:
        r = await asyncio.to_thread(api.index_start, get_paths(), mode=mode, path=path,
                                    rebuild=rebuild, restart=restart, client=client)
        if not r.get("ok"):
            return _error(r)
        out = {k: v for k, v in r.items() if k != "ok"}
        out["next"] = ("An indexing run is active. Call rag_index_status (wait_seconds up to "
                       f"{MAX_WAIT_SECONDS}) until its status is 'succeeded' or 'partial'; "
                       "the new documents are then published and searchable automatically.")
        return _json(out)

    async def rag_index_update(path: str = "", restart: bool = False, rebuild: bool = False) -> str:
        """Index new or changed documents (incremental) in the background.

        The indexer daemon converts each document to Markdown, chunks and embeds it and
        merges it into its collection; unchanged documents are skipped. When the run
        finishes the new generation is published and the search daemon switches to it
        without downtime. Only one run exists at a time: if one is active this call just
        reports it, unless restart=true, which stops it and starts over (already finished
        documents are not redone).

        Args:
            path: A registered collection's name, or a file or folder inside one (default:
                everything).
            restart: Stop the active run and start again.
            rebuild: Re-embed even unchanged documents.
        """
        return await _start("new", path, restart, rebuild)

    async def rag_index_rebuild(path: str = "", confirm: bool = False, restart: bool = False) -> str:
        """Delete and fully rebuild the indexes for every document under `path`.

        Slow (re-converts everything). Requires confirm=true. Prefer rag_index_update.
        """
        if not confirm:
            return _json({"started": False,
                          "error": "rag_index_rebuild rebuilds everything from scratch; call again "
                                   "with confirm=true if that is intended, or use "
                                   "rag_index_update for incremental updates."})
        return await _start("all", path, restart)

    async def rag_index_status(job_id: str = "", wait_seconds: int = 0) -> str:
        """Report progress of the current (or a given) indexing run.

        Returns the run's status, when it started and how long it has been running, the phase
        and current document with how long each has taken so far, and `documents` with the
        per-document timings (convert_s, embed_s, total_s) of the documents handled so far.

        Args:
            job_id: Id from rag_index_update / rag_index_rebuild. Empty = current/latest run.
            wait_seconds: Wait up to this many seconds (max 50) for the run to finish
                before answering; saves polling.
        """
        paths = get_paths()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0, min(int(wait_seconds), MAX_WAIT_SECONDS))
        while True:
            r = await asyncio.to_thread(api.index_status, paths, job_id=job_id, client=client)
            if not r.get("ok"):
                return _error(r)
            if not r.get("running") or loop.time() >= deadline:
                return _json({k: v for k, v in r.items() if k != "ok"})
            await asyncio.sleep(1.0)

    async def rag_index_cancel() -> str:
        """Stop the running indexing run (nothing is published; finished documents are kept)."""
        r = await asyncio.to_thread(api.index_cancel, get_paths(), client=client)
        return _json({k: v for k, v in r.items() if k != "ok"}) if r.get("ok") else _error(r)

    fns = [rag_list_collections, rag_describe_collection, rag_search, rag_grep, rag_read_document, rag_get_photo,
           rag_list_documents,
           rag_index_update, rag_index_rebuild, rag_index_status, rag_index_cancel]
    return {f.__name__: f for f in fns}


def build_server(profile: str = DEFAULT_PROFILE, tool_prefix: str = "",
                 client_id: str = "") -> Any:
    try:
        from mcp.server import MCPServer
    except ImportError:  # mcp 1.x
        from mcp.server.fastmcp import FastMCP as MCPServer  # type: ignore

    prof = get_profile(profile)
    client = normalize_client(client_id or os.environ.get("RAG_SEARCH_CLIENT") or prof.client)
    try:
        server = MCPServer("rag-search", instructions=instructions(prof))
    except TypeError:
        server = MCPServer("rag-search")
    for name, fn in make_tools(client).items():
        server.tool(name=tool_prefix + name)(fn)
    return server


def _profile_name(value: str) -> str:
    name = str(value).strip().lower()
    if not valid_client_name(name) or name in RESERVED_CLIENTS:
        raise argparse.ArgumentTypeError(
            f"{value!r} is not a usable client name (letters, digits, '_', '.', '-'; "
            f"not one of {', '.join(RESERVED_CLIENTS)})")
    return name


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="rag-search-mcp",
                                 description="rag-search MCP adapter (stdio).")
    ap.add_argument("--profile", type=_profile_name, default=DEFAULT_PROFILE, metavar="NAME",
                    help="the host's name, e.g. claude; it is the client identity that "
                         "selects which collections this adapter may see (manage with "
                         "`rag-search access`). Any new name is a new client. Default: claude")
    ap.add_argument("--client-id", default="", help="override the client identity")
    ap.add_argument("--tool-prefix", default="",
                    help="extra prefix for the tool names (they already start with rag_)")
    ap.add_argument("--home", default="", help="data folder (= $RAG_SEARCH_HOME)")
    ap.add_argument("--version", action="version", version=f"rag-search-mcp {__version__}")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)
    if args.home:
        os.environ["RAG_SEARCH_HOME"] = str(Path(args.home).expanduser().absolute())
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="[rag-search-mcp] %(levelname)s %(message)s")
    log.info("rag-search %s MCP adapter (profile=%s, home=%s)", __version__, args.profile,
             get_paths().home)
    try:
        server = build_server(args.profile, args.tool_prefix, args.client_id)
    except ImportError as exc:
        print(f"rag-search-mcp needs the 'mcp' package ({exc}); "
              "install it with: pip install 'rag-search[mcp]'", file=sys.stderr)
        raise SystemExit(1) from None
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
